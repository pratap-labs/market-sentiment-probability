import streamlit as st
import pandas as pd
import os
import re
from pathlib import Path
import glob
from collections import defaultdict
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# page config will be set by the combined hub view to avoid duplicate calls


CANONICAL_COLUMNS = {
    "symbol": ["symbol"],
    "date": ["date"],
    "expiry": ["expiry", "expiry_date"],
    "option_type": ["option type", "option_type", "optiontype"],
    "strike_price": ["strike", "strike price", "strike_price"],
    "open": ["open"],
    "high": ["high"],
    "low": ["low"],
    "close": ["close"],
    "ltp": ["ltp", "last_traded_price", "ltp "],
    "settle_price": ["settle price", "settle_price", "settle"],
    "no_of_contracts": ["no. of contracts", "no of contracts", "no_of_contracts", "no_of_contract"],
    "turnover": ["turnover", "turnover * in  ₹ lakhs", "turnover_in_lakhs"],
    "premium_turnover": ["premium turnover", "premium turnover ** in   ₹ lakhs", "premium_turnover"],
    "open_int": ["open int", "open_int", "open_interest"],
    "change_in_oi": ["change in oi", "change_in_oi"],
    "underlying_value": ["underlying value", "underlying_value"],
}


def _normalize_col(col: str) -> str:
    if not isinstance(col, str):
        return col
    s = col.lower().strip()
    # remove extra characters
    for ch in ["*", "**", "₹", "(", ")", ","]:
        s = s.replace(ch, "")
    s = s.replace("/", " ")
    s = " ".join(s.split())
    return s


def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Build a mapping from original column -> canonical target using
    # longest matching variant to avoid 'open int' -> 'open' collisions.
    col_map = {}
    cols = df.columns.tolist()
    for c in cols:
        nc = _normalize_col(c)
        best_match = None
        best_len = 0
        for key, variants in CANONICAL_COLUMNS.items():
            for v in variants:
                # match on word boundary to reduce false positives
                if re.search(r"\b" + re.escape(v) + r"\b", nc):
                    if len(v) > best_len:
                        best_len = len(v)
                        best_match = key
        if best_match:
            col_map[c] = best_match

    if not col_map:
        return df

    # Invert mapping to find original columns that map to the same target
    target_map = defaultdict(list)
    for orig, tgt in col_map.items():
        target_map[tgt].append(orig)

    new_df = df.copy()

    # If multiple originals mapped to 'turnover', try to detect the
    # premium turnover column by checking for the word 'premium' in the
    # original header. If found, move it to the 'premium_turnover' target.
    if 'turnover' in target_map and len(target_map['turnover']) > 1:
        origs = target_map['turnover']
        premium_idx = None
        for i, o in enumerate(origs):
            if 'premium' in _normalize_col(o):
                premium_idx = i
                break
        if premium_idx is not None:
            prem_orig = origs.pop(premium_idx)
            target_map.setdefault('premium_turnover', []).append(prem_orig)

    # Process mapped targets: coalesce duplicates (first non-null) and assign
    for tgt, orig_list in list(target_map.items()):
        if len(orig_list) == 1:
            # simple rename
            new_df = new_df.rename(columns={orig_list[0]: tgt})
        else:
            # coalesce across originals: prefer left-most non-null
            series = new_df[orig_list].bfill(axis=1).iloc[:, 0]
            new_df[tgt] = series
            # drop all original columns
            new_df = new_df.drop(columns=orig_list)

    return new_df


def load_all_data(dirs=None):
    """Load all excel/csv files from the given directories.

    If dirs is None, we try `database/data/` and fall back to `database/options_data/`.
    """
    if dirs is None:
        dirs = ["database/data", "database/options_data"]

    all_files = []
    for d in dirs:
        p = Path(d)
        if not p.exists():
            continue
        # glob for common spreadsheet formats
        for ext in ("*.xlsx", "*.xls", "*.csv", "*.xlsm"):
            all_files.extend(sorted([str(x) for x in p.glob(ext)]))

    if not all_files:
        return pd.DataFrame(), []

    frames = []
    errors = []
    for f in all_files:
        try:
            if f.lower().endswith(".csv"):
                df = pd.read_csv(f)
            else:
                df = pd.read_excel(f, sheet_name=0)
        except Exception as e:
            errors.append((f, str(e)))
            continue

        df = _map_columns(df)
        frames.append(df)

    if not frames:
        return pd.DataFrame(), errors

    combined = pd.concat(frames, ignore_index=True, sort=False)

    # Normalize date fields
    if "date" in combined.columns:
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    if "expiry" in combined.columns:
        combined["expiry"] = pd.to_datetime(combined["expiry"], errors="coerce")

    # Ensure strike_price numeric
    if "strike_price" in combined.columns:
        combined["strike_price"] = pd.to_numeric(combined["strike_price"], errors="coerce")

    # Clean and coerce common numeric columns that may contain commas or ₹ signs
    numeric_candidates = [
        "open", "high", "low", "close", "ltp", "settle_price",
        "no_of_contracts", "turnover", "premium_turnover", "open_int",
        "change_in_oi", "underlying_value"
    ]
    for col in numeric_candidates:
        if col in combined.columns:
            # convert to string, strip common non-numeric chars, then coerce
            combined[col] = combined[col].astype(str).str.replace(r"[^0-9.\-eE]", "", regex=True)
            combined[col] = pd.to_numeric(combined[col].replace("", pd.NA), errors="coerce")

    return combined, errors


def render():
    # Debug flag - set to True to show all tabs (Expiry Table, More Visuals)
    # Set to False to show only More Visuals tab
    DEBUG = 0
    

    df, errors = load_all_data()

    if errors:
        st.warning(f"Some files failed to load ({len(errors)}). See console for details.")
        for f, e in errors:
            st.text(f"Failed: {f} -> {e}")

    if df.empty:
        st.info("No files found in `database/data/` or `database/options_data/`. Place your Excel/CSV files there and reload.")
        return

    if DEBUG:
        # Debug mode: Show both tabs
        tab1, tab2 = st.tabs(["Expiry Table", "More Visuals (TBD)"])
        
        with tab1:
            st.subheader("Select Expiry and View Table")
            if "expiry" not in df.columns:
                st.error("No `Expiry` column detected after parsing. Make sure your files have the expected headers.")
            else:
                # list unique expiries
                expiries = sorted([x for x in df["expiry"].dropna().unique()])
                expiries_str = [pd.to_datetime(x).strftime("%Y-%m-%d") for x in expiries]
                choice = st.selectbox("Select Expiry", expiries_str)
                chosen_dt = pd.to_datetime(choice)
                filtered = df[df["expiry"] == chosen_dt].copy()

                # Show a simple preview table and basic metrics
                st.write(f"Rows for expiry {choice}: {len(filtered)}")
                st.dataframe(filtered, use_container_width=True)

        with tab2:
            # st.subheader("More Visuals")
            # st.markdown("Per-expiry: last 60 days CE vs PE open interest (bars) with underlying price (line) — one figure per expiry.")

            if not all(x in df.columns for x in ["date", "expiry", "option_type", "open_int"]):
                st.error("Required columns for plots not found. Need: date, expiry, option_type, open_int")
            else:
                expiries = sorted([x for x in df["expiry"].dropna().unique()])
                # allow user to select which expiries to plot (multi-select)
                sel = st.multiselect("Select expiries to plot (one or more)",
                                     [pd.to_datetime(x).strftime("%Y-%m-%d") for x in expiries],
                                     default=[pd.to_datetime(expiries[-1]).strftime("%Y-%m-%d")] if expiries else [])

                for choice in sel:
                    expiry_dt = pd.to_datetime(choice)
                    # filter last 60 days up to expiry
                    start_dt = expiry_dt - timedelta(days=60)
                    mask = (df["expiry"] == expiry_dt) & (df["date"] >= start_dt) & (df["date"] <= expiry_dt)
                    sub = df.loc[mask].copy()
                    if sub.empty:
                        st.info(f"No rows for expiry {choice} in the last 60 days")
                        continue

                    # Aggregate CE and PE open interest per date
                    sub["option_type"] = sub["option_type"].astype(str).str.upper().str.strip()
                    oi_pivot = sub.pivot_table(index="date", columns="option_type", values="open_int", aggfunc="sum").fillna(0)

                    # Ensure CE and PE columns exist
                    for col in ["CE", "PE"]:
                        if col not in oi_pivot.columns:
                            oi_pivot[col] = 0

                    # make sure index is datetime and sorted
                    oi_pivot.index = pd.to_datetime(oi_pivot.index)
                    oi_pivot = oi_pivot.sort_index()

                    # Underlying price per date (use last observed for that date)
                    if "underlying_value" in sub.columns:
                        underlying = sub.groupby("date")["underlying_value"].last().reindex(oi_pivot.index).ffill()
                    else:
                        underlying = pd.Series([pd.NA] * len(oi_pivot), index=oi_pivot.index)

                    # ensure numeric
                    for col in ["CE", "PE"]:
                        oi_pivot[col] = pd.to_numeric(oi_pivot[col], errors="coerce").fillna(0)
                    underlying = pd.to_numeric(underlying, errors="coerce")

                    # Build plotly figure: grouped bars for CE/PE and line for underlying
                    fig = make_subplots(specs=[[{"secondary_y": True}]])

                    fig.add_trace(
                        go.Bar(x=oi_pivot.index, y=oi_pivot["CE"], name="CE OI", marker_color="#75F37B"),
                        secondary_y=False,
                    )
                    fig.add_trace(
                        go.Bar(x=oi_pivot.index, y=oi_pivot["PE"], name="PE OI", marker_color="#E96767"),
                        secondary_y=False,
                    )

                    fig.add_trace(
                        go.Scatter(x=underlying.index, y=underlying.values, name="Underlying", mode="lines+markers", line=dict(color="#f1f1f1", width=2)),
                        secondary_y=True,
                    )

                    fig.update_layout(barmode="group", height=420, title_text=f"Expiry {choice} — CE vs PE OI (last 60 days)")
                    fig.update_xaxes(title_text="Date", tickangle=45, nticks=15)
                    fig.update_yaxes(title_text="Open Interest (contracts)", secondary_y=False)
                    fig.update_yaxes(title_text="Underlying Price", secondary_y=True)

                    # improve hover formatting
                    fig.update_traces(hovertemplate=None)

                    st.plotly_chart(fig, use_container_width=True)
    else:
        # Production mode: Show only More Visuals (no tabs)
        # st.subheader("More Visuals")
        # st.markdown("Per-expiry: last 60 days CE vs PE open interest (bars) with underlying price (line) — one figure per expiry.")

        if not all(x in df.columns for x in ["date", "expiry", "option_type", "open_int"]):
            st.error("Required columns for plots not found. Need: date, expiry, option_type, open_int")
        else:
            expiries = sorted([x for x in df["expiry"].dropna().unique()])
            # allow user to select which expiries to plot (multi-select)
            sel = st.multiselect("Select expiries to plot (one or more)",
                                 [pd.to_datetime(x).strftime("%Y-%m-%d") for x in expiries],
                                 default=[pd.to_datetime(expiries[-1]).strftime("%Y-%m-%d")] if expiries else [])

            for choice in sel:
                expiry_dt = pd.to_datetime(choice)
                # filter last 60 days up to expiry
                start_dt = expiry_dt - timedelta(days=60)
                mask = (df["expiry"] == expiry_dt) & (df["date"] >= start_dt) & (df["date"] <= expiry_dt)
                sub = df.loc[mask].copy()
                if sub.empty:
                    st.info(f"No rows for expiry {choice} in the last 60 days")
                    continue

                # Aggregate CE and PE open interest per date
                sub["option_type"] = sub["option_type"].astype(str).str.upper().str.strip()
                oi_pivot = sub.pivot_table(index="date", columns="option_type", values="open_int", aggfunc="sum").fillna(0)

                # Ensure CE and PE columns exist
                for col in ["CE", "PE"]:
                    if col not in oi_pivot.columns:
                        oi_pivot[col] = 0

                # make sure index is datetime and sorted
                oi_pivot.index = pd.to_datetime(oi_pivot.index)
                oi_pivot = oi_pivot.sort_index()

                # Underlying price per date (use last observed for that date)
                if "underlying_value" in sub.columns:
                    underlying = sub.groupby("date")["underlying_value"].last().reindex(oi_pivot.index).ffill()
                else:
                    underlying = pd.Series([pd.NA] * len(oi_pivot), index=oi_pivot.index)

                # ensure numeric
                for col in ["CE", "PE"]:
                    oi_pivot[col] = pd.to_numeric(oi_pivot[col], errors="coerce").fillna(0)
                underlying = pd.to_numeric(underlying, errors="coerce")

                # Build plotly figure: grouped bars for CE/PE and line for underlying
                fig = make_subplots(specs=[[{"secondary_y": True}]])

                fig.add_trace(
                    go.Bar(x=oi_pivot.index, y=oi_pivot["CE"], name="CE OI", marker_color="#75F37B"),
                    secondary_y=False,
                )
                fig.add_trace(
                    go.Bar(x=oi_pivot.index, y=oi_pivot["PE"], name="PE OI", marker_color="#E96767"),
                    secondary_y=False,
                )

                fig.add_trace(
                    go.Scatter(x=underlying.index, y=underlying.values, name="Underlying", mode="lines+markers", line=dict(color="#f1f1f1", width=2)),
                    secondary_y=True,
                )

                fig.update_layout(barmode="group", height=420, title_text=f"Expiry {choice} — CE vs PE OI (last 60 days)")
                fig.update_xaxes(title_text="Date", tickangle=45, nticks=15)
                fig.update_yaxes(title_text="Open Interest (contracts)", secondary_y=False)
                fig.update_yaxes(title_text="Underlying Price", secondary_y=True)

                # improve hover formatting
                fig.update_traces(hovertemplate=None)

                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    render()
