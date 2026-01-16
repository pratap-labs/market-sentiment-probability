"""Historical options performance tab from F&O CSV files."""

import calendar
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

import sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.utils import parse_tradingsymbol


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    def _norm(val: str) -> str:
        return re.sub(r"[^a-z0-9]", "", val.lower())

    normalized = {_norm(col): col for col in df.columns}
    for cand in candidates:
        key = _norm(cand)
        if key in normalized:
            return normalized[key]
    return None


def _to_number(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def _last_tuesday(year: int, month: int) -> datetime:
    last_day = datetime(year, month, calendar.monthrange(year, month)[1])
    while last_day.weekday() != 1:
        last_day -= timedelta(days=1)
    if last_day.month == 3 and last_day.day == 31:
        last_day -= timedelta(days=1)
    return last_day


def _parse_symbol_metadata(symbol: str) -> Dict[str, Optional[object]]:
    payload = {
        "symbol": symbol,
        "expiry_date": None,
        "option_type": None,
        "expiry_type": None,
        "strike": None,
    }
    if not symbol:
        return payload

    s = str(symbol).upper().strip()
    payload["symbol"] = s

    if s.endswith("FUT"):
        match = re.match(r"^NIFTY(\d{2})([A-Z]{3})FUT$", s)
        if match:
            year_2, mon = match.groups()
            year = 2000 + int(year_2)
            month_map = {
                "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
                "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
            }
            month = month_map.get(mon)
            if month:
                payload["expiry_date"] = _last_tuesday(year, month)
                payload["option_type"] = "FUT"
                payload["expiry_type"] = "quarterly" if month in {3, 6, 9, 12} else "monthly"
        return payload

    parsed = parse_tradingsymbol(s)
    if not parsed:
        return payload

    payload["expiry_date"] = parsed["expiry"]
    payload["option_type"] = parsed["option_type"]
    payload["strike"] = parsed.get("strike")

    if re.match(r"^NIFTY\d{2}[A-Z]{3}", s):
        month = payload["expiry_date"].month if payload["expiry_date"] else None
        payload["expiry_type"] = "quarterly" if month in {3, 6, 9, 12} else "monthly"
    else:
        payload["expiry_type"] = "weekly"

    return payload


def _prepare_trades(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    symbol_col = _find_column(df, ["Symbol"])
    realized_col = _find_column(df, ["Realized P&L", "Realized PnL", "Realized"])
    realized_pct_col = _find_column(df, ["Realized P&L Pct.", "Realized P&L Pct", "Realized PnL Pct", "Realized %"])

    if not symbol_col:
        warnings.append("Missing Symbol column in the CSV.")
        return pd.DataFrame(), warnings
    if not realized_col:
        warnings.append("Missing Realized P&L column in the CSV.")
        return pd.DataFrame(), warnings

    df = df.copy()
    df["symbol"] = df[symbol_col].astype(str)
    df["realized_pnl"] = _to_number(df[realized_col]).fillna(0.0)
    if realized_pct_col:
        df["realized_pnl_pct"] = _to_number(df[realized_pct_col])
    else:
        df["realized_pnl_pct"] = pd.NA

    parsed = df["symbol"].apply(_parse_symbol_metadata)
    df["expiry_date"] = parsed.apply(lambda item: item.get("expiry_date"))
    df["option_type"] = parsed.apply(lambda item: item.get("option_type"))
    df["expiry_type"] = parsed.apply(lambda item: item.get("expiry_type"))
    df["strike"] = parsed.apply(lambda item: item.get("strike"))

    df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce")

    missing_expiries = df["expiry_date"].isna().sum()
    if missing_expiries:
        warnings.append(f"{missing_expiries} rows are missing expiry parsing; filters may exclude them.")

    return df, warnings


def _monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    working = df.dropna(subset=["expiry_date"]).copy()
    if working.empty:
        return pd.DataFrame()
    working["month"] = working["expiry_date"].dt.to_period("M").dt.to_timestamp()
    grouped = working.groupby("month")
    summary = grouped["realized_pnl"].agg(
        total_pnl="sum",
        total_trades="count",
        avg_pnl="mean",
    ).reset_index()
    summary["win_rate"] = grouped.apply(lambda g: (g["realized_pnl"] > 0).mean() * 100).values
    summary["month_label"] = summary["month"].dt.strftime("%b %Y")
    return summary.sort_values("month")


def _render_metrics(df: pd.DataFrame, monthly: pd.DataFrame):
    pnl = df["realized_pnl"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    total_realized = pnl.sum()
    total_trades = len(df)
    win_rate = (pnl > 0).mean() * 100 if total_trades else 0.0
    avg_win = wins.mean() if not wins.empty else 0.0
    avg_loss = losses.mean() if not losses.empty else 0.0
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss else None
    largest_win = wins.max() if not wins.empty else 0.0
    largest_loss = losses.min() if not losses.empty else 0.0

    best_month = None
    worst_month = None
    if not monthly.empty:
        best = monthly.loc[monthly["total_pnl"].idxmax()]
        worst = monthly.loc[monthly["total_pnl"].idxmin()]
        best_month = f"{best['month_label']} (₹{best['total_pnl']:,.0f})"
        worst_month = f"{worst['month_label']} (₹{worst['total_pnl']:,.0f})"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Realized P&L", f"₹{total_realized:,.0f}")
        st.metric("Expiry Trades", f"{total_trades:,}")
    with col2:
        st.metric("Win Rate", f"{win_rate:.1f}%")
        st.metric("Largest Win", f"₹{largest_win:,.0f}")
    with col3:
        st.metric("Avg Profit (Win)", f"₹{avg_win:,.0f}")
        st.metric("Largest Loss", f"₹{largest_loss:,.0f}")
    with col4:
        st.metric("Avg Loss (Loss)", f"₹{avg_loss:,.0f}")
        st.metric("Profit Factor", f"{profit_factor:.2f}" if profit_factor is not None else "N/A")

    if best_month or worst_month:
        st.caption(
            f"Best Month: {best_month or 'N/A'} | Worst Month: {worst_month or 'N/A'}"
        )


def _aggregate_by_expiry(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("expiry_date", dropna=False)
    rows = []
    for expiry, group in grouped:
        option_types = group["option_type"].dropna().unique().tolist()
        expiry_types = group["expiry_type"].dropna().unique().tolist()
        rows.append({
            "expiry_date": expiry,
            "realized_pnl": group["realized_pnl"].sum(),
            "realized_pnl_pct": group["realized_pnl_pct"].mean(),
            "trade_count": len(group),
            "option_type": option_types[0] if len(option_types) == 1 else "MIXED",
            "expiry_type": expiry_types[0] if len(expiry_types) == 1 else "mixed",
            "strike": group["strike"].median(),
            "symbol": ", ".join(sorted(set(group["symbol"].dropna().astype(str).tolist()))[:3]),
        })
    return pd.DataFrame(rows)


def render_historical_performance_tab():
    """Render historical options performance analysis tab."""
    st.subheader("📈 Historical Options Performance")
    st.caption("Upload NSE F&O table CSV or load from a default tradebook file.")


    input_tabs = st.tabs(["📁 Upload CSV File", "💾 Use Local File"])
    df = None

    with input_tabs[0]:
        st.markdown("### 📤 Upload F&O CSV")
        st.info("Upload CSV in the F&O table format.")
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                with st.spinner("Loading uploaded CSV..."):
                    df = pd.read_csv(uploaded_file)
                    df = _normalize_df(df)
                st.success(f"✅ Loaded {len(df)} rows from {uploaded_file.name}")
            except Exception as exc:
                st.error(f"❌ Error reading uploaded file: {exc}")
                return

    with input_tabs[1]:
        st.markdown("### 💾 Use Local File")
        local_path = os.path.join(ROOT, "tradebook.csv")
        if not os.path.exists(local_path):
            st.error(f"❌ Local file not found: `{local_path}`")
            st.caption("Save your CSV as tradebook.csv at the project root.")
        else:
            file_size = os.path.getsize(local_path) / (1024 * 1024)
            st.info(f"📁 Found local file: `{local_path}` ({file_size:.2f} MB)")
            if st.sidebar.button("🔄 Load Local File", type="primary"):
                try:
                    with st.spinner("Loading local CSV..."):
                        df = pd.read_csv(local_path)
                        df = _normalize_df(df)
                    st.success(f"✅ Loaded {len(df)} rows from local file")
                except Exception as exc:
                    st.error(f"❌ Error reading local file: {exc}")
                    return

    if df is None or df.empty:
        st.warning("⚠️ No data loaded. Upload a CSV or load the local file.")
        return

    trades, warnings = _prepare_trades(df)
    if warnings:
        for warning in warnings:
            st.warning(warning)
    if trades.empty:
        return

    st.markdown("### Key Performance Metrics (per expiry)")
    expiry_trades = _aggregate_by_expiry(trades)
    monthly_summary = _monthly_summary(expiry_trades)
    _render_metrics(expiry_trades, monthly_summary)

    st.markdown("### Filters")
    min_date = trades["expiry_date"].min()
    max_date = trades["expiry_date"].max()
    date_range = None
    if pd.notna(min_date) and pd.notna(max_date):
        date_range = st.sidebar.date_input(
            "Expiry date range",
            value=(min_date.date(), max_date.date()),
        )

    option_types = ["All", "CE", "PE", "FUTURES"]
    option_type_filter = st.sidebar.selectbox("Option Type", options=option_types, index=0)
    min_pnl = float(trades["realized_pnl"].min())
    max_pnl = float(trades["realized_pnl"].max())
    pnl_min = st.sidebar.number_input("Min P&L", value=min_pnl)
    pnl_max = st.sidebar.number_input("Max P&L", value=max_pnl)
    symbol_query = st.sidebar.text_input("Symbol search", value="")

    filtered = trades.copy()
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        mask = filtered["expiry_date"].isna() | (
            (filtered["expiry_date"].dt.date >= start_date) &
            (filtered["expiry_date"].dt.date <= end_date)
        )
        filtered = filtered[mask]
    if option_type_filter != "All":
        target = "FUT" if option_type_filter == "FUTURES" else option_type_filter
        filtered = filtered[filtered["option_type"] == target]
    filtered = filtered[
        (filtered["realized_pnl"] >= pnl_min) & (filtered["realized_pnl"] <= pnl_max)
    ]
    if symbol_query:
        filtered = filtered[filtered["symbol"].str.contains(symbol_query, case=False, na=False)]

    if filtered.empty:
        st.warning("No trades match the filters.")
        return

    expiry_trades = _aggregate_by_expiry(filtered)
    if expiry_trades.empty:
        st.warning("No expiry-level trades match the filters.")
        return

    analysis_tabs = st.tabs([
        "📅 Monthly Summary",
        "📈 P&L Over Time",
        "🧭 Trade Analysis",
        "📋 Data Table"
    ])

    with analysis_tabs[0]:
        st.markdown("#### Monthly Performance Summary")
        monthly = _monthly_summary(expiry_trades)
        if monthly.empty:
            st.info("No monthly summary available for the selected filters.")
        else:
            st.dataframe(monthly[["month_label", "total_pnl", "win_rate", "total_trades", "avg_pnl"]], use_container_width=True)
            chart = px.bar(
                monthly,
                x="month_label",
                y="total_pnl",
                color=monthly["total_pnl"].apply(lambda v: "Profit" if v >= 0 else "Loss"),
                color_discrete_map={"Profit": "#2ca02c", "Loss": "#d62728"},
                labels={"month_label": "Month", "total_pnl": "Monthly P&L"},
                title="Monthly P&L"
            )
            chart.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(chart, use_container_width=True)

    with analysis_tabs[1]:
        st.markdown("#### P&L Over Time")
        timeline = expiry_trades.sort_values("expiry_date")
        timeline = timeline[timeline["expiry_date"].notna()]
        if timeline.empty:
            st.info("Expiry dates are missing for the selected filters.")
        else:
            timeline["cumulative_pnl"] = timeline["realized_pnl"].cumsum()
            line = px.line(
                timeline,
                x="expiry_date",
                y="cumulative_pnl",
                title="Cumulative P&L",
                labels={"expiry_date": "Expiry Date", "cumulative_pnl": "Cumulative P&L"}
            )
            st.plotly_chart(line, use_container_width=True)

            scatter = px.scatter(
                timeline,
                x="expiry_date",
                y="realized_pnl",
                color=timeline["realized_pnl"].apply(lambda v: "Profit" if v >= 0 else "Loss"),
                color_discrete_map={"Profit": "#2ca02c", "Loss": "#d62728"},
                title="Trade P&L by Expiry",
                labels={"expiry_date": "Expiry Date", "realized_pnl": "Trade P&L"}
            )
            st.plotly_chart(scatter, use_container_width=True)

            monthly = _monthly_summary(expiry_trades)
            if not monthly.empty:
                bar = px.bar(
                    monthly,
                    x="month_label",
                    y="total_pnl",
                    color=monthly["total_pnl"].apply(lambda v: "Profit" if v >= 0 else "Loss"),
                    color_discrete_map={"Profit": "#2ca02c", "Loss": "#d62728"},
                    title="Monthly P&L (Filtered)",
                    labels={"month_label": "Month", "total_pnl": "Monthly P&L"}
                )
                bar.update_layout(showlegend=False, xaxis_tickangle=-45)
                st.plotly_chart(bar, use_container_width=True)

    with analysis_tabs[2]:
        st.markdown("#### Trade Breakdown")
        by_type = expiry_trades.groupby("option_type")["realized_pnl"].agg(["count", "sum", "mean"]).reset_index()
        by_type = by_type.sort_values("sum", ascending=False)
        st.dataframe(by_type, use_container_width=True)

        type_chart = px.bar(
            by_type,
            x="option_type",
            y="sum",
            color=by_type["sum"].apply(lambda v: "Profit" if v >= 0 else "Loss"),
            color_discrete_map={"Profit": "#2ca02c", "Loss": "#d62728"},
            title="P&L by Option Type",
            labels={"option_type": "Type", "sum": "Total P&L"}
        )
        type_chart.update_layout(showlegend=False)
        st.plotly_chart(type_chart, use_container_width=True)

        strikes = expiry_trades.dropna(subset=["strike"]).copy()
        if not strikes.empty:
            min_strike = strikes["strike"].min()
            max_strike = strikes["strike"].max()
            step = 500 if max_strike - min_strike <= 20000 else 1000
            bins = list(range(int(min_strike // step * step), int(max_strike + step), step))
            strikes["strike_bucket"] = pd.cut(strikes["strike"], bins=bins, include_lowest=True)
            strike_summary = strikes.groupby("strike_bucket")["realized_pnl"].sum().reset_index()
            strike_summary["strike_bucket"] = strike_summary["strike_bucket"].astype(str)
            st.dataframe(strike_summary, use_container_width=True)
            strike_chart = px.bar(
                strike_summary,
                x="strike_bucket",
                y="realized_pnl",
                title="P&L by Strike Range",
                labels={"strike_bucket": "Strike Range", "realized_pnl": "Total P&L"}
            )
            strike_chart.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(strike_chart, use_container_width=True)

        expiry_summary = expiry_trades.groupby("expiry_type")["realized_pnl"].agg(["count", "sum", "mean"]).reset_index()
        st.dataframe(expiry_summary, use_container_width=True)

        if expiry_trades["realized_pnl_pct"].notna().any():
            pct_chart = px.histogram(
                expiry_trades.dropna(subset=["realized_pnl_pct"]),
                x="realized_pnl_pct",
                nbins=40,
                title="Distribution of Realized P&L %",
                labels={"realized_pnl_pct": "Realized P&L %"}
            )
            st.plotly_chart(pct_chart, use_container_width=True)

    with analysis_tabs[3]:
        st.markdown("#### Filtered Trades")
        display_cols = [
            "symbol", "expiry_date", "option_type", "expiry_type", "strike", "realized_pnl", "realized_pnl_pct"
        ]
        available_cols = [col for col in display_cols if col in expiry_trades.columns]
        st.dataframe(expiry_trades[available_cols].sort_values("expiry_date"), use_container_width=True)
