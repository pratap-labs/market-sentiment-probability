"""Derivatives Data Management Tab - Fetch and cache NIFTY, Futures, and Options data."""

import streamlit as st
import pandas as pd
import requests
import altair as alt
from datetime import datetime, timedelta, date
import calendar
from pathlib import Path
import os
import sys
import time
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.data import NSEDataFetcher

# Cache directory
CACHE_DIR = Path(ROOT) / "database" / "derivatives_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL = timedelta(days=7)


def get_cache_filename(data_type: str) -> Path:
    """Generate cache filename based on data type."""
    return CACHE_DIR / f"{data_type}.csv"


def get_cache_status(data_type: str) -> Dict[str, Optional[datetime]]:
    """Return cache metadata including freshness flag."""
    cache_file = get_cache_filename(data_type)
    if not cache_file.exists():
        return {
            "path": cache_file,
            "exists": False,
            "fresh": False,
            "last_updated": None,
            "expires_at": None
        }
    last_updated = datetime.fromtimestamp(cache_file.stat().st_mtime)
    expires_at = last_updated + CACHE_TTL
    is_fresh = datetime.now() <= expires_at
    return {
        "path": cache_file,
        "exists": True,
        "fresh": is_fresh,
        "last_updated": last_updated,
        "expires_at": expires_at
    }


def load_from_cache(data_type: str, silent: bool = False) -> pd.DataFrame:
    """Load data from cache if it exists and is within TTL."""
    status = get_cache_status(data_type)
    if not status["exists"]:
        if not silent:
            st.warning(f"⚠️ No cache found for {data_type}. Please fetch fresh data.")
        return pd.DataFrame()
    if not status["fresh"]:
        if not silent:
            st.warning(f"⚠️ Cache for {data_type} has expired. Fetch fresh data from the Derivatives Data tab.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(status["path"])
        if not silent:
            st.success(f"✅ Loaded from cache: {status['path'].name}")
        return df
    except Exception as exc:
        if not silent:
            st.error(f"❌ Failed to read cache for {data_type}: {exc}")
        return pd.DataFrame()


def save_to_cache(df: pd.DataFrame, data_type: str):
    """Save dataframe to cache file."""
    cache_file = get_cache_filename(data_type)
    df.to_csv(cache_file, index=False)
    st.success(f"💾 Saved to cache: {cache_file.name} (valid for 24h)")


def load_cached_derivatives_data_for_session() -> List[str]:
    """
    Load cached derivatives datasets into session state after login.
    
    Returns:
        List of data categories that are missing or expired.
    """
    missing_categories: List[str] = []

    # NIFTY OHLCV
    nifty_status = get_cache_status("nifty_ohlcv")
    if nifty_status["fresh"]:
        try:
            st.session_state["nifty_df_cache"] = pd.read_csv(nifty_status["path"])
        except Exception:
            missing_categories.append("NIFTY OHLCV")
            st.session_state.pop("nifty_df_cache", None)
    else:
        missing_categories.append("NIFTY OHLCV")
        st.session_state.pop("nifty_df_cache", None)

    # Futures
    futures_status = get_cache_status("nifty_futures")
    if futures_status["fresh"]:
        try:
            st.session_state["nifty_futures_df_cache"] = pd.read_csv(futures_status["path"])
        except Exception:
            missing_categories.append("NIFTY Futures")
            st.session_state.pop("nifty_futures_df_cache", None)
    else:
        missing_categories.append("NIFTY Futures")
        st.session_state.pop("nifty_futures_df_cache", None)

    # Options CE/PE
    ce_status = get_cache_status("nifty_options_ce")
    pe_status = get_cache_status("nifty_options_pe")

    ce_df = pd.DataFrame()
    pe_df = pd.DataFrame()

    if ce_status["fresh"]:
        try:
            ce_df = pd.read_csv(ce_status["path"])
            st.session_state["options_ce_df_cache"] = ce_df
        except Exception:
            ce_df = pd.DataFrame()
    else:
        st.session_state.pop("options_ce_df_cache", None)

    if ce_df.empty:
        missing_categories.append("NIFTY Options (CE)")

    if pe_status["fresh"]:
        try:
            pe_df = pd.read_csv(pe_status["path"])
            st.session_state["options_pe_df_cache"] = pe_df
        except Exception:
            pe_df = pd.DataFrame()
    else:
        st.session_state.pop("options_pe_df_cache", None)

    if pe_df.empty:
        missing_categories.append("NIFTY Options (PE)")

    if not ce_df.empty and not pe_df.empty:
        st.session_state["options_df_cache"] = pd.concat([ce_df, pe_df], ignore_index=True)
    elif not ce_df.empty:
        st.session_state["options_df_cache"] = ce_df.copy()
    elif not pe_df.empty:
        st.session_state["options_df_cache"] = pe_df.copy()
    else:
        st.session_state.pop("options_df_cache", None)

    return missing_categories


# ==================== NIFTY OHLCV SUBTAB ====================
def render_nifty_ohlcv_subtab():
    """Fetch NIFTY daily data from Kite for last 2 years."""
    
    cache_status = get_cache_status("nifty_ohlcv")
    if cache_status["fresh"]:
        df = load_from_cache("nifty_ohlcv")
        if not df.empty:
            st.session_state["nifty_df_cache"] = df
            st.dataframe(df, use_container_width=True, height=400)
            st.info(f"📊 Loaded {len(df)} rows from cache")
    else:
        st.caption("ℹ️ Cache unavailable or expired. Fetch fresh data to enable cache loading.")

    if st.sidebar.button("🔄 Fetch Fresh NIFTY (index) Data", key="nifty_fetch_fresh"):
        with st.spinner("Fetching NIFTY data from Kite..."):
                try:
                    # Check if Kite session exists in session state
                    kite_token = st.session_state.get("kite_access_token")
                    kite_key = st.session_state.get("kite_api_key")
                    
                    if not kite_token or not kite_key:
                        st.error("❌ Not logged in to Kite. Please login first from Login tab.")
                        st.info("💡 Go to the Login tab and authenticate with your Kite credentials.")
                        return
                    
                    # Initialize Kite Connect
                    try:
                        from kiteconnect import KiteConnect
                    except ImportError:
                        st.error("❌ KiteConnect library not installed. Run: pip install kiteconnect")
                        return
                    
                    kite = KiteConnect(api_key=kite_key)
                    kite.set_access_token(kite_token)
                    
                    # Calculate date range (last 2 years)
                    to_date = datetime.now()
                    from_date = to_date - timedelta(days=730)  # 2 years
                    
                    # Fetch NIFTY 50 historical data
                    # NIFTY 50 instrument token: 256265
                    instrument_token = 256265
                    interval = "day"
                    
                    st.info(f"📅 Fetching data from {from_date.date()} to {to_date.date()}")
                    
                    historical_data = kite.historical_data(
                        instrument_token=instrument_token,
                        from_date=from_date,
                        to_date=to_date,
                        interval=interval
                    )
                    
                    if historical_data:
                        df = pd.DataFrame(historical_data)
                        
                        # Save to cache
                        save_to_cache(df, "nifty_ohlcv")
                        cache_status = get_cache_status("nifty_ohlcv")
                        st.session_state["nifty_df_cache"] = df
                    
                        # Display data
                        st.dataframe(df, use_container_width=True, height=400)
                        st.success(f"✅ Fetched {len(df)} rows of NIFTY data")
                        
                        # Show summary stats
                        st.markdown("#### 📈 Summary Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Latest Close", f"₹{df['close'].iloc[-1]:.2f}")
                        with col2:
                            st.metric("2Y High", f"₹{df['high'].max():.2f}")
                        with col3:
                            st.metric("2Y Low", f"₹{df['low'].min():.2f}")
                        with col4:
                            avg_volume = df['volume'].mean()
                            st.metric("Avg Volume", f"{avg_volume:,.0f}")
                    else:
                        st.error("❌ No data received from Kite")
                        
                except Exception as e:
                    st.error(f"❌ Error fetching NIFTY data: {str(e)}")
    
    # Show cache info
    if cache_status["last_updated"]:
        mod_time = cache_status["last_updated"]
        st.caption(f"🕐 Cache last updated: {mod_time.strftime('%Y-%m-%d %H:%M:%S')} (expires after 24h)")
    else:
        st.caption("🕐 Cache last updated: never")


# ==================== FUTURES DATA SUBTAB ====================
def _last_tuesday(year: int, month: int) -> datetime:
    last_day = datetime(year, month, calendar.monthrange(year, month)[1])
    while last_day.weekday() != 1:
        last_day -= timedelta(days=1)
    if last_day.month == 3 and last_day.day == 31:
        last_day -= timedelta(days=1)
    return last_day


def _build_expiry_list() -> List[str]:
    today = datetime.now()
    expiries: List[datetime] = []
    year = today.year
    month = today.month
    while len(expiries) < 3:
        expiry = _last_tuesday(year, month)
        if expiry >= today:
            expiries.append(expiry)
        month += 1
        if month > 12:
            month = 1
            year += 1
    unique_expiries = sorted({expiry.date(): expiry for expiry in expiries}.values())
    return [expiry.strftime("%d-%b-%Y") for expiry in unique_expiries]


def _get_nifty_close_df(start_date: Optional[date] = None) -> pd.DataFrame:
    """Return NIFTY close series for overlay charts."""
    nifty_df = st.session_state.get("nifty_df_cache", pd.DataFrame())
    if nifty_df.empty or "date" not in nifty_df.columns or "close" not in nifty_df.columns:
        return pd.DataFrame()
    df = nifty_df[["date", "close"]].copy()
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date_dt", "close"]).sort_values("date_dt")
    if start_date:
        df = df[df["date_dt"].dt.date >= start_date]
    return df


def _coerce_date_columns(
    df: pd.DataFrame,
    date_col: str = "date",
    expiry_col: str = "expiry_date",
    fallback_expiry_col: str = "expiry",
) -> pd.DataFrame:
    """Normalize date columns for plotting."""
    if df.empty:
        return df
    df = df.copy()
    for col in (date_col, expiry_col):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    if expiry_col not in df.columns and fallback_expiry_col in df.columns:
        df[expiry_col] = pd.to_datetime(df[fallback_expiry_col], errors="coerce").dt.date
    return df.dropna(subset=[date_col, expiry_col])


def _plot_futures_oi_charts(df: pd.DataFrame, expiry_dates: List[date]):
    """Plot futures OI and change in OI for selected expiries."""
    if df.empty:
        st.info("No futures data available to plot OI trends.")
        return
    if not {"date", "open_interest", "change_in_oi"}.issubset(df.columns):
        st.info("Futures data missing OI columns needed for plotting.")
        return

    df = _coerce_date_columns(df)
    if df.empty:
        st.info("Futures data missing usable dates for plotting.")
        return

    if expiry_dates:
        df_filtered = df[df["expiry_date"].isin(expiry_dates)]
    else:
        df_filtered = df

    if df_filtered.empty:
        today = datetime.now().date()
        unique_expiries = sorted(df["expiry_date"].unique())
        next_expiries = [exp for exp in unique_expiries if exp >= today][:3]
        if not next_expiries:
            next_expiries = unique_expiries[-3:]
        df_filtered = df[df["expiry_date"].isin(next_expiries)]
        if df_filtered.empty:
            st.info("No futures data found for the available expiries.")
            return
        expiry_dates = next_expiries

    df_filtered["expiry_label"] = df_filtered["expiry_date"].apply(lambda d: d.strftime("%d-%b-%Y"))
    df_filtered["date_dt"] = pd.to_datetime(df_filtered["date"], errors="coerce")
    oi_series = (
        df_filtered.groupby(["date_dt", "expiry_label"])["open_interest"]
        .sum()
        .reset_index()
        .pivot(index="date_dt", columns="expiry_label", values="open_interest")
        .sort_index()
    )
    change_series = (
        df_filtered.groupby(["date_dt", "expiry_label"])["change_in_oi"]
        .sum()
        .reset_index()
        .pivot(index="date_dt", columns="expiry_label", values="change_in_oi")
        .sort_index()
    )

    st.markdown("### Futures OI Trends (Next 3 Expiries)")
    st.caption("Daily Open Interest by expiry")
    oi_long = oi_series.reset_index().melt(id_vars="date_dt", var_name="expiry", value_name="oi")
    min_date = oi_long["date_dt"].min().date() if not oi_long.empty else None
    nifty_df = _get_nifty_close_df(start_date=min_date)
    base_oi = alt.Chart(oi_long).mark_line().encode(
        x=alt.X("date_dt:T", title="Date"),
        y=alt.Y("oi:Q", title="Open Interest"),
        color=alt.Color("expiry:N", title="Expiry"),
    )
    if not nifty_df.empty:
        nifty_line = alt.Chart(nifty_df).mark_line(color="#2b2b2b").encode(
            x=alt.X("date_dt:T"),
            y=alt.Y("close:Q", title="NIFTY Close", scale=alt.Scale(zero=False)),
        )
        st.altair_chart(
            alt.layer(base_oi, nifty_line).resolve_scale(y="independent"),
            use_container_width=True,
        )
    else:
        st.altair_chart(base_oi, use_container_width=True)

    st.caption("Daily Change in Open Interest by expiry")
    change_long = change_series.reset_index().melt(id_vars="date_dt", var_name="expiry", value_name="change_oi")
    base_change = alt.Chart(change_long).mark_line().encode(
        x=alt.X("date_dt:T", title="Date"),
        y=alt.Y("change_oi:Q", title="Change in OI"),
        color=alt.Color("expiry:N", title="Expiry"),
    )
    if not nifty_df.empty:
        nifty_line = alt.Chart(nifty_df).mark_line(color="#2b2b2b").encode(
            x=alt.X("date_dt:T"),
            y=alt.Y("close:Q", title="NIFTY Close", scale=alt.Scale(zero=False)),
        )
        st.altair_chart(
            alt.layer(base_change, nifty_line).resolve_scale(y="independent"),
            use_container_width=True,
        )
    else:
        st.altair_chart(base_change, use_container_width=True)


def _plot_options_oi_charts(df: pd.DataFrame, expiry_dates: List[date], label: str):
    """Plot options total OI for selected expiries."""
    if df.empty:
        st.info(f"No {label} data available to plot OI trends.")
        return
    if not {"date", "open_interest"}.issubset(df.columns):
        st.info(f"{label} data missing OI columns needed for plotting.")
        return

    df = _coerce_date_columns(df)
    if df.empty:
        st.info(f"{label} data missing usable dates for plotting.")
        return

    if expiry_dates:
        df_filtered = df[df["expiry_date"].isin(expiry_dates)]
    else:
        df_filtered = df

    if df_filtered.empty:
        today = datetime.now().date()
        unique_expiries = sorted(df["expiry_date"].unique())
        next_expiries = [exp for exp in unique_expiries if exp >= today][:3]
        if not next_expiries:
            next_expiries = unique_expiries[-3:]
        df_filtered = df[df["expiry_date"].isin(next_expiries)]
        if df_filtered.empty:
            st.info(f"No {label} data found for the available expiries.")
            return

    df_filtered["expiry_label"] = df_filtered["expiry_date"].apply(lambda d: d.strftime("%d-%b-%Y"))
    df_filtered["date_dt"] = pd.to_datetime(df_filtered["date"], errors="coerce")
    oi_series = (
        df_filtered.groupby(["date_dt", "expiry_label"])["open_interest"]
        .sum()
        .reset_index()
        .pivot(index="date_dt", columns="expiry_label", values="open_interest")
        .sort_index()
    )
    st.markdown(f"### {label} Total OI (Next 3 Expiries)")
    st.caption("Aggregated open interest across all strikes")
    oi_long = oi_series.reset_index().melt(id_vars="date_dt", var_name="expiry", value_name="oi")
    nifty_df = _get_nifty_close_df()
    base_oi = alt.Chart(oi_long).mark_line().encode(
        x=alt.X("date_dt:T", title="Date"),
        y=alt.Y("oi:Q", title="Open Interest"),
        color=alt.Color("expiry:N", title="Expiry"),
    )
    if not nifty_df.empty:
        nifty_line = alt.Chart(nifty_df).mark_line(color="#2b2b2b").encode(
            x=alt.X("date_dt:T"),
            y=alt.Y("close:Q", title="NIFTY Close", scale=alt.Scale(zero=False)),
        )
        st.altair_chart(
            alt.layer(base_oi, nifty_line).resolve_scale(y="independent"),
            use_container_width=True,
        )
    else:
        st.altair_chart(base_oi, use_container_width=True)


def render_futures_data_subtab():
    """Fetch NIFTY futures data from NSE for current expiries."""
    
    expiries = _build_expiry_list()
    st.info(f"📅 Expiries: {', '.join(expiries)}")
    cache_status = get_cache_status("nifty_futures")
    if cache_status["fresh"]:
        df = load_from_cache("nifty_futures")
        if not df.empty:
            st.session_state["nifty_futures_df_cache"] = df
            st.dataframe(df, use_container_width=True, height=400)
            st.info(f"📊 Loaded {len(df)} rows from cache")
    else:
        st.caption("ℹ️ Cache unavailable or expired. Fetch fresh data to enable cache loading.")
    
    if st.sidebar.button("🔄 Fetch Fresh NIFTY Futures Data", key="futures_fetch_fresh"):
        with st.spinner("Fetching futures data from NSE..."):
                try:
                    fetcher = NSEDataFetcher()
                    all_futures_data = []
                    
                    for expiry_str in expiries:
                        st.write(f"Fetching data for expiry: {expiry_str}")
                        
                        # Parse expiry date
                        expiry_date = datetime.strptime(expiry_str, "%d-%b-%Y")
                        
                        # Calculate date range (90 days before expiry)
                        to_date = datetime.now()
                        from_date = to_date - timedelta(days=90)
                        
                        from_date_str = from_date.strftime('%d-%m-%Y')
                        to_date_str = to_date.strftime('%d-%m-%Y')
                        
                        # Use NSEDataFetcher to fetch data
                        raw_data = fetcher.fetch_futures_data(
                            from_date=from_date_str,
                            to_date=to_date_str,
                            symbol="NIFTY",
                            expiry_str=expiry_date,
                            year=expiry_date.year
                        )
                        
                        if raw_data and 'data' in raw_data and len(raw_data['data']) > 0:
                            df_expiry = fetcher.parse_futures_data(raw_data)
                            if not df_expiry.empty:
                                all_futures_data.append(df_expiry)
                                st.success(f"✅ Fetched {len(df_expiry)} rows for {expiry_str}")
                        else:
                            st.warning(f"⚠️ No data for {expiry_str}")
                        
                        time.sleep(1)
                    
                    if all_futures_data:
                        df_combined = pd.concat(all_futures_data, ignore_index=True)
                        save_to_cache(df_combined, "nifty_futures")
                        cache_status = get_cache_status("nifty_futures")
                        st.session_state["nifty_futures_df_cache"] = df_combined
                        st.dataframe(df_combined, use_container_width=True, height=400)
                        st.success(f"✅ Total {len(df_combined)} rows fetched")
                    else:
                        st.error("❌ No futures data fetched")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

    
    # Show cache info
    if cache_status["last_updated"]:
        mod_time = cache_status["last_updated"]
        st.caption(f"🕐 Cache last updated: {mod_time.strftime('%Y-%m-%d %H:%M:%S')} (expires after 24h)")
    else:
        st.caption("🕐 Cache last updated: never")

    futures_df = st.session_state.get("nifty_futures_df_cache", pd.DataFrame())
    expiry_dates = [datetime.strptime(expiry, "%d-%b-%Y").date() for expiry in expiries[:3]]
    _plot_futures_oi_charts(futures_df, expiry_dates)


# ==================== OPTIONS DATA SUBTAB ====================
def render_options_data_subtab():
    """Fetch NIFTY options data from NSE for current expiries."""
    expiries = _build_expiry_list()
    st.info(f"📅 Expiries: {', '.join(expiries)}")
    cache_status_ce = get_cache_status("nifty_options_ce")
    cache_status_pe = get_cache_status("nifty_options_pe")
    load_disabled = not (cache_status_ce["fresh"] and cache_status_pe["fresh"])
    
    if not load_disabled:
        df_ce = load_from_cache("nifty_options_ce")
        df_pe = load_from_cache("nifty_options_pe")
        
        if not df_ce.empty or not df_pe.empty:
            if not df_ce.empty:
                st.session_state["options_ce_df_cache"] = df_ce
            if not df_pe.empty:
                st.session_state["options_pe_df_cache"] = df_pe
            if not df_ce.empty and not df_pe.empty:
                st.session_state["options_df_cache"] = pd.concat([df_ce, df_pe], ignore_index=True)
            elif not df_ce.empty:
                st.session_state["options_df_cache"] = df_ce.copy()
            elif not df_pe.empty:
                st.session_state["options_df_cache"] = df_pe.copy()

            tab1, tab2 = st.tabs(["📈 Call Options (CE)", "📉 Put Options (PE)"])
            
            with tab1:
                if not df_ce.empty:
                    st.dataframe(df_ce, use_container_width=True, height=400)
                    st.info(f"📊 Loaded {len(df_ce)} CE rows from cache")
            
            with tab2:
                if not df_pe.empty:
                    st.dataframe(df_pe, use_container_width=True, height=400)
                    st.info(f"📊 Loaded {len(df_pe)} PE rows from cache")
            # Avoid rerun loops; cache will display on next interaction.
    else:
        st.caption("ℹ️ Cache unavailable or expired. Fetch fresh CE & PE data to enable cache loading.")
    
    if st.sidebar.button("🔄 Fetch Fresh NIFTY Options Data", key="options_fetch_fresh"):
        with st.spinner("Fetching options data from NSE..."):
            try:
                fetcher = NSEDataFetcher()
                all_ce_data = []
                all_pe_data = []
                
                for expiry_str in expiries:
                        st.write(f"Fetching options for expiry: {expiry_str}")
                        
                        # Parse expiry date
                        expiry_date = datetime.strptime(expiry_str, "%d-%b-%Y")
                        
                        # Calculate date range (90 days before expiry)
                        to_date = datetime.now()
                        from_date = to_date - timedelta(days=90)
                        
                        from_date_str = from_date.strftime('%d-%m-%Y')
                        to_date_str = to_date.strftime('%d-%m-%Y')
                        
                        # Fetch CE (Call Options)
                        raw_data_ce = fetcher.fetch_options_data(
                            from_date=from_date_str,
                            to_date=to_date_str,
                            symbol="NIFTY",
                            expiry_str=expiry_date,
                            option_type="CE",
                            year=expiry_date.year
                        )
                        
                        if raw_data_ce and 'data' in raw_data_ce and len(raw_data_ce['data']) > 0:
                            df_ce = fetcher.parse_options_data(raw_data_ce)
                            if not df_ce.empty:
                                all_ce_data.append(df_ce)
                                st.success(f"✅ CE: Fetched {len(df_ce)} rows for {expiry_str}")
                        else:
                            st.warning(f"⚠️ No CE data for {expiry_str}")
                        
                        time.sleep(1)
                        
                        # Fetch PE (Put Options)
                        raw_data_pe = fetcher.fetch_options_data(
                            from_date=from_date_str,
                            to_date=to_date_str,
                            symbol="NIFTY",
                            expiry_str=expiry_date,
                            option_type="PE",
                            year=expiry_date.year
                        )
                        
                        if raw_data_pe and 'data' in raw_data_pe and len(raw_data_pe['data']) > 0:
                            df_pe = fetcher.parse_options_data(raw_data_pe)
                            if not df_pe.empty:
                                all_pe_data.append(df_pe)
                                st.success(f"✅ PE: Fetched {len(df_pe)} rows for {expiry_str}")
                        else:
                            st.warning(f"⚠️ No PE data for {expiry_str}")
                        
                        time.sleep(1)
                    
                # Combine and save
                if all_ce_data:
                    df_ce_combined = pd.concat(all_ce_data, ignore_index=True)
                    save_to_cache(df_ce_combined, "nifty_options_ce")
                    cache_status_ce = get_cache_status("nifty_options_ce")
                    st.session_state["options_ce_df_cache"] = df_ce_combined
                
                if all_pe_data:
                    df_pe_combined = pd.concat(all_pe_data, ignore_index=True)
                    save_to_cache(df_pe_combined, "nifty_options_pe")
                    cache_status_pe = get_cache_status("nifty_options_pe")
                    st.session_state["options_pe_df_cache"] = df_pe_combined

                if all_ce_data and all_pe_data:
                    st.session_state["options_df_cache"] = pd.concat([df_ce_combined, df_pe_combined], ignore_index=True)
                elif all_ce_data:
                    st.session_state["options_df_cache"] = df_ce_combined.copy()
                elif all_pe_data:
                    st.session_state["options_df_cache"] = df_pe_combined.copy()
                
                # Display
                if all_ce_data or all_pe_data:
                    tab1, tab2 = st.tabs(["📈 Call Options (CE)", "📉 Put Options (PE)"])
                    
                    with tab1:
                        if all_ce_data:
                            st.dataframe(df_ce_combined, use_container_width=True, height=400)
                            st.success(f"✅ Total {len(df_ce_combined)} CE rows")
                    
                    with tab2:
                        if all_pe_data:
                            st.dataframe(df_pe_combined, use_container_width=True, height=400)
                            st.success(f"✅ Total {len(df_pe_combined)} PE rows")
                else:
                    st.error("❌ No options data fetched")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    options_ce_df = st.session_state.get("options_ce_df_cache", pd.DataFrame())
    options_pe_df = st.session_state.get("options_pe_df_cache", pd.DataFrame())
    expiry_dates = [datetime.strptime(expiry, "%d-%b-%Y").date() for expiry in expiries[:3]]
    st.markdown("---")
    tab1, tab2 = st.tabs(["📈 CE OI Trends", "📉 PE OI Trends"])
    with tab1:
        _plot_options_oi_charts(options_ce_df, expiry_dates, "Call (CE)")
    with tab2:
        _plot_options_oi_charts(options_pe_df, expiry_dates, "Put (PE)")

    # Show cache info
    cache_file_ce = cache_status_ce["path"]
    cache_file_pe = cache_status_pe["path"]
    
    col1, col2 = st.columns(2)
    with col1:
        if cache_status_ce["last_updated"]:
            mod_time = cache_status_ce["last_updated"]
            st.caption(f"🕐 CE cache updated: {mod_time.strftime('%Y-%m-%d %H:%M:%S')} (expires after 24h)")
        else:
            st.caption("🕐 CE cache updated: never")
    with col2:
        if cache_status_pe["last_updated"]:
            mod_time = cache_status_pe["last_updated"]
            st.caption(f"🕐 PE cache updated: {mod_time.strftime('%Y-%m-%d %H:%M:%S')} (expires after 24h)")
        else:
            st.caption("🕐 PE cache updated: never")


# ==================== MAIN TAB RENDER ====================
def render_derivatives_data_tab():
    """Main tab with 3 subtabs for different data sources."""
    st.markdown("Fetch and cache NIFTY derivatives data from Kite and NSE")

    st.session_state.setdefault("derivatives_active_tab", "NIFTY OHLCV")
    nav_cols = st.columns(3)
    labels = ["NIFTY OHLCV", "Futures Data", "Options Data"]
    for idx, label in enumerate(labels):
        with nav_cols[idx]:
            if st.button(
                label,
                key=f"derivatives_nav_{label}",
                use_container_width=True,
                type="primary" if st.session_state["derivatives_active_tab"] == label else "secondary",
            ):
                st.session_state["derivatives_active_tab"] = label

    active = st.session_state["derivatives_active_tab"]
    if active == "NIFTY OHLCV":
        render_nifty_ohlcv_subtab()
    elif active == "Futures Data":
        render_futures_data_subtab()
    else:
        render_options_data_subtab()
