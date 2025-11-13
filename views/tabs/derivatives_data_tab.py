"""Derivatives Data Management Tab - Fetch and cache NIFTY, Futures, and Options data."""

import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.data import NSEDataFetcher

# Cache directory
CACHE_DIR = Path(ROOT) / "database" / "derivatives_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_cache_filename(data_type: str, date: str = None) -> Path:
    """Generate cache filename based on data type and date."""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    return CACHE_DIR / f"{data_type}_{date}.csv"


def load_from_cache(data_type: str) -> pd.DataFrame:
    """Load data from most recent cache file."""
    cache_file = get_cache_filename(data_type)
    if cache_file.exists():
        df = pd.read_csv(cache_file)
        st.success(f"✅ Loaded from cache: {cache_file.name}")
        return df
    else:
        st.warning(f"⚠️ No cache found for {data_type}. Please fetch fresh data.")
        return pd.DataFrame()


def save_to_cache(df: pd.DataFrame, data_type: str):
    """Save dataframe to cache file."""
    cache_file = get_cache_filename(data_type)
    df.to_csv(cache_file, index=False)
    st.success(f"💾 Saved to cache: {cache_file.name}")


# ==================== NIFTY OHLCV SUBTAB ====================
def render_nifty_ohlcv_subtab():
    """Fetch NIFTY daily data from Kite for last 2 years."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📂 Load from Cache", key="nifty_load_cache"):
            df = load_from_cache("nifty_ohlcv")
            if not df.empty:
                st.dataframe(df, use_container_width=True, height=400)
                st.info(f"📊 Loaded {len(df)} rows")
    
    with col2:
        if st.button("🔄 Fetch Fresh Data", key="nifty_fetch_fresh"):
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
    cache_file = get_cache_filename("nifty_ohlcv")
    if cache_file.exists():
        mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        st.caption(f"🕐 Cache last updated: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")


# ==================== FUTURES DATA SUBTAB ====================
def render_futures_data_subtab():
    """Fetch NIFTY futures data from NSE for current expiries."""
    
    # Hardcoded expiries (next 3 monthly expiries)
    expiries = ["25-Nov-2025", "30-Dec-2025", "27-Jan-2026"]
    st.info(f"📅 Expiries: {', '.join(expiries)}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📂 Load from Cache", key="futures_load_cache"):
            df = load_from_cache("nifty_futures")
            if not df.empty:
                st.dataframe(df, use_container_width=True, height=400)
                st.info(f"📊 Loaded {len(df)} rows")
    
    with col2:
        if st.button("🔄 Fetch Fresh Data", key="futures_fetch_fresh"):
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
                        st.dataframe(df_combined, use_container_width=True, height=400)
                        st.success(f"✅ Total {len(df_combined)} rows fetched")
                    else:
                        st.error("❌ No futures data fetched")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

    
    # Show cache info
    cache_file = get_cache_filename("nifty_futures")
    if cache_file.exists():
        mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        st.caption(f"🕐 Cache last updated: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")


# ==================== OPTIONS DATA SUBTAB ====================
def render_options_data_subtab():
    """Fetch NIFTY options data from NSE for current expiries."""
    # Hardcoded expiries (next 3 monthly expiries)
    expiries = ["25-Nov-2025", "30-Dec-2025", "27-Jan-2026"]
    st.info(f"📅 Expiries: {', '.join(expiries)}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📂 Load from Cache", key="options_load_cache"):
            df_ce = load_from_cache("nifty_options_ce")
            df_pe = load_from_cache("nifty_options_pe")
            
            if not df_ce.empty or not df_pe.empty:
                tab1, tab2 = st.tabs(["📈 Call Options (CE)", "📉 Put Options (PE)"])
                
                with tab1:
                    if not df_ce.empty:
                        st.dataframe(df_ce, use_container_width=True, height=400)
                        st.info(f"📊 Loaded {len(df_ce)} CE rows")
                
                with tab2:
                    if not df_pe.empty:
                        st.dataframe(df_pe, use_container_width=True, height=400)
                        st.info(f"📊 Loaded {len(df_pe)} PE rows")
    
    with col2:
        if st.button("🔄 Fetch Fresh Data", key="options_fetch_fresh"):
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
                    
                    if all_pe_data:
                        df_pe_combined = pd.concat(all_pe_data, ignore_index=True)
                        save_to_cache(df_pe_combined, "nifty_options_pe")
                    
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
    
    # Show cache info
    cache_file_ce = get_cache_filename("nifty_options_ce")
    cache_file_pe = get_cache_filename("nifty_options_pe")
    
    col1, col2 = st.columns(2)
    with col1:
        if cache_file_ce.exists():
            mod_time = datetime.fromtimestamp(cache_file_ce.stat().st_mtime)
            st.caption(f"🕐 CE cache updated: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    with col2:
        if cache_file_pe.exists():
            mod_time = datetime.fromtimestamp(cache_file_pe.stat().st_mtime)
            st.caption(f"🕐 PE cache updated: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")


# ==================== MAIN TAB RENDER ====================
def render_derivatives_data_tab():
    """Main tab with 3 subtabs for different data sources."""
    st.markdown("Fetch and cache NIFTY derivatives data from Kite and NSE")
    
    # Create 3 subtabs
    tab1, tab2, tab3 = st.tabs([
        "📊 NIFTY OHLCV",
        "📈 Futures Data", 
        "📉 Options Data"
    ])
    
    with tab1:
        render_nifty_ohlcv_subtab()
    
    with tab2:
        render_futures_data_subtab()
    
    with tab3:
        render_options_data_subtab()
