"""
Streamlit view for logging in with Kite Connect and fetching positions.
Enhanced with Options Analytics Dashboard - Refactored with modular tabs.
"""

import os
import sys
import streamlit as st
import pandas as pd

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from kiteconnect import KiteConnect
except Exception:
    KiteConnect = None

try:
    from py_vollib.black_scholes import black_scholes as bs
    from py_vollib.black_scholes.implied_volatility import implied_volatility as iv
    from py_vollib.black_scholes.greeks import analytical as greeks
except Exception:
    bs = iv = greeks = None

# Import utility functions
from views.utils import (
    load_all_data,
    load_nifty_daily
)

# Import tab render functions
from views.tabs import (
    render_login_tab,
    render_overview_tab,
    render_positions_tab,
    render_portfolio_tab,
    render_diagnostics_tab,
    render_market_regime_tab,
    render_alerts_tab,
    render_advanced_analytics_tab,
    render_trade_history_tab,
    render_data_hub_tab
)

try:
    from views.tabs import data_hub_tab
except Exception as e:
    data_hub_tab = None
    print(f"Failed to import data_hub: {e}")


def render():
    """Main entrypoint for the Streamlit view."""
    st.title("🎯 Options Dashboard")

    # set wider page layout
    st.set_page_config(layout="wide")
    
    # Compact CSS styling for dashboard presentation
    st.markdown(
        """
        <style>
        /* Slightly reduce global base font for the dashboard */
        html, body, .stApp { font-size: 13px; }

        /* Metric label/value sizing */
        div[data-testid="stMetricValue"], div[data-testid="stMetricDelta"] {
            font-size: 18px !important;
            line-height: 1 !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 12px !important;
        }

        /* Headings slightly smaller */
        h1, h2, h3, h4, h5 { font-size: 1.05rem !important; }

        /* Table cells smaller */
        table, th, td, .stDataFrame, .element-container table { font-size: 12px !important; }

        /* Compact padding for common Streamlit containers */
        .css-1d391kg, .css-1lcbmhc, .css-18e3th9, .css-12w0qpk { padding: 0.35rem !important; margin: 0.05rem !important; }

        /* Force metric containers to inline-block and take ~24% width to show 4 per row */
        .css-1lcbmhc, .css-1d391kg, .css-18e3th9, .css-12w0qpk {
            display: inline-block !important;
            vertical-align: top !important;
            width: 24% !important;
            box-sizing: border-box !important;
            padding: 0.45rem !important;
        }

        /* Ensure metric inner blocks are compact; keep delta pill sized to content */
        div[data-testid="stMetricLabel"], div[data-testid="stMetricValue"] {
            padding: 0.05rem 0 !important;
            margin: 0 !important;
        }

        /* Make the delta pill inline and sized to its contents (avoid full-width pills) */
        div[data-testid="stMetricDelta"] { font-size: 12px !important; display: inline-flex !important; align-items: center; gap: 0.35rem; padding: 5px 10px}
        div[data-testid="stMetricDelta"] > span { padding: 0.12rem 1rem !important; border-radius: 999px !important; }

        /* Responsive fallbacks */
        @media (max-width: 900px) {
            .css-1lcbmhc, .css-1d391kg, .css-18e3th9, .css-12w0qpk { width: 48% !important; }
        }
        @media (max-width: 520px) {
            .css-1lcbmhc, .css-1d391kg, .css-18e3th9, .css-12w0qpk { width: 100% !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Check dependencies
    if KiteConnect is None:
        st.error("Missing dependency `kiteconnect`. Install with `pip install kiteconnect`.")
        return
    
    if bs is None or iv is None or greeks is None:
        st.error("Missing dependency `py_vollib`. Install with `pip install py_vollib`.")
        return

    # Initialize session state
    if "kite_login_initiated" not in st.session_state:
        st.session_state.kite_login_initiated = False
    if "kite_processing_token" not in st.session_state:
        st.session_state.kite_processing_token = False

    # Check for request_token in URL
    query_params = st.query_params
    incoming_request_token = query_params.get("request_token", None)

    # Exchange request token for access token
    if incoming_request_token and not st.session_state.kite_processing_token:
        st.session_state.kite_processing_token = True
        
        api_key = st.session_state.get("kite_api_key_stored") or os.getenv("KITE_API_KEY")
        api_secret = st.session_state.get("kite_api_secret_stored") or os.getenv("KITE_API_SECRET")
            
        if api_key and api_secret:
            try:
                with st.spinner("Exchanging request token for access token..."):
                    kite = KiteConnect(api_key=api_key)
                    data = kite.generate_session(incoming_request_token, api_secret=api_secret)
                    access_token = data.get("access_token")
                    
                    if access_token:
                        st.session_state["kite_access_token"] = access_token
                        st.session_state["kite_api_key"] = api_key
                        st.success("✅ Successfully logged in!")
                        
                        st.query_params.clear()
                        st.session_state.kite_login_initiated = False
                        st.session_state.kite_processing_token = False
                        st.rerun()
                    else:
                        st.error("Failed to obtain access token from Kite")
                        st.session_state.kite_processing_token = False
                        
            except Exception as e:
                st.error(f"Login failed: {e}")
                st.session_state.kite_processing_token = False
                st.query_params.clear()
        else:
            st.error("API credentials not found. Please enter them again and retry login.")
            st.session_state.kite_processing_token = False
            st.query_params.clear()

    # Load data
    with st.spinner("Loading options data..."):
        options_df, errors = load_all_data()
        
        # Setup Kite client if logged in
        kite_client = None
        kite_token = st.session_state.get("kite_access_token")
        kite_key = st.session_state.get("kite_api_key")
        if kite_token and kite_key:
            try:
                kite_client = KiteConnect(api_key=kite_key)
                kite_client.set_access_token(kite_token)
            except Exception:
                kite_client = None

        # Load NIFTY daily data from Kite if available
        if kite_client is not None:
            try:
                nifty_df = load_nifty_daily(_kite_client=kite_client)
            except RuntimeError as e:
                st.error(f"Failed to load NIFTY daily: {e}")
                nifty_df = pd.DataFrame()
                st.session_state["nifty_loaded_from"] = None
        else:
            nifty_df = pd.DataFrame()
            st.session_state["nifty_loaded_from"] = None
            st.info("Log in via the Login tab to load NIFTY daily price data from Kite.")

        # Store NIFTY data in session state for VaR calculations
        if not nifty_df.empty:
            st.session_state["nifty_loaded_from"] = "Kite API"
            st.session_state["nifty_df"] = nifty_df

            # Validate required columns
            required_cols = {"date", "close", "open", "high", "low"}
            present = set(nifty_df.columns.str.lower())
            missing = required_cols - present
            if missing:
                msg = f"NIFTY daily data missing required columns: {sorted(list(missing))}"
                st.error(msg)
                raise RuntimeError(msg)
    
    # Show data loading warnings if any
    if errors:
        with st.expander("⚠️ Data loading warnings", expanded=False):
            for file, err in errors:
                st.warning(f"{file}: {err}")

    # Create tabs
    tabs = st.tabs([
        "🔐 Login",
        "🧭 Overview",
        "📊 Positions", 
        "📈 Portfolio Overview",
        "🔍 Position Diagnostics", 
        "🌡️ Market Regime",
        "🚨 Risk Alerts",
        "🎯 Advanced Analytics",
        "📜 Trade History",
        "📂 Data Hub"
    ])

    # Render each tab
    with tabs[0]:
        render_login_tab()

    with tabs[1]:
        render_overview_tab(options_df, nifty_df)

    with tabs[2]:
        render_positions_tab(options_df, nifty_df)
    
    with tabs[3]:
        render_portfolio_tab()
    
    with tabs[4]:
        render_diagnostics_tab()
    
    with tabs[5]:
        render_market_regime_tab(options_df, nifty_df)
    
    with tabs[6]:
        render_alerts_tab()
    
    with tabs[7]:
        render_advanced_analytics_tab()
    
    with tabs[8]:
        render_trade_history_tab()
    
    with tabs[9]:
        render_data_hub_tab()


if __name__ == "__main__":
    render()
