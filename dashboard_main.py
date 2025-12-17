"""
Streamlit view for logging in with Kite Connect and fetching positions.
Enhanced with Options Analytics Dashboard - Refactored with modular tabs.
"""

import os
import sys
import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Cache directory for credentials
CACHE_DIR = Path(ROOT) / "database" / "derivatives_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
KITE_CREDS_FILE = CACHE_DIR / "kite_credentials.json"
KITE_TOKEN_TTL = timedelta(hours=12)

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
    render_data_hub_tab,
    render_kite_instruments_tab,
    render_nifty_overview_tab,
    render_risk_analysis_tab
)

from views.tabs.derivatives_data_tab import (
    render_derivatives_data_tab,
    load_cached_derivatives_data_for_session
)

try:
    from views.tabs import data_hub_tab
except Exception as e:
    data_hub_tab = None
    print(f"Failed to import data_hub: {e}")


def save_kite_credentials(access_token: str, api_key: str, saved_at: Optional[str] = None):
    """Save Kite credentials to persistent file."""
    try:
        saved_at = saved_at or datetime.now(timezone.utc).isoformat()
        creds = {
            "access_token": access_token,
            "api_key": api_key,
            "saved_at": saved_at
        }
        with open(KITE_CREDS_FILE, 'w') as f:
            json.dump(creds, f)
        return saved_at
    except Exception as e:
        st.error(f"Failed to save credentials: {e}")
        return None


def load_kite_credentials():
    """Load Kite credentials from persistent file."""
    try:
        if KITE_CREDS_FILE.exists():
            with open(KITE_CREDS_FILE, 'r') as f:
                creds = json.load(f)
                return (
                    creds.get("access_token"),
                    creds.get("api_key"),
                    creds.get("saved_at")
                )
    except Exception as e:
        st.error(f"Failed to load credentials: {e}")
    return None, None, None


def clear_kite_credentials():
    """Clear saved Kite credentials file."""
    try:
        if KITE_CREDS_FILE.exists():
            KITE_CREDS_FILE.unlink()
    except Exception as e:
        st.error(f"Failed to clear credentials: {e}")


def clear_kite_session_state():
    """Remove Kite auth info from Streamlit session state."""
    st.session_state.pop("kite_access_token", None)
    st.session_state.pop("kite_api_key", None)
    st.session_state.pop("kite_token_timestamp", None)


def is_token_expired(saved_at: Optional[str]) -> bool:
    """Return True if the stored Kite token timestamp is older than the TTL."""
    if not saved_at:
        return True
    try:
        saved_dt = datetime.fromisoformat(saved_at)
    except ValueError:
        return True
    if saved_dt.tzinfo is None:
        saved_dt = saved_dt.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - saved_dt >= KITE_TOKEN_TTL


def enforce_kite_token_ttl():
    """Expire the in-memory/session token if it is older than the TTL."""
    token_ts = st.session_state.get("kite_token_timestamp")
    token_present = st.session_state.get("kite_access_token") and st.session_state.get("kite_api_key")
    if not token_present:
        return
    if is_token_expired(token_ts):
        clear_kite_session_state()
        clear_kite_credentials()
        st.warning("Kite login expired. Please login again to continue.")


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
    
    # Load credentials from file if not in session
    if "kite_access_token" not in st.session_state or "kite_api_key" not in st.session_state:
        saved_token, saved_key, saved_at = load_kite_credentials()
        if saved_token and saved_key:
            if is_token_expired(saved_at):
                clear_kite_credentials()
                st.info("Saved Kite credentials have expired. Please login again.")
            else:
                st.session_state["kite_access_token"] = saved_token
                st.session_state["kite_api_key"] = saved_key
                st.session_state["kite_token_timestamp"] = saved_at
                st.success("✅ Loaded saved Kite credentials from file")

    # Ensure session tokens are still within TTL before rendering rest of the dashboard
    enforce_kite_token_ttl()

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
                        token_timestamp = datetime.now(timezone.utc).isoformat()
                        st.session_state["kite_token_timestamp"] = token_timestamp
                        
                        # Save credentials to persistent file
                        save_kite_credentials(access_token, api_key, token_timestamp)
                        
                        st.success("✅ Successfully logged in and saved credentials!")
                        
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

    # Initialize empty dataframes - data will be loaded from Derivatives Data tab
    # Each tab will load data from cache using their own reload buttons
    
    # Create tabs - Only show essential tabs
    tabs = st.tabs([
        "🔐 Login",
        "🧭 Overview",
        "📊 Positions", 
        "📈 Portfolio Overview",
        "🔍 Position Diagnostics",
        "🎯 Risk Analysis",
        "🌡️ Market Regime",
        "💾 Derivatives Data"
    ])

    # Render each tab
    with tabs[0]:
        render_login_tab()

    kite_logged_in = bool(st.session_state.get("kite_access_token") and st.session_state.get("kite_api_key"))

    if not kite_logged_in:
        disabled_msg = "🔒 Login with Kite to unlock this tab."
        st.warning("Kite authentication required. Use the Login tab to connect before accessing the dashboard.")
        for tab in tabs[1:]:
            with tab:
                st.info(disabled_msg)
        return

    missing_derivative_cache = load_cached_derivatives_data_for_session()
    if missing_derivative_cache:
        missing_str = ", ".join(missing_derivative_cache)
        st.info(f"⚠️ Derivatives cache missing or expired for: {missing_str}. Please load fresh data from the Derivatives Data tab.")

    with tabs[1]:
        render_overview_tab()

    with tabs[2]:
        render_positions_tab()
    
    with tabs[3]:
        render_portfolio_tab()
    
    with tabs[4]:
        render_diagnostics_tab()
    
    with tabs[5]:
        render_risk_analysis_tab()
    
    with tabs[6]:
        render_market_regime_tab()
    
    with tabs[7]:
        render_derivatives_data_tab()


if __name__ == "__main__":
    render()
    # Ensure session tokens are still within TTL
    enforce_kite_token_ttl()
