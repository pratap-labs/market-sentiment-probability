"""
Streamlit view for logging in with Kite Connect and fetching positions.
Enhanced with Options Analytics Dashboard - Refactored with modular tabs.
"""

import os
import sys
import streamlit as st
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional

# Add project root to path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def load_env_file(env_path: Path) -> None:
    """Load simple KEY=VALUE pairs into environment if not already set."""
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and (key not in os.environ or not os.environ.get(key)):
                os.environ[key] = value
    except Exception:
        return


load_env_file(Path(ROOT) / ".env")
try:
    if "KITE_API_KEY" in st.secrets and not os.getenv("KITE_API_KEY"):
        os.environ["KITE_API_KEY"] = st.secrets["KITE_API_KEY"]
    if "KITE_API_SECRET" in st.secrets and not os.getenv("KITE_API_SECRET"):
        os.environ["KITE_API_SECRET"] = st.secrets["KITE_API_SECRET"]
    if "KITE_API_KEY_2" in st.secrets and not os.getenv("KITE_API_KEY_2"):
        os.environ["KITE_API_KEY_2"] = st.secrets["KITE_API_KEY_2"]
    if "KITE_API_SECRET_2" in st.secrets and not os.getenv("KITE_API_SECRET_2"):
        os.environ["KITE_API_SECRET_2"] = st.secrets["KITE_API_SECRET_2"]
except Exception:
    pass

# Cache directory for credentials
CACHE_DIR = Path(ROOT) / "database" / "derivatives_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
KITE_CREDS_FILE = CACHE_DIR / "kite_credentials.json"
KITE_CREDS_FILE_2 = CACHE_DIR / "kite_credentials_2.json"
KITE_TOKEN_TTL = timedelta(hours=12)

try:
    from kiteconnect import KiteConnect
except Exception as e:
    KiteConnect = None
    print(f"⚠️ Failed to import kiteconnect: {e}")

try:
    from py_vollib.black_scholes import black_scholes as bs
    from py_vollib.black_scholes.implied_volatility import implied_volatility as iv
    from py_vollib.black_scholes.greeks import analytical as greeks
except Exception:
    bs = iv = greeks = None



# Import tab render functions
from views.tabs.login_tab import render_login_tab
from views.tabs.portfolio_dashboard_tab import render_portfolio_dashboard_tab
from views.tabs.diagnostics_tab import render_diagnostics_tab
from views.tabs.market_regime_tab import render_market_regime_tab
from views.tabs.risk_analysis_tab import render_risk_analysis_tab
from views.tabs.risk_buckets_tab import render_risk_buckets_tab
from views.tabs.trade_selector_tab import render_trade_selector_tab
from views.tabs.equities_tab import render_equities_tab
from views.tabs.portfolio_buckets_tab import render_portfolio_buckets_tab
from views.tabs.stress_testing_tab import render_stress_testing_tab
from views.tabs.hedge_lab_tab import render_hedge_lab_tab
from views.tabs.historical_performance_tab import render_historical_performance_tab
from views.tabs.greeks_debug_tab import render_greeks_debug_tab
from views.tabs.product_overview_tab import render_product_overview_tab
from views.tabs.derivatives_data_tab import render_derivatives_data_tab, load_cached_derivatives_data_for_session


KITE_ACCOUNT_PRIMARY = "primary"
KITE_ACCOUNT_SECONDARY = "secondary"


def _account_suffix(account: str) -> str:
    return "primary" if account == KITE_ACCOUNT_PRIMARY else "secondary"


def _account_session_key(base: str, account: str) -> str:
    return f"{base}_{_account_suffix(account)}"


def _account_env_key(account: str, base: str) -> str:
    if account == KITE_ACCOUNT_PRIMARY:
        return base
    return f"{base}_2"


def _account_creds_file(account: str) -> Path:
    return KITE_CREDS_FILE if account == KITE_ACCOUNT_PRIMARY else KITE_CREDS_FILE_2


def save_kite_credentials(
    access_token: str,
    api_key: str,
    saved_at: Optional[str] = None,
    account: str = KITE_ACCOUNT_PRIMARY,
):
    """Save Kite credentials to persistent file."""
    try:
        saved_at = saved_at or datetime.now(timezone.utc).isoformat()
        creds = {
            "access_token": access_token,
            "api_key": api_key,
            "saved_at": saved_at
        }
        with open(_account_creds_file(account), 'w') as f:
            json.dump(creds, f)
        return saved_at
    except Exception as e:
        st.error(f"Failed to save credentials: {e}")
        return None


def load_kite_credentials(account: str = KITE_ACCOUNT_PRIMARY):
    """Load Kite credentials from persistent file."""
    try:
        creds_file = _account_creds_file(account)
        if creds_file.exists():
            with open(creds_file, 'r') as f:
                creds = json.load(f)
                return (
                    creds.get("access_token"),
                    creds.get("api_key"),
                    creds.get("saved_at")
                )
    except Exception as e:
        st.error(f"Failed to load credentials: {e}")
    return None, None, None


def clear_kite_credentials(account: str = KITE_ACCOUNT_PRIMARY):
    """Clear saved Kite credentials file."""
    try:
        creds_file = _account_creds_file(account)
        if creds_file.exists():
            creds_file.unlink()
    except Exception as e:
        st.error(f"Failed to clear credentials: {e}")


def clear_kite_session_state():
    """Remove Kite auth info from Streamlit session state."""
    st.session_state.pop("kite_access_token", None)
    st.session_state.pop("kite_api_key", None)
    st.session_state.pop("kite_token_timestamp", None)


def clear_kite_account_state(account: str) -> None:
    st.session_state.pop(_account_session_key("kite_access_token", account), None)
    st.session_state.pop(_account_session_key("kite_api_key", account), None)
    st.session_state.pop(_account_session_key("kite_token_timestamp", account), None)


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
        active_account = st.session_state.get("kite_active_account", KITE_ACCOUNT_PRIMARY)
        clear_kite_session_state()
        clear_kite_account_state(active_account)
        clear_kite_credentials(active_account)
        st.warning("Kite login expired. Please login again to continue.")


def render():
    """Main entrypoint for the Streamlit view."""
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

        /* Allow custom big headings in Risk Analysis */
        .risk-big-title { font-size: 2.2rem !important; margin-bottom: 0.2rem; }
        .risk-big-subtitle { font-size: 1.2rem !important; margin-top: 0.4rem; }

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
    st.session_state.setdefault(_account_session_key("kite_login_initiated", KITE_ACCOUNT_PRIMARY), False)
    st.session_state.setdefault(_account_session_key("kite_login_initiated", KITE_ACCOUNT_SECONDARY), False)
    if "kite_processing_token" not in st.session_state:
        st.session_state.kite_processing_token = False
    st.session_state.setdefault("kite_active_account", KITE_ACCOUNT_PRIMARY)

    if st.session_state.get("kite_force_logout"):
        clear_kite_session_state()
        clear_kite_credentials(KITE_ACCOUNT_PRIMARY)
        clear_kite_credentials(KITE_ACCOUNT_SECONDARY)
        clear_kite_account_state(KITE_ACCOUNT_PRIMARY)
        clear_kite_account_state(KITE_ACCOUNT_SECONDARY)
        for key in (
            _account_session_key("kite_login_initiated", KITE_ACCOUNT_PRIMARY),
            _account_session_key("kite_login_initiated", KITE_ACCOUNT_SECONDARY),
            "kite_processing_token",
            _account_session_key("kite_api_key_stored", KITE_ACCOUNT_PRIMARY),
            _account_session_key("kite_api_secret_stored", KITE_ACCOUNT_PRIMARY),
            _account_session_key("kite_api_key_stored", KITE_ACCOUNT_SECONDARY),
            _account_session_key("kite_api_secret_stored", KITE_ACCOUNT_SECONDARY),
            "kite_login_account",
            "kite_active_account",
        ):
            st.session_state.pop(key, None)
        st.session_state.pop("kite_force_logout", None)

    def load_account_credentials(account: str) -> None:
        token_key = _account_session_key("kite_access_token", account)
        api_key_key = _account_session_key("kite_api_key", account)
        ts_key = _account_session_key("kite_token_timestamp", account)
        if token_key in st.session_state and api_key_key in st.session_state:
            return
        saved_token, saved_key, saved_at = load_kite_credentials(account)
        if saved_token and saved_key:
            if is_token_expired(saved_at):
                clear_kite_credentials(account)
                st.info(f"Saved Kite credentials ({account}) have expired. Please login again.")
            else:
                st.session_state[token_key] = saved_token
                st.session_state[api_key_key] = saved_key
                st.session_state[ts_key] = saved_at

    def apply_active_account(account: str) -> None:
        token_key = _account_session_key("kite_access_token", account)
        api_key_key = _account_session_key("kite_api_key", account)
        ts_key = _account_session_key("kite_token_timestamp", account)
        access_token = st.session_state.get(token_key)
        api_key = st.session_state.get(api_key_key)
        if access_token and api_key:
            st.session_state["kite_access_token"] = access_token
            st.session_state["kite_api_key"] = api_key
            st.session_state["kite_token_timestamp"] = st.session_state.get(ts_key)
        else:
            clear_kite_session_state()

    # Load credentials from file if not in session
    load_account_credentials(KITE_ACCOUNT_PRIMARY)
    load_account_credentials(KITE_ACCOUNT_SECONDARY)
    apply_active_account(st.session_state.get("kite_active_account", KITE_ACCOUNT_PRIMARY))

    # Ensure session tokens are still within TTL before rendering rest of the dashboard
    enforce_kite_token_ttl()

    # Check for request_token in URL
    query_params = st.query_params
    incoming_request_token = query_params.get("request_token", None)

    # Exchange request token for access token
    if incoming_request_token and not st.session_state.kite_processing_token:
        st.session_state.kite_processing_token = True

        account = st.session_state.get("kite_login_account", KITE_ACCOUNT_PRIMARY)
        api_key = st.session_state.get(_account_session_key("kite_api_key_stored", account)) or os.getenv(
            _account_env_key(account, "KITE_API_KEY")
        )
        api_secret = st.session_state.get(_account_session_key("kite_api_secret_stored", account)) or os.getenv(
            _account_env_key(account, "KITE_API_SECRET")
        )
            
        if api_key and api_secret:
            try:
                with st.spinner("Exchanging request token for access token..."):
                    kite = KiteConnect(api_key=api_key)
                    data = kite.generate_session(incoming_request_token, api_secret=api_secret)
                    access_token = data.get("access_token")
                    
                    if access_token:
                        token_timestamp = datetime.now(timezone.utc).isoformat()
                        st.session_state[_account_session_key("kite_access_token", account)] = access_token
                        st.session_state[_account_session_key("kite_api_key", account)] = api_key
                        st.session_state[_account_session_key("kite_token_timestamp", account)] = token_timestamp
                        if st.session_state.get("kite_active_account", KITE_ACCOUNT_PRIMARY) == account:
                            st.session_state["kite_access_token"] = access_token
                            st.session_state["kite_api_key"] = api_key
                            st.session_state["kite_token_timestamp"] = token_timestamp
                        
                        # Save credentials to persistent file
                        save_kite_credentials(access_token, api_key, token_timestamp, account=account)
                        
                        st.success("✅ Successfully logged in and saved credentials!")
                        
                        st.query_params.clear()
                        st.session_state[_account_session_key("kite_login_initiated", account)] = False
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
    
    # Top navigation bar (single active tab controls)
    tabs = [
        {"key": "login", "label": "🔐 Login"},
        {"key": "portfolio", "label": "📊 Portfolio"},
        {"key": "risk_analysis", "label": "🎯 Risk Analysis"},
        {"key": "equities", "label": "📈 Equities"},
        {"key": "risk_buckets", "label": "Risk Buckets (50/30/20)"},
        {"key": "portfolio_buckets", "label": "Portfolio Buckets"},
        {"key": "market_regime", "label": "🌡️ Market Regime"},
        {"key": "stress_testing", "label": "🧪 Stress Testing"},
        {"key": "diagnostics", "label": "🔍 Position Diagnostics"},
        {"key": "hedge_lab", "label": "🛡️ Hedge Lab"},
        {"key": "historical", "label": "📈 Historical Performance"},
        {"key": "greeks_debug", "label": "🧪 Greeks Debug"},
        {"key": "product_overview", "label": "✨ Product Overview"},
        {"key": "derivatives_data", "label": "💾 Derivatives Data"},
        {"key": "trade_selector", "label": "🎯 Trade Selector"},
    ]
    tab_map = {t["key"]: t["label"] for t in tabs}
    st.session_state.setdefault("active_tab_key", "login")
    if st.session_state["active_tab_key"] not in tab_map:
        st.session_state["active_tab_key"] = "login"

    visible_keys = ["login", "portfolio", "risk_analysis", "risk_buckets", "stress_testing"]
    overflow_keys = [t["key"] for t in tabs if t["key"] not in visible_keys]
    overflow_labels = ["⋯"] + [tab_map[key] for key in overflow_keys]
    if st.session_state["active_tab_key"] in overflow_keys:
        overflow_default = tab_map[st.session_state["active_tab_key"]]
    else:
        overflow_default = "⋯"

    st.markdown(
        """
        <style>
        /* Active button styling */
        button[key="nav_login"][aria-pressed="true"],
        button[key="nav_portfolio"][aria-pressed="true"],
        button[key="nav_risk_analysis"][aria-pressed="true"],
        button[key="nav_risk_buckets"][aria-pressed="true"],
        button[key="nav_stress_testing"][aria-pressed="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border: 2px solid #8b5cf6 !important;
            color: white !important;
            font-weight: 700 !important;
        }
        .controls-marker, .content-marker { display: none; }
        .controls-col {
            background: #182131;
            border: 1px solid #2a3548;
            border-radius: 12px;
            padding: 12px 16px;
        }
        .content-col {
            background: #0e141f;
            border: 1px solid #1e2836;
            border-radius: 12px;
            padding: 12px 16px;
        }
        [data-testid="stSidebar"] {
            background: #14171f;
        }
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 58px;
        }
        [data-testid="stSidebar"] { display: none; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    with st.container():
        nav_cols = st.columns(6)
        for idx, key in enumerate(visible_keys):
            label = tab_map[key]
            is_active = st.session_state["active_tab_key"] == key
            with nav_cols[idx]:
                if st.button(label, key=f"nav_{key}", use_container_width=True, type="primary" if is_active else "secondary"):
                    st.session_state["active_tab_key"] = key
                    st.rerun()
        with nav_cols[5]:
            selection = st.selectbox(
                "More",
                overflow_labels,
                index=overflow_labels.index(overflow_default),
                key="nav_more",
                label_visibility="collapsed",
            )
            if selection != "⋯":
                selected_key = next(
                    (k for k, v in tab_map.items() if v == selection),
                    None,
                )
                if selected_key:
                    st.session_state["active_tab_key"] = selected_key
    active_key = st.session_state["active_tab_key"]

    # Two-column layout: left controls, right content
    controls_col, content_col = st.columns([1, 4], gap="large")
    st.sidebar = controls_col

    with controls_col:
        st.markdown("<div class='controls-marker'></div>", unsafe_allow_html=True)
    with content_col:
        st.markdown("<div class='content-marker'></div>", unsafe_allow_html=True)

        if active_key == "login":
            render_login_tab()
            return

        kite_logged_in = bool(st.session_state.get("kite_access_token") and st.session_state.get("kite_api_key"))
        if not kite_logged_in:
            st.warning("Kite authentication required. Use the Login tab to connect before accessing the dashboard.")
            st.info("🔒 Login with Kite to unlock this tab.")
            return

        missing_derivative_cache = load_cached_derivatives_data_for_session()
        if missing_derivative_cache:
            missing_str = ", ".join(missing_derivative_cache)
            st.info(f"⚠️ Derivatives cache missing or expired for: {missing_str}. Please load fresh data from the Derivatives Data tab.")

        if active_key == "portfolio":
            render_portfolio_dashboard_tab()
        elif active_key == "risk_analysis":
            render_risk_analysis_tab()
        elif active_key == "equities":
            render_equities_tab()
        elif active_key == "risk_buckets":
            render_risk_buckets_tab()
        elif active_key == "portfolio_buckets":
            render_portfolio_buckets_tab()
        elif active_key == "market_regime":
            render_market_regime_tab()
        elif active_key == "stress_testing":
            render_stress_testing_tab()
        elif active_key == "diagnostics":
            render_diagnostics_tab()
        elif active_key == "hedge_lab":
            render_hedge_lab_tab()
        elif active_key == "historical":
            render_historical_performance_tab()
        elif active_key == "greeks_debug":
            render_greeks_debug_tab()
        elif active_key == "product_overview":
            render_product_overview_tab()
        elif active_key == "derivatives_data":
            render_derivatives_data_tab()
        elif active_key == "trade_selector":
            render_trade_selector_tab()

    # markers are hidden via CSS; no closing tags required
    st.markdown(
        """
        <script>
        (function() {
          const applyClasses = () => {
            const controlsMarker = document.querySelector('.controls-marker');
            const contentMarker = document.querySelector('.content-marker');
            if (controlsMarker) {
              const controlsCol = controlsMarker.closest('[data-testid="column"]');
              if (controlsCol) controlsCol.classList.add('controls-col');
            }
            if (contentMarker) {
              const contentCol = contentMarker.closest('[data-testid="column"]');
              if (contentCol) contentCol.classList.add('content-col');
            }
          };
          setTimeout(applyClasses, 50);
          setTimeout(applyClasses, 250);
        })();
        </script>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    render()
    # Ensure session tokens are still within TTL
    enforce_kite_token_ttl()
