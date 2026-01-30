"""
Streamlit view for logging in with Kite Connect and fetching positions.
Enhanced with Options Analytics Dashboard - Refactored with modular tabs.
"""

import os
import sys
import streamlit as st
import streamlit.components.v1 as components
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional
from streamlit_extras.stylable_container import stylable_container

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
    # Secondary account is no longer supported.
except Exception:
    pass

# Cache directory for credentials
CACHE_DIR = Path(ROOT) / "database" / "derivatives_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
KITE_CREDS_FILE = CACHE_DIR / "kite_credentials.json"
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
from views.tabs.market_regime_tab import render_market_regime_tab
from views.tabs.stress_testing_tab import render_pre_trade_analysis_tab
from views.tabs.trade_selector_tab import render_trade_selector_tab
from views.tabs.risk_buckets_tab import render_risk_buckets_tab
from views.tabs.equities_tab import render_equities_tab
from views.tabs.historical_performance_tab import render_historical_performance_tab
from views.tabs.product_overview_tab import render_product_overview_tab
from views.tabs.derivatives_data_tab import render_derivatives_data_tab, load_cached_derivatives_data_for_session


def save_kite_credentials(
    access_token: str,
    api_key: str,
    saved_at: Optional[str] = None,
):
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
    # set wider page layout
    st.set_page_config(layout="wide")

    # Load GammaShield institutional theme
    try:
        with open("assets/gammashield.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass  # Fallback to default styling

    # Load and encode logo
    import base64
    try:
        with open("assets/gammashield-logo.png", "rb") as f:
            logo_base64 = base64.b64encode(f.read()).decode()
            logo_src = f"data:image/png;base64,{logo_base64}"
    except FileNotFoundError:
        logo_src = ""

    st.sidebar.markdown(
        f"""
        <div class="gs-sidebar-brand" style="text-align: center; padding: 0.6rem 0 1rem 0; border-bottom: 1px solid var(--gs-border, #1F2A3D); margin: 0 0 2rem 0;">
            <div style="display: flex; align-items: center; justify-content: center; gap: 0.75rem; margin-bottom: 0.5rem; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
                {"<img src='" + logo_src + "' style='width: 100%; max-width: 180px; height: auto; object-fit: contain;' />" if logo_src else ""}                <div class="gs-sidebar-title" style="font-size: 2.2rem; font-weight: 800; margin: 0; background: linear-gradient(135deg, #2D7DFF 0%, #1B4FD8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; letter-spacing: -0.02em;">
                </div>
            </div>
            <div class="gs-sidebar-subtitle" style="font-size: 0.8rem; color: var(--gs-muted, #A9B7D0); margin-top: 0.25rem; font-weight: 500; letter-spacing: 0.04em;">
                Advanced Options Risk Engine
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Compact CSS styling for dashboard presentation
    st.markdown(
        """
        <style>
        /* Slightly reduce global base font for the dashboard */
        html, body, .stApp { font-size: 13px; }

        /* Remove default top padding above main content/nav */
        [data-testid="stAppViewContainer"] > .main,
        [data-testid="stAppViewContainer"] > .main > div,
        [data-testid="stMainBlockContainer"],
        [data-testid="stVerticalBlock"] {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }
        
        .main .block-container,
        .block-container {
            padding-top: 0 !important;
            margin-top: 0 !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        
        /* Navigation container styling */
        .gs-nav-container {
            background: var(--gs-surface, #0B1220) !important;
            border: 1px solid var(--gs-border, #1F2A3D) !important;
            border-radius: 12px !important;
            padding: 1rem 1.5rem !important;
            margin: 1rem 0 1.5rem 0 !important;
            display: none !important;
        }
        
        /* Nav button columns */
        .gs-nav-col {
            padding: 0 0.35rem !important;
        }

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

        /* Remove top padding above sidebar logo */
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            padding-top: 0 !important;
        }

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

    # Initialize session state (single account)
    st.session_state.setdefault("kite_login_initiated", False)
    st.session_state.setdefault("kite_redirect_pending", False)
    st.session_state.setdefault("kite_login_url", "")
    if "kite_processing_token" not in st.session_state:
        st.session_state.kite_processing_token = False

    if st.session_state.get("kite_force_logout"):
        clear_kite_session_state()
        clear_kite_credentials()
        for key in (
            "kite_login_initiated",
            "kite_redirect_pending",
            "kite_login_url",
            "kite_processing_token",
            "kite_api_key_stored",
            "kite_api_secret_stored",
        ):
            st.session_state.pop(key, None)
        st.session_state.pop("kite_force_logout", None)

    def load_saved_credentials() -> None:
        if st.session_state.get("kite_access_token") and st.session_state.get("kite_api_key"):
            return
        saved_token, saved_key, saved_at = load_kite_credentials()
        if saved_token and saved_key:
            if is_token_expired(saved_at):
                clear_kite_credentials()
                st.info("Saved Kite credentials have expired. Please login again.")
            else:
                st.session_state["kite_access_token"] = saved_token
                st.session_state["kite_api_key"] = saved_key
                st.session_state["kite_token_timestamp"] = saved_at

    # Load credentials from file if not in session
    load_saved_credentials()

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
                        token_timestamp = datetime.now(timezone.utc).isoformat()
                        st.session_state["kite_access_token"] = access_token
                        st.session_state["kite_api_key"] = api_key
                        st.session_state["kite_token_timestamp"] = token_timestamp
                        
                        # Save credentials to persistent file
                        save_kite_credentials(access_token, api_key, token_timestamp)
                        
                        st.success("✅ Successfully logged in and saved credentials!")
                        
                        st.query_params.clear()
                        st.session_state["kite_login_initiated"] = False
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
    
    # Navigation tabs (single active tab controls)
    tabs = [
        {"key": "login", "label": "🔐 Login"},
        {"key": "portfolio", "label": "📊 Portfolio"},
        {"key": "equities", "label": "📈 Equities"},
        {"key": "risk_buckets", "label": "🛡️ Risk Buckets (50/30/20)"},
        {"key": "pre_trade", "label": "🔬 Pre-Trade Analysis"},
        {"key": "market_regime", "label": "🌡️ Market Regime"},
        {"key": "historical", "label": "📈 Historical Performance"},
        {"key": "product_overview", "label": "✨ Product Overview"},
        {"key": "derivatives_data", "label": "💾 Data Source"},
    ]
    tab_map = {t["key"]: t["label"] for t in tabs}
    requested_tab = st.query_params.get("tab")
    if requested_tab in tab_map:
        st.session_state["active_tab_key"] = requested_tab
        st.query_params.pop("tab", None)
    st.session_state.setdefault("active_tab_key", "login")
    if st.session_state["active_tab_key"] not in tab_map:
        st.session_state["active_tab_key"] = "login"

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background: #14171f;
        }
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    visible_keys = ["portfolio", "equities", "risk_buckets", "pre_trade", "market_regime"]
    overflow_keys = [t["key"] for t in tabs if t["key"] not in visible_keys]
    overflow_labels = ["⋯"] + [tab_map[key] for key in overflow_keys]
    if st.session_state["active_tab_key"] in overflow_keys:
        overflow_default = tab_map[st.session_state["active_tab_key"]]
    else:
        overflow_default = "⋯"

    def set_active_tab(tab_key: str) -> None:
        st.session_state["active_tab_key"] = tab_key

    def handle_nav_more_change() -> None:
        selection = st.session_state.get("nav_more", "⋯")
        if selection != "⋯":
            selected_key = next(
                (k for k, v in tab_map.items() if v == selection),
                None,
            )
            if selected_key:
                st.session_state["active_tab_key"] = selected_key

    st.markdown('<div id="gs-nav" class="gs-nav">', unsafe_allow_html=True)
    nav_cols = st.columns(6)
    for idx, key in enumerate(visible_keys):
        label = tab_map[key]
        is_active = st.session_state["active_tab_key"] == key
        if is_active:
            background = "#1b4fd8"
            text_color = "#ffffff"
            hover_bg = "#2d7dff"
        else:
            background = "#1f2a3d"
            text_color = "#dbe6ff"
            hover_bg = "#2a3547"

        with nav_cols[idx]:
            with stylable_container(
                key=f"nav_btn_{key}",
                css_styles=f"""
                    button {{
                        width: 100%;
                        border-radius: 999px;
                        padding: 0 rem 0 arem;
                        background: {background} !important;
                        color: {text_color} !important;
                        border: 0 !important;
                        font-weight: 700;
                        font-size: 0.75rem;
                        transition: background 150ms ease, transform 150ms ease;
                        white-space: nowrap;
                        box-shadow: {"0 0 0 2px #2d7dff inset" if is_active else "none"};
                    }}
                    button:hover {{
                        background: {hover_bg} !important;
                        transform: translateY(-1px);
                    }}
                """,
            ):
                st.button(label, key=f"nav_{key}", on_click=set_active_tab, args=(key,))
    with nav_cols[5]:
        st.selectbox(
            "More",
            overflow_labels,
            index=overflow_labels.index(overflow_default),
            key="nav_more",
            label_visibility="collapsed",
            on_change=handle_nav_more_change,
        )
    st.markdown("</div>", unsafe_allow_html=True)
    active_key = st.session_state["active_tab_key"]

    if active_key == "login":
        st.markdown("---")
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
        st.info(
            f"⚠️ Derivatives cache missing or expired for: {missing_str}. "
            "Please load fresh data from the Derivatives Data tab."
        )

    if active_key == "portfolio":
        st.markdown("---")
        render_portfolio_dashboard_tab()
    elif active_key == "equities":
        st.markdown("---")
        render_equities_tab()
    elif active_key == "risk_buckets":
        st.markdown("---")
        render_risk_buckets_tab()
    elif active_key == "pre_trade":
        subtab = st.radio(
            "Pre-Trade Tool",
            ["Stress Testing", "Trade Selector"],
            horizontal=True,
            label_visibility="collapsed",
        )
        if subtab == "Stress Testing":
            st.markdown("---")
            render_pre_trade_analysis_tab()
        else:
            st.markdown("---")
            render_trade_selector_tab()
    elif active_key == "market_regime":
        st.markdown("---")
        render_market_regime_tab()
    elif active_key == "historical":
        st.markdown("---")
        render_historical_performance_tab()
    elif active_key == "product_overview":
        st.markdown("---")
        render_product_overview_tab()
    elif active_key == "derivatives_data":
        st.markdown("---")
        render_derivatives_data_tab()

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
