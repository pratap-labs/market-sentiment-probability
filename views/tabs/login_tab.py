"""Login tab for Kite Connect authentication."""

import os
import streamlit as st
from pathlib import Path
import streamlit.components.v1 as components

# Path to credentials file
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = Path(ROOT) / "database" / "derivatives_cache"
KITE_CREDS_FILE = CACHE_DIR / "kite_credentials.json"

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


def clear_kite_credentials():
    """Clear saved Kite credentials file."""
    try:
        if KITE_CREDS_FILE.exists():
            KITE_CREDS_FILE.unlink()
    except Exception as e:
        st.error(f"Failed to clear credentials: {e}")


def render_login_tab():
    """Render the login tab."""

    load_env_file(Path(ROOT) / ".env")
    if not st.session_state.get("kite_api_key_stored") and os.getenv("KITE_API_KEY"):
        st.session_state["kite_api_key_stored"] = os.getenv("KITE_API_KEY")
    if not st.session_state.get("kite_api_secret_stored") and os.getenv("KITE_API_SECRET"):
        st.session_state["kite_api_secret_stored"] = os.getenv("KITE_API_SECRET")

    st.session_state.setdefault("kite_login_initiated", False)
    st.session_state.setdefault("kite_redirect_pending", False)
    st.session_state.setdefault("kite_login_url", "")

    host = st.sidebar.text_input(
        "Redirect host",
        value="127.0.0.1",
        key="kite_redirect_host",
        disabled=True,
    )
    port = st.sidebar.number_input(
        "Redirect port",
        value=5174,
        min_value=1024,
        max_value=65535,
        key="kite_redirect_port",
        disabled=True,
    )

    st.sidebar.markdown("**Kite credentials**")
    api_key = st.sidebar.text_input(
        "API Key",
        value=st.session_state.get("kite_api_key_stored", "") or "",
        key="kite_api_key_input",
        disabled=True,
    )
    api_secret = st.sidebar.text_input(
        "API Secret",
        value=st.session_state.get("kite_api_secret_stored", "") or "",
        type="password",
        key="kite_api_secret_input",
        disabled=True,
    )
    if not api_key or not api_secret:
        st.info("Enter Kite API key/secret (or set env vars).")
    login_clicked = st.sidebar.button(
        "Login with Kite",
        disabled=st.session_state.get("kite_login_initiated", False),
        key="kite_login_button",
    )

    logout_clicked = st.sidebar.button("Logout (clear token)", disabled=True)
    force_logout_clicked = st.sidebar.button("Force logout (clear all)", disabled=True)

    if logout_clicked:
        st.session_state.pop("kite_access_token", None)
        st.session_state.pop("kite_api_key", None)
        st.session_state.pop("kite_token_timestamp", None)
        st.session_state["kite_login_initiated"] = False
        st.session_state.kite_processing_token = False
        st.query_params.clear()
        
        # Clear persistent credentials file
        clear_kite_credentials()
        
        st.success("Logged out and cleared saved credentials")
        st.rerun()

    if force_logout_clicked:
        st.session_state["kite_force_logout"] = True
        for key in (
            "kite_access_token",
            "kite_api_key",
            "kite_token_timestamp",
            "kite_login_initiated",
            "kite_processing_token",
            "kite_api_key_stored",
            "kite_api_secret_stored",
            "kite_access_token",
            "kite_api_key",
            "kite_token_timestamp",
            "kite_login_url",
            "kite_redirect_pending",
        ):
            st.session_state.pop(key, None)
        st.query_params.clear()
        clear_kite_credentials()
        st.success("Force logged out and cleared all Kite state")
        st.rerun()

    def _reset_session_for_new_login() -> None:
        # Reset cached data so the new account starts clean.
        for key in (
            "enriched_positions",
            "debug_enriched_positions",
            "options_df_cache",
            "options_cache_meta",
            "tba_saved_trades",
            "tba_use_saved_groups",
            "tba_trade_drawdown_df",
            "tba_trade_group_drawdown_df",
            "trade_selector_results",
            "trade_selector_inputs",
        ):
            st.session_state.pop(key, None)
        st.session_state.pop("kite_access_token", None)
        st.session_state.pop("kite_api_key", None)
        st.session_state.pop("kite_token_timestamp", None)
        clear_kite_credentials()

    def handle_login_click(api_key: str, api_secret: str) -> None:
        if not api_key or not api_secret:
            return
        _reset_session_for_new_login()
        st.session_state["kite_api_key_stored"] = api_key
        st.session_state["kite_api_secret_stored"] = api_secret
        st.session_state["kite_login_initiated"] = True
        st.session_state["kite_redirect_pending"] = True

        redirect_uri = f"http://{host}:{port}/"
        login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}&redirect_params=status%3Dlogin&redirect_uri={redirect_uri}"
        st.session_state["kite_login_url"] = login_url

    if login_clicked:
        handle_login_click(api_key, api_secret)

    if st.session_state.get("kite_login_initiated") and not st.session_state.get("kite_access_token"):
        login_url = st.session_state.get("kite_login_url")
        if st.session_state.get("kite_redirect_pending") and login_url:
            st.info("🔗 Redirecting to Kite login in this tab. After login, you'll be sent back here.")
            st.warning("⏳ Complete the login in the same browser tab. This page will update automatically after redirect.")
            components.html(
                f"""
                <script>
                window.top.location.href = "{login_url}";
                </script>
                """,
                height=0,
            )
            st.session_state["kite_redirect_pending"] = False
        if login_url:
            st.markdown(
                f"""
                <div style="margin-top: 0.5rem;">
                  <a href="{login_url}" target="_self"
                     style="display:inline-block;padding:0.4rem 0.8rem;border:1px solid #999;border-radius:6px;text-decoration:none;">
                     Open Kite login (same tab)
                  </a>
                </div>
                """,
                unsafe_allow_html=True,
            )

    logged_in = bool(st.session_state.get("kite_access_token"))
    if logged_in:
        st.success("✅ Account logged in.")
    if st.session_state.get("kite_login_initiated") and not logged_in:
        st.info("⏳ Waiting for login completion. Complete the login in your browser.")
