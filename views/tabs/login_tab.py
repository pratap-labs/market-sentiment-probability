"""Login tab for Kite Connect authentication."""

import os
import streamlit as st
from pathlib import Path
import streamlit.components.v1 as components

# Path to credentials file
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = Path(ROOT) / "database" / "derivatives_cache"
KITE_CREDS_FILE = CACHE_DIR / "kite_credentials.json"
KITE_CREDS_FILE_2 = CACHE_DIR / "kite_credentials_2.json"

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


def clear_kite_credentials(account: str = KITE_ACCOUNT_PRIMARY):
    """Clear saved Kite credentials file."""
    try:
        creds_file = _account_creds_file(account)
        if creds_file.exists():
            creds_file.unlink()
    except Exception as e:
        st.error(f"Failed to clear credentials: {e}")


def _apply_active_account(account: str) -> None:
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
        st.session_state.pop("kite_access_token", None)
        st.session_state.pop("kite_api_key", None)
        st.session_state.pop("kite_token_timestamp", None)


def render_login_tab():
    """Render the login tab."""
    st.subheader("Login")

    load_env_file(Path(ROOT) / ".env")
    for account in (KITE_ACCOUNT_PRIMARY, KITE_ACCOUNT_SECONDARY):
        key_key = _account_session_key("kite_api_key_stored", account)
        secret_key = _account_session_key("kite_api_secret_stored", account)
        env_key = _account_env_key(account, "KITE_API_KEY")
        env_secret = _account_env_key(account, "KITE_API_SECRET")
        if not st.session_state.get(key_key) and os.getenv(env_key):
            st.session_state[key_key] = os.getenv(env_key)
        if not st.session_state.get(secret_key) and os.getenv(env_secret):
            st.session_state[secret_key] = os.getenv(env_secret)

    for account in (KITE_ACCOUNT_PRIMARY, KITE_ACCOUNT_SECONDARY):
        st.session_state.setdefault(_account_session_key("kite_login_initiated", account), False)
        st.session_state.setdefault(_account_session_key("kite_redirect_pending", account), False)
        st.session_state.setdefault(_account_session_key("kite_login_url", account), "")
    st.session_state.setdefault("kite_active_account", KITE_ACCOUNT_PRIMARY)

    active_account = st.sidebar.selectbox(
        "Active account",
        [KITE_ACCOUNT_PRIMARY, KITE_ACCOUNT_SECONDARY],
        format_func=lambda value: "Primary (KITE_API_KEY)" if value == KITE_ACCOUNT_PRIMARY else "Secondary (KITE_API_KEY_2)",
        key="kite_active_account",
    )
    _apply_active_account(active_account)

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

    def render_account_controls(account: str):
        label = "Primary" if account == KITE_ACCOUNT_PRIMARY else "Secondary"
        default_api_key = st.session_state.get(_account_session_key("kite_api_key_stored", account), "") or os.getenv(
            _account_env_key(account, "KITE_API_KEY"), ""
        )
        default_api_secret = st.session_state.get(
            _account_session_key("kite_api_secret_stored", account), ""
        ) or os.getenv(_account_env_key(account, "KITE_API_SECRET"), "")
        st.sidebar.markdown(f"**{label} credentials**")
        api_key = st.sidebar.text_input(
            "API Key",
            value=default_api_key,
            key=_account_session_key("kite_api_key_input", account),
            disabled=True,
        )
        api_secret = st.sidebar.text_input(
            "API Secret",
            value=default_api_secret,
            type="password",
            key=_account_session_key("kite_api_secret_input", account),
            disabled=True,
        )
        if not api_key or not api_secret:
            st.info(f"Enter Kite API key/secret for {label} (or set env vars).")
        login_clicked = st.sidebar.button(
            f"Login with Kite ({label})",
            disabled=st.session_state.get(_account_session_key("kite_login_initiated", account), False),
            key=_account_session_key("kite_login_button", account),
        )
        return login_clicked, api_key, api_secret

    login_clicked_primary, api_key_primary, api_secret_primary = render_account_controls(KITE_ACCOUNT_PRIMARY)
    login_clicked_secondary, api_key_secondary, api_secret_secondary = render_account_controls(KITE_ACCOUNT_SECONDARY)

    logout_clicked = st.sidebar.button("Logout (clear token)", disabled=True)
    force_logout_clicked = st.sidebar.button("Force logout (clear all)", disabled=True)

    if logout_clicked:
        st.session_state.pop("kite_access_token", None)
        st.session_state.pop("kite_api_key", None)
        st.session_state.pop("kite_token_timestamp", None)
        st.session_state[_account_session_key("kite_login_initiated", KITE_ACCOUNT_PRIMARY)] = False
        st.session_state[_account_session_key("kite_login_initiated", KITE_ACCOUNT_SECONDARY)] = False
        st.session_state.kite_processing_token = False
        st.query_params.clear()
        
        # Clear persistent credentials file
        clear_kite_credentials(KITE_ACCOUNT_PRIMARY)
        
        st.success("Logged out and cleared saved credentials")
        st.rerun()

    if force_logout_clicked:
        st.session_state["kite_force_logout"] = True
        for key in (
            "kite_access_token",
            "kite_api_key",
            "kite_token_timestamp",
            _account_session_key("kite_login_initiated", KITE_ACCOUNT_PRIMARY),
            _account_session_key("kite_login_initiated", KITE_ACCOUNT_SECONDARY),
            "kite_processing_token",
            _account_session_key("kite_api_key_stored", KITE_ACCOUNT_PRIMARY),
            _account_session_key("kite_api_secret_stored", KITE_ACCOUNT_PRIMARY),
            _account_session_key("kite_api_key_stored", KITE_ACCOUNT_SECONDARY),
            _account_session_key("kite_api_secret_stored", KITE_ACCOUNT_SECONDARY),
            _account_session_key("kite_access_token", KITE_ACCOUNT_PRIMARY),
            _account_session_key("kite_api_key", KITE_ACCOUNT_PRIMARY),
            _account_session_key("kite_token_timestamp", KITE_ACCOUNT_PRIMARY),
            _account_session_key("kite_access_token", KITE_ACCOUNT_SECONDARY),
            _account_session_key("kite_api_key", KITE_ACCOUNT_SECONDARY),
            _account_session_key("kite_token_timestamp", KITE_ACCOUNT_SECONDARY),
            "kite_active_account",
        ):
            st.session_state.pop(key, None)
        st.query_params.clear()
        clear_kite_credentials(KITE_ACCOUNT_PRIMARY)
        clear_kite_credentials(KITE_ACCOUNT_SECONDARY)
        st.success("Force logged out and cleared all Kite state")
        st.rerun()

    def handle_login_click(account: str, api_key: str, api_secret: str) -> None:
        if not api_key or not api_secret:
            return
        st.session_state[_account_session_key("kite_api_key_stored", account)] = api_key
        st.session_state[_account_session_key("kite_api_secret_stored", account)] = api_secret
        st.session_state[_account_session_key("kite_login_initiated", account)] = True
        st.session_state[_account_session_key("kite_redirect_pending", account)] = True
        st.session_state["kite_login_account"] = account

        redirect_uri = f"http://{host}:{port}/"
        login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}&redirect_params=status%3Dlogin&redirect_uri={redirect_uri}"
        st.session_state[_account_session_key("kite_login_url", account)] = login_url

    if login_clicked_primary:
        handle_login_click(KITE_ACCOUNT_PRIMARY, api_key_primary, api_secret_primary)
    if login_clicked_secondary:
        handle_login_click(KITE_ACCOUNT_SECONDARY, api_key_secondary, api_secret_secondary)
        
    for account in (KITE_ACCOUNT_PRIMARY, KITE_ACCOUNT_SECONDARY):
        initiated_key = _account_session_key("kite_login_initiated", account)
        access_token_key = _account_session_key("kite_access_token", account)
        login_url_key = _account_session_key("kite_login_url", account)
        redirect_pending_key = _account_session_key("kite_redirect_pending", account)
        if st.session_state.get(initiated_key) and not st.session_state.get(access_token_key):
            login_url = st.session_state.get(login_url_key)
            if st.session_state.get(redirect_pending_key) and login_url:
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
                st.session_state[redirect_pending_key] = False
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

    primary_logged_in = bool(st.session_state.get(_account_session_key("kite_access_token", KITE_ACCOUNT_PRIMARY)))
    secondary_logged_in = bool(st.session_state.get(_account_session_key("kite_access_token", KITE_ACCOUNT_SECONDARY)))
    if primary_logged_in:
        st.success("✅ Primary account logged in.")
    if secondary_logged_in:
        st.success("✅ Secondary account logged in.")
    if st.session_state.get(_account_session_key("kite_login_initiated", KITE_ACCOUNT_PRIMARY)) and not primary_logged_in:
        st.info("⏳ Waiting for primary login completion. Complete the login in your browser.")
    if st.session_state.get(_account_session_key("kite_login_initiated", KITE_ACCOUNT_SECONDARY)) and not secondary_logged_in:
        st.info("⏳ Waiting for secondary login completion. Complete the login in your browser.")
