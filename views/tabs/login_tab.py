"""Login tab for Kite Connect authentication."""

import os
import webbrowser
import streamlit as st
from pathlib import Path
import json

# Path to credentials file
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = Path(ROOT) / "database" / "derivatives_cache"
KITE_CREDS_FILE = CACHE_DIR / "kite_credentials.json"


def clear_kite_credentials():
    """Clear saved Kite credentials file."""
    try:
        if KITE_CREDS_FILE.exists():
            KITE_CREDS_FILE.unlink()
    except Exception as e:
        st.error(f"Failed to clear credentials: {e}")


def render_login_tab():
    """Render the login tab."""
    st.subheader("Login")
    
    default_api_key = st.session_state.get("kite_api_key_stored", "") or os.getenv("KITE_API_KEY", "")
    default_api_secret = st.session_state.get("kite_api_secret_stored", "") or os.getenv("KITE_API_SECRET", "")
    
    api_key = st.text_input("API Key", value=default_api_key, key="kite_api_key_input")
    api_secret = st.text_input("API Secret", value=default_api_secret, type="password", key="kite_api_secret_input")
    
    st.caption("⚠️ **Important**: Your Kite app redirect URI must match your Streamlit URL (e.g., http://127.0.0.1:5174/)")
    
    host = st.text_input("Redirect host", value="127.0.0.1", key="kite_redirect_host")
    port = st.number_input("Redirect port", value=5174, min_value=1024, max_value=65535, key="kite_redirect_port")

    if not api_key or not api_secret:
        st.info("Enter Kite API key and secret (or set KITE_API_KEY/KITE_API_SECRET env vars).")

    col1, col2 = st.columns([1, 1])
    login_clicked = col1.button("Login with Kite", disabled=st.session_state.kite_login_initiated)
    logout_clicked = col2.button("Logout (clear token)")

    if logout_clicked:
        st.session_state.pop("kite_access_token", None)
        st.session_state.pop("kite_api_key", None)
        st.session_state.pop("kite_token_timestamp", None)
        st.session_state.kite_login_initiated = False
        st.session_state.kite_processing_token = False
        
        # Clear persistent credentials file
        clear_kite_credentials()
        
        st.success("Logged out and cleared saved credentials")
        st.rerun()

    if login_clicked and api_key and api_secret:
        st.session_state["kite_api_key_stored"] = api_key
        st.session_state["kite_api_secret_stored"] = api_secret
        st.session_state.kite_login_initiated = True
        
        redirect_uri = f"http://{host}:{port}/"
        login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}&redirect_params=status%3Dlogin&redirect_uri={redirect_uri}"
        
        st.info(f"🔗 Opening Kite login in your browser. After login, you'll be redirected back here.")
        
        webbrowser.open(login_url)
        
        st.warning("⏳ Complete the login in the browser window. This page will update automatically after redirect.")

    if st.session_state.get("kite_access_token"):
        st.success("✅ You are logged in. Other dashboard tabs are now available.")
    elif st.session_state.kite_login_initiated:
        st.info("⏳ Waiting for login completion. Complete the login in your browser.")
