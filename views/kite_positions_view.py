"""
Streamlit view for logging in with Kite Connect and fetching positions.

Place this file under `views/` so it can be imported by the app hub.

Behavior:
 - If not logged in (no `st.session_state['kite_access_token']`) show login UI.
 - User provides API key/secret (or use env vars), clicks Login -> browser opens Kite login URL.
 - Kite redirects back to Streamlit app with `request_token` in URL.
 - The app exchanges `request_token` for an access token and stores it.
 - After login the page shows a "Fetch latest positions" button that retrieves and displays positions.

Notes:
 - This view expects the developer to register the redirect URI in the Kite app settings 
   (e.g. http://127.0.0.1:5174/ - match your Streamlit port).
 - The redirect target should be the Streamlit app itself.
"""

import webbrowser
import os
import streamlit as st
import pandas as pd
from typing import Optional

try:
    from kiteconnect import KiteConnect
except Exception:
    KiteConnect = None


def render():
    """Main entrypoint for the Streamlit view."""
    st.header("Kite Connect — Positions")

    if KiteConnect is None:
        st.error("Missing dependency `kiteconnect`. Install with `pip install kiteconnect`.")
        return

    # Initialize session state
    if "kite_login_initiated" not in st.session_state:
        st.session_state.kite_login_initiated = False
    if "kite_processing_token" not in st.session_state:
        st.session_state.kite_processing_token = False

    # Check for request_token in URL (from Kite redirect)
    query_params = st.query_params
    incoming_request_token = query_params.get("request_token", None)

    # If we have a request_token and haven't processed it yet, exchange it
    if incoming_request_token and not st.session_state.kite_processing_token:
        st.session_state.kite_processing_token = True
        
        # Read from env vars OR previously stored values
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
                        
                        # Clear the request_token from URL
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

    # Two tabs: Login and Positions
    tab_login, tab_positions = st.tabs(["Login", "Positions"])

    # --- Login tab ---
    with tab_login:
        st.subheader("Login")
        
        # Get default values from env or session state
        default_api_key = st.session_state.get("kite_api_key_stored", "") or os.getenv("KITE_API_KEY", "")
        default_api_secret = st.session_state.get("kite_api_secret_stored", "") or os.getenv("KITE_API_SECRET", "")
        
        api_key = st.text_input("API Key", value=default_api_key, key="kite_api_key_input")
        api_secret = st.text_input("API Secret", value=default_api_secret, type="password", key="kite_api_secret_input")
        
        # Get current Streamlit host and port from browser URL
        st.caption("⚠️ **Important**: Your Kite app redirect URI must match your Streamlit URL (e.g., http://127.0.0.1:5174/)")
        
        host = st.text_input("Redirect host", value="127.0.0.1", key="kite_redirect_host", 
                            help="Should match the host where Streamlit is running")
        port = st.number_input("Redirect port", value=5174, min_value=1024, max_value=65535, 
                              key="kite_redirect_port",
                              help="Should match the port where Streamlit is running")

        if not api_key or not api_secret:
            st.info("Enter Kite API key and secret (or set KITE_API_KEY/KITE_API_SECRET env vars).")

        col1, col2 = st.columns([1, 1])
        login_clicked = col1.button("Login with Kite", disabled=st.session_state.kite_login_initiated)
        logout_clicked = col2.button("Logout (clear token)")

        if logout_clicked:
            st.session_state.pop("kite_access_token", None)
            st.session_state.pop("kite_api_key", None)
            st.session_state.kite_login_initiated = False
            st.session_state.kite_processing_token = False
            st.success("Logged out")
            st.rerun()

        if login_clicked and api_key and api_secret:
            # Store credentials in session state for later use
            st.session_state["kite_api_key_stored"] = api_key
            st.session_state["kite_api_secret_stored"] = api_secret
            st.session_state.kite_login_initiated = True
            
            # Build login URL
            redirect_uri = f"http://{host}:{port}/"
            login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}&redirect_params=status%3Dlogin&redirect_uri={redirect_uri}"
            
            st.info(f"🔗 Opening Kite login in your browser. After login, you'll be redirected back here.")
            st.info(f"📍 Redirect URI: {redirect_uri}")
            
            # Open browser
            webbrowser.open(login_url)
            
            st.warning("⏳ Complete the login in the browser window. This page will update automatically after redirect.")

        # Show current login status
        if st.session_state.get("kite_access_token"):
            st.success("✅ You are logged in. Switch to the Positions tab to fetch positions.")
        elif st.session_state.kite_login_initiated:
            st.info("⏳ Waiting for login completion. Complete the login in your browser.")

    # --- Positions tab ---
    with tab_positions:
        st.subheader("Positions")
        access_token = st.session_state.get("kite_access_token")
        
        if not access_token:
            st.warning("Not logged in. Please go to the Login tab and sign in first.")
        else:
            kite_api_key = st.session_state.get("kite_api_key")
            
            if not kite_api_key:
                st.error("API key not found in session. Please log in again.")
                return
                
            kite = KiteConnect(api_key=kite_api_key)
            kite.set_access_token(access_token)

            if st.button("Fetch latest positions"):
                try:
                    with st.spinner("Fetching positions..."):
                        positions = kite.positions()
                        net = positions.get("net", [])
                        day = positions.get("day", [])

                        if net:
                            st.subheader("Net positions")
                            st.dataframe(pd.DataFrame(net), use_container_width=True)

                        if day:
                            st.subheader("Day positions")
                            st.dataframe(pd.DataFrame(day), use_container_width=True)

                        if not net and not day:
                            st.info("No positions returned (empty list)")

                except Exception as e:
                    st.error(f"Failed to fetch positions: {e}")
                    if "403" in str(e) or "Invalid" in str(e):
                        st.warning("Your session may have expired. Please log in again.")
                        st.session_state.pop("kite_access_token", None)


if __name__ == "__main__":
    render()