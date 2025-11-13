"""
Streamlit helper to login to Kite (Zerodha) and fetch latest positions.

Usage:
  pip install kiteconnect streamlit
  export KITE_API_KEY=your_api_key
  export KITE_API_SECRET=your_api_secret
  streamlit run scripts/kite_positions.py

This script provides a minimal OAuth-like flow:
 - User supplies API key/secret (or set as env vars)
 - The app opens the Kite login URL in the browser
 - A small local HTTP server captures the redirect containing `request_token`
 - We exchange request_token + api_secret for an access_token via KiteConnect
 - Then we call `kite.positions()` and show the latest positions

Note: You must have a registered Kite Connect API app and allowed redirect URI set to
http://127.0.0.1:8080/ (or whichever host/port you choose).
"""

import os
import threading
import webbrowser
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import streamlit as st
import pandas as pd

try:
    from kiteconnect import KiteConnect
except Exception:
    KiteConnect = None


class _TokenHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler to capture `request_token` from redirect."""

    server_version = "KiteTokenServer/0.1"

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        req_tokens = qs.get("request_token") or qs.get("request_token\r\n")
        if req_tokens:
            token = req_tokens[0]
            # store on the server object for the waiting thread
            self.server.request_token = token
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h3>Login successful. You can close this window and return to the app.</h3></body></html>")
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Missing request_token")

    def log_message(self, format, *args):
        # silence HTTP server logging in Streamlit
        return


def _start_token_server(host: str, port: int, timeout: int = 120):
    """Start a temporary HTTP server to capture request_token.

    Returns the request_token or raises TimeoutError.
    """
    server = HTTPServer((host, port), _TokenHandler)
    server.request_token = None

    def run_server():
        try:
            server.handle_request()  # handle a single request then exit
        except Exception:
            pass

    thr = threading.Thread(target=run_server, daemon=True)
    thr.start()

    # wait for token to be populated by handler
    waited = 0
    poll = 0.5
    while waited < timeout:
        if getattr(server, "request_token", None):
            return server.request_token
        threading.Event().wait(poll)
        waited += poll

    raise TimeoutError("Timed out waiting for request_token on redirect")


def main():
    st.title("Kite Positions — Login & Fetch")

    if KiteConnect is None:
        st.error("Missing dependency: install with `pip install kiteconnect`")
        st.stop()

    api_key = st.text_input("Kite API Key", value=os.getenv("KITE_API_KEY", ""))
    api_secret = st.text_input("Kite API Secret", value=os.getenv("KITE_API_SECRET", ""), type="password")
    redirect_host = st.text_input("Redirect host (local)", value="127.0.0.1")
    redirect_port = st.number_input("Redirect port", value=5173, min_value=1024, max_value=65535)

    if not api_key or not api_secret:
        st.info("Enter your Kite API key and secret (or set KITE_API_KEY/KITE_API_SECRET env vars).")

    col1, col2 = st.columns([1, 1])
    login_clicked = col1.button("Login with Kite")
    clear_token = col2.button("Clear saved token")

    if clear_token:
        st.session_state.pop("kite_access_token", None)
        st.success("Cleared saved token")

    if login_clicked:
        try:
            kite = KiteConnect(api_key=api_key)
            # build login URL (Kite Connect expects redirect_uri registered in app)
            redirect_uri = f"http://{redirect_host}:{redirect_port}/"
            login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}&redirect_uri={redirect_uri}"

            st.info("Opening browser for Kite login — please complete login and allow redirect.")
            webbrowser.open(login_url)

            # start local server to capture request_token -- blocking wait
            with st.spinner("Waiting for login redirect and request_token..."):
                request_token = _start_token_server(redirect_host, int(redirect_port), timeout=180)

            st.success("Received request_token — exchanging for access token")
            # exchange for session (access token)
            data = kite.generate_session(request_token, api_secret=api_secret)
            access_token = data.get("access_token")
            if not access_token:
                st.error(f"Failed to obtain access token: {data}")
                return

            # persist in session_state for this Streamlit session
            st.session_state["kite_access_token"] = access_token
            st.success("Login complete — access token stored in session")

        except Exception as e:
            st.error(f"Login flow failed: {e}")

    # If we have an access token, allow fetching positions
    access_token = st.session_state.get("kite_access_token")
    if access_token:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)

        if st.button("Fetch latest positions"):
            try:
                positions = kite.positions()
                # positions is a dict with 'net' and 'day' depending; show both
                net = positions.get("net", [])
                day = positions.get("day", [])

                if net:
                    df_net = pd.DataFrame(net)
                    st.subheader("Net positions")
                    st.dataframe(df_net)

                if day:
                    df_day = pd.DataFrame(day)
                    st.subheader("Day positions")
                    st.dataframe(df_day)

                if not net and not day:
                    st.info("No positions returned by Kite API (empty list)")

            except Exception as e:
                st.error(f"Failed to fetch positions: {e}")


if __name__ == "__main__":
    main()
