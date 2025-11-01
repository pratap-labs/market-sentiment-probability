"""Positions tab for displaying current options positions."""

import streamlit as st
import pandas as pd

try:
    from kiteconnect import KiteConnect
except Exception:
    KiteConnect = None

import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from views.utils import enrich_position_with_greeks, calculate_market_regime


def render_positions_tab(options_df: pd.DataFrame, nifty_df: pd.DataFrame):
    """Render the positions tab with enriched data."""
    st.subheader("Positions")
    access_token = st.session_state.get("kite_access_token")
    
    if not access_token:
        st.warning("Not logged in. Please go to the Login tab and sign in first.")
        return
    
    kite_api_key = st.session_state.get("kite_api_key")
    if not kite_api_key:
        st.error("API key not found in session. Please log in again.")
        return
        
    kite = KiteConnect(api_key=kite_api_key)
    kite.set_access_token(access_token)

    if st.button("🔄 Fetch Latest Positions", type="primary"):
        try:
            with st.spinner("Fetching positions..."):
                positions = kite.positions()
                net_positions = positions.get("net", [])

                if not net_positions:
                    st.info("No positions returned")
                    return
                
                # Get current spot price
                market_regime = calculate_market_regime(options_df, nifty_df)
                current_spot = market_regime.get("current_spot", 25000)
                
                # Enrich positions with Greeks
                enriched = []
                for pos in net_positions:
                    enriched_pos = enrich_position_with_greeks(pos, options_df, current_spot)
                    enriched.append(enriched_pos)
                
                # Store in session state
                st.session_state["enriched_positions"] = enriched
                st.session_state["current_spot"] = current_spot
                
                st.success(f"✅ Loaded {len(enriched)} positions")
                st.info("💡 **Tip:** Switch to the **Overview** tab to see your complete portfolio analysis and recommendations.")

        except Exception as e:
            st.error(f"Failed to fetch positions: {e}")
            if "403" in str(e) or "Invalid" in str(e):
                st.warning("Your session may have expired. Please log in again.")
                st.session_state.pop("kite_access_token", None)
            return
    
    # Display enriched positions if available
    if "enriched_positions" in st.session_state:
        enriched = st.session_state["enriched_positions"]
        
        if enriched:
            # Create display dataframe
            display_cols = [
                "tradingsymbol", "quantity", "strike", "option_type", "dte",
                "last_price", "pnl", "implied_vol",
                "delta", "gamma", "vega", "theta",
                "position_delta", "position_gamma", "position_vega", "position_theta"
            ]
            
            display_data = []
            for pos in enriched:
                row = {col: pos.get(col, None) for col in display_cols}
                display_data.append(row)
            
            df = pd.DataFrame(display_data)
            
            # Format display
            if not df.empty:
                st.dataframe(
                    df.style.format({
                        "implied_vol": "{:.2%}",
                        "delta": "{:.3f}",
                        "gamma": "{:.4f}",
                        "vega": "{:.2f}",
                        "theta": "{:.2f}",
                        "position_delta": "{:.2f}",
                        "position_gamma": "{:.3f}",
                        "position_vega": "{:.1f}",
                        "position_theta": "{:.1f}",
                        "last_price": "{:.2f}",
                        "pnl": "{:.2f}"
                    }),
                    use_container_width=True
                )
