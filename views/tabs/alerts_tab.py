"""Risk alerts tab."""

import streamlit as st

import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from views.utils import calculate_portfolio_greeks


def render_alerts_tab():
    """Render risk alerts tab."""
    st.subheader("⚠️ Risk Alerts")
    
    if "enriched_positions" not in st.session_state:
        st.info("No positions loaded. Fetch positions from the Positions tab first.")
        return
    
    enriched = st.session_state["enriched_positions"]
    portfolio_greeks = calculate_portfolio_greeks(enriched)
    
    # Delta alert
    delta = portfolio_greeks["net_delta"]
    if abs(delta) > 15:
        st.error(f"🔴 High Delta Alert: Net Delta is {delta:.2f}. Consider hedging your portfolio.")
    elif abs(delta) > 8:
        st.warning(f"🟡 Moderate Delta Alert: Net Delta is {delta:.2f}. Monitor your exposure.")
    else:
        st.success(f"🟢 Delta is within safe limits: {delta:.2f}.")
    
    # Additional alerts can be added here
