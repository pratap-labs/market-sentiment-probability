"""Greeks debug tab for manual spot-driven recalculation."""

import os
import sys
from typing import Dict, List

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.utils import enrich_position_with_greeks


def _recompute_positions(positions: List[Dict], spot: float) -> List[Dict]:
    st.session_state["greeks_spot_override"] = spot
    refreshed = []
    for pos in positions:
        refreshed.append(enrich_position_with_greeks(pos, pd.DataFrame(), spot))
    return refreshed


def render_greeks_debug_tab():
    """Render a debug-only greeks recalculation panel."""
    st.subheader("🧪 Greeks Debug")
    st.caption("Recompute greeks using manual spot and last_price only. No auto-enrichment.")

    positions = st.session_state.get("enriched_positions", [])
    if not positions:
        st.info("Load positions first to debug greeks.")
        return

    spot = st.sidebar.number_input(
        "Spot (NIFTY)",
        value=float(st.session_state.get("current_spot", 25000)),
        step=10.0,
        format="%.2f",
        key="greeks_debug_spot",
    )

    if st.sidebar.button("Recompute Greeks", type="primary", key="greeks_debug_recompute"):
        st.session_state["debug_enriched_positions"] = _recompute_positions(positions, spot)
        st.success("Greeks recomputed using spot override.")

    debug_positions = st.session_state.get("debug_enriched_positions", [])
    if not debug_positions:
        st.info("Click 'Recompute Greeks' to populate debug values.")
        return

    symbols = sorted({pos.get("tradingsymbol") for pos in debug_positions if pos.get("tradingsymbol")})
    selected_symbol = st.sidebar.selectbox("Symbol", symbols, key="greeks_debug_symbol")
    match = next((p for p in debug_positions if p.get("tradingsymbol") == selected_symbol), None)
    if match:
        debug = match.get("iv_debug", {})
        debug_payload = {
            "spot_used": debug.get("spot_used"),
            "option_price_used": debug.get("option_price_used"),
            "time_to_expiry_years": debug.get("time_to_expiry"),
            "time_to_expiry_days": (
                float(debug.get("time_to_expiry", 0.0)) * 365.0
                if debug.get("time_to_expiry") is not None
                else None
            ),
            "expiry_date": debug.get("expiry_date"),
            "implied_vol_pct": (
                float(match.get("implied_vol")) * 100.0
                if match.get("implied_vol") is not None
                else None
            ),
            "last_price": match.get("last_price"),
        }
        st.json(debug_payload)

    display_cols = [
        "tradingsymbol", "quantity", "strike", "option_type", "expiry", "dte",
        "last_price", "implied_vol", "delta", "gamma", "vega", "theta",
        "position_delta", "position_gamma", "position_vega", "position_theta",
    ]
    df = pd.DataFrame([{col: pos.get(col) for col in display_cols} for pos in debug_positions])
    expiry_values = sorted(
        {
            exp.strftime("%Y-%m-%d")
            for exp in df["expiry"].dropna()
            if hasattr(exp, "strftime")
        }
    )
    expiry_options = ["All"] + expiry_values
    selected_expiry = st.sidebar.selectbox("Filter by Expiry", expiry_options, key="greeks_debug_expiry")
    if selected_expiry != "All":
        df = df[
            df["expiry"].apply(
                lambda exp: exp.strftime("%Y-%m-%d") if hasattr(exp, "strftime") else str(exp)
            )
            == selected_expiry
        ]
    st.dataframe(df, use_container_width=True)
