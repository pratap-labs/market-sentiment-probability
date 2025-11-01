"""Position diagnostics tab with action signals."""

import streamlit as st
import pandas as pd
from typing import Dict, Tuple

try:
    from kiteconnect import KiteConnect
except Exception:
    KiteConnect = None

import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from views.utils import (
    format_inr,
    get_action_signal,
    get_action_recommendation
)


def render_diagnostics_tab():
    """Render position diagnostics tab."""
    st.subheader("🔍 Position Diagnostics")
    
    if "enriched_positions" not in st.session_state:
        st.info("No positions loaded. Fetch positions from the Positions tab first.")
        return
    
    enriched = st.session_state["enriched_positions"]
    current_spot = st.session_state.get("current_spot", 25000)
    
    # Get account size for % calculations
    access_token = st.session_state.get("kite_access_token")
    kite_api_key = st.session_state.get("kite_api_key")
    
    margin_available = 500000
    margin_used = 320000
    
    if access_token and kite_api_key:
        try:
            kite = KiteConnect(api_key=kite_api_key)
            kite.set_access_token(access_token)
            margins = kite.margins()
            equity_margins = margins.get("equity", {})
            margin_available = equity_margins.get("available", {}).get("live_balance", 500000)
            margin_used = equity_margins.get("utilised", {}).get("debits", 0)
        except:
            pass
    
    account_size = margin_available + margin_used
    
    # Sorting and filtering options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sort_by = st.selectbox(
            "Sort by",
            ["Days to B/E", "Theta Efficiency %", "PnL", "Position Size", "Delta", "Action Priority"]
        )
    
    with col2:
        show_red_only = st.checkbox("Show only RED positions")
    
    with col3:
        show_losing_only = st.checkbox("Show only losing positions")
    
    # Calculate diagnostics for each position
    diagnostics = []
    for pos in enriched:
        theta = pos.get("position_theta", 0)
        pnl = pos.get("pnl", 0)
        dte = pos.get("dte", 0)
        delta = pos.get("delta", 0)
        gamma = pos.get("gamma", 0)
        vega = pos.get("vega", 0)
        position_delta = pos.get("position_delta", 0)
        strike = pos.get("strike", 0)
        quantity = pos.get("quantity", 0)
        iv = pos.get("implied_vol", 0)
        last_price = pos.get("last_price", 0)
        buy_price = pos.get("buy_price", 0) or pos.get("sell_price", 0)
        
        # Days to breakeven
        days_to_breakeven = abs(pnl / theta) if theta != 0 else 999
        
        # Theta efficiency
        theta_eff = (pnl / theta * 100) if theta != 0 else 0
        
        # Notional value
        notional = abs(quantity) * strike  # 50 = lot size
        
        # Position size as % of portfolio
        position_pct = (notional / account_size * 100) if account_size > 0 else 0
        
        # Spot distance
        spot_distance_pct = ((current_spot - strike) / current_spot * 100) if strike > 0 else 0
        
        # Probability of profit (rough approximation using delta)
        prob_of_profit = (1 - abs(delta)) * 100 if delta else 50
        
        # Premium change
        premium_change_pct = ((last_price - buy_price) / buy_price * 100) if buy_price > 0 else 0
        
        # Loss vs credit ratio (for short positions)
        if quantity < 0:  # Short position
            credit_received = buy_price * abs(quantity) 
            loss_vs_credit = abs(pnl) / credit_received if credit_received > 0 else 0
        else:
            loss_vs_credit = 0
        
        # Action signal
        action_signal, action_priority = get_action_signal(
            dte, pnl, theta, position_delta, days_to_breakeven, loss_vs_credit
        )
        
        # Status color
        if days_to_breakeven > dte:
            status = "🔴 RED"
        elif days_to_breakeven > dte * 0.5:
            status = "🟡 YELLOW"
        else:
            status = "🟢 GREEN"
        
        diagnostics.append({
            # Identity
            "Symbol": pos.get("tradingsymbol", ""),
            "Strike": strike,
            "Type": pos.get("option_type", ""),
            "Qty": quantity,
            
            # P&L & Pricing
            "PnL": pnl,
            "Entry": buy_price,
            "Current": last_price,
            "Chg %": premium_change_pct,
            
            # Time
            "DTE": dte,
            "Days to B/E": days_to_breakeven,
            
            # Theta Analysis
            "Theta/Day": theta,
            "Theta Eff %": theta_eff,
            
            # Greeks
            "Delta": delta,
            "Pos Delta": position_delta,
            "Gamma": gamma,
            "Vega": vega,
            
            # Position Sizing
            "Notional": notional,
            "% Portfolio": position_pct,
            
            # Risk Metrics
            "Spot Dist %": spot_distance_pct,
            "PoP %": prob_of_profit,
            "IV": iv,
            "Loss/Credit": loss_vs_credit,
            
            # Action
            "Action": action_signal,
            "Priority": action_priority,
            "Status": status,
            
            # Hidden sort keys
            "_sort_action_priority": action_priority
        })
    
    # Apply filters
    filtered = diagnostics
    if show_red_only:
        filtered = [d for d in filtered if "🔴" in d["Status"]]
    if show_losing_only:
        filtered = [d for d in filtered if d["PnL"] < 0]
    
    # Apply sorting
    sort_key_map = {
        "Days to B/E": "Days to B/E",
        "Theta Efficiency %": "Theta Eff %",
        "PnL": "PnL",
        "Position Size": "Notional",
        "Delta": "Pos Delta",
        "Action Priority": "_sort_action_priority"
    }
    
    sort_key = sort_key_map.get(sort_by, "Days to B/E")
    filtered = sorted(filtered, key=lambda x: abs(x[sort_key]), reverse=True)
    
    # Remove internal sort keys from display
    for d in filtered:
        d.pop("_sort_action_priority", None)
    
    df = pd.DataFrame(filtered)

    # Format currency columns for Indian style commas before display
    if "PnL" in df.columns:
        df["PnL"] = df["PnL"].apply(lambda x: format_inr(x))
    if "Notional" in df.columns:
        df["Notional"] = df["Notional"].apply(lambda x: format_inr(x))
    
    if df.empty:
        st.info("No positions match the selected filters.")
        return
    
    # Format and display (PnL/Notional already formatted as strings)
    st.dataframe(
        df.style.format({
            "Entry": "{:.2f}",
            "Current": "{:.2f}",
            "Chg %": "{:+.1f}%",
            "Theta/Day": "{:.2f}",
            "Days to B/E": "{:.1f}",
            "Theta Eff %": "{:.0f}%",
            "Delta": "{:.3f}",
            "Pos Delta": "{:.2f}",
            "Gamma": "{:.4f}",
            "Vega": "{:.2f}",
            "% Portfolio": "{:.1f}%",
            "Spot Dist %": "{:+.1f}%",
            "PoP %": "{:.0f}%",
            "IV": "{:.1%}",
            "Loss/Credit": "{:.2f}x"
        }),
        use_container_width=True,
        height=600
    )
    
    # Action summary
    st.markdown("### 🎯 Action Summary")
    
    action_counts = {}
    for d in diagnostics:
        action = d["Action"]
        action_counts[action] = action_counts.get(action, 0) + 1
    
    cols = st.columns(len(action_counts))
    for i, (action, count) in enumerate(action_counts.items()):
        with cols[i]:
            st.metric(action, count)
    
    # Top priority actions
    st.markdown("### ⚡ Top Priority Actions")
    priority_positions = [d for d in diagnostics if d["Priority"] >= 8]
    priority_positions = sorted(priority_positions, key=lambda x: x["Priority"], reverse=True)[:5]
    
    if priority_positions:
        for pos in priority_positions:
            with st.expander(f"🔴 {pos['Symbol']} - {pos['Action']} (Priority: {pos['Priority']}/10)"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**P&L:** {format_inr(pos['PnL'])}")
                    st.write(f"**DTE:** {pos['DTE']} days")
                    st.write(f"**Days to B/E:** {pos['Days to B/E']:.1f}")
                with col2:
                    st.write(f"**Position Delta:** {pos['Pos Delta']:.2f}")
                    st.write(f"**Theta/Day:** {format_inr(pos['Theta/Day'], decimals=2)}")
                    st.write(f"**Loss/Credit:** {pos['Loss/Credit']:.2f}x")
                
                st.markdown(f"**Recommended Action:** {get_action_recommendation(pos)}")
    else:
        st.success("✅ No high-priority actions needed")
