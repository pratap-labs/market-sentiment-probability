"""Risk Analysis Tab - Zone-based portfolio risk framework with normalized greeks."""

import streamlit as st
import pandas as pd
from collections import Counter
from typing import List, Dict

import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.utils import (
    calculate_portfolio_greeks,
    format_inr,
    build_threshold_report,
    get_weighted_scenarios,
    classify_history_bucket,
    compute_var_es_metrics,
)

from views.tabs.derivatives_data_tab import load_from_cache


def get_iv_regime(iv_percentile):
    """Classify IV regime based on IV percentile."""
    if iv_percentile < 20:
        return "Low IV", "🟢"
    elif iv_percentile < 40:
        return "Mid IV", "🟡"
    else:
        return "High IV", "🔴"


def classify_zone(theta_norm, gamma_norm, vega_norm, iv_regime):
    """
    Classify portfolio into Zone 1/2/3 based on normalized greeks.
    
    All greeks are normalized per ₹1L capital.
    
    Returns: (zone_number, zone_name, zone_color, status_message)
    """
    
    # ZONE 1 — SAFE / PROFESSIONAL
    zone1_theta_range = (120, 180)
    zone1_gamma_range = (0, -0.020)
    zone1_vega_ranges = {
        "Low IV": (0, -200),
        "Mid IV": (-200, -450),
        "High IV": (-450, -700)
    }
    
    # ZONE 2 — BALANCED / CONTROLLED
    zone2_theta_range = (180, 220)
    zone2_gamma_range = (-0.020, -0.035)
    zone2_vega_ranges = {
        "Low IV": (0, -350),
        "Mid IV": (-350, -650),
        "High IV": (-650, -900)
    }
    
    # ZONE 3 — AGGRESSIVE / FRAGILE
    zone3_theta_range = (220, 300)
    zone3_gamma_range = (-0.035, -0.055)
    zone3_vega_ranges = {
        "Low IV": (None, None),  # Avoid in Low IV
        "Mid IV": (-650, -1000),
        "High IV": (-1000, -1300)
    }
    
    # Check Zone 1
    theta_in_z1 = zone1_theta_range[0] <= theta_norm <= zone1_theta_range[1]
    gamma_in_z1 = zone1_gamma_range[1] <= gamma_norm <= zone1_gamma_range[0]
    vega_range_z1 = zone1_vega_ranges[iv_regime]
    vega_in_z1 = vega_range_z1[1] <= vega_norm <= vega_range_z1[0]
    
    if theta_in_z1 and gamma_in_z1 and vega_in_z1:
        return 1, "SAFE / PROFESSIONAL", "🟢", "Long-term survival, minimal drawdowns. 1% NIFTY move ≈ 1–2 days of theta."
    
    # Check Zone 2
    theta_in_z2 = zone2_theta_range[0] <= theta_norm <= zone2_theta_range[1]
    gamma_in_z2 = zone2_gamma_range[1] <= gamma_norm <= zone2_gamma_range[0]
    vega_range_z2 = zone2_vega_ranges[iv_regime]
    vega_in_z2 = vega_range_z2[1] <= vega_norm <= vega_range_z2[0]
    
    if theta_in_z2 and gamma_in_z2 and vega_in_z2:
        return 2, "BALANCED / CONTROLLED", "🟡", "Income focus with active management. 1% move ≈ 3–5 days of theta. Requires daily monitoring."
    
    # Check Zone 3
    theta_in_z3 = zone3_theta_range[0] <= theta_norm <= zone3_theta_range[1]
    gamma_in_z3 = zone3_gamma_range[1] <= gamma_norm <= zone3_gamma_range[0]
    vega_range_z3 = zone3_vega_ranges[iv_regime]
    
    # Zone 3 not allowed in Low IV
    if iv_regime == "Low IV":
        if theta_in_z3 or gamma_in_z3:
            return 3, "AGGRESSIVE / FRAGILE", "🔴", "❌ AVOID ZONE 3 IN LOW IV REGIME"
    else:
        vega_in_z3 = vega_range_z3[1] <= vega_norm <= vega_range_z3[0] if vega_range_z3[0] else False
        if theta_in_z3 and gamma_in_z3 and vega_in_z3:
            return 3, "AGGRESSIVE / FRAGILE", "🔴", "High income, short lifespan if unmanaged. 1% move wipes 1–2 weeks of theta. Emotion enters decisions."
    
    # Out of bounds
    return 0, "OUT OF BOUNDS", "⚠️", "Portfolio greeks outside defined zone limits. Review and adjust."


def render_risk_analysis_tab():
    """Render zone-based risk analysis tab."""
    
    st.header("🎯 Zone-Based Risk Analysis")
    
    # Check if positions are loaded
    if "enriched_positions" not in st.session_state:
        st.info("No positions loaded. Please fetch positions from the Positions tab first.")
        st.markdown("---")
        st.markdown("## 📋 Risk Framework Rules")
        st.info("Load positions to see your portfolio's zone classification and risk analysis.")
        render_rules_only()
        return
    
    enriched = st.session_state["enriched_positions"]
    portfolio_greeks = calculate_portfolio_greeks(enriched)
    current_spot = st.session_state.get("current_spot", 25000)
    
    # Get account size
    access_token = st.session_state.get("kite_access_token")
    kite_api_key = st.session_state.get("kite_api_key")
    
    margin_available = 1150000
    margin_used = 1300000
    
    if access_token and kite_api_key:
        try:
            from kiteconnect import KiteConnect
            kite = KiteConnect(api_key=kite_api_key)
            kite.set_access_token(access_token)
            margins = kite.margins()
            
            equity_margins = margins.get("equity", {})
            margin_available = equity_margins.get("available", {}).get("live_balance", 500000)
            margin_used = equity_margins.get("utilised", {}).get("debits", 0)
        except:
            pass
    
    account_size = margin_available + margin_used
    margin_deployed = margin_used if margin_used > 0 else account_size
    
    # Get greeks
    total_theta = portfolio_greeks["net_theta"]
    total_delta = portfolio_greeks["net_delta"]
    total_gamma = portfolio_greeks["net_gamma"]
    total_vega = portfolio_greeks["net_vega"]
    
    # ========== SECTION 1: CURRENT PORTFOLIO ANALYSIS ==========
    st.markdown("## 📊 Your Portfolio Analysis")
    
    # Calculate IV percentile
    iv_percentile = 30  # Default
    options_df_cache = st.session_state.get("options_df_cache", pd.DataFrame())
    if not options_df_cache.empty and "iv" in options_df_cache.columns:
        current_iv = options_df_cache["iv"].median()
        iv_percentile = 35
    
    iv_regime, iv_color = get_iv_regime(iv_percentile)
    
    st.markdown(f"### Current Volatility Regime: {iv_color} **{iv_regime}**")
    st.caption(f"IV Percentile: {iv_percentile:.0f}")
    
    # Normalize greeks per ₹1L
    capital_in_lakhs = account_size / 100000
    
    theta_norm = abs(total_theta) / capital_in_lakhs if capital_in_lakhs > 0 else 0
    gamma_norm = total_gamma / capital_in_lakhs if capital_in_lakhs > 0 else 0
    vega_norm = total_vega / capital_in_lakhs if capital_in_lakhs > 0 else 0
    
    # Classify zone first (needed for color coding)
    zone_num, zone_name, zone_color, zone_message = classify_zone(
        theta_norm, gamma_norm, vega_norm, iv_regime
    )
    
    # Display current normalized greeks in tabular format
    st.markdown("### Normalized Greeks (per ₹1L)")
    
    # Account size metric at top
    st.metric("Account Size", f"₹{capital_in_lakhs:.2f}L")
    
    # Define zone ranges
    zone1_ranges = {
        "Theta": (120, 180),
        "Gamma": (0, -0.020),
        "Vega (Low IV)": (0, -200),
        "Vega (Mid IV)": (-200, -450),
        "Vega (High IV)": (-450, -700)
    }
    
    zone2_ranges = {
        "Theta": (180, 220),
        "Gamma": (-0.020, -0.035),
        "Vega (Low IV)": (0, -350),
        "Vega (Mid IV)": (-350, -650),
        "Vega (High IV)": (-650, -900)
    }
    
    zone3_ranges = {
        "Theta": (220, 300),
        "Gamma": (-0.035, -0.055),
        "Vega (Low IV)": (None, None),
        "Vega (Mid IV)": (-650, -1000),
        "Vega (High IV)": (-1000, -1300)
    }
    
    # Determine vega key based on IV regime
    vega_key = f"Vega ({iv_regime})"
    
    # Helper function to get color based on value and zones
    def get_greek_color(value, greek_name):
        if greek_name == "Theta":
            z1 = zone1_ranges["Theta"]
            z2 = zone2_ranges["Theta"]
            z3 = zone3_ranges["Theta"]
            
            if z1[0] <= value <= z1[1]:
                return "🟢", "Zone 1"
            elif z2[0] <= value <= z2[1]:
                return "🟡", "Zone 2"
            elif z3[0] <= value <= z3[1]:
                return "🔴", "Zone 3"
            else:
                return "⚠️", "Out of Bounds"
        
        elif greek_name == "Gamma":
            z1 = zone1_ranges["Gamma"]
            z2 = zone2_ranges["Gamma"]
            z3 = zone3_ranges["Gamma"]
            
            if z1[1] <= value <= z1[0]:
                return "🟢", "Zone 1"
            elif z2[1] <= value <= z2[0]:
                return "🟡", "Zone 2"
            elif z3[1] <= value <= z3[0]:
                return "🔴", "Zone 3"
            else:
                return "⚠️", "Out of Bounds"
        
        elif greek_name == "Vega":
            z1 = zone1_ranges[vega_key]
            z2 = zone2_ranges[vega_key]
            z3 = zone3_ranges[vega_key]
            
            if z1[1] <= value <= z1[0]:
                return "🟢", "Zone 1"
            elif z2[1] <= value <= z2[0]:
                return "🟡", "Zone 2"
            elif z3[0] is not None and z3[1] <= value <= z3[0]:
                return "🔴", "Zone 3"
            else:
                return "⚠️", "Out of Bounds"
        
        return "⚪", "Unknown"
    
    # Get colors for each greek
    theta_color, theta_zone = get_greek_color(theta_norm, "Theta")
    gamma_color, gamma_zone = get_greek_color(gamma_norm, "Gamma")
    vega_color, vega_zone = get_greek_color(vega_norm, "Vega")
    
    # Create the comparison table
    theta_pct = theta_norm / 1000 if capital_in_lakhs > 0 else 0
    vega_pct = abs(vega_norm) / 1000 if capital_in_lakhs > 0 else 0

    greeks_comparison_df = pd.DataFrame({
        "Greek": ["Theta/Day", "Gamma", f"Vega ({iv_regime})"],
        "Zone 1 Range": [
            f"₹{zone1_ranges['Theta'][0]} – ₹{zone1_ranges['Theta'][1]}",
            f"{zone1_ranges['Gamma'][1]:.3f} to {zone1_ranges['Gamma'][0]:.3f}",
            f"₹{zone1_ranges[vega_key][1]} to ₹{zone1_ranges[vega_key][0]}"
        ],
        "Zone 2 Range": [
            f"₹{zone2_ranges['Theta'][0]} – ₹{zone2_ranges['Theta'][1]}",
            f"{zone2_ranges['Gamma'][1]:.3f} to {zone2_ranges['Gamma'][0]:.3f}",
            f"₹{zone2_ranges[vega_key][1]} to ₹{zone2_ranges[vega_key][0]}"
        ],
        "Zone 3 Range": [
            f"₹{zone3_ranges['Theta'][0]} – ₹{zone3_ranges['Theta'][1]}",
            f"{zone3_ranges['Gamma'][1]:.3f} to {zone3_ranges['Gamma'][0]:.3f}",
            "❌ AVOID" if zone3_ranges[vega_key][0] is None else f"₹{zone3_ranges[vega_key][1]} to ₹{zone3_ranges[vega_key][0]}"
        ],
        "Current Value (per ₹1L)": [
            f"₹{theta_norm:.0f} ({theta_pct:.2f}%/day)",
            f"{gamma_norm:.4f}",
            f"₹{vega_norm:.0f} ({vega_pct:.2f}% per IV pt)"
        ],
        "Status": [
            f"{theta_color} {theta_zone}",
            f"{gamma_color} {gamma_zone}",
            f"{vega_color} {vega_zone}"
        ]
    })
    
    # Display the table with custom styling
    st.dataframe(
        greeks_comparison_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Greek": st.column_config.TextColumn("Greek", width="medium"),
            "Zone 1 Range": st.column_config.TextColumn("🟢 Zone 1", width="medium"),
            "Zone 2 Range": st.column_config.TextColumn("🟡 Zone 2", width="medium"),
            "Zone 3 Range": st.column_config.TextColumn("🔴 Zone 3", width="medium"),
            "Current Value (per ₹1L)": st.column_config.TextColumn("Current", width="medium"),
            "Status": st.column_config.TextColumn("Status", width="medium")
        }
    )
    
    st.markdown("---")

    scenarios = get_weighted_scenarios(iv_regime)
    threshold_context = render_threshold_builder_panel(
        scenarios,
        portfolio_greeks,
        account_size,
        margin_deployed,
        current_spot,
    )
    
    render_expected_shortfall_panel(
        threshold_context,
        account_size,
        margin_deployed,
        scenarios
    )
    
    # Display zone classification
    st.markdown(f"## {zone_color} ZONE {zone_num} — {zone_name}")
    
    if zone_num == 0:
        st.error(zone_message)
    elif zone_num == 1:
        st.success(zone_message)
    elif zone_num == 2:
        st.warning(zone_message)
    elif zone_num == 3:
        st.error(zone_message)
    
    # Visual indicator of position within zones (removed - now in table)
    # st.markdown("### Zone Positioning")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        theta_z1 = (120, 180)
        theta_z2 = (180, 220)
        theta_z3 = (220, 300)
        
        if theta_z1[0] <= theta_norm <= theta_z1[1]:
            theta_zone = "🟢 Zone 1"
        elif theta_z2[0] <= theta_norm <= theta_z2[1]:
            theta_zone = "🟡 Zone 2"
        elif theta_z3[0] <= theta_norm <= theta_z3[1]:
            theta_zone = "🔴 Zone 3"
        else:
            theta_zone = "⚠️ Out of Bounds"
        
        st.metric("Theta Zone", theta_zone, f"₹{theta_norm:.0f}/₹1L")
    
    with col2:
        gamma_z1 = (0, -0.020)
        gamma_z2 = (-0.020, -0.035)
        gamma_z3 = (-0.035, -0.055)
        
        if gamma_z1[1] <= gamma_norm <= gamma_z1[0]:
            gamma_zone = "🟢 Zone 1"
        elif gamma_z2[1] <= gamma_norm <= gamma_z2[0]:
            gamma_zone = "🟡 Zone 2"
        elif gamma_z3[1] <= gamma_norm <= gamma_z3[0]:
            gamma_zone = "🔴 Zone 3"
        else:
            gamma_zone = "⚠️ Out of Bounds"
        
        st.metric("Gamma Zone", gamma_zone, f"{gamma_norm:.4f}")
    
    with col3:
        vega_ranges = {
            "Low IV": {"z1": (0, -200), "z2": (0, -350), "z3": (None, None)},
            "Mid IV": {"z1": (-200, -450), "z2": (-350, -650), "z3": (-650, -1000)},
            "High IV": {"z1": (-450, -700), "z2": (-650, -900), "z3": (-1000, -1300)}
        }
        
        vega_z1 = vega_ranges[iv_regime]["z1"]
        vega_z2 = vega_ranges[iv_regime]["z2"]
        vega_z3 = vega_ranges[iv_regime]["z3"]
        
        if vega_z1[1] <= vega_norm <= vega_z1[0]:
            vega_zone = "🟢 Zone 1"
        elif vega_z2[1] <= vega_norm <= vega_z2[0]:
            vega_zone = "🟡 Zone 2"
        elif vega_z3[0] and vega_z3[1] <= vega_norm <= vega_z3[0]:
            vega_zone = "🔴 Zone 3"
        else:
            vega_zone = "⚠️ Out of Bounds"
        
        st.metric(f"Vega Zone ({iv_regime})", vega_zone, f"₹{vega_norm:.0f}/₹1L")
    
    st.markdown("---")
    
    # ========== SECTION 2: ACTIONABLE INSIGHTS ==========
    st.markdown("## 💡 Actionable Insights")
    
    insights = []
    
    # Zone-specific insights
    if zone_num == 0:
        insights.append("### 🔴 CRITICAL: Portfolio Out of Bounds")
        insights.append("Your portfolio is outside all defined zones. This indicates excessive risk.")
        
        if theta_norm > 300:
            insights.append(f"- **Theta ({theta_norm:.0f})** is too high. Reduce short option exposure.")
        elif theta_norm < 120:
            insights.append(f"- **Theta ({theta_norm:.0f})** is too low. Consider adding income-generating positions.")
        
        if gamma_norm < -0.055:
            insights.append(f"- **Gamma ({gamma_norm:.4f})** is dangerously negative. Reduce short gamma exposure or add long options.")
        elif gamma_norm > 0:
            insights.append(f"- **Gamma ({gamma_norm:.4f})** is positive. Consider selling some long options for income.")
        
        if vega_norm < -1300:
            insights.append(f"- **Vega ({vega_norm:.0f})** is extremely negative. High risk from IV expansion.")
        
    elif zone_num == 1:
        insights.append("### ✅ Excellent: Safe Zone")
        insights.append("Portfolio is in the safest zone for long-term survival.")
        insights.append("- This setup can withstand market volatility with minimal drawdowns.")
        insights.append("- Continue monitoring but no urgent adjustments needed.")
        
        if theta_norm < 150:
            insights.append(f"- **Optimization**: Theta ({theta_norm:.0f}) is in lower range. Consider slight increase for better returns.")
        
    elif zone_num == 2:
        insights.append("### ⚠️ Good: Balanced Zone")
        insights.append("Portfolio is balanced but requires active monitoring.")
        insights.append("- Watch for 1%+ NIFTY moves — may need adjustments.")
        insights.append("- IV expansion will require action — prepare hedge strategies.")
        
        if theta_norm > 210:
            insights.append(f"- **Caution**: Theta ({theta_norm:.0f}) approaching Zone 3. Watch for overexposure.")
        
        if gamma_norm < -0.030:
            insights.append(f"- **Caution**: Gamma ({gamma_norm:.4f}) getting more negative. Monitor delta swings.")
        
    elif zone_num == 3:
        insights.append("### 🔴 WARNING: Aggressive Zone")
        
        if iv_regime == "Low IV":
            insights.append("- ❌ **CRITICAL**: Zone 3 is NOT ALLOWED in Low IV regime.")
            insights.append("- **Action Required**: Reduce exposure immediately.")
        else:
            insights.append("- High risk, high reward. This is not sustainable long-term.")
            insights.append("- 1% NIFTY move will wipe 1-2 weeks of theta collection.")
            insights.append("- **Recommendation**: Consider reducing to Zone 2 for better risk-adjusted returns.")
        
        if gamma_norm < -0.045:
            insights.append(f"- **URGENT**: Gamma ({gamma_norm:.4f}) very negative. Small moves = large delta changes.")
        
        if vega_norm < -1100:
            insights.append(f"- **URGENT**: Vega ({vega_norm:.0f}) exposure is extreme. Any IV spike will hurt badly.")
    
    for insight in insights:
        st.markdown(insight)
    
    st.markdown("---")
    
    # ========== SECTION 3: GREEK-SPECIFIC ANALYSIS ==========
    st.markdown("## 📈 Greek-Specific Impact Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Theta Impact")
        if theta_norm > 0:
            days_to_1pct = (account_size * 0.01) / abs(total_theta) if total_theta != 0 else 999
            st.metric("Daily Collection", format_inr(abs(total_theta)))
            st.metric("Days to Earn 1% Capital", f"{days_to_1pct:.1f} days")
            
            monthly_theta = abs(total_theta) * 30
            monthly_pct = (monthly_theta / account_size * 100) if account_size > 0 else 0
            st.metric("Monthly Theta (30 days)", format_inr(monthly_theta), f"{monthly_pct:.2f}% of capital")
    
    with col2:
        st.markdown("### Gamma Impact")
        if gamma_norm < -0.020:
            one_pct_move_delta_change = abs(gamma_norm * capital_in_lakhs * (current_spot * 0.01))
            st.metric("Delta Change (1% NIFTY)", f"{one_pct_move_delta_change:.0f} units")
            
            # Estimate P&L impact from gamma
            gamma_pnl_1pct = abs(total_gamma) * (current_spot * 0.01) ** 2 * 50  # Approximate
            st.metric("Gamma P&L (1% move)", format_inr(gamma_pnl_1pct))
            
            st.caption(f"Current Gamma: {gamma_norm:.4f} per ₹1L")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Vega Impact")
        if abs(vega_norm) > 200:
            iv_1pt_impact = abs(vega_norm * capital_in_lakhs * 1)
            iv_5pt_impact = abs(vega_norm * capital_in_lakhs * 5)
            
            st.metric("P&L Impact (1 IV point)", format_inr(iv_1pt_impact))
            st.metric("P&L Impact (5 IV points)", format_inr(iv_5pt_impact))
            
            st.caption(f"Consider hedging if expecting IV expansion in {iv_regime}")
    
    with col2:
        st.markdown("### Delta Impact")
        delta_dollars = abs(total_delta * current_spot)
        delta_pct = (delta_dollars / account_size * 100) if account_size > 0 else 0
        
        st.metric("Directional Exposure", format_inr(delta_dollars), f"{delta_pct:.1f}% of capital")
        
        one_pct_nifty_pnl = total_delta * (current_spot * 0.01) * 50  # Approximate lot size effect
        st.metric("P&L on 1% NIFTY Move", format_inr(one_pct_nifty_pnl))
        
        if delta_pct > 10:
            st.warning(f"⚠️ Directional exposure is {delta_pct:.1f}% - Consider neutralizing if not intentional")
    
    st.markdown("---")
    
    # ========== SECTION 4: RULES DISPLAY ==========
    st.markdown("## 📋 Zone-Based Risk Framework Rules")
    st.markdown("All greeks below are **normalized per ₹1 Lakh** of capital for consistent comparison across portfolio sizes.")
    
    # IV Regime classification
    st.markdown("### Volatility Regimes (Mandatory Context)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🟢 **Low IV**")
        st.caption("< 20 IV Percentile")
    with col2:
        st.markdown("🟡 **Mid IV**")
        st.caption("20–40 IV Percentile")
    with col3:
        st.markdown("🔴 **High IV**")
        st.caption("> 40 IV Percentile")
    
    st.markdown("---")
    
    # Zone definitions with visual tables
    st.markdown("### 🟢 ZONE 1 — SAFE / PROFESSIONAL")
    st.success("**Long-term survival, minimal drawdowns**")
    
    zone1_df = pd.DataFrame({
        "Greek": ["Theta", "Gamma", "Vega (Low IV)", "Vega (Mid IV)", "Vega (High IV)"],
        "Range (per ₹1L)": [
            "₹120 – ₹180 / day",
            "0 to −0.020",
            "0 to −200",
            "−200 to −450",
            "−450 to −700"
        ],
        "% of Capital": [
            "0.12% – 0.18% / day",
            "—",
            "—",
            "—",
            "—"
        ]
    })
    st.dataframe(zone1_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    **What this feels like:**
    - ✅ 1% NIFTY move ≈ 1–2 days of theta
    - ✅ IV spike ≈ annoyance, not damage
    - ✅ Adjustments optional
    """)
    
    st.markdown("---")
    
    st.markdown("### 🟡 ZONE 2 — BALANCED / CONTROLLED")
    st.warning("**Income focus with active management**")
    
    zone2_df = pd.DataFrame({
        "Greek": ["Theta", "Gamma", "Vega (Low IV)", "Vega (Mid IV)", "Vega (High IV)"],
        "Range (per ₹1L)": [
            "₹180 – ₹220 / day",
            "−0.020 to −0.035",
            "0 to −350",
            "−350 to −650",
            "−650 to −900"
        ],
        "% of Capital": [
            "0.18% – 0.22% / day",
            "—",
            "—",
            "—",
            "—"
        ]
    })
    st.dataframe(zone2_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    **What this feels like:**
    - ⚠️ 1% move ≈ 3–5 days of theta
    - ⚠️ IV expansion needs action
    - ⚠️ Requires daily monitoring
    """)
    
    st.markdown("---")
    
    st.markdown("### 🔴 ZONE 3 — AGGRESSIVE / FRAGILE")
    st.error("**High income, short lifespan if unmanaged**")
    
    zone3_df = pd.DataFrame({
        "Greek": ["Theta", "Gamma", "Vega (Low IV)", "Vega (Mid IV)", "Vega (High IV)"],
        "Range (per ₹1L)": [
            "₹220 – ₹300 / day",
            "−0.035 to −0.055",
            "❌ AVOID",
            "−650 to −1,000",
            "−1,000 to −1,300"
        ],
        "% of Capital": [
            "> 0.22% / day",
            "—",
            "—",
            "—",
            "—"
        ]
    })
    st.dataframe(zone3_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    **What this feels like:**
    - 🔴 1% move wipes 1–2 weeks of theta
    - 🔴 Gaps cause forced adjustments
    - 🔴 Emotion enters decisions
    - 🔴 **NOT ALLOWED in Low IV regime**
    """)
    
    st.markdown("---")


def render_threshold_builder_panel(
    scenarios,
    portfolio_greeks,
    account_size,
    margin_deployed,
    spot,
):
    """Render threshold builder derived from master NAV limits."""
    st.markdown("## 🧱 Threshold Builder (Derived from Master NAV Limit)")
    if not scenarios:
        st.info("No scenarios available. Run scenario stress tests first.")
        return

    if account_size <= 0 or margin_deployed <= 0:
        st.warning("NAV or margin unavailable. Cannot compute threshold breaches.")
        return

    controls = st.columns(4)
    master_pct = controls[0].number_input(
        "Master loss budget (% NAV)",
        value=1.0,
        step=0.1,
        format="%.2f",
        key="threshold_master_pct",
    )
    hard_stop_pct = controls[1].number_input(
        "Hard stop (% NAV)",
        value=1.2,
        step=0.1,
        format="%.2f",
        key="threshold_hard_stop_pct",
    )
    normal_share = controls[2].number_input(
        "Normal share (fraction of master)",
        value=0.5,
        step=0.05,
        format="%.2f",
        key="threshold_normal_share",
    )
    stress_share = controls[3].number_input(
        "Stress share (fraction of master)",
        value=0.9,
        step=0.05,
        format="%.2f",
        key="threshold_stress_share",
    )

    derived = build_threshold_report(
        portfolio={
            "delta": portfolio_greeks.get("net_delta", 0.0),
            "gamma": portfolio_greeks.get("net_gamma", 0.0),
            "vega": portfolio_greeks.get("net_vega", 0.0),
            "spot": spot,
            "nav": account_size,
            "margin": margin_deployed,
        },
        scenarios=[
            {
                "name": scenario.name,
                "dS_pct": scenario.ds_pct,
                "dIV_pts": scenario.div_pts,
                "type": scenario.category.upper(),
            }
            for scenario in scenarios
        ],
        master_pct=master_pct,
        hard_stop_pct=hard_stop_pct,
        normal_share=normal_share,
        stress_share=stress_share,
    )

    thresholds = derived["thresholds"]
    st.caption(
        f"Normal threshold: {thresholds['limitA']:.2f}% | "
        f"Stress threshold: {thresholds['limitB']:.2f}% | "
        f"Extreme threshold: {thresholds['limitC']:.2f}%"
    )

    worst = derived["worst"]
    worst_name = worst["scenario"]["name"] if worst else "N/A"
    worst_loss_value = worst["pnl_total"] if worst else 0.0
    worst_loss_value = min(0.0, worst_loss_value)
    worst_loss_pct = derived["worst_loss_pct"]

    summary_cols = st.columns(4)
    summary_cols[0].metric("Worst Scenario", worst_name)
    summary_cols[1].metric("Worst Loss (₹)", format_inr(worst_loss_value))
    summary_cols[2].metric("Worst Loss % NAV", f"{worst_loss_pct:.2f}%")
    summary_cols[3].metric("Failed Scenarios", str(derived["fail_count"]))

    within_master = derived["within_master"]
    status_badge = "PASS" if within_master else "FAIL"
    badge_icon = "🟢" if within_master else "🔴"
    st.markdown(f"**Within Master Rule?** {badge_icon} {status_badge}")

    table_rows = []
    for row in derived["rows"]:
        scenario = row["scenario"]
        table_rows.append(
            {
                "Scenario": scenario["name"],
                "Bucket": row["bucket"],
                "dS% / dIV": f"{scenario['dS_pct']:+.2f}% / {scenario['dIV_pts']:+.1f}",
                "Δ P&L (₹)": format_inr(row["pnl_delta"]),
                "Γ P&L (₹)": format_inr(row["pnl_gamma"]),
                "Vega P&L (₹)": format_inr(row["pnl_vega"]),
                "Total P&L (₹)": format_inr(row["pnl_total"]),
                "Loss % NAV": row["loss_pct_nav"],
                "Threshold % NAV": row["threshold_pct"],
                "Status": row["status"],
            }
        )

    threshold_df = pd.DataFrame(table_rows)
    derived["table_df"] = threshold_df
    return derived


def render_expected_shortfall_panel(threshold_context, account_size, margin_deployed, scenarios):
    """Render probability-weighted ES/VaR panel."""
    st.markdown("## 📉 Expected Shortfall (Historical-calibrated)")

    if not threshold_context:
        st.info("Run the Threshold Builder to compute scenario P&L before calibrating ES.")
        return

    derived_rows = threshold_context.get("rows") if isinstance(threshold_context, dict) else threshold_context
    if not derived_rows:
        st.info("No scenario rows available for ES calculation.")
        return

    lookback = st.number_input("Lookback trading days", value=504, min_value=126, max_value=756, step=21, key="es_lookback_days")
    smoothing_enabled = st.checkbox("Apply EWMA smoothing", value=False, key="es_smoothing_toggle")
    smoothing_span = 63
    if smoothing_enabled:
        smoothing_span = st.slider("EWMA span (days)", min_value=21, max_value=min(252, lookback), value=63, step=7, key="es_smoothing_span")

    bucket_probs, history_count, used_fallback = compute_historical_bucket_probabilities(
        lookback=int(lookback),
        smoothing_enabled=smoothing_enabled,
        smoothing_span=int(smoothing_span),
    )

    p_caption = f"Bucket probabilities (lookback {history_count}d): "
    p_caption += f"A {bucket_probs['A']*100:.1f}%, B {bucket_probs['B']*100:.1f}%, C {bucket_probs['C']*100:.1f}%"
    st.caption(p_caption)
    if used_fallback:
        st.warning("Historical cache unavailable; using default manual probabilities (A/B/C = 60/30/10).")

    bucket_counts = Counter(row["bucket"] for row in derived_rows)
    for row in derived_rows:
        bucket = row["bucket"]
        bucket_prob = bucket_probs.get(bucket, 0.0)
        count = bucket_counts.get(bucket, 0)
        if count and bucket_prob > 0:
            row["probability"] = bucket_prob / count
        else:
            row["probability"] = 0.0

    total_prob = sum(row.get("probability", 0.0) for row in derived_rows)
    if abs(total_prob - 1.0) > 0.05:
        st.warning(f"Scenario probabilities sum to {total_prob:.2f}; expected ≈ 1.0.")

    loss_distribution = [
        {
            "loss_pct": row["loss_pct_nav"],
            "prob": row["probability"],
            "scenario": row["scenario"],
            "bucket": row["bucket"],
        }
        for row in derived_rows
    ]

    metrics = compute_var_es_metrics(loss_distribution, account_size)

    limit_cols = st.columns(4)
    es99_limit = limit_cols[0].number_input("ES99 limit (% NAV)", value=1.0, step=0.1, format="%.2f", key="es99_limit")
    es95_limit = limit_cols[1].number_input("ES95 limit (% NAV)", value=0.7, step=0.1, format="%.2f", key="es95_limit")
    var99_limit = limit_cols[2].number_input("VaR99 limit (% NAV)", value=1.2, step=0.1, format="%.2f", key="var99_limit")
    var95_limit = limit_cols[3].number_input("VaR95 limit (% NAV)", value=0.7, step=0.1, format="%.2f", key="var95_limit")

    breach_flags = [
        metrics["ES99"] > es99_limit,
        metrics["ES95"] > es95_limit,
        metrics["VaR99"] > var99_limit,
    ]
    breach_count = sum(1 for flag in breach_flags if flag)
    if breach_count == 0:
        badge = ("SAFE", "🟢")
    elif breach_count == 1:
        badge = ("WATCH", "🟡")
    else:
        badge = ("REDUCE", "🔴")

    render_risk_governance_block(metrics, var99_limit, es99_limit, var95_limit, threshold_context)

    summary_cols = st.columns(4)
    summary_cols[0].metric("ES99", f"{metrics['ES99']:.2f}%", format_inr(metrics["ES99Value"]))
    summary_cols[1].metric("VaR99", f"{metrics['VaR99']:.2f}%", format_inr(metrics["VaR99Value"]))
    summary_cols[2].metric("ES95", f"{metrics['ES95']:.2f}%", format_inr(metrics["ES95Value"]))
    summary_cols[3].metric("VaR95", f"{metrics['VaR95']:.2f}%", format_inr(metrics["VaR95Value"]))

    st.markdown(f"**ES Status:** {badge[1]} {badge[0]}")

    tail_set = metrics.get("tail_set_99", [])
    expected_loss, prob_sum, bucket_shares, scenario_tail_contribs = compute_es_attribution(
        derived_rows, tail_set, tail_level=0.99, es_value=metrics["ES99"]
    )

    st.caption(f"Expected loss (portfolio mean): {expected_loss:.3f}% NAV")

    if metrics["ES99"] > 0 and bucket_shares:
        share_text = "ES99 driven by: " + ", ".join(
            [
                f"{bucket} {bucket_shares.get(bucket, 0.0) * 100:.1f}%"
                for bucket in ["A", "B", "C"]
            ]
        )
        st.markdown(share_text)

    bucket_share_sum = sum(bucket_shares.values())
    tol = 1e-6
    inconsistencies = []
    if metrics["ES99"] + tol < metrics["VaR99"]:
        inconsistencies.append("ES99 < VaR99")
    if metrics["ES95"] + tol < metrics["VaR95"]:
        inconsistencies.append("ES95 < VaR95")
    if expected_loss > metrics["ES95"] + 1e-3:
        inconsistencies.append("Expected loss exceeds ES95")
    if abs(prob_sum - 1.0) > 0.05:
        inconsistencies.append("Scenario probabilities do not sum to 1")
    if metrics["ES99"] > 0 and abs(bucket_share_sum - 1.0) > 0.05:
        inconsistencies.append("Bucket tail shares do not sum to 100%")

    if inconsistencies:
        print("ES attribution inconsistencies:", inconsistencies)
        st.warning("⚠️ ES attribution inconsistent. Check scenario probabilities or data quality.")

    if scenario_tail_contribs:
        tail_table = sorted(
            [
                {
                    "Scenario": entry["scenario"]["name"],
                    "Bucket": entry["bucket"],
                    "Loss % NAV": entry["loss_pct"],
                    "Prob (tail)": entry["tail_prob"],
                    "TailContrib": entry["tail_contribution"],
                    "Share of ES99": entry["share_of_es"],
                }
                for entry in scenario_tail_contribs
            ],
            key=lambda x: x["TailContrib"],
            reverse=True,
        )[:8]
        tail_df = pd.DataFrame(
            [
                {
                    **row,
                    "Loss % NAV": f"{row['Loss % NAV']:.2f}%",
                    "Prob (tail)": f"{row['Prob (tail)']*100:.2f}%",
                    "TailContrib": f"{row['TailContrib']:.3f}%",
                    "Share of ES99": f"{row['Share of ES99']*100:.2f}%",
                }
                for row in tail_table
            ]
        )
        st.markdown("### ES99 Attribution")
        st.dataframe(tail_df, use_container_width=True, hide_index=True)
    else:
        st.info("No tail contributions available for ES99.")

    if metrics["ES99"] > es99_limit and scenario_tail_contribs:
        dominant_bucket = max(bucket_shares, key=bucket_shares.get)
        dominant_entry = max(scenario_tail_contribs, key=lambda x: x["tail_contribution"])
        reduction_needed = compute_needed_reduction(metrics["ES99"], es99_limit)
        st.warning(
            f"Primary driver: Bucket {dominant_bucket} / {dominant_entry['scenario']['name']}.\n"
            f"Required ES99 reduction: {reduction_needed * 100:.1f}%. Reduce Bucket {dominant_bucket} tail risk to pass ES99."
        )

    scenario_table_df = threshold_context.get("table_df")
    if scenario_table_df is not None and not scenario_table_df.empty:
        display_df = scenario_table_df.copy()
        display_df["Loss % NAV"] = display_df["Loss % NAV"].map(lambda x: f"{x:.2f}%")
        display_df["Threshold % NAV"] = display_df["Threshold % NAV"].map(lambda x: f"{x:.2f}%")
        prob_values = [f"{row.get('probability', 0.0) * 100:.2f}%" for row in derived_rows]
        if len(prob_values) == len(display_df):
            display_df["Probability"] = prob_values
        st.markdown("### Scenario Table (with calibrated probabilities)")
        st.dataframe(display_df, use_container_width=True, hide_index=True)


DEFAULT_BUCKET_PROBS = {"A": 0.6, "B": 0.3, "C": 0.1}


def compute_historical_bucket_probabilities(lookback: int, smoothing_enabled: bool, smoothing_span: int):
    """Load cached history and compute empirical bucket probabilities."""
    history_df = load_from_cache("nifty_ohlcv")
    if history_df is None or history_df.empty:
        return DEFAULT_BUCKET_PROBS.copy(), 0, True
    df = history_df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
    if "returnPct" not in df.columns and "close" in df.columns:
        df["returnPct"] = df["close"].pct_change()
    if "ivChange" not in df.columns:
        if "ivClose" in df.columns:
            df["ivChange"] = df["ivClose"].diff()
        else:
            df["ivChange"] = 0.0
    df = df.dropna(subset=["returnPct", "ivChange"])
    if df.empty:
        return DEFAULT_BUCKET_PROBS.copy(), 0, True
    df = df.tail(lookback)
    if df.empty:
        return DEFAULT_BUCKET_PROBS.copy(), 0, True
    weights = [1.0] * len(df)
    if smoothing_enabled:
        weights = exponential_weights(len(df), smoothing_span)
    df = df.reset_index(drop=True)
    df["weight"] = weights
    df["bucket"] = df.apply(lambda row: classify_history_bucket(float(row["returnPct"]), float(row["ivChange"])), axis=1)
    counts = df.groupby("bucket")["weight"].sum().to_dict()
    total_weight = sum(counts.values())
    if total_weight <= 0:
        return DEFAULT_BUCKET_PROBS.copy(), len(df), True
    probs = {bucket: counts.get(bucket, 0.0) / total_weight for bucket in ["A", "B", "C"]}
    return probs, len(df), False


def exponential_weights(length: int, span: int) -> List[float]:
    """Return exponentially decaying weights for EWMA counts."""
    if length <= 0:
        return []
    alpha = 2.0 / (span + 1.0)
    weights = []
    for idx in range(length):
        power = length - 1 - idx
        weights.append((1 - alpha) ** power)
    return weights


def render_risk_governance_block(metrics, var99_limit, es99_limit, var95_limit, threshold_context):
    """Display ordered governance rules."""
    st.markdown("### Risk Governance")
    hard_stop_pass = metrics["VaR99"] <= var99_limit
    tail_pass = metrics["ES99"] <= es99_limit
    normal_pass = metrics["VaR95"] <= var95_limit
    scenario_pass = threshold_context.get("fail_count", 0) == 0

    statuses = [
        ("Hard Stop", hard_stop_pass, f"VaR99 {metrics['VaR99']:.2f}% ≤ {var99_limit:.2f}%"),
        ("Tail Limit", tail_pass, f"ES99 {metrics['ES99']:.2f}% ≤ {es99_limit:.2f}%"),
        ("Normal-Day Limit", normal_pass, f"VaR95 {metrics['VaR95']:.2f}% ≤ {var95_limit:.2f}%"),
        ("Scenario Caps", scenario_pass, f"Bucket breaches: {threshold_context.get('fail_count', 0)}"),
    ]

    for label, passed, detail in statuses:
        icon = "✅" if passed else "❌"
        st.markdown(f"{icon} **{label}:** {detail}")

    if not hard_stop_pass:
        decision = "FORCED REDUCE (hard stop)"
    elif not tail_pass:
        decision = "REDUCE (tail risk)"
    elif not normal_pass:
        decision = "TRIM (normal-day risk)"
    else:
        decision = "OK"

    st.markdown(f"**Governance Decision:** {decision}")


def compute_es_attribution(rows, tail_entries, tail_level: float, es_value: float):
    """Compute expected loss and ES tail attribution."""
    prob_sum = sum(row.get("probability", 0.0) for row in rows)
    expected_loss = sum(
        row.get("probability", 0.0) * row["loss_pct_nav"] for row in rows
    )
    tail_prob = 1.0 - tail_level
    tail_loss_total = sum(entry["tail_contribution"] for entry in tail_entries)

    bucket_shares = {"A": 0.0, "B": 0.0, "C": 0.0}
    scenario_entries = []

    if tail_loss_total > 0 and es_value > 0 and tail_prob > 0:
        for entry in tail_entries:
            bucket = entry["bucket"]
            share = entry["tail_contribution"] / tail_loss_total
            bucket_shares[bucket] = bucket_shares.get(bucket, 0.0) + share
            scenario_entries.append(
                {
                    "scenario": entry["scenario"],
                    "bucket": bucket,
                    "loss_pct": entry["loss_pct"],
                    "tail_prob": entry["tail_prob"],
                    "tail_contribution": entry["tail_contribution"],
                    "share_of_es": share,
                }
            )
    else:
        scenario_entries = []
    return expected_loss, prob_sum, bucket_shares, scenario_entries


def compute_needed_reduction(es_value: float, limit: float) -> float:
    """Fractional reduction needed to meet ES limit."""
    if es_value <= 0 or es_value <= limit:
        return 0.0
    return max(0.0, (es_value - limit) / es_value)


def render_rules_only():
    """Display just the rules when no positions loaded."""
    st.markdown("### Volatility Regimes")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🟢 **Low IV**")
        st.caption("< 20 IV Percentile")
    with col2:
        st.markdown("🟡 **Mid IV**")
        st.caption("20–40 IV Percentile")
    with col3:
        st.markdown("🔴 **High IV**")
        st.caption("> 40 IV Percentile")
    
    st.markdown("---")
    
    # Zone 1
    with st.expander("🟢 ZONE 1 — SAFE / PROFESSIONAL", expanded=True):
        st.markdown("**Long-term survival, minimal drawdowns**")
        zone1_df = pd.DataFrame({
            "Greek": ["Theta", "Gamma", "Vega (Low IV)", "Vega (Mid IV)", "Vega (High IV)"],
            "Range (per ₹1L)": [
                "₹120 – ₹180 / day",
                "0 to −0.020",
                "0 to −200",
                "−200 to −450",
                "−450 to −700"
            ]
        })
        st.dataframe(zone1_df, use_container_width=True, hide_index=True)
        st.caption("1% NIFTY move ≈ 1–2 days of theta")
    
    # Zone 2
    with st.expander("🟡 ZONE 2 — BALANCED / CONTROLLED", expanded=True):
        st.markdown("**Income focus with active management**")
        zone2_df = pd.DataFrame({
            "Greek": ["Theta", "Gamma", "Vega (Low IV)", "Vega (Mid IV)", "Vega (High IV)"],
            "Range (per ₹1L)": [
                "₹180 – ₹220 / day",
                "−0.020 to −0.035",
                "0 to −350",
                "−350 to −650",
                "−650 to −900"
            ]
        })
        st.dataframe(zone2_df, use_container_width=True, hide_index=True)
        st.caption("1% move ≈ 3–5 days of theta | Requires daily monitoring")
    
    # Zone 3
    with st.expander("🔴 ZONE 3 — AGGRESSIVE / FRAGILE", expanded=True):
        st.markdown("**High income, short lifespan if unmanaged**")
        zone3_df = pd.DataFrame({
            "Greek": ["Theta", "Gamma", "Vega (Low IV)", "Vega (Mid IV)", "Vega (High IV)"],
            "Range (per ₹1L)": [
                "₹220 – ₹300 / day",
                "−0.035 to −0.055",
                "❌ AVOID",
                "−650 to −1,000",
                "−1,000 to −1,300"
            ]
        })
        st.dataframe(zone3_df, use_container_width=True, hide_index=True)
        st.caption("1% move wipes 1–2 weeks of theta | NOT ALLOWED in Low IV")
