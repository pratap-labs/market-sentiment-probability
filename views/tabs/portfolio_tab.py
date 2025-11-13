"""Portfolio overview tab with comprehensive metrics."""

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

from scripts.utils import (
    calculate_portfolio_greeks,
    calculate_var,
    calculate_stress_pnl,
    format_inr,
    DEFAULT_LOT_SIZE
)


def render_portfolio_tab():
    """Render portfolio overview tab with complete metrics."""
    st.subheader("📈 Portfolio Overview")
    
    if "enriched_positions" not in st.session_state:
        st.info("No positions loaded. Fetch positions from the Positions tab first.")
        return
    
    enriched = st.session_state["enriched_positions"]
    portfolio_greeks = calculate_portfolio_greeks(enriched)
    current_spot = st.session_state.get("current_spot", 25000)
    
    # Get account info from Kite (margins)
    access_token = st.session_state.get("kite_access_token")
    kite_api_key = st.session_state.get("kite_api_key")
    
    margin_available = 1150000  # Default, will fetch from Kite
    margin_used = 1300000  # Default, will fetch from Kite
    
    if access_token and kite_api_key:
        try:
            kite = KiteConnect(api_key=kite_api_key)
            kite.set_access_token(access_token)
            margins = kite.margins()
            
            equity_margins = margins.get("equity", {})
            margin_available = equity_margins.get("available", {}).get("live_balance", 500000)
            margin_used = equity_margins.get("utilised", {}).get("debits", 0)
        except:
            pass  # Use defaults if fetch fails
    
    account_size = margin_available + margin_used
    
    # Calculate portfolio metrics
    total_pnl = sum(p.get("pnl", 0) for p in enriched)
    total_theta = portfolio_greeks["net_theta"]
    total_delta = portfolio_greeks["net_delta"]
    total_gamma = portfolio_greeks["net_gamma"]
    total_vega = portfolio_greeks["net_vega"]
    
    # Days to recover
    days_to_recover = abs(total_pnl / total_theta) if total_theta != 0 else 999
    
    # Theta efficiency
    theta_efficiency = (total_pnl / total_theta * 100) if total_theta != 0 else 0
    
    # ROI calculations
    roi_pct = (total_pnl / account_size * 100) if account_size > 0 else 0
    
    # Assume average DTE for annualization
    avg_dte = sum(p.get("dte", 0) for p in enriched) / len(enriched) if enriched else 30
    days_in_trade = max(30 - avg_dte, 1)  # Approximate
    roi_annualized = (total_pnl / account_size) / (days_in_trade / 365) * 100 if account_size > 0 else 0
    
    # Delta dollars
    delta_dollars = total_delta * current_spot
    
    # Notional exposure
    notional_exposure = sum(
        abs(p.get("quantity", 0)) * p.get("strike", 0) 
        for p in enriched
    )
    
    # Leverage ratio (Notional / Account)
    leverage_ratio = (notional_exposure / account_size) if account_size > 0 else 0
    
    # Margin utilization
    margin_util_pct = (margin_used / account_size * 100) if account_size > 0 else 0
    
    # Theta as % of capital
    theta_pct_capital = (abs(total_theta) / account_size * 100) if account_size > 0 else 0
    
    # ========== SECTION 1: CAPITAL & PERFORMANCE ==========
    st.markdown("### 💰 Capital & Performance")

    # First row: 4 metrics per row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Account Size", format_inr(account_size))

    with col2:
        margin_color = "🟢" if margin_util_pct < 50 else ("🟡" if margin_util_pct < 70 else "🔴")
        st.metric("Margin Used", format_inr(margin_used), f"{margin_util_pct:.1f}% {margin_color}")

    with col3:
        pnl_color = "inverse" if total_pnl < 0 else "normal"
        st.metric("Net P&L", format_inr(total_pnl), f"{roi_pct:.2f}%")

    with col4:
        st.metric("Theta/Day", format_inr(total_theta), f"{theta_pct_capital:.2f}% of capital")

    # Second row: 4 columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("ROI (Annualized)", f"{roi_annualized:.1f}%")

    with col2:
        recovery_color = "🔴" if days_to_recover > avg_dte else ("🟡" if days_to_recover > avg_dte * 0.5 else "🟢")
        st.metric("Days to Recover", f"{days_to_recover:.1f} {recovery_color}", f"Avg DTE: {avg_dte:.0f}")

    with col3:
        eff_color = "🔴" if theta_efficiency < -200 else ("🟡" if theta_efficiency < -100 else "🟢")
        st.metric("Theta Efficiency", f"{theta_efficiency:.0f}% {eff_color}")

    with col4:
        leverage_color = "🟢" if leverage_ratio < 50 else ("🟡" if leverage_ratio < 100 else "🔴")
        st.metric("Notional Exposure", format_inr(notional_exposure), f"{leverage_ratio:.0f} × capital {leverage_color}")
    
    # ========== SECTION 2: GREEKS & RISK ==========
    st.markdown("### 📊 Greeks & Risk")
    st.caption(f"Delta conversions assume lot size = {DEFAULT_LOT_SIZE} for ₹/pt calculation. If your positions already include lot-size in total_delta, ₹/pt equals Net Delta.")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta_abs = abs(total_delta)
        delta_status = "🟢" if delta_abs < 40 else ("🟡" if delta_abs < 100 else "🔴")
        st.metric("Net Delta (units)", f"{total_delta:.2f} {delta_status}")

    # Compute presentation-only conversions (no greeks math changed)
    # total_delta already includes lot-size scaling (positions are stored scaled),
    # so do NOT multiply by DEFAULT_LOT_SIZE here — use total_delta directly.
    rupee_per_point_by_lot = total_delta
    delta_notional = total_delta * current_spot

    with col2:
        st.metric("Delta (₹/pt)", format_inr(rupee_per_point_by_lot))

    with col3:
        gamma_status = "⚠️" if total_gamma < -0.5 else ""
        st.metric("Net Gamma", f"{total_gamma:.3f} {gamma_status}")

    with col4:
        st.metric("Net Vega", format_inr(total_vega))

    # Separate row for Delta Notional and Vega % Capital to keep 4-per-row layout consistent
    col1, col2, col3, col4 = st.columns(4)
    # percent of account for Delta Notional (use abs for percent sizing)
    delta_notional_pct = (abs(delta_notional) / account_size * 100) if account_size > 0 else 0
    vega_pct = (abs(total_vega) / account_size * 100) if account_size > 0 else 0
    with col1:
        st.metric("Delta Notional (₹)", format_inr(delta_notional), f"{delta_notional_pct:.1f}% of account")
    with col2:
        st.metric("Vega % Capital", f"{vega_pct:.2f}%")
    
    # Greeks breakdown chart
    # st.markdown("#### Greeks Breakdown")
    greeks_data = pd.DataFrame({
        "Greek": ["Delta", "Gamma×100", "Vega/100", "Theta"],
        "Value": [total_delta, total_gamma * 100, total_vega / 100, total_theta]
    })
    # st.bar_chart(greeks_data.set_index("Greek"))
    
    # ========== SECTION 3: RISK ANALYSIS ==========
    st.markdown("### ⚠️ Risk Analysis")
    
    # Get NIFTY data for VaR calculation (uses realized volatility)
    nifty_df = st.session_state.get("nifty_df", pd.DataFrame())
    
    # Calculate VaR (uses realized volatility from NIFTY data)
    var_95 = calculate_var(enriched, current_spot, nifty_df)
    
    # Stress test scenarios
    stress_up_2 = calculate_stress_pnl(enriched, current_spot, 1.02, 0)
    stress_down_2 = calculate_stress_pnl(enriched, current_spot, 0.98, 0)
    stress_iv_up = calculate_stress_pnl(enriched, current_spot, 1.0, 0.05)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("VaR (95%)", format_inr(var_95))
    
    with col2:
        st.metric("+2% NIFTY Move", format_inr(stress_up_2))
    
    with col3:
        st.metric("-2% NIFTY Move", format_inr(stress_down_2))
    
    with col4:
        st.metric("+5 IV Points", format_inr(stress_iv_up))
    
    # Risk status summary
    st.markdown("#### Risk Status Summary")
    
    risk_alerts = []
    
    # Delta check
    if abs(total_delta) > 100:
        risk_alerts.append("🔴 **CRITICAL**: Net Delta > ±100 - High directional risk")
    elif abs(total_delta) > 40:
        risk_alerts.append("🟡 **WARNING**: Net Delta > ±40 - Monitor directional exposure")
    else:
        risk_alerts.append("🟢 **OK**: Delta is neutral")
    
    # Margin check
    if margin_util_pct > 80:
        risk_alerts.append("🔴 **CRITICAL**: Margin utilization > 70% - Limited room for adjustments")
    elif margin_util_pct > 50:
        risk_alerts.append("🟡 **WARNING**: Margin utilization > 50%")
    else:
        risk_alerts.append("🟢 **OK**: Margin utilization healthy")
    
    # Recovery check
    if days_to_recover > avg_dte:
        risk_alerts.append("🔴 **CRITICAL**: Cannot recover losses by expiry with current theta")
    elif days_to_recover > avg_dte * 0.7:
        risk_alerts.append("🟡 **WARNING**: Tight timeline to recover losses")
    else:
        risk_alerts.append("🟢 **OK**: Recovery timeline manageable")
    
    # Theta efficiency check
    if theta_efficiency < -200:
        risk_alerts.append("🔴 **CRITICAL**: Theta efficiency < -200% - Directional problem, not time decay")
    elif theta_efficiency < -100:
        risk_alerts.append("🟡 **WARNING**: Theta efficiency negative")
    
    # Gamma check
    if total_gamma < -0.5 and avg_dte < 7:
        risk_alerts.append("🔴 **CRITICAL**: High negative gamma near expiry - Risk of rapid delta changes")
    
    for alert in risk_alerts:
        st.markdown(alert)
    
    # ========== SECTION 4: POSITION CONCENTRATION ==========
    st.markdown("### 🎯 Position Concentration")
    
    # Group by expiry
    expiry_groups = {}
    for pos in enriched:
        expiry = pos.get("expiry")
        if expiry:
            expiry_str = expiry.strftime("%Y-%m-%d")
            if expiry_str not in expiry_groups:
                expiry_groups[expiry_str] = {"count": 0, "pnl": 0, "notional": 0}
            expiry_groups[expiry_str]["count"] += 1
            expiry_groups[expiry_str]["pnl"] += pos.get("pnl", 0)
            expiry_groups[expiry_str]["notional"] += abs(pos.get("quantity", 0)) * pos.get("strike", 0) 
    
    # calculate leverage per expiry
    for exp, data in expiry_groups.items():
        data["leverage"] = (data["notional"] / account_size) if account_size > 0 else 0


    if expiry_groups:
        expiry_df = pd.DataFrame([
            {
                "Expiry": exp,
                "Positions": data["count"],
                "P&L": data["pnl"],
                "Notional": f"{data['notional']}",
                "Leverage (× capital)": f"{data['leverage']:.0f}"

            }
            for exp, data in expiry_groups.items()
        ])
        # Format currency columns using Indian grouping before display
        if "P&L" in expiry_df.columns:
            expiry_df["P&L"] = expiry_df["P&L"].apply(lambda x: format_inr(x))
        if "Notional" in expiry_df.columns:
            expiry_df["Notional"] = expiry_df["Notional"].apply(lambda x: format_inr(x))

        st.dataframe(expiry_df, use_container_width=True)
    
    # Largest positions
    st.markdown("#### Largest Positions by Notional")
    positions_by_size = sorted(
        enriched,
        key=lambda p: abs(p.get("quantity", 0)) * p.get("strike", 0),
        reverse=True
    )[:5]
    
    for pos in positions_by_size:
        notional = abs(pos.get("quantity", 0)) * pos.get("strike", 0)
        pct_portfolio = (notional / account_size * 100) if account_size > 0 else 0
        st.write(f"- {pos.get('tradingsymbol')}: {format_inr(notional)} ({pct_portfolio:.1f}% of portfolio)")
