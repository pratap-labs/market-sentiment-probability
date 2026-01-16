"""Risk Analysis Tab - Zone-based portfolio risk framework with normalized greeks."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
from typing import List, Dict, Tuple

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
try:
    from py_vollib.black_scholes import black_scholes as bs
except Exception:
    bs = None

from views.tabs.derivatives_data_tab import load_from_cache as _load_from_cache


STRATEGY_MAX_DRAWDOWN = 12.0  # % NAV
DEFAULT_ES99_LIMIT = 4.0  # % NAV
VAR99_LIMIT = 5.0  # % NAV
VAR95_LIMIT = 2.5  # % NAV
DEFAULT_THRESHOLD_MASTER_PCT = 1.0
DEFAULT_THRESHOLD_HARD_STOP_PCT = 1.2
DEFAULT_THRESHOLD_NORMAL_SHARE = 0.5
DEFAULT_THRESHOLD_STRESS_SHARE = 0.9


def _load_cache_silent(data_type: str) -> pd.DataFrame:
    return _load_from_cache(data_type, silent=True)


def get_active_es_limit() -> float:
    """Return the ES99 limit currently set in session."""
    return float(st.session_state.get("strategy_es_limit", DEFAULT_ES99_LIMIT))


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


BUCKET_DEFINITIONS = [
    {
        "Bucket": "A",
        "Scenario": "Calm",
        "Move": "|Return| ≤ 0.5% and |IV proxy| ≤ 1",
        "Note": "Low-vol day; small drift and muted intraday range.",
    },
    {
        "Bucket": "B",
        "Scenario": "Normal",
        "Move": "|Return| ≤ 1.0% and |IV proxy| ≤ 2",
        "Note": "Routine session with modest moves.",
    },
    {
        "Bucket": "C",
        "Scenario": "Elevated",
        "Move": "|Return| ≤ 1.5% and |IV proxy| ≤ 3",
        "Note": "Elevated range with IV pickup.",
    },
    {
        "Bucket": "D",
        "Scenario": "Stress",
        "Move": "|Return| ≤ 2.5% and |IV proxy| ≤ 5",
        "Note": "Large range day; risk controls should trigger.",
    },
    {
        "Bucket": "E",
        "Scenario": "Gap / Tail",
        "Move": "|Return| > 2.5% or |IV proxy| > 5",
        "Note": "Tail event bucket; gap or volatility shock.",
    },
]


def render_risk_analysis_tab():
    """Render zone-based risk analysis tab."""
    
    st.markdown(
        "<h1 class='risk-big-title'>📉 Expected Shortfall (Historical-calibrated)</h1>",
        unsafe_allow_html=True,
    )

    # Sync positions button
    if st.sidebar.button("🔄 Sync Positions", key="risk_analysis_sync_positions", help="Fetch latest positions from Kite", use_container_width=True):
        st.rerun()
    
    current_es_limit = st.sidebar.number_input(
        "ES99 limit (% NAV)",
        min_value=1.0,
        max_value=12.0,
        step=0.1,
        format="%.1f",
        value=get_active_es_limit(),
        key="strategy_es_limit",
    )
    lookback_days = st.sidebar.number_input(
        "Lookback trading days",
        value=504,
        min_value=126,
        max_value=756,
        step=21,
        key="risk_es_lookback_days",
    )
    
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
    
    # Calculate IV percentile and regime before ES computation
    iv_percentile = 30  # Default
    options_df_cache = st.session_state.get("options_df_cache", pd.DataFrame())
    if not options_df_cache.empty and "iv" in options_df_cache.columns:
        current_iv = options_df_cache["iv"].median()
        iv_percentile = 35

    iv_regime, iv_color = get_iv_regime(iv_percentile)

    scenarios = get_weighted_scenarios(iv_regime)
    threshold_context = build_threshold_report(
        portfolio={
            "delta": portfolio_greeks.get("net_delta", 0.0),
            "gamma": portfolio_greeks.get("net_gamma", 0.0),
            "vega": portfolio_greeks.get("net_vega", 0.0),
            "spot": current_spot,
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
        master_pct=current_es_limit,
        hard_stop_pct=current_es_limit * 1.2,
        normal_share=DEFAULT_THRESHOLD_NORMAL_SHARE,
        stress_share=DEFAULT_THRESHOLD_STRESS_SHARE,
    )

    st.markdown("### Bucket Definitions")
    st.dataframe(pd.DataFrame(BUCKET_DEFINITIONS), use_container_width=True, hide_index=True)

    render_expected_shortfall_panel(
        threshold_context.get("rows", []),
        account_size,
        scenarios,
        key_prefix="risk_",
        es_limit=current_es_limit,
        lookback_override=int(lookback_days),
    )

    st.markdown("### Scenario Table (with calibrated probabilities)")
    derived_rows = threshold_context.get("rows", [])
    repriced_map: Dict[str, Dict[str, object]] = {}
    repriced_skipped = {"missing_fields": 0, "no_price": 0}
    if bs is not None:
        positions = st.session_state.get("enriched_positions", [])
        if positions:
            repriced_rows, repriced_skipped = _repriced_scenario_rows(
                positions,
                scenarios,
                current_spot,
                account_size,
            )
            repriced_map = {row["Scenario"]: row for row in repriced_rows}
    def _parse_inr(val: str) -> float:
        cleaned = str(val).replace("₹", "").replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return 0.0

    derived_rows = sorted(
        derived_rows,
        key=lambda row: _parse_inr(
            repriced_map.get(row["scenario"]["name"], {}).get("Repriced P&L (₹)", 0.0)
        ),
        reverse=False,
    )
    if derived_rows:
        table_rows = []
        for row in derived_rows:
            scenario = row["scenario"]
            status = row.get("status", "INFO")
            if status == "PASS":
                status_display = "🟢 PASS"
            elif status == "FAIL":
                status_display = "🔴 FAIL"
            else:
                status_display = "ℹ️ INFO"
            repriced = repriced_map.get(scenario["name"])
            table_rows.append(
                {
                    "Scenario": scenario["name"],
                    "Bucket": row["bucket"],
                    "dS% / dIV": f"{scenario['dS_pct']:+.2f}% / {scenario['dIV_pts']:+.1f}",
                    "Δ P&L (₹)": format_inr(row["pnl_delta"]),
                    "Γ P&L (₹)": format_inr(row["pnl_gamma"]),
                    "Vega P&L (₹)": format_inr(row["pnl_vega"]),
                    "Total P&L (₹)": format_inr(row["pnl_total"]),
                    "Repriced P&L (₹)": repriced.get("Repriced P&L (₹)") if repriced else "—",
                    "Repriced Loss % NAV": repriced.get("Repriced Loss % NAV") if repriced else "—",
                    "Loss % NAV": f"{row['loss_pct_nav']:.2f}%",
                    "Threshold % NAV": f"{row['threshold_pct']:.2f}%",
                    "Probability": f"{row.get('probability', 0.0) * 100:.2f}%",
                    "Status": status_display,
                }
            )
        scenario_df = pd.DataFrame(table_rows)

        def _status_bg(val: str) -> str:
            val = str(val).upper()
            if "FAIL" in val:
                return "background-color: #f8d7da; color: #842029;"
            if "PASS" in val:
                return "background-color: #d1e7dd; color: #0f5132;"
            return ""

        styled = scenario_df.style.applymap(_status_bg, subset=["Status"])
        st.dataframe(styled, use_container_width=True, hide_index=True, height=320)
        if repriced_skipped["missing_fields"] or repriced_skipped["no_price"]:
            st.caption(
                f"Repricing skipped: {repriced_skipped['missing_fields']} missing fields, "
                f"{repriced_skipped['no_price']} missing prices."
            )
    else:
        st.info("No scenario rows available to display.")

    # ========== SECTION 1: CURRENT PORTFOLIO ANALYSIS ==========
    st.markdown("---")
    st.markdown(
        "<h1 class='risk-big-title'>🧭 Zone-Based Testing</h1>",
        unsafe_allow_html=True,
    )
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


def render_expected_shortfall_panel(
    derived_rows,
    account_size,
    scenarios,
    key_prefix: str = "",
    es_limit: float = DEFAULT_ES99_LIMIT,
    lookback_override: int = None,
):
    """Render probability-weighted ES/VaR panel."""

    if not derived_rows:
        st.info("No scenario rows available for ES calculation.")
        return

    lookback = lookback_override or 504

    bucket_probs, history_count, used_fallback = compute_historical_bucket_probabilities(
        lookback=int(lookback),
        smoothing_enabled=False,
        smoothing_span=63,
    )

    st.markdown("### Bucket Probabilities")
    prob_cols = st.columns(6)
    prob_cols[0].metric("Lookback", f"{history_count}d")
    prob_cols[1].metric("Bucket A", f"{bucket_probs['A']*100:.1f}%")
    prob_cols[2].metric("Bucket B", f"{bucket_probs['B']*100:.1f}%")
    prob_cols[3].metric("Bucket C", f"{bucket_probs['C']*100:.1f}%")
    prob_cols[4].metric("Bucket D", f"{bucket_probs['D']*100:.1f}%")
    prob_cols[5].metric("Bucket E", f"{bucket_probs['E']*100:.1f}%")
    # Fallback message suppressed per UX request.

    bucket_colors = {
        "A": "#2ca02c",
        "B": "#98df8a",
        "C": "#ffbf00",
        "D": "#ff7f0e",
        "E": "#d62728",
    }

    history_df = _build_bucket_history(lookback)
    if not history_df.empty:
        st.markdown("### NIFTY Lookback Bucket Map")
        drift_pct = history_df["returnPct"] * 100
        bins = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        labels = [
            "-10 to -9",
            "-9 to -8",
            "-8 to -7",
            "-7 to -6",
            "-6 to -5",
            "-5 to -4",
            "-4 to -3",
            "-3 to -2",
            "-2 to -1",
            "-1 to -0.5",
            "-0.5 to 0.5",
            "0.5 to 1",
            "1 to 2",
            "2 to 3",
            "3 to 4",
            "4 to 5",
            "5 to 6",
            "6 to 7",
            "7 to 8",
            "8 to 9",
            "9 to 10",
        ]
        drift_bins = pd.cut(drift_pct, bins=bins, labels=labels, include_lowest=True)
        drift_counts = drift_bins.value_counts().reindex(labels, fill_value=0)
        drift_df = drift_counts.rename_axis("Drift (%)").reset_index(name="Count")
        bin_midpoints = []
        for label in drift_df["Drift (%)"]:
            parts = str(label).replace("to", "").split()
            try:
                low = float(parts[0])
                high = float(parts[1])
                bin_midpoints.append((low + high) / 2.0)
            except Exception:
                bin_midpoints.append(0.0)
        drift_df["bucket"] = [
            classify_history_bucket(mid / 100.0, 0.0) for mid in bin_midpoints
        ]
        count_chart = px.bar(
            drift_df,
            x="Drift (%)",
            y="Count",
            title="Raw Drift Counts (Lookback)",
            labels={"Count": "Days"},
            text="Count",
            color="bucket",
            color_discrete_map=bucket_colors,
        )
        count_chart.update_traces(textposition="outside")
        count_chart.update_layout(
            xaxis_tickangle=-45,
            xaxis=dict(categoryorder="array", categoryarray=labels),
        )
        st.plotly_chart(
            count_chart,
            use_container_width=True,
            key=f"{key_prefix}bucket_drift_chart_{lookback}",
        )

        ohlc_fig = make_subplots(rows=1, cols=1)
        for bucket, color in bucket_colors.items():
            subset = history_df[history_df["bucket"] == bucket]
            if subset.empty:
                continue
            ohlc_fig.add_trace(
                go.Candlestick(
                    x=subset["date"],
                    open=subset["open"],
                    high=subset["high"],
                    low=subset["low"],
                    close=subset["close"],
                    name=f"Bucket {bucket}",
                    increasing_line_color=color,
                    decreasing_line_color=color,
                )
            )
        ohlc_fig.update_layout(
            title="NIFTY OHLC with Bucket Classification",
            showlegend=True,
            xaxis_rangeslider_visible=True,
            dragmode="pan",
        )
        st.plotly_chart(
            ohlc_fig,
            use_container_width=True,
            key=f"{key_prefix}bucket_ohlc_chart_{lookback}",
        )
    else:
        st.info("No NIFTY history available to render bucket distribution.")

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
    status_info = evaluate_strategy_status(metrics, es_limit)

    render_risk_governance_block(metrics, None, status_info, es_limit)

    summary_cols = st.columns(4)
    summary_cols[0].metric("ES99", f"{metrics['ES99']:.2f}%", format_inr(metrics["ES99Value"]))
    summary_cols[1].metric("VaR99", f"{metrics['VaR99']:.2f}%", format_inr(metrics["VaR99Value"]))
    summary_cols[2].metric("ES95", f"{metrics['ES95']:.2f}%", format_inr(metrics["ES95Value"]))
    summary_cols[3].metric("VaR95", f"{metrics['VaR95']:.2f}%", format_inr(metrics["VaR95Value"]))

    status_label, status_icon, detail_text = status_info
    st.markdown(f"**Risk Status:** {status_icon} {status_label}")
    st.caption(detail_text)
    if status_label == "REDUCE SIZE" and metrics["ES99"] > 0:
        scale = es_limit / max(metrics["ES99"], 1e-6)
        st.warning(f"To pass ES99, reduce position size to {scale*100:.0f}% of current.")

    tail_set = metrics.get("tail_set_99", [])
    expected_loss, prob_sum, bucket_shares, scenario_tail_contribs = compute_es_attribution(
        derived_rows, tail_set, tail_level=0.99, es_value=metrics["ES99"]
    )

    st.caption(f"Expected loss (portfolio mean): {expected_loss:.3f}% NAV")

    if metrics["ES99"] > 0 and bucket_shares:
        share_text = "ES99 driven by: " + ", ".join(
            [
                f"{bucket} {bucket_shares.get(bucket, 0.0) * 100:.1f}%"
                for bucket in ["A", "B", "C", "D", "E"]
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

    if metrics["ES99"] > es_limit and scenario_tail_contribs:
        dominant_bucket = max(bucket_shares, key=bucket_shares.get)
        dominant_entry = max(scenario_tail_contribs, key=lambda x: x["tail_contribution"])
        reduction_needed = compute_needed_reduction(metrics["ES99"], es_limit)
        st.warning(
            f"Primary driver: Bucket {dominant_bucket} / {dominant_entry['scenario']['name']}.\n"
            f"Required ES99 reduction: {reduction_needed * 100:.1f}%. Reduce Bucket {dominant_bucket} tail risk to pass ES99."
        )

    # Scenario table is rendered in the parent tab using derived rows.


def _build_bucket_history(lookback: int) -> pd.DataFrame:
    history_df = _load_cache_silent("nifty_ohlcv")
    if history_df is None or history_df.empty:
        return pd.DataFrame()
    df = history_df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
    if "returnPct" not in df.columns and "close" in df.columns:
        df["returnPct"] = df["close"].pct_change()
    if "ivChange" not in df.columns:
        if "high" in df.columns and "low" in df.columns and "close" in df.columns:
            df["intraday_vol"] = (df["high"] - df["low"]) / df["close"] * 100
            df["ivChange"] = df["intraday_vol"].diff()
        else:
            df["ivChange"] = 0.0
    df = df.dropna(subset=["returnPct", "ivChange", "close"])
    if df.empty:
        return pd.DataFrame()
    df = df.tail(lookback).reset_index(drop=True)
    df["bucket"] = df.apply(
        lambda row: classify_history_bucket(float(row["returnPct"]), float(row["ivChange"])), axis=1
    )
    return df


def _repriced_scenario_rows(
    positions: List[Dict[str, object]],
    scenarios: List[object],
    spot: float,
    capital: float,
    risk_free_rate: float = 0.07,
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    rows: List[Dict[str, object]] = []
    skipped = {"missing_fields": 0, "no_price": 0}
    if not positions or spot <= 0:
        return rows, skipped

    for scenario in scenarios:
        scenario_spot = spot * (1 + scenario.ds_pct / 100.0)
        scenario_iv_shift = scenario.div_pts / 100.0
        total_pnl = 0.0
        used_positions = 0

        for pos in positions:
            option_type = pos.get("option_type")
            strike = pos.get("strike")
            tte = pos.get("time_to_expiry")
            iv = pos.get("implied_vol")
            last_price = pos.get("last_price", 0.0)
            qty = pos.get("quantity", 0)

            if option_type not in {"CE", "PE"}:
                continue
            if strike is None or tte is None or iv is None:
                skipped["missing_fields"] += 1
                continue
            if last_price is None:
                skipped["no_price"] += 1
                continue

            shock_iv = max(float(iv) + scenario_iv_shift, 0.0001)
            flag = "c" if option_type == "CE" else "p"
            try:
                new_price = bs(flag, scenario_spot, float(strike), float(tte), risk_free_rate, shock_iv)
            except Exception:
                continue

            total_pnl += (new_price - float(last_price)) * float(qty)
            used_positions += 1

        loss_pct = (-total_pnl / capital * 100.0) if total_pnl < 0 and capital > 0 else 0.0
        rows.append(
            {
                "Scenario": scenario.name,
                "Repriced P&L (₹)": format_inr(total_pnl),
                "Repriced Loss % NAV": f"{loss_pct:.2f}%",
                "Positions Used": used_positions,
            }
        )

    return rows, skipped


DEFAULT_BUCKET_PROBS = {"A": 0.45, "B": 0.25, "C": 0.15, "D": 0.10, "E": 0.05}


def compute_historical_bucket_probabilities(lookback: int, smoothing_enabled: bool, smoothing_span: int):
    """Load cached history and compute empirical bucket probabilities.
    
    Uses intraday volatility (high-low range) as proxy for IV changes since
    KiteConnect OHLCV data doesn't include implied volatility.
    """
    history_df = _load_cache_silent("nifty_ohlcv")
    if history_df is None or history_df.empty:
        return DEFAULT_BUCKET_PROBS.copy(), 0, True
    df = history_df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
    
    # Calculate daily return percentage
    if "returnPct" not in df.columns and "close" in df.columns:
        df["returnPct"] = df["close"].pct_change()
    
    # Use intraday range as volatility proxy (no IV data available from KiteConnect)
    if "ivChange" not in df.columns:
        if "high" in df.columns and "low" in df.columns and "close" in df.columns:
            # Normalized high-low range as volatility measure
            df["intraday_vol"] = (df["high"] - df["low"]) / df["close"] * 100
            df["ivChange"] = df["intraday_vol"].diff()
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
    probs = {bucket: counts.get(bucket, 0.0) / total_weight for bucket in ["A", "B", "C", "D", "E"]}
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


def evaluate_strategy_status(metrics, es_limit: float = DEFAULT_ES99_LIMIT):
    """Return status tuple (label, icon, detail) based on fixed strategy limits."""
    var99 = metrics.get("VaR99", 0.0)
    es99 = metrics.get("ES99", 0.0)
    var95 = metrics.get("VaR95", 0.0)

    if var99 > VAR99_LIMIT:
        detail = f"VaR99 {var99:.2f}% exceeds hard stop {VAR99_LIMIT:.1f}%."
        return "FORCED REDUCE", "🛑", detail
    if es99 > es_limit:
        detail = f"ES99 {es99:.2f}% exceeds limit {es_limit:.1f}%."
        return "REDUCE SIZE", "🔴", detail
    if var95 > VAR95_LIMIT:
        detail = f"VaR95 {var95:.2f}% exceeds limit {VAR95_LIMIT:.1f}%."
        return "TRIM RISK", "🟡", detail
    detail = (
        f"Within limits: VaR99 {var99:.2f}% ≤ {VAR99_LIMIT:.1f}%, "
        f"ES99 {es99:.2f}% ≤ {es_limit:.1f}%, "
        f"VaR95 {var95:.2f}% ≤ {VAR95_LIMIT:.1f}%."
    )
    return "OK", "🟢", detail


def render_risk_governance_block(metrics, threshold_context, status_info=None, es_limit: float = DEFAULT_ES99_LIMIT):
    """Display simplified governance using fixed single-strategy limits."""
    if status_info is None:
        status_info = evaluate_strategy_status(metrics, es_limit)

    status_label, status_icon, detail = status_info
    st.markdown(f"**Status:** {status_icon} {status_label}")
    st.caption(detail)

    if status_label == "REDUCE SIZE" and metrics["ES99"] > 0:
        scale = es_limit / max(metrics["ES99"], 1e-6)
        st.warning(f"To pass ES99, reduce position size to {scale*100:.0f}% of current.")


def compute_es_attribution(rows, tail_entries, tail_level: float, es_value: float):
    """Compute expected loss and ES tail attribution."""
    prob_sum = sum(row.get("probability", 0.0) for row in rows)
    expected_loss = sum(
        row.get("probability", 0.0) * row["loss_pct_nav"] for row in rows
    )
    tail_prob = 1.0 - tail_level
    tail_loss_total = sum(entry["tail_contribution"] for entry in tail_entries)

    bucket_shares = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0}
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
