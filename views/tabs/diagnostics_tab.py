"""Position diagnostics tab with action signals and strategy analysis."""

import streamlit as st
import pandas as pd
from typing import Dict, Tuple, List
from datetime import datetime
from collections import defaultdict

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
    format_inr,
    get_action_signal,
    get_action_recommendation
)


def detect_iron_condor_strategies(positions: List[Dict]) -> Dict[str, List[Dict]]:
    """Detect Iron Condor strategies by grouping positions by expiry.
    
    An Iron Condor consists of:
    - 1 short call + 1 long call (higher strike) = Call Credit Spread
    - 1 short put + 1 long put (lower strike) = Put Credit Spread
    
    Returns: Dict mapping expiry -> list of strategy dicts
    """
    # Group positions by expiry
    by_expiry = defaultdict(list)
    for pos in positions:
        expiry = pos.get("expiry")
        if expiry:
            expiry_key = expiry.strftime("%Y-%m-%d") if hasattr(expiry, 'strftime') else str(expiry)
            by_expiry[expiry_key].append(pos)
    
    strategies = {}
    
    for expiry_key, expiry_positions in by_expiry.items():
        # Separate calls and puts
        calls = [p for p in expiry_positions if p.get("option_type") == "CE"]
        puts = [p for p in expiry_positions if p.get("option_type") == "PE"]
        
        # Sort by strike
        calls = sorted(calls, key=lambda x: x.get("strike", 0))
        puts = sorted(puts, key=lambda x: x.get("strike", 0))
        
        # Detect call spread (short call + long call)
        call_spread = None
        if len(calls) >= 2:
            short_calls = [c for c in calls if c.get("quantity", 0) < 0]
            long_calls = [c for c in calls if c.get("quantity", 0) > 0]
            
            if short_calls and long_calls:
                # Match short call with higher strike long call
                for sc in short_calls:
                    for lc in long_calls:
                        if lc.get("strike", 0) > sc.get("strike", 0):
                            call_spread = {
                                "type": "Call Credit Spread",
                                "short": sc,
                                "long": lc,
                                "short_strike": sc.get("strike"),
                                "long_strike": lc.get("strike"),
                                "width": lc.get("strike", 0) - sc.get("strike", 0),
                                "net_qty": abs(sc.get("quantity", 0))
                            }
                            break
                    if call_spread:
                        break
        
        # Detect put spread (short put + long put)
        put_spread = None
        if len(puts) >= 2:
            short_puts = [p for p in puts if p.get("quantity", 0) < 0]
            long_puts = [p for p in puts if p.get("quantity", 0) > 0]
            
            if short_puts and long_puts:
                # Match short put with lower strike long put
                for sp in short_puts:
                    for lp in long_puts:
                        if lp.get("strike", 0) < sp.get("strike", 0):
                            put_spread = {
                                "type": "Put Credit Spread",
                                "short": sp,
                                "long": lp,
                                "short_strike": sp.get("strike"),
                                "long_strike": lp.get("strike"),
                                "width": sp.get("strike", 0) - lp.get("strike", 0),
                                "net_qty": abs(sp.get("quantity", 0))
                            }
                            break
                    if put_spread:
                        break
        
        # Combine into Iron Condor if both spreads exist
        if call_spread and put_spread:
            strategies[expiry_key] = [{
                "strategy": "Iron Condor",
                "call_spread": call_spread,
                "put_spread": put_spread,
                "all_positions": expiry_positions,
                "dte": expiry_positions[0].get("dte", 0) if expiry_positions else 0
            }]
        elif call_spread:
            strategies[expiry_key] = [{
                "strategy": "Call Credit Spread",
                "call_spread": call_spread,
                "all_positions": expiry_positions,
                "dte": expiry_positions[0].get("dte", 0) if expiry_positions else 0
            }]
        elif put_spread:
            strategies[expiry_key] = [{
                "strategy": "Put Credit Spread",
                "put_spread": put_spread,
                "all_positions": expiry_positions,
                "dte": expiry_positions[0].get("dte", 0) if expiry_positions else 0
            }]
        else:
            # Individual positions
            strategies[expiry_key] = [{
                "strategy": "Individual Positions",
                "all_positions": expiry_positions,
                "dte": expiry_positions[0].get("dte", 0) if expiry_positions else 0
            }]
    
    return strategies


def analyze_strategy(strategy_data: Dict, current_spot: float) -> Dict:
    """Analyze a strategy (Iron Condor, spreads, or individual positions) and provide recommendations.
    
    Returns analysis with P&L, Greeks, status, and action recommendations.
    """
    strategy_type = strategy_data.get("strategy", "Unknown")
    all_positions = strategy_data.get("all_positions", [])
    
    if not all_positions:
        return {}
    
    # Aggregate metrics
    total_pnl = sum(p.get("pnl", 0) for p in all_positions)
    net_delta = sum(p.get("position_delta", 0) for p in all_positions)
    net_gamma = sum(p.get("position_gamma", 0) for p in all_positions)
    net_vega = sum(p.get("position_vega", 0) for p in all_positions)
    net_theta = sum(p.get("position_theta", 0) for p in all_positions)
    dte = strategy_data.get("dte", 0)
    
    # Calculate metrics
    days_to_breakeven = abs(total_pnl / net_theta) if net_theta != 0 else 999
    theta_eff = (total_pnl / net_theta * 100) if net_theta != 0 else 0
    
    # Status
    if days_to_breakeven > dte and dte > 0:
        status = "🔴 Cannot Recover"
        priority = 9
        action = "CLOSE/ROLL"
    elif days_to_breakeven > dte * 0.7:
        status = "🟡 At Risk"
        priority = 5
        action = "MONITOR"
    elif total_pnl > 0 and theta_eff > 50:
        status = "🟢 Take Profit"
        priority = 3
        action = "CONSIDER CLOSING"
    else:
        status = "🟢 On Track"
        priority = 1
        action = "HOLD"
    
    # Strategy-specific analysis
    analysis = {
        "strategy": strategy_type,
        "dte": dte,
        "total_pnl": total_pnl,
        "net_delta": net_delta,
        "net_gamma": net_gamma,
        "net_vega": net_vega,
        "net_theta": net_theta,
        "days_to_breakeven": days_to_breakeven,
        "theta_eff": theta_eff,
        "status": status,
        "priority": priority,
        "action": action,
        "num_positions": len(all_positions)
    }
    
    # Iron Condor specific metrics
    if strategy_type == "Iron Condor":
        call_spread = strategy_data.get("call_spread", {})
        put_spread = strategy_data.get("put_spread", {})
        
        call_short_strike = call_spread.get("short_strike", 0)
        put_short_strike = put_spread.get("short_strike", 0)
        
        # Distance to short strikes
        call_distance = ((call_short_strike - current_spot) / current_spot * 100) if current_spot > 0 else 0
        put_distance = ((current_spot - put_short_strike) / current_spot * 100) if current_spot > 0 else 0
        
        # Minimum distance to either short strike (safety margin)
        min_distance = min(abs(call_distance), abs(put_distance))
        
        # Width of spreads
        call_width = call_spread.get("width", 0)
        put_width = put_spread.get("width", 0)
        
        # Max risk per side
        max_risk_call = call_width * abs(call_spread.get("net_qty", 0))
        max_risk_put = put_width * abs(put_spread.get("net_qty", 0))
        max_risk_total = max_risk_call + max_risk_put
        
        # Risk/reward
        credit_received = abs(total_pnl) if total_pnl < 0 else 0
        risk_reward = (max_risk_total / credit_received) if credit_received > 0 else 0
        
        analysis.update({
            "call_short_strike": call_short_strike,
            "put_short_strike": put_short_strike,
            "call_distance_pct": call_distance,
            "put_distance_pct": put_distance,
            "min_distance_pct": min_distance,
            "call_width": call_width,
            "put_width": put_width,
            "max_risk": max_risk_total,
            "risk_reward": risk_reward
        })
        
        # Adjust priority based on distance to short strikes
        if min_distance < 2:  # Within 2% of either short strike
            analysis["priority"] = max(analysis["priority"], 8)
            analysis["status"] = "🔴 Danger Zone"
            analysis["action"] = "CLOSE/ADJUST"
        elif min_distance < 5:  # Within 5%
            analysis["priority"] = max(analysis["priority"], 6)
            analysis["status"] = "🟡 Close to Short Strike"
            analysis["action"] = "MONITOR CLOSELY"
    
    return analysis


def render_diagnostics_tab():
    """Render position diagnostics tab with strategy analysis."""
    st.subheader("🔍 Position Diagnostics & Strategy Analysis")
    
    if "enriched_positions" not in st.session_state:
        st.info("No positions loaded. Fetch positions from the Positions tab first.")
        return
    
    enriched = st.session_state["enriched_positions"]
    
    if not enriched:
        st.info("No positions to analyze.")
        return
    
    current_spot = st.session_state.get("current_spot", 24000)
    
    # Validate spot price
    if current_spot == 0 or pd.isna(current_spot):
        st.error("⚠️ Invalid spot price (0 or NaN). Please refresh positions.")
        current_spot = 24000
        st.warning(f"Using default spot price: ₹{current_spot:,.2f}")
    
    st.info(f"📊 Current NIFTY Spot: **₹{current_spot:,.2f}**")
    
    # Detect strategies
    strategies = detect_iron_condor_strategies(enriched)
    
    # Get unique expiries for filter
    expiries = sorted(strategies.keys())
    
    # Add expiry filter
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        selected_expiry = st.selectbox(
            "📅 Select Expiry to Analyze",
            ["All Expiries"] + expiries,
            help="Filter positions by expiry date"
        )
    
    with col2:
        view_mode = st.radio(
            "View Mode",
            ["Strategy View", "Individual Positions"],
            horizontal=True,
            help="Strategy View groups positions into spreads/Iron Condors"
        )
    
    with col3:
        show_only_risky = st.checkbox("⚠️ Only Risky", help="Show only positions with priority >= 5")
    
    st.markdown("---")
    
    # Filter by expiry
    if selected_expiry == "All Expiries":
        filtered_strategies = strategies
    else:
        filtered_strategies = {selected_expiry: strategies[selected_expiry]}
    
    if view_mode == "Strategy View":
        render_strategy_view(filtered_strategies, current_spot, show_only_risky)
    else:
        render_individual_view(enriched, selected_expiry, current_spot, show_only_risky)


def render_strategy_view(strategies: Dict, current_spot: float, show_only_risky: bool):
    """Render strategy-based view (Iron Condors, spreads, etc.)."""
    
    st.subheader("📊 Strategy Analysis by Expiry")
    
    if not strategies:
        st.info("No strategies detected.")
        return
    
    # Analyze all strategies
    strategy_analyses = []
    for expiry_key, strategy_list in strategies.items():
        for strategy_data in strategy_list:
            analysis = analyze_strategy(strategy_data, current_spot)
            if analysis:
                analysis["expiry"] = expiry_key
                strategy_analyses.append(analysis)
    
    # Filter risky
    if show_only_risky:
        strategy_analyses = [s for s in strategy_analyses if s.get("priority", 0) >= 5]
    
    if not strategy_analyses:
        st.success("✅ No risky strategies found!")
        return
    
    # Sort by priority
    strategy_analyses = sorted(strategy_analyses, key=lambda x: x.get("priority", 0), reverse=True)
    
    # Display each strategy
    for analysis in strategy_analyses:
        with st.expander(
            f"{analysis['status']} {analysis['strategy']} - Expiry: {analysis['expiry']} (DTE: {analysis['dte']}) | P&L: {format_inr(analysis['total_pnl'])}",
            expanded=analysis.get("priority", 0) >= 7
        ):
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total P&L", format_inr(analysis["total_pnl"]))
                st.metric("Net Theta/Day", format_inr(analysis["net_theta"], decimals=2))
            
            with col2:
                st.metric("Days to B/E", f"{analysis['days_to_breakeven']:.1f}")
                st.metric("Theta Efficiency", f"{analysis['theta_eff']:.0f}%")
            
            with col3:
                st.metric("Net Delta", f"{analysis['net_delta']:.2f}")
                st.metric("Net Vega", f"{analysis['net_vega']:.2f}")
            
            with col4:
                st.metric("Priority", f"{analysis['priority']}/10")
                st.metric("# Positions", analysis["num_positions"])
            
            # Iron Condor specific details
            if analysis["strategy"] == "Iron Condor":
                st.markdown("---")
                st.markdown("#### 🦅 Iron Condor Details")
                
                ic_col1, ic_col2, ic_col3 = st.columns(3)
                
                with ic_col1:
                    st.markdown("**Put Side**")
                    st.write(f"Short Strike: ₹{analysis['put_short_strike']:.0f}")
                    st.write(f"Distance: {analysis['put_distance_pct']:+.1f}%")
                    st.write(f"Width: ₹{analysis['put_width']:.0f}")
                
                with ic_col2:
                    st.markdown("**Call Side**")
                    st.write(f"Short Strike: ₹{analysis['call_short_strike']:.0f}")
                    st.write(f"Distance: {analysis['call_distance_pct']:+.1f}%")
                    st.write(f"Width: ₹{analysis['call_width']:.0f}")
                
                with ic_col3:
                    st.markdown("**Risk Metrics**")
                    st.write(f"Max Risk: {format_inr(analysis['max_risk'])}")
                    st.write(f"Min Distance: {analysis['min_distance_pct']:.1f}%")
                    st.write(f"Risk/Reward: {analysis['risk_reward']:.2f}")
                
                # Visual indicator for safety
                min_dist = analysis['min_distance_pct']
                if min_dist < 2:
                    st.error(f"🚨 **DANGER:** Spot within {min_dist:.1f}% of short strike!")
                elif min_dist < 5:
                    st.warning(f"⚠️ **CAUTION:** Spot within {min_dist:.1f}% of short strike")
                else:
                    st.success(f"✅ **SAFE:** Spot is {min_dist:.1f}% from nearest short strike")
            
            # Action recommendation
            st.markdown("---")
            st.markdown(f"### 🎯 Recommended Action: **{analysis['action']}**")
            
            # Detailed recommendation
            if analysis['action'] == "CLOSE/ROLL":
                st.error(f"⚠️ **Close or roll this position.** Cannot recover {format_inr(abs(analysis['total_pnl']))} loss in {analysis['dte']} days with current theta of {format_inr(analysis['net_theta'], decimals=2)}/day.")
            elif analysis['action'] == "CLOSE/ADJUST":
                st.error(f"🚨 **Immediate action required.** Spot is too close to short strike. Consider closing, adjusting strikes, or adding hedges.")
            elif analysis['action'] == "CONSIDER CLOSING":
                st.info(f"💰 **Consider taking profit.** Captured {analysis['theta_eff']:.0f}% of potential theta with {analysis['dte']} days remaining.")
            elif analysis['action'] == "MONITOR CLOSELY":
                st.warning(f"👁️ **Monitor closely.** Position approaching danger zone.")
            else:
                st.success(f"✓ **Hold position.** Continue theta decay harvesting. On track to profit.")


def render_individual_view(enriched: List[Dict], selected_expiry: str, current_spot: float, show_only_risky: bool):
    """Render individual position view (original diagnostics table)."""
    
    # Filter by expiry if selected
    if selected_expiry != "All Expiries":
        filtered_positions = [
            p for p in enriched 
            if p.get("expiry") and p["expiry"].strftime("%Y-%m-%d") == selected_expiry
        ]
    else:
        filtered_positions = enriched
    
    if not filtered_positions:
        st.info(f"No positions found for {selected_expiry}")
        return
    
    st.subheader(f"📋 Individual Positions{' - ' + selected_expiry if selected_expiry != 'All Expiries' else ''}")
    
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
    
    # Calculate diagnostics for each position
    diagnostics = []
    for pos in filtered_positions:
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
        
        # Spot distance (safe division)
        if current_spot > 0 and strike > 0:
            spot_distance_pct = ((current_spot - strike) / current_spot * 100)
        else:
            spot_distance_pct = 0
        
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
    
    # Apply risk filter
    if show_only_risky:
        diagnostics = [d for d in diagnostics if d["Priority"] >= 5]
    
    if not diagnostics:
        st.success("✅ No risky positions found!")
        return
    
    # Sort by priority (highest first)
    diagnostics = sorted(diagnostics, key=lambda x: x["Priority"], reverse=True)
    
    # Remove internal sort keys from display
    for d in diagnostics:
        d.pop("_sort_action_priority", None)
    
    df = pd.DataFrame(diagnostics)
    
    # Keep numeric PnL for action recommendations before formatting
    pnl_numeric_map = {i: d["PnL"] for i, d in enumerate(diagnostics)}

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
            # Create a copy with numeric PnL for the recommendation function
            pos_for_recommendation = pos.copy()
            # Ensure PnL is numeric (it should be from diagnostics list)
            
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
                
                st.markdown(f"**Recommended Action:** {get_action_recommendation(pos_for_recommendation)}")
    else:
        st.success("✅ No high-priority actions needed")
