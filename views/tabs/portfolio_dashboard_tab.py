"""Combined portfolio dashboard tab: overview, portfolio, and positions."""

import streamlit as st
import pandas as pd
from typing import Tuple, List, Dict

try:
    from kiteconnect import KiteConnect
except Exception:
    KiteConnect = None

from scripts.utils import (
    enrich_position_with_greeks,
    calculate_market_regime,
    calculate_portfolio_greeks,
    calculate_var,
    calculate_stress_pnl,
    format_inr,
    DEFAULT_LOT_SIZE,
)

from .overview_tab import get_market_signal, get_alignment_status, get_portfolio_health_status


@st.cache_data(ttl=259200)
def _fetch_positions(api_key: str, access_token: str) -> List[Dict[str, object]]:
    if KiteConnect is None:
        return []
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    positions = kite.positions()
    return positions.get("net", []) or []


def _fetch_positions_fresh(api_key: str, access_token: str) -> List[Dict[str, object]]:
    if KiteConnect is None:
        return []
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    positions = kite.positions()
    return positions.get("net", []) or []


def _get_margin_snapshot() -> Tuple[float, float]:
    access_token = st.session_state.get("kite_access_token")
    api_key = st.session_state.get("kite_api_key")
    margin_available = 1150000.0
    margin_used = 1300000.0
    if access_token and api_key and KiteConnect is not None:
        try:
            kite = KiteConnect(api_key=api_key)
            kite.set_access_token(access_token)
            margins = kite.margins()
            equity = margins.get("equity", {})
            margin_available = equity.get("available", {}).get("live_balance", margin_available)
            margin_used = equity.get("utilised", {}).get("debits", margin_used)
        except Exception:
            pass
    return float(margin_available), float(margin_used)


def _render_capital_performance(enriched: List[Dict], account_size: float, margin_used: float) -> None:
    total_pnl = sum(p.get("pnl", 0) for p in enriched)
    portfolio_greeks = calculate_portfolio_greeks(enriched)
    total_theta = portfolio_greeks["net_theta"]

    avg_dte = sum(p.get("dte", 0) for p in enriched) / len(enriched) if enriched else 30
    days_in_trade = max(30 - avg_dte, 1)
    roi_pct = (total_pnl / account_size * 100) if account_size > 0 else 0
    roi_annualized = (total_pnl / account_size) / (days_in_trade / 365) * 100 if account_size > 0 else 0

    days_to_recover = abs(total_pnl / total_theta) if total_theta != 0 else 999
    theta_efficiency = (total_pnl / total_theta * 100) if total_theta != 0 else 0

    margin_util_pct = (margin_used / account_size * 100) if account_size > 0 else 0
    theta_pct_capital = (abs(total_theta) / account_size * 100) if account_size > 0 else 0

    notional_exposure = sum(abs(p.get("quantity", 0)) * p.get("strike", 0) for p in enriched)
    leverage_ratio = (notional_exposure / account_size) if account_size > 0 else 0

    st.subheader("Portfolio Overview — Capital & Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Account Size", format_inr(account_size))
    margin_color = "🟢" if margin_util_pct < 50 else ("🟡" if margin_util_pct < 70 else "🔴")
    col2.metric("Margin Used", format_inr(margin_used), f"{margin_util_pct:.1f}% {margin_color}")
    pnl_color = "inverse" if total_pnl < 0 else "normal"
    col3.metric("Net P&L", format_inr(total_pnl), f"{roi_pct:.2f}%", delta_color=pnl_color)
    col4.metric("Theta/Day", format_inr(total_theta), f"{theta_pct_capital:.2f}% of capital")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROI (Annualized)", f"{roi_annualized:.1f}%")
    recovery_color = "🔴" if days_to_recover > avg_dte else ("🟡" if days_to_recover > avg_dte * 0.5 else "🟢")
    col2.metric("Days to Recover", f"{days_to_recover:.1f} {recovery_color}", f"Avg DTE: {avg_dte:.0f}")
    eff_color = "🔴" if theta_efficiency < -200 else ("🟡" if theta_efficiency < -100 else "🟢")
    col3.metric("Theta Efficiency", f"{theta_efficiency:.0f}% {eff_color}")
    leverage_color = "🟢" if leverage_ratio < 50 else ("🟡" if leverage_ratio < 100 else "🔴")
    col4.metric("Notional Exposure", format_inr(notional_exposure), f"{leverage_ratio:.0f} × capital {leverage_color}")


def _render_greeks_risk(enriched: List[Dict], account_size: float, current_spot: float) -> None:
    portfolio_greeks = calculate_portfolio_greeks(enriched)
    total_delta = portfolio_greeks["net_delta"]
    total_gamma = portfolio_greeks["net_gamma"]
    total_vega = portfolio_greeks["net_vega"]
    total_theta = portfolio_greeks["net_theta"]

    st.subheader("Portfolio Overview — Greeks & Risk")
    st.caption(
        f"Delta conversions assume lot size = {DEFAULT_LOT_SIZE} for ₹/pt calculation. "
        "If your positions already include lot-size in total_delta, ₹/pt equals Net Delta."
    )
    col1, col2, col3, col4 = st.columns(4)
    delta_abs = abs(total_delta)
    delta_status = "🟢" if delta_abs < 40 else ("🟡" if delta_abs < 100 else "🔴")
    col1.metric("Net Delta (units)", f"{total_delta:.2f} {delta_status}")
    col2.metric("Delta (₹/pt)", format_inr(total_delta))
    gamma_status = "⚠️" if total_gamma < -0.5 else ""
    col3.metric("Net Gamma", f"{total_gamma:.3f} {gamma_status}")
    col4.metric("Net Vega", format_inr(total_vega))

    col1, col2, col3, col4 = st.columns(4)
    delta_notional = total_delta * current_spot
    delta_notional_pct = (abs(delta_notional) / account_size * 100) if account_size > 0 else 0
    vega_pct = (abs(total_vega) / account_size * 100) if account_size > 0 else 0
    col1.metric("Delta Notional (₹)", format_inr(delta_notional), f"{delta_notional_pct:.1f}% of account")
    col2.metric("Vega % Capital", f"{vega_pct:.2f}%")
    col3.metric("Net Theta", format_inr(total_theta))
    col4.metric("Total Positions", f"{portfolio_greeks['total_positions']}")

    st.markdown("#### Risk Analysis")
    nifty_df = st.session_state.get("nifty_df", pd.DataFrame())
    var_95 = calculate_var(enriched, current_spot, nifty_df)
    stress_up_2 = calculate_stress_pnl(enriched, current_spot, 1.02, 0)
    stress_down_2 = calculate_stress_pnl(enriched, current_spot, 0.98, 0)
    stress_iv_up = calculate_stress_pnl(enriched, current_spot, 1.0, 0.05)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("VaR (95%)", format_inr(var_95))
    col2.metric("+2% NIFTY Move", format_inr(stress_up_2))
    col3.metric("-2% NIFTY Move", format_inr(stress_down_2))
    col4.metric("+5 IV Points", format_inr(stress_iv_up))


def _render_market_weather(regime_data: Dict, current_spot: float) -> None:
    st.subheader("Market Weather")
    if not regime_data:
        st.warning("⚠️ Insufficient market data for regime analysis")
        return

    signal_emoji, signal_text, recommendation = get_market_signal(regime_data)
    col1, col2, col3 = st.columns([1, 2, 3])
    with col1:
        st.markdown(f"### {signal_emoji}")
        st.caption("Market Signal")
    with col2:
        st.metric("Market Regime", regime_data.get("market_regime", "Unknown"))
        st.caption(signal_text)
    with col3:
        st.info(f"**💬 Recommendation:** {recommendation}")

    st.markdown("---")
    metric_cols = st.columns(5)
    iv_rank = regime_data.get("iv_rank", 50)
    iv_arrow = "⬆️" if iv_rank > 70 else ("⬇️" if iv_rank < 30 else "➡️")
    metric_cols[0].metric("📊 IV Rank", f"{iv_rank:.0f}", delta=f"{iv_arrow} {'Expensive' if iv_rank > 70 else ('Cheap' if iv_rank < 30 else 'Fair')}")
    vrp = regime_data.get("vrp", 0)
    metric_cols[1].metric("⚡ VRP", f"{vrp * 100:+.1f}%", delta="IV > RV" if vrp > 0.03 else ("RV > IV" if vrp < -0.03 else "Balanced"))
    term = regime_data.get("term_structure", 0)
    term_label = "Backwardation" if term < -0.02 else ("Contango" if term > 0.02 else "Flat")
    metric_cols[2].metric("⏳ Term", f"{term * 100:+.1f}%", delta=term_label)
    skew = regime_data.get("skew", 0)
    skew_label = "Puts Rich" if skew > 0.01 else ("Calls Rich" if skew < -0.01 else "Balanced")
    metric_cols[3].metric("⚖️ Skew", f"{skew * 100:+.1f}%", delta=skew_label)
    pcr = regime_data.get("pcr_oi", 1.0)
    pcr_label = "Bearish" if pcr > 1.1 else ("Bullish" if pcr < 0.9 else "Neutral")
    metric_cols[4].metric("🗓️ PCR (OI)", f"{pcr:.2f}", delta=pcr_label)

    st.markdown("---")
    detail_cols = st.columns(4)
    detail_cols[0].metric("Current Spot", f"₹{current_spot:.0f}")
    rsi = regime_data.get("rsi", 50)
    rsi_status = "Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral")
    detail_cols[1].metric("RSI (14)", f"{rsi:.0f}", delta=rsi_status)
    max_pain = regime_data.get("max_pain_strike", current_spot)
    detail_cols[2].metric("Max Pain", f"₹{max_pain:.0f}")
    rv = regime_data.get("realized_vol", 0.15) * 100
    detail_cols[3].metric("Realized Vol (30d)", f"{rv:.1f}%")


def _render_alignment_recos_summary(
    regime_data: Dict,
    enriched: List[Dict],
    portfolio_greeks: Dict,
    current_spot: float,
    var_value: float,
    var_pct: float,
    health_status: str,
) -> None:
    st.subheader("Portfolio-Market Alignment")
    near_expiry_count = sum(1 for p in enriched if p.get("dte", 999) < 7)
    if regime_data:
        alignments = get_alignment_status(
            regime_data.get("iv_rank", 50),
            portfolio_greeks["net_vega"],
            regime_data.get("pcr_oi", 1.0),
            portfolio_greeks["net_delta"],
            regime_data.get("term_structure", 0),
            near_expiry_count
        )
        alignment_df = pd.DataFrame(alignments)
        st.dataframe(
            alignment_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Status": st.column_config.TextColumn("Status", width="small"),
                "Action": st.column_config.TextColumn("Action", width="large"),
            },
        )

    st.markdown("---")
    st.subheader("Next Step Recommendations")
    recommendations: List[Dict[str, str]] = []
    if regime_data:
        iv_rank = regime_data.get("iv_rank", 50)
        vrp = regime_data.get("vrp", 0)
        term_structure = regime_data.get("term_structure", 0)
        skew = regime_data.get("skew", 0)

        if iv_rank > 70 and vrp > 0.05:
            recommendations.append({"Type": "Market Strategy", "Recommendation": "🔴 Sell Volatility (prefer far expiry credit spreads)"})
        elif iv_rank < 30 and vrp < -0.05:
            recommendations.append({"Type": "Market Strategy", "Recommendation": "🟢 Buy Volatility (prefer debit spreads, long options)"})
        else:
            recommendations.append({"Type": "Market Strategy", "Recommendation": "🟡 Neutral — Consider Iron Condors or balanced spreads"})

        net_delta = portfolio_greeks["net_delta"]
        if abs(net_delta) > 20:
            delta_adjust = format_inr(abs(net_delta * current_spot))
            if net_delta > 20:
                recommendations.append({"Type": "Portfolio Adjustment", "Recommendation": f"⚠️ Reduce long delta by ~{delta_adjust}; add short calls or bearish spreads"})
            else:
                recommendations.append({"Type": "Portfolio Adjustment", "Recommendation": f"⚠️ Reduce short delta by ~{delta_adjust}; add long calls or bullish spreads"})
        else:
            recommendations.append({"Type": "Portfolio Adjustment", "Recommendation": "✅ Delta exposure within acceptable range"})

        if var_pct > 5:
            recommendations.append({"Type": "Risk Flag", "Recommendation": f"🚨 High VaR ({var_pct:.1f}%) — Consider reducing position sizes"})
        elif near_expiry_count > 5 and term_structure < -0.02:
            recommendations.append({"Type": "Risk Flag", "Recommendation": f"⚠️ {near_expiry_count} near-expiry positions in backwardation — Roll to next month"})
        else:
            recommendations.append({"Type": "Risk Flag", "Recommendation": "✅ Risk metrics within acceptable limits"})

        if abs(skew) > 0.015:
            if skew > 0:
                recommendations.append({"Type": "Opportunities", "Recommendation": f"💡 Puts overpriced by {skew*100:.1f}% — Short put spreads attractive"})
            else:
                recommendations.append({"Type": "Opportunities", "Recommendation": f"💡 Calls overpriced by {abs(skew)*100:.1f}% — Short call spreads attractive"})

    if recommendations:
        rec_df = pd.DataFrame(recommendations)
        st.dataframe(
            rec_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Type": st.column_config.TextColumn("Type", width="medium"),
                "Recommendation": st.column_config.TextColumn("Recommendation", width="large"),
            },
        )

    st.markdown("---")
    st.subheader("AI Summary")
    st.info(
        f"""
        Portfolio currently {health_status.lower()} with {format_inr(sum(p.get("pnl", 0) for p in enriched))} P&L and
        {portfolio_greeks['total_positions']} active positions.

        Market regime is **{regime_data.get('market_regime', 'Unknown')}** with IV Rank at {regime_data.get('iv_rank', 50):.0f}
        and VRP at {regime_data.get('vrp', 0)*100:+.1f}%.

        {'✅ Portfolio volatility stance aligned with market conditions.' if (regime_data.get('iv_rank', 50) > 70 and portfolio_greeks['net_vega'] < 0) or (regime_data.get('iv_rank', 50) < 30 and portfolio_greeks['net_vega'] > 0) else '⚠️ Consider adjusting volatility exposure to align with market regime.'}

        {'⚠️ Delta exposure elevated — consider hedging or reducing directional risk.' if abs(portfolio_greeks['net_delta']) > 20 else '✅ Delta exposure manageable.'}

        {f'🔄 Roll {near_expiry_count} near-expiry positions to avoid gamma risk.' if near_expiry_count > 5 and regime_data.get('term_structure', 0) < -0.02 else ''}
        """
    )


def _render_positions(enriched: List[Dict]) -> None:
    st.subheader("Positions")
    display_cols = [
        "tradingsymbol", "quantity", "strike", "option_type", "expiry", "dte",
        "last_price", "pnl", "implied_vol",
        "delta", "gamma", "vega", "theta",
        "position_delta", "position_gamma", "position_vega", "position_theta",
    ]

    display_data = []
    for pos in enriched:
        row = {col: pos.get(col, None) for col in display_cols}
        display_data.append(row)

    df = pd.DataFrame(display_data)
    expiry_values = sorted(
        {
            exp.strftime("%Y-%m-%d")
            for exp in df["expiry"].dropna()
            if hasattr(exp, "strftime")
        }
    )
    expiry_options = ["All"] + expiry_values
    selected_expiry = st.sidebar.selectbox("Filter positions by expiry", expiry_options, key="portfolio_dashboard_positions_expiry")
    if selected_expiry != "All":
        df = df[
            df["expiry"].apply(
                lambda exp: exp.strftime("%Y-%m-%d") if hasattr(exp, "strftime") else str(exp)
            )
            == selected_expiry
        ]
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
                "pnl": "{:.2f}",
            }),
            use_container_width=True,
        )


def render_portfolio_dashboard_tab():
    """Render the merged Portfolio Dashboard tab with reorganized sections."""

    options_df = st.session_state.get("options_df_cache", pd.DataFrame())
    nifty_df = st.session_state.get("nifty_df_cache", pd.DataFrame())

    access_token = st.session_state.get("kite_access_token")
    kite_api_key = st.session_state.get("kite_api_key")

    col_a, col_b = st.sidebar.columns(2)
    fetch_positions_cached = col_a.button("📦 Load Cache", type="secondary", use_container_width=True)
    fetch_positions_fresh = col_b.button("🔄 Load Fresh", type="primary", use_container_width=True)
    if st.sidebar.button("🔄 Refresh", key="portfolio_dashboard_refresh"):
        st.rerun()
    # if st.sidebar.button("🧹 Clear Positions", key="portfolio_dashboard_clear_positions"):
    #     st.session_state.pop("enriched_positions", None)
    #     st.session_state.pop("current_spot", None)
    #     st.rerun()

    if fetch_positions_cached or fetch_positions_fresh:
        if not access_token or not kite_api_key:
            st.warning("Not logged in. Please go to the Login tab and sign in first.")
        elif options_df is None or options_df.empty:
            st.warning("⚠️ No options data loaded. Please load options data from the Derivatives Data tab.")
        else:
            try:
                with st.spinner("Fetching positions..."):
                    if fetch_positions_fresh:
                        _fetch_positions.clear()
                        net_positions = _fetch_positions_fresh(kite_api_key, access_token)
                        _fetch_positions(kite_api_key, access_token)
                    else:
                        net_positions = _fetch_positions(kite_api_key, access_token)

                    if not net_positions:
                        st.info("No positions returned.")
                    else:
                        market_regime = calculate_market_regime(options_df, nifty_df)
                        current_spot = market_regime.get("current_spot", 25000)
                        if nifty_df is not None and not nifty_df.empty and "close" in nifty_df.columns:
                            try:
                                current_spot = float(nifty_df["close"].dropna().iloc[-1])
                            except Exception:
                                pass

                        enriched = []
                        for pos in net_positions:
                            enriched_pos = enrich_position_with_greeks(pos, options_df, current_spot)
                            enriched.append(enriched_pos)

                        st.session_state["enriched_positions"] = enriched
                        st.session_state["current_spot"] = current_spot
                        st.success(f"✅ Loaded {len(enriched)} positions")
            except Exception as exc:
                st.error(f"Failed to fetch positions: {exc}")
                if "403" in str(exc) or "Invalid" in str(exc):
                    st.warning("Your session may have expired. Please log in again.")
                    st.session_state.pop("kite_access_token", None)

    enriched = st.session_state.get("enriched_positions", [])
    if not enriched:
        st.info("No positions loaded yet. Click “Fetch Latest Positions” above.")
        return

    margin_available, margin_used = _get_margin_snapshot()
    account_size = margin_available + margin_used
    st.session_state["account_size"] = account_size
    st.session_state["margin_used"] = margin_used
    market_regime = calculate_market_regime(options_df, nifty_df)
    current_spot = market_regime.get("current_spot", st.session_state.get("current_spot", 25000))
    portfolio_greeks = calculate_portfolio_greeks(enriched)

    total_value = sum(abs(p.get("pnl", 0)) for p in enriched) or 100000
    var_value = calculate_var(enriched, current_spot, nifty_df)
    var_pct = (var_value / total_value) * 100

    health_emoji, health_status = get_portfolio_health_status(
        {"theta_efficiency": (sum(p.get("pnl", 0) for p in enriched) / portfolio_greeks["net_theta"] * 100) if portfolio_greeks["net_theta"] else 0},
        var_pct,
        (margin_used / account_size * 100) if account_size > 0 else 0,
    )

    st.markdown("---")
    _render_capital_performance(enriched, account_size, margin_used)

    st.markdown("---")
    _render_greeks_risk(enriched, account_size, current_spot)

    st.markdown("---")
    _render_market_weather(market_regime, current_spot)

    st.markdown("---")
    _render_alignment_recos_summary(
        market_regime,
        enriched,
        portfolio_greeks,
        current_spot,
        var_value,
        var_pct,
        health_status,
    )

    st.markdown("---")
    _render_positions(enriched)

    # Greeks Debug Section
    st.markdown("---")
    with st.expander("🧪 Greeks Debug (Manual Recalculation)", expanded=False):
        st.caption("Recompute greeks using manual spot and last_price only. No auto-enrichment.")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            debug_spot = st.number_input(
                "Spot (NIFTY)",
                value=float(current_spot),
                step=10.0,
                format="%.2f",
                key="greeks_debug_spot",
            )
        with col2:
            if st.button("Recompute Greeks", type="primary", key="greeks_debug_recompute"):
                st.session_state["greeks_spot_override"] = debug_spot
                refreshed = []
                for pos in enriched:
                    refreshed.append(enrich_position_with_greeks(pos, pd.DataFrame(), debug_spot))
                st.session_state["debug_enriched_positions"] = refreshed
                st.success("Greeks recomputed using spot override.")
        
        debug_positions = st.session_state.get("debug_enriched_positions", [])
        if debug_positions:
            # IV Debug Payload for selected symbol
            symbols = sorted({pos.get("tradingsymbol") for pos in debug_positions if pos.get("tradingsymbol")})
            if symbols:
                selected_symbol = st.selectbox("Symbol for IV Debug", symbols, key="greeks_debug_symbol")
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
            
            # Positions table with greeks
            display_cols = [
                "tradingsymbol", "quantity", "strike", "option_type", "expiry", "dte",
                "last_price", "implied_vol", "delta", "gamma", "vega", "theta",
                "position_delta", "position_gamma", "position_vega", "position_theta",
            ]
            df = pd.DataFrame([{col: pos.get(col) for col in display_cols} for pos in debug_positions])
            
            # Expiry filter
            expiry_values = sorted(
                {
                    exp.strftime("%Y-%m-%d")
                    for exp in df["expiry"].dropna()
                    if hasattr(exp, "strftime")
                }
            )
            if expiry_values:
                expiry_options = ["All"] + expiry_values
                selected_expiry = st.selectbox("Filter by Expiry", expiry_options, key="greeks_debug_expiry")
                if selected_expiry != "All":
                    df = df[
                        df["expiry"].apply(
                            lambda exp: exp.strftime("%Y-%m-%d") if hasattr(exp, "strftime") else str(exp)
                        )
                        == selected_expiry
                    ]
            
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Click 'Recompute Greeks' to populate debug values.")
