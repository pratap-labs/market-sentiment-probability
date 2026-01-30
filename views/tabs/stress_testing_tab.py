"""Pre-Trade Analysis tab combining stress testing and trade selection."""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st
import math

from scripts.utils import (
    format_inr,
    get_weighted_scenarios,
    build_threshold_report,
    classify_bucket,
    DEFAULT_LOT_SIZE,
    enrich_position_with_greeks,
)
try:
    from py_vollib.black_scholes import black_scholes as bs
except Exception:
    bs = None
from views.tabs.risk_analysis_tab import (
    classify_zone,
    get_iv_regime,
    compute_historical_bucket_probabilities,
    render_expected_shortfall_panel,
    DEFAULT_ES99_LIMIT,
    DEFAULT_THRESHOLD_NORMAL_SHARE,
    DEFAULT_THRESHOLD_STRESS_SHARE,
)


ZONE_RANGES = {
    "Theta": {
        "Zone 1": (120, 180),
        "Zone 2": (180, 220),
        "Zone 3": (220, 300),
    },
    "Gamma": {
        "Zone 1": (0.0, -0.020),
        "Zone 2": (-0.020, -0.035),
        "Zone 3": (-0.035, -0.055),
    },
}

VEGA_RANGES_BY_IV = {
    "Low IV": {
        "Zone 1": (0, -200),
        "Zone 2": (0, -350),
        "Zone 3": (None, None),
    },
    "Mid IV": {
        "Zone 1": (-200, -450),
        "Zone 2": (-350, -650),
        "Zone 3": (-650, -1000),
    },
    "High IV": {
        "Zone 1": (-450, -700),
        "Zone 2": (-650, -900),
        "Zone 3": (-1000, -1300),
    },
}

BUCKET_SCENARIOS = [
    {
        "bucket": "A",
        "name": "Bucket A – Mild drift (-0.5% spot, +1 IV)",
        "ds_pct": -0.5,
        "d_iv": 1.0,
        "note": "Low-vol day with manageable reversion risk.",
    },
    {
        "bucket": "B",
        "name": "Bucket B – Normal stress (-1.0% spot, +2 IV)",
        "ds_pct": -1.0,
        "d_iv": 2.0,
        "note": "Healthy pullback; requires routine adjustments.",
    },
    {
        "bucket": "C",
        "name": "Bucket C – Vol build (-1.5% spot, +3 IV)",
        "ds_pct": -1.5,
        "d_iv": 3.0,
        "note": "Sustained selling with IV pickup.",
    },
    {
        "bucket": "D",
        "name": "Bucket D – Large shock (-2.5% spot, +5 IV)",
        "ds_pct": -2.5,
        "d_iv": 5.0,
        "note": "Event-driven sell-off; risk controls must trigger.",
    },
    {
        "bucket": "E",
        "name": "Bucket E – Gap risk (-5.0% spot, +8 IV)",
        "ds_pct": -5.0,
        "d_iv": 8.0,
        "note": "Severe gap down scenario. Survival check.",
    },
]

DEFAULT_INPUTS = {
    "stress_delta_input": 0.0,
    "stress_theta_input": 0.0,
    "stress_gamma_input": 0.0,
    "stress_vega_input": 0.0,
    "stress_capital_input": 1000000.0,
}

INSIGHTS = {
    0: [
        "Greeks outside calibrated ranges. Normalize exposures before entering a trade.",
        "Reduce size or add hedges until theta, gamma, and vega sit inside a zone band.",
    ],
    1: [
        "Portfolio aligns with professional posture. Maintain discipline on position sizing.",
        "Consider gradually scaling only if IV percentile trends higher.",
    ],
    2: [
        "Income heavy posture demands daily supervision.",
        "Trim short gamma or add long vega hedges ahead of event-heavy weeks.",
    ],
    3: [
        "High income comes with fragility. Pre-plan stop-loss and repair flows.",
        "Cut size rapidly on 1% adverse move; let delta hedges absorb spikes.",
    ],
}


def _ensure_default_inputs():
    """Ensure Streamlit session contains default manual inputs."""
    positions = st.session_state.get("enriched_positions", [])
    inferred_lot_size = None
    if positions:
        qty_values = [
            int(abs(p.get("quantity", 0) or 0))
            for p in positions
            if p.get("quantity") not in (None, 0)
        ]
        if qty_values:
            try:
                inferred_lot_size = qty_values[0]
                for qty in qty_values[1:]:
                    inferred_lot_size = math.gcd(inferred_lot_size, qty)
                if inferred_lot_size <= 1:
                    inferred_lot_size = None
            except Exception:
                inferred_lot_size = None
    positions_signature = (
        len(positions),
        sum(p.get("quantity", 0) or 0 for p in positions),
        round(sum(abs(float(p.get("last_price", 0) or 0)) for p in positions), 2),
    )
    if st.session_state.get("stress_inputs_signature") != positions_signature:
        if positions:
            net_delta = 0.0
            net_theta = 0.0
            net_gamma = 0.0
            net_vega = 0.0
            for pos in positions:
                qty = pos.get("quantity", 0) or 0
                lot_size = pos.get("lot_size") or inferred_lot_size or DEFAULT_LOT_SIZE
                if lot_size and abs(qty) >= lot_size and abs(qty) % lot_size == 0:
                    lots = qty / lot_size
                else:
                    lots = qty
                units = lots * lot_size if lot_size else qty
                delta = pos.get("delta")
                gamma = pos.get("gamma")
                vega = pos.get("vega")
                theta = pos.get("theta")
                if delta is not None:
                    net_delta += float(delta) * units
                else:
                    net_delta += float(pos.get("position_delta", 0.0))
                if gamma is not None:
                    net_gamma += float(gamma) * units
                else:
                    net_gamma += float(pos.get("position_gamma", 0.0))
                if vega is not None:
                    net_vega += float(vega) * units
                else:
                    net_vega += float(pos.get("position_vega", 0.0))
                if theta is not None:
                    net_theta += float(theta) * units
                else:
                    net_theta += float(pos.get("position_theta", 0.0))
            st.session_state["stress_delta_input"] = float(net_delta)
            st.session_state["stress_theta_input"] = float(net_theta)
            st.session_state["stress_gamma_input"] = float(net_gamma)
            st.session_state["stress_vega_input"] = float(net_vega)
            account_size = st.session_state.get("account_size")
            if account_size:
                st.session_state["stress_capital_input"] = float(account_size)
        st.session_state["stress_inputs_signature"] = positions_signature

    for key, default in DEFAULT_INPUTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _compute_normalized(delta: float, theta: float, gamma: float, vega: float, capital: float) -> Dict[str, float]:
    """Return normalized greeks per ₹1L invested."""
    capital_lakhs = capital / 100000 if capital else 0
    if capital_lakhs <= 0:
        return {"delta": 0.0, "theta": 0.0, "gamma": 0.0, "vega": 0.0, "capital_lakhs": 0.0}
    return {
        "delta": delta / capital_lakhs,
        "theta": abs(theta) / capital_lakhs,
        "gamma": gamma / capital_lakhs,
        "vega": vega / capital_lakhs,
        "capital_lakhs": capital_lakhs,
    }


def _get_greek_status(value: float, greek: str, iv_regime: str) -> Tuple[str, str]:
    """Return emoji + label describing which zone a greek sits in."""
    if greek == "Theta":
        ranges = ZONE_RANGES["Theta"]
        if ranges["Zone 1"][0] <= value <= ranges["Zone 1"][1]:
            return "🟢", "Zone 1"
        if ranges["Zone 2"][0] <= value <= ranges["Zone 2"][1]:
            return "🟡", "Zone 2"
        if ranges["Zone 3"][0] <= value <= ranges["Zone 3"][1]:
            return "🔴", "Zone 3"
    elif greek == "Gamma":
        ranges = ZONE_RANGES["Gamma"]
        if ranges["Zone 1"][1] <= value <= ranges["Zone 1"][0]:
            return "🟢", "Zone 1"
        if ranges["Zone 2"][1] <= value <= ranges["Zone 2"][0]:
            return "🟡", "Zone 2"
        if ranges["Zone 3"][1] <= value <= ranges["Zone 3"][0]:
            return "🔴", "Zone 3"
    elif greek == "Vega":
        ranges = VEGA_RANGES_BY_IV.get(iv_regime, VEGA_RANGES_BY_IV["Mid IV"])
        zone1 = ranges["Zone 1"]
        zone2 = ranges["Zone 2"]
        zone3 = ranges["Zone 3"]
        if zone1[1] <= value <= zone1[0]:
            return "🟢", "Zone 1"
        if zone2[1] <= value <= zone2[0]:
            return "🟡", "Zone 2"
        if zone3[0] is not None and zone3[1] <= value <= zone3[0]:
            return "🔴", "Zone 3"
    return "⚠️", "Out of Range"


def _build_comparison_table(theta_norm: float, gamma_norm: float, vega_norm: float, iv_regime: str) -> pd.DataFrame:
    """Return dataframe comparing current normalized greeks vs zone ranges."""
    vega_ranges = VEGA_RANGES_BY_IV.get(iv_regime, VEGA_RANGES_BY_IV["Mid IV"])
    theta_status = _get_greek_status(theta_norm, "Theta", iv_regime)
    gamma_status = _get_greek_status(gamma_norm, "Gamma", iv_regime)
    vega_status = _get_greek_status(vega_norm, "Vega", iv_regime)

    theta_pct_day = theta_norm / 1000 if theta_norm else 0.0
    theta_pct_month = theta_pct_day * 20

    theta_rows = [
        f"₹{ZONE_RANGES['Theta']['Zone 1'][0]} – ₹{ZONE_RANGES['Theta']['Zone 1'][1]}",
        f"₹{ZONE_RANGES['Theta']['Zone 2'][0]} – ₹{ZONE_RANGES['Theta']['Zone 2'][1]}",
        f"₹{ZONE_RANGES['Theta']['Zone 3'][0]} – ₹{ZONE_RANGES['Theta']['Zone 3'][1]}",
    ]
    gamma_rows = [
        f"{ZONE_RANGES['Gamma']['Zone 1'][1]:.3f} to {ZONE_RANGES['Gamma']['Zone 1'][0]:.3f}",
        f"{ZONE_RANGES['Gamma']['Zone 2'][1]:.3f} to {ZONE_RANGES['Gamma']['Zone 2'][0]:.3f}",
        f"{ZONE_RANGES['Gamma']['Zone 3'][1]:.3f} to {ZONE_RANGES['Gamma']['Zone 3'][0]:.3f}",
    ]

    vega_rows = [
        f"₹{vega_ranges['Zone 1'][1]} to ₹{vega_ranges['Zone 1'][0]}",
        f"₹{vega_ranges['Zone 2'][1]} to ₹{vega_ranges['Zone 2'][0]}",
        "❌ AVOID" if vega_ranges["Zone 3"][0] is None else f"₹{vega_ranges['Zone 3'][1]} to ₹{vega_ranges['Zone 3'][0]}",
    ]

    data = [
        {
            "Metric": "Theta / ₹1L",
            "Zone 1 Range": theta_rows[0],
            "Zone 2 Range": theta_rows[1],
            "Zone 3 Range": theta_rows[2],
            "Your Position": (
                f"{theta_status[0]} ₹{theta_norm:.0f} "
                f"({theta_pct_day:.2f}%/day, {theta_pct_month:.2f}%/month)"
            ),
        },
        {
            "Metric": "Gamma / ₹1L",
            "Zone 1 Range": gamma_rows[0],
            "Zone 2 Range": gamma_rows[1],
            "Zone 3 Range": gamma_rows[2],
            "Your Position": f"{gamma_status[0]} {gamma_norm:.4f}",
        },
        {
            "Metric": f"Vega ({iv_regime}) / ₹1L",
            "Zone 1 Range": vega_rows[0],
            "Zone 2 Range": vega_rows[1],
            "Zone 3 Range": vega_rows[2],
            "Your Position": f"{vega_status[0]} ₹{vega_norm:.0f}",
        },
    ]
    return pd.DataFrame(data)


def _compute_bucket_rows(delta: float, gamma: float, vega: float, capital: float, spot: float, bucket_probs: Dict[str, float]) -> List[Dict[str, str]]:
    """Estimate bucket level scenario PnL using manual greeks."""
    if capital <= 0 or spot <= 0:
        return []

    rows: List[Dict[str, str]] = []
    for scenario in BUCKET_SCENARIOS:
        d_s = spot * (scenario["ds_pct"] / 100.0)
        pnl_delta = delta * d_s
        pnl_gamma = 0.5 * gamma * (d_s ** 2)
        pnl_vega = vega * scenario["d_iv"]
        total = pnl_delta + pnl_gamma + pnl_vega
        loss_pct = (-total / capital * 100.0) if total < 0 else 0.0
        dominant = max(
            [("Delta", abs(pnl_delta)), ("Gamma", abs(pnl_gamma)), ("Vega", abs(pnl_vega))],
            key=lambda item: item[1],
        )[0]

        prob = bucket_probs.get(scenario["bucket"], 0.0)
        rows.append(
            {
                "Bucket": scenario["bucket"],
                "Scenario": scenario["name"],
                "Probability": f"{prob * 100:.1f}%",
                "Δ P&L (₹)": format_inr(total),
                "Loss % Capital": f"{loss_pct:.2f}%",
                "Primary Driver": dominant,
                "Context": scenario["note"],
            }
        )
    return rows


def _render_risk_warnings(zone_num: int, iv_regime: str, theta_norm: float, gamma_norm: float, vega_norm: float):
    """Display red alerts for attempted limit breaches."""
    warnings = []
    if zone_num == 3 and iv_regime == "Low IV":
        warnings.append("Zone 3 posture is blocked in Low IV. Scale down theta or wait for IV expansion.")
    if theta_norm > ZONE_RANGES["Theta"]["Zone 3"][1]:
        warnings.append("Theta exceeds Zone 3 ceiling (₹300/₹1L). Lower size or add calendars.")
    if gamma_norm < ZONE_RANGES["Gamma"]["Zone 3"][1]:
        warnings.append("Short gamma is beyond risk guardrail (-0.055/₹1L). Reduce near-expiry shorts.")
    vega_ceiling = VEGA_RANGES_BY_IV.get(iv_regime, VEGA_RANGES_BY_IV["Mid IV"])["Zone 3"][1]
    if vega_ceiling is not None and vega_norm < vega_ceiling:
        warnings.append("Vega is beyond extreme short limit. Add long vega or trim short straddles.")

    if warnings:
        st.markdown("### 🔴 Risk Warnings")
        for msg in warnings:
            st.error(msg)


def _repriced_scenario_rows(
    positions: List[Dict[str, object]],
    scenarios: List[object],
    spot: float,
    capital: float,
    thresholds: Dict[str, float],
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
        bucket = classify_bucket(
            {"type": scenario.category.upper(), "dS_pct": scenario.ds_pct, "dIV_pts": scenario.div_pts}
        )
        threshold_pct = thresholds.get(f"limit{bucket}", thresholds.get("limitE", 0.0))
        status = "PASS"
        if scenario.category.upper() == "IV" and scenario.div_pts < 0:
            status = "INFO"
        elif loss_pct > threshold_pct:
            status = "FAIL"
        rows.append(
            {
                "Scenario": scenario.name,
                "Bucket": bucket,
                "dS% / dIV": f"{scenario.ds_pct:+.2f}% / {scenario.div_pts:+.1f}",
                "Repriced P&L (₹)": format_inr(total_pnl),
                "Loss % Capital": f"{loss_pct:.2f}%",
                "Threshold % NAV": f"{threshold_pct:.2f}%",
                "Status": status,
                "Positions Used": used_positions,
            }
        )

    return rows, skipped


def _repriced_position_audit(
    positions: List[Dict[str, object]],
    scenario: object,
    spot: float,
    risk_free_rate: float = 0.07,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if not positions or spot <= 0:
        return pd.DataFrame()

    scenario_spot = spot * (1 + scenario.ds_pct / 100.0)
    scenario_iv_shift = scenario.div_pts / 100.0

    for pos in positions:
        option_type = pos.get("option_type")
        strike = pos.get("strike")
        tte = pos.get("time_to_expiry")
        iv = pos.get("implied_vol")
        last_price = pos.get("last_price", 0.0)
        qty = pos.get("quantity", 0)
        symbol = pos.get("tradingsymbol") or pos.get("symbol") or "—"

        if option_type not in {"CE", "PE"}:
            continue
        if strike is None or tte is None or iv is None or last_price is None:
            continue

        shock_iv = max(float(iv) + scenario_iv_shift, 0.0001)
        flag = "c" if option_type == "CE" else "p"
        try:
            new_price = bs(flag, scenario_spot, float(strike), float(tte), risk_free_rate, shock_iv)
        except Exception:
            continue

        pnl = (new_price - float(last_price)) * float(qty)
        rows.append(
            {
                "Symbol": symbol,
                "Type": option_type,
                "Qty": qty,
                "Strike": strike,
                "TTE (days)": round(float(tte) * 365, 1),
                "IV": float(iv),
                "Last Price": float(last_price),
                "Repriced Price": float(new_price),
                "P&L (₹)": float(pnl),
            }
        )

    return pd.DataFrame(rows)


def render_pre_trade_analysis_tab():
    """Render pre-trade stress testing analysis."""
    st.header("🔬 Pre-Trade Analysis")
    st.markdown("### Analyze Positions Before Entry")

    if st.sidebar.button("🧹 Reset Derivatives Cache", key="stress_reset_cache"):
        st.session_state.pop("options_df_cache", None)
        st.session_state.pop("nifty_df_cache", None)
        st.session_state.pop("current_spot", None)
        st.session_state.pop("enriched_positions", None)
        st.session_state.pop("stress_inputs_signature", None)
        st.rerun()
    current_es_limit = st.sidebar.number_input(
        "ES99 limit (% NAV)",
        min_value=1.0,
        max_value=12.0,
        step=0.1,
        format="%.1f",
        value=float(st.session_state.get("strategy_es_limit", DEFAULT_ES99_LIMIT)),
        key="stress_es_limit",
    )
    _ensure_default_inputs()

    spot = st.session_state.get("current_spot", 25000)
    nifty_df_cache = st.session_state.get("nifty_df_cache")
    if isinstance(nifty_df_cache, pd.DataFrame) and not nifty_df_cache.empty and "close" in nifty_df_cache.columns:
        try:
            spot = float(nifty_df_cache["close"].dropna().iloc[-1])
        except Exception:
            pass
    spot_input = st.sidebar.number_input(
        "Spot Override (NIFTY)",
        value=float(spot),
        step=10.0,
        format="%.2f",
        key="stress_spot_override",
    )
    spot = spot_input

    st.caption("Input tentative position greeks and capital to preview zone placement before executing trades.")

    def _sync_from_portfolio():
        positions = st.session_state.get("enriched_positions", [])
        if positions:
            net_delta = 0.0
            net_theta = 0.0
            net_gamma = 0.0
            net_vega = 0.0
            qty_values = [
                int(abs(p.get("quantity", 0) or 0))
                for p in positions
                if p.get("quantity") not in (None, 0)
            ]
            inferred_lot_size = None
            if qty_values:
                try:
                    inferred_lot_size = qty_values[0]
                    for qty in qty_values[1:]:
                        inferred_lot_size = math.gcd(inferred_lot_size, qty)
                    if inferred_lot_size <= 1:
                        inferred_lot_size = None
                except Exception:
                    inferred_lot_size = None
            for pos in positions:
                qty = pos.get("quantity", 0) or 0
                lot_size = pos.get("lot_size") or inferred_lot_size or DEFAULT_LOT_SIZE
                if lot_size and abs(qty) >= lot_size and abs(qty) % lot_size == 0:
                    lots = qty / lot_size
                else:
                    lots = qty
                units = lots * lot_size if lot_size else qty
                delta_val = pos.get("delta")
                gamma_val = pos.get("gamma")
                vega_val = pos.get("vega")
                theta_val = pos.get("theta")
                if delta_val is not None:
                    net_delta += float(delta_val) * units
                else:
                    net_delta += float(pos.get("position_delta", 0.0))
                if gamma_val is not None:
                    net_gamma += float(gamma_val) * units
                else:
                    net_gamma += float(pos.get("position_gamma", 0.0))
                if vega_val is not None:
                    net_vega += float(vega_val) * units
                else:
                    net_vega += float(pos.get("position_vega", 0.0))
                if theta_val is not None:
                    net_theta += float(theta_val) * units
                else:
                    net_theta += float(pos.get("position_theta", 0.0))
            st.session_state["stress_delta_input"] = float(net_delta)
            st.session_state["stress_theta_input"] = float(net_theta)
            st.session_state["stress_gamma_input"] = float(net_gamma)
            st.session_state["stress_vega_input"] = float(net_vega)
        account_size = st.session_state.get("account_size")
        if account_size:
            st.session_state["stress_capital_input"] = float(account_size)
        st.session_state["stress_inputs_signature"] = (
            len(positions),
            sum(p.get("quantity", 0) or 0 for p in positions),
            round(sum(abs(float(p.get("last_price", 0) or 0)) for p in positions), 2),
        )

    def _reset_inputs():
        for key, default in DEFAULT_INPUTS.items():
            st.session_state[key] = default

    st.caption(f"NIFTY close (cached): {spot:.2f}")

    input_cols = st.columns(5)
    delta = st.sidebar.number_input(
        "Delta",
        value=float(st.session_state["stress_delta_input"]),
        step=10.0,
        format="%.2f",
        key="stress_delta_input",
    )
    theta = st.sidebar.number_input(
        "Theta (₹/day)",
        value=float(st.session_state["stress_theta_input"]),
        step=10.0,
        format="%.2f",
        key="stress_theta_input",
    )
    gamma = st.sidebar.number_input(
        "Gamma",
        value=float(st.session_state["stress_gamma_input"]),
        step=0.001,
        format="%.4f",
        key="stress_gamma_input",
    )
    vega = st.sidebar.number_input(
        "Vega",
        value=float(st.session_state["stress_vega_input"]),
        step=50.0,
        format="%.2f",
        key="stress_vega_input",
    )
    capital = st.sidebar.number_input(
        "Capital Allocated (₹)",
        value=float(st.session_state["stress_capital_input"]),
        step=50000.0,
        min_value=0.0,
        format="%.0f",
        key="stress_capital_input",
    )

    action_cols = st.columns([3, 1, 1, 1])
    if st.sidebar.button("Recompute Greeks", type="secondary", key="stress_recompute_btn"):
        positions = st.session_state.get("enriched_positions", [])
        current_spot = st.session_state.get("stress_spot_override", spot)
        if positions:
            refreshed = []
            for pos in positions:
                refreshed.append(enrich_position_with_greeks(pos, pd.DataFrame(), current_spot))
            st.session_state["enriched_positions"] = refreshed
            st.session_state["stress_inputs_signature"] = None
            st.rerun()
        else:
            st.warning("Load positions before recomputing greeks.")

    st.sidebar.button(
        "Sync from Portfolio",
        type="primary",
        on_click=_sync_from_portfolio,
        key="stress_sync_btn",
    )
    st.sidebar.button(
        "Reset Inputs",
        type="secondary",
        on_click=_reset_inputs,
        key="stress_reset_btn",
    )

    normalized = _compute_normalized(delta, theta, gamma, vega, capital)
    capital_lakhs = normalized["capital_lakhs"]

    if capital <= 0:
        st.warning("Enter capital allocation to unlock normalized risk analysis.")
        return

    # Determine IV percentile and regime (reuse cached data if available)
    iv_percentile = 30
    options_df_cache = st.session_state.get("options_df_cache")
    if isinstance(options_df_cache, pd.DataFrame) and not options_df_cache.empty and "iv" in options_df_cache.columns:
        iv_percentile = 35
    iv_regime, iv_color = get_iv_regime(iv_percentile)

    theta_norm = normalized["theta"]
    gamma_norm = normalized["gamma"]
    vega_norm = normalized["vega"]

    zone_num, zone_name, zone_color, zone_message = classify_zone(theta_norm, gamma_norm, vega_norm, iv_regime)

    st.markdown("### Normalized Greeks (per ₹1L capital)")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Account Size", f"₹{capital_lakhs:.2f}L")
    metric_cols[1].metric("Theta / ₹1L", f"₹{theta_norm:.0f}")
    metric_cols[2].metric("Gamma / ₹1L", f"{gamma_norm:.4f}")
    metric_cols[3].metric("Vega / ₹1L", f"₹{vega_norm:.0f}")

    st.markdown("---")
    st.markdown("### Zone Classification")
    
    # Zone badge with colored background (green for Zone 1, red for Zone 2/3)
    zone_bg_colors = {
        1: "#2d5f3f",  # Green
        2: "#8b2e2e",  # Red
        3: "#8b2e2e",  # Red
    }
    zone_text = f"Zone {zone_num} – {zone_name}"
    st.markdown(
        f"""<div style="background-color: {zone_bg_colors.get(zone_num, '#8b2e2e')}; 
        padding: 15px; border-radius: 12px; text-align: center; margin-bottom: 15px;">
        <h3 style="margin: 0; color: white;">{zone_color} {zone_text}</h3>
        <p style="margin: 5px 0 0 0; color: #e0e0e0;">{zone_message}</p>
        </div>""",
        unsafe_allow_html=True
    )

    # IV regime badge (green for Low, red for Mid/High)
    iv_bg_colors = {
        "Low IV": "#2d5f3f",   # Green
        "Mid IV": "#8b2e2e",   # Red
        "High IV": "#8b2e2e",  # Red
    }
    st.markdown(
        f"""<div style="background-color: {iv_bg_colors.get(iv_regime, '#8b2e2e')}; 
        padding: 12px; border-radius: 12px; text-align: center; margin-bottom: 15px;">
        <h4 style="margin: 0; color: white;">{iv_color} {iv_regime}</h4>
        <p style="margin: 5px 0 0 0; color: #e0e0e0;">IV Percentile (est): {iv_percentile:.0f}</p>
        </div>""",
        unsafe_allow_html=True
    )

    st.markdown("### Actionable Insights")
    for tip in INSIGHTS.get(zone_num, INSIGHTS[0]):
        st.markdown(f"- {tip}")

    st.markdown("### Greek Impact Analysis")
    impact_cols = st.columns(3)
    impact_cols[0].metric("Theta decay (₹/day)", f"₹{theta:.0f}", delta=f"{theta_norm:.0f} / ₹1L")
    impact_cols[1].metric("Gamma risk", f"{gamma:.4f}", delta=f"{gamma_norm:.4f} / ₹1L")
    impact_cols[2].metric("Vega exposure", f"₹{vega:.0f}", delta=f"{vega_norm:.0f} / ₹1L")

    comparison_df = _build_comparison_table(theta_norm, gamma_norm, vega_norm, iv_regime)
    st.markdown("### Zone Comparison Table")
    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Metric": st.column_config.TextColumn("Metric"),
            "Zone 1 Range": st.column_config.TextColumn("🟢 Zone 1"),
            "Zone 2 Range": st.column_config.TextColumn("🟡 Zone 2"),
            "Zone 3 Range": st.column_config.TextColumn("🔴 Zone 3"),
            "Your Position": st.column_config.TextColumn("Your Position"),
        },
    )

    st.markdown("### Scenario Analysis (Buckets A/B/C/D/E)")
    bucket_probs, history_count, used_fallback = compute_historical_bucket_probabilities(
        lookback=252,
        smoothing_enabled=False,
        smoothing_span=63,
    )
    bucket_rows = _compute_bucket_rows(delta, gamma, vega, capital, st.session_state.get("current_spot", 25000), bucket_probs)
    st.caption(
        f"Historical bucket mix ({history_count}d lookback): "
        f"A {bucket_probs['A']*100:.1f}%, B {bucket_probs['B']*100:.1f}%, "
        f"C {bucket_probs['C']*100:.1f}%, D {bucket_probs['D']*100:.1f}%, "
        f"E {bucket_probs['E']*100:.1f}%"
    )
    if used_fallback:
        st.warning("Historical cache unavailable; using default bucket probabilities.")
    if bucket_rows:
        scenario_df = pd.DataFrame(bucket_rows)
        st.dataframe(scenario_df, use_container_width=True, hide_index=True)
        worst_row = max(bucket_rows, key=lambda r: float(r["Loss % Capital"].rstrip("%")))
        st.caption(
            f"Worst bucket from inputs: {worst_row['Bucket']} "
            f"({worst_row['Scenario']}) → {worst_row['Loss % Capital']}"
        )
    else:
        st.info("Insufficient data to compute scenario impacts. Verify capital and spot inputs.")

    _render_risk_warnings(zone_num, iv_regime, theta_norm, gamma_norm, vega_norm)

    st.markdown("---")
    st.markdown("## 🔬 Full Scenario Stress Testing")

    scenarios = get_weighted_scenarios(iv_regime)
    portfolio_greeks = {
        "net_delta": delta,
        "net_gamma": gamma,
        "net_theta": theta,
        "net_vega": vega,
    }
    account_size = capital
    margin_deployed = capital

    threshold_context = build_threshold_report(
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
        master_pct=current_es_limit,
        hard_stop_pct=current_es_limit * 1.2,
        normal_share=DEFAULT_THRESHOLD_NORMAL_SHARE,
        stress_share=DEFAULT_THRESHOLD_STRESS_SHARE,
    )

    repriced_rows: List[Dict[str, object]] = []
    repriced_map: Dict[str, Dict[str, object]] = {}
    repriced_skipped = {"missing_fields": 0, "no_price": 0}
    if bs is not None:
        positions = st.session_state.get("enriched_positions", [])
        if positions:
            repriced_rows, repriced_skipped = _repriced_scenario_rows(
                positions,
                scenarios,
                spot,
                capital,
                threshold_context.get("thresholds", {}),
            )
            repriced_map = {row["Scenario"]: row for row in repriced_rows}

    render_expected_shortfall_panel(
        threshold_context.get("rows", []),
        account_size,
        scenarios,
        key_prefix="stress_",
        es_limit=current_es_limit,
    )

    st.markdown("### Scenario Table (with calibrated probabilities)")
    def _parse_inr(val: str) -> float:
        cleaned = str(val).replace("₹", "").replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return 0.0

    derived_rows = threshold_context.get("rows", [])
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
                    "Repriced Loss % NAV": repriced.get("Loss % Capital") if repriced else "—",
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
    else:
        st.info("No scenario rows available to display.")

    st.markdown("---")
    st.markdown("## 🔄 Scenario Repricing (Non-linear Greeks)")
    st.caption(
        "Reprices each option at shocked spot/IV to capture gamma/vega changes. "
        "Repriced P&L is shown inside the main scenario table."
    )
    if bs is None:
        st.warning("Black-Scholes library unavailable; repricing panel disabled.")
        return

    positions = st.session_state.get("enriched_positions", [])
    if not positions:
        st.info("Load positions first to run repriced scenario analysis.")
        return

    if repriced_rows:
        if repriced_skipped["missing_fields"] or repriced_skipped["no_price"]:
            st.caption(
                f"Skipped: {repriced_skipped['missing_fields']} missing fields, {repriced_skipped['no_price']} missing prices."
            )
        chart_df = pd.DataFrame(repriced_rows)
        chart_df["Loss % NAV"] = (
            chart_df["Loss % Capital"].str.replace("%", "", regex=False).astype(float)
        )
        chart_df["Threshold % NAV"] = (
            chart_df["Threshold % NAV"].str.replace("%", "", regex=False).astype(float)
        )
        chart_df["P&L"] = chart_df["Repriced P&L (₹)"].str.replace(",", "", regex=False)
        chart_df["P&L"] = chart_df["P&L"].str.replace("₹", "", regex=False).astype(float)

        st.markdown("### Scenario P&L Chart")
        chart = px.bar(
            chart_df,
            x="Scenario",
            y="P&L",
            color=chart_df["P&L"].apply(lambda v: "Profit" if v >= 0 else "Loss"),
            color_discrete_map={"Profit": "#2ca02c", "Loss": "#d62728"},
            title="Repriced Scenario P&L",
            labels={"P&L": "P&L (₹)"},
        )
        chart.update_layout(showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(chart, use_container_width=True, key="repriced_pnl_chart")

        loss_chart = px.scatter(
            chart_df,
            x="Scenario",
            y="Loss % NAV",
            color=chart_df["Loss % NAV"].apply(lambda v: "Breach" if v > 0 else "OK"),
            color_discrete_map={"Breach": "#d62728", "OK": "#2ca02c"},
            title="Loss % NAV vs Threshold",
            labels={"Loss % NAV": "Loss % NAV"},
        )
        loss_chart.add_scatter(
            x=chart_df["Scenario"],
            y=chart_df["Threshold % NAV"],
            mode="lines+markers",
            name="Threshold",
            line=dict(color="#1f77b4", dash="dash"),
        )
        loss_chart.update_layout(showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(loss_chart, use_container_width=True, key="repriced_loss_chart")
    else:
        st.info("No repriced scenarios available. Verify positions have IV and strike data.")

    st.markdown("### Scenario Repricing Audit")
    st.caption("Select a scenario to review per-position repricing inputs and P&L.")
    if bs is None:
        st.warning("Black-Scholes library unavailable; audit panel disabled.")
        return
    if not positions:
        st.info("Load positions first to run the audit.")
        return
    scenario_names = [scenario.name for scenario in scenarios]
    selected_name = st.sidebar.selectbox("Scenario to audit", scenario_names, key="repriced_audit_scenario")
    selected = next((s for s in scenarios if s.name == selected_name), None)
    if selected:
        audit_df = _repriced_position_audit(positions, selected, spot)
        if audit_df.empty:
            st.info("No option positions available for repricing audit.")
        else:
            total_pnl = audit_df["P&L (₹)"].sum()
            st.metric("Scenario P&L (sum)", f"₹{total_pnl:,.0f}")
            st.dataframe(audit_df, use_container_width=True, hide_index=True)

    st.markdown("### IV Debug")
    st.caption("Inspect the spot, price, and time inputs used for IV back-solve.")
    symbols = sorted({pos.get("tradingsymbol") for pos in positions if pos.get("tradingsymbol")})
    if symbols:
        default_symbol = symbols[0]
        selected_symbol = st.sidebar.selectbox("Symbol", symbols, index=0, key="iv_debug_symbol")
        match = next((p for p in positions if p.get("tradingsymbol") == selected_symbol), None)
        if match:
            debug = match.get("iv_debug", {})
            debug_payload = {
                "spot_used": debug.get("spot_used"),
                "spot_from_nifty_df": float(nifty_df_cache["close"].dropna().iloc[-1]) if isinstance(nifty_df_cache, pd.DataFrame) and not nifty_df_cache.empty and "close" in nifty_df_cache.columns else None,
                "option_price_used": debug.get("option_price_used"),
                "time_to_expiry_years": debug.get("time_to_expiry"),
                "time_to_expiry_days": (
                    float(debug.get("time_to_expiry", 0.0)) * 365.0
                    if debug.get("time_to_expiry") is not None
                    else None
                ),
                "expiry_date": debug.get("expiry_date"),
                "match_count": debug.get("match_count"),
                "implied_vol_pct": (
                    float(match.get("implied_vol")) * 100.0
                    if match.get("implied_vol") is not None
                    else None
                ),
                "last_price": match.get("last_price"),
            }
            st.json(debug_payload)
