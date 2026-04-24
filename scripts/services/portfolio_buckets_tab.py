"""Portfolio Buckets Tab - trade-level ES99 + zone bucketing overlay."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional

import pandas as pd
from scripts.utils.optional_streamlit import st

from scripts.utils import (
    calculate_portfolio_greeks,
    compute_var_es_metrics,
    get_weighted_scenarios,
    classify_bucket,
)
from scripts.utils.formatters import format_inr
from scripts.utils.option_pricing import price_option
from scripts.services.risk_analysis_tab import (
    DEFAULT_BUCKET_PROBS,
    classify_zone,
    compute_historical_bucket_probabilities,
    get_iv_regime,
)


def _init_portfolio_bucket_state() -> None:
    """Seed session-state defaults for the Portfolio Buckets tab."""
    st.session_state.setdefault("pb_total_capital", 1_000_000.0)
    st.session_state.setdefault("pb_alloc_low", 50.0)
    st.session_state.setdefault("pb_alloc_med", 30.0)
    st.session_state.setdefault("pb_alloc_high", 20.0)
    st.session_state.setdefault("pb_es99_limit_low", 2.0)
    st.session_state.setdefault("pb_es99_limit_med", 3.0)
    st.session_state.setdefault("pb_es99_limit_high", 5.0)
    st.session_state.setdefault("pb_trade_es99_low_max", 1.0)
    st.session_state.setdefault("pb_trade_es99_med_max", 2.0)
    st.session_state.setdefault(
        "pb_zone_low_labels", "SAFE / PROFESSIONAL,BALANCED / CONTROLLED"
    )
    st.session_state.setdefault("pb_zone_amber_labels", "BALANCED / CONTROLLED")
    st.session_state.setdefault(
        "pb_zone_red_labels", "AGGRESSIVE / FRAGILE,OUT OF BOUNDS"
    )
    st.session_state.setdefault("pb_allow_new_low", True)
    st.session_state.setdefault("pb_allow_new_med", True)
    st.session_state.setdefault("pb_allow_new_high", True)
    st.session_state.setdefault("pb_es_lookback_days", 504)
    st.session_state.setdefault("pb_spot_input", 0.0)
    st.session_state.setdefault("pb_target_date", datetime.now().date())
    st.session_state.setdefault("pb_forward_override", 0.0)


def _sync_total_capital_from_account() -> None:
    """Prefer account size from other tabs if available."""
    account_size = st.session_state.get("account_size")
    if account_size and account_size > 0:
        st.session_state["pb_total_capital"] = float(account_size)
        return
    margin_used = st.session_state.get("margin_used")
    margin_available = st.session_state.get("margin_available")
    if margin_used is not None and margin_available is not None:
        try:
            st.session_state["pb_total_capital"] = float(margin_used) + float(margin_available)
        except Exception:
            return


def _parse_label_list(value: str) -> List[str]:
    return [item.strip().lower() for item in (value or "").split(",") if item.strip()]


def _resolve_target_date(target_date: Optional[object]) -> date:
    if isinstance(target_date, date):
        return target_date
    if isinstance(target_date, datetime):
        return target_date.date()
    return datetime.now().date()


def _resolve_time_to_expiry(leg: Dict[str, object], target_date: date) -> Optional[float]:
    expiry = leg.get("expiry")
    if isinstance(expiry, datetime):
        expiry_date = expiry.date()
    elif isinstance(expiry, date):
        expiry_date = expiry
    else:
        expiry_date = None
    if expiry_date is not None:
        days = max((expiry_date - target_date).days, 0)
        return max(days / 365.0, 1.0 / 365.0)
    dte = leg.get("dte")
    try:
        dte_val = float(dte)
        return max(dte_val / 365.0, 1.0 / 365.0) if dte_val >= 0 else None
    except Exception:
        pass
    tte = leg.get("time_to_expiry")
    try:
        return float(tte)
    except Exception:
        return None


def _derive_underlying(position: Dict[str, object]) -> str:
    for key in ("underlying", "name", "symbol", "exchange"):
        val = position.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip().upper()
    symbol = str(position.get("tradingsymbol", "")).upper()
    if symbol:
        prefix = ""
        for ch in symbol:
            if ch.isalpha():
                prefix += ch
            else:
                break
        return prefix or symbol
    return "UNKNOWN"


def _extract_expiry(position: Dict[str, object]) -> str:
    expiry = position.get("expiry")
    if isinstance(expiry, datetime):
        return expiry.strftime("%Y-%m-%d")
    if isinstance(expiry, str) and expiry.strip():
        return expiry
    return "UNKNOWN"


def _extract_margin(position: Dict[str, object]) -> float:
    for key in ("margin", "margin_used", "margin_utilised", "exposure"):
        val = position.get(key)
        try:
            if val is not None:
                return float(val)
        except Exception:
            continue
    return 0.0


def _group_positions_by_trade(
    positions: List[Dict[str, object]],
    key_func=None,
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], Dict[str, object]] = {}
    for pos in positions:
        try:
            qty_val = float(pos.get("quantity") or 0.0)
        except Exception:
            qty_val = 0.0
        if qty_val == 0.0:
            continue
        if key_func:
            key = key_func(pos)
            if not isinstance(key, tuple):
                key = (key,)
        else:
            underlying = _derive_underlying(pos)
            expiry_label = _extract_expiry(pos)
            key = (underlying, expiry_label)
        if len(key) == 1:
            key = (key[0], "UNKNOWN")
        if key not in grouped:
            grouped[key] = {
                "underlying": key[0],
                "expiry": key[1],
                "legs": [],
            }
            if len(key) > 2:
                grouped[key]["option_side"] = key[2]
        grouped[key]["legs"].append(pos)
    return list(grouped.values())


def _compute_trade_es99(
    legs: List[Dict[str, object]],
    account_size: float,
    scenarios: List[object],
    iv_regime: str,
    lookback_days: int,
    spot_override: Optional[float] = None,
    forward_override: Optional[float] = None,
    target_date: Optional[object] = None,
    t_override_days: Optional[float] = None,
    iv_snapshot_override: Optional[List[Optional[float]]] = None,
    return_debug: bool = False,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[str]]:
    spot = float(spot_override or 0.0)
    if spot <= 0 or not scenarios:
        if return_debug:
            return None, None, None, None, None, "invalid_spot" if spot <= 0 else "scenario_empty"
        return 0.0, 0.0, 0.0, 0.0, 0.0

    target_day = _resolve_target_date(target_date)
    bucket_probs, _, _ = compute_historical_bucket_probabilities(
        lookback=int(lookback_days),
        smoothing_enabled=False,
        smoothing_span=63,
    )
    if not bucket_probs:
        bucket_probs = DEFAULT_BUCKET_PROBS.copy()

    bucket_counts: Dict[str, int] = defaultdict(int)
    scenario_buckets: Dict[str, str] = {}
    for scenario in scenarios:
        bucket = classify_bucket(
            {"type": scenario.category.upper(), "dS_pct": scenario.ds_pct, "dIV_pts": scenario.div_pts}
        )
        scenario_buckets[scenario.name] = bucket
        bucket_counts[bucket] += 1

    scenario_probs: Dict[str, float] = {}
    for scenario in scenarios:
        bucket = scenario_buckets.get(scenario.name, "E")
        bucket_prob = bucket_probs.get(bucket, 0.0)
        count = bucket_counts.get(bucket, 0)
        scenario_probs[scenario.name] = bucket_prob / count if count and bucket_prob > 0 else 0.0

    risk_free_rate = 0.07
    forward_base = float(forward_override) if forward_override and forward_override > 0 else None
    scenario_pnls: Dict[str, float] = {}
    used_total = 0
    error_reason = None

    for scenario in scenarios:
        base_pnl = 0.0
        for leg in legs:
            try:
                base_pnl += float(leg.get("pnl", leg.get("m2m", 0.0)) or 0.0)
            except Exception:
                continue
        scenario_spot = spot * (1 + scenario.ds_pct / 100.0)
        scenario_forward = forward_base * (1 + scenario.ds_pct / 100.0) if forward_base else None
        scenario_iv_shift = scenario.div_pts / 100.0
        total_pnl = base_pnl
        used_positions = 0

        for idx, leg in enumerate(legs):
            option_type = leg.get("option_type") or leg.get("instrument_type")
            strike = leg.get("strike", leg.get("strike_price"))
            iv = leg.get("implied_vol")
            if iv is None:
                iv = leg.get("iv")
            if iv is None:
                iv = leg.get("implied_volatility")
            if iv_snapshot_override is not None:
                try:
                    iv_override = iv_snapshot_override[idx]
                except Exception:
                    iv_override = None
                if iv_override is not None:
                    iv = iv_override
            last_price = leg.get("last_price")
            if last_price is None:
                last_price = leg.get("ltp")
            if last_price is None:
                last_price = leg.get("close")
            qty = leg.get("quantity", 0)

            if option_type not in {"CE", "PE"}:
                continue
            if strike is None or iv is None or last_price is None:
                continue

            if t_override_days is not None:
                try:
                    tte = max(float(t_override_days) / 365.0, 1.0 / 365.0)
                except Exception:
                    tte = None
            else:
                tte = _resolve_time_to_expiry(leg, target_day)
            if tte is None:
                continue

            shock_iv = max(float(iv) + scenario_iv_shift, 0.0001)
            flag = "c" if option_type == "CE" else "p"
            try:
                new_price = price_option(
                    flag,
                    float(scenario_spot),
                    float(strike),
                    float(tte),
                    risk_free_rate,
                    float(shock_iv),
                    model="black76",
                    forward=scenario_forward,
                )
            except Exception:
                continue
            if new_price is None:
                continue

            total_pnl += (new_price - float(last_price)) * float(qty)
            used_positions += 1
            used_total += 1

        if used_positions == 0:
            total_pnl = base_pnl
        scenario_pnls[scenario.name] = total_pnl
    if used_total == 0:
        error_reason = "missing_chain"
        if return_debug:
            return None, None, None, None, None, error_reason
        return 0.0, 0.0, 0.0, 0.0, 0.0

    loss_distribution = []
    expected_pnl_inr = 0.0
    loss_prob = 0.0
    loss_weighted = 0.0
    for scenario in scenarios:
        if scenario.name not in scenario_pnls:
            continue
        pnl_value = scenario_pnls.get(scenario.name, 0.0)
        prob = scenario_probs.get(scenario.name, 0.0)
        expected_pnl_inr += pnl_value * prob
        if pnl_value < 0:
            loss_weighted += (-pnl_value) * prob
            loss_prob += prob
        loss_pct = (-pnl_value / account_size * 100.0) if pnl_value < 0 and account_size else 0.0
        loss_distribution.append({"loss_pct": loss_pct, "prob": prob, "scenario": scenario})

    if not loss_distribution:
        error_reason = "scenario_empty"
        if return_debug:
            return None, None, None, None, None, error_reason
        return 0.0, 0.0, 0.0, 0.0, 0.0
    metrics = compute_var_es_metrics(loss_distribution, account_size)
    expected_pnl_pct = (expected_pnl_inr / account_size * 100.0) if account_size else 0.0
    mean_loss_inr = (loss_weighted / loss_prob) if loss_prob > 0 else 0.0
    result = (
        float(metrics.get("ES99", 0.0)),
        float(metrics.get("ES99Value", 0.0)),
        float(expected_pnl_pct),
        float(expected_pnl_inr),
        float(mean_loss_inr),
    )
    if return_debug:
        return (*result, error_reason)
    return result


def _classify_trade_zone(
    legs: List[Dict[str, object]], account_size: float, iv_regime: str
) -> Tuple[str, int]:
    trade_greeks = calculate_portfolio_greeks(legs)
    capital_in_lakhs = account_size / 100000 if account_size else 0.0
    theta_norm = abs(trade_greeks.get("net_theta", 0.0)) / capital_in_lakhs if capital_in_lakhs else 0.0
    gamma_norm = trade_greeks.get("net_gamma", 0.0) / capital_in_lakhs if capital_in_lakhs else 0.0
    vega_norm = trade_greeks.get("net_vega", 0.0) / capital_in_lakhs if capital_in_lakhs else 0.0
    zone_num, zone_name, _, _ = classify_zone(theta_norm, gamma_norm, vega_norm, iv_regime)
    return zone_name, zone_num


def _assign_bucket(
    es99_pct: float,
    zone_label: str,
    low_max: float,
    med_max: float,
    low_zones: List[str],
    amber_zones: List[str],
    red_zones: List[str],
) -> str:
    zone_key = (zone_label or "").strip().lower()
    if es99_pct > med_max or zone_key in red_zones:
        return "HIGH"
    if es99_pct <= low_max and zone_key in low_zones:
        return "LOW"
    if zone_key in amber_zones or es99_pct <= med_max:
        return "MED"
    return "MED"


def render_portfolio_buckets_tab() -> None:
    """Render the Portfolio Buckets tab."""
    _init_portfolio_bucket_state()
    _sync_total_capital_from_account()

    st.markdown("## Portfolio Buckets")
    st.caption("Institutional-style 50/30/20 overlay driven by trade ES99 and zone classification.")

    # Sync positions button
    if st.sidebar.button("🔄 Sync Positions", key="portfolio_buckets_sync_positions", help="Fetch latest positions from Kite", use_container_width=True, type="primary"):
        st.rerun()

    positions = st.session_state.get("enriched_positions", [])
    if not positions:
        st.warning("No positions loaded. Fetch positions from the Portfolio tab to populate this view.")
        return

    # Sidebar controls for thresholds, allocations, and zone mappings.

    account_size = st.session_state.get("account_size")
    if account_size and account_size > 0:
        st.session_state["pb_total_capital"] = float(account_size)
        total_capital = st.sidebar.number_input(
            "Total capital (INR)",
            min_value=0.0,
            step=50_000.0,
            format="%.2f",
            key="pb_total_capital",
            disabled=True,
            help="Synced from Portfolio tab account size",
        )
    else:
        total_capital = st.sidebar.number_input(
            "Total capital (INR)",
            min_value=0.0,
            step=50_000.0,
            format="%.2f",
            key="pb_total_capital",
        )
    spot_default = float(st.session_state.get("pb_spot_input") or st.session_state.get("current_spot") or 0.0)
    st.sidebar.number_input(
        "Spot price (override)",
        min_value=0.0,
        step=10.0,
        format="%.2f",
        value=spot_default,
        key="pb_spot_input",
        help="Overrides spot used in trade ES99 for this tab.",
    )
    st.sidebar.date_input(
        "Target date (repricing)",
        value=st.session_state.get("pb_target_date", datetime.now().date()),
        key="pb_target_date",
        help="Used for Black-76 repricing (target date time to expiry).",
    )
    st.sidebar.number_input(
        "Target-day futures price (optional)",
        min_value=0.0,
        step=10.0,
        format="%.2f",
        value=float(st.session_state.get("pb_forward_override", 0.0) or 0.0),
        key="pb_forward_override",
        help="Leave 0 to use spot-derived forward. Scenario shocks apply to this base.",
    )
    alloc_low = st.sidebar.number_input("Low bucket allocation %", min_value=0.0, max_value=100.0, step=1.0, key="pb_alloc_low")
    alloc_med = st.sidebar.number_input("Medium bucket allocation %", min_value=0.0, max_value=100.0, step=1.0, key="pb_alloc_med")
    alloc_high = st.sidebar.number_input("High bucket allocation %", min_value=0.0, max_value=100.0, step=1.0, key="pb_alloc_high")
    trade_low_max = st.sidebar.number_input("LOW trade ES99 max %", min_value=0.0, step=0.1, key="pb_trade_es99_low_max")
    trade_med_max = st.sidebar.number_input("MED trade ES99 max %", min_value=0.0, step=0.1, key="pb_trade_es99_med_max")
    es_limit_low = st.sidebar.number_input("Bucket ES99 limit (Low) %", min_value=0.0, step=0.1, key="pb_es99_limit_low")
    es_limit_med = st.sidebar.number_input("Bucket ES99 limit (Med) %", min_value=0.0, step=0.1, key="pb_es99_limit_med")
    es_limit_high = st.sidebar.number_input("Bucket ES99 limit (High) %", min_value=0.0, step=0.1, key="pb_es99_limit_high")
    zone_low_labels = st.sidebar.text_input(
        "Zone labels for LOW (comma-separated)",
        key="pb_zone_low_labels",
    )
    zone_amber_labels = st.sidebar.text_input(
        "Zone labels for AMBER (comma-separated)",
        key="pb_zone_amber_labels",
    )
    zone_red_labels = st.sidebar.text_input(
        "Zone labels for RED (comma-separated)",
        key="pb_zone_red_labels",
    )
    lookback_days = st.sidebar.number_input(
        "ES99 lookback days",
        min_value=126,
        max_value=756,
        step=21,
        key="pb_es_lookback_days",
    )

    allocation_total = alloc_low + alloc_med + alloc_high
    if abs(allocation_total - 100.0) > 0.01:
        st.error(f"Bucket allocations must sum to 100%. Current total: {allocation_total:.2f}%")

    low_zone_labels = _parse_label_list(zone_low_labels)
    amber_zone_labels = _parse_label_list(zone_amber_labels)
    red_zone_labels = _parse_label_list(zone_red_labels)

    options_df_cache = st.session_state.get("options_df_cache", pd.DataFrame())
    iv_percentile = 35
    if isinstance(options_df_cache, pd.DataFrame) and not options_df_cache.empty:
        if "iv" in options_df_cache.columns:
            iv_percentile = 35
    iv_regime, _ = get_iv_regime(iv_percentile)
    scenarios = get_weighted_scenarios(iv_regime)
    spot_override = st.session_state.get("pb_spot_input")
    forward_override = st.session_state.get("pb_forward_override")
    target_date = st.session_state.get("pb_target_date")

    trades = _group_positions_by_trade(positions)
    trade_rows = []
    total_legs = 0
    for trade in trades:
        legs = trade["legs"]
        total_legs += len(legs)
        try:
            es99_pct, es99_value, _, _, _ = _compute_trade_es99(
                legs,
                total_capital,
                scenarios,
                iv_regime,
                int(lookback_days),
                spot_override=spot_override,
                forward_override=forward_override,
                target_date=target_date,
            )
        except Exception:
            es99_pct, es99_value = 0.0, 0.0
        try:
            zone_label, _ = _classify_trade_zone(legs, total_capital, iv_regime)
        except Exception:
            zone_label = "UNKNOWN"
        bucket = _assign_bucket(
            es99_pct,
            zone_label,
            float(trade_low_max),
            float(trade_med_max),
            low_zone_labels,
            amber_zone_labels,
            red_zone_labels,
        )
        mtd_pnl = sum(float(leg.get("pnl", leg.get("m2m", 0.0)) or 0.0) for leg in legs)
        margin_total = sum(_extract_margin(leg) for leg in legs)
        trade_rows.append(
            {
                "underlying": trade["underlying"],
                "expiry": trade["expiry"],
                "legs": len(legs),
                "bucket": bucket,
                "trade_es99_inr": es99_value,
                "trade_es99_pct": es99_pct,
                "zone_label": zone_label,
                "mtd_pnl": mtd_pnl,
                "margin": margin_total if margin_total > 0 else None,
                "legs_detail": legs,
            }
        )

    trades_df = pd.DataFrame(trade_rows)
    if trades_df.empty:
        st.warning("No trades available after grouping positions by expiry and underlying.")
        return

    trades_df["risk_weighted_usage"] = trades_df.apply(
        lambda row: abs(row["trade_es99_inr"]) if pd.isna(row["margin"]) else float(row["margin"]),
        axis=1,
    )

    total_es99_inr = trades_df["trade_es99_inr"].sum()
    total_es99_pct = (total_es99_inr / total_capital * 100.0) if total_capital else 0.0

    es99_by_bucket = trades_df.groupby("bucket")["trade_es99_inr"].sum().to_dict()
    es99_by_underlying = trades_df.groupby("underlying")["trade_es99_inr"].sum().sort_values(ascending=False)
    usage_by_bucket = trades_df.groupby("bucket")["risk_weighted_usage"].sum().to_dict()

    top_underlying = es99_by_underlying.index[0] if not es99_by_underlying.empty else "N/A"

    margin_used = st.session_state.get("margin_used")
    margin_available = st.session_state.get("margin_available")
    if margin_available is None and margin_used is not None and total_capital:
        try:
            margin_available = max(float(total_capital) - float(margin_used), 0.0)
        except Exception:
            margin_available = None
    available_capital = None
    if margin_used is not None:
        try:
            available_capital = max(float(total_capital) - float(margin_used), 0.0)
        except Exception:
            available_capital = None

    kpi_cols = st.columns(6)
    kpi_cols[0].metric("Total Capital", format_inr(total_capital))
    kpi_cols[1].metric("Portfolio ES99", f"{total_es99_pct:.2f}%", format_inr(total_es99_inr))
    kpi_cols[2].metric("# Trades", f"{len(trades)}")
    kpi_cols[3].metric("# Legs", f"{total_legs}")
    kpi_cols[4].metric("Top Risk Underlying", top_underlying)
    if margin_used is not None:
        kpi_cols[5].metric("Margin Used", format_inr(margin_used))
    else:
        kpi_cols[5].metric("Margin Used", "—")
    if margin_available is not None:
        st.caption(f"Available margin: {format_inr(margin_available)}")

    st.markdown("### Bucket Status")
    status_cols = st.columns(3)
    bucket_limits = {
        "LOW": es_limit_low,
        "MED": es_limit_med,
        "HIGH": es_limit_high,
    }
    bucket_allocs = {
        "LOW": alloc_low,
        "MED": alloc_med,
        "HIGH": alloc_high,
    }
    for col, bucket in zip(status_cols, ["LOW", "MED", "HIGH"]):
        bucket_es99 = es99_by_bucket.get(bucket, 0.0)
        bucket_es99_pct = (bucket_es99 / total_capital * 100.0) if total_capital else 0.0
        with col:
            st.subheader(bucket)
            bucket_target_capital = total_capital * (bucket_allocs[bucket] / 100.0)
            st.metric(
                "Current Capital Usage",
                format_inr(usage_by_bucket.get(bucket, 0.0)),
                delta_color="off",
            )
            st.caption(f"Target {bucket_allocs[bucket]:.1f}% | {format_inr(bucket_target_capital)}")
            es99_target_inr = total_capital * (float(bucket_limits[bucket]) / 100.0)
            st.metric(
                "Current ES99",
                f"{bucket_es99_pct:.2f}% | {format_inr(bucket_es99)}",
                delta_color="off",
            )
            st.caption(f"Target {bucket_limits[bucket]:.2f}% | {format_inr(es99_target_inr)}")
            bucket_margin = trades_df.loc[trades_df["bucket"] == bucket, "margin"]
            usage_label = "Margin Usage" if bucket_margin.notna().all() else "Risk-Weighted Usage"
            st.metric(usage_label, format_inr(usage_by_bucket.get(bucket, 0.0)))
            current_usage = usage_by_bucket.get(bucket, 0.0)
            if bucket_target_capital > 0 and current_usage > bucket_target_capital:
                st.error("Over target capital usage")

    total_current_usage = sum(usage_by_bucket.values())
    unallocated_capital = max(total_capital - total_current_usage, 0.0)
    unallocated_pct = (unallocated_capital / total_capital * 100.0) if total_capital else 0.0
    st.caption(
        f"Unallocated (by risk-weighted usage): {format_inr(unallocated_capital)} ({unallocated_pct:.1f}%)"
    )

    st.markdown("### Trades (Grouped by Expiry + Underlying)")
    display_df = trades_df.drop(columns=["legs_detail"])
    display_df = display_df.rename(
        columns={
            "legs": "#legs",
            "trade_es99_inr": "trade_es99_inr",
            "trade_es99_pct": "trade_es99_pct",
            "zone_label": "zone_label",
            "mtd_pnl": "mtd_pnl",
        }
    )
    st.dataframe(
        display_df.style.format(
            {
                "trade_es99_inr": lambda v: format_inr(v),
                "trade_es99_pct": "{:.2f}%",
                "mtd_pnl": lambda v: format_inr(v),
                "margin": lambda v: format_inr(v) if pd.notna(v) else "—",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Trade Legs Detail")
    for row in trade_rows:
        header = f"{row['underlying']} {row['expiry']} • {row['bucket']}"
        with st.expander(header):
            legs_df = pd.DataFrame(
                [
                    {
                        "symbol": leg.get("tradingsymbol"),
                        "strike": leg.get("strike"),
                        "type": leg.get("option_type") or leg.get("instrument_type"),
                        "qty": leg.get("quantity"),
                        "avg_price": leg.get("average_price"),
                        "ltp": leg.get("last_price"),
                    }
                    for leg in row["legs_detail"]
                ]
            )
            st.dataframe(legs_df, use_container_width=True, hide_index=True)

    st.markdown("### ES99 Distribution")
    bucket_chart_df = pd.DataFrame(
        {"ES99 (INR)": es99_by_bucket}
    ).reindex(["LOW", "MED", "HIGH"])
    st.bar_chart(bucket_chart_df)

    if not es99_by_underlying.empty:
        top_underlying_df = es99_by_underlying.head(10).to_frame("ES99 (INR)")
        st.bar_chart(top_underlying_df)

    bucket_counts_df = trades_df["bucket"].value_counts().to_frame("Trades")
    st.bar_chart(bucket_counts_df)
