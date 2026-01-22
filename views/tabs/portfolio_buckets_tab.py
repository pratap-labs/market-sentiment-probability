"""Portfolio Buckets Tab - trade-level ES99 + zone bucketing overlay."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

from scripts.utils import (
    build_threshold_report,
    calculate_portfolio_greeks,
    compute_var_es_metrics,
    get_weighted_scenarios,
)
from scripts.utils.formatters import format_inr
from views.tabs.risk_analysis_tab import (
    DEFAULT_BUCKET_PROBS,
    DEFAULT_ES99_LIMIT,
    DEFAULT_THRESHOLD_NORMAL_SHARE,
    DEFAULT_THRESHOLD_STRESS_SHARE,
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
) -> Tuple[float, float, float, float]:
    trade_greeks = calculate_portfolio_greeks(legs)
    spot = next((leg.get("spot_price") for leg in legs if leg.get("spot_price")), 0.0)
    threshold_context = build_threshold_report(
        portfolio={
            "delta": trade_greeks.get("net_delta", 0.0),
            "gamma": trade_greeks.get("net_gamma", 0.0),
            "vega": trade_greeks.get("net_vega", 0.0),
            "spot": float(spot or 0.0),
            "nav": float(account_size),
            "margin": 0.0,
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
        master_pct=float(st.session_state.get("strategy_es_limit", DEFAULT_ES99_LIMIT)),
        hard_stop_pct=float(st.session_state.get("strategy_es_limit", DEFAULT_ES99_LIMIT)) * 1.2,
        normal_share=DEFAULT_THRESHOLD_NORMAL_SHARE,
        stress_share=DEFAULT_THRESHOLD_STRESS_SHARE,
    )
    derived_rows = threshold_context.get("rows", [])
    if not derived_rows:
        return 0.0, 0.0, 0.0, 0.0

    bucket_probs, _, _ = compute_historical_bucket_probabilities(
        lookback=int(lookback_days),
        smoothing_enabled=False,
        smoothing_span=63,
    )
    if not bucket_probs:
        bucket_probs = DEFAULT_BUCKET_PROBS.copy()
    bucket_counts = defaultdict(int)
    for row in derived_rows:
        bucket_counts[row["bucket"]] += 1
    for row in derived_rows:
        bucket = row["bucket"]
        bucket_prob = bucket_probs.get(bucket, 0.0)
        count = bucket_counts.get(bucket, 0)
        row["probability"] = bucket_prob / count if count and bucket_prob > 0 else 0.0

    loss_distribution = [
        {"loss_pct": row["loss_pct_nav"], "prob": row["probability"], "scenario": row["scenario"]}
        for row in derived_rows
    ]
    metrics = compute_var_es_metrics(loss_distribution, account_size)
    expected_pnl_inr = sum(row.get("pnl_total", 0.0) * row.get("probability", 0.0) for row in derived_rows)
    expected_pnl_pct = (expected_pnl_inr / account_size * 100.0) if account_size else 0.0
    return (
        float(metrics.get("ES99", 0.0)),
        float(metrics.get("ES99Value", 0.0)),
        float(expected_pnl_pct),
        float(expected_pnl_inr),
    )


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

    trades = _group_positions_by_trade(positions)
    trade_rows = []
    total_legs = 0
    for trade in trades:
        legs = trade["legs"]
        total_legs += len(legs)
        try:
            es99_pct, es99_value, _, _ = _compute_trade_es99(
                legs, total_capital, scenarios, iv_regime, int(lookback_days)
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
