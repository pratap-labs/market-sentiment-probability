"""TBA Tab - Institutional Risk View with portfolio, bucket, and trade levels."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import json

from scripts.utils import (
    build_threshold_report,
    calculate_portfolio_greeks,
    compute_var_es_metrics,
    get_weighted_scenarios,
    enrich_position_with_greeks,
)
from scripts.utils.formatters import format_inr
from views.tabs.portfolio_buckets_tab import (
    _classify_trade_zone,
    _compute_trade_es99,
    _derive_underlying,
    _extract_margin,
    _group_positions_by_trade,
)
from views.tabs.risk_analysis_tab import (
    DEFAULT_BUCKET_PROBS,
    DEFAULT_ES99_LIMIT,
    DEFAULT_THRESHOLD_NORMAL_SHARE,
    DEFAULT_THRESHOLD_STRESS_SHARE,
    compute_historical_bucket_probabilities,
    get_iv_regime,
)
from views.tabs import risk_analysis_tab as ra_tab
from scripts.utils.stress_testing import Scenario


BUCKET_ORDER = ["Low", "Med", "High"]
BUCKET_KEYS = {"Low": "low", "Med": "med", "High": "high"}


@dataclass
class SimulationConfig:
    horizon_days: int
    paths: int
    iv_mode: str
    iv_shock: float
    include_spot_shocks: bool


def _init_tba_state() -> None:
    st.session_state.setdefault("tba_total_capital", 1_750_000.0)
    st.session_state.setdefault("tba_alloc_low", 50.0)
    st.session_state.setdefault("tba_alloc_med", 30.0)
    st.session_state.setdefault("tba_alloc_high", 20.0)
    st.session_state.setdefault("tba_portfolio_es_limit", 4.0)
    st.session_state.setdefault("tba_bucket_es_limit_low", 2.0)
    st.session_state.setdefault("tba_bucket_es_limit_med", 3.0)
    st.session_state.setdefault("tba_bucket_es_limit_high", 5.0)
    st.session_state.setdefault("tba_trade_low_max", 1.0)
    st.session_state.setdefault("tba_trade_med_max", 2.0)
    st.session_state.setdefault("tba_zone_map", "")
    st.session_state.setdefault("tba_sim_days", 10)
    st.session_state.setdefault("tba_sim_paths", 2000)
    st.session_state.setdefault("tba_iv_mode", "IV Flat")
    st.session_state.setdefault("tba_iv_shock", 2.0)
    st.session_state.setdefault("tba_spot_overlay", False)
    st.session_state.setdefault("tba_spot_input", 0.0)
    st.session_state.setdefault("tba_bucket_sim_target", "Portfolio")
    st.session_state.setdefault("tba_trade_filter_underlying", "All")
    st.session_state.setdefault("tba_trade_filter_week", "All")
    st.session_state.setdefault("tba_trade_filter_side", "All")
    st.session_state.setdefault("tba_trade_filter_bucket", "All")
    st.session_state.setdefault("tba_trade_filter_zone", "All")
    st.session_state.setdefault("tba_trade_sort", "trade_es99_inr")
    st.session_state.setdefault("tba_selected_trade", "")
    if "tba_saved_trades" not in st.session_state:
        st.session_state["tba_saved_trades"] = _load_saved_groups()
    st.session_state.setdefault("tba_trade_group_name", "")
    st.session_state.setdefault("tba_trade_group_selection", [])
    st.session_state.setdefault("tba_clear_trade_group", False)
    st.session_state.setdefault("tba_use_saved_groups", False)


def _saved_groups_path() -> Path:
    return Path("database") / "trade_groups.json"


def _load_saved_groups() -> List[Dict[str, object]]:
    path = _saved_groups_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_saved_groups(groups: List[Dict[str, object]]) -> None:
    path = _saved_groups_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(groups, ensure_ascii=True, indent=2), encoding="utf-8")
    except Exception:
        return


def _sync_total_capital_from_account() -> None:
    account_size = st.session_state.get("account_size")
    if account_size and account_size > 0:
        st.session_state["tba_total_capital"] = float(account_size)
        return
    margin_used = st.session_state.get("margin_used")
    margin_available = st.session_state.get("margin_available")
    if margin_used is not None and margin_available is not None:
        try:
            st.session_state["tba_total_capital"] = float(margin_used) + float(margin_available)
        except Exception:
            return


def load_positions() -> List[Dict[str, object]]:
    return st.session_state.get("enriched_positions", [])


def _iso_week_id(expiry: Optional[object]) -> str:
    if isinstance(expiry, datetime):
        dt = expiry
    else:
        try:
            dt = pd.to_datetime(expiry, errors="coerce")
        except Exception:
            return "UNKNOWN"
    if dt is None or pd.isna(dt):
        return "UNKNOWN"
    iso_year, iso_week, _ = dt.isocalendar()
    return f"{iso_year}-W{int(iso_week):02d}"


def _derive_option_side(position: Dict[str, object]) -> str:
    raw = position.get("option_type") or position.get("instrument_type")
    if raw is None:
        return "OTHER"
    value = str(raw).upper().strip()
    if value in {"CE", "PE"}:
        return value
    return value or "OTHER"


def _position_label(position: Dict[str, object], idx: int) -> str:
    symbol = position.get("tradingsymbol") or "UNKNOWN"
    expiry = position.get("expiry")
    if isinstance(expiry, datetime):
        expiry_label = expiry.strftime("%Y-%m-%d")
    else:
        expiry_label = str(expiry) if expiry else "—"
    option_type = position.get("option_type") or position.get("instrument_type") or ""
    qty = position.get("quantity") or 0
    strike = position.get("strike") or ""
    return f"{idx} | {symbol} | {option_type} {strike} | {expiry_label} | qty {qty}"


def build_trades_from_saved_groups(
    positions: List[Dict[str, object]],
    saved_groups: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    if not saved_groups:
        return []
    trades = []
    for group in saved_groups:
        leg_ids = set(group.get("legs", []))
        legs = []
        for idx, pos in enumerate(positions):
            pos_id = str(idx)
            if pos_id in leg_ids:
                legs.append(pos)
        if not legs:
            continue
        expiries = []
        for leg in legs:
            exp = leg.get("expiry")
            if isinstance(exp, datetime):
                expiries.append(exp)
            else:
                try:
                    dt = pd.to_datetime(exp, errors="coerce")
                except Exception:
                    continue
                if dt is None or pd.isna(dt):
                    continue
                expiries.append(dt)
        trade_name = group.get("name") or "Manual Trade"
        trades.append(
            {
                "underlying": _derive_underlying(legs[0]),
                "expiry": trade_name,
                "legs": legs,
                "option_side": trade_name,
                "earliest_expiry": min(expiries).strftime("%Y-%m-%d") if expiries else "—",
                "latest_expiry": max(expiries).strftime("%Y-%m-%d") if expiries else "—",
                "trade_name": trade_name,
            }
        )
    return trades


def build_trades_using_existing_grouping(positions: List[Dict[str, object]]) -> List[Dict[str, object]]:
    def _key_func(pos: Dict[str, object]) -> Tuple[str, str, str]:
        underlying = _derive_underlying(pos)
        expiry = pos.get("expiry")
        week_id = _iso_week_id(expiry)
        option_side = _derive_option_side(pos)
        return (underlying, week_id, option_side)

    trades = _group_positions_by_trade(positions, key_func=_key_func)
    for trade in trades:
        expiries = []
        for leg in trade["legs"]:
            exp = leg.get("expiry")
            if isinstance(exp, datetime):
                expiries.append(exp)
                continue
            try:
                dt = pd.to_datetime(exp, errors="coerce")
            except Exception:
                continue
            if dt is None or pd.isna(dt):
                continue
            expiries.append(dt)
        if expiries:
            trade["earliest_expiry"] = min(expiries).strftime("%Y-%m-%d")
            trade["latest_expiry"] = max(expiries).strftime("%Y-%m-%d")
        else:
            trade["earliest_expiry"] = "—"
            trade["latest_expiry"] = "—"
    return trades


def _parse_zone_map(raw: str) -> Dict[str, str]:
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return {str(k).strip(): str(v).strip().lower() for k, v in data.items()}
    except Exception:
        pass
    mappings = {}
    for chunk in raw.split(","):
        if "=" not in chunk:
            continue
        k, v = chunk.split("=", 1)
        mappings[k.strip()] = v.strip().lower()
    return mappings


 


def _label_theta_zone(theta_per_lakh: float) -> str:
    if theta_per_lakh < 120:
        return "TOO_SAFE"
    if 120 <= theta_per_lakh <= 180:
        return "Z1"
    if 180 < theta_per_lakh <= 220:
        return "Z2"
    if 220 < theta_per_lakh <= 300:
        return "Z3"
    return "OUT"


def _label_gamma_zone(gamma_per_lakh: float) -> str:
    if gamma_per_lakh > -0.020:
        return "TOO_SAFE"
    if -0.020 <= gamma_per_lakh <= 0:
        return "Z1"
    if -0.035 <= gamma_per_lakh < -0.020:
        return "Z2"
    if -0.055 <= gamma_per_lakh < -0.035:
        return "Z3"
    return "OUT"


def _label_vega_zone(vega_per_lakh: float, iv_regime: str) -> str:
    ranges = {
        "Low IV": {"Z1": (0, -200), "Z2": (0, -350), "Z3": (None, None)},
        "Mid IV": {"Z1": (-200, -450), "Z2": (-350, -650), "Z3": (-650, -1000)},
        "High IV": {"Z1": (-450, -700), "Z2": (-650, -900), "Z3": (-1000, -1300)},
    }
    regime = ranges.get(iv_regime, ranges["Mid IV"])
    z1_low, z1_high = regime["Z1"]
    if z1_low is not None and z1_high is not None and vega_per_lakh > z1_low:
        return "TOO_SAFE"
    for label, rng in regime.items():
        if rng[0] is None or rng[1] is None:
            continue
        low, high = rng
        if high <= vega_per_lakh <= low:
            return label
    return "OUT"


def _zone_rules_table(iv_regime: str) -> pd.DataFrame:
    zone1_theta_range = (120, 180)
    zone1_gamma_range = (0, -0.020)
    zone1_vega_ranges = {
        "Low IV": (0, -200),
        "Mid IV": (-200, -450),
        "High IV": (-450, -700),
    }
    zone2_theta_range = (180, 220)
    zone2_gamma_range = (-0.020, -0.035)
    zone2_vega_ranges = {
        "Low IV": (0, -350),
        "Mid IV": (-350, -650),
        "High IV": (-650, -900),
    }
    zone3_theta_range = (220, 300)
    zone3_gamma_range = (-0.035, -0.055)
    zone3_vega_ranges = {
        "Low IV": (None, None),
        "Mid IV": (-650, -1000),
        "High IV": (-1000, -1300),
    }

    def _fmt_range(rng: Tuple[Optional[float], Optional[float]]) -> str:
        low, high = rng
        if low is None or high is None:
            return "Avoid"
        return f"{low:.0f} to {high:.0f}"

    return pd.DataFrame(
        [
            {
                "Zone": "Zone 1 (SAFE / PROFESSIONAL)",
                "Theta (per 1L)": f"{zone1_theta_range[0]} to {zone1_theta_range[1]}",
                "Gamma": f"{zone1_gamma_range[1]:.3f} to {zone1_gamma_range[0]:.3f}",
                "Vega (per 1L)": _fmt_range(zone1_vega_ranges[iv_regime]),
            },
            {
                "Zone": "Zone 2 (BALANCED / CONTROLLED)",
                "Theta (per 1L)": f"{zone2_theta_range[0]} to {zone2_theta_range[1]}",
                "Gamma": f"{zone2_gamma_range[1]:.3f} to {zone2_gamma_range[0]:.3f}",
                "Vega (per 1L)": _fmt_range(zone2_vega_ranges[iv_regime]),
            },
            {
                "Zone": "Zone 3 (AGGRESSIVE / FRAGILE)",
                "Theta (per 1L)": f"{zone3_theta_range[0]} to {zone3_theta_range[1]}",
                "Gamma": f"{zone3_gamma_range[1]:.3f} to {zone3_gamma_range[0]:.3f}",
                "Vega (per 1L)": _fmt_range(zone3_vega_ranges[iv_regime]),
            },
        ]
    )


def _compute_trade_sim_metrics(
    legs: List[Dict[str, object]],
    config: SimulationConfig,
    account_size: float,
) -> Tuple[float, float, float, float, float]:
    if not legs or account_size <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    sigma, sigma_source = _get_sigma_source(legs)
    iv_shift = 0.0
    if config.iv_mode == "IV Up Shock":
        iv_shift = config.iv_shock
    elif config.iv_mode == "IV Down Shock":
        iv_shift = -config.iv_shock
    if "IV" not in sigma_source and iv_shift != 0.0:
        iv_shift = math.copysign(max(sigma * 100.0 * 0.10, 0.5), iv_shift)

    spot = _get_spot_from_positions(legs)
    if not spot:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    greeks = calculate_portfolio_greeks(legs)
    delta = greeks.get("net_delta", 0.0)
    gamma = greeks.get("net_gamma", 0.0)
    vega = greeks.get("net_vega", 0.0)
    theta = greeks.get("net_theta", 0.0)

    dt = 1.0 / 252.0
    paths = max(200, int(config.paths))
    horizon = max(1, min(int(config.horizon_days), 20))
    z = np.random.normal(size=(paths, horizon))
    drift = 0.0
    increments = (drift - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z
    log_paths = np.cumsum(increments, axis=1)
    s_paths = spot * np.exp(log_paths)

    pnl_paths = np.zeros_like(s_paths)
    for t in range(horizon):
        dS = s_paths[:, t] - spot
        pnl_paths[:, t] = delta * dS + 0.5 * gamma * (dS ** 2) + vega * iv_shift + theta * (t + 1)

    pnl_tn = pnl_paths[:, -1]
    expected_pnl_inr = float(np.mean(pnl_tn))
    expected_pnl_pct = (expected_pnl_inr / account_size * 100.0) if account_size else 0.0
    mean_loss_inr = float(-np.mean(pnl_tn[pnl_tn < 0])) if np.any(pnl_tn < 0) else 0.0

    worst_count = max(1, int(math.ceil(0.01 * len(pnl_tn))))
    worst_slice = np.sort(pnl_tn)[:worst_count]
    es99_tail_mean = float(np.mean(worst_slice)) if worst_slice.size else 0.0
    es99_inr = abs(es99_tail_mean)
    es99_pct = (es99_inr / account_size * 100.0) if account_size else 0.0

    return expected_pnl_inr, expected_pnl_pct, es99_inr, es99_pct, es99_tail_mean, mean_loss_inr


def compute_trade_risk(
    trades: List[Dict[str, object]],
    account_size: float,
    lookback_days: int,
) -> Tuple[pd.DataFrame, Dict[str, List[Dict[str, object]]]]:
    options_df_cache = st.session_state.get("options_df_cache", pd.DataFrame())
    iv_percentile = 35
    if isinstance(options_df_cache, pd.DataFrame) and not options_df_cache.empty and "iv" in options_df_cache.columns:
        iv_percentile = 35
    iv_regime, _ = get_iv_regime(iv_percentile)
    scenarios = get_weighted_scenarios(iv_regime)
    sim_cfg = SimulationConfig(
        horizon_days=int(st.session_state.get("tba_sim_days", 10)),
        paths=int(st.session_state.get("tba_sim_paths", 2000)),
        iv_mode=st.session_state.get("tba_iv_mode", "IV Flat"),
        iv_shock=float(st.session_state.get("tba_iv_shock", 2.0)),
        include_spot_shocks=False,
    )
    horizon_days = max(1, min(int(sim_cfg.horizon_days), 20))

    rows = []
    legs_by_trade: Dict[str, List[Dict[str, object]]] = {}
    for trade in trades:
        legs = trade["legs"]
        option_side = trade.get("option_side", "OTHER")
        trade_name = trade.get("trade_name", "")
        trade_id = (
            f"{trade['underlying']}|{trade['expiry']}|{trade_name}"
            if trade_name
            else f"{trade['underlying']}|{trade['expiry']}|{option_side}"
        )
        legs_by_trade[trade_id] = legs
        try:
            es99_pct, es99_value, _, _ = _compute_trade_es99(
                legs, account_size, scenarios, iv_regime, int(lookback_days)
            )
            exp_pnl_value, exp_pnl_pct, _, _, es99_tail_mean, mean_loss_inr = _compute_trade_sim_metrics(
                legs, sim_cfg, account_size
            )
            trade_greeks = calculate_portfolio_greeks(legs)
            capital_in_lakhs = account_size / 100000.0 if account_size else 0.0
            theta_per_lakh = abs(trade_greeks.get("net_theta", 0.0)) / capital_in_lakhs if capital_in_lakhs else 0.0
            gamma_per_lakh = trade_greeks.get("net_gamma", 0.0) / capital_in_lakhs if capital_in_lakhs else 0.0
            vega_per_lakh = trade_greeks.get("net_vega", 0.0) / capital_in_lakhs if capital_in_lakhs else 0.0
            theta_label = _label_theta_zone(theta_per_lakh)
            gamma_label = _label_gamma_zone(gamma_per_lakh)
            vega_label = _label_vega_zone(vega_per_lakh, iv_regime)
            theta_carry = float(trade_greeks.get("net_theta", 0.0)) * horizon_days
            zone_label, _ = _classify_trade_zone(legs, account_size, iv_regime)
            risk_error = ""
        except Exception as exc:
            es99_pct, es99_value = 0.0, 0.0
            exp_pnl_pct, exp_pnl_value = 0.0, 0.0
            es99_tail_mean = 0.0
            mean_loss_inr = 0.0
            theta_carry = 0.0
            theta_per_lakh = 0.0
            gamma_per_lakh = 0.0
            vega_per_lakh = 0.0
            theta_label = "OUT"
            gamma_label = "OUT"
            vega_label = "OUT"
            trade_greeks = {"net_theta": 0.0, "net_gamma": 0.0, "net_vega": 0.0}
            zone_label = "UNKNOWN"
            risk_error = str(exc)
        mtd_pnl = sum(float(leg.get("pnl", leg.get("m2m", 0.0)) or 0.0) for leg in legs)
        premium_received = 0.0
        for leg in legs:
            try:
                qty = float(leg.get("quantity") or 0.0)
                avg_price = float(leg.get("average_price") or 0.0)
            except Exception:
                continue
            premium_received += -avg_price * qty
        dte_values = []
        for leg in legs:
            try:
                dte = float(leg.get("dte"))
            except Exception:
                continue
            if dte > 0:
                dte_values.append(dte)
        min_dte = min(dte_values) if dte_values else 0.0
        premium_prorated = 0.0
        if min_dte > 0:
            premium_prorated = premium_received * min(horizon_days, min_dte) / min_dte
        exp_pnl_value += premium_prorated
        exp_pnl_pct = (exp_pnl_value / account_size * 100.0) if account_size else 0.0
        margin_total = sum(_extract_margin(leg) for leg in legs)
        rows.append(
            {
                "trade_id": trade_id,
                "underlying": trade["underlying"],
                "week_id": trade["expiry"],
                "option_side": trade_name or option_side,
                "earliest_expiry": trade.get("earliest_expiry", "—"),
                "latest_expiry": trade.get("latest_expiry", "—"),
                "legs": len(legs),
                "trade_es99_inr": es99_value,
                "trade_es99_pct": es99_pct,
                "expected_pnl_inr": exp_pnl_value,
                "expected_pnl_pct": exp_pnl_pct,
                "risk_reward": (exp_pnl_value / abs(es99_value)) if es99_value else 0.0,
                "es99_tail_pnl_inr": es99_tail_mean,
                "tail_loss_inr": abs(es99_tail_mean) if es99_tail_mean < 0 else 0.0,
                "theta_carry_inr": theta_carry,
                "mean_loss_inr": mean_loss_inr,
                "mean_tail_ratio": (mean_loss_inr / abs(es99_tail_mean))
                if es99_tail_mean < 0
                else 0.0,
                "net_theta": trade_greeks.get("net_theta", 0.0),
                "net_gamma": trade_greeks.get("net_gamma", 0.0),
                "net_vega": trade_greeks.get("net_vega", 0.0),
                "theta_per_lakh": theta_per_lakh,
                "gamma_per_lakh": gamma_per_lakh,
                "vega_per_lakh": vega_per_lakh,
                "theta_label": theta_label,
                "gamma_label": gamma_label,
                "vega_label": vega_label,
                "zone_label": zone_label,
                "risk_error": risk_error,
                "mtd_pnl": mtd_pnl,
                "premium_received_inr": premium_received,
                "premium_prorated_inr": premium_prorated,
                "margin": margin_total if margin_total > 0 else None,
                "legs_detail": legs,
            }
        )
    return pd.DataFrame(rows), legs_by_trade


def assign_buckets(
    df_trades: pd.DataFrame,
    total_capital: float,
    allocations: Dict[str, float],
    thresholds: Dict[str, float],
    zone_map: Dict[str, str],
) -> pd.DataFrame:
    df = df_trades.copy()
    df["bucket"] = "Med"

    low_cap = total_capital * allocations[BUCKET_KEYS["Low"]] / 100.0
    med_cap = total_capital * allocations[BUCKET_KEYS["Med"]] / 100.0
    high_cap = total_capital * allocations[BUCKET_KEYS["High"]] / 100.0

    def _apply_bucket(row):
        zone_override = zone_map.get(str(row.get("zone_label", "")).strip(), "")
        if zone_override in {"low", "med", "high"}:
            target_cap = {"low": low_cap, "med": med_cap, "high": high_cap}[zone_override]
            trade_pct = row["trade_es99_inr"] / target_cap * 100.0 if target_cap > 0 else 0.0
            return zone_override.capitalize(), trade_pct
        trade_pct_low = row["trade_es99_inr"] / low_cap * 100.0 if low_cap > 0 else 0.0
        trade_pct_med = row["trade_es99_inr"] / med_cap * 100.0 if med_cap > 0 else 0.0
        trade_pct_high = row["trade_es99_inr"] / high_cap * 100.0 if high_cap > 0 else 0.0
        if trade_pct_low <= thresholds["low"]:
            return "Low", trade_pct_low
        if trade_pct_med <= thresholds["med"]:
            return "Med", trade_pct_med
        return "High", trade_pct_high

    results = df.apply(_apply_bucket, axis=1, result_type="expand")
    df["bucket"] = results[0]
    df["trade_es99_pct_of_bucket_cap"] = results[1]
    return df


def aggregate_portfolio(df_trades: pd.DataFrame) -> Dict[str, object]:
    portfolio_es99_inr = df_trades["trade_es99_inr"].sum()
    return {
        "portfolio_es99_inr": portfolio_es99_inr,
        "by_underlying": df_trades.groupby("underlying")["trade_es99_inr"].sum().sort_values(ascending=False),
        "by_trade": df_trades.groupby("trade_id")["trade_es99_inr"].sum().sort_values(ascending=False),
        "by_bucket": df_trades.groupby("bucket")["trade_es99_inr"].sum(),
    }


def aggregate_buckets(df_trades: pd.DataFrame, total_capital: float, allocations: Dict[str, float], bucket_limits: Dict[str, float] = None, total_margin_used: Optional[float] = None) -> pd.DataFrame:
    """
    Aggregate trades into buckets with capital allocation and ES99 metrics.
    
    Args:
        df_trades: DataFrame with trades and bucket assignments
        total_capital: Total account capital
        allocations: Target allocation percentages per bucket
        bucket_limits: ES99 limits per bucket
        total_margin_used: Actual margin used from Kite API (for accurate capital %)
    """
    # Calculate bucket metrics first pass to get total usage for proportional distribution
    bucket_usage = {}
    total_usage = 0.0
    
    for bucket in BUCKET_ORDER:
        bucket_df = df_trades[df_trades["bucket"] == bucket]
        # Use margin if available, otherwise ES99 as proxy
        if bucket_df["margin"].notna().any():
            usage = bucket_df["margin"].fillna(0.0).sum()
        else:
            usage = bucket_df["trade_es99_inr"].abs().sum()
        bucket_usage[bucket] = usage
        total_usage += usage
    
    rows = []
    for bucket in BUCKET_ORDER:
        bucket_capital = total_capital * allocations[BUCKET_KEYS[bucket]] / 100.0
        bucket_df = df_trades[df_trades["bucket"] == bucket]
        bucket_es99 = bucket_df["trade_es99_inr"].sum()
        bucket_pct = bucket_es99 / bucket_capital * 100.0 if bucket_capital else 0.0
        
        # Calculate actual allocated capital using portfolio margin if available
        if total_margin_used is not None and total_usage > 0:
            # Distribute total portfolio margin proportionally based on bucket usage
            actual_allocated_capital = total_margin_used * (bucket_usage[bucket] / total_usage)
        else:
            # Fallback to summing individual margins or ES99
            actual_allocated_capital = bucket_usage[bucket]
        
        # Calculate target allocation and delta
        bucket_limit = bucket_limits.get(bucket.lower(), 0.0) if bucket_limits else 0.0
        target_es99_inr = bucket_capital * bucket_limit / 100.0 if bucket_capital else 0.0
        delta_pct = bucket_pct - bucket_limit if bucket_limit else 0.0
        
        # Calculate capital allocation percentage (relative to total capital)
        actual_capital_pct = (actual_allocated_capital / total_capital * 100.0) if total_capital else 0.0
        target_capital_pct = allocations[BUCKET_KEYS[bucket]]
        capital_delta_pct = actual_capital_pct - target_capital_pct
        
        rows.append(
            {
                "bucket": bucket,
                "target_capital_pct": target_capital_pct,
                "bucket_capital_inr": bucket_capital,
                "actual_allocated_capital": actual_allocated_capital,
                "actual_capital_pct": actual_capital_pct,
                "capital_delta_pct": capital_delta_pct,
                "target_es99_inr": target_es99_inr,
                "target_es99_pct": bucket_limit,
                "bucket_es99_inr": bucket_es99,
                "bucket_es99_pct_of_bucket_capital": bucket_pct,
                "delta_pct": delta_pct,
                "trades": len(bucket_df),
                "legs": bucket_df["legs"].sum() if not bucket_df.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def _compute_portfolio_es99(positions: List[Dict[str, object]], account_size: float, lookback_days: int) -> Tuple[float, float]:
    if not positions:
        return 0.0, 0.0
    options_df_cache = st.session_state.get("options_df_cache", pd.DataFrame())
    iv_percentile = 35
    if isinstance(options_df_cache, pd.DataFrame) and not options_df_cache.empty and "iv" in options_df_cache.columns:
        iv_percentile = 35
    iv_regime, _ = get_iv_regime(iv_percentile)
    scenarios = get_weighted_scenarios(iv_regime)
    trade_greeks = calculate_portfolio_greeks(positions)
    spot = _get_spot_from_positions(positions)
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
        return 0.0, 0.0
    bucket_probs, _, _ = compute_historical_bucket_probabilities(
        lookback=int(lookback_days),
        smoothing_enabled=False,
        smoothing_span=63,
    )
    if not bucket_probs:
        bucket_probs = DEFAULT_BUCKET_PROBS.copy()
    bucket_counts = {}
    for row in derived_rows:
        bucket_counts[row["bucket"]] = bucket_counts.get(row["bucket"], 0) + 1
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
    return float(metrics.get("ES99", 0.0)), float(metrics.get("ES99Value", 0.0))


def _get_sigma_source(positions: List[Dict[str, object]]) -> Tuple[float, str]:
    options_df_cache = st.session_state.get("options_df_cache", pd.DataFrame())
    if isinstance(options_df_cache, pd.DataFrame) and not options_df_cache.empty and "iv" in options_df_cache.columns:
        iv = float(options_df_cache["iv"].dropna().median())
        return max(iv, 0.01), "ATM IV (options cache)"
    implieds = [p.get("implied_vol") for p in positions if p.get("implied_vol")]
    if implieds:
        iv = float(pd.Series(implieds).median())
        return max(iv, 0.01), "Position IV (median)"
    nifty_df = st.session_state.get("nifty_df_cache", pd.DataFrame())
    if isinstance(nifty_df, pd.DataFrame) and not nifty_df.empty and "close" in nifty_df.columns:
        returns = nifty_df["close"].pct_change().dropna()
        if not returns.empty:
            rv = float(returns.std() * math.sqrt(252))
            return max(rv, 0.01), "Realized vol (NIFTY)"
    return 0.15, "Fallback 15% vol"


def _get_spot_fallback(positions: List[Dict[str, object]]) -> float:
    spot = st.session_state.get("current_spot", 0.0)
    if spot:
        return float(spot)
    return float(next((p.get("spot_price") for p in positions if p.get("spot_price")), 0.0) or 0.0)


def _get_spot_from_positions(positions: List[Dict[str, object]]) -> float:
    spot_override = st.session_state.get("tba_spot_input", 0.0)
    if spot_override:
        return float(spot_override)
    return _get_spot_fallback(positions)


def _recompute_positions_with_spot(
    positions: List[Dict[str, object]],
    spot: float,
) -> List[Dict[str, object]]:
    options_df_cache = st.session_state.get("options_df_cache", pd.DataFrame())
    if spot <= 0 or not isinstance(options_df_cache, pd.DataFrame) or options_df_cache.empty:
        return positions
    recomputed = []
    for pos in positions:
        try:
            recomputed.append(enrich_position_with_greeks(pos, options_df_cache, float(spot)))
        except Exception:
            recomputed.append(pos)
    return recomputed


def _render_scenario_table(
    positions: List[Dict[str, object]],
    capital: float,
    lookback_days: int,
    es_limit_pct: float,
    title: str,
) -> None:
    st.markdown(f"### {title}")
    if not positions or capital <= 0:
        st.info("No positions available for scenario analysis.")
        return

    spot = _get_spot_from_positions(positions)
    if spot <= 0:
        st.info("Scenario analysis unavailable due to missing spot.")
        return

    options_df_cache = st.session_state.get("options_df_cache", pd.DataFrame())
    iv_percentile = 35
    if isinstance(options_df_cache, pd.DataFrame) and not options_df_cache.empty and "iv" in options_df_cache.columns:
        iv_percentile = 35
    iv_regime, _ = get_iv_regime(iv_percentile)
    scenarios = get_weighted_scenarios(iv_regime)

    greeks = calculate_portfolio_greeks(positions)
    threshold_context = build_threshold_report(
        portfolio={
            "delta": greeks.get("net_delta", 0.0),
            "gamma": greeks.get("net_gamma", 0.0),
            "vega": greeks.get("net_vega", 0.0),
            "spot": spot,
            "nav": capital,
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
        master_pct=es_limit_pct,
        hard_stop_pct=es_limit_pct * 1.2,
        normal_share=DEFAULT_THRESHOLD_NORMAL_SHARE,
        stress_share=DEFAULT_THRESHOLD_STRESS_SHARE,
    )
    derived_rows = threshold_context.get("rows", [])
    if not derived_rows:
        st.info("No scenario rows available to display.")
        return

    bucket_probs, _, _ = compute_historical_bucket_probabilities(
        lookback=int(lookback_days),
        smoothing_enabled=False,
        smoothing_span=63,
    )
    if not bucket_probs:
        bucket_probs = DEFAULT_BUCKET_PROBS.copy()

    bucket_counts = {}
    for row in derived_rows:
        bucket_counts[row["bucket"]] = bucket_counts.get(row["bucket"], 0) + 1
    for row in derived_rows:
        bucket = row["bucket"]
        bucket_prob = bucket_probs.get(bucket, 0.0)
        count = bucket_counts.get(bucket, 0)
        row["probability"] = bucket_prob / count if count and bucket_prob > 0 else 0.0

    repriced_map: Dict[str, Dict[str, object]] = {}
    repriced_skipped = {"missing_fields": 0, "no_price": 0}
    if getattr(ra_tab, "bs", None) is not None:
        try:
            repriced_rows, repriced_skipped = ra_tab._repriced_scenario_rows(
                positions,
                scenarios,
                spot,
                capital,
            )
            repriced_map = {row["Scenario"]: row for row in repriced_rows}
        except Exception:
            repriced_map = {}

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

    table_rows = []
    for row in derived_rows:
        scenario = row["scenario"]
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
            }
        )
    scenario_df = pd.DataFrame(table_rows)
    st.dataframe(scenario_df, use_container_width=True, hide_index=True, height=400)
    if repriced_skipped["missing_fields"] or repriced_skipped["no_price"]:
        st.caption(
            f"Repricing skipped: {repriced_skipped['missing_fields']} missing fields, "
            f"{repriced_skipped['no_price']} missing prices."
        )


def simulate_forward_pnl(
    positions: List[Dict[str, object]],
    config: SimulationConfig,
    limit_pct: Optional[float],
    limit_base: Optional[float],
) -> Optional[Dict[str, object]]:
    if not positions:
        return None
    sigma, sigma_source = _get_sigma_source(positions)
    iv_shift = 0.0
    if config.iv_mode == "IV Up Shock":
        iv_shift = config.iv_shock
    elif config.iv_mode == "IV Down Shock":
        iv_shift = -config.iv_shock
    if "IV" not in sigma_source and iv_shift != 0.0:
        iv_shift = math.copysign(max(sigma * 100.0 * 0.10, 0.5), iv_shift)
    spot = _get_spot_from_positions(positions)
    if not spot:
        return None

    greeks = calculate_portfolio_greeks(positions)
    delta = greeks.get("net_delta", 0.0)
    gamma = greeks.get("net_gamma", 0.0)
    vega = greeks.get("net_vega", 0.0)
    theta = greeks.get("net_theta", 0.0)

    dt = 1.0 / 252.0
    paths = max(200, int(config.paths))
    horizon = max(1, min(int(config.horizon_days), 20))
    z = np.random.normal(size=(paths, horizon))
    drift = 0.0
    increments = (drift - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z
    log_paths = np.cumsum(increments, axis=1)
    s_paths = spot * np.exp(log_paths)

    pnl_paths = np.zeros_like(s_paths)
    for t in range(horizon):
        dS = s_paths[:, t] - spot
        pnl_paths[:, t] = delta * dS + 0.5 * gamma * (dS ** 2) + vega * iv_shift + theta * (t + 1)

    pnl_tn = pnl_paths[:, -1]
    summary = {
        "mean": float(np.mean(pnl_tn)),
        "median": float(np.median(pnl_tn)),
        "p5": float(np.percentile(pnl_tn, 5)),
        "p1": float(np.percentile(pnl_tn, 1)),
        "prob_loss": float(np.mean(pnl_tn < 0)),
    }
    if limit_pct is not None and limit_base is not None:
        limit_inr = (limit_pct / 100.0) * limit_base
        summary["prob_breach"] = float(np.mean(pnl_tn < -limit_inr))
    else:
        summary["prob_breach"] = None

    fan = pd.DataFrame(
        {
            "Day": np.arange(1, horizon + 1),
            "P50": np.percentile(pnl_paths, 50, axis=0),
            "P10": np.percentile(pnl_paths, 10, axis=0),
            "P1": np.percentile(pnl_paths, 1, axis=0),
        }
    )

    shock_table = None
    if config.include_spot_shocks:
        shocks = [-0.035, -0.02, 0.02, 0.035]
        scenario_objs = [
            Scenario(f"Shock {shock*100:+.2f}%", ds_pct=shock * 100.0, div_pts=iv_shift, category="combined")
            for shock in shocks
        ]
        try:
            if getattr(ra_tab, "bs", None) is not None:
                repriced_rows, _ = ra_tab._repriced_scenario_rows(
                    positions,
                    scenario_objs,
                    spot,
                    limit_base if limit_base is not None else 0.0,
                )
                if repriced_rows:
                    shock_table = pd.DataFrame(repriced_rows)[["Scenario", "Repriced P&L (₹)"]]
        except Exception:
            shock_rows = []
            for shock in shocks:
                dS = spot * shock
                pnl = delta * dS + 0.5 * gamma * (dS ** 2) + vega * iv_shift + theta * horizon
                shock_rows.append({"Shock": f"{shock*100:+.2f}%", "P&L (₹)": pnl})
            shock_table = pd.DataFrame(shock_rows)
        
    return {
        "summary": summary,
        "fan": fan,
        "pnl_paths": pnl_paths,
        "sigma": sigma,
        "sigma_source": sigma_source,
        "iv_shift": iv_shift,
        "shock_table": shock_table,
    }


def _breach_day(curve: List[float], limit_inr: float) -> Optional[int]:
    for idx, val in enumerate(curve, start=1):
        if val <= limit_inr:
            return idx
    return None


def _format_breach(day: Optional[int], horizon: int) -> str:
    return f"{day}d" if day is not None else f"No breach in {horizon}d"


def _render_gate_settings(prefix: str) -> Dict[str, float]:
    defaults = {
        "gate_tail_ratio_watch": 30.0,
        "gate_tail_ratio_fail": 60.0,
        "gate_prob_loss_watch": 40.0,
        "gate_prob_loss_fail": 50.0,
        "gate_portfolio_breach_prob": 15.0,
        "gate_bucket_breach_prob_low": 5.0,
        "gate_bucket_breach_prob_med": 10.0,
        "gate_bucket_breach_prob_high": 15.0,
        "gate_p1_breach_fail_days": 2,
        "gate_portfolio_p10_fail_days": 4,
        "gate_bucket_p10_fail_days_med": 6,
        "gate_bucket_p10_fail_days_high": 3,
    }
    config: Dict[str, float] = {}
    with st.expander("Gate settings", expanded=False):
        for key, default in defaults.items():
            widget_key = f"{prefix}_{key}"
            st.session_state.setdefault(key, default)
            val = st.number_input(
                key.replace("gate_", "").replace("_", " ").title(),
                value=float(st.session_state.get(key, default)),
                step=1.0,
                key=widget_key,
            )
            st.session_state[key] = val
            config[key] = float(val)
    return config


def evaluate_gates_portfolio(
    kpis: Dict[str, float],
    limits: Dict[str, float],
    config: Dict[str, float],
) -> Tuple[List[Dict[str, str]], str, str]:
    gates: List[Dict[str, str]] = []
    actions = {
        "G1": "Reduce size / add hedges / re-bucket",
        "G2": "Reduce gamma (avoid near-expiry, widen strikes)",
        "G3": "Add convexity (long wings/long vol)",
        "G4": "Require higher IV or reduce exposure",
        "G5": "Reduce allocation to meet ES99 limit",
    }

    p1_breach = kpis.get("p1_breach_day")
    p10_breach = kpis.get("p10_breach_day")
    prob_breach = kpis.get("prob_breach_pct", 0.0)
    prob_loss = kpis.get("prob_loss_pct", 0.0)
    p1_horizon = kpis.get("p1_horizon", 0.0)
    p50_horizon = kpis.get("p50_horizon", 0.0)
    mean = kpis.get("mean", 0.0)
    median = kpis.get("median", 0.0)
    tail_ratio = kpis.get("tail_ratio", 0.0)
    total_capital = limits.get("total_capital", 0.0)
    portfolio_limit_inr = limits.get("portfolio_limit_inr", 0.0)

    # G1 Survivability
    status = "PASS"
    reasons = []
    if p1_breach is not None and p1_breach <= config["gate_p1_breach_fail_days"]:
        status = "FAIL"
        reasons.append("P1 breach ≤ 2d")
    if prob_breach > config["gate_portfolio_breach_prob"]:
        status = "FAIL"
        reasons.append("Prob breach > limit")
    if p1_horizon <= 1.25 * portfolio_limit_inr:
        status = "FAIL"
        reasons.append("P1 deeper than limit")
    p1_limit = 1.25 * portfolio_limit_inr
    gates.append(
        {
            "Gate": "G1 Survivability",
            "Value/Threshold": (
                f"P1 breach day: {p1_breach or '—'} (≤{int(config['gate_p1_breach_fail_days'])}) | "
                f"Prob breach: {prob_breach:.1f}% (limit {config['gate_portfolio_breach_prob']:.0f}%) | "
                f"P1 horizon: {format_inr(p1_horizon)} (≤{format_inr(p1_limit)})"
            ),
            "Status": status,
            "Reason": "; ".join(reasons) or "Within limits",
            "Action": actions["G1"],
        }
    )

    # G2 Speed of damage
    status = "PASS"
    reasons = []
    if p10_breach is not None and p10_breach <= config["gate_portfolio_p10_fail_days"]:
        status = "FAIL"
        reasons.append("P10 breach ≤ 4d")
    elif p10_breach is not None or (config["gate_prob_loss_watch"] <= prob_loss <= config["gate_prob_loss_fail"]):
        status = "WATCH"
        reasons.append("P10 breach within horizon or prob loss elevated")
    gates.append(
        {
            "Gate": "G2 Speed of damage",
            "Value/Threshold": (
                f"P10 breach day: {p10_breach or '—'} (FAIL ≤{int(config['gate_portfolio_p10_fail_days'])}) | "
                f"Prob loss: {prob_loss:.1f}% (watch ≥{config['gate_prob_loss_watch']:.0f}%)"
            ),
            "Status": status,
            "Reason": "; ".join(reasons) or "No near-term breach",
            "Action": actions["G2"],
        }
    )

    # G3 Asymmetry / convexity
    status = "PASS"
    reasons = []
    if tail_ratio > config["gate_tail_ratio_fail"]:
        status = "FAIL"
        reasons.append("Tail ratio > 60")
    elif tail_ratio > config["gate_tail_ratio_watch"]:
        status = "WATCH"
        reasons.append("Tail ratio 30–60")
    if mean < 0 and median > 0:
        status = "WATCH" if status == "PASS" else status
        reasons.append("Mean < 0 with Median > 0")
    gates.append(
        {
            "Gate": "G3 Asymmetry / convexity",
            "Value/Threshold": (
                f"Tail ratio: {tail_ratio:.1f} (watch>{config['gate_tail_ratio_watch']:.0f}, "
                f"fail>{config['gate_tail_ratio_fail']:.0f}) | Mean/Median: {format_inr(mean)} / {format_inr(median)}"
            ),
            "Status": status,
            "Reason": "; ".join(reasons) or "Balanced tail",
            "Action": actions["G3"],
        }
    )

    # G4 Carry quality
    status = "PASS"
    reasons = []
    median_pct = (median / total_capital * 100.0) if total_capital else 0.0
    if median <= 0 and mean < 0:
        status = "FAIL"
        reasons.append("Median ≤ 0 and Mean < 0")
    elif prob_loss > config["gate_prob_loss_watch"] or median_pct < 0.05:
        status = "WATCH"
        reasons.append("Prob loss high or median too small")
    gates.append(
        {
            "Gate": "G4 Carry quality",
            "Value/Threshold": (
                f"Median: {format_inr(median)} | Mean: {format_inr(mean)} | "
                f"Prob loss: {prob_loss:.1f}% (watch ≥{config['gate_prob_loss_watch']:.0f}%)"
            ),
            "Status": status,
            "Reason": "; ".join(reasons) or "Carry acceptable",
            "Action": actions["G4"],
        }
    )

    # G5 Sizing (informational)
    status = "PASS"
    reasons = []
    es99_pct = limits.get("portfolio_es99_pct")
    limit_pct = limits.get("portfolio_limit_pct")
    if es99_pct is None or limit_pct is None:
        status = "WATCH"
        reasons.append("ES99 unavailable")
    else:
        if es99_pct > limit_pct:
            status = "FAIL"
            reasons.append("ES99 above limit")
        elif es99_pct > 0.8 * limit_pct:
            status = "WATCH"
            reasons.append("ES99 near limit")
    gates.append(
        {
            "Gate": "G5 Sizing",
            "Value/Threshold": (
                f"ES99: {es99_pct:.2f}% (limit {limit_pct:.2f}%)"
                if es99_pct is not None and limit_pct is not None
                else "ES99: —"
            ),
            "Status": status,
            "Reason": "; ".join(reasons) or "Within ES99 limit",
            "Action": actions["G5"],
        }
    )

    overall = "PASS"
    if any(g["Status"] == "FAIL" for g in gates[:3]):
        overall = "FAIL"
    elif any(g["Status"] == "WATCH" for g in gates):
        overall = "WATCH"

    suggested_action = ""
    first_fail = next((g for g in gates if g["Status"] == "FAIL"), None)
    if first_fail:
        gate_key = first_fail["Gate"].split()[0]
        suggested_action = actions.get(gate_key, "")
    return gates, overall, suggested_action


def evaluate_gates_bucket(
    kpis: Dict[str, float],
    limits: Dict[str, float],
    config: Dict[str, float],
    bucket_name: str,
) -> Tuple[List[Dict[str, str]], str, str]:
    gates: List[Dict[str, str]] = []
    actions = {
        "G1": "Reduce size / add hedges / re-bucket",
        "G2": "Reduce gamma (avoid near-expiry, widen strikes)",
        "G3": "Add convexity (long wings/long vol)",
        "G4": "Require higher IV or reduce exposure",
        "G5": "Reduce allocation to meet ES99 limit",
    }

    p1_breach = kpis.get("p1_breach_day")
    p10_breach = kpis.get("p10_breach_day")
    prob_breach = kpis.get("prob_breach_pct", 0.0)
    prob_loss = kpis.get("prob_loss_pct", 0.0)
    p1_horizon = kpis.get("p1_horizon", 0.0)
    p50_horizon = kpis.get("p50_horizon", 0.0)
    mean = kpis.get("mean", 0.0)
    median = kpis.get("median", 0.0)
    tail_ratio = kpis.get("tail_ratio", 0.0)
    bucket_limit_inr = limits.get("bucket_limit_inr", 0.0)

    bucket_key = bucket_name.lower()
    breach_prob_limit = config.get(f"gate_bucket_breach_prob_{bucket_key}", 10.0)

    # G1 Survivability
    status = "PASS"
    reasons = []
    if p1_breach is not None and p1_breach <= config["gate_p1_breach_fail_days"]:
        status = "FAIL"
        reasons.append("P1 breach ≤ 2d")
    if prob_breach > breach_prob_limit:
        status = "FAIL"
        reasons.append("Prob breach > limit")
    if p1_horizon <= 1.25 * bucket_limit_inr:
        status = "FAIL"
        reasons.append("P1 deeper than limit")
    p1_limit = 1.25 * bucket_limit_inr
    gates.append(
        {
            "Gate": "G1 Survivability",
            "Value/Threshold": (
                f"P1 breach day: {p1_breach or '—'} (≤{int(config['gate_p1_breach_fail_days'])}) | "
                f"Prob breach: {prob_breach:.1f}% (limit {breach_prob_limit:.0f}%) | "
                f"P1 horizon: {format_inr(p1_horizon)} (≤{format_inr(p1_limit)})"
            ),
            "Status": status,
            "Reason": "; ".join(reasons) or "Within limits",
            "Action": actions["G1"],
        }
    )

    # G2 Speed of damage
    status = "PASS"
    reasons = []
    if bucket_key == "low":
        if p10_breach is not None:
            status = "FAIL"
            reasons.append("Any P10 breach")
    elif bucket_key == "med":
        if p10_breach is not None and p10_breach <= config["gate_bucket_p10_fail_days_med"]:
            status = "FAIL"
            reasons.append("P10 breach ≤ 6d")
        elif p10_breach is not None:
            status = "WATCH"
            reasons.append("P10 breach within horizon")
    else:
        if p10_breach is not None and p10_breach <= config["gate_bucket_p10_fail_days_high"]:
            status = "FAIL"
            reasons.append("P10 breach ≤ 3d")
        elif p10_breach is not None:
            status = "WATCH"
            reasons.append("P10 breach within horizon")
    if bucket_key == "low":
        p10_rule = "FAIL if any breach"
    elif bucket_key == "med":
        p10_rule = f"FAIL ≤{int(config['gate_bucket_p10_fail_days_med'])}"
    else:
        p10_rule = f"FAIL ≤{int(config['gate_bucket_p10_fail_days_high'])}"
    gates.append(
        {
            "Gate": "G2 Speed of damage",
            "Value/Threshold": f"P10 breach day: {p10_breach or '—'} ({p10_rule})",
            "Status": status,
            "Reason": "; ".join(reasons) or "No near-term breach",
            "Action": actions["G2"],
        }
    )

    # G3 Asymmetry / convexity
    status = "PASS"
    reasons = []
    if tail_ratio > config["gate_tail_ratio_fail"]:
        status = "FAIL"
        reasons.append("Tail ratio > 60")
    elif tail_ratio > config["gate_tail_ratio_watch"]:
        status = "WATCH"
        reasons.append("Tail ratio 30–60")
    if mean < 0 and median > 0:
        status = "WATCH" if status == "PASS" else status
        reasons.append("Mean < 0 with Median > 0")
    gates.append(
        {
            "Gate": "G3 Asymmetry / convexity",
            "Value/Threshold": (
                f"Tail ratio: {tail_ratio:.1f} (watch>{config['gate_tail_ratio_watch']:.0f}, "
                f"fail>{config['gate_tail_ratio_fail']:.0f}) | Mean/Median: {format_inr(mean)} / {format_inr(median)}"
            ),
            "Status": status,
            "Reason": "; ".join(reasons) or "Balanced tail",
            "Action": actions["G3"],
        }
    )

    # G4 Carry quality
    status = "PASS"
    reasons = []
    if median <= 0 and mean < 0:
        status = "FAIL"
        reasons.append("Median ≤ 0 and Mean < 0")
    else:
        if bucket_key in {"low", "med"}:
            if prob_loss > config["gate_prob_loss_fail"]:
                status = "FAIL"
                reasons.append("Prob loss > 50%")
            elif prob_loss > config["gate_prob_loss_watch"]:
                status = "WATCH"
                reasons.append("Prob loss > 40%")
        else:
            if prob_loss > config["gate_prob_loss_fail"]:
                status = "WATCH"
                reasons.append("Prob loss > 50% (High bucket)")
    gates.append(
        {
            "Gate": "G4 Carry quality",
            "Value/Threshold": (
                f"Median: {format_inr(median)} | Mean: {format_inr(mean)} | "
                f"Prob loss: {prob_loss:.1f}% (watch ≥{config['gate_prob_loss_watch']:.0f}%)"
            ),
            "Status": status,
            "Reason": "; ".join(reasons) or "Carry acceptable",
            "Action": actions["G4"],
        }
    )

    # G5 Sizing
    status = "PASS"
    reasons = []
    bucket_es99_pct = limits.get("bucket_es99_pct")
    bucket_limit_pct = limits.get("bucket_limit_pct")
    if bucket_es99_pct is None or bucket_limit_pct is None:
        status = "WATCH"
        reasons.append("ES99 unavailable")
    else:
        if bucket_es99_pct > bucket_limit_pct:
            status = "FAIL"
            reasons.append("ES99 above limit")
        elif bucket_es99_pct > 0.8 * bucket_limit_pct:
            status = "WATCH"
            reasons.append("ES99 near limit")
    gates.append(
        {
            "Gate": "G5 Sizing",
            "Value/Threshold": (
                f"ES99: {bucket_es99_pct:.2f}% (limit {bucket_limit_pct:.2f}%)"
                if bucket_es99_pct is not None and bucket_limit_pct is not None
                else "ES99: —"
            ),
            "Status": status,
            "Reason": "; ".join(reasons) or "Within ES99 limit",
            "Action": actions["G5"],
        }
    )

    overall = "PASS"
    if any(g["Status"] == "FAIL" for g in gates[:3]):
        overall = "FAIL"
    elif any(g["Status"] == "WATCH" for g in gates):
        overall = "WATCH"

    suggested_action = ""
    first_fail = next((g for g in gates if g["Status"] == "FAIL"), None)
    if first_fail:
        gate_key = first_fail["Gate"].split()[0]
        suggested_action = actions.get(gate_key, "")
    return gates, overall, suggested_action


def _select_representative_paths(pnl_paths: np.ndarray, num_samples: int) -> List[int]:
    num_paths = pnl_paths.shape[0]
    if num_paths <= 0:
        return []
    final_pnl = pnl_paths[:, -1]
    targets = [
        (np.percentile(final_pnl, 50), 3),
        (np.percentile(final_pnl, 10), 3),
        (np.percentile(final_pnl, 1), 2),
    ]
    selected: List[int] = []
    used = set()
    for target, count in targets:
        order = np.argsort(np.abs(final_pnl - target))
        picks = 0
        for idx in order:
            if idx in used:
                continue
            selected.append(int(idx))
            used.add(int(idx))
            picks += 1
            if picks >= count:
                break
    remaining = [i for i in range(num_paths) if i not in used]
    if remaining and len(selected) < num_samples:
        extra = np.random.choice(
            remaining, size=min(len(remaining), num_samples - len(selected)), replace=False
        )
        selected.extend([int(i) for i in extra])
    return selected[: min(num_samples, num_paths)]


def _render_simulation(
    sim: Optional[Dict[str, object]],
    title: str,
    portfolio_limit_inr: Optional[float],
    bucket_limit_inr: Optional[float],
    bucket_label: Optional[str],
    key_prefix: str,
    gate_config: Optional[Dict[str, float]] = None,
    gate_limits: Optional[Dict[str, float]] = None,
) -> None:
    st.markdown(f"#### {title}")
    if sim is None:
        st.info("Forward simulation unavailable due to missing IV/pricer.")
        return
    fan = sim.get("fan")
    if not isinstance(fan, pd.DataFrame) or fan.empty:
        st.error("Forward simulation fan chart data missing. Please rerun the simulation.")
        return
    required_cols = {"P1", "P10", "P50"}
    if not required_cols.issubset(set(fan.columns)):
        st.error("Forward simulation curves missing (P1/P10/P50). Please rerun the simulation.")
        return

    summary = sim["summary"]
    cols = st.columns(5)
    cols[0].metric(
        "Mean P&L",
        format_inr(summary["mean"]),
        help=(
            "Definition: Average P&L across all simulated paths at T+N.\n"
            "Why: Captures whether you are being paid for risk on average.\n"
            "Interpret: Mean < 0 with Median > 0 ⇒ frequent small wins, rare large losses (short convexity).\n"
            "Action if bad: reduce short-gamma exposure, widen strikes, add wings/hedge, avoid low-IV regimes.\n"
            "Rule of thumb: For income books, mean should be ≥ 0 over the chosen horizon."
        ),
    )
    cols[1].metric(
        "Median P&L",
        format_inr(summary["median"]),
        help=(
            "Definition: 50th percentile (‘typical’) P&L at T+N.\n"
            "Why: Shows the most common outcome, not the best.\n"
            "Interpret: Small/flat median means theta is weak vs path risk.\n"
            "Action if low: don’t rely on ‘carry’; either increase edge (enter only at higher IV) or reduce exposure.\n"
            "Rule: Median should be meaningfully positive if strategy is carry-driven."
        ),
    )
    cols[2].metric(
        "5% Quantile",
        format_inr(summary["p5"]),
        help=(
            "Definition: P&L level that only 5% of scenarios are worse than (bad-but-plausible).\n"
            "Why: Measures drawdown you should expect occasionally.\n"
            "Action if too negative: cut size, shorten holding window, avoid event weeks, add defined-risk hedges.\n"
            "Rule: P5 loss should not exceed your bucket monthly loss tolerance."
        ),
    )
    cols[3].metric(
        "1% Quantile",
        format_inr(summary["p1"]),
        help=(
            "Definition: Extreme tail P&L (1 in 100 scenarios).\n"
            "Why: Survivability metric; shows tail blow-up magnitude.\n"
            "Action if too negative: move trade to High bucket, cap exposure, buy tail hedges, or avoid this structure.\n"
            "Rule: P1 should be compatible with ‘I can continue trading after this happens’."
        ),
    )
    cols[4].metric(
        "Prob. Loss",
        f"{summary['prob_loss']*100:.1f}%",
        help=(
            "Definition: % of scenarios with P&L < 0 at T+N.\n"
            "Why: A carry strategy should win more often than it loses.\n"
            "Action if high: stop selling cheap theta; require higher IV, reduce gamma, or switch to defined-risk directional spreads.\n"
            "Rule: <35–40% is healthy carry; 40–50% watch; >50% weak carry."
        ),
    )
    if portfolio_limit_inr is not None and summary.get("prob_breach") is not None:
        st.metric(
            "Prob. Breach limit",
            f"{summary['prob_breach']*100:.1f}%",
            help=(
                "Definition: % of scenarios that cross the configured loss limit within the horizon.\n"
                "Why: Directly measures ‘how often you violate risk policy’.\n"
                "Action if high: lower position size, tighten bucket limits, reduce correlated exposure, add hedges, shorten horizon.\n"
                "Rule: Low bucket <5%, Medium <10%, High <15% (approx)."
            ),
        )

    p1 = fan["P1"].tolist()
    p10 = fan["P10"].tolist()
    p50 = fan["P50"].tolist()
    horizon = len(p50)
    if not horizon:
        st.error("Forward simulation curves are empty. Please rerun the simulation.")
        return

    tail_ratio = abs(p1[-1]) / max(1.0, abs(p50[-1]))
    p10_port = None
    p01_port = None
    if bucket_limit_inr is not None and bucket_label:
        p10_bucket = _breach_day(p10, bucket_limit_inr)
        p01_bucket = _breach_day(p1, bucket_limit_inr)
        p10_port = _breach_day(p10, portfolio_limit_inr) if portfolio_limit_inr is not None else None
        p01_port = _breach_day(p1, portfolio_limit_inr) if portfolio_limit_inr is not None else None
        kpi_cols = st.columns(3)
        kpi_cols[0].metric(
            "P10 breach: bucket / portfolio",
            f"{_format_breach(p10_bucket, horizon)} / {_format_breach(p10_port, horizon) if portfolio_limit_inr is not None else '—'}",
            help=(
                "Definition: First day when the P10 curve (10% worst outcomes) crosses the loss limit.\n"
                "Why: ‘Time-to-trouble’ for normal bad periods.\n"
                "Action if small: you have little time to adjust—reduce gamma, avoid near-expiry shorts, widen wings, reduce leverage.\n"
                "Rule: For Low bucket, breach day should be beyond horizon (no breach)."
            ),
        )
        kpi_cols[1].metric(
            "P1 breach: bucket / portfolio",
            f"{_format_breach(p01_bucket, horizon)} / {_format_breach(p01_port, horizon) if portfolio_limit_inr is not None else '—'}",
            help=(
                "Definition: First day when the P1 curve (1% tail) crosses the loss limit.\n"
                "Why: Measures gap/tail speed—how fast the worst case overwhelms you.\n"
                "Action if 1–2 days: treat as tail-dominated; use only High bucket + hard stops/hedges, or redesign structure.\n"
                "Rule: Ideally ‘No breach in N days’ for Low/Med buckets."
            ),
        )
        kpi_cols[2].metric(
            "Tail ratio |P1| / |P50| (T+N)",
            f"{tail_ratio:.2f}",
            help=(
                "Definition: Tail severity relative to typical outcome at horizon.\n"
                "Why: Flags ‘pennies in front of steamroller’ profiles.\n"
                "Action if high: add convexity (long wings/long vol), widen strikes, reduce size, avoid short-dated exposure.\n"
                "Rule: <30 healthy income, 30–60 aggressive, >60 tail-dominated."
            ),
        )
    else:
        p10_port = _breach_day(p10, portfolio_limit_inr) if portfolio_limit_inr is not None else None
        p01_port = _breach_day(p1, portfolio_limit_inr) if portfolio_limit_inr is not None else None
        kpi_cols = st.columns(3)
        kpi_cols[0].metric(
            "P10 days to breach (portfolio)",
            _format_breach(p10_port, horizon),
            help=(
                "Definition: First day when the P10 curve (10% worst outcomes) crosses the loss limit.\n"
                "Why: ‘Time-to-trouble’ for normal bad periods.\n"
                "Action if small: you have little time to adjust—reduce gamma, avoid near-expiry shorts, widen wings, reduce leverage.\n"
                "Rule: For Low bucket, breach day should be beyond horizon (no breach)."
            ),
        )
        kpi_cols[1].metric(
            "P1 days to breach (portfolio)",
            _format_breach(p01_port, horizon),
            help=(
                "Definition: First day when the P1 curve (1% tail) crosses the loss limit.\n"
                "Why: Measures gap/tail speed—how fast the worst case overwhelms you.\n"
                "Action if 1–2 days: treat as tail-dominated; use only High bucket + hard stops/hedges, or redesign structure.\n"
                "Rule: Ideally ‘No breach in N days’ for Low/Med buckets."
            ),
        )
        kpi_cols[2].metric(
            "Tail ratio |P1| / |P50| (T+N)",
            f"{tail_ratio:.2f}",
            help=(
                "Definition: Tail severity relative to typical outcome at horizon.\n"
                "Why: Flags ‘pennies in front of steamroller’ profiles.\n"
                "Action if high: add convexity (long wings/long vol), widen strikes, reduce size, avoid short-dated exposure.\n"
                "Rule: <30 healthy income, 30–60 aggressive, >60 tail-dominated."
            ),
        )

    gate_config = gate_config or {}
    gate_limits = gate_limits or {}
    kpis = {
        "p1_horizon": float(p1[-1]),
        "p10_horizon": float(p10[-1]),
        "p50_horizon": float(p50[-1]),
        "mean": float(summary["mean"]),
        "median": float(summary["median"]),
        "prob_loss_pct": float(summary["prob_loss"] * 100.0),
        "prob_breach_pct": float(summary.get("prob_breach") or 0.0) * 100.0,
        "p1_breach_day": p01_bucket if bucket_limit_inr is not None and bucket_label else p01_port,
        "p10_breach_day": p10_bucket if bucket_limit_inr is not None and bucket_label else p10_port,
        "tail_ratio": float(tail_ratio),
    }

    st.markdown("##### Decision Gates")
    if bucket_label:
        gates, overall, suggested = evaluate_gates_bucket(kpis, gate_limits, gate_config, bucket_label)
    else:
        gates, overall, suggested = evaluate_gates_portfolio(kpis, gate_limits, gate_config)
    badge = {"PASS": "✅", "WATCH": "⚠️", "FAIL": "❌"}
    gate_df = pd.DataFrame(
        [
            {
                "Gate": g["Gate"],
                "Value/Threshold": g.get("Value/Threshold", "—"),
                "Status": f"{badge.get(g['Status'], '')} {g['Status']}",
                "Reason": g["Reason"],
                "Action": g["Action"],
            }
            for g in gates
        ]
    )
    st.dataframe(gate_df, use_container_width=True, hide_index=True)
    st.markdown(f"**Overall verdict:** {badge.get(overall, '')} {overall}")
    if suggested:
        st.caption(f"Suggested action: {suggested}")

    st.markdown("##### P&L Distribution Over Time")
    plot_df = fan.set_index("Day").copy()
    if portfolio_limit_inr is not None:
        plot_df["Portfolio limit (₹)"] = float(portfolio_limit_inr)
    if bucket_limit_inr is not None and bucket_label:
        plot_df[f"{bucket_label} limit (₹)"] = float(bucket_limit_inr)
    
    st.session_state.setdefault(f"{key_prefix}_show_paths", False)
    st.session_state.setdefault(f"{key_prefix}_num_paths", 10)
    pnl_paths = sim.get("pnl_paths")
    show_paths = st.checkbox(
        "Show sample simulated paths (illustrative)",
        value=st.session_state.get(f"{key_prefix}_show_paths", False),
        key=f"{key_prefix}_show_paths",
        disabled=pnl_paths is None,
    )
    st.caption(
        "These are randomly selected paths for intuition only; percentiles are the decision metric."
    )
    if pnl_paths is None:
        st.info("Sample paths unavailable (missing pnl_paths).")
    sample_df = None
    if show_paths:
        pnl_paths = np.asarray(pnl_paths)
        if pnl_paths.ndim == 2 and pnl_paths.shape[0] > 0:
            max_slider = min(20, pnl_paths.shape[0])
            num_samples = st.slider(
                "Sample paths",
                min_value=3,
                max_value=max_slider,
                value=min(int(st.session_state.get(f"{key_prefix}_num_paths", 10)), max_slider),
                key=f"{key_prefix}_num_paths",
            )
            sample_idx = _select_representative_paths(pnl_paths, int(num_samples))
            if sample_idx:
                matrix = pnl_paths[sample_idx, :].T
                days = plot_df.index.to_numpy()
                num_paths = matrix.shape[1]
                sample_df = pd.DataFrame(
                    {
                        "Day": np.tile(days, num_paths),
                        "Path": [f"P{i+1}" for i in range(num_paths) for _ in range(len(days))],
                        "Value": matrix.flatten(order="F"),
                        "Series": "Sample paths (illustrative)",
                    }
                )
        else:
            st.info("Sample paths unavailable (invalid pnl_paths shape).")

    # Create Altair chart with colored tooltips
    plot_df_reset = plot_df.reset_index()
    
    # Create selection for nearest point
    nearest = alt.selection_point(nearest=True, on='mouseover', fields=['Day'], empty=False)
    
    # Melt for line drawing and tooltip with colors
    melted = plot_df_reset.melt(id_vars=['Day'], var_name='Series', value_name='Value')
    
    # Define color scale to ensure consistent colors
    color_domain = list(plot_df.columns)
    if sample_df is not None and not sample_df.empty:
        color_domain = color_domain + ["Sample paths (illustrative)"]
    color_scale = alt.Scale(domain=color_domain)
    
    # Base line chart
    lines = alt.Chart(melted).mark_line().encode(
        x=alt.X('Day:Q', title='Day'),
        y=alt.Y('Value:Q', title='P&L (₹)'),
        color=alt.Color('Series:N', scale=color_scale, legend=alt.Legend(title=None))
    )

    sample_lines = None
    if sample_df is not None and not sample_df.empty:
        sample_lines = alt.Chart(sample_df).mark_line(opacity=0.15).encode(
            x=alt.X("Day:Q"),
            y=alt.Y("Value:Q"),
            detail=alt.Detail("Path:N"),
            color=alt.Color("Series:N", legend=alt.Legend(title=None)),
        )
    
    # Points on hover with colors shown in tooltip
    points = alt.Chart(melted).mark_circle(size=100).encode(
        x=alt.X('Day:Q'),
        y=alt.Y('Value:Q'),
        color=alt.Color('Series:N', scale=color_scale, legend=None),
        opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
        tooltip=[
            alt.Tooltip('Day:Q', title='Day'),
            alt.Tooltip('Series:N', title='color'),
            alt.Tooltip('Value:Q', title='value', format=',.2f')
        ]
    ).add_params(nearest)
    
    chart_layers = [lines, points]
    if sample_lines is not None:
        chart_layers = [sample_lines] + chart_layers
    chart = alt.layer(*chart_layers).properties(height=400).interactive()
    
    st.altair_chart(chart, use_container_width=True)
    
    with st.expander("Assumptions"):
        st.markdown(
            f"- Model: risk-neutral GBM (drift=0)\n"
            f"- Sigma source: {sim['sigma_source']}\n"
            f"- IV mode: {st.session_state.get('tba_iv_mode')}\n"
            f"- Paths: {st.session_state.get('tba_sim_paths')}, Horizon: {st.session_state.get('tba_sim_days')} days"
        )


def render_risk_buckets_tab() -> None:
    _init_tba_state()
    _sync_total_capital_from_account()

    st.markdown("## TBA — Institutional Risk View")
    
    # Sync positions button
    if st.sidebar.button("🔄 Sync Positions", key="risk_buckets_sync_positions", help="Fetch latest positions from Kite", use_container_width=True, type="primary"):
        st.rerun()
    
    positions = load_positions()
    if not positions:
        st.warning("No positions loaded. Fetch positions from Portfolio tab first.")
        return

    if not st.session_state.get("tba_spot_input"):
        fallback_spot = _get_spot_fallback(positions)
        if fallback_spot > 0:
            st.session_state["tba_spot_input"] = fallback_spot
    spot_override = float(st.session_state.get("tba_spot_input") or 0.0)
    if spot_override > 0:
        if st.session_state.get("tba_spot_recomputed_for") != spot_override:
            positions = _recompute_positions_with_spot(positions, spot_override)
            st.session_state["enriched_positions"] = positions
            st.session_state["current_spot"] = spot_override
            st.session_state["tba_spot_recomputed_for"] = spot_override

    controls = st.sidebar
    with controls.expander("Portfolio Controls", expanded=False):
        total_capital = st.number_input(
            "Total capital (INR)",
            min_value=0.0,
            step=50_000.0,
            format="%.2f",
            value=st.session_state["tba_total_capital"],
            key="tba_total_capital",
        )
        st.number_input(
            "Spot price (override)",
            min_value=0.0,
            step=10.0,
            value=st.session_state.get("tba_spot_input", 0.0),
            key="tba_spot_input",
            help="Overrides spot used in ES99/scenario calculations for this tab.",
        )
        alloc_low = st.number_input("Low bucket allocation %", min_value=0.0, max_value=100.0, step=1.0, value=st.session_state["tba_alloc_low"], key="tba_alloc_low")
        alloc_med = st.number_input("Med bucket allocation %", min_value=0.0, max_value=100.0, step=1.0, value=st.session_state["tba_alloc_med"], key="tba_alloc_med")
        alloc_high = st.number_input("High bucket allocation %", min_value=0.0, max_value=100.0, step=1.0, value=st.session_state["tba_alloc_high"], key="tba_alloc_high")
        portfolio_es_limit = st.number_input("Portfolio ES99 limit %", min_value=0.0, step=0.1, value=st.session_state["tba_portfolio_es_limit"], key="tba_portfolio_es_limit")
    allocations = {"low": alloc_low, "med": alloc_med, "high": alloc_high}
    if abs(sum(allocations.values()) - 100.0) > 0.01:
        st.error("Bucket allocation must sum to 100%.")

    with controls.expander("Forward Simulation Controls", expanded=False):
        st.number_input("Horizon (days)", min_value=1, max_value=20, step=1, value=st.session_state["tba_sim_days"], key="tba_sim_days")
        st.number_input("Simulation paths", min_value=200, max_value=10000, step=100, value=st.session_state["tba_sim_paths"], key="tba_sim_paths")
        st.selectbox("IV mode", ["IV Flat", "IV Up Shock", "IV Down Shock"], index=["IV Flat", "IV Up Shock", "IV Down Shock"].index(st.session_state["tba_iv_mode"]), key="tba_iv_mode")
        st.number_input("IV shock (vol points)", min_value=0.0, step=0.5, value=st.session_state["tba_iv_shock"], key="tba_iv_shock")
        st.caption("Note: Scenario analysis (±2%, ±3.5% shocks) is always included")

    saved_groups = st.session_state.get("tba_saved_trades", [])
    use_saved = st.session_state.get("tba_use_saved_groups", False)
    trades = build_trades_using_existing_grouping(positions)
    if use_saved and saved_groups:
        trades += build_trades_from_saved_groups(positions, saved_groups)
    trades_df, legs_by_trade = compute_trade_risk(trades, float(total_capital), lookback_days=504)

    thresholds = {
        "low": float(st.session_state.get("tba_trade_low_max")),
        "med": float(st.session_state.get("tba_trade_med_max")),
    }
    zone_map = _parse_zone_map(st.session_state.get("tba_zone_map", ""))
    trades_df = assign_buckets(trades_df, float(total_capital), allocations, thresholds, zone_map)

    subtab1, subtab2, subtab3, subtab4 = st.tabs(
        ["Portfolio Level", "Bucket Level", "Trade Level", "Meta Information"]
    )

    with subtab1:
        st.markdown("### Portfolio ES99")
        portfolio_es_pct, portfolio_es_inr = _compute_portfolio_es99(positions, float(total_capital), 504)
        if portfolio_es_inr == 0:
            portfolio_es_inr = aggregate_portfolio(trades_df)["portfolio_es99_inr"]
        portfolio_es_pct = (portfolio_es_inr / total_capital * 100.0) if total_capital else 0.0
        
        # Display portfolio metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Portfolio ES99", f"{portfolio_es_pct:.2f}%", format_inr(portfolio_es_inr))
        with col2:
            margin_used = st.session_state.get("margin_used")
            if margin_used is not None:
                margin_pct = (float(margin_used) / total_capital * 100.0) if total_capital else 0.0
                st.metric("Margin Used", f"{margin_pct:.1f}%", format_inr(margin_used))
        
        if portfolio_es_pct > portfolio_es_limit:
            st.error("Portfolio kill switch: RED")
        else:
            st.success("Portfolio kill switch: OK")

        agg = aggregate_portfolio(trades_df)
        
        # ES99 charts in 2 columns
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("### Top ES99 Contributors (Underlying)")
            data = agg["by_underlying"].head(10).reset_index()
            data.columns = ['name', 'value']
            chart = alt.Chart(data).mark_bar(size=50).encode(
                x=alt.X('name:N', title=None),
                y=alt.Y('value:Q', title=None)
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
        with chart_col2:
            st.markdown("### ES99 by Bucket")
            data = agg["by_bucket"].reset_index()
            data.columns = ['name', 'value']
            chart = alt.Chart(data).mark_bar(size=50).encode(
                x=alt.X('name:N', title=None),
                y=alt.Y('value:Q', title=None)
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
        
        # Trade week table below
        st.markdown("### Top ES99 Contributors (Trade Week)")
        top_trades = (
            trades_df.groupby(["underlying", "week_id", "option_side"])["trade_es99_inr"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
            .reset_index()
        )
        st.dataframe(top_trades, use_container_width=True, hide_index=True)

        st.markdown("### Forward 10-Day Risk/Return")
        gate_config = _render_gate_settings("portfolio_gates")
        sim_cfg = SimulationConfig(
            horizon_days=int(st.session_state.get("tba_sim_days")),
            paths=int(st.session_state.get("tba_sim_paths")),
            iv_mode=st.session_state.get("tba_iv_mode"),
            iv_shock=float(st.session_state.get("tba_iv_shock")),
            include_spot_shocks=True,  # Always include scenarios for portfolio level
        )
        sim = simulate_forward_pnl(positions, sim_cfg, portfolio_es_limit, total_capital)
        portfolio_limit_inr = -float(total_capital) * float(portfolio_es_limit) / 100.0
        _render_simulation(
            sim,
            "Portfolio Forward Simulation",
            portfolio_limit_inr,
            None,
            None,
            "tba_portfolio_sim",
            gate_config=gate_config,
            gate_limits={
                "total_capital": float(total_capital),
                "portfolio_limit_inr": portfolio_limit_inr,
                "portfolio_limit_pct": float(portfolio_es_limit),
                "portfolio_es99_pct": float(portfolio_es_pct),
            },
        )
        _render_scenario_table(
            positions,
            float(total_capital),
            lookback_days=504,
            es_limit_pct=float(portfolio_es_limit),
            title="Scenario Table (with calibrated probabilities)",
        )

    with subtab2:
        with controls.expander("Bucket Controls", expanded=False):
            st.session_state.setdefault("tba_bucket_es_limit_low", 2.0)
            st.session_state.setdefault("tba_bucket_es_limit_med", 3.0)
            st.session_state.setdefault("tba_bucket_es_limit_high", 5.0)
            st.number_input("Low bucket ES99 limit %", min_value=0.0, step=0.1, value=st.session_state["tba_bucket_es_limit_low"], key="tba_bucket_es_limit_low")
            st.number_input("Med bucket ES99 limit %", min_value=0.0, step=0.1, value=st.session_state["tba_bucket_es_limit_med"], key="tba_bucket_es_limit_med")
            st.number_input("High bucket ES99 limit %", min_value=0.0, step=0.1, value=st.session_state["tba_bucket_es_limit_high"], key="tba_bucket_es_limit_high")
            st.number_input("Low trade ES99 max %", min_value=0.0, step=0.1, value=st.session_state["tba_trade_low_max"], key="tba_trade_low_max")
            st.number_input("Med trade ES99 max %", min_value=0.0, step=0.1, value=st.session_state["tba_trade_med_max"], key="tba_trade_med_max")
            st.text_area("Zone → tier mapping (JSON or key=val)", value=st.session_state["tba_zone_map"], key="tba_zone_map")
        if st.session_state.get("tba_zone_map") and not zone_map:
            st.warning("Zone mapping invalid; ignoring zone overrides.")

        bucket_limits = {
            "low": st.session_state.get("tba_bucket_es_limit_low", 2.0),
            "med": st.session_state.get("tba_bucket_es_limit_med", 3.0),
            "high": st.session_state.get("tba_bucket_es_limit_high", 5.0),
        }
        
        # Get actual margin used from portfolio
        margin_used = st.session_state.get("margin_used")
        total_margin_used = float(margin_used) if margin_used is not None else None
        
        bucket_df = aggregate_buckets(trades_df, float(total_capital), allocations, bucket_limits, total_margin_used)
        bucket_df["kill_switch"] = bucket_df["bucket"].apply(
            lambda b: "RED"
            if bucket_df.loc[bucket_df["bucket"] == b, "bucket_es99_pct_of_bucket_capital"].values[0]
            > st.session_state.get(f"tba_bucket_es_limit_{b.lower()}", 0.0)
            else "OK"
        )
        
        # Table 1: Bucket Sizing (Capital Allocation)
        st.markdown("### Bucket Sizing - Capital Allocation")
        sizing_df = bucket_df[["bucket", "target_capital_pct", "actual_capital_pct", "capital_delta_pct", "trades", "legs"]].copy()
        sizing_df["target_capital_pct"] = sizing_df["target_capital_pct"].apply(lambda x: f"{int(round(x))}%")
        sizing_df["actual_capital_pct"] = sizing_df["actual_capital_pct"].apply(lambda x: f"{int(round(x))}%")
        sizing_df["capital_delta_pct"] = sizing_df["capital_delta_pct"].apply(lambda x: f"{int(round(x)):+}%")
        sizing_df.columns = ["Bucket", "Target %", "Actual %", "Delta %", "Trades", "Legs"]
        st.dataframe(sizing_df, use_container_width=True, hide_index=True)
        
        # Table 2: ES99 Risk per Bucket
        st.markdown("### ES99 Risk per Bucket")
        es99_df = bucket_df[["bucket", "target_es99_pct", "bucket_es99_pct_of_bucket_capital", "delta_pct", "kill_switch"]].copy()
        es99_df["target_es99_pct"] = es99_df["target_es99_pct"].apply(lambda x: f"{int(round(x))}%")
        es99_df["bucket_es99_pct_of_bucket_capital"] = es99_df["bucket_es99_pct_of_bucket_capital"].apply(lambda x: f"{int(round(x))}%")
        es99_df["delta_pct"] = es99_df["delta_pct"].apply(lambda x: f"{int(round(x)):+}%")
        es99_df.columns = ["Bucket", "Target ES99 %", "Actual ES99 %", "Delta %", "Kill Switch"]
        st.dataframe(es99_df, use_container_width=True, hide_index=True)

        usage_df = trades_df.copy()
        if usage_df["margin"].notna().any():
            usage_df["usage_inr"] = usage_df["margin"].fillna(0.0)
            usage_label = "Margin usage"
        else:
            usage_df["usage_inr"] = usage_df["trade_es99_inr"].abs()
            usage_label = "Risk-weighted usage (ES99)"
        usage_summary = usage_df.groupby("bucket")["usage_inr"].sum()
        st.caption(f"{usage_label}: {', '.join([f'{b} {format_inr(v)}' for b, v in usage_summary.items()])}")

        # Bucket charts in 2 columns
        bucket_chart_col1, bucket_chart_col2 = st.columns(2)
        with bucket_chart_col1:
            st.markdown("### Bucket ES99")
            data = bucket_df.set_index("bucket")["bucket_es99_inr"].reset_index()
            data.columns = ['name', 'value']
            chart = alt.Chart(data).mark_bar(size=50).encode(
                x=alt.X('name:N', title=None),
                y=alt.Y('value:Q', title=None)
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
        with bucket_chart_col2:
            st.markdown("### Bucket Trades")
            data = bucket_df.set_index("bucket")["trades"].reset_index()
            data.columns = ['name', 'value']
            chart = alt.Chart(data).mark_bar(size=50).encode(
                x=alt.X('name:N', title=None),
                y=alt.Y('value:Q', title=None)
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)

        st.markdown("### Forward Simulation & Scenario Table by Bucket")
        gate_config = _render_gate_settings("bucket_gates")
        sim_cfg = SimulationConfig(
            horizon_days=int(st.session_state.get("tba_sim_days")),
            paths=int(st.session_state.get("tba_sim_paths")),
            iv_mode=st.session_state.get("tba_iv_mode"),
            iv_shock=float(st.session_state.get("tba_iv_shock")),
            include_spot_shocks=True,  # Always include scenarios for bucket level
        )
        allocations_total = sum(allocations.values())
        allocations_ok = abs(allocations_total - 100.0) <= 0.01
        if not allocations_ok:
            st.warning("Bucket allocations do not sum to 100%; bucket limits may be incorrect.")
        portfolio_limit_inr = -float(total_capital) * float(portfolio_es_limit) / 100.0
        bucket_tabs = st.tabs([f"{bucket} Bucket" for bucket in BUCKET_ORDER])
        for tab, bucket in zip(bucket_tabs, BUCKET_ORDER):
            with tab:
                bucket_positions: List[Dict[str, object]] = []
                for _, row in trades_df[trades_df["bucket"] == bucket].iterrows():
                    bucket_positions.extend(row["legs_detail"])
                bucket_limit = st.session_state.get(f"tba_bucket_es_limit_{bucket.lower()}", None)
                bucket_cap = total_capital * allocations[BUCKET_KEYS[bucket]] / 100.0
                sim = simulate_forward_pnl(bucket_positions, sim_cfg, bucket_limit, bucket_cap)
                bucket_limit_inr = None
                if allocations_ok and bucket_limit is not None:
                    bucket_limit_inr = -float(bucket_cap) * float(bucket_limit) / 100.0
                elif not allocations_ok:
                    st.warning("Skipping bucket loss limit line due to invalid allocations.")
                _render_simulation(
                    sim,
                    f"{bucket} Bucket Forward Simulation",
                    portfolio_limit_inr,
                    bucket_limit_inr,
                    bucket,
                    f"tba_bucket_sim_{bucket.lower()}",
                    gate_config=gate_config,
                    gate_limits={
                        "bucket_limit_inr": bucket_limit_inr or 0.0,
                        "bucket_limit_pct": float(bucket_limit or 0.0),
                        "bucket_es99_pct": float(
                            bucket_df.loc[bucket_df["bucket"] == bucket, "bucket_es99_pct_of_bucket_capital"]
                            .values[0]
                            if not bucket_df.empty
                            else 0.0
                        ),
                    },
                )
                _render_scenario_table(
                    bucket_positions,
                    float(bucket_cap),
                    lookback_days=504,
                    es_limit_pct=float(bucket_limit or 0.0),
                    title="Scenario Table (with calibrated probabilities)",
                )

    with subtab3:
        st.markdown("### Trade Level")
        st.caption(
            "Expected PnL is the mean simulated PnL at the end of the current horizon "
            f"({int(st.session_state.get('tba_sim_days', 10))} days), includes theta carry, "
            "and adds pro-rated premium received."
        )
        with st.expander("Zone rules (per 1L capital)"):
            options_df_cache = st.session_state.get("options_df_cache", pd.DataFrame())
            iv_percentile = 35
            if (
                isinstance(options_df_cache, pd.DataFrame)
                and not options_df_cache.empty
                and "iv" in options_df_cache.columns
            ):
                iv_percentile = 35
            iv_regime, _ = get_iv_regime(iv_percentile)
            st.caption(f"IV regime: {iv_regime}")
            st.dataframe(_zone_rules_table(iv_regime), use_container_width=True, hide_index=True)
        with st.expander("Trade Grouping (Manual)", expanded=False):
            saved_groups = st.session_state.get("tba_saved_trades", [])
            grouped_leg_ids = {leg_id for group in saved_groups for leg_id in group.get("legs", [])}
            if st.session_state.get("tba_clear_trade_group"):
                for idx in range(len(positions)):
                    key = f"tba_leg_select_{idx}"
                    if str(idx) not in grouped_leg_ids:
                        st.session_state[key] = False
                st.session_state["tba_trade_group_name"] = ""
                st.session_state["tba_clear_trade_group"] = False
            st.caption("Select legs to group (grouped legs are locked).")
            selected_legs = []
            for idx, pos in enumerate(positions):
                pos_id = str(idx)
                label = _position_label(pos, idx)
                is_grouped = pos_id in grouped_leg_ids
                checkbox_key = f"tba_leg_select_{idx}"
                checked = st.checkbox(
                    f"{label}{' (grouped)' if is_grouped else ''}",
                    value=is_grouped or bool(st.session_state.get(checkbox_key, False)),
                    key=checkbox_key,
                    disabled=is_grouped,
                )
                if checked and not is_grouped:
                    selected_legs.append(pos_id)
            group_name = st.text_input("Trade name", key="tba_trade_group_name")
            if st.button("Save trade group", type="secondary"):
                if not selected_legs:
                    st.warning("Select at least one leg to save a trade group.")
                else:
                    saved_groups.append({"name": group_name or "Manual Trade", "legs": selected_legs})
                    st.session_state["tba_saved_trades"] = saved_groups
                    _save_saved_groups(saved_groups)
                    st.session_state["tba_clear_trade_group"] = True
                    st.success("Trade group saved.")
            apply_cols = st.columns([0.5, 0.5])
            if apply_cols[0].button("Apply groupings", type="primary"):
                st.session_state["tba_use_saved_groups"] = True
            if apply_cols[1].button("Use auto grouping"):
                st.session_state["tba_use_saved_groups"] = False
            if saved_groups:
                st.caption("Saved groups (only these trades will be shown):")
                for idx, group in enumerate(saved_groups):
                    label = group.get("name") or f"Group {idx + 1}"
                    cols = st.columns([0.8, 0.2])
                    cols[0].write(label)
                    leg_labels = []
                    for leg_id in group.get("legs", []):
                        try:
                            leg_idx = int(leg_id)
                        except Exception:
                            continue
                        if 0 <= leg_idx < len(positions):
                            leg_labels.append(_position_label(positions[leg_idx], leg_idx))
                    if leg_labels:
                        cols[0].caption(" | ".join(leg_labels))
                    if cols[1].button("Remove", key=f"tba_remove_group_{idx}"):
                        st.session_state["tba_saved_trades"] = [
                            g for g_i, g in enumerate(saved_groups) if g_i != idx
                        ]
                        _save_saved_groups(st.session_state["tba_saved_trades"])
                        if not st.session_state.get("tba_saved_trades"):
                            st.session_state["tba_use_saved_groups"] = False
        chart_controls = st.columns(3)
        with chart_controls[0]:
            norm_mode = st.selectbox(
                "Normalize",
                ["Absolute INR", "Per 1L capital", "Per bucket cap %"],
                key="tba_trade_chart_norm",
            )
        with chart_controls[1]:
            log_loss_axis = st.toggle("Log scale (loss axis)", value=False, key="tba_trade_chart_log")
        with chart_controls[2]:
            tail_cap_label = "Tail cap (INR)"
            if norm_mode == "Per 1L capital":
                tail_cap_label = "Tail cap (per 1L)"
            elif norm_mode == "Per bucket cap %":
                tail_cap_label = "Tail cap (%)"
            tail_cap_inr = st.number_input(
                tail_cap_label,
                min_value=0.0,
                step=1_000.0 if norm_mode == "Absolute INR" else 1.0,
                value=float(st.session_state.get("tba_trade_tail_cap_inr", 200_000.0)),
                key="tba_trade_tail_cap_inr",
            )
        toggle_cols = st.columns(2)
        with toggle_cols[0]:
            show_individual_trades = st.toggle(
                "Show individual trades",
                value=True,
                key="tba_show_individual_trades",
            )
        with toggle_cols[1]:
            show_group_trades = st.toggle(
                "Show group trades",
                value=True,
                key="tba_show_group_trades",
            )

        charts_container = st.container()
        chart_df = trades_df.copy()
        group_names = {g.get("name") for g in saved_groups if g.get("name")}
        if saved_groups and st.session_state.get("tba_use_saved_groups", False):
            if not show_individual_trades:
                chart_df = chart_df[chart_df["option_side"].isin(group_names)]
            if not show_group_trades and group_names:
                chart_df = chart_df[~chart_df["option_side"].isin(group_names)]
        chart_df["theta_per_lakh_label"] = chart_df.apply(
            lambda row: f"{row['theta_per_lakh']:.1f} ({row['theta_label']})"
            if row["theta_per_lakh"] is not None
            else "—",
            axis=1,
        )
        chart_df["gamma_per_lakh_label"] = chart_df.apply(
            lambda row: f"{row['gamma_per_lakh']:.4f} ({row['gamma_label']})"
            if row["gamma_per_lakh"] is not None
            else "—",
            axis=1,
        )
        chart_df["vega_per_lakh_label"] = chart_df.apply(
            lambda row: f"{row['vega_per_lakh']:.1f} ({row['vega_label']})"
            if row["vega_per_lakh"] is not None
            else "—",
            axis=1,
        )
        chart_df["label_expiry"] = chart_df["earliest_expiry"].fillna("—")
        chart_df.loc[chart_df["label_expiry"].isin(["—", "UNKNOWN"]), "label_expiry"] = chart_df["week_id"]
        chart_df["trade_label"] = (
            chart_df["underlying"].astype(str)
            + " | "
            + chart_df["label_expiry"].astype(str)
            + " | "
            + chart_df["option_side"].astype(str)
        )
        chart_df["capital_inr"] = chart_df["margin"].fillna(0.0)
        chart_df.loc[chart_df["capital_inr"] <= 0, "capital_inr"] = chart_df["premium_received_inr"].abs()
        chart_df["tail_loss_inr"] = chart_df["tail_loss_inr"].fillna(0.0)
        chart_df["risk_reward"] = chart_df.apply(
            lambda row: (row["expected_pnl_inr"] / row["tail_loss_inr"])
            if row["tail_loss_inr"]
            else 0.0,
            axis=1,
        )
        if norm_mode == "Per 1L capital":
            norm_factor = chart_df["capital_inr"].replace(0, np.nan) / 100000.0
            chart_df["x_value"] = chart_df["expected_pnl_inr"] / norm_factor
            chart_df["mean_loss_value"] = chart_df["mean_loss_inr"] / norm_factor
            chart_df["tail_loss_value"] = chart_df["tail_loss_inr"] / norm_factor
            x_title = "Mean PnL (per 1L)"
            y_mean_title = "Mean Loss (per 1L)"
            y_tail_title = "Tail Loss (per 1L)"
        elif norm_mode == "Per bucket cap %":
            bucket_cap_inr = chart_df.apply(
                lambda row: (row["trade_es99_inr"] / (row["trade_es99_pct_of_bucket_cap"] / 100.0))
                if row["trade_es99_pct_of_bucket_cap"]
                else np.nan,
                axis=1,
            )
            chart_df["x_value"] = chart_df.get("expected_pnl_pct")
            chart_df["mean_loss_value"] = (chart_df["mean_loss_inr"] / bucket_cap_inr) * 100.0
            chart_df["tail_loss_value"] = chart_df.get("trade_es99_pct_of_bucket_cap")
            x_title = "Mean PnL (% bucket cap)"
            y_mean_title = "Mean Loss (% bucket cap)"
            y_tail_title = "Tail Loss (% bucket cap)"
        else:
            chart_df["x_value"] = chart_df["expected_pnl_inr"]
            chart_df["mean_loss_value"] = chart_df["mean_loss_inr"]
            chart_df["tail_loss_value"] = chart_df["tail_loss_inr"]
            x_title = "Mean PnL (INR)"
            y_mean_title = "Mean Loss (INR)"
            y_tail_title = "Tail Loss (INR)"

        tooltip_cols = [
            alt.Tooltip("trade_id:N", title="Trade"),
            alt.Tooltip("week_id:N", title="Week"),
            alt.Tooltip("option_side:N", title="Side"),
            alt.Tooltip("bucket:N", title="Bucket"),
            alt.Tooltip("zone_label:N", title="Zone"),
            alt.Tooltip("legs:Q", title="Legs"),
            alt.Tooltip("expected_pnl_inr:Q", title="Mean PnL", format=",.0f"),
            alt.Tooltip("tail_loss_inr:Q", title="Tail Loss", format=",.0f"),
            alt.Tooltip("mean_loss_inr:Q", title="Mean Loss", format=",.0f"),
            alt.Tooltip("premium_received_inr:Q", title="Premium", format=",.0f"),
            alt.Tooltip("capital_inr:Q", title="Capital", format=",.0f"),
            alt.Tooltip("risk_reward:Q", title="Reward/Risk", format=".3f"),
            alt.Tooltip("mean_tail_ratio:Q", title="Mean/Tail", format=".3f"),
            alt.Tooltip("theta_per_lakh:Q", title="Theta/1L", format=",.1f"),
            alt.Tooltip("gamma_per_lakh:Q", title="Gamma/1L", format=",.4f"),
            alt.Tooltip("vega_per_lakh:Q", title="Vega/1L", format=",.1f"),
        ]

        def _build_overlays(x_min: float, x_max: float, y_max: float, loss_axis_title: str) -> List[alt.Chart]:
            overlays: List[alt.Chart] = []
            if x_min < 0:
                overlays.append(
                    alt.Chart(pd.DataFrame({"x1": [x_min], "x2": [0], "y1": [0], "y2": [y_max]}))
                    .mark_rect(opacity=0.08, color="#FF6B6B")
                    .encode(x="x1:Q", x2="x2:Q", y="y1:Q", y2="y2:Q")
                )
            if tail_cap_inr > 0 and tail_cap_inr < y_max:
                overlays.append(
                    alt.Chart(pd.DataFrame({"x1": [x_min], "x2": [x_max], "y1": [tail_cap_inr], "y2": [y_max]}))
                    .mark_rect(opacity=0.08, color="#FF6B6B")
                    .encode(x="x1:Q", x2="x2:Q", y="y1:Q", y2="y2:Q")
                )
                overlays.append(
                    alt.Chart(pd.DataFrame({"y": [tail_cap_inr]}))
                    .mark_rule(stroke="#FF6B6B", strokeWidth=1)
                    .encode(y="y:Q")
                )
            overlays.append(
                alt.Chart(pd.DataFrame({"x": [0]}))
                .mark_rule(stroke="#8A8A8A", strokeWidth=1, strokeDash=[4, 4])
                .encode(x="x:Q")
            )
            if not log_loss_axis:
                overlays.append(
                    alt.Chart(pd.DataFrame({"y": [0]}))
                    .mark_rule(stroke="#8A8A8A", strokeWidth=1, strokeDash=[4, 4])
                    .encode(y="y:Q")
                )
            return overlays

        def _build_rr_overlays(x_min: float, x_max: float, y_max: float) -> List[alt.Chart]:
            rr_levels = [0.15, 0.30, 0.40]
            overlays: List[alt.Chart] = []
            if y_max <= 0:
                return overlays
            x_left = max(0.0, x_min)
            x_right = 10000
            if x_right <= x_left:
                return overlays
            x_vals = np.linspace(x_left, x_right, 80)
            line_rows = []
            for rr in rr_levels:
                for x_val in x_vals:
                    y_val = x_val / rr if rr else 0.0
                    if 0 <= y_val <= y_max:
                        line_rows.append({"x": x_val, "y": y_val, "rr": f"{rr:.2f}"})
            if line_rows:
                overlays.append(
                    alt.Chart(pd.DataFrame(line_rows))
                    .mark_line(strokeDash=[4, 4])
                    .encode(
                        x="x:Q",
                        y="y:Q",
                        detail="rr:N",
                        color=alt.Color(
                            "rr:N",
                            scale=alt.Scale(range=["#B38E5D", "#5BAE6B", "#6F8EDC"]),
                            legend=None,
                        ),
                    )
                )
                label_rows = []
                for rr, color in zip(rr_levels, ["#B38E5D", "#5BAE6B", "#6F8EDC"]):
                    y_val = min(y_max, x_right / rr) if rr else y_max
                    label_rows.append({"x": x_right, "y": y_val, "label": f"RR {rr:.2f}", "color": color})
                overlays.append(
                    alt.Chart(pd.DataFrame(label_rows))
                    .mark_text(align="right", dx=-6, dy=-6, fontSize=10, fontWeight="bold")
                    .encode(
                        x="x:Q",
                        y="y:Q",
                        text="label:N",
                        color=alt.Color("color:N", scale=None, legend=None),
                    )
                )

            zone_defs = [
                (
                    0.0,
                    0.15,
                    "Zone A: Very cheap premium. Mean PnL small relative to tail risk. "
                    "Small spot or vol moves quickly flip EV negative; slow bleed with rare large losses. "
                    "Requires very tight risk caps.",
                    "#3A1F24",
                ),
                (
                    0.15,
                    0.30,
                    "Zone B: Preferred short-vol regime. Balanced credit vs tail risk. "
                    "Trades remain bottom-right under moderate spot moves; manageable gamma. "
                    "Best persistence of positive EV.",
                    "#243622",
                ),
                (
                    0.30,
                    0.40,
                    "Zone C: High credit but fragile. Positive EV only near entry. "
                    "One-sided spot moves or IV expansion rapidly increase tail loss; "
                    "needs fast exits and active management.",
                    "#3B2F1A",
                ),
                (
                    0.40,
                    None,
                    "Zone D: Gamma-dominant. Near-spot exposure. EV highly unstable; "
                    "small moves cause sharp jumps in tail loss. Treat as directional, not probability trade.",
                    "#2C2F3D",
                ),
            ]
            shade_rows = []
            label_rows = []
            for low_rr, high_rr, label, color in zone_defs:
                y_top = y_max if high_rr is None else max(0.0, x_right / high_rr)
                y_bottom = max(0.0, x_right / low_rr) if low_rr > 0 else 0.0
                y_top = min(y_top, y_max)
                y_bottom = min(y_bottom, y_max)
                if y_bottom >= y_top:
                    continue
                shade_rows.append(
                    {
                        "x1": x_left,
                        "x2": x_right,
                        "y1": y_bottom,
                        "y2": y_top,
                        "label": label,
                        "color": color,
                    }
                )
                label_rows.append(
                    {
                        "x": x_left + (x_right - x_left) * 0.05,
                        "y": (y_bottom + y_top) / 2.0,
                        "label": label,
                    }
                )
            if shade_rows:
                overlays.append(
                    alt.Chart(pd.DataFrame(shade_rows))
                    .mark_rect(opacity=0.10)
                    .encode(
                        x="x1:Q",
                        x2="x2:Q",
                        y="y1:Q",
                        y2="y2:Q",
                        color=alt.Color("color:N", scale=None, legend=None),
                    )
                )
            return overlays

        for chart_key, y_field, y_title in [
            ("tail_loss", "tail_loss_value", y_tail_title),
        ]:
            plot_df = chart_df.copy()
            if log_loss_axis:
                plot_df = plot_df[plot_df[y_field] > 0]
            x_min = float(plot_df["x_value"].min()) if not plot_df.empty else 0.0
            x_max = float(plot_df["x_value"].max()) if not plot_df.empty else 1.0
            y_max = float(plot_df[y_field].max()) if not plot_df.empty else 1.0
            positive_df = plot_df[plot_df["expected_pnl_inr"] >= 0]
            negative_df = plot_df[plot_df["expected_pnl_inr"] < 0]
            base_positive = (
                alt.Chart(positive_df)
                .mark_circle(opacity=0.85, strokeWidth=1.2)
                .encode(
                    x=alt.X("x_value:Q", title=x_title, scale=alt.Scale(domain=[-10000, 10000])),
                    y=alt.Y(
                        f"{y_field}:Q",
                        title=y_title,
                        scale=alt.Scale(type="log") if log_loss_axis else alt.Scale(zero=True),
                    ),
                    color=alt.Color(
                        "risk_reward:Q",
                        legend=alt.Legend(title="Reward/Risk"),
                        scale=alt.Scale(scheme="redyellowgreen", clamp=True),
                    ),
                    shape=alt.Shape("option_side:N", legend=alt.Legend(title="Side")),
                    tooltip=tooltip_cols,
                )
                .properties(height=300, width="container")
            )
            base_negative = (
                alt.Chart(negative_df)
                .mark_circle(opacity=0.85, strokeWidth=1.2, color="#d9534f")
                .encode(
                    x=alt.X("x_value:Q", title=x_title, scale=alt.Scale(domain=[-10000, 10000])),
                    y=alt.Y(
                        f"{y_field}:Q",
                        title=y_title,
                        scale=alt.Scale(type="log") if log_loss_axis else alt.Scale(zero=True),
                    ),
                    shape=alt.Shape("option_side:N", legend=alt.Legend(title="Side")),
                    tooltip=tooltip_cols,
                )
                .properties(height=300, width="container")
            )
            labels = (
                alt.Chart(plot_df)
                .mark_text(align="left", dx=6, dy=-6, color="#BFC7D5", fontSize=10)
                .encode(
                    x=alt.X("x_value:Q"),
                    y=alt.Y(f"{y_field}:Q"),
                    text=alt.Text("trade_label:N"),
                )
            )
            overlays = _build_overlays(x_min, x_max, y_max, y_title)
            rr_overlays = _build_rr_overlays(x_min, x_max, y_max) if chart_key == "tail_loss" else []
            layers = overlays + rr_overlays + [base_positive, base_negative, labels]
            chart = alt.layer(*layers)
            with charts_container:
                title_cols = st.columns([0.9, 0.1])
                with title_cols[0]:
                    st.markdown("#### Tail Loss vs Mean PnL")
                with title_cols[1]:
                    with st.popover("ℹ️ Zones", use_container_width=False):
                        st.markdown("""
                        **Zone A (RR < 0.15)**: Very cheap premium. Mean PnL small relative to tail risk. 
                        Small spot or vol moves quickly flip EV negative; slow bleed with rare large losses. 
                        Requires very tight risk caps.
                        
                        **Zone B (0.15–0.30)**: Preferred short-vol regime. Balanced credit vs tail risk. 
                        Trades remain bottom-right under moderate spot moves; manageable gamma. 
                        Best persistence of positive EV.
                        
                        **Zone C (0.30–0.40)**: High credit but fragile. Positive EV only near entry. 
                        One-sided spot moves or IV expansion rapidly increase tail loss; 
                        needs fast exits and active management.
                        
                        **Zone D (RR > 0.40)**: Gamma-dominant. Near-spot exposure. EV highly unstable; 
                        small moves cause sharp jumps in tail loss. Treat as directional, not probability trade.
                        """)
                chart = chart.properties(height=420)
                st.altair_chart(chart, use_container_width=True)
        filter_cols = st.columns(5)
        underlyings = ["All"] + sorted(trades_df["underlying"].dropna().unique().tolist())
        weeks = ["All"] + sorted(trades_df["week_id"].dropna().unique().tolist())
        sides = ["All"] + sorted(trades_df["option_side"].dropna().unique().tolist())
        buckets = ["All"] + sorted(trades_df["bucket"].dropna().unique().tolist())
        zones = ["All"] + sorted(trades_df["zone_label"].dropna().unique().tolist())
        filter_underlying = filter_cols[0].selectbox("Underlying", underlyings, key="tba_trade_filter_underlying")
        filter_week = filter_cols[1].selectbox("Week", weeks, key="tba_trade_filter_week")
        filter_side = filter_cols[2].selectbox("Side", sides, key="tba_trade_filter_side")
        filter_bucket = filter_cols[3].selectbox("Bucket", buckets, key="tba_trade_filter_bucket")
        filter_zone = filter_cols[4].selectbox("Zone", zones, key="tba_trade_filter_zone")

        df = trades_df.copy()
        if filter_underlying != "All":
            df = df[df["underlying"] == filter_underlying]
        if filter_week != "All":
            df = df[df["week_id"] == filter_week]
        if filter_side != "All":
            df = df[df["option_side"] == filter_side]
        if filter_bucket != "All":
            df = df[df["bucket"] == filter_bucket]
        if filter_zone != "All":
            df = df[df["zone_label"] == filter_zone]

        sort_choice = st.selectbox(
            "Sort by",
            [
                "trade_es99_inr",
                "trade_es99_pct_of_bucket_cap",
                "expected_pnl_inr",
                "risk_reward",
                "theta_carry_inr",
                "mean_loss_inr",
                "mean_tail_ratio",
                "premium_received_inr",
                "premium_prorated_inr",
                "theta_per_lakh",
                "gamma_per_lakh",
                "vega_per_lakh",
                "margin",
                "week_id",
            ],
            key="tba_trade_sort",
        )
        df = df.sort_values(sort_choice, ascending=False)
        df["theta_per_lakh_label"] = df.apply(
            lambda row: f"{row['theta_per_lakh']:.1f} ({row['theta_label']})"
            if row.get("theta_per_lakh") is not None
            else "—",
            axis=1,
        )
        df["gamma_per_lakh_label"] = df.apply(
            lambda row: f"{row['gamma_per_lakh']:.4f} ({row['gamma_label']})"
            if row.get("gamma_per_lakh") is not None
            else "—",
            axis=1,
        )
        df["vega_per_lakh_label"] = df.apply(
            lambda row: f"{row['vega_per_lakh']:.1f} ({row['vega_label']})"
            if row.get("vega_per_lakh") is not None
            else "—",
            axis=1,
        )
        st.dataframe(
            df[
                [
                    "underlying",
                    "week_id",
                    "option_side",
                    "earliest_expiry",
                    "latest_expiry",
                    "legs",
                    "bucket",
                    "trade_es99_inr",
                    "trade_es99_pct_of_bucket_cap",
                    "expected_pnl_inr",
                    "expected_pnl_pct",
                    "risk_reward",
                    "theta_carry_inr",
                    "mean_loss_inr",
                    "mean_tail_ratio",
                    "premium_received_inr",
                    "premium_prorated_inr",
                    "theta_per_lakh_label",
                    "gamma_per_lakh_label",
                    "vega_per_lakh_label",
                    "margin",
                    "zone_label",
                    "mtd_pnl",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download trades CSV",
            df.to_csv(index=False),
            file_name="tba_trades.csv",
        )

        for _, row in df.iterrows():
            with st.expander(
                f"{row['underlying']} | {row['week_id']} | {row['option_side']} | {row['bucket']}"
            ):
                legs_df = pd.DataFrame(
                    [
                        {
                            "symbol": leg.get("tradingsymbol"),
                            "strike": leg.get("strike"),
                            "type": leg.get("option_type") or leg.get("instrument_type"),
                            "qty": leg.get("quantity"),
                            "avg_price": leg.get("average_price"),
                            "ltp": leg.get("last_price"),
                            "pnl": leg.get("pnl", leg.get("m2m")),
                            "delta": leg.get("delta"),
                            "gamma": leg.get("gamma"),
                            "vega": leg.get("vega"),
                            "theta": leg.get("theta"),
                            "pos_delta": leg.get("position_delta"),
                            "pos_gamma": leg.get("position_gamma"),
                            "pos_vega": leg.get("position_vega"),
                            "pos_theta": leg.get("position_theta"),
                        }
                        for leg in row["legs_detail"]
                    ]
                )
                st.dataframe(legs_df, use_container_width=True, hide_index=True)

        st.info("Forward simulation is available at Portfolio and Bucket levels only.")

    with subtab4:
        st.markdown("### How to Read This Tab (Process)")
        st.markdown(
            "- Start at **Portfolio Level**: check total capital, overall ES99, and forward-sim KPIs.\n"
            "- Then go to **Bucket Level**: verify capital allocations and ES99 vs bucket limits.\n"
            "- Finally, review **Trade Level** to see which weekly trades drive risk and their zone labels."
        )

        st.markdown("### How to Think About a Trade (Core)")
        st.markdown(
            "- Ask one brutal question: **If I had no position right now, would I enter this trade at this price?**\n"
            "- If yes: keep it (even add).\n"
            "- If no: exit immediately.\n"
            "- Your PnL history is irrelevant."
        )
        st.markdown("### A Cleaner Decision Framework")
        st.markdown(
            "- **Step 1: Freeze the past**\n"
            "  - Ignore entry price, max profit, and unrealized loss.\n"
            "  - They are gone.\n"
            "- **Step 2: Compare distributions**\n"
            "  - Trade A (current): future EV, tail loss, drawdown.\n"
            "  - Trade B (new): future EV, tail loss, drawdown.\n"
            "  - Pick the higher EV per unit tail risk. Always.\n"
            "- **Step 3: Direction != structure**\n"
            "  - If you still believe bullish, express it with fresh risk.\n"
            "  - Do not cling to a broken structure."
        )
        st.markdown("### Hard Truth (But Freeing)")
        st.markdown(
            "- Booking a loss is not being wrong.\n"
            "- Holding a bad trade is being undisciplined.\n"
            "- The market already took the money. You are just deciding whether to risk more."
        )

        st.markdown("### ES99 (Stress-Scenario Based)")
        st.markdown(
            "- **Step 1 (Greeks)**: compute portfolio delta/gamma/vega on current positions.\n"
            "- **Step 2 (Scenario shocks)**: apply predefined spot/IV shocks by IV regime (from the same stress engine).\n"
            "- **Step 3 (Scenario P&L)**: estimate P&L per scenario using the greek-based formula.\n"
            "- **Step 4 (Probabilities)**: weight each scenario by historical bucket frequencies (lookback window).\n"
            "- **Step 5 (ES99)**: build the loss distribution and take the **expected loss of the worst 1% tail**.\n"
            "- **Output**: ES99 in INR and as % of capital; **not** from forward simulation paths."
        )

        st.markdown("### Forward Simulation KPIs (Portfolio & Buckets)")
        st.markdown(
            "- **Mean / Median**: expected and typical P&L at horizon.\n"
            "- **P5 / P1**: tail P&L quantiles (bad-but-plausible and extreme tail).\n"
            "- **Prob. Loss**: chance of negative P&L over the horizon.\n"
            "- **Prob. Breach**: chance of crossing the configured loss limit.\n"
            "- **Days to breach (P10/P1)**: how fast a bad or extreme tail reaches the limit.\n"
            "- **Tail ratio**: |P1| / |P50| at horizon (tail severity)."
        )

        st.markdown("### Historical Bucket Frequencies (Scenario Weights)")
        st.markdown(
            "- Scenarios are classified into historical buckets (A–E) based on spot/IV moves.\n"
            "- The frequency of each bucket over the lookback window becomes the probability weight.\n"
            "- These weights are used to compute ES99 from the stress-scenario loss distribution.\n"
            "- Bucket probabilities change with the lookback window and recent market regime."
        )
        st.markdown("#### Historical Buckets (used for ES99 weighting)")
        bucket_meta = pd.DataFrame(
            [
                {
                    "Bucket": "A",
                    "Label": "Calm",
                    "Definition": "|Return| ≤ 0.5% and |IV proxy| ≤ 1",
                    "Notes": "Low-vol day; small drift and muted intraday range.",
                },
                {
                    "Bucket": "B",
                    "Label": "Normal",
                    "Definition": "|Return| ≤ 1.0% and |IV proxy| ≤ 2",
                    "Notes": "Routine session with modest moves.",
                },
                {
                    "Bucket": "C",
                    "Label": "Elevated",
                    "Definition": "|Return| ≤ 1.5% and |IV proxy| ≤ 3",
                    "Notes": "Elevated range with IV pickup.",
                },
                {
                    "Bucket": "D",
                    "Label": "Stress",
                    "Definition": "|Return| ≤ 2.5% and |IV proxy| ≤ 5",
                    "Notes": "Large range day; risk controls should trigger.",
                },
                {
                    "Bucket": "E",
                    "Label": "Gap / Tail",
                    "Definition": "|Return| > 2.5% or |IV proxy| > 5",
                    "Notes": "Tail event bucket; gap or volatility shock.",
                },
            ]
        )
        st.dataframe(bucket_meta, use_container_width=True, hide_index=True)

        st.markdown("### Capital Allocations & Limits")
        st.markdown(
            f"- **Allocations**: Low {st.session_state.get('tba_alloc_low', 50.0):.0f}% / "
            f"Med {st.session_state.get('tba_alloc_med', 30.0):.0f}% / "
            f"High {st.session_state.get('tba_alloc_high', 20.0):.0f}%.\n"
            f"- **Portfolio ES99 limit**: {st.session_state.get('tba_portfolio_es_limit', 4.0):.1f}% of total capital.\n"
            f"- **Bucket ES99 limits**: Low {st.session_state.get('tba_bucket_es_limit_low', 2.0):.1f}% / "
            f"Med {st.session_state.get('tba_bucket_es_limit_med', 3.0):.1f}% / "
            f"High {st.session_state.get('tba_bucket_es_limit_high', 5.0):.1f}% of bucket capital."
        )

        st.markdown("### Decision Gates (Pass / Watch / Fail)")
        st.markdown(
            "- **G1 Survivability**: P1 breach day ≤ "
            f"{int(st.session_state.get('gate_p1_breach_fail_days', 2))} is FAIL; "
            "Prob breach > ES99 limit is FAIL; P1 horizon loss deeper than 1.25× ES99 limit is FAIL.\n"
            "- **G2 Speed of damage**: P10 breach day is compared to bucket rules "
            f"(Low: any breach FAIL; Med: ≤{int(st.session_state.get('gate_bucket_p10_fail_days_med', 6))} FAIL; "
            f"High: ≤{int(st.session_state.get('gate_bucket_p10_fail_days_high', 3))} FAIL). "
            f"Portfolio FAIL if ≤{int(st.session_state.get('gate_portfolio_p10_fail_days', 4))}.\n"
            "- **G3 Asymmetry / convexity**: tail ratio WATCH if >"
            f"{int(st.session_state.get('gate_tail_ratio_watch', 30))}, FAIL if >"
            f"{int(st.session_state.get('gate_tail_ratio_fail', 60))}; Mean < 0 and Median > 0 is WATCH.\n"
            "- **G4 Carry quality**: Prob loss WATCH if >"
            f"{int(st.session_state.get('gate_prob_loss_watch', 40))}% and FAIL if >"
            f"{int(st.session_state.get('gate_prob_loss_fail', 50))}%; "
            "Median ≤ 0 and Mean < 0 is FAIL.\n"
            "- **G5 Sizing**: ES99 vs configured limits (Portfolio limit % and Bucket limit %).\n"
            "- Gate thresholds are bucket-aware where noted, and editable in **Gate settings** under each forward simulation."
        )
