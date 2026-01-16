"""Scenario stress testing utilities for the risk analysis tab."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union


@dataclass(frozen=True)
class Scenario:
    """Describes a stress scenario."""

    name: str
    ds_pct: float = 0.0
    div_pts: float = 0.0
    category: str = "price"  # price | iv | combined | gap


DEFAULT_SCENARIOS: List[Scenario] = [
    # Price-only
    Scenario("NIFTY +1.0%", ds_pct=1.0, div_pts=0.0, category="price"),
    Scenario("NIFTY -1.0%", ds_pct=-1.0, div_pts=0.0, category="price"),
    Scenario("NIFTY +1.5%", ds_pct=1.5, div_pts=0.0, category="price"),
    Scenario("NIFTY -1.5%", ds_pct=-1.5, div_pts=0.0, category="price"),
    # IV-only
    Scenario("IV +2 pts", ds_pct=0.0, div_pts=2.0, category="iv"),
    Scenario("IV +5 pts", ds_pct=0.0, div_pts=5.0, category="iv"),
    Scenario("IV -2 pts", ds_pct=0.0, div_pts=-2.0, category="iv"),
    # Combined
    Scenario("NIFTY +1% & IV +2", ds_pct=1.0, div_pts=2.0, category="combined"),
    Scenario("NIFTY -1% & IV +2", ds_pct=-1.0, div_pts=2.0, category="combined"),
    Scenario("NIFTY +1.5% & IV +3", ds_pct=1.5, div_pts=3.0, category="combined"),
    Scenario("NIFTY -1.5% & IV +3", ds_pct=-1.5, div_pts=3.0, category="combined"),
    # Gap risk
    Scenario("Gap Down -2% & IV +5", ds_pct=-2.0, div_pts=5.0, category="gap"),
    Scenario("Gap Up +2% & IV +4", ds_pct=2.0, div_pts=4.0, category="gap"),
    Scenario("Gap Down -5% & IV +8", ds_pct=-5.0, div_pts=8.0, category="gap"),
    Scenario("Gap Up +5% & IV +7", ds_pct=5.0, div_pts=7.0, category="gap"),
    Scenario("Gap Down -10% & IV +12", ds_pct=-10.0, div_pts=12.0, category="gap"),
    Scenario("Gap Up +10% & IV +10", ds_pct=10.0, div_pts=10.0, category="gap"),
]


STRESS_LIMIT_DEFAULTS: Dict[str, float] = {
    "price": 0.50,
    "iv": 0.40,
    "combined": 1.00,
    "gap": 1.20,
}

REGIME_NOTES: Dict[str, str] = {
    "Low IV": "emphasizing IV expansion and combined shocks.",
    "Mid IV": "balanced stress set.",
    "High IV": "highlighting price shocks and IV crush scenarios.",
}


def get_regime_note(regime: str) -> str:
    """Return descriptive note for current regime."""
    return REGIME_NOTES.get(regime, "balanced stress set.")


def get_weighted_scenarios(regime: str) -> List[Scenario]:
    """Return the default scenarios ordered by emphasis for the IV regime."""

    def weight(s: Scenario) -> int:
        if regime == "Low IV":
            if s.category in {"iv", "combined"} and s.div_pts > 0:
                return 0
            if s.category == "gap":
                return 1
            return 2
        if regime == "High IV":
            if s.category == "price":
                return 0
            if s.category == "combined" and s.div_pts <= 0:
                return 1
            return 2
        # Mid IV or unknown
        return 1

    return sorted(DEFAULT_SCENARIOS, key=weight)


def classify_scenario_limit(
    scenario: Scenario, limits: Optional[Dict[str, float]] = None
) -> float:
    """Return the max loss % limit for the given scenario."""
    limits = limits or STRESS_LIMIT_DEFAULTS
    return limits.get(scenario.category, limits["combined"])


def compute_scenario_pnl(
    portfolio: Dict[str, float], scenario: Scenario
) -> Dict[str, float]:
    """Compute scenario P&L contribution breakdown."""
    spot = portfolio.get("spot", 0.0)
    delta = portfolio.get("delta", 0.0)
    gamma = portfolio.get("gamma", 0.0)
    vega = portfolio.get("vega", 0.0)
    account_size = portfolio.get("account_size", 0.0)
    margin_deployed = portfolio.get("margin_deployed", 0.0)

    d_s = spot * (scenario.ds_pct / 100.0)
    d_iv = scenario.div_pts

    pnl_delta = delta * d_s
    pnl_gamma = 0.5 * gamma * (d_s ** 2)
    pnl_vega = vega * d_iv
    total = pnl_delta + pnl_gamma + pnl_vega

    pct_account = (total / account_size * 100.0) if account_size else 0.0
    pct_margin = (
        (total / margin_deployed * 100.0) if margin_deployed else 0.0
    )
    loss_pct_account = (-pct_account) if total < 0 else 0.0
    loss_pct_margin = (-pct_margin) if total < 0 else 0.0

    return {
        "scenario": scenario,
        "delta_pnl": pnl_delta,
        "gamma_pnl": pnl_gamma,
        "vega_pnl": pnl_vega,
        "total_pnl": total,
        "pct_account": pct_account,
        "pct_margin": pct_margin,
        "loss_pct_account": loss_pct_account,
        "loss_pct_margin": loss_pct_margin,
    }


def build_stress_report(
    portfolio: Dict[str, float],
    scenarios: List[Scenario],
    limits: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Build aggregated stress report."""
    limits = limits or STRESS_LIMIT_DEFAULTS
    results = []

    for scenario in scenarios:
        breakdown = compute_scenario_pnl(portfolio, scenario)
        limit_pct = classify_scenario_limit(scenario, limits)
        status = "PASS"
        if breakdown["loss_pct_account"] > limit_pct:
            status = "FAIL"
        results.append({**breakdown, "limit_pct": limit_pct, "status": status})

    # Sort worst to best (most negative first)
    sorted_results = sorted(results, key=lambda r: r["total_pnl"])
    worst = sorted_results[0] if sorted_results else None
    fail_count = sum(1 for r in results if r["status"] == "FAIL")
    worst_loss_pct = (
        worst["loss_pct_account"] if worst else 0.0
    )
    worst_loss_value = worst["total_pnl"] if worst else 0.0

    badge = "SAFE"
    if fail_count > 0 or worst_loss_pct > limits.get("gap", 1.2):
        badge = "REDUCE"
    elif worst_loss_pct >= 0.8:
        badge = "WATCH"

    zone_compliant = fail_count == 0

    return {
        "results": sorted_results,
        "worst": worst,
        "fail_count": fail_count,
        "worst_loss_pct": worst_loss_pct,
        "worst_loss_value": worst_loss_value,
        "badge": badge,
        "zone_compliant": zone_compliant,
    }


def generate_stress_suggestions(results: List[Dict[str, object]]) -> List[str]:
    """Provide deterministic suggestions based on failing scenarios."""
    if not results:
        return ["No stress scenarios evaluated."]

    iv_fail = sum(
        1
        for r in results
        if r["status"] == "FAIL" and r["scenario"].div_pts > 0
    )
    price_down_fail = sum(
        1
        for r in results
        if r["status"] == "FAIL" and r["scenario"].ds_pct < 0
    )
    gamma_dominant = sum(
        1
        for r in results
        if r["status"] == "FAIL"
        and abs(r["gamma_pnl"]) >= 0.4 * abs(r["total_pnl"])
        and r["total_pnl"] < 0
    )

    suggestions: List[str] = []

    if iv_fail:
        suggestions.append(
            "Reduce net vega exposure or add long vega hedges (IV shock failures)."
        )
    if price_down_fail:
        suggestions.append(
            "Tighten downside delta/gamma or add protective puts (price-down stress)."
        )
    if gamma_dominant:
        suggestions.append(
            "Gamma drives the loss — cut near-expiry size or widen hedging wings."
        )
    if not suggestions:
        suggestions.append(
            "Stress set within limits — continue monitoring as market regime evolves."
        )

    return suggestions


def normalize_ds_pct(ds_pct: float) -> float:
    """Ensure dS percentage is expressed as decimal move."""
    # Treat values like 1.0 as 1% move, otherwise assume already decimal.
    if abs(ds_pct) > 0.25:
        return ds_pct / 100.0
    return ds_pct


def classify_bucket(scenario: Dict[str, Union[str, float]]) -> str:
    """Classify scenario into buckets A/B/C/D/E."""
    scenario_type = str(scenario.get("type", "PRICE")).upper()
    ds_pct = normalize_ds_pct(float(scenario.get("dS_pct", 0.0)))
    abs_ds = abs(ds_pct)
    d_iv = float(scenario.get("dIV_pts", 0.0))

    if scenario_type == "GAP":
        return "E"
    if abs_ds > 0.05 or d_iv > 8.0:
        return "E"
    if abs_ds > 0.025 or d_iv > 5.0:
        return "D"

    if scenario_type == "IV":
        if d_iv <= 0:
            return "A"  # informational but treat as A for display
        if d_iv <= 1.0:
            return "A"
        if d_iv <= 2.0:
            return "B"
        if d_iv <= 3.0:
            return "C"
        return "D"

    if scenario_type == "PRICE":
        if abs_ds <= 0.005:
            return "A"
        if abs_ds <= 0.010:
            return "B"
        if abs_ds <= 0.015:
            return "C"
        return "D"

    if scenario_type == "COMBINED":
        if abs_ds <= 0.005 and d_iv <= 1.0:
            return "A"
        if abs_ds <= 0.010 and d_iv <= 2.0:
            return "B"
        if abs_ds <= 0.015 and d_iv <= 3.0:
            return "C"
        return "D"

    return "C"


def derive_thresholds(
    master_pct: float,
    hard_stop_pct: float,
    normal_share: float,
    stress_share: float,
) -> Dict[str, float]:
    """Compute scenario thresholds from master rules."""
    limit_a = master_pct * normal_share
    limit_b = master_pct * stress_share
    limit_c = hard_stop_pct
    limit_d = hard_stop_pct
    limit_e = hard_stop_pct
    return {
        "limitA": limit_a,
        "limitB": limit_b,
        "limitC": limit_c,
        "limitD": limit_d,
        "limitE": limit_e,
    }


def compute_threshold_scenario_pnl(
    portfolio: Dict[str, float],
    scenario: Dict[str, Union[str, float]],
) -> Dict[str, float]:
    """Compute P&L contributions for threshold builder scenarios."""
    spot = portfolio.get("spot", 0.0)
    delta = portfolio.get("delta", 0.0)
    gamma = portfolio.get("gamma", 0.0)
    vega = portfolio.get("vega", 0.0)
    nav = portfolio.get("nav", 0.0)
    margin = portfolio.get("margin", 0.0)

    ds_pct = normalize_ds_pct(float(scenario.get("dS_pct", 0.0)))
    d_iv = float(scenario.get("dIV_pts", 0.0))
    d_s = spot * ds_pct

    pnl_delta = delta * d_s
    pnl_gamma = 0.5 * gamma * (d_s ** 2)
    pnl_vega = vega * d_iv
    total = pnl_delta + pnl_gamma + pnl_vega

    loss_pct_nav = (-total / nav * 100.0) if total < 0 and nav else 0.0
    loss_pct_margin = (-total / margin * 100.0) if total < 0 and margin else 0.0

    return {
        "pnl_delta": pnl_delta,
        "pnl_gamma": pnl_gamma,
        "pnl_vega": pnl_vega,
        "pnl_total": total,
        "loss_pct_nav": loss_pct_nav,
        "loss_pct_margin": loss_pct_margin,
    }


def build_threshold_report(
    portfolio: Dict[str, float],
    scenarios: List[Dict[str, Union[str, float]]],
    master_pct: float,
    hard_stop_pct: float,
    normal_share: float,
    stress_share: float,
) -> Dict[str, object]:
    """Evaluate all scenarios using derived thresholds."""
    thresholds = derive_thresholds(master_pct, hard_stop_pct, normal_share, stress_share)
    rows: List[Dict[str, object]] = []

    for scenario in scenarios:
        bucket = classify_bucket(scenario)
        pnl = compute_threshold_scenario_pnl(portfolio, scenario)
        if bucket == "A":
            threshold_pct = thresholds["limitA"]
        elif bucket == "B":
            threshold_pct = thresholds["limitB"]
        elif bucket == "C":
            threshold_pct = thresholds["limitC"]
        elif bucket == "D":
            threshold_pct = thresholds["limitD"]
        else:
            threshold_pct = thresholds["limitE"]

        status = "PASS"
        if scenario.get("type", "").upper() == "IV" and float(scenario.get("dIV_pts", 0.0)) < 0:
            status = "INFO"
        elif pnl["loss_pct_nav"] > threshold_pct:
            status = "FAIL"

        rows.append(
            {
                "scenario": scenario,
                "bucket": bucket,
                "threshold_pct": threshold_pct,
                **pnl,
                "status": status,
            }
        )

    sorted_rows = sorted(rows, key=lambda r: r["pnl_total"])
    worst = sorted_rows[0] if sorted_rows else None
    fail_count = sum(1 for r in sorted_rows if r["status"] == "FAIL")
    ab_fail_count = sum(
        1 for r in sorted_rows if r["status"] == "FAIL" and r["bucket"] in {"A", "B"}
    )
    worst_loss_pct = worst["loss_pct_nav"] if worst else 0.0
    within_master = (
        worst_loss_pct <= hard_stop_pct and ab_fail_count == 0
    )

    return {
        "rows": sorted_rows,
        "thresholds": thresholds,
        "worst": worst,
        "fail_count": fail_count,
        "ab_fail_count": ab_fail_count,
        "worst_loss_pct": worst_loss_pct,
        "within_master": within_master,
    }


def classify_history_bucket(return_pct: float, iv_change: float) -> str:
    """Assign historical move to bucket A/B/C/D/E."""
    abs_ret = abs(return_pct)
    abs_iv = abs(iv_change)
    if abs_ret <= 0.005 and abs_iv <= 1.0:
        return "A"
    if abs_ret <= 0.01 and abs_iv <= 2.0:
        return "B"
    if abs_ret <= 0.015 and abs_iv <= 3.0:
        return "C"
    if abs_ret <= 0.025 and abs_iv <= 5.0:
        return "D"
    return "E"


def normalize_probabilities(prob_map: Dict[str, float]) -> Dict[str, float]:
    """Ensure probabilities sum to 1."""
    total = sum(prob_map.values())
    if total <= 0:
        return {k: 0.0 for k in prob_map}
    return {k: v / total for k, v in prob_map.items()}


def compute_var_es_metrics(
    losses: List[Dict[str, float]], nav: float, levels: Tuple[float, float] = (0.95, 0.99)
) -> Dict[str, float]:
    """Compute VaR/ES for supplied loss distribution (loss % NAV)."""
    if not losses:
        return {
            "VaR95": 0.0,
            "VaR99": 0.0,
            "ES95": 0.0,
            "ES99": 0.0,
            "tail_set_95": [],
            "tail_set_99": [],
        }
    sorted_losses = sorted(losses, key=lambda x: x["loss_pct"])

    def var_at(level: float) -> float:
        target = level
        cum = 0.0
        for item in sorted_losses:
            cum += item["prob"]
            if cum >= target:
                return item["loss_pct"]
        return sorted_losses[-1]["loss_pct"]

    def es_at(level: float) -> Tuple[float, List[Dict[str, float]]]:
        tail_prob = 1.0 - level
        if tail_prob <= 0:
            return 0.0, []
        remaining = tail_prob
        tail_sum = 0.0
        tail_entries: List[Dict[str, float]] = []
        for item in reversed(sorted_losses):
            loss = item["loss_pct"]
            prob = item["prob"]
            if loss <= 0 or prob <= 0:
                continue
            take = min(prob, remaining)
            if take <= 0:
                continue
            contribution = take * loss
            entry = dict(item)
            entry["tail_prob"] = take
            entry["tail_contribution"] = contribution
            tail_entries.append(entry)
            tail_sum += contribution
            remaining -= take
            if remaining <= 0:
                break
        es_value = (tail_sum / tail_prob) if tail_prob > 0 and tail_sum > 0 else 0.0
        return es_value, tail_entries

    var95 = var_at(levels[0])
    var99 = var_at(levels[1])
    es95, tail95 = es_at(levels[0])
    es99, tail99 = es_at(levels[1])

    return {
        "VaR95": var95,
        "VaR99": var99,
        "ES95": es95,
        "ES99": es99,
        "VaR95Value": (var95 / 100.0) * nav,
        "VaR99Value": (var99 / 100.0) * nav,
        "ES95Value": (es95 / 100.0) * nav,
        "ES99Value": (es99 / 100.0) * nav,
        "tail_set_95": tail95,
        "tail_set_99": tail99,
    }
