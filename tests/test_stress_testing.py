"""Unit tests for scenario stress testing utilities."""

from scripts.utils.stress_testing import (
    Scenario,
    STRESS_LIMIT_DEFAULTS,
    classify_bucket,
    classify_history_bucket,
    classify_scenario_limit,
    compute_scenario_pnl,
    compute_var_es_metrics,
    derive_thresholds,
)


def test_compute_scenario_pnl_short_vega_loses_on_iv_up():
    portfolio = {
        "delta": 0.0,
        "gamma": 0.0,
        "vega": -1500.0,  # short vega
        "spot": 20000.0,
        "account_size": 1_000_000.0,
        "margin_deployed": 500_000.0,
    }
    scenario = Scenario("IV +2", ds_pct=0.0, div_pts=2.0, category="iv")
    breakdown = compute_scenario_pnl(portfolio, scenario)
    assert breakdown["delta_pnl"] == 0.0
    assert breakdown["gamma_pnl"] == 0.0
    assert breakdown["vega_pnl"] < 0  # short vega loses on IV spike
    assert breakdown["loss_pct_account"] > 0


def test_compute_scenario_pnl_gamma_convexity_increase():
    portfolio = {
        "delta": 0.0,
        "gamma": -10.0,  # short gamma
        "vega": 0.0,
        "spot": 20000.0,
        "account_size": 1_000_000.0,
        "margin_deployed": 1_000_000.0,
    }
    scenario_small = Scenario("Move 1%", ds_pct=1.0, div_pts=0.0, category="price")
    scenario_large = Scenario("Move 2%", ds_pct=2.0, div_pts=0.0, category="price")
    pnl_small = compute_scenario_pnl(portfolio, scenario_small)["total_pnl"]
    pnl_large = compute_scenario_pnl(portfolio, scenario_large)["total_pnl"]
    assert pnl_small < 0
    assert pnl_large < 0
    # Gamma loss should grow roughly with square of move (≈4x here)
    assert abs(pnl_large) > abs(pnl_small) * 3.5


def test_classify_scenario_limit_routing():
    price_scenario = Scenario("Price", ds_pct=1.0, div_pts=0.0, category="price")
    iv_scenario = Scenario("IV", ds_pct=0.0, div_pts=2.0, category="iv")
    combined_scenario = Scenario("Combo", ds_pct=1.0, div_pts=2.0, category="combined")
    gap_scenario = Scenario("Gap", ds_pct=-2.0, div_pts=5.0, category="gap")

    assert (
        classify_scenario_limit(price_scenario, STRESS_LIMIT_DEFAULTS)
        == STRESS_LIMIT_DEFAULTS["price"]
    )
    assert (
        classify_scenario_limit(iv_scenario, STRESS_LIMIT_DEFAULTS)
        == STRESS_LIMIT_DEFAULTS["iv"]
    )
    assert (
        classify_scenario_limit(combined_scenario, STRESS_LIMIT_DEFAULTS)
        == STRESS_LIMIT_DEFAULTS["combined"]
    )
    assert (
        classify_scenario_limit(gap_scenario, STRESS_LIMIT_DEFAULTS)
        == STRESS_LIMIT_DEFAULTS["gap"]
    )


def test_classify_bucket_assignments():
    scenario_price = {"type": "PRICE", "dS_pct": 1.0, "dIV_pts": 0}
    scenario_price_big = {"type": "PRICE", "dS_pct": 1.6, "dIV_pts": 0}
    scenario_iv = {"type": "IV", "dS_pct": 0, "dIV_pts": 2.5}
    scenario_gap = {"type": "GAP", "dS_pct": -2.5, "dIV_pts": 6}
    assert classify_bucket(scenario_price) == "A"
    assert classify_bucket(scenario_price_big) == "C"
    assert classify_bucket(scenario_iv) == "B"
    assert classify_bucket(scenario_gap) == "C"


def test_derive_thresholds_from_master():
    thresholds = derive_thresholds(master_pct=1.0, hard_stop_pct=1.2, normal_share=0.5, stress_share=0.9)
    assert thresholds["limitA"] == 0.5
    assert thresholds["limitB"] == 0.9
    assert thresholds["limitC"] == 1.2


def test_classify_history_bucket_rules():
    assert classify_history_bucket(0.005, 1.0) == "A"
    assert classify_history_bucket(0.012, 1.0) == "B"
    assert classify_history_bucket(0.02, 1.0) == "C"
    assert classify_history_bucket(0.005, 6.0) == "C"


def test_compute_var_es_metrics():
    losses = [
        {"loss_pct": 0.5, "prob": 0.5},
        {"loss_pct": 1.0, "prob": 0.3},
        {"loss_pct": 2.0, "prob": 0.2},
    ]
    metrics = compute_var_es_metrics(losses, nav=1_000_000)
    assert metrics["VaR95"] >= 1.0
    assert metrics["VaR99"] >= metrics["VaR95"]
    assert metrics["ES99"] >= metrics["VaR99"]
    assert metrics["ES99Value"] > 0
