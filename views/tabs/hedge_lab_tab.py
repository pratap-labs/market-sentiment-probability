"""Hedge Lab - sandboxed hedge design and comparison."""

from __future__ import annotations

import calendar
from collections import Counter
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

try:
    from kiteconnect import KiteConnect
except Exception:
    KiteConnect = None

from scripts.utils import (
    calculate_portfolio_greeks,
    build_threshold_report,
    get_weighted_scenarios,
    compute_var_es_metrics,
    DEFAULT_LOT_SIZE,
    format_inr,
    calculate_greeks,
)
from views.tabs.risk_analysis_tab import compute_historical_bucket_probabilities

HEDGE_DISCLAIMER = (
    "Hedge Lab is for design and comparison only.  \n"
    "Risk Analysis reflects the live portfolio state."
)

FIXED_STRIKE_CANDIDATES = [24000, 25000, 26000, 27000, 28000, 30000]
LOT_CANDIDATES = [1, 2, 3]


def _determine_iv_percentile(options_df: pd.DataFrame) -> float:
    if options_df is None or options_df.empty or "iv" not in options_df.columns:
        return 30.0
    iv_series = options_df["iv"].dropna()
    if iv_series.empty:
        return 35.0
    return float(min(max(iv_series.median(), 5.0), 95.0))


def _get_iv_regime(percentile: float) -> str:
    if percentile < 20:
        return "Low IV"
    if percentile < 40:
        return "Mid IV"
    return "High IV"


def _spot_from_options(options_df: pd.DataFrame, fallback: float) -> float:
    if fallback and fallback > 0:
        return fallback
    if options_df is None or options_df.empty:
        return fallback
    spot_col = "underlying_value" if "underlying_value" in options_df.columns else None
    if spot_col:
        latest = options_df.sort_values("date").iloc[-1]
        val = latest.get(spot_col)
        if pd.notna(val):
            return float(val)
    if "close" in options_df.columns:
        return float(options_df["close"].dropna().iloc[-1])
    return fallback


def _monthly_expiry_schedule(today: datetime) -> List[pd.Timestamp]:
    """Return last-Tuesday expiries from current month through Dec 2026."""
    expiries: List[pd.Timestamp] = []
    year = today.year
    month = today.month
    while year <= 2026:
        last_day = datetime(year, month, calendar.monthrange(year, month)[1])
        # Roll back to Tuesday (weekday=1)
        while last_day.weekday() != 1:
            last_day -= timedelta(days=1)
        if last_day.month == 3 and last_day.day == 31:
            last_day -= timedelta(days=1)
        if last_day >= today:
            expiries.append(pd.Timestamp(last_day))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return expiries


def _resolve_option_quote(options_df: pd.DataFrame, strike: float, expiry: pd.Timestamp, option_type: str) -> Optional[pd.Series]:
    if options_df is None or options_df.empty:
        return None
    df = options_df.copy()
    if "expiry_date" not in df.columns:
        return None
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])
    matches = df[
        (df["expiry_date"] == expiry) &
        (df["strike_price"] == strike) &
        (df["option_type"] == option_type.upper())
    ]
    if matches.empty:
        return None
    latest = matches.sort_values("date" if "date" in matches.columns else "timestamp").iloc[-1]
    return latest


def _compute_account_snapshot() -> Tuple[float, float]:
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
    account_size = margin_available + margin_used
    margin_deployed = margin_used if margin_used > 0 else account_size
    return account_size, margin_deployed


def _format_tradingsymbol(expiry: datetime, strike: int, option_type: str) -> Optional[str]:
    if not isinstance(expiry, datetime):
        return None
    year_code = expiry.strftime("%y")
    month_code = expiry.strftime("%b").upper()
    strike_int = int(strike)
    opt = option_type.upper()
    return f"NIFTY{year_code}{month_code}{strike_int}{opt}"


def _get_kite_client() -> Optional["KiteConnect"]:
    if KiteConnect is None:
        return None
    api_key = st.session_state.get("kite_api_key")
    access_token = st.session_state.get("kite_access_token")
    if not api_key or not access_token:
        return None
    try:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        return kite
    except Exception as exc:
        st.warning(f"Kite authentication invalid for Hedge Lab quotes: {exc}")
        return None


def _fetch_kite_quotes(strikes: List[int], expiry: pd.Timestamp, option_type: str) -> Dict[int, Dict[str, float]]:
    kite = _get_kite_client()
    if kite is None or not strikes:
        return {}
    instruments = []
    strike_map = {}
    for strike in strikes:
        symbol = _format_tradingsymbol(expiry.to_pydatetime(), strike, option_type)
        if not symbol:
            continue
        token = f"NFO:{symbol}"
        instruments.append(token)
        strike_map[token] = strike
    if not instruments:
        return {}
    try:
        quote_data = kite.quote(instruments)
    except Exception as exc:
        st.warning(f"Failed to fetch Kite quotes: {exc}")
        return {}
    results: Dict[int, Dict[str, float]] = {}
    for instrument, payload in quote_data.items():
        strike = strike_map.get(instrument)
        if strike is None:
            continue
        last_price = payload.get("last_price") or payload.get("last_traded_price")
        if last_price is None:
            last_price = payload.get("ohlc", {}).get("close")
        oi_value = payload.get("oi") or payload.get("open_interest") or payload.get("net_change") or 0
        results[strike] = {"ltp": last_price, "oi": oi_value}
    return results


def _build_scenario_payload(regime: str) -> Tuple[List[Dict], List]:
    scenarios = get_weighted_scenarios(regime)
    scenario_dicts = [
        {
            "name": scenario.name,
            "dS_pct": scenario.ds_pct,
            "dIV_pts": scenario.div_pts,
            "type": scenario.category.upper(),
        }
        for scenario in scenarios
    ]
    return scenario_dicts, scenarios


def _run_es_pipeline(
    portfolio_greeks: Dict[str, float],
    account_size: float,
    margin_deployed: float,
    scenario_dicts: List[Dict],
    bucket_probs: Dict[str, float],
    master_pct: float,
    hard_stop_pct: float,
    normal_share: float,
    stress_share: float,
):
    if account_size <= 0 or margin_deployed <= 0:
        return {
            "threshold": None,
            "metrics": {"ES99": 0.0, "VaR99": 0.0, "ES95": 0.0, "VaR95": 0.0},
            "derived_rows": [],
        }
    context = build_threshold_report(
        portfolio={
            "delta": portfolio_greeks.get("net_delta", 0.0),
            "gamma": portfolio_greeks.get("net_gamma", 0.0),
            "vega": portfolio_greeks.get("net_vega", 0.0),
            "spot": portfolio_greeks.get("spot", 0.0),
            "nav": account_size,
            "margin": margin_deployed,
        },
        scenarios=scenario_dicts,
        master_pct=master_pct,
        hard_stop_pct=hard_stop_pct,
        normal_share=normal_share,
        stress_share=stress_share,
    )
    derived_rows = context.get("rows", [])
    if not derived_rows:
        return {"threshold": context, "metrics": {"ES99": 0.0, "VaR99": 0.0, "ES95": 0.0, "VaR95": 0.0}, "derived_rows": []}

    bucket_counts = Counter(row["bucket"] for row in derived_rows)
    for row in derived_rows:
        bucket = row["bucket"]
        prob_share = bucket_probs.get(bucket, 0.0)
        count = bucket_counts.get(bucket, 0)
        if prob_share > 0 and count > 0:
            row["probability"] = prob_share / count
        else:
            row["probability"] = 0.0

    loss_distribution = [
        {"loss_pct": row["loss_pct_nav"], "prob": row.get("probability", 0.0), "scenario": row["scenario"], "bucket": row["bucket"]}
        for row in derived_rows
    ]
    metrics = compute_var_es_metrics(loss_distribution, account_size)
    return {"threshold": context, "metrics": metrics, "derived_rows": derived_rows}


def _format_pct(value: float) -> str:
    return f"{value:.2f}%" if value or value == 0 else "—"


def _generate_candidate_rows(
    strike_values: List[int],
    expiry: pd.Timestamp,
    expiry_label: str,
    spot: float,
    base_greeks: Dict[str, float],
    base_eval: Dict,
    account_size: float,
    margin_deployed: float,
    scenario_dicts: List[Dict],
    bucket_probs: Dict[str, float],
    options_df: pd.DataFrame,
    max_premium: float,
    target_es: float,
    kite_quotes: Dict[int, Dict[str, float]],
    master_pct: float,
    hard_stop_pct: float,
    normal_share: float,
    stress_share: float,
    option_type: str,
) -> Tuple[List[Dict], Dict[str, Dict], Dict[str, object], Dict[str, List[Dict]]]:
    rows: List[Dict] = []
    payload_lookup: Dict[str, Dict] = {}
    diagnostics = {"missing_quotes": set(), "premium_filtered": 0, "no_reduction": 0}
    scenario_rows_lookup: Dict[str, List[Dict]] = {}
    base_metrics = base_eval["metrics"]
    base_es99 = base_metrics.get("ES99", 0.0)
    now = datetime.now()

    for strike in strike_values:
        quote = _resolve_option_quote(options_df, strike, expiry, option_type)
        kite_quote = kite_quotes.get(strike)
        price = None
        if kite_quote and kite_quote.get("ltp"):
            price = float(kite_quote.get("ltp"))
        if price is None:
            for col in ("ltp", "close", "last_price", "price"):
                if quote is not None and col in quote and pd.notna(quote[col]) and quote[col] > 0:
                    price = float(quote[col])
                    break
        if price is None or price <= 0:
            diagnostics["missing_quotes"].add(strike)
            continue
        implied_vol = float(quote.get("iv", 0.20)) if quote is not None and pd.notna(quote.get("iv", 0.20)) else 0.20
        dte = max((expiry - now).days, 1)
        time_to_expiry = dte / 365.0

        greeks = calculate_greeks(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            implied_vol=implied_vol,
            option_type=option_type,
        )
        for lots in LOT_CANDIDATES:
            quantity = lots * DEFAULT_LOT_SIZE
            premium = price * quantity
            if max_premium > 0 and premium > max_premium:
                diagnostics["premium_filtered"] += 1
                continue
            scaled_delta = greeks.get("delta", 0.0) * quantity
            scaled_gamma = greeks.get("gamma", 0.0) * quantity
            scaled_vega = greeks.get("vega", 0.0) * quantity
            scaled_theta = greeks.get("theta", 0.0) * quantity

            temp_greeks = {
                "net_delta": base_greeks["net_delta"] + scaled_delta,
                "net_gamma": base_greeks["net_gamma"] + scaled_gamma,
                "net_vega": base_greeks["net_vega"] + scaled_vega,
                "net_theta": base_greeks["net_theta"] + scaled_theta,
                "spot": spot,
            }

            eval_result = _run_es_pipeline(
                temp_greeks,
                account_size,
                margin_deployed,
                scenario_dicts,
                bucket_probs,
                master_pct,
                hard_stop_pct,
                normal_share,
                stress_share,
            )
            metrics = eval_result["metrics"]
            hedged_es99 = metrics.get("ES99", 0.0)
            hedged_var99 = metrics.get("VaR99", 0.0)
            worst_loss_pct = eval_result["threshold"]["worst_loss_pct"] if eval_result["threshold"] else 0.0
            delta_es99 = base_es99 - hedged_es99
            if delta_es99 <= 0:
                diagnostics["no_reduction"] += 1
                continue
            scenario_rows = eval_result.get("derived_rows", [])
            scenario_pass = bool(scenario_rows) and all(row.get("status") != "FAIL" for row in scenario_rows)
            efficiency_value = ((delta_es99 / 100.0) * account_size / premium) if premium > 0 else 0.0
            premium_pct = (premium / account_size * 100.0) if account_size > 0 else 0.0
            if target_es > 0:
                meets_target = scenario_pass and hedged_es99 <= target_es
            else:
                meets_target = scenario_pass
            row_id = f"{strike}_{expiry.strftime('%Y%m%d')}_{lots}"

            oi_value = 0.0
            if kite_quote and kite_quote.get("oi") is not None:
                oi_value = float(kite_quote.get("oi"))
            elif quote is not None and "oi" in quote and pd.notna(quote["oi"]):
                oi_value = float(quote["oi"])

            rows.append(
                {
                    "Hedge ID": row_id,
                    "Strike": int(strike),
                    "Lots": lots,
                    "Expiry": expiry,
                    "Expiry Label": expiry_label,
                    "Option Type": option_type.upper(),
                    "Premium ₹": premium,
                    "Premium % NAV": premium_pct,
                    "Hedged ES99": hedged_es99,
                    "Δ ES99": delta_es99,
                    "ES Reduction per ₹": efficiency_value,
                    "Hedged VaR99": hedged_var99,
                    "Worst Loss % NAV": worst_loss_pct,
                    "Theta impact": scaled_theta,
                    "Vega impact": scaled_vega,
                    "Delta impact": scaled_delta,
                    "Margin impact ₹": premium,
                    "Meets Target": meets_target,
                    "Hedge OI": oi_value,
                }
            )

            payload_lookup[row_id] = {
                "strike": int(strike),
                "lots": lots,
                "quantity": quantity,
                "price": price,
                "expiry": expiry,
                "expiry_str": expiry.strftime("%d %b %Y"),
                "option_type": option_type.upper(),
                "greeks": {
                    "delta": scaled_delta,
                    "gamma": scaled_gamma,
                    "vega": scaled_vega,
                    "theta": scaled_theta,
                },
                "premium": premium,
            }
            scenario_rows_lookup[row_id] = scenario_rows
    diagnostics["missing_quotes"] = sorted(diagnostics["missing_quotes"])
    return rows, payload_lookup, diagnostics, scenario_rows_lookup


def _prepare_df(row_list: List[Dict], best_id: Optional[str]) -> pd.DataFrame:
    if not row_list:
        return pd.DataFrame()
    df = pd.DataFrame(row_list)
    df["Premium ₹"] = df["Premium ₹"].map(lambda x: format_inr(x, decimals=0))
    df["Premium % NAV"] = df["Premium % NAV"].map(_format_pct)
    df["Hedged ES99"] = df["Hedged ES99"].map(_format_pct)
    df["Δ ES99"] = df["Δ ES99"].map(_format_pct)
    df["Hedged VaR99"] = df["Hedged VaR99"].map(_format_pct)
    df["Worst Loss % NAV"] = df["Worst Loss % NAV"].map(_format_pct)
    df["ES Reduction per ₹"] = df["ES Reduction per ₹"].map(lambda x: f"{x:.4f}")
    df["Theta impact"] = df["Theta impact"].map(lambda x: f"{x:+.0f}")
    df["Vega impact"] = df["Vega impact"].map(lambda x: f"{x:+.0f}")
    df["Delta impact"] = df["Delta impact"].map(lambda x: f"{x:+.0f}")
    df["Margin impact ₹"] = df["Margin impact ₹"].map(lambda x: format_inr(x, decimals=0))
    df["Hedge OI"] = df["Hedge OI"].map(lambda x: f"{float(x):,.0f}" if x is not None else "0")
    df["Best Hedge"] = df["Hedge ID"].apply(lambda hid: "⭐" if hid == best_id else "")
    df["Meets Target"] = df["Meets Target"].map(lambda x: "🎯" if x else "")
    display_cols = [
        "Strike",
        "Lots",
        "Option Type",
        "Premium ₹",
        "Premium % NAV",
        "Hedged ES99",
        "Δ ES99",
        "ES Reduction per ₹",
        "Hedged VaR99",
        "Worst Loss % NAV",
        "Theta impact",
        "Vega impact",
        "Delta impact",
        "Margin impact ₹",
        "Hedge OI",
        "Best Hedge",
        "Meets Target",
    ]
    return df[display_cols + ["Hedge ID"]]


def _top_per_expiry(rows: List[Dict], target_es: float, scenario_rows_lookup: Dict[str, List[Dict]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    grouped: Dict[str, List[Dict]] = {}
    for row in rows:
        label = row.get("Expiry Label")
        if not label:
            expiry = row.get("Expiry")
            label = expiry.strftime("%d %b %Y") if isinstance(expiry, (datetime, pd.Timestamp)) else str(expiry)
        grouped.setdefault(label, []).append(row)
    summary_rows = []
    for expiry_label, items in grouped.items():
        top_items = sorted(items, key=lambda r: r["ES Reduction per ₹"], reverse=True)[:3]
        for rank, entry in enumerate(top_items, start=1):
            scenario_rows = scenario_rows_lookup.get(entry["Hedge ID"], [])
            scenario_pass = bool(scenario_rows) and all(row.get("status") != "FAIL" for row in scenario_rows)
            meets_target = False
            if target_es > 0:
                meets_target = scenario_pass and entry.get("Hedged ES99", 999) <= target_es
            else:
                meets_target = scenario_pass
            summary_rows.append(
                {
                    "Expiry": expiry_label,
                    "Rank": rank,
                    "Strike": entry["Strike"],
                    "Type": entry.get("Option Type", "PE"),
                    "Lots": entry["Lots"],
                    "Premium ₹": format_inr(entry["Premium ₹"], decimals=0),
                    "Δ ES99": f"{entry['Δ ES99']:.2f}%",
                    "ES Redn / ₹": f"{entry['ES Reduction per ₹']:.4f}",
                    "Hedged ES99": f"{entry['Hedged ES99']:.2f}%",
                    "Target Pass": "✅" if meets_target else "—",
                }
            )
    return pd.DataFrame(summary_rows)


def _scenario_rows_to_df(rows: List[Dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    table = []
    for entry in rows:
        scenario = entry.get("scenario", {})
        table.append(
            {
                "Scenario": scenario.get("name", "N/A"),
                "Bucket": entry.get("bucket"),
                "dS% / dIV": f"{scenario.get('dS_pct', 0):+.2f}% / {scenario.get('dIV_pts', 0):+.1f}",
                "Δ P&L (₹)": format_inr(entry.get("pnl_delta", 0.0)),
                "Γ P&L (₹)": format_inr(entry.get("pnl_gamma", 0.0)),
                "Vega P&L (₹)": format_inr(entry.get("pnl_vega", 0.0)),
                "Total P&L (₹)": format_inr(entry.get("pnl_total", 0.0)),
                "Loss % NAV": f"{entry.get('loss_pct_nav', 0.0):.2f}%",
                "Threshold % NAV": f"{entry.get('threshold_pct', 0.0):.2f}%",
                "Status": entry.get("status"),
                "Probability": entry.get("probability", 0.0),
            }
        )
    return pd.DataFrame(table)


def _apply_hedge_to_portfolio(payload: Dict):
    expiry = payload["expiry"]
    expiry_str = expiry.strftime("%d%b%y").upper()
    strike = payload["strike"]
    opt = payload.get("option_type", "PE")
    tradingsymbol = f"HEDGE_{expiry_str}_{strike}{opt}"
    new_position = {
        "tradingsymbol": tradingsymbol,
        "quantity": payload["quantity"],
        "product": "NRML",
        "exchange": "NFO",
        "instrument_type": opt,
        "option_type": opt,
        "strike_price": strike,
        "expiry": expiry,
        "last_price": payload["price"],
        "position_delta": payload["greeks"]["delta"],
        "position_gamma": payload["greeks"]["gamma"],
        "position_vega": payload["greeks"]["vega"],
        "position_theta": payload["greeks"]["theta"],
        "source": "HEDGE_LAB",
    }
    enriched = list(st.session_state.get("enriched_positions", []))
    enriched.append(new_position)
    st.session_state["enriched_positions"] = enriched


def render_hedge_lab_tab():
    """Render Hedge Lab sandbox UI."""
    st.header("🛡️ Hedge Lab – Sandbox Hedging")
    st.caption(HEDGE_DISCLAIMER)

    # Sync positions button
    if st.sidebar.button("🔄 Sync Positions", key="hedge_lab_sync_positions", help="Fetch latest positions from Kite", use_container_width=True):
        st.rerun()

    if "enriched_positions" not in st.session_state:
        st.info("Load live positions first to design hedges.")
        return

    options_df = st.session_state.get("options_df_cache")
    if options_df is None or options_df.empty:
        st.warning("Derivatives cache unavailable. Load data in Derivatives tab before using Hedge Lab.")
        return
    if not st.session_state.get("kite_access_token") or not st.session_state.get("kite_api_key"):
        st.info("Login to Kite for live OI/price data; falling back to cached values where necessary.")

    positions = st.session_state["enriched_positions"]
    spot = _spot_from_options(options_df, st.session_state.get("current_spot", 0.0))
    if not spot or spot <= 0:
        st.error("Unable to determine current spot. Refresh derivatives data.")
        return

    today = datetime.now()
    expiries = _monthly_expiry_schedule(today)
    if not expiries:
        st.error("Unable to build monthly expiry calendar.")
        return

    default_selection = expiries[: min(3, len(expiries))]
    selected_expiries = st.sidebar.multiselect(
        "Hedge expiries to evaluate",
        options=expiries,
        default=default_selection,
        format_func=lambda x: x.strftime("%d %b %Y"),
        key="hedge_lab_expiries",
    )
    if not selected_expiries:
        st.warning("Select at least one expiry to evaluate hedges.")
        return

    col_inputs = st.columns(3)
    target_es = st.sidebar.number_input(
        "Target ES99 (% NAV) (optional)",
        min_value=0.0,
        max_value=12.0,
        step=0.1,
        value=float(st.session_state.get("strategy_es_limit", 0.0) or 0.0),
        help="Set to zero to disable target filtering.",
    )
    max_premium = st.sidebar.number_input(
        "Max hedge premium (₹)",
        min_value=0.0,
        step=5000.0,
        value=0.0,
        help="Hedges costing more than this will be discarded.",
    )
    instrument_type = st.sidebar.selectbox(
        "Hedge instrument",
        options=["PE", "CE"],
        index=0,
        key="hedge_lab_option_type",
    )

    thresh_cols = st.columns(4)
    master_pct = st.sidebar.number_input("Master loss budget (% NAV)", value=1.0, step=0.1, format="%.2f")
    hard_stop_pct = st.sidebar.number_input("Hard stop (% NAV)", value=1.2, step=0.1, format="%.2f")
    normal_share = st.sidebar.number_input("Normal share (fraction of master)", value=0.5, step=0.05, format="%.2f")
    stress_share = st.sidebar.number_input("Stress share (fraction of master)", value=0.9, step=0.05, format="%.2f")

    strike_candidates = [strike for strike in FIXED_STRIKE_CANDIDATES if strike > 0]
    st.markdown(f"**Strikes evaluated:** {', '.join(str(x) for x in strike_candidates)}")

    account_size, margin_deployed = _compute_account_snapshot()
    base_greeks = calculate_portfolio_greeks(positions)
    base_greeks["spot"] = spot

    iv_pct = _determine_iv_percentile(options_df)
    regime = _get_iv_regime(iv_pct)
    scenario_dicts, _ = _build_scenario_payload(regime)
    prob_cols = st.columns(3)
    lookback = st.sidebar.number_input(
        "Historical lookback (days)",
        value=504,
        min_value=126,
        max_value=756,
        step=21,
        key="hedge_lab_lookback",
    )
    smoothing_enabled = st.sidebar.checkbox(
        "Apply EWMA smoothing",
        value=False,
        key="hedge_lab_smoothing",
    )
    smoothing_span = 63
    if smoothing_enabled:
        smoothing_span = int(
            st.sidebar.slider(
                "EWMA span (days)",
                min_value=21,
                max_value=min(252, int(lookback)),
                value=63,
                step=7,
                key="hedge_lab_smoothing_span",
            )
        )
    bucket_probs, history_count, used_fallback = compute_historical_bucket_probabilities(
        lookback=int(lookback),
        smoothing_enabled=bool(smoothing_enabled),
        smoothing_span=int(smoothing_span),
    )
    st.caption(
        f"Bucket probabilities (lookback {history_count}d): "
        f"A {bucket_probs['A']*100:.2f}%, B {bucket_probs['B']*100:.2f}%, C {bucket_probs['C']*100:.2f}%"
    )
    if used_fallback:
        st.warning("Historical cache unavailable; using default manual probabilities (A/B/C = 60/30/10).")

    base_eval = _run_es_pipeline(
        base_greeks,
        account_size,
        margin_deployed,
        scenario_dicts,
        bucket_probs,
        master_pct,
        hard_stop_pct,
        normal_share,
        stress_share,
    )
    base_metrics = base_eval["metrics"]

    summary_cols = st.columns(4)
    summary_cols[0].metric("Base ES99", f"{base_metrics.get('ES99', 0.0):.2f}%")
    summary_cols[1].metric("Base VaR99", f"{base_metrics.get('VaR99', 0.0):.2f}%")
    summary_cols[2].metric("Base VaR95", f"{base_metrics.get('VaR95', 0.0):.2f}%")
    worst_pct = base_eval["threshold"]["worst_loss_pct"] if base_eval["threshold"] else 0.0
    summary_cols[3].metric("Worst Loss % NAV", f"{worst_pct:.2f}%")

    all_rows: List[Dict] = []
    payload_lookup: Dict[str, Dict] = {}
    scenario_rows_lookup: Dict[str, List[Dict]] = {}
    diag_messages: List[str] = []

    for expiry_choice in selected_expiries:
        expiry_label = expiry_choice.strftime("%d %b %Y")
        kite_quotes = _fetch_kite_quotes(strike_candidates, expiry_choice, instrument_type)
        rows, payloads, diag, scenario_map = _generate_candidate_rows(
            strike_candidates,
            expiry_choice,
            expiry_label,
            spot,
            base_greeks,
            base_eval,
            account_size,
            margin_deployed,
            scenario_dicts,
            bucket_probs,
            options_df,
            max_premium,
            target_es,
            kite_quotes,
            master_pct,
            hard_stop_pct,
            normal_share,
            stress_share,
            instrument_type,
        )
        if not rows:
            label = expiry_choice.strftime("%d %b %Y")
            if diag["missing_quotes"]:
                diag_messages.append(f"{label}: missing data for {', '.join(str(x) for x in diag['missing_quotes'])}.")
            if diag["premium_filtered"]:
                diag_messages.append(f"{label}: {diag['premium_filtered']} hedges exceeded premium cap.")
            if diag["no_reduction"]:
                diag_messages.append(f"{label}: {diag['no_reduction']} hedges failed to reduce ES99.")
        all_rows.extend(rows)
        payload_lookup.update(payloads)
        scenario_rows_lookup.update(scenario_map)

    if not all_rows:
        msg = "No hedge candidates reduced ES99 under the current filters."
        if diag_messages:
            msg += " " + " ".join(diag_messages)
        st.warning(msg)
        return

    all_rows.sort(key=lambda r: r["ES Reduction per ₹"], reverse=True)
    best_id = all_rows[0]["Hedge ID"]
    table_df = _prepare_df(all_rows, best_id)

    st.markdown("### Hedge Candidates")
    st.dataframe(table_df.drop(columns=["Hedge ID"]), use_container_width=True, hide_index=True)

    top_summary = _top_per_expiry(all_rows, target_es, scenario_rows_lookup)
    if not top_summary.empty:
        st.markdown("### Top 3 Hedges per Expiry")
        st.dataframe(top_summary, use_container_width=True, hide_index=True)

    featured_row = all_rows[0]
    target_hits = [r for r in all_rows if r["Meets Target"]]
    if target_es > 0 and target_hits:
        featured_row = target_hits[0]
        st.success(
            f"⭐ Best hedge meeting target: Strike {featured_row['Strike']} {featured_row['Option Type']} × {featured_row['Lots']} lots "
            f"({featured_row['Expiry Label']}) (ES reduction per ₹ {featured_row['ES Reduction per ₹']:.4f})"
        )
        st.info(f"🎯 {len(target_hits)} hedges meet target ES99 ≤ {target_es:.1f}%.")
    else:
        st.success(
            f"⭐ Best hedge (efficiency): Strike {featured_row['Strike']} {featured_row['Option Type']} × {featured_row['Lots']} lots "
            f"({featured_row['Expiry Label']}) (ES reduction per ₹ {featured_row['ES Reduction per ₹']:.4f})"
        )
        if target_es > 0 and not target_hits:
            st.warning("No hedge meets the target ES99 threshold yet; showing most efficient candidate overall.")

    candidate_pool = target_hits if target_es > 0 else all_rows
    if target_es > 0 and not target_hits:
        candidate_pool = []
    candidate_labels = {
        row["Hedge ID"]: f"{row['Strike']} {row['Option Type']} × {row['Lots']} lots ({row['Expiry Label']})"
        for row in candidate_pool
    }
    selection_options = [""] + list(candidate_labels.keys())
    selection = st.sidebar.selectbox(
        "Select hedge to apply",
        options=selection_options,
        format_func=lambda x: "Select..." if x == "" else candidate_labels.get(x, x),
        key="hedge_lab_selection",
    )
    st.caption("Applying a hedge will add a synthetic long put position to your live portfolio.")

    featured_id = featured_row["Hedge ID"]
    active_id = selection if selection else featured_id

    st.markdown("### Scenario Comparison")
    base_df = _scenario_rows_to_df(base_eval.get("derived_rows", []))
    if not base_df.empty:
        st.caption("Current Portfolio Scenarios")
        df_display = base_df.copy()
        df_display["Probability"] = df_display["Probability"].map(lambda x: f"{x*100:.2f}%")
        styled = df_display.style.applymap(
            lambda val: "background-color: #d1e7dd; color: #0f5132;" if str(val).upper() == "PASS"
            else "background-color: #f8d7da; color: #842029;" if str(val).upper() == "FAIL"
            else "",
            subset=["Status"],
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)
    hedged_rows = scenario_rows_lookup.get(active_id, [])
    hedged_df = _scenario_rows_to_df(hedged_rows)
    if not hedged_df.empty:
        st.caption(f"Hedged Scenarios ({candidate_labels.get(active_id, 'Selected Hedge')})")
        df_display = hedged_df.copy()
        df_display["Probability"] = df_display["Probability"].map(lambda x: f"{x*100:.2f}%")
        styled = df_display.style.applymap(
            lambda val: "background-color: #d1e7dd; color: #0f5132;" if str(val).upper() == "PASS"
            else "background-color: #f8d7da; color: #842029;" if str(val).upper() == "FAIL"
            else "",
            subset=["Status"],
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.info("Select a hedge to view scenario impacts.")

    apply_disabled = selection == "" or selection not in payload_lookup
    if st.sidebar.button("Apply Hedge to Portfolio", disabled=apply_disabled):
        payload = payload_lookup.get(selection)
        if not payload:
            st.error("Please select a hedge first.")
        else:
            _apply_hedge_to_portfolio(payload)
            st.success("Hedge applied to portfolio. Re-run Risk Analysis to evaluate the live state.")
            st.rerun()
