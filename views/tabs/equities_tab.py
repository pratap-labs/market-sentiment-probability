"""Equities tab - equity sleeve risk/health view."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

try:
    from kiteconnect import KiteConnect
except Exception:
    KiteConnect = None

from scripts.utils.formatters import format_inr
from scripts.utils.stress_testing import (
    classify_bucket,
    compute_var_es_metrics,
    get_weighted_scenarios,
)
from views.tabs.risk_analysis_tab import (
    DEFAULT_BUCKET_PROBS,
    compute_historical_bucket_probabilities,
    get_iv_regime,
)


def _is_equity_cash(pos: Dict[str, object]) -> bool:
    exchange = str(pos.get("exchange", "")).upper()
    segment = str(pos.get("segment", "")).upper()
    return "NSE" in exchange or "BSE" in exchange or "NSE" in segment or "BSE" in segment


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
    except Exception:
        return None


@st.cache_data(ttl=900)
def _fetch_holdings(api_key: str, access_token: str) -> List[Dict[str, object]]:
    if KiteConnect is None:
        return []
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    data = kite.holdings()
    return data or []


@st.cache_data(ttl=21600)
def _fetch_history(
    instrument_token: int,
    from_date: datetime,
    to_date: datetime,
    api_key: str,
    access_token: str,
) -> pd.DataFrame:
    if KiteConnect is None:
        return pd.DataFrame()
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    data = kite.historical_data(
        instrument_token=instrument_token,
        from_date=from_date,
        to_date=to_date,
        interval="day",
    )
    return pd.DataFrame(data or [])


def _compute_drawdown_from_peak(
    pos: Dict[str, object],
    lookback_days: int,
) -> Tuple[Optional[float], Optional[str]]:
    instrument_token = pos.get("instrument_token")
    if not instrument_token:
        return None, "N/A (missing token)"
    kite = _get_kite_client()
    if kite is None:
        return None, "N/A (Kite history unavailable)"

    entry_date = pos.get("entry_date")
    to_date = datetime.now()
    if entry_date:
        try:
            from_date = pd.to_datetime(entry_date)
        except Exception:
            from_date = to_date - timedelta(days=int(lookback_days))
    else:
        from_date = to_date - timedelta(days=int(lookback_days))

    try:
        df = _fetch_history(
            int(instrument_token),
            from_date,
            to_date,
            st.session_state.get("kite_api_key", ""),
            st.session_state.get("kite_access_token", ""),
        )
    except Exception:
        return None, "N/A (history fetch failed)"

    if df.empty or "close" not in df.columns:
        return None, "N/A (needs price history)"

    peak = df["close"].max()
    ltp = pos.get("last_price") or pos.get("ltp") or pos.get("mark_price")
    if ltp is None or not peak:
        return None, "N/A"
    drawdown_pct = (float(ltp) - float(peak)) / float(peak) * 100.0
    return drawdown_pct, None


def _compute_time_under_water_days(
    pos: Dict[str, object],
    lookback_days: int,
) -> Tuple[Optional[object], bool, Optional[str]]:
    avg_cost = pos.get("average_price") or pos.get("avg_price") or pos.get("cost_price")
    if not avg_cost:
        return None, False, "N/A (missing cost)"
    instrument_token = pos.get("instrument_token")
    if not instrument_token:
        return None, False, "N/A (missing token)"

    kite = _get_kite_client()
    if kite is None:
        return None, False, "N/A (Kite history unavailable)"

    to_date = datetime.now()
    from_date = to_date - timedelta(days=int(lookback_days))

    try:
        df = _fetch_history(
            int(instrument_token),
            from_date,
            to_date,
            st.session_state.get("kite_api_key", ""),
            st.session_state.get("kite_access_token", ""),
        )
    except Exception:
        return None, False, "N/A (history fetch failed)"

    if df.empty or "close" not in df.columns or "date" not in df.columns:
        return None, False, "N/A (history unavailable)"

    df = df.dropna(subset=["close", "date"]).copy()
    if df.empty:
        return None, False, "N/A (history unavailable)"

    df["date"] = pd.to_datetime(df["date"])
    last_date = df["date"].max()
    breakeven = df[df["close"] >= float(avg_cost)]
    if breakeven.empty:
        return f">{lookback_days}d", True, None
    last_breakeven = breakeven["date"].max()
    return (last_date - last_breakeven).days, False, None


def _scenario_probabilities(lookback_days: int) -> Dict[str, float]:
    bucket_probs, _, _ = compute_historical_bucket_probabilities(
        lookback=int(lookback_days),
        smoothing_enabled=False,
        smoothing_span=63,
    )
    if not bucket_probs:
        bucket_probs = DEFAULT_BUCKET_PROBS.copy()
    return bucket_probs


def _compute_equity_es99_per_symbol(
    equities: List[Dict[str, object]],
    lookback_days: int,
) -> Dict[str, float]:
    options_df_cache = st.session_state.get("options_df_cache", pd.DataFrame())
    iv_percentile = 35
    if isinstance(options_df_cache, pd.DataFrame) and not options_df_cache.empty and "iv" in options_df_cache.columns:
        iv_percentile = 35
    iv_regime, _ = get_iv_regime(iv_percentile)
    scenarios = get_weighted_scenarios(iv_regime)
    bucket_probs = _scenario_probabilities(lookback_days)
    bucket_counts: Dict[str, int] = {}
    for scenario in scenarios:
        bucket = classify_bucket(
            {"type": scenario.category.upper(), "dS_pct": scenario.ds_pct, "dIV_pts": scenario.div_pts}
        )
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    scenario_probs: List[Tuple[float, float]] = []
    for scenario in scenarios:
        bucket = classify_bucket(
            {"type": scenario.category.upper(), "dS_pct": scenario.ds_pct, "dIV_pts": scenario.div_pts}
        )
        bucket_prob = bucket_probs.get(bucket, 0.0)
        count = bucket_counts.get(bucket, 0)
        prob = bucket_prob / count if count and bucket_prob > 0 else 0.0
        scenario_probs.append((scenario.ds_pct, prob))

    es99_map: Dict[str, float] = {}
    for pos in equities:
        symbol = pos.get("tradingsymbol") or pos.get("symbol") or pos.get("instrument") or "—"
        qty = float(pos.get("quantity") or pos.get("qty") or 0.0)
        avg_cost = pos.get("average_price") or pos.get("avg_price") or pos.get("cost_price")
        ltp = pos.get("last_price") or pos.get("ltp") or pos.get("mark_price") or avg_cost or 0.0
        nav = float(avg_cost) * qty if avg_cost else float(ltp) * qty
        if nav <= 0 or qty == 0:
            es99_map[symbol] = None
            continue
        losses = []
        for ds_pct, prob in scenario_probs:
            pnl = qty * float(ltp) * (ds_pct / 100.0)
            loss_pct = (-pnl / nav * 100.0) if pnl < 0 else 0.0
            losses.append({"loss_pct": loss_pct, "prob": prob})
        metrics = compute_var_es_metrics(losses, nav)
        es99_map[symbol] = float(metrics.get("ES99Value", 0.0))
    return es99_map


def render_equities_tab() -> None:
    st.markdown("## Equities")

    api_key = st.session_state.get("kite_api_key")
    access_token = st.session_state.get("kite_access_token")
    if not api_key or not access_token or KiteConnect is None:
        st.info("Kite not connected. Login to load equity holdings.")
        return

    try:
        holdings = _fetch_holdings(api_key, access_token)
    except Exception:
        holdings = []
    equities = [p for p in holdings if _is_equity_cash(p)] if holdings else []
    if not equities:
        st.info("No equity holdings found.")
        return

    st.session_state.setdefault("equities_shock_levels", [-5, -10, -20, -30])
    st.session_state.setdefault("equities_current_shock", -10)
    st.session_state.setdefault("equities_lookback_days", 365)
    st.session_state.setdefault("equities_capital_base", "auto")

    controls = st.sidebar
    with controls.expander("Equities Controls", expanded=False):
        shock_levels = st.multiselect(
            "Shock levels (%)",
            options=[-5, -10, -20, -30],
            default=st.session_state.get("equities_shock_levels", [-5, -10, -20, -30]),
            key="equities_shock_levels",
        )
        current_shock = st.selectbox(
            "Current shock",
            options=shock_levels or [-10],
            key="equities_current_shock",
        )
        lookback_days = st.slider(
            "History lookback (days)",
            min_value=90,
            max_value=730,
            value=int(st.session_state.get("equities_lookback_days", 365)),
            key="equities_lookback_days",
        )
        capital_base = st.selectbox(
            "Equity sleeve capital base",
            options=["auto", "cost_basis", "market_value"],
            format_func=lambda v: {
                "auto": "Auto (cost if available)",
                "cost_basis": "Cost basis",
                "market_value": "Market value",
            }[v],
            key="equities_capital_base",
        )
        if st.button("Refresh history", key="equities_refresh_history"):
            _fetch_history.clear()

    rows = []
    es99_available = False
    history_unavailable = False
    drawdown_history_unavailable = False
    es99_map = _compute_equity_es99_per_symbol(equities, int(lookback_days))
    for pos in equities:
        symbol = pos.get("tradingsymbol") or pos.get("symbol") or pos.get("instrument") or "—"
        qty = float(pos.get("quantity") or pos.get("qty") or 0.0)
        avg_cost = pos.get("average_price") or pos.get("avg_price") or pos.get("cost_price")
        ltp = pos.get("last_price") or pos.get("ltp") or pos.get("mark_price") or avg_cost or 0.0
        market_value = float(ltp) * qty if ltp is not None else 0.0
        cost_basis = float(avg_cost) * qty if avg_cost is not None else None
        unreal_pnl = (float(ltp) - float(avg_cost)) * qty if avg_cost is not None else None
        unreal_pnl_pct = ((float(ltp) - float(avg_cost)) / float(avg_cost) * 100.0) if avg_cost else None
        return_vs_cost_pct = ((float(ltp) - float(avg_cost)) / float(avg_cost) * 100.0) if avg_cost else None
        stress_loss = float(ltp) * qty * (float(current_shock) / 100.0)

        es99_inr = es99_map.get(symbol)
        if es99_inr is not None:
            es99_available = True

        time_under_water, censored, tuw_note = _compute_time_under_water_days(pos, lookback_days)
        drawdown_from_peak, dd_note = _compute_drawdown_from_peak(pos, lookback_days)
        if dd_note and "Kite history unavailable" in dd_note:
            drawdown_history_unavailable = True
        if tuw_note:
            time_under_water_display = tuw_note
            if "Kite history unavailable" in tuw_note or "history fetch failed" in tuw_note:
                history_unavailable = True
        elif censored:
            time_under_water_display = time_under_water
        else:
            time_under_water_display = f"{time_under_water}d" if time_under_water is not None else "N/A"

        rows.append(
            {
                "symbol": symbol,
                "qty": qty,
                "avg_cost": avg_cost,
                "ltp": ltp,
                "market_value": market_value,
                "unreal_pnl": unreal_pnl,
                "unreal_pnl_pct": unreal_pnl_pct,
                "return_vs_cost_pct": return_vs_cost_pct,
                "drawdown_pct": drawdown_from_peak,
                "time_under_water": time_under_water_display,
                "es99_inr": es99_inr,
                "es99_pct": None,
                "stress_loss": stress_loss,
                "cost_basis": cost_basis,
            }
        )

    df = pd.DataFrame(rows)
    equity_value = df["market_value"].sum()
    cost_basis_sum = df["cost_basis"].dropna().sum() if df["cost_basis"].notna().any() else None

    if capital_base == "cost_basis" and cost_basis_sum is not None:
        equity_capital_base = cost_basis_sum
        base_label = "Cost basis"
    elif capital_base == "market_value":
        equity_capital_base = equity_value
        base_label = "Market value"
    else:
        equity_capital_base = cost_basis_sum if cost_basis_sum is not None else equity_value
        base_label = "Cost basis" if cost_basis_sum is not None else "Market value"

    total_portfolio_value = st.session_state.get("account_size")
    denom_label = "Account size"
    if not total_portfolio_value:
        total_portfolio_value = st.session_state.get("tba_total_capital")
        denom_label = "Total capital"
    if not total_portfolio_value:
        total_portfolio_value = equity_value
        denom_label = "Equity value"

    allocation_pct = (equity_value / total_portfolio_value * 100.0) if total_portfolio_value else 0.0
    sleeve_drawdown_pct = None
    if cost_basis_sum:
        sleeve_drawdown_pct = df["unreal_pnl"].dropna().sum() / cost_basis_sum * 100.0

    stress_loss_inr_sleeve = df["stress_loss"].sum()
    stress_loss_pct = (stress_loss_inr_sleeve / equity_capital_base * 100.0) if equity_capital_base else 0.0

    shock_levels = shock_levels or [-10]
    shock_losses = pd.Series(
        {lvl: (df["market_value"] * (lvl / 100.0)).sum() for lvl in shock_levels},
        name="stress_loss_inr",
    )

    underwater_count = 0
    weighted_tuw = 0.0
    weight_total = 0.0
    for _, row in df.iterrows():
        if row["return_vs_cost_pct"] is not None and row["return_vs_cost_pct"] < 0:
            underwater_count += 1
        tuw_val = row["time_under_water"]
        if isinstance(tuw_val, str) and tuw_val.startswith(">"):
            try:
                tuw_numeric = float(tuw_val.replace(">", "").replace("d", ""))
            except Exception:
                tuw_numeric = None
        elif isinstance(tuw_val, str) and tuw_val.endswith("d"):
            try:
                tuw_numeric = float(tuw_val.replace("d", ""))
            except Exception:
                tuw_numeric = None
        elif isinstance(tuw_val, (int, float)):
            tuw_numeric = float(tuw_val)
        else:
            tuw_numeric = None

        if tuw_numeric is not None and row["market_value"] > 0:
            weighted_tuw += tuw_numeric * row["market_value"]
            weight_total += row["market_value"]

    pct_underwater = (underwater_count / len(df) * 100.0) if len(df) else 0.0
    weighted_tuw_days = (weighted_tuw / weight_total) if weight_total else None

    portfolio_es99_inr = st.session_state.get("portfolio_es99_inr")
    equity_es99_inr = df["es99_inr"].dropna().sum() if es99_available else None
    equity_es99_pct = None
    if equity_es99_inr is not None and portfolio_es99_inr:
        equity_es99_pct = equity_es99_inr / float(portfolio_es99_inr) * 100.0

    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Equity sleeve value", format_inr(equity_value))
    kpi_cols[1].metric("Allocation %", f"{allocation_pct:.2f}%", help=f"Denominator: {denom_label}")
    kpi_cols[2].metric(
        "Sleeve drawdown %",
        f"{sleeve_drawdown_pct:.2f}%" if sleeve_drawdown_pct is not None else "N/A",
    )
    kpi_cols[3].metric(
        "Sleeve stress loss",
        f"{format_inr(stress_loss_inr_sleeve)} ({stress_loss_pct:.2f}%)",
    )
    kpi_cols[4].metric(
        "Equity ES99 contrib %",
        f"{equity_es99_pct:.2f}%" if equity_es99_pct is not None else "N/A",
    )

    stress_top = (
        df.set_index("symbol")["stress_loss"]
        .sort_values()
        .head(5)
        .index.tolist()
    )
    st.caption(
        f"Capital base: {format_inr(equity_capital_base)} ({base_label}). "
        f"Top stress contributors: {', '.join(stress_top) if stress_top else 'N/A'}."
    )

    st.markdown("### Under-water Summary")
    under_cols = st.columns(2)
    under_cols[0].metric("Holdings under water", f"{pct_underwater:.1f}%")
    under_cols[1].metric(
        "Weighted avg time under water",
        f"{weighted_tuw_days:.0f}d" if weighted_tuw_days is not None else "N/A",
    )

    if history_unavailable:
        st.warning("Time under water: Kite history unavailable (check login/credentials).")
    if drawdown_history_unavailable:
        st.warning("Drawdown needs Kite history; currently unavailable.")

    if not es99_available:
        st.caption("ES99 contribution not available for equities (unable to compute per-symbol ES99).")
    elif equity_es99_inr is not None and portfolio_es99_inr:
        df.loc[df["es99_inr"].notna(), "es99_pct"] = (
            df.loc[df["es99_inr"].notna(), "es99_inr"] / float(portfolio_es99_inr) * 100.0
        )

    st.markdown("### Equity Scenario Table")
    table_df = df.copy()
    table_df["avg_cost"] = table_df["avg_cost"].apply(lambda v: format_inr(v) if v is not None else "N/A")
    table_df["ltp"] = table_df["ltp"].apply(lambda v: format_inr(v) if v is not None else "N/A")
    table_df["market_value"] = table_df["market_value"].apply(format_inr)
    table_df["unreal_pnl"] = table_df["unreal_pnl"].apply(lambda v: format_inr(v) if v is not None else "N/A")
    table_df["unreal_pnl_pct"] = table_df["unreal_pnl_pct"].apply(lambda v: f"{v:.2f}%" if v is not None else "N/A")
    table_df["return_vs_cost_pct"] = table_df["return_vs_cost_pct"].apply(lambda v: f"{v:.2f}%" if v is not None else "N/A")
    table_df["drawdown_pct"] = table_df["drawdown_pct"].apply(lambda v: f"{v:.2f}%" if v is not None else "N/A (needs price history)")
    table_df["stress_loss"] = table_df["stress_loss"].apply(format_inr)
    table_df["stress_pct_stock"] = df["stress_loss"] / df["market_value"] * 100.0
    table_df["stress_contrib_pct"] = df["stress_loss"] / stress_loss_inr_sleeve * 100.0 if stress_loss_inr_sleeve else 0.0
    table_df["concentration_flag"] = table_df["stress_contrib_pct"].apply(lambda v: "⚠️" if abs(v) > 10 else "")
    table_df["stress_pct_stock"] = table_df["stress_pct_stock"].apply(lambda v: f"{v:.2f}%" if pd.notna(v) else "N/A")
    table_df["stress_contrib_pct"] = table_df["stress_contrib_pct"].apply(lambda v: f"{v:.2f}%" if pd.notna(v) else "N/A")
    table_df["es99_inr"] = table_df["es99_inr"].apply(lambda v: format_inr(v) if v is not None else "N/A")
    table_df["es99_pct"] = table_df["es99_pct"].apply(lambda v: f"{v:.2f}%" if v is not None else "N/A")

    display_cols = [
        "symbol",
        "qty",
        "avg_cost",
        "ltp",
        "market_value",
        "unreal_pnl",
        "unreal_pnl_pct",
        "return_vs_cost_pct",
        "drawdown_pct",
        "time_under_water",
        "stress_loss",
        "stress_pct_stock",
        "stress_contrib_pct",
        "concentration_flag",
        "es99_inr",
        "es99_pct",
    ]
    st.dataframe(
        table_df[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "return_vs_cost_pct": st.column_config.TextColumn("Return vs Cost (%)"),
            "drawdown_pct": st.column_config.TextColumn(
                "Drawdown from Peak (%)",
                help="Peak-to-trough drawdown from highest price since entry (or lookback). Requires price history.",
            ),
        },
        column_order=display_cols,
        column_config_overrides={"symbol": st.column_config.TextColumn("Symbol", pinned=True)},
    )

    st.markdown("### Scenario Comparison (Sleeve Stress Loss)")
    st.bar_chart(shock_losses)

    st.markdown("### Allocation by Holding")
    alloc_series = df.set_index("symbol")["market_value"].sort_values(ascending=False).head(10)
    st.bar_chart(alloc_series)

    st.markdown("### Top Stress Contributors (Current Shock)")
    top_stress = df.set_index("symbol")["stress_loss"].sort_values().head(10)
    st.bar_chart(top_stress)

    st.markdown("### Risk Concentration KPIs")
    top1_pct = (alloc_series.iloc[0] / equity_value * 100.0) if not alloc_series.empty and equity_value else 0.0
    top5_pct = (alloc_series.head(5).sum() / equity_value * 100.0) if equity_value else 0.0
    stress_abs = df.set_index("symbol")["stress_loss"].abs().sort_values(ascending=False)
    total_stress_abs = stress_abs.sum()
    top1_stress_pct = (stress_abs.iloc[0] / total_stress_abs * 100.0) if not stress_abs.empty and total_stress_abs else 0.0
    top5_stress_pct = (stress_abs.head(5).sum() / total_stress_abs * 100.0) if total_stress_abs else 0.0
    kpi2 = st.columns(4)
    kpi2[0].metric("Top 1 holding %", f"{top1_pct:.2f}%")
    kpi2[1].metric("Top 5 holdings %", f"{top5_pct:.2f}%")
    kpi2[2].metric("Top 1 stress contrib %", f"{top1_stress_pct:.2f}%")
    kpi2[3].metric("Top 5 stress contrib %", f"{top5_stress_pct:.2f}%")

    if equity_es99_inr is not None:
        st.markdown("### ES99 Contribution by Holding")
        es99_series = df.set_index("symbol")["es99_inr"].dropna().sort_values(ascending=False).head(10)
        if not es99_series.empty:
            st.bar_chart(es99_series)
