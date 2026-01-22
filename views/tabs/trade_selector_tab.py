"""Trade Selector tab for screening vertical spreads."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


@dataclass
class PricingConfig:
    mode: str
    slippage_per_leg: float
    brokerage_per_leg: float


@dataclass
class SimConfig:
    n_paths: int
    rf: float
    jump_enabled: bool
    cvar_level: float


def _get_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _normalize_chain(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    strike_col = _get_col(df, ["strike_price", "strike"])
    expiry_col = _get_col(df, ["expiry", "expiry_date"])
    opt_type_col = _get_col(df, ["option_type", "type"])
    bid_col = _get_col(df, ["bid", "best_buy_price", "buy_price", "bid_price"])
    ask_col = _get_col(df, ["ask", "best_sell_price", "sell_price", "ask_price"])
    iv_col = _get_col(df, ["iv", "implied_volatility"])
    oi_col = _get_col(df, ["open_int", "oi", "open_interest"])
    vol_col = _get_col(df, ["volume", "vol", "volume_traded"])
    ltp_col = _get_col(df, ["last_price", "ltp"])
    spot_col = _get_col(df, ["underlying_value", "spot", "underlying"])
    underlying_col = _get_col(df, ["underlying", "symbol", "name"])

    if strike_col is None or expiry_col is None or opt_type_col is None:
        return pd.DataFrame()

    df["strike"] = pd.to_numeric(df[strike_col], errors="coerce")
    df["expiry"] = pd.to_datetime(df[expiry_col], errors="coerce")
    df["option_type"] = df[opt_type_col].astype(str).str.upper().str.strip()

    df["bid"] = pd.to_numeric(df[bid_col], errors="coerce") if bid_col else np.nan
    df["ask"] = pd.to_numeric(df[ask_col], errors="coerce") if ask_col else np.nan
    df["ltp"] = pd.to_numeric(df[ltp_col], errors="coerce") if ltp_col else np.nan
    close_col = _get_col(df, ["close"])
    if close_col:
        df["close"] = pd.to_numeric(df[close_col], errors="coerce")
    else:
        df["close"] = np.nan
    df["price"] = df["close"]

    df["iv"] = pd.to_numeric(df[iv_col], errors="coerce") if iv_col else np.nan
    if df["iv"].median(skipna=True) > 3:
        df["iv"] = df["iv"] / 100.0
    df["iv"] = df["iv"].fillna(df["iv"].median()).fillna(0.2)

    df["oi"] = pd.to_numeric(df[oi_col], errors="coerce") if oi_col else np.nan
    df["volume"] = pd.to_numeric(df[vol_col], errors="coerce") if vol_col else np.nan

    df["spot"] = pd.to_numeric(df[spot_col], errors="coerce") if spot_col else np.nan
    if underlying_col:
        df["underlying"] = df[underlying_col].astype(str).str.upper()
    else:
        df["underlying"] = "NIFTY"

    df = df.dropna(subset=["strike", "expiry", "option_type"])
    return df


def _pick_strike(strikes: np.ndarray, target: float, direction: str) -> Optional[float]:
    if strikes.size == 0:
        return None
    if direction == "ge":
        candidates = strikes[strikes >= target]
        return float(candidates.min()) if candidates.size else None
    if direction == "le":
        candidates = strikes[strikes <= target]
        return float(candidates.max()) if candidates.size else None
    idx = np.abs(strikes - target).argmin()
    return float(strikes[idx])


def _strategy_spec(strategy: str) -> Dict[str, object]:
    if strategy == "Bull Put Spread":
        return {"type": "PE", "short_dir": "le", "long_dir": "le", "sign": "credit", "long_offset": -1}
    if strategy == "Bear Put Spread":
        return {"type": "PE", "short_dir": "le", "long_dir": "ge", "sign": "debit", "long_offset": 1}
    if strategy == "Bull Call Spread":
        return {"type": "CE", "short_dir": "ge", "long_dir": "le", "sign": "debit", "long_offset": -1}
    return {"type": "CE", "short_dir": "ge", "long_dir": "ge", "sign": "credit", "long_offset": 1}


def _leg_payoff(option_type: str, strike: float, spot: np.ndarray) -> np.ndarray:
    if option_type == "CE":
        return np.maximum(spot - strike, 0.0)
    return np.maximum(strike - spot, 0.0)


def _simulate_pnl(
    spot: float,
    days: int,
    sigma: float,
    rf: float,
    n_paths: int,
    jump_enabled: bool,
    premium: float,
    short_type: str,
    short_strike: float,
    long_type: str,
    long_strike: float,
    rng: np.random.Generator,
    z_base: np.ndarray,
    jump_mask: np.ndarray,
    jump_z: np.ndarray,
    short_mult: float,
    long_mult: float,
) -> np.ndarray:
    if days <= 0 or sigma <= 0:
        return np.array([premium])
    t = days / 365.0
    drift = (rf - 0.5 * sigma * sigma) * t
    vol_term = sigma * np.sqrt(t) * z_base
    jump_term = 0.0
    if jump_enabled:
        jump_term = jump_mask * (2.0 * sigma) * jump_z
    spot_t = spot * np.exp(drift + vol_term + jump_term)
    short_payoff = _leg_payoff(short_type, short_strike, spot_t) * short_mult
    long_payoff = _leg_payoff(long_type, long_strike, spot_t) * long_mult
    return premium + long_payoff - short_payoff


def _compute_spread_metrics(
    strategy: str,
    expiry: pd.Timestamp,
    spot: float,
    chain: pd.DataFrame,
    distances: List[int],
    widths: List[int],
    pricing: PricingConfig,
    sim_cfg: SimConfig,
    max_margin: float,
    max_spread_pct: float,
    min_oi: float,
    min_volume: float,
    lot_size: float,
    short_lots: float,
    wing_lots: float,
) -> pd.DataFrame:
    spec = _strategy_spec(strategy)
    opt_type = spec["type"]
    chain = chain[chain["option_type"] == opt_type].copy()
    if chain.empty:
        return pd.DataFrame()

    strikes = np.sort(chain["strike"].unique())
    chain_map = {row["strike"]: row for row in chain.to_dict(orient="records")}

    combos = []
    rng = np.random.default_rng(42)
    z_base = rng.standard_normal(sim_cfg.n_paths)
    jump_mask = rng.random(sim_cfg.n_paths) < 0.01
    jump_z = rng.standard_normal(sim_cfg.n_paths)

    days_to_expiry = max((expiry.date() - datetime.now().date()).days, 1)

    for dist in distances:
        short_target = spot - dist if opt_type == "PE" else spot + dist
        short_strike = _pick_strike(strikes, short_target, spec["short_dir"])
        if short_strike is None:
            continue
        for width in widths:
            long_target = short_strike + (width * spec["long_offset"])
            long_strike = _pick_strike(strikes, long_target, spec["long_dir"])
            if long_strike is None or long_strike == short_strike:
                continue
            short_row = chain_map.get(short_strike)
            long_row = chain_map.get(long_strike)
            if not short_row or not long_row:
                continue
            short_price = float(short_row.get("price") or 0.0)
            long_price = float(long_row.get("price") or 0.0)
            if short_price <= 0 or long_price <= 0:
                continue
            short_mult = lot_size * short_lots
            long_mult = lot_size * wing_lots
            premium = (short_price * short_mult) - (long_price * long_mult)
            costs = (pricing.slippage_per_leg + pricing.brokerage_per_leg) * (short_lots + wing_lots)
            premium -= costs

            # Validate strategy-specific constraints
            if strategy in {"Bull Put Spread", "Bear Call Spread"}:
                # Credit spreads must have positive premium
                if premium <= 0:
                    continue
                # Bull Put: short_strike > long_strike, Bear Call: short_strike < long_strike
                if strategy == "Bull Put Spread" and short_strike <= long_strike:
                    continue
                if strategy == "Bear Call Spread" and short_strike >= long_strike:
                    continue
            else:
                # Debit spreads must have negative premium (we pay)
                if premium >= 0:
                    continue
                # Bear Put: long_strike > short_strike, Bull Call: long_strike < short_strike
                if strategy == "Bear Put Spread" and long_strike <= short_strike:
                    continue
                if strategy == "Bull Call Spread" and long_strike >= short_strike:
                    continue

            width_val = abs(short_strike - long_strike)
            if strategy in {"Bull Put Spread", "Bear Call Spread"}:
                max_profit = premium
                max_loss = (width_val * short_mult) - premium
                breakeven = short_strike - (premium / short_mult) if strategy == "Bull Put Spread" else short_strike + (premium / short_mult)
            else:
                debit = -premium
                max_profit = (width_val * long_mult) - debit
                max_loss = debit
                breakeven = long_strike - (debit / long_mult) if strategy == "Bear Put Spread" else long_strike + (debit / long_mult)

            spread_pct = 0.0
            oi_min = min(float(short_row.get("oi") or 0.0), float(long_row.get("oi") or 0.0))
            vol_min = min(float(short_row.get("volume") or 0.0), float(long_row.get("volume") or 0.0))

            if max_margin > 0 and abs(max_loss) > max_margin:
                continue
            if max_spread_pct > 0 and spread_pct > max_spread_pct:
                continue
            if min_oi > 0 and oi_min < min_oi:
                continue
            if min_volume > 0 and vol_min < min_volume:
                continue

            sigma = float(short_row.get("iv") or 0.0)
            pnl = _simulate_pnl(
                spot,
                days_to_expiry,
                sigma,
                sim_cfg.rf,
                sim_cfg.n_paths,
                sim_cfg.jump_enabled,
                premium,
                opt_type,
                short_strike,
                opt_type,
                long_strike,
                rng,
                z_base,
                jump_mask,
                jump_z,
                short_mult,
                long_mult,
            )
            expected_pnl = float(np.mean(pnl))
            pop = float(np.mean(pnl > 0))
            losses = -pnl
            if losses.size:
                var = float(np.quantile(losses, sim_cfg.cvar_level))
                tail = losses[losses >= var]
                cvar = float(np.mean(tail)) if tail.size else 0.0
            else:
                cvar = 0.0
            tail_loss = abs(cvar)
            ev_over_cvar = expected_pnl / tail_loss if tail_loss else 0.0
            rr = expected_pnl / tail_loss if tail_loss else 0.0

            # Calculate net greeks
            # Try multiple possible column names for greeks
            short_delta = float(short_row.get("delta") or short_row.get("greeks.delta") or 0.0)
            short_gamma = float(short_row.get("gamma") or short_row.get("greeks.gamma") or 0.0)
            short_theta = float(short_row.get("theta") or short_row.get("greeks.theta") or 0.0)
            short_vega = float(short_row.get("vega") or short_row.get("greeks.vega") or 0.0)
            
            long_delta = float(long_row.get("delta") or long_row.get("greeks.delta") or 0.0)
            long_gamma = float(long_row.get("gamma") or long_row.get("greeks.gamma") or 0.0)
            long_theta = float(long_row.get("theta") or long_row.get("greeks.theta") or 0.0)
            long_vega = float(long_row.get("vega") or long_row.get("greeks.vega") or 0.0)
            
            net_delta = (long_delta * long_mult) - (short_delta * short_mult)
            net_gamma = (long_gamma * long_mult) - (short_gamma * short_mult)
            net_theta = (long_theta * long_mult) - (short_theta * short_mult)
            net_vega = (long_vega * long_mult) - (short_vega * short_mult)

            combos.append(
                {
                    "strategy": strategy,
                    "expiry": expiry.strftime("%Y-%m-%d"),
                    "short_strike": short_strike,
                    "long_strike": long_strike,
                    "short_price": short_price,
                    "long_price": long_price,
                    "width": width_val,
                    "premium": premium,
                    "max_profit": max_profit,
                    "max_loss": max_loss,
                    "tail_loss": tail_loss,
                    "rr": rr,
                    "breakeven": breakeven,
                    "pop": pop,
                    "expected_pnl": expected_pnl,
                    "cvar": cvar,
                    "ev_over_cvar": ev_over_cvar,
                    "oi_min": oi_min,
                    "vol_min": vol_min,
                    "spread_pct": spread_pct,
                    "net_delta": net_delta,
                    "net_gamma": net_gamma,
                    "net_theta": net_theta,
                    "net_vega": net_vega,
                }
            )

    return pd.DataFrame(combos)


def _preview_selected_legs(
    strategy: str,
    expiry: pd.Timestamp,
    spot: float,
    chain: pd.DataFrame,
    distances: List[int],
    widths: List[int],
) -> pd.DataFrame:
    spec = _strategy_spec(strategy)
    opt_type = spec["type"]
    chain = chain[(chain["option_type"] == opt_type) & (chain["expiry"] == expiry)].copy()
    if chain.empty:
        return pd.DataFrame()

    strikes = np.sort(chain["strike"].unique())
    chain_map = {row["strike"]: row for row in chain.to_dict(orient="records")}
    rows = []
    for dist in distances:
        short_target = spot - dist if opt_type == "PE" else spot + dist
        short_strike = _pick_strike(strikes, short_target, spec["short_dir"])
        if short_strike is None:
            continue
        for width in widths:
            long_target = short_strike + (width * spec["long_offset"])
            long_strike = _pick_strike(strikes, long_target, spec["long_dir"])
            if long_strike is None or long_strike == short_strike:
                continue
            short_row = chain_map.get(short_strike, {})
            long_row = chain_map.get(long_strike, {})
            rows.append(
                {
                    "short_strike": short_strike,
                    "long_strike": long_strike,
                    "short_price": short_row.get("price"),
                    "long_price": long_row.get("price"),
                    "short_oi": short_row.get("oi"),
                    "long_oi": long_row.get("oi"),
                    "short_volume": short_row.get("volume"),
                    "long_volume": long_row.get("volume"),
                }
            )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _build_trade_table(
    chain: pd.DataFrame,
    strategy: str,
    expiry: str,
    spot: float,
    distances: Tuple[int, ...],
    widths: Tuple[int, ...],
    pricing_mode: str,
    slippage_per_leg: float,
    brokerage_per_leg: float,
    n_paths: int,
    rf: float,
    jump_enabled: bool,
    cvar_level: float,
    max_margin: float,
    max_spread_pct: float,
    min_oi: float,
    min_volume: float,
    lot_size: float,
    short_lots: float,
    wing_lots: float,
) -> pd.DataFrame:
    if chain.empty:
        return pd.DataFrame()
    expiry_dt = pd.to_datetime(expiry, errors="coerce")
    if pd.isna(expiry_dt):
        return pd.DataFrame()
    pricing = PricingConfig(pricing_mode, slippage_per_leg, brokerage_per_leg)
    sim_cfg = SimConfig(n_paths, rf, jump_enabled, cvar_level)
    
    # Filter by expiry
    df = chain[chain["expiry"] == expiry_dt].copy()
    
    # Filter for latest data date if timestamp/date column exists
    date_cols = [col for col in df.columns if col.lower() in ['timestamp', 'date', 'datetime', 'last_updated']]
    if date_cols and not df.empty:
        date_col = date_cols[0]
        latest_date = df[date_col].max()
        df = df[df[date_col] == latest_date]
    
    return _compute_spread_metrics(
        strategy,
        expiry_dt,
        spot,
        df,
        list(distances),
        list(widths),
        pricing,
        sim_cfg,
        max_margin,
        max_spread_pct,
        min_oi,
        min_volume,
        lot_size,
        short_lots,
        wing_lots,
    )


def render_trade_selector_tab() -> None:
    st.markdown("## Trade Selector")
    options_df = st.session_state.get("options_df_cache", pd.DataFrame())
    chain = _normalize_chain(options_df)
    if chain.empty:
        st.warning("Options cache is empty. Load data from the Derivatives Data tab.")
        return

    underlyings = sorted(chain["underlying"].dropna().unique().tolist())
    with st.sidebar.form("trade_selector_form"):
        underlying = st.selectbox("Underlying", underlyings)
        chain = chain[chain["underlying"] == underlying]

        spot_default = float(chain["spot"].dropna().median()) if not chain["spot"].dropna().empty else 0.0
        spot = st.number_input("Current spot", min_value=0.0, value=spot_default, step=10.0)

        expiries = sorted(chain["expiry"].dropna().unique())
        today = datetime.now().date()
        expiries = [e for e in expiries if e.date() >= today]
        next_expiries = expiries[:3] if len(expiries) >= 3 else expiries
        expiry_choice = st.selectbox(
            "Expiry",
            [e.strftime("%Y-%m-%d") for e in next_expiries],
        )

        strategy = st.selectbox(
            "Strategy",
            [
                "Bull Spread",
                "Bear Spread",
            ],
        )

        st.markdown("### Short Strike Scan")
        short_cols = st.columns(3)
        short_min = short_cols[0].number_input("Min distance", value=-700, step=50, key="short_min")
        short_max = short_cols[1].number_input("Max distance", value=700, step=50, key="short_max")
        short_step = short_cols[2].number_input("Step", value=100, step=50, key="short_step", min_value=50)
        
        if short_step > 0:
            distances = tuple(range(int(short_min), int(short_max) + 1, int(short_step)))
        else:
            distances = (int(short_min),)

        st.markdown("### Wing Width Scan")
        wing_cols = st.columns(3)
        wing_min = wing_cols[0].number_input("Min width", value=-1000, step=50, key="wing_min")
        wing_max = wing_cols[1].number_input("Max width", value=1000, step=50, key="wing_max")
        wing_step = wing_cols[2].number_input("Step", value=100, step=50, key="wing_step", min_value=50)
        
        if wing_step > 0:
            widths = tuple(range(int(wing_min), int(wing_max) + 1, int(wing_step)))
        else:
            widths = (int(wing_min),)
        
        st.caption(f"Total combinations: {len(distances)} strikes × {len(widths)} widths = {len(distances) * len(widths)} trades")

        lot_size = st.number_input("Lot size", value=50, step=1)
        short_lots = st.number_input("Short leg lots", value=1, step=1)
        wing_lots = st.number_input("Wing leg lots", value=1, step=1)

        pricing_mode = st.selectbox("Pricing mode", ["MID", "CONSERVATIVE"], disabled=True)
        slippage_per_leg = st.number_input("Slippage per leg", value=0.0, step=1.0)
        brokerage_per_leg = st.number_input("Brokerage per leg", value=0.0, step=1.0)

        st.markdown("### Forward Simulation")
        n_paths = st.number_input("Paths", value=5000, step=500)
        rf = st.number_input("Risk-free drift", value=0.0, step=0.005, format="%.3f")
        jump_enabled = st.toggle("Enable jumps", value=False)
        cvar_level = st.selectbox("CVaR level", [0.95, 0.99], index=1)

        st.markdown("### Constraints")
        max_margin = st.number_input("Max margin (proxy)", value=0.0, step=10000.0)
        max_spread_pct = st.number_input("Max bid/ask spread %", value=0.0, step=0.01)
        min_oi = st.number_input("Min OI", value=0.0, step=10.0)
        min_volume = st.number_input("Min Volume", value=0.0, step=10.0)
        min_pop = st.number_input("Min PoP %", value=0.0, min_value=0.0, max_value=100.0, step=5.0, help="Minimum probability of profit")
        positive_ev_only = st.toggle("Expected PnL > 0", value=True)

        submitted = st.form_submit_button("Run Trade Selector")

    if submitted:
        # Generate spreads for both PE and CE
        strategies_to_run = []
        if strategy == "Bull Spread":
            strategies_to_run = ["Bull Put Spread", "Bull Call Spread"]
        elif strategy == "Bear Spread":
            strategies_to_run = ["Bear Put Spread", "Bear Call Spread"]
        
        all_results = []
        for strat in strategies_to_run:
            results = _build_trade_table(
                chain,
                strat,
                expiry_choice,
                spot,
                distances,
                widths,
                pricing_mode,
                slippage_per_leg,
                brokerage_per_leg,
                int(n_paths),
                float(rf),
                jump_enabled,
                float(cvar_level),
                float(max_margin),
                float(max_spread_pct),
                float(min_oi),
                float(min_volume),
                float(lot_size),
                float(short_lots),
                float(wing_lots),
            )
            if not results.empty:
                all_results.append(results)
        
        if all_results:
            results = pd.concat(all_results, ignore_index=True)
        else:
            results = pd.DataFrame()
        
        st.session_state["trade_selector_results"] = results
        st.session_state["trade_selector_positive_only"] = positive_ev_only
        st.session_state["trade_selector_min_pop"] = min_pop
    else:
        results = st.session_state.get("trade_selector_results", pd.DataFrame())
        positive_ev_only = st.session_state.get("trade_selector_positive_only", True)
        min_pop = st.session_state.get("trade_selector_min_pop", 0.0)

    with st.expander("Debug: Selected legs preview", expanded=False):
        expiry_dt = pd.to_datetime(expiry_choice, errors="coerce")
        expiry_chain = chain[chain["expiry"] == expiry_dt].copy()
        st.caption(
            f"Rows for expiry: {len(expiry_chain)} | "
            f"Price>0: {(expiry_chain['price'] > 0).sum()} | "
            f"Close>0: {(expiry_chain['close'] > 0).sum()}"
        )
        if not expiry_chain.empty:
            st.dataframe(
                expiry_chain[
                    [
                        "option_type",
                        "strike",
                        "price",
                        "close",
                        "ltp",
                        "oi",
                        "volume",
                    ]
                ].head(10),
                use_container_width=True,
                hide_index=True,
            )
        preview_df = _preview_selected_legs(
            strategy,
            expiry_dt,
            spot,
            chain,
            list(distances),
            list(widths),
        )
        if preview_df.empty:
            st.info("No candidate legs found for the current inputs.")
        else:
            st.dataframe(preview_df, use_container_width=True, hide_index=True)

    if results.empty:
        st.info("No trades match the current filters.")
        return

    # Apply filters
    if positive_ev_only:
        results = results[results["expected_pnl"] > 0]
    
    if min_pop > 0:
        results = results[results["pop"] >= (min_pop / 100.0)]
    
    if results.empty:
        st.info("No trades match the current filters.")
        return

    results = results.sort_values(["ev_over_cvar", "expected_pnl"], ascending=[False, False])
    results = results.reset_index(drop=True)
    results["row_id"] = results.index.astype(int)

    st.markdown("### Candidate Spreads")
    
    # Build column list dynamically based on what's available
    base_cols = [
        "strategy",
        "expiry",
        "short_strike",
    ]
    
    # Add price columns if available
    if "short_price" in results.columns:
        base_cols.append("short_price")
    
    base_cols.append("long_strike")
    
    if "long_price" in results.columns:
        base_cols.append("long_price")
    
    base_cols.extend([
        "width",
        "premium",
        "max_profit",
        "max_loss",
        "tail_loss",
        "rr",
        "breakeven",
        "pop",
        "expected_pnl",
    ])
    
    greek_cols = ["net_delta", "net_gamma", "net_theta", "net_vega"]
    available_greek_cols = [col for col in greek_cols if col in results.columns]
    
    other_cols = [
        "cvar",
        "ev_over_cvar",
        "oi_min",
        "vol_min",
        "spread_pct",
    ]
    
    display_cols = base_cols + available_greek_cols + other_cols
    display_df = results[display_cols].copy()
    
    # Format currency columns
    for col in ["premium", "max_profit", "max_loss", "tail_loss", "cvar"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "—")
    
    if "expected_pnl" in display_df.columns:
        display_df["expected_pnl"] = display_df["expected_pnl"].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "—")
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Tail Loss vs Mean PnL")
    chart_df = results.copy()
    
    # Exclude combinations where pop = 1 (100% probability)
    chart_df = chart_df[chart_df["pop"] < 1.0]
    
    if chart_df.empty:
        st.warning("No trades to display after filtering out 100% probability trades.")
    else:
        tooltip_cols = [
            alt.Tooltip("strategy:N"),
            alt.Tooltip("expiry:N"),
            alt.Tooltip("short_strike:Q"),
            alt.Tooltip("long_strike:Q"),
            alt.Tooltip("premium:Q", format=",.2f"),
            alt.Tooltip("rr:Q", format=".2f"),
            alt.Tooltip("pop:Q", format=".2%"),
            alt.Tooltip("ev_over_cvar:Q", format=".2f"),
            alt.Tooltip("expected_pnl:Q", format=",.0f"),
            alt.Tooltip("tail_loss:Q", format=",.0f"),
        ]

        base = (
            alt.Chart(chart_df)
            .mark_circle(opacity=0.85)
            .encode(
                x=alt.X("expected_pnl:Q", title="Expected PnL"),
                y=alt.Y("tail_loss:Q", title="Tail Loss (CVaR)", scale=alt.Scale(domain=[0, 200000]), axis=alt.Axis(format="~s")),
                size=alt.Size("pop:Q", scale=alt.Scale(range=[50, 500]), legend=alt.Legend(title="PoP")),
                color=alt.Color("ev_over_cvar:Q", legend=alt.Legend(title="EV/CVaR"), scale=alt.Scale(domain=[0, 1], scheme="redyellowgreen")),
                tooltip=tooltip_cols,
            )
        )

        rr_lines = []
        rr_levels = [0.15, 0.30, 0.50]
        x_max = max(1.0, float(chart_df["expected_pnl"].max()))
        x_vals = np.linspace(0, x_max, 60)
        for rr in rr_levels:
            y_vals = x_vals / rr
            rr_lines.append(pd.DataFrame({"x": x_vals, "y": y_vals, "rr": f"RR {rr:.2f}"}))
        rr_df = pd.concat(rr_lines, ignore_index=True)
        rr_chart = (
            alt.Chart(rr_df)
            .mark_line(strokeDash=[4, 4])
            .encode(
                x="x:Q",
                y="y:Q",
                color=alt.Color("rr:N", legend=None),
            )
        )

        zone_defs = [
            (0.0, 0.15, "Zone A", "#3A1F24"),
            (0.15, 0.30, "Zone B", "#243622"),
            (0.30, 0.40, "Zone C", "#3B2F1A"),
            (0.40, None, "Zone D", "#2C2F3D"),
        ]
        y_min = float(chart_df["tail_loss"].min())
        y_max = float(chart_df["tail_loss"].max())
        zone_rows = []
        label_rows = []
        for low_rr, high_rr, label, color in zone_defs:
            y_low = x_vals / low_rr if low_rr else np.zeros_like(x_vals)
            if high_rr is None:
                y_high = np.full_like(x_vals, y_max * 1.2)
            else:
                y_high = x_vals / high_rr
            zone_rows.append(
                pd.DataFrame(
                    {
                        "x": x_vals,
                        "y1": y_low,
                        "y2": y_high,
                        "zone": label,
                        "color": color,
                    }
                )
            )
            label_x = x_max * 0.7
            label_y = float(np.mean([(label_x / (high_rr or 0.001)), (label_x / (low_rr or 0.001))]))
            label_rows.append({"x": label_x, "y": label_y, "label": label, "color": color})
        zone_df = pd.concat(zone_rows, ignore_index=True)
        zone_chart = (
            alt.Chart(zone_df)
            .mark_area(opacity=0.10)
            .encode(
                x="x:Q",
                y="y1:Q",
                y2="y2:Q",
                color=alt.Color("color:N", scale=None, legend=None),
            )
        )
        zone_labels = (
            alt.Chart(pd.DataFrame(label_rows))
            .mark_text(align="left", dx=6, color="#BFC7D5", fontSize=10)
            .encode(x="x:Q", y="y:Q", text="label:N")
        )

        chart = alt.layer(zone_chart, rr_chart, base, zone_labels).properties(
            height=420,
            padding={"left": 10, "top": 10, "right": 10, "bottom": 10}
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown("### Top Candidates")
    top_k = st.number_input("Top K", value=10, min_value=1, step=1)
    top_df = results.head(int(top_k)).copy()
    
    # Format top candidates table
    for col in ["premium", "max_profit", "max_loss", "tail_loss", "cvar"]:
        if col in top_df.columns:
            top_df[col] = top_df[col].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "—")
    
    if "expected_pnl" in top_df.columns:
        top_df["expected_pnl"] = top_df["expected_pnl"].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "—")
    
    st.dataframe(top_df, use_container_width=True, hide_index=True)

    selected_ids = st.multiselect("Select rows to export", options=results["row_id"].tolist())
    export_df = results[results["row_id"].isin(selected_ids)] if selected_ids else top_df
    csv_data = export_df.to_csv(index=False)
    st.download_button("Download CSV", csv_data, file_name="trade_selector.csv")
    st.download_button("Download JSON", export_df.to_json(orient="records"), file_name="trade_selector.json")
