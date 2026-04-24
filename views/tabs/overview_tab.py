"""
🧭 Overview Tab - The Ultimate Decision Summary
Provides one-glance decision signals on market conditions, portfolio health, and immediate actions.
"""

import json
import pandas as pd
import plotly.graph_objects as go
from typing import Any, Dict, List, Optional

# Import utility functions
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.utils.portfolio_metrics import calculate_portfolio_greeks, calculate_market_regime
from scripts.utils.risk_calculations import calculate_var
from scripts.utils.formatters import format_inr
from scripts.utils.optional_streamlit import st
from scripts.utils.greeks import calculate_implied_volatility
from pathlib import Path
from datetime import datetime


# Cache directory
CACHE_DIR = Path(ROOT) / "database" / "derivatives_cache"
EQUITY_HISTORY_CACHE_DIR = CACHE_DIR / "equity_history_cache"
NIFTY50_WEIGHTS_FILE = Path(ROOT) / "data" / "nifty50_weights.json"


def load_from_cache(data_type: str) -> pd.DataFrame:
    """Load data from most recent cache file."""
    cache_file = CACHE_DIR / f"{data_type}_{datetime.now().strftime('%Y-%m-%d')}.csv"
    if cache_file.exists():
        df = pd.read_csv(cache_file)
        return df
    else:
        return pd.DataFrame()


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
        if pd.isna(out):
            return None
        return out
    except Exception:
        return None


def _direction(curr: Any, prev: Any, tolerance: float = 1e-9) -> str:
    curr_f = _safe_float(curr)
    prev_f = _safe_float(prev)
    if curr_f is None or prev_f is None:
        return "n/a"
    diff = curr_f - prev_f
    if abs(diff) <= tolerance:
        return "flat"
    return "up" if diff > 0 else "down"


def _direction_badge(direction: str) -> str:
    mapping = {
        "up": "↑ Up",
        "down": "↓ Down",
        "flat": "→ Flat",
        "strong": "Strong",
        "weak": "Weak",
        "n/a": "N/A",
    }
    return mapping.get(direction, str(direction))


def _load_nifty50_weights() -> List[Dict[str, Any]]:
    if not NIFTY50_WEIGHTS_FILE.exists():
        return []
    try:
        payload = json.loads(NIFTY50_WEIGHTS_FILE.read_text())
    except Exception:
        return []
    if not isinstance(payload, list):
        return []

    rows: List[Dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").upper().strip()
        weight_pct = _safe_float(row.get("weight_pct"))
        if not symbol or weight_pct is None or weight_pct <= 0:
            continue
        rows.append({"symbol": symbol, "weight": weight_pct / 100.0})
    return rows


def _equity_history_path(symbol: str) -> Optional[Path]:
    candidates = [symbol]
    if symbol == "MM":
        candidates.append("M&M")
    elif symbol == "M&M":
        candidates.append("MM")
    for candidate in candidates:
        path = EQUITY_HISTORY_CACHE_DIR / f"{candidate}.csv"
        if path.exists():
            return path
    return None


def _build_market_breadth_map() -> Dict[str, Dict[str, float]]:
    weights = _load_nifty50_weights()
    if not weights:
        return {}

    total_weight = sum(float(item["weight"]) for item in weights)
    breadth_by_date: Dict[str, Dict[str, float]] = {}

    for item in weights:
        symbol = str(item["symbol"])
        weight = float(item["weight"])
        path = _equity_history_path(symbol)
        if path is None:
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty or "date" not in df.columns or "close" not in df.columns:
            continue
        work = df.copy()
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work["close"] = pd.to_numeric(work["close"], errors="coerce")
        work["volume"] = pd.to_numeric(work.get("volume"), errors="coerce")
        work = work.dropna(subset=["date", "close"]).sort_values("date")
        work["ret"] = work["close"].pct_change()

        for _, row in work.iterrows():
            ret = _safe_float(row.get("ret"))
            if ret is None:
                continue
            date_key = pd.to_datetime(row["date"]).date().isoformat()
            bucket = breadth_by_date.setdefault(
                date_key,
                {
                    "up_weight": 0.0,
                    "down_weight": 0.0,
                    "up_count": 0.0,
                    "down_count": 0.0,
                    "weighted_volume": 0.0,
                    "total_weight": total_weight,
                },
            )
            if ret > 0:
                bucket["up_weight"] += weight
                bucket["up_count"] += 1
            elif ret < 0:
                bucket["down_weight"] += weight
                bucket["down_count"] += 1

            volume = _safe_float(row.get("volume"))
            if volume is not None and volume > 0:
                bucket["weighted_volume"] += volume * weight

    return breadth_by_date


def _build_futures_snapshot(futures_df: pd.DataFrame) -> pd.DataFrame:
    if futures_df.empty or "date" not in futures_df.columns:
        return pd.DataFrame(columns=["date", "front_fut_close", "front_fut_oi"])

    work = futures_df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["expiry"] = pd.to_datetime(work.get("expiry"), errors="coerce")
    work["open_interest"] = pd.to_numeric(work.get("open_interest"), errors="coerce")
    work["open_int"] = pd.to_numeric(work.get("open_int"), errors="coerce")
    work["close"] = pd.to_numeric(work.get("close"), errors="coerce")
    work = work.dropna(subset=["date"]).sort_values(["date", "expiry"])
    work["oi_value"] = work["open_interest"].fillna(work["open_int"])
    front = work.groupby("date", as_index=False).first()
    front["date"] = front["date"].dt.date.astype(str)
    return front.rename(columns={"close": "front_fut_close", "oi_value": "front_fut_oi"})[
        ["date", "front_fut_close", "front_fut_oi"]
    ]


def _build_vix_proxy_map(options_df: pd.DataFrame, max_days: int = 45) -> Dict[str, float]:
    if options_df.empty or "date" not in options_df.columns:
        return {}

    work = options_df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["strike_price"] = pd.to_numeric(work.get("strike_price"), errors="coerce")
    work["ltp"] = pd.to_numeric(work.get("ltp"), errors="coerce")
    work["underlying_value"] = pd.to_numeric(work.get("underlying_value"), errors="coerce")
    work = work.dropna(subset=["date", "strike_price", "ltp", "underlying_value"])
    if work.empty:
        return {}

    out: Dict[str, float] = {}
    unique_dates = sorted(work["date"].dropna().unique())[-max_days:]
    for day in unique_dates:
        day_options = work[work["date"] == day].copy()
        if day_options.empty:
            continue
        spot = _safe_float(day_options["underlying_value"].iloc[0])
        if spot is None or spot <= 0:
            continue
        atm = round(spot / 50.0) * 50.0
        near_atm = day_options[
            (day_options["strike_price"] >= atm - 100.0) &
            (day_options["strike_price"] <= atm + 100.0)
        ]
        ivs: List[float] = []
        for _, row in near_atm.iterrows():
            try:
                iv = calculate_implied_volatility(
                    row["ltp"],
                    spot,
                    row["strike_price"],
                    0.027,
                    row["option_type"],
                )
                if iv and iv > 0:
                    ivs.append(float(iv) * 100.0)
            except Exception:
                continue
        if ivs:
            out[pd.to_datetime(day).date().isoformat()] = float(sum(ivs) / len(ivs))
    return out


def _classify_daily_market_row(row: Dict[str, Any]) -> str:
    price_dir = str(row.get("price_dir") or "n/a")
    oi_dir = str(row.get("oi_dir") or "n/a")
    vix_dir = str(row.get("vix_dir") or "n/a")
    breadth_signal = str(row.get("breadth_signal") or "weak")
    usdinr_dir = str(row.get("usdinr_dir") or "n/a")

    if vix_dir == "up" and (price_dir == "n/a" or oi_dir == "n/a" or breadth_signal == "weak"):
        return "🟣 Uncertain / Trap"
    if price_dir == "up" and oi_dir == "up" and vix_dir == "down" and breadth_signal == "strong":
        return "🟢 Strong Bullish"
    if price_dir == "up" and oi_dir == "down" and vix_dir in {"up", "flat", "n/a"}:
        return "🟡 Weak / Fake Bullish"
    if price_dir == "down" and oi_dir == "up" and vix_dir == "up":
        return "🔴 Strong Bearish"
    if price_dir == "down" and oi_dir == "down":
        return "⚪ Mild Bearish"
    if usdinr_dir == "up" and price_dir == "up" and breadth_signal == "weak":
        return "🟣 Uncertain / Trap"
    if price_dir == "up" and oi_dir == "up":
        return "🟢 Bullish Bias"
    if price_dir == "down" and oi_dir == "up":
        return "🔴 Bearish Bias"
    return "🟣 Mixed / Unclear"


def build_daily_market_tracker(
    nifty_df: pd.DataFrame,
    futures_df: pd.DataFrame,
    options_df: pd.DataFrame,
    history_days: int = 20,
) -> pd.DataFrame:
    if nifty_df.empty or futures_df.empty:
        return pd.DataFrame()

    spot = nifty_df.copy()
    spot["date"] = pd.to_datetime(spot["date"], errors="coerce")
    spot["close"] = pd.to_numeric(spot.get("close"), errors="coerce")
    spot = spot.dropna(subset=["date", "close"]).sort_values("date")
    spot["date"] = spot["date"].dt.date.astype(str)
    spot = spot[["date", "close"]].rename(columns={"close": "nifty_close"})
    spot["prev_nifty_close"] = spot["nifty_close"].shift(1)
    spot["price_dir"] = [
        _direction(curr, prev) for curr, prev in zip(spot["nifty_close"], spot["prev_nifty_close"])
    ]

    futures = _build_futures_snapshot(futures_df)
    if futures.empty:
        return pd.DataFrame()
    futures = futures.sort_values("date")
    futures["prev_front_fut_oi"] = futures["front_fut_oi"].shift(1)
    futures["oi_dir"] = [
        _direction(curr, prev) for curr, prev in zip(futures["front_fut_oi"], futures["prev_front_fut_oi"])
    ]

    merged = spot.merge(futures, on="date", how="inner").sort_values("date")
    if merged.empty:
        return merged

    breadth_map = _build_market_breadth_map()
    vix_map = _build_vix_proxy_map(options_df)

    vix_series = []
    for date_key in merged["date"]:
        vix_series.append(vix_map.get(str(date_key)))
    merged["vix_proxy"] = vix_series
    merged["prev_vix_proxy"] = merged["vix_proxy"].shift(1)
    merged["vix_dir"] = [
        _direction(curr, prev) for curr, prev in zip(merged["vix_proxy"], merged["prev_vix_proxy"])
    ]

    breadth_signal = []
    breadth_detail = []
    weighted_volume = []
    for _, row in merged.iterrows():
        date_key = str(row["date"])
        price_dir = str(row.get("price_dir") or "n/a")
        stats = breadth_map.get(date_key) or {}
        up_weight = float(stats.get("up_weight") or 0.0)
        down_weight = float(stats.get("down_weight") or 0.0)
        total_weight = float(stats.get("total_weight") or 1.0)
        if price_dir == "up":
            strength = up_weight / total_weight if total_weight > 0 else 0.0
        elif price_dir == "down":
            strength = down_weight / total_weight if total_weight > 0 else 0.0
        else:
            strength = max(up_weight, down_weight) / total_weight if total_weight > 0 else 0.0
        breadth_signal.append("strong" if strength >= 0.55 else "weak")
        breadth_detail.append(f"{strength * 100:.0f}%")
        weighted_volume.append(float(stats.get("weighted_volume") or 0.0))

    merged["breadth_signal"] = breadth_signal
    merged["breadth_strength"] = breadth_detail
    merged["weighted_constituent_volume"] = weighted_volume
    merged["usdinr_dir"] = "n/a"

    core_logic = []
    for _, row in merged.iterrows():
        price_dir = str(row.get("price_dir") or "n/a")
        oi_dir = str(row.get("oi_dir") or "n/a")
        if price_dir == "up" and oi_dir == "up":
            core_logic.append("Long build-up")
        elif price_dir == "up" and oi_dir == "down":
            core_logic.append("Short covering")
        elif price_dir == "down" and oi_dir == "up":
            core_logic.append("Short build-up")
        elif price_dir == "down" and oi_dir == "down":
            core_logic.append("Long unwinding")
        else:
            core_logic.append("Mixed")
    merged["core_logic"] = core_logic

    merged["daily_classification"] = [
        _classify_daily_market_row(row) for row in merged.to_dict(orient="records")
    ]
    merged["nifty_signal"] = merged["price_dir"].map(_direction_badge)
    merged["futures_oi_signal"] = merged["oi_dir"].map(_direction_badge)
    merged["vix_signal"] = merged["vix_dir"].map(_direction_badge)
    merged["breadth_label"] = [
        f"{_direction_badge(signal)} ({strength})"
        for signal, strength in zip(merged["breadth_signal"], merged["breadth_strength"])
    ]
    merged["usdinr_signal"] = "N/A"
    merged["weighted_constituent_volume_fmt"] = [
        f"{float(val):,.0f}" if _safe_float(val) is not None else "—"
        for val in merged["weighted_constituent_volume"]
    ]

    return merged.tail(history_days).reset_index(drop=True)



def get_market_signal(regime_data: Dict) -> tuple[str, str, str]:
    """Generate market signal and recommendation based on regime data.
    
    Returns: (signal_emoji, signal_text, recommendation)
    """
    if not regime_data:
        return "⚪", "Unknown", "Insufficient data for market analysis"
    
    iv_rank = regime_data.get("iv_rank", 50)
    vrp = regime_data.get("vrp", 0)
    term_structure = regime_data.get("term_structure", 0)
    skew = regime_data.get("skew", 0)
    pcr_oi = regime_data.get("pcr_oi", 1.0)
    
    # Classify market condition
    signal_parts = []
    
    # Volatility level
    if iv_rank > 70:
        signal_parts.append("High IV")
        vol_bias = "expensive"
    elif iv_rank < 30:
        signal_parts.append("Low IV")
        vol_bias = "cheap"
    else:
        signal_parts.append("Moderate IV")
        vol_bias = "fair"
    
    # VRP direction
    if vrp > 0.05:
        signal_parts.append("IV > RV")
    elif vrp < -0.05:
        signal_parts.append("RV > IV")
    
    # Skew sentiment
    if skew > 0.01:
        signal_parts.append("Put Skew")
        sentiment = "Fear"
    elif skew < -0.01:
        signal_parts.append("Call Skew")
        sentiment = "Greed"
    else:
        sentiment = "Neutral"
    
    signal_text = ", ".join(signal_parts)
    
    # Signal emoji
    if iv_rank > 70 and vrp > 0.05:
        emoji = "🔴"  # Sell vol regime
    elif iv_rank < 30 and vrp < -0.05:
        emoji = "🟢"  # Buy vol regime
    elif abs(vrp) < 0.03:
        emoji = "🟡"  # Neutral
    else:
        emoji = "🟠"  # Mixed
    
    # Generate recommendation
    recommendations = []
    
    if iv_rank > 70:
        recommendations.append(f"Volatility {vol_bias} (IV Rank {iv_rank:.0f})")
        if vrp > 0.05:
            recommendations.append("Favor premium selling strategies")
    elif iv_rank < 30:
        recommendations.append(f"Volatility {vol_bias} (IV Rank {iv_rank:.0f})")
        if vrp < -0.05:
            recommendations.append("Favor premium buying / debit spreads")
    
    if term_structure < -0.02:
        recommendations.append("⚠️ Backwardation detected — avoid near-term short gamma")
    elif term_structure > 0.03:
        recommendations.append("Normal term structure — safe to sell near expiries")
    
    if abs(skew) > 0.01:
        if skew > 0:
            recommendations.append(f"Puts expensive by {skew*100:.1f}% — consider selling put spreads")
        else:
            recommendations.append(f"Calls expensive by {abs(skew)*100:.1f}% — consider selling call spreads")
    
    if pcr_oi > 1.2:
        recommendations.append("PCR suggests bearish positioning")
    elif pcr_oi < 0.8:
        recommendations.append("PCR suggests bullish positioning")
    
    recommendation = " | ".join(recommendations) if recommendations else "Market conditions neutral — balanced approach"
    
    return emoji, signal_text, recommendation


def get_portfolio_health_status(metrics: Dict, var_pct: float, margin_pct: float) -> tuple[str, str]:
    """Determine portfolio health status.
    
    Returns: (status_emoji, status_text)
    """
    risk_flags = 0
    
    # Check VaR
    if var_pct > 5:
        risk_flags += 2
    elif var_pct > 3:
        risk_flags += 1
    
    # Check margin
    if margin_pct > 85:
        risk_flags += 2
    elif margin_pct > 70:
        risk_flags += 1
    
    # Check theta efficiency (if available)
    theta_eff = metrics.get("theta_efficiency", 0)
    if abs(theta_eff) > 100:  # Too much deviation
        risk_flags += 1
    
    # Determine status
    if risk_flags == 0:
        return "🟢", "Healthy"
    elif risk_flags <= 2:
        return "🟡", "Moderate Risk"
    else:
        return "🔴", "High Risk"


def get_alignment_status(market_iv_rank: float, portfolio_vega: float,
                         market_pcr: float, portfolio_delta: float,
                         term_structure: float, near_expiry_positions: int) -> List[Dict]:
    """Check portfolio alignment with market conditions.
    
    Returns list of alignment checks with status and recommendations.
    """
    alignments = []
    
    # 1. Volatility Alignment
    market_vol_bias = "High IV" if market_iv_rank > 70 else ("Low IV" if market_iv_rank < 30 else "Moderate IV")
    portfolio_vol_stance = "Short Vega" if portfolio_vega < 0 else ("Long Vega" if portfolio_vega > 0 else "Neutral")
    
    vol_aligned = (market_iv_rank > 70 and portfolio_vega < 0) or (market_iv_rank < 30 and portfolio_vega > 0)
    vol_misaligned = (market_iv_rank > 70 and portfolio_vega > 0) or (market_iv_rank < 30 and portfolio_vega < 0)
    
    if vol_aligned:
        vol_status = "✅ Aligned"
        vol_action = "Maintain current volatility exposure"
    elif vol_misaligned:
        vol_status = "❌ Misaligned"
        if market_iv_rank > 70:
            vol_action = "Reduce long vega exposure; consider credit spreads"
        else:
            vol_action = "Reduce short vega exposure; consider debit spreads"
    else:
        vol_status = "⚪ Neutral"
        vol_action = "Monitor for directional opportunity"
    
    alignments.append({
        "Category": "Volatility",
        "Market": market_vol_bias,
        "Portfolio": portfolio_vol_stance,
        "Status": vol_status,
        "Action": vol_action
    })
    
    # 2. Directional Alignment
    market_dir_bias = "Bearish" if market_pcr > 1.1 else ("Bullish" if market_pcr < 0.9 else "Neutral")
    portfolio_dir_stance = "Long Delta" if portfolio_delta > 10 else ("Short Delta" if portfolio_delta < -10 else "Neutral")
    
    dir_aligned = (market_pcr > 1.1 and portfolio_delta < 0) or (market_pcr < 0.9 and portfolio_delta > 0)
    dir_misaligned = (market_pcr > 1.1 and portfolio_delta > 10) or (market_pcr < 0.9 and portfolio_delta < -10)
    
    if dir_aligned:
        dir_status = "✅ Aligned"
        dir_action = "Hold directional bias"
    elif dir_misaligned:
        dir_status = "❌ Misaligned"
        if market_pcr > 1.1:
            dir_action = f"Reduce long delta by ~₹{abs(portfolio_delta*1000):.0f}; add bearish positions"
        else:
            dir_action = f"Reduce short delta by ~₹{abs(portfolio_delta*1000):.0f}; add bullish positions"
    else:
        dir_status = "✅ Aligned"
        dir_action = "Neutral positioning appropriate"
    
    alignments.append({
        "Category": "Direction",
        "Market": market_dir_bias,
        "Portfolio": portfolio_dir_stance,
        "Status": dir_status,
        "Action": dir_action
    })
    
    # 3. Term Structure Alignment
    market_term = "Backwardation" if term_structure < -0.02 else ("Contango" if term_structure > 0.02 else "Flat")
    portfolio_term_risk = "Heavy Near-Expiry" if near_expiry_positions > 5 else ("Balanced" if near_expiry_positions > 0 else "Far-Dated")
    
    term_risk = term_structure < -0.02 and near_expiry_positions > 3
    
    if term_risk:
        term_status = "⚠️ Risk"
        term_action = f"Roll {near_expiry_positions} near-expiry positions to next month"
    elif term_structure < -0.02:
        term_status = "⚪ Caution"
        term_action = "Avoid adding near-term short gamma positions"
    else:
        term_status = "✅ Aligned"
        term_action = "Term structure supports current positioning"
    
    alignments.append({
        "Category": "Term Structure",
        "Market": market_term,
        "Portfolio": portfolio_term_risk,
        "Status": term_status,
        "Action": term_action
    })
    
    # 4. Margin Status
    # (Would need actual margin data - placeholder)
    alignments.append({
        "Category": "Margin",
        "Market": "N/A",
        "Portfolio": "Monitor",
        "Status": "✅ OK",
        "Action": "Maintain buffer"
    })
    
    return alignments


def _render_daily_market_tracker(nifty_df: pd.DataFrame, options_df: pd.DataFrame) -> None:
    futures_df = st.session_state.get("nifty_futures_df_cache")
    if not isinstance(futures_df, pd.DataFrame) or futures_df.empty:
        futures_df = load_from_cache("nifty_futures")

    tracker_df = build_daily_market_tracker(nifty_df=nifty_df, futures_df=futures_df, options_df=options_df, history_days=20)

    st.markdown("---")
    st.subheader("🧭 Daily Market Tracker")
    st.caption("Tracks NIFTY, front-month futures OI, VIX proxy, constituent breadth, and USDINR feed status.")

    if tracker_df.empty:
        st.warning("⚠️ Daily tracker unavailable. NIFTY and futures cache are required.")
        return

    latest = tracker_df.iloc[-1]
    top_cols = st.columns(5)
    top_cols[0].metric("NIFTY", f"₹{float(latest['nifty_close']):,.0f}", latest["nifty_signal"])
    top_cols[1].metric("Futures OI", f"{float(latest['front_fut_oi']):,.0f}", latest["futures_oi_signal"])
    top_cols[2].metric(
        "VIX",
        f"{float(latest['vix_proxy']):.1f}" if _safe_float(latest.get("vix_proxy")) is not None else "N/A",
        latest["vix_signal"],
    )
    top_cols[3].metric("Breadth", latest["breadth_label"], latest["core_logic"])
    top_cols[4].metric("USDINR", latest["usdinr_signal"], "Feed missing")

    st.info(
        f"**Latest classification:** {latest['daily_classification']} | "
        f"Weighted constituent volume: {latest['weighted_constituent_volume_fmt']}"
    )

    display_df = tracker_df.copy()
    display_df = display_df.rename(
        columns={
            "date": "Date",
            "nifty_signal": "NIFTY",
            "futures_oi_signal": "Futures OI",
            "vix_signal": "VIX",
            "breadth_label": "Breadth",
            "usdinr_signal": "USDINR",
            "core_logic": "Core Logic",
            "daily_classification": "Daily Classification",
            "weighted_constituent_volume_fmt": "Weighted Volume",
        }
    )
    st.dataframe(
        display_df[
            [
                "Date",
                "NIFTY",
                "Futures OI",
                "VIX",
                "Breadth",
                "USDINR",
                "Core Logic",
                "Daily Classification",
                "Weighted Volume",
            ]
        ].sort_values("Date", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("`VIX` is derived from near-ATM options IV until a dedicated India VIX feed is added. `USDINR` is shown as unavailable because no local feed exists.")


def render_overview_tab(options_df: pd.DataFrame = None, nifty_df: pd.DataFrame = None):
    """Render the Ultimate Overview Tab with key decision metrics."""
    
    st.header("🧭 Portfolio Overview — Decision View")

    if st.sidebar.button("🔄 Reload Data", key="overview_reload_data", help="Load latest derivatives data from cache"):
        with st.spinner("Loading data from cache..."):
            # Load options data (combine CE and PE)
            df_ce = load_from_cache("nifty_options_ce")
            df_pe = load_from_cache("nifty_options_pe")
            
            if not df_ce.empty and not df_pe.empty:
                options_df = pd.concat([df_ce, df_pe], ignore_index=True)
                st.session_state["options_df_cache"] = options_df
                st.success(f"✅ Loaded {len(options_df)} options records")
            elif not df_ce.empty:
                options_df = df_ce
                st.session_state["options_df_cache"] = options_df
                st.success(f"✅ Loaded {len(options_df)} CE records")
            elif not df_pe.empty:
                options_df = df_pe
                st.session_state["options_df_cache"] = options_df
                st.success(f"✅ Loaded {len(options_df)} PE records")
            else:
                st.warning("⚠️ No options data in cache")
            
            # Load NIFTY OHLCV data
            nifty_df = load_from_cache("nifty_ohlcv")
            if not nifty_df.empty:
                st.session_state["nifty_df_cache"] = nifty_df
                st.success(f"✅ Loaded {len(nifty_df)} NIFTY records")
            else:
                st.warning("⚠️ No NIFTY data in cache")
            
            futures_df = load_from_cache("nifty_futures")
            if not futures_df.empty:
                st.session_state["nifty_futures_df_cache"] = futures_df
                st.success(f"✅ Loaded {len(futures_df)} futures records")
            else:
                st.warning("⚠️ No NIFTY futures data in cache")
    if st.sidebar.button("🔄 Refresh", key="overview_refresh", help="Refresh overview with latest data"):
        st.rerun()
    
    # Use cached data if available
    if "options_df_cache" in st.session_state:
        options_df = st.session_state["options_df_cache"]
    if "nifty_df_cache" in st.session_state:
        nifty_df = st.session_state["nifty_df_cache"]
    if "nifty_futures_df_cache" not in st.session_state:
        futures_df = load_from_cache("nifty_futures")
        if not futures_df.empty:
            st.session_state["nifty_futures_df_cache"] = futures_df
    
    # Initialize empty dataframes if None
    if options_df is None:
        options_df = pd.DataFrame()
    if nifty_df is None:
        nifty_df = pd.DataFrame()
    
    # Check if positions exist
    positions = st.session_state.get("enriched_positions", [])
    
    if not positions:
        st.warning("⚠️ No positions loaded. Please load positions in the **Positions** tab first.")
        
        # Show market data even without positions
        st.info("💡 **Tip:** After fetching positions in the Positions tab, return here to see your complete portfolio overview.")
        
        # Show limited market regime data
        st.markdown("---")
        st.subheader("🌤️ Market Weather (Preview)")
        
        regime_data = calculate_market_regime(options_df, nifty_df)
        
        if regime_data:
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
            
            # Market metrics grid
            st.markdown("---")
            
            metric_cols = st.columns(5)
            
            with metric_cols[0]:
                iv_rank = regime_data.get("iv_rank", 50)
                iv_arrow = "⬆️" if iv_rank > 70 else ("⬇️" if iv_rank < 30 else "➡️")
                st.metric(
                    "📊 IV Rank",
                    f"{iv_rank:.0f}",
                    delta=f"{iv_arrow} {'Expensive' if iv_rank > 70 else ('Cheap' if iv_rank < 30 else 'Fair')}"
                )
            
            with metric_cols[1]:
                vrp = regime_data.get("vrp", 0)
                vrp_pct = vrp * 100
                st.metric(
                    "⚡ VRP",
                    f"{vrp_pct:+.1f}%",
                    delta="IV > RV" if vrp > 0.03 else ("RV > IV" if vrp < -0.03 else "Balanced")
                )
            
            with metric_cols[2]:
                term = regime_data.get("term_structure", 0)
                term_pct = term * 100
                term_label = "Backwardation" if term < -0.02 else ("Contango" if term > 0.02 else "Flat")
                st.metric(
                    "⏳ Term",
                    f"{term_pct:+.1f}%",
                    delta=term_label
                )
            
            with metric_cols[3]:
                skew = regime_data.get("skew", 0)
                skew_pct = skew * 100
                skew_label = "Puts Rich" if skew > 0.01 else ("Calls Rich" if skew < -0.01 else "Balanced")
                st.metric(
                    "⚖️ Skew",
                    f"{skew_pct:+.1f}%",
                    delta=skew_label
                )
            
            with metric_cols[4]:
                pcr = regime_data.get("pcr_oi", 1.0)
                pcr_label = "Bearish" if pcr > 1.1 else ("Bullish" if pcr < 0.9 else "Neutral")
                st.metric(
                    "🗓️ PCR (OI)",
                    f"{pcr:.2f}",
                    delta=pcr_label
                )
        
        return
    
    # Calculate market regime
    regime_data = calculate_market_regime(options_df, nifty_df)
    
    # Calculate portfolio metrics
    portfolio_greeks = calculate_portfolio_greeks(positions)
    
    # Get current spot
    current_spot = regime_data.get("current_spot", 19500)
    
    # Calculate VaR
    var_value = calculate_var(positions, current_spot, nifty_df)
    
    # Calculate portfolio value for percentages
    total_pnl = sum(p.get("pnl", 0) for p in positions)
    total_value = sum(abs(p.get("pnl", 0)) for p in positions) or 100000
    var_pct = (var_value / total_value) * 100
    
    # Count near-expiry positions (DTE < 7)
    near_expiry_count = sum(1 for p in positions if p.get("dte", 999) < 7)
    
    # ========== 1️⃣ TOP SECTION — MARKET WEATHER ==========
    st.subheader("🌤️ Market Weather")
    
    if regime_data:
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
        
        # Market metrics grid
        st.markdown("---")
        
        metric_cols = st.columns(5)
        
        with metric_cols[0]:
            iv_rank = regime_data.get("iv_rank", 50)
            iv_arrow = "⬆️" if iv_rank > 70 else ("⬇️" if iv_rank < 30 else "➡️")
            st.metric(
                "📊 IV Rank",
                f"{iv_rank:.0f}",
                delta=f"{iv_arrow} {'Expensive' if iv_rank > 70 else ('Cheap' if iv_rank < 30 else 'Fair')}"
            )
        
        with metric_cols[1]:
            vrp = regime_data.get("vrp", 0)
            vrp_pct = vrp * 100
            st.metric(
                "⚡ VRP",
                f"{vrp_pct:+.1f}%",
                delta="IV > RV" if vrp > 0.03 else ("RV > IV" if vrp < -0.03 else "Balanced")
            )
        
        with metric_cols[2]:
            term = regime_data.get("term_structure", 0)
            term_pct = term * 100
            term_label = "Backwardation" if term < -0.02 else ("Contango" if term > 0.02 else "Flat")
            st.metric(
                "⏳ Term",
                f"{term_pct:+.1f}%",
                delta=term_label
            )
        
        with metric_cols[3]:
            skew = regime_data.get("skew", 0)
            skew_pct = skew * 100
            skew_label = "Puts Rich" if skew > 0.01 else ("Calls Rich" if skew < -0.01 else "Balanced")
            st.metric(
                "⚖️ Skew",
                f"{skew_pct:+.1f}%",
                delta=skew_label
            )
        
        with metric_cols[4]:
            pcr = regime_data.get("pcr_oi", 1.0)
            pcr_label = "Bearish" if pcr > 1.1 else ("Bullish" if pcr < 0.9 else "Neutral")
            st.metric(
                "🗓️ PCR (OI)",
                f"{pcr:.2f}",
                delta=pcr_label
            )
        
        # Additional market metrics
        st.markdown("---")
        detail_cols = st.columns(4)
        
        with detail_cols[0]:
            st.metric("Current Spot", f"₹{current_spot:.0f}")
        
        with detail_cols[1]:
            rsi = regime_data.get("rsi", 50)
            rsi_status = "Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral")
            st.metric("RSI (14)", f"{rsi:.0f}", delta=rsi_status)
        
        with detail_cols[2]:
            max_pain = regime_data.get("max_pain_strike", current_spot)
            st.metric("Max Pain", f"₹{max_pain:.0f}")
        
        with detail_cols[3]:
            rv = regime_data.get("realized_vol", 0.15) * 100
            st.metric("Realized Vol (30d)", f"{rv:.1f}%")
    
    else:
        st.warning("⚠️ Insufficient market data for regime analysis")

    _render_daily_market_tracker(nifty_df=nifty_df, options_df=options_df)
    
    # ========== 2️⃣ MIDDLE SECTION — PORTFOLIO HEALTH SNAPSHOT ==========
    st.markdown("---")
    st.subheader("💼 Portfolio Health Snapshot")
    
    # Calculate additional metrics
    net_delta = portfolio_greeks["net_delta"]
    net_vega = portfolio_greeks["net_vega"]
    net_theta = portfolio_greeks["net_theta"]
    net_gamma = portfolio_greeks["net_gamma"]
    
    # Calculate theta efficiency (actual P&L vs expected theta decay)
    theta_efficiency = (total_pnl / net_theta * 100) if net_theta != 0 else 0
    
    # Calculate gamma-theta ratio
    gamma_theta_ratio = net_gamma / net_theta if net_theta != 0 else 0
    
    # Estimate days to recover (if losing money)
    days_to_recover = abs(total_pnl / net_theta) if total_pnl < 0 and net_theta > 0 else 0
    
    # Placeholder margin (would come from actual margin data)
    margin_used_pct = 62.0  # Placeholder
    
    # Portfolio health status
    health_emoji, health_status = get_portfolio_health_status(
        {"theta_efficiency": theta_efficiency},
        var_pct,
        margin_used_pct
    )
    
    # Display portfolio status
    status_cols = st.columns([1, 4])
    with status_cols[0]:
        st.markdown(f"### {health_emoji}")
        st.caption("Portfolio Status")
    with status_cols[1]:
        st.markdown(f"### {health_status}")
    
    st.markdown("---")
    
    # Portfolio metrics grid
    portfolio_cols = st.columns(4)
    
    with portfolio_cols[0]:
        pnl_color = "normal" if total_pnl >= 0 else "inverse"
        st.metric(
            "💰 Net P&L",
            format_inr(total_pnl),
            delta="Profitable" if total_pnl > 0 else "Loss",
            delta_color=pnl_color
        )
    
    with portfolio_cols[1]:
        delta_exposure = net_delta * current_spot
        st.metric(
            "⚖️ Delta (₹)",
            format_inr(delta_exposure),
            delta="Long" if delta_exposure > 0 else ("Short" if delta_exposure < 0 else "Neutral")
        )
    
    with portfolio_cols[2]:
        vega_status = "Short Vol" if net_vega < 0 else ("Long Vol" if net_vega > 0 else "Neutral")
        st.metric(
            "⚡ Vega",
            format_inr(net_vega),
            delta=vega_status
        )
    
    with portfolio_cols[3]:
        st.metric(
            "🕒 Theta/Day",
            format_inr(net_theta),
            delta="Positive Decay" if net_theta > 0 else "Negative Decay"
        )
    
    # Second row of portfolio metrics
    portfolio_cols2 = st.columns(4)
    
    with portfolio_cols2[0]:
        theta_status = "On Track" if abs(theta_efficiency) < 100 else "Deviated"
        st.metric(
            "🧮 Theta Efficiency",
            f"{theta_efficiency:.0f}%",
            delta=theta_status
        )
    
    with portfolio_cols2[1]:
        gamma_status = "Stable" if abs(gamma_theta_ratio) < 1 else "High Gamma"
        st.metric(
            "📊 Gamma/Theta",
            f"{gamma_theta_ratio:.2f}",
            delta=gamma_status
        )
    
    with portfolio_cols2[2]:
        margin_status = "High" if margin_used_pct > 80 else ("Moderate" if margin_used_pct > 60 else "Low")
        st.metric(
            "💸 Margin Used",
            f"{margin_used_pct:.0f}%",
            delta=margin_status
        )
    
    with portfolio_cols2[3]:
        var_status = "High" if var_pct > 5 else ("Moderate" if var_pct > 2 else "Low")
        st.metric(
            "🧯 VaR (95%)",
            f"{format_inr(-var_value)} ({var_pct:.1f}%)",
            delta=var_status
        )
    
    # Days to recover metric (if applicable)
    if days_to_recover > 0:
        recovery_cols = st.columns(4)
        with recovery_cols[0]:
            max_days = max([p.get("dte", 0) for p in positions]) if positions else 30
            recovery_status = "OK" if days_to_recover < max_days * 0.5 else "At Risk"
            st.metric(
                "📅 Days to Recover",
                f"{days_to_recover:.0f} of {max_days}",
                delta=recovery_status
            )
    
    # Visual add-ons
    st.markdown("---")
    
    # Greeks distribution pie chart
    viz_cols = st.columns([2, 1])
    
    with viz_cols[0]:
        st.markdown("**Greeks Contribution**")
        
        # Create pie chart for Greeks
        greeks_values = [
            abs(net_delta * current_spot * 0.01),  # Scale delta to comparable magnitude
            abs(net_gamma * 100),  # Scale gamma
            abs(net_vega),
            abs(net_theta)
        ]
        
        fig = go.Figure(data=[go.Pie(
            labels=['Delta', 'Gamma', 'Vega', 'Theta'],
            values=greeks_values,
            hole=0.3,
            marker=dict(colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
        )])
        
        fig.update_layout(
            showlegend=True,
            height=250,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with viz_cols[1]:
        st.markdown("**Portfolio Summary**")
        st.markdown(f"""
        - **Total Positions:** {len(positions)}
        - **Near Expiry (<7d):** {near_expiry_count}
        - **Net P&L:** {format_inr(total_pnl)}
        - **Total VaR:** {format_inr(var_value)}
        """)
    
    # ========== 3️⃣ BOTTOM SECTION — ALIGNMENT CHECK ==========
    st.markdown("---")
    st.subheader("🔄 Portfolio-Market Alignment")
    
    if regime_data:
        alignments = get_alignment_status(
            regime_data.get("iv_rank", 50),
            net_vega,
            regime_data.get("pcr_oi", 1.0),
            net_delta,
            regime_data.get("term_structure", 0),
            near_expiry_count
        )
        
        # Display alignment table
        alignment_df = pd.DataFrame(alignments)
        
        # Style the dataframe
        st.dataframe(
            alignment_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Status": st.column_config.TextColumn(
                    "Status",
                    width="small"
                ),
                "Action": st.column_config.TextColumn(
                    "Action",
                    width="large"
                )
            }
        )
    
    # ========== 4️⃣ ACTION BOX — NEXT STEP RECOMMENDATION ==========
    st.markdown("---")
    st.subheader("🎯 Next Step Recommendations")
    
    recommendations = []
    
    # Market strategy bias
    if regime_data:
        iv_rank = regime_data.get("iv_rank", 50)
        vrp = regime_data.get("vrp", 0)
        term_structure = regime_data.get("term_structure", 0)
        skew = regime_data.get("skew", 0)
        
        if iv_rank > 70 and vrp > 0.05:
            recommendations.append({
                "Type": "Market Strategy",
                "Recommendation": "🔴 Sell Volatility (prefer far expiry credit spreads)"
            })
        elif iv_rank < 30 and vrp < -0.05:
            recommendations.append({
                "Type": "Market Strategy",
                "Recommendation": "🟢 Buy Volatility (prefer debit spreads, long options)"
            })
        else:
            recommendations.append({
                "Type": "Market Strategy",
                "Recommendation": "🟡 Neutral — Consider Iron Condors or balanced spreads"
            })
        
        # Portfolio adjustment
        if abs(net_delta) > 20:
            delta_adjust = format_inr(abs(net_delta * current_spot))
            if net_delta > 20:
                recommendations.append({
                    "Type": "Portfolio Adjustment",
                    "Recommendation": f"⚠️ Reduce long delta by ~{delta_adjust}; add short calls or bearish spreads"
                })
            else:
                recommendations.append({
                    "Type": "Portfolio Adjustment",
                    "Recommendation": f"⚠️ Reduce short delta by ~{delta_adjust}; add long calls or bullish spreads"
                })
        else:
            recommendations.append({
                "Type": "Portfolio Adjustment",
                "Recommendation": "✅ Delta exposure within acceptable range"
            })
        
        # Risk flag
        if var_pct > 5:
            recommendations.append({
                "Type": "Risk Flag",
                "Recommendation": f"🚨 High VaR ({var_pct:.1f}%) — Consider reducing position sizes"
            })
        elif near_expiry_count > 5 and term_structure < -0.02:
            recommendations.append({
                "Type": "Risk Flag",
                "Recommendation": f"⚠️ {near_expiry_count} near-expiry positions in backwardation — Roll to next month"
            })
        else:
            recommendations.append({
                "Type": "Risk Flag",
                "Recommendation": "✅ Risk metrics within acceptable limits"
            })
        
        # Opportunities
        if abs(skew) > 0.015:
            if skew > 0:
                recommendations.append({
                    "Type": "Opportunities",
                    "Recommendation": f"💡 Puts overpriced by {skew*100:.1f}% — Short put spreads attractive"
                })
            else:
                recommendations.append({
                    "Type": "Opportunities",
                    "Recommendation": f"💡 Calls overpriced by {abs(skew)*100:.1f}% — Short call spreads attractive"
                })
    
    # Display recommendations
    if recommendations:
        rec_df = pd.DataFrame(recommendations)
        st.dataframe(
            rec_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Type": st.column_config.TextColumn("Type", width="medium"),
                "Recommendation": st.column_config.TextColumn("Recommendation", width="large")
            }
        )
    
    # Auto summary message
    st.markdown("---")
    st.info(f"""
    **🧠 AI Summary:**
    
    Portfolio currently {health_status.lower()} with {format_inr(total_pnl)} P&L and {portfolio_greeks['total_positions']} active positions.
    
    Market regime is **{regime_data.get('market_regime', 'Unknown')}** with IV Rank at {regime_data.get('iv_rank', 50):.0f} 
    and VRP at {regime_data.get('vrp', 0)*100:+.1f}%.
    
    {'✅ Portfolio volatility stance aligned with market conditions.' if (regime_data.get('iv_rank', 50) > 70 and net_vega < 0) or (regime_data.get('iv_rank', 50) < 30 and net_vega > 0) else '⚠️ Consider adjusting volatility exposure to align with market regime.'}
    
    {'⚠️ Delta exposure elevated — consider hedging or reducing directional risk.' if abs(net_delta) > 20 else '✅ Delta exposure manageable.'}
    
    {f'🔄 Roll {near_expiry_count} near-expiry positions to avoid gamma risk.' if near_expiry_count > 5 and regime_data.get('term_structure', 0) < -0.02 else ''}
    """)
