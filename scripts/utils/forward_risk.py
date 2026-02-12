from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.utils.greeks import calculate_implied_volatility
from scripts.utils.parsers import parse_tradingsymbol

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


TRADING_DAYS_PER_YEAR = 252.0
MIN_VOL = 0.03
MAX_VOL = 2.50
MIN_DAILY_VAR = (MIN_VOL / math.sqrt(TRADING_DAYS_PER_YEAR)) ** 2
MAX_DAILY_VAR = (MAX_VOL / math.sqrt(TRADING_DAYS_PER_YEAR)) ** 2
ADV_MC_VERSION = "adv_mc_v5_2026-02-11"
logger = logging.getLogger("gammashield.api")


@dataclass
class AdvancedForwardConfig:
    horizon_days: int = 10
    n_paths: int = 2000
    dt: float = 1.0 / TRADING_DAYS_PER_YEAR
    seed: int = 42
    risk_free_rate: float = 0.0
    dividend_yield: float = 0.0
    daily_loss_limit: Optional[float] = None
    total_loss_limit: Optional[float] = None
    engines: Tuple[str, ...] = ("fhs", "garch", "egarch", "gjr", "heston", "bates")
    pnl_modes: Tuple[str, ...] = ("greeks", "repricing")
    iv_rules: Tuple[str, ...] = ("surface", "flat")
    repricing_models: Tuple[str, ...] = ("bs", "bs76")
    use_evt_overlay: bool = True
    simulate_surface: bool = True
    heston_kappa: float = 2.0
    heston_theta: float = 0.045
    heston_xi: float = 0.45
    heston_rho: float = -0.60
    bates_jump_lambda: float = 0.20
    bates_jump_mu: float = -0.03
    bates_jump_sigma: float = 0.08
    surface_rho_to_spot: float = -0.35
    surface_factor_vol_level: float = 0.35
    surface_factor_vol_skew: float = 0.28
    surface_factor_vol_curv: float = 0.22
    surface_factor_kappa: float = 2.0
    margin_model: str = "none"
    tx_cost_per_leg: float = 0.0
    slippage_per_leg: float = 0.0
    repricing_anchor: str = "market_t0"
    debug_logging: bool = True
    debug_mode_filter: Optional[str] = "repricing"
    debug_skip_engines: Tuple[str, ...] = ("gbm",)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _as_date(v: Any) -> Optional[date]:
    if isinstance(v, date):
        return v
    if isinstance(v, datetime):
        return v.date()
    if v is None:
        return None
    try:
        return pd.to_datetime(v, errors="coerce").date()
    except Exception:
        return None


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        out = float(v)
        if math.isfinite(out):
            return out
    except Exception:
        pass
    return default


def _parse_future_symbol(symbol: str) -> Optional[Dict[str, Any]]:
    s = (symbol or "").upper().strip()
    if not s.endswith("FUT"):
        return None
    month_map = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    if len(s) >= 12 and s.startswith("NIFTY"):
        yy = s[5:7]
        mon = s[7:10]
        if yy.isdigit() and mon in month_map:
            year = 2000 + int(yy)
            month = month_map[mon]
            last_day = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
            while last_day.weekday() != 1:
                last_day -= pd.Timedelta(days=1)
            return {"expiry": last_day.date()}
    return {"expiry": None}


def _nearest_option_row(
    options_df: pd.DataFrame,
    expiry: Optional[date],
    strike: float,
    option_type: str,
) -> Optional[pd.Series]:
    if options_df.empty:
        return None
    df = options_df
    if "option_type" not in df.columns or "strike_price" not in df.columns:
        return None
    pick = df[df["option_type"].astype(str).str.upper() == str(option_type).upper()].copy()
    if pick.empty:
        return None
    if expiry is not None:
        if "expiry" in pick.columns:
            pick["expiry_norm"] = pd.to_datetime(pick["expiry"], errors="coerce").dt.date
        elif "expiry_date" in pick.columns:
            pick["expiry_norm"] = pd.to_datetime(pick["expiry_date"], errors="coerce").dt.date
        else:
            pick["expiry_norm"] = None
        exp_match = pick[pick["expiry_norm"] == expiry]
        if not exp_match.empty:
            pick = exp_match
    pick["strike_dist"] = (pick["strike_price"].astype(float) - float(strike)).abs()
    pick = pick.sort_values("strike_dist")
    if pick.empty:
        return None
    return pick.iloc[0]


def _build_legs(
    positions: List[Dict[str, Any]],
    options_df: pd.DataFrame,
    spot: float,
    risk_free_rate: float,
) -> List[Dict[str, Any]]:
    legs: List[Dict[str, Any]] = []
    today = datetime.now().date()
    for p in positions:
        qty = _safe_float(p.get("quantity"), 0.0)
        if qty == 0:
            continue
        symbol = str(p.get("tradingsymbol", "") or "").upper()
        parsed_opt = parse_tradingsymbol(symbol)
        if parsed_opt:
            strike = _safe_float(parsed_opt.get("strike"), 0.0)
            expiry_dt = parsed_opt.get("expiry")
            expiry = expiry_dt.date() if isinstance(expiry_dt, datetime) else _as_date(expiry_dt)
            option_type = str(parsed_opt.get("option_type", "CE")).upper()
            tte = max((_as_date(expiry) - today).days / 365.0, 1.0 / 365.0) if expiry else 30.0 / 365.0
            px0 = _safe_float(p.get("last_price"), _safe_float(p.get("average_price"), 0.0))
            iv0 = _safe_float(p.get("implied_vol"), 0.0)
            if iv0 > 1.0:
                iv0 /= 100.0
            if iv0 <= 0.0 and px0 > 0 and strike > 0:
                iv_calc = calculate_implied_volatility(px0, spot, strike, tte, option_type, risk_free_rate=risk_free_rate)
                iv0 = _safe_float(iv_calc, 0.0)
            if iv0 <= 0.0:
                row = _nearest_option_row(options_df, expiry, strike, option_type)
                if row is not None:
                    px_ref = _safe_float(row.get("ltp"), _safe_float(row.get("close"), 0.0))
                    if px_ref > 0:
                        iv_calc = calculate_implied_volatility(px_ref, spot, strike, tte, option_type, risk_free_rate=risk_free_rate)
                        iv0 = _safe_float(iv_calc, 0.0)
            iv0 = float(np.clip(iv0 if iv0 > 0 else 0.20, MIN_VOL, MAX_VOL))
            legs.append(
                {
                    "symbol": symbol,
                    "kind": "option",
                    "qty": qty,
                    "strike": strike,
                    "expiry": expiry,
                    "flag": "c" if option_type == "CE" else "p",
                    "price0": px0,
                    "iv0": iv0,
                }
            )
            continue

        parsed_fut = _parse_future_symbol(symbol)
        if parsed_fut:
            fut_price0 = _safe_float(p.get("last_price"), _safe_float(p.get("average_price"), spot))
            legs.append(
                {
                    "symbol": symbol,
                    "kind": "future",
                    "qty": qty,
                    "expiry": _as_date(parsed_fut.get("expiry")),
                    "price0": fut_price0 if fut_price0 > 0 else spot,
                }
            )
    return legs


def _log_leg_inputs(legs: List[Dict[str, Any]], spot0: float, cfg: AdvancedForwardConfig) -> None:
    if not cfg.debug_logging:
        return
    option_count = sum(1 for leg in legs if leg.get("kind") == "option")
    future_count = sum(1 for leg in legs if leg.get("kind") == "future")
    logger.info(
        "[ADV_SIM][INPUT] legs_total=%d option_legs=%d future_legs=%d spot0=%.6f r=%s q=%s",
        len(legs),
        option_count,
        future_count,
        float(spot0),
        cfg.risk_free_rate,
        cfg.dividend_yield,
    )
    today = datetime.now().date()
    for idx, leg in enumerate(legs):
        if leg.get("kind") == "future":
            logger.info(
                "[ADV_SIM][INPUT][LEG] i=%d kind=future symbol=%s qty=%.6f price0=%.6f expiry=%s",
                idx,
                str(leg.get("symbol", "")),
                float(leg.get("qty", 0.0)),
                float(leg.get("price0", spot0)),
                str(leg.get("expiry")),
            )
            continue
        expiry = leg.get("expiry")
        dte = max((expiry - today).days if isinstance(expiry, date) else 30, 0)
        tte = max(dte / 365.0, 1.0 / 365.0)
        logger.info(
            (
                "[ADV_SIM][INPUT][LEG] i=%d kind=option symbol=%s qty=%.6f flag=%s strike=%.6f "
                "expiry=%s dte=%d tte=%.8f iv0=%.8f price0=%.8f"
            ),
            idx,
            str(leg.get("symbol", "")),
            float(leg.get("qty", 0.0)),
            str(leg.get("flag", "")),
            float(leg.get("strike", 0.0)),
            str(expiry),
            dte,
            tte,
            float(leg.get("iv0", 0.0)),
            float(leg.get("price0", 0.0)),
        )


def _log_engine_inputs(
    engine: str,
    spot_paths: np.ndarray,
    returns_paths: np.ndarray,
    factor_shocks: Optional[np.ndarray],
    cfg: AdvancedForwardConfig,
) -> None:
    if not cfg.debug_logging:
        return
    s1 = spot_paths[:, 0] if spot_paths.shape[1] >= 1 else np.array([], dtype=float)
    st = spot_paths[:, -1] if spot_paths.shape[1] >= 1 else np.array([], dtype=float)
    r1 = returns_paths[:, 0] if returns_paths.shape[1] >= 1 else np.array([], dtype=float)
    rt = returns_paths[:, -1] if returns_paths.shape[1] >= 1 else np.array([], dtype=float)
    logger.info(
        (
            "[ADV_SIM][ENGINE_INPUT] engine=%s paths=%d horizon=%d "
            "s1[min=%.6f mean=%.6f max=%.6f] sT[min=%.6f mean=%.6f max=%.6f] "
            "r1[mean=%.8f std=%.8f] rT[mean=%.8f std=%.8f]"
        ),
        engine,
        int(spot_paths.shape[0]),
        int(spot_paths.shape[1]),
        float(np.min(s1)) if s1.size else float("nan"),
        float(np.mean(s1)) if s1.size else float("nan"),
        float(np.max(s1)) if s1.size else float("nan"),
        float(np.min(st)) if st.size else float("nan"),
        float(np.mean(st)) if st.size else float("nan"),
        float(np.max(st)) if st.size else float("nan"),
        float(np.mean(r1)) if r1.size else float("nan"),
        float(np.std(r1)) if r1.size else float("nan"),
        float(np.mean(rt)) if rt.size else float("nan"),
        float(np.std(rt)) if rt.size else float("nan"),
    )
    if factor_shocks is not None and factor_shocks.size:
        logger.info(
            (
                "[ADV_SIM][ENGINE_INPUT] engine=%s surface[level_mean=%.8f level_std=%.8f "
                "skew_mean=%.8f skew_std=%.8f curv_mean=%.8f curv_std=%.8f]"
            ),
            engine,
            float(np.mean(factor_shocks[:, :, 0])),
            float(np.std(factor_shocks[:, :, 0])),
            float(np.mean(factor_shocks[:, :, 1])),
            float(np.std(factor_shocks[:, :, 1])),
            float(np.mean(factor_shocks[:, :, 2])),
            float(np.std(factor_shocks[:, :, 2])),
        )


def _prepare_returns(spot_history: pd.DataFrame) -> np.ndarray:
    if spot_history is None or spot_history.empty or "close" not in spot_history.columns:
        return np.array([], dtype=float)
    df = spot_history.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")
    close = pd.to_numeric(df["close"], errors="coerce")
    close = close.replace([np.inf, -np.inf], np.nan).dropna()
    if len(close) < 40:
        return np.array([], dtype=float)
    rets = np.log(close / close.shift(1)).dropna().to_numpy(dtype=float)
    return rets[np.isfinite(rets)]


def _fit_evt_left_tail(z_hist: np.ndarray, tail_q: float = 0.10) -> Optional[Dict[str, float]]:
    if z_hist.size < 100:
        return None
    z = z_hist[np.isfinite(z_hist)]
    if z.size < 100:
        return None
    u = float(np.quantile(z, tail_q))
    tail = z[z < u]
    if tail.size < 50:
        return None
    exceed = (u - tail).astype(float)
    m1 = float(np.mean(exceed))
    m2 = float(np.var(exceed))
    if m1 <= 1e-9:
        return None
    if m2 <= m1 * m1:
        xi = 0.05
        beta = max(m1 * (1 - xi), 1e-6)
    else:
        xi = 0.5 * (1.0 - (m1 * m1 / m2))
        xi = float(np.clip(xi, -0.20, 0.50))
        beta = m1 * (1.0 - xi)
        beta = float(max(beta, 1e-6))
    return {"u": u, "xi": xi, "beta": beta, "tail_prob": float(tail.size / z.size)}


def _sample_gpd(size: int, xi: float, beta: float, rng: np.random.Generator) -> np.ndarray:
    u = np.clip(rng.random(size=size), 1e-9, 1.0 - 1e-9)
    if abs(xi) < 1e-8:
        return -beta * np.log(1.0 - u)
    return (beta / xi) * ((1.0 - u) ** (-xi) - 1.0)


def _overlay_evt_on_z(z: np.ndarray, evt: Optional[Dict[str, float]], rng: np.random.Generator) -> np.ndarray:
    if evt is None:
        return z
    out = z.copy()
    mask = rng.random(size=out.shape) < evt["tail_prob"]
    if not np.any(mask):
        return out
    draw = _sample_gpd(int(mask.sum()), evt["xi"], evt["beta"], rng)
    out[mask] = evt["u"] - draw
    return out


def _vol_fit_errors(pred_sigma_ann: np.ndarray, returns: np.ndarray) -> Dict[str, Optional[float]]:
    if pred_sigma_ann.size == 0 or returns.size == 0:
        return {"vol_mae": None, "vol_rmse": None}
    n = min(pred_sigma_ann.size, returns.size)
    pred = pred_sigma_ann[-n:]
    real = np.abs(returns[-n:]) * math.sqrt(TRADING_DAYS_PER_YEAR)
    err = pred - real
    return {
        "vol_mae": float(np.mean(np.abs(err))),
        "vol_rmse": float(np.sqrt(np.mean(err * err))),
    }


def _simulate_engine_gbm(
    spot0: float,
    returns_hist: np.ndarray,
    cfg: AdvancedForwardConfig,
    rng: np.random.Generator,
    evt: Optional[Dict[str, float]],
) -> Dict[str, Any]:
    sigma = max(np.std(returns_hist) * math.sqrt(TRADING_DAYS_PER_YEAR), 0.08) if returns_hist.size else 0.20
    z = rng.standard_normal(size=(cfg.n_paths, cfg.horizon_days))
    z = _overlay_evt_on_z(z, evt if cfg.use_evt_overlay else None, rng)
    rets = (-0.5 * sigma * sigma) * cfg.dt + sigma * math.sqrt(cfg.dt) * z
    log_paths = np.cumsum(rets, axis=1)
    s_paths = spot0 * np.exp(log_paths)
    pred = np.full(max(len(returns_hist), 1), sigma, dtype=float)
    return {
        "spot_paths": s_paths,
        "returns_paths": rets,
        "sigma_paths_ann": np.full_like(s_paths, sigma),
        "fit": _vol_fit_errors(pred, returns_hist),
    }


def _simulate_engine_fhs(
    spot0: float,
    returns_hist: np.ndarray,
    cfg: AdvancedForwardConfig,
    rng: np.random.Generator,
    evt: Optional[Dict[str, float]],
) -> Dict[str, Any]:
    if returns_hist.size < 80:
        return _simulate_engine_gbm(spot0, returns_hist, cfg, rng, evt)
    lam = 0.94
    r = returns_hist
    h = np.zeros_like(r)
    h[0] = max(np.var(r), 1e-10)
    for t in range(1, len(r)):
        h[t] = lam * h[t - 1] + (1.0 - lam) * (r[t - 1] ** 2)
    sig = np.sqrt(np.maximum(h, 1e-12))
    z_hist = r / sig
    z_hist = z_hist[np.isfinite(z_hist)]
    if z_hist.size < 40:
        return _simulate_engine_gbm(spot0, returns_hist, cfg, rng, evt)
    sigma_prev = np.full(cfg.n_paths, sig[-1], dtype=float)
    rets = np.zeros((cfg.n_paths, cfg.horizon_days), dtype=float)
    sig_ann = np.zeros_like(rets)
    for t in range(cfg.horizon_days):
        z = rng.choice(z_hist, size=cfg.n_paths, replace=True)
        if cfg.use_evt_overlay:
            z = _overlay_evt_on_z(z, evt, rng)
        rt = sigma_prev * z
        rets[:, t] = rt
        sig_ann[:, t] = sigma_prev * math.sqrt(TRADING_DAYS_PER_YEAR)
        sigma_prev = np.sqrt(np.maximum(lam * sigma_prev**2 + (1.0 - lam) * rt**2, 1e-12))
    log_paths = np.cumsum(rets, axis=1)
    s_paths = spot0 * np.exp(log_paths)
    pred_sigma_ann = sig * math.sqrt(TRADING_DAYS_PER_YEAR)
    return {
        "spot_paths": s_paths,
        "returns_paths": rets,
        "sigma_paths_ann": sig_ann,
        "fit": _vol_fit_errors(pred_sigma_ann, returns_hist),
    }


def _simulate_engine_garch_family(
    family: str,
    spot0: float,
    returns_hist: np.ndarray,
    cfg: AdvancedForwardConfig,
    rng: np.random.Generator,
    evt: Optional[Dict[str, float]],
) -> Dict[str, Any]:
    if returns_hist.size < 80:
        return _simulate_engine_gbm(spot0, returns_hist, cfg, rng, evt)
    r = returns_hist
    var_u = float(np.clip(np.var(r), MIN_DAILY_VAR, MAX_DAILY_VAR))
    if family == "garch":
        alpha, beta, gamma = 0.07, 0.90, 0.00
    elif family == "gjr":
        alpha, beta, gamma = 0.05, 0.88, 0.10
    else:
        alpha, beta, gamma = 0.10, 0.95, -0.10
    omega = max(var_u * max(1e-6, 1.0 - alpha - beta - 0.5 * max(gamma, 0.0)), MIN_DAILY_VAR)
    min_log_var = math.log(MIN_DAILY_VAR)
    max_log_var = math.log(MAX_DAILY_VAR)

    pred_h = np.zeros_like(r)
    pred_h[0] = var_u
    for t in range(1, len(r)):
        if family == "egarch":
            z_prev = r[t - 1] / math.sqrt(max(pred_h[t - 1], MIN_DAILY_VAR))
            z_prev = float(np.clip(z_prev, -12.0, 12.0))
            e_abs = math.sqrt(2.0 / math.pi)
            log_h = math.log(max(pred_h[t - 1], MIN_DAILY_VAR))
            log_h = math.log(max(omega, 1e-12)) + beta * log_h + alpha * (abs(z_prev) - e_abs) + gamma * z_prev
            log_h = float(np.clip(log_h, min_log_var, max_log_var))
            pred_h[t] = float(np.clip(math.exp(log_h), MIN_DAILY_VAR, MAX_DAILY_VAR))
        else:
            ind = 1.0 if (r[t - 1] < 0 and family == "gjr") else 0.0
            pred_h[t] = max(
                omega + alpha * (r[t - 1] ** 2) + gamma * ind * (r[t - 1] ** 2) + beta * pred_h[t - 1],
                MIN_DAILY_VAR,
            )
            pred_h[t] = float(np.clip(pred_h[t], MIN_DAILY_VAR, MAX_DAILY_VAR))

    h_prev = np.full(cfg.n_paths, pred_h[-1], dtype=float)
    r_prev = np.full(cfg.n_paths, r[-1], dtype=float)
    rets = np.zeros((cfg.n_paths, cfg.horizon_days), dtype=float)
    sig_ann = np.zeros_like(rets)
    for t in range(cfg.horizon_days):
        z = rng.standard_normal(size=cfg.n_paths)
        if cfg.use_evt_overlay:
            z = _overlay_evt_on_z(z, evt, rng)
        if family == "egarch":
            e_abs = math.sqrt(2.0 / math.pi)
            z_prev = r_prev / np.sqrt(np.maximum(h_prev, MIN_DAILY_VAR))
            z_prev = np.clip(z_prev, -12.0, 12.0)
            log_h = np.log(np.maximum(omega, MIN_DAILY_VAR)) + beta * np.log(np.maximum(h_prev, MIN_DAILY_VAR))
            log_h = log_h + alpha * (np.abs(z_prev) - e_abs) + gamma * z_prev
            log_h = np.clip(log_h, min_log_var, max_log_var)
            h_curr = np.exp(log_h)
            h_curr = np.clip(h_curr, MIN_DAILY_VAR, MAX_DAILY_VAR)
        else:
            ind = ((r_prev < 0) & (family == "gjr")).astype(float)
            h_curr = omega + alpha * (r_prev**2) + gamma * ind * (r_prev**2) + beta * h_prev
            h_curr = np.maximum(h_curr, MIN_DAILY_VAR)
            h_curr = np.clip(h_curr, MIN_DAILY_VAR, MAX_DAILY_VAR)
        rt = np.sqrt(h_curr) * z
        rets[:, t] = rt
        sig_ann[:, t] = np.sqrt(h_curr) * math.sqrt(TRADING_DAYS_PER_YEAR)
        h_prev = h_curr
        r_prev = rt
    log_paths = np.cumsum(rets, axis=1)
    s_paths = spot0 * np.exp(log_paths)
    pred_sigma_ann = np.sqrt(np.maximum(pred_h, 1e-12)) * math.sqrt(TRADING_DAYS_PER_YEAR)
    return {
        "spot_paths": s_paths,
        "returns_paths": rets,
        "sigma_paths_ann": sig_ann,
        "fit": _vol_fit_errors(pred_sigma_ann, returns_hist),
    }


def _simulate_engine_heston(
    spot0: float,
    cfg: AdvancedForwardConfig,
    rng: np.random.Generator,
    with_jumps: bool = False,
) -> Dict[str, Any]:
    kappa = max(cfg.heston_kappa, 1e-6)
    theta = max(cfg.heston_theta, 1e-6)
    xi = max(cfg.heston_xi, 1e-6)
    rho = float(np.clip(cfg.heston_rho, -0.99, 0.99))

    z1 = rng.standard_normal(size=(cfg.n_paths, cfg.horizon_days))
    z2 = rng.standard_normal(size=(cfg.n_paths, cfg.horizon_days))
    wv = rho * z1 + math.sqrt(1.0 - rho * rho) * z2

    s_paths = np.zeros((cfg.n_paths, cfg.horizon_days), dtype=float)
    rets = np.zeros_like(s_paths)
    sig_ann = np.zeros_like(s_paths)
    s_prev = np.full(cfg.n_paths, spot0, dtype=float)
    v_prev = np.full(cfg.n_paths, theta, dtype=float)
    for t in range(cfg.horizon_days):
        v_pos = np.maximum(v_prev, 0.0)
        dv = kappa * (theta - v_pos) * cfg.dt + xi * np.sqrt(v_pos) * math.sqrt(cfg.dt) * wv[:, t]
        v_curr = np.maximum(v_prev + dv, 0.0)
        rt = (-0.5 * v_pos) * cfg.dt + np.sqrt(v_pos * cfg.dt) * z1[:, t]
        if with_jumps:
            n_jump = rng.poisson(cfg.bates_jump_lambda * cfg.dt, size=cfg.n_paths)
            jump = n_jump * (cfg.bates_jump_mu + cfg.bates_jump_sigma * rng.standard_normal(cfg.n_paths))
            rt = rt + jump
        s_curr = s_prev * np.exp(rt)
        s_paths[:, t] = s_curr
        rets[:, t] = rt
        sig_ann[:, t] = np.sqrt(np.maximum(v_curr, 1e-12))
        s_prev = s_curr
        v_prev = v_curr
    return {
        "spot_paths": s_paths,
        "returns_paths": rets,
        "sigma_paths_ann": sig_ann,
        "fit": {"vol_mae": None, "vol_rmse": None},
    }


def _fit_surface_factors(options_df: pd.DataFrame, spot: float, risk_free_rate: float) -> Dict[str, Any]:
    out = {
        "base": np.array([0.20, -0.04, 0.15, 0.00], dtype=float),
        "rmse": None,
        "points": 0,
    }
    if options_df is None or options_df.empty:
        return out
    df = options_df.copy()
    for c in ("strike_price", "ltp", "close"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "expiry" in df.columns:
        df["expiry_norm"] = pd.to_datetime(df["expiry"], errors="coerce").dt.date
    elif "expiry_date" in df.columns:
        df["expiry_norm"] = pd.to_datetime(df["expiry_date"], errors="coerce").dt.date
    else:
        return out
    if "option_type" not in df.columns:
        return out
    df = df.dropna(subset=["strike_price", "expiry_norm", "option_type"])
    if df.empty:
        return out
    today = datetime.now().date()
    df["tte"] = (pd.to_datetime(df["expiry_norm"]) - pd.Timestamp(today)).dt.days / 365.0
    df = df[df["tte"] > 0.01]
    if df.empty:
        return out
    df["m"] = np.log(np.maximum(df["strike_price"], 1.0) / max(spot, 1.0))
    df = df[np.abs(df["m"]) <= 0.35]
    if df.empty:
        return out
    df = df.sort_values(["expiry_norm", "strike_price"]).groupby(["expiry_norm", "option_type"]).head(40)

    xs = []
    ys = []
    for _, row in df.iterrows():
        px = _safe_float(row.get("ltp"), _safe_float(row.get("close"), 0.0))
        if px <= 0:
            continue
        t = float(max(row.get("tte", 0.0), 1.0 / 365.0))
        strike = _safe_float(row.get("strike_price"), 0.0)
        if strike <= 0:
            continue
        opt_type = str(row.get("option_type", "CE")).upper()
        iv = calculate_implied_volatility(px, spot, strike, t, opt_type, risk_free_rate=risk_free_rate)
        iv = _safe_float(iv, 0.0)
        if iv <= 0:
            continue
        m = float(row.get("m", 0.0))
        xs.append([1.0, m, m * m, math.sqrt(t)])
        ys.append(iv)
    if len(xs) < 30:
        return out
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ beta
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    out["base"] = beta.astype(float)
    out["rmse"] = rmse
    out["points"] = int(len(y))
    return out


def _simulate_surface_factor_shocks(
    returns_paths: np.ndarray,
    cfg: AdvancedForwardConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    paths, horizon = returns_paths.shape
    z_spot = returns_paths / max(np.std(returns_paths), 1e-8)
    sqrt_dt = math.sqrt(cfg.dt)
    rho = float(np.clip(cfg.surface_rho_to_spot, -0.99, 0.99))
    shocks = np.zeros((paths, horizon, 3), dtype=float)
    f = np.zeros((paths, 3), dtype=float)
    kappa_dt = cfg.surface_factor_kappa * cfg.dt
    for t in range(horizon):
        e0 = z_spot[:, t]
        e1 = rng.standard_normal(paths)
        e2 = rng.standard_normal(paths)
        e3 = rng.standard_normal(paths)
        lvl = cfg.surface_factor_vol_level * sqrt_dt * (rho * e0 + math.sqrt(1.0 - rho * rho) * e1)
        skw = cfg.surface_factor_vol_skew * sqrt_dt * e2
        cur = cfg.surface_factor_vol_curv * sqrt_dt * e3
        f[:, 0] = (1.0 - kappa_dt) * f[:, 0] + lvl
        f[:, 1] = (1.0 - kappa_dt) * f[:, 1] + skw
        f[:, 2] = (1.0 - kappa_dt) * f[:, 2] + cur
        shocks[:, t, :] = f
    return shocks


def _bs_price_vec(flag: str, s: np.ndarray, k: float, t: float, r: float, sigma: np.ndarray) -> np.ndarray:
    s = np.maximum(s, 1e-8)
    k = max(k, 1e-8)
    t = max(t, 1e-8)
    sigma = np.maximum(sigma, 1e-6)
    sqrt_t = math.sqrt(t)
    d1 = (np.log(s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    nd1 = _norm_cdf(d1)
    nd2 = _norm_cdf(d2)
    disc = math.exp(-r * t)
    if flag == "c":
        return s * nd1 - k * disc * nd2
    return k * disc * _norm_cdf(-d2) - s * _norm_cdf(-d1)


def _b76_price_vec(flag: str, fwd: np.ndarray, k: float, t: float, r: float, sigma: np.ndarray) -> np.ndarray:
    fwd = np.maximum(fwd, 1e-8)
    k = max(k, 1e-8)
    t = max(t, 1e-8)
    sigma = np.maximum(sigma, 1e-6)
    sqrt_t = math.sqrt(t)
    d1 = (np.log(fwd / k) + 0.5 * sigma * sigma * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    nd1 = _norm_cdf(d1)
    nd2 = _norm_cdf(d2)
    disc = math.exp(-r * t)
    if flag == "c":
        return disc * (fwd * nd1 - k * nd2)
    return disc * (k * _norm_cdf(-d2) - fwd * _norm_cdf(-d1))


def _bs_greeks_vec(flag: str, s: np.ndarray, k: float, t: float, r: float, sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s = np.maximum(s, 1e-8)
    k = max(k, 1e-8)
    t = max(t, 1e-8)
    sigma = np.maximum(sigma, 1e-6)
    sqrt_t = math.sqrt(t)
    d1 = (np.log(s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    pdf = _norm_pdf(d1)
    if flag == "c":
        delta = _norm_cdf(d1)
        theta = -(s * pdf * sigma) / (2.0 * sqrt_t) - r * k * math.exp(-r * t) * _norm_cdf(d2)
    else:
        delta = _norm_cdf(d1) - 1.0
        theta = -(s * pdf * sigma) / (2.0 * sqrt_t) + r * k * math.exp(-r * t) * _norm_cdf(-d2)
    gamma = pdf / (s * sigma * sqrt_t)
    vega = s * pdf * sqrt_t
    return delta, gamma, vega, theta


def _effective_iv_for_leg(
    leg: Dict[str, Any],
    spot_vec: np.ndarray,
    tte: float,
    surface_base: np.ndarray,
    factor_shocks: Optional[np.ndarray],
    t: int,
) -> np.ndarray:
    if factor_shocks is None:
        return np.full_like(spot_vec, float(leg["iv0"]), dtype=float)
    m = np.log(np.maximum(float(leg["strike"]), 1e-8) / np.maximum(spot_vec, 1e-8))
    lvl = surface_base[0] + factor_shocks[:, t, 0]
    skw = surface_base[1] + factor_shocks[:, t, 1]
    cur = surface_base[2] + factor_shocks[:, t, 2]
    trm = surface_base[3]
    iv = lvl + skw * m + cur * (m * m) + trm * math.sqrt(max(tte, 1e-8))
    return np.clip(iv, MIN_VOL, MAX_VOL)


def _simulate_mode_repricing(
    legs: List[Dict[str, Any]],
    spot_paths: np.ndarray,
    spot0: float,
    surface_base: np.ndarray,
    factor_shocks: Optional[np.ndarray],
    cfg: AdvancedForwardConfig,
    engine_name: Optional[str] = None,
    pricing_model: str = "bs",
) -> np.ndarray:
    paths, horizon = spot_paths.shape
    pnl = np.zeros((paths, horizon), dtype=float)
    today = datetime.now().date()
    initial_value = np.zeros(paths, dtype=float)
    initial_market = 0.0
    initial_model_iv0 = 0.0
    initial_model_surface = 0.0
    for idx, leg in enumerate(legs):
        qty = float(leg.get("qty", 0.0))
        if leg["kind"] == "future":
            px_market = float(leg.get("price0", spot0))
            px_iv0 = px_market
            px_surface = px_market
            # Anchor repricing P&L to current marked futures price.
            initial_value += qty * px_market
        else:
            # Anchor repricing P&L to current market option price to avoid model-calibration bias.
            px_market = float(leg.get("price0", 0.0))
            expiry = leg.get("expiry")
            dte0 = max((expiry - today).days if isinstance(expiry, date) else 30, 0)
            tte0 = max(dte0 / 365.0, 1.0 / 365.0)
            iv0 = float(np.clip(float(leg.get("iv0", 0.0)), MIN_VOL, MAX_VOL))
            spot_vec = np.asarray([spot0], dtype=float)
            if pricing_model == "bs76":
                fwd0 = spot_vec * math.exp((cfg.risk_free_rate - cfg.dividend_yield) * tte0)
                px_iv0 = float(
                    _b76_price_vec(
                        leg["flag"],
                        fwd0,
                        float(leg["strike"]),
                        tte0,
                        cfg.risk_free_rate,
                        np.asarray([iv0], dtype=float),
                    )[0]
                )
            else:
                px_iv0 = float(
                    _bs_price_vec(
                        leg["flag"],
                        spot_vec,
                        float(leg["strike"]),
                        tte0,
                        cfg.risk_free_rate,
                        np.asarray([iv0], dtype=float),
                    )[0]
                )
            iv_surface = float(
                _effective_iv_for_leg(
                    leg,
                    spot_vec,
                    tte0,
                    surface_base,
                    None,
                    0,
                )[0]
            )
            if pricing_model == "bs76":
                px_surface = float(
                    _b76_price_vec(
                        leg["flag"],
                        fwd0,
                        float(leg["strike"]),
                        tte0,
                        cfg.risk_free_rate,
                        np.asarray([iv_surface], dtype=float),
                    )[0]
                )
            else:
                px_surface = float(
                    _bs_price_vec(
                        leg["flag"],
                        spot_vec,
                        float(leg["strike"]),
                        tte0,
                        cfg.risk_free_rate,
                        np.asarray([iv_surface], dtype=float),
                    )[0]
                )
            initial_value += qty * px_market
        initial_market += qty * px_market
        initial_model_iv0 += qty * px_iv0
        initial_model_surface += qty * px_surface
        engine = str(engine_name or "unknown")
        skip_engine = engine in set(cfg.debug_skip_engines or ())
        if cfg.debug_logging and not skip_engine:
            logger.info(
                (
                    "[ADV_SIM][REPRICE_INPUT] engine=%s leg=%d kind=%s symbol=%s qty=%.6f "
                    "pricing_model=%s strike=%s expiry=%s price0=%.8f iv0=%s px_iv0=%.8f px_surface_t0=%.8f "
                    "contrib_market=%.8f contrib_iv0=%.8f contrib_surface=%.8f"
                ),
                engine,
                idx,
                str(leg.get("kind")),
                str(leg.get("symbol", "")),
                qty,
                pricing_model,
                str(leg.get("strike")),
                str(leg.get("expiry")),
                px_market,
                str(leg.get("iv0")),
                px_iv0,
                px_surface,
                qty * px_market,
                qty * px_iv0,
                qty * px_surface,
            )
    if cfg.debug_logging and str(engine_name or "unknown") not in set(cfg.debug_skip_engines or ()):
        logger.info(
            (
                "[ADV_SIM][REPRICE_BASE] engine=%s anchor=%s baseline_market=%.8f baseline_model_iv0=%.8f "
                "baseline_model_surface=%.8f pricing_model=%s gap_model_iv0_minus_market=%.8f gap_model_surface_minus_market=%.8f"
            ),
            str(engine_name or "unknown"),
            str(cfg.repricing_anchor),
            initial_market,
            initial_model_iv0,
            initial_model_surface,
            pricing_model,
            initial_model_iv0 - initial_market,
            initial_model_surface - initial_market,
        )
    # NOTE: repricing_anchor is intentionally market_t0 to keep P&L mark-to-market consistent.
    for t in range(horizon):
        s_t = spot_paths[:, t]
        value = np.zeros(paths, dtype=float)
        for leg in legs:
            if leg["kind"] == "future":
                value += leg["qty"] * s_t
                continue
            expiry = leg.get("expiry")
            dte = max((expiry - today).days if isinstance(expiry, date) else 30, 0)
            tte = max(dte / 365.0 - ((t + 1) / TRADING_DAYS_PER_YEAR), 1.0 / 365.0)
            iv = _effective_iv_for_leg(leg, s_t, tte, surface_base, factor_shocks, t)
            if pricing_model == "bs76":
                fwd_t = s_t * np.exp((cfg.risk_free_rate - cfg.dividend_yield) * tte)
                px = _b76_price_vec(leg["flag"], fwd_t, float(leg["strike"]), tte, cfg.risk_free_rate, iv)
            else:
                px = _bs_price_vec(leg["flag"], s_t, float(leg["strike"]), tte, cfg.risk_free_rate, iv)
            value += leg["qty"] * px
        pnl[:, t] = value - initial_value
        if (
            cfg.debug_logging
            and str(engine_name or "unknown") not in set(cfg.debug_skip_engines or ())
            and (t == 0 or t == horizon - 1)
        ):
            logger.info(
                (
                    "[ADV_SIM][REPRICE_PATH] engine=%s day=%d value[min=%.8f mean=%.8f max=%.8f] "
                    "pnl[min=%.8f mean=%.8f max=%.8f]"
                ),
                str(engine_name or "unknown"),
                t + 1,
                float(np.min(value)),
                float(np.mean(value)),
                float(np.max(value)),
                float(np.min(pnl[:, t])),
                float(np.mean(pnl[:, t])),
                float(np.max(pnl[:, t])),
            )
    cost = max(cfg.tx_cost_per_leg + cfg.slippage_per_leg, 0.0) * float(len(legs))
    if cost > 0:
        pnl = pnl - cost
    return pnl


def _simulate_mode_greeks(
    legs: List[Dict[str, Any]],
    spot_paths: np.ndarray,
    spot0: float,
    surface_base: np.ndarray,
    factor_shocks: Optional[np.ndarray],
    cfg: AdvancedForwardConfig,
) -> np.ndarray:
    paths, horizon = spot_paths.shape
    pnl = np.zeros((paths, horizon), dtype=float)
    today = datetime.now().date()
    spot_prev = np.full(paths, float(max(spot0, 1e-6)), dtype=float)
    prev_sigmas: List[np.ndarray] = []
    prev_tte: List[float] = []
    for leg in legs:
        if leg["kind"] == "future":
            prev_sigmas.append(np.zeros(paths, dtype=float))
            prev_tte.append(0.0)
        else:
            expiry = leg.get("expiry")
            dte0 = max((expiry - today).days if isinstance(expiry, date) else 30, 0)
            tte0 = max(dte0 / 365.0, 1.0 / 365.0)
            prev_tte.append(tte0)
            prev_sigmas.append(np.full(paths, float(leg["iv0"]), dtype=float))

    cumulative = np.zeros(paths, dtype=float)
    for t in range(horizon):
        s_t = spot_paths[:, t]
        d_s = s_t - spot_prev
        inc = np.zeros(paths, dtype=float)
        for i, leg in enumerate(legs):
            if leg["kind"] == "future":
                inc += leg["qty"] * d_s
                continue
            tte_prev = max(prev_tte[i], 1.0 / 365.0)
            sigma_prev = np.clip(prev_sigmas[i], MIN_VOL, MAX_VOL)
            delta, gamma, vega, theta = _bs_greeks_vec(
                leg["flag"], spot_prev, float(leg["strike"]), tte_prev, cfg.risk_free_rate, sigma_prev
            )
            expiry = leg.get("expiry")
            dte = max((expiry - today).days if isinstance(expiry, date) else 30, 0)
            tte_curr = max(dte / 365.0 - ((t + 1) / TRADING_DAYS_PER_YEAR), 1.0 / 365.0)
            sigma_curr = _effective_iv_for_leg(leg, s_t, tte_curr, surface_base, factor_shocks, t)
            d_sigma = sigma_curr - sigma_prev
            leg_inc = delta * d_s + 0.5 * gamma * (d_s**2) + vega * d_sigma + theta * cfg.dt
            inc += leg["qty"] * leg_inc
            prev_sigmas[i] = sigma_curr
            prev_tte[i] = tte_curr
        cumulative += inc
        pnl[:, t] = cumulative
        spot_prev = s_t
    cost = max(cfg.tx_cost_per_leg + cfg.slippage_per_leg, 0.0) * float(len(legs))
    if cost > 0:
        pnl = pnl - cost
    return pnl


def _distribution_kpis(
    pnl_paths: np.ndarray,
    daily_loss_limit: Optional[float],
    total_loss_limit: Optional[float],
) -> Dict[str, Any]:
    term = pnl_paths[:, -1]
    q = {
        "p95": float(np.percentile(term, 95)),
        "p50": float(np.percentile(term, 50)),
        "p10": float(np.percentile(term, 10)),
        "p5": float(np.percentile(term, 5)),
        "p1": float(np.percentile(term, 1)),
        "p0_5": float(np.percentile(term, 0.5)),
    }
    var95 = q["p5"]
    var99 = q["p1"]
    es95_mask = term <= var95
    es99_mask = term <= var99
    es95 = float(np.mean(term[es95_mask])) if np.any(es95_mask) else float(var95)
    es99 = float(np.mean(term[es99_mask])) if np.any(es99_mask) else float(var99)
    daily_inc = np.diff(np.hstack([np.zeros((pnl_paths.shape[0], 1)), pnl_paths]), axis=1)
    if daily_loss_limit is not None:
        breach_daily = np.any(daily_inc <= -abs(float(daily_loss_limit)), axis=1)
    else:
        breach_daily = np.zeros(pnl_paths.shape[0], dtype=bool)
    if total_loss_limit is not None:
        breach_total_path = pnl_paths <= -abs(float(total_loss_limit))
        breach_total = np.any(breach_total_path, axis=1)
        first = np.where(breach_total_path, np.arange(1, pnl_paths.shape[1] + 1), np.inf)
        first_day = np.min(first, axis=1)
        first_day = first_day[np.isfinite(first_day)]
    else:
        breach_total = np.zeros(pnl_paths.shape[0], dtype=bool)
        first_day = np.array([], dtype=float)

    running_max = np.maximum.accumulate(np.hstack([np.zeros((pnl_paths.shape[0], 1)), pnl_paths]), axis=1)
    curve = np.hstack([np.zeros((pnl_paths.shape[0], 1)), pnl_paths])
    dd = curve - running_max
    max_dd = np.min(dd, axis=1)
    fan = {
        "p50": np.percentile(pnl_paths, 50, axis=0).tolist(),
        "p10": np.percentile(pnl_paths, 10, axis=0).tolist(),
        "p5": np.percentile(pnl_paths, 5, axis=0).tolist(),
        "p1": np.percentile(pnl_paths, 1, axis=0).tolist(),
    }

    mu = float(np.mean(term))
    std = float(np.std(term))
    centered = term - mu
    m2 = float(np.mean(centered**2))
    m3 = float(np.mean(centered**3))
    m4 = float(np.mean(centered**4))
    skew = (m3 / (m2 ** 1.5)) if m2 > 1e-12 else 0.0
    kurt = (m4 / (m2**2) - 3.0) if m2 > 1e-12 else 0.0

    return {
        "mean": mu,
        "median": float(np.median(term)),
        "stdev": std,
        "skew": float(skew),
        "kurtosis": float(kurt),
        "quantiles": q,
        "var95": var95,
        "var99": var99,
        "es95": es95,
        "es99": es99,
        "prob_loss": float(np.mean(term < 0.0)),
        "prob_breach_daily": float(np.mean(breach_daily)),
        "prob_breach_total": float(np.mean(breach_total)),
        "worst_path_pnl": float(np.min(term)),
        "max_drawdown": {
            "p50": float(np.percentile(max_dd, 50)),
            "p5": float(np.percentile(max_dd, 5)),
            "p1": float(np.percentile(max_dd, 1)),
        },
        "breach_time": {
            "p50": float(np.percentile(first_day, 50)) if first_day.size else None,
            "p10": float(np.percentile(first_day, 10)) if first_day.size else None,
            "p1": float(np.percentile(first_day, 1)) if first_day.size else None,
        },
        "fan": fan,
    }


def _approx_error_kpis(greeks_term: np.ndarray, full_term: np.ndarray) -> Dict[str, float]:
    diff = greeks_term - full_term
    abs_diff = np.abs(diff)
    return {
        "mae": float(np.mean(abs_diff)),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "p95_abs_error": float(np.percentile(abs_diff, 95)),
        "mean_signed_error": float(np.mean(diff)),
    }


def _empirical_cdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.sort(values.astype(float))
    n = len(x)
    y = np.arange(1, n + 1, dtype=float) / float(max(n, 1))
    return x, y


def _downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int = 700) -> Tuple[np.ndarray, np.ndarray]:
    n = len(x)
    if n <= max_points:
        return x, y
    idx = np.linspace(0, n - 1, max_points, dtype=int)
    return x[idx], y[idx]


def _build_overlay_plot_payload(engine_plot_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if not engine_plot_data:
        return {"cdf": [], "tail": [], "tail_xlim": None}
    p10_values = [float(p["p10"]) for p in engine_plot_data.values() if np.isfinite(float(p["p10"]))]
    x_min_values = [float(np.min(p["terminal"])) for p in engine_plot_data.values()]
    x_min = min(x_min_values) if x_min_values else -1.0
    x_max = min(p10_values) if p10_values else max(x_min + 1.0, -0.1)
    if not np.isfinite(x_max) or x_max <= x_min:
        x_max = x_min + max(1.0, abs(x_min) * 0.1)

    cdf_payload: List[Dict[str, Any]] = []
    tail_payload: List[Dict[str, Any]] = []
    for eng, payload in engine_plot_data.items():
        term = payload["terminal"]
        x, y = _empirical_cdf(term)
        x, y = _downsample_xy(x, y)
        cdf_payload.append(
            {
                "engine": eng,
                "x": x.astype(float).tolist(),
                "cdf": y.astype(float).tolist(),
                "var95": float(payload["var95"]),
                "var99": float(payload["var99"]),
            }
        )
        mask = (x >= x_min) & (x <= x_max)
        xt = x[mask]
        yt = y[mask]
        if xt.size == 0:
            xt = x[: min(80, len(x))]
            yt = y[: min(80, len(y))]
        ccdf = 1.0 - yt
        tail_payload.append(
            {
                "engine": eng,
                "x": xt.astype(float).tolist(),
                "cdf": yt.astype(float).tolist(),
                "ccdf": ccdf.astype(float).tolist(),
                "es99": float(payload["es99"]),
            }
        )
    return {
        "cdf": cdf_payload,
        "tail": tail_payload,
        "tail_xlim": [float(x_min), float(x_max)],
    }


def _save_overlay_plots(engine_plot_data: Dict[str, Dict[str, float]]) -> Dict[str, Optional[str]]:
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    cdf_path = out_dir / "overlay_cdf.png"
    tail_path = out_dir / "overlay_tail.png"
    if plt is None or not engine_plot_data:
        return {
            "overlay_cdf": str(cdf_path) if cdf_path.exists() else None,
            "overlay_tail": str(tail_path) if tail_path.exists() else None,
        }

    # Plot 1: overlay empirical CDF + VaR markers
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    for eng, payload in engine_plot_data.items():
        term = payload["terminal"]
        x, y = _empirical_cdf(term)
        line = ax1.plot(x, y, linewidth=1.8, label=eng.upper())[0]
        color = line.get_color()
        ax1.axvline(float(payload["var99"]), color=color, linestyle="--", linewidth=1.2, alpha=0.55)
        ax1.axvline(float(payload["var95"]), color=color, linestyle=":", linewidth=1.2, alpha=0.55)
    ax1.set_title("Empirical CDF Overlay by Engine")
    ax1.set_xlabel("Terminal P&L")
    ax1.set_ylabel("CDF")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="lower right", ncol=2, fontsize=9)
    fig1.tight_layout()
    fig1.savefig(cdf_path, dpi=150)
    plt.close(fig1)

    # Plot 2: tail-only zoom CDF + CCDF with ES99 annotations
    p10_values = [float(p["p10"]) for p in engine_plot_data.values() if np.isfinite(float(p["p10"]))]
    x_min_values = [float(np.min(p["terminal"])) for p in engine_plot_data.values()]
    x_min = min(x_min_values) if x_min_values else -1.0
    x_max = min(p10_values) if p10_values else max(x_min + 1.0, -0.1)
    if not np.isfinite(x_max) or x_max <= x_min:
        x_max = x_min + max(1.0, abs(x_min) * 0.1)

    fig2, (ax_cdf, ax_ccdf) = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    for eng, payload in engine_plot_data.items():
        term = payload["terminal"]
        x, y = _empirical_cdf(term)
        ccdf = 1.0 - y
        line = ax_cdf.plot(x, y, linewidth=1.8, label=eng.upper())[0]
        color = line.get_color()
        ax_ccdf.plot(x, ccdf, linewidth=1.8, label=eng.upper(), color=color)
        es99 = float(payload["es99"])
        ax_cdf.axvline(es99, color=color, linestyle="--", linewidth=1.1, alpha=0.6)
        ax_ccdf.axvline(es99, color=color, linestyle="--", linewidth=1.1, alpha=0.6)
        ax_cdf.annotate(f"{eng.upper()} ES99", xy=(es99, 0.08), xytext=(4, 4), textcoords="offset points", color=color, fontsize=8, rotation=90)
    ax_cdf.set_xlim(x_min, x_max)
    ax_ccdf.set_xlim(x_min, x_max)
    ax_cdf.set_title("Tail CDF Zoom")
    ax_ccdf.set_title("Tail CCDF Zoom")
    ax_cdf.set_xlabel("Terminal P&L")
    ax_ccdf.set_xlabel("Terminal P&L")
    ax_cdf.set_ylabel("CDF")
    ax_ccdf.set_ylabel("CCDF")
    ax_cdf.grid(alpha=0.25)
    ax_ccdf.grid(alpha=0.25)
    ax_cdf.legend(loc="lower right", fontsize=9)
    fig2.tight_layout()
    fig2.savefig(tail_path, dpi=150)
    plt.close(fig2)

    return {"overlay_cdf": str(cdf_path), "overlay_tail": str(tail_path)}


def run_advanced_forward_risk(
    positions: List[Dict[str, Any]],
    options_df: pd.DataFrame,
    spot_history_df: pd.DataFrame,
    spot: float,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = AdvancedForwardConfig(**(config or {}))
    rng_master = np.random.default_rng(cfg.seed)

    returns_hist = _prepare_returns(spot_history_df)
    if returns_hist.size == 0:
        sigma_fallback = 0.20 / math.sqrt(TRADING_DAYS_PER_YEAR)
        returns_hist = rng_master.normal(0.0, sigma_fallback, size=260)
    evt = _fit_evt_left_tail((returns_hist - np.mean(returns_hist)) / max(np.std(returns_hist), 1e-8))
    legs = _build_legs(positions, options_df, spot, cfg.risk_free_rate)
    if cfg.debug_logging and cfg.debug_mode_filter in (None, "all"):
        _log_leg_inputs(legs, spot, cfg)
    if not legs:
        return {
            "status": "no_legs",
            "config": cfg.__dict__,
            "engines": {},
            "summary_rows": [],
        }

    surface_fit = _fit_surface_factors(options_df, spot, cfg.risk_free_rate)
    base_surface = np.asarray(surface_fit["base"], dtype=float)
    engines_out: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []
    engine_plot_data: Dict[str, Dict[str, Any]] = {}
    repricing_sanity: Dict[str, Dict[str, float]] = {}

    engine_order = list(dict.fromkeys(["gbm", *list(cfg.engines)]))
    for idx, eng in enumerate(engine_order):
        rng = np.random.default_rng(cfg.seed + 101 * (idx + 1))
        if eng == "fhs":
            engine_sim = _simulate_engine_fhs(spot, returns_hist, cfg, rng, evt)
        elif eng == "garch":
            engine_sim = _simulate_engine_garch_family("garch", spot, returns_hist, cfg, rng, evt)
        elif eng == "egarch":
            engine_sim = _simulate_engine_garch_family("egarch", spot, returns_hist, cfg, rng, evt)
        elif eng == "gjr":
            engine_sim = _simulate_engine_garch_family("gjr", spot, returns_hist, cfg, rng, evt)
        elif eng == "heston":
            engine_sim = _simulate_engine_heston(spot, cfg, rng, with_jumps=False)
        elif eng == "bates":
            engine_sim = _simulate_engine_heston(spot, cfg, rng, with_jumps=True)
        else:
            engine_sim = _simulate_engine_gbm(spot, returns_hist, cfg, rng, evt)

        spot_paths = engine_sim["spot_paths"]
        returns_paths = engine_sim["returns_paths"]
        factor_shocks = None
        if cfg.simulate_surface:
            factor_shocks = _simulate_surface_factor_shocks(returns_paths, cfg, rng)
        if cfg.debug_logging and cfg.debug_mode_filter in (None, "all") and eng not in set(cfg.debug_skip_engines or ()):
            _log_engine_inputs(eng, spot_paths, returns_paths, factor_shocks, cfg)

        for iv_rule in cfg.iv_rules:
            use_surface = str(iv_rule).lower() == "surface"
            factor_shocks_eff = factor_shocks if use_surface else None
            mode_results: Dict[str, Any] = {}
            pnl_repricing_bs = None
            pnl_greeks = None
            scenario_engine = f"{eng}|{iv_rule}"

            for mode in cfg.pnl_modes:
                if mode == "repricing":
                    for model_name in cfg.repricing_models:
                        mode_key = f"repricing_{model_name}"
                        pnl_repricing = _simulate_mode_repricing(
                            legs,
                            spot_paths,
                            spot,
                            base_surface,
                            factor_shocks_eff,
                            cfg,
                            engine_name=scenario_engine,
                            pricing_model=model_name,
                        )
                        if model_name == "bs":
                            pnl_repricing_bs = pnl_repricing
                        k = _distribution_kpis(pnl_repricing, cfg.daily_loss_limit, cfg.total_loss_limit)
                        term = pnl_repricing[:, -1]
                        if (
                            cfg.debug_logging
                            and (cfg.debug_mode_filter in (None, "repricing"))
                            and eng not in set(cfg.debug_skip_engines or ())
                        ):
                            sorted_term = np.sort(term)
                            n = sorted_term.size
                            lo = sorted_term[: min(5, n)].tolist() if n else []
                            hi = sorted_term[max(0, n - 5):].tolist() if n else []
                            neg_count = int(np.sum(term < 0))
                            logger.info(
                                (
                                    "[ADV_SIM][TERM_DEBUG] engine=%s iv_rule=%s mode=%s n=%d neg_count=%d neg_pct=%.6f "
                                    "min=%.8f p1=%.8f p5=%.8f median=%.8f p95=%.8f p99=%.8f max=%.8f "
                                    "sample_low=%s sample_high=%s"
                                ),
                                eng,
                                iv_rule,
                                mode_key,
                                n,
                                neg_count,
                                (neg_count / n) if n else 0.0,
                                float(np.min(term)) if n else float("nan"),
                                float(np.percentile(term, 1)) if n else float("nan"),
                                float(np.percentile(term, 5)) if n else float("nan"),
                                float(np.percentile(term, 50)) if n else float("nan"),
                                float(np.percentile(term, 95)) if n else float("nan"),
                                float(np.percentile(term, 99)) if n else float("nan"),
                                float(np.max(term)) if n else float("nan"),
                                lo,
                                hi,
                            )
                        if term.size > 3000:
                            pick = np.linspace(0, term.size - 1, 3000, dtype=int)
                            term_sample = term[pick]
                        else:
                            term_sample = term
                        mode_results[mode_key] = {
                            "kpis": k,
                            "terminal_pnl_sample": term_sample.astype(float).tolist(),
                            "mode_base": "repricing",
                            "pricing_model": model_name,
                            "iv_rule": iv_rule,
                        }
                else:
                    mode_key = "greeks"
                    pnl_greeks = _simulate_mode_greeks(legs, spot_paths, spot, base_surface, factor_shocks_eff, cfg)
                    k = _distribution_kpis(pnl_greeks, cfg.daily_loss_limit, cfg.total_loss_limit)
                    term = pnl_greeks[:, -1]
                    if term.size > 3000:
                        pick = np.linspace(0, term.size - 1, 3000, dtype=int)
                        term_sample = term[pick]
                    else:
                        term_sample = term
                    mode_results[mode_key] = {
                        "kpis": k,
                        "terminal_pnl_sample": term_sample.astype(float).tolist(),
                        "mode_base": "greeks",
                        "pricing_model": "bs",
                        "iv_rule": iv_rule,
                    }

            approx_error = None
            if pnl_repricing_bs is not None and pnl_greeks is not None:
                approx_error = _approx_error_kpis(pnl_greeks[:, -1], pnl_repricing_bs[:, -1])
                if "greeks" in mode_results:
                    mode_results["greeks"]["kpis"]["approx_error_vs_repricing"] = approx_error

            fit_block = dict(engine_sim.get("fit") or {})
            fit_block["surface_iv_rmse"] = surface_fit.get("rmse")
            fit_block["surface_points"] = surface_fit.get("points")
            fit_block["iv_rule"] = iv_rule
            if factor_shocks_eff is not None:
                fit_block["surface_factor_stability"] = {
                    "level_std": float(np.std(factor_shocks_eff[:, :, 0])),
                    "skew_std": float(np.std(factor_shocks_eff[:, :, 1])),
                    "curvature_std": float(np.std(factor_shocks_eff[:, :, 2])),
                }
            else:
                fit_block["surface_factor_stability"] = None

            engines_out[scenario_engine] = {
                "engine_base": eng,
                "iv_rule": iv_rule,
                "fit_kpis": fit_block,
                "modes": mode_results,
            }

            # Overlay default: repricing BS if present, then repricing BS76, then greeks.
            overlay_key = "repricing_bs" if "repricing_bs" in mode_results else ("repricing_bs76" if "repricing_bs76" in mode_results else "greeks")
            overlay_payload = mode_results.get(overlay_key) or {}
            term = np.asarray(overlay_payload.get("terminal_pnl_sample") or [], dtype=float)
            k_sel = overlay_payload.get("kpis") or {}
            if term.size:
                engine_plot_data[scenario_engine] = {
                    "terminal": term.astype(float),
                    "var95": float(k_sel.get("var95")),
                    "var99": float(k_sel.get("var99")),
                    "es99": float(k_sel.get("es99")),
                    "p10": float((k_sel.get("quantiles") or {}).get("p10", np.percentile(term, 10))),
                }
            if "repricing_bs" in mode_results:
                kr = mode_results["repricing_bs"]["kpis"]
                repricing_sanity[scenario_engine] = {
                    "p5": float((kr.get("quantiles") or {}).get("p5", kr.get("var95", np.nan))),
                    "p1": float((kr.get("quantiles") or {}).get("p1", kr.get("var99", np.nan))),
                    "prob_loss": float(kr.get("prob_loss", np.nan)),
                    "worst_path_pnl": float(kr.get("worst_path_pnl", np.nan)),
                    "mean": float(kr.get("mean", np.nan)),
                }

            for mode_key, payload in mode_results.items():
                k = payload["kpis"]
                row = {
                    "engine": scenario_engine,
                    "engine_base": eng,
                    "mode": payload.get("mode_base", mode_key),
                    "mode_variant": mode_key,
                    "iv_rule": iv_rule,
                    "pricing_model": payload.get("pricing_model", "bs"),
                    "mean": k["mean"],
                    "median": k["median"],
                    "stdev": k["stdev"],
                    "var95": k["var95"],
                    "var99": k["var99"],
                    "es95": k["es95"],
                    "es99": k["es99"],
                    "p50": k["quantiles"]["p50"],
                    "p10": k["quantiles"]["p10"],
                    "p5": k["quantiles"]["p5"],
                    "p1": k["quantiles"]["p1"],
                    "p0_5": k["quantiles"]["p0_5"],
                    "prob_loss": k["prob_loss"],
                    "prob_breach_daily": k["prob_breach_daily"],
                    "prob_breach_total": k["prob_breach_total"],
                    "worst_path_pnl": k["worst_path_pnl"],
                    "maxdd_p50": k["max_drawdown"]["p50"],
                    "maxdd_p5": k["max_drawdown"]["p5"],
                    "maxdd_p1": k["max_drawdown"]["p1"],
                    "breach_t_p50": k["breach_time"]["p50"],
                    "breach_t_p10": k["breach_time"]["p10"],
                    "breach_t_p1": k["breach_time"]["p1"],
                    "vol_mae": fit_block.get("vol_mae"),
                    "vol_rmse": fit_block.get("vol_rmse"),
                    "surface_iv_rmse": fit_block.get("surface_iv_rmse"),
                }
                if approx_error and mode_key == "greeks":
                    row["greeks_err_mae"] = approx_error["mae"]
                    row["greeks_err_rmse"] = approx_error["rmse"]
                    row["greeks_err_p95_abs"] = approx_error["p95_abs_error"]
                summary_rows.append(row)

    artifacts = _save_overlay_plots(engine_plot_data)
    overlay_plots = _build_overlay_plot_payload(engine_plot_data)

    return {
        "model_version": ADV_MC_VERSION,
        "status": "ok",
        "config": cfg.__dict__,
        "surface_model": {
            "name": "svi_style_factorized_quadratic_proxy",
            "base_factors": {
                "level": float(base_surface[0]),
                "skew": float(base_surface[1]),
                "curvature": float(base_surface[2]),
                "term": float(base_surface[3]),
            },
            "fit_rmse": surface_fit.get("rmse"),
            "fit_points": surface_fit.get("points"),
        },
        "evt_overlay": evt if cfg.use_evt_overlay else None,
        "artifacts": artifacts,
        "overlay_plots": overlay_plots,
        "repricing_sanity": repricing_sanity,
        "engines": engines_out,
        "summary_rows": summary_rows,
    }
