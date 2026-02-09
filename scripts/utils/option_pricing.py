"""Option pricing helpers for scenario repricing."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

import pandas as pd

try:
    from py_vollib.black_scholes import black_scholes as bs
except Exception:
    bs = None


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black76_price(
    flag: str,
    forward: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    implied_vol: float,
) -> Optional[float]:
    if forward <= 0 or strike <= 0:
        return None
    t = max(float(time_to_expiry), 0.0)
    vol = max(float(implied_vol), 0.0)
    if t == 0.0 or vol == 0.0:
        intrinsic = max(0.0, forward - strike) if flag == "c" else max(0.0, strike - forward)
        return math.exp(-risk_free_rate * t) * intrinsic
    sigma_sqrt_t = vol * math.sqrt(t)
    d1 = (math.log(forward / strike) + 0.5 * sigma_sqrt_t * sigma_sqrt_t) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    if flag == "c":
        price = forward * _norm_cdf(d1) - strike * _norm_cdf(d2)
    else:
        price = strike * _norm_cdf(-d2) - forward * _norm_cdf(-d1)
    return math.exp(-risk_free_rate * t) * price


def price_option(
    flag: str,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    implied_vol: float,
    model: str = "black76",
    forward: Optional[float] = None,
) -> Optional[float]:
    model_key = (model or "").lower().replace("-", "_")
    if model_key in {"black76", "black_76", "black"}:
        fwd = float(forward) if forward is not None else float(spot) * math.exp(risk_free_rate * time_to_expiry)
        return black76_price(flag, fwd, strike, time_to_expiry, risk_free_rate, implied_vol)
    if model_key in {"black_scholes", "bs", "blackscholes"}:
        if bs is None:
            return None
        return bs(flag, spot, strike, time_to_expiry, risk_free_rate, implied_vol)
    return None


def pricing_model_available(model: str) -> bool:
    model_key = (model or "").lower().replace("-", "_")
    if model_key in {"black76", "black_76", "black"}:
        return True
    if model_key in {"black_scholes", "bs", "blackscholes"}:
        return bs is not None
    return False


def infer_forward_from_futures_df(futures_df: pd.DataFrame, spot: float) -> Optional[float]:
    if futures_df is None or futures_df.empty:
        return None
    df = futures_df.copy()
    expiry_col = None
    if "expiry" in df.columns:
        expiry_col = "expiry"
    elif "expiry_date" in df.columns:
        expiry_col = "expiry_date"

    if expiry_col:
        df[expiry_col] = pd.to_datetime(df[expiry_col], errors="coerce").dt.date
        df = df.dropna(subset=[expiry_col])
        if df.empty:
            return None
        today = datetime.now().date()
        future_df = df[df[expiry_col] >= today]
        if future_df.empty:
            future_df = df
        nearest_expiry = min(future_df[expiry_col])
        df = future_df[future_df[expiry_col] == nearest_expiry]

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")

    if df.empty:
        return None

    row = df.iloc[-1]
    for col in ("ltp", "close", "settle_price", "prev_close"):
        try:
            val = float(row.get(col, 0) or 0)
        except Exception:
            val = 0.0
        if val > 0:
            return val

    if spot and spot > 0:
        return float(spot)
    return None
