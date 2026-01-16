"""Greeks calculation utilities using Black-Scholes model."""

import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional, Dict, List
try:
    import streamlit as st
except Exception:
    st = None

try:
    from py_vollib.black_scholes import black_scholes as bs
    from py_vollib.black_scholes.implied_volatility import implied_volatility as iv
    from py_vollib.black_scholes.greeks import analytical as greeks
except Exception:
    bs = iv = greeks = None

from .parsers import parse_tradingsymbol


RISK_FREE_RATE = 0.06
CALENDAR_DAYS_PER_YEAR = 365.0


def calculate_implied_volatility(
    option_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    option_type: str,
    risk_free_rate: float = RISK_FREE_RATE,
) -> Optional[float]:
    """Calculate implied volatility using py_vollib.
    
    Args:
        option_price: Current option price
        spot: Spot price of underlying
        strike: Strike price
        time_to_expiry: Time to expiry in years
        option_type: 'CE' for call, 'PE' for put
        risk_free_rate: Risk-free interest rate (default 7%)
    
    Returns:
        Implied volatility as decimal (e.g., 0.20 for 20%)
        None if calculation fails
    """
    if iv is None:
        return None
    
    try:
        flag = 'c' if option_type == 'CE' else 'p'
        calculated_iv = iv(option_price, spot, strike, time_to_expiry, risk_free_rate, flag)
        return calculated_iv
    except Exception as e:
        return None


def calculate_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    implied_vol: float,
    option_type: str,
    risk_free_rate: float = RISK_FREE_RATE,
) -> Dict[str, float]:
    """Calculate option greeks using py_vollib.
    
    Args:
        spot: Spot price of underlying
        strike: Strike price
        time_to_expiry: Time to expiry in years
        implied_vol: Implied volatility
        option_type: 'CE' for call, 'PE' for put
        risk_free_rate: Risk-free interest rate (default 7%)
    
    Returns:
        Dict with keys: delta, gamma, vega, theta, vega_raw, theta_raw
    """
    if greeks is None:
        return {"delta": 0, "gamma": 0, "vega": 0, "theta": 0}
    
    try:
        flag = 'c' if option_type == 'CE' else 'p'
        
        print(f"DEBUG calculate_greeks: flag={flag}, spot={spot}, strike={strike}, t={time_to_expiry}, r={risk_free_rate}, iv={implied_vol}")
        
        # Calculate greeks using py_vollib
        delta = greeks.delta(flag, spot, strike, time_to_expiry, risk_free_rate, implied_vol)
        gamma = greeks.gamma(flag, spot, strike, time_to_expiry, risk_free_rate, implied_vol)
        vega_raw = greeks.vega(flag, spot, strike, time_to_expiry, risk_free_rate, implied_vol)
        theta_raw = greeks.theta(flag, spot, strike, time_to_expiry, risk_free_rate, implied_vol)
        
        print(f"DEBUG calculate_greeks result: delta={delta}, gamma={gamma}, vega={vega_raw}, theta={theta_raw}")

        # py_vollib returns vega per 1% vol (0.01) and theta per day.
        # Keep theta as-is to match broker UI units.
        try:
            vega_per_pct = float(vega_raw)
        except Exception:
            vega_per_pct = float(vega_raw or 0)
        try:
            theta_per_day = float(theta_raw)
        except Exception:
            theta_per_day = float(theta_raw or 0)

        return {
            "delta": delta,
            "gamma": gamma,
            # Standardized units for consumers (per 1% vol, per day)
            "vega": vega_per_pct,
            "theta": theta_per_day,
            # Keep raw values for debugging
            "vega_raw": vega_raw,
            "theta_raw": theta_raw
        }
    except Exception as e:
        print(f"DEBUG: calculate_greeks exception: {e}")
        return {"delta": 0, "gamma": 0, "vega": 0, "theta": 0}


def enrich_position_with_greeks(
    position: Dict, 
    options_df: pd.DataFrame, 
    current_spot: float
) -> Dict:
    """Enrich a single Kite position with parsed data + Greeks.
    
    Args:
        position: Dict containing position data from Kite
        options_df: DataFrame with options market data
        current_spot: Current spot price of underlying
    
    Returns:
        Enhanced position dict with Greeks and parsed fields
    """
    symbol = position.get("tradingsymbol", "")
    if st is not None:
        spot_override = st.session_state.get("greeks_spot_override")
        if spot_override is None:
            spot_override = st.session_state.get("stress_spot_override")
        if spot_override is not None:
            current_spot = float(spot_override)
    parsed = parse_tradingsymbol(symbol)
    
    if not parsed:
        print(f"DEBUG: Failed to parse symbol: {symbol}")
        return {**position, "error": "Could not parse symbol"}
    
    strike = parsed["strike"]
    expiry = parsed["expiry"]
    option_type = parsed["option_type"]
    
    # Calculate time to expiry
    today = datetime.now()
    dte = max((expiry - today).days, 0)
    expiry_dt = expiry
    try:
        if expiry_dt.tzinfo is None:
            expiry_dt = expiry_dt.replace(tzinfo=ZoneInfo("Asia/Kolkata"))
        expiry_dt = expiry_dt.replace(hour=15, minute=30, second=0, microsecond=0)
        now_dt = datetime.now(ZoneInfo("Asia/Kolkata"))
        minutes_to_expiry = max((expiry_dt - now_dt).total_seconds() / 60.0, 0.0)
        time_to_expiry = max((minutes_to_expiry / (60.0 * 24.0)) / CALENDAR_DAYS_PER_YEAR, 1.0 / CALENDAR_DAYS_PER_YEAR)
    except Exception:
        time_to_expiry = max(dte / CALENDAR_DAYS_PER_YEAR, 1.0 / CALENDAR_DAYS_PER_YEAR)
    
    # Get current option price
    last_price = position.get("last_price", 0)
    
    # Debug: Print position details
    print(f"DEBUG: Processing {symbol}")
    print(f"  - Parsed: strike={strike}, expiry={expiry.date()}, type={option_type}")
    print(f"  - DTE={dte}, spot={current_spot}, last_price={last_price}")
    
    # Use position last_price only for IV calculation.
    option_price = last_price
    
    # Calculate IV if not sourced from options_df
    implied_vol = calculate_implied_volatility(
        option_price, current_spot, strike, time_to_expiry, option_type
    )
    
    print(f"  - Option price used: {option_price}, calculated IV: {implied_vol}")
    
    if implied_vol is None or implied_vol <= 0:
        implied_vol = 0.20  # Default 20% if calculation fails
        print(f"  - Using default IV: {implied_vol}")
    
    # Calculate Greeks
    position_greeks = calculate_greeks(
        current_spot, strike, time_to_expiry, implied_vol, option_type
    )
    
    print(f"  - Calculated greeks: delta={position_greeks.get('delta')}, gamma={position_greeks.get('gamma')}, vega={position_greeks.get('vega')}, theta={position_greeks.get('theta')}")

    # Determine position size (net quantity, negative for short)
    quantity = position.get("quantity", 0) or 0

    # Scale Greeks by quantity
    scaled_greeks = {}
    for k, v in position_greeks.items():
        # only scale the primary greek names (delta,gamma,vega,theta)
        if k in ("delta", "gamma", "vega", "theta"):
            scaled_greeks[f"position_{k}"] = v * quantity
        else:
            # keep any raw helpers intact
            scaled_greeks[k] = v

    # Debug output
    try:
        print(f"DEBUG: {symbol} qty={quantity} "
              f"greeks_raw={{{k: position_greeks.get(k) for k in ('delta','gamma','vega','theta')}}} "
              f"scaled={{{k: scaled_greeks.get('position_'+k) for k in ('delta','gamma','vega','theta')}}}")
    except Exception:
        pass
    
    return {
        **position,
        "strike": strike,
        "expiry": expiry,
        "option_type": option_type,
        "dte": dte,
        "time_to_expiry": time_to_expiry,
        "implied_vol": implied_vol,
        "spot_price": current_spot,
        "iv_debug": {
            "spot_used": current_spot,
            "option_price_used": option_price,
            "time_to_expiry": time_to_expiry,
            "expiry_date": expiry.date().isoformat(),
            "match_count": 0,
        },
        # include raw greeks and scaled position-level greeks
        **position_greeks,
        **scaled_greeks
    }
