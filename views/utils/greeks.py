"""Greeks calculation utilities using Black-Scholes model."""

import pandas as pd
from datetime import datetime
from typing import Optional, Dict, List

try:
    from py_vollib.black_scholes import black_scholes as bs
    from py_vollib.black_scholes.implied_volatility import implied_volatility as iv
    from py_vollib.black_scholes.greeks import analytical as greeks
except Exception:
    bs = iv = greeks = None

from .parsers import parse_tradingsymbol


def calculate_implied_volatility(
    option_price: float, 
    spot: float, 
    strike: float, 
    time_to_expiry: float, 
    option_type: str, 
    risk_free_rate: float = 0.07
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
    risk_free_rate: float = 0.07
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
        
        # Calculate greeks using py_vollib
        delta = greeks.delta(flag, spot, strike, time_to_expiry, risk_free_rate, implied_vol)
        gamma = greeks.gamma(flag, spot, strike, time_to_expiry, risk_free_rate, implied_vol)
        vega_raw = greeks.vega(flag, spot, strike, time_to_expiry, risk_free_rate, implied_vol)
        theta_raw = greeks.theta(flag, spot, strike, time_to_expiry, risk_free_rate, implied_vol)

        # py_vollib returns vega as change in price for a change in volatility of 1.0
        # (i.e. 100 percentage points). Most UIs report vega per 1% change in vol.
        # Likewise theta from py_vollib is typically per-year; convert to per-day.
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
    parsed = parse_tradingsymbol(symbol)
    
    if not parsed:
        return {**position, "error": "Could not parse symbol"}
    
    strike = parsed["strike"]
    expiry = parsed["expiry"]
    option_type = parsed["option_type"]
    
    # Calculate time to expiry
    today = datetime.now()
    dte = (expiry - today).days
    time_to_expiry = max(dte / 365.0, 0.001)  # Avoid division by zero
    
    # Get current option price
    last_price = position.get("last_price", 0)
    
    # Try to get IV from options_df if available
    matching_options = options_df[
        (options_df["strike_price"] == strike) &
        (options_df["expiry"] == expiry) &
        (options_df["option_type"] == option_type)
    ]
    
    if not matching_options.empty:
        latest = matching_options.sort_values("date", ascending=False).iloc[0]
        option_price = latest.get("ltp", last_price) or last_price
    else:
        option_price = last_price
    
    # Calculate IV
    implied_vol = calculate_implied_volatility(
        option_price, current_spot, strike, time_to_expiry, option_type
    )
    
    if implied_vol is None or implied_vol <= 0:
        implied_vol = 0.20  # Default 20% if calculation fails
    
    # Calculate Greeks
    position_greeks = calculate_greeks(
        current_spot, strike, time_to_expiry, implied_vol, option_type
    )

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
        # include raw greeks and scaled position-level greeks
        **position_greeks,
        **scaled_greeks
    }
