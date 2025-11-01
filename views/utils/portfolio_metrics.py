"""Portfolio metrics and market regime calculations."""

import numpy as np
import pandas as pd
import streamlit as st
from typing import List, Dict
from .greeks import calculate_implied_volatility


def calculate_portfolio_greeks(enriched_positions: List[Dict]) -> Dict:
    """Aggregate Greeks across all positions."""
    if not enriched_positions:
        return {
            "net_delta": 0,
            "net_gamma": 0,
            "net_vega": 0,
            "net_theta": 0,
            "total_positions": 0
        }
    
    net_delta = sum(p.get("position_delta", 0) for p in enriched_positions)
    net_gamma = sum(p.get("position_gamma", 0) for p in enriched_positions)
    net_vega = sum(p.get("position_vega", 0) for p in enriched_positions)
    net_theta = sum(p.get("position_theta", 0) for p in enriched_positions)
    
    return {
        "net_delta": net_delta,
        "net_gamma": net_gamma,
        "net_vega": net_vega,
        "net_theta": net_theta,
        "total_positions": len(enriched_positions)
    }


def calculate_market_regime(options_df: pd.DataFrame, nifty_df: pd.DataFrame) -> Dict:
    """Calculate comprehensive market regime indicators including PCR, skew, term structure, and trend."""
    if options_df.empty or nifty_df.empty:
        return {}
    
    # Get latest ATM IV
    latest_date = options_df["date"].max()
    latest_options = options_df[options_df["date"] == latest_date]
    
    if latest_options.empty:
        return {}
    
    # Determine current spot — prefer NIFTY price from nifty_df if available
    current_spot = None
    if nifty_df is not None and not nifty_df.empty and "close" in nifty_df.columns:
        try:
            last_close = nifty_df["close"].dropna().iloc[-1]
            current_spot = float(last_close)
            print(f"DEBUG: using spot from nifty_df.close = {current_spot}")
        except Exception as e:
            print(f"DEBUG: failed to extract spot from nifty_df: {e}")
            current_spot = None

    # Fallback to options underlying_value if nifty_df doesn't provide a valid spot
    if current_spot is None:
        try:
            current_spot = latest_options["underlying_value"].iloc[0]
            if pd.isna(current_spot):
                raise ValueError("underlying_value is NaN")
            print(f"DEBUG: using spot from options_df.underlying_value = {current_spot}")
        except Exception as e:
            msg = "DEBUG: latest spot missing from both nifty_df.close and options_df.underlying_value — cannot compute market regime"
            print(msg)
            try:
                st.warning("Latest underlying spot is missing in both NIFTY prices and options data. Ensure upstream data ingestion includes a valid spot price.")
            except Exception:
                pass
            return {}
    
    # Find ATM options
    atm_strike = round(current_spot / 50) * 50  # Round to nearest 50
    atm_options = latest_options[
        (latest_options["strike_price"] >= atm_strike - 100) &
        (latest_options["strike_price"] <= atm_strike + 100)
    ]
    
    # Calculate IV from ATM options
    current_iv_list = []
    for _, row in atm_options.iterrows():
        try:
            opt_iv = calculate_implied_volatility(
                row["ltp"], current_spot, row["strike_price"],
                0.027, row["option_type"]  # Approximate 10 days
            )
            if opt_iv and opt_iv > 0:
                current_iv_list.append(opt_iv)
        except:
            continue
    
    current_iv = np.mean(current_iv_list) if current_iv_list else 0.20
    
    # Calculate historical IV over last 90 days
    hist_ivs = []
    skipped_days = []
    for date in options_df["date"].unique()[-90:]:
        day_options = options_df[options_df["date"] == date]
        if day_options.empty:
            continue

        day_spot = day_options["underlying_value"].iloc[0]
        if pd.isna(day_spot):
            skipped_days.append(date)
            print(f"DEBUG: skipping date {date} because underlying_value is NaN")
            continue
        day_atm = round(day_spot / 50) * 50
        day_atm_options = day_options[
            (day_options["strike_price"] >= day_atm - 100) &
            (day_options["strike_price"] <= day_atm + 100)
        ]
        
        for _, row in day_atm_options.iterrows():
            try:
                opt_iv = calculate_implied_volatility(
                    row["ltp"], day_spot, row["strike_price"],
                    0.027, row["option_type"]
                )
                if opt_iv and opt_iv > 0:
                    hist_ivs.append(opt_iv)
            except:
                continue
    
    if hist_ivs:
        iv_min = min(hist_ivs)
        iv_max = max(hist_ivs)
        iv_rank = (current_iv - iv_min) / (iv_max - iv_min) * 100 if iv_max > iv_min else 50
    else:
        iv_rank = 50

    if skipped_days:
        print(f"DEBUG: skipped {len(skipped_days)} historical days due to missing underlying_value. Examples: {skipped_days[:5]}")
    
    # Calculate realized volatility from NIFTY daily
    if "close" in nifty_df.columns and len(nifty_df) > 30:
        returns = nifty_df["close"].pct_change().dropna()[-30:]
        realized_vol = returns.std() * np.sqrt(252)
    else:
        realized_vol = 0.15
    
    # VRP = IV - RV
    vrp = current_iv - realized_vol
    
    # ========== 1. PUT-CALL RATIO (PCR) ==========
    # Calculate by volume and by open interest
    pcr_volume = 0
    pcr_oi = 0
    try:
        puts = latest_options[latest_options["option_type"] == "PE"]
        calls = latest_options[latest_options["option_type"] == "CE"]
        
        # PCR by volume (if available)
        if "no_of_contracts" in latest_options.columns:
            put_volume = puts["no_of_contracts"].sum()
            call_volume = calls["no_of_contracts"].sum()
            pcr_volume = put_volume / call_volume if call_volume > 0 else 0
        
        # PCR by OI
        if "open_int" in latest_options.columns:
            put_oi = puts["open_int"].sum()
            call_oi = calls["open_int"].sum()
            pcr_oi = put_oi / call_oi if call_oi > 0 else 0
    except Exception as e:
        print(f"DEBUG: PCR calculation failed: {e}")
    
    # ========== 2. VOLATILITY SKEW (25-delta risk reversal) ==========
    skew = 0
    put_25d_iv = 0
    call_25d_iv = 0
    try:
        # Find 25-delta strikes (approximately OTM by 3-5%)
        otm_put_strike = atm_strike - (atm_strike * 0.03)  # ~3% OTM put
        otm_call_strike = atm_strike + (atm_strike * 0.03)  # ~3% OTM call
        
        # Get closest strikes
        put_options = latest_options[
            (latest_options["option_type"] == "PE") &
            (latest_options["strike_price"] >= otm_put_strike - 100) &
            (latest_options["strike_price"] <= otm_put_strike + 100)
        ]
        call_options = latest_options[
            (latest_options["option_type"] == "CE") &
            (latest_options["strike_price"] >= otm_call_strike - 100) &
            (latest_options["strike_price"] <= otm_call_strike + 100)
        ]
        
        # Calculate IVs for OTM options
        put_ivs = []
        for _, row in put_options.iterrows():
            try:
                put_iv = calculate_implied_volatility(
                    row["ltp"], current_spot, row["strike_price"], 0.027, "PE"
                )
                if put_iv and put_iv > 0:
                    put_ivs.append(put_iv)
            except:
                continue
        
        call_ivs = []
        for _, row in call_options.iterrows():
            try:
                call_iv = calculate_implied_volatility(
                    row["ltp"], current_spot, row["strike_price"], 0.027, "CE"
                )
                if call_iv and call_iv > 0:
                    call_ivs.append(call_iv)
            except:
                continue
        
        if put_ivs and call_ivs:
            put_25d_iv = np.mean(put_ivs)
            call_25d_iv = np.mean(call_ivs)
            skew = put_25d_iv - call_25d_iv  # Positive = put skew (fear)
    except Exception as e:
        print(f"DEBUG: Skew calculation failed: {e}")
    
    # ========== 3. TERM STRUCTURE ==========
    # Compare near-term vs next-term IV
    term_structure = 0
    near_iv = current_iv
    far_iv = current_iv
    try:
        # Get unique expiries sorted
        expiries = sorted(options_df["expiry"].dropna().unique())
        if len(expiries) >= 2:
            near_expiry = expiries[0]
            far_expiry = expiries[1]
            
            # Calculate IV for next month
            far_options = options_df[options_df["expiry"] == far_expiry]
            if not far_options.empty:
                far_spot = far_options["underlying_value"].iloc[0]
                if not pd.isna(far_spot):
                    far_atm = round(far_spot / 50) * 50
                    far_atm_options = far_options[
                        (far_options["strike_price"] >= far_atm - 100) &
                        (far_options["strike_price"] <= far_atm + 100)
                    ]
                    
                    far_ivs = []
                    for _, row in far_atm_options.iterrows():
                        try:
                            far_opt_iv = calculate_implied_volatility(
                                row["ltp"], far_spot, row["strike_price"], 0.08, row["option_type"]
                            )
                            if far_opt_iv and far_opt_iv > 0:
                                far_ivs.append(far_opt_iv)
                        except:
                            continue
                    
                    if far_ivs:
                        far_iv = np.mean(far_ivs)
                        # Positive = contango (normal), Negative = backwardation (stress)
                        term_structure = far_iv - near_iv
    except Exception as e:
        print(f"DEBUG: Term structure calculation failed: {e}")
    
    # ========== 4. MARKET REGIME CLASSIFICATION ==========
    regime_name = "Unknown"
    try:
        # Classify based on IV rank, VRP, and realized vol
        if iv_rank > 70:
            if vrp > 0.05:
                regime_name = "High Vol - Sell Premium"
            else:
                regime_name = "High Vol - Caution"
        elif iv_rank < 30:
            if vrp < -0.05:
                regime_name = "Low Vol - Buy Premium"
            else:
                regime_name = "Low Vol - Range"
        else:
            if abs(vrp) < 0.03:
                regime_name = "Neutral - Balanced"
            elif vrp > 0:
                regime_name = "Elevated IV - Sell Bias"
            else:
                regime_name = "Compressed IV - Buy Bias"
    except Exception as e:
        print(f"DEBUG: Regime classification failed: {e}")
    
    # ========== 5. NIFTY TREND INDICATORS ==========
    sma_20 = 0
    sma_50 = 0
    rsi = 50
    atr = 0
    try:
        if "close" in nifty_df.columns and len(nifty_df) >= 50:
            # 20-day and 50-day SMAs
            sma_20 = nifty_df["close"].iloc[-20:].mean()
            sma_50 = nifty_df["close"].iloc[-50:].mean()
            
            # RSI (14-period)
            if len(nifty_df) >= 14:
                delta = nifty_df["close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs.iloc[-1])) if not pd.isna(rs.iloc[-1]) and loss.iloc[-1] != 0 else 50
            
            # ATR (14-period) - measure of volatility
            if "high" in nifty_df.columns and "low" in nifty_df.columns:
                high_low = nifty_df["high"] - nifty_df["low"]
                high_close = np.abs(nifty_df["high"] - nifty_df["close"].shift())
                low_close = np.abs(nifty_df["low"] - nifty_df["close"].shift())
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = ranges.max(axis=1)
                atr = true_range.rolling(14).mean().iloc[-1]
    except Exception as e:
        print(f"DEBUG: Trend indicators calculation failed: {e}")
    
    # ========== 6. VOLATILITY PERCENTILE ==========
    rv_percentile = 50
    try:
        if "close" in nifty_df.columns and len(nifty_df) > 60:
            # Calculate 10, 20, 30 day realized vols for last 60 days
            historical_rvs = []
            for i in range(30, len(nifty_df)):
                window_returns = nifty_df["close"].iloc[i-30:i].pct_change().dropna()
                if len(window_returns) >= 20:
                    window_rv = window_returns.std() * np.sqrt(252)
                    historical_rvs.append(window_rv)
            
            if historical_rvs and realized_vol:
                # Percentile of current realized vol
                below = sum(1 for rv in historical_rvs if rv < realized_vol)
                rv_percentile = (below / len(historical_rvs)) * 100
    except Exception as e:
        print(f"DEBUG: RV percentile calculation failed: {e}")
    
    # ========== 7. MAX PAIN ==========
    max_pain_strike = atm_strike
    try:
        if "open_int" in latest_options.columns:
            # Calculate max pain (strike with maximum total OI pain for option writers)
            strikes = sorted(latest_options["strike_price"].dropna().unique())
            min_pain = float('inf')
            
            for strike in strikes:
                # Calculate total pain at this strike
                call_pain = 0
                put_pain = 0
                
                # Calls: pain if strike > call_strike
                calls_below = latest_options[
                    (latest_options["option_type"] == "CE") &
                    (latest_options["strike_price"] < strike)
                ]
                call_pain = sum((strike - row["strike_price"]) * row["open_int"] 
                               for _, row in calls_below.iterrows() 
                               if not pd.isna(row["open_int"]))
                
                # Puts: pain if strike < put_strike
                puts_above = latest_options[
                    (latest_options["option_type"] == "PE") &
                    (latest_options["strike_price"] > strike)
                ]
                put_pain = sum((row["strike_price"] - strike) * row["open_int"] 
                              for _, row in puts_above.iterrows() 
                              if not pd.isna(row["open_int"]))
                
                total_pain = call_pain + put_pain
                if total_pain < min_pain:
                    min_pain = total_pain
                    max_pain_strike = strike
    except Exception as e:
        print(f"DEBUG: Max pain calculation failed: {e}")
    
    return {
        # Original metrics
        "current_iv": current_iv,
        "iv_rank": iv_rank,
        "realized_vol": realized_vol,
        "vrp": vrp,
        "current_spot": current_spot,
        
        # 1. PCR
        "pcr_volume": pcr_volume,
        "pcr_oi": pcr_oi,
        
        # 2. Skew
        "skew": skew,
        "put_25d_iv": put_25d_iv,
        "call_25d_iv": call_25d_iv,
        
        # 3. Term Structure
        "term_structure": term_structure,
        "near_iv": near_iv,
        "far_iv": far_iv,
        
        # 4. Market Regime
        "market_regime": regime_name,
        
        # 5. Trend Indicators
        "sma_20": sma_20,
        "sma_50": sma_50,
        "rsi": rsi,
        "atr": atr,
        
        # 6. Volatility Percentile
        "rv_percentile": rv_percentile,
        
        # 7. Max Pain
        "max_pain_strike": max_pain_strike,
    }
