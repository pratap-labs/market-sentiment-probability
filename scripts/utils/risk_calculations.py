"""Risk calculation utilities for portfolio analysis."""

import pandas as pd
from typing import List, Dict, Optional, Tuple
from .formatters import format_inr


def calculate_var(positions: List[Dict], spot: float, nifty_df: Optional[pd.DataFrame] = None, confidence: float = 0.95) -> float:
    """Calculate portfolio VaR using realized volatility from NIFTY data.
    
    Args:
        positions: List of enriched position dictionaries
        spot: Current NIFTY spot price
        nifty_df: NIFTY daily price data (optional, will use default vol if not provided)
        confidence: Confidence level (default 0.95 for 95% VaR)
    
    Returns:
        VaR estimate in rupees (95% confidence = 2-sigma move by default)
    """
    total_delta = sum(p.get("position_delta", 0) for p in positions)
    total_gamma = sum(p.get("position_gamma", 0) for p in positions)
    total_vega = sum(p.get("position_vega", 0) for p in positions)
    
    # Calculate realized daily volatility from NIFTY data
    if nifty_df is not None and not nifty_df.empty and "close" in nifty_df.columns and len(nifty_df) > 30:
        returns = nifty_df["close"].pct_change().dropna()[-30:]  # Last 30 days
        daily_vol = returns.std()
    else:
        # Conservative fallback: 1.5% daily vol (typical for NIFTY in normal markets)
        daily_vol = 0.015
    

    # 2-sigma move for 95% confidence
    spot_move_2sigma = spot * (2 * daily_vol)
    
    
    print(f'DEBUG VaR: daily_vol={daily_vol:.4f} ({daily_vol*100:.2f}%)')
    print(f'DEBUG VaR: total_delta={total_delta:.2f}, total_gamma={total_gamma:.4f}, total_vega={total_vega:.2f}')
    print(f'DEBUG VaR: spot={spot:.0f}, spot_move_2sigma={spot_move_2sigma:.0f} ({spot_move_2sigma/spot*100:.2f}%)')
    
    # IV move assumption: 5 volatility points (0.05) for stress scenario
    iv_move = 0.05
    
    # Linear approximation: ΔP ≈ Delta×ΔS + 0.5×Gamma×ΔS² + Vega×ΔIV
    delta_contribution = total_delta * spot_move_2sigma
    gamma_contribution = 0.5 * total_gamma * (spot_move_2sigma ** 2)
    vega_contribution = total_vega * iv_move
    
    print(f'DEBUG VaR components: delta={delta_contribution:.0f}, gamma={gamma_contribution:.0f}, vega={vega_contribution:.0f}')
    
    var = abs(delta_contribution + gamma_contribution + vega_contribution)
    
    print(f'DEBUG VaR: final VaR={var:.0f}')
    
    return var


def calculate_stress_pnl(positions: List[Dict], spot: float, spot_multiplier: float, iv_change: float) -> float:
    """Calculate P&L change under stress scenario.
    
    Returns the expected P&L change (not total P&L) for the given stress scenario.
    Positive = profit, Negative = loss.
    """
    new_spot = spot * spot_multiplier
    spot_change = new_spot - spot
    
    total_pnl_change = 0
    for pos in positions:
        delta = pos.get("position_delta", 0)
        gamma = pos.get("position_gamma", 0)
        vega = pos.get("position_vega", 0)
        
        # P&L change = Delta×ΔS + 0.5×Gamma×ΔS² + Vega×ΔIV
        pnl_change = delta * spot_change + 0.5 * gamma * (spot_change ** 2) + vega * iv_change
        total_pnl_change += pnl_change
    
    return total_pnl_change


def get_action_signal(dte: int, pnl: float, theta: float, position_delta: float, 
                     days_to_breakeven: float, loss_vs_credit: float) -> Tuple[str, int]:
    """
    Determine action signal and priority (0-10).
    
    Args:
        dte: Days to expiry
        pnl: Current P&L
        theta: Position theta (per-day decay)
        position_delta: Position delta (delta × quantity) - scaled directional exposure
        days_to_breakeven: Days needed to recover losses via theta
        loss_vs_credit: Loss as multiple of credit received (for short positions)
    
    Returns: (action_string, priority_score)
    """
    priority = 0
    action = "HOLD"
    
    # Check critical conditions
    # 1. Cannot recover by expiry
    if days_to_breakeven > dte and dte > 0:
        priority += 5
        action = "CLOSE/ROLL"
    
    # 2. High directional risk - position_delta is total delta (delta × quantity)
    # Threshold of 20 means position gains/loses ₹20 per 1-point NIFTY move
    # (e.g., 20 delta = ₹1000 P&L on 50-point NIFTY move)
    # This catches positions contributing >50% of typical portfolio delta exposure
    if abs(position_delta) > 20:
        priority += 3
        action = "ADJUST/HEDGE"
    
    # 3. Loss exceeds credit by 2x - catastrophic for short positions
    if loss_vs_credit > 2:
        priority += 4
        action = "CLOSE"
    
    # 4. Near expiry with losses - close to avoid gamma risk
    if dte < 3 and pnl < 0:
        priority += 3
        action = "CLOSE"
    
    # 5. Profitable with time remaining - consider taking profit
    if dte > 21 and pnl > 0 and theta != 0:
        if pnl / theta > dte * 0.5:  # Captured >50% of potential theta
            priority += 1
            action = "TAKE PROFIT"
    
    # If multiple conditions, prioritize most severe action
    if priority == 0:
        action = "HOLD"
    
    return action, min(priority, 10)


def get_action_recommendation(pos: Dict) -> str:
    """Get detailed action recommendation for a position."""
    action = pos["Action"]
    dte = pos["DTE"]
    pnl = pos["PnL"]
    delta = pos["Pos Delta"]
    
    if action == "CLOSE":
        return f"⚠️ **Close this position immediately**. Loss exceeds acceptable limits or cannot recover by expiry."
    
    elif action == "CLOSE/ROLL":
        return f"🔄 **Close and roll to next expiry**. Cannot recover {format_inr(abs(pnl))} loss in {dte} days with current theta."
    
    elif action == "ADJUST/HEDGE":
        return f"⚡ **Hedge delta exposure**. Position delta of {delta:.2f} is too high. Consider buying protective options."
    
    elif action == "TAKE PROFIT":
        return f"✅ **Take profit early**. Position is profitable with {dte} DTE remaining. Lock in gains."
    
    else:
        return "✓ **Hold position**. Continue monitoring theta decay."
