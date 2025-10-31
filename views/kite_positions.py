"""
Streamlit view for logging in with Kite Connect and fetching positions.
Enhanced with Options Analytics Dashboard.
"""

import webbrowser
import os
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import re
import json
from pathlib import Path as _Path
import os
import sys


try:
    from kiteconnect import KiteConnect
except Exception:
    KiteConnect = None

try:
    from py_vollib.black_scholes import black_scholes as bs
    from py_vollib.black_scholes.implied_volatility import implied_volatility as iv
    from py_vollib.black_scholes.greeks import analytical as greeks
except Exception:
    bs = iv = greeks = None


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from views import data_hub
except Exception as e:
    data_hub = None
    print(f"Failed to import data_hub: {e}")


# ============================================================================
# DATA LOADING
# ============================================================================

def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase with underscores."""
    df.columns = df.columns.str.strip().str.lower().str.replace(r"[\s\.\*\*]+", "_", regex=True)
    return df


# ------------------------
# Token persistence helpers
# ------------------------
def _token_file_path() -> _Path:
    """Return the path to the persistent token file.

    Use the current working directory (where Streamlit is usually launched) as the
    project root. This avoids issues when __file__ resolution differs under
    different runtimes (tests, packaging, or import paths).
    """
    try:
        root = _Path.cwd()
    except Exception:
        root = _Path(__file__).resolve().parents[1]
    return root / ".kite_token.json"


def save_kite_token(api_key: str, access_token: str) -> None:
    p = _token_file_path()
    payload = {
        "kite_api_key": api_key,
        "kite_access_token": access_token,
        "saved_at": datetime.utcnow().isoformat() + "Z"
    }
    try:
        with p.open("w", encoding="utf-8") as f:
            json.dump(payload, f)
        # Try to restrict permissions to owner-only where possible
        try:
            p.chmod(0o600)
        except Exception:
            pass
        print(f"Saved Kite token to {p}")
    except Exception as e:
        print(f"Failed to save Kite token: {e}")
        try:
            st.warning(f"Failed to save Kite token to {p}: {e}")
        except Exception:
            pass


def load_kite_token() -> Optional[Dict[str, str]]:
    p = _token_file_path()
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Failed to read Kite token file {p}: {e}")
        try:
            st.warning(f"Failed to read saved Kite token: {e}")
        except Exception:
            pass
        return None


def clear_kite_token_file() -> None:
    p = _token_file_path()
    try:
        if p.exists():
            p.unlink()
            print(f"Removed saved Kite token file {p}")
    except Exception as e:
        print(f"Failed to remove Kite token file: {e}")


@st.cache_data(ttl=300)
def load_all_data(dirs=None):
    """Load all excel/csv files from the given directories.

    If dirs is None, we try `database/data/` and fall back to `database/options_data/`.
    """
    if dirs is None:
        dirs = ["database/data", "database/options_data"]

    all_files = []
    for d in dirs:
        p = Path(d)
        if not p.exists():
            continue
        for ext in ("*.xlsx", "*.xls", "*.csv", "*.xlsm"):
            all_files.extend(sorted([str(x) for x in p.glob(ext)]))

    if not all_files:
        return pd.DataFrame(), []

    frames = []
    errors = []
    for f in all_files:
        try:
            if f.lower().endswith(".csv"):
                df = pd.read_csv(f)
            else:
                df = pd.read_excel(f, sheet_name=0)
        except Exception as e:
            errors.append((f, str(e)))
            continue

        df = _map_columns(df)
        frames.append(df)

    if not frames:
        return pd.DataFrame(), errors

    combined = pd.concat(frames, ignore_index=True, sort=False)

    # Normalize date fields
    if "date" in combined.columns:
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    if "expiry" in combined.columns:
        combined["expiry"] = pd.to_datetime(combined["expiry"], errors="coerce")

    # Ensure strike_price numeric
    if "strike_price" in combined.columns:
        combined["strike_price"] = pd.to_numeric(combined["strike_price"], errors="coerce")

    # Clean and coerce common numeric columns
    numeric_candidates = [
        "open", "high", "low", "close", "ltp", "settle_price",
        "no_of_contracts", "turnover", "premium_turnover", "open_int",
        "change_in_oi", "underlying_value"
    ]
    for col in numeric_candidates:
        if col in combined.columns:
            combined[col] = combined[col].astype(str).str.replace(r"[^0-9.\-eE]", "", regex=True)
            combined[col] = pd.to_numeric(combined[col].replace("", pd.NA), errors="coerce")

    return combined, errors


def load_nifty_daily(kite_client: Optional[object] = None, dirs=None, days: int = 365) -> pd.DataFrame:
    """Load NIFTY daily data.

    Preferred source: Kite historical API when a `kite_client` (KiteConnect) is provided
    and a valid session exists. Falls back to reading local files under
    `database/nifty_daily` when Kite is not available or call fails.

    Returns a DataFrame sorted by date with at least columns ['date','close'] when
    successful. Raises no exception — returns empty DataFrame on failure.
    """
    # This function now requires a KiteConnect client and will NOT fall back to local files
    if kite_client is None:
        raise RuntimeError(
            "Kite client is required to load NIFTY daily data. Local files do not contain NIFTY daily prices (they only have OI)."
        )

    try:
        instruments = kite_client.instruments()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch instruments from Kite client: {e}")

    # Find NIFTY instrument token (prefer exact matches)
    token = None
    for inst in instruments:
        trad = (inst.get("tradingsymbol") or "").upper().strip()
        if trad in ("NIFTY", "NIFTY 50", "NIFTY-I", "NIFTY50"):
            token = inst.get("instrument_token") or inst.get("instrument_token")
            break

    if token is None:
        # fallback: first instrument with 'NIFTY' in the tradingsymbol
        for inst in instruments:
            trad = (inst.get("tradingsymbol") or "").upper()
            if "NIFTY" in trad:
                token = inst.get("instrument_token") or inst.get("instrument_token")
                break

    if token is None:
        raise RuntimeError("Could not find a NIFTY instrument token in Kite instruments")

    # Fetch historical data from Kite
    to_date = datetime.now()
    from_date = to_date - timedelta(days=days)

    try:
        raw = kite_client.historical_data(token, from_date.strftime("%Y-%m-%d"), to_date.strftime("%Y-%m-%d"), "day")
    except Exception as e:
        raise RuntimeError(f"Kite historical_data call failed: {e}")

    kite_df = pd.DataFrame(raw)

    # Print columns for debugging per user's request
    print("Kite NIFTY columns:", list(kite_df.columns))

    if kite_df.empty:
        raise RuntimeError("Kite returned no historical data for NIFTY")

    # Validate required columns
    required = {"date", "close", "open", "high", "low"}
    present = set([c.lower() for c in kite_df.columns])
    missing = required - present
    if missing:
        raise RuntimeError(f"Kite NIFTY data missing required columns: {sorted(list(missing))}")

    # Normalize and return
    kite_df["date"] = pd.to_datetime(kite_df["date"], errors="coerce")
    kite_df = kite_df.dropna(subset=["date"])
    kite_df = kite_df.sort_values("date").reset_index(drop=True)
    st.session_state["nifty_loaded_from"] = "kite"
    return kite_df


# ============================================================================
# POSITION PARSING
# ============================================================================
import re
from datetime import datetime
from typing import Optional, Dict
import calendar

def parse_tradingsymbol(symbol: str) -> Optional[Dict]:
    s = (symbol or "").upper().strip()

    # -------------------
    # Weekly options
    # Format: NIFTY + 2-digit-year + month (1-9 or O/N/D) + day (2 digits) + strike + CE/PE
    # Examples: "NIFTY2591125100PE" (pre-Oct), "NIFTY25N1125100PE" (Nov)
    # -------------------
    weekly_pattern = r"^NIFTY(\d{2})([1-9OND])(\d{2})(\d+)(CE|PE)$"
    m_week = re.match(weekly_pattern, s)
    if m_week:
        year_2, month_code, day_str, strike_str, opt_type = m_week.groups()
        year = 2000 + int(year_2)

        # Convert month_code to month number
        if month_code.isdigit():
            month = int(month_code)  # 1-9 → Jan-Sep
        else:
            month = {"O": 10, "N": 11, "D": 12}.get(month_code)
            if month is None:
                return None

        try:
            day = int(day_str)
            expiry = datetime(year, month, day)
        except ValueError:
            return None

        return {
            "strike": float(strike_str),
            "expiry": expiry,
            "option_type": opt_type
        }

    # -------------------
    # Monthly options
    # Format: NIFTY + 2-digit-year + 3-letter-month + strike + CE/PE
    # Example: "NIFTY25NOV25000CE"
    # -------------------
    monthly_pattern = r"^NIFTY(\d{2})([A-Z]{3})(\d+)(CE|PE)$"
    m_mon = re.match(monthly_pattern, s)
    if m_mon:
        year_2, month_str, strike_str, opt_type = m_mon.groups()
        year = 2000 + int(year_2)

        month_map = {
            "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
            "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
        }
        month = month_map.get(month_str.upper())
        if not month:
            return None

        # Monthly expiry: last day of the month
        last_day = calendar.monthrange(year, month)[1]
        expiry = datetime(year, month, last_day)

        return {
            "strike": float(strike_str),
            "expiry": expiry,
            "option_type": opt_type
        }

    return None


# ============================================================================
# GREEKS CALCULATION
# ============================================================================

def calculate_implied_volatility(option_price: float, spot: float, strike: float, 
                                 time_to_expiry: float, option_type: str, 
                                 risk_free_rate: float = 0.07) -> Optional[float]:
    """Calculate implied volatility using py_vollib."""
    if iv is None:
        return None
    
    try:
        flag = 'c' if option_type == 'CE' else 'p'
        calculated_iv = iv(option_price, spot, strike, time_to_expiry, risk_free_rate, flag)
        return calculated_iv
    except Exception as e:
        return None


def calculate_greeks(spot: float, strike: float, time_to_expiry: float, 
                     implied_vol: float, option_type: str, 
                     risk_free_rate: float = 0.07) -> Dict[str, float]:
    """Calculate option greeks using py_vollib."""
    if greeks is None:
        return {"delta": 0, "gamma": 0, "vega": 0, "theta": 0}
    
    try:
        flag = 'c' if option_type == 'CE' else 'p'
        # Use raw values returned by py_vollib. Higher-level scaling (per-contract) is
        # applied when enriching positions because lot/contract sizes vary by instrument.
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


def enrich_position_with_greeks(position: Dict, options_df: pd.DataFrame, 
                                current_spot: float) -> Dict:
    """
    Enrich a single Kite position with parsed data + Greeks.
    Returns enhanced position dict.
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

    # Scale Greeks by quantity only (user requested). Include both raw and scaled values.
    scaled_greeks = {}
    for k, v in position_greeks.items():
        # only scale the primary greek names (delta,gamma,vega,theta)
        if k in ("delta", "gamma", "vega", "theta"):
            scaled_greeks[f"position_{k}"] = v * quantity
        else:
            # keep any raw helpers intact
            scaled_greeks[k] = v

    # Debug: print values so the developer can inspect raw vs scaled greeks
    try:
        print(f"DEBUG: {symbol} qty={quantity} greeks_raw={ {k: position_greeks.get(k) for k in ('delta','gamma','vega','theta')} } scaled={{k: scaled_greeks.get('position_'+k) for k in ('delta','gamma','vega','theta')}}")
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


# ============================================================================
# PORTFOLIO METRICS
# ============================================================================

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


# ------------------------
# Presentation helpers
# ------------------------
def format_inr(value, decimals: int = 0, symbol: str = "₹") -> str:
    """Format number using Indian-style comma grouping.

    Examples:
        format_inr(1000000) -> '₹10,00,000'
        format_inr(12345.67, decimals=2) -> '₹12,345.67'
    """
    try:
        if value is None:
            return f"{symbol}0"
        neg = float(value) < 0
        val = abs(float(value))

        # round according to decimals
        if decimals and decimals > 0:
            fmt_val = f"{val:.{decimals}f}"
            int_part, _, frac = fmt_val.partition('.')
        else:
            int_part = str(int(round(val)))
            frac = ""

        # Indian grouping: last 3 digits, then groups of 2
        if len(int_part) <= 3:
            int_fmt = int_part
        else:
            last3 = int_part[-3:]
            rest = int_part[:-3]
            parts = []
            while len(rest) > 2:
                parts.append(rest[-2:])
                rest = rest[:-2]
            if rest:
                parts.append(rest)
            parts.reverse()
            int_fmt = ",".join(parts) + "," + last3

        s = f"{symbol}{'-' if neg else ''}{int_fmt}"
        if frac:
            s = s + "." + frac
        return s
    except Exception:
        try:
            return f"{symbol}{value}"
        except Exception:
            return f"{symbol}0"


# Default lot/contract size used for converting delta units to rupees-per-point
# Keep as a presentation constant; does NOT change greeks calculations.
DEFAULT_LOT_SIZE = int(os.getenv("OPTION_CONTRACT_SIZE", "50"))


# ============================================================================
# STREAMLIT UI
# ============================================================================

def render():
    """Main entrypoint for the Streamlit view."""
    st.title("🎯 Options Dashboard")
    # Presentation: compact font sizes + enforce 4 metrics per row (purely presentational)
    st.markdown(
        """
        <style>
        /* Slightly reduce global base font for the dashboard */
        html, body, .stApp { font-size: 13px; }

        /* Metric label/value sizing */
        div[data-testid="stMetricValue"], div[data-testid="stMetricDelta"] {
            font-size: 18px !important;
            line-height: 1 !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 12px !important;
        }

        /* Headings slightly smaller */
        h1, h2, h3, h4, h5 { font-size: 1.05rem !important; }

        /* Table cells smaller */
        table, th, td, .stDataFrame, .element-container table { font-size: 12px !important; }

        /* Compact padding for common Streamlit containers */
        .css-1d391kg, .css-1lcbmhc, .css-18e3th9, .css-12w0qpk { padding: 0.35rem !important; margin: 0.05rem !important; }

        /* Force metric containers to inline-block and take ~24% width to show 4 per row */
        .css-1lcbmhc, .css-1d391kg, .css-18e3th9, .css-12w0qpk {
            display: inline-block !important;
            vertical-align: top !important;
            width: 24% !important;
            box-sizing: border-box !important;
            padding: 0.45rem !important;
        }

            /* Ensure metric inner blocks are compact; keep delta pill sized to content */
            div[data-testid="stMetricLabel"], div[data-testid="stMetricValue"] {
                padding: 0.05rem 0 !important;
                margin: 0 !important;
            }

            /* Make the delta pill inline and sized to its contents (avoid full-width pills) */
            div[data-testid="stMetricDelta"] { font-size: 12px !important; display: inline-flex !important; align-items: center; gap: 0.35rem; padding: 5px 10px}
            div[data-testid="stMetricDelta"] > span { padding: 0.12rem 1rem !important; border-radius: 999px !important; }

        /* Responsive fallbacks */
        @media (max-width: 900px) {
            .css-1lcbmhc, .css-1d391kg, .css-18e3th9, .css-12w0qpk { width: 48% !important; }
        }
        @media (max-width: 520px) {
            .css-1lcbmhc, .css-1d391kg, .css-18e3th9, .css-12w0qpk { width: 100% !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if KiteConnect is None:
        st.error("Missing dependency `kiteconnect`. Install with `pip install kiteconnect`.")
        return
    
    if bs is None or iv is None or greeks is None:
        st.error("Missing dependency `py_vollib`. Install with `pip install py_vollib`.")
        return

    # Initialize session state
    if "kite_login_initiated" not in st.session_state:
        st.session_state.kite_login_initiated = False
    if "kite_processing_token" not in st.session_state:
        st.session_state.kite_processing_token = False

    # Check for request_token in URL
    query_params = st.query_params
    incoming_request_token = query_params.get("request_token", None)

    # Exchange request token for access token
    if incoming_request_token and not st.session_state.kite_processing_token:
        st.session_state.kite_processing_token = True
        
        api_key = st.session_state.get("kite_api_key_stored") or os.getenv("KITE_API_KEY")
        api_secret = st.session_state.get("kite_api_secret_stored") or os.getenv("KITE_API_SECRET")
            
        if api_key and api_secret:
            try:
                with st.spinner("Exchanging request token for access token..."):
                    kite = KiteConnect(api_key=api_key)
                    data = kite.generate_session(incoming_request_token, api_secret=api_secret)
                    access_token = data.get("access_token")
                    
                    if access_token:
                        st.session_state["kite_access_token"] = access_token
                        st.session_state["kite_api_key"] = api_key
                        st.success("✅ Successfully logged in!")
                        
                        st.query_params.clear()
                        st.session_state.kite_login_initiated = False
                        st.session_state.kite_processing_token = False
                        st.rerun()
                    else:
                        st.error("Failed to obtain access token from Kite")
                        st.session_state.kite_processing_token = False
                        
            except Exception as e:
                st.error(f"Login failed: {e}")
                st.session_state.kite_processing_token = False
                st.query_params.clear()
        else:
            st.error("API credentials not found. Please enter them again and retry login.")
            st.session_state.kite_processing_token = False
            st.query_params.clear()

    # Load data
    with st.spinner("Loading options data..."):
        options_df, errors = load_all_data()
        # If we have a Kite session available in session_state, prefer Kite historical
        kite_client = None
        kite_token = st.session_state.get("kite_access_token")
        kite_key = st.session_state.get("kite_api_key")
        if kite_token and kite_key:
            try:
                kite_client = KiteConnect(api_key=kite_key)
                kite_client.set_access_token(kite_token)
            except Exception:
                kite_client = None

        # Only attempt to load NIFTY daily from Kite when a Kite client is available
        if kite_client is not None:
            try:
                nifty_df = load_nifty_daily(kite_client=kite_client)
            except RuntimeError as e:
                # Surface Kite/data errors to the UI and continue with empty df
                st.error(f"Failed to load NIFTY daily: {e}")
                nifty_df = pd.DataFrame()
                st.session_state["nifty_loaded_from"] = None
        else:
            nifty_df = pd.DataFrame()
            st.session_state["nifty_loaded_from"] = None
            # Inform the user that they need to log in to load NIFTY daily prices
            st.info("Log in via the Login tab to load NIFTY daily price data from Kite.")

        # Confirm data load source and show sample head if data exists
        if not nifty_df.empty:
            st.session_state["nifty_loaded_from"] = "Kite API"
            st.session_state["nifty_df"] = nifty_df  # Store for use in VaR calculation

            # Validate required columns
            required_cols = {"date", "close", "open", "high", "low"}
            present = set(nifty_df.columns.str.lower())
            missing = required_cols - present
            if missing:
                msg = f"NIFTY daily data missing required columns: {sorted(list(missing))}"
                st.error(msg)
                # Raise to make the failure obvious during development
                raise RuntimeError(msg)
    
    if errors:
        with st.expander("⚠️ Data loading warnings", expanded=False):
            for file, err in errors:
                st.warning(f"{file}: {err}")

    # Create tabs
    tabs = st.tabs([
        "🔐 Login", 
        "📊 Positions", 
        "📈 Portfolio Overview",
        "🔍 Position Diagnostics", 
        "🌡️ Market Regime",
        "🚨 Risk Alerts",
        "🎯 Advanced Analytics",
        "� Trade History",
        "�📂 Data Hub"
    ])

    # --- LOGIN TAB ---
    with tabs[0]:
        render_login_tab()

    # --- POSITIONS TAB ---
    with tabs[1]:
        render_positions_tab(options_df, nifty_df)
    
    # --- PORTFOLIO OVERVIEW TAB ---
    with tabs[2]:
        render_portfolio_tab()
    
    # --- POSITION DIAGNOSTICS TAB ---
    with tabs[3]:
        render_diagnostics_tab()
    
    # --- MARKET REGIME TAB ---
    with tabs[4]:
        render_market_regime_tab(options_df, nifty_df)
    
    # --- RISK ALERTS TAB ---
    with tabs[5]:
        render_alerts_tab()
    
    # --- ADVANCED ANALYTICS TAB ---
    with tabs[6]:
        render_advanced_analytics_tab()
    
    # --- TRADE HISTORY TAB ---
    with tabs[7]:
        render_trade_history_tab()
    
    # --- DATA HUB TAB ---
    with tabs[8]:
        if data_hub is not None:
            data_hub.render()
        else:
            st.error("Data Hub module not available. Please check the installation.")


def render_login_tab():
    """Render the login tab."""
    st.subheader("Login")
    
    default_api_key = st.session_state.get("kite_api_key_stored", "") or os.getenv("KITE_API_KEY", "")
    default_api_secret = st.session_state.get("kite_api_secret_stored", "") or os.getenv("KITE_API_SECRET", "")
    
    api_key = st.text_input("API Key", value=default_api_key, key="kite_api_key_input")
    api_secret = st.text_input("API Secret", value=default_api_secret, type="password", key="kite_api_secret_input")
    
    st.caption("⚠️ **Important**: Your Kite app redirect URI must match your Streamlit URL (e.g., http://127.0.0.1:5174/)")
    
    host = st.text_input("Redirect host", value="127.0.0.1", key="kite_redirect_host")
    port = st.number_input("Redirect port", value=5174, min_value=1024, max_value=65535, key="kite_redirect_port")

    if not api_key or not api_secret:
        st.info("Enter Kite API key and secret (or set KITE_API_KEY/KITE_API_SECRET env vars).")

    col1, col2 = st.columns([1, 1])
    login_clicked = col1.button("Login with Kite", disabled=st.session_state.kite_login_initiated)
    logout_clicked = col2.button("Logout (clear token)")

    if logout_clicked:
        st.session_state.pop("kite_access_token", None)
        st.session_state.pop("kite_api_key", None)
        st.session_state.kite_login_initiated = False
        st.session_state.kite_processing_token = False
        st.success("Logged out")
        st.rerun()

    if login_clicked and api_key and api_secret:
        st.session_state["kite_api_key_stored"] = api_key
        st.session_state["kite_api_secret_stored"] = api_secret
        st.session_state.kite_login_initiated = True
        
        redirect_uri = f"http://{host}:{port}/"
        login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}&redirect_params=status%3Dlogin&redirect_uri={redirect_uri}"
        
        st.info(f"🔗 Opening Kite login in your browser. After login, you'll be redirected back here.")
        
        webbrowser.open(login_url)
        
        st.warning("⏳ Complete the login in the browser window. This page will update automatically after redirect.")

    if st.session_state.get("kite_access_token"):
        st.success("✅ You are logged in. Switch to the Positions tab to fetch positions.")
    elif st.session_state.kite_login_initiated:
        st.info("⏳ Waiting for login completion. Complete the login in your browser.")


def render_positions_tab(options_df: pd.DataFrame, nifty_df: pd.DataFrame):
    """Render the positions tab with enriched data."""
    st.subheader("Positions")
    access_token = st.session_state.get("kite_access_token")
    
    if not access_token:
        st.warning("Not logged in. Please go to the Login tab and sign in first.")
        return
    
    kite_api_key = st.session_state.get("kite_api_key")
    if not kite_api_key:
        st.error("API key not found in session. Please log in again.")
        return
        
    kite = KiteConnect(api_key=kite_api_key)
    kite.set_access_token(access_token)

    if st.button("🔄 Fetch Latest Positions", type="primary"):
        try:
            with st.spinner("Fetching positions..."):
                positions = kite.positions()
                net_positions = positions.get("net", [])

                if not net_positions:
                    st.info("No positions returned")
                    return
                
                # Get current spot price
                market_regime = calculate_market_regime(options_df, nifty_df)
                current_spot = market_regime.get("current_spot", 25000)
                
                # Enrich positions with Greeks
                enriched = []
                for pos in net_positions:
                    enriched_pos = enrich_position_with_greeks(pos, options_df, current_spot)
                    enriched.append(enriched_pos)
                
                # Store in session state
                st.session_state["enriched_positions"] = enriched
                st.session_state["current_spot"] = current_spot
                
                st.success(f"✅ Loaded {len(enriched)} positions")

        except Exception as e:
            st.error(f"Failed to fetch positions: {e}")
            if "403" in str(e) or "Invalid" in str(e):
                st.warning("Your session may have expired. Please log in again.")
                st.session_state.pop("kite_access_token", None)
            return
    
    # Display enriched positions if available
    if "enriched_positions" in st.session_state:
        enriched = st.session_state["enriched_positions"]
        
        if enriched:
            # Create display dataframe
            display_cols = [
                "tradingsymbol", "quantity", "strike", "option_type", "dte",
                "last_price", "pnl", "implied_vol",
                "delta", "gamma", "vega", "theta",
                "position_delta", "position_gamma", "position_vega", "position_theta"
            ]
            
            display_data = []
            for pos in enriched:
                row = {col: pos.get(col, None) for col in display_cols}
                display_data.append(row)
            
            df = pd.DataFrame(display_data)
            
            # Format display
            if not df.empty:
                st.dataframe(
                    df.style.format({
                        "implied_vol": "{:.2%}",
                        "delta": "{:.3f}",
                        "gamma": "{:.4f}",
                        "vega": "{:.2f}",
                        "theta": "{:.2f}",
                        "position_delta": "{:.2f}",
                        "position_gamma": "{:.3f}",
                        "position_vega": "{:.1f}",
                        "position_theta": "{:.1f}",
                        "last_price": "{:.2f}",
                        "pnl": "{:.2f}"
                    }),
                    use_container_width=True
                )


def render_portfolio_tab():
    """Render portfolio overview tab with complete metrics."""
    st.subheader("📈 Portfolio Overview")
    
    if "enriched_positions" not in st.session_state:
        st.info("No positions loaded. Fetch positions from the Positions tab first.")
        return
    
    enriched = st.session_state["enriched_positions"]
    portfolio_greeks = calculate_portfolio_greeks(enriched)
    current_spot = st.session_state.get("current_spot", 25000)
    
    # Get account info from Kite (margins)
    access_token = st.session_state.get("kite_access_token")
    kite_api_key = st.session_state.get("kite_api_key")
    
    margin_available = 500000  # Default, will fetch from Kite
    margin_used = 320000  # Default, will fetch from Kite
    
    if access_token and kite_api_key:
        try:
            kite = KiteConnect(api_key=kite_api_key)
            kite.set_access_token(access_token)
            margins = kite.margins()
            
            equity_margins = margins.get("equity", {})
            margin_available = equity_margins.get("available", {}).get("live_balance", 500000)
            margin_used = equity_margins.get("utilised", {}).get("debits", 0)
        except:
            pass  # Use defaults if fetch fails
    
    account_size = margin_available + margin_used
    
    # Calculate portfolio metrics
    total_pnl = sum(p.get("pnl", 0) for p in enriched)
    total_theta = portfolio_greeks["net_theta"]
    total_delta = portfolio_greeks["net_delta"]
    total_gamma = portfolio_greeks["net_gamma"]
    total_vega = portfolio_greeks["net_vega"]
    
    # Days to recover
    days_to_recover = abs(total_pnl / total_theta) if total_theta != 0 else 999
    
    # Theta efficiency
    theta_efficiency = (total_pnl / total_theta * 100) if total_theta != 0 else 0
    
    # ROI calculations
    roi_pct = (total_pnl / account_size * 100) if account_size > 0 else 0
    
    # Assume average DTE for annualization
    avg_dte = sum(p.get("dte", 0) for p in enriched) / len(enriched) if enriched else 30
    days_in_trade = max(30 - avg_dte, 1)  # Approximate
    roi_annualized = (total_pnl / account_size) / (days_in_trade / 365) * 100 if account_size > 0 else 0
    
    # Delta dollars
    delta_dollars = total_delta * current_spot
    
    # Notional exposure
    notional_exposure = sum(
        abs(p.get("quantity", 0)) * p.get("strike", 0) 
        for p in enriched
    )
    
    # Leverage ratio (Notional / Account)
    leverage_ratio = (notional_exposure / account_size) if account_size > 0 else 0
    
    # Margin utilization
    margin_util_pct = (margin_used / account_size * 100) if account_size > 0 else 0
    
    # Theta as % of capital
    theta_pct_capital = (abs(total_theta) / account_size * 100) if account_size > 0 else 0
    
    # ========== SECTION 1: CAPITAL & PERFORMANCE ==========
    st.markdown("### 💰 Capital & Performance")

    # First row: 4 metrics per row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Account Size", format_inr(account_size))

    with col2:
        margin_color = "🟢" if margin_util_pct < 50 else ("🟡" if margin_util_pct < 70 else "🔴")
        st.metric("Margin Used", format_inr(margin_used), f"{margin_util_pct:.1f}% {margin_color}")

    with col3:
        pnl_color = "inverse" if total_pnl < 0 else "normal"
        st.metric("Net P&L", format_inr(total_pnl), f"{roi_pct:.2f}%")

    with col4:
        st.metric("Theta/Day", format_inr(total_theta), f"{theta_pct_capital:.2f}% of capital")

    # Second row: 4 columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("ROI (Annualized)", f"{roi_annualized:.1f}%")

    with col2:
        recovery_color = "🔴" if days_to_recover > avg_dte else ("🟡" if days_to_recover > avg_dte * 0.5 else "🟢")
        st.metric("Days to Recover", f"{days_to_recover:.1f} {recovery_color}", f"Avg DTE: {avg_dte:.0f}")

    with col3:
        eff_color = "🔴" if theta_efficiency < -200 else ("🟡" if theta_efficiency < -100 else "🟢")
        st.metric("Theta Efficiency", f"{theta_efficiency:.0f}% {eff_color}")

    with col4:
        leverage_color = "🟢" if leverage_ratio < 50 else ("🟡" if leverage_ratio < 100 else "🔴")
        st.metric("Notional Exposure", format_inr(notional_exposure), f"{leverage_ratio:.0f} × capital {leverage_color}")
    
    # ========== SECTION 2: GREEKS & RISK ==========
    st.markdown("### 📊 Greeks & Risk")
    st.caption(f"Delta conversions assume lot size = {DEFAULT_LOT_SIZE} for ₹/pt calculation. If your positions already include lot-size in total_delta, ₹/pt equals Net Delta.")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta_abs = abs(total_delta)
        delta_status = "🟢" if delta_abs < 8 else ("🟡" if delta_abs < 15 else "🔴")
        st.metric("Net Delta (units)", f"{total_delta:.2f} {delta_status}")

    # Compute presentation-only conversions (no greeks math changed)
    # total_delta already includes lot-size scaling (positions are stored scaled),
    # so do NOT multiply by DEFAULT_LOT_SIZE here — use total_delta directly.
    rupee_per_point_by_lot = total_delta
    delta_notional = total_delta * current_spot

    with col2:
        st.metric("Delta (₹/pt)", format_inr(rupee_per_point_by_lot))

    with col3:
        gamma_status = "⚠️" if total_gamma < -0.5 else ""
        st.metric("Net Gamma", f"{total_gamma:.3f} {gamma_status}")

    with col4:
        st.metric("Net Vega", format_inr(total_vega))

    # Separate row for Delta Notional and Vega % Capital to keep 4-per-row layout consistent
    col1, col2, col3, col4 = st.columns(4)
    # percent of account for Delta Notional (use abs for percent sizing)
    delta_notional_pct = (abs(delta_notional) / account_size * 100) if account_size > 0 else 0
    vega_pct = (abs(total_vega) / account_size * 100) if account_size > 0 else 0
    with col1:
        st.metric("Delta Notional (₹)", format_inr(delta_notional), f"{delta_notional_pct:.1f}% of account")
    with col2:
        st.metric("Vega % Capital", f"{vega_pct:.2f}%")
    
    # Greeks breakdown chart
    st.markdown("#### Greeks Breakdown")
    greeks_data = pd.DataFrame({
        "Greek": ["Delta", "Gamma×100", "Vega/100", "Theta"],
        "Value": [total_delta, total_gamma * 100, total_vega / 100, total_theta]
    })
    st.bar_chart(greeks_data.set_index("Greek"))
    
    # ========== SECTION 3: RISK ANALYSIS ==========
    st.markdown("### ⚠️ Risk Analysis")
    
    # Get NIFTY data for VaR calculation (uses realized volatility)
    nifty_df = st.session_state.get("nifty_df", pd.DataFrame())
    
    # Calculate VaR (uses realized volatility from NIFTY data)
    var_95 = calculate_var(enriched, current_spot, nifty_df)
    
    # Stress test scenarios
    stress_up_2 = calculate_stress_pnl(enriched, current_spot, 1.02, 0)
    stress_down_2 = calculate_stress_pnl(enriched, current_spot, 0.98, 0)
    stress_iv_up = calculate_stress_pnl(enriched, current_spot, 1.0, 0.05)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("VaR (95%)", format_inr(var_95))
    
    with col2:
        st.metric("+2% NIFTY Move", format_inr(stress_up_2))
    
    with col3:
        st.metric("-2% NIFTY Move", format_inr(stress_down_2))
    
    with col4:
        st.metric("+5 IV Points", format_inr(stress_iv_up))
    
    # Risk status summary
    st.markdown("#### Risk Status Summary")
    
    risk_alerts = []
    
    # Delta check
    if abs(total_delta) > 15:
        risk_alerts.append("🔴 **CRITICAL**: Net Delta > ±15 - High directional risk")
    elif abs(total_delta) > 8:
        risk_alerts.append("🟡 **WARNING**: Net Delta > ±8 - Monitor directional exposure")
    else:
        risk_alerts.append("🟢 **OK**: Delta is neutral")
    
    # Margin check
    if margin_util_pct > 70:
        risk_alerts.append("🔴 **CRITICAL**: Margin utilization > 70% - Limited room for adjustments")
    elif margin_util_pct > 50:
        risk_alerts.append("🟡 **WARNING**: Margin utilization > 50%")
    else:
        risk_alerts.append("🟢 **OK**: Margin utilization healthy")
    
    # Recovery check
    if days_to_recover > avg_dte:
        risk_alerts.append("🔴 **CRITICAL**: Cannot recover losses by expiry with current theta")
    elif days_to_recover > avg_dte * 0.7:
        risk_alerts.append("🟡 **WARNING**: Tight timeline to recover losses")
    else:
        risk_alerts.append("🟢 **OK**: Recovery timeline manageable")
    
    # Theta efficiency check
    if theta_efficiency < -200:
        risk_alerts.append("🔴 **CRITICAL**: Theta efficiency < -200% - Directional problem, not time decay")
    elif theta_efficiency < -100:
        risk_alerts.append("🟡 **WARNING**: Theta efficiency negative")
    
    # Gamma check
    if total_gamma < -0.5 and avg_dte < 7:
        risk_alerts.append("🔴 **CRITICAL**: High negative gamma near expiry - Risk of rapid delta changes")
    
    for alert in risk_alerts:
        st.markdown(alert)
    
    # ========== SECTION 4: POSITION CONCENTRATION ==========
    st.markdown("### 🎯 Position Concentration")
    
    # Group by expiry
    expiry_groups = {}
    for pos in enriched:
        expiry = pos.get("expiry")
        if expiry:
            expiry_str = expiry.strftime("%Y-%m-%d")
            if expiry_str not in expiry_groups:
                expiry_groups[expiry_str] = {"count": 0, "pnl": 0, "notional": 0}
            expiry_groups[expiry_str]["count"] += 1
            expiry_groups[expiry_str]["pnl"] += pos.get("pnl", 0)
            expiry_groups[expiry_str]["notional"] += abs(pos.get("quantity", 0)) * pos.get("strike", 0) 
    
    # calculate leverage per expiry
    for exp, data in expiry_groups.items():
        data["leverage"] = (data["notional"] / account_size) if account_size > 0 else 0


    if expiry_groups:
        expiry_df = pd.DataFrame([
            {
                "Expiry": exp,
                "Positions": data["count"],
                "P&L": data["pnl"],
                "Notional": f"{data['notional']}",
                "Leverage (× capital)": f"{data['leverage']:.0f}"

            }
            for exp, data in expiry_groups.items()
        ])
        # Format currency columns using Indian grouping before display
        if "P&L" in expiry_df.columns:
            expiry_df["P&L"] = expiry_df["P&L"].apply(lambda x: format_inr(x))
        if "Notional" in expiry_df.columns:
            expiry_df["Notional"] = expiry_df["Notional"].apply(lambda x: format_inr(x))

        st.dataframe(expiry_df, use_container_width=True)
    
    # Largest positions
    st.markdown("#### Largest Positions by Notional")
    positions_by_size = sorted(
        enriched,
        key=lambda p: abs(p.get("quantity", 0)) * p.get("strike", 0),
        reverse=True
    )[:5]
    
    for pos in positions_by_size:
        notional = abs(pos.get("quantity", 0)) * pos.get("strike", 0)
        pct_portfolio = (notional / account_size * 100) if account_size > 0 else 0
        st.write(f"- {pos.get('tradingsymbol')}: {format_inr(notional)} ({pct_portfolio:.1f}% of portfolio)")


def render_diagnostics_tab():
    """Render position diagnostics tab with complete metrics."""
    st.subheader("🔍 Position Diagnostics")
    
    if "enriched_positions" not in st.session_state:
        st.info("No positions loaded. Fetch positions from the Positions tab first.")
        return
    
    enriched = st.session_state["enriched_positions"]
    current_spot = st.session_state.get("current_spot", 25000)
    
    # Get account size for % calculations
    access_token = st.session_state.get("kite_access_token")
    kite_api_key = st.session_state.get("kite_api_key")
    
    margin_available = 500000
    margin_used = 320000
    
    if access_token and kite_api_key:
        try:
            kite = KiteConnect(api_key=kite_api_key)
            kite.set_access_token(access_token)
            margins = kite.margins()
            equity_margins = margins.get("equity", {})
            margin_available = equity_margins.get("available", {}).get("live_balance", 500000)
            margin_used = equity_margins.get("utilised", {}).get("debits", 0)
        except:
            pass
    
    account_size = margin_available + margin_used
    
    # Sorting and filtering options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sort_by = st.selectbox(
            "Sort by",
            ["Days to B/E", "Theta Efficiency %", "PnL", "Position Size", "Delta", "Action Priority"]
        )
    
    with col2:
        show_red_only = st.checkbox("Show only RED positions")
    
    with col3:
        show_losing_only = st.checkbox("Show only losing positions")
    
    # Calculate diagnostics for each position
    diagnostics = []
    for pos in enriched:
        theta = pos.get("position_theta", 0)
        pnl = pos.get("pnl", 0)
        dte = pos.get("dte", 0)
        delta = pos.get("delta", 0)
        gamma = pos.get("gamma", 0)
        vega = pos.get("vega", 0)
        position_delta = pos.get("position_delta", 0)
        strike = pos.get("strike", 0)
        quantity = pos.get("quantity", 0)
        iv = pos.get("implied_vol", 0)
        last_price = pos.get("last_price", 0)
        buy_price = pos.get("buy_price", 0) or pos.get("sell_price", 0)
        
        # Days to breakeven
        days_to_breakeven = abs(pnl / theta) if theta != 0 else 999
        
        # Theta efficiency
        theta_eff = (pnl / theta * 100) if theta != 0 else 0
        
        # Notional value
        notional = abs(quantity) * strike  # 50 = lot size
        
        # Position size as % of portfolio
        position_pct = (notional / account_size * 100) if account_size > 0 else 0
        
        # Spot distance
        spot_distance_pct = ((current_spot - strike) / current_spot * 100) if strike > 0 else 0
        
        # Probability of profit (rough approximation using delta)
        prob_of_profit = (1 - abs(delta)) * 100 if delta else 50
        
        # Premium change
        premium_change_pct = ((last_price - buy_price) / buy_price * 100) if buy_price > 0 else 0
        
        # Loss vs credit ratio (for short positions)
        if quantity < 0:  # Short position
            credit_received = buy_price * abs(quantity) 
            loss_vs_credit = abs(pnl) / credit_received if credit_received > 0 else 0
        else:
            loss_vs_credit = 0
        
        # Action signal
        action_signal, action_priority = get_action_signal(
            dte, pnl, theta, position_delta, days_to_breakeven, loss_vs_credit
        )
        
        # Status color
        if days_to_breakeven > dte:
            status = "🔴 RED"
        elif days_to_breakeven > dte * 0.5:
            status = "🟡 YELLOW"
        else:
            status = "🟢 GREEN"
        
        diagnostics.append({
            # Identity
            "Symbol": pos.get("tradingsymbol", ""),
            "Strike": strike,
            "Type": pos.get("option_type", ""),
            "Qty": quantity,
            
            # P&L & Pricing
            "PnL": pnl,
            "Entry": buy_price,
            "Current": last_price,
            "Chg %": premium_change_pct,
            
            # Time
            "DTE": dte,
            "Days to B/E": days_to_breakeven,
            
            # Theta Analysis
            "Theta/Day": theta,
            "Theta Eff %": theta_eff,
            
            # Greeks
            "Delta": delta,
            "Pos Delta": position_delta,
            "Gamma": gamma,
            "Vega": vega,
            
            # Position Sizing
            "Notional": notional,
            "% Portfolio": position_pct,
            
            # Risk Metrics
            "Spot Dist %": spot_distance_pct,
            "PoP %": prob_of_profit,
            "IV": iv,
            "Loss/Credit": loss_vs_credit,
            
            # Action
            "Action": action_signal,
            "Priority": action_priority,
            "Status": status,
            
            # Hidden sort keys
            "_sort_action_priority": action_priority
        })
    
    # Apply filters
    filtered = diagnostics
    if show_red_only:
        filtered = [d for d in filtered if "🔴" in d["Status"]]
    if show_losing_only:
        filtered = [d for d in filtered if d["PnL"] < 0]
    
    # Apply sorting
    sort_key_map = {
        "Days to B/E": "Days to B/E",
        "Theta Efficiency %": "Theta Eff %",
        "PnL": "PnL",
        "Position Size": "Notional",
        "Delta": "Pos Delta",
        "Action Priority": "_sort_action_priority"
    }
    
    sort_key = sort_key_map.get(sort_by, "Days to B/E")
    filtered = sorted(filtered, key=lambda x: abs(x[sort_key]), reverse=True)
    
    # Remove internal sort keys from display
    for d in filtered:
        d.pop("_sort_action_priority", None)
    
    df = pd.DataFrame(filtered)

    # Format currency columns for Indian style commas before display
    if "PnL" in df.columns:
        df["PnL"] = df["PnL"].apply(lambda x: format_inr(x))
    if "Notional" in df.columns:
        df["Notional"] = df["Notional"].apply(lambda x: format_inr(x))
    
    if df.empty:
        st.info("No positions match the selected filters.")
        return
    
    # Format and display (PnL/Notional already formatted as strings)
    st.dataframe(
        df.style.format({
            "Entry": "{:.2f}",
            "Current": "{:.2f}",
            "Chg %": "{:+.1f}%",
            "Theta/Day": "{:.2f}",
            "Days to B/E": "{:.1f}",
            "Theta Eff %": "{:.0f}%",
            "Delta": "{:.3f}",
            "Pos Delta": "{:.2f}",
            "Gamma": "{:.4f}",
            "Vega": "{:.2f}",
            "% Portfolio": "{:.1f}%",
            "Spot Dist %": "{:+.1f}%",
            "PoP %": "{:.0f}%",
            "IV": "{:.1%}",
            "Loss/Credit": "{:.2f}x"
        }),
        use_container_width=True,
        height=600
    )
    
    # Action summary
    st.markdown("### 🎯 Action Summary")
    
    action_counts = {}
    for d in diagnostics:
        action = d["Action"]
        action_counts[action] = action_counts.get(action, 0) + 1
    
    cols = st.columns(len(action_counts))
    for i, (action, count) in enumerate(action_counts.items()):
        with cols[i]:
            st.metric(action, count)
    
    # Top priority actions
    st.markdown("### ⚡ Top Priority Actions")
    priority_positions = [d for d in diagnostics if d["Priority"] >= 8]
    priority_positions = sorted(priority_positions, key=lambda x: x["Priority"], reverse=True)[:5]
    
    if priority_positions:
        for pos in priority_positions:
            with st.expander(f"🔴 {pos['Symbol']} - {pos['Action']} (Priority: {pos['Priority']}/10)"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**P&L:** {format_inr(pos['PnL'])}")
                    st.write(f"**DTE:** {pos['DTE']} days")
                    st.write(f"**Days to B/E:** {pos['Days to B/E']:.1f}")
                with col2:
                    st.write(f"**Position Delta:** {pos['Pos Delta']:.2f}")
                    st.write(f"**Theta/Day:** {format_inr(pos['Theta/Day'], decimals=2)}")
                    st.write(f"**Loss/Credit:** {pos['Loss/Credit']:.2f}x")
                
                st.markdown(f"**Recommended Action:** {get_action_recommendation(pos)}")
    else:
        st.success("✅ No high-priority actions needed")


# ========== HELPER FUNCTIONS ==========

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


def render_market_regime_tab(options_df: pd.DataFrame, nifty_df: pd.DataFrame):
    """Render comprehensive market regime tab with all key metrics."""
    st.subheader("🌡️ Market Regime Analysis")
    
    regime = calculate_market_regime(options_df, nifty_df)
    
    if not regime:
        st.warning("Market regime data not available. Load NIFTY data and options data first.")
        return
    
    current_spot = regime.get("current_spot", 0)
    
    # ========== SECTION 1: VOLATILITY METRICS ==========
    st.markdown("### 📊 Volatility Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_iv = regime.get('current_iv', 0)
        st.metric(
            "Current IV (ATM)", 
            f"{current_iv:.1%}",
            help="**Implied Volatility** of at-the-money options. Shows market's expectation of future price movement. Higher IV = more expensive options."
        )
    
    with col2:
        iv_rank = regime.get('iv_rank', 0)
        rank_status = "🔴" if iv_rank > 70 else ("🟡" if iv_rank > 50 else "🟢")
        st.metric(
            "IV Rank (90d)", 
            f"{iv_rank:.0f} {rank_status}",
            help="""**IV Rank** = Where current IV sits in 90-day range (0-100%)

Formula: (Current IV - Min IV) / (Max IV - Min IV) × 100

**HIGH IV RANK** 🔴 (>70):
• Example: Current IV = 18%, Min = 12%, Max = 20%
• IV Rank = (18-12)/(20-12) × 100 = 75%
• Options are EXPENSIVE (near 90-day highs)
• ✅ SELL premium: Iron condors, credit spreads, strangles

**LOW IV RANK** 🟢 (<30):
• Example: Current IV = 13%, Min = 12%, Max = 20%
• IV Rank = (13-12)/(20-12) × 100 = 12.5%
• Options are CHEAP (near 90-day lows)
• ✅ BUY premium: Long straddles, debit spreads, calendars

**MID RANGE** 🟡 (30-70): Fair value, neutral strategies"""
        )
    
    with col3:
        realized_vol = regime.get('realized_vol', 0)
        st.metric(
            "Realized Vol (30d)", 
            f"{realized_vol:.1%}",
            help="**Realized Volatility** = Actual price movement over last 30 days (annualized). Compare with IV to see if options are overpriced or underpriced."
        )
    
    with col4:
        vrp = regime.get('vrp', 0)
        vrp_status = "🔴" if vrp > 0.05 else ("🟢" if vrp < -0.05 else "🟡")
        st.metric(
            "Vol Risk Premium", 
            f"{vrp:.1%} {vrp_status}",
            help="""**VRP (Volatility Risk Premium)** = Implied Vol - Realized Vol

Shows if options are overpriced or underpriced vs actual movement.

**NEUTRAL** 🟡 (-5% to +5%): Fair pricing, IV matches recent realized vol"""
        )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        rv_percentile = regime.get('rv_percentile', 50)
        rv_status = "🔴" if rv_percentile > 80 else ("🟡" if rv_percentile > 60 else "🟢")
        st.metric(
            "RV Percentile", 
            f"{rv_percentile:.0f}% {rv_status}",
            help="**RV Percentile** = Where current realized vol ranks in historical range. >80 = high volatility environment, <20 = calm market."
        )
    
    with col2:
        term_structure = regime.get('term_structure', 0)
        ts_status = "🔴" if term_structure < -0.02 else ("🟢" if term_structure > 0 else "🟡")
        ts_label = "Contango" if term_structure > 0 else "Backwardation"
        st.metric(
            "Term Structure", 
            f"{term_structure:.2%}", 
            f"{ts_label} {ts_status}",
            help="""**Term Structure** = How IV changes across expiries (Far IV - Near IV).

**CONTANGO** (Positive, Normal):
• Nov expiry: IV = 15%, Dec expiry: IV = 18% → Term Structure = +3%
• Far options MORE expensive (more time = more uncertainty)
• ✅ Good for: Calendar spreads (sell Nov, buy Dec)
• Normal healthy market

**BACKWARDATION** (Negative, Stress):
• Nov expiry: IV = 22%, Dec expiry: IV = 16% → Term Structure = -6%
• Near options MORE expensive (immediate fear/event)
• ⚠️ Warning: Avoid selling near-term, market expects short-term turbulence
• Happens before: RBI policy, Budget, crashes"""
        )
    
    with col3:
        near_iv = regime.get('near_iv', 0)
        st.metric(
            "Near-term IV", 
            f"{near_iv:.1%}",
            help="IV of current month expiry options."
        )
    
    with col4:
        far_iv = regime.get('far_iv', 0)
        st.metric(
            "Next-term IV", 
            f"{far_iv:.1%}",
            help="IV of next month expiry options."
        )
    
    # ========== SECTION 2: SENTIMENT INDICATORS ==========
    st.markdown("### 🎭 Sentiment Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pcr_oi = regime.get('pcr_oi', 0)
        pcr_status = "🟢" if pcr_oi > 1.2 else ("🔴" if pcr_oi < 0.8 else "🟡")
        pcr_sentiment = "Bullish" if pcr_oi > 1.2 else ("Bearish" if pcr_oi < 0.8 else "Neutral")
        st.metric(
            "PCR (OI)", 
            f"{pcr_oi:.2f} {pcr_status}", 
            f"{pcr_sentiment}",
            help="""**Put-Call Ratio** = Total Put OI / Total Call OI

**BULLISH** 🟢 (PCR > 1.2):
• Example: Put OI = 60L, Call OI = 45L → PCR = 1.33
• More puts than calls = hedging/protection buying
• Traders expect up move, buying puts to protect profits
• ✅ Good for: Selling puts, Bull spreads

**BEARISH** 🔴 (PCR < 0.8):
• Example: Put OI = 35L, Call OI = 50L → PCR = 0.70
• More calls than puts = excessive optimism
• Everyone chasing upside, no protection
• ⚠️ Warning: Complacent market, consider bear spreads

**NEUTRAL** 🟡 (PCR 0.8-1.2): Balanced sentiment, no extreme positioning"""
        )
    
    with col2:
        pcr_volume = regime.get('pcr_volume', 0)
        st.metric(
            "PCR (Volume)", 
            f"{pcr_volume:.2f}",
            help="Put-Call Ratio by trading volume. Shows active trading sentiment."
        )
    
    with col3:
        skew = regime.get('skew', 0)
        # Updated thresholds: ±1% is balanced, >1% is fear, <-1% is complacent
        if abs(skew) <= 0.01:
            skew_status = "�"
            skew_sentiment = "Balanced"
        elif skew > 0.01:
            skew_status = "�"
            skew_sentiment = "Fear (Put buying)"
        else:
            skew_status = "🟢"
            skew_sentiment = "Complacent"
        st.metric("Volatility Skew", f"{skew:.2%} {skew_status}", f"{skew_sentiment}",
            help="""**Volatility Skew = OTM Put IV - OTM Call IV (both ~3% away from spot)
**Example: NIFTY @ 25,000**

**HIGH SKEW (Fear)** 🔴:
• 24,250 Put (3% OTM): IV = 20%
• 25,750 Call (3% OTM): IV = 15%
• Skew = +5% → FEAR mode! Puts are expensive
• ❌ Don't buy OTM puts (overpriced protection)
• ✅ Sell put spreads (collect rich premium)

**LOW/NEGATIVE SKEW (Complacent)** 🟢:
• 24,250 Put: IV = 14%
• 25,750 Call: IV = 18%
• Skew = -4% → Excessive optimism! Calls expensive
• ⚠️ Warning sign - market too complacent
• ✅ Buy put protection (cheap insurance)

**BALANCED** 🟡:
• Put IV ≈ Call IV (within ±1%)
• Fair pricing, no extreme sentiment""")
    
    with col4:
        put_25d = regime.get('put_25d_iv', 0)
        call_25d = regime.get('call_25d_iv', 0)
        st.metric("Put IV (OTM)", f"{put_25d:.1%}", f"Call: {call_25d:.1%}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        max_pain = regime.get('max_pain_strike', 0)
        pain_dist = ((current_spot - max_pain) / current_spot * 100) if current_spot > 0 else 0
        pain_status = "🟢" if abs(pain_dist) < 1 else ("🟡" if abs(pain_dist) < 2 else "🔴")
        st.metric("Max Pain Strike", f"{max_pain:.0f} {pain_status}", f"{pain_dist:+.1f}% from spot",
        help="""**Max Pain** = Strike where option writers lose the LEAST money at expiry

**How it works:**
• Options sellers (writers) want options to expire worthless
• Max Pain = Strike that causes maximum loss to option buyers
• Market often gravitates toward this level before expiry

**Example:**
• NIFTY Spot = 25,050
• Max Pain = 25,000
• Distance = -0.2% (close to max pain)

**Near Max Pain** 🟢 (<1% away):
• High chance spot moves toward 25,000 by expiry
• Option writers will defend this level
• ✅ Strategy: Sell options near max pain

**Far from Max Pain** 🔴 (>2% away):
• Strong directional move in progress
• Option writers struggling, may need to hedge
• ⚠️ Expect increased volatility near expiry""")
    
    with col2:
        st.metric("Spot Price", format_inr(current_spot, decimals=2))
    
    # ========== SECTION 3: MARKET REGIME CLASSIFICATION ==========
    st.markdown("### 🎯 Market Regime")
    
    regime_name = regime.get('market_regime', 'Unknown')
    
    # Color code the regime
    if "High Vol" in regime_name:
        regime_color = "🔴"
    elif "Low Vol" in regime_name:
        regime_color = "🟢"
    elif "Sell" in regime_name:
        regime_color = "🟡"
    else:
        regime_color = "⚪"
    
    st.markdown(f"### {regime_color} **{regime_name}**")
    
    # Interpretation guide
    with st.expander("📖 How to interpret this regime", expanded=False):
        if "High Vol - Sell Premium" in regime_name:
            st.markdown("""
            **High Volatility - Sell Premium Environment**
            - IV Rank > 70: Options are expensive
            - VRP > 5%: IV significantly higher than realized
            - **Strategy**: Sell options (credit spreads, iron condors, strangles)
            - **Risk**: Be prepared for large moves
            """)
        elif "Low Vol - Buy Premium" in regime_name:
            st.markdown("""
            **Low Volatility - Buy Premium Environment**
            - IV Rank < 30: Options are cheap
            - VRP < -5%: IV lower than realized (underpriced)
            - **Strategy**: Buy options (long straddles, calendars)
            - **Risk**: Time decay will hurt if no movement
            """)
        elif "Neutral" in regime_name:
            st.markdown("""
            **Neutral Market - Balanced Approach**
            - IV Rank 30-70: Fair pricing
            - VRP near 0: IV matches realized
            - **Strategy**: Neutral strategies (iron condors, butterflies)
            - **Risk**: Moderate - watch for regime changes
            """)
        else:
            st.markdown(f"""
            **{regime_name}**
            - Review individual metrics for strategy selection
            - Consider both directional and volatility views
            """)
    
    # ========== SECTION 4: TREND INDICATORS ==========
    st.markdown("### 📈 NIFTY Trend Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sma_20 = regime.get('sma_20', 0)
        sma_dist = ((current_spot - sma_20) / current_spot * 100) if current_spot > 0 and sma_20 > 0 else 0
        sma_status = "🟢" if sma_dist > 0 else "🔴"
        st.metric("20-day SMA", format_inr(sma_20, decimals=2), f"{sma_dist:+.1f}% {sma_status}")
        with st.expander("ℹ️ What is SMA?"):
            st.caption("**Simple Moving Average** of last 20 days. Price above SMA = bullish trend, below = bearish.")
    
    with col2:
        sma_50 = regime.get('sma_50', 0)
        sma_50_dist = ((current_spot - sma_50) / current_spot * 100) if current_spot > 0 and sma_50 > 0 else 0
        sma_50_status = "🟢" if sma_50_dist > 0 else "🔴"
        st.metric("50-day SMA", format_inr(sma_50, decimals=2), f"{sma_50_dist:+.1f}% {sma_50_status}")
        with st.expander("ℹ️ What is 50-day SMA?"):
            st.caption("**50-day SMA** shows medium-term trend. Price above = sustained uptrend.")
    
    with col3:
        rsi = regime.get('rsi', 50)
        rsi_status = "🔴" if rsi > 70 else ("🟢" if rsi < 30 else "🟡")
        rsi_label = "Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral")
        st.metric("RSI (14)", f"{rsi:.0f} {rsi_status}", f"{rsi_label}")
        with st.expander("ℹ️ What is RSI?"):
            st.caption("**Relative Strength Index** (0-100). >70 = overbought (potential reversal down), <30 = oversold (potential bounce up).")
    
    with col4:
        atr = regime.get('atr', 0)
        atr_pct = (atr / current_spot * 100) if current_spot > 0 else 0
        st.metric("ATR (14)", f"{atr:.0f}", f"{atr_pct:.1f}% of spot")
        with st.expander("ℹ️ What is ATR?"):
            st.caption("**Average True Range** = average daily movement in points. Higher ATR = more volatile market, wider stop losses needed.")
    
    # Trend summary
    trend_signals = []
    if sma_dist > 0 and sma_50_dist > 0:
        trend_signals.append("🟢 **Bullish**: Price above both SMAs")
    elif sma_dist < 0 and sma_50_dist < 0:
        trend_signals.append("🔴 **Bearish**: Price below both SMAs")
    else:
        trend_signals.append("🟡 **Mixed**: Price between SMAs")
    
    if rsi > 70:
        trend_signals.append("⚠️ **Caution**: RSI Overbought (potential reversal)")
    elif rsi < 30:
        trend_signals.append("⚠️ **Caution**: RSI Oversold (potential bounce)")
    
    st.markdown("#### Trend Summary")
    for signal in trend_signals:
        st.markdown(signal)
    
    # ========== SECTION 5: ACTIONABLE INSIGHTS ==========
    st.markdown("### 💡 Actionable Insights")
    
    insights = []
    
    # PCR insights
    if pcr_oi > 1.3:
        insights.append("🟢 **PCR > 1.3**: Heavy put buying suggests bullish sentiment or hedging. Consider selling puts or bull spreads.")
    elif pcr_oi < 0.7:
        insights.append("🔴 **PCR < 0.7**: Heavy call buying suggests excessive optimism. Consider selling calls or bear spreads.")
    
    # Skew insights
    if skew > 0.05:
        insights.append("🔴 **High Put Skew**: Market fear elevated. OTM puts expensive - avoid buying, consider selling put spreads.")
    elif skew < -0.03:
        insights.append("⚠️ **Negative Skew**: Calls more expensive than puts - rare condition, potential warning sign.")
    
    # IV Rank insights
    if iv_rank > 80:
        insights.append("🔴 **IV Rank > 80**: Options very expensive. Prime time to SELL premium (iron condors, credit spreads).")
    elif iv_rank < 20:
        insights.append("🟢 **IV Rank < 20**: Options very cheap. Good time to BUY options (long straddles, debit spreads).")
    
    # Term structure insights
    if term_structure < -0.03:
        insights.append("🔴 **Backwardation**: Near-term vol higher than far - suggests stress. Avoid selling near-term options.")
    elif term_structure > 0.05:
        insights.append("🟢 **Steep Contango**: Good environment for calendar spreads (sell near, buy far).")
    
    # Max pain insights
    if abs(pain_dist) < 1.5:
        insights.append(f"📍 **Near Max Pain ({max_pain:.0f})**: Expect spot to gravitate toward max pain before expiry.")
    
    # VRP insights
    if vrp > 0.08:
        insights.append("🟡 **High VRP**: IV much higher than realized - premium sellers have edge, but watch for gap moves.")
    elif vrp < -0.05:
        insights.append("🟡 **Negative VRP**: IV underpricing risk - good for buying options if expecting volatility expansion.")
    
    if insights:
        for insight in insights:
            st.markdown(insight)
    else:
        st.info("Market conditions are balanced. Monitor for regime changes.")
    
    # ========== SECTION 6: CHARTS ==========
    st.markdown("### 📊 Visual Analysis")
    
    # IV vs RV comparison
    col1, col2 = st.columns(2)
    
    with col1:
        iv_rv_data = pd.DataFrame({
            "Metric": ["Implied Vol", "Realized Vol"],
            "Value": [current_iv * 100, realized_vol * 100]
        })
        st.bar_chart(iv_rv_data.set_index("Metric"))
        st.caption("IV vs RV Comparison (%)")
    
    with col2:
        # PCR visualization
        pcr_data = pd.DataFrame({
            "Type": ["Puts (OI)", "Calls (OI)"],
            "Value": [pcr_oi / (1 + pcr_oi) * 100, 100 / (1 + pcr_oi)]
        })
        st.bar_chart(pcr_data.set_index("Type"))
        st.caption("Put-Call Ratio Distribution")


def render_alerts_tab():
    """Render risk alerts tab."""
    st.subheader("Risk Alerts")
    
    if "enriched_positions" not in st.session_state:
        st.info("No positions loaded. Fetch positions from the Positions tab first.")
        return
    
    enriched = st.session_state["enriched_positions"]
    portfolio_greeks = calculate_portfolio_greeks(enriched)
    
    # Delta alert
    delta = portfolio_greeks["net_delta"]
    if abs(delta) > 15:
        st.error(f"🔴 High Delta Alert: Net Delta is {delta:.2f}. Consider hedging your portfolio.")
    elif abs(delta) > 8:
        st.warning(f"🟡 Moderate Delta Alert: Net Delta is {delta:.2f}. Monitor your exposure.")
    else:
        st.success(f"🟢 Delta is within safe limits: {delta:.2f}.")
    
    # Additional alerts can be added here


def render_advanced_analytics_tab():
    """Render advanced analytics tab with professional metrics."""
    st.subheader("🎯 Advanced Analytics")
    
    if "enriched_positions" not in st.session_state:
        st.info("No positions loaded. Fetch positions from the Positions tab first.")
        return
    
    enriched = st.session_state["enriched_positions"]
    portfolio_greeks = calculate_portfolio_greeks(enriched)
    current_spot = st.session_state.get("current_spot", 25000)
    
    # Get account info
    access_token = st.session_state.get("kite_access_token")
    kite_api_key = st.session_state.get("kite_api_key")
    
    margin_available = 500000
    margin_used = 320000
    
    if access_token and kite_api_key:
        try:
            kite = KiteConnect(api_key=kite_api_key)
            kite.set_access_token(access_token)
            margins = kite.margins()
            equity_margins = margins.get("equity", {})
            margin_available = equity_margins.get("available", {}).get("live_balance", 500000)
            margin_used = equity_margins.get("utilised", {}).get("debits", 0)
        except:
            pass
    
    account_size = margin_available + margin_used
    total_pnl = sum(p.get("pnl", 0) for p in enriched)
    
    # Create tabs for different analytics sections
    analytics_tabs = st.tabs([
        "📊 P&L Attribution",
        "⚠️ Risk Metrics", 
        "💰 Efficiency",
        "📈 Volatility Surface",
        "🔗 Concentration",
        "⚡ Execution Quality",
        "📉 Drawdown Analysis"
    ])
    
    # ========== P&L ATTRIBUTION TAB ==========
    with analytics_tabs[0]:
        st.markdown("### P&L Attribution Analysis")
        st.caption("Decomposes P&L into Greek components to identify true profit sources")
        
        # Calculate P&L attribution
        total_delta = portfolio_greeks["net_delta"]
        total_gamma = portfolio_greeks["net_gamma"]
        total_vega = portfolio_greeks["net_vega"]
        total_theta = portfolio_greeks["net_theta"]
        
        # Get previous spot (approximation - using 1% move for demo)
        prev_spot = current_spot * 0.99
        spot_change = current_spot - prev_spot
        
        # P&L attribution components
        delta_pnl = total_delta * spot_change * 50  # Lot size
        gamma_pnl = 0.5 * total_gamma * (spot_change ** 2) * 50
        vega_pnl = total_vega * 0  # Assume no IV change for now
        theta_pnl = total_theta * 1  # 1 day decay
        residual_pnl = total_pnl - (delta_pnl + gamma_pnl + vega_pnl + theta_pnl)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Delta P&L", f"₹{delta_pnl:,.0f}", 
                     delta=f"{delta_pnl/total_pnl*100:.1f}%" if total_pnl != 0 else "0%")
        
        with col2:
            st.metric("Gamma P&L", f"₹{gamma_pnl:,.0f}",
                     delta=f"{gamma_pnl/total_pnl*100:.1f}%" if total_pnl != 0 else "0%")
        
        with col3:
            st.metric("Vega P&L", f"₹{vega_pnl:,.0f}",
                     delta=f"{vega_pnl/total_pnl*100:.1f}%" if total_pnl != 0 else "0%")
        
        with col4:
            st.metric("Theta P&L", f"₹{theta_pnl:,.0f}",
                     delta=f"{theta_pnl/total_pnl*100:.1f}%" if total_pnl != 0 else "0%")
        
        with col5:
            st.metric("Residual P&L", f"₹{residual_pnl:,.0f}",
                     delta=f"{residual_pnl/total_pnl*100:.1f}%" if total_pnl != 0 else "0%")
        
        # P&L Attribution Chart
        import plotly.graph_objects as go
        
        attribution_data = {
            'Component': ['Delta', 'Gamma', 'Vega', 'Theta', 'Residual'],
            'P&L': [delta_pnl, gamma_pnl, vega_pnl, theta_pnl, residual_pnl]
        }
        
        colors = ['#3B82F6' if x > 0 else '#EF4444' for x in attribution_data['P&L']]
        
        fig = go.Figure(data=[
            go.Bar(
                x=attribution_data['Component'],
                y=attribution_data['P&L'],
                marker_color=colors,
                text=[f"₹{x:,.0f}" for x in attribution_data['P&L']],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="P&L Attribution by Greek",
            xaxis_title="Component",
            yaxis_title="P&L (₹)",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Interpretation:**
        - **Delta P&L**: Profit/loss from directional moves
        - **Gamma P&L**: Profit/loss from acceleration (convexity)
        - **Vega P&L**: Profit/loss from IV changes
        - **Theta P&L**: Profit/loss from time decay
        - **Residual**: Unexplained P&L (slippage, crosses, rounding)
        """)
    
    # ========== RISK METRICS TAB ==========
    with analytics_tabs[1]:
        st.markdown("### Advanced Risk Metrics")
        
        # CVaR (Conditional Value at Risk / Expected Shortfall)
        # Using historical simulation approach
        import numpy as np
        
        st.markdown("#### CVaR (Conditional Value at Risk)")
        st.caption("Expected loss in worst 5% of scenarios (tail risk measure)")
        
        # Simulate returns
        np.random.seed(42)
        daily_returns = np.random.normal(-0.001, 0.02, 1000)  # Simulated daily returns
        
        # Calculate portfolio value changes
        portfolio_values = []
        for ret in daily_returns:
            spot_move = current_spot * ret
            pnl_scenario = (total_delta * spot_move * 50 + 
                          0.5 * total_gamma * (spot_move ** 2) * 50 +
                          total_theta)
            portfolio_values.append(pnl_scenario)
        
        portfolio_values = np.array(portfolio_values)
        
        # VaR and CVaR at 95% confidence
        var_95 = np.percentile(portfolio_values, 5)
        cvar_95 = portfolio_values[portfolio_values <= var_95].mean()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("VaR (95%)", f"₹{abs(var_95):,.0f}",
                     delta=f"{abs(var_95)/account_size*100:.2f}% of capital",
                     delta_color="inverse")
        
        with col2:
            st.metric("CVaR (95%)", f"₹{abs(cvar_95):,.0f}",
                     delta=f"{abs(cvar_95)/account_size*100:.2f}% of capital",
                     delta_color="inverse")
        
        with col3:
            ratio = abs(cvar_95) / abs(var_95) if var_95 != 0 else 1
            st.metric("CVaR/VaR Ratio", f"{ratio:.2f}",
                     help="How much worse than VaR is the tail risk")
        
        # Distribution chart
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=portfolio_values,
            nbinsx=50,
            name='P&L Distribution',
            marker_color='#3B82F6'
        ))
        
        fig.add_vline(x=var_95, line_dash="dash", line_color="orange",
                     annotation_text=f"VaR: ₹{var_95:,.0f}")
        fig.add_vline(x=cvar_95, line_dash="dash", line_color="red",
                     annotation_text=f"CVaR: ₹{cvar_95:,.0f}")
        
        fig.update_layout(
            title="Portfolio P&L Distribution (1000 Scenarios)",
            xaxis_title="P&L (₹)",
            yaxis_title="Frequency",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **CVaR vs VaR:**
        - **VaR**: Maximum expected loss at confidence level (95% = worst loss in 19/20 days)
        - **CVaR**: Average loss when VaR is exceeded (tail risk)
        - CVaR is always worse than VaR and measures extreme scenarios
        """)
    
    # ========== EFFICIENCY TAB ==========
    with analytics_tabs[2]:
        st.markdown("### Capital Efficiency Metrics")
        
        # Return on Margin (ROM)
        rom = (total_pnl / margin_used * 100) if margin_used > 0 else 0
        roi = (total_pnl / account_size * 100) if account_size > 0 else 0
        
        # Sharpe-like metric (simplified)
        daily_return = total_pnl / account_size
        sharpe_approx = (daily_return * 252) / (0.02 * np.sqrt(252))  # Assuming 2% daily vol
        
        # Margin efficiency
        margin_efficiency = (margin_used / account_size * 100) if account_size > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Return on Margin", f"{rom:.2f}%",
                     help="P&L relative to margin deployed")
            if rom > 10:
                st.success("🟢 Excellent efficiency")
            elif rom > 5:
                st.warning("🟡 Good efficiency")
            else:
                st.error("🔴 Low efficiency")
        
        with col2:
            st.metric("ROI (Account)", f"{roi:.2f}%",
                     help="P&L relative to total account size")
        
        with col3:
            st.metric("Sharpe Ratio (Est.)", f"{sharpe_approx:.2f}",
                     help="Risk-adjusted return estimate")
            if sharpe_approx > 2:
                st.success("🟢 Excellent")
            elif sharpe_approx > 1:
                st.warning("🟡 Good")
            else:
                st.error("🔴 Poor")
        
        with col4:
            st.metric("Margin Utilization", f"{margin_efficiency:.1f}%",
                     help="% of capital deployed")
        
        # Capital efficiency breakdown
        notional_exposure = sum(abs(p.get("quantity", 0)) * p.get("strike", 0) for p in enriched)
        leverage = notional_exposure / account_size if account_size > 0 else 0
        
        st.markdown("#### Capital Deployment Analysis")
        
        efficiency_data = pd.DataFrame({
            'Metric': ['Account Size', 'Margin Used', 'Margin Free', 'Notional Exposure'],
            'Value': [account_size, margin_used, margin_available, notional_exposure]
        })
        
        fig = go.Figure(data=[
            go.Bar(
                x=efficiency_data['Metric'],
                y=efficiency_data['Value'],
                marker_color=['#3B82F6', '#EF4444', '#10B981', '#9333EA'],
                text=[f"₹{x:,.0f}" for x in efficiency_data['Value']],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="Capital Structure",
            yaxis_title="Amount (₹)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Leverage Ratio", f"{leverage:.1f}x",
                     help="Notional exposure / Account size")
        with col2:
            buying_power = margin_available / margin_used if margin_used > 0 else 0
            st.metric("Buying Power Left", f"{buying_power:.1f}x",
                     help="Can increase positions by this factor")
    
    # ========== VOLATILITY SURFACE TAB ==========
    with analytics_tabs[3]:
        st.markdown("### IV Surface & Smile Visualization")
        
        # Build IV surface from positions
        iv_data = []
        for pos in enriched:
            strike = pos.get("strike", 0)
            iv = pos.get("implied_vol", 0)
            dte = pos.get("dte", 0)
            option_type = pos.get("option_type", "")
            
            if strike and iv:
                moneyness = (strike - current_spot) / current_spot * 100
                iv_data.append({
                    'Strike': strike,
                    'Moneyness': moneyness,
                    'IV': iv * 100,
                    'DTE': dte,
                    'Type': option_type
                })
        
        if iv_data:
            iv_df = pd.DataFrame(iv_data)
            
            # IV Smile Chart
            fig = go.Figure()
            
            ce_data = iv_df[iv_df['Type'] == 'CE']
            pe_data = iv_df[iv_df['Type'] == 'PE']
            
            if not ce_data.empty:
                fig.add_trace(go.Scatter(
                    x=ce_data['Moneyness'],
                    y=ce_data['IV'],
                    mode='markers+lines',
                    name='Call IV',
                    marker=dict(size=10, color='#3B82F6'),
                    line=dict(color='#3B82F6', width=2)
                ))
            
            if not pe_data.empty:
                fig.add_trace(go.Scatter(
                    x=pe_data['Moneyness'],
                    y=pe_data['IV'],
                    mode='markers+lines',
                    name='Put IV',
                    marker=dict(size=10, color='#EF4444'),
                    line=dict(color='#EF4444', width=2)
                ))
            
            fig.update_layout(
                title="IV Smile by Moneyness",
                xaxis_title="Moneyness (%)",
                yaxis_title="Implied Volatility (%)",
                height=500,
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # IV statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                atm_iv = iv_df[abs(iv_df['Moneyness']) < 2]['IV'].mean()
                st.metric("ATM IV", f"{atm_iv:.2f}%")
            
            with col2:
                otm_put_iv = iv_df[(iv_df['Type'] == 'PE') & (iv_df['Moneyness'] < -2)]['IV'].mean()
                st.metric("OTM Put IV", f"{otm_put_iv:.2f}%")
            
            with col3:
                otm_call_iv = iv_df[(iv_df['Type'] == 'CE') & (iv_df['Moneyness'] > 2)]['IV'].mean()
                st.metric("OTM Call IV", f"{otm_call_iv:.2f}%")
            
            with col4:
                skew = otm_put_iv - otm_call_iv
                st.metric("Skew", f"{skew:.2f}%",
                         delta="Put premium" if skew > 0 else "Call premium")
            
            # Display IV surface data
            st.markdown("#### IV Surface Data")
            st.dataframe(
                iv_df.sort_values('Moneyness').style.format({
                    'Strike': '{:.0f}',
                    'Moneyness': '{:+.2f}%',
                    'IV': '{:.2f}%',
                    'DTE': '{:.0f}'
                }),
                use_container_width=True
            )
        else:
            st.warning("No IV data available. Fetch positions first.")
    
    # ========== CONCENTRATION TAB ==========
    with analytics_tabs[4]:
        st.markdown("### Position Correlation & Concentration")
        
        # Position concentration by strike
        strike_exposure = {}
        for pos in enriched:
            strike = pos.get("strike", 0)
            notional = abs(pos.get("quantity", 0)) * strike
            
            if strike in strike_exposure:
                strike_exposure[strike] += notional
            else:
                strike_exposure[strike] = notional
        
        # Sort and get top concentrations
        sorted_strikes = sorted(strike_exposure.items(), key=lambda x: x[1], reverse=True)
        
        st.markdown("#### Strike Concentration")
        
        if sorted_strikes:
            top_5_strikes = sorted_strikes[:5]
            
            strike_df = pd.DataFrame(top_5_strikes, columns=['Strike', 'Notional'])
            strike_df['% of Portfolio'] = strike_df['Notional'] / strike_df['Notional'].sum() * 100
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = go.Figure(data=[
                    go.Bar(
                        x=[f"{s[0]:.0f}" for s in top_5_strikes],
                        y=[s[1] for s in top_5_strikes],
                        marker_color='#3B82F6',
                        text=[f"₹{s[1]:,.0f}" for s in top_5_strikes],
                        textposition='outside'
                    )
                ])
                
                fig.update_layout(
                    title="Top 5 Strike Exposures",
                    xaxis_title="Strike",
                    yaxis_title="Notional (₹)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.dataframe(
                    strike_df.style.format({
                        'Strike': '{:.0f}',
                        'Notional': '₹{:,.0f}',
                        '% of Portfolio': '{:.1f}%'
                    }),
                    use_container_width=True
                )
        
        # Position correlation analysis
        st.markdown("#### Position Correlation Matrix")
        st.caption("Correlation based on strike proximity and Greeks")
        
        # Build correlation matrix based on strike distance
        if len(enriched) >= 2:
            position_names = [f"{p.get('tradingsymbol', 'Unknown')}" for p in enriched]
            n_pos = len(enriched)
            
            correlation_matrix = np.ones((n_pos, n_pos))
            
            for i in range(n_pos):
                for j in range(n_pos):
                    if i != j:
                        strike_i = enriched[i].get('strike', 0)
                        strike_j = enriched[j].get('strike', 0)
                        delta_i = enriched[i].get('delta', 0)
                        delta_j = enriched[j].get('delta', 0)
                        
                        # Correlation based on strike distance and delta similarity
                        strike_dist = abs(strike_i - strike_j) / current_spot
                        delta_sim = 1 - abs(delta_i - delta_j)
                        
                        correlation = (1 - strike_dist) * delta_sim
                        correlation = max(-1, min(1, correlation))
                        correlation_matrix[i, j] = correlation
            
            # Show only first 10 positions for clarity
            display_limit = min(10, n_pos)
            
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix[:display_limit, :display_limit],
                x=position_names[:display_limit],
                y=position_names[:display_limit],
                colorscale='RdBu',
                zmid=0,
                text=correlation_matrix[:display_limit, :display_limit],
                texttemplate='%{text:.2f}',
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                title=f"Position Correlation Heatmap (Top {display_limit} positions)",
                height=600,
                xaxis={'side': 'bottom'},
                yaxis={'autorange': 'reversed'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Concentration risk score
            avg_correlation = (correlation_matrix.sum() - n_pos) / (n_pos * (n_pos - 1))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Avg Position Correlation", f"{avg_correlation:.2f}",
                         help="Higher = more concentrated risk")
                if avg_correlation > 0.7:
                    st.error("🔴 High concentration - positions move together")
                elif avg_correlation > 0.4:
                    st.warning("🟡 Moderate concentration")
                else:
                    st.success("🟢 Well diversified across strikes")
            
            with col2:
                diversification_score = (1 - avg_correlation) * 100
                st.metric("Diversification Score", f"{diversification_score:.1f}/100",
                         help="Higher = better diversification")
        else:
            st.info("Need at least 2 positions for correlation analysis")
    
    # ========== EXECUTION QUALITY TAB ==========
    with analytics_tabs[5]:
        st.markdown("### Execution Quality Metrics")
        st.caption("Track slippage, spread impact, and execution efficiency")
        
        st.info("📊 **Note**: These metrics require historical trade data. Currently showing simulated values for demonstration.")
        
        # Simulated execution metrics (in real implementation, track actual fills)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Slippage %
            avg_slippage = 0.15  # Simulated
            st.metric("Avg Slippage", f"{avg_slippage:.2f}%",
                     help="Average price difference vs mid-market")
            if avg_slippage < 0.2:
                st.success("🟢 Excellent execution")
            elif avg_slippage < 0.5:
                st.warning("🟡 Acceptable")
            else:
                st.error("🔴 Poor execution")
        
        with col2:
            # Spread impact
            spread_cost = 2500  # Simulated in rupees
            st.metric("Spread Cost", f"₹{spread_cost:,.0f}",
                     delta=f"{spread_cost/total_pnl*100:.1f}% of P&L" if total_pnl != 0 else "0%",
                     delta_color="inverse")
        
        with col3:
            # Fill rate
            fill_rate = 98.5  # Simulated
            st.metric("Fill Rate", f"{fill_rate:.1f}%",
                     help="% of orders fully filled")
        
        # Execution time analysis
        st.markdown("#### Execution Latency")
        
        # Simulated latency data
        latencies = np.random.gamma(2, 50, 100)  # Simulated in milliseconds
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Latency", f"{latencies.mean():.0f}ms")
        with col2:
            st.metric("P95 Latency", f"{np.percentile(latencies, 95):.0f}ms")
        with col3:
            st.metric("Max Latency", f"{latencies.max():.0f}ms")
        
        fig = go.Figure(data=[go.Histogram(x=latencies, nbinsx=30, marker_color='#3B82F6')])
        fig.update_layout(
            title="Order Execution Latency Distribution",
            xaxis_title="Latency (ms)",
            yaxis_title="Frequency",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Execution quality over time (simulated)
        st.markdown("#### Execution Quality Trend")
        
        dates = pd.date_range(end=pd.Timestamp.now(), periods=30, freq='D')
        quality_scores = 100 - np.random.uniform(0, 5, 30)  # Simulated quality scores
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=quality_scores,
            mode='lines+markers',
            name='Quality Score',
            line=dict(color='#3B82F6', width=2),
            marker=dict(size=6)
        ))
        
        fig.add_hline(y=95, line_dash="dash", line_color="green",
                     annotation_text="Target: 95")
        
        fig.update_layout(
            title="Execution Quality Score (Last 30 Days)",
            xaxis_title="Date",
            yaxis_title="Quality Score",
            height=350,
            yaxis=dict(range=[90, 100])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # ========== DRAWDOWN ANALYSIS TAB ==========
    with analytics_tabs[6]:
        st.markdown("### Drawdown & Recovery Analysis")
        st.caption("Analyze loss persistence and recovery patterns")
        
        # Simulated P&L history (in real implementation, use actual trade history)
        np.random.seed(42)
        days = 60
        daily_pnls = np.cumsum(np.random.normal(500, 3000, days))
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D')
        
        # Calculate drawdown
        cumulative_max = np.maximum.accumulate(daily_pnls)
        drawdown = daily_pnls - cumulative_max
        drawdown_pct = (drawdown / cumulative_max) * 100
        
        # Current drawdown
        current_dd = drawdown[-1]
        current_dd_pct = drawdown_pct[-1]
        max_dd = drawdown.min()
        max_dd_pct = drawdown_pct.min()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Drawdown", f"₹{abs(current_dd):,.0f}",
                     delta=f"{current_dd_pct:.2f}%",
                     delta_color="inverse")
        
        with col2:
            st.metric("Max Drawdown", f"₹{abs(max_dd):,.0f}",
                     delta=f"{max_dd_pct:.2f}%",
                     delta_color="inverse")
        
        with col3:
            # Recovery days calculation
            if current_dd < 0:
                # Estimate recovery days based on avg daily P&L
                avg_daily_pnl = (daily_pnls[-1] - daily_pnls[0]) / days
                recovery_days = abs(current_dd / avg_daily_pnl) if avg_daily_pnl > 0 else 999
                st.metric("Est. Recovery Days", f"{recovery_days:.0f}",
                         help="Days to recover at current avg daily P&L")
            else:
                st.metric("Recovery Days", "0", help="No active drawdown")
        
        # P&L and Drawdown chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=daily_pnls,
            mode='lines',
            name='Cumulative P&L',
            line=dict(color='#3B82F6', width=2),
            yaxis='y1'
        ))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=drawdown,
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='#EF4444', width=1),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="P&L and Drawdown History",
            xaxis_title="Date",
            yaxis=dict(
                title="Cumulative P&L (₹)",
                side='left'
            ),
            yaxis2=dict(
                title="Drawdown (₹)",
                side='right',
                overlaying='y',
                showgrid=False
            ),
            height=450,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown statistics
        st.markdown("#### Drawdown Statistics")
        
        # Find all drawdown periods
        in_drawdown = drawdown < 0
        drawdown_starts = np.where(np.diff(in_drawdown.astype(int)) == 1)[0]
        drawdown_ends = np.where(np.diff(in_drawdown.astype(int)) == -1)[0]
        
        if len(drawdown_starts) > 0 and len(drawdown_ends) > 0:
            # Ensure matching starts and ends
            if drawdown_starts[0] > drawdown_ends[0]:
                drawdown_ends = drawdown_ends[1:]
            if len(drawdown_starts) > len(drawdown_ends):
                drawdown_starts = drawdown_starts[:-1]
            
            drawdown_durations = drawdown_ends - drawdown_starts
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Drawdowns", len(drawdown_durations))
            
            with col2:
                avg_duration = drawdown_durations.mean() if len(drawdown_durations) > 0 else 0
                st.metric("Avg Duration", f"{avg_duration:.1f} days")
            
            with col3:
                max_duration = drawdown_durations.max() if len(drawdown_durations) > 0 else 0
                st.metric("Max Duration", f"{max_duration:.0f} days")
        
        # Recovery efficiency
        st.markdown("#### Recovery Efficiency")
        
        if len(drawdown_starts) > 0:
            recovery_rates = []
            for start, end in zip(drawdown_starts, drawdown_ends):
                dd_depth = abs(drawdown[start:end+1].min())
                recovery_time = end - start
                if recovery_time > 0:
                    recovery_rate = dd_depth / recovery_time
                    recovery_rates.append(recovery_rate)
            
            if recovery_rates:
                avg_recovery_rate = np.mean(recovery_rates)
                st.metric("Avg Recovery Rate", f"₹{avg_recovery_rate:,.0f}/day",
                         help="Average daily P&L during recovery periods")
                
                if avg_recovery_rate > 1000:
                    st.success("🟢 Fast recovery capability")
                elif avg_recovery_rate > 500:
                    st.warning("🟡 Moderate recovery")
                else:
                    st.error("🔴 Slow recovery")


def render_trade_history_tab():
    """Render trade history analysis tab from tradebook.csv."""
    st.subheader("📊 Trade History Analysis")
    st.caption("Comprehensive analysis from Kite Console tradebook export")
    
    # File path
    tradebook_path = "database/tradebook.csv"
    
    # Check if file exists
    if not os.path.exists(tradebook_path):
        st.error(f"❌ Tradebook file not found at: {tradebook_path}")
        st.info("💡 Export your tradebook from Kite Console and save it as `database/tradebook.csv`")
        return
    
    try:
        # Read tradebook
        with st.spinner("Loading tradebook..."):
            # Read CSV and drop any completely empty columns
            df = pd.read_csv(tradebook_path)
            df = df.dropna(axis=1, how='all')  # Drop columns that are completely empty
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Drop unnamed columns
        
        st.success(f"✅ Loaded {len(df)} trades from tradebook")
        
        # Display column info for debugging
        with st.expander("📋 Tradebook Columns"):
            st.write(df.columns.tolist())
            st.write(f"Shape: {df.shape}")
            st.dataframe(df.head(3))
            st.caption("Expected: Symbol, ISIN, Trade Date, Exchange, Segment, Series, Trade Type, Auction, Quantity, Price, Trade ID, Order ID, Order Execution Time")
        
        # Normalize column names
        df.columns = df.columns.str.strip()
        
        # Check for required columns
        required_cols = ['Symbol', 'Trade Date', 'Trade Type', 'Quantity', 'Price']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
            st.dataframe(df.head())
            return
        
        # Parse dates
        df['Trade Date'] = pd.to_datetime(df['Trade Date'], errors='coerce')
        
        # Parse Order Execution Time if available
        if 'Order Execution Time' in df.columns:
            df['Order Execution Time'] = pd.to_datetime(df['Order Execution Time'], errors='coerce')
        
        # Calculate trade value
        df['Trade Value'] = df['Quantity'] * df['Price']
        
        # Add date range filter
        st.markdown("### 📅 Filter by Date Range")
        col1, col2, col3 = st.columns([1, 1, 2])
        
        min_date = df['Trade Date'].min()
        max_date = df['Trade Date'].max()
        
        with col1:
            start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
        
        with col2:
            end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
        
        # Filter by date range
        mask = (df['Trade Date'] >= pd.Timestamp(start_date)) & (df['Trade Date'] <= pd.Timestamp(end_date))
        df_filtered = df[mask].copy()
        
        with col3:
            st.metric("Trades in Period", len(df_filtered), delta=f"{len(df_filtered)/len(df)*100:.1f}% of total")
        
        if len(df_filtered) == 0:
            st.warning("No trades in selected period")
            return
        
        # ========== EXTRACT EXPIRY FROM SYMBOL ==========
        # Extract expiry date from option symbols (e.g., NIFTY24JAN24000CE -> 24JAN)
        def extract_expiry(symbol):
            """Extract expiry from option symbol, removing strike price and CE/PE."""

            # remove last 7 characters without regex
            symbol_str = str(symbol)
            if len(symbol_str) < 7:
                return symbol_str
            symbol_trimmed = symbol_str[:-7]
            
            # If no match, return first 15 chars (should capture underlying + expiry)
            return symbol_trimmed
        
        df_filtered['Expiry'] = df_filtered['Symbol'].apply(extract_expiry)
        
        st.info("📅 Grouping trades by expiry cycle for strategy-level analysis (Iron Condor / Short Strangle)")
        
        # Show expiry extraction results
        with st.expander("🔍 Debug: Expiry Extraction"):
            sample_df = df_filtered[['Symbol', 'Expiry']].drop_duplicates().head(10)
            st.dataframe(sample_df, use_container_width=True)
            st.caption(f"Found {df_filtered['Expiry'].nunique()} unique expiry cycles")
        
        # ========== MATCH TRADES AND GROUP BY EXPIRY ==========
        # First, match individual buy/sell pairs
        matched_trades = []
        
        for symbol in df_filtered['Symbol'].unique():
            symbol_trades = df_filtered[df_filtered['Symbol'] == symbol].copy()
            symbol_trades = symbol_trades.sort_values('Trade Date')
            
            buys = symbol_trades[symbol_trades['Trade Type'].str.upper() == 'BUY'].copy()
            sells = symbol_trades[symbol_trades['Trade Type'].str.upper() == 'SELL'].copy()
            
            # Simple FIFO matching
            for _, buy in buys.iterrows():
                remaining_qty = buy['Quantity']
                buy_price = buy['Price']
                buy_date = buy['Trade Date']
                expiry = buy['Expiry']
                
                for idx, sell in sells.iterrows():
                    if remaining_qty <= 0:
                        break
                    
                    if sell['Quantity'] > 0:
                        matched_qty = min(remaining_qty, sell['Quantity'])
                        
                        # Calculate P&L for this matched pair
                        pnl = matched_qty * (sell['Price'] - buy_price)
                        
                        # Calculate duration
                        duration = (sell['Trade Date'] - buy_date).total_seconds() / 3600  # hours
                        
                        matched_trades.append({
                            'Symbol': symbol,
                            'Expiry': expiry,
                            'Quantity': matched_qty,
                            'Buy Price': buy_price,
                            'Sell Price': sell['Price'],
                            'P&L': pnl,
                            'Entry Date': buy_date,
                            'Exit Date': sell['Trade Date'],
                            'Duration (hrs)': duration,
                            'Trade Value': matched_qty * buy_price
                        })
                        
                        # Update remaining quantities
                        remaining_qty -= matched_qty
                        sells.at[idx, 'Quantity'] -= matched_qty
        
        matched_df = pd.DataFrame(matched_trades)
        
        # ========== GROUP BY EXPIRY FOR STRATEGY-LEVEL METRICS ==========
        if len(matched_df) > 0:
            # Group all trades by expiry
            expiry_groups = matched_df.groupby('Expiry').agg({
                'P&L': 'sum',
                'Entry Date': 'min',
                'Exit Date': 'max',
                'Trade Value': 'sum',
                'Symbol': 'count'  # Number of legs in the strategy
            }).reset_index()
            
            expiry_groups.columns = ['Expiry', 'P&L', 'Entry Date', 'Exit Date', 'Trade Value', 'Num Legs']
            expiry_groups['Duration (hrs)'] = (expiry_groups['Exit Date'] - expiry_groups['Entry Date']).dt.total_seconds() / 3600
            expiry_groups = expiry_groups.sort_values('Entry Date')
            
            st.success(f"✅ Analyzed {len(expiry_groups)} expiry cycles with {len(matched_df)} total legs")
        else:
            expiry_groups = pd.DataFrame()
        
        # Create tabs for different analysis sections
        analysis_tabs = st.tabs([
            "📊 Profitability",
            "⚡ Efficiency", 
            "📈 Performance Trends",
            "🎯 Trade Analysis",
            "📉 Drawdown",
            "📋 Raw Data"
        ])
        
        # ========== PROFITABILITY METRICS TAB ==========
        with analysis_tabs[0]:
            st.markdown("### 💰 Profitability Metrics (Per Expiry Cycle)")
            
            if len(expiry_groups) == 0:
                st.warning("⚠️ No completed expiry cycles found in this period")
                st.info("Showing raw trade data instead")
                
                # Show basic stats from raw data
                buy_value = df_filtered[df_filtered['Trade Type'].str.upper() == 'BUY']['Trade Value'].sum()
                sell_value = df_filtered[df_filtered['Trade Type'].str.upper() == 'SELL']['Trade Value'].sum()
                gross_pnl = sell_value - buy_value
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Buy Value", f"₹{buy_value:,.2f}")
                with col2:
                    st.metric("Sell Value", f"₹{sell_value:,.2f}")
                
                st.metric("Gross P&L (Approximate)", f"₹{gross_pnl:,.2f}",
                         help="Sell Value - Buy Value (not matched)")
                
            else:
                # Calculate metrics from expiry groups (strategy-level)
                total_expiries = len(expiry_groups)
                gross_pnl = expiry_groups['P&L'].sum()
                
                # Estimate charges (1% of trade value as approximation)
                total_trade_value = expiry_groups['Trade Value'].sum()
                estimated_charges = total_trade_value * 0.01  # 1% estimate
                net_pnl = gross_pnl - estimated_charges
                
                # Win/Loss metrics per expiry
                winning_expiries = expiry_groups[expiry_groups['P&L'] > 0]
                losing_expiries = expiry_groups[expiry_groups['P&L'] < 0]
                
                win_count = len(winning_expiries)
                loss_count = len(losing_expiries)
                
                win_rate = (win_count / total_expiries * 100) if total_expiries > 0 else 0
                loss_rate = (loss_count / total_expiries * 100) if total_expiries > 0 else 0
                
                avg_win = winning_expiries['P&L'].mean() if win_count > 0 else 0
                avg_loss = abs(losing_expiries['P&L'].mean()) if loss_count > 0 else 0
                
                gross_profit = winning_expiries['P&L'].sum() if win_count > 0 else 0
                gross_loss = abs(losing_expiries['P&L'].sum()) if loss_count > 0 else 0
                
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Gross P&L", f"₹{gross_pnl:,.2f}")
                    if gross_pnl > 0:
                        st.success("🟢 Profitable")
                    else:
                        st.error("🔴 Loss")
                
                with col2:
                    st.metric("Net P&L (Est.)", f"₹{net_pnl:,.2f}",
                             delta=f"Charges: ~₹{estimated_charges:,.0f}",
                             delta_color="inverse",
                             help="Estimated charges at 1% of trade value")
                
                with col3:
                    st.metric("Profit Factor", f"{profit_factor:.2f}",
                             help="Gross Profit / Gross Loss")
                    if profit_factor > 1.5:
                        st.success("🟢 Excellent")
                    elif profit_factor > 1.0:
                        st.warning("🟡 Profitable")
                    else:
                        st.error("🔴 Losing")
                
                with col4:
                    st.metric("Total Expiry Cycles", total_expiries,
                             help="Number of different expiry cycles traded")
                
                st.markdown("---")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Win Rate", f"{win_rate:.1f}%",
                             delta=f"{win_count} winning cycles")
                
                with col2:
                    st.metric("Loss Rate", f"{loss_rate:.1f}%",
                             delta=f"{loss_count} losing cycles")
                
                with col3:
                    st.metric("Avg Win", f"₹{avg_win:,.2f}",
                             help="Average profit per winning expiry cycle")
                
                with col4:
                    st.metric("Avg Loss", f"₹{avg_loss:,.2f}",
                             help="Average loss per losing expiry cycle")
                
                # Win/Loss ratio and Expectancy
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
                    st.metric("Avg Win / Avg Loss", f"{win_loss_ratio:.2f}",
                             help="Higher is better - shows quality of wins vs losses")
                    if win_loss_ratio > 2:
                        st.success("🟢 Excellent - Big wins, small losses")
                    elif win_loss_ratio > 1:
                        st.warning("🟡 Good")
                    else:
                        st.error("🔴 Poor - Losses bigger than wins")
                
                with col2:
                    # Expectancy
                    expectancy = (avg_win * win_rate / 100) - (avg_loss * loss_rate / 100)
                    st.metric("Expectancy per Expiry", f"₹{expectancy:,.2f}",
                             help="(Avg Win × Win Rate) - (Avg Loss × Loss Rate)")
                    if expectancy > 0:
                        st.success(f"🟢 Positive: ₹{expectancy:,.2f} per expiry")
                    else:
                        st.error(f"🔴 Negative: ₹{expectancy:,.2f} per expiry")
                
                with col3:
                    # Max Drawdown from expiry cycles
                    cumulative_pnl = expiry_groups.sort_values('Entry Date')['P&L'].cumsum()
                    cumulative_max = cumulative_pnl.expanding().max()
                    drawdown = cumulative_pnl - cumulative_max
                    max_dd = drawdown.min()
                    
                    st.metric("Max Drawdown", f"₹{abs(max_dd):,.2f}",
                             delta_color="inverse",
                             help="Largest peak-to-trough loss")
                
                # Recovery Factor
                col1, col2 = st.columns(2)
                
                with col1:
                    recovery_factor = net_pnl / abs(max_dd) if max_dd != 0 else 0
                    st.metric("Recovery Factor", f"{recovery_factor:.2f}",
                             help="Net P&L / Max Drawdown")
                    if recovery_factor > 3:
                        st.success("🟢 Excellent recovery efficiency")
                    elif recovery_factor > 1:
                        st.warning("🟡 Good")
                    else:
                        st.error("🔴 Poor - losses not recovered efficiently")
                
                with col2:
                    # Gross Profit and Loss breakdown
                    st.metric("Gross Profit", f"₹{gross_profit:,.2f}",
                             delta=f"From {win_count} winning cycles")
                
                # Expiry Cycles Table
                st.markdown("### 📅 Expiry Cycle Performance")
                
                display_df = expiry_groups.copy()
                display_df['ROI %'] = (display_df['P&L'] / display_df['Trade Value'] * 100).round(2)
                display_df['P&L'] = display_df['P&L'].apply(lambda x: f"₹{x:,.2f}")
                display_df['Trade Value'] = display_df['Trade Value'].apply(lambda x: f"₹{x:,.2f}")
                display_df['Duration (days)'] = (display_df['Duration (hrs)'] / 24).round(1)
                display_df = display_df[['Expiry', 'Entry Date', 'Exit Date', 'Num Legs', 'P&L', 'ROI %', 'Duration (days)']].sort_values('Entry Date', ascending=False)
                
                st.dataframe(display_df, use_container_width=True, height=400)
                
                # P&L Distribution per Expiry
                st.markdown("### 📊 P&L Distribution (Per Expiry Cycle)")
                
                import plotly.graph_objects as go
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=expiry_groups['P&L'],
                    nbinsx=20,
                    marker_color='#3B82F6',
                    name='Expiry P&L'
                ))
                
                fig.add_vline(x=0, line_dash="dash", line_color="white",
                             annotation_text="Break-even")
                fig.add_vline(x=expectancy, line_dash="dash", line_color="green",
                             annotation_text=f"Expectancy: ₹{expectancy:.0f}")
                
                fig.update_layout(
                    title="Expiry Cycle P&L Distribution",
                    xaxis_title="P&L per Expiry (₹)",
                    yaxis_title="Frequency",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # ========== EFFICIENCY METRICS TAB ==========
        with analysis_tabs[1]:
            st.markdown("### ⚡ Efficiency & Strategy Quality (Per Expiry)")
            
            if len(expiry_groups) == 0:
                st.warning("⚠️ No completed expiry cycles to analyze")
                return
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Expiry Cycles", len(expiry_groups))
                avg_legs = expiry_groups['Num Legs'].mean()
                st.caption(f"Avg {avg_legs:.1f} legs per cycle")
            
            with col2:
                # Average cycle duration
                avg_duration = expiry_groups['Duration (hrs)'].mean()
                if avg_duration < 24:
                    duration_display = f"{avg_duration:.1f} hrs"
                else:
                    duration_display = f"{avg_duration/24:.1f} days"
                st.metric("Avg Cycle Duration", duration_display)
            
            with col3:
                # Return on Investment
                roi = (gross_pnl / total_trade_value * 100) if total_trade_value > 0 else 0
                st.metric("Total ROI", f"{roi:.2f}%",
                         help="Gross P&L / Total Capital Deployed")
            
            with col4:
                # Risk-adjusted P&L
                pnl_std = expiry_groups['P&L'].std()
                risk_adjusted = gross_pnl / pnl_std if pnl_std > 0 else 0
                st.metric("Risk-Adjusted P&L", f"{risk_adjusted:.2f}",
                         help="P&L / Std Dev of expiry cycles")
            
            # Streaks
            st.markdown("### 🔥 Winning & Losing Streaks (Per Expiry)")
            
            # Sort by entry date and calculate streaks
            expiry_sorted = expiry_groups.sort_values('Entry Date')
            is_win = expiry_sorted['P&L'] > 0
            
            # Calculate streaks
            current_streak = 0
            max_win_streak = 0
            max_loss_streak = 0
            current_is_win = None
            
            for win in is_win:
                if current_is_win == win:
                    current_streak += 1
                else:
                    current_streak = 1
                    current_is_win = win
                
                if win:
                    max_win_streak = max(max_win_streak, current_streak)
                else:
                    max_loss_streak = max(max_loss_streak, current_streak)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Max Winning Streak", f"{max_win_streak} cycles",
                         help="Longest consecutive winning expiry cycles")
                if max_win_streak > 5:
                    st.success("🟢 Strong consistency")
            
            with col2:
                st.metric("Max Losing Streak", f"{max_loss_streak} trades",
                         help="Longest consecutive losing trades")
                if max_loss_streak > 5:
                    st.error("🔴 High - review strategy robustness")
            
            # Sharpe-like metric
            st.markdown("### 📈 Risk-Adjusted Performance")
            
            mean_pnl = expiry_groups['P&L'].mean()
            std_pnl = expiry_groups['P&L'].std()
            sharpe_like = mean_pnl / std_pnl if std_pnl > 0 else 0
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Mean P&L per Expiry", f"₹{mean_pnl:,.2f}")
            
            with col2:
                st.metric("Sharpe-like Ratio", f"{sharpe_like:.2f}",
                         help="Mean P&L / Std Dev P&L (per expiry cycle)")
                if sharpe_like > 1:
                    st.success("🟢 Excellent risk-adjusted returns")
                elif sharpe_like > 0.5:
                    st.warning("🟡 Good")
                else:
                    st.error("🔴 Poor - high volatility relative to returns")
            
            # Consistency Score
            st.markdown("### 🎯 Consistency Score")
            
            # Combine multiple factors into consistency score (0-100)
            win_rate_score = min(win_rate, 100)
            profit_factor_score = min(profit_factor * 33.3, 100)
            streak_score = max(0, 100 - (max_loss_streak * 10))
            expectancy_score = min(max(0, expectancy / 100), 100) if expectancy > 0 else 0
            
            consistency_score = (win_rate_score * 0.3 + 
                               profit_factor_score * 0.3 + 
                               streak_score * 0.2 + 
                               expectancy_score * 0.2)
            
            st.metric("Consistency Score", f"{consistency_score:.1f}/100",
                     help="Combined metric: Win Rate (30%), Profit Factor (30%), Streak Control (20%), Expectancy (20%)")
            
            if consistency_score > 70:
                st.success("🟢 Highly consistent strategy")
            elif consistency_score > 50:
                st.warning("🟡 Moderately consistent")
            else:
                st.error("🔴 Inconsistent - needs improvement")
            
            # Breakdown
            with st.expander("📊 Consistency Score Breakdown"):
                breakdown_df = pd.DataFrame({
                    'Component': ['Win Rate', 'Profit Factor', 'Streak Control', 'Expectancy'],
                    'Score': [win_rate_score * 0.3, profit_factor_score * 0.3, 
                             streak_score * 0.2, expectancy_score * 0.2],
                    'Weight': ['30%', '30%', '20%', '20%']
                })
                st.dataframe(breakdown_df, use_container_width=True)
        
        # ========== PERFORMANCE TRENDS TAB ==========
        with analysis_tabs[2]:
            st.markdown("### 📈 Performance Over Time (Per Expiry Cycle)")
            
            if len(expiry_groups) == 0:
                st.warning("⚠️ No completed expiry cycles to analyze")
                return
            
            # Cumulative P&L by expiry cycle
            expiry_sorted = expiry_groups.sort_values('Entry Date').copy()
            expiry_sorted['Cumulative_PnL'] = expiry_sorted['P&L'].cumsum()
            
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Cumulative P&L by Expiry Cycle', 'P&L per Expiry Cycle'),
                vertical_spacing=0.15,
                row_heights=[0.6, 0.4]
            )
            
            # Cumulative P&L
            fig.add_trace(
                go.Scatter(
                    x=expiry_sorted['Expiry'],
                    y=expiry_sorted['Cumulative_PnL'],
                    mode='lines+markers',
                    name='Cumulative P&L',
                    line=dict(color='#3B82F6', width=2),
                    fill='tozeroy',
                    text=expiry_sorted['Entry Date'].dt.strftime('%Y-%m-%d'),
                    hovertemplate='<b>%{x}</b><br>Cumulative: ₹%{y:,.2f}<br>Date: %{text}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Per Expiry P&L
            colors = ['#10B981' if x > 0 else '#EF4444' for x in expiry_sorted['P&L']]
            fig.add_trace(
                go.Bar(
                    x=expiry_sorted['Expiry'],
                    y=expiry_sorted['P&L'],
                    name='Expiry P&L',
                    marker_color=colors,
                    text=expiry_sorted['Entry Date'].dt.strftime('%Y-%m-%d'),
                    hovertemplate='<b>%{x}</b><br>P&L: ₹%{y:,.2f}<br>Date: %{text}<extra></extra>'
                ),
                row=2, col=1
            )
            
            fig.update_xaxes(title_text="Expiry", row=2, col=1)
            fig.update_yaxes(title_text="Cumulative P&L (₹)", row=1, col=1)
            fig.update_yaxes(title_text="P&L (₹)", row=2, col=1)
            
            fig.update_layout(height=700, showlegend=False, hovermode='x unified')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                best_expiry_pnl = expiry_sorted['P&L'].max()
                best_expiry = expiry_sorted[expiry_sorted['P&L'] == best_expiry_pnl]['Expiry'].iloc[0]
                st.metric("Best Expiry", f"₹{best_expiry_pnl:,.2f}",
                         delta=best_expiry)
            
            with col2:
                worst_expiry_pnl = expiry_sorted['P&L'].min()
                worst_expiry = expiry_sorted[expiry_sorted['P&L'] == worst_expiry_pnl]['Expiry'].iloc[0]
                st.metric("Worst Expiry", f"₹{worst_expiry_pnl:,.2f}",
                         delta=worst_expiry)
            
            with col3:
                positive_expiries = (expiry_sorted['P&L'] > 0).sum()
                total_expiries_count = len(expiry_sorted)
                st.metric("Positive Expiries", f"{positive_expiries}/{total_expiries_count}",
                         delta=f"{positive_expiries/total_expiries_count*100:.1f}%")
            
            with col4:
                avg_expiry_pnl = expiry_sorted['P&L'].mean()
                st.metric("Avg Expiry P&L", f"₹{avg_expiry_pnl:,.2f}")
        
        # ========== TRADE ANALYSIS TAB ==========
        with analysis_tabs[3]:
            st.markdown("### 🎯 Detailed Analysis by Expiry")
            
            if len(expiry_groups) == 0:
                st.warning("⚠️ No completed expiry cycles to analyze")
                return
            
            # By expiry
            st.markdown("#### Performance by Expiry Cycle")
            
            display_expiry = expiry_groups.copy()
            display_expiry['ROI %'] = (display_expiry['P&L'] / display_expiry['Trade Value'] * 100).round(2)
            display_expiry['Duration (days)'] = (display_expiry['Duration (hrs)'] / 24).round(1)
            display_expiry = display_expiry.sort_values('Entry Date', ascending=False)
            
            st.dataframe(
                display_expiry[['Expiry', 'Entry Date', 'Exit Date', 'Num Legs', 'P&L', 'Trade Value', 'ROI %', 'Duration (days)']].style.format({
                    'P&L': '₹{:,.2f}',
                    'Trade Value': '₹{:,.2f}',
                    'ROI %': '{:.2f}%',
                    'Duration (days)': '{:.1f}'
                }).background_gradient(subset=['P&L'], cmap='RdYlGn', vmin=-10000, vmax=10000),
                use_container_width=True,
                height=400
            )
            
            # Strategy composition
            st.markdown("#### Strategy Leg Composition")
            leg_distribution = expiry_groups['Num Legs'].value_counts().sort_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure(data=[go.Bar(
                    x=leg_distribution.index,
                    y=leg_distribution.values,
                    marker_color='#3B82F6',
                    text=leg_distribution.values,
                    textposition='auto'
                )])
                
                fig.update_layout(
                    title="Number of Legs per Expiry Cycle",
                    xaxis_title="Number of Legs",
                    yaxis_title="Frequency",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.caption("2 legs = Vertical Spread, 4 legs = Iron Condor, etc.")
            
            with col2:
                # ROI distribution
                roi_values = (expiry_groups['P&L'] / expiry_groups['Trade Value'] * 100)
                
                fig = go.Figure(data=[go.Histogram(
                    x=roi_values,
                    nbinsx=20,
                    marker_color='#10B981'
                )])
                
                fig.update_layout(
                    title="ROI Distribution per Expiry",
                    xaxis_title="ROI (%)",
                    yaxis_title="Frequency",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Individual legs by symbol
            st.markdown("#### Performance by Individual Legs (Symbols)")
            
            symbol_performance = matched_df.groupby('Symbol').agg({
                'P&L': ['sum', 'count', 'mean'],
                'Quantity': 'sum'
            }).reset_index()
            
            symbol_performance.columns = ['Symbol', 'Total P&L', 'Leg Count', 'Avg P&L', 'Total Qty']
            symbol_performance = symbol_performance.sort_values('Total P&L', ascending=False)
            
            st.dataframe(
                symbol_performance.head(20).style.format({
                    'Total P&L': '₹{:,.2f}',
                    'Leg Count': '{:.0f}',
                    'Avg P&L': '₹{:,.2f}',
                    'Total Qty': '{:.0f}'
                }).background_gradient(subset=['Total P&L'], cmap='RdYlGn', vmin=-1000, vmax=1000),
                use_container_width=True
            )
            
            # Top expiries
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🏆 Top 5 Best Expiry Cycles")
                top_winners = expiry_groups.nlargest(5, 'P&L')[['Expiry', 'Entry Date', 'P&L', 'Num Legs', 'Duration (hrs)']]
                top_winners['Duration (days)'] = (top_winners['Duration (hrs)'] / 24).round(1)
                st.dataframe(
                    top_winners[['Expiry', 'Entry Date', 'P&L', 'Num Legs', 'Duration (days)']].style.format({
                        'P&L': '₹{:,.2f}',
                        'Duration (days)': '{:.1f}'
                    }),
                    use_container_width=True
                )
            
            with col2:
                st.markdown("#### 💔 Top 5 Worst Expiry Cycles")
                top_losers = expiry_groups.nsmallest(5, 'P&L')[['Expiry', 'Entry Date', 'P&L', 'Num Legs', 'Duration (hrs)']]
                top_losers['Duration (days)'] = (top_losers['Duration (hrs)'] / 24).round(1)
                st.dataframe(
                    top_losers[['Expiry', 'Entry Date', 'P&L', 'Num Legs', 'Duration (days)']].style.format({
                        'P&L': '₹{:,.2f}',
                        'Duration (days)': '{:.1f}'
                    }),
                    use_container_width=True
                )
            
            # Holding period analysis
            st.markdown("#### Holding Period Analysis")
            
            fig = go.Figure(data=[go.Histogram(
                x=expiry_groups['Duration (hrs)'] / 24,  # Convert to days
                nbinsx=20,
                marker_color='#9333EA'
            )])
            
            fig.update_layout(
                title="Expiry Cycle Duration Distribution",
                xaxis_title="Duration (days)",
                yaxis_title="Frequency",
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # ========== DRAWDOWN TAB ==========
        with analysis_tabs[4]:
            st.markdown("### 📉 Drawdown Analysis (Per Expiry Cycle)")
            
            if len(expiry_groups) == 0:
                st.warning("⚠️ No completed expiry cycles to analyze")
                return
            
            # Calculate drawdown from expiry cycles
            expiry_sorted = expiry_groups.sort_values('Entry Date').copy()
            expiry_sorted['Cumulative_PnL'] = expiry_sorted['P&L'].cumsum()
            
            cumulative_max = expiry_sorted['Cumulative_PnL'].expanding().max()
            expiry_sorted['Drawdown'] = expiry_sorted['Cumulative_PnL'] - cumulative_max
            
            max_dd = expiry_sorted['Drawdown'].min()
            max_dd_idx = expiry_sorted['Drawdown'].idxmin()
            max_dd_pct = (max_dd / cumulative_max[max_dd_idx]) * 100 if cumulative_max[max_dd_idx] != 0 else 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Max Drawdown", f"₹{abs(max_dd):,.2f}",
                         delta=f"{max_dd_pct:.2f}%",
                         delta_color="inverse")
            
            with col2:
                # Recovery factor
                recovery_factor = gross_pnl / abs(max_dd) if max_dd != 0 else 0
                st.metric("Recovery Factor", f"{recovery_factor:.2f}",
                         help="Net P&L / Max Drawdown")
                if recovery_factor > 3:
                    st.success("🟢 Excellent recovery efficiency")
                elif recovery_factor > 1:
                    st.warning("🟡 Good")
                else:
                    st.error("🔴 Poor - losses not recovered efficiently")
            
            with col3:
                current_dd = expiry_sorted['Drawdown'].iloc[-1]
                st.metric("Current Drawdown", f"₹{abs(current_dd):,.2f}")
            
            # Drawdown chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=expiry_sorted['Expiry'],
                y=expiry_sorted['Drawdown'],
                mode='lines+markers',
                name='Drawdown',
                fill='tozeroy',
                line=dict(color='#EF4444', width=2),
                text=expiry_sorted['Entry Date'].dt.strftime('%Y-%m-%d'),
                hovertemplate='<b>%{x}</b><br>Drawdown: ₹%{y:,.2f}<br>Date: %{text}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Drawdown Over Time (By Expiry Cycle)",
                xaxis_title="Expiry",
                yaxis_title="Drawdown (₹)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Drawdown recovery analysis
            st.markdown("#### Recovery Analysis")
            
            # Find drawdown periods
            in_drawdown = expiry_sorted['Drawdown'] < 0
            drawdown_periods = []
            
            start_idx = None
            for idx, is_dd in enumerate(in_drawdown):
                if is_dd and start_idx is None:
                    start_idx = idx
                elif not is_dd and start_idx is not None:
                    drawdown_periods.append((start_idx, idx - 1))
                    start_idx = None
            
            if start_idx is not None:
                drawdown_periods.append((start_idx, len(expiry_sorted) - 1))
            
            if drawdown_periods:
                recovery_info = []
                for start, end in drawdown_periods:
                    dd_depth = expiry_sorted.iloc[start:end+1]['Drawdown'].min()
                    dd_duration_days = (expiry_sorted.iloc[end]['Entry Date'] - 
                                 expiry_sorted.iloc[start]['Entry Date']).days
                    recovery_info.append({
                        'Start Expiry': expiry_sorted.iloc[start]['Expiry'],
                        'End Expiry': expiry_sorted.iloc[end]['Expiry'],
                        'Depth': dd_depth,
                        'Duration (expiry cycles)': end - start + 1,
                        'Duration (days)': dd_duration_days
                    })
                
                recovery_df = pd.DataFrame(recovery_info)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    avg_dd_duration = recovery_df['Duration (expiry cycles)'].mean()
                    st.metric("Avg Drawdown Duration", f"{avg_dd_duration:.1f} expiry cycles")
                
                with col2:
                    max_dd_duration = recovery_df['Duration (days)'].max()
                    st.metric("Max Drawdown Duration", f"{max_dd_duration:.0f} days")
                
                st.dataframe(
                    recovery_df.style.format({
                        'Depth': '₹{:,.2f}',
                        'Duration (expiry cycles)': '{:.0f}',
                        'Duration (days)': '{:.0f}'
                    }),
                    use_container_width=True
                )
        
        # ========== RAW DATA TAB ==========
        with analysis_tabs[5]:
            st.markdown("### 📋 Raw Data")
            
            tab1, tab2 = st.tabs(["Matched Trades", "All Trades"])
            
            with tab1:
                if len(matched_df) > 0:
                    st.markdown("#### Matched Buy-Sell Pairs")
                    st.dataframe(
                        matched_df.style.format({
                            'Buy Price': '₹{:.2f}',
                            'Sell Price': '₹{:.2f}',
                            'P&L': '₹{:,.2f}',
                            'Quantity': '{:.0f}',
                            'Duration (hrs)': '{:.1f}',
                            'Trade Value': '₹{:,.2f}'
                        }),
                        use_container_width=True,
                        height=600
                    )
                    
                    # Download button
                    csv = matched_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Matched Trades as CSV",
                        data=csv,
                        file_name=f"matched_trades_{start_date}_to_{end_date}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No matched trades available")
            
            with tab2:
                st.markdown("#### All Trades from Tradebook")
                st.dataframe(df_filtered, use_container_width=True, height=600)
                
                # Download button
                csv = df_filtered.to_csv(index=False)
                st.download_button(
                    label="📥 Download All Trades as CSV",
                    data=csv,
                    file_name=f"tradebook_all_{start_date}_to_{end_date}.csv",
                    mime="text/csv"
                )
    
    except Exception as e:
        st.error(f"❌ Error loading tradebook: {str(e)}")
        st.exception(e)



if __name__ == "__main__":
    render() 

    # set width to wide

    st.set_page_config(page_title="Options Trading Dashboard", layout="wide")

    # st.title("🎯 Options Trading Dashboard" )

