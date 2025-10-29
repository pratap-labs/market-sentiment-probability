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

def parse_tradingsymbol(symbol: str) -> Optional[Dict]:
    """
    Parse Kite tradingsymbol like 'NIFTY24NOV25000CE' or 'NIFTY21JUN15400PE'
    Returns: {strike: float, expiry: datetime, option_type: 'CE'/'PE'}
    """
    s = (symbol or "").upper().strip()

    # Weekly pattern: NIFTY + 2-digit-year + single-letter-month-code + DD + strike + CE/PE
    # Example: NIFTY25N0426400CE -> year=25, month_code=N, day=04, strike=26400
    weekly_pattern = r"^NIFTY(\d{2})([A-Z])(\d{2})(\d+)(CE|PE)$"
    m_week = re.match(weekly_pattern, s)
    if m_week:
        year_2, month_code, day_str, strike_str, opt_type = m_week.groups()
        year = 2000 + int(year_2)

        # Standard single-letter month codes used in futures/options (Futures month codes):
        # F=Jan, G=Feb, H=Mar, J=Apr, K=May, M=Jun, N=Jul, Q=Aug, U=Sep, V=Oct, X=Nov, Z=Dec
        month_code_map = {
            "J": 1, "F": 2, "M": 3, "J": 4, "X": 5, "M": 6,
            "N": 7, "A": 8, "S": 9, "O": 10, "N": 11, "D": 12
        }

        month = month_code_map.get(month_code)
        if month is None:
            return None

        try:
            day = int(day_str)
            expiry = datetime(year, month, day)
        except Exception:
            # invalid date (e.g., day out of range)
            return None

        return {
            "strike": float(strike_str),
            "expiry": expiry,
            "option_type": opt_type
        }

    # Monthly pattern: NIFTY + 2-digit-year + 3-letter-month + strike + CE/PE
    monthly_pattern = r"^NIFTY(\d{2})([A-Z]{3})(\d+)(CE|PE)$"
    m_mon = re.match(monthly_pattern, s)
    if not m_mon:
        return None

    year_2, month_str, strike_str, opt_type = m_mon.groups()
    year = 2000 + int(year_2)

    month_map = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
    }
    month = month_map.get(month_str.upper())
    if not month:
        return None

    import calendar
    last_day = calendar.monthrange(year, month)[1]
    # Use last calendar day of the month for monthly expiries (user preference)
    expiry = datetime(year, month, last_day)

    return {
        "strike": float(strike_str),
        "expiry": expiry,
        "option_type": opt_type
    }


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
    """Calculate market regime indicators."""
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
            # log skipped days for debugging — this indicates missing underlying spot in options data
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

    # If we skipped historical days, surface a short warning to the UI and print a summary
    if skipped_days:
        print(f"DEBUG: skipped {len(skipped_days)} historical days due to missing underlying_value. Examples: {skipped_days[:5]}")
        try:
            st.warning(f"Skipped {len(skipped_days)} historical days with missing 'underlying_value' when calculating IV history — check options data completeness.")
        except Exception:
            pass
    
    # Calculate realized volatility from NIFTY daily
    if "close" in nifty_df.columns and len(nifty_df) > 30:
        returns = nifty_df["close"].pct_change().dropna()[-30:]
        realized_vol = returns.std() * np.sqrt(252)
    else:
        realized_vol = 0.15
    
    # VRP = IV - RV
    vrp = current_iv - realized_vol
    
    return {
        "current_iv": current_iv,
        "iv_rank": iv_rank,
        "realized_vol": realized_vol,
        "vrp": vrp,
        "current_spot": current_spot
    }


# ============================================================================
# STREAMLIT UI
# ============================================================================

def render():
    """Main entrypoint for the Streamlit view."""
    st.title("🎯 Options Dashboard")

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
        "🚨 Risk Alerts"
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
    """Render portfolio overview tab."""
    st.subheader("Portfolio Overview")
    
    if "enriched_positions" not in st.session_state:
        st.info("No positions loaded. Fetch positions from the Positions tab first.")
        return
    
    enriched = st.session_state["enriched_positions"]
    portfolio_greeks = calculate_portfolio_greeks(enriched)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta = portfolio_greeks["net_delta"]
        # Streamlit in this environment doesn't accept `delta_on_color`.
        # Show the signed delta as the `delta` parameter instead.
        st.metric("Net Delta", f"{delta:.2f}", f"{delta:+.2f}")
    
    with col2:
        gamma = portfolio_greeks["net_gamma"]
        st.metric("Net Gamma", f"{gamma:.3f}")
    
    with col3:
        vega = portfolio_greeks["net_vega"]
        st.metric("Net Vega", f"{vega:.0f}")
    
    with col4:
        theta = portfolio_greeks["net_theta"]
        st.metric("Net Theta", f"{theta:.0f}")
    
    # Greeks visualization
    st.subheader("Greeks Breakdown")
    greeks_df = pd.DataFrame({
        "Metric": ["Delta", "Gamma", "Vega", "Theta"],
        "Value": [
            portfolio_greeks["net_delta"],
            portfolio_greeks["net_gamma"],
            portfolio_greeks["net_vega"],
            portfolio_greeks["net_theta"]
        ]
    })
    
    st.bar_chart(greeks_df.set_index("Metric"))
    
    # Risk zones
    st.subheader("Risk Status")
    
    # Delta zone
    delta_abs = abs(portfolio_greeks["net_delta"])
    if delta_abs < 8:
        st.success("🟢 Delta: GREEN - Portfolio is well balanced")
    elif delta_abs < 15:
        st.warning("🟡 Delta: YELLOW - Monitor for directional risk")
    else:
        st.error("🔴 Delta: RED - High directional exposure, consider adjustment")


def render_diagnostics_tab():
    """Render position diagnostics tab."""
    st.subheader("Position Diagnostics")
    
    if "enriched_positions" not in st.session_state:
        st.info("No positions loaded. Fetch positions from the Positions tab first.")
        return
    
    enriched = st.session_state["enriched_positions"]
    
    # Calculate diagnostics for each position
    diagnostics = []
    for pos in enriched:
        theta = pos.get("position_theta", 0)
        pnl = pos.get("pnl", 0)
        dte = pos.get("dte", 0)
        
        # Days to breakeven
        days_to_breakeven = abs(pnl / theta) if theta != 0 else 999
        
        # Theta efficiency
        theta_eff = (pnl / theta * 100) if theta != 0 else 0
        
        diagnostics.append({
            "Symbol": pos.get("tradingsymbol", ""),
            "PnL": pnl,
            "Theta/Day": theta,
            "DTE": dte,
            "Days to B/E": days_to_breakeven,
            "Theta Efficiency %": theta_eff,
            "Status": "🔴 RED" if days_to_breakeven > dte else ("🟡 YELLOW" if days_to_breakeven > dte * 0.5 else "🟢 GREEN")
        })
    
    df = pd.DataFrame(diagnostics)
    st.dataframe(
        df.style.format({
            "PnL": "{:.2f}",
            "Theta/Day": "{:.2f}",
            "Days to B/E": "{:.1f}",
            "Theta Efficiency %": "{:.1f}%"
        }),
        use_container_width=True
    )


def render_market_regime_tab(options_df: pd.DataFrame, nifty_df: pd.DataFrame):
    """Render market regime tab."""
    st.subheader("Market Regime Analysis")
    
    regime = calculate_market_regime(options_df, nifty_df)
    
    if not regime:
        st.warning("Insufficient data to calculate market regime.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current IV", f"{regime.get('current_iv', 0):.2%}")
    
    with col2:
        iv_rank = regime.get('iv_rank', 50)
        st.metric("IV Rank", f"{iv_rank:.1f}%")
        st.metric("IV Percentile", f"{regime.get('iv_percentile', 0):.1f}%")
        st.metric("Market Regime", regime.get("market_regime", "Unknown"))
    
    with col3:
        st.metric("Realized Vol", f"{regime.get('realized_vol', 0):.2%}")
    with col4:
        st.metric("Volatility Risk Premium (VRP)", f"{regime.get('vrp', 0):.2%}")


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


if __name__ == "__main__":
    render()
    # st.title("🎯 Options Trading Dashboard" )
