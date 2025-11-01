"""Data loading utilities for options and NIFTY data."""

import os
import sys
import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Tuple

# Add root to path for imports
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase with underscores."""
    df.columns = df.columns.str.strip().str.lower().str.replace(r"[\s\.\*\*]+", "_", regex=True)
    return df


@st.cache_data(ttl=300)
def load_all_data(dirs=None) -> Tuple[pd.DataFrame, List]:
    """Load all excel/csv files from the given directories.

    If dirs is None, we try `database/data/` and fall back to `database/options_data/`.
    
    Returns:
        Tuple of (combined DataFrame, list of errors)
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


def load_nifty_daily(_kite_client: Optional[object] = None, dirs=None, days: int = 365) -> pd.DataFrame:
    """Load NIFTY daily data from Kite historical API.

    Args:
        _kite_client: KiteConnect client instance (prefixed with _ to skip hashing)
        dirs: Not used (kept for compatibility)
        days: Number of days of historical data to fetch

    Returns:
        DataFrame with NIFTY daily OHLC data

    Raises:
        RuntimeError: If Kite client is not provided or data fetch fails
    """
    if _kite_client is None:
        raise RuntimeError(
            "Kite client is required to load NIFTY daily data. "
            "Local files do not contain NIFTY daily prices (they only have OI)."
        )

    try:
        instruments = _kite_client.instruments()
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
        raw = _kite_client.historical_data(
            token, 
            from_date.strftime("%Y-%m-%d"), 
            to_date.strftime("%Y-%m-%d"), 
            "day"
        )
    except Exception as e:
        raise RuntimeError(f"Kite historical_data call failed: {e}")

    kite_df = pd.DataFrame(raw)

    # Print columns for debugging
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
