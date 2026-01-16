"""Symbol parsing utilities for options trading symbols."""

import re
import calendar
from datetime import datetime, timedelta
from typing import Optional, Dict


def parse_tradingsymbol(symbol: str) -> Optional[Dict]:
    """Parse NIFTY option trading symbol.
    
    Supports both weekly and monthly options formats:
    - Weekly: NIFTY + 2-digit-year + month (1-9 or O/N/D) + day (2 digits) + strike + CE/PE
      Example: NIFTY2591125100PE, NIFTY25N1125100PE
    - Monthly: NIFTY + 2-digit-year + 3-letter-month + strike + CE/PE
      Example: NIFTY25NOV25000CE
    
    Returns:
        Dict with keys: strike, expiry, option_type
        None if parsing fails
    """
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

        # Monthly expiry: last Tuesday of the month
        last_day = datetime(year, month, calendar.monthrange(year, month)[1])
        while last_day.weekday() != 1:
            last_day -= timedelta(days=1)
        if last_day.month == 3 and last_day.day == 31:
            last_day -= timedelta(days=1)
        expiry = last_day

        return {
            "strike": float(strike_str),
            "expiry": expiry,
            "option_type": opt_type
        }

    return None
