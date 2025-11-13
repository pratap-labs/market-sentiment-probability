"""Formatting utilities and constants."""

import os


def format_inr(value, decimals: int = 0, symbol: str = "₹") -> str:
    """Format number using Indian-style comma grouping.
    
    Args:
        value: Number to format
        decimals: Number of decimal places
        symbol: Currency symbol (default: ₹)
    
    Returns:
        Formatted string with Indian number formatting
    
    Examples:
        >>> format_inr(1000000)
        '₹10,00,000'
        >>> format_inr(12345.67, decimals=2)
        '₹12,345.67'
        >>> format_inr(-50000)
        '₹-50,000'
    """
    try:
        if value is None:
            return f"{symbol}0"
        neg = float(value) < 0
        val = abs(float(value))

        # Round according to decimals
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
DEFAULT_LOT_SIZE = int(os.getenv("OPTION_CONTRACT_SIZE", "75"))
