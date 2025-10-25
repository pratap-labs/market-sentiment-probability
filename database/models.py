import os
from datetime import datetime, date
from typing import Dict, Any
from supabase import create_client, Client

# Load credentials
SUPABASE_URL = 'https://ehojxadxaqolqqshlzcd.supabase.co'
SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVob2p4YWR4YXFvbHFxc2hsemNkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjExMzIzMDYsImV4cCI6MjA3NjcwODMwNn0.nWeFny73-GybqDWqKVe96j4Ojph-Z8maGfw1sYDtC1M'


if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("❌ SUPABASE_URL or SUPABASE_KEY not set in environment")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------- Utility Helpers ---------- #
def upsert(table: str, record: Dict[str, Any]):
    """Generic upsert (insert or update).

    Args:
        table: table name
        record: dict payload or list of records
    """
    # Add updated_at timestamp instead of a generic 'timestamp' column
    # Some tables (like options_data) don't have a 'timestamp' column in the schema.
    if isinstance(record, dict):
        if 'updated_at' not in record:
            record['updated_at'] = datetime.utcnow().isoformat()
    elif isinstance(record, list):
        for r in record:
            if 'updated_at' not in r:
                r['updated_at'] = datetime.utcnow().isoformat()


    res = supabase.table(table).upsert(record).execute()
    
    # Check for errors properly
    if hasattr(res, "error") and res.error:
        print(f"❌ Error upserting into {table}: {res.error}")
    # elif res.status_code >= 400:
        # print(f"❌ HTTP Error {res.status_code} when upserting into {table}: {res.data}")
    else:
        print(f"✅ Upsert successful into {table}")
    
    return res


def select_all(table: str, limit: int = 100):
    """Fetch rows from a table"""
    res = supabase.table(table).select("*").limit(limit).execute()
    return res.data


def get_table_stats():
    """Count rows in each table"""
    tables = ["futures_data", "options_data", "fii_futures_data", "fii_options_data"]
    stats = {}
    for t in tables:
        res = supabase.table(t).select("id").execute()
        stats[t] = len(res.data)
    return stats


# ---------- Table-specific wrappers ---------- #

def insert_futures_data(data: Dict[str, Any]):
    return upsert("futures_data", data)


def insert_options_data(data: Dict[str, Any]):
    """
    Normalize incoming option record keys and upsert into `options_data` table.

    Accepts either raw NSE keys (FH_*) or the generic keys produced by the fetcher
    (e.g., 'date', 'expiry_date', 'strike_price', 'option_type', 'open_interest', ...)
    and maps them to the database column names prefixed with `fh_`.
    """
    def _as_date_str(v):
        if v is None:
            return None
        if isinstance(v, str):
            return v
        try:
            return v.strftime('%Y-%m-%d')
        except Exception:
            return str(v)

    def _as_ts(v):
        if v is None:
            return None
        if isinstance(v, str):
            return v
        try:
            return v.isoformat()
        except Exception:
            return str(v)

    mapped = {
        'fh_instrument': 'OPTIDX',
        'fh_symbol': data.get('fh_symbol') or data.get('symbol') or data.get('FH_SYMBOL'),
        'fh_expiry_dt': _as_date_str(data.get('fh_expiry_dt') or data.get('expiry_date') or data.get('FH_EXPIRY_DT')),
        'fh_strike_price': data.get('fh_strike_price') or data.get('strike_price') or data.get('FH_STRIKE_PRICE'),
        'fh_option_type': data.get('fh_option_type') or data.get('option_type') or data.get('FH_OPTION_TYPE'),
        'fh_market_type': data.get('fh_market_type') or data.get('FH_MARKET_TYPE'),

        'fh_opening_price': data.get('fh_opening_price') or data.get('FH_OPENING_PRICE'),
        'fh_trade_high_price': data.get('fh_trade_high_price') or data.get('FH_TRADE_HIGH_PRICE'),
        'fh_trade_low_price': data.get('fh_trade_low_price') or data.get('FH_TRADE_LOW_PRICE'),
        'fh_closing_price': data.get('fh_closing_price') or data.get('FH_CLOSING_PRICE'),
        'fh_last_traded_price': data.get('fh_last_traded_price') or data.get('FH_LAST_TRADED_PRICE'),
        'fh_prev_cls': data.get('fh_prev_cls') or data.get('FH_PREV_CLS'),
        'fh_settle_price': data.get('fh_settle_price') or data.get('FH_SETTLE_PRICE'),

        'fh_tot_traded_qty': data.get('fh_tot_traded_qty') or data.get('volume') or data.get('FH_TOT_TRADED_QTY'),
        'fh_tot_traded_val': data.get('fh_tot_traded_val') or data.get('FH_TOT_TRADED_VAL'),
        'fh_open_int': data.get('fh_open_int') or data.get('open_interest') or data.get('FH_OPEN_INT'),
        'fh_change_in_oi': data.get('fh_change_in_oi') or data.get('change_in_oi') or data.get('FH_CHANGE_IN_OI'),
        'fh_market_lot': data.get('fh_market_lot') or data.get('FH_MARKET_LOT'),

        'fh_timestamp': _as_date_str(data.get('fh_timestamp') or data.get('date') or data.get('FH_TIMESTAMP')),
        'fh_timestamp_order': _as_ts(data.get('fh_timestamp_order') or data.get('timestamp') or data.get('FH_TIMESTAMP_ORDER')),
        'fh_underlying_value': data.get('fh_underlying_value') or data.get('underlying_value') or data.get('FH_UNDERLYING_VALUE'),
        'calculated_premium_val': data.get('calculated_premium_val') or data.get('CALCULATED_PREMIUM_VAL'),
    }

    # Remove None values to keep upsert payload small
    payload = {k: v for k, v in mapped.items() if v is not None}

    return upsert("options_data", payload)


def insert_fii_futures_data(data: Dict[str, Any]):
    return upsert("fii_futures_data", data)


def insert_fii_options_data(data: Dict[str, Any]):
    return upsert("fii_options_data", data)


# ---------- Example Usage ---------- #

if __name__ == "__main__":
    print("✅ Supabase connected:", SUPABASE_URL)
    print("📊 Table stats:", get_table_stats())

    # Sample insert
    record = {
        "date": str(date.today()),
        "symbol": "NIFTY",
        "expiry_date": "2025-10-30",
        "strike_price": 22500.0,
        "option_type": "CE",
        "open_interest": 11000,
        "change_in_oi": 300,
        "volume": 1000,
        "underlying_value": 22050.55,
    }

    insert_futures_data(record)
    print("✅ Record inserted successfully")
