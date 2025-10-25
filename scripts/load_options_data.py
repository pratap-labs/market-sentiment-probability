import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.models import insert_options_data, get_table_stats
from utils.nse_options_fetcher import NSEOptionsFetcher
from datetime import datetime

def main():
    """
    Main script to fetch NSE options data and store in Supabase
    """
    print("\n" + "="*60)
    print("NSE OPTIONS DATA LOADER (Supabase version)")
    print("="*60 + "\n")
    
    # Step 1: Initialize NSE Options Fetcher
    print("Step 1: Initializing NSE Options Fetcher...")
    fetcher = NSEOptionsFetcher()
    print("✓ Fetcher initialized\n")
    
    # Step 2: Fetch data (hardcoded expiries inside the fetcher)
    print("Step 2: Fetching options data from NSE (CE + PE for hardcoded expiries)...")
    df = fetcher.fetch_multiple_expiries(symbol='NIFTY')
    
    if df.empty:
        print("\n⚠️  WARNING: No data was fetched from NSE API")
        print("Possible reasons:")
        print("  - NSE API is down or blocking requests")
        print("  - Network issues or rate limiting")
        print("  - Weekend or holiday")
        return
    
    # Step 3: Store in Supabase
    print(f"Step 3: Uploading {len(df)} records to Supabase...\n")

    records_added = 0
    errors = 0
    import pandas as pd

    for _, row in df.iterrows():
        try:
            # Map DataFrame FH_* columns directly to DB columns expected by insert_options_data
            # Normalize expiry and timestamp formats where possible
            fh_expiry = row.get('FH_EXPIRY_DT') or row.get('FH_EXPIRY_DATE') or row.get('expiry_date')
            try:
                if isinstance(fh_expiry, str) and '-' in fh_expiry:
                    parsed_expiry = datetime.strptime(fh_expiry, '%d-%b-%Y').date()
                    fh_expiry_str = parsed_expiry.isoformat()
                elif hasattr(fh_expiry, 'strftime'):
                    fh_expiry_str = fh_expiry.strftime('%Y-%m-%d')
                else:
                    fh_expiry_str = str(fh_expiry) if fh_expiry is not None else None
            except Exception:
                fh_expiry_str = str(fh_expiry) if fh_expiry is not None else None

            fh_timestamp = row.get('FH_TIMESTAMP') or row.get('timestamp') or row.get('date')
            fh_timestamp_order = row.get('FH_TIMESTAMP_ORDER') or row.get('timestamp_order')

            def _as_float(v):
                try:
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        return None
                    return float(v)
                except Exception:
                    return None

            def _as_int(v):
                try:
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        return None
                    return int(float(v))
                except Exception:
                    return None

            record = {
                'fh_instrument': row.get('FH_INSTRUMENT') or row.get('FH_INSTRUMENT_TYPE') or row.get('fh_instrument') or 'OPTIDX',
                'fh_symbol': row.get('FH_SYMBOL') or row.get('symbol') or row.get('fh_symbol'),
                'fh_expiry_dt': fh_expiry_str,
                'fh_strike_price': _as_float(row.get('FH_STRIKE_PRICE') if 'FH_STRIKE_PRICE' in row else row.get('strike_price') or row.get('fh_strike_price')),
                'fh_option_type': row.get('FH_OPTION_TYPE') or row.get('option_type') or row.get('fh_option_type'),
                'fh_market_type': row.get('FH_MARKET_TYPE') or row.get('market_type') or row.get('fh_market_type'),

                'fh_opening_price': _as_float(row.get('FH_OPENING_PRICE') or row.get('fh_opening_price')),
                'fh_trade_high_price': _as_float(row.get('FH_TRADE_HIGH_PRICE') or row.get('fh_trade_high_price')),
                'fh_trade_low_price': _as_float(row.get('FH_TRADE_LOW_PRICE') or row.get('fh_trade_low_price')),
                'fh_closing_price': _as_float(row.get('FH_CLOSING_PRICE') or row.get('fh_closing_price')),
                'fh_last_traded_price': _as_float(row.get('FH_LAST_TRADED_PRICE') or row.get('fh_last_traded_price')),
                'fh_prev_cls': _as_float(row.get('FH_PREV_CLS') or row.get('fh_prev_cls')),
                'fh_settle_price': _as_float(row.get('FH_SETTLE_PRICE') or row.get('fh_settle_price')),

                'fh_tot_traded_qty': _as_int(row.get('FH_TOT_TRADED_QTY') or row.get('fh_tot_traded_qty')),
                'fh_tot_traded_val': _as_float(row.get('FH_TOT_TRADED_VAL') or row.get('fh_tot_traded_val')),
                'fh_open_int': _as_int(row.get('FH_OPEN_INT') if 'FH_OPEN_INT' in row else row.get('open_interest') or row.get('fh_open_int')),
                'fh_change_in_oi': _as_int(row.get('FH_CHANGE_IN_OI') or row.get('fh_change_in_oi')),
                'fh_market_lot': _as_int(row.get('FH_MARKET_LOT') or row.get('fh_market_lot')),

                'fh_timestamp': fh_timestamp,
                'fh_timestamp_order': fh_timestamp_order,
                'fh_underlying_value': _as_float(row.get('FH_UNDERLYING_VALUE') if 'FH_UNDERLYING_VALUE' in row else row.get('underlying_value') or row.get('fh_underlying_value')),
                'calculated_premium_val': _as_float(row.get('CALCULATED_PREMIUM_VAL') or row.get('calculated_premium_val')),
            }

            # Remove None values to keep payload small
            payload = {k: v for k, v in record.items() if v is not None}

            # Validate required NOT NULL columns for options_data table
            missing = []
            for req in ('fh_instrument', 'fh_symbol', 'fh_expiry_dt', 'fh_strike_price'):
                if req not in payload or payload.get(req) is None:
                    missing.append(req)

            if missing:
                errors += 1
                print(f"✗ Skipping row due to missing required fields: {missing}")
                continue

            res = insert_options_data(payload)
            if not getattr(res, 'error', False):
                records_added += 1
            else:
                errors += 1
                print(f"✗ Error inserting record: {res.error}")
        except Exception as e:
            errors += 1
            print(f"✗ Exception: {e}")
    
    print(f"\n✓ Successfully added {records_added} records")
    if errors:
        print(f"⚠️ {errors} records failed to insert")
    
    # Step 4: Summary
    print("\nStep 4: Table summary\n---------------------------")
    stats = get_table_stats()
    for table, count in stats.items():
        print(f"{table}: {count:,} rows")

    print("\n" + "="*60)
    print("DATA LOAD COMPLETED!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
