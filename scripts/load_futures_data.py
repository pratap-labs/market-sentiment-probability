import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.models import insert_futures_data, get_table_stats
from views.utils.nse_fetcher import NSEDataFetcher
from datetime import datetime

def main():
    """
    Main script to fetch NSE futures data and store in Supabase
    """
    print("\n" + "="*60)
    print("NSE FUTURES DATA LOADER (Supabase version)")
    print("="*60 + "\n")
    
    # Step 1: Initialize NSE Data Fetcher
    print("Step 1: Initializing NSE Data Fetcher...")
    fetcher = NSEDataFetcher()
    print("✓ Fetcher initialized\n")
    
    # Step 2: Fetch data
    print("Step 2: Fetching futures data from NSE...")
    df = fetcher.fetch_multiple_expiries(symbol='NIFTY', months_back=3)
    
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
    
    for _, row in df.iterrows():
        try:
            record = {
                "date": str(row["date"]),
                "instrument_type": row.get("instrument_type"),
                "symbol": row["symbol"],
                "expiry_date": str(row["expiry_date"]),
                "strike_price": float(row["strike_price"]) if not pd.isna(row["strike_price"]) else None,
                "option_type": row.get("option_type"),
                "open_interest": int(row["open_interest"]) if not pd.isna(row["open_interest"]) else None,
                "change_in_oi": int(row["change_in_oi"]) if not pd.isna(row["change_in_oi"]) else None,
                "volume": int(row["volume"]) if not pd.isna(row["volume"]) else None,
                "underlying_value": float(row["underlying_value"]) if not pd.isna(row["underlying_value"]) else None,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            res = insert_futures_data(record)
            if not res.error:
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
    import pandas as pd
    main()
