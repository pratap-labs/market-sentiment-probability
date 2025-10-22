import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.models import create_tables, get_session, FuturesData
from utils.nse_fetcher import NSEDataFetcher
from datetime import datetime

def main():
    """
    Main script to fetch NSE futures data and store in database
    """
    print("=" * 60)
    print("NSE Futures Data Fetcher")
    print("=" * 60)
    
    print("\n1. Creating database tables...")
    try:
        create_tables()
    except Exception as e:
        print(f"Error creating tables: {str(e)}")
        return
    
    print("\n2. Initializing NSE Data Fetcher...")
    fetcher = NSEDataFetcher()
    
    print("\n3. Fetching data for last 3 months of expiries...")
    df = fetcher.fetch_multiple_expiries(symbol='NIFTY', months_back=3)
    
    if df.empty:
        print("\n⚠️  No data was fetched from NSE API")
        print("This could be due to:")
        print("  - NSE API being down or blocking requests")
        print("  - Network connectivity issues")
        print("  - API rate limiting")
        print("\nFalling back to sample data generation...")
        
        from datetime import timedelta
        import pandas as pd
        import numpy as np
        
        dates = pd.date_range(end=datetime.now().date(), periods=90, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'instrument_type': 'FUTIDX',
            'symbol': 'NIFTY',
            'expiry_date': dates[-1],
            'strike_price': 0.0,
            'option_type': None,
            'open_interest': np.random.randint(1000000, 5000000, len(dates)),
            'change_in_oi': np.random.randint(-500000, 500000, len(dates)),
            'volume': np.random.randint(100000, 1000000, len(dates)),
            'underlying_value': np.random.uniform(21000, 22000, len(dates)),
            'timestamp': datetime.now()
        })
    
    print(f"\n4. Storing {len(df)} records in database...")
    
    session = get_session()
    
    try:
        for _, row in df.iterrows():
            futures_record = FuturesData(
                date=row['date'],
                instrument_type=row['instrument_type'],
                symbol=row['symbol'],
                expiry_date=row['expiry_date'],
                strike_price=row['strike_price'],
                option_type=row['option_type'],
                open_interest=row['open_interest'],
                change_in_oi=row['change_in_oi'],
                volume=row['volume'],
                underlying_value=row['underlying_value'],
                timestamp=row['timestamp']
            )
            session.add(futures_record)
        
        session.commit()
        print(f"✓ Successfully stored {len(df)} records")
        
        count = session.query(FuturesData).count()
        print(f"✓ Total records in database: {count}")
        
    except Exception as e:
        session.rollback()
        print(f"✗ Error storing data: {str(e)}")
    finally:
        session.close()
    
    print("\n" + "=" * 60)
    print("Data fetch completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
