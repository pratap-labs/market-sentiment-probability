import requests
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import time
import json
import gzip
import subprocess
import json
import subprocess
import json
import tempfile
import os




class NSEDataFetcher:
    """
    Fetch data from NSE India API for futures and options
    """
    
    def __init__(self):
        self.base_url = "https://www.nseindia.com"
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'X-Requested-With': 'XMLHttpRequest',
            'Connection': 'keep-alive',
            'Referer': 'https://www.nseindia.com/get-quotes/derivatives'
        }
        self._initialize_session()
    

    def _initialize_session(self):
        """Initialize session by visiting NSE homepage to get cookies"""
        
        # Update headers for initial request
        init_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        }
        
        try:
            print("Initializing NSE session...")
            
            # First request to homepage
            response = self.session.get(
                self.base_url, 
                headers=init_headers, 
                timeout=10
            )
            
            print(f"  Homepage Status: {response.status_code}")
            print(f"  Cookies received: {len(self.session.cookies)}")
            
            if self.session.cookies:
                print(f"  Cookie names: {list(self.session.cookies.keys())}")
            else:
                print("  ⚠ WARNING: No cookies received from homepage!")
                return False
            
            time.sleep(2)  # Increased delay
            
            return response.status_code == 200
            
        except Exception as e:
            print(f"❌ Failed to initialize session: {str(e)}")
            return False


    def get_monthly_expiry_dates(self, start_date, months_back=3):
        """
        Get NIFTY monthly expiry dates (hardcoded values)
        
        Args:
            start_date: Reference date
            months_back: Number of months to look back
            
        Returns:
            List of expiry dates (sorted) within the range
        """
        from datetime import datetime
        
        # Hardcoded monthly expiry dates from the options chain
        all_expiry_dates = [
            datetime(2025, 1, 30),   # 30-JAN-2025
            datetime(2025, 2, 27),   # 27-FEB-2025
            datetime(2025, 3, 27),   # 27-MAR-2025
            datetime(2025, 4, 24),   # 24-APR-2025
            datetime(2025, 5, 29),   # 29-MAY-2025
            datetime(2025, 6, 26),   # 26-JUN-2025
            datetime(2025, 7, 31),   # 31-JUL-2025
            datetime(2025, 8, 28),   # 28-AUG-2025
            datetime(2025, 9, 25),   # 25-SEP-2025
            datetime(2025, 9, 30),   # 30-SEP-2025
            datetime(2025, 10, 28),  # 28-OCT-2025
            datetime(2025, 11, 25),  # 25-NOV-2025
            datetime(2025, 12, 30),  # 30-DEC-2025
        ]
        
        # Filter dates based on start_date and months_back
        filtered_dates = []
        for expiry_date in all_expiry_dates:
            if expiry_date <= start_date:
                # Check if within months_back range
                months_diff = (start_date.year - expiry_date.year) * 12 + (start_date.month - expiry_date.month)
                if months_diff < months_back:
                    filtered_dates.append(expiry_date)
        
        return sorted(all_expiry_dates)


    def fetch_futures_data(self, from_date, to_date, symbol, expiry_str, year=2025):
        """
        Fetch futures data using curl with cookies
        """
        cookie_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        cookie_path = cookie_file.name
        cookie_file.close()
        
        try:
            # Get cookies from homepage
            subprocess.run([
                'curl', 'https://www.nseindia.com',
                '-c', cookie_path, '-s', '-o', '/dev/null',
                '-H', 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            ], timeout=10)
            
            # API call with cookies
            expiry_date = expiry_str.strftime('%d-%b-%Y').upper()  # '28-AUG-2025'RetryClaude does not have the ability to run the code it generates yet.P

            url = f"https://www.nseindia.com/api/historicalOR/foCPV?from={from_date}&to={to_date}&instrumentType=FUTIDX&symbol=NIFTY&year={year}&expiryDate={expiry_date}"
            print(f"URL: {url}")
            result = subprocess.run([
                'curl', url, '-b', cookie_path,
                '-H', 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                '-H', 'Accept: application/json',
                '-H', 'Referer: https://www.nseindia.com/get-quotes/derivatives',
                '--compressed', '-s'
            ], capture_output=True, text=True, timeout=30)


            data = json.loads(result.stdout)
            print(f"  ✓ {len(data.get('data', []))} records")
            return data
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return None
        finally:
            if os.path.exists(cookie_path):
                os.remove(cookie_path)


    def parse_futures_data(self, raw_data):
        """
        Parse raw futures data into structured DataFrame
        
        Args:
            raw_data: Raw JSON response from NSE API
            
        Returns:
            DataFrame with parsed futures data
        """
        if not raw_data or 'data' not in raw_data:
            return pd.DataFrame()
        
        records = []
        
        for item in raw_data['data']:
            try:
                record = {
                    'date': datetime.strptime(item.get('FH_TIMESTAMP', ''), '%d-%b-%Y').date() if item.get('FH_TIMESTAMP') else None,
                    'instrument_type': item.get('FH_INSTRUMENT_TYPE'),
                    'symbol': item.get('FH_SYMBOL'),
                    'expiry_date': datetime.strptime(item.get('FH_EXPIRY_DT', ''), '%d-%b-%Y').date() if item.get('FH_EXPIRY_DT') else None,
                    'strike_price': 0,
                    'option_type': item.get('FH_OPTION_TYPE'),
                    'open_interest': int(item.get('FH_OPEN_INT', 0)),
                    'change_in_oi': int(item.get('FH_CHANGE_IN_OI', 0)),
                    'volume': int(item.get('FH_TOT_TRADED_QTY', 0)),
                    'underlying_value': float(item.get('FH_UNDERLYING_VALUE', 0)),
                    'timestamp': datetime.now()
                }
                records.append(record)
            except Exception as e:
                print(f"  ⚠ Error parsing record: {str(e)}")
                continue
        
        df = pd.DataFrame(records)
        
        # Remove records with missing critical data
        if not df.empty:
            df = df.dropna(subset=['date', 'symbol', 'expiry_date'])
        
        return df
    

    def fetch_and_parse_futures(self, from_date, to_date, symbol='NIFTY', expiry_date=None):
        """
        Fetch and parse futures data in one step
        
        Args:
            from_date: Start date in format 'DD-MM-YYYY'
            to_date: End date in format 'DD-MM-YYYY'
            symbol: Symbol name (default: NIFTY)
            expiry_date: Expiry date as datetime object
            
        Returns:
            DataFrame with parsed futures data
        """
        raw_data = self.fetch_futures_data(from_date, to_date, symbol, expiry_date)
        
        if raw_data:
            df = self.parse_futures_data(raw_data)
            return df
        
        return pd.DataFrame()
    
    
    def fetch_multiple_expiries(self, symbol='NIFTY', months_back=10):
        """
        Fetch data for multiple monthly expiries
        
        Args:
            symbol: Symbol name (default: NIFTY)
            months_back: Number of months to look back
            
        Returns:
            Combined DataFrame with all expiries
        """
        end_date = datetime.now()
        expiry_dates = self.get_monthly_expiry_dates(end_date, months_back)
        
        all_data = []
        
        print(f"\n{'='*60}")
        print(f"Fetching {symbol} futures data for {months_back} expiries")
        print(f"{'='*60}\n")
        
        for idx, expiry in enumerate(expiry_dates, 1):
            # Fetch 90 days of data before each expiry
            from_date_dt = expiry - timedelta(days=90)
            to_date_dt = expiry
            
            from_date = from_date_dt.strftime('%d-%m-%Y')
            to_date = to_date_dt.strftime('%d-%m-%Y')
            
            print(f"[{idx}/{len(expiry_dates)}] Expiry: {expiry.strftime('%d-%b-%Y')}")
            
            df = self.fetch_and_parse_futures(from_date, to_date, symbol, expiry)
            
            if not df.empty:
                all_data.append(df)
            
            # Rate limiting - be nice to NSE servers
            if idx < len(expiry_dates):
                time.sleep(2)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Remove duplicates
            initial_count = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=['date', 'expiry_date', 'symbol'])
            final_count = len(combined_df)
            
            if initial_count > final_count:
                print(f"\n  ℹ Removed {initial_count - final_count} duplicate records")
            
            print(f"\n{'='*60}")
            print(f"✓ Total records fetched: {final_count}")
            print(f"{'='*60}\n")
            
            return combined_df
        
        print(f"\n{'='*60}")
        print(f"✗ No data fetched")
        print(f"{'='*60}\n")
        
        return pd.DataFrame()