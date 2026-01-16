import calendar
from datetime import datetime, timedelta
import json
import os
import pandas as pd
import requests
import subprocess
import tempfile
import time


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
        Get NIFTY monthly expiry dates (last Tuesday of the month).
        
        Args:
            start_date: Reference date
            months_back: Number of months to look back
            
        Returns:
            List of expiry dates (sorted) within the range
        """
        if isinstance(start_date, datetime):
            ref_date = start_date
        else:
            ref_date = datetime.combine(start_date, datetime.min.time())

        def _last_tuesday(year: int, month: int) -> datetime:
            last_day = datetime(year, month, calendar.monthrange(year, month)[1])
            while last_day.weekday() != 1:
                last_day -= timedelta(days=1)
            if last_day.month == 3 and last_day.day == 31:
                last_day -= timedelta(days=1)
            return last_day

        expiries = []
        year = ref_date.year
        month = ref_date.month
        while len(expiries) < months_back:
            expiry = _last_tuesday(year, month)
            if expiry <= ref_date:
                expiries.append(expiry)
            month -= 1
            if month < 1:
                month = 12
                year -= 1

        return sorted(expiries)


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
                    # Date and identification
                    'date': datetime.strptime(item.get('FH_TIMESTAMP', ''), '%d-%b-%Y').date() if item.get('FH_TIMESTAMP') else None,
                    'timestamp_order': item.get('FH_TIMESTAMP_ORDER'),
                    'instrument_type': item.get('FH_INSTRUMENT') or 'FUTIDX',
                    'symbol': item.get('FH_SYMBOL'),
                    'market_type': item.get('FH_MARKET_TYPE'),
                    
                    # Expiry
                    'expiry_date': datetime.strptime(item.get('FH_EXPIRY_DT', ''), '%d-%b-%Y').date() if item.get('FH_EXPIRY_DT') else None,
                    'expiry': datetime.strptime(item.get('FH_EXPIRY_DT', ''), '%d-%b-%Y').date() if item.get('FH_EXPIRY_DT') else None,  # Alias
                    'strike_price': 0,  # Futures don't have strikes
                    'option_type': None,
                    
                    # Price data
                    'open': float(item.get('FH_OPENING_PRICE', 0)),
                    'high': float(item.get('FH_TRADE_HIGH_PRICE', 0)),
                    'low': float(item.get('FH_TRADE_LOW_PRICE', 0)),
                    'close': float(item.get('FH_CLOSING_PRICE', 0)),
                    'ltp': float(item.get('FH_LAST_TRADED_PRICE', 0)),
                    'prev_close': float(item.get('FH_PREV_CLS', 0)),
                    'settle_price': float(item.get('FH_SETTLE_PRICE', 0)),
                    
                    # Volume and OI
                    'open_interest': int(item.get('FH_OPEN_INT', 0)),
                    'open_int': int(item.get('FH_OPEN_INT', 0)),  # Alias
                    'change_in_oi': int(item.get('FH_CHANGE_IN_OI', 0)),
                    'volume': int(item.get('FH_TOT_TRADED_QTY', 0)),
                    'no_of_contracts': int(item.get('FH_TOT_TRADED_QTY', 0)),  # Alias
                    'traded_value': float(item.get('FH_TOT_TRADED_VAL', 0)),
                    
                    # Other
                    'market_lot': int(item.get('FH_MARKET_LOT', 0)),
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


    def fetch_options_data(self, from_date, to_date, symbol, expiry_str, option_type, year=2025):
        """
        Fetch options data using curl with cookies
        
        Args:
            from_date: Start date in format 'DD-MM-YYYY'
            to_date: End date in format 'DD-MM-YYYY'
            symbol: Symbol name (e.g., 'NIFTY')
            expiry_str: Expiry date as datetime object
            option_type: 'CE' for Call or 'PE' for Put
            year: Year of expiry (default: 2025)
            
        Returns:
            Raw JSON response from NSE API or None if error
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
            expiry_date = expiry_str.strftime('%d-%b-%Y').upper()  # '28-AUG-2025'
            
            url = f"https://www.nseindia.com/api/historicalOR/foCPV?from={from_date}&to={to_date}&instrumentType=OPTIDX&symbol={symbol}&year={year}&expiryDate={expiry_date}&optionType={option_type}&csv=true"
            print(f"URL ({option_type}): {url}")
            
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


    def parse_options_data(self, raw_data):
        """
        Parse raw options data into structured DataFrame
        
        Args:
            raw_data: Raw JSON response from NSE API
            
        Returns:
            DataFrame with parsed options data
        """
        if not raw_data or 'data' not in raw_data:
            return pd.DataFrame()
        
        records = []
        
        for item in raw_data['data']:
            try:
                record = {
                    # Date and identification
                    'date': datetime.strptime(item.get('FH_TIMESTAMP', ''), '%d-%b-%Y').date() if item.get('FH_TIMESTAMP') else None,
                    'timestamp_order': item.get('FH_TIMESTAMP_ORDER'),
                    'instrument_type': item.get('FH_INSTRUMENT') or 'OPTIDX',
                    'symbol': item.get('FH_SYMBOL'),
                    'market_type': item.get('FH_MARKET_TYPE'),
                    
                    # Expiry and strike
                    'expiry_date': datetime.strptime(item.get('FH_EXPIRY_DT', ''), '%d-%b-%Y').date() if item.get('FH_EXPIRY_DT') else None,
                    'expiry': datetime.strptime(item.get('FH_EXPIRY_DT', ''), '%d-%b-%Y').date() if item.get('FH_EXPIRY_DT') else None,  # Alias for compatibility
                    'strike_price': float(item.get('FH_STRIKE_PRICE', 0)),
                    'option_type': item.get('FH_OPTION_TYPE'),
                    
                    # Price data
                    'open': float(item.get('FH_OPENING_PRICE', 0)),
                    'high': float(item.get('FH_TRADE_HIGH_PRICE', 0)),
                    'low': float(item.get('FH_TRADE_LOW_PRICE', 0)),
                    'close': float(item.get('FH_CLOSING_PRICE', 0)),
                    'ltp': float(item.get('FH_LAST_TRADED_PRICE', 0)),
                    'prev_close': float(item.get('FH_PREV_CLS', 0)),
                    'settle_price': float(item.get('FH_SETTLE_PRICE', 0)),
                    'calculated_premium': float(item.get('CALCULATED_PREMIUM_VAL', 0)),
                    
                    # Volume and OI
                    'open_interest': int(item.get('FH_OPEN_INT', 0)),
                    'open_int': int(item.get('FH_OPEN_INT', 0)),  # Alias for compatibility
                    'change_in_oi': int(item.get('FH_CHANGE_IN_OI', 0)),
                    'volume': int(item.get('FH_TOT_TRADED_QTY', 0)),
                    'no_of_contracts': int(item.get('FH_TOT_TRADED_QTY', 0)),  # Alias for compatibility
                    'traded_value': float(item.get('FH_TOT_TRADED_VAL', 0)),
                    
                    # Other
                    'market_lot': int(item.get('FH_MARKET_LOT', 0)),
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


    def fetch_and_parse_options(self, from_date, to_date, symbol='NIFTY', expiry_date=None, option_type='CE'):
        """
        Fetch and parse options data in one step
        
        Args:
            from_date: Start date in format 'DD-MM-YYYY'
            to_date: End date in format 'DD-MM-YYYY'
            symbol: Symbol name (default: NIFTY)
            expiry_date: Expiry date as datetime object
            option_type: 'CE' for Call or 'PE' for Put
            
        Returns:
            DataFrame with parsed options data
        """
        raw_data = self.fetch_options_data(from_date, to_date, symbol, expiry_date, option_type)
        
        if raw_data:
            df = self.parse_options_data(raw_data)
            return df
        
        return pd.DataFrame()
