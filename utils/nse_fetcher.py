import requests
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import time
import json

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
        try:
            self.session.get(self.base_url, headers=self.headers, timeout=10)
            time.sleep(1)
        except Exception as e:
            print(f"Warning: Could not initialize session: {str(e)}")
    
    def get_monthly_expiry_dates(self, start_date, months_back=3):
        """
        Calculate NIFTY monthly expiry dates (last Thursday of each month)
        """
        expiry_dates = []
        current_date = start_date
        
        for i in range(months_back):
            month_date = current_date - relativedelta(months=i)
            
            last_day = month_date.replace(day=1) + relativedelta(months=1) - timedelta(days=1)
            
            last_thursday = last_day
            while last_thursday.weekday() != 3:
                last_thursday -= timedelta(days=1)
            
            expiry_dates.append(last_thursday)
        
        return sorted(expiry_dates)
    
    def fetch_futures_data(self, from_date, to_date, symbol='NIFTY', expiry_date=None):
        """
        Fetch futures data from NSE API
        
        Args:
            from_date: Start date in format 'DD-MM-YYYY'
            to_date: End date in format 'DD-MM-YYYY'
            symbol: Symbol name (default: NIFTY)
            expiry_date: Expiry date in format 'DD-MMM-YYYY'
        """
        try:
            if expiry_date:
                expiry_str = expiry_date.strftime('%d-%b-%Y').upper()
                year = expiry_date.year
            else:
                expiry_str = ''
                year = datetime.now().year
            
            url = f"{self.base_url}/api/historicalOR/foCPV"
            
            params = {
                'from': from_date,
                'to': to_date,
                'instrumentType': 'FUTIDX',
                'symbol': symbol,
                'year': str(year),
                'expiryDate': expiry_str
            }
            
            print(f"Fetching data: {params}")
            
            response = self.session.get(
                url,
                params=params,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    return data
                except json.JSONDecodeError:
                    print(f"Failed to decode JSON response")
                    return None
            else:
                print(f"Request failed with status code: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return None
    
    def parse_futures_data(self, raw_data):
        """
        Parse raw futures data into structured format
        """
        if not raw_data or 'data' not in raw_data:
            return pd.DataFrame()
        
        records = []
        
        for item in raw_data['data']:
            record = {
                'date': datetime.strptime(item.get('FH_TIMESTAMP', ''), '%d-%b-%Y').date() if item.get('FH_TIMESTAMP') else None,
                'instrument_type': item.get('FH_INSTRUMENT_TYPE'),
                'symbol': item.get('FH_SYMBOL'),
                'expiry_date': datetime.strptime(item.get('FH_EXPIRY_DT', ''), '%d-%b-%Y').date() if item.get('FH_EXPIRY_DT') else None,
                'strike_price': float(item.get('FH_STRIKE_PRICE', 0)),
                'option_type': item.get('FH_OPTION_TYPE'),
                'open_interest': int(item.get('FH_OPEN_INT', 0)),
                'change_in_oi': int(item.get('FH_CHANGE_IN_OI', 0)),
                'volume': int(item.get('FH_CONTRACTS', 0)),
                'underlying_value': float(item.get('FH_UNDERLYING_VALUE', 0)),
                'timestamp': datetime.now()
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        return df
    
    def fetch_and_parse_futures(self, from_date, to_date, symbol='NIFTY', expiry_date=None):
        """
        Fetch and parse futures data in one step
        """
        raw_data = self.fetch_futures_data(from_date, to_date, symbol, expiry_date)
        
        if raw_data:
            df = self.parse_futures_data(raw_data)
            return df
        
        return pd.DataFrame()
    
    def fetch_multiple_expiries(self, symbol='NIFTY', months_back=3):
        """
        Fetch data for multiple monthly expiries
        """
        end_date = datetime.now()
        expiry_dates = self.get_monthly_expiry_dates(end_date, months_back)
        
        all_data = []
        
        for expiry in expiry_dates:
            from_date_dt = expiry - timedelta(days=90)
            to_date_dt = expiry
            
            from_date = from_date_dt.strftime('%d-%m-%Y')
            to_date = to_date_dt.strftime('%d-%m-%Y')
            
            print(f"\nFetching data for expiry: {expiry.strftime('%d-%b-%Y')}")
            
            df = self.fetch_and_parse_futures(from_date, to_date, symbol, expiry)
            
            if not df.empty:
                all_data.append(df)
                print(f"  Retrieved {len(df)} records")
            else:
                print(f"  No data retrieved")
            
            time.sleep(2)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['date', 'expiry_date', 'symbol'])
            return combined_df
        
        return pd.DataFrame()
