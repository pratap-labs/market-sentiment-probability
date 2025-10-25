import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import json
import subprocess
import tempfile
import os


class NSEOptionsFetcher:
    """
    Fetch options data from NSE India (historical FO endpoint).
    This mirrors the patterns used in `utils/nse_fetcher.py` but targets options (OPTIDX).
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
        """Initialize session by visiting NSE homepage to get cookies."""
        init_headers = {
            'User-Agent': self.headers['User-Agent'],
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }

        try:
            resp = self.session.get(self.base_url, headers=init_headers, timeout=10)
            time.sleep(1)
            return resp.status_code == 200
        except Exception:
            return False


    def fetch_options_data(self, from_date, to_date, symbol, expiry_date, year=None, option_type=None, strike_price=None):
        """
        Fetch options (OPTIDX) historical FO data using curl to preserve cookie handling.

        Args:
            from_date: 'DD-MM-YYYY'
            to_date: 'DD-MM-YYYY'
            symbol: e.g. 'NIFTY'
            expiry_date: datetime object
            year: optional year parameter for the API (int)

        Returns:
            Parsed JSON dict or None
        """
        cookie_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        cookie_path = cookie_file.name
        cookie_file.close()

        try:
            # Get cookies from homepage
            subprocess.run([
                'curl', 'https://www.nseindia.com',
                '-c', cookie_path, '-s', '-o', '/dev/null',
                '-H', f"User-Agent: {self.headers['User-Agent']}"
            ], timeout=10)

            expiry_str = expiry_date.strftime('%d-%b-%Y').upper()
            if year is None:
                year = expiry_date.year

            # Build URL and include optional params when provided
            url = (
                f"https://www.nseindia.com/api/historicalOR/foCPV?from={from_date}&to={to_date}"
                f"&instrumentType=OPTIDX&symbol={symbol}&year={year}&expiryDate={expiry_str}"
            )

            if option_type:
                # API expects optionType param like 'CE' or 'PE'
                url += f"&optionType={option_type}"

            if strike_price is not None:
                # Ensure integer/float formatting
                url += f"&strikePrice={int(strike_price)}"


            print(f'Fetching URL: {url}')

            result = subprocess.run([
                'curl', url, '-b', cookie_path,
                '-H', f"User-Agent: {self.headers['User-Agent']}",
                '-H', 'Accept: application/json',
                '-H', 'Referer: https://www.nseindia.com/get-quotes/derivatives',
                '--compressed', '-s'
            ], capture_output=True, text=True, timeout=30)

            print(f'Curl command completed with return code {result.returncode}')
            print(f'Curl stdout length: {len(result.stdout) if result.stdout else 0}')

            if not result.stdout:
                return None

            data = json.loads(result.stdout)
            return data
        except Exception as e:
            print(f"Error fetching options data: {e}")
            return None
        finally:
            if os.path.exists(cookie_path):
                os.remove(cookie_path)


    def parse_options_data(self, raw_data):
        """
        Convert raw JSON to a pandas DataFrame for options.
        The parser is tolerant to missing keys in the NSE response.
        """
        if not raw_data or 'data' not in raw_data:
            return pd.DataFrame()

        # Map all known FH_* fields (and sensible fallbacks) so the DataFrame columns
        # align with the DB table `options_data` which expects FH_* column names.
        def to_float(x):
            try:
                if x is None or x == '':
                    return None
                return float(x)
            except Exception:
                return None

        def to_int(x):
            try:
                if x is None or x == '':
                    return None
                return int(float(x))
            except Exception:
                return None

        records = []
        for item in raw_data.get('data', []):
            # Map fields directly from the NSE payload where possible
            r = {
                'FH_INSTRUMENT': item.get('FH_INSTRUMENT') or item.get('FH_INSTRUMENT_TYPE') or item.get('instrument'),
                'FH_SYMBOL': item.get('FH_SYMBOL') or item.get('symbol'),
                'FH_EXPIRY_DT': item.get('FH_EXPIRY_DT') or item.get('FH_EXPIRY_DATE') or item.get('expiry_date'),
                'FH_STRIKE_PRICE': to_float(item.get('FH_STRIKE_PRICE') or item.get('FH_STRIKE_PR') or item.get('STRIKE_PRICE') or item.get('strike_price')),
                'FH_OPTION_TYPE': item.get('FH_OPTION_TYPE') or item.get('option_type'),
                'FH_MARKET_TYPE': item.get('FH_MARKET_TYPE') or item.get('market_type') or item.get('FH_MARKET_TYPE'),

                'FH_OPENING_PRICE': to_float(item.get('FH_OPENING_PRICE')),
                'FH_TRADE_HIGH_PRICE': to_float(item.get('FH_TRADE_HIGH_PRICE')),
                'FH_TRADE_LOW_PRICE': to_float(item.get('FH_TRADE_LOW_PRICE')),
                'FH_CLOSING_PRICE': to_float(item.get('FH_CLOSING_PRICE')),
                'FH_LAST_TRADED_PRICE': to_float(item.get('FH_LAST_TRADED_PRICE')),
                'FH_PREV_CLS': to_float(item.get('FH_PREV_CLS')),
                'FH_SETTLE_PRICE': to_float(item.get('FH_SETTLE_PRICE')),

                'FH_TOT_TRADED_QTY': to_int(item.get('FH_TOT_TRADED_QTY') or item.get('TOT_TRADED_QTY') or item.get('total_traded_qty')),
                'FH_TOT_TRADED_VAL': to_float(item.get('FH_TOT_TRADED_VAL') or item.get('TOT_TRADED_VAL')),
                'FH_OPEN_INT': to_int(item.get('FH_OPEN_INT') or item.get('OPEN_INT') or item.get('open_interest')),
                'FH_CHANGE_IN_OI': to_int(item.get('FH_CHANGE_IN_OI') or item.get('CHANGE_IN_OI') or item.get('change_in_oi')),
                'FH_MARKET_LOT': to_int(item.get('FH_MARKET_LOT') or item.get('MARKET_LOT')),

                'FH_TIMESTAMP': item.get('FH_TIMESTAMP') or item.get('timestamp') or item.get('date'),
                'FH_TIMESTAMP_ORDER': item.get('FH_TIMESTAMP_ORDER') or item.get('timestamp_order') or item.get('FH_TIMESTAMP_ORDER'),
                'FH_UNDERLYING_VALUE': to_float(item.get('FH_UNDERLYING_VALUE') or item.get('underlying_value')),
                'CALCULATED_PREMIUM_VAL': to_float(item.get('CALCULATED_PREMIUM_VAL') or item.get('calculated_premium_val')),
            }

            records.append(r)

        df = pd.DataFrame(records)
        return df


    def fetch_and_parse_options(self, from_date, to_date, symbol='NIFTY', expiry_date=None, option_type=None, strike_price=None):
        """Fetch & parse options; supports filtering by option_type and strike_price when provided."""
        raw = self.fetch_options_data(from_date, to_date, symbol, expiry_date, option_type=option_type, strike_price=strike_price)
        if raw:
            return self.parse_options_data(raw)
        return pd.DataFrame()


    def fetch_multiple_expiries(self, symbol='NIFTY', months_back=6):
        """
        Fetch options data for a hardcoded set of monthly expiries.

        For each expiry we fetch both CE and PE (no strike filtering) and combine results.
        """
        # Hardcoded expiry timestamps (use these exact expiry dates)
        expiries = [
            datetime(2025, 1, 30),
            datetime(2025, 2, 27),
            datetime(2025, 3, 27),
            datetime(2025, 4, 24),
            datetime(2025, 5, 29),
            datetime(2025, 6, 26),
            datetime(2025, 7, 31),
            datetime(2025, 8, 28),
            datetime(2025, 9, 25),
            datetime(2025, 10, 28),
            datetime(2025, 11, 27),
            datetime(2025, 12, 24),
        ]

        all_data = []

        for idx, expiry in enumerate(expiries, start=1):
            from_dt = (expiry - timedelta(days=90)).strftime('%d-%m-%Y')
            to_dt = expiry.strftime('%d-%m-%Y')

            # Fetch CE and PE for this expiry
            # LOG
            print(f"Fetching options for expiry {expiry.strftime('%d-%b-%Y')} ({idx}/{len(expiries)})...")
            try:
                df_ce = self.fetch_and_parse_options(from_dt, to_dt, symbol, expiry, option_type='CE')
                print(f"Fetched {len(df_ce)} CE records for expiry {expiry.strftime('%d-%b-%Y')}")
            except Exception as e:
                print(f"Error fetching CE for {expiry}: {e}")
                df_ce = pd.DataFrame()

            try:
                df_pe = self.fetch_and_parse_options(from_dt, to_dt, symbol, expiry, option_type='PE')
                print(f"Fetched {len(df_pe)} PE records for expiry {expiry.strftime('%d-%b-%Y')}")
            except Exception as e:
                print(f"Error fetching PE for {expiry}: {e}")
                df_pe = pd.DataFrame()

            if not df_ce.empty:
                all_data.append(df_ce)
            if not df_pe.empty:
                all_data.append(df_pe)

            # print first few rows
            print(f"CE Data Sample:\n{df_ce.head()}")
            print(f"PE Data Sample:\n{df_pe.head()}")

            # Be polite to the NSE servers
            if idx < len(expiries):
                time.sleep(2)

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)

            # Prefer FH_* column names since parser returns those. Fall back to generic names
            possible_subsets = [
                ['FH_TIMESTAMP', 'FH_EXPIRY_DT', 'FH_SYMBOL', 'FH_STRIKE_PRICE', 'FH_OPTION_TYPE'],
                ['FH_TIMESTAMP', 'FH_EXPIRY_DT', 'FH_SYMBOL', 'FH_STRIKE_PRICE'],
                ['FH_EXPIRY_DT', 'FH_SYMBOL', 'FH_STRIKE_PRICE', 'FH_OPTION_TYPE']
            ]

            subset_cols = None
            for cols in possible_subsets:
                if all(c in combined.columns for c in cols):
                    subset_cols = cols
                    break

            if subset_cols:
                combined = combined.drop_duplicates(subset=subset_cols)
            else:
                # Last resort: full-row dedupe
                combined = combined.drop_duplicates()

            return combined

        return pd.DataFrame()
