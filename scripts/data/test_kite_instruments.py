"""
Test script to list all instrument tokens for futures using Kite API
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime
from typing import List, Dict

try:
    from kiteconnect import KiteConnect
except ImportError:
    print("❌ kiteconnect not installed. Install with: pip install kiteconnect")
    sys.exit(1)


def get_kite_client():
    """Get authenticated Kite client from Streamlit session or environment."""
    
    # First try to get from Streamlit session state (if running in Streamlit context)
    try:
        import streamlit as st
        api_key = st.session_state.get("kite_api_key")
        access_token = st.session_state.get("kite_access_token")
        
        if api_key and access_token:
            print("✅ Found Kite credentials in Streamlit session")
        else:
            print("⚠️ No Kite credentials found in Streamlit session")
            api_key = None
            access_token = None
    except ImportError:
        # Not running in Streamlit context
        print("ℹ️ Not running in Streamlit context, checking environment variables")
        api_key = None
        access_token = None
    except Exception as e:
        print(f"⚠️ Error accessing Streamlit session: {e}")
        api_key = None
        access_token = None
    
    # Fallback to environment variables
    if not api_key:
        api_key = os.getenv("KITE_API_KEY")
    if not access_token:
        access_token = os.getenv("KITE_ACCESS_TOKEN")
    
    # Last resort: ask user for input
    if not api_key:
        api_key = input("Enter your Kite API Key: ").strip()
    
    if not access_token:
        access_token = input("Enter your Kite Access Token: ").strip()
    
    if not api_key or not access_token:
        print("❌ API Key and Access Token are required")
        return None
    
    try:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        
        # Test the connection
        profile = kite.profile()
        print(f"✅ Connected to Kite API as: {profile.get('user_name', 'Unknown')}")
        
        return kite
    except Exception as e:
        print(f"❌ Failed to connect to Kite API: {e}")
        return None


def list_futures_instruments(kite: KiteConnect) -> List[Dict]:
    """
    List all futures instruments from Kite API
    
    Args:
        kite: Authenticated KiteConnect instance
    
    Returns:
        List of futures instrument dictionaries
    """
    try:
        print("\n📡 Fetching all instruments from Kite API...")
        
        # Get all instruments
        instruments = kite.instruments()
        
        print(f"✅ Retrieved {len(instruments)} total instruments")
        
        # Filter for futures
        futures_instruments = []
        
        for instrument in instruments:
            # Check if it's a futures instrument
            if (instrument.get('instrument_type') == 'FUT' or 
                'FUT' in instrument.get('tradingsymbol', '')):
                futures_instruments.append(instrument)
        
        print(f"🎯 Found {len(futures_instruments)} futures instruments")
        
        return futures_instruments
        
    except Exception as e:
        print(f"❌ Error fetching instruments: {e}")
        return []


def filter_nifty_futures(futures_instruments: List[Dict]) -> List[Dict]:
    """
    Filter futures instruments to get only NIFTY futures
    
    Args:
        futures_instruments: List of all futures instruments
    
    Returns:
        List of NIFTY futures instruments
    """
    nifty_futures = []
    
    for instrument in futures_instruments:
        symbol = instrument.get('name', '').upper()
        tradingsymbol = instrument.get('tradingsymbol', '').upper()
        
        # Filter for NIFTY futures (main index, not sectoral)
        if (symbol == 'NIFTY' or 
            (tradingsymbol.startswith('NIFTY') and 
             not any(sector in tradingsymbol for sector in [
                 'BANK', 'IT', 'PHARMA', 'AUTO', 'FMCG', 'METAL', 
                 'REALTY', 'ENERGY', 'INFRA', 'MIDCAP', 'SMLCAP', 'NEXT'
             ]))):
            nifty_futures.append(instrument)
    
    return nifty_futures


def display_futures_summary(futures_instruments: List[Dict]):
    """
    Display a summary of futures instruments
    
    Args:
        futures_instruments: List of futures instruments
    """
    if not futures_instruments:
        print("❌ No futures instruments found")
        return
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(futures_instruments)
    
    print("\n" + "="*80)
    print("FUTURES INSTRUMENTS SUMMARY")
    print("="*80)
    
    # Basic stats
    print(f"\n📊 Total Instruments: {len(df)}")
    
    # Group by exchange
    if 'exchange' in df.columns:
        exchange_counts = df['exchange'].value_counts()
        print(f"\n📈 By Exchange:")
        for exchange, count in exchange_counts.items():
            print(f"   {exchange}: {count}")
    
    # Group by underlying
    if 'name' in df.columns:
        underlying_counts = df['name'].value_counts().head(10)
        print(f"\n🎯 Top 10 Underlying Assets:")
        for underlying, count in underlying_counts.items():
            print(f"   {underlying}: {count}")
    
    # Group by expiry
    if 'expiry' in df.columns:
        # Convert expiry to datetime for sorting
        df['expiry_date'] = pd.to_datetime(df['expiry'], errors='coerce')
        current_expiries = df[df['expiry_date'] >= datetime.now()]
        
        if not current_expiries.empty:
            expiry_counts = current_expiries['expiry'].value_counts().head(5)
            print(f"\n📅 Next 5 Expiry Dates:")
            for expiry, count in expiry_counts.items():
                print(f"   {expiry}: {count} instruments")
    
    print("\n" + "="*80)


def display_nifty_futures_detail(nifty_futures: List[Dict]):
    """
    Display detailed information about NIFTY futures
    
    Args:
        nifty_futures: List of NIFTY futures instruments
    """
    if not nifty_futures:
        print("❌ No NIFTY futures found")
        return
    
    print("\n" + "="*80)
    print("NIFTY FUTURES DETAILED LIST")
    print("="*80)
    
    # Sort by expiry date
    df = pd.DataFrame(nifty_futures)
    
    if 'expiry' in df.columns:
        df['expiry_date'] = pd.to_datetime(df['expiry'], errors='coerce')
        df = df.sort_values('expiry_date')
    
    print(f"\n{'Instrument Token':<15} {'Trading Symbol':<20} {'Expiry':<12} {'Exchange':<8} {'Lot Size':<8}")
    print("-" * 80)
    
    for _, instrument in df.iterrows():
        token = instrument.get('instrument_token', 'N/A')
        symbol = instrument.get('tradingsymbol', 'N/A')
        expiry = instrument.get('expiry', 'N/A')
        exchange = instrument.get('exchange', 'N/A')
        lot_size = instrument.get('lot_size', 'N/A')
        
        print(f"{token:<15} {symbol:<20} {expiry:<12} {exchange:<8} {lot_size:<8}")
    
    print("\n" + "="*80)


def save_instruments_to_csv(instruments: List[Dict], filename: str = "kite_futures_instruments.csv"):
    """
    Save instruments data to CSV file
    
    Args:
        instruments: List of instrument dictionaries
        filename: Output filename
    """
    if not instruments:
        print("❌ No instruments to save")
        return
    
    try:
        df = pd.DataFrame(instruments)
        df.to_csv(filename, index=False)
        print(f"💾 Saved {len(instruments)} instruments to {filename}")
    except Exception as e:
        print(f"❌ Error saving to CSV: {e}")


def main():
    """
    Main function to test Kite API instruments fetching
    """
    print("🚀 Kite API Futures Instruments Test")
    print("="*50)
    
    # Step 1: Get authenticated Kite client
    kite = get_kite_client()
    if not kite:
        return
    
    # Step 2: List all futures instruments
    futures_instruments = list_futures_instruments(kite)
    if not futures_instruments:
        return
    
    # Step 3: Display summary
    display_futures_summary(futures_instruments)
    
    # Step 4: Filter and display NIFTY futures
    nifty_futures = filter_nifty_futures(futures_instruments)
    display_nifty_futures_detail(nifty_futures)
    
    # Step 5: Save to CSV
    choice = input("\n💾 Save all futures instruments to CSV? (y/n): ").strip().lower()
    if choice in ['y', 'yes']:
        save_instruments_to_csv(futures_instruments)
    
    choice = input("💾 Save NIFTY futures to CSV? (y/n): ").strip().lower()
    if choice in ['y', 'yes']:
        save_instruments_to_csv(nifty_futures, "nifty_futures_instruments.csv")
    
    # Step 6: Test historical data fetching for one instrument
    if nifty_futures:
        print("\n🧪 Testing Historical Data Fetch for one NIFTY Future...")
        test_instrument = nifty_futures[0]
        test_historical_data(kite, test_instrument)
    
    print("\n✅ Test completed!")


def test_historical_data(kite: KiteConnect, instrument: Dict):
    """
    Test fetching historical data for a specific instrument
    
    Args:
        kite: Authenticated KiteConnect instance
        instrument: Instrument dictionary
    """
    try:
        token = instrument.get('instrument_token')
        symbol = instrument.get('tradingsymbol')
        
        print(f"\n📊 Testing historical data for {symbol} (Token: {token})")
        
        # Get last 5 days of data
        from datetime import datetime, timedelta
        
        to_date = datetime.now()
        from_date = to_date - timedelta(days=5)
        
        # Fetch historical data
        historical_data = kite.historical_data(
            instrument_token=token,
            from_date=from_date,
            to_date=to_date,
            interval="day"
        )
        
        if historical_data:
            print(f"✅ Retrieved {len(historical_data)} days of data")
            print("\nSample data:")
            for i, candle in enumerate(historical_data[-3:]):  # Show last 3 days
                date = candle.get('date', 'N/A')
                open_price = candle.get('open', 'N/A')
                high = candle.get('high', 'N/A')
                low = candle.get('low', 'N/A')
                close = candle.get('close', 'N/A')
                volume = candle.get('volume', 'N/A')
                oi = candle.get('oi', 'N/A')
                
                print(f"  {date}: O={open_price}, H={high}, L={low}, C={close}, Vol={volume}, OI={oi}")
        else:
            print("⚠️ No historical data retrieved")
            
    except Exception as e:
        print(f"❌ Error fetching historical data: {e}")


if __name__ == "__main__":
    main()