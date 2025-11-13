"""
Kite Instruments tab for loading NIFTY futures data from Kite API
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import io
import zipfile

try:
    from kiteconnect import KiteConnect
except ImportError:
    KiteConnect = None


def get_kite_client_from_session():
    """
    Get authenticated Kite client from Streamlit session state.
    
    Returns:
        KiteConnect instance if successful, None otherwise
    """
    if not KiteConnect:
        st.error("❌ kiteconnect not installed. Install with: `pip install kiteconnect`")
        return None
    
    try:
        api_key = st.session_state.get("kite_api_key")
        access_token = st.session_state.get("kite_access_token")
        
        if not api_key or not access_token:
            return None
        
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        
        # Test the connection
        profile = kite.profile()
        st.session_state["kite_profile"] = profile
        
        return kite
        
    except Exception as e:
        st.error(f"❌ Failed to connect to Kite API: {e}")
        return None


def fetch_all_instruments(kite) -> List[Dict]:
    """
    Fetch all instruments from Kite API.
    
    Args:
        kite: Authenticated KiteConnect instance
    
    Returns:
        List of all instrument dictionaries
    """
    try:
        instruments = kite.instruments()
        return instruments
    except Exception as e:
        st.error(f"❌ Error fetching instruments: {e}")
        return []


def filter_instruments_by_type(instruments: List[Dict], instrument_type: str) -> List[Dict]:
    """
    Filter instruments by type.
    
    Args:
        instruments: List of all instruments
        instrument_type: 'FUT' for futures, 'CE'/'PE' for options
    
    Returns:
        List of filtered instruments
    """
    if instrument_type in ['CE', 'PE']:
        # Options
        filtered = [inst for inst in instruments if inst.get('instrument_type') in ['CE', 'PE']]
    else:
        # Futures or other types
        filtered = [inst for inst in instruments if inst.get('instrument_type') == instrument_type]
    
    return filtered


def fetch_historical_data_for_futures(kite, futures_list: List[Dict]) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical data for each NIFTY futures contract.
    
    Args:
        kite: KiteConnect instance
        futures_list: List of NIFTY futures instruments
    
    Returns:
        Dict mapping expiry dates to DataFrames with historical data
    """
    historical_data = {}
    
    for future in futures_list:
        try:
            instrument_token = future.get('instrument_token')
            expiry = future.get('expiry', 'Unknown')
            symbol = future.get('tradingsymbol', 'Unknown')
            
            # Get last 30 days of data
            to_date = datetime.now()
            from_date = to_date - timedelta(days=30)
            
            # Fetch OHLC data
            data = kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval="day"
            )
            
            if data:
                df = pd.DataFrame(data)
                df['symbol'] = symbol
                df['expiry'] = expiry
                df['instrument_token'] = instrument_token
                
                # Group by expiry date
                expiry_key = str(expiry)
                if expiry_key not in historical_data:
                    historical_data[expiry_key] = []
                historical_data[expiry_key].append(df)
                
        except Exception as e:
            st.warning(f"Failed to fetch data for {future.get('tradingsymbol', 'Unknown')}: {str(e)}")
            continue
    
    # Combine DataFrames for each expiry
    combined_data = {}
    for expiry, dfs in historical_data.items():
        if dfs:
            combined_data[expiry] = pd.concat(dfs, ignore_index=True)
    
    return combined_data


def generate_csv_downloads(historical_data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    """
    Generate CSV files for each expiry and create download links.
    
    Args:
        historical_data: Dict mapping expiry dates to DataFrames
    
    Returns:
        Dict mapping expiry dates to CSV file paths/content
    """
    csv_files = {}
    
    for expiry, df in historical_data.items():
        if df is not None and len(df) > 0:
            try:
                # Clean expiry date for filename
                clean_expiry = str(expiry).replace('-', '_').replace(':', '_').replace(' ', '_')
                filename = f"NIFTY_Futures_{clean_expiry}.csv"
                
                # Convert to CSV
                csv_content = df.to_csv(index=False)
                csv_files[expiry] = {
                    'filename': filename,
                    'content': csv_content,
                    'dataframe': df
                }
            except Exception as e:
                st.error(f"Error generating CSV for expiry {expiry}: {str(e)}")
                continue
    
    return csv_files


def create_zip_download(csv_files: Dict[str, Dict]) -> bytes:
    """
    Create a ZIP file containing all CSV files.
    
    Args:
        csv_files: Dict with CSV content for each expiry
    
    Returns:
        ZIP file content as bytes
    """
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for expiry, file_info in csv_files.items():
            zip_file.writestr(file_info['filename'], file_info['content'])
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def get_nifty_futures_only(instruments: List[Dict]) -> List[Dict]:
    """
    Filter instruments to get only NIFTY futures (main index).
    
    Args:
        instruments: List of all instruments
    
    Returns:
        List of NIFTY futures instruments sorted by expiry
    """
    nifty_futures = []
    
    for instrument in instruments:
        # Must be futures
        if instrument.get('instrument_type') != 'FUT':
            continue
            
        symbol = instrument.get('name', '').upper()
        tradingsymbol = instrument.get('tradingsymbol', '').upper()
        
        # Must be NIFTY main index (not sectoral)
        if (symbol == 'NIFTY' or 
            (tradingsymbol.startswith('NIFTY') and 
             not any(sector in tradingsymbol for sector in [
                 'BANK', 'IT', 'PHARMA', 'AUTO', 'FMCG', 'METAL', 
                 'REALTY', 'ENERGY', 'INFRA', 'MIDCAP', 'SMLCAP', 'NEXT',
                 'PSU', 'CPSE', 'MNC', 'PVTBANK', 'FINANCIAL'
             ]))):
            nifty_futures.append(instrument)
    
    # Sort by expiry date
    try:
        nifty_futures.sort(key=lambda x: pd.to_datetime(x.get('expiry', '1900-01-01')))
    except:
        pass
    
    return nifty_futures


def test_historical_data(kite, instrument: Dict, days: int = 5):
    """
    Test fetching historical data for a specific instrument.
    
    Args:
        kite: Authenticated KiteConnect instance
        instrument: Instrument dictionary
        days: Number of days of historical data to fetch
    
    Returns:
        DataFrame with historical data or None if failed
    """
    try:
        token = instrument.get('instrument_token')
        
        # Get historical data
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        historical_data = kite.historical_data(
            instrument_token=token,
            from_date=from_date,
            to_date=to_date,
            interval="day"
        )
        
        if historical_data:
            return pd.DataFrame(historical_data)
        else:
            return None
            
    except Exception as e:
        st.error(f"❌ Error fetching historical data: {e}")
        return None


def render_kite_instruments_tab():
    """Render the NIFTY Futures Data tab."""
    
    st.header("� NIFTY Futures Data Export")
    st.markdown("Fetch historical data for all NIFTY futures contracts and download as CSV files (one per expiry)")
    
    # Check if user is logged in
    if not st.session_state.get("kite_access_token"):
        st.warning("⚠️ Not logged in to Kite. Please go to the **Login** tab and sign in first.")
        
        # Add refresh button to check for login
        if st.button("🔄 Refresh Login Status", help="Check if you've logged in via the Login tab"):
            st.rerun()
        return
    
    # Show connection status
    profile = st.session_state.get("kite_profile")
    if profile:
        st.success(f"✅ Connected to Kite API as: **{profile.get('user_name', 'Unknown')}**")
    
    # Settings for data fetch
    st.markdown("### ⚙️ Data Fetch Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        days_back = st.number_input(
            "� Days of Historical Data",
            min_value=1,
            max_value=365,
            value=30,
            help="Number of days of historical data to fetch"
        )
    
    with col2:
        interval = st.selectbox(
            "📊 Data Interval",
            ["day", "minute", "3minute", "5minute", "10minute", "15minute", "30minute", "60minute"],
            help="Time interval for the data"
        )
    
    with col3:
        if st.button("🚀 Fetch NIFTY Futures Data", type="primary"):
            with st.spinner("🔄 Connecting to Kite API..."):
                kite = get_kite_client_from_session()
                
                if not kite:
                    st.error("❌ Failed to connect to Kite API. Please check your login status.")
                    return
            
            with st.spinner("📡 Fetching all instruments..."):
                all_instruments = fetch_all_instruments(kite)
                
                if not all_instruments:
                    st.error("❌ No instruments retrieved from Kite API")
                    return
                
                st.success(f"✅ Retrieved **{len(all_instruments):,}** total instruments")
            
            with st.spinner("🎯 Filtering NIFTY futures..."):
                nifty_futures = get_nifty_futures_only(all_instruments)
                
                if not nifty_futures:
                    st.error("❌ No NIFTY futures found")
                    return
                
                st.success(f"✅ Found **{len(nifty_futures)}** NIFTY futures contracts")
                
                # Display futures summary
                df_futures = pd.DataFrame(nifty_futures)
                st.markdown("#### 📋 NIFTY Futures Contracts Found")
                
                # Show expiry wise breakdown
                if 'expiry' in df_futures.columns:
                    expiry_counts = df_futures['expiry'].value_counts().sort_index()
                    st.markdown("**By Expiry Date:**")
                    for expiry, count in expiry_counts.items():
                        st.write(f"- {expiry}: {count} contract(s)")
                
                # Show table of contracts
                display_cols = ['tradingsymbol', 'expiry', 'lot_size', 'tick_size']
                available_cols = [col for col in display_cols if col in df_futures.columns]
                st.dataframe(df_futures[available_cols], use_container_width=True)
            
            with st.spinner(f"� Fetching {days_back} days of historical data..."):
                # Modify fetch function to accept interval and days
                historical_data = fetch_historical_data_for_futures_custom(
                    kite, nifty_futures, days_back, interval
                )
                
                if not historical_data:
                    st.error("❌ No historical data retrieved")
                    return
                
                st.success(f"✅ Retrieved historical data for **{len(historical_data)}** expiry dates")
            
            with st.spinner("📄 Generating CSV files..."):
                csv_files = generate_csv_downloads(historical_data)
                
                if not csv_files:
                    st.error("❌ Failed to generate CSV files")
                    return
                
                st.success(f"✅ Generated **{len(csv_files)}** CSV files")
                
                # Store in session state for download
                st.session_state["nifty_futures_csv"] = csv_files
                st.session_state["nifty_futures_data"] = historical_data
    
    # Display download section if CSV files are available
    if st.session_state.get("nifty_futures_csv"):
        st.markdown("---")
        st.markdown("### 📥 Download CSV Files")
        
        csv_files = st.session_state["nifty_futures_csv"]
        historical_data = st.session_state["nifty_futures_data"]
        
        # Summary of generated files
        st.markdown("#### 📊 Data Summary")
        
        for expiry, file_info in csv_files.items():
            df = file_info['dataframe']
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Expiry", expiry)
            with col2:
                st.metric("Records", len(df))
            with col3:
                if 'volume' in df.columns:
                    total_volume = df['volume'].sum()
                    st.metric("Total Volume", f"{total_volume:,.0f}")
            with col4:
                # Download button for individual CSV
                st.download_button(
                    label=f"� Download {expiry}",
                    data=file_info['content'],
                    file_name=file_info['filename'],
                    mime="text/csv"
                )
        
        # Download all as ZIP
        st.markdown("#### 📦 Download All Files")
        
        if st.button("🗜️ Generate ZIP File"):
            with st.spinner("📦 Creating ZIP file..."):
                zip_content = create_zip_download(csv_files)
                
                st.download_button(
                    label="📥 Download All CSV Files (ZIP)",
                    data=zip_content,
                    file_name=f"NIFTY_Futures_Data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip"
                )
        
        # Option to clear data
        if st.button("🗑️ Clear Data", help="Clear downloaded data from session"):
            if "nifty_futures_csv" in st.session_state:
                del st.session_state["nifty_futures_csv"]
            if "nifty_futures_data" in st.session_state:
                del st.session_state["nifty_futures_data"]
            st.success("✅ Data cleared from session")
            st.rerun()
    
    # Help section
    st.markdown("---")
    st.markdown("### ❓ How to Use")
    st.markdown("""
    1. **Login**: Ensure you're logged in via the Login tab
    2. **Configure**: Set the number of days and data interval
    3. **Fetch**: Click 'Fetch NIFTY Futures Data' to get all NIFTY futures contracts and their historical data
    4. **Download**: Use individual download buttons or create a ZIP file with all data
    
    **Note**: Each CSV file contains historical OHLC data, volume, and open interest for futures contracts expiring on the same date.
    """)


def fetch_historical_data_for_futures_custom(kite, futures_list: List[Dict], days_back: int, interval: str) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical data for each NIFTY futures contract with custom parameters.
    
    Args:
        kite: KiteConnect instance
        futures_list: List of NIFTY futures instruments
        days_back: Number of days to fetch historical data
        interval: Data interval (day, minute, etc.)
    
    Returns:
        Dict mapping expiry dates to DataFrames with historical data
    """
    historical_data = {}
    
    for future in futures_list:
        try:
            instrument_token = future.get('instrument_token')
            expiry = future.get('expiry', 'Unknown')
            symbol = future.get('tradingsymbol', 'Unknown')
            
            # Get historical data for specified days
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days_back)
            
            # Fetch OHLC data
            data = kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            
            if data:
                df = pd.DataFrame(data)
                df['symbol'] = symbol
                df['expiry'] = expiry
                df['instrument_token'] = instrument_token
                
                # Group by expiry date
                expiry_key = str(expiry)
                if expiry_key not in historical_data:
                    historical_data[expiry_key] = []
                historical_data[expiry_key].append(df)
                
        except Exception as e:
            st.warning(f"Failed to fetch data for {future.get('tradingsymbol', 'Unknown')}: {str(e)}")
            continue
    
    # Combine DataFrames for each expiry
    combined_data = {}
    for expiry, dfs in historical_data.items():
        if dfs:
            combined_data[expiry] = pd.concat(dfs, ignore_index=True)
    
    return combined_data
