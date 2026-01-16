"""
Simple NIFTY Overview tab - loads data directly from NSE API
"""

import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import time


# NSE API endpoint
NSE_OPTIONS_API = "https://www.nseindia.com/api/historicalOR/foCPV"

# Headers to mimic browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Referer': 'https://www.nseindia.com/report-detail/eq_security'
}


def get_nse_session():
    """Create session with NSE cookies."""
    session = requests.Session()
    session.headers.update(HEADERS)
    try:
        session.get('https://www.nseindia.com', timeout=10)
        time.sleep(1)
        return session
    except Exception as e:
        st.error(f"❌ Error creating NSE session: {e}")
        return None


def get_active_expiries(months_ahead=3):
    """Get next N months of NIFTY expiries (last Tuesday)."""
    expiries = []
    current_date = datetime.now()
    
    for i in range(months_ahead + 1):
        target_month = current_date.month + i
        target_year = current_date.year
        
        while target_month > 12:
            target_month -= 12
            target_year += 1
        
        # Find last Tuesday
        if target_month == 12:
            last_day = 31
        else:
            next_month = datetime(target_year, target_month + 1 if target_month < 12 else 1, 1)
            last_day = (next_month - timedelta(days=1)).day
        
        for day in range(last_day, 0, -1):
            date = datetime(target_year, target_month, day)
            if date.weekday() == 1:  # Tuesday
                expiry_date = date
                if expiry_date.month == 3 and expiry_date.day == 31:
                    expiry_date -= timedelta(days=1)
                if expiry_date >= current_date.replace(hour=0, minute=0, second=0, microsecond=0):
                    expiries.append({
                        'date': expiry_date.strftime('%d-%b-%Y').upper(),
                        'datetime': expiry_date
                    })
                break
    
    return expiries


def fetch_options_data(session, from_date, to_date, expiry_date, option_type='CE'):
    """Fetch options data from NSE API."""
    try:
        expiry_dt = datetime.strptime(expiry_date, '%d-%b-%Y')
        year = expiry_dt.year
        
        params = {
            'from': from_date,
            'to': to_date,
            'instrumentType': 'OPTIDX',
            'symbol': 'NIFTY',
            'year': str(year),
            'expiryDate': expiry_date,
            'optionType': option_type
        }
        
        # Build URL for debugging
        url = NSE_OPTIONS_API + '?' + '&'.join([f"{k}={v}" for k, v in params.items()])
        st.info(f"📡 API URL: {url}")
        
        response = session.get(NSE_OPTIONS_API, params=params, timeout=30)
        
        st.write(f"Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            st.write(f"Response keys: {list(data.keys())}")
            
            if 'data' in data and len(data['data']) > 0:
                df = pd.DataFrame(data['data'])
                df['option_type'] = option_type
                df['expiry'] = expiry_date
                return df
            else:
                st.warning(f"No data in response for {option_type}")
                st.json(data)
        else:
            st.error(f"HTTP {response.status_code}: {response.text[:500]}")
        return None
            
    except Exception as e:
        st.error(f"Error fetching {option_type} data: {e}")
        return None


def render_nifty_overview_tab():
    """Render NIFTY Overview tab with live NSE data."""
    
    st.header("📈 NIFTY Overview - Live NSE Data")

    
    # Hardcoded active expiries
    active_expiries = [
        "28-NOV-2024",
        "05-DEC-2024",
        "12-DEC-2024",
        "19-DEC-2024",
        "26-DEC-2024",
        "02-JAN-2025",
        "09-JAN-2025",
        "16-JAN-2025",
        "23-JAN-2025",
        "30-JAN-2025"
    ]
    
    st.success(f"📅 {len(active_expiries)} active expiries available")
    
    # Expiry selector
    selected_expiry = st.sidebar.selectbox(
        "Select Expiry:",
        active_expiries
    )
    
    # Date range
    to_date = datetime.now()
    from_date = to_date - timedelta(days=90)
    from_date_str = from_date.strftime('%d-%m-%Y')
    to_date_str = to_date.strftime('%d-%m-%Y')
    
    # Load button
    if st.sidebar.button("📊 Load Data", type="primary"):
        with st.spinner(f"� Fetching data for {selected_expiry}..."):
            session = get_nse_session()
            
            if not session:
                st.error("❌ Failed to create NSE session")
                return
            
            # Fetch CE data
            time.sleep(2)
            ce_df = fetch_options_data(session, from_date_str, to_date_str, selected_expiry, 'CE')
            
            # Fetch PE data
            time.sleep(2)
            pe_df = fetch_options_data(session, from_date_str, to_date_str, selected_expiry, 'PE')
            
            # Combine
            dfs = []
            if ce_df is not None:
                dfs.append(ce_df)
            if pe_df is not None:
                dfs.append(pe_df)
            
            if not dfs:
                st.error("❌ No data received from NSE")
                return
            
            combined_df = pd.concat(dfs, ignore_index=True)
            st.session_state['nse_data'] = combined_df
            st.session_state['selected_expiry'] = selected_expiry
            st.success(f"✅ Loaded {len(combined_df):,} records")
    
    # Show data if loaded
    if 'nse_data' in st.session_state:
        df = st.session_state['nse_data']
        
        st.markdown("### 📊 Raw Data")
        
        # Show columns
        with st.expander("📋 Columns"):
            st.write(list(df.columns))
        
        # Show data
        st.dataframe(df.head(100), use_container_width=True)
        
        # Stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            if 'option_type' in df.columns:
                ce_count = len(df[df['option_type'] == 'CE'])
                st.metric("CE Options", f"{ce_count:,}")
        with col3:
            if 'option_type' in df.columns:
                pe_count = len(df[df['option_type'] == 'PE'])
                st.metric("PE Options", f"{pe_count:,}")
