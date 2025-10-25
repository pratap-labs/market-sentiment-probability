import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
from datetime import datetime, timedelta
import sys
import os   
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.models import supabase
from utils.nse_fetcher import NSEDataFetcher

def get_monthly_expiry_dates(start_date=None, months_back=3):
    if start_date is None:
        start_date = datetime.now()
    
    all_expiry_dates = [
        datetime(2025, 1, 30), 
        datetime(2025, 2, 27),
        datetime(2025, 3, 27),
        datetime(2025, 4, 24),
        datetime(2025, 5, 29),
        datetime(2025, 6, 26),
        datetime(2025, 7, 31),
        datetime(2025, 8, 28),
        datetime(2025, 9, 30),
        datetime(2025, 10, 28),
        datetime(2025, 11, 25),
        datetime(2025, 12, 30)
    ]
    
    return all_expiry_dates


def fetch_and_store_futures_data(symbol, expiry_date):
    try:
        fetcher = NSEDataFetcher()
        to_date_dt = expiry_date
        from_date_dt = expiry_date - timedelta(days=90)
        
        from_date = from_date_dt.strftime('%d-%m-%Y')
        to_date = to_date_dt.strftime('%d-%m-%Y')
        
        df = fetcher.fetch_and_parse_futures(from_date, to_date, symbol, expiry_date)
        
        if df.empty:
            return False, "No data retrieved from NSE API"
        
        records = df.to_dict('records')
        
        for record in records:
            if isinstance(record.get('date'), datetime):
                record['date'] = record['date'].isoformat()
            if isinstance(record.get('expiry_date'), datetime):
                record['expiry_date'] = record['expiry_date'].isoformat()
            if isinstance(record.get('timestamp'), datetime):
                record['timestamp'] = record['timestamp'].isoformat()
        
        # Upsert without specifying ON CONFLICT here to avoid DB errors
        # if the matching unique constraint/index does not exist.
        result = supabase.table('futures_data').upsert(records).execute()

        if result:
            return True, f"Successfully synced {len(records)} records"
        else:
            return False, "Failed to sync data to Supabase"

    except Exception as e:
        return False, f"Error: {str(e)}"


def display_table_data(table_name):
    st.subheader(f"📋 {table_name}")
    
    try:
        data = supabase.table('futures_data').select('*', count='exact').order('date', desc=True).limit(10).execute()
        
        if data.data:
            df = pd.DataFrame(data.data)
            total_count = data.count
            
            st.info(f"Total records in database: **{total_count:,}** | Showing latest 10")
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.warning(f"No data found in {table_name}")
            
    except Exception as e:
        st.error(f"Error loading {table_name}: {str(e)}")


def plot_fii_futures_chart(symbol='NIFTY'):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)  # Fetch wider range to cover all expiries
        
        # Fetch all data for the symbol
        result = supabase.table('futures_data')\
            .select('date, expiry_date, underlying_value, open_interest, change_in_oi')\
            .eq('symbol', symbol)\
            .order('date', desc=False)\
            .execute()
        
        if not result.data:
            st.warning("No data available for the selected period")
            return
        
        df = pd.DataFrame(result.data)
        df['date'] = pd.to_datetime(df['date'])
        df['expiry_date'] = pd.to_datetime(df['expiry_date'])
        
        # Get unique expiry dates
        expiries = sorted(df['expiry_date'].unique())

        print(expiries)
        
        if len(expiries) == 0:
            st.warning("No expiry dates found")
            return
        
        # Plot each expiry separately
        for expiry in expiries:
            expiry_df = df[df['expiry_date'] == expiry].copy()
            
            # Filter last 60 days from expiry
            expiry_start = expiry - timedelta(days=60)
            expiry_df = expiry_df[(expiry_df['date'] >= expiry_start) & (expiry_df['date'] <= expiry)]
            
            if expiry_df.empty:
                continue
            
            # Aggregate daily data
            daily_data = expiry_df.groupby('date').agg({
                'underlying_value': 'last',
                'change_in_oi': 'sum'
            }).reset_index()
            
            # Create subplot
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # NIFTY price line
            fig.add_trace(
                go.Scatter(
                    x=daily_data['date'], y=daily_data['underlying_value'],
                    name=symbol, line=dict(color='#9333EA', width=2), mode='lines'
                ), secondary_y=False
            )
            
            # Change in OI bars
            colors = ['#3B82F6' if x > 0 else '#EF4444' for x in daily_data['change_in_oi']]
            fig.add_trace(
                go.Bar(
                    x=daily_data['date'], y=daily_data['change_in_oi'],
                    name='Change in OI', marker=dict(color=colors),
                    showlegend=False
                ), secondary_y=True
            )
            
            # Layout
            expiry_str = expiry.strftime('%d-%b-%Y')
            fig.update_layout(
                title=dict(
                    text=f'{symbol} Futures - Expiry: {expiry_str}',
                    font=dict(size=18, color='white')
                ),
                template='plotly_dark',
                height=400,
                hovermode='x unified',
                plot_bgcolor='#1E1E1E',
                paper_bgcolor='#0E1117',
                font=dict(color='white'),
                margin=dict(l=50, r=50, t=60, b=40),
                xaxis=dict(showgrid=True, gridcolor='#333333', title='')
            )
            
            # Fixed y-axis range for NIFTY
            fig.update_yaxes(
                title_text=symbol,
                secondary_y=False,
                showgrid=True,
                gridcolor='#333333',
                title_font=dict(color='#9333EA'),
            )
            
            fig.update_yaxes(
                title_text="Change in OI",
                secondary_y=True,
                showgrid=False,
                title_font=dict(color='#3B82F6')
            )
            
            # Display metrics and chart side by side
            col1, col2 = st.columns([10, 1])
            
            with col1:
                st.plotly_chart(fig, use_container_width=True)
        
            
            st.divider()
            
    except Exception as e:
        st.error(f"Error plotting charts: {str(e)}")


def render():
    st.title("📥 Data Loader")
    st.markdown("View existing data and load new futures data from NSE")
    st.set_page_config(layout="wide")
    # increase width of each tab

    
    tab1, tab2, tab3 = st.tabs(["Futures Data", "FII Futures Chart", "Load New Data"])
    
    with tab1:
        display_table_data("Futures Data")
    
    with tab2:
        st.subheader("📊 FII Futures Analysis")
        col1, col2 = st.columns([1, 3])
        with col1:
            symbol = st.selectbox("Symbol", ["NIFTY", "BANKNIFTY", "FINNIFTY"], key="chart_symbol")
        plot_fii_futures_chart(symbol)
    
    with tab3:
        st.subheader("🔄 Load New Data")
        col1, col2 = st.columns(2)
        
        with col1:
            symbol = st.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY", "FINNIFTY"], index=0)
        
        with col2:
            expiry_dates = get_monthly_expiry_dates(months_back=3)
            expiry_options = [date.strftime('%d-%b-%Y') for date in expiry_dates]
            selected_expiry_str = st.selectbox("Select Expiry Date", expiry_options, index=0)
        
        selected_expiry = datetime.strptime(selected_expiry_str, '%d-%b-%Y')
        st.info(f"📅 Data will be fetched from {(selected_expiry - timedelta(days=90)).strftime('%d-%b-%Y')} to {selected_expiry_str}")
        
        if st.button("🚀 Fetch & Load Data", type="primary", use_container_width=True):
            with st.spinner(f"Fetching {symbol} data for {selected_expiry_str}..."):
                success, message = fetch_and_store_futures_data(symbol, selected_expiry)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)


if __name__ == "__main__":
    render()