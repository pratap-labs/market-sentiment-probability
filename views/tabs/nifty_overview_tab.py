"""
NIFTY Overview tab for predicting future movements and analyzing max pain
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import os
from typing import Dict, List, Tuple

import sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import the data loader from derivatives tab
from views.tabs.derivatives_data_tab import load_from_cache


def load_options_data() -> pd.DataFrame:
    """
    Load options data using cache from derivatives data tab
    
    Returns:
        Combined DataFrame with all options data
    """
    try:
        # Load from cache
        ce_df = load_from_cache("nifty_options_ce")
        pe_df = load_from_cache("nifty_options_pe")
        
        if ce_df.empty and pe_df.empty:
            st.warning("⚠️ No options data in cache. Please fetch data from the Derivatives Data tab first.")
            return pd.DataFrame()
        
        # Combine CE and PE data
        df = pd.concat([ce_df, pe_df], ignore_index=True) if not ce_df.empty and not pe_df.empty else (ce_df if not ce_df.empty else pe_df)
        
        return df
    
    except Exception as e:
        st.error(f"Error loading options data: {str(e)}")
        return pd.DataFrame()


def get_expiry_dates_from_data(df: pd.DataFrame) -> List[str]:
    """
    Extract unique expiry dates from options data
    
    Args:
        df: Options DataFrame
    
    Returns:
        List of expiry dates
    """
    if 'expiry' in df.columns:
        # Convert to datetime and sort
        expiry_dates = pd.to_datetime(df['expiry']).dt.strftime('%Y-%m-%d').unique()
        return sorted(expiry_dates)
    
    return []


def is_expiry_too_old(expiry_date: str, days_threshold: int = 30) -> bool:
    """
    Check if an expiry date is too old (more than threshold days in the past)
    
    Args:
        expiry_date: Expiry date string
        days_threshold: Number of days threshold
    
    Returns:
        True if expiry is too old
    """
    try:
        expiry_dt = pd.to_datetime(expiry_date)
        today = datetime.now()
        days_diff = (today - expiry_dt).days
        return days_diff > days_threshold
    except:
        # If we can't parse the date, don't exclude it
        return False


def calculate_max_pain(df: pd.DataFrame, expiry_date: str = None) -> Dict:
    """
    Calculate max pain for a given options DataFrame
    
    Args:
        df: DataFrame with options data
        expiry_date: Optional specific expiry date to filter
    
    Returns:
        Dict with max pain info
    """
    if df.empty:
        return {"max_pain": None, "strikes": [], "total_pain": []}
    
    try:
        # Filter by expiry if specified
        if expiry_date:
            expiry_dt = pd.to_datetime(expiry_date)
            df = df[df['expiry'] == expiry_dt].copy()
        
        if df.empty:
            return {"max_pain": None, "strikes": [], "total_pain": []}
        
        # Get unique strike prices
        if 'strike_price' not in df.columns:
            return {"max_pain": None, "strikes": [], "total_pain": []}
        
        strikes = sorted(df['strike_price'].dropna().unique())
        
        if len(strikes) == 0:
            return {"max_pain": None, "strikes": [], "total_pain": []}
        
        max_pain_data = []
        
        for strike in strikes:
            total_pain = 0
            
            # Calculate pain for this strike price
            for _, row in df.iterrows():
                oi = row.get('open_int', 0)
                strike_price = row.get('strike_price', 0)
                option_type = str(row.get('option_type', '')).upper()
                
                if pd.notna(oi) and oi > 0 and pd.notna(strike_price):
                    if option_type == 'CE' and strike > strike_price:
                        # ITM Call - pain for option writers
                        total_pain += (strike - strike_price) * oi
                    elif option_type == 'PE' and strike < strike_price:
                        # ITM Put - pain for option writers  
                        total_pain += (strike_price - strike) * oi
            
            max_pain_data.append({
                'strike': strike,
                'total_pain': total_pain
            })
        
        # Convert to DataFrame for easier handling
        pain_df = pd.DataFrame(max_pain_data)
        
        if pain_df.empty:
            return {"max_pain": None, "strikes": [], "total_pain": []}
        
        # Find max pain (minimum total pain)
        max_pain_strike = pain_df.loc[pain_df['total_pain'].idxmin(), 'strike']
        
        return {
            "max_pain": max_pain_strike,
            "strikes": pain_df['strike'].tolist(),
            "total_pain": pain_df['total_pain'].tolist(),
            "pain_data": pain_df
        }
    
    except Exception as e:
        st.warning(f"Error calculating max pain: {str(e)}")
        return {"max_pain": None, "strikes": [], "total_pain": []}


def analyze_max_pain_trends(df):
    """Analyze max pain trends across multiple expiry dates."""
    max_pain_results = []
    
    expiry_dates = get_expiry_dates_from_data(df)
    
    for expiry_date in expiry_dates:
        # Skip if expiry is more than 30 days in the past
        if is_expiry_too_old(expiry_date, days_threshold=30):
            continue
        
        # Filter data for this expiry
        expiry_data = df[df['expiry'] == expiry_date]
        
        if len(expiry_data) == 0:
            continue
        
        # Calculate max pain
        max_pain_result = calculate_max_pain(expiry_data)
        max_pain_value = max_pain_result.get('max_pain') if max_pain_result else None
        
        if max_pain_value is not None and max_pain_value > 0:
            # Get additional metrics
            latest_date = expiry_data['date'].max() if 'date' in expiry_data.columns else None
            underlying_value = expiry_data['underlying_value'].iloc[0] if 'underlying_value' in expiry_data.columns and not expiry_data['underlying_value'].isna().all() else None
            total_oi = expiry_data['open_int'].sum() if 'open_int' in expiry_data.columns else 0
            
            max_pain_results.append({
                'expiry_date': expiry_date,
                'max_pain': max_pain_value,
                'underlying_value': underlying_value,
                'total_oi': total_oi,
                'latest_date': latest_date
            })
    
    # Sort by expiry date
    max_pain_results.sort(key=lambda x: pd.to_datetime(x['expiry_date']))
    return max_pain_results


def plot_max_pain_trends(max_pain_trends: list) -> go.Figure:
    """
    Plot max pain trends over time for each expiry
    
    Args:
        max_pain_trends: List with max pain trend data
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    if not max_pain_trends:
        fig.add_annotation(
            text="No max pain data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Convert list data to plot format
    expiry_dates = [item['expiry_date'] for item in max_pain_trends]
    max_pain_values = [item['max_pain'] for item in max_pain_trends]
    underlying_values = [item.get('underlying_value', 0) for item in max_pain_trends]
    
    # Create bar chart for max pain vs underlying
    fig = go.Figure()
    
    # Add max pain bars
    fig.add_trace(go.Bar(
        x=expiry_dates,
        y=max_pain_values,
        name='Max Pain Level',
        marker_color='lightblue',
        yaxis='y',
        hovertemplate='<b>Max Pain</b><br>Expiry: %{x}<br>Level: ₹%{y:,.0f}<extra></extra>'
    ))
    
    # Add underlying value line if available
    if any(val and val > 0 for val in underlying_values):
        fig.add_trace(go.Scatter(
            x=expiry_dates,
            y=underlying_values,
            mode='lines+markers',
            name='Underlying Value',
            line=dict(color='red', width=2),
            marker=dict(size=6),
            yaxis='y',
            hovertemplate='<b>Underlying</b><br>Expiry: %{x}<br>Level: ₹%{y:,.0f}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title='Max Pain Levels vs Underlying Value',
        xaxis_title='Expiry Date',
        yaxis_title='Price Level (₹)',
        hovermode='x',
        showlegend=True,
        height=400,
        xaxis=dict(tickangle=45)
    )
    
    return fig


def plot_current_max_pain_distribution(max_pain_trends: list) -> go.Figure:
    """
    Plot current max pain distribution across different expiries
    
    Args:
        max_pain_trends: List with max pain trend data
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    if not max_pain_trends:
        return fig
    
    expiries = []
    max_pains = []
    days_to_expiry = []
    
    for item in max_pain_trends:
        expiry_date = item['expiry_date']
        max_pain = item['max_pain']
        
        try:
            expiry_dt = pd.to_datetime(expiry_date)
            days_remaining = (expiry_dt - datetime.now()).days
        except:
            days_remaining = 0
        
        expiries.append(expiry_date)
        max_pains.append(max_pain)
        days_to_expiry.append(days_remaining)
    
    if expiries:
        # Create bar chart
        fig.add_trace(go.Bar(
            x=expiries,
            y=max_pains,
            text=[f'{mp:,.0f}' for mp in max_pains],
            textposition='auto',
            name='Max Pain Strike',
            hovertemplate='<b>%{x}</b><br>Max Pain: ₹%{y:,.0f}<br>Days to Expiry: %{customdata}<extra></extra>',
            customdata=days_to_expiry
        ))
    
    fig.update_layout(
        title='Current Max Pain by Expiry',
        xaxis_title='Expiry Date',
        yaxis_title='Max Pain Strike Price (₹)',
        height=400,
        xaxis={'categoryorder': 'category ascending'}
    )
    
    return fig


def render_nifty_overview_tab():
    """Render the NIFTY Overview tab."""
    
    st.header("📈 NIFTY Overview & Future Movement Prediction")
    st.markdown("Analysis of max pain trends and prediction of NIFTY movements based on options data")
    
    # Load options data using the same loader as Data Hub
    with st.spinner("📊 Loading options data..."):
        df = load_options_data()
    
    if df.empty:
        st.error("❌ No options data available. Please ensure options data files are present in `database/options_data/`")
        st.info("""
        **Expected file structure:**
        ```
        database/options_data/
        ├── OPTIDX_NIFTY_CE_01-Dec-2024_TO_28-Feb-2025.csv
        ├── OPTIDX_NIFTY_PE_01-Dec-2024_TO_28-Feb-2025.csv
        ├── OPTIDX_NIFTY_CE_01-Jan-2025_TO_31-Mar-2025.csv
        └── OPTIDX_NIFTY_PE_01-Jan-2025_TO_31-Mar-2025.csv
        ```
        """)
        return
    
    st.success(f"✅ Loaded {len(df)} options records from CSV files")
    
    # Show data summary
    with st.expander("📋 Data Summary"):
        expiry_dates = get_expiry_dates_from_data(df)
        ce_count = len(df[df['option_type'] == 'CE']) if 'option_type' in df.columns else 0
        pe_count = len(df[df['option_type'] == 'PE']) if 'option_type' in df.columns else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Unique Expiries", len(expiry_dates))
        with col3:
            st.metric("CE Options", f"{ce_count:,}")
        with col4:
            st.metric("PE Options", f"{pe_count:,}")
        
        if expiry_dates:
            st.write("**Expiry Dates:**", ", ".join(expiry_dates[:10]) + ("..." if len(expiry_dates) > 10 else ""))
        
        # Show data structure
        if len(df) > 0:
            st.markdown("**Sample Data Structure:**")
            sample_columns = [col for col in ['date', 'expiry', 'option_type', 'strike_price', 'open_int', 'underlying_value'] if col in df.columns]
            if sample_columns:
                st.dataframe(df[sample_columns].head(3), use_container_width=True)
    
    # Analyze max pain trends
    with st.spinner("🔍 Analyzing max pain trends..."):
        max_pain_trends = analyze_max_pain_trends(df)
    
    if not max_pain_trends:
        st.warning("⚠️ No max pain trends could be calculated from the available data")
        st.info("This could happen if:")
        st.write("- No expiry dates found in the data")
        st.write("- All expiries are more than 30 days in the past") 
        st.write("- Missing required columns: 'expiry', 'option_type', 'strike_price', 'open_int'")
        return
    
    st.success(f"✅ Analyzed max pain for {len(max_pain_trends)} expiry dates")
    
    # Create analysis tabs
    analysis_tabs = st.tabs([
        "📊 Max Pain Trends",
        "📈 Current Distribution", 
        "🎯 Detailed Analysis",
        "🔮 Predictions"
    ])
    
    # ========== MAX PAIN TRENDS TAB ==========
    with analysis_tabs[0]:
        st.markdown("### 📊 Max Pain Trends Over Time")
        st.info("Track how max pain levels change over the last 30 days for each expiry")
        
        # Plot max pain trends
        trends_fig = plot_max_pain_trends(max_pain_trends)
        st.plotly_chart(trends_fig, use_container_width=True)
        
        # Summary table
        st.markdown("#### 📋 Max Pain Summary")
        
        summary_data = []
        for item in max_pain_trends:
            expiry_date = item['expiry_date']
            max_pain = item['max_pain']
            underlying_value = item.get('underlying_value', 0)
            total_oi = item.get('total_oi', 0)
            
            try:
                expiry_dt = pd.to_datetime(expiry_date)
                days_remaining = (expiry_dt - datetime.now()).days
            except:
                days_remaining = 0
            
            # Calculate difference from underlying
            price_diff = 0
            if underlying_value and underlying_value > 0:
                price_diff = max_pain - underlying_value
            
            summary_data.append({
                'Expiry': expiry_date,
                'Days Remaining': days_remaining,
                'Current Max Pain': f"₹{max_pain:,.0f}",
                'Current NIFTY': f"₹{underlying_value:,.0f}" if underlying_value else "N/A",
                'Difference': f"₹{price_diff:+,.0f}" if underlying_value else "N/A",
                'Total OI': f"{total_oi:,.0f}" if total_oi else "N/A"
            })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(
                summary_df.style.format({
                    'Days Remaining': '{:.0f}'
                }),
                use_container_width=True
            )
    
    # ========== CURRENT DISTRIBUTION TAB ==========
    with analysis_tabs[1]:
        st.markdown("### 📈 Current Max Pain Distribution")
        st.info("Compare current max pain levels across different expiry dates")
        
        # Plot current max pain distribution
        distribution_fig = plot_current_max_pain_distribution(max_pain_trends)
        st.plotly_chart(distribution_fig, use_container_width=True)
        
        # Key insights
        st.markdown("#### 🔍 Key Insights")
        
        if max_pain_trends:
            # Get all current max pain values
            current_max_pains = []
            for expiry_date, trend_data in max_pain_trends.items():
                if trend_data['data_points']:
                    current_max_pains.append(trend_data['data_points'][-1]['max_pain'])
            
            if current_max_pains:
                avg_max_pain = np.mean(current_max_pains)
                max_max_pain = max(current_max_pains)
                min_max_pain = min(current_max_pains)
                pain_range = max_max_pain - min_max_pain
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Average Max Pain", f"₹{avg_max_pain:,.0f}")
                
                with col2:
                    st.metric("Highest Max Pain", f"₹{max_max_pain:,.0f}")
                
                with col3:
                    st.metric("Lowest Max Pain", f"₹{min_max_pain:,.0f}")
                
                with col4:
                    st.metric("Range", f"₹{pain_range:,.0f}")
                
                # Analysis insights
                if pain_range < 500:
                    st.success("🟢 **Tight Range**: Max pain levels are closely clustered, indicating consensus")
                elif pain_range < 1000:
                    st.warning("🟡 **Moderate Range**: Some divergence in max pain across expiries")
                else:
                    st.error("🔴 **Wide Range**: Significant divergence in max pain levels")
    
    # ========== DETAILED ANALYSIS TAB ==========
    with analysis_tabs[2]:
        st.markdown("### 🎯 Detailed Max Pain Analysis")
        
        # Select expiry for detailed analysis
        expiry_options = [item['expiry_date'] for item in max_pain_trends]
        selected_expiry = st.selectbox(
            "Choose expiry for detailed analysis:",
            expiry_options,
            help="Select an expiry date to see detailed max pain breakdown"
        )
        
        if selected_expiry:
            # Find the selected expiry data
            selected_data = None
            for item in max_pain_trends:
                if item['expiry_date'] == selected_expiry:
                    selected_data = item
                    break
            
            if selected_data:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Max pain info
                    max_pain_value = selected_data['max_pain']
                    underlying_value = selected_data.get('underlying_value', 0)
                    total_oi = selected_data.get('total_oi', 0)
                    
                    try:
                        expiry_dt = pd.to_datetime(selected_expiry)
                        days_remaining = (expiry_dt - datetime.now()).days
                    except:
                        days_remaining = 0
                
                # Get detailed pain data by recalculating for this expiry
                expiry_data = df[df['expiry'] == selected_expiry]
                max_pain_result = calculate_max_pain(expiry_data)
                pain_data = max_pain_result.get('pain_data', pd.DataFrame())
                
                if not pain_data.empty:
                    with col1:
                        st.metric("Max Pain Strike", f"₹{max_pain_value:,.0f}")
                        st.metric("Days to Expiry", f"{days_remaining}")
                        st.metric("Expiry Date", selected_expiry)
                        if underlying_value:
                            diff = max_pain_value - underlying_value
                            st.metric("vs Current NIFTY", f"₹{diff:+,.0f}")
                    
                    with col2:
                        # Pain distribution chart
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=pain_data['strike'],
                            y=pain_data['total_pain'],
                            mode='lines+markers',
                            name='Total Pain',
                            line=dict(color='red', width=2),
                            marker=dict(size=4)
                        ))
                        
                        # Highlight max pain point
                        max_pain_row = pain_data[pain_data['strike'] == max_pain_value]
                        if not max_pain_row.empty:
                            fig.add_trace(go.Scatter(
                                x=[max_pain_value],
                                y=[max_pain_row['total_pain'].iloc[0]],
                                mode='markers',
                                name='Max Pain',
                                marker=dict(color='green', size=12, symbol='diamond')
                            ))
                        
                        fig.update_layout(
                            title=f'Pain Distribution - {selected_expiry}',
                            xaxis_title='Strike Price (₹)',
                            yaxis_title='Total Pain',
                            height=350
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Pain data table
                    st.markdown("#### Pain by Strike Price")
                    
                    # Add percentage columns
                    display_pain_data = pain_data.copy()
                    display_pain_data['pain_millions'] = display_pain_data['total_pain'] / 1_000_000
                    display_pain_data = display_pain_data.sort_values('total_pain')
                    
                    # Show top 20 strikes with lowest pain
                    st.dataframe(
                        display_pain_data.head(20)[['strike', 'pain_millions']].style.format({
                            'strike': '₹{:,.0f}',
                            'pain_millions': '{:.2f}M'
                        }).background_gradient(subset=['pain_millions'], cmap='RdYlGn_r'),
                        use_container_width=True
                    )
                
                else:
                    st.warning("No detailed pain data available for this expiry")
            else:
                st.error("No data available for selected expiry")
    
    # ========== PREDICTIONS TAB ==========
    with analysis_tabs[3]:
        st.markdown("### 🔮 NIFTY Movement Predictions")
        st.info("Based on max pain analysis and options data trends")
        
        # Calculate predictions based on max pain
        if max_pain_trends:
            st.markdown("#### 📊 Max Pain Based Predictions")
            
            predictions = []
            for item in max_pain_trends:
                expiry_date = item['expiry_date']
                max_pain = item['max_pain']
                
                try:
                    expiry_dt = pd.to_datetime(expiry_date)
                    days_remaining = (expiry_dt - datetime.now()).days
                except:
                    days_remaining = 0
                
                # Simple prediction logic (can be enhanced)
                if days_remaining > 0:
                    predictions.append({
                        'expiry': expiry_date,
                        'days_remaining': days_remaining,
                        'max_pain': max_pain,
                        'prediction': f"Expected to trade around ₹{max_pain:,.0f}"
                    })
            
            if predictions:
                # Sort by days remaining
                predictions.sort(key=lambda x: x['days_remaining'])
                
                for pred in predictions[:5]:  # Show next 5 expiries
                    with st.container():
                        col1, col2, col3 = st.columns([1, 1, 2])
                        
                        with col1:
                            st.metric("Expiry", pred['expiry'])
                        
                        with col2:
                            st.metric("Days Left", pred['days_remaining'])
                        
                        with col3:
                            st.info(f"**Max Pain Target**: {pred['prediction']}")
                        
                        st.markdown("---")
            
            st.markdown("#### ⚠️ Important Notes")
            st.warning("""
            **Disclaimer**: These predictions are based solely on max pain analysis and should not be used as the only factor for trading decisions.
            
            **Max Pain Theory**: Suggests that prices tend to move towards the strike price where options writers (sellers) would experience the least loss.
            
            **Limitations**:
            - Market sentiment can override max pain
            - External news and events impact prices
            - Liquidity and volume matter
            - Time decay affects options differently
            
            **Recommendation**: Use this as one factor among many in your analysis.
            """)
        
        else:
            st.warning("No prediction data available")