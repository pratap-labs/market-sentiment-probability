import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
from io import StringIO

from utils.db_data_processor import DatabaseDataProcessor
from utils.model_trainer import ModelTrainer

# Configure page
st.set_page_config(
    page_title="Market Sentiment Probability",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_and_process_data():
    """Load and process data from database"""
    try:
        processor = DatabaseDataProcessor()
        processed_data = processor.load_and_process_data()
        processor.close()
        return processed_data
    except Exception as e:
        st.error(f"Error loading data from database: {str(e)}")
        st.info("Make sure to run: python scripts/fetch_nse_data.py")
        return None

@st.cache_resource
def train_model(data):
    """Train the sentiment prediction model"""
    try:
        trainer = ModelTrainer()
        model, scaler, feature_cols = trainer.train_model(data)
        return model, scaler, feature_cols
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None

def create_probability_gauge(probability, title="Bullish Probability"):
    """Create a probability gauge using Plotly"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "green" if probability > 0.5 else "red"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 100], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_probability_trend_chart(data):
    """Create historical probability trend chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['date'],
        y=data['bullish_probability'],
        mode='lines+markers',
        name='Bullish Probability',
        line=dict(color='green', width=2),
        marker=dict(size=4)
    ))
    
    # Add horizontal line at 50%
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Neutral (50%)")
    
    fig.update_layout(
        title="Historical Market Sentiment Probability",
        xaxis_title="Date",
        yaxis_title="Bullish Probability",
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        hovermode='x unified',
        height=400
    )
    
    return fig

def create_feature_importance_chart(feature_importance, feature_names):
    """Create feature importance visualization"""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': abs(feature_importance)
    }).sort_values('importance', ascending=True)
    
    fig = px.bar(
        importance_df,
        x='importance',
        y='feature',
        orientation='h',
        title="Feature Importance in Sentiment Prediction",
        labels={'importance': 'Absolute Coefficient Value', 'feature': 'Features'}
    )
    
    fig.update_layout(height=300)
    return fig

def create_oi_analysis_chart(data, expiry_date):
    """
    Create combined chart showing NIFTY price, FII net OI, and daily OI changes
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            'NIFTY Price Movement',
            'Total Futures Open Interest',
            'Daily OI Change'
        ),
        row_heights=[0.33, 0.33, 0.34]
    )
    
    fig.add_trace(
        go.Scatter(
            x=data['date'],
            y=data['underlying_value'],
            mode='lines',
            name='NIFTY Price',
            line=dict(color='blue', width=2),
            hovertemplate='%{x}<br>Price: %{y:,.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data['date'],
            y=data['open_interest'],
            mode='lines',
            name='Total OI',
            line=dict(color='green', width=2),
            fill='tozeroy',
            hovertemplate='%{x}<br>Total OI: %{y:,.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    colors = ['red' if x < 0 else 'green' for x in data['change_in_oi']]
    fig.add_trace(
        go.Bar(
            x=data['date'],
            y=data['change_in_oi'],
            name='Daily OI Change',
            marker_color=colors,
            hovertemplate='%{x}<br>OI Change: %{y:,.0f}<extra></extra>'
        ),
        row=3, col=1
    )
    
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Open Interest", row=2, col=1)
    fig.update_yaxes(title_text="OI Change", row=3, col=1)
    
    fig.update_layout(
        height=800,
        showlegend=True,
        hovermode='x unified',
        title_text=f"Futures Analysis - Expiry: {expiry_date.strftime('%d-%b-%Y') if expiry_date else 'All'}",
        title_font_size=16
    )
    
    return fig

@st.cache_data
def load_raw_futures_data():
    """Load raw futures data from database for OI analysis"""
    try:
        from database.models import get_session, FuturesData
        session = get_session()
        
        query = session.query(FuturesData).order_by(FuturesData.date)
        data = []
        for record in query.all():
            data.append({
                'date': record.date,
                'symbol': record.symbol,
                'open_interest': record.open_interest,
                'change_in_oi': record.change_in_oi,
                'volume': record.volume,
                'underlying_value': record.underlying_value,
                'expiry_date': record.expiry_date
            })
        
        session.close()
        
        if not data:
            return None
            
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df['expiry_date'] = pd.to_datetime(df['expiry_date'])
        
        return df
    except Exception as e:
        st.error(f"Error loading raw futures data: {str(e)}")
        return None

def main():
    st.title("📊 Market Sentiment Probability Dashboard")
    st.markdown("Predict market sentiment using NSE Futures and Options Open Interest data")
    
    with st.sidebar:
        st.markdown("### 🔄 Data Source")
        st.info("📊 Data loaded from PostgreSQL database")
        
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["📈 Sentiment Prediction", "📊 OI Analysis"])
    
    # Load data
    with st.spinner("Loading and processing market data..."):
        data = load_and_process_data()
    
    if data is None:
        st.error("Failed to load data. Please check the data files.")
        return
    
    # TAB 1: Sentiment Prediction
    with tab1:
        # Train model
        with st.spinner("Training machine learning model..."):
            model, scaler, feature_cols = train_model(data)
        
        if model is None:
            st.error("Failed to train the prediction model.")
            return
        
        # Generate predictions for all dates
        processor = DatabaseDataProcessor()
        trainer = ModelTrainer()
        
        # Prepare features for prediction
        features_data = data[feature_cols].copy()
        features_scaled = scaler.transform(features_data)
        probabilities = model.predict_proba(features_scaled)[:, 1]  # Probability of bullish class
        
        # Add probabilities to data
        data['bullish_probability'] = probabilities
        
        # Date selector
        st.markdown("### 📅 Select Date for Prediction")
        
        min_date = data['date'].min()
        max_date = data['date'].max()
        selected_date = st.date_input(
            "Select date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="sentiment_date"
        )
        
        # Convert to datetime for comparison
        selected_date = pd.to_datetime(selected_date)
        
        # Filter data for selected date
        selected_row = data[data['date'] == selected_date]
        
        if selected_row.empty:
            st.warning(f"No data available for {selected_date.strftime('%Y-%m-%d')}")
            return
        
        selected_prob = selected_row['bullish_probability'].iloc[0]
        
        # Main dashboard layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader(f"📈 Prediction for {selected_date.strftime('%Y-%m-%d')}")
            
            # Probability gauge
            gauge_fig = create_probability_gauge(selected_prob)
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Sentiment interpretation
            if selected_prob > 0.7:
                sentiment = "🟢 Strongly Bullish"
                color = "green"
            elif selected_prob > 0.5:
                sentiment = "🔵 Moderately Bullish"
                color = "blue"
            elif selected_prob > 0.3:
                sentiment = "🟡 Moderately Bearish"
                color = "orange"
            else:
                sentiment = "🔴 Strongly Bearish"
                color = "red"
            
            st.markdown(f"### {sentiment}")
            st.markdown(f"**Confidence:** {selected_prob:.1%}")
            
            # Key metrics for selected date
            st.subheader("📊 Key Metrics")
            metrics_data = selected_row.iloc[0]
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("FII Net", f"₹{metrics_data.get('fii_net', 0):.0f}Cr", 
                         f"{metrics_data.get('fii_change', 0):.0f}Cr")
                st.metric("PCR", f"{metrics_data.get('pcr', 0):.2f}")
            
            with col_b:
                st.metric("FII 3D Avg", f"₹{metrics_data.get('fii_3d_avg', 0):.0f}Cr")
                st.metric("PCR 3D Avg", f"{metrics_data.get('pcr_3d_avg', 0):.2f}")
        
        with col2:
            st.subheader("📈 Historical Sentiment Trend")
            
            # Historical probability chart
            trend_fig = create_probability_trend_chart(data)
            
            # Highlight selected date
            trend_fig.add_vline(
                x=selected_date, 
                line_dash="dash", 
                line_color="red",
                annotation_text="Selected Date"
            )
            
            st.plotly_chart(trend_fig, use_container_width=True)
            
            # Feature importance
            st.subheader("🎯 Model Feature Importance")
            importance_fig = create_feature_importance_chart(
                model.coef_[0], feature_cols
            )
            st.plotly_chart(importance_fig, use_container_width=True)
        
        # Download section
        st.subheader("💾 Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download predictions CSV
            predictions_csv = data[['date', 'bullish_probability', 'fii_net', 'pcr']].copy()
            predictions_csv['sentiment'] = predictions_csv['bullish_probability'].apply(
                lambda x: 'Bullish' if x > 0.5 else 'Bearish'
            )
            
            csv_buffer = StringIO()
            predictions_csv.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv_buffer.getvalue(),
                file_name=f"market_sentiment_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Download raw data
            raw_csv_buffer = StringIO()
            data.to_csv(raw_csv_buffer, index=False)
            
            st.download_button(
                label="📥 Download Raw Data CSV",
                data=raw_csv_buffer.getvalue(),
                file_name=f"market_data_processed_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col3:
            st.info("💡 **Model Info**\n\nAlgorithm: Logistic Regression\n\nFeatures: FII data, PCR, Moving averages")
        
        # Model performance metrics
        with st.expander("📊 Model Performance Details"):
            trainer = ModelTrainer()
            y_true = data['target']
            features_scaled = scaler.transform(data[feature_cols])
            y_pred = model.predict(features_scaled)
            y_prob = model.predict_proba(features_scaled)[:, 1]
            
            accuracy = trainer.calculate_accuracy(y_true, y_pred)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{accuracy:.1%}")
            with col2:
                st.metric("Total Predictions", len(y_true))
            with col3:
                bullish_count = sum(y_true)
                st.metric("Bullish Days", f"{bullish_count} ({bullish_count/len(y_true):.1%})")
    
    # TAB 2: OI Analysis
    with tab2:
        st.markdown("### 📊 Futures Open Interest Analysis")
        st.markdown("Analyze NIFTY price movements and OI changes by expiry date")
        
        # Load raw futures data
        raw_data = load_raw_futures_data()
        
        if raw_data is None or raw_data.empty:
            st.error("No futures data available. Please run: python scripts/fetch_nse_data.py")
            return
        
        # Expiry selector
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown("#### Select Expiry Date")
            
            available_expiries = sorted(raw_data['expiry_date'].unique(), reverse=True)
            expiry_options = [exp.strftime('%d-%b-%Y') for exp in available_expiries]
            
            selected_expiry_str = st.selectbox(
                "Expiry",
                options=expiry_options,
                key="expiry_selector"
            )
            
            selected_expiry = pd.to_datetime(selected_expiry_str, format='%d-%b-%Y')
            
            # Filter data for selected expiry
            filtered_data = raw_data[raw_data['expiry_date'] == selected_expiry].copy()
            filtered_data = filtered_data.sort_values('date')
            
            # Filter to last 3 months from expiry
            three_months_ago = selected_expiry - pd.Timedelta(days=90)
            filtered_data = filtered_data[filtered_data['date'] >= three_months_ago]
            
            # Display stats
            st.metric("Data Points", len(filtered_data))
            if len(filtered_data) > 0:
                date_span = (filtered_data['date'].max() - filtered_data['date'].min()).days
                st.metric("Date Range", f"{date_span} days")
            else:
                st.metric("Date Range", "N/A")
            
            if not filtered_data.empty:
                st.metric("Latest OI", f"{filtered_data['open_interest'].iloc[-1]:,.0f}")
                st.metric("Latest Price", f"₹{filtered_data['underlying_value'].iloc[-1]:,.2f}")
        
        with col2:
            if filtered_data.empty:
                st.warning("No data available for selected expiry")
            else:
                # Create combined OI analysis chart
                oi_chart = create_oi_analysis_chart(filtered_data, selected_expiry)
                st.plotly_chart(oi_chart, use_container_width=True)
                
                # Summary statistics
                st.markdown("### 📈 Summary Statistics")
                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    st.metric("Avg Price", f"₹{filtered_data['underlying_value'].mean():,.2f}")
                with col_b:
                    st.metric("Price Range", f"₹{filtered_data['underlying_value'].max() - filtered_data['underlying_value'].min():,.2f}")
                with col_c:
                    total_oi_change = filtered_data['change_in_oi'].sum()
                    st.metric("Net OI Change", f"{total_oi_change:,.0f}")
                with col_d:
                    st.metric("Avg Daily Volume", f"{filtered_data['volume'].mean():,.0f}")

if __name__ == "__main__":
    main()
