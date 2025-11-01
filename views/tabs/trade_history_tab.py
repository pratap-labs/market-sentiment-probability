"""Trade history analysis tab from tradebook.csv."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

import sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def render_trade_history_tab():
    """Render trade history analysis tab from tradebook.csv."""
    st.subheader("📊 Trade History Analysis")
    st.caption("Comprehensive analysis from Kite Console tradebook export")
    
    # File path
    tradebook_path = "database/tradebook.csv"
    
    # Check if file exists
    if not os.path.exists(tradebook_path):
        st.error(f"❌ Tradebook file not found at: {tradebook_path}")
        st.info("💡 Export your tradebook from Kite Console and save it as `database/tradebook.csv`")
        return
    
    try:
        # Read tradebook
        with st.spinner("Loading tradebook..."):
            # Read CSV and drop any completely empty columns
            df = pd.read_csv(tradebook_path)
            df = df.dropna(axis=1, how='all')  # Drop columns that are completely empty
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Drop unnamed columns
        
        st.success(f"✅ Loaded {len(df)} trades from tradebook")
        
        # Display column info for debugging
        with st.expander("📋 Tradebook Columns"):
            st.write(df.columns.tolist())
            st.write(f"Shape: {df.shape}")
            st.dataframe(df.head(3))
            st.caption("Expected: Symbol, ISIN, Trade Date, Exchange, Segment, Series, Trade Type, Auction, Quantity, Price, Trade ID, Order ID, Order Execution Time")
        
        # Normalize column names
        df.columns = df.columns.str.strip()
        
        # Check for required columns
        required_cols = ['Symbol', 'Trade Date', 'Trade Type', 'Quantity', 'Price']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
            st.dataframe(df.head())
            return
        
        # Parse dates
        df['Trade Date'] = pd.to_datetime(df['Trade Date'], errors='coerce')
        
        # Parse Order Execution Time if available
        if 'Order Execution Time' in df.columns:
            df['Order Execution Time'] = pd.to_datetime(df['Order Execution Time'], errors='coerce')
        
        # Calculate trade value
        df['Trade Value'] = df['Quantity'] * df['Price']
        
        # Add date range filter
        st.markdown("### 📅 Filter by Date Range")
        col1, col2, col3 = st.columns([1, 1, 2])
        
        min_date = df['Trade Date'].min()
        max_date = df['Trade Date'].max()
        
        with col1:
            start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
        
        with col2:
            end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
        
        # Filter by date range
        mask = (df['Trade Date'] >= pd.Timestamp(start_date)) & (df['Trade Date'] <= pd.Timestamp(end_date))
        df_filtered = df[mask].copy()
        
        with col3:
            st.metric("Trades in Period", len(df_filtered), delta=f"{len(df_filtered)/len(df)*100:.1f}% of total")
        
        if len(df_filtered) == 0:
            st.warning("No trades in selected period")
            return
        
        # ========== EXTRACT EXPIRY FROM SYMBOL ==========
        # Extract expiry date from option symbols (e.g., NIFTY24JAN24000CE -> 24JAN)
        def extract_expiry(symbol):
            """Extract expiry from option symbol, removing strike price and CE/PE."""

            # remove last 7 characters without regex
            symbol_str = str(symbol)
            if len(symbol_str) < 7:
                return symbol_str
            symbol_trimmed = symbol_str[:-7]
            
            # If no match, return first 15 chars (should capture underlying + expiry)
            return symbol_trimmed
        
        df_filtered['Expiry'] = df_filtered['Symbol'].apply(extract_expiry)
        
        st.info("📅 Grouping trades by expiry cycle for strategy-level analysis (Iron Condor / Short Strangle)")
        
        # Show expiry extraction results
        with st.expander("🔍 Debug: Expiry Extraction"):
            sample_df = df_filtered[['Symbol', 'Expiry']].drop_duplicates().head(10)
            st.dataframe(sample_df, use_container_width=True)
            st.caption(f"Found {df_filtered['Expiry'].nunique()} unique expiry cycles")
        
        # ========== MATCH TRADES AND GROUP BY EXPIRY ==========
        # First, match individual buy/sell pairs
        matched_trades = []
        
        for symbol in df_filtered['Symbol'].unique():
            symbol_trades = df_filtered[df_filtered['Symbol'] == symbol].copy()
            symbol_trades = symbol_trades.sort_values('Trade Date')
            
            buys = symbol_trades[symbol_trades['Trade Type'].str.upper() == 'BUY'].copy()
            sells = symbol_trades[symbol_trades['Trade Type'].str.upper() == 'SELL'].copy()
            
            # Simple FIFO matching
            for _, buy in buys.iterrows():
                remaining_qty = buy['Quantity']
                buy_price = buy['Price']
                buy_date = buy['Trade Date']
                expiry = buy['Expiry']
                
                for idx, sell in sells.iterrows():
                    if remaining_qty <= 0:
                        break
                    
                    if sell['Quantity'] > 0:
                        matched_qty = min(remaining_qty, sell['Quantity'])
                        
                        # Calculate P&L for this matched pair
                        pnl = matched_qty * (sell['Price'] - buy_price)
                        
                        # Calculate duration
                        duration = (sell['Trade Date'] - buy_date).total_seconds() / 3600  # hours
                        
                        matched_trades.append({
                            'Symbol': symbol,
                            'Expiry': expiry,
                            'Quantity': matched_qty,
                            'Buy Price': buy_price,
                            'Sell Price': sell['Price'],
                            'P&L': pnl,
                            'Entry Date': buy_date,
                            'Exit Date': sell['Trade Date'],
                            'Duration (hrs)': duration,
                            'Trade Value': matched_qty * buy_price
                        })
                        
                        # Update remaining quantities
                        remaining_qty -= matched_qty
                        sells.at[idx, 'Quantity'] -= matched_qty
        
        matched_df = pd.DataFrame(matched_trades)
        
        # ========== GROUP BY EXPIRY FOR STRATEGY-LEVEL METRICS ==========
        if len(matched_df) > 0:
            # Group all trades by expiry
            expiry_groups = matched_df.groupby('Expiry').agg({
                'P&L': 'sum',
                'Entry Date': 'min',
                'Exit Date': 'max',
                'Trade Value': 'sum',
                'Symbol': 'count'  # Number of legs in the strategy
            }).reset_index()
            
            expiry_groups.columns = ['Expiry', 'P&L', 'Entry Date', 'Exit Date', 'Trade Value', 'Num Legs']
            expiry_groups['Duration (hrs)'] = (expiry_groups['Exit Date'] - expiry_groups['Entry Date']).dt.total_seconds() / 3600
            expiry_groups = expiry_groups.sort_values('Entry Date')
            
            st.success(f"✅ Analyzed {len(expiry_groups)} expiry cycles with {len(matched_df)} total legs")
            
            # Store for use in metrics calculation
            total_expiries = len(expiry_groups)
            gross_pnl = expiry_groups['P&L'].sum()
            total_trade_value = expiry_groups['Trade Value'].sum()
            estimated_charges = total_trade_value * 0.01
            net_pnl = gross_pnl - estimated_charges
            winning_expiries = expiry_groups[expiry_groups['P&L'] > 0]
            losing_expiries = expiry_groups[expiry_groups['P&L'] < 0]
            win_count = len(winning_expiries)
            loss_count = len(losing_expiries)
            win_rate = (win_count / total_expiries * 100) if total_expiries > 0 else 0
            loss_rate = (loss_count / total_expiries * 100) if total_expiries > 0 else 0
            avg_win = winning_expiries['P&L'].mean() if win_count > 0 else 0
            avg_loss = abs(losing_expiries['P&L'].mean()) if loss_count > 0 else 0
            gross_profit = winning_expiries['P&L'].sum() if win_count > 0 else 0
            gross_loss = abs(losing_expiries['P&L'].sum()) if loss_count > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            expectancy = (avg_win * win_rate / 100) - (avg_loss * loss_rate / 100)
        else:
            expiry_groups = pd.DataFrame()
            total_expiries = 0
            gross_pnl = 0
            total_trade_value = 0
            estimated_charges = 0
            net_pnl = 0
            win_count = 0
            loss_count = 0
            win_rate = 0
            loss_rate = 0
            avg_win = 0
            avg_loss = 0
            gross_profit = 0
            gross_loss = 0
            profit_factor = 0
            expectancy = 0
        
        # Create tabs for different analysis sections
        analysis_tabs = st.tabs([
            "📊 Profitability",
            "⚡ Efficiency", 
            "📈 Performance Trends",
            "🎯 Trade Analysis",
            "📉 Drawdown",
            "📋 Raw Data"
        ])
        
        # ========== PROFITABILITY METRICS TAB ==========
        with analysis_tabs[0]:
            st.markdown("### 💰 Profitability Metrics (Per Expiry Cycle)")
            
            if len(expiry_groups) == 0:
                st.warning("⚠️ No completed expiry cycles found in this period")
                st.info("Showing raw trade data instead")
                
                # Show basic stats from raw data
                buy_value = df_filtered[df_filtered['Trade Type'].str.upper() == 'BUY']['Trade Value'].sum()
                sell_value = df_filtered[df_filtered['Trade Type'].str.upper() == 'SELL']['Trade Value'].sum()
                gross_pnl_raw = sell_value - buy_value
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Buy Value", f"₹{buy_value:,.2f}")
                with col2:
                    st.metric("Sell Value", f"₹{sell_value:,.2f}")
                
                st.metric("Gross P&L (Approximate)", f"₹{gross_pnl_raw:,.2f}",
                         help="Sell Value - Buy Value (not matched)")
                
            else:
                # Calculate metrics from expiry groups (strategy-level)
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Gross P&L", f"₹{gross_pnl:,.2f}")
                    if gross_pnl > 0:
                        st.success("🟢 Profitable")
                    else:
                        st.error("🔴 Loss")
                
                with col2:
                    st.metric("Net P&L (Est.)", f"₹{net_pnl:,.2f}",
                             delta=f"Charges: ~₹{estimated_charges:,.0f}",
                             delta_color="inverse",
                             help="Estimated charges at 1% of trade value")
                
                with col3:
                    st.metric("Profit Factor", f"{profit_factor:.2f}",
                             help="Gross Profit / Gross Loss")
                    if profit_factor > 1.5:
                        st.success("🟢 Excellent")
                    elif profit_factor > 1.0:
                        st.warning("🟡 Profitable")
                    else:
                        st.error("🔴 Losing")
                
                with col4:
                    st.metric("Total Expiry Cycles", total_expiries,
                             help="Number of different expiry cycles traded")
                
                st.markdown("---")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Win Rate", f"{win_rate:.1f}%",
                             delta=f"{win_count} winning cycles")
                
                with col2:
                    st.metric("Loss Rate", f"{loss_rate:.1f}%",
                             delta=f"{loss_count} losing cycles")
                
                with col3:
                    st.metric("Avg Win", f"₹{avg_win:,.2f}",
                             help="Average profit per winning expiry cycle")
                
                with col4:
                    st.metric("Avg Loss", f"₹{avg_loss:,.2f}",
                             help="Average loss per losing expiry cycle")
                
                # Win/Loss ratio and Expectancy
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
                    st.metric("Avg Win / Avg Loss", f"{win_loss_ratio:.2f}",
                             help="Higher is better - shows quality of wins vs losses")
                    if win_loss_ratio > 2:
                        st.success("🟢 Excellent - Big wins, small losses")
                    elif win_loss_ratio > 1:
                        st.warning("🟡 Good")
                    else:
                        st.error("🔴 Poor - Losses bigger than wins")
                
                with col2:
                    # Expectancy
                    st.metric("Expectancy per Expiry", f"₹{expectancy:,.2f}",
                             help="(Avg Win × Win Rate) - (Avg Loss × Loss Rate)")
                    if expectancy > 0:
                        st.success(f"🟢 Positive: ₹{expectancy:,.2f} per expiry")
                    else:
                        st.error(f"🔴 Negative: ₹{expectancy:,.2f} per expiry")
                
                with col3:
                    # Max Drawdown from expiry cycles
                    cumulative_pnl = expiry_groups.sort_values('Entry Date')['P&L'].cumsum()
                    cumulative_max = cumulative_pnl.expanding().max()
                    drawdown = cumulative_pnl - cumulative_max
                    max_dd = drawdown.min()
                    
                    st.metric("Max Drawdown", f"₹{abs(max_dd):,.2f}",
                             delta_color="inverse",
                             help="Largest peak-to-trough loss")
                
                # Recovery Factor
                col1, col2 = st.columns(2)
                
                with col1:
                    recovery_factor = net_pnl / abs(max_dd) if max_dd != 0 else 0
                    st.metric("Recovery Factor", f"{recovery_factor:.2f}",
                             help="Net P&L / Max Drawdown")
                    if recovery_factor > 3:
                        st.success("🟢 Excellent recovery efficiency")
                    elif recovery_factor > 1:
                        st.warning("🟡 Good")
                    else:
                        st.error("🔴 Poor - losses not recovered efficiently")
                
                with col2:
                    # Gross Profit and Loss breakdown
                    st.metric("Gross Profit", f"₹{gross_profit:,.2f}",
                             delta=f"From {win_count} winning cycles")
                
                # Expiry Cycles Table
                st.markdown("### 📅 Expiry Cycle Performance")
                
                display_df = expiry_groups.copy()
                display_df['ROI %'] = (display_df['P&L'] / display_df['Trade Value'] * 100).round(2)
                display_df['P&L'] = display_df['P&L'].apply(lambda x: f"₹{x:,.2f}")
                display_df['Trade Value'] = display_df['Trade Value'].apply(lambda x: f"₹{x:,.2f}")
                display_df['Duration (days)'] = (display_df['Duration (hrs)'] / 24).round(1)
                display_df = display_df[['Expiry', 'Entry Date', 'Exit Date', 'Num Legs', 'P&L', 'ROI %', 'Duration (days)']].sort_values('Entry Date', ascending=False)
                
                st.dataframe(display_df, use_container_width=True, height=400)
                
                # P&L Distribution per Expiry
                st.markdown("### 📊 P&L Distribution (Per Expiry Cycle)")
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=expiry_groups['P&L'],
                    nbinsx=20,
                    marker_color='#3B82F6',
                    name='Expiry P&L'
                ))
                
                fig.add_vline(x=0, line_dash="dash", line_color="white",
                             annotation_text="Break-even")
                fig.add_vline(x=expectancy, line_dash="dash", line_color="green",
                             annotation_text=f"Expectancy: ₹{expectancy:.0f}")
                
                fig.update_layout(
                    title="Expiry Cycle P&L Distribution",
                    xaxis_title="P&L per Expiry (₹)",
                    yaxis_title="Frequency",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # ========== EFFICIENCY METRICS TAB ==========
        with analysis_tabs[1]:
            st.markdown("### ⚡ Efficiency & Strategy Quality (Per Expiry)")
            
            if len(expiry_groups) == 0:
                st.warning("⚠️ No completed expiry cycles to analyze")
                return
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Expiry Cycles", len(expiry_groups))
                avg_legs = expiry_groups['Num Legs'].mean()
                st.caption(f"Avg {avg_legs:.1f} legs per cycle")
            
            with col2:
                # Average cycle duration
                avg_duration = expiry_groups['Duration (hrs)'].mean()
                if avg_duration < 24:
                    duration_display = f"{avg_duration:.1f} hrs"
                else:
                    duration_display = f"{avg_duration/24:.1f} days"
                st.metric("Avg Cycle Duration", duration_display)
            
            with col3:
                # Return on Investment
                roi = (gross_pnl / total_trade_value * 100) if total_trade_value > 0 else 0
                st.metric("Total ROI", f"{roi:.2f}%",
                         help="Gross P&L / Total Capital Deployed")
            
            with col4:
                # Risk-adjusted P&L
                pnl_std = expiry_groups['P&L'].std()
                risk_adjusted = gross_pnl / pnl_std if pnl_std > 0 else 0
                st.metric("Risk-Adjusted P&L", f"{risk_adjusted:.2f}",
                         help="P&L / Std Dev of expiry cycles")
            
            # Streaks
            st.markdown("### 🔥 Winning & Losing Streaks (Per Expiry)")
            
            # Sort by entry date and calculate streaks
            expiry_sorted = expiry_groups.sort_values('Entry Date')
            is_win = expiry_sorted['P&L'] > 0
            
            # Calculate streaks
            current_streak = 0
            max_win_streak = 0
            max_loss_streak = 0
            current_is_win = None
            
            for win in is_win:
                if current_is_win == win:
                    current_streak += 1
                else:
                    current_streak = 1
                    current_is_win = win
                
                if win:
                    max_win_streak = max(max_win_streak, current_streak)
                else:
                    max_loss_streak = max(max_loss_streak, current_streak)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Max Winning Streak", f"{max_win_streak} cycles",
                         help="Longest consecutive winning expiry cycles")
                if max_win_streak > 5:
                    st.success("🟢 Strong consistency")
            
            with col2:
                st.metric("Max Losing Streak", f"{max_loss_streak} cycles",
                         help="Longest consecutive losing expiry cycles")
                if max_loss_streak > 5:
                    st.error("🔴 High - review strategy robustness")
            
            # Sharpe-like metric
            st.markdown("### 📈 Risk-Adjusted Performance")
            
            mean_pnl = expiry_groups['P&L'].mean()
            std_pnl = expiry_groups['P&L'].std()
            sharpe_like = mean_pnl / std_pnl if std_pnl > 0 else 0
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Mean P&L per Expiry", f"₹{mean_pnl:,.2f}")
            
            with col2:
                st.metric("Sharpe-like Ratio", f"{sharpe_like:.2f}",
                         help="Mean P&L / Std Dev P&L (per expiry cycle)")
                if sharpe_like > 1:
                    st.success("🟢 Excellent risk-adjusted returns")
                elif sharpe_like > 0.5:
                    st.warning("🟡 Good")
                else:
                    st.error("🔴 Poor - high volatility relative to returns")
            
            # Consistency Score
            st.markdown("### 🎯 Consistency Score")
            
            # Combine multiple factors into consistency score (0-100)
            win_rate_score = min(win_rate, 100)
            profit_factor_score = min(profit_factor * 33.3, 100)
            streak_score = max(0, 100 - (max_loss_streak * 10))
            expectancy_score = min(max(0, expectancy / 100), 100) if expectancy > 0 else 0
            
            consistency_score = (win_rate_score * 0.3 + 
                               profit_factor_score * 0.3 + 
                               streak_score * 0.2 + 
                               expectancy_score * 0.2)
            
            st.metric("Consistency Score", f"{consistency_score:.1f}/100",
                     help="Combined metric: Win Rate (30%), Profit Factor (30%), Streak Control (20%), Expectancy (20%)")
            
            if consistency_score > 70:
                st.success("🟢 Highly consistent strategy")
            elif consistency_score > 50:
                st.warning("🟡 Moderately consistent")
            else:
                st.error("🔴 Inconsistent - needs improvement")
            
            # Breakdown
            with st.expander("📊 Consistency Score Breakdown"):
                breakdown_df = pd.DataFrame({
                    'Component': ['Win Rate', 'Profit Factor', 'Streak Control', 'Expectancy'],
                    'Score': [win_rate_score * 0.3, profit_factor_score * 0.3, 
                             streak_score * 0.2, expectancy_score * 0.2],
                    'Weight': ['30%', '30%', '20%', '20%']
                })
                st.dataframe(breakdown_df, use_container_width=True)
        
        # ========== PERFORMANCE TRENDS TAB ==========
        with analysis_tabs[2]:
            st.markdown("### 📈 Performance Over Time (Per Expiry Cycle)")
            
            if len(expiry_groups) == 0:
                st.warning("⚠️ No completed expiry cycles to analyze")
                return
            
            # Cumulative P&L by expiry cycle
            expiry_sorted = expiry_groups.sort_values('Entry Date').copy()
            expiry_sorted['Cumulative_PnL'] = expiry_sorted['P&L'].cumsum()
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Cumulative P&L by Expiry Cycle', 'P&L per Expiry Cycle'),
                vertical_spacing=0.15,
                row_heights=[0.6, 0.4]
            )
            
            # Cumulative P&L
            fig.add_trace(
                go.Scatter(
                    x=expiry_sorted['Expiry'],
                    y=expiry_sorted['Cumulative_PnL'],
                    mode='lines+markers',
                    name='Cumulative P&L',
                    line=dict(color='#3B82F6', width=2),
                    fill='tozeroy',
                    text=expiry_sorted['Entry Date'].dt.strftime('%Y-%m-%d'),
                    hovertemplate='<b>%{x}</b><br>Cumulative: ₹%{y:,.2f}<br>Date: %{text}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Per Expiry P&L
            colors = ['#10B981' if x > 0 else '#EF4444' for x in expiry_sorted['P&L']]
            fig.add_trace(
                go.Bar(
                    x=expiry_sorted['Expiry'],
                    y=expiry_sorted['P&L'],
                    name='Expiry P&L',
                    marker_color=colors,
                    text=expiry_sorted['Entry Date'].dt.strftime('%Y-%m-%d'),
                    hovertemplate='<b>%{x}</b><br>P&L: ₹%{y:,.2f}<br>Date: %{text}<extra></extra>'
                ),
                row=2, col=1
            )
            
            fig.update_xaxes(title_text="Expiry", row=2, col=1)
            fig.update_yaxes(title_text="Cumulative P&L (₹)", row=1, col=1)
            fig.update_yaxes(title_text="P&L (₹)", row=2, col=1)
            
            fig.update_layout(height=700, showlegend=False, hovermode='x unified')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                best_expiry_pnl = expiry_sorted['P&L'].max()
                best_expiry = expiry_sorted[expiry_sorted['P&L'] == best_expiry_pnl]['Expiry'].iloc[0]
                st.metric("Best Expiry", f"₹{best_expiry_pnl:,.2f}",
                         delta=best_expiry)
            
            with col2:
                worst_expiry_pnl = expiry_sorted['P&L'].min()
                worst_expiry = expiry_sorted[expiry_sorted['P&L'] == worst_expiry_pnl]['Expiry'].iloc[0]
                st.metric("Worst Expiry", f"₹{worst_expiry_pnl:,.2f}",
                         delta=worst_expiry)
            
            with col3:
                positive_expiries = (expiry_sorted['P&L'] > 0).sum()
                total_expiries_count = len(expiry_sorted)
                st.metric("Positive Expiries", f"{positive_expiries}/{total_expiries_count}",
                         delta=f"{positive_expiries/total_expiries_count*100:.1f}%")
            
            with col4:
                avg_expiry_pnl = expiry_sorted['P&L'].mean()
                st.metric("Avg Expiry P&L", f"₹{avg_expiry_pnl:,.2f}")
        
        # ========== TRADE ANALYSIS TAB ==========
        with analysis_tabs[3]:
            st.markdown("### 🎯 Detailed Analysis by Expiry")
            
            if len(expiry_groups) == 0:
                st.warning("⚠️ No completed expiry cycles to analyze")
                return
            
            # By expiry
            st.markdown("#### Performance by Expiry Cycle")
            
            display_expiry = expiry_groups.copy()
            display_expiry['ROI %'] = (display_expiry['P&L'] / display_expiry['Trade Value'] * 100).round(2)
            display_expiry['Duration (days)'] = (display_expiry['Duration (hrs)'] / 24).round(1)
            display_expiry = display_expiry.sort_values('Entry Date', ascending=False)
            
            st.dataframe(
                display_expiry[['Expiry', 'Entry Date', 'Exit Date', 'Num Legs', 'P&L', 'Trade Value', 'ROI %', 'Duration (days)']].style.format({
                    'P&L': '₹{:,.2f}',
                    'Trade Value': '₹{:,.2f}',
                    'ROI %': '{:.2f}%',
                    'Duration (days)': '{:.1f}'
                }).background_gradient(subset=['P&L'], cmap='RdYlGn', vmin=-10000, vmax=10000),
                use_container_width=True,
                height=400
            )
            
            # Strategy composition
            st.markdown("#### Strategy Leg Composition")
            leg_distribution = expiry_groups['Num Legs'].value_counts().sort_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure(data=[go.Bar(
                    x=leg_distribution.index,
                    y=leg_distribution.values,
                    marker_color='#3B82F6',
                    text=leg_distribution.values,
                    textposition='auto'
                )])
                
                fig.update_layout(
                    title="Number of Legs per Expiry Cycle",
                    xaxis_title="Number of Legs",
                    yaxis_title="Frequency",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.caption("2 legs = Vertical Spread, 4 legs = Iron Condor, etc.")
            
            with col2:
                # ROI distribution
                roi_values = (expiry_groups['P&L'] / expiry_groups['Trade Value'] * 100)
                
                fig = go.Figure(data=[go.Histogram(
                    x=roi_values,
                    nbinsx=20,
                    marker_color='#10B981'
                )])
                
                fig.update_layout(
                    title="ROI Distribution per Expiry",
                    xaxis_title="ROI (%)",
                    yaxis_title="Frequency",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Individual legs by symbol
            st.markdown("#### Performance by Individual Legs (Symbols)")
            
            symbol_performance = matched_df.groupby('Symbol').agg({
                'P&L': ['sum', 'count', 'mean'],
                'Quantity': 'sum'
            }).reset_index()
            
            symbol_performance.columns = ['Symbol', 'Total P&L', 'Leg Count', 'Avg P&L', 'Total Qty']
            symbol_performance = symbol_performance.sort_values('Total P&L', ascending=False)
            
            st.dataframe(
                symbol_performance.head(20).style.format({
                    'Total P&L': '₹{:,.2f}',
                    'Leg Count': '{:.0f}',
                    'Avg P&L': '₹{:,.2f}',
                    'Total Qty': '{:.0f}'
                }).background_gradient(subset=['Total P&L'], cmap='RdYlGn', vmin=-1000, vmax=1000),
                use_container_width=True
            )
            
            # Top expiries
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🏆 Top 5 Best Expiry Cycles")
                top_winners = expiry_groups.nlargest(5, 'P&L')[['Expiry', 'Entry Date', 'P&L', 'Num Legs', 'Duration (hrs)']]
                top_winners['Duration (days)'] = (top_winners['Duration (hrs)'] / 24).round(1)
                st.dataframe(
                    top_winners[['Expiry', 'Entry Date', 'P&L', 'Num Legs', 'Duration (days)']].style.format({
                        'P&L': '₹{:,.2f}',
                        'Duration (days)': '{:.1f}'
                    }),
                    use_container_width=True
                )
            
            with col2:
                st.markdown("#### 💔 Top 5 Worst Expiry Cycles")
                top_losers = expiry_groups.nsmallest(5, 'P&L')[['Expiry', 'Entry Date', 'P&L', 'Num Legs', 'Duration (hrs)']]
                top_losers['Duration (days)'] = (top_losers['Duration (hrs)'] / 24).round(1)
                st.dataframe(
                    top_losers[['Expiry', 'Entry Date', 'P&L', 'Num Legs', 'Duration (days)']].style.format({
                        'P&L': '₹{:,.2f}',
                        'Duration (days)': '{:.1f}'
                    }),
                    use_container_width=True
                )
            
            # Holding period analysis
            st.markdown("#### Holding Period Analysis")
            
            fig = go.Figure(data=[go.Histogram(
                x=expiry_groups['Duration (hrs)'] / 24,  # Convert to days
                nbinsx=20,
                marker_color='#9333EA'
            )])
            
            fig.update_layout(
                title="Expiry Cycle Duration Distribution",
                xaxis_title="Duration (days)",
                yaxis_title="Frequency",
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # ========== DRAWDOWN TAB ==========
        with analysis_tabs[4]:
            st.markdown("### 📉 Drawdown Analysis (Per Expiry Cycle)")
            
            if len(expiry_groups) == 0:
                st.warning("⚠️ No completed expiry cycles to analyze")
                return
            
            # Calculate drawdown from expiry cycles
            expiry_sorted = expiry_groups.sort_values('Entry Date').copy()
            expiry_sorted['Cumulative_PnL'] = expiry_sorted['P&L'].cumsum()
            
            cumulative_max = expiry_sorted['Cumulative_PnL'].expanding().max()
            expiry_sorted['Drawdown'] = expiry_sorted['Cumulative_PnL'] - cumulative_max
            
            max_dd = expiry_sorted['Drawdown'].min()
            max_dd_idx = expiry_sorted['Drawdown'].idxmin()
            max_dd_pct = (max_dd / cumulative_max[max_dd_idx]) * 100 if cumulative_max[max_dd_idx] != 0 else 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Max Drawdown", f"₹{abs(max_dd):,.2f}",
                         delta=f"{max_dd_pct:.2f}%",
                         delta_color="inverse")
            
            with col2:
                # Recovery factor
                recovery_factor = gross_pnl / abs(max_dd) if max_dd != 0 else 0
                st.metric("Recovery Factor", f"{recovery_factor:.2f}",
                         help="Net P&L / Max Drawdown")
                if recovery_factor > 3:
                    st.success("🟢 Excellent recovery efficiency")
                elif recovery_factor > 1:
                    st.warning("🟡 Good")
                else:
                    st.error("🔴 Poor - losses not recovered efficiently")
            
            with col3:
                current_dd = expiry_sorted['Drawdown'].iloc[-1]
                st.metric("Current Drawdown", f"₹{abs(current_dd):,.2f}")
            
            # Drawdown chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=expiry_sorted['Expiry'],
                y=expiry_sorted['Drawdown'],
                mode='lines+markers',
                name='Drawdown',
                fill='tozeroy',
                line=dict(color='#EF4444', width=2),
                text=expiry_sorted['Entry Date'].dt.strftime('%Y-%m-%d'),
                hovertemplate='<b>%{x}</b><br>Drawdown: ₹%{y:,.2f}<br>Date: %{text}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Drawdown Over Time (By Expiry Cycle)",
                xaxis_title="Expiry",
                yaxis_title="Drawdown (₹)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Drawdown recovery analysis
            st.markdown("#### Recovery Analysis")
            
            # Find drawdown periods
            in_drawdown = expiry_sorted['Drawdown'] < 0
            drawdown_periods = []
            
            start_idx = None
            for idx, is_dd in enumerate(in_drawdown):
                if is_dd and start_idx is None:
                    start_idx = idx
                elif not is_dd and start_idx is not None:
                    drawdown_periods.append((start_idx, idx - 1))
                    start_idx = None
            
            if start_idx is not None:
                drawdown_periods.append((start_idx, len(expiry_sorted) - 1))
            
            if drawdown_periods:
                recovery_info = []
                for start, end in drawdown_periods:
                    dd_depth = expiry_sorted.iloc[start:end+1]['Drawdown'].min()
                    dd_duration_days = (expiry_sorted.iloc[end]['Entry Date'] - 
                                 expiry_sorted.iloc[start]['Entry Date']).days
                    recovery_info.append({
                        'Start Expiry': expiry_sorted.iloc[start]['Expiry'],
                        'End Expiry': expiry_sorted.iloc[end]['Expiry'],
                        'Depth': dd_depth,
                        'Duration (expiry cycles)': end - start + 1,
                        'Duration (days)': dd_duration_days
                    })
                
                recovery_df = pd.DataFrame(recovery_info)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    avg_dd_duration = recovery_df['Duration (expiry cycles)'].mean()
                    st.metric("Avg Drawdown Duration", f"{avg_dd_duration:.1f} expiry cycles")
                
                with col2:
                    max_dd_duration = recovery_df['Duration (days)'].max()
                    st.metric("Max Drawdown Duration", f"{max_dd_duration:.0f} days")
                
                st.dataframe(
                    recovery_df.style.format({
                        'Depth': '₹{:,.2f}',
                        'Duration (expiry cycles)': '{:.0f}',
                        'Duration (days)': '{:.0f}'
                    }),
                    use_container_width=True
                )
        
        # ========== RAW DATA TAB ==========
        with analysis_tabs[5]:
            st.markdown("### 📋 Raw Data")
            
            tab1, tab2 = st.tabs(["Matched Trades", "All Trades"])
            
            with tab1:
                if len(matched_df) > 0:
                    st.markdown("#### Matched Buy-Sell Pairs")
                    st.dataframe(
                        matched_df.style.format({
                            'Buy Price': '₹{:.2f}',
                            'Sell Price': '₹{:.2f}',
                            'P&L': '₹{:,.2f}',
                            'Quantity': '{:.0f}',
                            'Duration (hrs)': '{:.1f}',
                            'Trade Value': '₹{:,.2f}'
                        }),
                        use_container_width=True,
                        height=600
                    )
                    
                    # Download button
                    csv = matched_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Matched Trades as CSV",
                        data=csv,
                        file_name=f"matched_trades_{start_date}_to_{end_date}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No matched trades available")
            
            with tab2:
                st.markdown("#### All Trades from Tradebook")
                st.dataframe(df_filtered, use_container_width=True, height=600)
                
                # Download button
                csv = df_filtered.to_csv(index=False)
                st.download_button(
                    label="📥 Download All Trades as CSV",
                    data=csv,
                    file_name=f"tradebook_all_{start_date}_to_{end_date}.csv",
                    mime="text/csv"
                )
    
    except Exception as e:
        st.error(f"❌ Error loading tradebook: {str(e)}")
        st.exception(e)
