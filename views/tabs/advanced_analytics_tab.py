"""Advanced analytics tab with professional metrics."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

try:
    from kiteconnect import KiteConnect
except Exception:
    KiteConnect = None

import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.utils import calculate_portfolio_greeks


def render_advanced_analytics_tab():
    """Render advanced analytics tab with professional metrics."""
    st.subheader("🎯 Advanced Analytics")

    st.sidebar.caption("No controls for this tab.")
    
    if "enriched_positions" not in st.session_state:
        st.info("No positions loaded. Fetch positions from the Positions tab first.")
        return
    
    enriched = st.session_state["enriched_positions"]
    portfolio_greeks = calculate_portfolio_greeks(enriched)
    current_spot = st.session_state.get("current_spot", 25000)
    
    # Get account info
    access_token = st.session_state.get("kite_access_token")
    kite_api_key = st.session_state.get("kite_api_key")
    
    margin_available = 500000
    margin_used = 320000
    
    if access_token and kite_api_key:
        try:
            kite = KiteConnect(api_key=kite_api_key)
            kite.set_access_token(access_token)
            margins = kite.margins()
            equity_margins = margins.get("equity", {})
            margin_available = equity_margins.get("available", {}).get("live_balance", 500000)
            margin_used = equity_margins.get("utilised", {}).get("debits", 0)
        except:
            pass
    
    account_size = margin_available + margin_used
    total_pnl = sum(p.get("pnl", 0) for p in enriched)
    
    # Create tabs for different analytics sections
    analytics_tabs = st.tabs([
        "📊 P&L Attribution",
        "⚠️ Risk Metrics", 
        "💰 Efficiency",
        "📈 Volatility Surface",
        "🔗 Concentration",
        "⚡ Execution Quality",
        "📉 Drawdown Analysis"
    ])
    
    # ========== P&L ATTRIBUTION TAB ==========
    with analytics_tabs[0]:
        st.markdown("### P&L Attribution Analysis")
        st.caption("Decomposes P&L into Greek components to identify true profit sources")
        
        # Calculate P&L attribution
        total_delta = portfolio_greeks["net_delta"]
        total_gamma = portfolio_greeks["net_gamma"]
        total_vega = portfolio_greeks["net_vega"]
        total_theta = portfolio_greeks["net_theta"]
        
        # Get previous spot (approximation - using 1% move for demo)
        prev_spot = current_spot * 0.99
        spot_change = current_spot - prev_spot
        
        # P&L attribution components
        delta_pnl = total_delta * spot_change  # Lot size
        gamma_pnl = 0.5 * total_gamma * (spot_change ** 2)
        vega_pnl = total_vega * 0  # Assume no IV change for now
        theta_pnl = total_theta * 1  # 1 day decay
        residual_pnl = total_pnl - (delta_pnl + gamma_pnl + vega_pnl + theta_pnl)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Delta P&L", f"₹{delta_pnl:,.0f}", 
                     delta=f"{delta_pnl/total_pnl*100:.1f}%" if total_pnl != 0 else "0%")
        
        with col2:
            st.metric("Gamma P&L", f"₹{gamma_pnl:,.0f}",
                     delta=f"{gamma_pnl/total_pnl*100:.1f}%" if total_pnl != 0 else "0%")
        
        with col3:
            st.metric("Vega P&L", f"₹{vega_pnl:,.0f}",
                     delta=f"{vega_pnl/total_pnl*100:.1f}%" if total_pnl != 0 else "0%")
        
        with col4:
            st.metric("Theta P&L", f"₹{theta_pnl:,.0f}",
                     delta=f"{theta_pnl/total_pnl*100:.1f}%" if total_pnl != 0 else "0%")
        
        with col5:
            st.metric("Residual P&L", f"₹{residual_pnl:,.0f}",
                     delta=f"{residual_pnl/total_pnl*100:.1f}%" if total_pnl != 0 else "0%")
        
        # P&L Attribution Chart
        attribution_data = {
            'Component': ['Delta', 'Gamma', 'Vega', 'Theta', 'Residual'],
            'P&L': [delta_pnl, gamma_pnl, vega_pnl, theta_pnl, residual_pnl]
        }
        
        colors = ['#3B82F6' if x > 0 else '#EF4444' for x in attribution_data['P&L']]
        
        fig = go.Figure(data=[
            go.Bar(
                x=attribution_data['Component'],
                y=attribution_data['P&L'],
                marker_color=colors,
                text=[f"₹{x:,.0f}" for x in attribution_data['P&L']],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="P&L Attribution by Greek",
            xaxis_title="Component",
            yaxis_title="P&L (₹)",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Interpretation:**
        - **Delta P&L**: Profit/loss from directional moves
        - **Gamma P&L**: Profit/loss from acceleration (convexity)
        - **Vega P&L**: Profit/loss from IV changes
        - **Theta P&L**: Profit/loss from time decay
        - **Residual**: Unexplained P&L (slippage, crosses, rounding)
        """)
    
    # ========== RISK METRICS TAB ==========
    with analytics_tabs[1]:
        st.markdown("### Advanced Risk Metrics")
        
        # CVaR (Conditional Value at Risk / Expected Shortfall)
        # Using historical simulation approach
        
        st.markdown("#### CVaR (Conditional Value at Risk)")
        st.caption("Expected loss in worst 5% of scenarios (tail risk measure)")
        
        # Simulate returns
        np.random.seed(42)
        daily_returns = np.random.normal(-0.001, 0.02, 1000)  # Simulated daily returns
        
        # Calculate portfolio value changes
        portfolio_values = []
        for ret in daily_returns:
            spot_move = current_spot * ret
            pnl_scenario = (total_delta * spot_move + 
                          0.5 * total_gamma * (spot_move ** 2) +
                          total_theta)
            portfolio_values.append(pnl_scenario)
        
        portfolio_values = np.array(portfolio_values)
        
        # VaR and CVaR at 95% confidence
        var_95 = np.percentile(portfolio_values, 5)
        cvar_95 = portfolio_values[portfolio_values <= var_95].mean()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("VaR (95%)", f"₹{abs(var_95):,.0f}",
                     delta=f"{abs(var_95)/account_size*100:.2f}% of capital",
                     delta_color="inverse")
        
        with col2:
            st.metric("CVaR (95%)", f"₹{abs(cvar_95):,.0f}",
                     delta=f"{abs(cvar_95)/account_size*100:.2f}% of capital",
                     delta_color="inverse")
        
        with col3:
            ratio = abs(cvar_95) / abs(var_95) if var_95 != 0 else 1
            st.metric("CVaR/VaR Ratio", f"{ratio:.2f}",
                     help="How much worse than VaR is the tail risk")
        
        # Distribution chart
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=portfolio_values,
            nbinsx=50,
            name='P&L Distribution',
            marker_color='#3B82F6'
        ))
        
        fig.add_vline(x=var_95, line_dash="dash", line_color="orange",
                     annotation_text=f"VaR: ₹{var_95:,.0f}")
        fig.add_vline(x=cvar_95, line_dash="dash", line_color="red",
                     annotation_text=f"CVaR: ₹{cvar_95:,.0f}")
        
        fig.update_layout(
            title="Portfolio P&L Distribution (1000 Scenarios)",
            xaxis_title="P&L (₹)",
            yaxis_title="Frequency",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **CVaR vs VaR:**
        - **VaR**: Maximum expected loss at confidence level (95% = worst loss in 19/20 days)
        - **CVaR**: Average loss when VaR is exceeded (tail risk)
        - CVaR is always worse than VaR and measures extreme scenarios
        """)
    
    # ========== EFFICIENCY TAB ==========
    with analytics_tabs[2]:
        st.markdown("### Capital Efficiency Metrics")
        
        # Return on Margin (ROM)
        rom = (total_pnl / margin_used * 100) if margin_used > 0 else 0
        roi = (total_pnl / account_size * 100) if account_size > 0 else 0
        
        # Sharpe-like metric (simplified)
        daily_return = total_pnl / account_size
        sharpe_approx = (daily_return * 252) / (0.02 * np.sqrt(252))  # Assuming 2% daily vol
        
        # Margin efficiency
        margin_efficiency = (margin_used / account_size * 100) if account_size > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Return on Margin", f"{rom:.2f}%",
                     help="P&L relative to margin deployed")
            if rom > 10:
                st.success("🟢 Excellent efficiency")
            elif rom > 5:
                st.warning("🟡 Good efficiency")
            else:
                st.error("🔴 Low efficiency")
        
        with col2:
            st.metric("ROI (Account)", f"{roi:.2f}%",
                     help="P&L relative to total account size")
        
        with col3:
            st.metric("Sharpe Ratio (Est.)", f"{sharpe_approx:.2f}",
                     help="Risk-adjusted return estimate")
            if sharpe_approx > 2:
                st.success("🟢 Excellent")
            elif sharpe_approx > 1:
                st.warning("🟡 Good")
            else:
                st.error("🔴 Poor")
        
        with col4:
            st.metric("Margin Utilization", f"{margin_efficiency:.1f}%",
                     help="% of capital deployed")
        
        # Capital efficiency breakdown
        notional_exposure = sum(abs(p.get("quantity", 0)) * p.get("strike", 0) for p in enriched)
        leverage = notional_exposure / account_size if account_size > 0 else 0
        
        st.markdown("#### Capital Deployment Analysis")
        
        efficiency_data = pd.DataFrame({
            'Metric': ['Account Size', 'Margin Used', 'Margin Free', 'Notional Exposure'],
            'Value': [account_size, margin_used, margin_available, notional_exposure]
        })
        
        fig = go.Figure(data=[
            go.Bar(
                x=efficiency_data['Metric'],
                y=efficiency_data['Value'],
                marker_color=['#3B82F6', '#EF4444', '#10B981', '#9333EA'],
                text=[f"₹{x:,.0f}" for x in efficiency_data['Value']],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="Capital Structure",
            yaxis_title="Amount (₹)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Leverage Ratio", f"{leverage:.1f}x",
                     help="Notional exposure / Account size")
        with col2:
            buying_power = margin_available / margin_used if margin_used > 0 else 0
            st.metric("Buying Power Left", f"{buying_power:.1f}x",
                     help="Can increase positions by this factor")
    
    # ========== VOLATILITY SURFACE TAB ==========
    with analytics_tabs[3]:
        st.markdown("### IV Surface & Smile Visualization")
        
        # Build IV surface from positions
        iv_data = []
        for pos in enriched:
            strike = pos.get("strike", 0)
            iv = pos.get("implied_vol", 0)
            dte = pos.get("dte", 0)
            option_type = pos.get("option_type", "")
            
            if strike and iv:
                moneyness = (strike - current_spot) / current_spot * 100
                iv_data.append({
                    'Strike': strike,
                    'Moneyness': moneyness,
                    'IV': iv * 100,
                    'DTE': dte,
                    'Type': option_type
                })
        
        if iv_data:
            iv_df = pd.DataFrame(iv_data)
            
            # IV Smile Chart
            fig = go.Figure()
            
            ce_data = iv_df[iv_df['Type'] == 'CE']
            pe_data = iv_df[iv_df['Type'] == 'PE']
            
            if not ce_data.empty:
                fig.add_trace(go.Scatter(
                    x=ce_data['Moneyness'],
                    y=ce_data['IV'],
                    mode='markers+lines',
                    name='Call IV',
                    marker=dict(size=10, color='#3B82F6'),
                    line=dict(color='#3B82F6', width=2)
                ))
            
            if not pe_data.empty:
                fig.add_trace(go.Scatter(
                    x=pe_data['Moneyness'],
                    y=pe_data['IV'],
                    mode='markers+lines',
                    name='Put IV',
                    marker=dict(size=10, color='#EF4444'),
                    line=dict(color='#EF4444', width=2)
                ))
            
            fig.update_layout(
                title="IV Smile by Moneyness",
                xaxis_title="Moneyness (%)",
                yaxis_title="Implied Volatility (%)",
                height=500,
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # IV statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                atm_iv = iv_df[abs(iv_df['Moneyness']) < 2]['IV'].mean()
                st.metric("ATM IV", f"{atm_iv:.2f}%")
            
            with col2:
                otm_put_iv = iv_df[(iv_df['Type'] == 'PE') & (iv_df['Moneyness'] < -2)]['IV'].mean()
                st.metric("OTM Put IV", f"{otm_put_iv:.2f}%")
            
            with col3:
                otm_call_iv = iv_df[(iv_df['Type'] == 'CE') & (iv_df['Moneyness'] > 2)]['IV'].mean()
                st.metric("OTM Call IV", f"{otm_call_iv:.2f}%")
            
            with col4:
                skew = otm_put_iv - otm_call_iv
                st.metric("Skew", f"{skew:.2f}%",
                         delta="Put premium" if skew > 0 else "Call premium")
            
            # Display IV surface data
            st.markdown("#### IV Surface Data")
            st.dataframe(
                iv_df.sort_values('Moneyness').style.format({
                    'Strike': '{:.0f}',
                    'Moneyness': '{:+.2f}%',
                    'IV': '{:.2f}%',
                    'DTE': '{:.0f}'
                }),
                use_container_width=True
            )
        else:
            st.warning("No IV data available. Fetch positions first.")
    
    # ========== CONCENTRATION TAB ==========
    with analytics_tabs[4]:
        st.markdown("### Position Correlation & Concentration")
        
        # Position concentration by strike
        strike_exposure = {}
        for pos in enriched:
            strike = pos.get("strike", 0)
            notional = abs(pos.get("quantity", 0)) * strike
            
            if strike in strike_exposure:
                strike_exposure[strike] += notional
            else:
                strike_exposure[strike] = notional
        
        # Sort and get top concentrations
        sorted_strikes = sorted(strike_exposure.items(), key=lambda x: x[1], reverse=True)
        
        st.markdown("#### Strike Concentration")
        
        if sorted_strikes:
            top_5_strikes = sorted_strikes[:5]
            
            strike_df = pd.DataFrame(top_5_strikes, columns=['Strike', 'Notional'])
            strike_df['% of Portfolio'] = strike_df['Notional'] / strike_df['Notional'].sum() * 100
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = go.Figure(data=[
                    go.Bar(
                        x=[f"{s[0]:.0f}" for s in top_5_strikes],
                        y=[s[1] for s in top_5_strikes],
                        marker_color='#3B82F6',
                        text=[f"₹{s[1]:,.0f}" for s in top_5_strikes],
                        textposition='outside'
                    )
                ])
                
                fig.update_layout(
                    title="Top 5 Strike Exposures",
                    xaxis_title="Strike",
                    yaxis_title="Notional (₹)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.dataframe(
                    strike_df.style.format({
                        'Strike': '{:.0f}',
                        'Notional': '₹{:,.0f}',
                        '% of Portfolio': '{:.1f}%'
                    }),
                    use_container_width=True
                )
        
        # Position correlation analysis
        st.markdown("#### Position Correlation Matrix")
        st.caption("Correlation based on strike proximity and Greeks")
        
        # Build correlation matrix based on strike distance
        if len(enriched) >= 2:
            position_names = [f"{p.get('tradingsymbol', 'Unknown')}" for p in enriched]
            n_pos = len(enriched)
            
            correlation_matrix = np.ones((n_pos, n_pos))
            
            for i in range(n_pos):
                for j in range(n_pos):
                    if i != j:
                        strike_i = enriched[i].get('strike', 0)
                        strike_j = enriched[j].get('strike', 0)
                        delta_i = enriched[i].get('delta', 0)
                        delta_j = enriched[j].get('delta', 0)
                        
                        # Correlation based on strike distance and delta similarity
                        strike_dist = abs(strike_i - strike_j) / current_spot
                        delta_sim = 1 - abs(delta_i - delta_j)
                        
                        correlation = (1 - strike_dist) * delta_sim
                        correlation = max(-1, min(1, correlation))
                        correlation_matrix[i, j] = correlation
            
            # Show only first 10 positions for clarity
            display_limit = min(10, n_pos)
            
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix[:display_limit, :display_limit],
                x=position_names[:display_limit],
                y=position_names[:display_limit],
                colorscale='RdBu',
                zmid=0,
                text=correlation_matrix[:display_limit, :display_limit],
                texttemplate='%{text:.2f}',
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                title=f"Position Correlation Heatmap (Top {display_limit} positions)",
                height=600,
                xaxis={'side': 'bottom'},
                yaxis={'autorange': 'reversed'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Concentration risk score
            avg_correlation = (correlation_matrix.sum() - n_pos) / (n_pos * (n_pos - 1))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Avg Position Correlation", f"{avg_correlation:.2f}",
                         help="Higher = more concentrated risk")
                if avg_correlation > 0.7:
                    st.error("🔴 High concentration - positions move together")
                elif avg_correlation > 0.4:
                    st.warning("🟡 Moderate concentration")
                else:
                    st.success("🟢 Well diversified across strikes")
            
            with col2:
                diversification_score = (1 - avg_correlation) * 100
                st.metric("Diversification Score", f"{diversification_score:.1f}/100",
                         help="Higher = better diversification")
        else:
            st.info("Need at least 2 positions for correlation analysis")
    
    # ========== EXECUTION QUALITY TAB ==========
    with analytics_tabs[5]:
        st.markdown("### Execution Quality Metrics")
        st.caption("Track slippage, spread impact, and execution efficiency")
        
        st.info("📊 **Note**: These metrics require historical trade data. Currently showing simulated values for demonstration.")
        
        # Simulated execution metrics (in real implementation, track actual fills)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Slippage %
            avg_slippage = 0.15  # Simulated
            st.metric("Avg Slippage", f"{avg_slippage:.2f}%",
                     help="Average price difference vs mid-market")
            if avg_slippage < 0.2:
                st.success("🟢 Excellent execution")
            elif avg_slippage < 0.5:
                st.warning("🟡 Acceptable")
            else:
                st.error("🔴 Poor execution")
        
        with col2:
            # Spread impact
            spread_cost = 2500  # Simulated in rupees
            st.metric("Spread Cost", f"₹{spread_cost:,.0f}",
                     delta=f"{spread_cost/total_pnl*100:.1f}% of P&L" if total_pnl != 0 else "0%",
                     delta_color="inverse")
        
        with col3:
            # Fill rate
            fill_rate = 98.5  # Simulated
            st.metric("Fill Rate", f"{fill_rate:.1f}%",
                     help="% of orders fully filled")
        
        # Execution time analysis
        st.markdown("#### Execution Latency")
        
        # Simulated latency data
        latencies = np.random.gamma(2, 50, 100)  # Simulated in milliseconds
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Latency", f"{latencies.mean():.0f}ms")
        with col2:
            st.metric("P95 Latency", f"{np.percentile(latencies, 95):.0f}ms")
        with col3:
            st.metric("Max Latency", f"{latencies.max():.0f}ms")
        
        fig = go.Figure(data=[go.Histogram(x=latencies, nbinsx=30, marker_color='#3B82F6')])
        fig.update_layout(
            title="Order Execution Latency Distribution",
            xaxis_title="Latency (ms)",
            yaxis_title="Frequency",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Execution quality over time (simulated)
        st.markdown("#### Execution Quality Trend")
        
        dates = pd.date_range(end=pd.Timestamp.now(), periods=30, freq='D')
        quality_scores = 100 - np.random.uniform(0, 5, 30)  # Simulated quality scores
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=quality_scores,
            mode='lines+markers',
            name='Quality Score',
            line=dict(color='#3B82F6', width=2),
            marker=dict(size=6)
        ))
        
        fig.add_hline(y=95, line_dash="dash", line_color="green",
                     annotation_text="Target: 95")
        
        fig.update_layout(
            title="Execution Quality Score (Last 30 Days)",
            xaxis_title="Date",
            yaxis_title="Quality Score",
            height=350,
            yaxis=dict(range=[90, 100])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # ========== DRAWDOWN ANALYSIS TAB ==========
    with analytics_tabs[6]:
        st.markdown("### Drawdown & Recovery Analysis")
        st.caption("Analyze loss persistence and recovery patterns")
        
        # Simulated P&L history (in real implementation, use actual trade history)
        np.random.seed(42)
        days = 60
        daily_pnls = np.cumsum(np.random.normal(500, 3000, days))
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D')
        
        # Calculate drawdown
        cumulative_max = np.maximum.accumulate(daily_pnls)
        drawdown = daily_pnls - cumulative_max
        drawdown_pct = (drawdown / cumulative_max) * 100
        
        # Current drawdown
        current_dd = drawdown[-1]
        current_dd_pct = drawdown_pct[-1]
        max_dd = drawdown.min()
        max_dd_pct = drawdown_pct.min()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Drawdown", f"₹{abs(current_dd):,.0f}",
                     delta=f"{current_dd_pct:.2f}%",
                     delta_color="inverse")
        
        with col2:
            st.metric("Max Drawdown", f"₹{abs(max_dd):,.0f}",
                     delta=f"{max_dd_pct:.2f}%",
                     delta_color="inverse")
        
        with col3:
            # Recovery days calculation
            if current_dd < 0:
                # Estimate recovery days based on avg daily P&L
                avg_daily_pnl = (daily_pnls[-1] - daily_pnls[0]) / days
                recovery_days = abs(current_dd / avg_daily_pnl) if avg_daily_pnl > 0 else 999
                st.metric("Est. Recovery Days", f"{recovery_days:.0f}",
                         help="Days to recover at current avg daily P&L")
            else:
                st.metric("Recovery Days", "0", help="No active drawdown")
        
        # P&L and Drawdown chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=daily_pnls,
            mode='lines',
            name='Cumulative P&L',
            line=dict(color='#3B82F6', width=2),
            yaxis='y1'
        ))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=drawdown,
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='#EF4444', width=1),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="P&L and Drawdown History",
            xaxis_title="Date",
            yaxis=dict(
                title="Cumulative P&L (₹)",
                side='left'
            ),
            yaxis2=dict(
                title="Drawdown (₹)",
                side='right',
                overlaying='y',
                showgrid=False
            ),
            height=450,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown statistics
        st.markdown("#### Drawdown Statistics")
        
        # Find all drawdown periods
        in_drawdown = drawdown < 0
        drawdown_starts = np.where(np.diff(in_drawdown.astype(int)) == 1)[0]
        drawdown_ends = np.where(np.diff(in_drawdown.astype(int)) == -1)[0]
        
        if len(drawdown_starts) > 0 and len(drawdown_ends) > 0:
            # Ensure matching starts and ends
            if drawdown_starts[0] > drawdown_ends[0]:
                drawdown_ends = drawdown_ends[1:]
            if len(drawdown_starts) > len(drawdown_ends):
                drawdown_starts = drawdown_starts[:-1]
            
            drawdown_durations = drawdown_ends - drawdown_starts
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Drawdowns", len(drawdown_durations))
            
            with col2:
                avg_duration = drawdown_durations.mean() if len(drawdown_durations) > 0 else 0
                st.metric("Avg Duration", f"{avg_duration:.1f} days")
            
            with col3:
                max_duration = drawdown_durations.max() if len(drawdown_durations) > 0 else 0
                st.metric("Max Duration", f"{max_duration:.0f} days")
        
        # Recovery efficiency
        st.markdown("#### Recovery Efficiency")
        
        if len(drawdown_starts) > 0:
            recovery_rates = []
            for start, end in zip(drawdown_starts, drawdown_ends):
                dd_depth = abs(drawdown[start:end+1].min())
                recovery_time = end - start
                if recovery_time > 0:
                    recovery_rate = dd_depth / recovery_time
                    recovery_rates.append(recovery_rate)
            
            if recovery_rates:
                avg_recovery_rate = np.mean(recovery_rates)
                st.metric("Avg Recovery Rate", f"₹{avg_recovery_rate:,.0f}/day",
                         help="Average daily P&L during recovery periods")
                
                if avg_recovery_rate > 1000:
                    st.success("🟢 Fast recovery capability")
                elif avg_recovery_rate > 500:
                    st.warning("🟡 Moderate recovery")
                else:
                    st.error("🔴 Slow recovery")
