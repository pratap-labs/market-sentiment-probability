"""Market regime analysis tab."""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.utils import calculate_market_regime, format_inr


# Cache directory
CACHE_DIR = Path(ROOT) / "database" / "derivatives_cache"


def load_from_cache(data_type: str) -> pd.DataFrame:
    """Load data from most recent cache file."""
    cache_file = CACHE_DIR / f"{data_type}_{datetime.now().strftime('%Y-%m-%d')}.csv"
    if cache_file.exists():
        df = pd.read_csv(cache_file)
        return df
    else:
        return pd.DataFrame()


def render_market_regime_tab(options_df: pd.DataFrame = None, nifty_df: pd.DataFrame = None):
    """Render comprehensive market regime tab with all key metrics."""
    st.subheader("🌡️ Market Regime Analysis")

    
    # Add reload data button
    if st.sidebar.button("🔄 Reload Data", key="regime_reload_data", help="Load latest derivatives data from cache"):
        with st.spinner("Loading data from cache..."):
            # Load options data (combine CE and PE)
            df_ce = load_from_cache("nifty_options_ce")
            df_pe = load_from_cache("nifty_options_pe")
            
            if not df_ce.empty and not df_pe.empty:
                options_df = pd.concat([df_ce, df_pe], ignore_index=True)
                st.session_state["options_df_cache"] = options_df
                st.success(f"✅ Loaded {len(options_df)} options records")
            elif not df_ce.empty:
                options_df = df_ce
                st.session_state["options_df_cache"] = options_df
                st.success(f"✅ Loaded {len(options_df)} CE records")
            elif not df_pe.empty:
                options_df = df_pe
                st.session_state["options_df_cache"] = options_df
                st.success(f"✅ Loaded {len(options_df)} PE records")
            else:
                st.warning("⚠️ No options data in cache")
            
            # Load NIFTY OHLCV data
            nifty_df = load_from_cache("nifty_ohlcv")
            if not nifty_df.empty:
                st.session_state["nifty_df_cache"] = nifty_df
                st.success(f"✅ Loaded {len(nifty_df)} NIFTY records")
            else:
                st.warning("⚠️ No NIFTY data in cache")
    
    # Use cached data if available
    if "options_df_cache" in st.session_state:
        options_df = st.session_state["options_df_cache"]
    if "nifty_df_cache" in st.session_state:
        nifty_df = st.session_state["nifty_df_cache"]
    
    # Initialize empty dataframes if None
    if options_df is None:
        options_df = pd.DataFrame()
    if nifty_df is None:
        nifty_df = pd.DataFrame()
    
    regime = calculate_market_regime(options_df, nifty_df)
    
    if not regime:
        st.warning("⚠️ Market regime data not available. Please click 'Reload Data' or go to the 'Derivatives Data' tab to load options data first.")
        st.info("💡 You need both NIFTY OHLCV and Options data for market regime analysis.")
        return
    
    current_spot = regime.get("current_spot", 0)
    
    # ========== SECTION 1: VOLATILITY METRICS ==========
    st.markdown("### 📊 Volatility Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_iv = regime.get('current_iv', 0)
        st.metric(
            "Current IV (ATM)", 
            f"{current_iv:.1%}",
            help="**Implied Volatility** of at-the-money options. Shows market's expectation of future price movement. Higher IV = more expensive options."
        )
    
    with col2:
        iv_rank = regime.get('iv_rank', 0)
        rank_status = "🔴" if iv_rank > 70 else ("🟡" if iv_rank > 50 else "🟢")
        st.metric(
            "IV Rank (90d)", 
            f"{iv_rank:.0f} {rank_status}",
            help="""**IV Rank** = Where current IV sits in 90-day range (0-100%)

Formula: (Current IV - Min IV) / (Max IV - Min IV) × 100

**HIGH IV RANK** 🔴 (>70):
• Example: Current IV = 18%, Min = 12%, Max = 20%
• IV Rank = (18-12)/(20-12) × 100 = 75%
• Options are EXPENSIVE (near 90-day highs)
• ✅ SELL premium: Iron condors, credit spreads, strangles

**LOW IV RANK** 🟢 (<30):
• Example: Current IV = 13%, Min = 12%, Max = 20%
• IV Rank = (13-12)/(20-12) × 100 = 12.5%
• Options are CHEAP (near 90-day lows)
• ✅ BUY premium: Long straddles, debit spreads, calendars

**MID RANGE** 🟡 (30-70): Fair value, neutral strategies"""
        )
    
    with col3:
        realized_vol = regime.get('realized_vol', 0)
        st.metric(
            "Realized Vol (30d)", 
            f"{realized_vol:.1%}",
            help="**Realized Volatility** = Actual price movement over last 30 days (annualized). Compare with IV to see if options are overpriced or underpriced."
        )
    
    with col4:
        vrp = regime.get('vrp', 0)
        vrp_status = "🔴" if vrp > 0.05 else ("🟢" if vrp < -0.05 else "🟡")
        st.metric(
            "Vol Risk Premium", 
            f"{vrp:.1%} {vrp_status}",
            help="""**VRP (Volatility Risk Premium)** = Implied Vol - Realized Vol

Shows if options are overpriced or underpriced vs actual movement.

**NEUTRAL** 🟡 (-5% to +5%): Fair pricing, IV matches recent realized vol"""
        )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        rv_percentile = regime.get('rv_percentile', 50)
        rv_status = "🔴" if rv_percentile > 80 else ("🟡" if rv_percentile > 60 else "🟢")
        st.metric(
            "RV Percentile", 
            f"{rv_percentile:.0f}% {rv_status}",
            help="**RV Percentile** = Where current realized vol ranks in historical range. >80 = high volatility environment, <20 = calm market."
        )
    
    with col2:
        term_structure = regime.get('term_structure', 0)
        ts_status = "🔴" if term_structure < -0.02 else ("🟢" if term_structure > 0 else "🟡")
        ts_label = "Contango" if term_structure > 0 else "Backwardation"
        st.metric(
            "Term Structure", 
            f"{term_structure:.2%}", 
            f"{ts_label} {ts_status}",
            help="""**Term Structure** = How IV changes across expiries (Far IV - Near IV).

**CONTANGO** (Positive, Normal):
• Nov expiry: IV = 15%, Dec expiry: IV = 18% → Term Structure = +3%
• Far options MORE expensive (more time = more uncertainty)
• ✅ Good for: Calendar spreads (sell Nov, buy Dec)
• Normal healthy market

**BACKWARDATION** (Negative, Stress):
• Nov expiry: IV = 22%, Dec expiry: IV = 16% → Term Structure = -6%
• Near options MORE expensive (immediate fear/event)
• ⚠️ Warning: Avoid selling near-term, market expects short-term turbulence
• Happens before: RBI policy, Budget, crashes"""
        )
    
    with col3:
        near_iv = regime.get('near_iv', 0)
        st.metric(
            "Near-term IV", 
            f"{near_iv:.1%}",
            help="IV of current month expiry options."
        )
    
    with col4:
        far_iv = regime.get('far_iv', 0)
        st.metric(
            "Next-term IV", 
            f"{far_iv:.1%}",
            help="IV of next month expiry options."
        )
    
    # ========== SECTION 2: SENTIMENT INDICATORS ==========
    st.markdown("### 🎭 Sentiment Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pcr_oi = regime.get('pcr_oi', 0)
        pcr_status = "🟢" if pcr_oi > 1.2 else ("🔴" if pcr_oi < 0.8 else "🟡")
        pcr_sentiment = "Bullish" if pcr_oi > 1.2 else ("Bearish" if pcr_oi < 0.8 else "Neutral")
        st.metric(
            "PCR (OI)", 
            f"{pcr_oi:.2f} {pcr_status}", 
            f"{pcr_sentiment}",
            help="""**Put-Call Ratio** = Total Put OI / Total Call OI

**BULLISH** 🟢 (PCR > 1.2):
• Example: Put OI = 60L, Call OI = 45L → PCR = 1.33
• More puts than calls = hedging/protection buying
• Traders expect up move, buying puts to protect profits
• ✅ Good for: Selling puts, Bull spreads

**BEARISH** 🔴 (PCR < 0.8):
• Example: Put OI = 35L, Call OI = 50L → PCR = 0.70
• More calls than puts = excessive optimism
• Everyone chasing upside, no protection
• ⚠️ Warning: Complacent market, consider bear spreads

**NEUTRAL** 🟡 (PCR 0.8-1.2): Balanced sentiment, no extreme positioning"""
        )
    
    with col2:
        pcr_volume = regime.get('pcr_volume', 0)
        st.metric(
            "PCR (Volume)", 
            f"{pcr_volume:.2f}",
            help="Put-Call Ratio by trading volume. Shows active trading sentiment."
        )
    
    with col3:
        skew = regime.get('skew', 0)
        # Updated thresholds: ±1% is balanced, >1% is fear, <-1% is complacent
        if abs(skew) <= 0.01:
            skew_status = "🟡"
            skew_sentiment = "Balanced"
        elif skew > 0.01:
            skew_status = "🔴"
            skew_sentiment = "Fear (Put buying)"
        else:
            skew_status = "🟢"
            skew_sentiment = "Complacent"
        st.metric("Volatility Skew", f"{skew:.2%} {skew_status}", f"{skew_sentiment}",
            help="""**Volatility Skew = OTM Put IV - OTM Call IV (both ~3% away from spot)
**Example: NIFTY @ 25,000**

**HIGH SKEW (Fear)** 🔴:
• 24,250 Put (3% OTM): IV = 20%
• 25,750 Call (3% OTM): IV = 15%
• Skew = +5% → FEAR mode! Puts are expensive
• ❌ Don't buy OTM puts (overpriced protection)
• ✅ Sell put spreads (collect rich premium)

**LOW/NEGATIVE SKEW (Complacent)** 🟢:
• 24,250 Put: IV = 14%
• 25,750 Call: IV = 18%
• Skew = -4% → Excessive optimism! Calls expensive
• ⚠️ Warning sign - market too complacent
• ✅ Buy put protection (cheap insurance)

**BALANCED** 🟡:
• Put IV ≈ Call IV (within ±1%)
• Fair pricing, no extreme sentiment""")
    
    with col4:
        put_25d = regime.get('put_25d_iv', 0)
        call_25d = regime.get('call_25d_iv', 0)
        st.metric("Put IV (OTM)", f"{put_25d:.1%}", f"Call: {call_25d:.1%}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        max_pain = regime.get('max_pain_strike', 0)
        pain_dist = ((current_spot - max_pain) / current_spot * 100) if current_spot > 0 else 0
        pain_status = "🟢" if abs(pain_dist) < 1 else ("🟡" if abs(pain_dist) < 2 else "🔴")
        st.metric("Max Pain Strike", f"{max_pain:.0f} {pain_status}", f"{pain_dist:+.1f}% from spot",
        help="""**Max Pain** = Strike where option writers lose the LEAST money at expiry

**How it works:**
• Options sellers (writers) want options to expire worthless
• Max Pain = Strike that causes maximum loss to option buyers
• Market often gravitates toward this level before expiry

**Example:**
• NIFTY Spot = 25,050
• Max Pain = 25,000
• Distance = -0.2% (close to max pain)

**Near Max Pain** 🟢 (<1% away):
• High chance spot moves toward 25,000 by expiry
• Option writers will defend this level
• ✅ Strategy: Sell options near max pain

**Far from Max Pain** 🔴 (>2% away):
• Strong directional move in progress
• Option writers struggling, may need to hedge
• ⚠️ Expect increased volatility near expiry""")
    
    with col2:
        st.metric("Spot Price", format_inr(current_spot, decimals=2))
    
    # ========== SECTION 3: MARKET REGIME CLASSIFICATION ==========
    st.markdown("### 🎯 Market Regime")
    
    regime_name = regime.get('market_regime', 'Unknown')
    
    # Color code the regime
    if "High Vol" in regime_name:
        regime_color = "🔴"
    elif "Low Vol" in regime_name:
        regime_color = "🟢"
    elif "Sell" in regime_name:
        regime_color = "🟡"
    else:
        regime_color = "⚪"
    
    st.markdown(f"### {regime_color} **{regime_name}**")
    
    # Interpretation guide
    with st.expander("📖 How to interpret this regime", expanded=False):
        if "High Vol - Sell Premium" in regime_name:
            st.markdown("""
            **High Volatility - Sell Premium Environment**
            - IV Rank > 70: Options are expensive
            - VRP > 5%: IV significantly higher than realized
            - **Strategy**: Sell options (credit spreads, iron condors, strangles)
            - **Risk**: Be prepared for large moves
            """)
        elif "Low Vol - Buy Premium" in regime_name:
            st.markdown("""
            **Low Volatility - Buy Premium Environment**
            - IV Rank < 30: Options are cheap
            - VRP < -5%: IV lower than realized (underpriced)
            - **Strategy**: Buy options (long straddles, calendars)
            - **Risk**: Time decay will hurt if no movement
            """)
        elif "Neutral" in regime_name:
            st.markdown("""
            **Neutral Market - Balanced Approach**
            - IV Rank 30-70: Fair pricing
            - VRP near 0: IV matches realized
            - **Strategy**: Neutral strategies (iron condors, butterflies)
            - **Risk**: Moderate - watch for regime changes
            """)
        else:
            st.markdown(f"""
            **{regime_name}**
            - Review individual metrics for strategy selection
            - Consider both directional and volatility views
            """)
    
    # ========== SECTION 4: TREND INDICATORS ==========
    st.markdown("### 📈 NIFTY Trend Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sma_20 = regime.get('sma_20', 0)
        sma_dist = ((current_spot - sma_20) / current_spot * 100) if current_spot > 0 and sma_20 > 0 else 0
        sma_status = "🟢" if sma_dist > 0 else "🔴"
        st.metric("20-day SMA", format_inr(sma_20, decimals=2), f"{sma_dist:+.1f}% {sma_status}")
        with st.expander("ℹ️ What is SMA?"):
            st.caption("**Simple Moving Average** of last 20 days. Price above SMA = bullish trend, below = bearish.")
    
    with col2:
        sma_50 = regime.get('sma_50', 0)
        sma_50_dist = ((current_spot - sma_50) / current_spot * 100) if current_spot > 0 and sma_50 > 0 else 0
        sma_50_status = "🟢" if sma_50_dist > 0 else "🔴"
        st.metric("50-day SMA", format_inr(sma_50, decimals=2), f"{sma_50_dist:+.1f}% {sma_50_status}")
        with st.expander("ℹ️ What is 50-day SMA?"):
            st.caption("**50-day SMA** shows medium-term trend. Price above = sustained uptrend.")
    
    with col3:
        rsi = regime.get('rsi', 50)
        rsi_status = "🔴" if rsi > 70 else ("🟢" if rsi < 30 else "🟡")
        rsi_label = "Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral")
        st.metric("RSI (14)", f"{rsi:.0f} {rsi_status}", f"{rsi_label}")
        with st.expander("ℹ️ What is RSI?"):
            st.caption("**Relative Strength Index** (0-100). >70 = overbought (potential reversal down), <30 = oversold (potential bounce up).")
    
    with col4:
        atr = regime.get('atr', 0)
        atr_pct = (atr / current_spot * 100) if current_spot > 0 else 0
        st.metric("ATR (14)", f"{atr:.0f}", f"{atr_pct:.1f}% of spot")
        with st.expander("ℹ️ What is ATR?"):
            st.caption("**Average True Range** = average daily movement in points. Higher ATR = more volatile market, wider stop losses needed.")
    
    # Trend summary
    trend_signals = []
    if sma_dist > 0 and sma_50_dist > 0:
        trend_signals.append("🟢 **Bullish**: Price above both SMAs")
    elif sma_dist < 0 and sma_50_dist < 0:
        trend_signals.append("🔴 **Bearish**: Price below both SMAs")
    else:
        trend_signals.append("🟡 **Mixed**: Price between SMAs")
    
    if rsi > 70:
        trend_signals.append("⚠️ **Caution**: RSI Overbought (potential reversal)")
    elif rsi < 30:
        trend_signals.append("⚠️ **Caution**: RSI Oversold (potential bounce)")
    
    st.markdown("#### Trend Summary")
    for signal in trend_signals:
        st.markdown(signal)
    
    # ========== SECTION 5: ACTIONABLE INSIGHTS ==========
    st.markdown("### 💡 Actionable Insights")
    
    insights = []
    
    # PCR insights
    if pcr_oi > 1.3:
        insights.append("🟢 **PCR > 1.3**: Heavy put buying suggests bullish sentiment or hedging. Consider selling puts or bull spreads.")
    elif pcr_oi < 0.7:
        insights.append("🔴 **PCR < 0.7**: Heavy call buying suggests excessive optimism. Consider selling calls or bear spreads.")
    
    # Skew insights
    if skew > 0.05:
        insights.append("🔴 **High Put Skew**: Market fear elevated. OTM puts expensive - avoid buying, consider selling put spreads.")
    elif skew < -0.03:
        insights.append("⚠️ **Negative Skew**: Calls more expensive than puts - rare condition, potential warning sign.")
    
    # IV Rank insights
    if iv_rank > 80:
        insights.append("🔴 **IV Rank > 80**: Options very expensive. Prime time to SELL premium (iron condors, credit spreads).")
    elif iv_rank < 20:
        insights.append("🟢 **IV Rank < 20**: Options very cheap. Good time to BUY options (long straddles, debit spreads).")
    
    # Term structure insights
    if term_structure < -0.03:
        insights.append("🔴 **Backwardation**: Near-term vol higher than far - suggests stress. Avoid selling near-term options.")
    elif term_structure > 0.05:
        insights.append("🟢 **Steep Contango**: Good environment for calendar spreads (sell near, buy far).")
    
    # Max pain insights
    if abs(pain_dist) < 1.5:
        insights.append(f"📍 **Near Max Pain ({max_pain:.0f})**: Expect spot to gravitate toward max pain before expiry.")
    
    # VRP insights
    if vrp > 0.08:
        insights.append("🟡 **High VRP**: IV much higher than realized - premium sellers have edge, but watch for gap moves.")
    elif vrp < -0.05:
        insights.append("🟡 **Negative VRP**: IV underpricing risk - good for buying options if expecting volatility expansion.")
    
    if insights:
        for insight in insights:
            st.markdown(insight)
    else:
        st.info("Market conditions are balanced. Monitor for regime changes.")
    
    # ========== SECTION 6: CHARTS ==========
    st.markdown("### 📊 Visual Analysis")
    
    # Volatility over time with Gaussian prediction
    st.markdown("#### 📈 Volatility Over Time with Gaussian Prediction")
    
    try:
        import plotly.graph_objects as go
        from scipy import stats
        import numpy as np
        
        # Get historical NIFTY data for volatility calculation
        if not nifty_df.empty and 'close' in nifty_df.columns:
            # Ensure date column exists and is datetime
            if 'date' not in nifty_df.columns and nifty_df.index.name == 'date':
                nifty_df = nifty_df.reset_index()
            
            # Convert date to datetime if it's not already
            if 'date' in nifty_df.columns:
                nifty_df['date'] = pd.to_datetime(nifty_df['date'])
            else:
                st.warning("Date column not found in NIFTY data")
                raise ValueError("No date column")
            
            # Calculate rolling volatility (30-day window)
            nifty_returns = nifty_df['close'].pct_change().dropna()
            rolling_vol = nifty_returns.rolling(window=30).std() * np.sqrt(252)  # Annualized
            
            # Get last 90 days of data - create DataFrame with date from nifty_df
            # Align rolling_vol with nifty_df dates
            vol_data = pd.DataFrame({
                'date': nifty_df['date'].iloc[rolling_vol.index],
                'volatility': rolling_vol.values * 100  # Convert to percentage
            }).dropna()
            
            # Get last 90 rows
            vol_data = vol_data.tail(90).reset_index(drop=True)
            
            if len(vol_data) > 30:
                # Fit Gaussian distribution to historical volatility
                vol_values = vol_data['volatility'].dropna()
                mu, sigma = stats.norm.fit(vol_values)
                
                # Create prediction bands (±1σ, ±2σ)
                upper_1sigma = mu + sigma
                lower_1sigma = mu - sigma
                upper_2sigma = mu + 2*sigma
                lower_2sigma = max(0, mu - 2*sigma)  # Volatility can't be negative
                
                # Define regime thresholds
                # Low vol: < lower_1sigma
                # Medium vol: lower_1sigma to upper_1sigma
                # High vol: > upper_1sigma
                low_vol_threshold = lower_1sigma
                high_vol_threshold = upper_1sigma
                
                # Create the plot
                fig = go.Figure()
                
                # Add regime background shading
                # Low volatility zone (bottom to low threshold) - Green
                fig.add_hrect(
                    y0=0, y1=low_vol_threshold,
                    fillcolor="rgba(0, 255, 0, 0.1)",
                    layer="below",
                    line_width=0,
                    annotation_text="LOW VOL<br>Gamma Scalping",
                    annotation_position="right",
                    annotation_font_size=10,
                    annotation_font_color="rgba(0, 255, 0, 0.7)"
                )
                
                # Medium volatility zone (low to high threshold) - Yellow
                fig.add_hrect(
                    y0=low_vol_threshold, y1=high_vol_threshold,
                    fillcolor="rgba(255, 255, 0, 0.1)",
                    layer="below",
                    line_width=0,
                    annotation_text="MEDIUM VOL<br>Short Vol",
                    annotation_position="right",
                    annotation_font_size=10,
                    annotation_font_color="rgba(255, 255, 0, 0.7)"
                )
                
                # High volatility zone (high threshold to top) - Red
                fig.add_hrect(
                    y0=high_vol_threshold, y1=max(vol_data['volatility'].max(), upper_2sigma) * 1.1,
                    fillcolor="rgba(255, 0, 0, 0.1)",
                    layer="below",
                    line_width=0,
                    annotation_text="HIGH VOL<br>Short Vol",
                    annotation_position="right",
                    annotation_font_size=10,
                    annotation_font_color="rgba(255, 0, 0, 0.7)"
                )
                
                # Historical volatility line - main data
                fig.add_trace(go.Scatter(
                    x=vol_data['date'],
                    y=vol_data['volatility'],
                    mode='lines',
                    name='Realized Volatility',
                    line=dict(color='#00BFFF', width=3),  # Bright blue
                    hovertemplate='<b>%{x|%d %b %Y}</b><br>RV: %{y:.2f}%<extra></extra>'
                ))
                
                # Mean line - green dashed
                fig.add_trace(go.Scatter(
                    x=vol_data['date'],
                    y=[mu] * len(vol_data),
                    mode='lines',
                    name=f'Mean: {mu:.1f}%',
                    line=dict(color='#00FF00', width=2, dash='dash'),
                    hovertemplate='<b>Mean</b>: %{y:.2f}%<extra></extra>'
                ))
                
                # Current IV line - purple dashed (reference line)
                fig.add_trace(go.Scatter(
                    x=vol_data['date'],
                    y=[current_iv * 100] * len(vol_data),
                    mode='lines',
                    name=f'Current IV: {current_iv*100:.1f}%',
                    line=dict(color='#FF00FF', width=2, dash='dash'),
                    hovertemplate='<b>Current Implied Vol</b>: %{y:.2f}%<extra></extra>'
                ))
                
                # +1σ line - orange dotted
                fig.add_trace(go.Scatter(
                    x=vol_data['date'],
                    y=[upper_1sigma] * len(vol_data),
                    mode='lines',
                    name=f'+1σ: {upper_1sigma:.1f}%',
                    line=dict(color='#FFA500', width=1.5, dash='dot'),
                    hovertemplate='<b>+1σ</b>: %{y:.2f}%<extra></extra>'
                ))
                
                # -1σ line - orange dotted
                fig.add_trace(go.Scatter(
                    x=vol_data['date'],
                    y=[lower_1sigma] * len(vol_data),
                    mode='lines',
                    name=f'-1σ: {lower_1sigma:.1f}%',
                    line=dict(color='#FFA500', width=1.5, dash='dot'),
                    hovertemplate='<b>-1σ</b>: %{y:.2f}%<extra></extra>'
                ))
                
                # +2σ line - red dotted
                fig.add_trace(go.Scatter(
                    x=vol_data['date'],
                    y=[upper_2sigma] * len(vol_data),
                    mode='lines',
                    name=f'+2σ: {upper_2sigma:.1f}%',
                    line=dict(color='#FF0000', width=1.5, dash='dot'),
                    hovertemplate='<b>+2σ</b>: %{y:.2f}%<extra></extra>'
                ))
                
                # -2σ line - red dotted
                fig.add_trace(go.Scatter(
                    x=vol_data['date'],
                    y=[lower_2sigma] * len(vol_data),
                    mode='lines',
                    name=f'-2σ: {lower_2sigma:.1f}%',
                    line=dict(color='#FF0000', width=1.5, dash='dot'),
                    hovertemplate='<b>-2σ</b>: %{y:.2f}%<extra></extra>'
                ))
                
                # Get date range for title
                start_date = vol_data['date'].min().strftime('%d %b %Y')
                end_date = vol_data['date'].max().strftime('%d %b %Y')
                
                fig.update_layout(
                    title={
                        'text': f"90-Day Realized Volatility ({start_date} to {end_date})",
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 20, 'color': 'white'}
                    },
                    xaxis_title="Date",
                    yaxis_title="Volatility (%)",
                    height=600,
                    hovermode='x unified',
                    showlegend=True,
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        bgcolor="rgba(0,0,0,0.7)",
                        bordercolor="white",
                        borderwidth=1,
                        font=dict(size=11)
                    ),
                    plot_bgcolor='rgba(20,20,20,0.5)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128,128,128,0.3)',
                        showline=True,
                        linewidth=2,
                        linecolor='white',
                        tickformat='%d %b %y',
                        dtick=7*24*60*60*1000,  # Weekly ticks
                        tickangle=-45
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128,128,128,0.3)',
                        showline=True,
                        linewidth=2,
                        linecolor='white',
                        zeroline=False
                    ),
                    font=dict(color='white', size=12)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretation
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Gaussian Mean (μ)", f"{mu:.2f}%")
                    st.caption("Expected volatility level")
                
                with col2:
                    st.metric("Std Dev (σ)", f"{sigma:.2f}%")
                    st.caption("Volatility of volatility")
                
                with col3:
                    current_rv = vol_data['volatility'].iloc[-1]
                    z_score = (current_rv - mu) / sigma if sigma > 0 else 0
                    st.metric("Current Z-Score", f"{z_score:.2f}")
                    st.caption("How many σ from mean")
                
                # Actionable insights based on Gaussian analysis
                st.markdown("#### 🎯 Volatility Regime Interpretation")
                
                if z_score > 2:
                    st.error(f"""
                    **🔴 Extreme High Volatility (>{upper_2sigma:.1f}%)**
                    - Current RV is {z_score:.1f}σ above normal
                    - Only ~2.5% of time volatility this high
                    - **Action**: Prime time to SELL premium (very expensive options)
                    - **Warning**: Large moves happening, use wider stops
                    """)
                elif z_score > 1:
                    st.warning(f"""
                    **🟡 High Volatility ({lower_1sigma:.1f}% - {upper_2sigma:.1f}%)**
                    - Current RV is {z_score:.1f}σ above normal
                    - Options likely expensive
                    - **Action**: Favor selling strategies, be selective on buys
                    """)
                elif z_score < -1:
                    st.success(f"""
                    **🟢 Low Volatility (< {lower_1sigma:.1f}%)**
                    - Current RV is {abs(z_score):.1f}σ below normal
                    - Options likely cheap
                    - **Action**: Good time to BUY options (cheap protection/speculation)
                    - **Watch for**: Volatility expansion (reversion to mean)
                    """)
                else:
                    st.info(f"""
                    **⚪ Normal Volatility Range ({lower_1sigma:.1f}% - {upper_1sigma:.1f}%)**
                    - Current RV is {abs(z_score):.1f}σ from mean
                    - Within 68% probability band
                    - **Action**: Neutral strategies, wait for extremes
                    """)
                
                # Mean reversion probability
                if abs(z_score) > 1:
                    reversion_prob = stats.norm.cdf(0) - stats.norm.cdf(z_score)
                    st.markdown(f"""
                    📊 **Mean Reversion Probability**: {abs(reversion_prob)*100:.1f}%
                    
                    Volatility tends to revert to mean ({mu:.1f}%). Current extreme suggests 
                    {'decrease' if z_score > 0 else 'increase'} in volatility likely over next 2-4 weeks.
                    """)
                
            else:
                st.warning("Not enough historical data for Gaussian analysis (need 30+ days)")
        else:
            st.warning("NIFTY data not available for volatility analysis")
    
    except Exception as e:
        st.error(f"Error in volatility analysis: {e}")
    
    # IV vs RV comparison
    st.markdown("#### 📊 IV vs RV Comparison")
    col1, col2 = st.columns(2)
    
    with col1:
        iv_rv_data = pd.DataFrame({
            "Metric": ["Implied Vol", "Realized Vol"],
            "Value": [current_iv * 100, realized_vol * 100]
        })
        st.bar_chart(iv_rv_data.set_index("Metric"))
        st.caption("IV vs RV Comparison (%)")
    
    with col2:
        # PCR visualization
        pcr_data = pd.DataFrame({
            "Type": ["Puts (OI)", "Calls (OI)"],
            "Value": [pcr_oi / (1 + pcr_oi) * 100, 100 / (1 + pcr_oi)]
        })
        st.bar_chart(pcr_data.set_index("Type"))
        st.caption("Put-Call Ratio Distribution")
    
    # ========== REGIME CLASSIFICATION SUMMARY ==========
    st.markdown("---")
    st.markdown("## 🎯 Regime Classification & Trading Strategy")
    
    # Determine current regime based on realized volatility
    try:
        if not nifty_df.empty and 'close' in nifty_df.columns:
            # Calculate current realized volatility
            nifty_returns = nifty_df['close'].pct_change().dropna()
            rolling_vol = nifty_returns.rolling(window=30).std() * np.sqrt(252)
            current_rv = rolling_vol.iloc[-1] * 100
            
            # Get thresholds from earlier calculation
            vol_values = rolling_vol.dropna() * 100
            mu, sigma = stats.norm.fit(vol_values)
            low_vol_threshold = mu - sigma
            high_vol_threshold = mu + sigma
            
            # Classify regime
            if current_rv < low_vol_threshold:
                regime = "LOW VOLATILITY"
                regime_color = "🟢"
                strategy = "Gamma Scalping"
                explanation = """
                **Strategy: Gamma Scalping**
                - Volatility is compressed and likely to expand
                - Buy ATM straddles/strangles to capture gamma
                - Delta-hedge frequently to capture profits from price swings
                - Look for breakout opportunities
                - Options are relatively cheap - good time to buy
                """
                risk_note = "⚠️ Risk: Market may stay range-bound longer than expected"
            elif current_rv > high_vol_threshold:
                regime = "HIGH VOLATILITY"
                regime_color = "🔴"
                strategy = "Short Vol (Premium Selling)"
                explanation = """
                **Strategy: Short Volatility (Premium Selling)**
                - Volatility is elevated and likely to contract
                - Sell OTM options (credit spreads, iron condors)
                - Use defined risk strategies due to higher realized moves
                - Premium is expensive - advantage to sellers
                - Watch for mean reversion back to normal levels
                """
                risk_note = "⚠️ Risk: Use wider stops, large moves happening. Consider hedging."
            else:
                regime = "MEDIUM VOLATILITY"
                regime_color = "🟡"
                strategy = "Short Vol (Selective Premium Selling)"
                explanation = """
                **Strategy: Short Volatility (Selective)**
                - Volatility in normal range with slight bias to contract
                - Sell slightly OTM options or credit spreads
                - Good risk-reward for premium selling strategies
                - Monitor for regime shifts to extremes
                - Balance directional and non-directional strategies
                """
                risk_note = "⚠️ Risk: Watch for regime change - be ready to adjust"
            
            # Display regime summary
            st.markdown(f"### {regime_color} Current Regime: **{regime}**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Current Realized Vol",
                    f"{current_rv:.2f}%",
                    delta=f"{current_rv - mu:.2f}% vs mean"
                )
            
            with col2:
                st.metric(
                    "Regime Boundaries",
                    f"{low_vol_threshold:.1f}% - {high_vol_threshold:.1f}%",
                    delta="±1σ range"
                )
            
            with col3:
                st.metric(
                    "Recommended Strategy",
                    strategy,
                    delta=None
                )
            
            # Strategy details
            st.markdown(explanation)
            st.info(risk_note)
            
            # Additional context
            st.markdown("#### 📋 Strategy Guidelines by Regime")
            
            guidelines_df = pd.DataFrame({
                "Regime": ["🟢 Low Vol", "🟡 Medium Vol", "🔴 High Vol"],
                "Range": [f"< {low_vol_threshold:.1f}%", f"{low_vol_threshold:.1f}% - {high_vol_threshold:.1f}%", f"> {high_vol_threshold:.1f}%"],
                "Primary Strategy": ["Gamma Scalping", "Short Vol (Selective)", "Short Vol (Aggressive)"],
                "Action": ["Buy ATM Options", "Sell OTM Options", "Sell Premium w/ Hedges"],
                "Market Outlook": ["Range-bound → Breakout", "Normal Range", "High Moves → Calm"]
            })
            
            st.dataframe(guidelines_df, use_container_width=True, hide_index=True)
            
        else:
            st.warning("Unable to calculate regime classification - NIFTY data not available")
    
    except Exception as e:
        st.error(f"Error calculating regime summary: {e}")
