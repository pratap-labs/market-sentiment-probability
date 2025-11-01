"""Market regime analysis tab."""

import streamlit as st
import pandas as pd

import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from views.utils import calculate_market_regime, format_inr


def render_market_regime_tab(options_df: pd.DataFrame, nifty_df: pd.DataFrame):
    """Render comprehensive market regime tab with all key metrics."""
    st.subheader("🌡️ Market Regime Analysis")
    
    regime = calculate_market_regime(options_df, nifty_df)
    
    if not regime:
        st.warning("Market regime data not available. Load NIFTY data and options data first.")
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
    
    # IV vs RV comparison
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
