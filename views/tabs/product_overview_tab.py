"""Comprehensive product documentation for investors."""

import streamlit as st


def render_product_overview_tab():
    """Render comprehensive product documentation for investors."""
    st.header("✨ TBA Institutional Options Risk Management System")
    st.caption("Systematic Short-Volatility Income with Disciplined Risk Controls")

    st.sidebar.caption("No controls for this tab.")
    
    # Executive Summary
    st.markdown("---")
    st.subheader("📋 Executive Summary")
    st.markdown(
        """
        **TBA** is an institutional-grade options income strategy that systematically sells premium on index options 
        while managing tail risk through multi-layered controls. The system targets **10-15% annualized returns** 
        with **4-6% maximum drawdown risk** by harvesting time decay (theta) and maintaining positive expected value 
        across weekly trade cycles.
        
        Unlike directional trading, this is a **process-driven income business** that generates consistent cash flow 
        from option premium while protecting capital through real-time risk monitoring and automated position limits.
        """
    )
    
    # What It Is
    st.markdown("---")
    st.subheader("🎯 What This System Does")
    st.markdown(
        """
        TBA is a **short-volatility income strategy** that operates like an insurance company:
        
        - **Premium Collection**: Sell options (primarily out-of-the-money puts and calls) to collect upfront premium
        - **Time Decay Harvesting**: Earn daily theta as options lose value approaching expiration
        - **Risk-Adjusted Sizing**: Size positions based on capital allocated to risk buckets (Low/Med/High)
        - **Tail Risk Management**: Continuously monitor and limit extreme loss scenarios (ES99)
        - **Zone Discipline**: Ensure trades stay within safe greek exposure ranges (theta/gamma/vega zones)
        
        **This is NOT**:
        - Market timing or directional betting
        - Unlimited risk option selling
        - High-frequency trading
        - Leveraged speculation
        
        **This IS**:
        - Systematic premium harvesting with defined risk
        - Business-like income generation
        - Probability-driven trade selection
        - Multi-layered risk governance
        """
    )
    
    # How It Makes Money
    st.markdown("---")
    st.subheader("💰 How Returns Are Generated")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Primary Return Drivers:**")
        st.markdown(
            """
            1. **Theta Decay** (70-80% of returns)
               - Daily time decay on short options
               - Target: 150-220 theta per 1L capital
               - Compounds over weekly cycles
            
            2. **Premium Collection** (Entry edge)
               - Sell options with positive expected PnL
               - Target reward/risk ratio > 0.15
               - Mean PnL must exceed tail loss risk
            
            3. **Probability Edge**
               - Sell OTM options with <50% prob of profit for buyer
               - Statistical edge from implied vol > realized vol
               - Mean reversion in volatility regimes
            """
        )
    
    with col2:
        st.markdown("**Return Expectations:**")
        st.markdown(
            """
            **Conservative Scenario** (Low risk bucket focus)
            - Monthly: 0.8% - 1.2%
            - Annualized: 10% - 15%
            - Max drawdown: 3% - 4%
            
            **Balanced Scenario** (Mixed buckets)
            - Monthly: 1.2% - 1.8%
            - Annualized: 15% - 22%
            - Max drawdown: 4% - 6%
            
            **Aggressive Scenario** (Higher risk buckets)
            - Monthly: 1.8% - 2.5%
            - Annualized: 22% - 30%
            - Max drawdown: 6% - 8%
            
            *Returns highly dependent on risk appetite and ES99 limits*
            """
        )
    
    # Risk Management Framework
    st.markdown("---")
    st.subheader("🛡️ Multi-Layer Risk Management Framework")
    
    st.markdown("**Layer 1: Capital Allocation & Bucket Limits**")
    st.markdown(
        """
        Capital is divided into three risk buckets with distinct ES99 limits:
        
        | Bucket | Capital Allocation | ES99 Limit | Character | Use Case |
        |--------|-------------------|------------|-----------|----------|
        | **Low** | 50% | 2% | Conservative theta harvesting | Stable base income |
        | **Med** | 30% | 3% | Balanced risk-reward | Core trades |
        | **High** | 20% | 5% | Higher premium capture | Opportunistic |
        
        **ES99 (Expected Shortfall 99%)** = Average loss in worst 1% of scenarios
        - Calculated using greek-based stress scenarios
        - Weighted by historical market regime frequencies
        - Hard limit: Portfolio ES99 must stay below 4% of total capital
        """
    )
    
    st.markdown("**Layer 2: Zone-Based Greek Controls (Per 1L Capital)**")
    st.markdown(
        """
        Even if ES99 passes, trades must stay within safe greek exposure zones:
        
        **Theta Zones** (Income generation)
        - Z1 (Safe): 120-180 per 1L → Optimal income/risk balance
        - Z2 (Acceptable): 180-220 per 1L → Higher income, manageable risk
        - Z3 (Stretched): 220-300 per 1L → Premium chase, tight monitoring
        - OUT: <120 (too safe) or >300 (excessive risk)
        
        **Gamma Zones** (Convexity risk)
        - Z1 (Safe): 0 to -0.020 per 1L → Minimal spot sensitivity
        - Z2 (Acceptable): -0.020 to -0.035 per 1L → Controlled acceleration
        - Z3 (Stretched): -0.035 to -0.055 per 1L → Near-spot risk
        - OUT: >-0.020 (too hedged) or <-0.055 (gamma bomb)
        
        **Vega Zones** (Vol regime dependent)
        - Low IV: Focus on small short vega positions
        - Mid IV: Balanced vega exposure
        - High IV: Aggressive short vega (vol normalization)
        - Zones adapt to current IV percentile
        """
    )
    
    st.markdown("**Layer 3: Trade-Level Gates (Pass/Watch/Fail)**")
    st.markdown(
        """
        Every trade evaluated through three decision gates:
        
        **G1: Survivability** (Can we survive tail events?)
        - P1 breach day ≤ 2 days = FAIL (too fragile)
        - Prob(breach) > ES99 limit = FAIL
        - P1 horizon loss > 1.25× ES99 limit = FAIL
        
        **G2: Speed of Damage** (How fast can we get hurt?)
        - Low bucket: ANY P10 breach = FAIL
        - Med bucket: P10 breach ≤ 6 days = FAIL
        - High bucket: P10 breach ≤ 3 days = FAIL
        - Portfolio: P10 breach ≤ 4 days = FAIL
        
        **G3: Asymmetry** (Is risk/reward acceptable?)
        - Tail ratio |P1|/|P50| thresholds by bucket
        - Reward/Risk ratio must be > 0.15
        - Mean PnL must justify tail loss exposure
        
        Trade must PASS all three gates to be added to portfolio
        """
    )
    
    # Navigation Guide
    st.markdown("---")
    st.subheader("📊 Dashboard Navigation Guide")
    
    st.markdown(
        """
        The system is organized into focused tabs for different workflows:
        
        **Core Tabs** (Daily Operations):
        - **📊 Portfolio**: Real-time P&L, margin usage, capital performance, greeks summary, positions table
        - **Risk Buckets (50/30/20)**: Multi-level risk hierarchy (Portfolio → Bucket → Trade → Meta)
        - **🔬 Pre-Trade Analysis**: Stress test hypothetical trades before entry
        - **🌡️ Market Regime**: IV regime, drift, volatility analysis
        - **📈 Historical Performance**: P&L analysis, trade outcome distribution
        
        **Supporting Tabs**:
        - **📈 Equities**: Equity holdings tracking (separate from options)
        - **💾 Data Source**: NIFTY options chain data browser
        - **✨ Product Overview**: This documentation tab
        """
    )
    
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Portfolio Tab", "Risk Buckets Tab", "🔬 Pre-Trade Analysis", "Supporting Tabs"])
    
    with tab1:
        st.markdown("**� Portfolio Dashboard - Live Position Monitoring**")
        st.markdown(
            """
            **Purpose**: Real-time portfolio health snapshot and position details
            
            **Main Sections**:
            
            1. **Capital & Performance**
               - Margin available vs used
               - Total P&L and ROI
               - Days in trade, daily theta capture
               - Annualized return projections
            
            2. **Greeks & Risk Summary**
               - Portfolio delta, gamma, vega, theta
               - VaR (Value at Risk) metrics
               - Stress scenario P&L (±1%, ±2%, ±3% spot moves)
               
            3. **Market Weather**
               - Current IV percentile and regime (Low/Mid/High)
               - Market drift and signal
               - Alignment status with portfolio positioning
            
            4. **Alignment & Recommendations**
               - Portfolio health status (Pass/Watch/Fail)
               - Regime-based position recommendations
               - Risk-reward efficiency score
            
            5. **Positions Table**
               - All active options positions
               - Expiry, strike, greeks, P&L per position
               - Filterable and sortable
            
            6. **🧪 Greeks Debug** (Collapsed expander at bottom)
               - Manual spot override for greek recalculation
               - IV calculation diagnostics
               - Position greeks table with expiry filters
               - Used for troubleshooting greek calculations
            
            **When to Use**:
            - Start of day: Quick portfolio health check
            - Throughout day: Monitor P&L and greek exposures
            - Before adding trades: Check current exposures
            - Troubleshooting: Use Greeks Debug for calculation verification
            """
        )
    
    with tab2:
        st.markdown("**Risk Buckets (50/30/20) - Four-Level Risk Hierarchy**")
        st.markdown(
            """
            **Purpose**: Institutional risk management across Portfolio → Bucket → Trade → Meta levels
            
            **SUBTAB 1: Portfolio Level** (Top-down aggregate view)
            
            *Layer 1 - Forward Simulation*
            - Gate configuration and decision logic display
            - Scenario distribution table: Mean/Median/P1/P5/P10 PnL
            - Days to breach analysis
            - Probability metrics (loss, breach, survival)
            
            *Layer 2 - ES99 Portfolio Metrics*
            - Portfolio ES99 vs 4% hard limit
            - Kill switch status and breach analysis
            - ES99 contribution by bucket (stacked bar chart)
            - Top 10 trade-weeks by ES99 contribution
            
            *Layer 3 - Greek-Based Zone Analysis*
            - Current IV regime classification
            - Normalized greeks per ₹1L capital
            - Zone classification table (Z1/Z2/Z3/OUT for theta/gamma/vega)
            - Bucket definitions and ES99 limits reference table
            
            *Historical Bucket Analysis*
            - Bucket probability distribution by IV regime
            - Drift distribution histogram
            - Historical NIFTY OHLC candlestick charts by bucket expiry
            
            **When to Use Portfolio Level**:
            - Morning check: Portfolio ES99 vs 4% limit
            - Pre-trade: Forward sim to see if new trade fits
            - Post-loss: Zone analysis to understand risk drift
            - Strategic: Historical analysis for regime-based allocation
            
            ---
            
            **SUBTAB 2: Bucket Level** (Capital allocation control)
            
            *Bucket Status Cards* (3-column grid)
            - Capital usage and ES99 for each bucket (Low/Med/High)
            - Kill switch warnings when bucket exceeds its limit
            - Visual status indicators
            
            *Bucket Tables & Charts*
            - Capital allocation by bucket (50% Low / 30% Med / 20% High)
            - ES99 risk and trade count by bucket
            - ES99 bar chart and trade count bar chart
            
            *Forward Simulation by Bucket*
            - Scenario analysis per bucket with gate configuration
            - Mean/Median/P1/P5/P10 PnL tables
            - Days to breach and probability metrics per bucket
            
            **When to Use Bucket Level**:
            - Adding trades: Which bucket has ES99 capacity?
            - Bucket limit breach: Which trades to cut?
            - Rebalancing: Verify 50/30/20 allocation
            - Gate failures: Isolate which bucket is problematic
            
            **Decision Logic**:
            - Low bucket ES99 > 2%? → Cut lowest-performing trades
            - Med bucket ES99 > 3%? → Reduce position sizes
            - High bucket ES99 > 5%? → Exit highest-risk trades immediately
            
            ---
            
            **SUBTAB 3: Trade Level** (Individual trade quality)
            
            *Chart Controls & Visualization*
            - Toggle: Use saved trade groups vs individual positions
            - Tail Loss vs Mean PnL scatter chart
            - X-axis: Expected PnL (want positive = right side)
            - Y-axis: Tail Loss/ES99 (want low = top of chart)
            - Color: Reward/Risk ratio (green good, red bad)
            - RR threshold lines: 0.15, 0.30, 0.40
            - Risk zones: A/B/C/D (B = sweet spot)
            
            *Trade Table*
            - All positions with trade metrics
            - Zone labels (Z1/Z2/Z3/OUT) for theta/gamma/vega
            - Gate status (PASS/WATCH/FAIL)
            - Greeks per ₹1L capital
            - Forward sim metrics per trade
            
            *Trade Grouping* (Manual expander)
            - Combine related positions (spreads, rolls, etc.)
            - View grouped trades as single units
            - Combined risk metrics for position groups
            
            **When to Use Trade Level**:
            - Daily review: Which trades have negative mean PnL? → Exit
            - Risk management: Which trades contribute most to ES99?
            - Zone monitoring: Are trades in Z1/Z2 (good) or Z3/OUT (bad)?
            - Exit decisions: Would I enter this trade now at current metrics?
            
            **Decision Logic**:
            - Mean PnL < 0? → EXIT (no longer positive EV)
            - Reward/Risk < 0.15? → WATCH or EXIT (insufficient edge)
            - Any FAIL gate? → Cannot add new similar trades
            - Zone OUT? → Don't add capital, review existing positions
            
            ---
            
            **SUBTAB 4: Meta Information**
            - Methodology documentation
            - Zone definitions and thresholds
            - Gate logic explanations
            - Bucket configuration details
            - Calculation formulas and assumptions
            """
        )
    
    with tab3:
        st.markdown("**🔬 Pre-Trade Analysis - Stress Testing Before Entry**")
        st.markdown(
            """
            **Purpose**: Analyze hypothetical positions BEFORE entering them
            
            **Main Features**:
            
            1. **Manual Greek Inputs**
               - Enter hypothetical position greeks (delta, gamma, vega, theta)
               - Specify quantity, strike, expiry
               - Test "what-if" scenarios without executing trades
            
            2. **Zone Preview**
               - See which zones (Z1/Z2/Z3/OUT) hypothetical position falls into
               - Normalized greeks per ₹1L capital
               - Theta/Gamma/Vega classification before commitment
            
            3. **Bucket Scenario Analysis**
               - Test how adding position affects each bucket (A/B/C/D/E)
               - Forward simulation with hypothetical position included
               - Mean/P1/P5 PnL projections
               - Gate pass/fail predictions
            
            4. **Full Stress Testing**
               - ES99 calculation with new position
               - VaR metrics and tail loss estimates
               - Scenario repricing across spot/vol/time changes
               - Black-Scholes option pricing for scenarios
            
            **Workflow**:
            1. Get greeks from broker/pricing tool for intended position
            2. Input greeks into Pre-Trade Analysis
            3. Review zone classification (want Z1 or Z2)
            4. Check bucket scenarios (which bucket to assign?)
            5. Run full stress test to see ES99 impact
            6. If all checks pass → Execute trade
            7. If any red flags → Don't trade or adjust size
            
            **When to Use**:
            - Before EVERY new trade entry
            - Testing position size adjustments
            - Comparing multiple trade candidates
            - Understanding tail risk of complex spreads
            - Validating trades suggested by other sources
            
            **Red Flags to Avoid**:
            - Pre-trade ES99 pushes portfolio over 4% limit
            - Position falls in Zone 3 or OUT zones
            - Any bucket shows FAIL gates with position added
            - Tail loss disproportionate to expected PnL
            - Adding to already-stressed bucket
            """
        )
    
    with tab4:
        st.markdown("**Supporting Tabs Overview**")
        st.markdown(
            """
            **📈 Equities Tab**
            - Separate equity holdings tracking
            - Not integrated with options risk analysis
            - Basic P&L and position management
            
            **🌡️ Market Regime Tab**
            - Deep dive into current market conditions
            - IV percentile analysis and historical context
            - Drift calculation and trend analysis
            - Volatility term structure
            - Used to inform risk appetite and trade selection
            
            **📈 Historical Performance Tab**
            - Historical P&L tracking over time
            - Trade outcome analysis (winners vs losers)
            - ES99 efficiency: Returns generated per unit of tail risk
            - Performance attribution by bucket
            - Learn from past trades to improve future decisions
            
            **💾 Data Source Tab**
            - Browse NIFTY options chain data
            - Historical derivatives OHLC data
            - Verify data quality and availability
            - Source for manual analysis or external tools
            - Primarily for data validation and exploration
            
            **✨ Product Overview Tab**
            - This documentation (you are here)
            - Comprehensive system guide for investors
            - Methodology explanations
            - Risk framework documentation
            - Navigation help and workflow guides
            """
        )
    
    # Risk vs Return Framework
    st.markdown("---")
    st.subheader("⚖️ Risk-Return Framework")
    
    st.markdown(
        """
        **Core Principle**: Returns scale with risk, but risk must be managed systematically
        
        | ES99 Limit | Expected Monthly Return | Expected Annual Return | Max Drawdown Risk | Bucket Allocation |
        |------------|------------------------|----------------------|-------------------|-------------------|
        | 2% | 0.8% - 1.2% | 10% - 15% | 3% - 4% | Low bucket only |
        | 3% | 1.0% - 1.5% | 12% - 18% | 4% - 5% | Low + Med |
        | 4% | 1.2% - 1.8% | 15% - 22% | 5% - 6% | Balanced mix |
        | 5% | 1.5% - 2.0% | 18% - 25% | 6% - 8% | Med + High focus |
        | 6% | 1.8% - 2.5% | 22% - 30% | 7% - 9% | High risk appetite |
        
        **Key Relationships**:
        - Higher theta targets → Higher ES99 → Higher returns but larger drawdowns
        - Lower gamma (more negative) → Near spot risk → Need faster exits
        - Higher short vega → Vol regime dependent → Best in high IV
        - More capital deployed → Better returns but less dry powder for opportunities
        
        **Optimal Operating Range** (for most institutional investors):
        - Portfolio ES99: 3.5% - 4.0% of capital
        - Expected annual return: 15% - 22%
        - Max drawdown: 5% - 6%
        - Theta per 1L: 150 - 200
        - Recovery time from max DD: 3-4 months
        """
    )
    
    # Workflow
    st.markdown("---")
    st.subheader("🔄 Daily Workflow & Monitoring")
    
    st.markdown(
        """
        **Morning Routine** (5-10 minutes):
        1. **Portfolio Tab**: Check total P&L, margin usage, greek exposures
        2. **Risk Buckets → Portfolio Level**: Check ES99 vs 4% limit, review Layer 1-3 metrics
        3. **Risk Buckets → Trade Level**: Identify trades with negative mean PnL → flag for exit
        4. **Chart Review**: Look for trades drifting into bad zones or failing gates
        
        **Pre-Trade Evaluation** (per new trade):
        1. **Pre-Trade Analysis**: Input hypothetical greeks and stress test
        2. Check zone classification (want Z1 or Z2, avoid Z3 and OUT)
        3. Review bucket scenarios (which bucket to assign?)
        4. Verify Reward/Risk ratio > 0.15
        5. Check ES99 impact on portfolio and bucket
        6. **Risk Buckets → Bucket Level**: Verify selected bucket has capacity
        7. If all checks pass → Execute trade
        
        **Intraday Monitoring**:
        - Market moves > 1%: Check Portfolio tab greeks and Risk Buckets ES99
        - VIX spike: Review Market Regime tab and vega exposures
        - Negative P&L on any position: Evaluate exit in Risk Buckets Trade Level
        
        **End of Day Review**:
        - **Portfolio Tab**: Update positions with closing prices
        - **Risk Buckets → Portfolio Level**: Review Layer 2 ES99 metrics
        - **Risk Buckets → Trade Level**: Check gate status changes (PASS → WATCH → FAIL)
        - Plan next day: Which expiries are coming, what needs adjustment
        - Document any exits/adjustments and reasoning
        
        **Weekly Review**:
        - **Historical Performance**: P&L attribution and trade outcomes
        - **Risk Buckets → Trade Level**: Gate failure patterns and zone drift analysis
        - ES99 efficiency: Returns generated per unit of tail risk
        - Strategy refinement: Learn from winners and losers
        """
    )
    
    # Why This Is Different
    st.markdown("---")
    st.subheader("🎖️ Why This Is Institutional-Grade")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Traditional Options Trading (Retail)**")
        st.markdown(
            """
            ❌ Directional bets on market moves
            
            ❌ Unlimited risk (naked options)
            
            ❌ No systematic position sizing
            
            ❌ Emotional exit decisions
            
            ❌ No tail risk measurement
            
            ❌ Chase high returns without risk context
            
            ❌ No greek management
            
            ❌ Hope-based holding through losses
            """
        )
    
    with col2:
        st.markdown("**TBA System (Institutional)**")
        st.markdown(
            """
            ✅ Probability-driven premium harvesting
            
            ✅ Defined risk with ES99 limits
            
            ✅ Capital allocated by risk bucket
            
            ✅ Rules-based exit criteria (gates)
            
            ✅ Real-time tail risk monitoring
            
            ✅ Returns scaled to risk appetite
            
            ✅ Zone-based greek controls
            
            ✅ Process-driven cutting of losers
            """
        )
    
    st.markdown(
        """
        **The Edge Is Process, Not Prediction**:
        - Consistent application of risk rules beats market timing
        - Systematic position sizing prevents catastrophic losses
        - Multi-layer controls ensure no single trade dominates risk
        - Forward simulation reveals risks before they materialize
        - Gate discipline forces exit of deteriorating trades
        
        This is designed to behave like a **risk-managed income business**, 
        not a trading strategy dependent on being "right" about market direction.
        """
    )
    
    # Footer
    st.markdown("---")
    st.caption(
        "📚 For detailed methodology on ES99 calculation, zone definitions, and gate logic, "
        "see the **Risk Buckets → Meta Information** subtab."
    )
