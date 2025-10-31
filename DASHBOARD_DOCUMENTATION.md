# Options Trading Dashboard Documentation

## Overview
This dashboard provides comprehensive options trading analytics, position management, risk monitoring, and market data visualization for NIFTY index options.

---

## Main Dashboard Tabs

### 1. 🔐 Login Tab
**Purpose**: Authenticate with Kite Connect API to access live trading data.

**Features**:
- Kite API key and secret input
- OAuth-based login flow via Zerodha Kite
- Access token management and persistence
- Session state management

**Metrics Tracked**: None (Authentication only)

---

### 2. 📊 Positions Tab
**Purpose**: Display current options positions with Greeks and market data.

**Features**:
- Fetch live positions from Kite Connect
- Parse trading symbols (NIFTY options)
- Calculate real-time Greeks for each position
- Enrich positions with IV and market data

**Metrics Tracked**:
- **Trading Symbol**: Option contract identifier
- **Quantity**: Position size (positive = long, negative = short)
- **Strike Price**: Option strike price
- **Option Type**: CE (Call) or PE (Put)
- **DTE (Days to Expiry)**: Time remaining until expiration
- **Last Price**: Current market price
- **P&L**: Profit/Loss for the position
- **Implied Volatility (IV)**: Market-implied volatility (%)
- **Delta**: Rate of change in option price per ₹1 move in underlying
- **Gamma**: Rate of change in delta per ₹1 move in underlying
- **Vega**: Sensitivity to 1% change in volatility
- **Theta**: Time decay per day
- **Position Greeks**: Scaled Greeks (Greek × Quantity)

---

### 3. 📈 Portfolio Overview Tab
**Purpose**: Aggregate portfolio-level metrics, risk analysis, and capital management.

**Sections**:

#### A. Capital & Performance (8 metrics)
1. **Account Size**: Total capital (margin available + used) in ₹

2. **Margin Used**: Capital deployed
   - Shows absolute amount and utilization %
   - Color codes:
     - 🟢 <50%: Healthy
     - 🟡 50-70%: Warning
     - 🔴 >70%: Critical

3. **Net P&L**: Total profit/loss across all positions in ₹
   - Shows ROI % (P&L / Account Size)

4. **Theta/Day**: Daily time decay income in ₹
   - Shows as % of capital
   - Positive theta = earning from time decay
   - Negative theta = paying for time decay

5. **ROI (Annualized)**: Projected annual return percentage
   - Formula: (P&L / Account Size) / (Days in Trade / 365) × 100

6. **Days to Recover**: Time needed to recover losses via theta decay
   - Shows average DTE for comparison
   - 🔴 >DTE: Cannot recover by expiry
   - 🟡 >DTE×0.5: Tight timeline
   - 🟢 <DTE×0.5: Manageable

7. **Theta Efficiency**: P&L relative to theta income (%)
   - Formula: (P&L / Theta) × 100
   - 🔴 <-200%: Directional problem
   - 🟡 -100% to -200%: Negative efficiency
   - 🟢 >-100%: Recovering

8. **Notional Exposure**: Total position value (sum of all strike × quantity)
   - Shows leverage ratio (Notional / Account Size)
   - 🟢 <50×: Conservative
   - 🟡 50-100×: Moderate
   - 🔴 >100×: Aggressive

#### B. Greeks & Risk (6 metrics + 1 chart)
9. **Net Delta (units)**: Portfolio directional exposure
   - Sum of all position deltas
   - Range: Can exceed ±1 for multi-position portfolio
   - 🟢 <±8: Neutral
   - 🟡 ±8 to ±15: Watch
   - 🔴 >±15: High directional risk

10. **Delta (₹/pt)**: Delta converted to rupees per NIFTY point
    - Shows capital at risk per 1-point move
    - Assumes lot size = 50 (configurable)

11. **Net Gamma**: Portfolio convexity
    - Rate of delta change
    - Negative gamma = risk accelerates with movement
    - ⚠️ Special alert if gamma <-0.5 near expiry (<7 DTE)

12. **Net Vega**: Portfolio volatility sensitivity in ₹
    - Total capital gain/loss for 1% IV change

13. **Delta Notional (₹)**: Delta exposure in rupees
    - Formula: Net Delta × Current Spot × Lot Size
    - Shows % of account

14. **Vega % Capital**: Vega as percentage of account size
    - Shows vulnerability to IV changes

15. **Greeks Breakdown Chart**: Bar chart visualization
    - Displays: Delta, Gamma×100, Vega/100, Theta
    - Scaled for visual comparison

#### C. Risk Analysis (4 metrics + alert system)
16. **VaR (95%)**: Value at Risk - potential 1-day loss at 95% confidence
    - Uses realized volatility from NIFTY data
    - Statistical estimate of tail risk

17. **+2% NIFTY Move**: P&L impact if NIFTY rises 2%
    - Stress test using delta/gamma
    - Shows profit/loss in ₹

18. **-2% NIFTY Move**: P&L impact if NIFTY falls 2%
    - Stress test using delta/gamma
    - Shows profit/loss in ₹

19. **+5 IV Points**: P&L impact if IV increases by 5%
    - Stress test using vega
    - Shows profit/loss in ₹

**Risk Status Summary** (Real-time alerts):
- Delta breach (>±8, >±15)
- Margin utilization (>50%, >70%)
- Recovery timeline vs expiry
- Theta efficiency (<-100%, <-200%)
- Gamma risk near expiry (gamma <-0.5 with DTE <7)

#### D. Position Concentration (2 sections)
20. **Expiry Table**: Breakdown by expiry date
    - Positions count
    - P&L per expiry in ₹
    - Notional exposure per expiry
    - Leverage ratio per expiry (× capital)

21. **Largest Positions by Notional**: Top 5 positions
    - Shows symbol
    - Notional value in ₹
    - % of portfolio

**Total Portfolio Overview Metrics: 21**

---

### 4. � Position Diagnostics Tab
**Purpose**: Identify problematic positions and prioritize actions with detailed per-position analysis.

**Features**:
- Sort positions by multiple criteria
- Filter for red/losing positions only
- Action priority signals with color coding

**All Metrics Tracked (23 metrics per position)**:

#### Identity & Basic Info (4 metrics)
1. **Symbol**: Trading symbol (e.g., NIFTY24NOV25000CE)
2. **Strike**: Strike price
3. **Type**: CE (Call) or PE (Put)
4. **Qty**: Position size (negative = short, positive = long)

#### P&L & Pricing (4 metrics)
5. **PnL**: Current profit/loss in ₹
6. **Entry**: Entry price (buy price or sell price)
7. **Current**: Current market price (LTP)
8. **Chg %**: Premium change percentage since entry

#### Time Metrics (2 metrics)
9. **DTE**: Days to expiry (calendar days)
10. **Days to B/E**: Days to break-even via theta decay
    - Formula: |P&L / Theta|
    - Critical if > DTE (cannot recover)

#### Theta Analysis (2 metrics)
11. **Theta/Day**: Daily time decay per contract
12. **Theta Eff %**: Theta efficiency percentage
    - Formula: (P&L / Theta) × 100
    - Negative = losing faster than theta can recover

#### Greeks (4 metrics)
13. **Delta**: Option delta (0 to 1 for calls, -1 to 0 for puts)
14. **Pos Delta**: Position delta (Delta × Quantity)
15. **Gamma**: Rate of delta change
16. **Vega**: IV sensitivity

#### Position Sizing (2 metrics)
17. **Notional**: Position notional value in ₹
    - Formula: |Quantity × Strike|
18. **% Portfolio**: Position size as % of total account

#### Risk Metrics (4 metrics)
19. **Spot Dist %**: Distance from current spot price
    - Formula: ((Spot - Strike) / Spot) × 100
    - Indicates how far ITM/OTM

20. **PoP %**: Probability of Profit (rough approximation)
    - Formula: (1 - |Delta|) × 100
    - Higher delta = lower PoP for that leg

21. **IV**: Implied volatility of the option

22. **Loss/Credit**: Loss-to-credit ratio (for short positions)
    - Shows if losing more than premium collected
    - N/A for long positions

#### Action Signals (2 metrics)
23. **Action**: Recommended action signal
    - "CLOSE NOW" 🔴: Urgent exit
    - "WATCH" 🟡: Monitor closely
    - "HOLD" 🟢: OK to maintain
    - "ADJUST" 🟡: Consider adjustment

24. **Priority**: Numerical action priority score
    - Higher score = more urgent attention needed
    - Used for sorting by urgency

25. **Status**: Overall position health with color indicators
    - Contains risk factors:
      - 🔴 RED: Critical issue (P&L <-50%, can't recover, extreme delta, near expiry + losing)
      - 🟡 YELLOW: Warning (P&L -20% to -50%, tight recovery, high delta)
      - 🟢 GREEN: Healthy (profitable or manageable)

**Sorting Options**:
- Days to B/E
- Theta Efficiency %
- PnL
- Position Size (Notional)
- Delta (Position Delta)
- Action Priority

**Filtering Options**:
- Show only RED positions
- Show only losing positions

**Total Position Diagnostics Metrics: 25 per position**

---

### 5. 🌡️ Market Regime Tab
**Purpose**: Analyze market conditions and volatility environment.

**Sections**:

#### A. Volatility Metrics (4 metrics)
1. **Current IV (ATM)**: At-the-money implied volatility percentage
   - Shows market's expectation of future price movement
   - Higher IV = more expensive options

2. **IV Rank (90d)**: Current IV percentile over 90-day range (0-100%)
   - Formula: (Current IV - Min IV) / (Max IV - Min IV) × 100
   - 🔴 >70: HIGH - Options expensive, sell premium strategies
   - 🟡 30-70: MID - Fair value, neutral strategies
   - 🟢 <30: LOW - Options cheap, buy premium strategies

3. **Realized Vol (30d)**: Actual price movement over last 30 days (annualized)
   - Compare with IV to identify over/under-priced options

4. **Vol Risk Premium (VRP)**: Implied Vol - Realized Vol
   - Shows if options are overpriced vs actual movement
   - 🔴 >+5%: IV too high, consider selling
   - 🟡 -5% to +5%: Fair pricing
   - 🟢 <-5%: IV too low, consider buying

#### B. Additional Volatility Metrics (4 metrics)
5. **RV Percentile**: Current realized vol rank in historical range
   - >80: High volatility environment
   - <20: Calm market

6. **Term Structure**: Far IV - Near IV (difference between expiries)
   - Positive (Contango): Normal market, far options more expensive
   - Negative (Backwardation): Market stress, near options more expensive
   - 🟢 >0: Normal/Healthy
   - 🟡 -2% to 0: Watch
   - 🔴 <-2%: Stress mode

7. **Near-term IV**: IV of current month expiry options

8. **Next-term IV**: IV of next month expiry options

#### C. Sentiment Indicators (4 metrics)
9. **PCR (OI)**: Put Open Interest / Call Open Interest ratio
   - 🟢 >1.2: Bullish (more hedging, expect up move)
   - 🟡 0.8-1.2: Neutral/Balanced
   - 🔴 <0.8: Bearish (excessive optimism, complacent)

10. **PCR (Volume)**: Put Volume / Call Volume ratio
    - Shows active trading sentiment

11. **Volatility Skew**: OTM Put IV - OTM Call IV (~3% away from spot)
    - 🔴 >+1%: Fear mode, puts expensive
    - 🟡 -1% to +1%: Balanced
    - 🟢 <-1%: Complacent, calls expensive

12. **Put IV (OTM)**: IV of out-of-money put options
    - Displayed with Call IV for comparison

#### D. Max Pain & Price Reference (2 metrics)
13. **Max Pain Strike**: Strike with maximum pain for option writers
    - Market often gravitates toward this level before expiry
    - Shows % distance from current spot
    - 🟢 <1% away: At max pain
    - 🟡 1-2% away: Near max pain
    - 🔴 >2% away: Far from max pain

14. **Spot Price**: Current NIFTY spot price (reference)

#### E. Market Regime Classification (1 metric)
15. **Market Regime**: Combined assessment with strategy recommendation
    - Based on IV rank, VRP, and volatility levels
    - Examples:
      - "High Vol - Sell Premium"
      - "Low Vol - Buy Premium"
      - "Neutral - Balanced"
      - "High Vol - Caution"
      - "Elevated IV - Sell Bias"
      - "Compressed IV - Buy Bias"

#### F. Trend Indicators (4 metrics)
16. **20-day SMA**: 20-day simple moving average of NIFTY
    - Shows short-term trend
    - Displays % distance from current price

17. **50-day SMA**: 50-day simple moving average of NIFTY
    - Shows medium-term trend
    - Displays % distance from current price

18. **RSI (14)**: Relative Strength Index (14-period)
    - 🔴 >70: Overbought
    - 🟡 30-70: Neutral
    - 🟢 <30: Oversold

19. **ATR (14)**: Average True Range (14-period volatility measure)
    - Shows average daily range
    - Displayed as absolute value and % of spot

**Total Market Regime Metrics: 19**

---

### 6. 🚨 Risk Alerts Tab
**Purpose**: Real-time alerts for portfolio risk breaches.

**Alert Types**:
- **Delta Breach**: Net delta exceeds safe thresholds (±8, ±15)
- **Margin Warning**: Margin utilization above 50% or 70%
- **Recovery Risk**: Cannot recover losses by expiry
- **Gamma Risk**: High negative gamma near expiry
- **Theta Inefficiency**: Losses exceed multiple of theta income
- **Position Size**: Individual positions too large (>20% portfolio)
- **Days to Expiry**: Positions approaching expiry (< 7 days)

---

## Data Hub Tab (📂 Data Hub)

### Purpose
Centralized data management for futures and options historical data with visualization tools.

### Sub-Tabs

#### 1. Futures Data Tab
**Purpose**: Load and visualize NIFTY/BANKNIFTY/FINNIFTY futures data.

**Features**:
- Fetch futures data from NSE API via Kite
- Store data in Supabase database
- View historical data by expiry
- FII futures position analysis

**Metrics Tracked**:
- **Date**: Trading date
- **Expiry Date**: Contract expiry
- **Underlying Value**: NIFTY/BANKNIFTY price
- **Open Interest (OI)**: Total open contracts
- **Change in OI**: Daily OI change (indicates buying/selling)
- **Symbol**: Futures symbol (NIFTY/BANKNIFTY/FINNIFTY)

**Visualizations**:
- Per-expiry charts (last 60 days):
  - Underlying price (line chart)
  - Change in OI (bar chart - blue=longs, red=shorts)
- Combined dual-axis view

#### 2. Options Data Tab
**Purpose**: Load and analyze options chain data from local files.

**Features**:
- Load Excel/CSV files from `database/data/` or `database/options_data/`
- Parse and normalize options data
- Multi-expiry analysis

**Metrics Tracked**:
- **Symbol**: NIFTY
- **Date**: Trading date
- **Expiry**: Option expiry date
- **Option Type**: CE or PE
- **Strike Price**: Strike price
- **Open, High, Low, Close**: OHLC prices
- **LTP (Last Traded Price)**: Current price
- **Settle Price**: Settlement price
- **No. of Contracts**: Trading volume
- **Turnover**: Total traded value
- **Premium Turnover**: Premium value traded
- **Open Interest (OI)**: Total open contracts at strike
- **Change in OI**: Daily OI change
- **Underlying Value**: Spot price

**Visualizations**:
- **CE vs PE Open Interest**: Grouped bar charts showing call/put OI over last 60 days
- **Underlying Price**: Line chart overlay on OI bars
- **Per-Strike OI**: Daily snapshot of OI distribution across strikes (23k-27k range)
  - Date navigation with previous/next buttons
  - Weekday display
  - CE/PE side-by-side comparison

#### 3. Analysis Tab
**Purpose**: Combined futures and options analysis per expiry.

**Features**:
- Unified view of futures and options for same expiry
- Compare underlying movement with options activity
- Identify divergences and patterns

**Metrics Tracked**:
- All metrics from Futures + Options tabs
- Cross-market analysis:
  - Futures OI buildup vs Options OI
  - Price action correlation
  - OI change patterns across both markets

**Visualizations**:
- Stacked charts per expiry:
  1. Futures: Underlying + Change in OI
  2. Options: CE/PE OI + Underlying price
  3. Per-Strike OI snapshot for selected date

---




## Key Concepts & Definitions

### Greeks
- **Delta**: Measures directional exposure. Range: -1 to +1 for single option, portfolio can exceed
- **Gamma**: Measures delta change rate. Negative gamma = risk increases as market moves
- **Vega**: Volatility sensitivity. High vega = big impact from IV changes
- **Theta**: Time decay. Positive theta = earning from time, negative theta = paying time

### Risk Metrics
- **VaR (Value at Risk)**: Statistical estimate of potential loss
- **Notional Exposure**: Total underlying value controlled by options positions
- **Leverage Ratio**: Notional exposure divided by account size

### Volatility Metrics
- **Implied Volatility (IV)**: Market's expectation of future volatility
- **Realized Volatility (RV)**: Actual historical volatility from price returns
- **IV Rank**: Current IV's percentile ranking over lookback period
- **VRP (Volatility Risk Premium)**: Difference between implied and realized volatility

### Market Indicators
- **PCR (Put-Call Ratio)**: Ratio of put to call activity (volume or OI)
- **Skew**: Difference in IV between OTM puts and calls
- **Term Structure**: Relationship between near and far month IVs
- **Max Pain**: Strike where option holders lose most, writers gain most

### Position Metrics
- **DTE (Days to Expiry)**: Calendar days until option expiration
- **Theta Efficiency**: How well theta income covers losses (>100% = recovering)
- **Days to Break-Even**: Time needed to recover via theta decay
- **ROI (Return on Investment)**: Percentage return on capital

---

## Color Coding & Status Indicators

### Risk Status Colors
- 🟢 **Green**: Healthy/Safe range
- 🟡 **Yellow**: Warning/Monitor
- 🔴 **Red**: Critical/Urgent action needed

### Chart Colors
- **#3B82F6 (Blue)**: Positive values, call options, longs
- **#EF4444 (Red)**: Negative values, put options, shorts
- **#75F37B (Green)**: Put OI in analysis
- **#E96767 (Red)**: Call OI in analysis
- **#9333EA (Purple)**: NIFTY price line
- **#F1F1F1 (White)**: Underlying price overlays

---

## Data Sources

1. **Kite Connect API**: Live positions, NIFTY daily data, instruments
2. **Supabase Database**: Futures historical data storage
3. **Local Files**: Options chain data (Excel/CSV in `database/data/`)
4. **NSE Data Fetcher**: Historical futures data from NSE website

---

## Usage Tips

1. **Always login first** via Login tab to access live data
2. **Fetch positions** regularly to keep Greeks updated
3. **Monitor Risk Alerts** tab for real-time warnings
4. **Check Market Regime** before taking new positions
5. **Use Position Diagnostics** to identify problem positions early
6. **Review Portfolio Overview** daily for overall health
7. **Set DEBUG flags** to True in `futures_data_loader.py` and `options_data_loader.py` to access all data management features

---

## Technical Notes

- Greeks calculated using Black-Scholes model via `py_vollib`
- Default risk-free rate: 7%
- Default lot size: 50 (configurable via `OPTION_CONTRACT_SIZE` env var)
- Token persistence: `.kite_token.json` in project root
- Indian currency formatting: ₹10,00,000 (lakhs-crores system)
- All times in IST (Indian Standard Time)
