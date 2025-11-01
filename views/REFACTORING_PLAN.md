# Kite Positions Refactoring Plan

## Overview
Refactoring `kite_positions.py` (4041 lines) into modular structure with separate files for tabs and utilities.

## New Structure

```
views/
├── kite_positions.py          # Main orchestrator (reduced to ~300 lines)
├── tabs/                       # Tab render functions
│   ├── __init__.py
│   ├── login_tab.py
│   ├── positions_tab.py
│   ├── portfolio_tab.py
│   ├── diagnostics_tab.py
│   ├── market_regime_tab.py
│   ├── alerts_tab.py
│   ├── advanced_analytics_tab.py
│   └── trade_history_tab.py
└── utils/                      # Utility functions
    ├── __init__.py
    ├── token_manager.py        # ✅ CREATED - Token persistence
    ├── data_loaders.py          # ✅ CREATED - Data loading utilities
    ├── parsers.py               # ✅ CREATED - Symbol parsing
    ├── greeks.py                # ✅ CREATED - Greeks calculations
    ├── portfolio_metrics.py     # TO CREATE - Portfolio aggregations & market regime
    ├── risk_calculations.py     # TO CREATE - VaR, stress tests, action signals
    └── formatters.py            # TO CREATE - INR formatting, constants
```

## Files Created ✅

### 1. utils/__init__.py
- Centralizes all utility imports
- Clean API for importing from tabs

### 2. utils/token_manager.py  
- `save_kite_token()` - Save credentials to .kite_token.json
- `load_kite_token()` - Load saved credentials
- `clear_kite_token_file()` - Remove token file
- `_token_file_path()` - Get token file path

### 3. utils/data_loaders.py
- `load_all_data()` - Load options data from Excel/CSV
- `load_nifty_daily()` - Fetch NIFTY historical data from Kite API
- `_map_columns()` - Normalize column names

### 4. utils/parsers.py
- `parse_tradingsymbol()` - Parse NIFTY option symbols (weekly & monthly)
  - Weekly: NIFTY25N1125100PE
  - Monthly: NIFTY25NOV25000CE

### 5. utils/greeks.py
- `calculate_implied_volatility()` - Calculate IV using Black-Scholes
- `calculate_greeks()` - Calculate delta, gamma, vega, theta
- `enrich_position_with_greeks()` - Enrich Kite position with Greeks

## Files To Create 📋

### 6. utils/portfolio_metrics.py
Functions:
- `calculate_portfolio_greeks(enriched_positions)` - Aggregate Greeks
- `calculate_market_regime(options_df, nifty_df)` - Market indicators
  - PCR (Put-Call Ratio)
  - Volatility skew
  - Term structure
  - Regime classification
  - Trend indicators (SMA, RSI, ATR)
  - Max pain calculation

### 7. utils/risk_calculations.py
Functions:
- `calculate_var(positions, spot, nifty_df, confidence)` - Value at Risk
- `calculate_stress_pnl(positions, spot, spot_multiplier, iv_change)` - Stress testing
- `get_action_signal(dte, pnl, theta, position_delta, days_to_breakeven, loss_vs_credit)` - Action priority
- `get_action_recommendation(pos)` - Position recommendations

### 8. utils/formatters.py
Functions:
- `format_inr(value, decimals, symbol)` - Indian currency formatting
Constants:
- `DEFAULT_LOT_SIZE` - Option contract size (default: 50)

### 9. tabs/__init__.py
Exports all tab render functions

### 10. tabs/login_tab.py
Function:
- `render_login_tab()` - Authentication UI

### 11. tabs/positions_tab.py
Function:
- `render_positions_tab(options_df, nifty_df)` - Display positions with Greeks

### 12. tabs/portfolio_tab.py
Function:
- `render_portfolio_tab()` - Portfolio overview metrics

### 13. tabs/diagnostics_tab.py
Function:
- `render_diagnostics_tab()` - Position diagnostics and action priorities

### 14. tabs/market_regime_tab.py
Function:
- `render_market_regime_tab(options_df, nifty_df)` - Market regime analysis

### 15. tabs/alerts_tab.py
Function:
- `render_alerts_tab()` - Risk alerts

### 16. tabs/advanced_analytics_tab.py
Function:
- `render_advanced_analytics_tab()` - Advanced analytics dashboard

### 17. tabs/trade_history_tab.py
Function:
- `render_trade_history_tab()` - Trade history analysis from CSV

### 18. kite_positions.py (Refactored Main File)
- Import all utilities and tabs
- Main `render()` function
- Session state management
- OAuth flow handling
- Tab orchestration

## Benefits

1. **Maintainability**: Each tab in separate file (~200-400 lines each)
2. **Testability**: Utilities can be tested independently
3. **Reusability**: Utils can be imported by other modules
4. **Collaboration**: Multiple developers can work on different tabs
5. **Code Discovery**: Easier to find specific functionality
6. **Performance**: Selective imports reduce memory footprint

## Migration Strategy

1. ✅ Create utils/ directory with utility modules
2. ⏳ Create tabs/ directory with tab modules  
3. ⏳ Refactor main kite_positions.py to import from modules
4. ⏳ Test each tab individually
5. ⏳ Test full integration
6. ⏳ Remove old kite_positions.py code (backup first)

## Next Steps

1. Create remaining utils files (portfolio_metrics, risk_calculations, formatters)
2. Create tabs/ directory and tab modules
3. Refactor main kite_positions.py
4. Run tests and verify functionality
5. Update imports in other files if needed

## Line Count Estimates

- Original file: 4041 lines
- After refactoring:
  - kite_positions.py: ~300 lines
  - utils/*.py: ~1500 lines total
  - tabs/*.py: ~2200 lines total
  - Total: ~4000 lines (same code, better organized)
