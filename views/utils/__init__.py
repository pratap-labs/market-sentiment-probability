"""Utility functions for Options Trading Dashboard."""

from .data_loaders import (
    load_all_data,
    load_nifty_daily,
    _map_columns
)

from .token_manager import (
    save_kite_token,
    load_kite_token,
    clear_kite_token_file,
    _token_file_path
)

from .parsers import parse_tradingsymbol

from .greeks import (
    calculate_implied_volatility,
    calculate_greeks,
    enrich_position_with_greeks
)

from .portfolio_metrics import (
    calculate_portfolio_greeks,
    calculate_market_regime
)

from .risk_calculations import (
    calculate_var,
    calculate_stress_pnl,
    get_action_signal,
    get_action_recommendation
)

from .formatters import (
    format_inr,
    DEFAULT_LOT_SIZE
)

__all__ = [
    # Data loaders
    'load_all_data',
    'load_nifty_daily',
    '_map_columns',
    
    # Token management
    'save_kite_token',
    'load_kite_token',
    'clear_kite_token_file',
    '_token_file_path',
    
    # Parsers
    'parse_tradingsymbol',
    
    # Greeks
    'calculate_implied_volatility',
    'calculate_greeks',
    'enrich_position_with_greeks',
    
    # Portfolio metrics
    'calculate_portfolio_greeks',
    'calculate_market_regime',
    
    # Risk calculations
    'calculate_var',
    'calculate_stress_pnl',
    'get_action_signal',
    'get_action_recommendation',
    
    # Formatters
    'format_inr',
    'DEFAULT_LOT_SIZE',
]
