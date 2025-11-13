"""Tab render functions for Options Trading Dashboard."""

from .login_tab import render_login_tab
from .overview_tab import render_overview_tab
from .positions_tab import render_positions_tab
from .portfolio_tab import render_portfolio_tab
from .diagnostics_tab import render_diagnostics_tab
from .market_regime_tab import render_market_regime_tab
from .alerts_tab import render_alerts_tab
from .advanced_analytics_tab import render_advanced_analytics_tab
from .trade_history_tab import render_trade_history_tab
from .data_hub_tab import render_data_hub_tab
from .kite_instruments_tab import render_kite_instruments_tab
from .nifty_overview_simple import render_nifty_overview_tab
from .derivatives_data_tab import render_derivatives_data_tab

__all__ = [
    'render_login_tab',
    'render_overview_tab',
    'render_positions_tab',
    'render_portfolio_tab',
    'render_diagnostics_tab',
    'render_market_regime_tab',
    'render_alerts_tab',
    'render_advanced_analytics_tab',
    'render_trade_history_tab',
    'render_data_hub_tab',
    'render_kite_instruments_tab',
    'render_nifty_overview_tab',
    'render_derivatives_data_tab'
]
