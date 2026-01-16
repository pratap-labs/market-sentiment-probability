"""Tab render functions for Options Trading Dashboard."""

from .login_tab import render_login_tab
from .overview_tab import render_overview_tab
from .positions_tab import render_positions_tab
from .portfolio_tab import render_portfolio_tab
from .portfolio_dashboard_tab import render_portfolio_dashboard_tab
from .diagnostics_tab import render_diagnostics_tab
from .market_regime_tab import render_market_regime_tab
from .alerts_tab import render_alerts_tab
from .advanced_analytics_tab import render_advanced_analytics_tab
from .trade_history_tab import render_trade_history_tab
from .data_hub_tab import render_data_hub_tab
from .kite_instruments_tab import render_kite_instruments_tab
from .nifty_overview_simple import render_nifty_overview_tab
from .derivatives_data_tab import render_derivatives_data_tab
from .risk_analysis_tab import render_risk_analysis_tab
from .risk_buckets_tab import render_risk_buckets_tab
from .portfolio_buckets_tab import render_portfolio_buckets_tab
from .stress_testing_tab import render_stress_testing_tab
from .hedge_lab_tab import render_hedge_lab_tab
from .historical_performance_tab import render_historical_performance_tab
from .greeks_debug_tab import render_greeks_debug_tab
from .product_overview_tab import render_product_overview_tab

__all__ = [
    'render_login_tab',
    'render_overview_tab',
    'render_positions_tab',
    'render_portfolio_tab',
    'render_portfolio_dashboard_tab',
    'render_diagnostics_tab',
    'render_market_regime_tab',
    'render_alerts_tab',
    'render_advanced_analytics_tab',
    'render_trade_history_tab',
    'render_data_hub_tab',
    'render_kite_instruments_tab',
    'render_nifty_overview_tab',
    'render_derivatives_data_tab',
    'render_risk_analysis_tab',
    'render_risk_buckets_tab',
    'render_portfolio_buckets_tab',
    'render_stress_testing_tab',
    'render_hedge_lab_tab',
    'render_historical_performance_tab',
    'render_greeks_debug_tab',
    'render_product_overview_tab'
]
