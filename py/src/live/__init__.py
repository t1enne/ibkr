"""Live trading package — thin batch reconcile cycle over the existing layers.

Reuses the backtest data/screen/portfolio layers and adds only the live edges
(portfolio source, signals bridge) plus (later) the pure reconcile core.
"""

from __future__ import annotations

from src.live.portfolio_source import (
    MockPortfolioSource,
    PortfolioSource,
    load_mock_portfolio,
)
from src.live.result import Err, Ok, Result
from src.live.signals import ACTIONABLE, live_signals
from src.live.types import (
    FeedError,
    LiveConfig,
    LiveSignal,
    OrderIntent,
    PortfolioSnapshot,
    PortfolioView,
    SignalAction,
)

__all__ = [
    "ACTIONABLE",
    "Err",
    "FeedError",
    "LiveConfig",
    "LiveSignal",
    "MockPortfolioSource",
    "Ok",
    "OrderIntent",
    "PortfolioSnapshot",
    "PortfolioSource",
    "PortfolioView",
    "Result",
    "SignalAction",
    "live_signals",
    "load_mock_portfolio",
]
