"""Live trading package — thin batch reconcile cycle over the existing layers.

Reuses the backtest data/screen/portfolio layers and adds only the live edges
(the ``LiveAdapter`` seam, signals bridge) plus the pure reconcile core.
"""

from __future__ import annotations

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
    "Ok",
    "OrderIntent",
    "PortfolioSnapshot",
    "PortfolioView",
    "Result",
    "SignalAction",
    "live_signals",
]
