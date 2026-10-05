"""IBKR adapter (plan phases 2-3): client/gateway wiring, replay, order placement.

Phase 2 shipped the read path (mapping, trade replay, portfolio source); phase 3
adds ``orders`` (pure intent -> ticket mapping + status decoding) and ``broker``
(``IbkrBroker``: MKT submission, the reply-confirmation loop, the bounded fill
wait). LMT carry-over, cancel/modify, resting stops and brackets stay phase 4.
"""

from src.live.adapters.ibkr.broker import IbkrBroker
from src.live.adapters.ibkr.trades import BookRow, Execution, StrategyBook, reconcile

__all__ = [
    "BookRow",
    "Execution",
    "IbkrBroker",
    "StrategyBook",
    "reconcile",
]
