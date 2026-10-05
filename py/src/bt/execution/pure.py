"""Backward-compatible re-exports of the signal/risk fill functions.

The implementations moved to ``src/bt/exchange/sim.py`` (the ``SimExchange``
adapter) so the backtest engine composes one typed ``Broker`` instead of calling
module-level functions. This module re-exports them so existing importers — the
live backtest-parity path and the exec tests — keep working unchanged.
"""

from src.bt.exchange.sim import (  # noqa: F401
    calculate_adverse_selection,
    execute_risk_event,
    execute_signal,
    is_buy_fill,
)
from src.exec.friction import apply_friction, commission_for_fill  # noqa: F401

__all__ = [
    "apply_friction",
    "calculate_adverse_selection",
    "commission_for_fill",
    "execute_risk_event",
    "execute_signal",
    "is_buy_fill",
]
