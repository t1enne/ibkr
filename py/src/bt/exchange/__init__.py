"""Backtest broker adapter — the sim exchange over the shared exec core."""

from src.bt.exchange.sim import (
    SimExchange,
    calculate_adverse_selection,
    default_exchange,
    execute_risk_event,
    execute_signal,
    is_buy_fill,
)

__all__ = [
    "SimExchange",
    "calculate_adverse_selection",
    "default_exchange",
    "execute_risk_event",
    "execute_signal",
    "is_buy_fill",
]
