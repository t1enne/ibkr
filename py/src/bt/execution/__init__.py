"""Execution module - functional implementation.

This module provides pure functions for order execution.
All functions are immutable - they return new state rather than mutating input.
"""

# State types
from src.bt.state import (
    ExecutionParams,
    create_execution_params,
)

# Pure functions — implementations now live on the SimExchange adapter; these
# are the module-level re-exports (see ``src/bt/exchange/sim.py``).
from src.bt.exchange.sim import (
    execute_signal,
    execute_risk_event,
    calculate_adverse_selection,
)

__all__ = [
    # State types
    "ExecutionParams",
    # Factories
    "create_execution_params",
    # Pure functions
    "execute_signal",
    "execute_risk_event",
    "calculate_adverse_selection",
]
