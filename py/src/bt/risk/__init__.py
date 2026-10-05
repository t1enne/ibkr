"""Risk management module - functional implementation.

This module provides pure functions for risk management.
All functions are immutable - they return new state rather than mutating input.
"""

# State types
from src.bt.state import (
    StopLossEvent,
    TakeProfitEvent,
    RiskConfig,
    create_risk_config,
)

# Pure functions
from src.bt.risk.pure import (
    check_risk,
    check_position_risk,
    update_trailing_stop,
)

# Injectable engine seam
from src.bt.risk.handlers import RiskHandler, default_risk_handler

# For backward compatibility - these are deprecated
# Use the functional versions above instead
RiskManager = None  # Removed - use pure functions
RiskManagerProps = None  # Removed - use RiskConfig
RiskEvent = None  # Removed - use tuple of events

__all__ = [
    # State types
    "StopLossEvent",
    "TakeProfitEvent",
    "RiskConfig",
    # Factories
    "create_risk_config",
    # Pure functions
    "check_risk",
    "check_position_risk",
    "update_trailing_stop",
    # Injectable engine seam
    "RiskHandler",
    "default_risk_handler",
    # Deprecated (for migration only)
    "RiskManager",
    "RiskManagerProps",
    "RiskEvent",
]
