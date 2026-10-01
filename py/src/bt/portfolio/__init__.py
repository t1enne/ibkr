"""Portfolio module - functional implementation.

This module provides pure functions for portfolio state management.
All functions are immutable - they return new state rather than mutating input.
"""

# State types
from src.bt.state import (
    PortfolioState,
    Position,
    Trade,
    EquityPoint,
    ActionType,
    TradeStatus,
    TradeExitReason,
    create_initial_portfolio,
)

# Pure functions
from src.bt.portfolio.pure import (
    apply_fill,
    next_position_id,
    update_prices,
    mark_to_market_list,
    calculate_equity,
    calculate_positions_value,
    iter_positions,
    count_positions,
    get_symbol_positions,
)

# For backward compatibility - these are deprecated
# Use the functional versions above instead
Portfolio = None  # Removed - use PortfolioState with pure functions
PortfolioProps = None  # Removed - use factory functions

__all__ = [
    # State types
    "PortfolioState",
    "Position",
    "Trade",
    "EquityPoint",
    "ActionType",
    "TradeStatus",
    "TradeExitReason",
    # Factories
    "create_initial_portfolio",
    # Pure functions
    "apply_fill",
    "next_position_id",
    "update_prices",
    "mark_to_market_list",
    "calculate_equity",
    "calculate_positions_value",
    "iter_positions",
    "count_positions",
    "get_symbol_positions",
    # Deprecated (for migration only)
    "Portfolio",
    "PortfolioProps",
]
