"""Factory functions for creating initial states."""

from typing import List, Optional

import pandas as pd

from src.bt.engine.candle_store import CandleStore
from src.bt.state.types import (
    PortfolioState,
    BacktestState,
    EquityPoint,
    ExecutionParams,
    RiskConfig,
    CommissionModel,
    FixedCommission,
    PerShareCommission,
)


def create_initial_portfolio(
    initial_capital: float, start_timestamp: pd.Timestamp
) -> PortfolioState:
    """Create initial empty portfolio state."""
    return PortfolioState(
        cash=initial_capital,
        positions={},
        trades=(),
        equity_curve=(
            EquityPoint(
                timestamp=start_timestamp,
                equity=initial_capital,
                cash=initial_capital,
                positions_value=0.0,
            ),
        ),
        initial_capital=initial_capital,
    )


def create_initial_backtest_state(
    symbols: List[str],
    initial_capital: float,
    start_timestamp: pd.Timestamp,
    rolling_window_size: Optional[int] = None,
) -> BacktestState:
    """Create initial backtest state."""
    return BacktestState(
        portfolio=create_initial_portfolio(initial_capital, start_timestamp),
        timestamp=None,
        pending_signals={},
        risk_events=(),
        candles=CandleStore({}),
    )


def build_commission_model(
    flat: float,
    per_share: float | None,
    min_per_fill: float,
    max_pct_of_value: float | None,
) -> CommissionModel:
    """Pick the commission shape: per-share when ``per_share`` is set, else flat.

    ``per_share=None`` keeps the legacy flat ``commission`` knob working, so
    existing strategy JSONs are unaffected.
    """
    if per_share is None:
        return FixedCommission(flat)
    return PerShareCommission(
        per_share=per_share,
        min_per_fill=min_per_fill,
        max_pct_of_value=max_pct_of_value,
    )


def create_execution_params(
    spread_bps: float = 5.0,
    slippage_bps: float = 2.0,
    fixed_commission: float = 0.5,
    commission_model: CommissionModel | None = None,
) -> ExecutionParams:
    """Create execution parameters.

    ``commission_model`` wins when given; otherwise a flat ``fixed_commission``
    model is built, preserving the pre-union behavior for bare callers.
    """
    model = (
        commission_model
        if commission_model is not None
        else FixedCommission(fixed_commission)
    )
    return ExecutionParams(
        spread_bps=spread_bps,
        slippage_bps=slippage_bps,
        commission_model=model,
    )


def create_risk_config(
    stop_loss_pct: float, take_profit_pct: float, trailing_stop: bool = False
) -> RiskConfig:
    """Create risk configuration."""
    return RiskConfig(
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        trailing_stop=trailing_stop,
    )
