"""State dataclasses for functional backtesting engine.

All state is immutable (frozen dataclasses) to enable:
- Easy testing and debugging
- State snapshots and replay
- Time travel debugging
- Deterministic behavior
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional, Any, Protocol
from enum import Enum

import pandas as pd

from src.bt.engine.candle_store import CandleStore


class ActionType(Enum):
    long = "long"
    short = "short"
    close = "close"
    rebalance = "rebalance"  # net delta adjustment to existing position


class TradeStatus(Enum):
    open = "open"
    closed = "closed"
    stopped = "stopped"


class TradeExitReason(Enum):
    sl = "sl"
    tp = "tp"
    end = "end"
    regression = "regression"
    none = "none"
    signal = "signal"


@dataclass(frozen=True)
class Candle:
    """A single OHLCV bar."""

    timestamp: pd.Timestamp
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    interval: Optional[str] = None  # e.g., "1h", "4h"


@dataclass(frozen=True)
class Position:
    """An open position."""

    symbol: str
    qty: float
    entry_price: float
    entry_time: pd.Timestamp
    stop_loss: Optional[float]
    take_profit: Optional[float]
    last_price: float
    type: ActionType
    position_id: str = ""  # unique identifier, e.g. "SPY_1714003200.0"
    tag: str = ""  # optional strategy-facing label for lot targeting, e.g. "spy-r1"


@dataclass
class Trade:
    """A completed trade record."""

    entry_time: pd.Timestamp
    entry_price: float
    exit_time: Optional[pd.Timestamp]
    exit_price: Optional[float]
    last_price: float
    symbol: str
    position: ActionType
    qty: float
    stop_loss: float
    take_profit: float
    pnl: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    reason: Optional[str] = ""
    status: TradeStatus = TradeStatus.open
    close_reason: Optional[Any] = None
    position_id: str = ""  # links back to the Position that generated this trade


@dataclass(frozen=True)
class EquityPoint:
    """Single point in equity curve."""

    timestamp: pd.Timestamp
    equity: float
    cash: float
    positions_value: float


@dataclass(frozen=True)
class PortfolioState:
    """Immutable portfolio state."""

    cash: float
    positions: Dict[str, Tuple[Position, ...]]  # Symbol -> tuple of Positions
    trades: Tuple[Trade, ...]
    equity_curve: Tuple[EquityPoint, ...]
    initial_capital: float


class PortfolioView(Protocol):
    """Read-only portfolio surface shared by backtest and live.

    ``PortfolioState`` satisfies this **structurally**, so the SAME ``reconcile``
    and sizing code runs on a backtest book and a broker snapshot alike. Keep
    this surface minimal: every field here is a contract both sides must keep.
    Read-only by design — mutation goes through ``apply_fill``.
    """

    @property
    def cash(self) -> float: ...

    @property
    def positions(self) -> Dict[str, Tuple[Position, ...]]: ...  # symbol -> LOTs

    @property
    def initial_capital(self) -> float: ...


@dataclass(frozen=True)
class TradeSignal:
    """Signal to enter/exit a position."""

    action: ActionType
    symbol: str
    timestamp: pd.Timestamp
    price: float
    qty: float = 0.0
    reason: Optional[Any] = None
    z_score: Optional[float] = None
    hedge_beta: Optional[float] = None
    fill_at_next_open: bool = True
    position_id: Optional[str] = (
        None  # target specific position; None = open new or close by symbol
    )
    stop_loss: Optional[float] = None  # explicit SL for new positions
    take_profit: Optional[float] = None  # explicit TP for new positions
    tag: str = ""  # optional strategy-facing lot label (stored on the Position)
    # For next-open **close** signals that model an intra-bar stop trigger: the
    # level the stop fired at, plus the position side. When both are set,
    # ``execute_signal`` fills at the adverse worse-of the stop level and the
    # next open (mirroring ``execute_risk_event`` gap-through math) instead of
    # naively at next open — so a gap through the stop is not understated and a
    # delayed fill never improves on the intended intra-bar stop exit.
    fill_guard_price: Optional[float] = None
    fill_guard_is_long: Optional[bool] = None


@dataclass(frozen=True)
class FillEvent:
    """Result of executing a signal."""

    signal: TradeSignal
    filled_qty: float
    executed_price: float
    commission: float
    slippage: float
    timestamp: pd.Timestamp


@dataclass(frozen=True)
class StopLossEvent:
    """Risk event for stop loss."""

    symbol: str
    timestamp: pd.Timestamp
    trigger_price: float
    reason: str = "sl"
    position_id: str = ""  # links to the Position that triggered
    position_qty: float = 0.0  # absolute qty of the triggering position
    position_type: Optional[ActionType] = None  # long/short of the triggering position


@dataclass(frozen=True)
class TakeProfitEvent:
    """Risk event for take profit."""

    symbol: str
    timestamp: pd.Timestamp
    trigger_price: float
    reason: str = "tp"
    position_id: str = ""  # links to the Position that triggered
    position_qty: float = 0.0  # absolute qty of the triggering position
    position_type: Optional[ActionType] = None  # long/short of the triggering position


RiskEvent = Tuple[StopLossEvent, TakeProfitEvent]


@dataclass(frozen=True)
class ExecutionParams:
    """Parameters for execution."""

    spread_bps: float = 5.0
    slippage_bps: float = 2.0
    fixed_commission: float = 0.5


@dataclass(frozen=True)
class RiskConfig:
    """Parameters for risk management."""

    stop_loss_pct: float
    take_profit_pct: float
    trailing_stop: bool = False


@dataclass(frozen=True)
class BacktestState:
    """Complete state snapshot at any point in time."""

    portfolio: PortfolioState
    timestamp: Optional[pd.Timestamp]
    pending_signals: Dict[str, Tuple[TradeSignal, ...]]  # symbol -> queued signals
    risk_events: Tuple[Any, ...]  # RiskEvent tuple
    candles: CandleStore = field(default_factory=lambda: CandleStore({}))


@dataclass(frozen=True)
class PortfolioResult:
    """Final portfolio results."""

    total_return: float
    sharpe_ratio: float
    trades: Tuple[Trade, ...]
    equity_curve: pd.Series
    annual_return: float = 0.0
    annual_volatility: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    stability: float = 0.0
    omega_ratio: float = 0.0
    capital_utilization: float = 0.0


@dataclass(frozen=True)
class BacktestResults:
    """Complete backtest results."""

    pf: PortfolioResult
    data: CandleStore  # full OHLCV history (cursor at end-of-data post-backtest); also accessible via final_state.candles
    final_state: BacktestState
    z_scores: Optional[pd.DataFrame] = None
    regimes: Optional[pd.DataFrame] = None
    benchmark_curves: dict[str, "pd.Series"] = field(default_factory=dict)
    # The config the run was executed with. Loose ``Any`` to avoid a
    # state -> bt.types (StrategyConfig) import edge; consumed post-run by
    # ``output.render_plot_json`` to resolve the strategy's plot fn + params.
    config: Any = None
