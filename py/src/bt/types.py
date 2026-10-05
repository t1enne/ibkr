from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Protocol, Tuple, Union
import pandas as pd

from src.bt.state import (  # noqa: F401
    Candle,
    TradeSignal,
    FillEvent,
    StopLossEvent,
    TakeProfitEvent,
    ExecutionParams,
    BacktestState,
    PortfolioState,
    PortfolioResult,
    BacktestResults,
    ActionType,
    TradeStatus,
    TradeExitReason,
    RiskConfig,
    CommissionModel,
    build_commission_model,
)


class ZScoreState:
    scores: List[float]
    timestamps: List[pd.Timestamp]
    scores_synced: List[float]
    timestamps_synced: List[pd.Timestamp]


class RegimeState:
    labels: List[Optional[int]]
    probs: List[Optional[List[float]]]
    timestamps: List[pd.Timestamp]


@dataclass(frozen=True)
class EngineWindow:
    """The engine's evaluation window.

    ``warmup_bars`` is how many bars are walked BEFORE ``test_start`` with the
    strategy invoked but unable to trade (see ``Backtest.window``).
    """

    warmup_bars: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp


#: Broker names ``StrategyConfig.broker`` accepts; a ``Literal`` so a typo is a
#: type error, with a runtime re-check for untyped JSON input.
BrokerName = Literal["sim", "ibkr"]


@dataclass
class StrategyConfig:
    name: str
    strategy_type: str
    symbols: list[str]
    initial_capital: float
    commission: float
    # Calendar duration string (e.g. "1y", "6m", "90d") walked before
    # ``trading_start`` with the strategy invoked but trading suppressed. The
    # engine derives ``warmup_bars`` from it at the config's base interval
    # (``src.bt.warmup``); a strategy's OWN bar-count readiness gate is what
    # actually guarantees enough accumulated state.
    warmup: str
    trading_start: str
    trading_end: str
    bars: list[str]
    # the strategy_params will be passed to the strategy raw.
    # Position sizing + stop-loss/take-profit live here (strategy-owned,
    # per-trade). They are NOT config-level fields.
    strategy_params: dict
    rolling_window_size: Optional[int] = None
    benchmark_symbols: list[str] = field(default_factory=lambda: ["SPY"])
    # Research control: when False, a multi-open cohort that exceeds available
    # cash is NOT uniformly scaled — fills settle sequentially and the
    # over-cash tail is rejected (order-sensitive). Default True keeps the
    # order-invariant cohort scaling that shipped. See ``_scale_opens``.
    cohort_scaling: bool = True
    # Execution friction. ``spread_bps``/``slippage_bps`` are the config-visible
    # levers for the hardcoded execution defaults; ``commission_per_share`` opts
    # into the IBKR-style per-share charge (with ``commission_min`` floor and
    # ``commission_max_pct`` percent-of-value cap). ``None`` keeps the legacy
    # flat ``commission`` fallback, so existing strategy JSONs are unaffected.
    spread_bps: float = 5.0
    slippage_bps: float = 2.0
    commission_per_share: float | None = None
    commission_min: float = 0.0
    commission_max_pct: float | None = None
    # Which broker the strategy trades through: ``"sim"`` (the candle
    # ``SimExchange`` backtest adapter) or ``"ibkr"`` (the live edge). Default
    # preserves every existing JSON's behaviour. An unrecognised value fails
    # loudly at construction rather than silently falling back to the sim.
    # NOTE: the live CLI (``ibkr live run``) resolves its adapter from the
    # config's ``broker`` key through ``resolve_broker`` (phase 2); it reads the
    # key once, not this field.
    broker: BrokerName = "sim"

    def __post_init__(self) -> None:
        if self.broker not in ("sim", "ibkr"):
            raise ValueError(
                f"Unknown broker {self.broker!r}; expected one of ['ibkr', 'sim']"
            )


def commission_model_from_config(cfg: StrategyConfig) -> CommissionModel:
    """Resolve the strategy config's commission shape into a tagged model.

    ``commission_per_share`` set -> per-share model; otherwise the flat
    ``commission`` fallback, keeping every legacy JSON numerically unchanged.
    """
    return build_commission_model(
        cfg.commission,
        cfg.commission_per_share,
        cfg.commission_min,
        cfg.commission_max_pct,
    )


RiskEvent = Union[StopLossEvent, TakeProfitEvent]


class ExecutionFn(Protocol):
    """Protocol for signal execution function."""

    def __call__(
        self,
        signal: TradeSignal,
        tick: Candle,
        exec_params: ExecutionParams,
    ) -> FillEvent: ...


class PositionSizerFn(Protocol):
    """Protocol for position sizing and fill application."""

    def __call__(
        self,
        portfolio: PortfolioState,
        fill: FillEvent,
        sizing_params: Dict[str, float],
    ) -> PortfolioState: ...


class RiskCheckFn(Protocol):
    """Protocol for risk checking function."""

    def __call__(
        self,
        portfolio: PortfolioState,
        tick: Candle,
        config: RiskConfig,
    ) -> Tuple[Tuple[RiskEvent, ...], PortfolioState]: ...


class DataLoaderFn(Protocol):
    """Protocol for data loading function."""

    def __call__(
        self,
        symbols: List[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        bar: str,
    ) -> pd.DataFrame: ...
