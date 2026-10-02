"""Live domain types — the shared vocabulary between the live edges and the pure core.

One portfolio interface end to end: live reuses the backtest ``PortfolioState``
and ``Position`` (no parallel ``LivePortfolio``), so ``apply_fill`` /
``apply_fills`` / ``equity_of`` apply unchanged to a live book and ``reconcile``
depends only on the minimal :class:`PortfolioView` Protocol that
``PortfolioState`` satisfies structurally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import pandas as pd

from src.bt.state import ActionType, PortfolioState, Position

#: Actions the screen can emit that require a live decision. ``flat`` is never
#: produced: absence of a signal is HOLD downstream.
SignalAction = Literal["long", "short", "close"]


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
    def positions(self) -> dict[str, tuple[Position, ...]]: ...  # symbol -> LOTs

    @property
    def initial_capital(self) -> float: ...


@dataclass(frozen=True)
class PortfolioSnapshot:
    """A live broker read: the shared ``PortfolioState`` + when it was read.

    Live deliberately reuses ``PortfolioState``/``Position`` rather than a
    parallel ``LivePortfolio``: one portfolio interface end to end, so
    ``apply_fill`` / ``apply_fills`` / ``equity_of`` apply unchanged to a live
    book. ``as_of`` is the only live-only addition. ``trades`` /
    ``equity_curve`` may be empty on a live read.
    """

    portfolio: PortfolioState
    as_of: pd.Timestamp


@dataclass(frozen=True)
class LiveSignal:
    """Actionable intent from the screen (mirrors ``ScreenRow``, actionable only).

    ``qty`` is always ``0.0`` from the current screen bridge: a ``ScreenRow``
    carries no quantity, so sizing must come from config via ``SizingParams``.
    A ``0.0`` here means "unsized — size at reconcile".
    """

    symbol: str
    action: SignalAction
    score: float
    reasons: tuple[str, ...]
    signal_ts: pd.Timestamp | None
    price: float  # ref price (last close of the decision bar)
    qty: float  # absolute shares; 0.0 = unsized (size from config)
    stop_loss: float | None = None
    take_profit: float | None = None
    position_id: str | None = None
    tag: str = ""


@dataclass(frozen=True)
class OrderIntent:
    """A single order to submit, after reconciliation. Never a batch."""

    symbol: str
    action: ActionType  # long / short / close
    qty: float  # absolute shares, always > 0
    ref_price: float
    reason: str  # e.g. "open long (flat->long)" / "close all lots"
    position_id: str | None = None  # target lot for a close; None = whole symbol
    stop_loss: float | None = None
    take_profit: float | None = None
    tag: str = ""


@dataclass(frozen=True)
class FeedError:
    """A typed edge failure (portfolio fetch / order placement / data staleness)."""

    kind: Literal["auth", "rate_limit", "transport", "bad_fixture", "stale_data"]
    message: str
    symbol: str | None = None


@dataclass(frozen=True)
class LiveConfig:
    """Everything one live cycle needs, resolved from the strategy JSON."""

    strategy_type: str
    symbols: tuple[str, ...]
    initial_capital: float
    strategy_params: dict[str, object]  # the screen's strategy params (verbatim)
    bars: tuple[str, ...]  # bars[0] = signal interval
    warmup: str  # screen warm-up window, e.g. "1y"
    commission: float = 0.5  # fixed per-fill commission, matches StrategyConfig
    # sizing (used only when a LiveSignal.qty == 0.0)
    size_mode: Literal["equity", "cash", "fixed"] = "equity"
    size: float = 0.0
    max_symbol_allocation: float = 1.0
    portfolio_path: str = ""  # MockPortfolioSource fixture path
    mode: Literal["paper", "live"] = "paper"
