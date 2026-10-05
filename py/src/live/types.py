"""Live domain types — the shared vocabulary between the live edges and the pure core.

One portfolio interface end to end: live reuses the backtest ``PortfolioState``
and ``Position`` (no parallel ``LivePortfolio``), so ``apply_fill`` /
``apply_fills`` / ``equity_of`` apply unchanged to a live book and ``reconcile``
depends only on the minimal :class:`PortfolioView` Protocol that
``PortfolioState`` satisfies structurally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import pandas as pd

from src.bt.state import ActionType, PortfolioState
from src.bt.state import PortfolioView as PortfolioView  # re-exported live vocabulary
from src.exec.types import OrderType

#: Actions the screen can emit that require a live decision. ``flat`` is never
#: produced: absence of a signal is HOLD downstream.
SignalAction = Literal["long", "short", "close"]


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
    #: The shared order vocabulary. Phase 3 places MKT only; an LMT intent is
    #: refused by the IBKR adapter until carry-over policy exists (phase 4).
    order_type: OrderType = OrderType.MKT


@dataclass(frozen=True)
class FeedError:
    """A typed edge failure (portfolio fetch / order placement / data staleness)."""

    kind: Literal[
        "auth",
        "rate_limit",
        "transport",
        "bad_fixture",
        "stale_data",
        "rejected",
        "unfilled",
        "timeout",
    ]
    message: str
    symbol: str | None = None


#: Every ``FeedError.kind`` a value may carry.
_FEED_KINDS = frozenset(
    {
        "auth",
        "rate_limit",
        "transport",
        "bad_fixture",
        "stale_data",
        "rejected",
        "unfilled",
        "timeout",
    }
)


def feed_error(kind: str, message: str, symbol: str | None = None) -> FeedError:
    """Build a typed ``FeedError`` from a client ``ErrorKind`` (a validating identity).

    The client's kinds (``auth``/``rate_limit``/``transport``) are a subset of
    ``FeedError``'s, so this is the ONE definition of the mapping — previously
    duplicated in ``data.ibkr.gateway`` and ``adapters.ibkr.portfolio_source``.
    An unknown kind degrades to ``transport`` (never invent a kind the type
    forbids).
    """
    resolved = kind if kind in _FEED_KINDS else "transport"
    return FeedError(
        kind=cast(
            "Literal['auth', 'rate_limit', 'transport', 'bad_fixture', 'stale_data', 'rejected', 'unfilled', 'timeout']",
            resolved,
        ),
        message=message,
        symbol=symbol,
    )


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
    # Execution friction mirroring StrategyConfig: per-share commission opts in
    # via ``commission_per_share``; ``None`` keeps the flat ``commission``.
    spread_bps: float = 5.0
    slippage_bps: float = 2.0
    commission_per_share: float | None = None
    commission_min: float = 0.0
    commission_max_pct: float | None = None
    # sizing (used only when a LiveSignal.qty == 0.0)
    size_mode: Literal["equity", "cash", "fixed"] = "equity"
    size: float = 0.0
    max_symbol_allocation: float = 1.0
    portfolio_path: str = ""  # MockPortfolioSource fixture path
    mode: Literal["paper", "live"] = "paper"
    # Which broker this config trades through. Mirrors ``StrategyConfig.broker``
    # (phase 1.5) and is the field ``ibkr live run`` resolves its adapter from
    # when no ``--adapter`` flag is given.
    broker: Literal["sim", "ibkr"] = "sim"
