"""Live domain types — the shared vocabulary between the live edges and the pure core.

One portfolio interface end to end: live reuses the backtest ``PortfolioState``
and ``Position`` (no parallel ``LivePortfolio``), so ``apply_fill`` /
``apply_fills`` / ``equity_of`` apply unchanged to a live book and ``reconcile``
depends only on the minimal :class:`PortfolioView` Protocol that
``PortfolioState`` satisfies structurally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast, get_args

import pandas as pd

from src.bt.state import ActionType, ExecutionParams, PortfolioState
from src.bt.state import PortfolioView as PortfolioView  # re-exported live vocabulary
from src.bt.state.factories import build_commission_model, create_execution_params
from src.config import live_adapter
from src.exec.types import OrderType

if TYPE_CHECKING:
    # Type-only: ``scope`` imports ``LiveConfig`` from here, so a runtime import
    # would be a cycle. ``AdapterName`` is the ONE definition of the adapter
    # vocabulary (never redefined in this module).
    from src.live.scope import AdapterName

#: Actions the screen can emit that require a live decision. ``flat`` is never
#: produced: absence of a signal is HOLD downstream.
SignalAction = Literal["long", "short", "close"]

#: Which source produced a cost figure (plan §7.3). ``broker_executions`` is the
#: broker's own per-execution number (exact); ``modelled`` is the sim
#: ``CommissionModel`` approximation.
CostSource = Literal["broker_executions", "modelled"]


@dataclass(frozen=True)
class CostProvenance:
    """Which source produced each cost a live run reports (plan §7.3).

    A live run can MIX sources, so this names BOTH sides rather than a single
    tag: the IBKR read path books the broker's exact per-execution commission
    into cash/lots (``bookkeeping``), while order sizing still runs the sim
    commission model (``sizing``) — the broker does not report a fee for an
    order it has not filled yet. A ``sim`` run is modelled on both sides. Two
    explicit fields, never a lone "mixed" token, is what keeps the mix
    unambiguous.
    """

    bookkeeping: CostSource
    sizing: CostSource


#: The all-modelled provenance: a sim run books and sizes from the same model.
MODELLED_COST = CostProvenance(bookkeeping="modelled", sizing="modelled")


def cost_provenance(adapter: str) -> CostProvenance:
    """Cost provenance for the adapter that ran: broker-exact book on IBKR only."""
    if adapter == "ibkr":
        return CostProvenance(bookkeeping="broker_executions", sizing="modelled")
    return MODELLED_COST


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

    ``qty`` is the row's executable share count: > 0 when the strategy sized
    its own open (``ctx.long(..., size=, size_mode=)``), ``0.0`` when the open
    is engine-sized. A ``0.0`` here means "unsized — size at reconcile".
    """

    symbol: str
    action: SignalAction
    score: float
    reasons: tuple[str, ...]
    signal_ts: pd.Timestamp | None
    price: float  # ref price (last close of the decision bar)
    qty: float  # absolute shares; 0.0 = unsized (size from config)
    #: The newest data bar this signal was decided on. AUDIT ONLY on the live
    #: path: the order cOID is bar-free (``src.live.identity`` — a scope-token-
    #: attempt), NOT anchored here, so a new bar never mints a new ref (INV-2).
    bar_ts: pd.Timestamp | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    #: Target lot for a close, in the LIVE book's id space (conid). The screen
    #: bridge never sets it — a ``ScreenRow`` id is a backtest lot id, which
    #: matches no live lot; absent means "close the symbol's whole owned book".
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
    #: The decision bar this intent was derived from (see ``LiveSignal.bar_ts``).
    #: AUDIT ONLY on the live path — the broker does NOT anchor the cOID on it
    #: (the cOID is the bar-free ``identity`` ref), so this is recorded for audit
    #: and for a DAY order's rollover check, never for identity (INV-2).
    decision_ts: pd.Timestamp | None = None
    #: The shared order vocabulary. Phase 3 places MKT only; an LMT intent is
    #: refused by the IBKR adapter until carry-over policy exists (phase 4).
    order_type: OrderType = OrderType.MKT
    #: On an OPEN, the scope's funded cash the sizer clamped the quantity against
    #: at decision time (the SAME ``PortfolioView.cash`` reconcile passed to
    #: ``sized_signal``). The IBKR edge re-checks the order notional against it,
    #: so it bounds the DECISION notional: a notional within the tolerance, the
    #: whole-share round-up, a funding close that never fills, and a mis-modelled
    #: commission are all outside what this bound can guarantee. ``None`` on a
    #: close, and on an open from a caller that cannot state it — the edge then
    #: refuses (fail-closed) rather than trading unbounded.
    cash_bound: float | None = None


#: Every ``FeedError.kind`` a value may carry. The ONE definition: ``FeedError``
#: uses it directly and ``feed_error`` validates against it, so the set and the
#: Literal cannot drift apart.
FeedKind = Literal[
    "auth",
    "rate_limit",
    "transport",
    "bad_fixture",
    "stale_data",
    "rejected",
    "unfilled",
    "timeout",
    "unresolved",
    #: The account net and the ledger's booked exposure on a conid disagree: a
    #: fill we cannot see may be live (or our book is ahead of the account). An
    #: OPEN is refused rather than placed on top of an unexplained position. The
    #: same kind carries a refused CLOSE (the account net cannot absorb the
    #: reducing order), so the reducing side surfaces symmetrically.
    "divergence",
    #: An OPEN dropped by a STRUCTURAL cohort rule (no provable shared cash bound,
    #: or a scaled qty flooring to 0 shares) — distinct from a broker ``rejected``,
    #: which is genuine cash exhaustion for this bar and re-mints next cycle.
    "unfunded",
]

#: Every ``FeedError.kind`` a value may carry, derived from the Literal above.
_FEED_KINDS = frozenset(get_args(FeedKind))


@dataclass(frozen=True)
class FeedError:
    """A typed edge failure (portfolio fetch / order placement / data staleness)."""

    kind: FeedKind
    message: str
    symbol: str | None = None
    #: Shares the order actually filled, when the edge knows the number (``None``
    #: when it does not). A partial fill is a normal outcome rather than a failure
    #: kind, so it travels here and is reported as a shortfall instead of living
    #: only inside ``message``.
    filled_qty: float | None = None


def feed_error(
    kind: str,
    message: str,
    symbol: str | None = None,
    filled_qty: float | None = None,
) -> FeedError:
    """Build a typed ``FeedError`` from a client ``ErrorKind`` (a validating identity).

    The client's kinds (``auth``/``rate_limit``/``transport``) are a subset of
    ``FeedError``'s, so this is the ONE definition of the mapping — previously
    duplicated in ``data.ibkr.gateway`` and ``adapters.ibkr.portfolio_source``.
    An unknown kind degrades to ``transport`` (never invent a kind the type
    forbids).
    """
    resolved = kind if kind in _FEED_KINDS else "transport"
    return FeedError(
        kind=cast("FeedKind", resolved),
        message=message,
        symbol=symbol,
        filled_qty=filled_qty,
    )


@dataclass(frozen=True)
class LiveConfig:
    """Everything one live cycle needs, resolved from the strategy JSON."""

    strategy_type: str
    symbols: tuple[str, ...]
    initial_capital: float
    strategy_params: dict[str, object]  # screen strategy params; never mutated
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
    # The resolved backend this config trades through (plan §2.2). The CLI
    # overrides it with ``--adapter`` when that flag is passed explicitly; absent
    # a flag this is what ``resolve_adapter_name`` resolves, falling back to the
    # back-compat ``broker`` key below, then ``config.toml``'s ``[live] adapter``.
    adapter: AdapterName = cast("AdapterName", live_adapter())
    # Which broker this config TRADES through, kept as the back-compat source for
    # adapter resolution (``resolve_broker``) and as a description of the config.
    # A run's actual backend is ``adapter`` — this no longer selects it.
    broker: Literal["sim", "ibkr"] = cast("Literal['sim', 'ibkr']", live_adapter())
    #: The config's ``name`` — the ``<config_name>`` scope segment.
    config_name: str = ""


def exec_params_of(config: LiveConfig) -> ExecutionParams:
    """The execution params a live cycle fills with, from its config.

    ONE construction, shared by ``reconcile``'s sizing book and the CLI's sim
    broker, so sizing and settlement agree byte-for-byte (finding L8).
    """
    return create_execution_params(
        spread_bps=config.spread_bps,
        slippage_bps=config.slippage_bps,
        fixed_commission=config.commission,
        commission_model=build_commission_model(
            config.commission,
            config.commission_per_share,
            config.commission_min,
            config.commission_max_pct,
        ),
    )
