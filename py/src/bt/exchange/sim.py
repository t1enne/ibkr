"""The backtest broker adapter: matcher + friction + cohort cash scaling.

``SimExchange`` is the backtest's implementation of the shared ``Broker`` port
(and of the bt-local ``FillSurface``). It owns three things:

- the legacy fill surface the engine drives (``execute_signal`` /
  ``execute_risk_event`` / ``apply_fill``), preserved byte-for-byte from the
  old ``src/bt/execution/pure.py`` so no strategy's numbers move, now using the
  shared friction core;
- cohort settlement (``settle_cohort``) — the order-invariant ``apply_fills``
  cash scaling that is a property of the SIM book, not of the shared core;
- the new order surface (``submit``/``cancel``/``fills``/``match_bar``) backed by
  the shared pure candle matcher in ``src/exec/matching.py``.

Only the plain MKT "next bar open" fill is shared with the matcher; the
same-bar and guard-price base-price paths stay here because they are
backtest-only behaviours with no broker analogue.

``Fill.price`` has ONE meaning across both surfaces: an EXECUTED price (base
plus friction), with ``commission``/``spread``/``slippage`` costs populated. The
pure matcher returns the frictionless base; ``match_bar`` is what applies the
adapter's friction, reusing the same adverse rule as the signal path.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import pandas as pd

from src.bt.portfolio.pure import (
    FillRejection,
    ScaleRecord,
    apply_fill,
    apply_fills,
)
from src.bt.state.types import (
    ActionType,
    Candle,
    CommissionModel,
    ExecutionParams,
    FillEvent,
    PortfolioState,
    TradeSignal,
)
from src.bt.types import RiskEvent
from src.exec.friction import apply_friction, commission_for_fill
from src.exec.matching import match_bar as _match_bar
from src.exec.types import (
    Fill,
    OrderAck,
    OrderRequest,
    OrderSide,
    OrderState,
    OrderType,
    RejectReason,
)

#: Adverse slippage multiplier shared by the signal and risk paths.
_ADVERSE_MULTIPLIER = 1.5


def is_buy_fill(action: ActionType, position_side: ActionType | None) -> bool:
    """Whether a fill *adds* (buys) rather than reduces (sells).

    Opens are decided by the action; a close/rebalance is a buy iff it acts on a
    short (buy-to-cover), else a sell. ``position_side=None`` defaults to a sell.
    """
    if action == ActionType.long:
        return True
    if action == ActionType.short:
        return False
    return position_side == ActionType.short


def _adverse_move(action: ActionType, open_: float, close: float) -> bool:
    """Whether a bar moved adversely for ``action`` by more than 0.1%."""
    price_move = close - open_
    percent_move = price_move / open_ if open_ != 0 else 0
    if action == ActionType.long:
        return percent_move < -0.001
    if action == ActionType.short:
        return percent_move > 0.001
    return False


def calculate_adverse_selection(signal: TradeSignal, tick: Candle) -> bool:
    """Determine if slippage should be adverse."""
    return _adverse_move(signal.action, tick.open, tick.close)


def _side_action(side: OrderSide) -> ActionType:
    """Map a shared order side onto the signal action that shares its friction."""
    return ActionType.long if side is OrderSide.BUY else ActionType.short


def _signal_is_buy(signal: TradeSignal) -> bool:
    """The buy/sell direction the signal's friction leans on."""
    side = signal.position_side
    if side is None and signal.fill_guard_is_long is not None:
        side = ActionType.long if signal.fill_guard_is_long else ActionType.short
    if signal.action == ActionType.rebalance:
        return signal.qty > 0  # a rebalance adds when its delta is positive
    return is_buy_fill(signal.action, side)


def _fill_event(
    signal: TradeSignal,
    *,
    tick: Candle,
    base_price: float,
    qty: float,
    is_buy: bool,
    params: ExecutionParams,
    adverse_multiplier: float,
) -> FillEvent:
    """Apply friction + commission to a base price and build the ``FillEvent``.

    The ONE place a bt fill is priced, so the signal, risk and matcher paths
    cannot charge costs differently.
    """
    friction = apply_friction(
        base_price,
        is_buy=is_buy,
        spread_bps=params.spread_bps,
        slippage_bps=params.slippage_bps,
        qty=qty,
        adverse_multiplier=adverse_multiplier,
    )
    return FillEvent(
        signal=signal,
        filled_qty=qty,
        executed_price=friction.executed_price,
        commission=commission_for_fill(
            params.commission_model, qty, friction.executed_price
        ),
        slippage=friction.slippage_cost,
        spread=friction.spread_cost,
        timestamp=tick.timestamp,
    )


def _signal_base_price(signal: TradeSignal, tick: Candle) -> float:
    """The pre-friction base price for a signal fill.

    Next-open fills use ``tick.open``; same-bar fills use ``signal.price``. A
    next-open **close** that models an intra-bar stop trigger
    (``fill_guard_price`` set) fills at the adverse worse-of the guard and the
    next open, mirroring :func:`execute_risk_event` gap-through math: a long
    close (sell) fills at ``min(guard, open)`` (the stop level, or lower if the
    bar gapped down through it), a short close (buy) at ``max(guard, open)``.
    """
    if (
        signal.action == ActionType.close
        and signal.fill_at_next_open
        and signal.fill_guard_price is not None
        and signal.fill_guard_is_long is not None
    ):
        if signal.fill_guard_is_long:
            return min(signal.fill_guard_price, tick.open)
        return max(signal.fill_guard_price, tick.open)
    return tick.open if signal.fill_at_next_open else signal.price


def execute_signal(
    signal: TradeSignal, tick: Candle, params: ExecutionParams
) -> FillEvent:
    """Convert signal to fill with slippage/spread.

    Pure function - no side effects. When ``signal.fill_at_next_open`` is True,
    the base price is this (fill) bar's open; otherwise ``signal.price``.
    """
    adverse = calculate_adverse_selection(signal, tick)
    return _fill_event(
        signal,
        tick=tick,
        base_price=_signal_base_price(signal, tick),
        qty=signal.qty,
        is_buy=_signal_is_buy(signal),
        params=params,
        adverse_multiplier=_ADVERSE_MULTIPLIER if adverse else 1.0,
    )


def _risk_base_price(event: RiskEvent, tick: Candle) -> float:
    """Trigger price adjusted for a gap through the level (worse-of open).

    A stop that is gapped through is filled at the open, not the trigger;
    a take-profit gap fills favorably through the level.
    """
    trigger = event.trigger_price
    is_long = event.position_type == ActionType.long
    if event.reason == "sl":
        return min(trigger, tick.open) if is_long else max(trigger, tick.open)
    return max(trigger, tick.open) if is_long else min(trigger, tick.open)


def execute_risk_event(
    event: RiskEvent, tick: Candle, params: ExecutionParams
) -> FillEvent:
    """Convert risk event (SL/TP) into fill event, modeling intra-bar gaps.

    A stop-loss/take-profit event fires because the bar's high/low crossed the
    trigger level. When the bar *gaps through* the level, the real fill is not
    the trigger price but the worse-of-trigger-and-open price:

    - Long stop  (trigger when price falls to ``trigger``): a bar that opens
      below the stop has already gapped past it downward — fill at the open.
    - Short stop (trigger when price rises to ``trigger``): a bar that opens
      above the stop has gapped past it upward — fill at the open.
    - Take-profit gaps go *favorably* through the level: a conservative fill
      takes at least the trigger, and at the gap-open when the open is already
      beyond it (e.g. a long TP gapped open above the target is filled at the
      higher open).

    Without this, a stop that is gapped through is happily filled at the
    trigger every time, which systematically overstates P&L on gap days.
    """
    is_long = event.position_type == ActionType.long
    # A stop/take-profit always closes against the position: a sell for a long,
    # a buy-to-cover for a short; slippage is always adverse.
    signal = TradeSignal(
        action=ActionType.close,
        symbol=event.symbol,
        timestamp=event.timestamp,
        price=event.trigger_price,
        reason=event.reason,
        position_id=event.position_id,
        position_side=event.position_type,
        qty=event.position_qty,
    )
    return _fill_event(
        signal,
        tick=tick,
        base_price=_risk_base_price(event, tick),
        qty=event.position_qty,
        is_buy=not is_long,
        params=params,
        adverse_multiplier=_ADVERSE_MULTIPLIER,
    )


class SimExchange:
    """Backtest ``Broker``: prices fills, settles cohorts, matches new orders.

    The engine drives it through ``execute_signal``/``execute_risk_event``/
    ``apply_fill``/``settle_cohort`` (the ``FillSurface`` port). The order-port
    methods (``submit``/``cancel``/``fills``) and ``match_bar`` are the new
    MKT/LMT surface: ``submit`` stages a validated order, ``match_bar`` resolves
    a *pending* order against a bar through the shared matcher, applies friction,
    and records the fill.
    """

    def __init__(self) -> None:
        self._pending: dict[str, OrderRequest] = {}
        self._fills: list[Fill] = []

    # ── Broker port ──────────────────────────────────────────────────────
    def submit(self, order: OrderRequest) -> OrderAck:
        """Stage an order for matching; reject invalid or duplicate refs.

        An ``LMT`` with no ``limit_price`` can never fill, and a duplicate
        ``order_ref`` would silently replace a live order — both are rejected as
        a value (``INVALID``), never by raising.
        """
        if order.order_type is OrderType.LMT and order.limit_price is None:
            return self._reject(order.order_ref)
        if order.order_ref in self._pending:
            return self._reject(order.order_ref)
        self._pending[order.order_ref] = order
        return OrderAck(
            order_ref=order.order_ref, accepted=True, state=OrderState.PENDING
        )

    def cancel(self, order_ref: str) -> OrderAck:
        """Drop a staged order; unknown refs are rejected, not raised."""
        if self._pending.pop(order_ref, None) is None:
            return OrderAck(
                order_ref=order_ref,
                accepted=False,
                state=OrderState.REJECTED,
                reason=RejectReason.UNKNOWN_ORDER,
            )
        return OrderAck(order_ref=order_ref, accepted=True, state=OrderState.CANCELLED)

    def fills(self) -> tuple[Fill, ...]:
        """Every fill the exchange has produced this session."""
        return tuple(self._fills)

    def close(self) -> None:
        """End the session: drop pending orders (positions are not flattened)."""
        self._pending.clear()

    @staticmethod
    def _reject(order_ref: str) -> OrderAck:
        return OrderAck(
            order_ref=order_ref,
            accepted=False,
            state=OrderState.REJECTED,
            reason=RejectReason.INVALID,
        )

    # ── Exchange-shaped surface (friction-bearing, so not the pure Exchange) ──
    def match_bar(
        self,
        order: OrderRequest,
        bars: pd.DataFrame,
        *,
        spread_bps: float,
        slippage_bps: float,
        commission_model: CommissionModel,
    ) -> Optional[Fill]:
        """Resolve a *pending* order against a bar; record the executed fill.

        Only orders that were submitted and are still pending resolve — a
        cancelled or unknown ``order_ref`` returns ``None``. The returned
        ``Fill.price`` is ALWAYS an executed price (base plus half-spread and
        slippage, adverse-selected with the SAME rule as the signal path) with
        ``commission``/``spread``/``slippage`` costs populated, so ``Fill.price``
        means one thing wherever it appears.
        """
        if order.order_ref not in self._pending:
            return None
        matched = _match_bar(order, bars)
        if matched is None:
            return None
        row = bars.iloc[0]
        adverse = _adverse_move(_side_action(order.side), row["open"], row["close"])
        friction = apply_friction(
            matched.price,
            is_buy=order.side is OrderSide.BUY,
            spread_bps=spread_bps,
            slippage_bps=slippage_bps,
            qty=order.qty,
            adverse_multiplier=_ADVERSE_MULTIPLIER if adverse else 1.0,
        )
        fill = replace(
            matched,
            price=friction.executed_price,
            commission=commission_for_fill(
                commission_model, order.qty, friction.executed_price
            ),
            spread=friction.spread_cost,
            slippage=friction.slippage_cost,
        )
        self._pending.pop(order.order_ref, None)
        self._fills.append(fill)
        return fill

    # ── Fill surface (parity path) ───────────────────────────────────────
    def execute_signal(
        self, signal: TradeSignal, tick: Candle, params: ExecutionParams
    ) -> FillEvent:
        """Price one signal; delegates to the shared signal-fill function."""
        return execute_signal(signal, tick, params)

    def execute_risk_event(
        self, event: RiskEvent, tick: Candle, params: ExecutionParams
    ) -> FillEvent:
        """Price one SL/TP event; delegates to the shared gap-through function."""
        return execute_risk_event(event, tick, params)

    def apply_fill(self, portfolio: PortfolioState, fill: FillEvent) -> PortfolioState:
        """Fold one fill into the book (single-fill path used by risk closes)."""
        return apply_fill(portfolio, fill)

    def settle_cohort(
        self,
        portfolio: PortfolioState,
        fills: tuple[FillEvent, ...],
        *,
        scale_cohorts: bool = True,
        commission_model: CommissionModel,
    ) -> tuple[PortfolioState, tuple[FillRejection, ...], tuple[ScaleRecord, ...]]:
        """Settle a whole bar's fills atomically via the shared cohort scaler.

        ``commission_model`` is required: the cohort scaler commissions each fill
        and a silent default would quietly charge the wrong costs.
        """
        return apply_fills(
            portfolio,
            fills,
            scale_cohorts=scale_cohorts,
            commission_model=commission_model,
        )


def default_exchange() -> SimExchange:
    """Create the default backtest exchange (matcher + friction + cohort scale)."""
    return SimExchange()


__all__ = [
    "SimExchange",
    "calculate_adverse_selection",
    "default_exchange",
    "execute_risk_event",
    "execute_signal",
    "is_buy_fill",
]
