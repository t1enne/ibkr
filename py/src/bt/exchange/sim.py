"""The backtest broker adapter: matcher + friction + cohort cash scaling.

``SimExchange`` is the backtest's implementation of the shared ``Broker`` port.
It owns three things:

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
"""

from __future__ import annotations

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
    FixedCommission,
    PortfolioState,
    TradeSignal,
)
from src.exec.friction import apply_friction, commission_for_fill
from src.exec.matching import match_bar as _match_bar
from src.exec.types import (
    Fill,
    OrderAck,
    OrderRequest,
    OrderState,
    RejectReason,
)


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


def calculate_adverse_selection(signal: TradeSignal, tick: Candle) -> bool:
    """Determine if slippage should be adverse."""
    price_move = tick.close - tick.open
    percent_move = price_move / tick.open if tick.open != 0 else 0

    if signal.action == ActionType.long:
        return percent_move < -0.001
    elif signal.action == ActionType.short:
        return percent_move > 0.001
    return False


def execute_signal(
    signal: TradeSignal, tick: Candle, params: ExecutionParams
) -> FillEvent:
    """Convert signal to fill with slippage/spread.

    Pure function - no side effects.

    When signal.fill_at_next_open is True, uses tick.open as the base price
    (realistic: signals generated at close fill at next bar's open).
    Otherwise uses signal.price (same-bar fill at signal generation price).

    A next-open **close** that models an intra-bar stop trigger
    (``fill_guard_price`` set) fills at the adverse worse-of the guard and the
    next open, mirroring :func:`execute_risk_event` gap-through math: a long
    close (sell) fills at ``min(guard, open)`` (the stop level, or lower if the
    bar gapped down through it), a short close (buy) at ``max(guard, open)``.
    """
    base_price = tick.open if signal.fill_at_next_open else signal.price
    if (
        signal.action == ActionType.close
        and signal.fill_at_next_open
        and signal.fill_guard_price is not None
        and signal.fill_guard_is_long is not None
    ):
        if signal.fill_guard_is_long:
            base_price = min(signal.fill_guard_price, tick.open)
        else:
            base_price = max(signal.fill_guard_price, tick.open)

    qty = signal.qty
    side = signal.position_side
    if side is None and signal.fill_guard_is_long is not None:
        side = ActionType.long if signal.fill_guard_is_long else ActionType.short
    if signal.action == ActionType.rebalance:
        is_buy = signal.qty > 0  # a rebalance adds when its delta is positive
    else:
        is_buy = is_buy_fill(signal.action, side)

    adverse = calculate_adverse_selection(signal, tick)
    friction = apply_friction(
        base_price,
        is_buy=is_buy,
        spread_bps=params.spread_bps,
        slippage_bps=params.slippage_bps,
        qty=qty,
        adverse_multiplier=1.5 if adverse else 1.0,
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


def execute_risk_event(
    event: object, tick: Candle, params: ExecutionParams
) -> FillEvent:
    """Convert risk event (SL/TP) into fill event, modeling intra-bar gaps.

    A stop-loss/take-profit event fires because the bar's high/low crossed
    the trigger level. When the bar *gaps through* the level, the real fill
    is not the trigger price but the worse-of-trigger-and-open price:

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
    pid = getattr(event, "position_id", "")
    qty = getattr(event, "position_qty", 0.0)
    position_type = getattr(event, "position_type", None)

    trigger = getattr(event, "trigger_price")
    is_stop = getattr(event, "reason", "") == "sl"
    is_long = position_type == ActionType.long

    # Determine the fill base price, accounting for a gap through the level.
    if is_stop:
        # Adverse direction: the loss-worse side of the trigger.
        if is_long:
            # Long stop: an open below the stop means we filled at the gap-open
            # (worse than trigger). Guard against tick.open falling below trigger.
            fill_base = min(trigger, tick.open)
        else:
            # Short stop: an open above the stop means we filled at the gap-open
            # (worse than trigger, i.e. higher buy-back price = bigger loss).
            fill_base = max(trigger, tick.open)
    else:
        # Take-profit: favorable direction. Take at least the trigger; if the
        # open already gapped past it favorably, capture the better open.
        if is_long:
            # Long TP: open above target is a favorable gap.
            fill_base = max(trigger, tick.open)
        else:
            # Short TP: open below target is a favorable gap.
            fill_base = min(trigger, tick.open)

    # A stop/take-profit always closes against the position, so the fill is a
    # sell for a long and a buy-to-cover for a short; slippage is always adverse.
    friction = apply_friction(
        fill_base,
        is_buy=not is_long,
        spread_bps=params.spread_bps,
        slippage_bps=params.slippage_bps,
        qty=qty,
        adverse_multiplier=1.5,
    )

    signal = TradeSignal(
        action=ActionType.close,
        symbol=getattr(event, "symbol"),
        timestamp=getattr(event, "timestamp"),
        price=trigger,
        reason=getattr(event, "reason", None),
        position_id=pid,
        position_side=position_type,
        qty=qty,
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


class SimExchange:
    """Backtest ``Broker``: prices fills, settles cohorts, matches new orders.

    The engine drives it through ``execute_signal``/``execute_risk_event``
    (parity path) and ``settle_cohort`` (cash scaling). The order-port methods
    (``submit``/``cancel``/``fills``) and ``match_bar`` are the new MKT/LMT
    surface: ``submit`` stages an order, ``match_bar`` resolves it against a bar
    through the shared matcher and records the fill.
    """

    def __init__(self) -> None:
        self._pending: dict[str, OrderRequest] = {}
        self._fills: list[Fill] = []

    # ── Broker port ──────────────────────────────────────────────────────
    def submit(self, order: OrderRequest) -> OrderAck:
        """Stage an order for matching; ack it as pending."""
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

    # ── Exchange port ────────────────────────────────────────────────────
    def match_bar(self, order: OrderRequest, bars: pd.DataFrame) -> Optional[Fill]:
        """Resolve one order against a bar via the shared matcher; record fills."""
        fill = _match_bar(order, bars)
        if fill is not None:
            self._pending.pop(order.order_ref, None)
            self._fills.append(fill)
        return fill

    # ── Legacy fill surface (parity path) ────────────────────────────────
    def execute_signal(
        self, signal: TradeSignal, tick: Candle, params: ExecutionParams
    ) -> FillEvent:
        """Price one signal; delegates to the shared signal-fill function."""
        return execute_signal(signal, tick, params)

    def execute_risk_event(
        self, event: object, tick: Candle, params: ExecutionParams
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
        commission_model: CommissionModel = FixedCommission(0.5),
    ) -> tuple[PortfolioState, tuple[FillRejection, ...], tuple[ScaleRecord, ...]]:
        """Settle a whole bar's fills atomically via the shared cohort scaler."""
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
