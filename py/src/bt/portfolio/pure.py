"""Pure portfolio functions.

All functions are pure: they take state and inputs, return new state.
No mutations, no side effects.

positions dict: Dict[str, Tuple[Position, ...]] — symbol → tuple of Positions.
Multiple positions per symbol are supported (e.g. partial entries, net rebalancing).
"""

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple as TupleT

import math
import pandas as pd

from src.bt.state.types import (
    PortfolioState,
    Position,
    Trade,
    TradeSignal,
    FillEvent,
    EquityPoint,
    ActionType,
    TradeStatus,
    TradeExitReason,
)


@dataclass(frozen=True)
class FillRejection:
    """Pure record of a fill the portfolio REFUSED to apply.

    Currently only raised by ``_open_position`` when the fill would drive cash
    negative: the fill is silently dropped (portfolio returned unchanged). The
    record exists so an I/O layer can report it — a pure portfolio function
    must not write to stderr itself (see ``describe_open_rejection``).
    """

    symbol: str
    timestamp: pd.Timestamp
    qty: float
    cash_used: float
    available_cash: float


def describe_open_rejection(
    portfolio: PortfolioState,
    fill: FillEvent,
) -> Optional[FillRejection]:
    """Report a fill that ``_open_position`` would reject for insufficient cash.

    Pure and side-effect-free — it only *describes* the rejection; the engine
    owns the stderr write. Single source of truth for the cash guard, shared
    with ``_open_position`` so the two can never drift apart.

    Returns ``None`` when the fill would apply (non-open signal, non-positive
    qty, or enough cash).
    """
    signal = fill.signal
    if signal.action not in (ActionType.long, ActionType.short):
        return None
    qty = round(signal.qty, 4)
    if qty <= 0:
        return None
    cash_used = qty * fill.executed_price + fill.commission
    if portfolio.cash - cash_used >= 0:
        return None
    return FillRejection(
        symbol=signal.symbol,
        timestamp=fill.timestamp,
        qty=qty,
        cash_used=cash_used,
        available_cash=portfolio.cash,
    )


def apply_fill(
    portfolio: PortfolioState,
    fill: FillEvent,
) -> PortfolioState:
    """Apply a fill to portfolio, return new state.

    Pure function - input portfolio is not modified.

    signal.qty must be set explicitly (>0) for open/rebalance signals.
    signal.stop_loss / signal.take_profit are used directly for new positions.
    No fallback sizing or SL/TP levels - strategies provide everything.
    """
    signal = fill.signal

    if signal.action == ActionType.close:
        return _close_position(portfolio, fill)

    if signal.action == ActionType.rebalance:
        return _rebalance_position(portfolio, fill)

    return _open_position(portfolio, fill)


def apply_fills(
    portfolio: PortfolioState,
    fills: tuple[FillEvent, ...],
) -> tuple[PortfolioState, TupleT[FillRejection, ...]]:
    """Apply a whole timestamp cohort of fills atomically. Pure.

    One call settles every fill that executes at the SAME phase of the same bar
    (Stage 4's next-open batch, or Stage 6's same-bar batch — never both, see
    ``run_backtest``), so the result cannot depend on the order of ``fills``.

    Ordering rule: NON-OPENS FIRST. Closes and rebalance/reduces are applied
    before any open, so capital a close frees is available to an open in the
    same cohort (the engine drains a bar in ``config.symbols`` order; settling
    closes first removes that order from "does this entry fit?"). Non-opens are
    NEVER scaled — they free or rebalance capital, they do not compete for it.

    Allocation: the opening fills' notional (``qty * executed_price``) is summed
    ONCE from the post-close book and a SINGLE ``scale = min(1, cash /
    requested)`` is applied to every open's qty. Dividing by the cohort's open
    count alone is not enough — once the book is invested, CASH is the binding
    constraint and the tail fill would still be dropped. Scaling every open by
    the same factor, derived from one pre-execution book, is what makes the
    batch order-invariant.

    Legacy guard: a cohort with <= 1 open is applied UNSCALED and untrimmed, so
    a lone order fills or is rejected exactly as before — the vast majority of
    strategies emit one signal per bar and are numerically untouched. SL/TP are
    PRICE levels and are never scaled; only ``qty`` changes. Scaled qtys are
    floored to the 4 dp ``_open_position`` rounds to, so a permutation cannot
    tip a fill over the cash edge by rounding.

    Returns the new portfolio plus the rejections for "genuine exhaustion"
    (an open that still does not fit at the shared scale) — the engine reports
    those; this function never does I/O.

    Caveat — scaling is NOT rejection. An over-cash multi-open cohort is
    divided by one shared ``scale`` (silent); only a lone open rejects. A
    risk-sized entry lands at ``scale × plan`` — $ risk/reward, risk budget
    and cost-adjusted RR off-plan, per-share R unchanged. Grouping, not
    arithmetic, is order-sensitive: results under rejections are advisory,
    never compared across symbol permutations or config edits.
    """
    non_opens = tuple(
        f for f in fills if f.signal.action not in (ActionType.long, ActionType.short)
    )
    opens = tuple(
        f for f in fills if f.signal.action in (ActionType.long, ActionType.short)
    )
    # Canonical order (symbol) so the resulting book is byte-identical for every
    # permutation of the input cohort — ``positions`` is a dict (order-free), but
    # ``trades`` is an append-ordered tuple, so the APPLY order must be fixed.
    for fill in sorted(non_opens, key=lambda f: f.signal.symbol):
        portfolio = apply_fill(portfolio, fill)
    opens = _scale_opens(portfolio, opens)
    rejections: list[FillRejection] = []
    for fill in sorted(opens, key=lambda f: f.signal.symbol):
        rejected = describe_open_rejection(portfolio, fill)
        if rejected is not None:
            rejections.append(rejected)
        portfolio = _open_position(portfolio, fill)
    return portfolio, tuple(rejections)


def _scale_opens(
    portfolio: PortfolioState, opens: tuple[FillEvent, ...]
) -> tuple[FillEvent, ...]:
    """Divide a cohort's opens by one shared, pre-execution cash scale.

    Returns ``opens`` unchanged for a lone open (or none) — the legacy path that
    keeps single-signal strategies bit-identical. Otherwise scales each open's
    qty by ``scale = min(1, cash / requested)`` and floors to 4 dp so the scaled
    result is deterministic and can never round UP past the cash edge.
    """
    if len(opens) <= 1:
        # Lone-open guard: full-size-or-reject, never scaled. Same intent
        # fills full alone, "scale × plan" with a peer (see ``apply_fills``).
        return opens
    # Reserve the fixed commission of every open up front: scaling the notional
    # by ``(cash - total_commission) / requested`` makes the cohort's total cost
    # (notional + commission) land exactly on ``cash``. Without the reservation a
    # batch that fits to the cent still leaves the last fill a commission short
    # and it is rejected — a rounding artefact, not genuine exhaustion.
    commission = sum(f.commission for f in opens)
    requested = sum(max(f.signal.qty, 0.0) * f.executed_price for f in opens)
    budget = portfolio.cash - commission
    if requested <= 0 or budget <= 0:
        return opens
    scale = min(1.0, budget / requested)
    if scale >= 1.0:
        return opens
    return tuple(
        replace(
            f,
            signal=replace(f.signal, qty=math.floor(f.signal.qty * scale * 1e4) / 1e4),
        )
        for f in opens
    )


def next_position_id(symbol: str, ts: pd.Timestamp, seq: int) -> str:
    """Deterministic, run-local unique lot handle for an auto-opened position.

    ``seq`` MUST be a counter carried by the append-only book itself — callers
    pass ``len(portfolio.trades)`` at open time, which grows by exactly one per
    open and is never truncated mid-run. This is what makes the id unique even
    when two opens share a symbol AND a fill timestamp (same-bar multi-open
    cohort): a timestamp-only id collides and the second lot becomes unreachable
    by ``position_id`` in ``_close_position``/``_rebalance_position``.

    Kept here (a pure function in ``src/bt/portfolio``) so the live
    ``SimulatedBroker`` can import the same shape later.
    """
    return f"{symbol}_{ts.timestamp()}_{seq}"


def _open_position(
    portfolio: PortfolioState,
    fill: FillEvent,
) -> PortfolioState:
    """Open a new position from fill - appends to symbol's position tuple.

    signal.qty and signal.sl/tp must be set explicitly by the strategy.
    No fallback sizing or SL/TP levels.
    """
    signal = fill.signal

    qty = round(signal.qty, 4)
    if qty <= 0:
        return portfolio
    # Insufficient cash: drop the fill (portfolio unchanged). The guard is
    # shared with ``describe_open_rejection`` so the engine can report the
    # rejection without this pure function doing I/O.
    if describe_open_rejection(portfolio, fill) is not None:
        return portfolio
    cash_used = qty * fill.executed_price + fill.commission
    new_cash = portfolio.cash - cash_used

    # Generate position_id from signal if provided, else auto-generate a
    # run-unique handle. The uniqueness source is the book's own append-only
    # trade count (one trade appended per open, never truncated during a run).
    pid = signal.position_id or next_position_id(
        signal.symbol, fill.timestamp, len(portfolio.trades)
    )

    # Map the open action to a real position side. `rebalance` is a lifecycle
    # action (net-delta), not a side — a fresh open via rebalance (no existing
    # position) is a `long` here (shannons_demon rebalances longs). This matters
    # because calculate_positions_value/update_prices branch on the side type:
    # a `rebalance`-typed Position would be misvalued as a short.
    position_type = (
        ActionType.short if signal.action == ActionType.short else ActionType.long
    )

    # Create position — SL/TP from signal, None if not set.
    # Mark explicit so risk module doesn't override signal-provided levels.
    position = Position(
        symbol=signal.symbol,
        qty=qty,
        entry_price=fill.executed_price,
        entry_time=fill.timestamp,
        stop_loss=signal.stop_loss,
        take_profit=signal.take_profit,
        last_price=fill.executed_price,
        type=position_type,
        position_id=pid,
        tag=signal.tag,
    )

    # Create trade record
    trade = Trade(
        entry_time=signal.timestamp,
        entry_price=fill.executed_price,
        exit_time=None,
        exit_price=None,
        last_price=fill.executed_price,
        symbol=signal.symbol,
        position=signal.action,
        qty=qty,
        stop_loss=signal.stop_loss or 0.0,
        take_profit=signal.take_profit or 0.0,
        pnl=0.0,
        commission=fill.commission,
        slippage=fill.slippage,
        reason=signal.reason,
        status=TradeStatus.open,
        close_reason=None,
        position_id=pid,
    )

    # Build new state — append position to symbol's tuple
    new_positions = dict(portfolio.positions)
    existing = new_positions.get(signal.symbol, ())
    new_positions[signal.symbol] = existing + (position,)

    return PortfolioState(
        cash=new_cash,
        positions=new_positions,
        trades=portfolio.trades + (trade,),
        equity_curve=portfolio.equity_curve,
        initial_capital=portfolio.initial_capital,
    )


def _close_position(portfolio: PortfolioState, fill: FillEvent) -> PortfolioState:
    """Close a position from fill.

    Requires position_id on the fill signal. Matches the exact position
    by position_id and removes it from the symbol's tuple. If no
    position_id is provided, raises ValueError.
    """
    symbol = fill.signal.symbol
    pid = fill.signal.position_id

    if not pid:
        raise ValueError(
            f"_close_position requires position_id on TradeSignal. "
            f"Signal for {symbol} has no position_id set."
        )

    positions_tuple = portfolio.positions.get(symbol)
    if not positions_tuple:
        return portfolio

    # Find target position by position_id
    target_idx: Optional[int] = None
    target: Optional[Position] = None
    for i, pos in enumerate(positions_tuple):
        if pos.position_id == pid:
            target_idx = i
            target = pos
            break

    if target is None:
        raise ValueError(
            f"Position {pid} not found for symbol {symbol}. "
            f"Available: {[p.position_id for p in positions_tuple]}"
        )

    # Calculate PnL
    is_long = target.type == ActionType.long
    qty = abs(target.qty)

    if is_long:
        pnl = (fill.executed_price - target.entry_price) * qty
        cash_change = qty * fill.executed_price - fill.commission
    else:
        pnl = (target.entry_price - fill.executed_price) * qty
        cash_change = (qty * target.entry_price) + pnl - fill.commission

    # Update trade record — match by position_id
    updated_trades = list(portfolio.trades)
    for i, trade in enumerate(updated_trades):
        if (
            trade.status == TradeStatus.open
            and trade.position_id == pid
            and trade.symbol == symbol
        ):
            updated_trades[i] = _close_trade(trade, fill, pnl)
            break

    # Build new positions — remove only the target position from the tuple
    assert target_idx is not None  # guaranteed by target-is-None check above
    new_positions = dict(portfolio.positions)
    remaining = positions_tuple[:target_idx] + positions_tuple[target_idx + 1 :]
    if remaining:
        new_positions[symbol] = remaining
    else:
        del new_positions[symbol]

    new_cash = portfolio.cash + cash_change

    return PortfolioState(
        cash=new_cash,
        positions=new_positions,
        trades=tuple(updated_trades),
        equity_curve=portfolio.equity_curve,
        initial_capital=portfolio.initial_capital,
    )


def _rebalance_position(
    portfolio: PortfolioState,
    fill: FillEvent,
) -> PortfolioState:
    """Adjust position quantity by delta (positive = add, negative = reduce).

    Requires signal.position_id to target the exact position.
    Signal.qty is the DELTA, not the absolute quantity:
      - +qty: add to position (buy more)
      - -qty: reduce position (sell some or all)

    When the resulting qty <= 0, the position is fully closed.
    When the resulting qty > 0, the position is adjusted in place with a
    time-weighted average cost (TWAC): adds re-average the surviving entry
    price; reduces keep it (selling does not change the cost basis of what
    remains). The single open trade/position is updated — never closed and
    reopened — so each rebalance edits the same position_id instead of
    fabricating a fresh trade preserving an ancient entry price.

    Cash bookkeeping (P0-2):
      - add     (delta>0): cash -= delta*price + commission; no PnL realized.
      - reduce  (delta<0): cash += reduce_qty*price - commission;
                           realizes PnL on the reduced shares only.

    Realized P&L is booked only on an actual full close (``new_qty <= 0``),
    so strategy-level per-trade PnL reflects true round-trips, not
    rebalance revaluation.
    """
    symbol = fill.signal.symbol
    pid = fill.signal.position_id

    positions_tuple = portfolio.positions.get(symbol)

    if not positions_tuple:
        # No existing position — treat as open if delta positive
        if fill.signal.qty > 0:
            return _open_position(portfolio, fill)
        return portfolio

    if not pid:
        raise ValueError(
            f"_rebalance_position requires position_id on TradeSignal. "
            f"Signal for {symbol} has no position_id set "
            f"but positions exist: {[p.position_id for p in positions_tuple]}"
        )

    # Find target position by position_id
    target_idx: Optional[int] = None
    target: Optional[Position] = None
    for i, pos in enumerate(positions_tuple):
        if pos.position_id == pid:
            target_idx = i
            target = pos
            break

    if target is None:
        raise ValueError(
            f"Position {pid} not found for symbol {symbol}. "
            f"Available: {[p.position_id for p in positions_tuple]}"
        )

    delta = round(fill.signal.qty, 4)
    if delta == 0:
        return portfolio

    new_qty = round(target.qty + delta, 4)

    if new_qty <= 0:
        # Full close — build a close fill for the remaining position
        close_fill = FillEvent(
            signal=TradeSignal(
                action=ActionType.close,
                symbol=fill.signal.symbol,
                timestamp=fill.signal.timestamp,
                price=fill.executed_price,
                qty=abs(target.qty),
                reason=fill.signal.reason,
                position_id=pid,
            ),
            filled_qty=abs(target.qty),
            executed_price=fill.executed_price,
            commission=fill.commission,
            slippage=fill.slippage,
            timestamp=fill.timestamp,
        )
        return _close_position(portfolio, close_fill)

    # Partial adjustment: keep the SAME position/trade, update in place with
    # time-weighted average cost (TWAC). No close/reopen, no fabricated PnL.
    is_long = target.type == ActionType.long

    if delta > 0:
        # Adding: cash goes out; no PnL realized.
        cash_change = -(delta * fill.executed_price + fill.commission)
        # TWAC: re-average cost basis over old + new shares.
        new_entry = (
            (target.qty * target.entry_price) + (delta * fill.executed_price)
        ) / new_qty
        new_entry_time = fill.timestamp
    else:
        # Reducing: cash comes in and the removed shares realize PnL; the
        # remaining cost basis is unchanged (selling doesn't re-average basis).
        reduce_qty = abs(delta)
        if is_long:
            realized = (fill.executed_price - target.entry_price) * reduce_qty
        else:
            realized = (target.entry_price - fill.executed_price) * reduce_qty
        cash_change = reduce_qty * fill.executed_price - fill.commission
        new_entry = target.entry_price
        new_entry_time = target.entry_time

    new_cash = portfolio.cash + cash_change

    # Update the single open trade in place: qty and (on add) cost basis. The
    # realized ``realized`` PnL on a partial reduce is recorded against the
    # open trade's pnl so the final close books the full round-trip P&L.
    updated_trades = list(portfolio.trades)
    trade_rebased = False
    for i, trade in enumerate(updated_trades):
        if (
            trade.status == TradeStatus.open
            and trade.position_id == pid
            and trade.symbol == symbol
        ):
            updated_trades[i] = Trade(
                entry_time=trade.entry_time,
                entry_price=new_entry,
                exit_time=trade.exit_time,
                exit_price=trade.exit_price,
                last_price=fill.executed_price,
                reason=trade.reason,
                symbol=trade.symbol,
                position=trade.position,
                qty=new_qty,
                stop_loss=trade.stop_loss,
                take_profit=trade.take_profit,
                pnl=trade.pnl + (realized if delta < 0 else 0.0),
                commission=trade.commission + fill.commission,
                slippage=trade.slippage + fill.slippage,
                status=trade.status,
                close_reason=trade.close_reason,
                position_id=trade.position_id,
            )
            trade_rebased = True
            break

    if not trade_rebased:
        # Defensive: no open trade matched yet (e.g. directly-built portfolio),
        # so create one rather than dropping the position's cost trail.
        updated_trades.append(
            Trade(
                entry_time=new_entry_time,
                entry_price=new_entry,
                exit_time=None,
                exit_price=None,
                last_price=fill.executed_price,
                reason=fill.signal.reason,
                symbol=symbol,
                position=target.type,
                qty=new_qty,
                stop_loss=target.stop_loss or 0.0,
                take_profit=target.take_profit or 0.0,
                pnl=0.0,
                status=TradeStatus.open,
                close_reason=None,
                position_id=pid,
            )
        )

    # Update the surviving position in place (same position_id).
    new_position = Position(
        symbol=symbol,
        qty=new_qty,
        entry_price=new_entry,
        entry_time=new_entry_time,
        stop_loss=target.stop_loss,
        take_profit=target.take_profit,
        last_price=fill.executed_price,
        type=target.type,
        position_id=pid,
        tag=target.tag,
    )

    # Replace the old position with the updated one in the tuple.
    assert target_idx is not None
    new_positions = dict(portfolio.positions)
    new_tup = (
        positions_tuple[:target_idx]
        + (new_position,)
        + positions_tuple[target_idx + 1 :]
    )
    new_positions[symbol] = new_tup

    return PortfolioState(
        cash=new_cash,
        positions=new_positions,
        trades=tuple(updated_trades),
        equity_curve=portfolio.equity_curve,
        initial_capital=portfolio.initial_capital,
    )


def _close_trade(trade: Trade, fill: FillEvent, pnl: float) -> Trade:
    """Return trade closed by fill."""
    return Trade(
        entry_time=trade.entry_time,
        entry_price=trade.entry_price,
        exit_time=fill.timestamp,
        exit_price=fill.executed_price,
        last_price=fill.executed_price,
        reason=trade.reason,
        symbol=trade.symbol,
        position=trade.position,
        qty=trade.qty,
        stop_loss=trade.stop_loss,
        take_profit=trade.take_profit,
        # Accumulate: any PnL realized on earlier partial reduces (already
        # folded into ``trade.pnl``) plus this full-close PnL on the remaining
        # basis forms the true round-trip result.
        pnl=trade.pnl + pnl,
        commission=trade.commission + fill.commission,
        slippage=trade.slippage + fill.slippage,
        status=TradeStatus.closed,
        close_reason=fill.signal.reason or TradeExitReason.none,
        position_id=trade.position_id,
    )


def _equity_point_for(portfolio: PortfolioState, tick) -> tuple[dict, EquityPoint]:
    """Update position last_prices for tick's symbol and build the EquityPoint.

    Returns ``(updated_positions, equity_point)``. Shared by ``update_prices``
    (tuple-backed) and the engine's list-buffered mark-to-market so the per-candle
    equity computation lives in one place.
    """
    new_positions = dict(portfolio.positions)

    symbol_positions = portfolio.positions.get(tick.symbol)
    if symbol_positions:
        new_positions[tick.symbol] = tuple(
            Position(
                symbol=pos.symbol,
                qty=pos.qty,
                entry_price=pos.entry_price,
                entry_time=pos.entry_time,
                stop_loss=pos.stop_loss,
                take_profit=pos.take_profit,
                last_price=tick.close,
                type=pos.type,
                position_id=pos.position_id,
                tag=pos.tag,
            )
            for pos in symbol_positions
        )

    positions_value = calculate_positions_value(new_positions)
    equity = portfolio.cash + positions_value

    equity_point = EquityPoint(
        timestamp=tick.timestamp,
        equity=equity,
        cash=portfolio.cash,
        positions_value=positions_value,
    )
    return new_positions, equity_point


def update_prices(portfolio: PortfolioState, tick) -> PortfolioState:
    """Update position prices and equity curve with new tick.

    Appends an ``EquityPoint`` to the (immutable) tuple -- correct but O(n)
    per candle. The engine host path uses ``mark_to_market_list`` / an engine
    buffer for long runs to avoid rebuilding the tuple every candle.
    """
    new_positions, equity_point = _equity_point_for(portfolio, tick)

    return PortfolioState(
        cash=portfolio.cash,
        positions=new_positions,
        trades=portfolio.trades,
        equity_curve=portfolio.equity_curve + (equity_point,),
        initial_capital=portfolio.initial_capital,
    )


def mark_to_market_list(
    portfolio: PortfolioState, tick, eq_buffer: list
) -> PortfolioState:
    """Same as ``update_prices`` but appends into a caller-owned ``eq_buffer``.

    Returns a copy of the portfolio with updated positions and an EMPTY
    equity_curve (the buffer holds the curve; the engine freezes it at finalize).
    O(1) per candle — the hot-path replacement for ``update_prices`` in long runs.
    """
    new_positions, equity_point = _equity_point_for(portfolio, tick)
    eq_buffer.append(equity_point)
    return PortfolioState(
        cash=portfolio.cash,
        positions=new_positions,
        trades=portfolio.trades,
        equity_curve=(),
        initial_capital=portfolio.initial_capital,
    )


def calculate_positions_value(positions: Dict[str, TupleT[Position, ...]]) -> float:
    """Calculate total value of all positions across all symbols."""
    value = 0.0
    for positions_tuple in positions.values():
        for position in positions_tuple:
            if position.type == ActionType.long:
                value += position.qty * position.last_price
            else:
                upnl = position.qty * (position.entry_price - position.last_price)
                value += (position.qty * position.entry_price) + upnl
    return value


def calculate_equity(portfolio: PortfolioState) -> float:
    """Calculate total equity."""
    return portfolio.cash + calculate_positions_value(portfolio.positions)


# ---------------------------------------------------------------------------
# Multi-position helpers — iterate all positions across all symbols
# ---------------------------------------------------------------------------


def iter_positions(
    portfolio: PortfolioState,
) -> Dict[str, TupleT[Position, ...]]:
    """Return positions dict directly — iterate via .items(), .values(), .keys().

    Each value is Tuple[Position, ...].
    """
    return portfolio.positions


def count_positions(portfolio: PortfolioState) -> int:
    """Total number of individual positions across all symbols."""
    return sum(len(tup) for tup in portfolio.positions.values())


def get_symbol_positions(
    portfolio: PortfolioState, symbol: str
) -> TupleT[Position, ...]:
    """Return all positions for a symbol. Empty tuple if none."""
    return portfolio.positions.get(symbol, ())


def resolve_lot(
    positions: TupleT[Position, ...],
    lot: Optional[str] = None,
    tag: str = "",
) -> Optional[Position]:
    """Resolve a specific lot from a symbol's position tuple.

    Matching precedence:
      1. ``lot`` — exact ``position_id`` match (canonical handle).
      2. ``tag`` — exact ``Position.tag`` match (strategy-facing label).
      3. Neither — the newest (last) position.

    Returns ``None`` when ``positions`` is empty, when ``lot``/``tag`` is
    given but matches nothing, or when a non-empty ``tuple`` has only older
    lots and ``lot``/``tag`` are empty. This is the shared resolver behind the
    DSL's ``ctx.partial_close`` lot targeting.
    """
    if not positions:
        return None
    if lot:
        return next((p for p in positions if p.position_id == lot), None)
    if tag:
        return next((p for p in positions if p.tag == tag), None)
    return positions[-1]


def net_quantity(portfolio: PortfolioState, symbol: str) -> float:
    """Net signed position size for ``symbol`` across all lots.

    Long lots count positive, short lots negative; the DSL's ``Position.qty``
    is always a positive magnitude (the side lives in ``Position.type``).
    Returns 0.0 when the symbol has no open lots.
    """
    total = 0.0
    for p in portfolio.positions.get(symbol, ()):
        if p.type == ActionType.long:
            total += p.qty
        else:
            total -= p.qty
    return total


def avg_entry(portfolio: PortfolioState, symbol: str) -> Optional[float]:
    """Quantity-weighted average entry price across ``symbol``'s lots.

    ``Position.qty`` is stored as a positive magnitude for both sides, so the
    VWAP uses absolute quantities. Returns ``None`` when the symbol is flat.
    """
    lots = portfolio.positions.get(symbol, ())
    if not lots:
        return None
    total_qty = sum(abs(p.qty) for p in lots)
    if total_qty <= 0:
        return None
    return sum(p.qty * p.entry_price for p in lots) / total_qty
