"""Pure write planner — a cycle's placement results as the rows the book takes.

No I/O, no sqlite: :func:`plan_result_writes` FOLDS a cycle's ``OrderResult``s in
order and returns the intended writes as VALUES (:class:`FillWrite`). The ledger
then applies that tuple in ONE transaction, so the money decisions are testable
without a database and the apply step is a dumb, atomic projection.

Each ``ok`` result with a fill and a ``position_id`` mints the fill rows the sqlite
fold stores: an OPEN emits ``<pid>:open`` (BUY for a long, SELL for a short) and a
CLOSE emits ``<pid>:close`` with the mirror side. The fold keyed by
``position_id`` then reproduces the book ``book_from_executions`` reads, so our
OWN record and the sim account book stay comparable by the divergence oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from src.bt.state import ActionType
from src.exec.types import OrderSide
from src.live.identity import IntentRecord
from src.live.pure import OrderResult
from src.live.types import ExecutionRecord

if TYPE_CHECKING:
    # Type-only: ``src.live.adapters.ibkr`` imports this package's siblings, so a
    # runtime import here would close a cycle.
    from src.live.adapters.ibkr.trades import Execution


@dataclass(frozen=True)
class FillWrite:
    """Insert one derived fill, keyed so a re-plan is a no-op."""

    record: ExecutionRecord


#: The attempt of the FIRST order an intent ever places. A re-send bumps past the
#: stored attempt, so a fresh key (or a pruned row) restarts here.
FIRST_ATTEMPT = 0


def next_attempt(existing: IntentRecord | None) -> int:
    """The attempt the next order for an intent mints: one past the stored record.

    Pure, so "a re-send never reuses an attempt" (and therefore never reuses a
    cOID) is testable without a store. ``None`` is a key with no stored record —
    an unwritten intent, or one whose terminal row was pruned — which starts over
    at :data:`FIRST_ATTEMPT`.
    """
    return FIRST_ATTEMPT if existing is None else existing.attempt + 1


def plan_result_writes(
    scope: str,
    results: tuple[OrderResult, ...],
    now: pd.Timestamp,
) -> tuple[FillWrite, ...]:
    """The fill writes a cycle's *results* imply (pure, in order).

    A failed result, or one with no fill, or one with no ``position_id`` writes
    nothing. An OPEN emits ``<pid>:open`` (BUY for a long, SELL for a short); a
    CLOSE emits ``<pid>:close`` with the mirror side, at *now*. Each fill's cash
    delta follows the ONE signed-cash rule (:func:`execution_cash_delta`).
    """
    writes: list[FillWrite] = []
    for result in results:
        write = _fill_write(scope, result, now)
        if write is not None:
            writes.append(write)
    return tuple(writes)


def _fill_write(scope: str, result: OrderResult, now: pd.Timestamp) -> FillWrite | None:
    """One result as its fill write, or ``None`` when it mints no fill.

    A close carries its own id on the intent; an open on the result (the broker
    minted it). Either way a missing id or fill leaves nothing to book.
    """
    if not result.ok:
        return None
    fill = result.fill
    if fill is None:
        return None
    opens = result.intent.action is not ActionType.close
    pid = result.position_id if opens else result.intent.position_id
    if not pid:
        return None
    if opens:
        side = (
            OrderSide.BUY if result.intent.action is ActionType.long else OrderSide.SELL
        )
    else:
        # A close's side is the mirror of the lot it closes; the lot's side rides
        # on the fill's ``position_side`` (``intent_to_signal`` resolves it).
        position_side = fill.signal.position_side
        side = OrderSide.SELL if position_side is ActionType.long else OrderSide.BUY
    commission = fill.commission or 0.0
    return FillWrite(
        ExecutionRecord(
            scope=scope,
            execution_id=f"{pid}:{'open' if opens else 'close'}",
            conid=None,
            position_id=pid,
            symbol=result.intent.symbol,
            side=side.value,
            qty=fill.filled_qty,
            price=fill.executed_price,
            commission=commission,
            cash_delta=_cash_delta(
                side, fill.filled_qty, fill.executed_price, commission
            ),
            ts=fill.timestamp,
        )
    )


def execution_cash_delta(execution: Execution) -> float:
    """Signed cash flow of one execution: a SELL credits, a BUY debits (net of fee)."""
    return _cash_delta(
        execution.side, execution.qty, execution.price, execution.commission
    )


def _cash_delta(side: OrderSide, qty: float, price: float, commission: float) -> float:
    """The ONE signed-cash rule: a SELL credits, a BUY debits, both net of fee."""
    gross = qty * price
    return (gross - commission) if side is OrderSide.SELL else -(gross + commission)


__all__ = [
    "FIRST_ATTEMPT",
    "FillWrite",
    "execution_cash_delta",
    "next_attempt",
    "plan_result_writes",
]
