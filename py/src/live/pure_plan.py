"""Pure write planner — a cycle's placement results as the rows the book takes.

No I/O, no sqlite: :func:`plan_result_writes` FOLDS a cycle's ``OrderResult``s in
order over in-memory lot state and returns the intended writes as VALUES
(:class:`LotOpen` / :class:`LotClose` / :class:`FillWrite`). The ledger then
applies that tuple in ONE transaction, so the money decisions are testable
without a database and the apply step is a dumb, atomic projection.

The fold mirrors what the book would hold if each write were applied and re-read
immediately — the semantics the ledger had when every step was its own
transaction (a batch that opens then closes the same lot, or closes it twice,
must land the same rows either way).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

import pandas as pd

from src.bt.state import ActionType
from src.exec.types import OrderSide
from src.live.identity import IntentRecord
from src.live.pure import OrderResult
from src.live.types import ExecutionRecord, SimLot

if TYPE_CHECKING:
    # Type-only: ``src.live.adapters.ibkr`` imports this package's siblings, so a
    # runtime import here would close a cycle.
    from src.live.adapters.ibkr.trades import Execution


@dataclass(frozen=True)
class LotOpen:
    """Upsert a sim lot row (the account book's open, or a re-open)."""

    lot: SimLot


@dataclass(frozen=True)
class LotClose:
    """Stamp ``position_id``'s lot closed, with the exit leg when one was reported."""

    position_id: str
    closed_at: pd.Timestamp
    exit_price: float | None = None
    commission: float | None = None


@dataclass(frozen=True)
class FillWrite:
    """Insert one derived fill, keyed so a re-plan is a no-op."""

    record: ExecutionRecord


#: Every intended write of one cycle's results, applied in tuple order.
ResultWrite = LotOpen | LotClose | FillWrite

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
    stored_lots: tuple[SimLot, ...],
    now: pd.Timestamp,
) -> tuple[ResultWrite, ...]:
    """The writes a cycle's *results* imply over *stored_lots* (pure, in order).

    A failed result writes nothing. An OPEN with no ``position_id`` can never be
    targeted by a close, so it records nothing; a CLOSE with no id, or one naming
    a lot the scope does not own, changes nothing (the unknown id is a ledger
    no-op, preserved here as "no write" rather than an UPDATE that matches no
    row). Each touched lot's implied fills are then re-derived from the lot's
    POST-write state, so an open followed by a close in ONE batch writes both
    legs — and a close never erases an exit already recorded.
    """
    lots = list(stored_lots)
    writes: list[ResultWrite] = []
    for result in results:
        if not result.ok:
            continue
        pid = _touched_pid(result)
        if pid is None:
            continue
        previous = next((lot for lot in lots if lot.position_id == pid), None)
        updated = _apply_result(result, pid, previous, now)
        if updated is None:
            continue  # a close naming no owned lot: the stored book is unchanged
        lots = [updated if lot.position_id == pid else lot for lot in lots]
        if previous is None:
            lots.append(updated)  # a lot this cycle opened is now part of the fold
        writes.append(_lot_write(result, pid, previous, updated, now))
        writes.extend(FillWrite(record) for record in sim_fill_records(scope, updated))
    return tuple(writes)


def _lot_write(
    result: OrderResult,
    pid: str,
    previous: SimLot | None,
    updated: SimLot,
    now: pd.Timestamp,
) -> LotOpen | LotClose:
    """The lot write *result* implies: an upsert for an open, a close otherwise.

    A close carries only the exit the UPDATE would set — the fields that CHANGED
    — so "already recorded" stays recorded and a close that reported nothing
    writes no exit column at all (the pre-fold no-op UPDATE preserved both).
    """
    if _opens(result):
        return LotOpen(updated)
    return LotClose(
        pid,
        now,
        exit_price=_changed(previous, updated, "exit_price"),
        commission=_changed(previous, updated, "exit_commission"),
    )


def _changed(previous: SimLot | None, updated: SimLot, field: str) -> float | None:
    """*updated*'s *field* when the close moved it, else ``None`` (no write)."""
    before = None if previous is None else getattr(previous, field)
    after = getattr(updated, field)
    return cast("float | None", after) if after != before else None


def _opens(result: OrderResult) -> bool:
    """Whether *result* OPENED (upserted) its lot rather than closing it."""
    return result.intent.action is not ActionType.close


def _touched_pid(result: OrderResult) -> str | None:
    """The lot *result* touches, or ``None`` when it must write nothing.

    A close names its target on the intent; an open names the lot the broker
    minted. Either way a falsy id leaves nothing addressable behind.
    """
    if result.intent.action is ActionType.close:
        return result.intent.position_id or None
    return result.position_id or None


def _apply_result(
    result: OrderResult, pid: str, previous: SimLot | None, now: pd.Timestamp
) -> SimLot | None:
    """The post-write state of the lot *result* touched, or ``None`` if none does."""
    if not _opens(result):
        return None if previous is None else _closed_lot(previous, result, now)
    return _result_lot(result)


def _closed_lot(existing: SimLot, result: OrderResult, now: pd.Timestamp) -> SimLot:
    """*existing* with the close stamped: ``closed_at`` always, the exit only if given.

    A close that does not say what it exited at leaves the recorded exit alone —
    re-marking a closed lot is idempotent, never an erasure.
    """
    fill = result.fill
    closed = replace(existing, closed_at=now)
    if fill is None:
        return closed
    if fill.executed_price is not None:
        closed = replace(closed, exit_price=fill.executed_price)
    if fill.commission is not None:
        closed = replace(closed, exit_commission=fill.commission)
    return closed


def _result_lot(result: OrderResult) -> SimLot:
    """The lot an OPEN fill created, from the fill and the intent behind it.

    A result with no fill still records the lot (ownership is never lost to
    missing detail): size and entry simply stay unknown, and the row reads as an
    ownership-only one rather than inventing a position.
    """
    fill = result.fill
    return SimLot(
        position_id=cast("str", result.position_id),
        symbol=result.intent.symbol,
        side=result.intent.action.value,
        qty=fill.filled_qty if fill is not None else None,
        entry_price=fill.executed_price if fill is not None else None,
        stop_loss=result.intent.stop_loss,
        take_profit=result.intent.take_profit,
        tag=result.intent.tag or None,
        opened_at=fill.timestamp if fill is not None else None,
        entry_commission=fill.commission if fill is not None else None,
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


def sim_fill_records(scope: str, lot: SimLot) -> tuple[ExecutionRecord, ...]:
    """The fills one sim lot implies: its entry, plus its exit when it has one.

    A lot without detail (an ownership-only row) implies nothing — there is no
    size or price to book, and inventing one would put a phantom fill in the
    scope's history. An open long DEBITS cash on the entry and will CREDIT it on
    the exit; a short is the mirror, so the side of each leg follows the lot.
    """
    if not lot.has_detail:
        return ()
    qty = cast("float", lot.qty)
    entry = cast("float", lot.entry_price)
    short = lot.side == "short"
    records = [
        _sim_fill(
            scope,
            f"{lot.position_id}:open",
            lot,
            OrderSide.SELL if short else OrderSide.BUY,
            qty,
            entry,
            lot.entry_commission,
            lot.opened_at,
        )
    ]
    if lot.exit_price is not None:
        records.append(
            _sim_fill(
                scope,
                f"{lot.position_id}:close",
                lot,
                OrderSide.BUY if short else OrderSide.SELL,
                qty,
                lot.exit_price,
                lot.exit_commission,
                lot.closed_at,
            )
        )
    return tuple(records)


def _sim_fill(
    scope: str,
    execution_id: str,
    lot: SimLot,
    side: OrderSide,
    qty: float,
    price: float,
    commission: float | None,
    ts: pd.Timestamp | None,
) -> ExecutionRecord:
    """One sim leg as the same record shape a replayed IBKR fill produces."""
    fee = commission or 0.0
    return ExecutionRecord(
        scope=scope,
        execution_id=execution_id,
        conid=None,
        position_id=lot.position_id,
        symbol=lot.symbol or "",
        side=side.value,
        qty=qty,
        price=price,
        commission=fee,
        cash_delta=_cash_delta(side, qty, price, fee),
        ts=ts,
    )


__all__ = [
    "FIRST_ATTEMPT",
    "FillWrite",
    "LotClose",
    "LotOpen",
    "ResultWrite",
    "next_attempt",
    "plan_result_writes",
]
