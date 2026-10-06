"""Per-scope book reconciliation from the broker's own execution stream.

The strategy book is one row per ``(scope, conid)`` (plan rev 4.1 §3). It is
advanced by applying the ``/iserver/account/trades`` window — one record per
execution, carrying the ``order_ref`` we set as the client order id (``cOID``),
the broker's ``order_id``, ``size``, ``price``, ``commission`` and
``trade_time_r``. ``reconcile`` is the pure core: it takes the scope's durable
rows plus the confirmation window and returns the advanced book, so the caller
(an edge) owns the sqlite read/write.

Rules (plan §3):

- **Ours** = the execution's ``order_ref`` scope segment equals ``scope_tag(scope)``
  exactly (the ref minus its last two ``-``-separated parts). A ref that is not
  ours is ignored by design (a human's trade is not our book), never a mismatch
  and never a reason to place an order. Exact-segment matching keeps scope
  ``momentum`` from absorbing ``momentum-v2``'s fills, and the non-collapsing
  ``scope_tag`` keeps two scopes with one ``slug`` from sharing an owner.
- **One row per conid.** Open / add / reduce / close per side and sign; a
  round trip leaves ONE closed row, not two open lots. ``entry_price`` is the
  VWAP of the opening executions of the *current* open interval.
- **Any execution we cannot apply** raises a warning carrying its raw identity —
  never a silent drop, and never a reason to place an order.

Pure: no I/O, no clock. Input order does not matter (executions are applied in
``ts`` order).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import pandas as pd

from src.exec.refs import scope_tag
from src.exec.types import OrderSide

#: Absolute quantity below which a net position is considered flat (float noise).
_EPS = 1e-9


@dataclass(frozen=True)
class Execution:
    """One broker execution: a single fill of a single order."""

    execution_id: str
    order_id: str
    order_ref: str
    conid: int
    symbol: str
    side: OrderSide
    qty: float
    price: float
    commission: float
    ts: pd.Timestamp
    #: IBKR's per-execution reported account net (position) after this fill.
    #: Diagnostics only — the book is advanced by our own fills, never by it.
    account_position: float | None = None


@dataclass(frozen=True)
class BookRow:
    """One scope-owned position keyed by conid; a closed one keeps ``closed_at``.

    ``qty`` is absolute; the side lives on ``side``. Re-apply protection is the
    ``StrategyBook.applied`` execution-id set, not anything on the row.
    """

    scope: str
    conid: int
    symbol: str
    side: str  # "long" | "short"
    qty: float
    entry_price: float
    opened_at: pd.Timestamp | None
    closed_at: pd.Timestamp | None
    stop_loss: float | None
    take_profit: float | None
    tag: str
    order_ref: str

    @property
    def is_open(self) -> bool:
        return self.closed_at is None and self.qty > _EPS


@dataclass(frozen=True)
class StrategyBook:
    """The durable rows for one scope plus the applied-execution watermark set.

    ``applied`` (execution ids already folded in) is what makes re-applying the
    trades window a no-op; it is stored per scope so two scopes never share a
    watermark.
    """

    rows: tuple[BookRow, ...] = ()
    applied: frozenset[str] = frozenset()


def is_ours(scope: str, execution: Execution) -> bool:
    """Ownership: the ref's scope segment equals ``scope_tag(scope)`` exactly.

    A ref is ``{scope_tag}-{token}-{attempt}`` and neither the token nor the
    attempt contains a ``-``, so the scope segment is the ref minus its last two
    segments. An exact match (not a prefix) keeps scope ``momentum`` from
    claiming refs minted by ``momentum-v2`` / ``momentum-2``, and the
    non-collapsing tag keeps ``momentum_v2`` from claiming ``momentum-v2``'s.
    """
    return execution.order_ref.rsplit("-", 2)[0] == scope_tag(scope)


def _signed(execution: Execution) -> float:
    """Signed quantity: a BUY adds, a SELL subtracts."""
    return execution.qty if execution.side is OrderSide.BUY else -execution.qty


def _side_of(signed: float) -> str:
    return "long" if signed > 0 else "short"


def _apply(
    row: BookRow | None, execution: Execution
) -> tuple[BookRow, tuple[str, ...]]:
    """Fold one execution into a row (or open a new one); return row + warnings."""
    signed = _signed(execution)
    if row is None:
        return _open_row(execution, signed), ()
    warnings: list[str] = []
    if row.symbol and execution.symbol and row.symbol != execution.symbol:
        warnings.append(
            f"conid {execution.conid}: execution symbol {execution.symbol!r} does "
            f"not match stored {row.symbol!r} (execution {execution.execution_id})"
        )
    if row.closed_at is not None or row.qty <= _EPS:
        # Flat -> a reported re-entry: the same conid, a new open interval.
        warnings.append(
            f"conid {execution.conid} {execution.symbol}: reopened after flat "
            f"(execution {execution.execution_id}, {execution.side.value} "
            f"{execution.qty:g} @ {execution.price:g})"
        )
        return _open_row(execution, signed), tuple(warnings)

    cur_signed = row.qty if row.side == "long" else -row.qty
    if math.copysign(1.0, signed) == math.copysign(1.0, cur_signed):
        # Adding to the position: VWAP blends the current open interval.
        new_abs = row.qty + execution.qty
        blended = (
            row.entry_price * row.qty + execution.price * execution.qty
        ) / new_abs
        return (
            replace(
                row,
                qty=new_abs,
                entry_price=blended,
                order_ref=execution.order_ref,
            ),
            tuple(warnings),
        )
    if execution.qty < row.qty - _EPS:  # partial reduce
        return (
            replace(
                row,
                qty=row.qty - execution.qty,
                order_ref=execution.order_ref,
            ),
            tuple(warnings),
        )
    if abs(execution.qty - row.qty) <= _EPS:  # exact close
        return (
            replace(
                row,
                qty=0.0,
                closed_at=execution.ts,
                order_ref=execution.order_ref,
            ),
            tuple(warnings),
        )
    # Flip: the reduction exceeds the position, so the interval resets.
    warnings.append(
        f"conid {execution.conid} {execution.symbol}: flip "
        f"{row.side}->{_side_of(signed)} (execution {execution.execution_id}, "
        f"{execution.side.value} {execution.qty:g} @ {execution.price:g})"
    )
    return _open_row(execution, signed, residual=execution.qty - row.qty), tuple(
        warnings
    )


def _open_row(
    execution: Execution, signed: float, residual: float | None = None
) -> BookRow:
    """A fresh open interval from *execution* (``residual`` for the flip case)."""
    qty = execution.qty if residual is None else residual
    return BookRow(
        scope="",
        conid=execution.conid,
        symbol=execution.symbol,
        side=_side_of(signed),
        qty=qty,
        entry_price=execution.price,
        opened_at=execution.ts,
        closed_at=None,
        stop_loss=None,
        take_profit=None,
        tag="",
        order_ref=execution.order_ref,
    )


def reconcile(
    scope: str,
    executions: tuple[Execution, ...],
    book: StrategyBook,
) -> tuple[StrategyBook, tuple[str, ...]]:
    """Apply unseen executions to *book*; return the advanced book + warnings.

    Executions not owned by *scope* (their ref is not ours) and ids already in
    ``book.applied`` are skipped. The rest are folded in ``ts`` order so the
    advanced book is independent of input order.
    """
    if not scope:
        raise ValueError("scope must be non-empty (ownership + book key)")
    rows: dict[int, BookRow] = {r.conid: r for r in book.rows}
    applied: set[str] = set(book.applied)
    warnings: list[str] = []
    for execution in sorted(executions, key=lambda e: e.ts):
        if execution.execution_id in applied or not is_ours(scope, execution):
            continue
        advanced, row_warnings = _apply(rows.get(execution.conid), execution)
        rows[execution.conid] = replace(advanced, scope=scope)
        applied.add(execution.execution_id)
        warnings.extend(row_warnings)
    advanced_book = StrategyBook(
        rows=tuple(rows[conid] for conid in sorted(rows)),
        applied=frozenset(applied),
    )
    return advanced_book, tuple(warnings)


__all__ = [
    "BookRow",
    "Execution",
    "StrategyBook",
    "is_ours",
    "reconcile",
]
