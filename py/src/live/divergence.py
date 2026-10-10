"""Pure divergence oracle — our fill-derived book vs the account's lot book.

Two books describe the same scope and must agree:

- **ours** — the fold of the scope's stored fills (``live_execution``). Append-only
  and engine-owned, so it is the truth: nobody hand-edits a fill.
- **the account's** — the exposure surface the broker reports or the operator
  edits (the sim JSON book). A human editing qty, deleting a
  lot or adding one is exactly what moves it.

They disagree when a lot exists on one side only, or when a lot on both sides has
a materially different net quantity. Either way the engine must NOT re-size
silently onto the account's number: it surfaces a :class:`Divergence`, the cycle
is refused, and the operator resolves it. This module is the oracle only — it
reads no ledger, writes nothing, and is called by the cycle (never the reverse).

What the oracle actually sees depends on the adapter. On the sim path the
account side is the operator-editable JSON book, so a hit is a genuine human
disagreement. On the IBKR path the adapter assembles the account side from the
SAME broker trade window this fold is fed, so the two agree by construction and
a hit means our sqlite fold drifted from the broker stream — a
persistence-integrity check, not a broker-side edit. Do not expect the IBKR path
to catch edits it cannot see.

Both functions are pure: same inputs, same output, no I/O, no clock, no store.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, cast

import pandas as pd

from src.bt.state import ActionType, PortfolioState, Position
from src.live.adapters.ibkr.trades import BookRow
from src.live.ledger_base import LedgerReadError
from src.live.types import ExecutionRecord

#: Signed-quantity difference below which two books count as the same lot size.
#: Loose enough to absorb float summation noise over a long fill history; tight
#: enough that any real edit or partial fill is well outside it.
QTY_TOLERANCE = 1e-6

#: Which lot the account book lacks, or that both hold at different sizes.
DivergenceKind = Literal["missing_ours", "missing_account", "qty_mismatch"]

#: Open time for a lot whose fills carry no timestamp (a predates-``ts`` row).
_EPOCH = cast("pd.Timestamp", pd.Timestamp(0, unit="ms", tz="UTC"))

#: Absolute quantity below which a lot counts as flat (float summation noise).
_EPS = 1e-9

#: A lot's book identity: its symbol and its lot id (``None`` when unrecorded).
LotKey = tuple[str, str | None]


@dataclass(frozen=True)
class Divergence:
    """One place our book and the account book disagree on a lot.

    ``kind`` names the direction so a report says which side is short without
    re-deriving it: ``missing_ours`` is a lot the account holds that our fills
    do not account for (an untracked entry), ``missing_account`` is a lot we
    booked that the account does not carry (a manual delete), ``qty_mismatch``
    is the same lot at two different net sizes (a manual edit). The qty fields
    carry ``0.0`` for the side that lacks the lot.
    """

    symbol: str
    position_id: str | None
    ours_qty: float
    account_qty: float
    kind: DivergenceKind


@dataclass(frozen=True)
class _Fill:
    """One fill reduced to what the fold needs: signed size, price, timestamp."""

    signed_qty: float
    price: float
    ts: pd.Timestamp | None


@dataclass(frozen=True)
class _LotFold:
    """A lot's running fold: net size, entry basis, and the earliest fill time.

    ``entry_notional``/``entry_qty`` accumulate ONLY fills that open or extend
    size; a fill that reduces the lot is an exit and must not move the entry
    price it is measured against.
    """

    symbol: str
    signed_qty: float
    entry_notional: float
    entry_qty: float
    opened_at: pd.Timestamp | None

    def fold(self, fill: _Fill) -> _LotFold:
        """Add one fill: net its sign and extend the entry basis when it opens."""
        extending = self.signed_qty == 0 or fill.signed_qty * self.signed_qty > 0
        return _LotFold(
            symbol=self.symbol,
            signed_qty=self.signed_qty + fill.signed_qty,
            entry_notional=self.entry_notional
            + (abs(fill.signed_qty) * fill.price if extending else 0.0),
            entry_qty=self.entry_qty + (abs(fill.signed_qty) if extending else 0.0),
            opened_at=_earlier(self.opened_at, fill.ts),
        )

    @property
    def entry_price(self) -> float:
        """Size-weighted mean fill price, or ``0.0`` when no fill opened size."""
        return self.entry_notional / self.entry_qty if self.entry_qty else 0.0

    def to_position(self, position_id: str | None) -> Position:
        """The open lot this fold describes, at its size-weighted entry."""
        return Position(
            symbol=self.symbol,
            qty=abs(self.signed_qty),
            entry_price=self.entry_price,
            entry_time=self.opened_at or _EPOCH,
            stop_loss=None,
            take_profit=None,
            last_price=self.entry_price,
            type=ActionType.long if self.signed_qty > 0 else ActionType.short,
            position_id=position_id or "",
        )


def book_from_executions(
    executions: tuple[ExecutionRecord, ...],
) -> PortfolioState:
    """Fold stored fills into the book the engine OWNS, keyed by lot.

    Fills group by ``(symbol, position_id)``; each group's signed quantity is
    summed (``side`` drives the sign) and a group whose net is zero is CLOSED —
    no open ``Position`` is emitted for it, so a round trip folds away. A non-zero
    group becomes one open ``Position`` carrying its lot id, side, net size, the
    group's size-weighted entry price over its fills, and the earliest fill as the
    open time.

    Cash, trades and equity are NOT this module's job, so they are left at their
    neutral values rather than invented: only ``positions`` is populated.
    """
    folds: dict[LotKey, _LotFold] = {}
    for execution in executions:
        key = (execution.symbol, execution.position_id or None)
        fill = _Fill(
            signed_qty=_signed_qty(execution), price=execution.price, ts=execution.ts
        )
        folds[key] = folds.get(key, _empty_fold(key[0])).fold(fill)

    grouped: dict[str, list[Position]] = {}
    for (symbol, position_id), fold in folds.items():
        if abs(fold.signed_qty) <= QTY_TOLERANCE:
            continue
        grouped.setdefault(symbol, []).append(fold.to_position(position_id))
    return PortfolioState(
        cash=0.0,
        positions={sym: tuple(items) for sym, items in grouped.items()},
        trades=(),
        equity_curve=(),
        initial_capital=0.0,
    )


def book_rows_from_executions(
    records: tuple[ExecutionRecord, ...],
) -> tuple[BookRow, ...]:
    """Fold stored fills into one ``BookRow`` per lot, mirroring ``trades._apply``.

    The executions book is keyed by conid, so a lot id that is not an IBKR conid
    cannot be read as one: a non-numeric ``position_id`` is a sim lot on a
    conid-keyed book and raises :class:`LedgerReadError` rather than being
    dropped. Rows come back sorted by conid, a round trip leaves one CLOSED row
    (``closed_at`` at the fill that nets the lot to zero), and ``entry_price`` is
    the VWAP of the current open interval's extending fills. Non-numeric lot ids
    are rejected outright, so a sim fill must never reach this fold.
    """
    rows: dict[int, BookRow] = {}
    for record in sorted(records, key=lambda r: _sort_ts(r.ts)):
        conid = _conid_of(record.position_id)
        rows[conid] = _fold(rows.get(conid), record, conid)
    return tuple(rows[conid] for conid in sorted(rows))


def _sort_ts(ts: pd.Timestamp | None) -> pd.Timestamp:
    """A fill's fold instant: a missing timestamp sorts first (earliest)."""
    return _EPOCH if ts is None else ts


def _conid_of(position_id: str) -> int:
    """The conid a lot id names; a non-numeric id is unreadable on this book."""
    if not position_id.isdigit():
        raise LedgerReadError(
            f"live_execution lot id {position_id!r} is not an IBKR conid; the "
            "executions book cannot read it"
        )
    return int(position_id)


def _fold(row: BookRow | None, record: ExecutionRecord, conid: int) -> BookRow:
    """Apply one stored fill to *row* (or open a new one); a closed row stays closed.

    The net is signed (BUY adds, SELL subtracts). An extending fill blends the
    entry VWAP; a reducing fill keeps it; a fill that nets to zero stamps
    ``closed_at``; a flip resets the interval with the residual size. A fill after
    the lot is flat re-opens it, exactly as the broker's own fold would.
    """
    signed = _signed_qty(record)
    if row is None or row.closed_at is not None or row.qty <= _EPS:
        return _open_row(record, conid, signed, abs(signed))
    current = row.qty if row.side == "long" else -row.qty
    if signed * current > 0:
        total = row.qty + abs(signed)
        blended = (row.entry_price * row.qty + record.price * abs(signed)) / total
        return replace(row, qty=total, entry_price=blended)
    if abs(signed) < row.qty - _EPS:
        return replace(row, qty=row.qty - abs(signed))
    if abs(abs(signed) - row.qty) <= _EPS:
        return replace(row, qty=0.0, closed_at=_sort_ts(record.ts))
    return _open_row(record, conid, signed, abs(signed) - row.qty)


def _open_row(
    record: ExecutionRecord, conid: int, signed: float, qty: float
) -> BookRow:
    """A fresh open interval from *record* (``qty`` is the residual for a flip)."""
    return BookRow(
        scope=record.scope,
        conid=conid,
        symbol=record.symbol,
        side="long" if signed > 0 else "short",
        qty=qty,
        entry_price=record.price,
        opened_at=_sort_ts(record.ts),
        closed_at=None,
        stop_loss=None,
        take_profit=None,
        tag="",
        #: The stored fill carries no ``order_ref`` (that is minted per attempt at
        #: the identity layer); the row leaves it blank rather than inventing one.
        order_ref="",
    )


def guard_divergence(
    ours: PortfolioState, account: PortfolioState
) -> tuple[Divergence, ...]:
    """Every lot the two books disagree on, sorted by symbol then lot id.

    Comparison is per ``(symbol, position_id)`` on net SIGNED quantity and on
    existence: a lot in one book only diverges with the absent side reported as
    ``0.0``, and a lot in both diverges when the net sizes differ by more than
    :data:`QTY_TOLERANCE`. A lot id of ``None`` is a legitimate key (a fill whose
    lot id is unrecorded), so both books are read through the same normaliser and
    a ``None`` id compares like any other.

    Two books that agree — including two empty books, or two books holding only
    the same closed lots — return ``()``. The order is deterministic so a report
    is stable across runs.
    """
    ours_lots = _lots(ours)
    account_lots = _lots(account)
    found: list[Divergence] = []
    for key in sorted(set(ours_lots) | set(account_lots), key=_sort_key):
        ours_qty = ours_lots.get(key, 0.0)
        account_qty = account_lots.get(key, 0.0)
        if abs(ours_qty - account_qty) <= QTY_TOLERANCE:
            continue
        symbol, position_id = key
        found.append(
            Divergence(
                symbol=symbol,
                position_id=position_id,
                ours_qty=ours_qty,
                account_qty=account_qty,
                kind=_kind_of(ours_qty, account_qty),
            )
        )
    return tuple(found)


def _lots(book: PortfolioState) -> dict[LotKey, float]:
    """Net signed quantity per ``(symbol, lot id)`` for every lot in *book*."""
    return {
        (symbol, position.position_id or None): _position_net(position)
        for symbol, positions in book.positions.items()
        for position in positions
    }


def _position_net(position: Position) -> float:
    """Signed size of a book lot: negative for a short, positive for a long."""
    return position.qty if position.type is ActionType.long else -position.qty


def _sort_key(key: LotKey) -> tuple[str, str]:
    """Total order over lot keys: symbol first, then lot id (``None`` first)."""
    symbol, position_id = key
    return (symbol, position_id or "")


def _kind_of(ours_qty: float, account_qty: float) -> DivergenceKind:
    """Which side is missing the lot, or that both hold it at different sizes."""
    if abs(ours_qty) <= QTY_TOLERANCE:
        return "missing_ours"
    if abs(account_qty) <= QTY_TOLERANCE:
        return "missing_account"
    return "qty_mismatch"


def _signed_qty(execution: ExecutionRecord) -> float:
    """Signed fill size: a BUY adds, a SELL subtracts; qty is stored absolute."""
    return execution.qty if execution.side.upper() == "BUY" else -execution.qty


def _earlier(
    current: pd.Timestamp | None, candidate: pd.Timestamp | None
) -> pd.Timestamp | None:
    """The earlier of two fill timestamps, tolerating a missing one."""
    if candidate is None:
        return current
    if current is None:
        return candidate
    return candidate if candidate < current else current


#: The zero fold a lot's first fill starts from (empty notional basis, no time).
def _empty_fold(symbol: str) -> _LotFold:
    """The fold a lot starts from: nothing held, nothing paid, never opened."""
    return _LotFold(
        symbol=symbol,
        signed_qty=0.0,
        entry_notional=0.0,
        entry_qty=0.0,
        opened_at=None,
    )
