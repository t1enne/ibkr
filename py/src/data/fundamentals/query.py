"""Local-DB reads -> as-first-stated rows and series assembly.

The only I/O in this module is :func:`load_stated` (one local SQLite read per
symbol, once per run). Everything downstream — dedupe, series assembly,
snapshot reconstruction — is pure and operates on rows already in memory, so
the PIT rules are testable without a database.

Two PIT views come out of the same row set:

* **series** (:func:`build_series` / ``SeriesPIT``) — as first stated. The
  earliest filing of a period owns that period's value forever; a restatement
  in a later 10-K adds a row but never moves the curve.
* **current** (:func:`rows_to_snapshot` with ``as_first=False``) — the newest
  stated value per field, i.e. the restated as-of-cursor statement.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, cast

import pandas as pd
from peewee import SqliteDatabase

from src.data.fundamentals.schema import (
    BalanceSheet,
    CashFlow,
    Form,
    FundamentalRow,
    FundamentalSchema,
    Income,
    Statement,
    StatementSnapshot,
    _default_conn,
    _SNAPSHOT_BY_STATEMENT,
    _using,
    field_names,
    statement_of,
)
from src.timestamps import parse_timestamp

if TYPE_CHECKING:
    # Import-cycle guard: the DSL context runtime-imports this module's helpers,
    # so this annotation is TYPE_CHECKING-only.
    from src.bt.strategies.fundamentals_context import SeriesPIT


def load_stated(
    symbol: str, db_conn: SqliteDatabase | None = None
) -> tuple[FundamentalRow, ...]:
    """All stored rows for ``symbol``, oldest filing first.

    A single indexed read of the whole symbol (a few hundred rows even for a
    long history) — series are assembled in memory once per run rather than
    queried per access in the hot loop. Ticker is matched uppercase, as stored
    (the ``candle.ticker`` convention). An unmigrated DB (no ``fundamental``
    table yet) reads as "no fundamentals" rather than raising into the
    backtest preamble.
    """
    conn = db_conn if db_conn is not None else _default_conn()
    with _using(conn):
        if not conn.table_exists(FundamentalSchema):
            return ()
        query = (
            FundamentalSchema.select()
            .where(FundamentalSchema.ticker == symbol.upper())
            .order_by(FundamentalSchema.filed)
        )
        return tuple(_to_row(record) for record in query)


def _from_ms(ms: int) -> pd.Timestamp:
    """Epoch-milliseconds -> Timestamp (naive, the engine's clock convention)."""
    return cast(pd.Timestamp, pd.Timestamp(int(ms), unit="ms"))


def _to_row(record: FundamentalSchema) -> FundamentalRow:
    """peewee record -> canonical row (epoch-ms ints back to Timestamps).

    The casts are peewee's: a model attribute is statically a ``Field``
    descriptor even though ``select()`` hands back the hydrated value.
    """
    return FundamentalRow(
        ticker=cast(str, record.ticker),
        statement=cast(Statement, record.statement),
        field=cast(str, record.field),
        value=float(cast(float, record.value)),
        period_start=_from_ms(cast(int, record.period_start)),
        period_end=_from_ms(cast(int, record.period_end)),
        filed=_from_ms(cast(int, record.filed)),
        form=cast(Form, record.form),
    )


def as_first_stated(rows: Iterable[FundamentalRow]) -> tuple[FundamentalRow, ...]:
    """Keep only the earliest filing per ``(ticker, statement, field, span)``.

    The as-first-stated rule in one sort: a restatement (a later ``filed`` for
    an already-stated period) is dropped, so a prior period's value can never be
    rewritten by a subsequent 10-K. Ties on ``filed`` keep the first row seen in
    sorted order, which makes the result independent of caller ordering.
    """
    ranked = sorted(
        rows,
        key=lambda r: (
            r.ticker,
            r.statement,
            r.field,
            r.period_start,
            r.period_end,
            r.filed,
        ),
    )
    out: list[FundamentalRow] = []
    seen: set[tuple[str, str, str, pd.Timestamp, pd.Timestamp]] = set()
    for row in ranked:
        key = row.period_key()
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return tuple(out)


def build_series(
    rows: Sequence[FundamentalRow],
    statement: Statement,
    field: str,
) -> "SeriesPIT":
    """Cursor-free as-first-stated series for one statement field, periods ascending.

    The dedupe happens here (not in the store) so ``rows`` may hold every filing:
    the earliest filing of each period owns the curve, and a restatement never
    joins it. Surviving rows are period-ascending.

    ``filed`` is deliberately **not** asserted to ascend with ``period_end``.
    That holds on a dense 10-Q grid, but sparse SEC data breaks it: a cumulative
    annual fact for an older period is routinely filed *after* interim facts for
    newer periods (an FY2008 10-K filed 2010-03-18 while the FY2009 Q2 10-Q was
    filed 2009-08-20). ``SeriesPIT`` therefore decides visibility per row rather
    than treating it as one prefix boundary.

    Deferred import of ``SeriesPIT``: the DSL context module imports this one.
    """
    from src.bt.strategies.fundamentals_context import SeriesPIT

    matching = sorted(
        (
            row
            for row in as_first_stated(rows)
            if row.statement == statement and row.field == field
        ),
        key=lambda r: (r.period_end, r.filed),
    )
    return SeriesPIT(
        period=tuple(r.period_end for r in matching),
        window=tuple(matching),
    )


def rows_to_snapshot(
    rows: Iterable[FundamentalRow],
    ticker: str,
    period: pd.Timestamp,
    *,
    statement: Statement | None = None,
    as_first: bool = True,
) -> StatementSnapshot | None:
    """Reconstruct the statement snapshot for ``ticker`` at ``period``.

    ``as_first=True`` (the series view) takes the earliest filing of each field
    — the value the curve shows for that period. ``as_first=False`` takes the
    newest filing — the restated view ``Fundamentals.latest`` serves.

    ``statement`` narrows/validates the reconstruction target; when omitted it
    is inferred from the rows, and ``None`` is returned if they disagree (a
    period's income and balance rows are separate reconstructions, never mixed).
    Returns ``None`` when the period has no stored rows for that statement.
    """
    scoped = [
        r
        for r in rows
        if r.ticker == ticker
        and r.period_end == period
        and (statement is None or r.statement == statement)
    ]
    if not scoped:
        return None
    statement_name = statement or statement_of_rows(scoped)
    if statement_name is None:
        return None

    by_field: dict[str, FundamentalRow] = {}
    for row in scoped:
        if row.statement != statement_name:
            continue
        incumbent = by_field.get(row.field)
        if incumbent is None or _wins(row, incumbent, as_first):
            by_field[row.field] = row

    snapshot_cls = _SNAPSHOT_BY_STATEMENT[statement_name]
    return snapshot_cls(
        **{
            name: (None if (row := by_field.get(name)) is None else row.value)
            for name in field_names(statement_name)
        }
    )


def _wins(candidate: FundamentalRow, incumbent: FundamentalRow, as_first: bool) -> bool:
    """Does ``candidate`` supersede ``incumbent`` under the given PIT view?"""
    if as_first:
        return candidate.filed < incumbent.filed
    return candidate.filed > incumbent.filed


def statement_of_rows(rows: Sequence[FundamentalRow]) -> Statement | None:
    """The one statement shared by ``rows``, or None when they disagree."""
    statements = {row.statement for row in rows}
    if len(statements) != 1:
        return None
    return next(iter(statements))


def snapshot_to_rows(
    snapshot: StatementSnapshot,
    ticker: str,
    period_start: pd.Timestamp | str,
    period_end: pd.Timestamp | str,
    filed: pd.Timestamp | str,
    form: str,
) -> list[FundamentalRow]:
    """Inverse of :func:`rows_to_snapshot`: one row per non-None field.

    ``None`` fields are omitted — an absent field is the absence of a fact, not
    a zero, and writing 0.0 would fabricate one.
    """
    statement = statement_of(snapshot)
    start, end, filed_ts = (
        parse_timestamp(period_start),
        parse_timestamp(period_end),
        parse_timestamp(filed),
    )
    return [
        FundamentalRow(
            ticker=ticker.upper(),
            statement=statement,
            field=name,
            value=float(value),
            period_start=start,
            period_end=end,
            filed=filed_ts,
            form=cast(Form, form),
        )
        for name, value in vars(snapshot).items()
        if value is not None
    ]


#: Snapshot classes by statement, so callers build typed statements without
#: reaching into ``schema``.
SNAPSHOTS: dict[Statement, type] = dict(_SNAPSHOT_BY_STATEMENT)

__all__ = [
    "load_stated",
    "as_first_stated",
    "build_series",
    "rows_to_snapshot",
    "snapshot_to_rows",
    "statement_of_rows",
    "SNAPSHOTS",
    "Income",
    "BalanceSheet",
    "CashFlow",
]
