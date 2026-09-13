"""Fundamentals schema — typed dataclasses, the sparse row, and the DB write path.

Storage is deliberately **sparse fiscal rows**: one row per
``(ticker, statement, field, period)`` filing, ~5 rows per symbol per year
rather than a 365-day daily grid. The series is assembled from these rows at
load time (see :mod:`src.data.fundamentals.query`), so the fiscal axis is
honored end-to-end and a restatement stays a *separate* row (as-first-stated
PIT falls out of the data model instead of being patched on).

The per-statement dataclasses are the normalized lumibot field surface. Every
field defaults to ``None`` so a statement with only some tags present is still
a valid snapshot (partial XBRL coverage is the norm, not an error).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields
from typing import Literal, Sequence

import pandas as pd
from peewee import CharField, FloatField, IntegerField, Model, SqliteDatabase

from src.data.types import db

# ---------------------------------------------------------------------------
# normalized types
# ---------------------------------------------------------------------------

#: Which statement a snapshot/row belongs to (reconstruction target).
Statement = Literal["income", "balance", "cashflow"]

#: SEC form type. 10-Q is a quarter span, 10-K/20-F/40-F an annual span — the
#: pair disambiguates a cumulative YTD span from a standalone quarter.
Form = Literal["10-K", "10-Q", "20-F", "40-F", "8-K"]


@dataclass(frozen=True)
class Income:
    """Normalized income-statement snapshot (lumibot field names)."""

    revenue: float | None = None
    cost_of_revenue: float | None = None
    gross_profit: float | None = None
    operating_income: float | None = None
    net_income: float | None = None
    eps_basic: float | None = None
    eps_diluted: float | None = None


@dataclass(frozen=True)
class BalanceSheet:
    """Normalized balance-sheet snapshot (lumibot field names)."""

    assets: float | None = None
    current_assets: float | None = None
    cash: float | None = None
    liabilities: float | None = None
    current_liabilities: float | None = None
    debt: float | None = None
    equity: float | None = None
    shares_outstanding: float | None = None


@dataclass(frozen=True)
class CashFlow:
    """Normalized cash-flow snapshot (lumibot field names)."""

    operating_cash_flow: float | None = None
    #: Raw SEC sign as reported (capex is reported negative by US-GAAP filers).
    capex: float | None = None
    investing_cash_flow: float | None = None
    #: Derived ``operating_cash_flow - |capex|``; None unless both are present.
    free_cash_flow: float | None = None
    dividends_paid: float | None = None
    buybacks: float | None = None


#: Tagged union over the three statement snapshots. Dispatch by ``isinstance``
#: (see :func:`statement_of`) rather than a parallel discriminant field.
StatementSnapshot = Income | BalanceSheet | CashFlow


#: Snapshot classes by statement literal. Typed as ``type[StatementSnapshot]``
#: rather than bare ``type`` so ``fields()`` and construction stay checkable.
_SNAPSHOT_BY_STATEMENT: dict[Statement, type[StatementSnapshot]] = {
    "income": Income,
    "balance": BalanceSheet,
    "cashflow": CashFlow,
}


def statement_of(snapshot: StatementSnapshot) -> Statement:
    """Statement literal for a snapshot (tagging via its concrete type)."""
    for name, cls in _SNAPSHOT_BY_STATEMENT.items():
        if type(snapshot) is cls:
            return name
    raise TypeError(f"unknown snapshot type {type(snapshot).__name__}")


def field_names(statement: Statement) -> frozenset[str]:
    """The valid attribute names for ``statement`` (row-field validation set)."""
    return frozenset(f.name for f in fields(_SNAPSHOT_BY_STATEMENT[statement]))


# ---------------------------------------------------------------------------
# the canonical sparse row
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FundamentalRow:
    """One filed value: a single field of a single fiscal period as filed.

    ``filed`` is the PIT anchor (publication date). Several rows can share a
    ``(ticker, statement, field, period_start, period_end)`` key — they are
    successive filings of the same period, and the earliest wins on the series
    curve (as-first-stated); the newest wins on ``.latest()``.
    """

    ticker: str
    statement: Statement
    field: str
    value: float
    period_start: pd.Timestamp  # fiscal span OPEN (XBRL start)
    period_end: pd.Timestamp  # fiscal span CLOSE (XBRL end)
    filed: pd.Timestamp  # PIT anchor (publication date)
    form: Form

    def period_key(self) -> tuple[str, str, str, pd.Timestamp, pd.Timestamp]:
        """Identity of the fiscal period+field this row states."""
        return (
            self.ticker,
            self.statement,
            self.field,
            self.period_start,
            self.period_end,
        )


# ---------------------------------------------------------------------------
# peewee model + write path
# ---------------------------------------------------------------------------

# PIT dates are stored as epoch milliseconds (int), matching the existing
# ``candle.timestamp`` convention so both tables share one time encoding.
_MS_PER_DAY = 86_400_000

#: Natural key of a stated fact, enforced UNIQUE (mirrors
#: ``candle_ticker_timestamp_idx`` on the candle table, which is what makes the
#: candle insert idempotent). ``form`` is deliberately **excluded**: ``filed``
#: already disambiguates two filings of one period, while a period-only key would
#: collide the original with its restatement and destroy the PIT distinction.
NATURAL_KEY_COLUMNS: tuple[str, ...] = (
    "ticker",
    "statement",
    "field",
    "period_start",
    "period_end",
    "filed",
)

#: Name of the UNIQUE index backing :data:`NATURAL_KEY_COLUMNS`.
NATURAL_KEY_INDEX = "fundamental_natural_key_idx"

#: Peewee derives index names from the table+columns (an overlong hashed name),
#: so the index is created explicitly by :func:`bootstrap` under a stable name
#: instead. The DDL lives next to :data:`NATURAL_KEY_COLUMNS` so the two cannot
#: drift apart.
NATURAL_KEY_DDL = (
    f"CREATE UNIQUE INDEX IF NOT EXISTS {NATURAL_KEY_INDEX} "
    f"ON fundamental ({', '.join(NATURAL_KEY_COLUMNS)})"
)


class FundamentalSchema(Model):
    """Sparse fiscal fundamentals row (see :class:`FundamentalRow`)."""

    ticker = CharField()
    statement = CharField()
    field = CharField()
    value = FloatField()
    period_start = IntegerField()
    period_end = IntegerField()
    filed = IntegerField()
    form = CharField()

    class Meta:
        database = db
        table_name = "fundamental"


def _default_conn() -> SqliteDatabase:
    """The DB the model is currently bound to.

    Reading the model's own binding (rather than the module-level ``db``
    constant) keeps one source of truth: a caller that rebinds the model — tests
    pointing at a temp file — gets reads and writes against the same target.
    """
    return FundamentalSchema._meta.database


def bootstrap(db_conn=None) -> None:
    """Create the ``fundamental`` table and its natural-key UNIQUE index.

    Mirrors the candle convention: schema is created explicitly rather than by
    an import side effect, so a read-only run over an existing DB never pays a
    schema write. ``db_conn`` (a :class:`SqliteDatabase`) overrides the module
    default, which is what tests and any non-default DB use.

    The index is created separately from ``create_tables`` because that call
    does not ALTER an existing table: a DB migrated before the index existed
    already has the table and would otherwise never gain the constraint that
    makes :func:`insert_fundamentals` idempotent. ``IF NOT EXISTS`` keeps the
    whole thing idempotent and safe on a table created in the same call.

    A table that predates the constraint may already hold duplicate natural keys
    (the very rows the constraint prevents going forward), which would make
    ``CREATE UNIQUE INDEX`` fail. Those exact-duplicate rows are collapsed first
    — see :func:`_collapse_duplicate_keys`. Restatements are not duplicates (they
    differ in ``filed``) and are never touched.
    """
    conn = db_conn if db_conn is not None else _default_conn()
    with _using(conn):
        conn.create_tables([FundamentalSchema])
        _collapse_duplicate_keys(conn)
        conn.execute_sql(NATURAL_KEY_DDL)


#: Delete every row sharing a natural key except the lowest ``id`` (the first
#: write, which is what :func:`insert_fundamentals` would now have kept anyway).
#: A grouped subquery rather than a window function, so it runs on any SQLite.
_COLLAPSE_DUPES = (
    "DELETE FROM fundamental WHERE id NOT IN ("
    "SELECT MIN(id) FROM fundamental "
    f"GROUP BY {', '.join(NATURAL_KEY_COLUMNS)})"
)


def _collapse_duplicate_keys(conn: SqliteDatabase) -> int:
    """Remove exact-duplicate natural-key rows; return how many were removed.

    Needed only to make the UNIQUE index creatable on a table that predates it.
    Safe when there is nothing to collapse (the DELETE matches no rows) and on a
    table created in the same call (it is empty).
    """
    removed = conn.execute_sql(_COLLAPSE_DUPES).rowcount
    return int(removed) if removed and removed > 0 else 0


@contextmanager
def _using(conn: SqliteDatabase):
    """Bind the model to ``conn`` for the duration of the block.

    The model has one module-level database (production), so a caller pointing
    at another DB must temporarily rebind and always restore — including on
    failure, hence a context manager rather than a try/finally repeated at each
    call site.
    """
    original = FundamentalSchema._meta.database
    FundamentalSchema._meta.database = conn
    try:
        yield conn
    finally:
        FundamentalSchema._meta.database = original


def _to_ms(ts: pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).value // 1_000_000)


def _to_row(year: FundamentalRow) -> dict:
    return {
        "ticker": year.ticker.upper(),
        "statement": year.statement,
        "field": year.field,
        "value": year.value,
        "period_start": _to_ms(year.period_start),
        "period_end": _to_ms(year.period_end),
        "filed": _to_ms(year.filed),
        "form": year.form,
    }


def insert_fundamentals(
    rows: Sequence[FundamentalRow],
    batch_size: int = 500,
    db_conn: SqliteDatabase | None = None,
) -> int:
    """Atomically insert ``rows``, skipping facts already stored.

    **Idempotent**: rows whose natural key
    (:data:`NATURAL_KEY_COLUMNS` — ticker/statement/field/period span/``filed``)
    is already present are ignored (``ON CONFLICT DO NOTHING``), so re-running a
    download for unchanged data writes nothing. This is the same mechanism that
    makes the candle insert idempotent (a UNIQUE index + ``on_conflict_ignore``).

    Restatements are preserved rather than collapsed: two filings of one fiscal
    period differ in ``filed``, so they are two distinct facts and both survive
    (the restatement *is* the data). ``form`` is not part of the key for the same
    reason — ``filed`` already disambiguates.

    Returns the number of rows actually written (duplicates skipped by the
    constraint are not counted), so a caller can tell a real ingest from a no-op
    re-run.

    ``batch_size`` bounds the executemany bind-parameter count (SQLite's
    variable limit), like the candle chunked insert. ``db_conn`` targets a
    non-default DB (tests).
    """
    payload = [_to_row(r) for r in rows]
    if not payload:
        return 0
    conn = db_conn if db_conn is not None else _default_conn()
    written = 0
    with _using(conn):
        for i in range(0, len(payload), batch_size):
            batch = payload[i : i + batch_size]
            with conn.atomic():
                written += _insert_batch(batch)
    return written


def _insert_batch(batch: list[dict]) -> int:
    """Insert one batch, ignoring rows already present; return how many landed.

    peewee's ``on_conflict_ignore().execute()`` returns only the last rowid (its
    own cursor is not exposed), so the inserted count comes from diffing the
    table size around the statement. Kept inside the caller's ``atomic()`` block
    so the measurement and the write share one transaction.
    """
    before = FundamentalSchema.select().count()
    FundamentalSchema.insert_many(batch).on_conflict_ignore().execute()
    return FundamentalSchema.select().count() - before


__all__ = [
    "Statement",
    "Form",
    "Income",
    "BalanceSheet",
    "CashFlow",
    "StatementSnapshot",
    "FundamentalRow",
    "FundamentalSchema",
    "statement_of",
    "field_names",
    "NATURAL_KEY_COLUMNS",
    "NATURAL_KEY_INDEX",
    "NATURAL_KEY_DDL",
    "bootstrap",
    "insert_fundamentals",
]
