"""Idempotency and natural-key tests for the sparse fundamentals store.

The store must be safe to re-ingest: SEC filings are re-fetched routinely (a
new 10-K, a `dl` re-run, a cache miss), and every row already present must be
ignored rather than appended. A restatement must still survive — the natural key
includes ``filed`` precisely so two filings of the same fiscal period are two
distinct facts.

The natural key is ``(ticker, statement, field, period_start, period_end,
filed)`` enforced by a UNIQUE index, mirroring ``candle_ticker_timestamp_idx``
on the candle table (which is what makes the candle insert idempotent).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from peewee import SqliteDatabase

from src.data.fundamentals.schema import (
    Form,
    FundamentalRow,
    FundamentalSchema,
    NATURAL_KEY_INDEX,
    NATURAL_KEY_COLUMNS,
    _using,
    bootstrap,
    insert_fundamentals,
)
from src.utils import parse_timestamp


def ts(value: str) -> pd.Timestamp:
    """``pd.Timestamp`` narrowed past its ``NaTType`` union (repo convention)."""
    return parse_timestamp(value)


@pytest.fixture()
def conn(tmp_path: Path) -> SqliteDatabase:
    """A migrated, isolated fundamentals DB (never the production file)."""
    database = SqliteDatabase(str(tmp_path / "fund.db"))
    bootstrap(database)
    return database


def _row(
    value: float,
    filed: str,
    *,
    field: str = "net_income",
    period_start: str = "2023-01-01",
    period_end: str = "2023-03-31",
    form: Form = "10-Q",
    ticker: str = "DEMO",
) -> FundamentalRow:
    return FundamentalRow(
        ticker=ticker,
        statement="income",
        field=field,
        value=value,
        period_start=ts(period_start),
        period_end=ts(period_end),
        filed=ts(filed),
        form=form,
    )


def _count(conn: SqliteDatabase) -> int:
    with _using(conn):
        return FundamentalSchema.select().count()


def _indexes(conn: SqliteDatabase) -> list[tuple[str, str]]:
    """``(name, sql)`` for every index on the fundamentals table."""
    rows = conn.execute_sql(
        "SELECT name, sql FROM sqlite_master "
        "WHERE type = 'index' AND tbl_name = 'fundamental'"
    ).fetchall()
    return [(str(name), str(sql)) for name, sql in rows]


def test_inserting_the_same_rows_twice_is_idempotent(conn: SqliteDatabase) -> None:
    """A re-ingest of unchanged data must write nothing and add no rows."""
    rows = [_row(10.0, "2023-05-01"), _row(20.0, "2023-08-01", period_end="2023-06-30")]

    assert insert_fundamentals(rows, db_conn=conn) == 2
    assert _count(conn) == 2

    assert insert_fundamentals(rows, db_conn=conn) == 0
    assert _count(conn) == 2

    # And a third pass is still a no-op (the guard is the index, not a one-shot).
    assert insert_fundamentals(rows, db_conn=conn) == 0
    assert _count(conn) == 2


def test_restatement_is_preserved_across_a_double_insert(conn: SqliteDatabase) -> None:
    """Two filings of one period differ by ``filed`` — both are real facts."""
    original = _row(10.0, "2023-11-06")
    restated = _row(99.0, "2024-02-01", form="10-K")

    assert insert_fundamentals([original, restated], db_conn=conn) == 2
    assert insert_fundamentals([original, restated], db_conn=conn) == 0

    assert _count(conn) == 2
    values = sorted(_values(conn))
    assert values == [10.0, 99.0]


def _values(conn: SqliteDatabase) -> list[float]:
    with _using(conn):
        return [float(r.value) for r in FundamentalSchema.select()]


def test_a_new_filing_of_a_known_period_is_still_inserted(conn: SqliteDatabase) -> None:
    """The natural key must not swallow genuinely new facts (a later restatement)."""
    assert insert_fundamentals([_row(10.0, "2023-11-06")], db_conn=conn) == 1
    assert (
        insert_fundamentals([_row(10.0, "2024-02-01", form="10-K")], db_conn=conn) == 1
    )
    assert _count(conn) == 2


def test_duplicate_rows_within_one_batch_collapse(conn: SqliteDatabase) -> None:
    """A payload repeating a fact (multiple units/contexts) must not duplicate."""
    row = _row(10.0, "2023-05-01")
    written = insert_fundamentals([row, replace(row)], db_conn=conn)
    assert written == 1
    assert _count(conn) == 1


def test_identical_key_differing_only_in_value_or_form_collides(
    conn: SqliteDatabase,
) -> None:
    """``form`` is deliberately excluded from the key: ``filed`` disambiguates.

    Same period + same filing date is the same fact; a differing value or form
    is a correction of that one fact, not a second one to append.
    """
    assert insert_fundamentals([_row(10.0, "2023-05-01")], db_conn=conn) == 1
    assert insert_fundamentals([_row(11.0, "2023-05-01")], db_conn=conn) == 0
    assert (
        insert_fundamentals([_row(12.0, "2023-05-01", form="10-K")], db_conn=conn) == 0
    )
    assert _count(conn) == 1
    assert _values(conn) == [10.0]  # first write wins; the key is the identity


def test_bootstrap_creates_the_natural_key_index(conn: SqliteDatabase) -> None:
    """The UNIQUE index must actually land — it is what makes inserts idempotent."""
    indexes = dict(_indexes(conn))
    assert NATURAL_KEY_INDEX in indexes
    assert indexes[NATURAL_KEY_INDEX].upper().startswith("CREATE UNIQUE INDEX")
    for column in NATURAL_KEY_COLUMNS:
        assert column in indexes[NATURAL_KEY_INDEX]


def test_bootstrap_is_idempotent_and_leaves_one_index(tmp_path: Path) -> None:
    """Repeated bootstrap over an existing table must not duplicate the index."""
    database = SqliteDatabase(str(tmp_path / "twice.db"))
    bootstrap(database)
    bootstrap(database)
    bootstrap(database)

    names = [name for name, _ in _indexes(database)]
    assert names.count(NATURAL_KEY_INDEX) == 1


def test_bootstrap_upgrades_a_pre_existing_table_without_the_index(
    tmp_path: Path,
) -> None:
    """A table created before this change has no index; bootstrap must add it.

    ``create_tables`` does not ALTER an existing table, so the index has to be
    created separately for deployed DBs.
    """
    database = SqliteDatabase(str(tmp_path / "legacy.db"))
    database.execute_sql(
        "CREATE TABLE fundamental ("
        "id INTEGER PRIMARY KEY, ticker TEXT, statement TEXT, field TEXT, "
        "value REAL, period_start INTEGER, period_end INTEGER, filed INTEGER, "
        "form TEXT)"
    )
    assert _indexes(database) == []

    bootstrap(database)

    names = [name for name, _ in _indexes(database)]
    assert NATURAL_KEY_INDEX in names


def test_bootstrap_collapses_pre_existing_duplicate_keys(tmp_path: Path) -> None:
    """A table with duplicates predating the constraint must still be upgradable.

    Exact-duplicate natural keys (the rows the constraint now prevents) would
    make ``CREATE UNIQUE INDEX`` fail. They are collapsed to the first write,
    which is what the insert path itself would have kept — and what
    ``as_first_stated`` already resolved on read, so nothing observable changes.
    """
    database = SqliteDatabase(str(tmp_path / "dupes.db"))
    database.execute_sql(
        "CREATE TABLE fundamental ("
        "id INTEGER PRIMARY KEY, ticker TEXT, statement TEXT, field TEXT, "
        "value REAL, period_start INTEGER, period_end INTEGER, filed INTEGER, "
        "form TEXT)"
    )
    insert = (
        "INSERT INTO fundamental "
        "(ticker, statement, field, value, period_start, period_end, filed, form) "
        "VALUES ('DEMO', 'income', 'net_income', %s, 0, 1, 2, '10-Q')"
    )
    for value in (10.0, 10.0, 10.0):  # three ingests of one fact
        database.execute_sql(insert % value)
    # A genuine restatement shares the period but differs in ``filed``.
    database.execute_sql(
        "INSERT INTO fundamental "
        "(ticker, statement, field, value, period_start, period_end, filed, form) "
        "VALUES ('DEMO', 'income', 'net_income', 99.0, 0, 1, 3, '10-K')"
    )
    assert database.execute_sql("SELECT COUNT(*) FROM fundamental").fetchone()[0] == 4

    bootstrap(database)

    rows = database.execute_sql(
        "SELECT value, filed FROM fundamental ORDER BY filed"
    ).fetchall()
    assert rows == [(10.0, 2), (99.0, 3)]  # dupes collapsed, restatement kept
    names = [name for name, _ in _indexes(database)]
    assert NATURAL_KEY_INDEX in names


def test_bootstrap_on_an_empty_mismatched_table_is_a_no_op(tmp_path: Path) -> None:
    """Collapsing nothing must not error (the fresh-create path)."""
    database = SqliteDatabase(str(tmp_path / "empty.db"))
    bootstrap(database)
    bootstrap(database)
    assert _count(database) == 0
    assert [n for n, _ in _indexes(database)].count(NATURAL_KEY_INDEX) == 1


def test_the_unique_index_is_enforced_by_sqlite(conn: SqliteDatabase) -> None:
    """Belt-and-braces: SQLite itself rejects a direct duplicate natural key.

    Guards against the insert path silently succeeding because of a missing
    ``on_conflict_ignore`` — the constraint must be real at the DB level.
    """
    payload = {
        "ticker": "DEMO",
        "statement": "income",
        "field": "net_income",
        "value": 1.0,
        "period_start": 0,
        "period_end": 1,
        "filed": 2,
        "form": "10-Q",
    }
    with _using(conn):
        FundamentalSchema.insert(**payload).execute()
        with pytest.raises(Exception, match="UNIQUE"):
            FundamentalSchema.insert(**payload).execute()
