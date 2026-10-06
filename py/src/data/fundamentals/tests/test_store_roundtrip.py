"""Local-DB roundtrip tests for the sparse fundamentals store.

Exercises the real peewee write/read path against a temporary SQLite file (the
production DB is never touched), so a schema/encoding regression — wrong integer
unit, a lost ``form``, a dropped restatement — fails here.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from peewee import SqliteDatabase

from src.data.fundamentals.query import as_first_stated, load_stated
from src.data.fundamentals.schema import (
    Form,
    FundamentalRow,
    FundamentalSchema,
    Statement,
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
    field: str,
    value: float,
    period_end: str,
    filed: str,
    *,
    statement: Statement = "income",
    form: Form = "10-Q",
) -> FundamentalRow:
    return FundamentalRow(
        ticker="DEMO",
        statement=statement,
        field=field,
        value=value,
        period_start=ts("2023-01-01"),
        period_end=ts(period_end),
        filed=ts(filed),
        form=form,
    )


def test_roundtrip_preserves_every_field(conn: SqliteDatabase) -> None:
    row = FundamentalRow(
        ticker="demo",
        statement="balance",
        field="assets",
        value=1234.5,
        period_start=ts("2023-01-01"),
        period_end=ts("2023-03-31"),
        filed=ts("2023-05-01 16:30"),
        form="10-Q",
    )
    assert insert_fundamentals([row], db_conn=conn) == 1

    (stored,) = load_stated("DEMO", conn)
    # Exact equality covers the epoch-ms encoding, the form, and the timestamp
    # precision — nothing is allowed to drift through the DB.
    assert stored == replace(row, ticker="DEMO")


def test_insert_does_not_leak_the_test_binding_onto_the_model(
    conn: SqliteDatabase,
) -> None:
    """A test DB write must not rebind the model for the whole process."""
    from src.data.types import db

    insert_fundamentals(
        [_row("net_income", 1.0, "2023-03-31", "2023-05-01")], db_conn=conn
    )
    assert FundamentalSchema._meta.database is db


def test_reads_are_scoped_to_the_symbol(conn: SqliteDatabase) -> None:
    insert_fundamentals(
        [_row("net_income", 1.0, "2023-03-31", "2023-05-01")], db_conn=conn
    )
    other = FundamentalRow(
        ticker="OTHER",
        statement="income",
        field="net_income",
        value=2.0,
        period_start=ts("2023-01-01"),
        period_end=ts("2023-03-31"),
        filed=ts("2023-05-01"),
        form="10-Q",
    )
    insert_fundamentals([other], db_conn=conn)
    assert len(load_stated("DEMO", conn)) == 1
    assert load_stated("NOPE", conn) == ()


def test_restatements_survive_the_roundtrip_and_stay_deduped(
    conn: SqliteDatabase,
) -> None:
    """Both filings are stored; the as-first-stated read keeps only the first."""
    original = _row("net_income", 10.0, "2023-03-31", "2023-11-06")
    restated = _row("net_income", 99.0, "2023-03-31", "2024-02-01", form="10-K")
    insert_fundamentals([original, restated], db_conn=conn)

    stored = load_stated("DEMO", conn)
    assert len(stored) == 2  # the restatement is data, not noise
    assert stored[0].filed <= stored[1].filed  # oldest filing first
    assert {r.value for r in stored} == {10.0, 99.0}

    first_stated = as_first_stated(stored)
    assert len(first_stated) == 1
    assert first_stated[0].value == 10.0
    assert first_stated[0].form == "10-Q"


def test_unmigrated_db_reads_as_empty(tmp_path: Path) -> None:
    """A DB without the fundamentals table must not break the backtest preamble."""
    bare = SqliteDatabase(str(tmp_path / "bare.db"))
    assert load_stated("DEMO", bare) == ()


def test_empty_insert_writes_nothing(conn: SqliteDatabase) -> None:
    assert insert_fundamentals([]) == 0
    with _using(conn):
        assert FundamentalSchema.select().count() == 0


def test_insert_batches_large_payloads(conn: SqliteDatabase) -> None:
    """Batching must not lose or duplicate rows across the batch boundary."""
    rows = [
        FundamentalRow(
            ticker="DEMO",
            statement="income",
            field=field,
            value=float(i),
            period_start=ts("2023-01-01"),
            period_end=ts("2023-03-31"),
            filed=ts("2024-02-01"),
            form="10-K",
        )
        for i, field in enumerate(
            ["net_income", "revenue", "gross_profit", "operating_income", "eps_basic"]
        )
    ]
    assert insert_fundamentals(rows, batch_size=2, db_conn=conn) == len(rows)
    with _using(conn):
        assert FundamentalSchema.select().count() == len(rows)


pytestmark = pytest.mark.db
