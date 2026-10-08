"""Behaviour tests for the ledger schema re-key migration (plan §4.1 + §4.4).

Behaviour only: rows PRESERVED (counts), the legacy copies EXIST, the bare legacy
scope is re-keyed and aliased, and a read still writes nothing. No shape
assertions (no column lists) — those are characterisation and rot.

The pre-migration fixture is extracted from the operator's real ``data/db.sqlite``
so the test exercises LIVE data (12 sim lots, 5 conid positions, a bare scope),
never a hand-built approximation. It is read-only: the source file is constantly
rewritten by an external process, so nothing here asserts against it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from typing import cast

import pandas as pd

from src.live.ledger import LedgerReadError, SqliteLedger

#: Reads the operator's mutable live DB (``../data/db.sqlite``) — a ``db`` test,
#: excluded from ``make check`` (test-fast) like the other sqlite-backed suites.
pytestmark = pytest.mark.db

#: A cutoff before every stored epoch (so a pruning write deletes nothing).
_EPOCH = cast("pd.Timestamp", pd.Timestamp("1970-01-02T00:00:00Z"))

#: The operator's DB, relative to this file: ``py/src/live/tests/`` -> ``data/``.
_LIVE_DB = Path(__file__).resolve().parents[3].parent / "data" / "db.sqlite"

#: The live tables the re-key touches. ``live_position`` and ``live_execution``
#: are counted separately because the sim fold GROWS the former.
_LIVE_TABLES = (
    "live_strategy",
    "live_position",
    "live_execution",
    "live_cash",
    "live_order_intent",
    "live_sim_lot",
)

#: The bare scope the operator's book was written under (pre-charged grammar).
_LEGACY_SCOPE = "vwatr_hv_gated"
_REKEYED_SCOPE = "ibkr_vwatr_hv_gated_legacy"


def _copy_live_tables(dest: Path) -> dict[str, int]:
    """Copy every ``live_*`` table from the operator's DB into *dest*.

    Only the live tables are copied: the 400 MB candle table is irrelevant here and
    copying it would dominate the run. The rows are copied VERBATIM (``SELECT *``)
    so the pre-migration shape is exactly what the migration sees in production.
    Returns the row count per table as they were BEFORE the migration.
    """
    if not _LIVE_DB.exists():  # pragma: no cover - the repo ships the db
        pytest.skip(f"no live database at {_LIVE_DB}")
    source = sqlite3.connect(f"file:{_LIVE_DB}?mode=ro", uri=True)
    dest_con = sqlite3.connect(dest)
    before: dict[str, int] = {}
    try:
        for table in _LIVE_TABLES:
            schema_row = source.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            if schema_row is None:  # pragma: no cover - the db ships the tables
                continue
            dest_con.execute(schema_row[0])
            rows = source.execute(f"SELECT * FROM {table}").fetchall()
            before[table] = len(rows)
            if rows:
                placeholders = ",".join("?" * len(rows[0]))
                dest_con.executemany(
                    f"INSERT INTO {table} VALUES ({placeholders})", rows
                )
        dest_con.commit()
    finally:
        source.close()
        dest_con.close()
    return before


def _counts(path: Path) -> dict[str, int]:
    """Row count per table that exists, so a NEW table shows up as a key."""
    with sqlite3.connect(path) as con:
        names = [
            r[0]
            for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        ]
        return {
            n: con.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0] for n in names
        }


def _table_names(path: Path) -> set[str]:
    with sqlite3.connect(path) as con:
        return {
            r[0]
            for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }


def _migrate(path: Path) -> SqliteLedger:
    """Trigger the migration the way production does: the first WRITE.

    ``prune`` with a cutoff before every stored row is the cheapest write that
    still takes the lazy-DDL path, and it deletes nothing — so any count change
    is the migration's, never the trigger's.
    """
    ledger = SqliteLedger(path)
    ledger.prune(_EPOCH)
    return ledger


@pytest.fixture
def pre_migration(tmp_path: Path) -> tuple[Path, dict[str, int]]:
    """A scratch copy of the operator's live tables plus their pre-migration counts."""
    path = tmp_path / "copy.sqlite"
    return path, _copy_live_tables(path)


def test_migration_preserves_every_row_and_keeps_legacy_copies(
    pre_migration: tuple[Path, dict[str, int]],
) -> None:
    """No row is lost: every pre-migration row is in the new table or its kept copy.

    This is the whole contract of the re-key (plan §4.4): RENAME, never drop. The
    legacy copies are asserted by COUNT, not by shape, so re-running the migration
    or reshaping the legacy table cannot pass this test by accident.
    """
    path, before = pre_migration
    _migrate(path)
    after = _counts(path)

    # The un-keyed tables keep their counts exactly.
    for table in ("live_strategy", "live_execution", "live_cash", "live_order_intent"):
        assert after[table] == before[table], table

    # ``live_position`` GROWS by the folded sim lots, which are copied not moved.
    assert after["live_position"] == before["live_position"] + before["live_sim_lot"]

    # ...and the old books are still there, verbatim, under kept copies.
    assert after["live_position_legacy"] == before["live_position"]
    assert after["live_sim_lot_legacy"] == before["live_sim_lot"]
    assert before["live_position"] > 0 and before["live_sim_lot"] > 0, (
        "the fixture must exercise real rows, not an empty database"
    )

    # Every IBKR lot kept its identity across the re-key: ``position_id`` is the
    # conid, so a reconcile re-read cannot re-open a position it already holds.
    with sqlite3.connect(path) as con:
        ids = {
            r[0]
            for r in con.execute(
                "SELECT position_id FROM live_position WHERE scope=?", (_REKEYED_SCOPE,)
            )
        }
        legacy_ids = {
            r[0]
            for r in con.execute("SELECT CAST(conid AS TEXT) FROM live_position_legacy")
        }
    assert legacy_ids <= ids


def test_migration_rekeys_the_bare_legacy_scope_and_aliases_it(
    pre_migration: tuple[Path, dict[str, int]],
) -> None:
    """The pre-grammar bare scope is re-keyed, aliased, and still readable by name."""
    path, _ = pre_migration
    ledger = _migrate(path)

    # No live table still carries the bare scope...
    with sqlite3.connect(path) as con:
        for table in ("live_strategy", "live_position", "live_execution", "live_cash"):
            assert (
                con.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE scope=?", (_LEGACY_SCOPE,)
                ).fetchone()[0]
                == 0
            ), table
        alias = con.execute(
            "SELECT new_scope FROM live_scope_alias WHERE legacy_scope=?",
            (_LEGACY_SCOPE,),
        ).fetchone()
    assert alias == (_REKEYED_SCOPE,)

    # ...but the operator's config, which still names the bare scope, reads the
    # SAME book through the alias — otherwise the strategy would see itself flat
    # and re-open on top of live positions.
    assert ledger.sim_open_ids(_LEGACY_SCOPE) == ledger.sim_open_ids(_REKEYED_SCOPE)
    assert ledger.scopes_of_store() == (_REKEYED_SCOPE,)


def test_migration_is_idempotent(pre_migration: tuple[Path, dict[str, int]]) -> None:
    """A second boot re-reads the migrated schema: no re-key, no re-copy, no loss."""
    path, _ = pre_migration
    _migrate(path)
    once = _counts(path)
    # A second ledger against the already-migrated file takes the write path again.
    SqliteLedger(path).prune(_EPOCH)
    assert _counts(path) == once


def test_a_read_on_a_fresh_db_writes_nothing(tmp_path: Path) -> None:
    """--dry-run safety: reads never create the schema, tables or rows."""
    db = tmp_path / "fresh.sqlite"
    ledger = SqliteLedger(db)
    assert ledger.load_book("scope").rows == ()
    assert ledger.cash_of("scope", 1000.0) == 1000.0
    assert ledger.initial_capital_of("scope") == 0.0
    assert ledger.sim_open_ids("scope") == frozenset()
    assert ledger.scopes_of_store() == ()
    assert ledger.strategies_of("scope") == ()
    assert ledger.intents_of("scope") == ()
    assert ledger.load_open("scope") == ()
    assert ledger.net_exposure(1) == 0.0
    assert _table_names(db) == set()


def test_a_read_on_a_migrated_db_writes_nothing(
    pre_migration: tuple[Path, dict[str, int]],
) -> None:
    """A read against a MIGRATED db changes no row count (dry run over live data)."""
    path, _ = pre_migration
    _migrate(path)
    before = _counts(path)
    ledger = SqliteLedger(path)
    ledger.load_book(_LEGACY_SCOPE)
    ledger.sim_open_lots(_LEGACY_SCOPE)
    ledger.executions_of(_LEGACY_SCOPE)
    ledger.cash_of(_LEGACY_SCOPE)
    assert _counts(path) == before


def test_a_read_on_a_pre_migration_db_fails_loudly_and_writes_nothing(
    pre_migration: tuple[Path, dict[str, int]],
) -> None:
    """An UN-migrated conid book reads as a LOUD failure, never as a flat book.

    The re-key happens on the first WRITE, so a read (and therefore ``--dry-run``)
    against a pre-migration DB is a real, transitional case. Reporting it as an
    empty book would read downstream as "flat" and re-open every position, so it
    must raise — and still write nothing.
    """
    path, _ = pre_migration
    before = _counts(path)
    with pytest.raises(LedgerReadError):
        SqliteLedger(path).load_book(_LEGACY_SCOPE)
    assert _counts(path) == before


def test_the_folded_sim_lots_read_back_as_the_account_role(
    pre_migration: tuple[Path, dict[str, int]],
) -> None:
    """The folded sim lots are the ``account`` role, and the conid book the other."""
    path, before = pre_migration
    ledger = _migrate(path)
    lots = ledger.sim_open_lots(_LEGACY_SCOPE)
    with sqlite3.connect(path) as con:
        roles = dict(
            con.execute(
                "SELECT position_id, source FROM live_position WHERE scope=?",
                (_REKEYED_SCOPE,),
            ).fetchall()
        )
        folded = con.execute(
            "SELECT COUNT(*) FROM live_position WHERE source='account' AND scope=?",
            (_REKEYED_SCOPE,),
        ).fetchone()[0]
    assert folded == before["live_sim_lot"]
    assert all(roles[lot.position_id] == "account" for lot in lots)
    # The IBKR reconcile book reads the OTHER role, so the two never double-count:
    # a sim lot id is never a conid, so no sim lot can appear as a book row.
    assert all(
        roles.get(str(row.conid)) != "account"
        for row in ledger.load_book(_REKEYED_SCOPE).rows
    )
