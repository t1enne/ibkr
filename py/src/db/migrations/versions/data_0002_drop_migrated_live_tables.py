"""``data_0002`` — drop the live tables left orphaned in the candle file (D-Q7).

After the file split the ``live_*`` tables exist in TWO places: the live book in
``data/live.db`` (migrated, read, authoritative) and a stale shell of them in
``data/db.sqlite``, left behind because the split copied rather than moved.

Those shells are worse than clutter. Nothing migrates them — ``ibkr db migrate``
runs ``LIVE_MIGRATIONS`` against the live file and ``DATA_MIGRATIONS`` against this
one — so they keep whatever shape they were frozen in (a pre-``position_id``
``live_position``, for one) while looking exactly like a real book to any tool
pointed at the wrong file. A stale book that reads as FLAT re-opens positions.

So this migration drops them. That is a genuine exception to this repo's
rename-never-drop policy, and it is justified only by the guard below: the policy
protects rows no copy can rebuild, and these rows have a verified copy in the live
file. The guard refuses to drop ROWS unless the live file is present, has been
migrated as a book, and holds AT LEAST as many rows per table as this file does. A
data file whose book was never adopted therefore fails loudly instead of quietly
destroying the only copy.

The guard keys on rows, not on the tables' presence, so a FRESH file — where the
orphan tables do not exist, or exist empty — is dropped through without requiring a
live file at all.

``down`` is ``None``: the rows are not reconstructible from this file once dropped,
and a refused unwind beats a lossy one.
"""

from __future__ import annotations

import peewee

from src.db.migrations.types import MigrationRefused
from src.db.path import resolve_live_db_path

#: The name recorded in ``peewee_migration``.
NAME = "data_0002_drop_migrated_live_tables"

#: Tables whose rows land in a same-named live table, so the counts are directly
#: comparable.
_SHARED_TABLES = (
    "live_strategy",
    "live_execution",
    "live_cash",
    "live_order_intent",
    "live_position_legacy",
    "live_scope_alias",
)

#: Tables ``live_0001_baseline`` FOLDS into ``live_position`` (the pre-``position_id``
#: book becomes ``source='executions'`` rows, the sim lots ``source='account'``).
#: Their rows are not separately recoverable, so they are counted against that one
#: destination instead of a like-named table.
_FOLDED_TABLES = ("live_position", "live_sim_lot")

#: Every table this migration may drop, in one place.
_ORPHAN_TABLES = _SHARED_TABLES + _FOLDED_TABLES

#: The live migration whose bookkeeping row marks a file as a migrated book.
_BOOK_MIGRATION = "live_0001_baseline"


class OrphanDropRefused(MigrationRefused):
    """Dropping would have destroyed rows the live file does not hold."""


def _count(db: peewee.SqliteDatabase, table: str) -> int:
    """Rows in *table*, or ``0`` when it does not exist."""
    exists = db.execute_sql(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    if exists is None:
        return 0
    return int(db.execute_sql(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _rows_at_risk(db: peewee.SqliteDatabase) -> int:
    """Rows in this file's orphaned live tables — what a drop would destroy.

    A table that exists but is EMPTY carries nothing to lose, and an empty table
    is the normal case on a fresh file. The guard therefore keys on ROWS, not on
    the table's presence: gating a fresh file on the live file existing would make
    a first-ever ``ibkr db migrate`` fail for no reason.
    """
    return sum(_count(db, table) for table in _ORPHAN_TABLES)


def _assert_live_file_holds_the_book(db: peewee.SqliteDatabase) -> None:
    """Refuse unless the live file exists, is migrated, and covers every row here.

    The live file is opened READ-ONLY from this migration's point of view, but a
    missing file or an unmigrated one is exactly the case where dropping would lose
    the only copy — so both are refusals, not warnings.
    """
    live_path = resolve_live_db_path()
    if not live_path.exists():
        raise OrphanDropRefused(
            f"refusing to drop the orphaned live tables: no live file at {live_path} "
            "— this file may hold the only copy of the book"
        )
    live = peewee.SqliteDatabase(str(live_path))
    try:
        applied = live.execute_sql(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='peewee_migration'"
        ).fetchone()
        recorded = (
            bool(
                live.execute_sql(
                    "SELECT name FROM peewee_migration WHERE name=?", (_BOOK_MIGRATION,)
                ).fetchone()
            )
            if applied is not None
            else False
        )
        if not recorded:
            raise OrphanDropRefused(
                f"refusing to drop the orphaned live tables: {live_path} has no "
                f"{_BOOK_MIGRATION} bookkeeping row, so it is not a migrated book"
            )
        for table in _SHARED_TABLES:
            here, there = _count(db, table), _count(live, table)
            if here > there:
                raise OrphanDropRefused(
                    f"refusing to drop {table}: this file holds {here} row(s) but "
                    f"{live_path} holds {there}; adopt the book first"
                )
        folded_here = sum(_count(db, table) for table in _FOLDED_TABLES)
        if folded_here > _count(live, "live_position"):
            raise OrphanDropRefused(
                f"refusing to drop the folded live tables: this file holds "
                f"{folded_here} row(s) but {live_path} holds "
                f"{_count(live, 'live_position')} in live_position"
            )
    finally:
        live.close()


def up(db: peewee.SqliteDatabase) -> None:
    """Verify the live file owns the book, then drop this file's orphaned live tables.

    Runs before every ``DROP``, so a refusal leaves the schema untouched. Dropping
    EMPTY orphan tables needs no guard — there is nothing to lose, and on a fresh
    file that is every case — so the live file is only required once rows are at
    stake.
    """
    if _rows_at_risk(db) > 0:
        _assert_live_file_holds_the_book(db)
    for table in _ORPHAN_TABLES:
        db.execute_sql(f"DROP TABLE IF EXISTS {table}")


__all__ = ["NAME", "OrphanDropRefused", "up"]
