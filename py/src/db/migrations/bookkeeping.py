"""Bookkeeping: which migrations have run, recorded in OUR OWN table.

``peewee_migration`` mirrors the spirit of the TS-created ``kysely_migration``
(name + applied_at) but is a separate table on purpose (D-Q2): the kysely table
belongs to the TypeScript data pipeline and may be rewritten by it, whereas this
one is the peewee side's own record. Sharing it would couple two tools' rollout
state to one another.

Shape: ``name TEXT NOT NULL PRIMARY KEY, applied_at TEXT NOT NULL``. ``name`` is
the migration's registry name, so applying a name twice is impossible at the
schema level — the primary key IS the idempotence guard, not a check-then-act in
Python.
"""

from __future__ import annotations

from peewee import SqliteDatabase

MIGRATION_TABLE = "peewee_migration"

_DDL = (
    f"CREATE TABLE IF NOT EXISTS {MIGRATION_TABLE} ("
    "name TEXT NOT NULL PRIMARY KEY, "
    "applied_at TEXT NOT NULL)"
)


def ensure_table(db: SqliteDatabase) -> None:
    """Create the bookkeeping table if absent (idempotent, safe to call always)."""
    db.execute_sql(_DDL)


def applied_names(db: SqliteDatabase) -> frozenset[str]:
    """Every migration name recorded as applied (empty when the table is absent)."""
    if not _exists(db):
        return frozenset()
    rows = db.execute_sql(f"SELECT name FROM {MIGRATION_TABLE}").fetchall()
    return frozenset(str(row[0]) for row in rows)


def record(db: SqliteDatabase, name: str, applied_at: str) -> None:
    """Record *name* as applied. ``INSERT OR IGNORE``: re-recording is a no-op.

    ``applied_at`` is passed in (an ISO-8601 UTC string) rather than read from the
    clock here, so the record is a value the caller controls — deterministic in
    tests and immune to a host whose clock is wrong.
    """
    ensure_table(db)
    db.execute_sql(
        f"INSERT OR IGNORE INTO {MIGRATION_TABLE} (name, applied_at) VALUES (?, ?)",
        (name, applied_at),
    )


def unrecord(db: SqliteDatabase, name: str) -> None:
    """Drop *name*'s row — only ever called after its ``down`` succeeded."""
    if not _exists(db):
        return
    db.execute_sql(f"DELETE FROM {MIGRATION_TABLE} WHERE name = ?", (name,))


def _exists(db: SqliteDatabase) -> bool:
    """Whether the bookkeeping table exists (a read must not create it)."""
    return bool(
        db.execute_sql(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (MIGRATION_TABLE,),
        ).fetchall()
    )
