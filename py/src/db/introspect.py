"""Raw sqlite schema introspection — ``sqlite_master`` and ``PRAGMA`` reads.

Lifted out of ``src.live.ledger_base`` so the migration framework can ask
"does this table exist / what columns does it have" without importing anything
from ``src.live``. A migration must be able to argue about a legacy shape
WITHOUT the current models being importable, otherwise a model rename breaks the
migration that is supposed to handle it.

Pure reads: nothing here writes, so calling one is always safe on a ``--dry-run``.
"""

from __future__ import annotations

from peewee import SqliteDatabase


def table_exists(db: SqliteDatabase, name: str) -> bool:
    """Whether *name* is a table in *db*."""
    return bool(
        db.execute_sql(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchall()
    )


def table_columns(db: SqliteDatabase, name: str) -> set[str]:
    """The column names of *name* in *db* (empty when the table does not exist)."""
    return {str(row[1]) for row in db.execute_sql(f"PRAGMA table_info({name})")}


def primary_key_columns(db: SqliteDatabase, name: str) -> set[str]:
    """The primary-key column names of *name* (from ``PRAGMA table_info``).

    Empty for an absent table or one with no declared PK, so a caller that needs
    to tell those apart must ask :func:`table_exists` first.
    """
    return {
        str(row[1]) for row in db.execute_sql(f"PRAGMA table_info({name})") if row[5]
    }
