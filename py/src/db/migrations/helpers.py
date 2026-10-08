"""Additive DDL helpers for migrations.

Every helper here is a ``CREATE ... IF NOT EXISTS`` / ``ALTER ... ADD COLUMN`` —
additive by construction, so it can be run against an already-hand-migrated DB
without a guard, and re-run without an error. That property is what lets a
baseline absorb a schema it did not itself build.
"""

from __future__ import annotations

from peewee import SqliteDatabase

from src.db.introspect import table_exists


def add_column(
    db: SqliteDatabase,
    table: str,
    column: str,
    ddl_type: str,
    *,
    default: object | None = None,
    not_null: bool = False,
) -> bool:
    """Add *table*.*column* if absent; return whether the ALTER ran.

    No-op when the table does not exist (the table's own migration owns its
    creation) or the column is already there. ``not_null`` without ``default``
    would make SQLite refuse the ALTER on a non-empty table, so a literal default
    is REQUIRED there and enforced here rather than surfacing as an operator's
    first-cycle crash.
    """
    if not table_exists(db, table):
        return False
    columns = {str(row[1]) for row in db.execute_sql(f"PRAGMA table_info({table})")}
    if column in columns:
        return False
    if not_null and default is None:
        raise ValueError(
            f"add_column({table}.{column}): not_null without default — SQLite "
            "rejects that ALTER on a table with rows"
        )
    clause = f"{column} {ddl_type}"
    if default is not None:
        clause += (
            f" NOT NULL DEFAULT {_literal(default)}"
            if not_null
            else f" DEFAULT {_literal(default)}"
        )
    elif not_null:
        clause += " NOT NULL"
    db.execute_sql(f"ALTER TABLE {table} ADD COLUMN {clause}")
    return True


def _literal(value: object) -> str:
    """A SQL literal for *value* (str/int/float/bool only — a DDL type, not data).

    Deliberately narrow and not a general quoting helper: this is called with
    migration-authored constants, never with user input.
    """
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    raise TypeError(f"unsupported DDL default {value!r} ({type(value).__name__})")


__all__ = ["add_column"]
