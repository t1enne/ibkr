"""``live_0002`` — retire the pre-grammar artifacts the re-key left behind.

``live_0001_baseline`` converts an operator's pre-grammar book in place: bare
scopes are re-keyed, and every shape it cannot read is RENAMED to a ``*_legacy``
copy rather than dropped. Those copies (plus the ``live_scope_alias`` mapping)
are scaffolding for ONE conversion, and scaffolding that outlives the conversion
is worse than clutter:

* a ``*_legacy`` copy looks exactly like a real book to any tool pointed at it,
  so a reader that lands on one sees positions the live tables do not agree with;
* ``live_scope_alias`` exists only to let a *bare* scope reach a re-keyed book,
  and the charged ``<adapter>_<name>_<instance>`` grammar makes a bare scope
  unmintable — so no config can produce one to follow.

So this migration drops them. It is the same sanctioned exception to
rename-never-drop as ``data_0002``, and it carries a matching guard. A legacy copy
whose rows have no counterpart in the live table they fold into is the only record
of those rows, so it is KEPT and named loudly rather than dropped — and a bare
scope still present means the re-key never completed, which refuses the whole
migration.

``down`` is ``None``: a dropped copy is not reconstructible, and a refused unwind
beats a lossy one.
"""

from __future__ import annotations

import logging

import peewee

from src.db.migrations.types import MigrationRefused
from src.live.scope import parse_scope

logger = logging.getLogger("src.live.ledger")

#: The name recorded in ``peewee_migration``.
NAME = "live_0002_drop_legacy_artifacts"

#: Retired table names. ``_free_legacy_name`` appends ``_<n>`` when the canonical
#: copy already exists, so each entry matches itself AND its numbered siblings.
_RETIRED = (
    "live_position_legacy",
    "live_sim_lot_legacy",
    "live_order_intent_legacy",
    "live_scope_alias",
)

#: Tables a bare scope may survive in — the re-key's own working set.
_SCOPE_TABLES = (
    "live_strategy",
    "live_position",
    "live_execution",
    "live_cash",
    "live_order_intent",
    "live_sim_lot",
)


class LegacyArtifactRefused(MigrationRefused):
    """Dropping would have destroyed rows the live tables do not hold."""


def _tables(db: peewee.SqliteDatabase) -> tuple[str, ...]:
    """Every retired table present in *db*, canonical and numbered copies."""
    present = {
        str(row[0])
        for row in db.execute_sql("SELECT name FROM sqlite_master WHERE type='table'")
    }
    found = [
        name
        for name in present
        if any(name == base or name.startswith(f"{base}_") for base in _RETIRED)
    ]
    return tuple(sorted(found))


def _columns(db: peewee.SqliteDatabase, table: str) -> set[str]:
    return {str(row[1]) for row in db.execute_sql(f"PRAGMA table_info({table})")}


def _bare_scopes(db: peewee.SqliteDatabase) -> tuple[str, ...]:
    """Every DISTINCT scope that does not parse as the charged grammar."""
    found: set[str] = set()
    for table in _SCOPE_TABLES:
        if not _table_exists(db, table) or "scope" not in _columns(db, table):
            continue
        for (value,) in db.execute_sql(f"SELECT DISTINCT scope FROM {table}"):
            if value is not None and not parse_scope(str(value)):
                found.add(str(value))
    return tuple(sorted(found))


def _table_exists(db: peewee.SqliteDatabase, table: str) -> bool:
    row = db.execute_sql(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def _position_ids(db: peewee.SqliteDatabase, table: str) -> tuple[str, ...]:
    """The identity of every row in a legacy position copy.

    The current key is ``position_id``; the pre-re-key shape keys on ``conid``
    (the identity the re-key preserved verbatim as ``str(conid)``).
    """
    columns = _columns(db, table)
    if "position_id" in columns:
        key = "position_id"
    elif "conid" in columns:
        key = "CAST(conid AS TEXT)"
    else:  # an ownership-only copy: nothing to compare identities against
        return ()
    return tuple(
        str(row[0])
        for row in db.execute_sql(f"SELECT {key} FROM {table} WHERE {key} IS NOT NULL")
    )


def _intent_keys(db: peewee.SqliteDatabase, table: str) -> tuple[tuple[str, ...], ...]:
    """The identity of every row in a legacy intent copy."""
    columns = _columns(db, table)
    required = {"scope", "symbol", "action"}
    if not required <= columns:
        return ()
    position = "COALESCE(position_id, '')" if "position_id" in columns else "''"
    return tuple(
        tuple(str(part) for part in row)
        for row in db.execute_sql(
            f"SELECT scope, symbol, action, {position} FROM {table}"
        )
    )


def _losses(db: peewee.SqliteDatabase) -> dict[str, tuple[str, ...]]:
    """Per legacy table, the rows with no counterpart in the live table they fold into.

    A table that is not one of the two foldable shapes (the alias mapping, or a
    copy of an unknown shape) reads as lossless: the alias holds no book row, and
    a copy whose identities cannot be named is not compared rather than guessed at.
    """
    losses: dict[str, list[str]] = {table: [] for table in _tables(db)}
    if _table_exists(db, "live_position"):
        held = {
            str(row[0])
            for row in db.execute_sql("SELECT position_id FROM live_position")
        }
        for table in _tables(db):
            columns = _columns(db, table)
            if not {"symbol", "side", "qty"} <= columns or "state" in columns:
                continue
            losses[table] += [
                id_ for id_ in _position_ids(db, table) if id_ not in held
            ]
    if _table_exists(db, "live_order_intent"):
        held_intents = {
            tuple(str(part) for part in row)
            for row in db.execute_sql(
                "SELECT scope, symbol, action, COALESCE(position_id, '') "
                "FROM live_order_intent"
            )
        }
        for table in _tables(db):
            if "state" not in _columns(db, table):
                continue
            losses[table] += [
                "/".join(key)
                for key in _intent_keys(db, table)
                if key not in held_intents
            ]
    return {table: tuple(ids) for table, ids in losses.items()}


def up(db: peewee.SqliteDatabase) -> None:
    """Drop the retired artifacts; keep any copy whose rows the live tables lack."""
    retired = _tables(db)
    if not retired:
        return
    bare = _bare_scopes(db)
    if bare:
        raise LegacyArtifactRefused(
            f"bare pre-grammar scope(s) {list(bare)} are still in the live tables: "
            "the re-key did not complete, so the legacy copies are the only "
            "readable record of that book"
        )
    for table, lost in _losses(db).items():
        if not lost:
            db.execute_sql(f"DROP TABLE {table}")
            continue
        logger.warning(
            "legacy artifact %s holds %d row(s) with no counterpart in the live "
            "tables (e.g. %s): KEPT, never dropped — reconcile it by hand",
            table,
            len(lost),
            list(lost[:3]),
        )


__all__ = ["NAME", "up"]
