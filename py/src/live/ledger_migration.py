"""Lossless migrations for the sqlite ledger — preserve, never drop.

Every legacy shape is RENAMED to a kept copy (or altered additively) so no
durable row is ever lost; a pre-conid ``live_position`` cannot be read into the
conid-keyed book, but dropping it would erase durable position state the rolling
trades window cannot rebuild. The orchestration (:func:`migrate`) runs inside the
caller's single ``BEGIN IMMEDIATE`` transaction, so a racing migration process
serializes rather than dropping another's table.

The logger name is pinned to ``src.live.ledger`` (not this module) on purpose:
the migration is part of the ledger subsystem's operator-facing log stream, and
existing operators/alerting key on that name.
"""

from __future__ import annotations

import logging

import peewee

from src.live.ledger_base import (
    _primary_key_columns,
    _table_columns,
    _table_exists,
)
from src.live.ledger_sim import _rekey_sim_lots

logger = logging.getLogger("src.live.ledger")

#: Where a pre-conid ``live_position`` is preserved instead of dropped by migration.
#: A name, not a schema: the rows kept here are the legacy table verbatim.
_LEGACY_POSITION_TABLE = "live_position_legacy"

_LEGACY_INTENT_TABLE = "live_order_intent_legacy"


def _free_legacy_name(db: peewee.SqliteDatabase) -> str:
    """A table name to keep a legacy ``live_position`` under, never overwriting one.

    The canonical copy is ``live_position_legacy``; if that already exists (a
    legacy-shaped ``live_position`` re-created after a prior migration), a fresh
    ``live_position_legacy_<n>`` is chosen so EVERY copy is kept — a rename,
    never a drop, so no row is ever lost.
    """
    if not _table_exists(db, _LEGACY_POSITION_TABLE):
        return _LEGACY_POSITION_TABLE
    n = 1
    while _table_exists(db, f"{_LEGACY_POSITION_TABLE}_{n}"):
        n += 1
    return f"{_LEGACY_POSITION_TABLE}_{n}"


def _preserve_legacy_positions(db: peewee.SqliteDatabase) -> None:
    """Preserve a pre-conid ``live_position`` under a kept copy and warn.

    A pre-4.1 ``live_position`` cannot be read into the conid-keyed book, but
    dropping it would erase durable position state the rolling trades window
    cannot rebuild. RENAME it to a kept copy (a fresh name when one is already
    kept, so a re-created table is preserved too) instead of dropping, naming any
    still-open rows loudly: until an operator reconciles them, the strategy reads
    the symbol as flat and could open again on top of a live position.
    """
    columns = _table_columns(db, "live_position")
    target = _free_legacy_name(db)
    sql = "SELECT symbol, side, qty FROM live_position"
    for symbol, side, qty in db.execute_sql(
        sql + (" WHERE status = 'open'" if "status" in columns else "")
    ).fetchall():
        logger.warning(
            "preserved legacy live_position open row %s %s qty=%s: the "
            "conid-keyed book cannot reproduce it — reconcile it from %s before "
            "trading, or the open may be doubled",
            symbol,
            side,
            qty,
            target,
        )
    db.execute_sql(f"ALTER TABLE live_position RENAME TO {target}")


def _free_intent_legacy_name(db: peewee.SqliteDatabase) -> str:
    """A kept table name for a legacy ``live_order_intent``, never overwriting one."""
    if not _table_exists(db, _LEGACY_INTENT_TABLE):
        return _LEGACY_INTENT_TABLE
    n = 1
    while _table_exists(db, f"{_LEGACY_INTENT_TABLE}_{n}"):
        n += 1
    return f"{_LEGACY_INTENT_TABLE}_{n}"


def _rekey_order_intents(db: peewee.SqliteDatabase) -> str | None:
    """Rename a legacy ``(scope, token)`` intent table to a kept copy, or ``None``.

    Also normalises any NULL ``position_id`` to ``''`` in the legacy copy (so it
    survives reading as an open). No-op when the table is already keyed by the
    identity columns, or absent. RENAME, never drop.
    """
    if not _table_exists(db, "live_order_intent"):
        return None
    if _primary_key_columns(db, "live_order_intent") == {
        "scope",
        "symbol",
        "action",
        "position_id",
    }:
        return None
    legacy = _free_intent_legacy_name(db)
    db.execute_sql("ALTER TABLE live_order_intent RENAME TO " + legacy)
    db.execute_sql(f"UPDATE {legacy} SET position_id = '' WHERE position_id IS NULL")
    return legacy


def _add_intent_columns(db: peewee.SqliteDatabase) -> None:
    """Add the post-4.1 intent columns to an EXISTING identity-keyed table.

    Additive only (``ALTER ... ADD COLUMN``), never a drop, so a live table keeps
    its rows. No-op when the table is absent or still the legacy
    ``(scope, token)`` shape (that one is re-keyed rather than altered).
    """
    if not _table_exists(db, "live_order_intent"):
        return
    columns = _table_columns(db, "live_order_intent")
    if not {"scope", "symbol", "action", "position_id"} <= columns:
        return
    if "tif" not in columns:
        db.execute_sql(
            "ALTER TABLE live_order_intent ADD COLUMN tif TEXT NOT NULL DEFAULT 'DAY'"
        )
    if "stuck_cycles" not in columns:
        db.execute_sql(
            "ALTER TABLE live_order_intent ADD COLUMN stuck_cycles INTEGER NOT NULL "
            "DEFAULT 0"
        )


def _restore_intents(db: peewee.SqliteDatabase, legacy: str) -> None:
    """Copy re-keyed legacy intent rows into the fresh identity-keyed table.

    Runs after ``create_tables`` rebuilt ``live_order_intent``. Every legacy row
    is copied (a rename+copy, never a drop); ``OR IGNORE`` guards the (already
    handled) token-collision case where two identities would otherwise collide on
    the identity PK. A legacy table whose columns do not match the expected shape
    is left intact rather than raising on the first write of every cycle — its
    rows stay preserved under the kept copy.
    """
    required = {
        "scope",
        "token",
        "symbol",
        "action",
        "position_id",
        "state",
        "attempt",
        "order_ref",
        "order_id",
        "decision_ts",
        "updated_at",
    }
    columns = _table_columns(db, legacy)
    if not required <= columns:
        logger.warning(
            "legacy intent table %s has an unexpected shape (%s); rows preserved "
            "in place, not restored",
            legacy,
            sorted(columns),
        )
        return
    db.execute_sql(
        f"INSERT OR IGNORE INTO live_order_intent "
        f"(scope, token, symbol, action, position_id, state, attempt, order_ref, "
        f"order_id, decision_ts, tif, stuck_cycles, updated_at) "
        f"SELECT scope, token, symbol, action, COALESCE(position_id, ''), state, "
        f"attempt, order_ref, order_id, decision_ts, 'DAY', 0, updated_at "
        f"FROM {legacy}"
    )


def migrate(db: peewee.SqliteDatabase) -> str | None:
    """Preserve/alter/re-key every legacy shape; return the legacy intent table.

    Preserve a pre-4.1 position table, ALTER ``live_strategy`` for ``scope``,
    re-key ``live_sim_lot`` from the config hash to the scope, and rename a legacy
    ``(scope, token)`` intent table to the identity columns. Returns the legacy
    table name for the post-create row copy (:func:`restore_intents`), or ``None``.
    All additive or renames — never a drop.
    """
    if _table_exists(db, "live_position") and "conid" not in _table_columns(
        db, "live_position"
    ):
        _preserve_legacy_positions(db)
    if _table_exists(db, "live_strategy") and "scope" not in _table_columns(
        db, "live_strategy"
    ):
        db.execute_sql(
            "ALTER TABLE live_strategy ADD COLUMN scope TEXT NOT NULL DEFAULT ''"
        )
    _rekey_sim_lots(db)
    legacy = _rekey_order_intents(db)
    _add_intent_columns(db)
    return legacy
