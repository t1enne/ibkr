"""Lossless migrations for the sqlite ledger — preserve, never drop.

Every legacy shape is RENAMED to a kept copy (or altered additively) so no
durable row is ever lost; a pre-conid ``live_position`` cannot be read into the
``position_id``-keyed book, but dropping it would erase durable position state
the rolling trades window cannot rebuild. The orchestration (:func:`migrate`)
runs inside the caller's single ``BEGIN IMMEDIATE`` transaction, so a racing
migration process serializes rather than dropping another's table.

The re-key steps (plan §4.4) each follow the same shape: RENAME the legacy table
to a free ``*_legacy[_<n>]`` name, let ``create_tables`` build the current shape,
then COPY every row across (:func:`_restore_positions` / :func:`_restore_intents`).
A row count before the migration therefore equals the sum of the new table's and
its kept copy's rows after it.

The logger name is pinned to ``src.live.ledger`` (not this module) on purpose:
the migration is part of the ledger subsystem's operator-facing log stream, and
existing operators/alerting key on that name.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import peewee

from src.live.ledger_base import (
    _primary_key_columns,
    _table_columns,
    _table_exists,
)

logger = logging.getLogger("src.live.ledger")

#: Where a pre-conid ``live_position`` is preserved instead of dropped by migration.
#: A name, not a schema: the rows kept here are the legacy table verbatim.
_LEGACY_POSITION_TABLE = "live_position_legacy"

#: Where a pre-fold ``live_sim_lot`` is preserved instead of dropped.
_LEGACY_SIM_LOT_TABLE = "live_sim_lot_legacy"

_LEGACY_INTENT_TABLE = "live_order_intent_legacy"

#: The current ``live_position`` key. A table without it is re-keyed, not altered.
_POSITION_KEY = "position_id"

#: The ``live_strategy`` columns the charged-scope grammar (plan §4.1) adds.
_STRATEGY_SEGMENTS = (
    ("adapter", "TEXT"),
    ("config_name", "TEXT"),
    ("instance", "TEXT"),
)


@dataclass(frozen=True)
class Legacy:
    """The kept copies a migration renamed away, for the post-create row copy.

    Each field is the ``*_legacy`` table name when a re-key happened, else
    ``None`` (the table was already current, or absent).
    """

    positions: str | None = None
    sim_lots: str | None = None
    intents: str | None = None


def _free_legacy_name(db: peewee.SqliteDatabase, base: str) -> str:
    """A name to keep a legacy table under, never overwriting one.

    The canonical copy is *base* itself; if that already exists (a legacy-shaped
    table re-created after a prior migration), a fresh ``<base>_<n>`` is chosen so
    EVERY copy is kept — a rename, never a drop, so no row is ever lost.
    """
    if not _table_exists(db, base):
        return base
    n = 1
    while _table_exists(db, f"{base}_{n}"):
        n += 1
    return f"{base}_{n}"


def _preserve_legacy_positions(db: peewee.SqliteDatabase) -> None:
    """Preserve a pre-conid ``live_position`` under a kept copy and warn.

    A pre-4.1 ``live_position`` cannot be read into the book at all, but dropping
    it would erase durable position state the rolling trades window cannot
    rebuild. RENAME it to a kept copy (a fresh name when one is already kept, so a
    re-created table is preserved too) instead of dropping, naming any still-open
    rows loudly: until an operator reconciles them, the strategy reads the symbol
    as flat and could open again on top of a live position.
    """
    columns = _table_columns(db, "live_position")
    target = _free_legacy_name(db, _LEGACY_POSITION_TABLE)
    sql = "SELECT symbol, side, qty FROM live_position"
    for symbol, side, qty in db.execute_sql(
        sql + (" WHERE status = 'open'" if "status" in columns else "")
    ).fetchall():
        logger.warning(
            "preserved legacy live_position open row %s %s qty=%s: the "
            "position_id-keyed book cannot reproduce it — reconcile it from %s "
            "before trading, or the open may be doubled",
            symbol,
            side,
            qty,
            target,
        )
    db.execute_sql(f"ALTER TABLE live_position RENAME TO {target}")


def _position_columns(db: peewee.SqliteDatabase) -> set[str] | None:
    """The ``live_position`` columns, or ``None`` when the table is absent.

    The pre-conid guard needs to tell THREE shapes apart, so ``None`` (absent) and
    the empty set (absent-but-truthy confusion) must not conflate. A pre-4.1 table
    has neither ``conid`` nor the ``source`` role column the re-key introduced; the
    CURRENT table has ``source``, and the conid-keyed pre-re-key table has
    ``conid`` — treating either as pre-4.1 would rename the live book away on
    every boot.
    """
    if not _table_exists(db, "live_position"):
        return None
    return _table_columns(db, "live_position")


def _rekey_positions(db: peewee.SqliteDatabase) -> str | None:
    """Rename a conid-keyed ``live_position`` to a kept copy, or ``None``.

    The re-key replaces the ``conid INTEGER`` PK with a TEXT ``position_id`` (the
    row's identity is ``str(conid)``, so nothing is lost). RENAME, never drop;
    :func:`_restore_positions` copies the rows into the fresh table.
    """
    if not _table_exists(db, "live_position"):
        return None
    columns = _table_columns(db, "live_position")
    if _POSITION_KEY in columns:
        return None
    legacy = _free_legacy_name(db, _LEGACY_POSITION_TABLE)
    db.execute_sql(f"ALTER TABLE live_position RENAME TO {legacy}")
    return legacy


def _rekey_sim_lots(db: peewee.SqliteDatabase) -> str | None:
    """Re-key and then rename ``live_sim_lot`` to a kept copy, or ``None``.

    Two legacy shapes are handled before the rename, in this order:

    - a pre-4.1 table keyed by the config hash (``strategy_id``) is re-keyed to
      the stable ``scope`` in place (backfilled through the ``live_strategy``
      audit link; a row with no link keeps an empty scope — preserved, never
      dropped), and
    - a pre-fold table is missing the fill-detail columns, so they are ADDed
      (additive: an ownership row simply reads back ownership-only).

    Then the whole table is RENAMED to a kept copy — the fold into
    ``live_position`` copies its rows in :func:`_restore_positions`.
    """
    if _table_exists(db, "live_sim_lot"):
        columns = _table_columns(db, "live_sim_lot")
        if "scope" not in columns and "strategy_id" in columns:
            db.execute_sql(
                "ALTER TABLE live_sim_lot RENAME COLUMN strategy_id TO scope"
            )
            if _table_exists(db, "live_strategy"):
                db.execute_sql(
                    "UPDATE live_sim_lot SET scope = (SELECT s.scope FROM "
                    "live_strategy s WHERE s.strategy_id = live_sim_lot.scope) "
                    "WHERE scope IN (SELECT strategy_id FROM live_strategy)"
                )
            columns = _table_columns(db, "live_sim_lot")
        for name, sql_type in (
            ("symbol", "TEXT"),
            ("side", "TEXT"),
            ("qty", "REAL"),
            ("entry_price", "REAL"),
            ("stop_loss", "REAL"),
            ("take_profit", "REAL"),
            ("tag", "TEXT"),
            ("opened_at", "INTEGER"),
            ("entry_commission", "REAL"),
            ("exit_price", "REAL"),
            ("exit_commission", "REAL"),
        ):
            if name not in columns:
                db.execute_sql(f"ALTER TABLE live_sim_lot ADD COLUMN {name} {sql_type}")
    if not _table_exists(db, "live_sim_lot"):
        return None
    legacy = _free_legacy_name(db, _LEGACY_SIM_LOT_TABLE)
    db.execute_sql(f"ALTER TABLE live_sim_lot RENAME TO {legacy}")
    return legacy


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
    legacy = _free_legacy_name(db, _LEGACY_INTENT_TABLE)
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


def _add_strategy_columns(db: peewee.SqliteDatabase) -> None:
    """Add ``scope`` and the scope SEGMENTS to an EXISTING ``live_strategy``.

    Additive only. The segments are what let ``live status`` group scopes by
    adapter; a row written before they existed reads them as ``''``, which is the
    truthful answer ("this scope predates the charged grammar").
    """
    if not _table_exists(db, "live_strategy"):
        return
    columns = _table_columns(db, "live_strategy")
    if "scope" not in columns:
        db.execute_sql(
            "ALTER TABLE live_strategy ADD COLUMN scope TEXT NOT NULL DEFAULT ''"
        )
    for name, sql_type in _STRATEGY_SEGMENTS:
        if name not in columns:
            db.execute_sql(
                f"ALTER TABLE live_strategy ADD COLUMN {name} {sql_type} "
                f"NOT NULL DEFAULT ''"
            )


def _legacy_scopes(db: peewee.SqliteDatabase) -> tuple[str, ...]:
    """Every DISTINCT bare scope across the live tables, sorted.

    A ``scope`` is "charged" when it parses as the ``<adapter>_<name>_<instance>``
    triplet the grammar mints; anything else (``momentum``) is the pre-grammar
    bare scope two adapters once shared, and must be re-keyed. The ``live_sim_lot``
    source of the pre-4.1 shape is included too (it is folded, not ignored).
    """
    found: set[str] = set()
    for table in (
        "live_strategy",
        "live_position",
        "live_execution",
        "live_cash",
        "live_order_intent",
        "live_sim_lot",
    ):
        if not _table_exists(db, table):
            continue
        columns = _table_columns(db, table)
        if "scope" not in columns:
            continue
        for (value,) in db.execute_sql(f"SELECT DISTINCT scope FROM {table}"):
            if value is not None and _is_legacy_scope(str(value)):
                found.add(str(value))
    return tuple(sorted(found))


def _is_legacy_scope(scope: str) -> bool:
    """Whether *scope* predates the charged grammar (a bare ``momentum``).

    Delegates to :func:`src.live.scope.parse_scope` so the migration and the
    scope mint can never disagree about what a charged scope looks like.
    """
    from src.live.scope import parse_scope

    return bool(scope) and parse_scope(scope) is None


def _rekey_scopes(db: peewee.SqliteDatabase) -> int:
    """Re-key every bare legacy scope; return the count.

    A legacy bare scope ``X`` becomes ``ibkr_X_legacy`` (the adapter that ever
    placed real orders; the original config hash is unknowable from a bare scope,
    so the literal instance ``legacy`` is used) in EVERY live table.
    ``live_strategy`` additionally gets the parsed segments, so the re-keyed scope
    is immediately groupable by adapter.

    No alias is recorded: the charged grammar makes a bare scope unmintable, so
    nothing but a hand-typed ``--scope X`` can still name one. A config file
    mints its own ``<adapter>_<name>_<hash>`` scope and is unaffected — see
    ``live_0002_drop_legacy_artifacts`` for why the alias table is gone.
    """
    from src.live.scope import ScopeParts, parse_scope, scope_of, slug_segment

    rekeyed = 0
    for legacy in _legacy_scopes(db):
        slug = slug_segment(legacy) or "scope"
        new_scope = scope_of(ScopeParts("ibkr", slug, "legacy"))
        for table in (
            "live_strategy",
            "live_position",
            "live_execution",
            "live_cash",
            "live_order_intent",
            "live_sim_lot",
        ):
            if not _table_exists(db, table) or "scope" not in _table_columns(db, table):
                continue
            db.execute_sql(
                f"UPDATE {table} SET scope = ? WHERE scope = ?", (new_scope, legacy)
            )
        if _table_exists(db, "live_strategy"):
            db.execute_sql(
                "UPDATE live_strategy SET adapter = ?, config_name = ?, instance = ? "
                "WHERE scope = ?",
                ("ibkr", slug, "legacy", new_scope),
            )
        if not parse_scope(new_scope):  # pragma: no cover - scope_of is total
            raise RuntimeError(f"minted an unparseable scope {new_scope!r}")
        logger.warning(
            "re-keyed legacy scope %r to %r (plan §4.4 step 4): the book is "
            "unchanged, but the scope string moved — a run must address %r, never "
            "the bare %r",
            legacy,
            new_scope,
            new_scope,
            legacy,
        )
        rekeyed += 1
    return rekeyed


def _rekey_executions(db: peewee.SqliteDatabase) -> None:
    """Re-key ``live_execution`` from ``conid`` to ``position_id``, ADDITIVELY.

    A pure ``ALTER ... ADD COLUMN`` plus a backfill, never a rebuild: the rows are
    the immutable fill ledger (the only record of a fill there is), and the legacy
    ``conid`` column is left in place as IBKR provenance — no code reads it, and
    dropping it would mean rebuilding the one table that cannot be regenerated.
    A row already carrying a ``position_id`` is left alone, so this is idempotent.
    """
    if not _table_exists(db, "live_execution"):
        return
    columns = _table_columns(db, "live_execution")
    if _POSITION_KEY not in columns:
        db.execute_sql(
            "ALTER TABLE live_execution ADD COLUMN position_id TEXT NOT NULL DEFAULT ''"
        )
        columns = _table_columns(db, "live_execution")
    if "conid" in columns:
        db.execute_sql(
            "UPDATE live_execution SET position_id = CAST(conid AS TEXT) "
            "WHERE position_id = ''"
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


def _restore_positions(db: peewee.SqliteDatabase, legacy: Legacy) -> None:
    """Fold every kept legacy book row into the fresh ``live_position``.

    Runs after ``create_tables`` rebuilt the ONE book table. Two sources are
    copied, both ``OR REPLACE`` so a row re-created under the same identity cannot
    raise:

    - the conid-keyed ``live_position`` (``CAST(conid AS TEXT)`` — the identity is
      unchanged, which is what keeps the IBKR reconcile book from re-opening
      positions it already holds), written as the ``executions`` ROLE because its
      rows ARE the fold of ``live_execution``; and
    - the ``live_sim_lot`` fold, written as the ``account`` ROLE (the
      human/broker-editable surface).

    A kept table whose shape is unexpected is left intact rather than raising on
    the first write of every cycle; its rows stay preserved under the copy.
    """
    if legacy.positions is not None:
        _copy_legacy_rows(
            db,
            legacy.positions,
            _position_select(db, legacy.positions, "CAST(conid AS TEXT)", "executions"),
            required={"scope", "conid", "symbol", "side", "qty", "entry_price"},
        )
    if legacy.sim_lots is not None:
        _copy_legacy_rows(
            db,
            legacy.sim_lots,
            _position_select(db, legacy.sim_lots, "position_id", "account"),
            required={"scope", "position_id"},
        )


def _position_select(
    db: peewee.SqliteDatabase, legacy: str, position_id: str, source: str
) -> str:
    """The INSERT ... SELECT folding one kept table into the current book shape."""
    columns = _table_columns(db, legacy)
    #: Target column -> SELECT expression, in TARGET order, so the two lists line
    #: up positionally. NOT NULL columns are COALESCEd (a legacy sim row may hold
    #: NULL detail — an ownership-only lot); a column the legacy table lacks is
    #: filled with the target's own default rather than named (that would raise).
    picks: tuple[tuple[str, str], ...] = (
        ("symbol", "COALESCE(symbol, '')"),
        ("side", "COALESCE(side, '')"),
        ("qty", "COALESCE(qty, 0.0)"),
        ("entry_price", "COALESCE(entry_price, 0.0)"),
        ("stop_loss", "stop_loss"),
        ("take_profit", "take_profit"),
        ("tag", "COALESCE(tag, '')"),
        ("opened_at", "opened_at"),
        ("closed_at", "closed_at"),
        ("entry_commission", "entry_commission"),
        ("exit_price", "exit_price"),
        ("exit_commission", "exit_commission"),
    )
    named = [name for name, _ in picks if name in columns]
    values = [expression for name, expression in picks if name in columns]
    return (
        f"INSERT OR REPLACE INTO live_position "
        f"(scope, position_id, {', '.join(named)}, source, order_ref) "
        f"SELECT scope, {position_id}, {', '.join(values)}, '{source}', '' "
        f"FROM {legacy}"
    )


def _copy_legacy_rows(
    db: peewee.SqliteDatabase, legacy: str, sql: str, *, required: set[str]
) -> None:
    """Run one fold SELECT, leaving an unexpected legacy shape preserved in place."""
    columns = _table_columns(db, legacy)
    if not required <= columns:
        logger.warning(
            "legacy book table %s has an unexpected shape (%s); rows preserved in "
            "place, not folded",
            legacy,
            sorted(columns),
        )
        return
    db.execute_sql(sql)


def migrate(db: peewee.SqliteDatabase) -> Legacy:
    """Preserve/alter/re-key every legacy shape; return the kept copies.

    Preserve a pre-conid position table, ALTER ``live_strategy`` for ``scope`` and
    its segments, re-key every bare legacy scope, re-key/fold ``live_sim_lot``,
    re-key ``live_position`` from ``conid`` to ``position_id``, backfill
    ``live_execution.position_id`` from its ``conid``, and rename a legacy
    ``(scope, token)`` intent table to the identity columns. All additive or
    renames — never a drop.

    ORDER matters: the scope re-key runs FIRST, while every table still carries a
    ``scope`` column, so no table is left holding a bare scope; only then are the
    re-keyed tables renamed to their kept copies (a table renamed first would keep
    its old bare scope into the fold, orphaning its rows from the new one).

    The returned :class:`Legacy` names the kept copies whose rows the post-create
    step (:func:`_restore_positions` / :func:`_restore_intents`) copies across.
    """
    columns = _position_columns(db)
    if columns is not None and not {"conid", "source"} & columns:
        _preserve_legacy_positions(db)
    _add_strategy_columns(db)
    _rekey_scopes(db)
    sim_lots = _rekey_sim_lots(db)
    _rekey_executions(db)
    intents = _rekey_order_intents(db)
    _add_intent_columns(db)
    return Legacy(
        positions=_rekey_positions(db),
        sim_lots=sim_lots,
        intents=intents,
    )
