"""Behaviour tests for the migration framework's durability contract.

Only behaviour a broken runner would actually break is asserted here:

* applying twice applies once (bookkeeping is the idempotence guard);
* a failing ``up`` leaves NO record and no schema change (the batch is one txn);
* a ``down`` round trip unwinds exactly once, and an irreversible target is
  refused rather than partially applied;
* the live baseline builds the target schema on a fresh file, idempotently.

Deliberately NOT here (delete-on-sight per AGENTS.md): file-layout assertions,
"the runner calls up()" wiring, bookkeeping column types, and registry-order tests.
"""

from __future__ import annotations

from pathlib import Path

import peewee
import pytest

from src.db import introspect
from src.db.migrations import bookkeeping
from src.db.migrations.runner import pending, run_down, run_pending
from src.db.migrations.types import IrreversibleMigrationError, Migration
from src.db.migrations.versions import data_0001_baseline
from src.live.models import LIVE_TABLES

#: The one migration this repo has with a genuine ``down``, as a LOCAL registry.
#: Asserting the whole ``DATA_MIGRATIONS`` tuple would fail on every new migration,
#: which is exactly the registry-shape assertion AGENTS.md calls delete-on-sight.
_REVERSIBLE_ONLY: tuple[Migration, ...] = (
    Migration(
        name=data_0001_baseline.NAME,
        up=data_0001_baseline.up,
        down=data_0001_baseline.down,
    ),
)

pytestmark = pytest.mark.db


@pytest.fixture()
def db(tmp_path: Path) -> peewee.SqliteDatabase:
    """An isolated sqlite file (never the production DB)."""
    return peewee.SqliteDatabase(str(tmp_path / "m.sqlite"))


def _up_creates_table(db: peewee.SqliteDatabase) -> None:
    db.execute_sql("CREATE TABLE IF NOT EXISTS widget (id INTEGER PRIMARY KEY)")


def _drop_widget(db: peewee.SqliteDatabase) -> None:
    db.execute_sql("DROP TABLE IF EXISTS widget")


# ── 1. bookkeeping idempotence ────────────────────────────────────


def test_run_pending_twice_applies_once(db: peewee.SqliteDatabase) -> None:
    """A second run writes nothing: the bookkeeping row IS the guard.

    Guards the exact failure that would re-run a migration on every boot.
    """
    calls: list[int] = []

    def up(conn: peewee.SqliteDatabase) -> None:
        calls.append(1)
        _up_creates_table(conn)

    registry = (Migration(name="w_0001", up=up),)

    assert run_pending(db, registry) == ("w_0001",)
    assert calls == [1]

    assert run_pending(db, registry) == ()
    assert calls == [1], "a second run must not invoke up() again"
    (recorded,) = db.execute_sql(
        f"SELECT COUNT(*) FROM {bookkeeping.MIGRATION_TABLE}"
    ).fetchone()
    assert recorded == 1


# ── 2. the live baseline builds the target schema ─────────────────


def test_a_fresh_live_baseline_creates_every_target_table(
    db: peewee.SqliteDatabase,
) -> None:
    """A fresh file gets the target ``live_*`` tables, and only those.

    The live file carries no legacy shape, so the baseline is a plain
    ``create_tables(LIVE_MODELS)``. Asserted on the TABLE SET the baseline creates,
    because a missing table would make the ledger read the book as flat.
    """
    from src.db.migrations.versions.live_0001_baseline import NAME, up as live_up

    baseline = (Migration(name=NAME, up=live_up),)
    assert run_pending(db, baseline) == (NAME,)

    live_tables = {name for name in LIVE_TABLES if introspect.table_exists(db, name)}
    assert live_tables == set(LIVE_TABLES)


def test_the_live_baseline_is_idempotent_on_an_already_migrated_file(
    db: peewee.SqliteDatabase,
) -> None:
    """Re-running the baseline against its own schema is a no-op.

    ``create_tables`` is ``IF NOT EXISTS``, so an operator whose file is already
    current (but has no bookkeeping row) can re-run without losing rows.
    """
    from src.db.migrations.versions.live_0001_baseline import up as live_up
    from src.live.models import LiveStrategy

    with db.bind_ctx((LiveStrategy,)):
        db.create_tables((LiveStrategy,))
    db.execute_sql(
        "INSERT INTO live_strategy (scope, strategy_id, adapter, config_name, "
        "instance, name, initial_capital, created_at, updated_at) "
        "VALUES ('S', 'h', 'sim', 'n', 'i', 'n', 1000.0, 0, 0)"
    )

    live_up(db)  # no bookkeeping needed: the DDL is idempotent by itself
    (count,) = db.execute_sql("SELECT COUNT(*) FROM live_strategy").fetchone()
    assert count == 1


# ── 3. a raising up() rolls back ──────────────────────────────────


def test_a_failing_up_leaves_no_record_and_no_schema_change(
    db: peewee.SqliteDatabase,
) -> None:
    """An ``up`` that raises aborts the batch: no record, no table, no row.

    A partially committed migration would leave the bookkeeping lying about which
    schema a file actually has — the failure this transaction exists to prevent.
    """

    def boom(_conn: peewee.SqliteDatabase) -> None:
        raise RuntimeError("migration exploded")

    def good(conn: peewee.SqliteDatabase) -> None:
        _up_creates_table(conn)

    registry = (
        Migration(name="ok_0001", up=good),
        Migration(name="boom_0002", up=boom),
    )

    with pytest.raises(RuntimeError, match="exploded"):
        run_pending(db, registry)

    assert bookkeeping.applied_names(db) == frozenset(), (
        "the first migration must have rolled back with the failing one"
    )
    assert not introspect.table_exists(db, "widget")
    assert pending(db, registry) == registry, "both are still pending"


# ── 4. down: a round trip, and a refusal ──────────────────────────


def test_a_reversible_migration_round_trips(db: peewee.SqliteDatabase) -> None:
    """``up`` then ``down`` returns to the pre-migration state, once each.

    Exercised on the one genuinely reversible migration this repo has (the data
    marker, whose ``down`` drops nothing). Guards that ``run_down`` unwinds and
    unrecords exactly the applied set, and that a second ``down`` is a no-op.
    """
    assert run_pending(db, _REVERSIBLE_ONLY) == (data_0001_baseline.NAME,)
    assert introspect.table_exists(db, "candle")
    assert bookkeeping.applied_names(db) == frozenset({data_0001_baseline.NAME})

    assert run_down(db, _REVERSIBLE_ONLY) == (data_0001_baseline.NAME,)
    assert bookkeeping.applied_names(db) == frozenset()
    assert pending(db, _REVERSIBLE_ONLY) == _REVERSIBLE_ONLY

    # The marker's down drops nothing: the bulk table the TS pipeline owns stays.
    assert introspect.table_exists(db, "candle")
    assert run_down(db, _REVERSIBLE_ONLY) == (), "nothing applied, nothing to unwind"


def test_run_down_refuses_an_irreversible_target_before_writing(
    db: peewee.SqliteDatabase,
) -> None:
    """An irreversible migration is refused with NO partial unwind.

    A refusal that still unwound the reversible migrations ahead of it would leave
    the operator unable to tell where the schema stopped.
    """

    def forward(conn: peewee.SqliteDatabase) -> None:
        _up_creates_table(conn)

    registry = (
        Migration(name="rev_0001", up=forward, down=_drop_widget),
        Migration(name="keep_0002", up=forward, down=None),
    )

    run_pending(db, registry)
    assert introspect.table_exists(db, "widget")

    with pytest.raises(IrreversibleMigrationError, match="keep_0002"):
        run_down(db, registry)

    assert bookkeeping.applied_names(db) == {"rev_0001", "keep_0002"}, (
        "nothing may be unrecorded when the unwind is refused"
    )
    assert introspect.table_exists(db, "widget"), "no partial unwind"


def test_run_down_refuses_an_unknown_target(db: peewee.SqliteDatabase) -> None:
    """A target that was never applied is refused, not silently ignored."""
    run_pending(db, _REVERSIBLE_ONLY)
    with pytest.raises(IrreversibleMigrationError, match="never_applied"):
        run_down(db, _REVERSIBLE_ONLY, "never_applied")
    assert bookkeeping.applied_names(db) == {data_0001_baseline.NAME}
