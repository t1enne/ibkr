"""Behaviour tests for the migration framework's durability contract.

Only behaviour a broken runner would actually break is asserted here:

* applying twice applies once (bookkeeping is the idempotence guard);
* a baseline absorbs an already-hand-migrated DB — the operator's real situation,
  and the one the recognition guards exist for;
* a failing ``up`` leaves NO record and no schema change (the batch is one txn);
* a ``down`` round trip unwinds exactly once, and an irreversible target is
  refused rather than partially applied.

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

    Guards the exact failure that would re-run a re-key on every boot.
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


# ── 2. the baseline absorbs an already-migrated DB ────────────────


def test_a_hand_migrated_db_absorbs_the_baseline_without_new_legacy_tables(
    db: peewee.SqliteDatabase,
) -> None:
    """A DB already migrated by the old code path absorbs ``live_0001`` as a no-op.

    This is the operator's real case: their file has the CURRENT schema and NO
    ``peewee_migration`` table, so the baseline runs against it. It must leave the
    rows and their identities exactly as they were and create NO ``*_legacy``
    table — a ``_free_legacy_name``/``_rekey_positions`` guard regression would
    rename the live book away and read it as flat.
    """
    from src.db.migrations.versions.live_0001_baseline import up as live_up

    baseline = (Migration(name="live_0001_baseline", up=live_up),)
    # First application through the runner, so the bookkeeping row exists to drop.
    run_pending(db, baseline)
    db.execute_sql(
        "INSERT INTO live_position (scope, position_id, symbol, side, qty, "
        "entry_price, source, order_ref, tag) VALUES "
        "('S','42','AAPL','long',10.0,100.0,'executions','','')"
    )
    before = _counts(db)

    # The operator's file, then. Wipe ONLY the bookkeeping, as their file has none.
    db.execute_sql(f"DROP TABLE {bookkeeping.MIGRATION_TABLE}")

    assert run_pending(db, baseline) == ("live_0001_baseline",)
    after = _counts(db)

    assert after["live_position"] == before["live_position"] == 1
    assert not [name for name in after if name.endswith("_legacy")], (
        "an already-current book must not be re-keyed into a legacy copy"
    )
    (position_id,) = db.execute_sql("SELECT position_id FROM live_position").fetchone()
    assert position_id == "42", "the lot's identity must survive the absorption"
    assert "live_position" in db.get_tables()


# ── 2b. a failed schema build rolls the legacy rename back ────────


def test_a_failed_live_baseline_build_rolls_the_rename_back(
    db: peewee.SqliteDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The legacy rename and the current-schema build are ONE transaction.

    The live baseline renames a pre-4.1 ``live_position`` to its preserved
    ``*_legacy`` copy and then builds the current table. When the build fails the
    rename must roll back with it: a committed rename against an unbuilt schema
    leaves neither shape readable, so the book reads downstream as FLAT and every
    position is re-opened.
    """
    from src.db.migrations.versions.live_0001_baseline import up as live_up

    db.execute_sql(
        "CREATE TABLE live_position (strategy_id TEXT, position_id TEXT, "
        "symbol TEXT, side TEXT, qty REAL, status TEXT, "
        "PRIMARY KEY (strategy_id, position_id))"
    )
    db.execute_sql(
        "INSERT INTO live_position VALUES ('h1','lot-1','AAPL','long',10.0,'open')"
    )

    def explode(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("schema build failed mid-transaction")

    monkeypatch.setattr(db, "create_tables", explode)
    with pytest.raises(RuntimeError, match="mid-transaction"):
        run_pending(db, (Migration(name="live_0001_baseline", up=live_up),))

    (symbol,) = db.execute_sql("SELECT symbol FROM live_position").fetchone()
    assert symbol == "AAPL", "a legacy row must survive a failed schema build"
    assert bookkeeping.applied_names(db) == frozenset()


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

    The live baseline renames and folds durable rows; its only inverse would drop
    what it preserved. A refusal that still unwound the reversible migrations
    ahead of it would leave the operator unable to tell where the schema stopped.
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


def _counts(db: peewee.SqliteDatabase) -> dict[str, int]:
    """Row count per table, so a NEW table shows up as a key."""
    names = [
        str(row[0])
        for row in db.execute_sql("SELECT name FROM sqlite_master WHERE type='table'")
    ]
    return {
        name: int(db.execute_sql(f"SELECT COUNT(*) FROM {name}").fetchone()[0])
        for name in names
    }
