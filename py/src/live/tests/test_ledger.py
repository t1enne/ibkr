"""Tests for the per-scope sqlite book ledger (plan rev 4.1 §3)."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import cast

import pandas as pd
import pytest


from src.data.db import get_connection
from src.exec.refs import scope_tag
from src.exec.types import OrderSide
from src.live.adapters.ibkr.trades import Execution, StrategyBook, reconcile
from src.bt.state import ActionType
from src.live.identity import (
    IntentKey,
    IntentRecord,
    IntentState,
    order_ref,
)
from src.live.ledger import (
    LedgerReadError,
    SqliteLedger,
    config_hash,
    execution_cash_delta,
)
from src.live.lease import CycleInProgressError

TS = cast("pd.Timestamp", pd.Timestamp("2024-06-03T15:00:00Z"))
OLD = cast("pd.Timestamp", pd.Timestamp("2020-01-01T15:00:00Z"))


@pytest.fixture
def ledger(tmp_path: Path) -> SqliteLedger:
    return SqliteLedger(tmp_path / "ledger.sqlite")


def _exec(
    execution_id: str,
    *,
    conid: int = 265598,
    side: OrderSide = OrderSide.BUY,
    qty: float = 10.0,
    price: float = 100.0,
    scope: str = "S1",
    ts: pd.Timestamp = TS,
) -> Execution:
    return Execution(
        execution_id=execution_id,
        order_id="o" + execution_id,
        order_ref=f"{scope_tag(scope)}-20240603T150000-000",
        conid=conid,
        symbol="AAPL",
        side=side,
        qty=qty,
        price=price,
        commission=1.0,
        ts=ts,
    )


def _tables(path: Path) -> set[str]:
    with get_connection(path) as con:
        rows = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    return {r[0] for r in rows}


def test_reusing_a_scope_with_a_different_strategy_warns(
    ledger: SqliteLedger, caplog: pytest.LogCaptureFixture
) -> None:
    # Two configs sharing a name share one scope (book + cash + cOID prefix); a
    # different strategy_id on an existing scope must be loud, not silent (L7).
    ledger.ensure_strategy("hash-a", "momentum", "momentum", "paper")
    with caplog.at_level(logging.WARNING, logger="src.live.ledger"):
        ledger.ensure_strategy("hash-b", "momentum", "momentum", "paper")
    assert any("momentum" in record.getMessage() for record in caplog.records)


def test_same_scope_same_strategy_does_not_warn(
    ledger: SqliteLedger, caplog: pytest.LogCaptureFixture
) -> None:
    # The ordinary re-run (same config hash) must stay quiet.
    ledger.ensure_strategy("hash-a", "momentum", "momentum", "paper")
    with caplog.at_level(logging.WARNING, logger="src.live.ledger"):
        ledger.ensure_strategy("hash-a", "momentum", "momentum", "paper")
    assert not [
        record
        for record in caplog.records
        if "belongs to strategy" in record.getMessage()
    ]


def test_config_hash_order_insensitive_over_keys() -> None:
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def test_config_hash_differs_on_value_change() -> None:
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_constructing_a_ledger_writes_no_ddl(tmp_path: Path) -> None:
    # A --dry-run must write nothing, including no schema (plan §6 phase 3.5).
    db = tmp_path / "fresh.sqlite"
    SqliteLedger(db)
    assert _tables(db) == set()


def test_load_book_on_an_unwritten_db_is_empty(tmp_path: Path) -> None:
    SqliteLedger(tmp_path / "ledger.sqlite")
    assert SqliteLedger(tmp_path / "ledger.sqlite").load_book("S1") == StrategyBook()


def test_unwritten_reads_are_empty_not_errors(tmp_path: Path) -> None:
    # The dry-run case: an absent table means "unwritten", never a failure.
    ledger = SqliteLedger(tmp_path / "ledger.sqlite")
    assert ledger.cash_of("S1", 1000.0) == 1000.0
    assert ledger.initial_capital_of("S1") == 0.0
    assert ledger.sim_open_ids("S1") == frozenset()


def test_load_book_raises_on_a_shape_drifted_table(tmp_path: Path) -> None:
    # A table that exists but cannot be read (missing column) must NOT read as an
    # empty book: an empty book is "flat" downstream, i.e. re-open everything.
    db = tmp_path / "drift.sqlite"
    with get_connection(db) as con:
        con.execute("CREATE TABLE live_position (scope TEXT, wrong INTEGER)")
        con.execute("CREATE TABLE live_execution (scope TEXT, wrong INTEGER)")
    with pytest.raises(LedgerReadError):
        SqliteLedger(db).load_book("S1")


def test_cash_of_raises_on_a_shape_drifted_table(tmp_path: Path) -> None:
    db = tmp_path / "drift.sqlite"
    with get_connection(db) as con:
        con.execute("CREATE TABLE live_cash (scope TEXT, wrong INTEGER)")
    with pytest.raises(LedgerReadError):
        SqliteLedger(db).cash_of("S1", 1000.0)


def test_cycle_lease_refuses_a_second_holder(tmp_path: Path) -> None:
    # A cron overlap / racing human run must REFUSE, not both place off one book.
    path = tmp_path / "l.sqlite"
    with SqliteLedger(path).cycle_lease():
        with pytest.raises(CycleInProgressError):
            with SqliteLedger(path).cycle_lease():
                pass


def test_cycle_lease_releases_on_exit(tmp_path: Path) -> None:
    # The kernel drops the flock when the block ends, so a later cycle re-acquires.
    path = tmp_path / "l.sqlite"
    with SqliteLedger(path).cycle_lease():
        pass
    with SqliteLedger(path).cycle_lease():
        pass


def test_save_then_load_round_trips_one_row(ledger: SqliteLedger) -> None:
    ex = _exec("e1")
    book, _ = reconcile("S1", (ex,), StrategyBook())
    ledger.save_book("S1", book, (ex,), 100_000.0)

    loaded = ledger.load_book("S1")
    assert len(loaded.rows) == 1
    assert loaded.rows[0].conid == 265598
    assert loaded.applied == frozenset({"e1"})
    assert loaded.rows[0].opened_at == TS


def test_reapplying_the_window_after_reload_is_a_noop(ledger: SqliteLedger) -> None:
    ex = _exec("e1")
    book, _ = reconcile("S1", (ex,), StrategyBook())
    ledger.save_book("S1", book, (ex,), 100_000.0)

    reloaded = ledger.load_book("S1")
    again, warnings = reconcile("S1", (ex,), reloaded)
    assert again == reloaded and warnings == ()


def test_cash_is_initial_plus_scope_executions(ledger: SqliteLedger) -> None:
    buy = _exec("e1", side=OrderSide.BUY, qty=10, price=100.0)
    sell = _exec("e2", side=OrderSide.SELL, qty=10, price=110.0)
    book, _ = reconcile("S1", (buy, sell), StrategyBook())
    ledger.save_book("S1", book, (buy, sell), 100_000.0)
    expected = 100_000.0 + execution_cash_delta(buy) + execution_cash_delta(sell)
    assert ledger.cash_of("S1", 0.0) == pytest.approx(expected)


def test_executions_are_recorded_once_across_reruns(ledger: SqliteLedger) -> None:
    ex = _exec("e1")
    book, _ = reconcile("S1", (ex,), StrategyBook())
    ledger.save_book("S1", book, (ex,), 100_000.0)
    ledger.save_book("S1", book, (ex,), 100_000.0)  # re-save, same execution id
    assert ledger.cash_of("S1", 0.0) == pytest.approx(
        100_000.0 + execution_cash_delta(ex)
    )


def test_scopes_do_not_leak(ledger: SqliteLedger) -> None:
    a = _exec("e1", scope="alpha", conid=1)
    b = _exec("e2", scope="beta", conid=2)
    abook, _ = reconcile("alpha", (a,), StrategyBook())
    bbook, _ = reconcile("beta", (b,), StrategyBook())
    ledger.save_book("alpha", abook, (a,), 1000.0)
    ledger.save_book("beta", bbook, (b,), 2000.0)
    assert {r.conid for r in ledger.load_book("alpha").rows} == {1}
    assert {r.conid for r in ledger.load_book("beta").rows} == {2}


def test_two_ledgers_on_two_paths_do_not_retarget_each_other(tmp_path: Path) -> None:
    # peewee binds a model at CLASS level, so a single module-global database let a
    # second SqliteLedger silently retarget the first's connection (writes landing
    # in the wrong file, or a half-created schema). Each instance owns its db now.
    a = SqliteLedger(tmp_path / "a.sqlite")
    a.ensure_cash("S1", 111.0)
    b = SqliteLedger(tmp_path / "b.sqlite")
    b.ensure_cash("S1", 222.0)
    a.ensure_cash("S2", 333.0)  # a still writes to its OWN file

    assert a.initial_capital_of("S1") == 111.0
    assert b.initial_capital_of("S1") == 222.0
    assert a.initial_capital_of("S2") == 333.0
    assert b.initial_capital_of("S2") == 0.0  # b never saw a's later write


def test_prune_closed_deletes_only_old_closed(ledger: SqliteLedger) -> None:
    old = _exec("e1", ts=OLD)
    book, _ = reconcile("S1", (old,), StrategyBook())
    # close it immediately so it carries closed_at == OLD
    close = _exec("e2", side=OrderSide.SELL, ts=OLD)
    book, _ = reconcile("S1", (old, close), StrategyBook())
    ledger.save_book("S1", book, (old, close), 1000.0)
    deleted = ledger.prune_closed(TS)
    assert deleted == 1
    assert ledger.load_book("S1").rows == ()


def test_migration_preserves_incompatible_legacy_position_table(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # A pre-conid live_position cannot be read into the conid-keyed book, but its
    # durable open rows must NOT be dropped: they are preserved under a kept copy
    # and named loudly, since the rolling trades window cannot rebuild them.
    db = tmp_path / "legacy.sqlite"
    with get_connection(db) as con:
        con.execute(
            "CREATE TABLE live_position (strategy_id TEXT, position_id TEXT, "
            "symbol TEXT, side TEXT, qty REAL, status TEXT, "
            "PRIMARY KEY (strategy_id, position_id))"
        )
        con.execute(
            "INSERT INTO live_position VALUES "
            "('h1', 'lot-1', 'AAPL', 'long', 10.0, 'open')"
        )
        con.execute(
            "CREATE TABLE live_strategy (strategy_id TEXT PRIMARY KEY, name TEXT, "
            "mode TEXT, created_at INTEGER, last_cycle_at INTEGER)"
        )
    ledger = SqliteLedger(db)
    with caplog.at_level("WARNING", logger="src.live.ledger"):
        ledger.ensure_strategy("h1", "momentum", "phase", "paper")

    tables = _tables(db)
    assert "live_position_legacy" in tables  # preserved, not dropped
    assert {"live_position", "live_execution", "live_cash"} <= tables
    with get_connection(db) as con:
        kept = con.execute(
            "SELECT symbol, side, qty FROM live_position_legacy"
        ).fetchall()
        fresh = con.execute("SELECT * FROM live_position").fetchall()
    assert kept == [("AAPL", "long", 10.0)]
    assert fresh == []  # the new conid-keyed book starts empty, not erased
    assert any("AAPL" in r.message and "double" in r.message for r in caplog.records)
    # The migration is one-time: a second write neither re-warns nor loses rows.
    ledger.touch_cycle("h1", TS)
    assert "live_position_legacy" in _tables(db)


def test_migration_keeps_every_legacy_copy_never_dropping_rows(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # D7(b): a legacy-shaped live_position re-created WITH rows beside an already
    # kept copy must be preserved, not dropped. The old branch warned about the
    # open rows and then DROPped the table, losing them.
    db = tmp_path / "legacy.sqlite"
    legacy_schema = (
        "CREATE TABLE %s (strategy_id TEXT, position_id TEXT, symbol TEXT, "
        "side TEXT, qty REAL, status TEXT, PRIMARY KEY (strategy_id, position_id))"
    )
    with get_connection(db) as con:
        con.execute(legacy_schema % "live_position")
        con.execute(
            "INSERT INTO live_position VALUES ('h1','lot-1','AAPL','long',10.0,'open')"
        )
        con.execute(legacy_schema % "live_position_legacy")
        con.execute(
            "INSERT INTO live_position_legacy VALUES "
            "('h0','lot-0','MSFT','long',5.0,'open')"
        )
    ledger = SqliteLedger(db)
    with caplog.at_level("WARNING", logger="src.live.ledger"):
        ledger.ensure_strategy("h1", "momentum", "phase", "paper")

    tables = _tables(db)
    assert "live_position" in tables  # fresh conid-keyed book created
    assert "live_position_legacy" in tables  # the pre-existing copy kept
    assert "live_position_legacy_1" in tables  # the re-created one kept, not dropped
    with get_connection(db) as con:
        kept = con.execute("SELECT symbol FROM live_position_legacy").fetchall()
        recreated = con.execute("SELECT symbol FROM live_position_legacy_1").fetchall()
        fresh = con.execute("SELECT * FROM live_position").fetchall()
    assert kept == [("MSFT",)]
    assert recreated == [("AAPL",)]  # no row lost
    assert fresh == []  # the new book starts empty, not erased


def test_migration_check_and_action_are_one_atomic_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # D7(a): the "is-this-legacy" check and the rename run in ONE transaction, so a
    # concurrent process cannot see a half-migrated schema and no check-then-act
    # window exists. We prove ATOMICITY: when the DDL step fails the rename is
    # fully ROLLED BACK — before the fix the rename committed on its own and the
    # legacy table survived a failed first write. That, plus ``BEGIN IMMEDIATE``
    # taking SQLite's write lock up front, is what makes two racing migration
    # PROCESSES serialize instead of dropping each other's table.
    db = tmp_path / "atomic.sqlite"
    setup = get_connection(db)
    setup.execute(
        "CREATE TABLE live_position (strategy_id TEXT, position_id TEXT, "
        "symbol TEXT, side TEXT, qty REAL, status TEXT, "
        "PRIMARY KEY (strategy_id, position_id))"
    )
    setup.execute(
        "INSERT INTO live_position VALUES ('h1','lot-1','AAPL','long',10.0,'open')"
    )
    setup.commit()
    setup.close()
    ledger = SqliteLedger(db)

    def explode(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("schema build failed mid-transaction")

    monkeypatch.setattr(ledger._database, "create_tables", explode)
    with pytest.raises(RuntimeError, match="mid-transaction"):
        ledger.ensure_strategy("h1", "momentum", "phase", "paper")
    monkeypatch.undo()

    tables = _tables(db)
    assert "live_position" in tables  # the rename was rolled back
    assert "live_position_legacy" not in tables
    with get_connection(db) as con:
        kept = con.execute("SELECT symbol FROM live_position").fetchall()
    assert kept == [("AAPL",)]  # no row lost to a failed migration


def test_migration_rekeys_legacy_sim_lots_to_scope(tmp_path: Path) -> None:
    # Pre-4.1 sim ownership was keyed by the config hash (strategy_id); re-key to
    # the stable scope, preserving the rows (never dropped).
    db = tmp_path / "legacy.sqlite"
    with get_connection(db) as con:
        con.execute(
            "CREATE TABLE live_sim_lot (strategy_id TEXT, position_id TEXT, "
            "closed_at INTEGER, PRIMARY KEY (strategy_id, position_id))"
        )
        con.execute("INSERT INTO live_sim_lot VALUES ('hash1','lot-1',NULL)")
        con.execute("INSERT INTO live_sim_lot VALUES ('hash1','lot-2',123)")
        con.execute(
            "CREATE TABLE live_strategy (strategy_id TEXT PRIMARY KEY, "
            "scope TEXT NOT NULL DEFAULT '', name TEXT, mode TEXT, "
            "created_at INTEGER, last_cycle_at INTEGER)"
        )
        con.execute(
            "INSERT INTO live_strategy VALUES ('hash1','momentum','n','paper',0,NULL)"
        )
    ledger = SqliteLedger(db)
    ledger.record_sim_open("momentum", "lot-3")  # first write triggers migration
    assert ledger.sim_open_ids("momentum") == frozenset({"lot-1", "lot-3"})
    with get_connection(db) as con:
        cols = {r[1] for r in con.execute("PRAGMA table_info(live_sim_lot)")}
    assert "scope" in cols and "strategy_id" not in cols


# --- pending order intents (PendingIntents) ---------------------------------


def _key(
    symbol: str = "AAPL",
    action: ActionType = ActionType.long,
    pid: str | None = None,
) -> IntentKey:
    return IntentKey(scope="S1", symbol=symbol, action=action, position_id=pid)


def test_intent_table_is_created_lazily_and_is_named_for_this_feature(
    ledger: SqliteLedger, tmp_path: Path
) -> None:
    path = tmp_path / "ledger.sqlite"
    assert ledger.load(_key()) is None  # a read writes no DDL
    assert "live_order_intent" not in _tables(path)
    ledger.open_attempt(_key(), None, TS)
    assert "live_order_intent" in _tables(path)


def test_open_attempt_bumps_after_the_prior_record(ledger: SqliteLedger) -> None:
    key = _key()
    first = ledger.open_attempt(key, TS, TS)
    assert first.attempt == 0 and first.state is IntentState.PENDING
    assert first.order_ref == order_ref(key, 0)
    second = ledger.open_attempt(key, TS, TS)
    assert second.attempt == 1
    assert second.order_ref != first.order_ref  # a NEW cOID for the new attempt


def test_close_and_load_open_round_trip(ledger: SqliteLedger) -> None:
    key = _key()
    ledger.open_attempt(key, TS, TS)
    ledger.close(key, IntentState.WORKING, "97932", TS)
    record = ledger.load(key)
    assert record is not None
    assert record.state is IntentState.WORKING and record.order_id == "97932"
    assert ledger.load_open("S1") == (record,)


def test_close_unresolved_keeps_a_known_order_id(ledger: SqliteLedger) -> None:
    key = _key()
    ledger.open_attempt(key, TS, TS)
    ledger.close(key, IntentState.WORKING, "97932", TS)
    ledger.close(key, IntentState.UNRESOLVED, None, TS)
    record = ledger.load(key)
    assert record is not None
    assert record.state is IntentState.UNRESOLVED and record.order_id == "97932"


def test_terminal_records_are_excluded_from_load_open(ledger: SqliteLedger) -> None:
    key = _key()
    ledger.open_attempt(key, TS, TS)
    ledger.close(key, IntentState.FILLED, "97932", TS)
    assert ledger.load_open("S1") == ()


def test_prune_deletes_only_closed_old_rows(ledger: SqliteLedger) -> None:
    filled = _key("AAPL")
    open_key = _key("MSFT")
    ledger.open_attempt(filled, OLD, OLD)
    ledger.close(filled, IntentState.FILLED, "1", OLD)
    ledger.open_attempt(open_key, OLD, OLD)
    # A cutoff after the closed row but the OPEN row is never pruned.
    assert ledger.prune(TS) == 1
    assert ledger.load(filled) is None
    assert ledger.load(open_key) is not None


def test_save_overwrites_a_same_key_record(ledger: SqliteLedger) -> None:
    key = _key()
    ledger.open_attempt(key, TS, TS)
    ledger.save(
        IntentRecord(
            key=key,
            state=IntentState.WORKING,
            attempt=0,
            order_ref=order_ref(key, 0),
            order_id="97932",
            decision_ts=TS,
        )
    )
    record = ledger.load(key)
    assert record is not None and record.state is IntentState.WORKING


def test_load_warns_and_ignores_a_token_owned_by_a_foreign_identity(
    ledger: SqliteLedger, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Same (scope, token) but a different symbol/action/position_id is a crc32
    # collision: the foreign row is warned about and read as absent, never adopted.
    key = _key("AAPL")
    foreign = replace(key, symbol="MSFT")
    ledger.save(
        IntentRecord(
            key=foreign,
            state=IntentState.WORKING,
            attempt=0,
            order_ref=order_ref(foreign, 0),
            order_id="1",
            decision_ts=None,
        )
    )
    with get_connection(tmp_path / "ledger.sqlite") as con:  # force the collision
        con.execute(
            "UPDATE live_order_intent SET token=? WHERE token=?",
            (key.token(), foreign.token()),
        )
    with caplog.at_level(logging.WARNING):
        assert ledger.load(key) is None
    assert any("collides" in r.message for r in caplog.records)


pytestmark = pytest.mark.db
