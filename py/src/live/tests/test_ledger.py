"""Tests for the per-scope sqlite book ledger (plan rev 4.1 §3)."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd
import pytest


from src.data.db import get_connection
from src.exec.refs import slug
from src.exec.types import OrderSide
from src.live.adapters.ibkr.trades import Execution, StrategyBook, reconcile
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
        order_ref=f"{slug(scope)}-20240603T150000-000",
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


pytestmark = pytest.mark.db
