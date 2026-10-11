"""Tests for the per-scope sqlite book ledger.

Only db-level guarantees live here: atomicity, idempotence, fail-loud reads, the
lease, and the scope merge. The pure fold/planner logic is covered in
``test_divergence.py``/``test_pure_plan.py``; this file exercises the store that
applies it.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd
import peewee
import pytest


from src.bt.state import ActionType, FillEvent, PortfolioState, Position
from src.data.db import get_connection
from src.exec.refs import scope_tag
from src.exec.types import OrderSide
from src.live.adapters.ibkr.trades import Execution, StrategyBook, reconcile
from src.live import ledger as ledger_module
from src.live.ledger import (
    LedgerReadError,
    SqliteLedger,
)
from src.live.lease import CycleInProgressError
from src.live.tests.ledgers import live_ledger
from src.live.pure import OrderResult, intent_to_signal
from src.live.types import ExecutionRecord, OrderIntent

pytestmark = pytest.mark.db

TS = cast("pd.Timestamp", pd.Timestamp("2024-06-03T15:00:00Z"))
OLD = cast("pd.Timestamp", pd.Timestamp("2020-01-01T15:00:00Z"))


@pytest.fixture
def ledger(tmp_path: Path) -> SqliteLedger:
    return live_ledger(tmp_path / "ledger.sqlite")


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


def test_constructing_a_ledger_writes_no_ddl(tmp_path: Path) -> None:
    # A --dry-run must write nothing, including no schema.
    db = tmp_path / "fresh.sqlite"
    SqliteLedger(db)
    assert _tables(db) == set()


def test_a_write_on_an_unmigrated_file_fails_loud_and_writes_no_ddl(
    tmp_path: Path,
) -> None:
    """The schema is ``ibkr db migrate``'s job: the ledger never builds one.

    A ledger that quietly created its tables on the first write would pass every
    other test here while re-introducing the half-schema-under-a-live-order
    failure — so an unmigrated file must raise, and must leave ``sqlite_master``
    empty.
    """
    db = tmp_path / "unmigrated.sqlite"
    ledger = SqliteLedger(db)
    with pytest.raises(peewee.OperationalError, match="live_strategy"):
        ledger.ensure_strategy("h", "S1", "n", 1000.0)
    assert _tables(db) == set()


def test_load_book_raises_on_a_shape_drifted_table(tmp_path: Path) -> None:
    # A table that exists but cannot be read (missing column) must NOT read as an
    # empty book: an empty book is "flat" downstream, i.e. re-open everything.
    db = tmp_path / "drift.sqlite"
    with get_connection(db) as con:
        con.execute("CREATE TABLE live_execution (scope TEXT, wrong INTEGER)")
    with pytest.raises(LedgerReadError):
        SqliteLedger(db).load_book("S1")


def test_cash_of_raises_on_a_shape_drifted_table(tmp_path: Path) -> None:
    db = tmp_path / "drift.sqlite"
    with get_connection(db) as con:
        con.execute("CREATE TABLE live_strategy (scope TEXT, wrong INTEGER)")
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


def test_scopes_do_not_leak(ledger: SqliteLedger) -> None:
    a = _exec("e1", scope="alpha", conid=1)
    b = _exec("e2", scope="beta", conid=2)
    abook, _ = reconcile("alpha", (a,), StrategyBook())
    bbook, _ = reconcile("beta", (b,), StrategyBook())
    ledger.save_book("alpha", abook, (a,), 1000.0)
    ledger.save_book("beta", bbook, (b,), 2000.0)
    assert {r.conid for r in ledger.load_book("alpha").rows} == {1}
    assert {r.conid for r in ledger.load_book("beta").rows} == {2}


def test_the_executions_book_refuses_a_non_conid_row(ledger: SqliteLedger) -> None:
    """A sim lot in the IBKR book is a LOUD failure, never a silently-missing lot.

    An account-role row read as a book row would read downstream as "flat" and
    re-open the position, so the projection raises rather than skips it.
    """
    ledger.record_results("S1", (_open_result("AAPL_1"),), TS)
    with pytest.raises(LedgerReadError):
        ledger.load_book("S1")


def test_two_ledgers_on_two_paths_do_not_retarget_each_other(tmp_path: Path) -> None:
    # peewee binds a model at CLASS level, so a single module-global database let a
    # second SqliteLedger silently retarget the first's connection (writes landing
    # in the wrong file, or a half-created schema). Each instance owns its db now.
    a = live_ledger(tmp_path / "a.sqlite")
    a.ensure_strategy("h", "S1", "n", 111.0)
    b = live_ledger(tmp_path / "b.sqlite")
    b.ensure_strategy("h", "S1", "n", 222.0)
    a.ensure_strategy("h", "S2", "n", 333.0)  # a still writes to its OWN file

    assert a.initial_capital_of("S1") == 111.0
    assert b.initial_capital_of("S1") == 222.0
    assert a.initial_capital_of("S2") == 333.0
    assert b.initial_capital_of("S2") == 0.0  # b never saw a's later write


def test_a_later_ensure_strategy_never_resets_the_cash_seed(
    ledger: SqliteLedger,
) -> None:
    """The FIRST write seeds ``initial_capital``; a later cycle keeps it (plan §4)."""
    ledger.ensure_strategy("h1", "S1", "first", 1000.0)
    ledger.ensure_strategy("h2", "S1", "second", 9999.0)  # a config edit

    assert ledger.initial_capital_of("S1") == 1000.0
    (audit,) = ledger.strategies_of("S1")
    assert audit.strategy_id == "h2" and audit.name == "second"  # audit advances


def test_touch_cycle_stamps_the_scope_row(ledger: SqliteLedger) -> None:
    ledger.ensure_strategy("h", "S1", "n", 1000.0)
    ledger.touch_cycle("S1", TS)
    (audit,) = ledger.strategies_of("S1")
    assert audit.last_cycle_at == TS


def _result(pid: str | None, intent: OrderIntent, *, price: float) -> OrderResult:
    return OrderResult(
        intent=intent,
        fill=FillEvent(
            signal=intent_to_signal(intent, TS),
            filled_qty=10.0,
            executed_price=price,
            commission=1.0,
            slippage=0.0,
            timestamp=TS,
        ),
        ok=True,
        position_id=pid,
    )


def _open_result(pid: str = "L1") -> OrderResult:
    return _result(
        pid,
        OrderIntent(
            symbol="AAPL",
            action=ActionType.long,
            qty=10.0,
            ref_price=100.0,
            reason="test",
        ),
        price=100.0,
    )


def _close_result(pid: str = "L1", *, price: float = 110.0) -> OrderResult:
    intent = OrderIntent(
        symbol="AAPL",
        action=ActionType.close,
        qty=10.0,
        ref_price=price,
        reason="test",
        position_id=pid,
    )
    # The close's leg side is the lot's MIRROR, which ``intent_to_signal`` reads
    # off the book; build the long lot the close targets so it sells, not buys.
    book = PortfolioState(
        cash=0.0,
        positions={
            "AAPL": (
                Position(
                    symbol="AAPL",
                    qty=10.0,
                    entry_price=100.0,
                    entry_time=TS,
                    stop_loss=None,
                    take_profit=None,
                    last_price=100.0,
                    type=ActionType.long,
                    position_id=pid,
                ),
            )
        },
        trades=(),
        equity_curve=(),
        initial_capital=0.0,
    )
    return OrderResult(
        intent=intent,
        fill=FillEvent(
            signal=intent_to_signal(intent, TS, book),
            filled_qty=10.0,
            executed_price=price,
            commission=1.0,
            slippage=0.0,
            timestamp=TS,
        ),
        ok=True,
        position_id=None,
    )


def test_record_results_applies_every_write_of_a_cycle_atomically(
    ledger: SqliteLedger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One cycle's results land TOGETHER: the caller never sees a half-cycle.

    The failure is forced on the LAST write of a two-lot batch — after the first
    lot's row is already inserted — so a per-write transaction would leave that
    row behind and this asserts the whole batch rolled back. The patch wraps the
    real ``_insert_execution`` (the forced error is not the only write attempted),
    so the first lot's insert really is in flight when the second one raises.
    """
    ledger.record_results("S1", (_open_result("L1"),), TS)
    real = ledger_module._insert_execution
    calls = 0

    def fail_last(record: ExecutionRecord) -> None:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise peewee.IntegrityError("forced mid-cycle failure")
        real(record)

    monkeypatch.setattr(ledger_module, "_insert_execution", fail_last)
    with pytest.raises(peewee.IntegrityError):
        ledger.record_results("S1", (_open_result("L2"), _close_result("L2")), TS)
    monkeypatch.undo()
    assert ledger.owned_ids("S1") == frozenset({"L1"})
    assert {r.execution_id for r in ledger.executions_of("S1")} == {"L1:open"}


def test_re_running_the_same_results_changes_nothing(ledger: SqliteLedger) -> None:
    """IGNORE keeps a re-applied cycle idempotent (a re-run is a no-op)."""
    results = (_open_result("L1"), _close_result("L1"))
    ledger.record_results("S1", results, TS)
    before = ledger.executions_of("S1")
    ledger.record_results("S1", results, TS)
    assert ledger.executions_of("S1") == before
    assert len(ledger.executions_of("S1")) == 2  # entry + exit, no duplicates


def test_a_round_trip_leaves_one_closed_book_row(ledger: SqliteLedger) -> None:
    """The derived book keeps ONE row for a round trip, at net zero and closed."""
    ledger.record_results("S1", (_open_result("265598"),), TS)
    assert ledger.owned_ids("S1") == frozenset({"265598"})
    ledger.record_results("S1", (_close_result("265598"),), TS)

    assert ledger.owned_ids("S1") == frozenset()
    (row,) = ledger.load_book("S1").rows
    assert row.qty == 0.0 and row.closed_at is not None
