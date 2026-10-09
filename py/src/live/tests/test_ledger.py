"""Tests for the per-scope sqlite book ledger (plan rev 4.1 §3)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import pandas as pd
import peewee
import pytest


from src.bt.state import ActionType, FillEvent
from src.data.db import get_connection
from src.exec.refs import scope_tag
from src.exec.types import OrderSide
from src.live.adapters.ibkr.trades import Execution, StrategyBook, reconcile
from src.live.identity import (
    IntentKey,
    IntentRecord,
    IntentState,
    order_ref,
)
from src.live import ledger as ledger_module
from src.live.ledger import (
    LedgerReadError,
    SqliteLedger,
)
from src.live.lease import CycleInProgressError
from src.live.tests.ledgers import live_ledger
from src.live.pure import OrderResult, intent_to_signal
from src.live.types import ExecutionRecord, OrderIntent

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
    # A --dry-run must write nothing, including no schema (plan §6 phase 3.5).
    db = tmp_path / "fresh.sqlite"
    SqliteLedger(db)
    assert _tables(db) == set()


def test_a_write_on_an_unmigrated_file_fails_loud_and_writes_no_ddl(
    tmp_path: Path,
) -> None:
    """The schema is ``ibkr db migrate``'s job: the ledger never builds one.

    Pins the WHOLE post-migration-bootstrap contract. A ledger that quietly
    created its tables on the first write would pass every other test here while
    re-introducing the half-schema-under-a-live-order failure — so an unmigrated
    file must raise, and must leave ``sqlite_master`` empty.
    """
    db = tmp_path / "unmigrated.sqlite"
    ledger = SqliteLedger(db)
    with pytest.raises(peewee.OperationalError, match="live_cash"):
        ledger.ensure_cash("S1", 1000.0)
    assert _tables(db) == set()


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
    ledger.ensure_cash("S1", 1000.0)
    with get_connection(ledger.db_path) as con:
        con.execute(
            "INSERT INTO live_position (scope, position_id, symbol, side, qty, "
            "entry_price, tag, order_ref, source) VALUES "
            "('S1', 'AAPL_1', 'AAPL', 'long', 1.0, 1.0, '', '', 'executions')"
        )
    with pytest.raises(LedgerReadError):
        ledger.load_book("S1")


def test_two_ledgers_on_two_paths_do_not_retarget_each_other(tmp_path: Path) -> None:
    # peewee binds a model at CLASS level, so a single module-global database let a
    # second SqliteLedger silently retarget the first's connection (writes landing
    # in the wrong file, or a half-created schema). Each instance owns its db now.
    a = live_ledger(tmp_path / "a.sqlite")
    a.ensure_cash("S1", 111.0)
    b = live_ledger(tmp_path / "b.sqlite")
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
    return _result(
        None,
        OrderIntent(
            symbol="AAPL",
            action=ActionType.close,
            qty=10.0,
            ref_price=price,
            reason="test",
            position_id=pid,
        ),
        price=price,
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
    """REPLACE/IGNORE keep a re-applied cycle idempotent (a re-run is a no-op)."""
    results = (_open_result("L1"), _close_result("L1"))
    ledger.record_results("S1", results, TS)
    before = ledger.executions_of("S1")
    ledger.record_results("S1", results, TS)
    assert ledger.executions_of("S1") == before
    assert len(ledger.executions_of("S1")) == 2  # entry + exit, no duplicates


# --- pending order intents (PendingIntents) ---------------------------------


def _key(
    symbol: str = "AAPL",
    action: ActionType = ActionType.long,
    pid: str | None = None,
) -> IntentKey:
    return IntentKey(scope="S1", symbol=symbol, action=action, position_id=pid)


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


def test_a_token_collision_cannot_alias_two_distinct_identities(
    ledger: SqliteLedger, tmp_path: Path
) -> None:
    """D7: the identity columns key the table, so a crc32 collision never aliases.

    Two distinct intents whose TOKENS collide (forced here by rewriting one) stay
    separately addressable by IDENTITY — the token is a plain column, not the key
    — so a write for A can never overwrite B's ROW. This is a ROW-level guarantee
    ONLY: the broker still attributes a working order by the token-keyed ref
    prefix, so two colliding keys share a prefix there and their orders can still
    be bridged at the broker layer.
    """
    key = _key("AAPL")
    foreign = replace(key, symbol="MSFT")
    ledger.save(
        IntentRecord(
            key=key,
            state=IntentState.WORKING,
            attempt=0,
            order_ref=order_ref(key, 0),
            order_id="1",
            decision_ts=None,
        )
    )
    ledger.save(
        IntentRecord(
            key=foreign,
            state=IntentState.WORKING,
            attempt=0,
            order_ref=order_ref(foreign, 0),
            order_id="2",
            decision_ts=None,
        )
    )
    with get_connection(tmp_path / "ledger.sqlite") as con:  # force a collision
        con.execute(
            "UPDATE live_order_intent SET token=? WHERE symbol='AAPL'",
            (foreign.token(),),
        )
    first = ledger.load(key)
    second = ledger.load(foreign)
    assert first is not None and second is not None
    assert (first.order_id, second.order_id) == ("1", "2")  # neither aliased
    assert ledger.load_open("S1") == (first, second)


pytestmark = pytest.mark.db


def test_pruned_terminal_record_mints_the_next_attempt_at_zero(
    ledger: SqliteLedger,
) -> None:
    """A pruned CLOSED row is gone, so the next attempt restarts the counter."""
    key = _key("AAPL")
    ledger.open_attempt(key, OLD, OLD)
    ledger.close(key, IntentState.FILLED, "1", OLD)
    assert ledger.prune(TS) == 1
    assert ledger.load(key) is None
    # The counter restarts at 0 once the prior row is pruned (fresh cOID reuse).
    record = ledger.open_attempt(key, TS, TS)
    assert record.attempt == 0


def test_intent_record_roundtrips_tif_and_stuck_cycles(ledger: SqliteLedger) -> None:
    """N1: the new columns persist and read back (tif + wedged counter)."""
    key = _key("AAPL")
    ledger.save(
        IntentRecord(
            key=key,
            state=IntentState.WORKING,
            attempt=0,
            order_ref=order_ref(key, 0),
            order_id="1",
            decision_ts=TS,
            tif="GTC",
            stuck_cycles=2,
        )
    )
    record = ledger.load(key)
    assert record is not None
    assert (record.tif, record.stuck_cycles) == ("GTC", 2)
