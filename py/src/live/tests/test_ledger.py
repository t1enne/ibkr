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
from src.live.ledger import SimLot

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
    ledger.ensure_strategy("hash-a", "momentum", "momentum")
    with caplog.at_level(logging.WARNING, logger="src.live.ledger"):
        ledger.ensure_strategy("hash-b", "momentum", "momentum")
    assert any("momentum" in record.getMessage() for record in caplog.records)


def test_same_scope_same_strategy_does_not_warn(
    ledger: SqliteLedger, caplog: pytest.LogCaptureFixture
) -> None:
    # The ordinary re-run (same config hash) must stay quiet.
    ledger.ensure_strategy("hash-a", "momentum", "momentum")
    with caplog.at_level(logging.WARNING, logger="src.live.ledger"):
        ledger.ensure_strategy("hash-a", "momentum", "momentum")
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


def test_net_exposure_sums_every_scope_on_a_conid(ledger: SqliteLedger) -> None:
    # The OPEN cross-check needs the net ACROSS scopes: one long scope and one
    # short scope on the same conid net together, matching what a shared account
    # would show. An untouched conid is flat.
    long_ex = _exec("e1", scope="alpha", conid=7, side=OrderSide.BUY, qty=10.0)
    short_ex = _exec("e2", scope="beta", conid=7, side=OrderSide.SELL, qty=4.0)
    abook, _ = reconcile("alpha", (long_ex,), StrategyBook())
    bbook, _ = reconcile("beta", (short_ex,), StrategyBook())
    ledger.save_book("alpha", abook, (long_ex,), 1000.0)
    ledger.save_book("beta", bbook, (short_ex,), 1000.0)
    assert ledger.net_exposure(7) == 6.0
    assert ledger.net_exposure(999) == 0.0


def test_net_exposure_on_an_unwritten_db_is_flat(tmp_path: Path) -> None:
    assert SqliteLedger(tmp_path / "l.sqlite").net_exposure(1) == 0.0


def test_net_exposure_ignores_the_sim_account_role(ledger: SqliteLedger) -> None:
    """The IBKR exposure cross-check reads conids, never the sim lot ids.

    One book table holds both roles, so the cross-check must not let a sim lot
    contribute to a conid's booked net. A sim lot id is never numeric in practice;
    the point is that an executions row IS, and only it counts.
    """
    ex = _exec("e1", conid=7, side=OrderSide.BUY, qty=10.0, scope="alpha")
    book, _ = reconcile("alpha", (ex,), StrategyBook())
    ledger.save_book("alpha", book, (ex,), 1000.0)
    ledger.record_sim_lot(
        "sim_a_1",
        SimLot(position_id="7", symbol="AAPL", side="long", qty=5.0, entry_price=1.0),
    )
    assert ledger.net_exposure(7) == 10.0  # the executions role only


def test_the_executions_book_refuses_a_non_conid_row(ledger: SqliteLedger) -> None:
    """A sim lot in the IBKR book is a LOUD failure, never a silently-missing lot.

    An account-role row read as a book row would read downstream as "flat" and
    re-open the position, so the projection raises rather than skips it.
    """
    ledger.ensure_cash("S1", 1000.0)  # first write builds the schema
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
        ledger.ensure_strategy("h1", "momentum", "phase")
    monkeypatch.undo()

    with get_connection(db) as con:
        kept = con.execute("SELECT symbol FROM live_position").fetchall()
    assert kept == [("AAPL",)]  # no row lost to a failed migration


def test_folded_sim_lots_are_the_account_role_of_the_one_book(tmp_path: Path) -> None:
    """The fold lands in ``live_position`` and reads back with ``source='account'``.

    The IBKR reconcile book (``load_book``) is the ``executions`` role of the SAME
    table, so the two never double-count: a sim scope's lots must NOT appear as
    IBKR book rows.
    """
    ledger = SqliteLedger(tmp_path / "l.sqlite")
    ledger.record_sim_lot(
        "sim_a_1",
        SimLot(
            position_id="AAPL_1",
            symbol="AAPL",
            side="long",
            qty=3.0,
            entry_price=10.0,
            opened_at=TS,
        ),
    )
    with get_connection(tmp_path / "l.sqlite") as con:
        rows = con.execute("SELECT position_id, source FROM live_position").fetchall()
    assert rows == [("AAPL_1", "account")]
    assert ledger.sim_open_ids("sim_a_1") == frozenset({"AAPL_1"})
    assert ledger.load_book("sim_a_1").rows == ()  # not the executions role


def test_sim_lot_detail_round_trips_and_survives_a_bare_reopen(
    ledger: SqliteLedger,
) -> None:
    """A recorded lot keeps its fill detail; a bared ownership write never erases it."""
    opened = cast("pd.Timestamp", pd.Timestamp("2024-06-03T15:00:00Z"))
    ledger.record_sim_lot(
        "S1",
        SimLot(
            position_id="AAPL_7",
            symbol="AAPL",
            side="long",
            qty=3.0,
            entry_price=10.0,
            stop_loss=9.0,
            take_profit=12.0,
            tag="v",
            opened_at=opened,
        ),
    )
    ledger.record_sim_open("S1", "AAPL_7")  # re-open, less detail
    (lot,) = ledger.sim_open_lots("S1")
    assert (lot.symbol, lot.side, lot.qty, lot.entry_price) == (
        "AAPL",
        "long",
        3.0,
        10.0,
    )
    assert (lot.stop_loss, lot.take_profit, lot.tag) == (9.0, 12.0, "v")
    assert lot.opened_at == opened
    assert lot.has_detail is True

    ledger.mark_sim_closed("S1", "AAPL_7", opened)
    assert ledger.sim_open_ids("S1") == frozenset()
    assert ledger.sim_open_lots("S1") == ()


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
