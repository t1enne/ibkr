"""Tests for the side-by-side pf view: store reads, broker reads, rendering, CLI.

Regression tests pin the contracts a later edit must not break — a bare read
leaves no schema behind, ownership is read from the right table per adapter
(sim lots vs book rows), a partial schema fails loudly. Characterisation tests
pin the CURRENT display shape of ``render_pf`` (block names, the
``NOT-OURS``/``ours`` markers, the divergence lines), so a format edit is a
deliberate diff rather than an accident.
"""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from typing import cast

import pandas as pd
import pytest
from click.testing import CliRunner

from dataclasses import replace

from src.bt.state import ActionType
from src.shared.style import COLOR, strip_ansi
from src.data.db import get_connection
from src.data.ibkr.client import IbkrClient
from src.exec.types import OrderSide
from src.live.adapters.ibkr.trades import Execution, StrategyBook, reconcile
from src.exec.refs import scope_tag
from src.live.cli import (
    _store_and_broker,
    live_group,
    load_live_config,
    watch_pf_loop,
)
from src.live.identity import IntentKey, IntentState, order_ref
from src.live.ledger import ExecutionRecord, SqliteLedger
from src.live.pf import (
    BrokerLot,
    BrokerSide,
    PfReport,
    StoreLot,
    StoreSide,
    position_rows,
    read_ibkr_broker,
    read_sim_broker,
    read_store,
    render_pf,
    scope_stats,
)
from src.live.types import LiveConfig
from src.live.ledger_sim import SimLot
from src.live.portfolio_source import fixture_is_managed
from src.live.result import Ok

TS: pd.Timestamp = cast(pd.Timestamp, pd.Timestamp("2024-06-03T15:00:00Z"))

BASE_CONFIG = {
    "name": "pf_test",
    "strategy_type": "vwatr_div_dsl",
    "symbols": ["AAPL"],
    "initial_capital": 50000,
    "commission": 0.05,
    "warmup": "300d",
    "trading_start": "2020-09-01",
    "trading_end": "2026-09-01",
    "bars": ["1d"],
    "strategy_params": {"vwatr_period": 14},
}


def _cfg(tmp_path: Path, **live_keys: object) -> LiveConfig:
    """A minimal valid LiveConfig written to a temp file and re-loaded."""
    target = tmp_path / "cfg.json"
    target.write_text(json.dumps({**BASE_CONFIG, **live_keys}))
    return load_live_config(str(target))


def _tables(path: Path) -> set[str]:
    with get_connection(path) as con:
        rows = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    return {r[0] for r in rows}


def _write_fixture(path: Path, doc: dict[str, object]) -> str:
    path.write_text(json.dumps(doc))
    return str(path)


# --- (a) strategies_of on a fresh ledger -------------------------------------


def test_strategies_of_on_a_fresh_ledger_is_empty(tmp_path: Path) -> None:
    """Regression: an unwritten store reads as empty, never raises, never DDLs."""
    db = tmp_path / "fresh.sqlite"
    ledger = SqliteLedger(db)
    assert ledger.strategies_of("S1") == ()
    assert _tables(db) == set()  # a read must not create the schema


def test_strategies_of_reports_scope_rows_oldest_first(tmp_path: Path) -> None:
    ledger = SqliteLedger(tmp_path / "l.sqlite")
    ledger.ensure_strategy("hash-a", "S1", "S1", "paper")
    ledger.ensure_strategy("hash-b", "S1", "S1", "paper")
    ledger.ensure_strategy("hash-c", "S2", "S2", "paper")
    rows = ledger.strategies_of("S1")
    assert [r.strategy_id for r in rows] == ["hash-a", "hash-b"]
    assert all(r.scope == "S1" for r in rows)


def test_strategies_of_raises_on_a_shape_drifted_table(tmp_path: Path) -> None:
    """Only a MISSING table is empty; a drifted one must fail loudly."""
    from src.live.ledger import LedgerReadError

    db = tmp_path / "drift.sqlite"
    with get_connection(db) as con:
        con.execute("CREATE TABLE live_strategy (strategy_id TEXT, wrong INTEGER)")
    with pytest.raises(LedgerReadError):
        SqliteLedger(db).strategies_of("S1")


def test_scopes_of_store_is_empty_on_a_fresh_db(tmp_path: Path) -> None:
    """Regression: an unwritten store lists no scope and stays untouched."""
    db = tmp_path / "fresh.sqlite"
    assert SqliteLedger(db).scopes_of_store() == ()
    assert _tables(db) == set()


def test_scopes_of_store_sees_a_scope_with_only_executions(tmp_path: Path) -> None:
    """Regression: every scope-carrying table counts, not just strategy/cash."""
    ledger = SqliteLedger(tmp_path / "l.sqlite")
    ledger.ensure_strategy("hash-a", "S1", "S1", "paper")
    ledger.record_sim_open("S2", "AAPL_1")  # no strategy row, no cash row
    assert ledger.scopes_of_store() == ("S1", "S2")


def test_scopes_of_store_raises_on_a_shape_drifted_table(tmp_path: Path) -> None:
    """Only a MISSING table is skipped; a drifted one must fail loudly."""
    from src.live.ledger import LedgerReadError

    db = tmp_path / "drift.sqlite"
    with get_connection(db) as con:
        con.execute("CREATE TABLE live_cash (wrong INTEGER)")  # no ``scope`` column
    with pytest.raises(LedgerReadError):
        SqliteLedger(db).scopes_of_store()


def test_read_store_exposes_sim_lot_ids_not_just_a_count(tmp_path: Path) -> None:
    """Regression: the sim path writes NO book rows, so sim lots ARE ownership.

    Reporting only a count left the sim adapter comparing broker lots against an
    empty ``lots`` set, marking every owned fixture lot ``NOT-OURS``.
    """
    ledger = SqliteLedger(tmp_path / "l.sqlite")
    ledger.record_sim_open("S1", "AAPL_1")
    ledger.record_sim_open("S1", "TSLA_9")
    store = read_store(ledger, "S1")
    assert store.sim_open_ids == ("AAPL_1", "TSLA_9")
    assert store.lots == ()  # the sim path persists no conid book rows


# --- (b) read_store -----------------------------------------------------------


def test_read_store_reflects_cash_lots_and_strategy(tmp_path: Path) -> None:
    ledger = SqliteLedger(tmp_path / "l.sqlite")
    ledger.ensure_strategy("hash-a", "S1", "S1", "paper")
    ledger.ensure_cash("S1", 50000.0)
    key = IntentKey(scope="S1", symbol="AAPL", action=ActionType.long, position_id=None)
    ledger.save(_record(key))
    # A one-lot open booked directly through the public write path.
    execution = Execution(
        execution_id="e1",
        order_id="o1",
        order_ref=order_ref(key, 0),
        conid=265598,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=10.0,
        price=100.0,
        commission=1.0,
        ts=TS,
    )
    book, _ = reconcile("S1", (execution,), StrategyBook())
    ledger.save_book("S1", book, (execution,), 50000.0)

    store = read_store(ledger, "S1", initial_capital=50000.0)
    assert store.scope == "S1"
    assert store.db_path == str(tmp_path / "l.sqlite")
    assert store.initial_capital == 50000.0
    assert store.cash == 50000.0 - 1000.0 - 1.0
    assert len(store.strategy_rows) == 1
    assert store.strategy_rows[0].strategy_id == "hash-a"
    (lot,) = store.lots
    assert (lot.id, lot.symbol, lot.side, lot.qty) == ("265598", "AAPL", "long", 10.0)
    (intent,) = store.orders
    assert intent.key.symbol == "AAPL"
    assert len(store.trades) == 1
    assert store.trades[0].execution_id == "e1"


def _record(key: IntentKey):
    from src.live.identity import IntentRecord

    return IntentRecord(
        key=key,
        state=IntentState.PENDING,
        attempt=0,
        order_ref=order_ref(key, 0),
        order_id=None,
        decision_ts=TS,
    )


def test_read_store_on_a_fresh_ledger_falls_back_to_config(tmp_path: Path) -> None:
    ledger = SqliteLedger(tmp_path / "l.sqlite")
    store = read_store(ledger, "S1", initial_capital=50000.0)
    assert store.cash == 50000.0
    assert store.initial_capital == 50000.0
    assert store.lots == ()
    assert store.strategy_rows == ()


def test_a_sim_scope_reports_its_fills_cash_and_realized(tmp_path: Path) -> None:
    """A sim scope reads like a real one: fills, cash and P&L all present.

    Regression: the sim wrote no fills, so an open lot's own cost read as
    REALIZED profit and the scope's cash never moved below its seed.
    """
    ledger = SqliteLedger(tmp_path / "l.sqlite")
    ledger.ensure_cash("S1", 50000.0)
    ledger.record_sim_lot(
        "S1",
        SimLot(
            position_id="AAPL_1",
            symbol="AAPL",
            side="long",
            qty=10.0,
            entry_price=100.0,
            opened_at=TS,
            entry_commission=0.05,
        ),
    )

    store = read_store(ledger, "S1", initial_capital=50000.0)
    assert store.cash == 50000.0 - 1000.0 - 0.05
    assert [t.execution_id for t in store.trades] == ["AAPL_1:open"]
    (trade,) = store.trades
    assert (trade.symbol, trade.side, trade.conid) == ("AAPL", "BUY", None)
    assert trade.cash_delta == -1000.05
    # The open lot is not a result: the entry's own cost must not read as profit.
    assert scope_stats(store).realized_pnl == pytest.approx(0.0)
    assert scope_stats(store).commission == pytest.approx(0.05)

    ledger.mark_sim_closed("S1", "AAPL_1", TS, exit_price=110.0, commission=0.05)
    store = read_store(ledger, "S1", initial_capital=50000.0)
    assert store.cash == pytest.approx(50099.9)
    stats = scope_stats(store)
    assert stats.realized_pnl == pytest.approx(99.9)
    assert stats.trades == 2
    assert stats.wins == 1 and stats.losses == 0


def test_an_ownership_only_sim_lot_implies_no_fill(tmp_path: Path) -> None:
    """A row with no detail books NOTHING: an invented fill would be phantom P&L."""
    ledger = SqliteLedger(tmp_path / "l.sqlite")
    ledger.ensure_cash("S1", 500.0)
    ledger.record_sim_open("S1", "L1")
    store = read_store(ledger, "S1", initial_capital=500.0)
    assert store.trades == ()
    assert store.cash == 500.0
    assert scope_stats(store).realized_pnl == pytest.approx(0.0)


# --- (c) read_sim_broker ownership -------------------------------------------


def test_read_sim_broker_marks_owned_lots(tmp_path: Path) -> None:
    fixture = _write_fixture(
        tmp_path / "pf.json",
        {
            "cash": 40000.0,
            "positions": [
                {
                    "symbol": "AAPL",
                    "qty": 10,
                    "type": "long",
                    "entry_price": 100.0,
                    "position_id": "AAPL_1",
                    "last_price": 110.0,
                },
                {
                    "symbol": "TSLA",
                    "qty": 5,
                    "type": "short",
                    "entry_price": 200.0,
                    "position_id": "TSLA_9",
                    "last_price": 190.0,
                },
            ],
        },
    )
    cfg = _cfg(tmp_path, portfolio_path=fixture, mode="paper")
    side = _run(read_sim_broker(cfg, frozenset({"AAPL_1"})))
    assert side.adapter == "sim"
    assert side.cash == 40000.0
    owned = {lot.id: lot.owned for lot in side.positions}
    assert owned == {"AAPL_1": True, "TSLA_9": False}
    tsla = next(lot for lot in side.positions if lot.id == "TSLA_9")
    assert (tsla.side, tsla.qty) == ("short", 5.0)


def _run(result):
    """Drive an async builder to completion (tests run without an event loop)."""
    return asyncio.run(result)


# --- (d) render_pf -----------------------------------------------------------


def _broker() -> BrokerSide:
    """The broker side of a report fixture: one owned lot, one the store lacks."""
    return BrokerSide(
        adapter="sim",
        source="pf.json",
        account="",
        net_liquidation=None,
        cash=50000.0,
        positions=(
            BrokerLot("AAPL", "AAPL_1", 10.0, "long", 100.0, 110.0, 1100.0, True),
            BrokerLot("TSLA", "TSLA_9", 5.0, "short", 200.0, 190.0, 950.0, False),
        ),
        working_orders=(),
        ours_orders=(),
        warnings=(),
    )


def _store(scope: str = "S1") -> StoreSide:
    """Our store side of a report fixture: one open lot, one order, one fill."""
    key = IntentKey(
        scope=scope, symbol="AAPL", action=ActionType.long, position_id=None
    )
    return StoreSide(
        scope=scope,
        db_path="/tmp/l.sqlite",
        strategy_rows=(),
        cash=50000.0,
        initial_capital=50000.0,
        lots=(StoreLot("AAPL_1", "AAPL", "long", 10.0, 100.0, None, None, "", "r1"),),
        orders=(_record(key),),
        trades=(
            ExecutionRecord(
                scope=scope,
                execution_id="e1",
                conid=265598,
                symbol="AAPL",
                side="BUY",
                qty=10.0,
                price=100.0,
                commission=1.0,
                cash_delta=-1001.0,
                ts=TS,
            ),
        ),
        sim_open_ids=("AAPL_1",),
    )


def _report() -> PfReport:
    return PfReport(as_of=TS, stores=(_store(),), broker=_broker())


def _fill(
    symbol: str,
    side: str,
    qty: float,
    price: float,
    commission: float = 1.0,
) -> ExecutionRecord:
    """One stored fill with the cash delta its price and commission imply."""
    cost = qty * price
    delta = -(cost + commission) if side == "BUY" else cost - commission
    return ExecutionRecord(
        scope="S1",
        execution_id=f"e{side}{qty}",
        conid=1,
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        commission=commission,
        cash_delta=delta,
        ts=TS,
    )


def _side(
    lots: tuple[StoreLot, ...] = (),
    trades: tuple[ExecutionRecord, ...] = (),
    orders: tuple = (),
    cash: float = 50000.0,
) -> StoreSide:
    """A store side built by hand, so the P&L math is exercised without a ledger."""
    return StoreSide(
        scope="S1",
        db_path="/tmp/l.sqlite",
        strategy_rows=(),
        cash=cash,
        initial_capital=50000.0,
        lots=lots,
        orders=orders,
        trades=trades,
        sim_open_ids=(),
    )


def _lot(side: str = "long", qty: float = 10.0, entry: float = 100.0) -> StoreLot:
    return StoreLot("AAPL", "AAPL", side, qty, entry, None, None, "", "r1")


def _mark(symbol: str, price: float) -> BrokerSide:
    """A broker side carrying nothing but a last price for *symbol*."""
    return BrokerSide(
        adapter="sim",
        source="pf.json",
        account="",
        net_liquidation=None,
        cash=None,
        positions=(BrokerLot(symbol, symbol, 10.0, "long", 0.0, price, 0.0, True),),
        working_orders=(),
        ours_orders=(),
        warnings=(),
    )


def test_position_rows_round_trip_realizes_the_pnl() -> None:
    """Regression: a closed symbol's realized P&L is its sell cash minus its cost."""
    store = _side(
        trades=(
            _fill("AAPL", "BUY", 10.0, 100.0),
            _fill("AAPL", "SELL", 10.0, 110.0),
        )
    )
    (row,) = position_rows(store)
    assert (row.status, row.qty) == ("closed", 0.0)
    assert row.entry == 100.0 and row.last == 110.0  # entry avg, then exit avg
    assert row.realized == 98.0  # (1100 - 1) - (1000 + 1)


def test_open_lot_entry_commission_is_cost_not_loss() -> None:
    """Regression: an open lot's entry commission must not read as a realized loss."""
    store = _side(lots=(_lot(),), trades=(_fill("AAPL", "BUY", 10.0, 100.0),))
    (row,) = position_rows(store)
    assert row.realized == 0.0
    assert scope_stats(store).realized_pnl == 0.0


def test_open_short_marks_to_market_in_its_own_direction() -> None:
    """Regression: a short's open basis is cash RECEIVED, so a fall is profit."""
    store = _side(
        lots=(StoreLot("TSLA", "TSLA", "short", 10.0, 200.0, None, None, "", "r1"),),
        trades=(_fill("TSLA", "SELL", 10.0, 200.0),),
    )
    (row,) = position_rows(store, _mark("TSLA", 190.0))
    assert row.side == "short"
    assert row.realized == 0.0  # the proceeds are not a result yet
    assert row.unrealized == 100.0  # (190 - 200) * 10, short-signed


def test_unrealized_is_none_without_a_mark_never_a_guess() -> None:
    """A store-only read cannot know a mark: total P&L stays unknown, not partial."""
    store = _side(lots=(_lot(),), trades=(_fill("AAPL", "BUY", 10.0, 100.0),))
    (row,) = position_rows(store)
    assert row.unrealized is None
    stats = scope_stats(store)
    assert (stats.unrealized_pnl, stats.total_pnl, stats.total_return) == (
        None,
        None,
        None,
    )


def test_stats_tally_wins_and_losses_over_closed_symbols() -> None:
    """The win/loss tally counts CLOSED symbols only, by the sign of realized."""
    store = _side(
        lots=(_lot(),),
        trades=(
            _fill("AAPL", "BUY", 10.0, 100.0),
            _fill("MSFT", "BUY", 5.0, 50.0),
            _fill("MSFT", "SELL", 5.0, 60.0),
            _fill("NVDA", "BUY", 2.0, 30.0),
            _fill("NVDA", "SELL", 2.0, 20.0),
        ),
    )
    stats = scope_stats(store)
    assert (stats.wins, stats.losses) == (1, 1)
    assert stats.open_cost == 1000.0
    assert stats.commission == 5.0


def test_stats_totals_realized_when_nothing_is_open() -> None:
    """With no open lot a store-only read still states a total, from the fills."""
    store = _side(
        lots=(),
        trades=(
            _fill("AAPL", "BUY", 10.0, 100.0),
            _fill("AAPL", "SELL", 10.0, 110.0),
        ),
    )
    stats = scope_stats(store)
    assert (stats.unrealized_pnl, stats.total_pnl) == (0.0, 98.0)
    assert stats.wins == 1 and stats.losses == 0


def test_a_filled_intent_with_no_lot_is_still_a_row() -> None:
    """The merged view keeps the order trail: a symbol with no lot is not dropped."""
    key = IntentKey(scope="S1", symbol="AAPL", action=ActionType.long, position_id=None)
    store = _side(orders=(replace(_record(key), state=IntentState.FILLED),))
    (row,) = position_rows(store)
    assert (row.symbol, row.status, row.qty) == ("AAPL", "filled", 0.0)
    assert row.order_ref == order_ref(key, 0) and row.realized == 0.0


def test_render_pf_json_roundtrips() -> None:
    out = render_pf(_report(), "json")
    doc = json.loads(out)  # no unserializable object
    assert doc["broker"]["adapter"] == "sim"
    assert doc["stores"][0]["scope"] == "S1"
    # The raw per-store arrays stay for machine consumers...
    assert len(doc["stores"][0]["lots"]) == 1
    assert len(doc["stores"][0]["orders"]) == 1
    assert len(doc["stores"][0]["trades"]) == 1
    # ...alongside the same merged/derived views the text report renders.
    assert doc["stats"][0]["scope"] == "S1"
    (position,) = doc["positions"]
    assert position["symbol"] == "AAPL" and position["unrealized"] == 100.0
    assert doc["divergence"] == ["broker lot TSLA_9 TSLA not in our store"]


def test_render_pf_text_names_every_open_lot() -> None:
    """The merged view: one positions table, plus scopes and stats tables."""
    text = render_pf(_report(), "text")
    assert "broker:" in text and "scopes:" in text and "stats:" in text
    assert "broker positions:" in text and "positions:" in text
    for gone in ("lots:", "trades:", "stores:", "\norders:"):
        assert gone not in text  # the three redundant tables are one table now
    header = text.partition("\npositions:\n")[2].partition("\n")[0]
    assert header.split() == [
        "scope",
        "symbol",
        "side",
        "state",
        "qty",
        "entry",
        "last",
        "upnl",
        "rpnl",
        "sl",
        "tp",
        "order_id",
        "ref",
    ]
    row = text.partition("\npositions:\n")[2].splitlines()[2]
    assert row.split() == [
        "S1",
        "AAPL",
        "long",
        "open",
        "10",
        "100.0000",
        "110.0000",
        "100.00",
        "0.00",
        "-",
        "-",
        "-",
        "r1",
    ]
    assert "AAPL_1" in text and "TSLA_9" in text
    assert "ours" in text and "NOT-OURS" in text
    assert "divergence: broker lot TSLA_9 TSLA not in our store" in text
    assert "store lot" not in text  # AAPL_1 is present at both sides


def test_stats_table_carries_the_pnl_figures() -> None:
    """Regression: the report states P&L, not just counts.

    One lot, 10 @ 100 with a 1.00 commission, marked at 110: realized is 0 (the
    lot is still open, so its entry commission is cost, not a result) and the
    unrealized 100.00 comes from the broker's mark, making the total 100.00.
    """
    text = render_pf(_report(), "text")
    header = text.partition("\nstats:\n")[2].partition("\n")[0]
    assert header.split() == [
        "scope",
        "initial",
        "cash",
        "open_cost",
        "realized",
        "unreal",
        "total",
        "ret%",
        "comm",
        "lots",
        "trd",
        "sim",
        "win",
        "loss",
    ]
    row = text.partition("\nstats:\n")[2].splitlines()[2]
    assert row.split() == [
        "S1",
        "50000.00",
        "50000.00",
        "1000.00",
        "0.00",
        "100.00",
        "100.00",
        "0.20",
        "1.00",
        "1",
        "1",
        "1",
        "0",
        "0",
    ]


def test_render_pf_text_over_every_scope_has_no_broker_block() -> None:
    """Without a config the report is store-only: no broker block, no divergence."""
    text = render_pf(
        PfReport(as_of=TS, stores=(_store(), _store("S2")), broker=None), "text"
    )
    assert "broker:" not in text
    assert "scopes:" in text and "S1" in text and "S2" in text
    assert text.count("positions:") == 1  # one aggregated table over every scope
    assert "divergence: - (broker not read)" in text


def test_render_pf_text_surfaces_a_broker_read_failure() -> None:
    """A failed broker read degrades to warnings + an empty book, never a raise."""
    broken = replace(_broker(), positions=(), warnings=("bad_fixture: boom",))
    text = render_pf(PfReport(as_of=TS, stores=(_store(),), broker=broken), "text")
    assert "warning: bad_fixture: boom" in text
    assert "broker positions: none" in text


def test_render_pf_text_names_a_store_lot_the_broker_lacks() -> None:
    """A store lot the broker does not show is named, not silently dropped."""
    text = render_pf(
        PfReport(as_of=TS, stores=(_store(),), broker=replace(_broker(), positions=())),
        "text",
    )
    assert "divergence: store lot AAPL_1 AAPL not at the broker" in text


def test_a_sim_lot_dropped_from_the_fixture_is_named_and_still_shown() -> None:
    """The sim's OWN book is the store side, so a lot the fixture lost is visible.

    Regression: the sim store side used to be projected FROM the fixture, so a lot
    dropped from the file (or a wiped ledger) vanished instead of diverging.
    """
    lot = StoreLot(
        id="AAPL_1",
        symbol="AAPL",
        side="long",
        qty=10.0,
        entry_price=100.0,
        stop_loss=None,
        take_profit=None,
        tag="",
        order_ref="",
    )
    store = replace(_store(), lots=(), sim_open_ids=("AAPL_1",), sim_lots=(lot,))
    text = render_pf(
        PfReport(as_of=TS, stores=(store,), broker=replace(_broker(), positions=())),
        "text",
    )
    assert "divergence: store lot AAPL_1 AAPL not at the broker" in text
    row = next(
        line for line in text.splitlines() if line.startswith("S1") and "AAPL" in line
    )
    assert "open" in row and "10" in row


def test_ownership_only_sim_lot_still_diverges_by_id() -> None:
    """A sim row with no fill detail diverges by id, so a legacy row is not lost."""
    store = replace(_store(), lots=(), sim_open_ids=("LEGACY_1",), sim_lots=())
    text = render_pf(
        PfReport(as_of=TS, stores=(store,), broker=replace(_broker(), positions=())),
        "text",
    )
    assert "divergence: store lot LEGACY_1 - not at the broker" in text


def test_an_edited_fixture_cash_is_reported_as_a_divergence() -> None:
    """The sim broker's cash IS this scope's cash, so a mismatch is a divergence.

    Editing the fixture's ``cash`` is otherwise invisible: it moves sizing but
    would not show anywhere in the report.
    """
    text = render_pf(
        PfReport(
            as_of=TS,
            stores=(_store(),),
            broker=replace(_broker(), positions=(), cash=49000.0),
        ),
        "text",
    )
    assert "divergence: cash: broker 49000.00 vs store 50000.00 for S1" in text


def test_an_ibkr_scope_never_reports_a_cash_divergence() -> None:
    """The account's cash is shared by every strategy, so it is not comparable."""
    broker = _broker()
    text = render_pf(
        PfReport(
            as_of=TS,
            stores=(_store(),),
            broker=replace(
                broker, adapter="ibkr", positions=broker.positions[:1], cash=123.0
            ),
        ),
        "text",
    )
    assert "divergence: none" in text


def test_render_pf_text_reports_no_divergence() -> None:
    broker = _broker()
    text = render_pf(
        PfReport(
            as_of=TS,
            stores=(_store(),),
            broker=replace(broker, positions=broker.positions[:1]),
        ),
        "text",
    )
    assert "divergence: none" in text


# --- (e) CLI -----------------------------------------------------------------


def test_cli_pf_sim_renders_and_writes_no_ddl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a dry pf read must not create a schema (the store stays empty)."""
    db = tmp_path / "ledger.sqlite"
    fixture = _write_fixture(
        tmp_path / "pf.json",
        {
            "cash": 50000.0,
            "positions": [
                {
                    "symbol": "AAPL",
                    "qty": 10,
                    "type": "long",
                    "entry_price": 100.0,
                    "position_id": "AAPL_1",
                    "last_price": 110.0,
                }
            ],
        },
    )
    target = tmp_path / "cfg.json"
    target.write_text(
        json.dumps({**BASE_CONFIG, "portfolio_path": fixture, "mode": "paper"})
    )
    monkeypatch.setattr("src.live.cli.SqliteLedger", lambda *a, **k: SqliteLedger(db))

    out = CliRunner().invoke(live_group, ["pf", str(target), "--adapter", "sim"])
    assert out.exit_code == 0, out.output
    assert "broker:" in out.output and "scopes:" in out.output
    assert "NOT-OURS" in out.output
    assert _tables(db) == set()  # a read wrote no DDL

    as_json = CliRunner().invoke(
        live_group, ["pf", str(target), "--adapter", "sim", "-F", "json"]
    )
    assert as_json.exit_code == 0, as_json.output
    assert json.loads(as_json.output)["broker"]["adapter"] == "sim"


def test_cli_pf_sim_without_fixture_mints_one_like_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sim read with no portfolio_path mints ``pf_sim_<name>.json``, like ``run``.

    Regression: ``live pf`` used to warn and skip the broker block, so the same
    config rendered a broker for ``run`` but none for ``pf``.
    """
    target = tmp_path / "cfg.json"
    target.write_text(
        json.dumps({**BASE_CONFIG, "portfolio_path": "", "mode": "paper"})
    )
    monkeypatch.setattr(
        "src.live.cli.SqliteLedger", lambda *a, **k: SqliteLedger(tmp_path / "l.sqlite")
    )
    monkeypatch.setattr("src.live.cli.tempfile.gettempdir", lambda: str(tmp_path))
    out = CliRunner().invoke(live_group, ["pf", str(target), "--adapter", "sim"])
    assert out.exit_code == 0, out.output
    assert "scopes:" in out.output
    assert "broker read skipped" not in out.output
    assert "broker:" in out.output
    fixture = tmp_path / "pf_sim_pf_test.json"
    assert fixture_is_managed(fixture)
    assert json.loads(fixture.read_text())["cash"] == 50000.0


def test_store_and_broker_mints_the_watch_frame_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``--watch`` frame mints the same fixture, so a refresh reads a book."""
    monkeypatch.setattr("src.live.cli.tempfile.gettempdir", lambda: str(tmp_path))
    cfg = _cfg(tmp_path, portfolio_path="", mode="paper")
    ledger = SqliteLedger(tmp_path / "l.sqlite")
    stores, broker = _store_and_broker(ledger, cfg, "sim", BASE_CONFIG)
    assert len(stores) == 1
    assert broker is not None
    assert (broker.adapter, broker.cash) == ("sim", 50000.0)
    assert (tmp_path / "pf_sim_pf_test.json").exists()


def test_store_and_broker_without_a_config_stays_store_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No config, no name to mint from: the watch frame reads no broker."""
    monkeypatch.setattr(
        "src.live.cli.tempfile",
        type("T", (), {"gettempdir": staticmethod(lambda: str(tmp_path))}),
    )
    ledger = SqliteLedger(tmp_path / "l.sqlite")
    stores, broker = _store_and_broker(ledger, None, "sim")
    assert stores == ()
    assert broker is None


def test_cli_pf_sim_with_no_config_at_all_warns_and_stays_store_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no config there is no name to mint from: warn, stay store-only."""
    db = tmp_path / "ledger.sqlite"
    ledger = SqliteLedger(db)
    ledger.ensure_strategy("hash-a", "S1", "S1", "paper")
    monkeypatch.setattr("src.live.cli.SqliteLedger", lambda *a, **k: SqliteLedger(db))
    out = CliRunner().invoke(live_group, ["pf", "--adapter", "sim"])
    assert out.exit_code == 0, out.output
    assert "stats:" in out.output
    assert "broker:" in out.output and "no portfolio_path" in out.output


def test_cli_pf_ibkr_refuses_a_paper_config_on_a_live_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate mirrors ``run``: mode paper vs a live account is a hard refusal."""

    class _AccountClient:
        async def resolve_account(self) -> str:
            return "U12345"  # a LIVE account

    class _Gateway:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._client = _AccountClient()

        @property
        def client(self) -> _AccountClient:
            return self._client

        async def ensure_ready(self) -> object:
            raise AssertionError("must not probe readiness after an authz refusal")

        async def aclose(self) -> None:
            return None

    target = tmp_path / "cfg.json"
    target.write_text(json.dumps({**BASE_CONFIG, "mode": "paper"}))
    monkeypatch.setattr(
        "src.live.cli.SqliteLedger", lambda *a, **k: SqliteLedger(tmp_path / "l.sqlite")
    )
    monkeypatch.setattr("src.live.cli.IbkrGateway", _Gateway)

    out = CliRunner().invoke(live_group, ["pf", str(target), "--adapter", "ibkr"])
    assert out.exit_code == 1
    assert "refuses live account" in out.output


def test_cli_pf_without_config_lists_every_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: no CONFIG_PATH -> every store scope, and no broker read."""
    db = tmp_path / "ledger.sqlite"
    ledger = SqliteLedger(db)
    ledger.ensure_strategy("hash-a", "S1", "S1", "paper")
    ledger.ensure_cash("S1", 40000.0)
    ledger.ensure_strategy("hash-b", "S2", "S2", "paper")
    ledger.ensure_cash("S2", 10.0)
    monkeypatch.setattr("src.live.cli.SqliteLedger", lambda *a, **k: SqliteLedger(db))
    monkeypatch.setattr(
        "src.live.cli.IbkrGateway",
        lambda *a, **k: pytest.fail("no config must read no broker"),
    )

    out = CliRunner().invoke(live_group, ["pf"])
    assert out.exit_code == 0, out.output
    assert "S1" in out.output and "S2" in out.output
    assert "broker:" not in out.output
    assert "divergence: - (broker not read)" in out.output
    # With no config a sim read has no fixture: warn and stay store-only.
    scoped = CliRunner().invoke(live_group, ["pf", "--adapter", "sim"])
    assert scoped.exit_code == 0, scoped.output
    assert "stats:" in scoped.output
    assert "broker:" in scoped.output and "no portfolio_path" in scoped.output


def test_cli_pf_sim_marks_a_recorded_lot_ours(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a sim lot recorded by a cycle renders ``ours``, not ``NOT-OURS``.

    The sim broker's lots are keyed by the broker's synthetic ``position_id`` and
    ownership lives in ``live_sim_lot``; comparing against book rows (always empty
    on the sim path) marked every owned lot foreign.
    """
    db = tmp_path / "ledger.sqlite"
    ledger = SqliteLedger(db)
    ledger.record_sim_open("pf_test", "AAPL_1")
    fixture = _write_fixture(
        tmp_path / "pf.json",
        {
            "cash": 50000.0,
            "positions": [
                {
                    "symbol": "AAPL",
                    "qty": 10,
                    "type": "long",
                    "entry_price": 100.0,
                    "position_id": "AAPL_1",
                    "last_price": 110.0,
                },
                {
                    "symbol": "MSFT",
                    "qty": 3,
                    "type": "long",
                    "entry_price": 50.0,
                    "position_id": "EXOGENOUS_1",
                    "last_price": 55.0,
                },
            ],
        },
    )
    target = tmp_path / "cfg.json"
    target.write_text(
        json.dumps({**BASE_CONFIG, "portfolio_path": fixture, "mode": "paper"})
    )
    monkeypatch.setattr("src.live.cli.SqliteLedger", lambda *a, **k: SqliteLedger(db))

    out = CliRunner().invoke(live_group, ["pf", str(target), "--adapter", "sim"])
    assert out.exit_code == 0, out.output
    lines = out.output.splitlines()
    owned_row = next(line for line in lines if "AAPL_1" in line)
    exogenous_row = next(line for line in lines if "EXOGENOUS_1" in line)
    assert owned_row.split()[-1] == "ours"
    assert exogenous_row.split()[-1] == "NOT-OURS"
    # The exogenous fixture lot is the only divergence: AAPL_1 is ours.
    assert "AAPL_1" not in out.output.partition("divergence:")[2]


# --- ibkr broker read (stubbed client) ---------------------------------------


class _StubClient:
    """A minimal stand-in for the IBKR client: fixed raw responses, no HTTP."""

    def __init__(self) -> None:
        self.base_url = "https://localhost:5000/v1/api/"
        self.summary: dict[str, object] = {
            "netliquidation": {"amount": 12345.0, "currency": "USD"},
            "totalcashvalue": {"amount": 6789.0, "currency": "USD"},
        }
        # Two rows: one appears at the broker only (not in our store).
        self.positions: list[dict[str, object]] = [
            {
                "acctId": "DU1",
                "conid": 265598,
                "contractDesc": "AAPL",
                "position": 10,
                "avgCost": 100.0,
                "mktPrice": 110.0,
            },
            {
                "acctId": "DU1",
                "conid": 4815,
                "contractDesc": "MSFT",
                "position": -3,
                "avgCost": 50.0,
                "mktPrice": 55.0,
            },
        ]
        # The captured endpoint's own shape: our row carries a cOID in
        # ``order_ref``; another scope's row carries a different tag; an order
        # placed through the UI carries NO ``order_ref`` at all and is dropped.
        self.orders: list[dict[str, object]] = [
            {
                "order_ref": f"{scope_tag('S1')}-deadbeef-00",
                "orderId": 1,
                "conid": 265598,
                "ticker": "AAPL",
                "side": "BUY",
                "status": "Submitted",
                "filledQuantity": 4.0,
            },
            {
                "order_ref": f"{scope_tag('S2')}-cafebabe-00",
                "orderId": 2,
                "conid": 4815,
                "ticker": "MSFT",
                "side": "SELL",
                "status": "Submitted",
            },
            {
                "orderId": 3,
                "conid": 4815,
                "ticker": "MSFT",
                "side": "SELL",
                "status": "Cancelled",
            },
        ]

    async def portfolio_summary(self, account: str) -> dict[str, object]:
        return self.summary

    async def positions_all(self, account: str) -> list[dict[str, object]]:
        return self.positions

    async def open_orders(self) -> list[dict[str, object]]:
        return self.orders


def test_read_ibkr_broker_marks_owned_and_splits_orders(tmp_path: Path) -> None:
    """Ownership folds in by conid; our/foreign is decided by ``ref_is_ours``.

    The foreign row carries ANOTHER scope's tag (a different scope on a shared
    account) and a third row carries no ``order_ref`` at all — the captured
    endpoint's shape for an order placed through the UI, which is dropped rather
    than guessed at.
    """

    async def go() -> BrokerSide:
        return await read_ibkr_broker(
            cast("IbkrClient", _StubClient()), "DU1", ("S1",), frozenset({"265598"})
        )

    side = asyncio.run(go())
    assert side.adapter == "ibkr"
    assert side.account == "DU1"
    assert side.source == "https://localhost:5000/v1/api/"
    assert side.net_liquidation == 12345.0
    assert side.cash == 6789.0
    owned = {lot.id: lot.owned for lot in side.positions}
    assert owned == {"265598": True, "4815": False}
    assert [lot.side for lot in side.positions] == ["long", "short"]
    assert [o.order_ref for o in side.ours_orders] == [f"{scope_tag('S1')}-deadbeef-00"]
    assert len(side.working_orders) == 2  # the no-ref row is dropped, not shown


# --- (f) --watch polling refresh ---------------------------------------------


class _FakeSleeper:
    """A cadence seam that never sleeps: raises ``KeyboardInterrupt`` on the Nth tick."""

    def __init__(self, stop_after: int) -> None:
        self._stop_after = stop_after
        self.calls = 0
        self.intervals: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls += 1
        self.intervals.append(seconds)
        if self.calls >= self._stop_after:
            raise KeyboardInterrupt


def test_watch_loop_terminates_via_the_injected_sleeper() -> None:
    """Regression: the loop's cadence is the injected seam — Ctrl-C ends it, exit 0."""
    sleeper = _FakeSleeper(stop_after=3)
    frames: list[str] = []

    def frame() -> str:
        frames.append(f"frame-{len(frames)}")
        return frames[-1]

    out = io.StringIO()
    watch_pf_loop(frame, 2.5, tty=False, out=out, sleeper=sleeper)

    assert sleeper.calls == 3  # the fake sleeper is what stopped the loop
    assert sleeper.intervals == [2.5, 2.5, 2.5]
    assert frames == ["frame-0", "frame-1", "frame-2"]


def test_watch_loop_on_a_non_tty_emits_zero_escape_bytes() -> None:
    """Regression: a piped watch appends frames with a separator, never an ANSI byte."""
    sleeper = _FakeSleeper(stop_after=2)
    out = io.StringIO()
    watch_pf_loop(
        lambda: "as_of: 2024-06-03\nscopes:\n", 1.0, tty=False, out=out, sleeper=sleeper
    )

    text = out.getvalue()
    assert "\x1b" not in text  # not one escape byte into a cron log
    assert text == (
        "as_of: 2024-06-03\nscopes:\n\n" + "-" * 72 + "\nas_of: 2024-06-03\nscopes:\n\n"
    )


def test_watch_loop_on_a_tty_uses_and_restores_the_alternate_screen() -> None:
    """Regression: a TTY refresh hides the cursor, clears per tick and restores on exit."""
    sleeper = _FakeSleeper(stop_after=1)
    out = io.StringIO()
    watch_pf_loop(lambda: "frame", 1.0, tty=True, out=out, sleeper=sleeper)

    text = out.getvalue()
    assert text.startswith("\x1b[?1049h\x1b[?25l")  # enter alt screen, hide cursor
    assert text.endswith("\x1b[?25h\x1b[?1049l")  # show cursor, leave alt screen
    assert "\x1b[H\x1b[2J" in text  # cleared + homed per tick


def test_watch_loop_renders_a_failing_tick_and_keeps_looping() -> None:
    """Regression: a mid-watch read failure is an error line, never a dead loop."""
    sleeper = _FakeSleeper(stop_after=3)
    out = io.StringIO()
    seen: list[int] = []

    def frame() -> str:
        seen.append(len(seen))
        if len(seen) == 2:
            raise RuntimeError("broker hiccup")
        return f"tick-{len(seen)}"

    watch_pf_loop(frame, 1.0, tty=False, out=out, sleeper=sleeper)

    assert len(seen) == 3  # the loop survived the failing tick
    assert "tick-1" in out.getvalue()
    assert "error: broker hiccup" in out.getvalue()
    assert "tick-3" in out.getvalue()


def test_cli_watch_reuses_one_ledger_and_one_gateway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: N ticks build ONE ledger and gate ONE gateway (never per tick)."""
    db = tmp_path / "ledger.sqlite"
    ledger_builds: list[str] = []
    gateway_builds: list[int] = []

    def _ledger(*args: object, **kwargs: object) -> SqliteLedger:
        ledger_builds.append("built")
        return SqliteLedger(db)

    class _AccountClient:
        async def resolve_account(self) -> str:
            return "DU1234"

        async def aclose(self) -> None:
            return None

    class _Gateway:
        def __init__(self, *args: object, **kwargs: object) -> None:
            gateway_builds.append(1)
            self._client = _AccountClient()

        @property
        def client(self) -> _AccountClient:
            return self._client

        async def ensure_ready(self) -> object:
            return Ok(None)

        async def aclose(self) -> None:
            return None

    target = tmp_path / "cfg.json"
    target.write_text(json.dumps({**BASE_CONFIG, "mode": "paper"}))
    monkeypatch.setattr("src.live.cli.SqliteLedger", _ledger)
    monkeypatch.setattr("src.live.cli.IbkrGateway", _Gateway)
    monkeypatch.setattr("src.live.cli.time.sleep", _FakeSleeper(stop_after=3))

    out = CliRunner().invoke(
        live_group, ["pf", str(target), "--adapter", "ibkr", "--watch", "2"]
    )

    assert out.exit_code == 0, out.output
    assert len(ledger_builds) == 1  # ONE ledger across every tick
    assert len(gateway_builds) == 1  # ONE authenticated gateway across every tick
    assert _tables(db) == set()  # a watching read wrote no DDL


def test_cli_watch_rejects_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: an endless JSON document is refused, not looped."""
    monkeypatch.setattr(
        "src.live.cli.SqliteLedger", lambda *a, **k: SqliteLedger(tmp_path / "l.sqlite")
    )
    out = CliRunner().invoke(live_group, ["pf", "-F", "json", "--watch", "1"])
    assert out.exit_code == 1
    assert "--watch cannot be combined with --format json" in out.output


def test_cli_watch_requires_a_positive_interval() -> None:
    """Regression: ``--watch 0`` is a usage error (never a hot spin)."""
    out = CliRunner().invoke(live_group, ["pf", "--watch", "0"])
    assert out.exit_code == 2


def test_cli_one_shot_is_unchanged_without_watch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: no ``--watch`` -> today's one-shot path, byte-identical output."""
    db = tmp_path / "ledger.sqlite"
    ledger = SqliteLedger(db)
    ledger.ensure_strategy("hash-a", "S1", "S1", "paper")
    ledger.ensure_cash("S1", 40000.0)
    monkeypatch.setattr("src.live.cli.SqliteLedger", lambda *a, **k: SqliteLedger(db))

    out = CliRunner().invoke(live_group, ["pf"])
    assert out.exit_code == 0, out.output
    assert out.output.startswith("as_of: ")
    assert "scopes:" in out.output
    assert "\x1b" not in out.output
    assert not out.output.rstrip().endswith("-" * 72)  # no watch separator


def test_cli_watch_without_a_config_still_reads_the_ibkr_broker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a no-config ``--adapter ibkr`` watch reads the account, not the store alone.

    The one-shot path reads the broker whenever ``--adapter`` names one, config or
    not; a watch that silently narrowed to store-only would drop the divergence
    block the whole report exists to show.
    """
    db = tmp_path / "ledger.sqlite"
    gateway_builds: list[int] = []

    def _ledger(*args: object, **kwargs: object) -> SqliteLedger:
        return SqliteLedger(db)

    class _AccountClient:
        async def resolve_account(self) -> str:
            return "DU1234"

        async def aclose(self) -> None:
            return None

    class _Gateway:
        def __init__(self, *args: object, **kwargs: object) -> None:
            gateway_builds.append(1)
            self._client = _AccountClient()

        @property
        def client(self) -> _AccountClient:
            return self._client

        async def ensure_ready(self) -> object:
            return Ok(None)

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr("src.live.cli.SqliteLedger", _ledger)
    monkeypatch.setattr("src.live.cli.IbkrGateway", _Gateway)
    monkeypatch.setattr("src.live.cli.time.sleep", _FakeSleeper(stop_after=2))

    out = CliRunner().invoke(
        live_group, ["pf", "--adapter", "ibkr", "--allow-live", "--watch", "1"]
    )

    assert out.exit_code == 0, out.output
    assert len(gateway_builds) == 1  # the no-config watch gates ONE session
    assert _tables(db) == set()  # and still writes no DDL


# --- (g) styling: sparing, tty-gated, width-neutral ---------------------------


def test_render_pf_text_is_byte_clean_by_default() -> None:
    """Regression: styling is opt-in, so the plain render carries no escape byte."""
    text = render_pf(_report(), "text")
    assert "\x1b" not in text


def test_render_pf_json_ignores_a_styler() -> None:
    """Regression: `--format json` is parsed by a machine, so it is never styled."""
    out = render_pf(_report(), "json", COLOR)
    assert "\x1b" not in out
    assert json.loads(out)["positions"][0]["symbol"] == "AAPL"


def test_styling_is_sparing_and_leaves_the_layout_alone() -> None:
    """The coloured report is the plain one plus codes: same visible columns.

    Only a NAMED role is coloured (the block titles, the sign of a P&L, the
    state/owner markers), so the vast majority of cells carry no code at all.
    """
    plain = render_pf(_report(), "text")
    styled = render_pf(_report(), "text", COLOR)
    assert strip_ansi(styled) == plain
    assert [len(strip_ansi(line)) for line in styled.splitlines()] == [
        len(line) for line in plain.splitlines()
    ]
    coded = sum(1 for line in styled.splitlines() if "\x1b" in line)
    assert 0 < coded < len(styled.splitlines())  # some, not all


def test_a_losing_scopes_stats_read_red_and_a_winning_one_green() -> None:
    """The P&L columns carry the sign colour an operator scans for."""
    store = _side(
        lots=(_lot(),),
        trades=(
            _fill("AAPL", "BUY", 10.0, 100.0),
            _fill("MSFT", "BUY", 5.0, 50.0),
            _fill("MSFT", "SELL", 5.0, 60.0),
        ),
    )
    stats = scope_stats(store)
    assert stats.realized_pnl > 0  # MSFT round-tripped at a profit
    text = render_pf(PfReport(as_of=TS, stores=(store,)), "text", COLOR)
    assert "\x1b[32m" in text  # a green figure is present
    assert "\x1b[31m" not in text  # and nothing lost, so no red one
