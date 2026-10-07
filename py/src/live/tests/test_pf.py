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
import json
from pathlib import Path
from typing import cast

import pandas as pd
import pytest
from click.testing import CliRunner

from dataclasses import replace

from src.bt.state import ActionType
from src.data.db import get_connection
from src.data.ibkr.client import IbkrClient
from src.exec.types import OrderSide
from src.live.adapters.ibkr.trades import Execution, StrategyBook, reconcile
from src.exec.refs import scope_tag
from src.live.cli import live_group, load_live_config
from src.live.identity import IntentKey, IntentState, order_ref
from src.live.ledger import ExecutionRecord, SqliteLedger
from src.live.pf import (
    BrokerLot,
    BrokerSide,
    PfReport,
    StoreLot,
    StoreSide,
    read_ibkr_broker,
    read_sim_broker,
    read_store,
    render_pf,
)
from src.live.types import LiveConfig

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
        cash=40000.0,
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


def test_render_pf_json_roundtrips() -> None:
    out = render_pf(_report(), "json")
    doc = json.loads(out)  # no unserializable object
    assert doc["broker"]["adapter"] == "sim"
    assert doc["stores"][0]["scope"] == "S1"
    assert len(doc["stores"][0]["lots"]) == 1
    assert len(doc["stores"][0]["orders"]) == 1
    assert len(doc["stores"][0]["trades"]) == 1
    assert doc["divergence"] == ["broker lot TSLA_9 TSLA not in our store"]


def test_render_pf_text_names_every_open_lot() -> None:
    """Characterisation: positions and lots are TABLES, not one line per row."""
    text = render_pf(_report(), "text")
    assert "broker:" in text and "stores:" in text
    assert "positions:" in text and "lots:" in text
    assert "orders:" in text and "trades:" in text
    header = text.partition("positions:\n")[2].partition("\n")[0]
    assert header.split() == [
        "symbol",
        "id",
        "side",
        "qty",
        "avg",
        "last",
        "mktvalue",
        "owner",
    ]
    assert "AAPL_1" in text and "TSLA_9" in text
    assert "ours" in text and "NOT-OURS" in text
    assert "stores:" in text and "S1" in text
    assert "divergence: broker lot TSLA_9 TSLA not in our store" in text
    assert "store lot" not in text  # AAPL_1 is present at both sides


def test_render_pf_text_over_every_scope_has_no_broker_block() -> None:
    """Without a config the report is store-only: no broker block, no divergence."""
    text = render_pf(
        PfReport(as_of=TS, stores=(_store(), _store("S2")), broker=None), "text"
    )
    assert "broker:" not in text
    assert "stores:" in text and "S1" in text and "S2" in text
    assert text.count("lots:") == 1  # one aggregated table over every scope
    assert "divergence: - (broker not read)" in text


def test_render_pf_text_surfaces_a_broker_read_failure() -> None:
    """A failed broker read degrades to warnings + `positions: none`, never a raise."""
    broken = replace(_broker(), positions=(), warnings=("bad_fixture: boom",))
    text = render_pf(PfReport(as_of=TS, stores=(_store(),), broker=broken), "text")
    assert "warning: bad_fixture: boom" in text
    assert "positions: none" in text


def test_render_pf_text_names_a_store_lot_the_broker_lacks() -> None:
    """A store lot the broker does not show is named, not silently dropped."""
    text = render_pf(
        PfReport(as_of=TS, stores=(_store(),), broker=replace(_broker(), positions=())),
        "text",
    )
    assert "divergence: store lot AAPL_1 AAPL not at the broker" in text


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
    assert "broker:" in out.output and "stores:" in out.output
    assert "NOT-OURS" in out.output
    assert _tables(db) == set()  # a read wrote no DDL

    as_json = CliRunner().invoke(
        live_group, ["pf", str(target), "--adapter", "sim", "-F", "json"]
    )
    assert as_json.exit_code == 0, as_json.output
    assert json.loads(as_json.output)["broker"]["adapter"] == "sim"


def test_cli_pf_sim_without_fixture_warns_and_stays_store_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sim read with no portfolio_path degrades to a warning, not a usage error.

    The scope still renders from OUR store; only the broker block is skipped.
    """
    target = tmp_path / "cfg.json"
    target.write_text(
        json.dumps({**BASE_CONFIG, "portfolio_path": "", "mode": "paper"})
    )
    monkeypatch.setattr(
        "src.live.cli.SqliteLedger", lambda *a, **k: SqliteLedger(tmp_path / "l.sqlite")
    )
    out = CliRunner().invoke(live_group, ["pf", str(target), "--adapter", "sim"])
    assert out.exit_code == 0, out.output
    assert "stores:" in out.output
    assert "no portfolio_path" in out.output


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
    assert "stores:" in scoped.output
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
