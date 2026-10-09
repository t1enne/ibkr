"""Critical-path tests for the side-by-side pf view.

``pf.py`` is a display layer, so this file keeps only the tests that pin a NUMBER
(cash, P&L, realized/unrealized) or a DIVERGENCE that would be wrong if the code
were broken. Rendering, layout and JSON-shape assertions are deliberately absent.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from src.bt.state import ActionType
from src.data.db import get_connection
from src.exec.refs import scope_tag
from src.live.cli import _config_report, load_live_config
from src.live.identity import IntentKey, IntentRecord, IntentState, order_ref
from src.live.ledger import ExecutionRecord, SqliteLedger
from src.live.pf import (
    BrokerLot,
    BrokerSide,
    PfReport,
    StoreLot,
    StoreSide,
    position_rows,
    render_pf,
    scope_stats,
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


_SCOPE = "pf_test"


# --- (a) reading the store: numbers, fail-loud reads, ownership ----------------


def test_strategies_of_raises_on_a_shape_drifted_table(tmp_path: Path) -> None:
    """Only a MISSING table is empty; a drifted one must fail loudly."""
    from src.live.ledger import LedgerReadError

    db = tmp_path / "drift.sqlite"
    with get_connection(db) as con:
        con.execute("CREATE TABLE live_strategy (strategy_id TEXT, wrong INTEGER)")
    with pytest.raises(LedgerReadError):
        SqliteLedger(db).strategies_of("S1")


def test_scopes_of_store_raises_on_a_shape_drifted_table(tmp_path: Path) -> None:
    """Only a MISSING table is skipped; a drifted one must fail loudly."""
    from src.live.ledger import LedgerReadError

    db = tmp_path / "drift.sqlite"
    with get_connection(db) as con:
        con.execute("CREATE TABLE live_cash (wrong INTEGER)")  # no ``scope`` column
    with pytest.raises(LedgerReadError):
        SqliteLedger(db).scopes_of_store()


# --- (b) money math: position_rows / scope_stats ------------------------------


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


# --- (c) divergence: the report's whole reason to exist -----------------------


def _record(key: IntentKey) -> IntentRecord:
    return IntentRecord(
        key=key,
        state=IntentState.PENDING,
        attempt=0,
        order_ref=order_ref(key, 0),
        order_id=None,
        decision_ts=TS,
    )


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
    )


def _divergence(report: PfReport) -> str:
    """The report's whole divergence block, so a wording edit stays out of the test."""
    text = render_pf(report, "text")
    return text.partition("\ndivergence:")[2].strip()


def test_render_pf_text_reports_no_divergence() -> None:
    """Agreement is SILENT: a matching book reports no divergence (INV-4)."""
    broker = _broker()
    assert (
        _divergence(
            PfReport(
                as_of=TS,
                stores=(_store(),),
                broker=replace(broker, positions=broker.positions[:1], cash=50000.0),
            )
        )
        == "none"
    )


def test_a_store_lot_the_broker_lacks_diverges_by_id() -> None:
    """A store lot the broker does not carry diverges by id, so it is not lost."""
    store = replace(
        _store(),
        lots=(StoreLot("LEGACY_1", "AAPL", "long", 10.0, 100.0, None, None, "", ""),),
    )
    diverged = _divergence(
        PfReport(as_of=TS, stores=(store,), broker=replace(_broker(), positions=()))
    )
    assert "LEGACY_1" in diverged


def test_a_qty_edited_broker_lot_diverges_by_size() -> None:
    """A lot both books hold at different sizes is a qty divergence, not silence."""
    broker = replace(
        _broker(),
        positions=(
            BrokerLot("AAPL", "AAPL_1", 99.0, "long", 100.0, 110.0, 1100.0, True),
        ),
    )
    diverged = _divergence(PfReport(as_of=TS, stores=(_store(),), broker=broker))
    assert "qty mismatch" in diverged and "99" in diverged


def test_an_edited_fixture_cash_is_reported_as_a_divergence() -> None:
    """The sim broker's cash IS this scope's cash, so a mismatch is a divergence."""
    diverged = _divergence(
        PfReport(
            as_of=TS,
            stores=(_store(),),
            broker=replace(_broker(), positions=(), cash=49000.0),
        )
    )
    assert "49000" in diverged and "50000" in diverged


def test_an_ibkr_scope_never_reports_a_cash_divergence() -> None:
    """The account's cash is shared by every strategy, so it is not comparable."""
    broker = _broker()
    diverged = _divergence(
        PfReport(
            as_of=TS,
            stores=(_store(),),
            broker=replace(
                broker, adapter="ibkr", positions=broker.positions[:1], cash=123.0
            ),
        )
    )
    assert diverged == "none"


# --- (d) CLI pf: the resolved scope is the book a run writes ------------------


def _cfg(tmp_path: Path, **live_keys: object) -> LiveConfig:
    target = tmp_path / "cfg.json"
    target.write_text(json.dumps({**BASE_CONFIG, **live_keys}))
    return load_live_config(str(target))


def test_pf_adapter_flag_selects_the_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``pf --adapter sim`` reads the SIM scope a ``run --adapter sim`` wrote.

    The bug: the flag only chose the broker to READ, not the scope, so a sim run
    followed by ``pf --adapter sim`` reported the config's default (ibkr) scope.
    Asserted on the scope handed to ``read_store`` — the only place the choice is
    observable before any broker read.
    """
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(BASE_CONFIG))  # no adapter key: defaults to ibkr
    seen: list[str] = []

    def fake_read(ledger: object, scope: str, **kw: object) -> StoreSide:
        seen.append(scope)
        return _store(scope=scope)

    monkeypatch.setattr("src.live.cli.read_store", fake_read)
    monkeypatch.setattr("src.live.cli._pf_broker", lambda *a, **k: None)
    ledger = cast("SqliteLedger", object())

    _config_report(ledger, str(path), None)  # no flag: the config's own scope
    _config_report(ledger, str(path), "sim")  # flag: the sim scope

    assert seen[0].startswith("ibkr_pf_test_")
    assert seen[1].startswith("sim_pf_test_")


# --- (e) ibkr broker read: ownership by our own ref prefix (INV-5) ------------


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
        # The captured endpoint's shape: our row carries a cOID in ``order_ref``;
        # another scope's row carries a different tag; an order placed through the
        # UI carries NO ``order_ref`` at all and is dropped.
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

    async def positions_invalidate(self, account: str) -> None:
        return None

    async def open_orders(self) -> list[dict[str, object]]:
        return self.orders


# --- (f) sim broker side reads the store's own account rows -------------------
