"""Critical-path tests for the side-by-side pf view.

``pf.py`` is a display layer, so this file keeps only the tests that pin a NUMBER
(cash, P&L, realized/unrealized) or a DIVERGENCE that would be wrong if the code
were broken. Rendering, layout and JSON-shape assertions are deliberately absent.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from src.bt.state import ActionType
from src.data.db import get_connection
from src.data.ibkr.client import IbkrClient, IbkrError
from src.exec.refs import scope_tag
from src.live.cli import load_live_config
from src.live.identity import IntentKey, IntentRecord, IntentState, order_ref
from src.live.ledger import ExecutionRecord, SimLot, SqliteLedger
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
        sim_open_ids=("AAPL_1",),
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


def test_render_pf_text_names_a_store_lot_the_broker_lacks() -> None:
    """A store lot the broker does not show is named, not silently dropped."""
    diverged = _divergence(
        PfReport(as_of=TS, stores=(_store(),), broker=replace(_broker(), positions=()))
    )
    assert "AAPL_1" in diverged and "AAPL" in diverged


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
    diverged = _divergence(
        PfReport(as_of=TS, stores=(store,), broker=replace(_broker(), positions=()))
    )
    assert "AAPL_1" in diverged and "AAPL" in diverged


def test_ownership_only_sim_lot_still_diverges_by_id() -> None:
    """A sim row with no fill detail diverges by id, so a legacy row is not lost."""
    store = replace(_store(), lots=(), sim_open_ids=("LEGACY_1",), sim_lots=())
    diverged = _divergence(
        PfReport(as_of=TS, stores=(store,), broker=replace(_broker(), positions=()))
    )
    assert "LEGACY_1" in diverged


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


def test_cli_pf_sim_marks_a_recorded_lot_ours(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a sim lot recorded by a cycle renders ``ours``, not ``NOT-OURS``.

    The sim broker's lots are keyed by the broker's synthetic ``position_id`` and
    ownership lives in the scope's account rows; comparing against book rows
    (always empty on the sim path) marked every owned lot foreign. The scope is
    the RESOLVED one (``<adapter>_<config_name>_<hash>``) — the same key a cycle
    writes — so a ``pf`` read and a ``run`` agree on which book they describe.
    """
    from src.live.cli import _config_scope
    from click.testing import CliRunner

    from src.live.cli import live_group

    db = tmp_path / "ledger.sqlite"
    scope = _config_scope(_cfg(tmp_path, mode="paper"))
    ledger = SqliteLedger(db)
    ledger.record_sim_lot(
        scope,
        SimLot(
            position_id="AAPL_1",
            symbol="AAPL",
            side="long",
            qty=10.0,
            entry_price=100.0,
        ),
    )
    target = tmp_path / "cfg.json"
    target.write_text(json.dumps({**BASE_CONFIG, "mode": "paper"}))
    monkeypatch.setattr("src.live.cli.SqliteLedger", lambda *a, **k: SqliteLedger(db))

    out = CliRunner().invoke(live_group, ["pf", str(target), "--adapter", "sim"])
    assert out.exit_code == 0, out.output
    owned_row = next(line for line in out.output.splitlines() if "AAPL_1" in line)
    assert owned_row.split()[-1] == "ours"
    # Ours on both sides: no divergence is reported for the lot.
    assert "AAPL_1" not in out.output.partition("divergence:")[2]


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


def test_read_ibkr_broker_marks_owned_and_splits_orders() -> None:
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
    owned = {lot.id: lot.owned for lot in side.positions}
    assert owned == {"265598": True, "4815": False}
    assert [o.order_ref for o in side.ours_orders] == [f"{scope_tag('S1')}-deadbeef-00"]
    assert len(side.working_orders) == 2  # the no-ref row is dropped, not shown


def test_read_ibkr_broker_warns_when_the_refresh_fails() -> None:
    """A failed cache discard is a warning — the cached book still beats no book."""

    class _Refusing(_StubClient):
        async def positions_invalidate(self, account: str) -> None:
            raise IbkrError("rate_limit", "positions/invalidate: 503", "positions")

    async def go() -> BrokerSide:
        return await read_ibkr_broker(
            cast("IbkrClient", _Refusing()), "DU1", ("S1",), frozenset()
        )

    side = asyncio.run(go())
    assert side.positions
    assert any("positions refresh" in warning for warning in side.warnings)


# --- (f) sim broker side reads the store's own account rows -------------------


def test_read_sim_broker_reads_the_scopes_account_rows(tmp_path: Path) -> None:
    """The sim broker side IS the scope's account rows + its own cash (no fixture)."""
    ledger = SqliteLedger(tmp_path / "sim.sqlite")
    ledger.ensure_cash("sim_t_x", 40000.0)
    ledger.record_sim_lot(
        "sim_t_x",
        SimLot(
            position_id="AAPL_1",
            symbol="AAPL",
            side="long",
            qty=10.0,
            entry_price=100.0,
        ),
    )
    ledger.record_sim_lot(
        "sim_t_x",
        SimLot(
            position_id="TSLA_9",
            symbol="TSLA",
            side="short",
            qty=5.0,
            entry_price=200.0,
        ),
    )
    side = read_sim_broker(ledger, "sim_t_x")
    assert side.cash == 40000.0
    assert {lot.id: lot.owned for lot in side.positions} == {
        "AAPL_1": True,
        "TSLA_9": True,  # a scope's own account rows are all ours
    }
    tsla = next(lot for lot in side.positions if lot.id == "TSLA_9")
    assert (tsla.side, tsla.qty) == ("short", 5.0)


def test_read_sim_broker_skips_an_ownership_only_row(tmp_path: Path) -> None:
    """A row with no detail names a lot we own but cannot describe: no phantom lot."""
    ledger = SqliteLedger(tmp_path / "sim.sqlite")
    ledger.record_sim_open("sim_t_x", "L1")
    side = read_sim_broker(ledger, "sim_t_x")
    assert side.positions == ()
