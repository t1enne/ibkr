"""IBKR portfolio source: per-scope book -> PortfolioState (account is display-only)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import cast

import httpx
import pandas as pd
import pytest

import respx

from src.bt.state import ActionType
from src.data.ibkr.client import IbkrClient
from src.exec.types import OrderSide
from src.live.adapters.ibkr.mapping import IbkrPosition
from src.live.adapters.ibkr.portfolio_source import IbkrPortfolioSource, build_snapshot
from src.live.adapters.ibkr.trades import BookRow, Execution, StrategyBook, reconcile
from src.live.ledger import SqliteLedger
from src.live.result import Ok
from src.live.types import PortfolioSnapshot

BASE = "https://localhost:5000/v1/api/"
SCOPE = "momentum"
AS_OF = cast("pd.Timestamp", pd.Timestamp("2024-01-03T00:00:00Z"))


def _row() -> BookRow:
    ex = Execution(
        execution_id="e1",
        order_id="o1",
        order_ref=f"{SCOPE}-20240102T093000-000",
        conid=1,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=10,
        price=100.0,
        commission=1.0,
        ts=cast("pd.Timestamp", pd.Timestamp("2024-01-02T09:30:00Z")),
    )
    book, _ = reconcile(SCOPE, (ex,), StrategyBook())
    (row,) = book.rows
    return row


def test_build_snapshot_maps_open_rows_to_positions() -> None:
    book = build_snapshot(
        (_row(),),
        {1: 105.0},
        cash=5000.0,
        initial_capital=51000.0,
        positions=(IbkrPosition("DU1", 1, "AAPL", 10.0, 0.0),),
        warnings=(),
        as_of=AS_OF,
    )
    portfolio = book.snapshot.portfolio
    assert portfolio.cash == 5000.0
    assert portfolio.initial_capital == 51000.0
    assert book.snapshot.as_of == AS_OF
    (position,) = portfolio.positions["AAPL"]
    assert position.type is ActionType.long
    assert position.qty == 10.0
    assert position.position_id == "1"  # conid IS the lot handle
    assert position.last_price == 105.0
    assert book.warnings == ()


def test_build_snapshot_ignores_a_position_we_do_not_own() -> None:
    # A foreign holding appears in the positions endpoint but not our book: no
    # warning, no absorption (plan §7.12).
    book = build_snapshot(
        (_row(),),
        {},
        5000.0,
        51000.0,
        (IbkrPosition("DU1", 999, "GOOG", 3.0, 0.0),),
        (),
        AS_OF,
    )
    assert "GOOG" not in book.snapshot.portfolio.positions
    assert book.warnings == ()


@respx.mock
@pytest.mark.asyncio
async def test_fetch_advances_the_scope_book(tmp_path: Path) -> None:
    _mock_reads(
        positions=[
            {"conid": 1, "contractDesc": "AAPL", "position": 15, "avgCost": 100.0}
        ],
        trades=[
            {
                "execution_id": "e1",
                "order_id": "o1",
                "order_ref": f"{SCOPE}-20240102T093000-000",
                "conid": 1,
                "symbol": "AAPL",
                "side": "B",
                "size": 10,
                "price": 100.0,
                "commission": 1.0,
                "trade_time_r": 1704191400000,
            }
        ],
        summary={
            "netliquidation": {"amount": 51000.0},
            "totalcashvalue": {"amount": 5000.0},
        },
    )
    ledger = SqliteLedger(tmp_path / "l.sqlite")
    source = IbkrPortfolioSource(
        IbkrClient(base_url=BASE, account="DU1"),
        scope=SCOPE,
        ledger=ledger,
        initial_capital=51000.0,
        now=AS_OF,
    )
    result = await source.fetch()
    assert isinstance(result, Ok)
    snapshot = cast("PortfolioSnapshot", result.value)
    (position,) = snapshot.portfolio.positions["AAPL"]
    assert position.qty == 10.0  # our book, not the account net 15
    # cash is derived, not the account summary's 5000
    assert snapshot.portfolio.cash == 51000.0 - (10 * 100.0 + 1.0)
    # persisted
    assert {r.conid for r in ledger.load_book(SCOPE).rows} == {1}


@respx.mock
@pytest.mark.asyncio
async def test_fetch_dry_run_writes_nothing(tmp_path: Path) -> None:
    _mock_reads(
        positions=[],
        trades=[
            {
                "execution_id": "e1",
                "order_id": "o1",
                "order_ref": f"{SCOPE}-20240102T093000-000",
                "conid": 1,
                "symbol": "AAPL",
                "side": "B",
                "size": 10,
                "price": 100.0,
                "commission": 1.0,
                "trade_time_r": 1704191400000,
            }
        ],
        summary={"netliquidation": {"amount": 0.0}, "totalcashvalue": {"amount": 0.0}},
    )
    ledger = SqliteLedger(tmp_path / "l.sqlite")
    source = IbkrPortfolioSource(
        IbkrClient(base_url=BASE, account="DU1"),
        scope=SCOPE,
        ledger=ledger,
        initial_capital=51000.0,
        dry_run=True,
        now=AS_OF,
    )
    await source.fetch()
    assert ledger.load_book(SCOPE).rows == ()
    with sqlite3.connect(tmp_path / "l.sqlite") as con:
        names = {
            r[0]
            for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert names == set()  # no DDL written on a dry run


@respx.mock
@pytest.mark.asyncio
async def test_fetch_transport_failure_is_err(tmp_path: Path) -> None:
    respx.get(f"{BASE}portfolio/DU1/summary").mock(
        side_effect=httpx.ConnectTimeout("down")
    )
    source = IbkrPortfolioSource(
        IbkrClient(base_url=BASE, account="DU1"),
        scope=SCOPE,
        ledger=SqliteLedger(tmp_path / "l.sqlite"),
        initial_capital=0.0,
        now=AS_OF,
    )
    result = await source.fetch()
    assert not isinstance(result, Ok)
    assert result.error.kind == "transport"


def _mock_reads(
    *, positions: list[object], trades: list[object], summary: dict[str, object]
) -> None:
    respx.get(f"{BASE}portfolio/DU1/summary").mock(
        return_value=httpx.Response(200, json=summary)
    )
    respx.get(f"{BASE}portfolio/DU1/positions/0").mock(
        return_value=httpx.Response(200, json=positions)
    )
    respx.get(f"{BASE}iserver/account/trades").mock(
        return_value=httpx.Response(200, json=trades)
    )


pytestmark = pytest.mark.db
