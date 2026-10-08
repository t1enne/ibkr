"""IBKR portfolio source: per-scope book -> PortfolioState (account is display-only)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import cast

import httpx
import pandas as pd
import pytest

import respx

from src.data.ibkr.client import IbkrClient
from src.exec.refs import scope_tag
from src.exec.types import OrderSide
from src.live.adapters.ibkr.portfolio_source import IbkrPortfolioSource
from src.live.adapters.ibkr.trades import BookRow, Execution, StrategyBook, reconcile
from src.live.ledger import SqliteLedger

BASE = "https://localhost:5000/v1/api/"
SCOPE = "momentum"
AS_OF = cast("pd.Timestamp", pd.Timestamp("2024-01-03T00:00:00Z"))


def _row() -> BookRow:
    ex = Execution(
        execution_id="e1",
        order_id="o1",
        order_ref=f"{scope_tag(SCOPE)}-20240102T093000-000",
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


@respx.mock
@pytest.mark.asyncio
async def test_fetch_dry_run_writes_nothing(tmp_path: Path) -> None:
    _mock_reads(
        positions=[],
        trades=[
            {
                "execution_id": "e1",
                "order_id": "o1",
                "order_ref": f"{scope_tag(SCOPE)}-20240102T093000-000",
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
