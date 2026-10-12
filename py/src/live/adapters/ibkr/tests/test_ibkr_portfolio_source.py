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
from src.live.adapters.ibkr.mapping import IbkrPosition
from src.live.adapters.ibkr.portfolio_source import (
    IbkrPortfolioSource,
    build_snapshot,
)
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


def test_build_snapshot_warns_on_a_conid_we_hold_mismatching_the_account_net() -> None:
    """The account's own net is the read-only truth for a conid we hold.

    A disagreement is a WARNING, not a divergence: a shared account can carry a
    conid this scope does not own, so it must never fail a cycle. (Item 8: the
    fold-vs-own-rows oracle cannot see this, only the /positions net can.)
    """
    row = _row()  # AAPL conid 1, long 10 -> ours +10
    book = build_snapshot(
        (row,),
        marks={1: 100.0},
        cash=10000.0,
        initial_capital=50000.0,
        positions=(
            IbkrPosition(
                account="DU1", conid=1, symbol="AAPL", qty=4.0, avg_cost=100.0
            ),
        ),
        warnings=(),
        as_of=AS_OF,
    )
    assert any(
        "conid 1" in w and "ours +10" in w and "account +4" in w for w in book.warnings
    )


def test_build_snapshot_stays_silent_when_the_account_net_agrees() -> None:
    row = _row()  # ours +10
    book = build_snapshot(
        (row,),
        marks={1: 100.0},
        cash=10000.0,
        initial_capital=50000.0,
        positions=(
            IbkrPosition(
                account="DU1", conid=1, symbol="AAPL", qty=10.0, avg_cost=100.0
            ),
        ),
        warnings=(),
        as_of=AS_OF,
    )
    assert book.warnings == ()


def test_build_snapshot_ignores_a_position_no_scope_owns() -> None:
    """A conid only the account holds (another scope/a human) is never a mismatch.

    Our AAPL conid agrees with the account, so the ONLY foreign row is GOOG —
    ignored by design: this scope's book is the sizing authority, a position we
    do not own is not our disagreement to alarm on.
    """
    row = _row()  # ours AAPL conid 1, +10
    book = build_snapshot(
        (row,),
        marks={1: 100.0},
        cash=10000.0,
        initial_capital=50000.0,
        positions=(
            IbkrPosition(
                account="DU1", conid=1, symbol="AAPL", qty=10.0, avg_cost=100.0
            ),
            IbkrPosition(
                account="DU1", conid=99, symbol="GOOG", qty=20.0, avg_cost=500.0
            ),
        ),
        warnings=(),
        as_of=AS_OF,
    )
    assert book.warnings == ()


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
