"""IBKR portfolio source: replay -> PortfolioState, cross-check warns, never aborts."""

from __future__ import annotations

import httpx
import pandas as pd
from typing import cast
import pytest
import respx

from src.bt.state import ActionType
from src.data.ibkr.client import IbkrClient
from src.exec.types import OrderSide
from src.live.adapters.ibkr.mapping import IbkrPosition, IbkrSummary
from src.live.adapters.ibkr.portfolio_source import (
    IbkrPortfolioSource,
    build_snapshot,
    cross_check,
    signed_net,
)
from src.live.adapters.ibkr.trades import Execution
from src.live.result import Ok
from src.live.types import PortfolioSnapshot

BASE = "https://localhost:5000/v1/api/"
PREFIX = "abc12345"
AS_OF = cast("pd.Timestamp", pd.Timestamp("2024-01-03T00:00:00Z"))


def _exec(order_id: str, side: OrderSide, qty: float, price: float) -> Execution:
    return Execution(
        execution_id=f"e-{order_id}-{side.value}",
        order_id=order_id,
        order_ref=f"{PREFIX}-deadbeef-20240102T0930-000",
        symbol="AAPL",
        side=side,
        qty=qty,
        price=price,
        commission=1.0,
        ts=cast("pd.Timestamp", pd.Timestamp("2024-01-02T09:30:00Z")),
    )


def _lot_executions() -> tuple[Execution, ...]:
    return (_exec("o1", OrderSide.BUY, 10, 100.0),)


def test_signed_net_buys_minus_sells() -> None:
    from src.live.adapters.ibkr.trades import replay

    lots = replay(
        (
            _exec("o1", OrderSide.BUY, 10, 100.0),
            _exec("o2", OrderSide.SELL, 4, 50.0),
        ),
        ref_prefix=PREFIX,
    ).lots
    # o2 is on the same symbol, so the symbol nets +6.
    assert signed_net(lots) == {"AAPL": 6.0}


def test_cross_check_reports_mismatch_with_execution_detail() -> None:
    from src.live.adapters.ibkr.trades import replay

    lots = replay(_lot_executions(), ref_prefix=PREFIX).lots
    warnings = cross_check(lots, (IbkrPosition("DU1", 1, "AAPL", 15.0, 0.0),))
    assert len(warnings) == 1
    assert "AAPL" in warnings[0]
    assert "replay=10" in warnings[0]
    assert "positions=15" in warnings[0]
    assert "o1/" in warnings[0]  # the execution detail is carried


def test_cross_check_silent_when_nets_agree() -> None:
    from src.live.adapters.ibkr.trades import replay

    lots = replay(_lot_executions(), ref_prefix=PREFIX).lots
    assert cross_check(lots, (IbkrPosition("DU1", 1, "AAPL", 10.0, 0.0),)) == ()


def test_build_snapshot_maps_open_lots_to_positions() -> None:
    from src.live.adapters.ibkr.trades import replay

    lots = replay(_lot_executions(), ref_prefix=PREFIX).lots
    book = build_snapshot(
        lots,
        IbkrSummary("DU1", net_liquidation=51000.0, total_cash=5000.0),
        (IbkrPosition("DU1", 1, "AAPL", 10.0, 0.0),),
        (),
        AS_OF,
    )
    portfolio = book.snapshot.portfolio
    assert portfolio.cash == 5000.0
    assert portfolio.initial_capital == 51000.0
    assert book.snapshot.as_of == AS_OF
    (position,) = portfolio.positions["AAPL"]
    assert position.type is ActionType.long
    assert position.qty == 10.0
    assert position.position_id == "o1"  # broker order id IS the lot handle
    assert book.warnings == ()


def test_build_snapshot_keeps_a_mismatch_a_warning() -> None:
    """A net disagreement warns with detail; the snapshot is still returned."""
    from src.live.adapters.ibkr.trades import replay

    lots = replay(_lot_executions(), ref_prefix=PREFIX).lots
    book = build_snapshot(
        lots,
        IbkrSummary("DU1", 51000.0, 5000.0),
        (IbkrPosition("DU1", 1, "AAPL", 99.0, 0.0),),
        (),
        AS_OF,
    )
    assert len(book.warnings) == 1
    assert book.snapshot.portfolio.positions["AAPL"][0].qty == 10.0  # replay wins


@respx.mock
@pytest.mark.asyncio
async def test_fetch_replays_and_cross_checks() -> None:
    respx.get(f"{BASE}portfolio/DU1/summary").mock(
        return_value=httpx.Response(
            200,
            json={
                "netliquidation": {"amount": 51000.0},
                "totalcashvalue": {"amount": 5000.0},
            },
        )
    )
    respx.get(f"{BASE}portfolio/DU1/positions/0").mock(
        return_value=httpx.Response(
            200,
            json=[
                {"conid": 1, "contractDesc": "AAPL", "position": 15, "avgCost": 100.0}
            ],
        )
    )
    respx.get(f"{BASE}iserver/account/trades").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "execution_id": "e1",
                    "order_id": "o1",
                    "order_ref": f"{PREFIX}-deadbeef-20240102T0930-000",
                    "symbol": "AAPL",
                    "side": "B",
                    "size": 10,
                    "price": 100.0,
                    "commission": 1.0,
                    "trade_time_r": 1704191400000,
                }
            ],
        )
    )
    source = IbkrPortfolioSource(
        IbkrClient(base_url=BASE, account="DU1"), PREFIX, now=AS_OF
    )
    result = await source.fetch()
    assert isinstance(result, Ok)
    snapshot = cast("PortfolioSnapshot", result.value)
    portfolio = snapshot.portfolio
    assert portfolio.cash == 5000.0
    assert portfolio.positions["AAPL"][0].qty == 10.0  # replay, not positions=15


@respx.mock
@pytest.mark.asyncio
async def test_fetch_transport_failure_is_err() -> None:
    respx.get(f"{BASE}portfolio/DU1/summary").mock(
        side_effect=httpx.ConnectTimeout("down")
    )
    source = IbkrPortfolioSource(
        IbkrClient(base_url=BASE, account="DU1"), PREFIX, now=AS_OF
    )
    result = await source.fetch()
    assert not isinstance(result, Ok)
    assert result.error.kind == "transport"
