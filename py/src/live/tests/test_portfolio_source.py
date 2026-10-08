"""Tests for the pure fixture loader + the async mock source edge."""

from __future__ import annotations

import json
from typing import cast

import pandas as pd
import pytest

from src.bt.state import ActionType, PortfolioState, Position
from src.live.portfolio_source import (
    MockPortfolioSource,
    fixture_is_managed,
    load_mock_portfolio,
    write_mock_portfolio,
)
from src.live.result import Err, Ok
from src.live.types import FeedError, PortfolioSnapshot

# Annotated as the expanded union (not the ``Result`` alias): ty narrows
# ``isinstance(result, Err)`` on the concrete union but not through the alias.
FetchResult = Ok[PortfolioSnapshot, FeedError] | Err[PortfolioSnapshot, FeedError]

AS_OF: pd.Timestamp = cast(pd.Timestamp, pd.Timestamp("2024-06-03 16:00"))


def test_load_mock_portfolio_short_lot() -> None:
    """One short lot builds a real PortfolioState with the side on Position.type."""
    raw = {
        "cash": 1.0,
        "positions": [
            {
                "symbol": "AAPL",
                "qty": 2,
                "type": "short",
                "entry_price": 5,
                "position_id": "L1",
            }
        ],
    }
    snap = load_mock_portfolio(raw, AS_OF)

    assert isinstance(snap.portfolio, PortfolioState)
    assert snap.portfolio.cash == 1.0
    assert snap.portfolio.initial_capital == 1.0  # defaults to cash
    assert snap.portfolio.trades == ()
    assert snap.portfolio.equity_curve == ()

    (pos,) = snap.portfolio.positions["AAPL"]
    assert pos.symbol == "AAPL"
    assert pos.qty == 2.0  # stored positive
    assert pos.type is ActionType.short
    assert pos.entry_price == 5.0
    assert pos.last_price == 5.0  # defaults to entry_price
    assert pos.position_id == "L1"
    assert pos.entry_time == AS_OF  # defaults to as_of


def test_load_mock_portfolio_empty_positions() -> None:
    """No positions -> an empty book, not a crash."""
    snap = load_mock_portfolio({"cash": 100.0}, AS_OF)

    assert snap.portfolio.positions == {}
    assert snap.portfolio.cash == 100.0


def test_load_mock_portfolio_preserves_lot_order() -> None:
    """Multiple lots per symbol stay grouped in input order."""
    raw = {
        "cash": 10.0,
        "positions": [
            {
                "symbol": "AAPL",
                "qty": 1,
                "type": "long",
                "entry_price": 1,
                "position_id": "A",
            },
            {
                "symbol": "MSFT",
                "qty": 3,
                "type": "long",
                "entry_price": 2,
                "position_id": "B",
            },
            {
                "symbol": "AAPL",
                "qty": 2,
                "type": "long",
                "entry_price": 3,
                "position_id": "C",
            },
        ],
    }
    snap = load_mock_portfolio(raw, AS_OF)

    assert [p.position_id for p in snap.portfolio.positions["AAPL"]] == ["A", "C"]
    assert [p.position_id for p in snap.portfolio.positions["MSFT"]] == ["B"]


def test_load_mock_portfolio_rejects_bad_qty() -> None:
    raw = {
        "cash": 10.0,
        "positions": [{"symbol": "AAPL", "qty": 0, "type": "long", "entry_price": 1}],
    }
    with pytest.raises(ValueError):
        load_mock_portfolio(raw, AS_OF)


def test_load_mock_portfolio_rejects_bad_type() -> None:
    raw = {
        "cash": 10.0,
        "positions": [
            {"symbol": "AAPL", "qty": 1, "type": "sideways", "entry_price": 1}
        ],
    }
    with pytest.raises(ValueError):
        load_mock_portfolio(raw, AS_OF)


def test_load_mock_portfolio_rejects_negative_cash() -> None:
    with pytest.raises(ValueError):
        load_mock_portfolio({"cash": -1.0, "positions": []}, AS_OF)


def test_load_mock_portfolio_rejects_bad_timestamp() -> None:
    raw = {
        "cash": 10.0,
        "positions": [
            {
                "symbol": "AAPL",
                "qty": 1,
                "type": "long",
                "entry_price": 1,
                "entry_time": "not-a-date",
            }
        ],
    }
    with pytest.raises(ValueError):
        load_mock_portfolio(raw, AS_OF)


@pytest.mark.asyncio
async def test_mock_source_missing_path_returns_err() -> None:
    result: FetchResult = await MockPortfolioSource("/nonexistent/fixture.json").fetch()

    assert isinstance(result, Err)
    assert cast(FeedError, result.error).kind == "bad_fixture"


@pytest.mark.asyncio
async def test_mock_source_reads_fixture(tmp_path) -> None:
    path = tmp_path / "pf.json"
    path.write_text(
        json.dumps(
            {
                "cash": 5.0,
                "positions": [
                    {"symbol": "AAPL", "qty": 2, "type": "long", "entry_price": 3}
                ],
            }
        )
    )
    result: FetchResult = await MockPortfolioSource(str(path)).fetch()

    assert isinstance(result, Ok)
    snapshot = cast(PortfolioSnapshot, result.value)
    assert snapshot.portfolio.cash == 5.0
    assert snapshot.portfolio.positions["AAPL"][0].qty == 2.0


def test_write_then_load_round_trips_a_book(tmp_path) -> None:
    """A written fixture reads back as the same book, and is marked as OURS.

    Regression: the sim book used to have no writer at all, so the fixture it read
    could never carry a cycle's result forward.
    """
    path = tmp_path / "pf_sim.json"
    portfolio = PortfolioState(
        cash=1234.5,
        positions={
            "AAPL": (
                Position(
                    symbol="AAPL",
                    qty=3.0,
                    entry_price=10.0,
                    entry_time=AS_OF,
                    stop_loss=9.0,
                    take_profit=12.0,
                    last_price=11.0,
                    type=ActionType.short,
                    position_id="AAPL_7",
                    tag="v",
                ),
            )
        },
        trades=(),
        equity_curve=(),
        initial_capital=2000.0,
    )

    write_mock_portfolio(path, portfolio)
    assert fixture_is_managed(path)

    snapshot = load_mock_portfolio(json.loads(path.read_text()), AS_OF)
    assert snapshot.portfolio.cash == 1234.5
    assert snapshot.portfolio.initial_capital == 2000.0
    lot = snapshot.portfolio.positions["AAPL"][0]
    assert (lot.qty, lot.type, lot.position_id) == (3.0, ActionType.short, "AAPL_7")
    assert (lot.stop_loss, lot.take_profit, lot.tag) == (9.0, 12.0, "v")
    assert lot.entry_time == AS_OF


def test_fixture_is_managed_fails_closed(tmp_path) -> None:
    """Only a readable file carrying the marker is ours; anything else is not."""
    unmarked = tmp_path / "hand.json"
    unmarked.write_text(json.dumps({"cash": 1.0, "positions": []}))
    assert fixture_is_managed(unmarked) is False
    assert fixture_is_managed(tmp_path / "missing.json") is False
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert fixture_is_managed(broken) is False
