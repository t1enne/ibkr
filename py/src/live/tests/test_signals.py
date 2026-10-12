"""Tests for the live signals bridge's book seed (the screen's initial state).

The screen runs a SHORT trailing window, so a strategy starts flat inside it and
a lot entered earlier is invisible — an exit conditioned on holding that lot
would never fire. ``book=`` seeds the run with the live book's lots so the
strategy's own exit rule sees them. This file pins that seed; the engine's
side of it (a seeded book lets a held lot exit, a flat one does not) is pinned
in ``src/bt/engine/tests/test_bt_engine.py``.
"""

from __future__ import annotations

from typing import cast

import pandas as pd

from src.bt.state import ActionType, PortfolioState, Position
from src.bt.types import StrategyConfig
from src.live.signals import _seed_state

TS = cast(pd.Timestamp, pd.Timestamp("2025-06-10"))


def _cfg() -> StrategyConfig:
    return StrategyConfig(
        name="test",
        strategy_type="momentum_compression_breakout_dsl",
        symbols=["AAPL", "MSFT"],
        initial_capital=10000.0,
        commission=0.5,
        warmup="250d",
        trading_start="2019-12-31",
        trading_end="2026-08-08",
        bars=["1d"],
        strategy_params={},
    )


def _book() -> PortfolioState:
    """A live book holding one old AAPL lot with its own cash, as read live."""
    lot = Position(
        symbol="AAPL",
        qty=10.0,
        entry_price=100.0,
        entry_time=TS,
        stop_loss=None,
        take_profit=None,
        last_price=102.0,
        type=ActionType.long,
        position_id="265598",
    )
    return PortfolioState(
        cash=1234.0,
        positions={"AAPL": (lot,)},
        trades=(),
        equity_curve=(),
        initial_capital=5000.0,
    )


def test_seed_state_carries_the_books_lots_and_cash():
    # The strategy must read the BOOK's lots and the BOOK's cash — an engine
    # default here would re-size or re-open against a number nobody holds.
    book = _book()
    state = _seed_state(book, _cfg())

    assert state.portfolio is book  # never re-derived / re-seeded from scratch
    assert state.portfolio.cash == 1234.0
    assert [p.position_id for p in state.portfolio.positions["AAPL"]] == ["265598"]
