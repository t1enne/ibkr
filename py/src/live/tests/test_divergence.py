"""Behaviour tests for the pure divergence oracle (no I/O, no mocks, no DB).

Each test drives a REAL book pair through ``guard_divergence``/``book_from_executions``
and asserts what the operator would see: a clean cycle, or the lot that disagrees.
"""

from __future__ import annotations

from typing import cast

import pandas as pd

from src.bt.state import ActionType, PortfolioState, Position
from src.live.divergence import (
    Divergence,
    book_from_executions,
    guard_divergence,
)
from src.live.ledger import ExecutionRecord

TS = cast(pd.Timestamp, pd.Timestamp("2025-01-02T15:00:00Z"))


def _lot(
    symbol: str = "AAPL",
    position_id: str = "265598",
    qty: float = 10.0,
    long: bool = True,
    entry_price: float = 100.0,
    entry_time: pd.Timestamp = TS,
) -> Position:
    return Position(
        symbol=symbol,
        qty=qty,
        entry_price=entry_price,
        entry_time=entry_time,
        stop_loss=None,
        take_profit=None,
        last_price=entry_price,
        type=ActionType.long if long else ActionType.short,
        position_id=position_id,
    )


def _book(*lots: Position) -> PortfolioState:
    grouped: dict[str, list[Position]] = {}
    for lot in lots:
        grouped.setdefault(lot.symbol, []).append(lot)
    return PortfolioState(
        cash=0.0,
        positions={sym: tuple(items) for sym, items in grouped.items()},
        trades=(),
        equity_curve=(),
        initial_capital=0.0,
    )


def _fill(
    symbol: str = "AAPL",
    position_id: str = "265598",
    side: str = "BUY",
    qty: float = 10.0,
    price: float = 100.0,
    ts: pd.Timestamp | None = TS,
    execution_id: str = "e1",
) -> ExecutionRecord:
    return ExecutionRecord(
        scope="ibkr_momentum_1a2b3c4d",
        execution_id=execution_id,
        conid=int(position_id) if position_id.isdigit() else None,
        position_id=position_id,
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        commission=0.0,
        cash_delta=-(qty * price) if side == "BUY" else qty * price,
        ts=ts,
    )


def test_identical_books_have_no_divergence() -> None:
    ours = _book(_lot(), _lot(symbol="MSFT", position_id="7", long=False))
    account = _book(_lot(), _lot(symbol="MSFT", position_id="7", long=False))
    assert guard_divergence(ours, account) == ()


def test_empty_books_have_no_divergence() -> None:
    assert guard_divergence(_book(), _book()) == ()


def test_account_qty_edit_is_one_divergence() -> None:
    ours = _book(_lot(qty=10.0))
    account = _book(_lot(qty=7.0))
    assert guard_divergence(ours, account) == (
        Divergence(
            symbol="AAPL",
            position_id="265598",
            ours_qty=10.0,
            account_qty=7.0,
            kind="qty_mismatch",
        ),
    )


def test_lot_only_the_account_holds_diverges() -> None:
    ours = _book()
    account = _book(_lot(position_id="999", qty=5.0))
    (divergence,) = guard_divergence(ours, account)
    assert divergence.kind == "missing_ours"
    assert divergence.position_id == "999"
    assert (divergence.ours_qty, divergence.account_qty) == (0.0, 5.0)


def test_lot_only_we_hold_diverges() -> None:
    ours = _book(_lot(position_id="265598", qty=4.0))
    account = _book()
    (divergence,) = guard_divergence(ours, account)
    assert divergence.kind == "missing_account"
    assert divergence.position_id == "265598"
    assert (divergence.ours_qty, divergence.account_qty) == (4.0, 0.0)


def test_book_from_executions_folds_open_and_close_to_nothing() -> None:
    book = book_from_executions(
        (
            _fill(side="BUY", qty=10.0, price=100.0, execution_id="e1"),
            _fill(
                side="SELL",
                qty=10.0,
                price=105.0,
                execution_id="e2",
                ts=cast("pd.Timestamp", TS + pd.Timedelta(hours=1)),
            ),
        )
    )
    assert book.positions == {}


def test_book_from_executions_of_nothing_is_empty() -> None:
    assert book_from_executions(()) == _book()
