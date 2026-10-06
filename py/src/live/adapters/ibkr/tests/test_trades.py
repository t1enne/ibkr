"""Per-scope book reconciliation: open/add/reduce/close/reopen/flip — all pure."""

from __future__ import annotations

from typing import cast

import pandas as pd

from src.exec.types import OrderSide
from src.live.adapters.ibkr.trades import Execution, StrategyBook, is_ours, reconcile

SCOPE = "momentum"


def _exec(
    order_id: str,
    *,
    side: OrderSide,
    qty: float,
    price: float,
    ref: str = f"{SCOPE}-20240102T093000-000",
    ts: str = "2024-01-02T09:30:00Z",
    commission: float = 1.0,
    symbol: str = "AAPL",
    conid: int = 265598,
    execution_id: str | None = None,
) -> Execution:
    return Execution(
        execution_id=execution_id or f"exec-{order_id}-{ts}",
        order_id=order_id,
        order_ref=ref,
        conid=conid,
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        commission=commission,
        ts=cast("pd.Timestamp", pd.Timestamp(ts)),
    )


def test_is_ours_uses_the_scope_slug_prefix() -> None:
    assert is_ours(SCOPE, _exec("o", side=OrderSide.BUY, qty=1, price=1))
    foreign = _exec(
        "o", side=OrderSide.BUY, qty=1, price=1, ref="other-20240102T093000-000"
    )
    assert not is_ours(SCOPE, foreign)


def test_is_ours_does_not_claim_a_longer_slug_scope() -> None:
    # Scope "momentum" must NOT absorb refs minted by "momentum-v2"/"momentum-2"
    # (slug keeps the dash): a prefix match let one scope's book absorb another's
    # lots (finding 3). The scope segment must match exactly.
    longer = _exec(
        "o",
        side=OrderSide.BUY,
        qty=1,
        price=1,
        ref="momentum-v2-20240102T093000-000",
    )
    assert not is_ours("momentum", longer)
    assert is_ours("momentum_v2", longer)  # slug("momentum_v2") == "momentum-v2"
    assert is_ours("momentum", _exec("o", side=OrderSide.BUY, qty=1, price=1))


def test_single_buy_opens_one_row() -> None:
    book, warnings = reconcile(
        SCOPE, (_exec("o1", side=OrderSide.BUY, qty=10, price=100),), StrategyBook()
    )
    assert warnings == ()
    (row,) = book.rows
    assert row.conid == 265598
    assert row.side == "long"
    assert row.qty == 10
    assert row.entry_price == 100
    assert row.is_open


def test_round_trip_leaves_one_closed_row_not_two_lots() -> None:
    executions = (
        _exec("o1", side=OrderSide.BUY, qty=10, price=100, ts="2024-01-02T09:30:00Z"),
        _exec("o2", side=OrderSide.SELL, qty=10, price=110, ts="2024-01-02T15:00:00Z"),
    )
    book, _ = reconcile(SCOPE, executions, StrategyBook())
    assert len(book.rows) == 1
    (row,) = book.rows
    assert row.qty == 0
    assert not row.is_open
    assert row.closed_at == pd.Timestamp("2024-01-02T15:00:00Z")
    assert row.entry_price == 100


def test_entry_price_is_vwap_of_the_open_interval() -> None:
    executions = (
        _exec("o1", side=OrderSide.BUY, qty=10, price=100, ts="2024-01-02T09:30:00Z"),
        _exec("o2", side=OrderSide.BUY, qty=10, price=120, ts="2024-01-02T09:40:00Z"),
        _exec("o3", side=OrderSide.SELL, qty=5, price=130, ts="2024-01-02T10:00:00Z"),
    )
    book, _ = reconcile(SCOPE, executions, StrategyBook())
    (row,) = book.rows
    assert row.qty == 15
    assert row.entry_price == 110.0  # (10*100 + 10*120) / 20


def test_reapplying_the_window_is_a_noop() -> None:
    executions = (_exec("o1", side=OrderSide.BUY, qty=10, price=100),)
    book, _ = reconcile(SCOPE, executions, StrategyBook())
    again, warnings = reconcile(SCOPE, executions, book)
    assert again == book
    assert warnings == ()


def test_reopen_after_flat_is_reported() -> None:
    executions = (
        _exec("o1", side=OrderSide.BUY, qty=10, price=100, ts="2024-01-02T09:30:00Z"),
        _exec("o2", side=OrderSide.SELL, qty=10, price=110, ts="2024-01-02T10:00:00Z"),
        _exec("o3", side=OrderSide.BUY, qty=7, price=120, ts="2024-01-02T11:00:00Z"),
    )
    book, warnings = reconcile(SCOPE, executions, StrategyBook())
    (row,) = book.rows
    assert row.is_open and row.qty == 7 and row.entry_price == 120
    assert row.opened_at == pd.Timestamp("2024-01-02T11:00:00Z")
    assert any("reopened" in w for w in warnings)


def test_flip_is_reported_and_resets_the_interval() -> None:
    executions = (
        _exec("o1", side=OrderSide.BUY, qty=10, price=100, ts="2024-01-02T09:30:00Z"),
        _exec("o2", side=OrderSide.SELL, qty=15, price=110, ts="2024-01-02T10:00:00Z"),
    )
    book, warnings = reconcile(SCOPE, executions, StrategyBook())
    (row,) = book.rows
    assert row.side == "short" and row.qty == 5 and row.entry_price == 110
    assert any("flip" in w for w in warnings)


def test_foreign_executions_are_ignored_without_warning() -> None:
    foreign = _exec(
        "o1",
        side=OrderSide.BUY,
        qty=99,
        price=1,
        ref="someone-else-20240102T093000-000",
    )
    book, warnings = reconcile(SCOPE, (foreign,), StrategyBook())
    assert book.rows == () and warnings == ()


def test_input_order_does_not_matter() -> None:
    a = _exec("o1", side=OrderSide.BUY, qty=10, price=100, ts="2024-01-02T09:30:00Z")
    b = _exec("o2", side=OrderSide.SELL, qty=10, price=110, ts="2024-01-02T10:00:00Z")
    assert reconcile(SCOPE, (a, b), StrategyBook()) == reconcile(
        SCOPE, (b, a), StrategyBook()
    )


def test_empty_scope_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="scope"):
        reconcile("", (), StrategyBook())


def test_separate_scopes_keep_separate_books() -> None:
    a = _exec(
        "o1", side=OrderSide.BUY, qty=10, price=100, ref="alpha-20240102T093000-000"
    )
    b = _exec("o2", side=OrderSide.BUY, qty=5, price=50, ref="beta-20240102T093000-000")
    abook, _ = reconcile("alpha", (a, b), StrategyBook())
    bbook, _ = reconcile("beta", (a, b), StrategyBook())
    assert [r.qty for r in abook.rows] == [10]
    assert [r.qty for r in bbook.rows] == [5]
