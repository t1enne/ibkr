"""Per-scope book reconciliation: open/add/reduce/close/reopen/flip — all pure."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pandas as pd

from src.exec.refs import scope_tag
from src.exec.types import OrderSide
from src.live.adapters.ibkr.trades import Execution, StrategyBook, is_ours, reconcile
from src.live.identity import ref_is_ours

SCOPE = "momentum"


def _ref(scope: str, ts: str = "20240102T093000", seq: int = 0) -> str:
    """A ref minted by *scope* (its non-collapsing ``scope_tag`` prefix)."""
    return f"{scope_tag(scope)}-{ts}-{seq:03d}"


def _exec(
    order_id: str,
    *,
    side: OrderSide,
    qty: float,
    price: float,
    ref: str = _ref(SCOPE),
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


def test_is_ours_attributes_the_new_bar_free_ref() -> None:
    # cOID = scope_tag-token-attempt. The token and attempt are dashless, so
    # rsplit("-", 2)[0] yields the whole tag (dashes included) and attribution
    # is unaffected by the identity change.
    assert is_ours(
        SCOPE,
        _exec(
            "o",
            side=OrderSide.BUY,
            qty=1,
            price=1,
            ref=f"{scope_tag(SCOPE)}-ffb76999-00",
        ),
    )
    dashed = _exec(
        "o",
        side=OrderSide.BUY,
        qty=1,
        price=1,
        ref=f"{scope_tag('momentum-v2')}-1a2b3c4d-01",
    )
    assert is_ours("momentum-v2", dashed)
    assert not is_ours("momentum", dashed)


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


def test_flip_is_reported_and_resets_the_interval() -> None:
    executions = (
        _exec("o1", side=OrderSide.BUY, qty=10, price=100, ts="2024-01-02T09:30:00Z"),
        _exec("o2", side=OrderSide.SELL, qty=15, price=110, ts="2024-01-02T10:00:00Z"),
    )
    book, warnings = reconcile(SCOPE, executions, StrategyBook())
    (row,) = book.rows
    assert row.side == "short" and row.qty == 5 and row.entry_price == 110
    assert any("flip" in w for w in warnings)


def test_empty_scope_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="scope"):
        reconcile("", (), StrategyBook())


def test_a_captured_pre_upgrade_order_ref_is_not_ours() -> None:
    """A ref from the OLD (pre-identity-layer) scheme is never considered ours.

    The captured trade carries ``511350df-7f3f2b38-20261005T1749-000`` — a shape
    the current identity layer cannot mint, so neither the trades attribution nor
    the identity helpers may claim it.
    """
    raw = json.loads(
        (Path(__file__).parent / "fixtures" / "gateway_trades.json").read_text()
    )
    ref = raw["trades"][0]["order_ref"]
    assert ref == "511350df-7f3f2b38-20261005T1749-000"  # the captured value
    assert not ref_is_ours(SCOPE, ref)
    assert not is_ours(SCOPE, _exec("o", side=OrderSide.BUY, qty=1, price=1, ref=ref))
