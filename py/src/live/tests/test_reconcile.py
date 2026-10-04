"""Tests for the pure reconcile core (posture diff -> order intents)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import pandas as pd
import pytest

from src.bt.state import ActionType, PortfolioState, Position
from src.live.reconcile import (
    Side,
    current_side,
    reconcile,
    size_qty,
    target_side,
)
from src.live.types import LiveConfig, LiveSignal, PortfolioView, SignalAction

TS = cast("pd.Timestamp", pd.Timestamp("2024-06-03"))


def pf(cash: float, *lots: Position) -> PortfolioView:
    """A real ``PortfolioState`` seen through ``PortfolioView``.

    ``PortfolioState`` is a frozen dataclass whose fields are read-only, and
    ``PortfolioView`` declares its members as ``@property``, so the two match
    structurally with no cast.
    """
    grouped: dict[str, list[Position]] = {}
    for pos in lots:
        grouped.setdefault(pos.symbol, []).append(pos)
    state = PortfolioState(
        cash=cash,
        positions={sym: tuple(v) for sym, v in grouped.items()},
        trades=(),
        equity_curve=(),
        initial_capital=cash,
    )
    return state


def lot(
    symbol: str,
    qty: float,
    entry: float,
    side: ActionType,
    pid: str = "",
) -> Position:
    """One lot: qty positive, side on ``type``, ``last_price`` = entry."""
    return Position(
        symbol=symbol,
        qty=qty,
        entry_price=entry,
        entry_time=TS,
        stop_loss=None,
        take_profit=None,
        last_price=entry,
        type=side,
        position_id=pid,
    )


def _cfg(
    symbols: tuple[str, ...] = ("AAPL", "MSFT"),
    size: float = 0.0,
    size_mode: Literal["equity", "cash", "fixed"] = "equity",
) -> LiveConfig:
    return LiveConfig(
        strategy_type="momentum",
        symbols=symbols,
        initial_capital=100_000.0,
        strategy_params={},
        bars=("1d",),
        warmup="1y",
        size_mode=size_mode,
        size=size,
    )


CFG = _cfg()


def sig(
    action: SignalAction,
    qty: float = 0.0,
    symbol: str = "AAPL",
    pid: str | None = None,
) -> LiveSignal:
    return LiveSignal(
        symbol=symbol,
        action=action,
        score=1.0,
        reasons=(),
        signal_ts=TS,
        price=100.0,
        qty=qty,
        position_id=pid,
    )


def test_flat_to_long_opens() -> None:
    (order,) = reconcile((sig("long", qty=10.0),), pf(100_000.0), CFG)
    assert order.action is ActionType.long
    assert order.qty == 10.0
    assert order.position_id is None
    assert order.reason == "open long (flat->long)"


def test_long_to_close_closes_lot() -> None:
    book = pf(100_000.0, lot("AAPL", 10.0, 90.0, ActionType.long, pid="L1"))
    (order,) = reconcile((sig("close"),), book, CFG)
    assert order.action is ActionType.close
    assert order.qty == 10.0
    assert order.position_id == "L1"


def test_absent_signal_holds() -> None:
    book = pf(100_000.0, lot("AAPL", 10.0, 90.0, ActionType.long, pid="L1"))
    assert reconcile((), book, CFG) == ()


def test_flip_closes_before_open() -> None:
    book = pf(100_000.0, lot("AAPL", 5.0, 100.0, ActionType.short, pid="S1"))
    first, second = reconcile((sig("long", qty=3.0),), book, CFG)
    assert first.action is ActionType.close
    assert first.position_id == "S1"
    assert second.action is ActionType.long
    assert second.reason == "open long (short->long)"


def test_unsized_open_raises() -> None:
    with pytest.raises(ValueError, match="unsized open AAPL"):
        reconcile((sig("long", qty=0.0),), pf(100_000.0), CFG)


def test_sized_open_uses_config() -> None:
    cfg = _cfg(symbols=("AAPL",), size=0.5)
    book = pf(100_000.0)
    (order,) = reconcile((sig("long", qty=0.0),), book, cfg)
    assert order.qty == size_qty(100.0, book, cfg)
    assert order.qty == 500.0


def test_no_signal_leaves_side_unchanged() -> None:
    book = pf(100_000.0, lot("AAPL", 10.0, 90.0, ActionType.long, pid="L1"))
    assert current_side(book, "AAPL") == "long"
    intents = reconcile((sig("long", qty=1.0, symbol="MSFT"),), book, CFG)
    assert all(i.symbol == "MSFT" for i in intents)
    assert current_side(book, "AAPL") == "long"
    assert reconcile((), book, CFG) == ()


def test_determinism_lot_order_is_pinned() -> None:
    a = lot("AAPL", 10.0, 90.0, ActionType.long, pid="L1")
    b = lot("AAPL", 4.0, 95.0, ActionType.long, pid="L2")
    one = reconcile((sig("close"),), pf(100_000.0, a, b), CFG)
    two = reconcile((sig("close"),), pf(100_000.0, b, a), CFG)
    # Ordered tuple, not set: emission follows the BOOK's lot order, and the
    # same book yields the identical tuple every time (fully deterministic).
    assert tuple(i.position_id for i in one) == ("L1", "L2")
    assert tuple(i.position_id for i in two) == ("L2", "L1")
    assert one == reconcile((sig("close"),), pf(100_000.0, a, b), CFG)


def test_lot_without_broker_id_produces_no_close() -> None:
    book = pf(100_000.0, lot("AAPL", 10.0, 90.0, ActionType.long, pid=""))
    assert reconcile((sig("close"),), book, CFG) == ()


def test_flip_sizes_open_against_freed_cash() -> None:
    cfg = _cfg(symbols=("AAPL",), size=0.5)
    book = pf(0.0, lot("AAPL", 10.0, 100.0, ActionType.short, pid="S1"))
    close, open_ = reconcile((sig("long", qty=0.0),), book, cfg)
    assert close.action is ActionType.close
    assert close.position_id == "S1"
    assert open_.action is ActionType.long
    # Opens are sized against the ACTUAL post-close book: the close is priced
    # with the shared execute_signal and settled through apply_fills, so the
    # short lot is gone and cash is its real proceeds. The short close is a
    # BUY-to-cover, so friction leans up: 100 + 0.025 (half-spread) + 0.02
    # (slip) = 100.045. Short settlement: 10*100 + (100-100.045)*10 - 0.5
    # (commission) = 1000 - 0.45 - 0.5 = 999.05. size 0.5 -> 999.05*0.5/100
    # = 4.99525 -> 4.9952 (4 dp). Double-counting the lot would give 10.
    assert open_.qty == 4.9952


def test_size_qty_nan_price_is_zero() -> None:
    cfg = _cfg(symbols=("AAPL",), size=0.5)
    assert size_qty(float("nan"), pf(100_000.0), cfg) == 0.0


def test_owned_filter_excludes_foreign_lot() -> None:
    book = pf(
        100_000.0,
        lot("AAPL", 10.0, 90.0, ActionType.long, pid="L1"),
        lot("AAPL", 4.0, 95.0, ActionType.long, pid="OTHER"),
    )
    (order,) = reconcile((sig("close"),), book, CFG, owned=frozenset({"L1"}))
    assert order.position_id == "L1"


def test_unknown_symbol_asserts() -> None:
    with pytest.raises(AssertionError, match="TSLA"):
        reconcile((sig("long", qty=1.0, symbol="TSLA"),), pf(100_000.0), CFG)


def test_reconcile_accepts_structural_portfolio_view() -> None:
    @dataclass(frozen=True)
    class StandIn:
        cash: float
        positions: dict[str, tuple[Position, ...]]
        initial_capital: float

    held = lot("AAPL", 10.0, 90.0, ActionType.long, pid="L1")
    # Structural stand-in: same three members, none of PortfolioState's extras.
    stand_in: PortfolioView = StandIn(
        cash=1_000.0, positions={"AAPL": (held,)}, initial_capital=1_000.0
    )
    (order,) = reconcile((sig("close"),), stand_in, CFG)
    assert order.action is ActionType.close
    assert order.qty == 10.0

    # And a real PortfolioState works through the same code path.
    (from_state,) = reconcile((sig("close"),), pf(1_000.0, held), CFG)
    assert from_state == order


def test_side_helpers() -> None:
    short_book = pf(100.0, lot("AAPL", 3.0, 9.0, ActionType.short, pid="S1"))
    assert current_side(short_book, "AAPL") == "short"
    assert current_side(pf(100.0), "AAPL") == "flat"
    assert target_side(sig("close")) == "flat"
    assert target_side(sig("long")) == "long"
    assert target_side(sig("short")) == "short"
    _side: Side = current_side(pf(100.0), "AAPL")
    assert _side == "flat"


def test_close_signal_targets_named_lot_only() -> None:
    book = pf(
        100_000.0,
        lot("AAPL", 10.0, 90.0, ActionType.long, pid="L1"),
        lot("AAPL", 4.0, 95.0, ActionType.long, pid="L2"),
    )
    (order,) = reconcile((sig("close", pid="L2"),), book, CFG)
    assert order.position_id == "L2"
    assert order.qty == 4.0


def test_long_to_long_holds() -> None:
    book = pf(100_000.0, lot("AAPL", 10.0, 90.0, ActionType.long, pid="L1"))
    # Already long and asked for long: side-only reconcile HOLDs (no resize).
    assert reconcile((sig("long", qty=5.0),), book, CFG) == ()


def test_short_to_short_holds() -> None:
    book = pf(100_000.0, lot("AAPL", 10.0, 90.0, ActionType.short, pid="S1"))
    assert reconcile((sig("short", qty=5.0),), book, CFG) == ()


def test_empty_owned_closes_nothing() -> None:
    book = pf(100_000.0, lot("AAPL", 10.0, 90.0, ActionType.long, pid="L1"))
    # Ownership scoped to the empty set: no lot is ours, so nothing closes.
    assert reconcile((sig("close"),), book, CFG, owned=frozenset()) == ()


def test_foreign_only_book_opposite_open_holds() -> None:
    # The whole book on the symbol is foreign (not in ``owned``): a close is
    # not ours to emit, so the opposite-side open must be skipped (HOLD) rather
    # than doubling gross exposure without reaching the target posture.
    book = pf(100_000.0, lot("AAPL", 10.0, 90.0, ActionType.short, pid="OTHER"))
    assert reconcile((sig("long", qty=5.0),), book, CFG, owned=frozenset({"L1"})) == ()
    # Same posture with no scoping: the lot is ours, so the flip proceeds.
    flipped = reconcile((sig("long", qty=5.0),), book, CFG)
    assert [i.action for i in flipped] == [ActionType.close, ActionType.long]
