"""Tests for the pure reconcile core (posture diff -> order intents)."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal, cast

import pandas as pd
import pytest

from src.bt.state import ActionType, PortfolioState, Position
from src.live.reconcile import (
    UnknownSignalSymbol,
    reconcile,
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
    bar_ts: pd.Timestamp | None = TS,
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
        bar_ts=bar_ts,
    )


def test_flip_closes_before_open() -> None:
    book = pf(100_000.0, lot("AAPL", 5.0, 100.0, ActionType.short, pid="S1"))
    first, second = reconcile((sig("long", qty=3.0),), book, CFG)
    assert first.action is ActionType.close
    assert first.position_id == "S1"
    assert second.action is ActionType.long
    assert second.reason == "open long (short->long)"


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


def test_nan_price_open_is_refused() -> None:
    # The real open path guards a non-finite price: ``_open_intent`` zeroes it, the
    # sizer then returns <= 0 and reconcile RAISES rather than place an order sized
    # by garbage.
    bad = replace(sig("long", qty=0.0), price=float("nan"))
    with pytest.raises(ValueError, match="unsized open AAPL"):
        reconcile((bad,), pf(100_000.0), _cfg(symbols=("AAPL",), size=0.5))


def test_unknown_symbol_refused() -> None:
    # A stray symbol is rejected by a typed error, NOT an assert (which -O strips).
    with pytest.raises(UnknownSignalSymbol, match="TSLA"):
        reconcile((sig("long", qty=1.0, symbol="TSLA"),), pf(100_000.0), CFG)


def test_a_partial_entry_is_not_topped_up() -> None:
    """Decision (A): a partial entry stands; the sizing target is never chased.

    The posture diff compares SIDES while ``size`` expresses a target weight, so a
    book holding a fraction of the sizer's ask still matches the target side and
    HOLDs. Topping up would re-size on every equity/price tick; instead the
    residual is left alone and the shortfall is reported on the order result.
    Under-filling errs toward LESS exposure than intended, which is the safe
    direction for a risk-sized strategy.
    """
    book = pf(100_000.0, lot("AAPL", 1.0, 90.0, ActionType.long, pid="L1"))
    assert reconcile((sig("long", qty=500.0),), book, CFG) == ()


def test_empty_owned_closes_nothing() -> None:
    book = pf(100_000.0, lot("AAPL", 10.0, 90.0, ActionType.long, pid="L1"))
    # Ownership scoped to the empty set: no lot is ours, so nothing closes.
    assert reconcile((sig("close"),), book, CFG, owned=frozenset()) == ()
