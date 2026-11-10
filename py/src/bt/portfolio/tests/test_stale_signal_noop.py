"""A stale signal for a vanished lot must be a no-op, never a raise.

A strategy emits a reduce/close while a lot is live; the fill settles in a
later cohort, by which time that lot may already be gone (it hit its intrabar
stop, or was closed earlier in the same cohort). The engine used to raise
``Position <pid> not found``, which aborted the whole backtest from a harmless
race — a real strategy run died this way.
"""

from __future__ import annotations

from typing import cast

import pandas as pd
import pytest

from src.bt.portfolio.pure import apply_fill
from src.bt.state import (
    ActionType,
    EquityPoint,
    FillEvent,
    PortfolioState,
    Position,
    Trade,
    TradeSignal,
    TradeStatus,
)

SYMBOL = "DXCM"
LIVE_PID = "DXCM_live"
STALE_PID = "DXCM_gone"


def _ts(val: str) -> pd.Timestamp:
    result = cast(pd.Timestamp, pd.Timestamp(val))
    assert not pd.isna(result)
    return result


def _portfolio() -> PortfolioState:
    """One open long lot, with the matching open trade record."""
    position = Position(
        symbol=SYMBOL,
        qty=10.0,
        entry_price=100.0,
        entry_time=_ts("2021-01-04"),
        stop_loss=95.0,
        take_profit=120.0,
        last_price=100.0,
        type=ActionType.long,
        position_id=LIVE_PID,
    )
    trade = Trade(
        entry_time=_ts("2021-01-04"),
        entry_price=100.0,
        exit_time=None,
        exit_price=None,
        last_price=100.0,
        reason="",
        symbol=SYMBOL,
        position=ActionType.long,
        qty=10.0,
        stop_loss=95.0,
        take_profit=120.0,
        pnl=0.0,
        status=TradeStatus.open,
        position_id=LIVE_PID,
    )
    return PortfolioState(
        cash=99_000.0,
        positions={SYMBOL: (position,)},
        trades=(trade,),
        equity_curve=(
            EquityPoint(
                timestamp=_ts("2021-01-04"),
                equity=100_000.0,
                cash=99_000.0,
                positions_value=1_000.0,
            ),
        ),
        initial_capital=100_000.0,
    )


@pytest.mark.parametrize(
    ("action", "qty"),
    [
        (ActionType.close, 10.0),
        (ActionType.rebalance, -5.0),
        (ActionType.rebalance, 5.0),
    ],
)
def test_stale_position_id_is_a_noop(action: ActionType, qty: float) -> None:
    """A signal targeting a departed lot leaves the book untouched."""
    before = _portfolio()
    fill = FillEvent(
        signal=TradeSignal(
            action=action,
            symbol=SYMBOL,
            timestamp=_ts("2021-01-05"),
            price=110.0,
            qty=qty,
            position_id=STALE_PID,
        ),
        filled_qty=abs(qty),
        executed_price=110.0,
        commission=0.0,
        slippage=0.0,
        timestamp=_ts("2021-01-05"),
    )

    after = apply_fill(before, fill)

    assert after == before, "a stale-target fill must not mutate the portfolio"


def test_live_position_id_still_closes() -> None:
    """The no-op guard is narrow: a live target still settles normally."""
    before = _portfolio()
    fill = FillEvent(
        signal=TradeSignal(
            action=ActionType.close,
            symbol=SYMBOL,
            timestamp=_ts("2021-01-05"),
            price=110.0,
            qty=10.0,
            position_id=LIVE_PID,
        ),
        filled_qty=10.0,
        executed_price=110.0,
        commission=1.0,
        slippage=0.0,
        timestamp=_ts("2021-01-05"),
    )

    after = apply_fill(before, fill)

    assert SYMBOL not in after.positions
    assert after.cash > before.cash
    assert after.trades[0].status == TradeStatus.closed
