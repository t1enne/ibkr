"""Pure portfolio tests for ``apply_stop_updates`` — the stop_update arm path.

Critical paths only, each stated as a real regression:

- ratchet-or-refuse on BOTH sides (a silent widen would loosen a stop and pay
  a worse loss than authored — the exact bug the ratchet exists to block);
- per-lot targeting inside one symbol (a symbol-level clobber would re-arm a
  sibling lot's levels, so a multi-lot book's protection silently degrades);
- unknown lots and flat symbols skip silently (a stale arming order must not
  raise — a closed lot's ghost update would otherwise fail the whole bar);
- no cash/trades/equity mutation (an update that books any PnL or cash would
  double-count money — this is a lifecycle action, never a fill).
"""

from __future__ import annotations

from typing import cast

import pandas as pd

from src.bt.portfolio.pure import apply_stop_updates
from src.bt.state import (
    ActionType,
    EquityPoint,
    PortfolioState,
    Position,
    Trade,
    TradeSignal,
    TradeStatus,
)


def _ts(val: str) -> pd.Timestamp:
    """Cast a timestamp literal to the precise ``Timestamp`` type."""
    result = cast(pd.Timestamp, pd.Timestamp(val))
    assert not pd.isna(result)
    return result


def _lot(
    pid: str,
    symbol: str = "AAPL",
    side: ActionType = ActionType.long,
    sl: float | None = 90.0,
    tp: float | None = 110.0,
) -> Position:
    return Position(
        symbol=symbol,
        qty=10.0,
        entry_price=100.0,
        entry_time=_ts("2024-01-01"),
        stop_loss=sl,
        take_profit=tp,
        last_price=100.0,
        type=side,
        position_id=pid,
    )


def _signal(
    pid: str,
    symbol: str = "AAPL",
    sl: float | None = None,
    tp: float | None = None,
    action: ActionType = ActionType.stop_update,
) -> TradeSignal:
    return TradeSignal(
        action=action,
        symbol=symbol,
        timestamp=_ts("2024-01-02"),
        price=100.0,
        qty=0.0,
        stop_loss=sl,
        take_profit=tp,
        reason="arm",
        fill_at_next_open=False,
        position_side=ActionType.long,
        position_id=pid,
    )


def test_long_ratchet_tightens_refuses_widens_keeps_none() -> None:
    """Long: tighter SL/TP apply, wider are refused, None leaves the leg as-is.

    Regression: a widen that silently loosened a stop would let a position ride
    past the authored risk level; ``None`` re-arming must not wipe an existing
    level (that would disarm the position entirely).
    """
    portfolio = PortfolioState(
        cash=5000.0,
        positions={"AAPL": (_lot("L1"),)},
        trades=(),
        equity_curve=(),
        initial_capital=10000.0,
    )
    tight = apply_stop_updates(portfolio, (_signal("L1", sl=92.0, tp=108.0),))
    lot = tight.positions["AAPL"][0]
    assert lot.stop_loss == 92.0
    assert lot.take_profit == 108.0

    wide = apply_stop_updates(tight, (_signal("L1", sl=88.0, tp=112.0),))
    lot = wide.positions["AAPL"][0]
    assert lot.stop_loss == 92.0  # refuse the widen
    assert lot.take_profit == 108.0

    none_legs = apply_stop_updates(wide, (_signal("L1", sl=None, tp=None),))
    lot = none_legs.positions["AAPL"][0]
    assert lot.stop_loss == 92.0  # None = leave unchanged
    assert lot.take_profit == 108.0

    from_none = apply_stop_updates(
        PortfolioState(
            cash=5000.0,
            positions={"AAPL": (_lot("L2", sl=None, tp=None),)},
            trades=(),
            equity_curve=(),
            initial_capital=10000.0,
        ),
        (_signal("L2", sl=85.0, tp=115.0),),
    )
    lot = from_none.positions["AAPL"][0]
    assert lot.stop_loss == 85.0  # None old takes the new level outright
    assert lot.take_profit == 115.0


def test_short_ratchet_mirrors_long() -> None:
    """Short: stop ratchets DOWN, target UP — the mirror of a long.

    Regression: applying the long direction to a short would LOWER the short's
    protective stop (a stop moving toward price is a silent disarming).
    """
    portfolio = PortfolioState(
        cash=5000.0,
        positions={"AAPL": (_lot("S1", side=ActionType.short, sl=110.0, tp=90.0),)},
        trades=(),
        equity_curve=(),
        initial_capital=10000.0,
    )
    tight = apply_stop_updates(portfolio, (_signal("S1", sl=108.0, tp=92.0),))
    lot = tight.positions["AAPL"][0]
    assert lot.stop_loss == 108.0  # tightened (down)
    assert lot.take_profit == 92.0  # tightened (up)

    wide = apply_stop_updates(tight, (_signal("S1", sl=112.0, tp=88.0),))
    lot = wide.positions["AAPL"][0]
    assert lot.stop_loss == 108.0  # refused
    assert lot.take_profit == 92.0


def test_multi_lot_same_symbol_updates_only_the_named_lot() -> None:
    """Two lots on one symbol: the signal targets ONE position_id.

    Regression: a symbol-level update would clobber a sibling lot's tighter
    levels, so a position stacked across entries would silently lose its
    protection on every bar.
    """
    portfolio = PortfolioState(
        cash=5000.0,
        positions={
            "AAPL": (
                _lot("A1", sl=90.0, tp=110.0),
                _lot("A2", sl=85.0, tp=120.0),
            )
        },
        trades=(),
        equity_curve=(),
        initial_capital=10000.0,
    )
    updated = apply_stop_updates(portfolio, (_signal("A1", sl=93.0, tp=109.0),))
    a1, a2 = updated.positions["AAPL"]
    assert a1.stop_loss == 93.0 and a1.take_profit == 109.0
    assert a2.stop_loss == 85.0 and a2.take_profit == 120.0  # untouched


def test_unknown_position_id_and_flat_symbol_skip_silently() -> None:
    """Unknown position_id / flat symbol: signal dropped, never raises.

    Regression: a stale arming order for a lot the engine already closed must
    not abort the run (or corrupt the book) — the position is simply not
    re-armed. A non-stop_update signal in the tuple is a legal no-op too, so
    mixing batches never mis-routes.
    """
    portfolio = PortfolioState(
        cash=5000.0,
        positions={"AAPL": (_lot("L1"),)},
        trades=(),
        equity_curve=(),
        initial_capital=10000.0,
    )
    ghost = apply_stop_updates(portfolio, (_signal("GONE", sl=50.0),))
    assert ghost.positions is portfolio.positions
    assert ghost.positions["AAPL"][0].stop_loss == 90.0

    flat = apply_stop_updates(portfolio, (_signal("L1", symbol="MSFT", sl=50.0),))
    assert flat.positions is portfolio.positions

    mixed = apply_stop_updates(
        portfolio,
        (
            _signal("L1", sl=92.0),
            _signal("L1", sl=91.0, action=ActionType.close),
        ),
    )
    assert mixed.positions["AAPL"][0].stop_loss == 92.0  # close signal is a no-op


def test_stop_update_never_touches_cash_trades_or_equity() -> None:
    """Level updates leave cash, trades and equity_curve byte-identical.

    Regression: an update that booked cash/PnL would double-count money — the
    realized PnL belongs to the eventual close fill, not to re-arming a level.
    """
    trade = Trade(
        entry_time=_ts("2024-01-01"),
        entry_price=100.0,
        exit_time=None,
        exit_price=None,
        last_price=100.0,
        symbol="AAPL",
        position=ActionType.long,
        qty=10.0,
        stop_loss=90.0,
        take_profit=110.0,
        status=TradeStatus.open,
        position_id="L1",
    )
    point = EquityPoint(
        timestamp=_ts("2024-01-01"),
        equity=5000.0,
        cash=5000.0,
        positions_value=0.0,
    )
    portfolio = PortfolioState(
        cash=4000.0,
        positions={"AAPL": (_lot("L1"),), "MSFT": (_lot("M1", symbol="MSFT"),)},
        trades=(trade,),
        equity_curve=(point,),
        initial_capital=10000.0,
    )
    updated = apply_stop_updates(
        portfolio,
        (
            _signal("L1", sl=93.0, tp=109.0),
            _signal("M1", symbol="MSFT", sl=120.0, tp=180.0),
        ),
    )
    assert updated.cash == portfolio.cash
    assert updated.trades is portfolio.trades
    assert updated.equity_curve is portfolio.equity_curve
    assert updated.initial_capital == portfolio.initial_capital
