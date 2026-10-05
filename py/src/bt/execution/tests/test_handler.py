"""Tests for execution handler — critical path: long entry pays spread."""

from src.bt.execution import execute_signal, create_execution_params
from src.bt.state import TradeSignal, Candle, ActionType
from src.utils import get_ts


def test_long_entry_pays_spread():
    params = create_execution_params(
        spread_bps=10.0, slippage_bps=2.0, fixed_commission=0.5
    )
    signal = TradeSignal(
        action=ActionType.long,
        symbol="AAPL",
        z_score=2.0,
        timestamp=get_ts("2025-01-01"),
        price=100.0,
    )
    candle = Candle(
        timestamp=get_ts("2025-01-01"),
        symbol="AAPL",
        open=99.0,
        high=101.0,
        low=98.0,
        close=100.5,
        volume=1000,
    )
    fill = execute_signal(signal, candle, params)
    assert fill.executed_price > candle.open  # pays spread
    assert fill.commission == 0.5


# Priority 2 — guarded stop-close fill ("./research/bear_breakout_review.md").
#
# The manual ATR stop used to emit a plain next-open close, so a stop that
# triggered intra-bar filled at the *following day's open* — higher than the
# stop level for a long (misses the intended intra-bar exit) and never
# worse-of a gap through the level (understates tail risk). A close signal with
# ``fill_guard_price`` now fills at the adverse worse-of (stop, next open),
# mirroring ``execute_risk_event`` gap-through math.


def _close_signal(guard: float | None = None, is_long: bool | None = None):
    return TradeSignal(
        action=ActionType.close,
        symbol="GLD",
        timestamp=get_ts("2025-01-01"),
        price=180.0,
        fill_at_next_open=True,
        fill_guard_price=guard,
        fill_guard_is_long=is_long,
    )


def _candle(open_: float) -> Candle:
    return Candle(
        timestamp=get_ts("2025-01-02"),
        symbol="GLD",
        open=open_,
        high=open_ + 2.0,
        low=open_ - 2.0,
        close=open_ + 1.0,
        volume=1000,
    )


def test_guarded_long_stop_close_fills_worse_of_stop_and_open():
    params = create_execution_params(spread_bps=0.0, slippage_bps=0.0)
    # Long stop 185.0; next open 186.0 (higher). The stop fired intra-bar at
    # 185.0, so the fill must be the stop level, NOT the higher next open — the
    # delay must not improve the exit.
    fill = execute_signal(
        _close_signal(guard=185.0, is_long=True), _candle(186.0), params
    )
    assert fill.executed_price == 185.0

    # Long stop 185.0; next open 181.0 (gap DOWN through the stop). The gap loss
    # must be realized at the open, not softened back to the stop.
    fill = execute_signal(
        _close_signal(guard=185.0, is_long=True), _candle(181.0), params
    )
    assert fill.executed_price == 181.0


def test_guarded_short_stop_close_fills_worse_of_stop_and_open():
    params = create_execution_params(spread_bps=0.0, slippage_bps=0.0)
    # Short stop 120.0; next open 119.0 (lower). Short covers by buying; filling
    # at 119.0 would improve on the 120.0 stop touch — must fill at the stop.
    fill = execute_signal(
        _close_signal(guard=120.0, is_long=False), _candle(119.0), params
    )
    assert fill.executed_price == 120.0

    # Short stop 120.0; next open 123.0 (gap UP through the stop). Buy-back must
    # realize the gap loss at 123.0.
    fill = execute_signal(
        _close_signal(guard=120.0, is_long=False), _candle(123.0), params
    )
    assert fill.executed_price == 123.0


def test_un_guarded_close_still_fills_at_next_open():
    params = create_execution_params(spread_bps=0.0, slippage_bps=0.0)
    fill = execute_signal(_close_signal(guard=None), _candle(188.0), params)
    assert fill.executed_price == 188.0


def test_guarded_same_bar_close_ignores_guard():
    # Guard only applies to next-open fills; a same-bar close uses signal.price.
    params = create_execution_params(spread_bps=0.0, slippage_bps=0.0)
    sig = TradeSignal(
        action=ActionType.close,
        symbol="GLD",
        timestamp=get_ts("2025-01-01"),
        price=187.0,
        fill_at_next_open=False,
        fill_guard_price=189.0,
        fill_guard_is_long=True,
    )
    fill = execute_signal(sig, _candle(999.0), params)
    assert fill.executed_price == 187.0


def test_commission_models_re_exported_from_state_are_the_shared_ones():
    # ``src.bt.state`` re-exports the commission models defined in ``src.exec``
    # (identity preserved) so the bt state layer can speak the shared shapes
    # without ``src.exec`` importing the bt package.
    from src.bt.state.types import FixedCommission, PerShareCommission
    from src.exec import types as exec_types

    assert FixedCommission is exec_types.FixedCommission
    assert PerShareCommission is exec_types.PerShareCommission
