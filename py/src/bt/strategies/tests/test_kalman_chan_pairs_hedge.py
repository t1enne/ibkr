"""Item 11 of the Chan Ex. 3.3 reproduction: the hedge is a SHARE count.

Methodology: no engine, no filter. Build a real ``StrategyContext`` over a
two-symbol ``TaContext`` at a known price pair, call the strategy's own
``_open_spread`` with a known ``beta``, and read the emitted signals'
``qty`` -- the absolute share counts the engine will fill.

Why a unit test when strategies are normally validated by running: this is an
IDENTITY (``x_shares == floor(beta * y_shares)``), not an edge. Running the six
configs cannot pin it -- a notional hedge changes the P&L by a few percent and
looks like noise, while the identity either holds or does not.

The test is a REGRESSION guard: on the pre-fix code (``hedge = size*|beta|``,
a capital fraction, so ``x_shares = |beta|*size*capital/price_x``) it fails by
the ``price_y/price_x`` factor.
"""

from __future__ import annotations

import math
from typing import cast

import pandas as pd
import pytest

from src.bt.state import (
    BacktestState,
    Candle,
    create_initial_portfolio,
)
from src.bt.strategies.dsl import StrategyContext
from src.bt.strategies.kalman_chan_pairs_dsl import Params, _leg_sizes, _open_spread
from src.bt.strategies.ta_context import TaContext

X, Y = "GDX", "SLV"  # (regressor, observed leg), the configs' own order
CAPITAL = 100_000.0
# Picked to make the two answers differ by a wide margin: price_x/price_y = 1.6,
# so a notional hedge emits 1/1.6 = 62.5% of the reference's x shares.
PX, PY = 32.0, 20.0
BETA = 0.75


def _ts(value: str) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(value))


def _ctx(px: float, py: float) -> StrategyContext:
    """A flat book at a known two-leg price pair (one bar, both legs quoted)."""
    index = pd.DatetimeIndex([_ts("2024-01-02")])
    data = pd.concat(
        {
            X: pd.DataFrame(
                {
                    "open": [px],
                    "high": [px],
                    "low": [px],
                    "close": [px],
                    "volume": [1_000.0],
                },
                index=index,
            ),
            Y: pd.DataFrame(
                {
                    "open": [py],
                    "high": [py],
                    "low": [py],
                    "close": [py],
                    "volume": [1_000.0],
                },
                index=index,
            ),
        },
        axis=1,
    )
    portfolio = create_initial_portfolio(
        initial_capital=CAPITAL, start_timestamp=_ts("2024-01-01")
    )
    state = cast(BacktestState, type("S", (), {"portfolio": portfolio})())
    candle = Candle(
        timestamp=_ts("2024-01-02"),
        symbol=Y,
        open=py,
        high=py,
        low=py,
        close=py,
        volume=1_000.0,
        interval="1d",
    )
    ctx = StrategyContext(
        state=state,
        candle=candle,
        params=Params(pair=(X, Y)),
        ta=TaContext.from_data(data, (X, Y), "1d"),
        symbols=(X, Y),
        interval="1d",
    )
    ctx.shared = {}  # the engine's per-run holder; ``_open_spread`` records the side
    return ctx


def test_hedge_qty_is_floor_beta_times_y_shares():
    """REGRESSION: x shares == floor(beta * N) -- a SHARE ratio, not notional.

    N = size*capital/price_y = 0.3*100_000/20 = 1500 y shares.
    floor(0.75*1500) = 1125 x shares, i.e. 1125*32 = $36,000 of x against
    $30,000 of y (gearing beta*price_x/price_y = 1.2).
    The pre-fix notional hedge emitted 0.3*0.75*100_000/32 = 703.125 shares.
    """
    ctx = _ctx(PX, PY)
    _open_spread(ctx, X, Y, 1.0, BETA, PX, PY)
    signals = {signal.symbol: signal for signal in ctx._signals}
    y_shares = signals[Y].qty
    x_shares = signals[X].qty

    assert y_shares == pytest.approx(0.3 * CAPITAL / PY)  # 1500
    assert x_shares == pytest.approx(math.floor(BETA * y_shares))  # 1125
    # The share ratio is beta (the floor moves it by < 1 share)...
    assert x_shares / y_shares == pytest.approx(BETA, rel=1e-3)
    # ...and the NOTIONAL ratio is beta * price_x/price_y, NOT beta.
    notional_ratio = x_shares * PX / (y_shares * PY)
    assert notional_ratio == pytest.approx(BETA * PX / PY)  # 1.2
    assert notional_ratio != pytest.approx(BETA)


def test_hedge_side_follows_the_spread_side_with_the_same_share_count():
    """Both legs mirror with the same share counts (item 9's short spread)."""
    long_ctx = _ctx(PX, PY)
    short_ctx = _ctx(PX, PY)
    _open_spread(long_ctx, X, Y, 1.0, BETA, PX, PY)
    _open_spread(short_ctx, X, Y, -1.0, BETA, PX, PY)

    long_sig = {signal.symbol: signal for signal in long_ctx._signals}
    short_sig = {signal.symbol: signal for signal in short_ctx._signals}
    assert long_sig[X].qty == pytest.approx(short_sig[X].qty)
    assert long_sig[Y].qty == pytest.approx(short_sig[Y].qty)
    assert long_sig[X].action.value == "short"  # long the spread -> sell x
    assert long_sig[Y].action.value == "long"
    assert short_sig[X].action.value == "long"  # short the spread -> buy x
    assert short_sig[Y].action.value == "short"


def test_leg_sizes_floors_off_the_y_count_that_actually_fills():
    """The floor is on the ABSOLUTE count, applied to the rounded y qty."""
    # beta*y_shares = 0.9996 -> 0 hedge shares, which must raise, not go naked.
    with pytest.raises(AssertionError, match="floors to zero"):
        _leg_sizes(CAPITAL, PX, PY, 0.0006664, 0.3)
    # A fractional product drops the fraction: floor(1.9999) = 1 share.
    y_frac, x_frac = _leg_sizes(CAPITAL, PX, PY, 1.9999 / 1_500.0, 0.3)
    assert y_frac * CAPITAL / PY == pytest.approx(1_500.0)
    assert x_frac * CAPITAL / PX == pytest.approx(1.0)
