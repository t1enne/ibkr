"""VWATR (and its companions) in the shared, cursor-safe ``ctx.ta`` layer.

VWATR = ``smooth(TR * volume) / smooth(volume)`` with a PLAIN rolling-mean
``smooth`` (distinct from Wilder-smoothed ``ctx.ta.atr``), so the tests pin: the
hand-computed value, the NaN head convention, per-key memoisation,
cursor-truncation (no lookahead), the ``|den| > 1e-12`` zero-volume guard, and
that ``ctx.ta.plain_atr`` really is the plain mean rather than Wilder's.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.bt.engine.backtest import Backtest, run_backtest
from src.bt.engine.handlers import default_execution_handler, default_risk_handler
from src.bt.engine.utils import candle_generator
from src.bt.strategies.dsl import StrategyContext, strategy
from src.bt.strategies.ta_context import TaContext, init_ta
from src.bt.types import StrategyConfig

SYMBOL = "TST"


def _ctx(
    high: list[float],
    low: list[float],
    close: list[float],
    volume: list[float],
    symbol: str = SYMBOL,
) -> TaContext:
    """TaContext over a one-symbol synthetic feed (no DB, no engine)."""
    cols: dict[tuple[str, str], list[float]] = {
        (symbol, "open"): close,
        (symbol, "high"): high,
        (symbol, "low"): low,
        (symbol, "close"): close,
        (symbol, "volume"): volume,
    }
    df = pd.DataFrame(
        cols, index=pd.date_range("2024-01-02", periods=len(close), freq="D")
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns)  # type: ignore[arg-type]
    return TaContext.from_data(df, (symbol,), "1d")


def _hand_computable_ctx(period: int) -> tuple[TaContext, np.ndarray]:
    """A 6-bar feed whose TR and VWATR are computable by hand.

    Bars (high, low, close, volume)::

        0  (11,  9, 10,   0)   <- volume 0: TR*vol + vol both contribute 0
        1  (12, 10, 11, 100)   TR = max(2, |12-10|, |10-10|) = 2
        2  ( 9,  8,  8, 200)   TR = max(1, |9-11|, |8-11|)     = 3
        3  (15, 12, 14, 100)   TR = max(3, |15-8|, |12-8|)     = 7
        4  (16, 15, 15, 300)   TR = max(1, |16-14|, |15-14|)   = 2
        5  (20, 19, 19, 100)   TR = max(1, |20-15|, |19-15|)   = 5
    """
    high = [11.0, 12.0, 9.0, 15.0, 16.0, 20.0]
    low = [9.0, 10.0, 8.0, 12.0, 15.0, 19.0]
    close = [10.0, 11.0, 8.0, 14.0, 15.0, 19.0]
    volume = [0.0, 100.0, 200.0, 100.0, 300.0, 100.0]
    return _ctx(high, low, close, volume), np.array([np.nan, 2.0, 3.0, 7.0, 2.0, 5.0])


def test_true_range_matches_hand_computed_values() -> None:
    ctx, expected = _hand_computable_ctx(period=2)
    tr = ctx.true_range(SYMBOL).to_array()
    assert len(tr) == len(expected)
    assert np.isnan(tr[0])  # no prior close on the first bar
    np.testing.assert_allclose(tr[1:], expected[1:])


def test_vwatr_matches_hand_computed_value() -> None:
    """period=2, so ``smooth`` is the mean of the last two bars' TR*volume.

    At bar 5: num = (2*300 + 5*100) / 2 = 550, den = (300 + 100) / 2 = 200
    -> VWATR = 2.75. At bar 4: num = (7*100 + 2*300) / 2 = 650,
    den = (100 + 300) / 2 = 200 -> 3.25. At bar 3: num = (3*200 + 7*100) / 2 =
    650, den = (200 + 100) / 2 = 150 -> 4.3333...
    """
    ctx, _ = _hand_computable_ctx(period=2)
    arr = ctx.vwatr(SYMBOL, 2).to_array()
    assert np.isnan(arr[0]) and np.isnan(arr[1])  # period-1 head + TR's NaN
    # bar 2: num = (NaN*0 + 2*100)/2 -> NaN. bar 2 is the first finite one:
    # num = (0*0 + 2*100)/2? no -- TR[1]=2 with vol 100, TR[2]=3 with vol 200:
    # num = (2*100 + 3*200)/2 = 400, den = (100 + 200)/2 = 150 -> 8/3.
    np.testing.assert_allclose(arr[2:], [8.0 / 3.0, 65.0 / 15.0, 3.25, 2.75])


def test_vwatr_nan_head_is_first_period_values() -> None:
    """The NaN head is ``period`` bars: rolling warm-up + true range's own first bar.

    ``smooth`` needs ``period - 1`` prior bars and TR is NaN on bar 0, so the
    first finite VWATR lands at index ``period``. With ``period = 1`` the only
    NaN is bar 0 (TR's warm-up).
    """
    n = 40
    ctx = _ctx(
        high=[10.0 + i for i in range(n)],
        low=[9.0 + i for i in range(n)],
        close=[9.5 + i for i in range(n)],
        volume=[1000.0] * n,
    )
    for period in (1, 2, 5, 14):
        arr = ctx.vwatr(SYMBOL, period).to_array()
        assert np.all(np.isnan(arr[:period])), period
        assert np.all(np.isfinite(arr[period:])), period


def test_true_range_nan_is_not_skip_filled() -> None:
    """NaN propagates through the rolling mean (pandas ``skipna`` is NOT used).

    A plain ``Series.rolling().mean()`` would drop bar 0's NaN true range and
    publish a VWATR one bar early; the shipped indicator keeps the ``period``
    bar NaN head, which is what every existing backtest was computed against.
    """
    n = 12
    ctx = _ctx(
        high=[10.0 + i for i in range(n)],
        low=[9.0 + i for i in range(n)],
        close=[9.5 + i for i in range(n)],
        volume=[1000.0] * n,
    )
    assert np.isnan(ctx.true_range(SYMBOL).to_array()[0])
    arr = ctx.vwatr(SYMBOL, 4).to_array()
    assert np.isnan(arr[3])  # still inside the head, not skip-filled
    assert np.isfinite(arr[4])


def test_vwatr_zero_volume_window_is_nan_not_inf() -> None:
    """A window whose volume sums to ~0 has no meaningful VWATR -> NaN."""
    n = 10
    volume = [0.0] * 6 + [500.0] * 4
    ctx = _ctx(
        high=[10.0 + i for i in range(n)],
        low=[9.0 + i for i in range(n)],
        close=[9.5 + i for i in range(n)],
        volume=volume,
    )
    arr = ctx.vwatr(SYMBOL, 3).to_array()
    assert not np.any(np.isinf(arr))
    # Windows ending at bars 2..5 hold only zero-volume bars -> NaN.
    assert np.all(np.isnan(arr[2:6]))
    assert np.all(np.isfinite(arr[6:]))


def test_vwatr_empty_and_short_series() -> None:
    empty = _ctx([], [], [], [])
    assert len(empty.vwatr(SYMBOL, 14).to_array()) == 0
    short = _ctx([10.0, 11.0], [9.0, 10.0], [9.5, 10.5], [100.0, 100.0])
    arr = short.vwatr(SYMBOL, 14).to_array()  # period > len
    assert len(arr) == 2
    assert np.all(np.isnan(arr))


def test_vwatr_is_memoised_per_key() -> None:
    n = 30
    ctx = _ctx(
        high=[10.0 + i for i in range(n)],
        low=[9.0 + i for i in range(n)],
        close=[9.5 + i for i in range(n)],
        volume=[1000.0] * n,
    )
    before = ctx.compute_count
    first = ctx.vwatr(SYMBOL, 14).to_array()
    after_first = ctx.compute_count
    second = ctx.vwatr(SYMBOL, 14).to_array()
    assert after_first == before + 1  # one full-series compute
    assert ctx.compute_count == after_first  # cache hit, no recompute
    np.testing.assert_array_equal(first, second)
    # A different period is a different key -> its own compute.
    ctx.vwatr(SYMBOL, 7)
    assert ctx.compute_count == after_first + 1


def test_vwatr_read_never_exposes_future_bars(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cursor-truncated SeriesView hides every bar past the engine cursor."""
    periods = 30
    idx = pd.date_range("2024-01-02", periods=periods, freq="D")
    closes = [100.0 + i for i in range(periods)]
    data = pd.DataFrame(
        {
            (SYMBOL, "open"): closes,
            (SYMBOL, "high"): [c + 1 for c in closes],
            (SYMBOL, "low"): [c - 1 for c in closes],
            (SYMBOL, "close"): closes,
            (SYMBOL, "volume"): [1000.0] * periods,
        },
        index=idx,
    )
    data.columns = pd.MultiIndex.from_tuples(data.columns)  # type: ignore[arg-type]
    ctx = init_ta(data, (SYMBOL,), "1d")
    full = ctx.vwatr(SYMBOL, 5).to_array()

    # Walk the real engine: the probe records what the cursor exposes per bar.
    seen: list[tuple[int, float, float]] = []

    def on_candle(ctx_inner: StrategyContext) -> None:
        view = ctx_inner.ta.vwatr(SYMBOL, 5)
        seen.append((len(view), full[len(view) - 1], view.last()))

    adapter = strategy(bars="1d")(on_candle)
    # The probe is not a registered strategy type, so param resolution is stubbed
    # (the probe needs no params); everything else is the production path.
    monkeypatch.setattr("src.bt.engine.backtest.resolve_params", lambda _name, raw: raw)
    bt = Backtest(_probe_config())
    run_backtest(
        bt,
        candle_generator(data, bt.config),
        default_execution_handler(),
        default_risk_handler(),
        strategy_mod=SimpleNamespace(on_candle=adapter),
        ta=ctx,
    )

    assert [n for n, _, _ in seen] == list(range(1, periods + 1))
    # The visible tail is always the bar at the cursor, never a future one.
    for n, expected, actual in seen:
        assert (np.isnan(actual) and np.isnan(expected)) or actual == expected, (
            n,
            actual,
            expected,
        )


def _probe_config() -> StrategyConfig:
    return StrategyConfig(
        name="vwatr-cursor",
        strategy_type="_probe",
        symbols=[SYMBOL],
        initial_capital=10000.0,
        commission=0.0,
        warmup="0d",
        trading_start="2024-01-02",
        trading_end="2024-02-01",
        bars=["1d"],
        strategy_params={},
        benchmark_symbols=[],
    )


def test_vwatr_baseline_is_rolling_mean_of_vwatr() -> None:
    """The baseline is the trailing mean of the VWATR series itself."""
    n = 30
    ctx = _ctx(
        high=[10.0 + i for i in range(n)],
        low=[9.0 + i for i in range(n)],
        close=[9.5 + i for i in range(n)],
        volume=[1000.0 + 10 * (i % 3) for i in range(n)],
    )
    vwatr = ctx.vwatr(SYMBOL, 5).to_array()
    baseline = ctx.vwatr_baseline(SYMBOL, 5, 10).to_array()
    expected = pd.Series(vwatr).rolling(10).mean().to_numpy()
    assert np.array_equal(np.isnan(baseline), np.isnan(expected))
    np.testing.assert_allclose(
        baseline[~np.isnan(expected)], expected[~np.isnan(expected)]
    )


def test_plain_atr_is_rolling_mean_not_wilder() -> None:
    """``plain_atr`` is the simple mean of true range; ``atr`` is Wilder-smoothed."""
    n = 30
    ctx = _ctx(
        high=[10.0 + i for i in range(n)],
        low=[9.0 + i for i in range(n)],
        close=[9.5 + i for i in range(n)],
        volume=[1000.0] * n,
    )
    tr = ctx.true_range(SYMBOL).to_array()
    plain = ctx.plain_atr(SYMBOL, 5).to_array()
    expected = pd.Series(tr).rolling(5).mean().to_numpy()
    assert np.array_equal(np.isnan(plain), np.isnan(expected))
    np.testing.assert_allclose(
        plain[~np.isnan(expected)], expected[~np.isnan(expected)]
    )
    wilder = ctx.atr(SYMBOL, 5).to_array()
    last = ~np.isnan(wilder) & ~np.isnan(plain)
    assert not np.allclose(wilder[last], plain[last])
