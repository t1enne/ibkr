"""Parity: the incremental swing cache must be byte-identical to a cold
``_find_swings`` on every growing prefix, so no STRATEGY SIGNAL can change.

The optimization never alters emitted trades — this is the guard. The cup_handle
strategy is a PASS config, so any divergence here means a re-validation is due.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.bt.strategies.cup_handle_dsl import (
    _find_swings,
    _swings_incremental,
    Swing,
)


def _ser(a: np.ndarray) -> pd.Series:
    return pd.Series(a)


def _walk(n: int, seed: int) -> tuple[pd.Series, pd.Series]:
    """A pseudo-trending OHLC-ish high/low pair (contiguous, plausible)."""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1.2, n))
    high = close + rng.uniform(0, 0.8, n)
    low = close - rng.uniform(0, 0.8, n)
    return _ser(high), _ser(low)


@pytest.mark.parametrize("lookback", [1, 2, 3, 6])
@pytest.mark.parametrize("seed", [0, 7, 42, 2024])
def test_incremental_equals_cold_every_growing_prefix(lookback: int, seed: int) -> None:
    """Feed prefixes one bar longer at a time (true screen/backtest replay)."""
    n = 160
    high, low = _walk(n, seed)
    entry: tuple[int, list[Swing]] | None = None
    for end in range(2 * lookback + 1, n + 1):
        en, got = _swings_incremental(high.iloc[:end], low.iloc[:end], lookback, entry)
        assert en == end
        cold = _find_swings(high.iloc[:end], low.iloc[:end], lookback)
        assert got == cold, f"diverged at prefix {end} (lookback={lookback})"
        entry = (en, got)


@pytest.mark.parametrize("lookback", [2, 3])
def test_incremental_from_stale_entry_reconciles(lookback: int) -> None:
    """A stale cache (symbol held, then re-entered) still converges to cold."""
    n = 200
    high, low = _walk(n, 99)
    # Simulate: computed at bar 120, then silence (held) until 200.
    _, first = _swings_incremental(high.iloc[:120], low.iloc[:120], lookback, None)
    en, got = _swings_incremental(
        high.iloc[:200], low.iloc[:200], lookback, (120, first)
    )
    assert en == 200
    cold = _find_swings(high.iloc[:200], low.iloc[:200], lookback)
    assert got == cold


def test_repeat_same_length_is_idempotent() -> None:
    """Two calls at one length (position manage + recheck) return equal lists."""
    high, low = _walk(90, 3)
    _, first = _swings_incremental(high, low, 3, None)
    en, second = _swings_incremental(high, low, 3, (len(high), first))
    assert en == len(high)
    assert first == second


def test_short_series_and_collinear_flat() -> None:
    """Below the fractal minimum returns empty and never crashes; a constant
    (collinear) series produces no swings on both paths."""
    hi = _ser(np.full(10, 5.0))
    lo = _ser(np.full(10, 5.0))
    en, got = _swings_incremental(hi, lo, 3, None)
    assert en == 10 and got == []
    assert _find_swings(hi, lo, 3) == []


def _cup_generator(seed: int, n: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """An oscillating series with gentle drift that routinely forms real cups
    whose final bars break a rim (verified: several ``entry_ok`` prefixes).
    Enhances ``_walk``, which trends and rarely breaks out at the last bar."""
    rng = np.random.default_rng(seed)
    price = np.zeros(n)
    for i in range(1, n):
        price[i] = 0.3 * price[i - 1] + rng.normal(0, 1.0) + 0.18
    close = 100 + price
    high = close + np.abs(rng.normal(0.9, 0.2, n))
    low = close - np.abs(rng.normal(0.9, 0.2, n))
    return _ser(high), _ser(low), _ser(close)


def test_detector_seed_equals_cold_detector() -> None:
    """Feeding the incremental swing list as the seed into the full detector
    yields the identical CupHandleResult as the seedless (cold) call — the
    property that keeps emitted trade signals unchanged. Uses generator that
    produces real breakout (``entry_ok``) prefixes, so the equivalence holds
    on the exact path that emits a trade."""
    from src.bt.strategies.cup_handle_dsl import detect_cup_and_handle

    n = 130
    high, low, close = _cup_generator(12, n)

    entry_signals_seen = 0
    entry: tuple[int, list[Swing]] | None = None
    for end in range(40, n + 1):
        en, sw = _swings_incremental(high.iloc[:end], low.iloc[:end], 3, entry)
        entry = (en, sw)
        vol = _ser(np.ones(end))
        seeded = detect_cup_and_handle(
            high.iloc[:end],
            low.iloc[:end],
            close.iloc[:end],
            vol,
            swing_lookback=3,
            max_cup_depth_pct=0.5,
            min_cup_depth_pct=0.02,
            rim_tolerance_pct=0.2,
            min_mid_pivots=1,
            max_cup_bars=260,
            max_handle_bars=40,
            max_handle_drop_pct=0.2,
            handle_width_scale=0.5,
            handle_depth_scale=0.3,
            handle_width_floor=3,
            volume_confirm_breakout=False,
            swings=sw,
        )
        cold = detect_cup_and_handle(
            high.iloc[:end],
            low.iloc[:end],
            close.iloc[:end],
            vol,
            swing_lookback=3,
            max_cup_depth_pct=0.5,
            min_cup_depth_pct=0.02,
            rim_tolerance_pct=0.2,
            min_mid_pivots=1,
            max_cup_bars=260,
            max_handle_bars=40,
            max_handle_drop_pct=0.2,
            handle_width_scale=0.5,
            handle_depth_scale=0.3,
            handle_width_floor=3,
            volume_confirm_breakout=False,
        )
        assert seeded == cold, f"detector diverged at prefix {end}"
        if seeded.entry_ok:
            entry_signals_seen += 1

    # The generator must produce genuine entry signals, or this test is not
    # actually guarding the trade-emitting path.
    assert entry_signals_seen > 0
