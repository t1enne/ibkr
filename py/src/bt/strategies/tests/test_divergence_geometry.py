"""Unit tests for pivot-pair divergence geometry. No DB, no engine."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.bt.strategies.divergence_geometry import (
    DivergenceParams,
    PivotScanner,
    detect_divergence,
    evaluate_from_pivots,
    find_pivots,
)


def _p(**kw: object) -> DivergenceParams:
    base = DivergenceParams(
        pivot_bars=1,
        lookback=40,
        min_gap_bars=2,
        min_osc_gap=1.0,
        require_intervening=True,
    )
    return dataclasses.replace(base, **kw)


# --- find_pivots -------------------------------------------------------------


def test_find_pivots_highs_and_lows() -> None:
    prices = np.array([1.0, 2.0, 1.0, 2.0, 1.0])
    osc = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    highs = find_pivots(prices, osc, pivot_bars=1, high=True)
    lows = find_pivots(prices, osc, pivot_bars=1, high=False)
    assert [h.idx for h in highs] == [1, 3]
    assert [lo.idx for lo in lows] == [2]
    assert highs[0].osc == 20.0


def test_find_pivots_ignores_right_edge() -> None:
    # last bar cannot be confirmed -- its right window is not in the array
    prices = np.array([1.0, 2.0, 3.0])
    osc = np.array([1.0, 1.0, 1.0])
    assert find_pivots(prices, osc, pivot_bars=1, high=True) == []


def test_find_pivots_rejects_nan_window() -> None:
    prices = np.array([1.0, np.nan, 3.0])
    osc = np.array([1.0, 1.0, 1.0])
    assert find_pivots(prices, osc, pivot_bars=1, high=True) == []


def test_find_pivots_short_array() -> None:
    assert find_pivots(np.array([1.0]), np.array([1.0]), pivot_bars=3, high=True) == []


# --- bullish divergence ------------------------------------------------------


def _bull_series() -> tuple[np.ndarray, np.ndarray]:
    """low 10 @i5, rally to 14 @i8, lower low 9 @i11. MFI: 30 -> 55."""
    prices = np.array(
        [13.0, 12.0, 12.5, 11.5, 11.0, 10.0, 12.0, 13.5, 14.0, 13.0, 11.5, 9.0, 10.5]
    )
    osc = np.array(
        [50.0, 45.0, 44.0, 40.0, 35.0, 30.0, 40.0, 50.0, 55.0, 45.0, 40.0, 55.0, 60.0]
    )
    return prices, osc


def test_bullish_divergence_fires() -> None:
    prices, osc = _bull_series()
    hit = detect_divergence(prices, osc, "long", _p())
    assert hit is not None
    assert hit.price == pytest.approx(9.0)
    assert hit.osc == pytest.approx(55.0)


def test_bullish_rejects_when_mfi_confirms_new_low() -> None:
    prices, osc = _bull_series()
    osc = osc.copy()
    osc[11] = 24.0  # MFI makes a lower low too: no divergence
    assert detect_divergence(prices, osc, "long", _p()) is None


def test_bullish_rejects_when_price_makes_higher_low() -> None:
    prices, osc = _bull_series()
    prices = prices.copy()
    prices[11] = 11.0  # not a lower low vs 10.0
    assert detect_divergence(prices, osc, "long", _p()) is None


def test_two_pivots_always_imply_an_intervening_opposite_pivot() -> None:
    # Fractal pivots of width k cannot be adjacent: between any two same-type
    # pivots there is an opposite pivot. This is the wave structure, and it is
    # what makes a monotone leg (which yields only ONE confirmed pivot) fail.
    prices, osc = _bull_series()
    lows = find_pivots(prices, osc, pivot_bars=1, high=False)
    highs = find_pivots(prices, osc, pivot_bars=1, high=True)
    assert len(lows) >= 2
    l1, l2 = lows[-2], lows[-1]
    assert any(l1.idx < h.idx < l2.idx for h in highs)


def test_monotone_leg_yields_one_pivot_and_no_signal() -> None:
    # A straight decline has no confirmed swing low until it turns, so the
    # detector sees a single pivot and refuses -- the exact failure of the
    # half-split ``min(min(prior_half), min(recent_half))`` definition, which
    # happily fires here.
    prices = np.array([11.0, 10.5, 10.0, 9.6, 9.0, 8.2, 7.0, 7.6, 8.0])
    osc = np.array([50.0, 45.0, 30.0, 26.0, 24.0, 28.0, 55.0, 60.0, 62.0])
    assert len(find_pivots(prices, osc, pivot_bars=1, high=False)) == 1
    assert detect_divergence(prices, osc, "long", _p()) is None


def test_min_osc_gap_enforced() -> None:
    prices, osc = _bull_series()
    osc = osc.copy()
    osc[11] = 30.5  # improvement of only 0.5 over P1's 30.0
    assert detect_divergence(prices, osc, "long", _p(min_osc_gap=5.0)) is None
    assert detect_divergence(prices, osc, "long", _p(min_osc_gap=0.5)) is not None


def test_osc_gap_frac_normalises_gap() -> None:
    prices, osc = _bull_series()
    # osc range over the segment is wide (~30), so a frac floor can bite
    assert detect_divergence(prices, osc, "long", _p(min_osc_gap_frac=0.99)) is None


def test_gap_bars_bounds() -> None:
    prices, osc = _bull_series()
    # P1 at idx 5, P2 at idx 11 -> separation 6
    assert detect_divergence(prices, osc, "long", _p(max_gap_bars=3)) is None
    assert detect_divergence(prices, osc, "long", _p(min_gap_bars=7)) is None
    assert detect_divergence(prices, osc, "long", _p(min_gap_bars=2)) is not None


def test_max_age_rejects_stale_pivot() -> None:
    prices, osc = _bull_series()
    assert detect_divergence(prices, osc, "long", _p(max_age_bars=0)) is not None
    # P2 sits 1 bar before the end; age 0 is fine, but force a gap after it
    prices2 = np.concatenate([prices, np.array([11.0, 12.0, 13.0])])
    osc2 = np.concatenate([osc, np.array([65.0, 66.0, 67.0])])
    assert detect_divergence(prices2, osc2, "long", _p(max_age_bars=1)) is None


def test_price_extension_band() -> None:
    prices, osc = _bull_series()
    # penetration (10 - 9)/10 = 0.10
    assert detect_divergence(prices, osc, "long", _p(min_price_ext=0.05)) is not None
    assert detect_divergence(prices, osc, "long", _p(min_price_ext=0.20)) is None
    assert detect_divergence(prices, osc, "long", _p(max_price_ext=0.05)) is None
    assert detect_divergence(prices, osc, "long", _p(max_price_ext=0.20)) is not None


def test_osc_extreme_side_gate() -> None:
    prices, osc = _bull_series()
    assert detect_divergence(prices, osc, "long", _p(osc_extreme_side=True)) is not None
    assert (
        detect_divergence(
            prices, osc, "long", _p(osc_extreme_side=True, osc_extreme_floor=20.0)
        )
        is None
    )


def test_intervening_atr_depth() -> None:
    prices, osc = _bull_series()
    # intervening high 14.0 vs P1 low 10.0 -> bounce of 4.0
    assert (
        detect_divergence(prices, osc, "long", _p(min_intervening_atr=1.0), atr=2.0)
        is not None
    )
    assert (
        detect_divergence(prices, osc, "long", _p(min_intervening_atr=3.0), atr=2.0)
        is None
    )


# --- bearish divergence ------------------------------------------------------


def _bear_series() -> tuple[np.ndarray, np.ndarray]:
    """high 10 @i5, dip to 6 @i8, higher high 11 @i11. MFI: 70 -> 45."""
    prices = np.array(
        [7.0, 8.0, 7.5, 8.5, 9.0, 10.0, 8.0, 6.5, 6.0, 7.0, 8.5, 11.0, 9.5]
    )
    osc = np.array(
        [50.0, 55.0, 56.0, 60.0, 65.0, 70.0, 60.0, 50.0, 45.0, 55.0, 60.0, 45.0, 40.0]
    )
    return prices, osc


def test_bearish_divergence_fires() -> None:
    prices, osc = _bear_series()
    hit = detect_divergence(prices, osc, "short", _p())
    assert hit is not None
    assert hit.price == pytest.approx(11.0)
    assert hit.osc == pytest.approx(45.0)


def test_bearish_rejects_when_mfi_confirms_new_high() -> None:
    prices, osc = _bear_series()
    osc = osc.copy()
    osc[11] = 76.0
    assert detect_divergence(prices, osc, "short", _p()) is None


def test_bearish_requires_intervening_trough() -> None:
    # A straight advance confirms a single swing high, so no divergence pair
    # exists -- the mirror of test_monotone_leg_yields_one_pivot_and_no_signal.
    prices = np.array([8.0, 8.5, 9.0, 9.4, 10.0, 10.5, 11.0, 11.5, 11.2])
    osc = np.array([50.0, 55.0, 60.0, 65.0, 70.0, 68.0, 60.0, 45.0, 42.0])
    assert len(find_pivots(prices, osc, pivot_bars=1, high=True)) == 1
    assert detect_divergence(prices, osc, "short", _p()) is None


# --- degenerate input --------------------------------------------------------


def test_short_history_and_empty() -> None:
    assert detect_divergence(np.array([]), np.array([]), "long", _p()) is None
    assert detect_divergence(np.array([1.0]), np.array([1.0]), "long", _p()) is None


def test_mismatched_lengths() -> None:
    assert (
        detect_divergence(np.array([1.0, 2.0, 3.0]), np.array([1.0]), "long", _p())
        is None
    )


def test_nan_prices_fail_closed() -> None:
    prices, osc = _bull_series()
    prices = prices.copy()
    prices[8] = np.nan  # kill the intervening peak's confirmation window
    assert detect_divergence(prices, osc, "long", _p()) is None


def test_zero_pivot_bars_or_gap_is_disabled() -> None:
    prices, osc = _bull_series()
    assert detect_divergence(prices, osc, "long", _p(pivot_bars=0)) is None
    assert detect_divergence(prices, osc, "long", _p(min_gap_bars=0)) is None


# --- PivotScanner: incremental == batch, and detect parity -------------------


def test_scanner_matches_batch_over_every_prefix() -> None:
    rng = np.random.default_rng(7)
    prices = 100 + np.cumsum(rng.normal(0, 1, 400))
    osc = rng.uniform(20, 80, 400)
    for k in (1, 3):
        for high in (False, True):
            scanner = PivotScanner(k, high=high)
            for i in range(10, 401):
                scanner.extend(prices[:i], osc[:i])
                expected = [
                    (pv.idx, pv.price)
                    for pv in find_pivots(prices[:i], osc[:i], pivot_bars=k, high=high)
                ]
                got = [(pv.idx, pv.price) for pv in scanner.pivots]
                assert got == expected, (k, high, i)


def test_scanner_resets_on_shorter_series() -> None:
    prices, osc = _bull_series()
    scanner = PivotScanner(1, high=False)
    scanner.extend(prices, osc)
    assert scanner.pivots
    scanner.extend(prices[:4], osc[:4])
    assert all(pv.idx < 4 for pv in scanner.pivots)


def test_scanner_agrees_with_detect_divergence() -> None:
    rng = np.random.default_rng(3)
    prices = 100 + np.cumsum(rng.normal(0, 1, 500))
    osc = rng.uniform(20, 80, 500)
    geo = _p(pivot_bars=2, lookback=50, min_gap_bars=3, min_osc_gap=4.0)
    lows = PivotScanner(2, high=False)
    highs = PivotScanner(2, high=True)
    for i in range(60, 501):
        p, o = prices[:i], osc[:i]
        lows.extend(p, o)
        highs.extend(p, o)
        for side, own, other in (
            ("long", lows.pivots, highs.pivots),
            ("short", highs.pivots, lows.pivots),
        ):
            batch = detect_divergence(p, o, side, geo)
            incr = evaluate_from_pivots(own, other, p, o, side, geo)
            assert (batch is None) == (incr is None)
            if batch is not None and incr is not None:
                assert batch.idx == incr.idx


def test_scanner_disabled_width_is_inert() -> None:
    scanner = PivotScanner(0, high=False)
    prices, osc = _bull_series()
    scanner.extend(prices, osc)
    assert scanner.pivots == []
