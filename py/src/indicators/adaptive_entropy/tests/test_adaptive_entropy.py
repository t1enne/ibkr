"""Tests for the adaptive entropy trend indicator (batch + online)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators.adaptive_entropy.online import OnlineAdaptiveEntropy
from src.indicators.adaptive_entropy.pure import adaptive_entropy
from src.indicators.adaptive_entropy.types import (
    AdaptiveEntropyConfig,
    AdaptiveEntropyResult,
)
from src.indicators.adaptive_entropy.validate import NonFiniteBarError


def _frame(closes: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Build high/low from closes (1% range) for the batch API."""
    high = closes * 1.01
    low = closes * 0.99
    return closes, high, low


def test_batch_empty_input():
    closes = pd.Series([], dtype=float)
    high, low = closes.copy(), closes.copy()
    df = adaptive_entropy(closes, high, low)
    assert df.empty


def test_batch_constant_series_no_nan_blowup():
    """A flat series has zero return range -> entropy falls back to 0.5 and the
    indicator stays well-formed (no NaN explosion, no division-by-zero)."""
    closes = pd.Series([100.0] * 60)
    high, low = closes.copy(), closes.copy()
    df = adaptive_entropy(closes, high, low, AdaptiveEntropyConfig(lookback=25))
    assert len(df) == 60
    assert set(df.columns) == {
        "close",
        "entropy",
        "normalized_entropy",
        "trend_strength",
        "adaptive_ema",
        "atr",
        "fast_band_width",
        "slow_band_width",
        "inner_upper",
        "inner_lower",
        "outer_upper",
        "outer_lower",
        "trend",
    }
    # No trend ever triggers on a truly flat series.
    assert int(df["trend"].max()) == 0
    assert int(df["trend"].min()) == 0


def test_batch_warmup_nan_then_defined():
    """Bars before the (lookback - 1) return observations carry NaN entropy and
    empty/zero contributions; once the window fills they are defined."""
    rng = np.random.default_rng(7)
    closes = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 120))))
    high, low = closes * 1.01, closes * 0.99
    cfg = AdaptiveEntropyConfig(lookback=25)
    df = adaptive_entropy(closes, high, low, cfg)

    # First bar: entropy is NaN (no returns yet). By row ~30 the window is warm.
    assert pd.isna(df["entropy"].iloc[0])
    warm = df[df["entropy"].notna()]
    assert len(warm) > 0
    # Entropy is normalized to [0, 1]; trend strength in [0, 1].
    assert float(warm["entropy"].min()) >= 0.0
    assert float(warm["entropy"].max()) <= 1.0
    assert float(warm["trend_strength"].min()) >= 0.0
    assert float(warm["trend_strength"].max()) <= 1.0
    # Once warm, ATR and the adaptive EMA are positive and finite.
    assert float(warm["atr"].dropna().max()) > 0.0
    assert warm["adaptive_ema"].isna().sum() == 0


def test_batch_clear_uptrend_is_bullish():
    """A strong monotone uptrend should resolve to trend == 1: close sits above
    the inner-upper band once the window is warm."""
    closes = pd.Series([110.0 * 1.005**i for i in range(80)])
    high, low = closes * 1.01, closes * 0.99
    cfg = AdaptiveEntropyConfig(lookback=25)
    df = adaptive_entropy(closes, high, low, cfg)
    warm = df[df["entropy"].notna()]
    assert warm["trend"].iloc[-1] == 1


def test_batch_clear_downtrend_is_bearish():
    """A strong monotone downtrend should resolve to trend == -1."""
    closes = pd.Series([110.0 * 0.995**i for i in range(80)])
    high, low = closes * 1.01, closes * 0.99
    cfg = AdaptiveEntropyConfig(lookback=25)
    df = adaptive_entropy(closes, high, low, cfg)
    warm = df[df["entropy"].notna()]
    assert warm["trend"].iloc[-1] == -1


def test_batch_entropy_warmup_starts_exactly_at_lookback():
    """Entropy must be NaN on bars [0, lookback) and defined from bar lookback
    onward -- matching both the batch loop and the online class's `ready` gate.
    Pins the warmup boundary so a refactor can't silently shift it."""
    rng = np.random.default_rng(3)
    closes = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 80))))
    high, low = closes * 1.01, closes * 0.99
    cfg = AdaptiveEntropyConfig(lookback=25)
    df = adaptive_entropy(closes, high, low, cfg)

    entropy = df["entropy"]
    # Defined exactly from bar `lookback` onward (0-indexed).
    assert entropy.iloc[: cfg.lookback].isna().all()
    assert entropy.iloc[cfg.lookback :].notna().all()


def test_online_matches_batch():
    """Feed the same bars through OnlineAdaptiveEntropy and compare the final
    snapshot to the batch function's last warm row."""
    rng = np.random.default_rng(11)
    closes = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.012, 150))))
    high = closes * 1.01
    low = closes * 0.99
    cfg = AdaptiveEntropyConfig(lookback=25)

    df = adaptive_entropy(closes, high, low, cfg)
    batch = df[df["entropy"].notna()].iloc[-1]

    online = OnlineAdaptiveEntropy(cfg)
    for c, h, lo_ in zip(closes, high, low):
        snap = online.observe(c, h, lo_)

    assert snap is not None
    assert online.ready
    assert snap.trend == batch["trend"]
    assert abs(snap.adaptive_ema - float(batch["adaptive_ema"])) < 1e-6
    assert abs(snap.atr - float(batch["atr"])) < 1e-6
    assert abs(snap.entropy - float(batch["entropy"])) < 1e-9
    assert abs(snap.inner_upper - float(batch["inner_upper"])) < 1e-6
    assert abs(snap.inner_lower - float(batch["inner_lower"])) < 1e-6


def test_online_full_bar_for_bar_parity():
    """Every *returned* online snapshot must equal the batch row at the same
    bar once both are warm (beyond warmup both must agree exactly)."""
    closes = pd.Series([100.0 * 1.002**i + 0.5 * np.sin(i / 5.0) for i in range(120)])
    high = closes * 1.02
    low = closes * 0.98
    cfg = AdaptiveEntropyConfig(lookback=25)

    df = adaptive_entropy(closes, high, low, cfg)
    online = OnlineAdaptiveEntropy(cfg)

    rets: list[tuple[int, AdaptiveEntropyResult]] = []
    for i, (c, h, lo_) in enumerate(zip(closes, high, low)):
        snap = online.observe(c, h, lo_)
        if snap is not None:
            rets.append((i, snap))

    for i, snap in rets:
        row = df.iloc[i]
        if not np.isnan(row["entropy"]):
            assert abs(snap.adaptive_ema - float(row["adaptive_ema"])) < 1e-9
            assert abs(snap.entropy - float(row["entropy"])) < 1e-9
            assert abs(snap.atr - float(row["atr"])) < 1e-6
            assert snap.trend == int(row["trend"])


def test_online_reset_replays_identically():
    """After reset(), replaying the same bars yields the identical outcomes."""
    closes = pd.Series([100.0 * 1.003**i for i in range(90)])
    high = closes * 1.01
    low = closes * 0.99
    cfg = AdaptiveEntropyConfig(lookback=25)

    a = OnlineAdaptiveEntropy(cfg)
    results_a = [a.observe(c, h, lo_) for c, h, lo_ in zip(closes, high, low)]
    a.reset()
    results_b = [a.observe(c, h, lo_) for c, h, lo_ in zip(closes, high, low)]

    assert len(results_a) == len(results_b)
    for sa, sb in zip(results_a, results_b):
        assert sa.trend == sb.trend
        assert abs(sa.adaptive_ema - sb.adaptive_ema) < 1e-12
        assert abs(sa.atr - sb.atr) < 1e-12


def test_config_validation():
    with pytest.raises(ValueError):
        AdaptiveEntropyConfig(lookback=3)
    with pytest.raises(ValueError):
        AdaptiveEntropyConfig(num_bins=1)
    with pytest.raises(ValueError):
        AdaptiveEntropyConfig(fast_multiplier=0)


def test_online_rejects_non_finite_bar():
    """A non-finite price raises instead of poisoning the running EMA/ATR.

    Regression: the model's EMA and ATR are running scalars, so one ``nan`` bar
    made every later band comparison ``False`` and froze ``trend`` at its last
    value -- a regime gate that silently admitted or refused everything. The
    error must name the offending field and leave state untouched.
    """
    model = OnlineAdaptiveEntropy(AdaptiveEntropyConfig(lookback=5))
    model.observe(100.0, 101.0, 99.0)
    before = model.n_bars

    for kwargs, field in (
        ({"close": float("nan")}, "close"),
        ({"high": float("inf")}, "high"),
        ({"low": float("-inf")}, "low"),
    ):
        bar = {"close": 100.0, "high": 101.0, "low": 99.0} | kwargs
        with pytest.raises(NonFiniteBarError) as err:
            model.observe(bar["close"], bar["high"], bar["low"])
        assert field in str(err.value)
        assert "non-finite" in str(err.value)
    assert model.n_bars == before


def test_batch_rejects_non_finite_series():
    """The batch path refuses the same bars the online path refuses."""
    closes = pd.Series([100.0] * 40)
    high = closes * 1.01
    low = closes * 0.99
    high.iloc[7] = float("nan")
    with pytest.raises(NonFiniteBarError) as err:
        adaptive_entropy(closes, high, low, AdaptiveEntropyConfig(lookback=5))
    assert "high" in str(err.value)
    assert "index 7" in str(err.value)


def test_online_and_batch_agree_on_rejection():
    """Parity holds for the failure mode, not just the happy path."""
    closes = pd.Series([100.0 * 1.001**i for i in range(40)])
    high = closes * 1.01
    low = closes * 0.99
    bad_bar = 12
    closes_bad = closes.copy()
    closes_bad.iloc[bad_bar] = float("nan")

    with pytest.raises(NonFiniteBarError):
        adaptive_entropy(closes_bad, high, low, AdaptiveEntropyConfig(lookback=5))

    model = OnlineAdaptiveEntropy(AdaptiveEntropyConfig(lookback=5))
    with pytest.raises(NonFiniteBarError):
        for i, (c, h, lo_) in enumerate(zip(closes_bad, high, low)):
            model.observe(c, h, lo_)
    assert model.n_bars == bad_bar
