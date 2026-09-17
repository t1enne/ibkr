"""Unit tests for the divergence triggers in ``mfi_divergence_dsl``.

No DB, no engine. The OhlcvView is faked with numpy arrays and MFI is passed
straight in. Covers bullish (price lower-low + MFI higher-low) and bearish
mirror, plus the ``div_min_gap`` requirement and edge cases.
"""

from __future__ import annotations

import numpy as np

from src.bt.strategies.mfi_divergence_dsl import (
    Params,
    _bear_divergence,
    _bull_divergence,
    _cooling,
)


class _V:
    def __init__(self, a) -> None:
        self._a = np.asarray(a, dtype=float)

    def to_array(self) -> np.ndarray:
        return self._a


class _O:
    def __init__(self, close) -> None:
        self.close = _V(close)


def _p(
    *,
    warmup_bars: int = 2,
    div_look: int = 6,
    div_min_gap: float = 8.0,
    div_min_price_ext: float = 0.0,
) -> Params:
    return Params(
        warmup_bars=warmup_bars,
        div_look=div_look,
        div_min_gap=div_min_gap,
        div_min_price_ext=div_min_price_ext,
    )


def test_bull_divergence_fires_on_lower_low_higher_low() -> None:
    # 6-bar window split 3|3. prior price low 10, recent price low 9 (lower).
    # prior MFI low 20, recent MFI low 35 (higher by > gap).
    close = [11.0, 10.0, 11.0, 12.0, 9.0, 10.0]
    mfi = np.array([30.0, 20.0, 30.0, 40.0, 35.0, 45.0])
    assert _bull_divergence(_O(close), _p(), mfi, len(close)) is True


def test_bull_divergence_rejects_when_mfi_lower_low() -> None:
    close = [11.0, 10.0, 11.0, 12.0, 9.0, 10.0]
    mfi = np.array([30.0, 20.0, 30.0, 40.0, 18.0, 45.0])  # recent MFI low 18 < prior 20
    assert _bull_divergence(_O(close), _p(), mfi, len(close)) is False


def test_bull_divergence_rejects_when_price_not_lower_low() -> None:
    close = [11.0, 8.0, 11.0, 12.0, 10.0, 13.0]  # recent price low 10 > prior 8
    mfi = np.array([30.0, 20.0, 30.0, 40.0, 35.0, 45.0])
    assert _bull_divergence(_O(close), _p(), mfi, len(close)) is False


def test_bull_divergence_gap_threshold_matters() -> None:
    close = [11.0, 10.0, 11.0, 12.0, 9.0, 10.0]
    mfi = np.array([30.0, 20.0, 30.0, 40.0, 25.0, 45.0])  # lead = 5
    assert _bull_divergence(_O(close), _p(div_min_gap=4.0), mfi, len(close)) is True
    assert _bull_divergence(_O(close), _p(div_min_gap=8.0), mfi, len(close)) is False


def test_bull_divergence_short_history_returns_false() -> None:
    o = _O([10.0, 9.0])
    mfi = np.array([20.0, 30.0])
    assert _bull_divergence(o, _p(div_look=6), mfi, 2) is False


def test_bull_divergence_nan_returns_false() -> None:
    close = [11.0, 10.0, 11.0, 12.0, 9.0, np.nan]
    mfi = np.array([30.0, 20.0, 30.0, 40.0, 35.0, 45.0])
    assert _bull_divergence(_O(close), _p(), mfi, len(close)) is False


def test_bear_divergence_fires_on_higher_high_lower_high() -> None:
    # mirror: recent price high above prior, recent MFI high below prior.
    # window split 3|3: prior highs 20/80, recent highs 22/65 (>20, <80).
    close = [10.0, 20.0, 9.0, 8.0, 22.0, 12.0]
    mfi = np.array([50.0, 80.0, 45.0, 40.0, 65.0, 35.0])
    assert _bear_divergence(_O(close), _p(), mfi, len(close)) is True


def test_bear_divergence_rejects_without_price_higher_high() -> None:
    close = [10.0, 20.0, 9.0, 8.0, 12.0, 11.0]  # recent high 12 < prior 20
    mfi = np.array([50.0, 80.0, 45.0, 40.0, 65.0, 35.0])
    assert _bear_divergence(_O(close), _p(), mfi, len(close)) is False


# ---- re-entry cooldown ----------------------------------------------------


def test_cooldown_blocks_within_window() -> None:
    # exited at bar 10, 5-bar cooldown -> bars 10..14 blocked
    assert _cooling(10, 10, 5) is True
    assert _cooling(13, 10, 5) is True


def test_cooldown_releases_after_window() -> None:
    assert _cooling(15, 10, 5) is False
    assert _cooling(20, 10, 5) is False


def test_cooldown_disabled_when_zero_or_negative() -> None:
    assert _cooling(11, 10, 0) is False
    assert _cooling(11, 10, -3) is False


def test_cooldown_off_when_never_held() -> None:
    assert _cooling(100, -1, 20) is False


# ---- price-extension gate -------------------------------------------------


def test_bull_divergence_price_ext_rejects_shallow_low() -> None:
    # recent low 9 vs prior 10 -> extension only (10-9)/10 = 10%
    close = [11.0, 10.0, 11.0, 12.0, 9.0, 10.0]
    mfi = np.array([30.0, 20.0, 30.0, 40.0, 35.0, 45.0])
    assert _bull_divergence(_O(close), _p(div_min_price_ext=0.05), mfi, len(close)) is True
    assert _bull_divergence(_O(close), _p(div_min_price_ext=0.20), mfi, len(close)) is False


def test_bull_divergence_price_ext_off_by_default() -> None:
    close = [11.0, 10.0, 11.0, 12.0, 9.999, 10.0]  # barely lower low
    mfi = np.array([30.0, 20.0, 30.0, 40.0, 35.0, 45.0])
    assert _bull_divergence(_O(close), _p(), mfi, len(close)) is True


def test_bear_divergence_price_ext_rejects_shallow_high() -> None:
    # recent high 22 vs prior 20 -> extension (22-20)/20 = 10%
    close = [10.0, 20.0, 9.0, 8.0, 22.0, 12.0]
    mfi = np.array([50.0, 80.0, 45.0, 40.0, 65.0, 35.0])
    assert _bear_divergence(_O(close), _p(div_min_price_ext=0.05), mfi, len(close)) is True
    assert _bear_divergence(_O(close), _p(div_min_price_ext=0.20), mfi, len(close)) is False
