"""Unit tests for the pure helpers in ``pullback_freshlow_dsl``.

No DB, no engine — the OhlcvView is faked with numpy arrays. Covers the
fresh-low + drying-volume trigger, the volume-dryness check, and edge cases
(empty/short windows, all-NaN, zero volume baseline).
"""

from __future__ import annotations

import numpy as np

from src.bt.strategies.pullback_freshlow_dsl import Params, _fresh_pullback, _vol_dry


class _V:
    def __init__(self, a) -> None:
        self._a = np.asarray(a, dtype=float)

    def to_array(self) -> np.ndarray:
        return self._a


class _O:
    def __init__(self, close, volume) -> None:
        self.close = _V(close)
        self.volume = _V(volume)


def _p(
    *,
    warmup_bars: int = 2,
    fresh_look: int = 3,
    leg_look: int = 4,
    vol_dry: float = 1.0,
) -> Params:
    return Params(
        warmup_bars=warmup_bars,
        fresh_look=fresh_look,
        leg_look=leg_look,
        vol_dry=vol_dry,
    )


def test_fresh_pullback_fires_on_fresh_low_and_dry_volume() -> None:
    # prior closes 10,11,12 then a fresh low 9, on dried volume (5 vs ~20 mean).
    close = [10.0, 11.0, 12.0, 9.0]
    vol = [20.0, 20.0, 20.0, 5.0]
    o = _O(close, vol)
    assert _fresh_pullback(o, _p(), n=4) is True


def test_fresh_pullback_rejects_when_not_a_fresh_low() -> None:
    # today 10 is above the min of the prior fresh_look window (10) -> not fresh.
    close = [8.0, 11.0, 12.0, 10.0]
    vol = [20.0, 20.0, 20.0, 5.0]  # volume is dry, so only the low must fail
    o = _O(close, vol)
    assert _fresh_pullback(o, _p(), n=4) is False


def test_fresh_pullback_rejects_on_wet_volume() -> None:
    # fresh low but volume above the leg mean -> rejected.
    close = [10.0, 11.0, 12.0, 9.0]
    vol = [20.0, 20.0, 20.0, 25.0]
    o = _O(close, vol)
    assert _fresh_pullback(o, _p(), n=4) is False


def test_fresh_pullback_short_history_returns_false() -> None:
    o = _O([9.0, 8.0], [5.0, 5.0])
    assert _fresh_pullback(o, _p(), n=2) is False


def test_fresh_pullback_nan_prior_close_returns_false() -> None:
    close = [np.nan, 11.0, 12.0, 9.0]
    vol = [20.0, 20.0, 20.0, 5.0]
    o = _O(close, vol)
    assert _fresh_pullback(o, _p(), n=4) is False


def test_vol_dry_true_and_false() -> None:
    o = _O([1.0, 1.0, 1.0, 1.0], [10.0, 10.0, 10.0, 9.0])
    assert _vol_dry(o, _p(leg_look=3), n=4) is True
    o2 = _O([1.0, 1.0, 1.0, 1.0], [10.0, 10.0, 10.0, 11.0])
    assert _vol_dry(o2, _p(leg_look=3), n=4) is False


def test_vol_dry_zero_baseline_returns_false() -> None:
    o = _O([1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0])
    assert _vol_dry(o, _p(leg_look=3), n=4) is False


def test_vol_dry_nan_leg_returns_false() -> None:
    o = _O([1.0, 1.0, 1.0, 1.0], [np.nan, 10.0, 10.0, 5.0])
    assert _vol_dry(o, _p(leg_look=3), n=4) is False
