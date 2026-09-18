"""Divergence geometry: pivot-pair detection over a price + oscillator pair.

A divergence is **wave geometry**, not an aggregate comparison of two halves of
a window. The formal definition used here:

  * Two same-type swing pivots ``P1`` (older) and ``P2`` (newer), each a local
    extreme over ``±pivot_bars`` bars (a fractal), with the right side of the
    window fully behind the cursor so nothing is confirmed early.
  * ``P2`` is more extreme than ``P1`` **in price**
    (bullish: lower low; bearish: higher high).
  * The oscillator is *less* extreme at ``P2`` (bullish: higher low; bearish:
    lower high), by at least ``min_osc_gap``.
  * An **intervening opposite pivot** sits strictly between them. With a
    single fractal width this is *implied* by ``find_pivots`` -- two same-type
    pivots of width ``k`` are always separated by an opposite pivot -- so
    ``require_intervening`` is an invariant assertion rather than extra
    filtering. It earns its keep as an explicit, testable guarantee of the
    wave shape, and as the hook for ``min_intervening_atr``, which does add
    real filtering (a two-bar twitch is not a bounce).
  * The leg ages are bounded: ``P1..P2`` separation within
    ``[min_gap_bars, max_gap_bars]`` kills same-leg pairs and stale pivots.
  * ``P2`` is recent (``max_age_bars``) so the signal is tradeable.
  * Price penetration of ``P1`` is inside a band ``[min_price_ext,
    max_price_ext]``: a marginal fresh extreme is a divergence, a violent
    breakdown is a regime change, not absorption.
  * The oscillator is scale-normalised: the gap is either an absolute point
    floor or a fraction of the oscillator's own lookback range.

Everything here is pure: numpy in, a frozen result out. No engine, no state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Side = Literal["long", "short"]


@dataclass(frozen=True)
class Pivot:
    """A confirmed fractal swing point."""

    idx: int  # absolute bar index into the full series
    price: float  # close at the pivot
    osc: float  # oscillator value at the pivot


@dataclass(frozen=True)
class DivergenceParams:
    """Geometry knobs. Defaults are conservative; every gate can be disabled.

    ``0`` disables most ratio/band gates; ``min_gap_bars`` and ``pivot_bars``
    cannot be zero (they define the shape itself).
    """

    pivot_bars: int = 3  # bars of strictly-lower/higher closes each side
    lookback: int = 90  # bars of history scanned for pivots
    min_gap_bars: int = 5  # minimum P1 -> P2 separation
    max_gap_bars: int = 0  # maximum separation (0 = derive from lookback)
    max_age_bars: int = 0  # P2 must be this recent (0 = derive from lookback)
    min_osc_gap: float = 3.0  # absolute oscillator improvement required
    min_osc_gap_frac: float = 0.0  # osc improvement as fraction of osc range
    require_intervening: bool = True  # assert the opposite pivot exists (see note)
    min_intervening_atr: float = 0.0  # retrace depth of that pivot, in ATRs
    min_price_ext: float = 0.0  # min penetration of P1 by P2 (fraction)
    max_price_ext: float = 0.0  # max penetration (0 = unbounded)
    osc_extreme_side: bool = False  # require P1 in the oversold/overbought tail
    osc_extreme_floor: float = 35.0  # bullish: P1.osc <= floor
    osc_extreme_ceil: float = 65.0  # bearish: P1.osc >= ceil


def find_pivots(
    prices: np.ndarray,
    osc: np.ndarray,
    *,
    pivot_bars: int,
    high: bool,
) -> list[Pivot]:
    """Fractal pivots of ``prices``, tagged with the ``osc`` value there.

    ``high=True`` finds swing highs (``prices[i]`` strictly greater than the
    ``pivot_bars`` bars either side); ``high=False`` finds swing lows. Only
    pivots whose right window is fully inside the array are returned, so a
    caller at bar ``n`` never sees a pivot confirmed by future data.

    Non-finite prices or oscillator values anywhere in a candidate's window
    disqualify it (fails closed).
    """
    if pivot_bars < 1:
        return []
    m = prices.size
    if osc.size != m or m < 2 * pivot_bars + 1:
        return []

    out: list[Pivot] = []
    for pv in _pivot_indices(prices, pivot_bars, high=high):
        if not np.isfinite(osc[pv]):
            continue
        out.append(Pivot(idx=pv, price=float(prices[pv]), osc=float(osc[pv])))
    return out


def _pivot_indices(prices: np.ndarray, pivot_bars: int, *, high: bool) -> list[int]:
    """Vectorised fractal indices: bar strictly extreme over ``±pivot_bars``.

    ``O(n)`` numpy, not a Python loop over bars -- this is the hot path when
    called per bar. Non-finite centres are excluded; a window containing a NaN
    cannot confirm a pivot (comparisons against NaN are False, so ``np.all``
    fails closed for free).
    """
    m = prices.size
    if pivot_bars < 1 or m < 2 * pivot_bars + 1:
        return []
    cand = np.arange(pivot_bars, m - pivot_bars)
    if cand.size == 0:
        return []
    centres = prices[cand]
    # (cand.size, k) window matrix: left block then right block
    left = np.stack([prices[cand - 1 - r] for r in range(pivot_bars)], axis=1)
    right = np.stack([prices[cand + 1 + r] for r in range(pivot_bars)], axis=1)
    if high:
        ok = (centres > left.max(axis=1)) & (centres > right.max(axis=1))
    else:
        ok = (centres < left.min(axis=1)) & (centres < right.min(axis=1))
    ok &= np.isfinite(centres)
    return [int(i) for i in cand[ok]]


def _osc_range(osc: np.ndarray) -> float:
    """Peak-to-trough range of the finite oscillator values (0.0 if degenerate)."""
    finite = osc[np.isfinite(osc)]
    if finite.size == 0:
        return 0.0
    span = float(np.max(finite)) - float(np.min(finite))
    return span if span > 0 else 0.0


class PivotScanner:
    """Incrementally confirmed fractal pivots for one ``(symbol, side)``.

    Pivots are append-only: a bar becomes a confirmed pivot once ``pivot_bars``
    bars have closed to its right, and its status never changes afterwards. So
    the scanner only evaluates newly-completable bars and caches the rest,
    making the per-bar cost ``O(pivot_bars)`` instead of ``O(lookback)``.

    The caller owns the instance, feeds bars in order, and must hand in the
    cursor-truncated series. A shorter series than last time means a fresh run,
    so the cache resets itself.
    """

    __slots__ = ("_pivot_bars", "_high", "_seen", "_prices", "_pivots")

    def __init__(self, pivot_bars: int, high: bool) -> None:
        self._pivot_bars = pivot_bars
        self._high = high
        self._seen = 0  # bars already evaluated for confirmation
        self._prices: list[float] = []  # every price seen, in order
        self._pivots: list[Pivot] = []

    @property
    def pivots(self) -> list[Pivot]:
        return self._pivots

    @property
    def pivot_bars(self) -> int:
        return self._pivot_bars

    @property
    def high(self) -> bool:
        return self._high

    def extend(self, prices: np.ndarray, osc: np.ndarray) -> None:
        """Scan any newly-completable bars and cache confirmed pivots."""
        m = prices.size
        if osc.size != m or self._pivot_bars < 1:
            return
        if m < self._seen or m < len(self._prices):
            self._seen = 0
            self._prices = []
            self._pivots = []
        if m == self._seen:
            return
        if not self._prices:
            self._prices = [float(v) for v in prices]
        elif m > len(self._prices):
            self._prices.extend(float(v) for v in prices[len(self._prices) :])

        k = self._pivot_bars
        last = m - k  # bars strictly before this can be confirmed
        if last <= self._seen:
            self._seen = max(self._seen, last)
            return
        lo = max(self._seen, k)
        seg = np.asarray(self._prices[lo - k : last + k + 1], dtype=float)
        for rel in _pivot_indices(seg, k, high=self._high):
            i = lo + rel - k
            if np.isfinite(osc[i]):
                self._pivots.append(Pivot(i, float(self._prices[i]), float(osc[i])))
        self._seen = last


def _penetration(p1: float, p2: float, side: Side) -> float:
    """Price extension of ``P2`` beyond ``P1``, as a positive fraction.

    Bullish (lower low): ``(P1 - P2) / P1``. Bearish (higher high):
    ``(P2 - P1) / P1``. Returns a negative number when ``P2`` fails to extend.
    """
    if p1 <= 0:
        return float("-inf")
    if side == "long":
        return (p1 - p2) / p1
    return (p2 - p1) / p1


def detect_divergence(
    prices: np.ndarray,
    osc: np.ndarray,
    side: Side,
    params: DivergenceParams,
    *,
    atr: float = 0.0,
) -> Pivot | None:
    """Return the confirming newer pivot ``P2`` when a divergence is present.

    Convenience wrapper that rescans the window on every call -- correct but
    ``O(lookback)`` per bar. Hot paths (a strategy firing once per bar per
    symbol) should drive :class:`PivotScanner` instead and call
    :func:`evaluate_from_pivots`.

    ``prices`` and ``osc`` must already be cursor-truncated to the current bar
    (the last element is "now"). ``atr`` is optional and only consulted by the
    ``min_intervening_atr`` gate. Returns ``None`` when geometry does not hold.
    """
    p = params
    m = prices.size
    if p.pivot_bars < 1 or p.min_gap_bars < 1 or osc.size != m:
        return None
    if m < 2 * p.pivot_bars + 1:
        return None

    want_high = side == "short"
    return evaluate_from_pivots(
        find_pivots(prices, osc, pivot_bars=p.pivot_bars, high=want_high),
        find_pivots(prices, osc, pivot_bars=p.pivot_bars, high=not want_high),
        prices,
        osc,
        side,
        p,
        atr=atr,
    )


def evaluate_from_pivots(
    pivots: list[Pivot],
    opposite: list[Pivot],
    prices: np.ndarray,
    osc: np.ndarray,
    side: Side,
    p: DivergenceParams,
    *,
    atr: float = 0.0,
) -> Pivot | None:
    """Apply every divergence gate to pre-computed pivot lists.

    Pure and ``O(1)`` in the series length, so a strategy can maintain
    :class:`PivotScanner` pair and call this per bar without rescanning. The
    ``opposite`` list (the other pivot type) supplies the wave requirement.
    """
    if len(pivots) < 2:
        return None
    m = prices.size
    look = min(p.lookback if p.lookback > 0 else m, m)
    base = m - look
    seg_osc = osc[base:]

    p1, p2 = pivots[-2], pivots[-1]
    if p1.idx < base or p2.idx < base:
        return None

    # -- leg timing ----------------------------------------------------------
    gap = p2.idx - p1.idx
    if gap < p.min_gap_bars:
        return None
    max_gap = p.max_gap_bars if p.max_gap_bars > 0 else look
    if gap > max_gap:
        return None
    max_age = p.max_age_bars if p.max_age_bars > 0 else max_gap
    if (m - 1 - p2.idx) > max_age:
        return None

    # -- price geometry: P2 takes out P1 in the traded direction -------------
    ext = _penetration(p1.price, p2.price, side)
    if not np.isfinite(ext) or ext < p.min_price_ext:
        return None
    if p.max_price_ext > 0 and ext > p.max_price_ext:
        return None

    # -- oscillator geometry: P2 is less extreme -----------------------------
    improvement = p2.osc - p1.osc if side == "long" else p1.osc - p2.osc
    need = p.min_osc_gap
    if p.min_osc_gap_frac > 0:
        need = max(need, p.min_osc_gap_frac * _osc_range(seg_osc))
    if not (improvement >= need):
        return None

    # -- optional: P1 must sit in the oscillator's own tail ------------------
    if p.osc_extreme_side:
        if side == "long" and p1.osc > p.osc_extreme_floor:
            return None
        if side == "short" and p1.osc < p.osc_extreme_ceil:
            return None

    # -- the wave requirement: an opposite pivot between the two -------------
    # With a single fractal width this is structurally guaranteed; the branch
    # verifies it and hosts the min_intervening_atr depth gate.
    if p.require_intervening:
        inside = [q for q in opposite if p1.idx < q.idx < p2.idx]
        if not inside:
            return None
        if p.min_intervening_atr > 0 and atr > 0:
            # Long: the intervening peak must retrace at least this far above
            # the prior low (a real bounce). Short: mirror on the intervening
            # trough below the prior high.
            deepest = (
                max(q.price for q in inside)
                if side == "long"
                else min(q.price for q in inside)
            )
            if abs(deepest - p1.price) < p.min_intervening_atr * atr:
                return None

    return p2
