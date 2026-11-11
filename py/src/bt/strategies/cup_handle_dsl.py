"""Cup and Handle breakout — DSL-native, self-contained port.

A pure geometric detector on confirmed swing pivots (true base low, symmetric
base, rim-matched rims), a breakout that must be the bar that *crosses* the rim
on expanding volume, and a chandelier exit.

* The detector (``detect_cup_and_handle``) and the trend gate
  (``is_bull_market`` / ``uptrend_aligned``) are inlined so the module is
  self-contained; shape invariants are module constants, not parameters.
* Data is read cursor-safe through ``ctx.ohlcv`` (cursor-truncated arrays ->
  ``pd.Series`` for the pure detector) and ``ctx.ta.atr`` (one full-series ATR
  compute, O(1) per-candle reads).
* Cross-candle state (cooldowns, swings, chandelier highs/stops) lives in
  ``ctx.shared`` via ``@strategy(stateful=True)`` — the DSL replacement for a
  module-level ``GLOBAL`` dict.

Sizing / exit semantics (DSL convention):

* ``ctx.long(...)`` omits ``size`` so the engine's **shared sizing layer**
  (``SizingParams``: ``sizing_mode`` / ``size`` / ``max_symbol_allocation``)
  derives the share count; the strategy bounds its stop distance to the risk
  budget via ``cap_stop_dist`` against that same equity-scaled notional.
* ``ctx.long(sl=)`` takes an ABSOLUTE price (same convention as
  ``ctx.set_stops``). No take-profit is armed: a level fixed at entry caps the
  advance the pattern exists to capture, so the chandelier owns the exit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import src.indicators.ta as ta

from src.bt.strategies.dsl import strategy, StrategyContext
from src.bt.strategies.fundamentals_context import SeriesPIT
from src.bt.strategies.types import StrategyParams


STRATEGY_TYPE = "cup_handle_dsl"


# ---------------------------------------------------------------------------
# pure geometric detector + trend helpers (cursor-safe inputs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Swing:
    """A single turning point on the price series."""

    idx: int
    high: bool  # True = pivot high, False = pivot low
    level: float


@dataclass(frozen=True)
class Handle:
    """A recognized handle off a cup's right rim."""

    start_idx: int  # pivot-high index = the resistance/breakout line
    low_idx: int  # index of the handle's lowest bar
    low: float
    breakout_level: float  # sell above this
    cup_depth: float  # absolute price depth of the parent cup
    rim_level: float  # right-rim price (for reference / target calc)
    cup_left_idx: int  # left-rim index (volume baseline for the breakout)


@dataclass(frozen=True)
class CupHandleResult:
    cup_ok: bool
    handle_ok: bool
    entry_ok: bool
    handle: Optional[Handle]
    reason: str


def cap_stop_dist(
    natural_dist: float,
    risk_cap_dollars: float,
    qty: float,
    *,
    atr_val: float,
    min_stop_atr: float = 0.5,
) -> float:
    """Bounded stop distance so a long's stop-out can never over-lose."""
    cap_dist = risk_cap_dollars / qty if qty > 0 else float("inf")
    dist = min(natural_dist, cap_dist)
    return max(dist, min_stop_atr * atr_val)


def is_bull_market(
    closes: pd.Series,
    *,
    ma_span: int = 200,
    slope_span: int = 20,
    min_slope_frac: float = 0.0,
) -> bool:
    """Is the broad market in a bullish regime (price above a long MA)?"""
    if len(closes) < ma_span + slope_span:
        return False
    ma = ta.sma(closes, ma_span)
    last_ma = float(ma.iloc[-1])
    if last_ma != last_ma or last_ma <= 0:  # NaN guard
        return False
    if min_slope_frac > 0:
        past_ma = float(ma.iloc[-slope_span - 1])
        if past_ma != past_ma:  # NaN guard
            return False
        slope = (last_ma - past_ma) / past_ma
        if slope < min_slope_frac:
            return False
    return float(closes.iloc[-1]) > last_ma


def uptrend_aligned(
    sym_close: pd.Series,
    mkt_close: pd.Series,
    *,
    sym_span: int = 100,
    mkt_span: int = 200,
) -> bool:
    """Both symbol and market proxy close above their own long MA."""
    return is_bull_market(
        sym_close, ma_span=sym_span, slope_span=20, min_slope_frac=0.0
    ) and is_bull_market(mkt_close, ma_span=mkt_span, slope_span=20, min_slope_frac=0.0)


def _find_swings(highs: pd.Series, lows: pd.Series, lookback: int) -> list[Swing]:
    """Return pivot highs/lows using a symmetric fractal window."""
    n = len(highs)
    if n < 2 * lookback + 1:
        return []

    swings: list[Swing] = []
    highs_np = highs.to_numpy()
    lows_np = lows.to_numpy()
    for i in range(lookback, n - lookback):
        lo, hi = i - lookback, i + lookback
        hi_max = float(highs_np[lo:i].max()) if lo < i else float(highs_np[i])
        hi_max = max(
            hi_max,
            float(highs_np[i + 1 : hi + 1].max()) if i + 1 <= hi else -float("inf"),
        )
        lo_min = float(lows_np[lo:i].min()) if lo < i else float(lows_np[i])
        lo_min = min(
            lo_min,
            float(lows_np[i + 1 : hi + 1].min()) if i + 1 <= hi else float("inf"),
        )
        if highs_np[i] > hi_max:
            swings.append(Swing(i, True, float(highs_np[i])))
        elif lows_np[i] < lo_min:
            swings.append(Swing(i, False, float(lows_np[i])))
    return swings


def _swings_incremental(
    highs: pd.Series,
    lows: pd.Series,
    lookback: int,
    entry: tuple[int, list[Swing]] | None,
) -> tuple[int, list[Swing]]:
    """Swing list for the current series, re-derived incrementally from an
    earlier result for a shorter (or equal-length) series.

    Returns ``(series_len, swings)`` describing the *current* series so the
    caller can feed it back on the next bar. Output is guaranteed identical to
    a cold ``_find_swings(highs, lows, lookback)`` over the current series.

    Fractal invariant that makes this exact: ``_find_swings`` only decides a
    swing at index ``i`` once its full window ``[i - lookback, i + lookback]``
    lies inside the series (loop runs to ``n - lookback``). Once a swing with
    right edge ``i + lookback <= n_prev - 1`` is decided it can never change
    when bars are appended — no later bar enters that window. So an earlier
    result computed for ``n_prev`` bars is immutable up to index
    ``n_prev - lookback - 1``; everything from index ``max(lookback, n_prev -
    lookback)`` onward is re-derived (only the frontier actually needs it —
    appending adds exactly one decidable index per bar). ``None`` (no prior)
    falls back to a cold full scan.
    """
    n = len(highs)
    if n < 2 * lookback + 1:
        return (n, [])
    if entry is not None:
        prev_n, prev = entry
        if prev_n == n:
            return (n, list(prev))  # exact repeat within the same bar
        if prev_n < 2 * lookback + 1 or n < prev_n:
            return (n, _find_swings(highs, lows, lookback))
        kept = [s for s in prev if s.idx < prev_n - lookback]
        start = max(lookback, prev_n - lookback)
    else:
        kept = []
        start = lookback
    out = list(kept)
    highs_np = highs.to_numpy()
    lows_np = lows.to_numpy()
    for i in range(start, n - lookback):
        lo, hi = i - lookback, i + lookback
        hi_max = float(highs_np[lo:i].max()) if lo < i else float(highs_np[i])
        hi_max = max(
            hi_max,
            float(highs_np[i + 1 : hi + 1].max()) if i + 1 <= hi else -float("inf"),
        )
        lo_min = float(lows_np[lo:i].min()) if lo < i else float(lows_np[i])
        lo_min = min(
            lo_min,
            float(lows_np[i + 1 : hi + 1].min()) if i + 1 <= hi else float("inf"),
        )
        if highs_np[i] > hi_max:
            out.append(Swing(i, True, float(highs_np[i])))
        elif lows_np[i] < lo_min:
            out.append(Swing(i, False, float(lows_np[i])))
    return (n, out)


# Structural geometry constants. Shape invariants of the pattern, not fitted
# knobs, so they are hard-coded instead of exposed as parameters.
MIN_CUP_BARS = 20  # a base narrower than this is a flag, not a cup
MIN_BASE_POS = 0.25  # the base low must sit in the middle half of the base
HANDLE_MAX_WIDTH_FRAC = 0.5  # handle width <= half the cup width
HANDLE_MAX_DEPTH_FRAC = 0.5  # handle retrace <= half the cup depth
BREAKOUT_VOLUME_MULT = 1.5  # breakout volume vs the cup-body average
VIX_WINDOW = 200  # prints in the volatility-regime baseline
VIX_MIN_PRINTS = 60  # fewer visible prints -> the gate cannot be read
ANNUAL_MIN_DAYS = 300  # span length that marks a full-year fundamental


def _lowest_low(swings: Sequence[Swing], start: int, stop: int) -> Optional[Swing]:
    """Lowest pivot low strictly between two swing positions, or ``None``."""
    lowest: Optional[Swing] = None
    for b in range(start, stop):
        swing = swings[b]
        if swing.high:
            continue
        if lowest is None or swing.level < lowest.level:
            lowest = swing
    return lowest


def _check_cup(
    seg_low: float,
    left: Swing,
    bottom: Swing,
    right: Swing,
    *,
    max_cup_depth_pct: float,
    min_cup_depth_pct: float,
    rim_tolerance_pct: float,
) -> bool:
    """Validate a cup's geometry between a left rim, a bottom low and a right rim.

    Three invariants the depth/rim gates alone did not catch:

    * **the base low is the TRUE low of the base.** A pivot low that merely sits
      some percentage under the left rim is a shelf on the way down, not a cup
      bottom. Accepting one declares a fictional depth and defeats
      ``max_cup_depth_pct`` -- the real decline can be anything.
    * **the base is wide enough** to be a base at all.
    * **the base low is roughly centred**, so a cliff-then-crawl or a
      ramp-then-spike is not read as a cup.
    """
    if not (left.idx < bottom.idx < right.idx):
        return False
    span = right.idx - left.idx
    if span < MIN_CUP_BARS:
        return False
    if bottom.level > seg_low + 1e-9:
        return False
    depth = (left.level - bottom.level) / left.level
    if not (min_cup_depth_pct <= depth <= max_cup_depth_pct):
        return False
    if abs(right.level - left.level) / left.level > rim_tolerance_pct:
        return False
    pos = (bottom.idx - left.idx) / span
    return MIN_BASE_POS <= pos <= 1.0 - MIN_BASE_POS


def _find_handle(
    closes_np: np.ndarray,
    lows_np: np.ndarray,
    right: Swing,
    n: int,
    *,
    cup_left_idx: int,
    cup_depth: float,
    max_handle_bars: int,
    max_handle_drop_pct: float,
) -> Optional[Handle]:
    """Locate the pullback (handle) that coils under the right rim before now.

    Measured on raw bars over ``[rim + 2, n - 1)`` -- the current bar is the
    breakout candidate and is excluded, so a post-breakout pivot low can never
    be mistaken for a handle. Rejected when:

    * a close inside the window is at or above the rim (the coil is over),
    * there is no pullback at all (the low is the rim itself), or
    * the retrace exceeds half the cup depth or ``max_handle_drop_pct`` -- a
      handle may not cut into the lower half of the base.

    Width is capped by ``max_handle_bars`` and by half the parent cup's width.
    """
    if cup_depth <= 0 or right.level <= 0:
        return None
    cup_bars = max(1, n - 1 - cup_left_idx)
    window = min(max_handle_bars, max(2, int(HANDLE_MAX_WIDTH_FRAC * cup_bars)))
    start = right.idx + 2
    end = min(right.idx + window, n - 1)
    if end <= start:
        return None
    if float(closes_np[start:end].max()) > right.level:
        return None  # already back above resistance: nothing coiled
    lo_seg = lows_np[start:end]
    lo_idx = start + int(lo_seg.argmin())
    lo_level = float(lo_seg.min())
    retrace = right.level - lo_level
    if retrace <= 0:
        return None  # no pullback off the rim -> nothing to call a handle
    if retrace > HANDLE_MAX_DEPTH_FRAC * cup_depth:
        return None  # handle sliced into the lower half of the base
    if retrace / right.level > max_handle_drop_pct:
        return None
    return Handle(
        start_idx=right.idx,
        low_idx=lo_idx,
        low=lo_level,
        breakout_level=right.level,
        cup_depth=cup_depth,
        rim_level=right.level,
        cup_left_idx=cup_left_idx,
    )


def _volume_picks_up(volumes: pd.Series, left_idx: int, n: int) -> bool:
    """Breakout bar volume should expand over the cup-body average."""
    vol = volumes.to_numpy()
    if n < 2 or vol[-1] != vol[-1]:  # NaN guard
        return False
    base = vol[left_idx : n - 1]
    if len(base) == 0:
        return False
    avg = float(np.mean(base))
    return avg > 0 and float(vol[-1]) >= BREAKOUT_VOLUME_MULT * avg


def detect_cup_and_handle(
    highs: pd.Series,
    lows: pd.Series,
    closes: pd.Series,
    volumes: Optional[pd.Series],
    *,
    swing_lookback: int = 3,
    max_cup_depth_pct: float = 0.40,
    min_cup_depth_pct: float = 0.10,
    rim_tolerance_pct: float = 0.05,
    max_cup_bars: int = 260,
    max_handle_bars: int = 40,
    max_handle_drop_pct: float = 0.12,
    volume_confirm_breakout: bool = True,
    # Incremental hook: precomputed swing list for the FULL history ``highs``
    # (valid at this exact length). When supplied the O(n) re-fractal scan is
    # skipped; when omitted the detector falls back to a fresh ``_find_swings``
    # so its pure-function contract is unchanged.
    swings: Optional[Sequence[Swing]] = None,
) -> CupHandleResult:
    """Detect a cup-and-handle formation ending at (breaking on) the last bar."""
    n = len(highs)
    if not (len(lows) == n and len(closes) == n):
        return CupHandleResult(False, False, False, None, "length mismatch")
    if volumes is not None and len(volumes) != n:
        return CupHandleResult(False, False, False, None, "volume length mismatch")
    if n < 2 * swing_lookback + 1:
        return CupHandleResult(False, False, False, None, "insufficient data")

    if swings is None:
        swings = _find_swings(highs, lows, swing_lookback)
    closes_np = closes.to_numpy()
    last_close = float(closes_np[-1])

    best: Optional[Handle] = None
    cup_found = False

    lows_np = lows.to_numpy()

    # Iterate candidate right rims from most recent backwards.
    for j in range(len(swings) - 1, -1, -1):
        right = swings[j]
        if not right.high:
            continue
        if right.idx > n - 1 - swing_lookback - 1:
            continue
        # Running raw low over ``[left.idx, right.idx]``. The left rims are
        # visited right-to-left, so each step only absorbs the newly covered
        # slice -- giving the true base low without re-scanning per candidate.
        seg_low = right.level
        seg_from = right.idx + 1
        for i in range(j - 1, -1, -1):
            left = swings[i]
            if left.idx < seg_from:
                seg_low = min(seg_low, float(lows_np[left.idx : seg_from].min()))
                seg_from = left.idx
            if not left.high:
                continue
            if right.idx - left.idx > max_cup_bars:
                break
            bottom = _lowest_low(swings, i + 1, j)
            if bottom is None:
                continue
            if not _check_cup(
                seg_low,
                left,
                bottom,
                right,
                max_cup_depth_pct=max_cup_depth_pct,
                min_cup_depth_pct=min_cup_depth_pct,
                rim_tolerance_pct=rim_tolerance_pct,
            ):
                continue

            cup_found = True
            handle = _find_handle(
                closes_np,
                lows_np,
                right,
                n,
                cup_left_idx=left.idx,
                cup_depth=left.level - bottom.level,
                max_handle_bars=max_handle_bars,
                max_handle_drop_pct=max_handle_drop_pct,
            )
            if handle is None:
                continue
            if best is None or right.idx > best.start_idx:
                best = handle

    if best is None:
        reason = "handle" if cup_found else "no valid cup"
        return CupHandleResult(cup_found, False, False, None, f"no {reason}")

    if last_close <= best.breakout_level:
        return CupHandleResult(True, True, False, best, "no breakout yet")
    # The breakout is the bar that CROSSES the rim, not any later bar in the
    # same advance. Without this the trigger fires on an already-extended move
    # (the cached formation stays valid for as long as price holds above the
    # rim), which turns the entry into a chase.
    if float(closes_np[-2]) > best.breakout_level:
        return CupHandleResult(True, True, False, best, "already above the rim")

    if volume_confirm_breakout and volumes is not None:
        if not _volume_picks_up(volumes, best.cup_left_idx, n):
            return CupHandleResult(True, True, False, best, "no volume confirmation")

    return CupHandleResult(True, True, True, best, "breakout")


@dataclass(frozen=True)
class Params(StrategyParams):
    # Swing detection
    swing_lookback: int = 3
    # Cup geometry
    max_cup_depth_pct: float = 0.40
    min_cup_depth_pct: float = 0.10
    rim_tolerance_pct: float = 0.05
    max_cup_bars: int = 260
    # Handle geometry
    max_handle_bars: int = 40
    max_handle_drop_pct: float = 0.12
    # Volume confirmation
    volume_confirm_breakout: bool = True
    # Confirmations convert to SIZE, not to a filter: a signal is never dropped,
    # it is sized between 1x and (1 + size_boost_max)x by how many enabled gates
    # agree. All off by default (mult stays 1x).
    boost_volume_dryup: bool = False
    boost_vix_calm: bool = False
    boost_fundamentals: bool = False
    size_boost_max: float = 2.0
    # Sizing / risk
    per_symbol_size: float = 0.30
    max_risk_pct: float = 0.02
    min_stop_atr: float = 0.5
    # Exit — chandelier only
    chandelier_period: int = 22
    chandelier_atr_mult: float = 3.0
    # Warmup & cooldown
    warmup_bars: int = 60
    cooldown_bars: int = 5


# ---------------------------------------------------------------------------
# entry confirmations (volume contraction, volatility regime, fundamentals)
# ---------------------------------------------------------------------------


def _volume_contracts(volumes: pd.Series, left_idx: int, rim_idx: int, n: int) -> bool:
    """Handle volume must dry up relative to the cup body.

    The breakout test already demands expansion ON the breakout bar. This is the
    other half of the volume story: the coil under the rim trades quieter than
    the base it came out of. A handle on *rising* volume is distribution, not
    accumulation.
    """
    vol = volumes.to_numpy()
    cup = vol[left_idx : rim_idx + 1]
    handle = vol[rim_idx + 2 : n - 1]
    if len(cup) == 0 or len(handle) == 0:
        return False
    return float(np.mean(handle)) < float(np.mean(cup))


def _vix_calm(ctx: StrategyContext) -> bool:
    """Volatility regime: the latest VIX print at or under its own average.

    Cheap insurance against breaking out into a volatility spike, which is where
    a breakout's failure rate lives. Unreadable (too few visible prints) rejects
    rather than passing silently.
    """
    vix = ctx.macro.vix
    if len(vix) < VIX_MIN_PRINTS:
        return False
    window = vix[-VIX_WINDOW:]
    return vix.last() <= float(np.mean(window))


def _annual_values(series: SeriesPIT, years: int) -> list[float]:
    """Most recent ``years`` full-year values of a cursor-safe fiscal series.

    Filters on span length, not the SEC form string: ``period_start ->
    period_end`` is what decides whether a number is a full year.
    """
    spans = series.spans()
    values = series.values()
    annual = [
        float(value)
        for value, (start, end) in zip(values, spans)
        if (end - start).days >= ANNUAL_MIN_DAYS
    ]
    return annual[-years:]


def _fundamentals_grow(ctx: StrategyContext, sym: str) -> bool:
    """Revenue must be growing year over year on as-first-stated filings.

    Cursor-safe by construction: ``ctx.fundamentals`` hides every filing filed
    after the current bar. Fewer than two annual points means the growth cannot
    be read, which rejects — a confirmation that cannot be evaluated is not a
    confirmation.
    """
    # The statement object is memoised per symbol: building it walks the whole
    # filing history, and it is read for every breakout. Safe to hold across
    # bars because a ``SeriesPIT`` re-derives which periods are visible from the
    # live cursor on every read — the memo caches the rows, never the visibility.
    memo = ctx.shared.setdefault("_fund_revenue", {})
    revenue = memo.get(sym)
    if revenue is None:
        revenue = ctx.fundamentals.income(sym).revenue
        memo[sym] = revenue
    annual = _annual_values(revenue, 2)
    if len(annual) < 2:
        return False
    return annual[-1] > annual[0]


# ---------------------------------------------------------------------------
# data access helpers (cursor-safe; build pd.Series for the pure detector)
# ---------------------------------------------------------------------------


def _arrays(ctx: StrategyContext, sym: str) -> dict[str, pd.Series]:
    """Cursor-truncated OHLCV as pandas Series (matches pure fn inputs)."""
    o = ctx.ohlcv(sym)
    return {
        "close": pd.Series(o.close.to_array()),
        "high": pd.Series(o.high.to_array()),
        "low": pd.Series(o.low.to_array()),
        "volume": pd.Series(o.volume.to_array()),
    }


def _spy_closes(ctx: StrategyContext) -> pd.Series | None:
    """SPY daily closes (broad-market proxy), or None when SPY isn't traded."""
    if "SPY" not in ctx.symbols:
        return None
    arr = ctx.ohlcv("SPY").close.to_array()
    return pd.Series(arr) if len(arr) > 0 else None


def _market_regime_ok(ctx: StrategyContext) -> bool:
    """New entries only when a broad-market bull gate is open (SPY > 200MA)."""
    spy = _spy_closes(ctx)
    return True if spy is None else bool(is_bull_market(spy))


def _symbol_market_aligned(ctx: StrategyContext, closes: pd.Series) -> bool:
    """Symbol + SPY long-MA uptrend alignment (level-based)."""
    spy = _spy_closes(ctx)
    return True if spy is None else bool(uptrend_aligned(closes, spy))


def _atr_val(ctx: StrategyContext, sym: str, period: int) -> float:
    v = float(ctx.ta.atr(sym, period)[-1])
    return v if not np.isnan(v) and v > 0 else 0.0


# ---------------------------------------------------------------------------
# entry / position-management
# ---------------------------------------------------------------------------


def _maybe_enter(ctx: StrategyContext, sym: str, arr: dict[str, pd.Series]) -> None:
    params = ctx.params
    closes = arr["close"]
    if len(closes) < params.warmup_bars:
        return

    # Incremental swing maintenance: the fractal list for this symbol only
    # gains at most one decidable index per appended bar, so re-use the prior
    # bar's swings instead of re-fractaling the whole history. ``detect`` then
    # skips its internal cold ``_find_swings``. ``ctx.shared`` is per-run, so
    # warm-up across bars is safe and nothing bleeds between split/sweep runs.
    # Cheap NECESSARY condition for any breakout, before the detector runs.
    # The rim is a confirmed pivot high at ``<= n - 5``, so bars ``n-4 .. n-2``
    # are all at or right of ``rim + 1``: inside the pivot window or the handle
    # coil, where every close is below the rim level. A breakout bar (the only
    # bar that can trigger an entry) must therefore close above all three. A bar
    # that fails this cannot be a breakout, so nothing is lost by skipping it —
    # and the O(rims x lefts) geometry scan is skipped on most bars.
    if float(closes.iloc[-1]) <= float(arr["close"].iloc[-4:-1].max()):
        return

    cache = ctx.shared.setdefault("_cup_swings", {})
    entry = cache.get(sym)
    _, swings = _swings_incremental(
        arr["high"], arr["low"], params.swing_lookback, entry
    )
    cache[sym] = (len(arr["high"]), swings)

    result = detect_cup_and_handle(
        arr["high"],
        arr["low"],
        closes,
        arr["volume"],
        swing_lookback=params.swing_lookback,
        max_cup_depth_pct=params.max_cup_depth_pct,
        min_cup_depth_pct=params.min_cup_depth_pct,
        rim_tolerance_pct=params.rim_tolerance_pct,
        max_cup_bars=params.max_cup_bars,
        max_handle_bars=params.max_handle_bars,
        max_handle_drop_pct=params.max_handle_drop_pct,
        volume_confirm_breakout=params.volume_confirm_breakout,
        swings=swings,
    )
    if not result.entry_ok or result.handle is None:
        return
    handle = result.handle

    if not _symbol_market_aligned(ctx, closes):
        return

    # Confirmations size the bet instead of gating it: every signal still
    # trades, and the number of agreeing gates scales the notional from 1x up to
    # 1 + size_boost_max. A signal no gate agrees with is a 1x trade, never a
    # dropped one — so the setup count is unchanged and only capital allocation
    # moves.
    gates_enabled = 0
    gates_passed = 0
    if params.boost_volume_dryup:
        gates_enabled += 1
        gates_passed += int(
            _volume_contracts(
                arr["volume"], handle.cup_left_idx, handle.start_idx, len(closes)
            )
        )
    if params.boost_vix_calm:
        gates_enabled += 1
        gates_passed += int(_vix_calm(ctx))
    if params.boost_fundamentals:
        gates_enabled += 1
        gates_passed += int(_fundamentals_grow(ctx, sym))
    size_mult = (
        1.0
        if gates_enabled == 0
        else 1.0 + params.size_boost_max * gates_passed / gates_enabled
    )

    entry_price = float(closes.iloc[-1])

    atr_val = _atr_val(ctx, sym, params.chandelier_period)
    if atr_val <= 0:
        atr_val = max(handle.cup_depth * 0.1, 1e-9)

    # Equity-scaled notional used to bound the stop distance to the risk
    # budget. Mirrors the engine's shared equity sizing (`base*size/price`)
    # so the stop cap tracks the share count the engine will actually fill.
    initial = ctx.state.portfolio.initial_capital
    equity = ctx.state.portfolio.cash  # conservative proxy post-commit
    if equity <= 0 or entry_price <= 0:
        return
    qty = equity * params.per_symbol_size * size_mult / entry_price
    if qty <= 0 or ctx.state.portfolio.cash <= 0:
        return

    natural_dist = max(
        params.chandelier_atr_mult * atr_val, (entry_price - handle.low) * 1.05
    )
    risk_cap_dollars = initial * params.max_risk_pct
    stop_dist = cap_stop_dist(
        natural_dist,
        risk_cap_dollars,
        qty,
        atr_val=atr_val,
        min_stop_atr=params.min_stop_atr,
    )

    # No take-profit: the cup's whole edge is the post-breakout advance, and a
    # level armed at entry caps it. The chandelier (``_manage_position``) owns
    # the exit; ``sl`` is only the initial protective stop.
    sl_abs = entry_price - stop_dist if stop_dist > 0 else None

    ctx.shared["cooldowns"][sym] = params.cooldown_bars
    # No explicit ``size``: the engine's shared sizing layer (``SizingParams``)
    # computes the equity-based share count from ``sizing_mode``/``size``/
    # ``max_symbol_allocation`` in the strategy config.
    ctx.long(
        sym,
        size=params.per_symbol_size * size_mult,
        size_mode="equity",
        sl=sl_abs,
        reason=(
            f"breakout above {handle.breakout_level:.2f} "
            f"(depth={handle.cup_depth:.2f} · {result.reason} · {size_mult:.2f}x)"
        ),
    )


def _manage_position(ctx: StrategyContext, sym: str, arr: dict[str, pd.Series]) -> None:
    """Chandelier exit: trail ``chandelier_atr_mult`` ATRs under the highest high.

    The stop is ``running_high - mult * ATR`` where ``running_high`` is the
    highest high since entry (never the pre-breakout history, so a spike before
    the rim cannot park the stop above the entry price). It ratchets up only and
    is floored by the entry stop the engine already holds. A close at or below
    the stop exits, filled at the next open.

    No ADX branch and no take-profit: one exit rule, so a trend that keeps
    making highs keeps the position — which is the whole point of the pattern.
    """
    pos = ctx.position(sym)
    if pos is None:
        return
    params = ctx.params
    pid = pos.position_id
    close_price = float(arr["close"].iloc[-1])
    high_price = float(arr["high"].iloc[-1])

    atr_val = _atr_val(ctx, sym, params.chandelier_period)
    if atr_val <= 0:
        atr_val = max(close_price * 0.01, 1e-9)

    highs_seen = ctx.shared["trail_highs"]
    run_high = max(highs_seen.get(pid, -float("inf")), high_price)
    highs_seen[pid] = run_high

    candidate = run_high - params.chandelier_atr_mult * atr_val
    seed = float(pos.stop_loss) if pos.stop_loss else -float("inf")
    saved = ctx.shared["trail_stops"].get(pid, -float("inf"))
    trail_stop = max(candidate, seed, saved)
    ctx.shared["trail_stops"][pid] = trail_stop

    if close_price > trail_stop:
        return

    ctx.shared["trail_stops"].pop(pid, None)
    highs_seen.pop(pid, None)
    ctx.shared["cooldowns"][sym] = params.cooldown_bars
    ctx.close(
        sym,
        reason=(
            f"chandelier exit (close {close_price:.2f} <= {trail_stop:.2f}, "
            f"high {run_high:.2f} - {params.chandelier_atr_mult:g}xATR)"
        ),
    )


# ---------------------------------------------------------------------------
# DSL strategy
# ---------------------------------------------------------------------------


@strategy(stateful=True)
def on_candle(ctx: StrategyContext):
    # Warmup bars fill strategy state only — no fill can happen there, so the
    # DSL raises on any entry emitted during them. Accumulators above/below
    # still receive warmup bars because the engine feeds the store regardless.
    if ctx.phase == "warmup":
        return
    shared = ctx.shared
    shared.setdefault("cooldowns", {})
    shared.setdefault("trail_stops", {})
    shared.setdefault("trail_highs", {})

    symbols = sorted(ctx.symbols)

    # Bull-market gate, evaluated lazily: skip the O(n) SMA work entirely when
    # every symbol already holds a position (no possible new entry this bar).
    can_enter = any(ctx.position(sym) is None for sym in symbols)
    bull_ok = True if not can_enter else _market_regime_ok(ctx)

    for sym in symbols:
        if ctx.position(sym) is not None:
            _manage_position(ctx, sym, _arrays(ctx, sym))
            continue

        cd = shared["cooldowns"].get(sym, 0)
        if cd > 0:
            shared["cooldowns"][sym] = cd - 1
            continue

        if not bull_ok:
            continue
        _maybe_enter(ctx, sym, _arrays(ctx, sym))
