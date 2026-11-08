"""Entropy-gated VP breakdown — dynamic ATR trail (DSL).

SHORT-only order-flow breakdown: a close below the rolling Value Area Low on
expanding volume, held near the bar low and not in an uptrend, taken only while
the symbol's adaptive-entropy regime is bearish. Exit on a revert into the value
area or the chandelier ATR trail.

The entropy regime read is the core edge: the same breakdown taken in any
regime is a much weaker book, so ``min_trend`` is a structural bear-only read
rather than a tunable floor. The VIX gate (absolute level against its own
multi-year baseline, never a percentile rank — a rank normalises away the
crash-regime level it must refuse) is optional and off by default; it trades
return for drawdown when enabled.

Lookahead-free: VP, entropy and VIX reads all resolve at or before the cursor.
State lives in ``ctx.shared`` via ``@strategy(stateful=True)``, minted per
run/window, so there is no cross-window bleed and sweeps are worker-safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from src.bt.state.types import Position
from src.indicators.macro._shared import load_daily
from src.indicators.volume_profile.strategy import OnlineVP
from src.indicators.adaptive_entropy import AdaptiveEntropyConfig, OnlineAdaptiveEntropy
from src.bt.size.pure import risk_sized_qty
from src.bt.strategies.dsl import strategy, StrategyContext
from src.bt.strategies.utils import gross_gate
from src.bt.strategies.series import SeriesView
from src.bt.strategies.types import StrategyParams

STRATEGY_TYPE = "entropy_vp_breakdown_dsl"

# Keys for per-run state in ctx.shared (minted fresh per run/window).
_VP_KEY = "vp_profiles"  # dict[sym -> OnlineVP]
_SNAP_KEY = "vp_snaps"  # dict[sym -> VolumeProfileSnapshot] (once/bar)
_ENT_KEY = "entropy"  # dict[sym -> OnlineAdaptiveEntropy]
_ENT_RESULT_KEY = "entropy_result"  # dict[sym -> AdaptiveEntropyResult] (once/bar)
_POS_KEY = "positions"  # dict[sym -> dynamic-trail bookkeeping]
_VIX_CACHE_KEY = "entropy_vix_series"  # tuple[dates, values] for the VIX grid


@dataclass(frozen=True)
class Params(StrategyParams):
    """The 10 load-bearing knobs. Everything else is a structural constant.

    Ablation on the full book showed the rest inert or marginal (a few tenths of
    Sharpe, within noise): the VP geometry, the entropy binning, the warm-up
    length, the initial stop distance and the gross cap are all structural and
    live as module constants below. These ten each move the result materially.
    """

    # -- VP breakdown (the signal) --
    value_area_pct: float = 0.60  # how far the breakdown must clear fair value
    vol_expand_mult: float = 1.3  # breakdown volume >= this x average
    wick_check: float = 0.5  # close must sit in the lower fraction of the bar
    trend_lookback: int = 50  # concurrency: close vs its own rolling mean
    # -- regime gates --
    entropy_lookback: int = 25  # adaptive-entropy window
    max_stretch_atr: float = 2.0  # reject entries already stretched below EMA
    vix_max_ratio: float = 0.0  # VIX / multi-year baseline cap (0 = off)
    # -- exit / risk --
    trail_lookback: int = 10  # chandelier anchor window
    trail_atr_mult: float = 2.0  # chandelier ATR distance
    risk_pct: float = 0.02  # equity risked per trade (ATR-sized)


# ---------------------------------------------------------------------------
# structural constants — fixed, not fitted. Sweeping these is a curve-fit knob.
# ---------------------------------------------------------------------------

_VP_WINDOW = 100  # rolling profile window (bars)
_NUM_BINS = 50  # profile bin count; low sensitivity
_VP_WARMUP = 60  # bars before a profile is ready
_VOL_PERIOD = 20  # volume baseline lookback
_STRETCH_EMA = 20  # short EMA the stretch read is measured against
_ENTROPY_BINS = 10  # adaptive-entropy bin count (inert at 5..10)
MIN_TREND = -1  # bear-only quantised entropy regime — the core edge
ATR_PERIOD = 14  # ATR period for sizing + the trail
STOP_ATR_MULT = 2.0  # sizing stop distance (also the initial trail seed)
MAX_GROSS_EXPOSURE = 0.5  # aggregate |gross| cap vs initial capital
_VIX_BASELINE_YEARS = 5.0  # VIX baseline span


# ---------------------------------------------------------------------------
# pure helpers (typed; operate on cursor-truncated arrays / SeriesViews only)
# ---------------------------------------------------------------------------


def _last(v: object) -> float:
    """Last visible value of a SeriesView, or a bare float."""
    if isinstance(v, SeriesView):
        return float(v[-1])
    return float(cast(float, v))


def _bar_position_close(close: float, high: float, low: float) -> float:
    """Where the close sits within the bar's range, 0 (low) .. 1 (high)."""
    rng = high - low
    if rng <= 1e-12:
        return 1.0
    return (close - low) / rng


def _avg_volume(ctx: StrategyContext, sym: str, params: Params) -> float:
    """Mean volume over the prior ``vol_period`` bars (excl. the current one)."""
    vol = ctx.ohlcv(sym).volume.to_array()
    n = len(vol)
    if n < _VOL_PERIOD + 1:
        return float("nan")
    return float(np.mean(vol[-_VOL_PERIOD:-1]))


def _trend_up(ctx: StrategyContext, sym: str, params: Params) -> bool:
    """True when close sits above its own trailing mean (uptrend concurrency)."""
    close = ctx.ohlcv(sym).close.to_array()
    n = len(close)
    if n < params.trend_lookback + 1:
        return False
    ma = float(np.mean(close[-params.trend_lookback :]))
    return float(close[-1]) > ma


def _feed_entropy(ctx: StrategyContext, shared: dict, sym: str, o) -> None:
    """Feed the per-symbol online adaptive-entropy tracker the current bar.

    Called for **every** symbol on **every** bar (top of the per-symbol loop),
    so the tracker warms and its ``trend`` reflects the true ongoing regime —
    not just the sparse bars where a breakdown is being evaluated. The latest
    snapshot is cached for the entry gate to read synchronously.
    """
    trackers = shared.setdefault(_ENT_KEY, {})
    tr = trackers.get(sym)
    if tr is None:
        cfg = AdaptiveEntropyConfig(
            lookback=ctx.params.entropy_lookback,
            num_bins=_ENTROPY_BINS,
        )
        tr = OnlineAdaptiveEntropy(cfg)
        trackers[sym] = tr
    result = tr.observe(float(o.close[-1]), float(o.high[-1]), float(o.low[-1]))
    shared.setdefault(_ENT_RESULT_KEY, {})[sym] = result


def entropy_result(ctx: StrategyContext, shared: dict, sym: str) -> object:
    """The cached adaptive-entropy snapshot for ``sym`` after feeding this bar."""
    return shared.get(_ENT_RESULT_KEY, {}).get(sym)


def bear_gate_ok(result, min_trend: int) -> bool:
    """Adaptive-entropy regime gate: allow the short only above a trend floor.

    A short is permitted when the quantised trend is at or above ``min_trend``.
    The default ``-1`` is bear-only; ``0`` widens to flat/bear. During warmup the
    running ``trend == 0`` default passes any ``min_trend <= 0``.
    """
    return bool(getattr(result, "trend", 0) >= min_trend)


def _refresh_vp(ctx: StrategyContext, params: Params) -> None:
    """Feed each symbol's online VP once per candle; store the snapshots."""
    profs = ctx.shared.setdefault(_VP_KEY, {})
    snaps: dict[str, object] = {}
    for sym in ctx.symbols:
        prof = profs.get(sym)
        if not isinstance(prof, OnlineVP):
            prof = OnlineVP(
                num_bins=_NUM_BINS,
                value_area_pct=params.value_area_pct,
                window=_VP_WINDOW,
                warmup_bars=_VP_WARMUP,
            )
            profs[sym] = prof
        snaps[sym] = prof.observe(ctx.state, sym, ctx.interval)
    ctx.shared[_SNAP_KEY] = snaps


def _snap(ctx: StrategyContext, sym: str) -> object:
    return ctx.shared.get(_SNAP_KEY, {}).get(sym)


def trail_stop(
    highs: np.ndarray, atr: float, trail_lookback: int, mult: float
) -> float:
    """Chandelier ATR trail for a short: ``highestHigh - mult * ATR``.

    Anchors on the highest high of the most recent ``trail_lookback`` bars
    (cursor-truncated ``highs``) minus ``mult * ATR``. Returns NaN when ATR is
    non-finite/non-positive or no highs are visible.
    """
    if not np.isfinite(atr) or atr <= 0 or len(highs) < 1:
        return float("nan")
    lo = max(0, len(highs) - trail_lookback)
    anchor = float(highs[lo:].max())
    return anchor - mult * atr


def _stretch_ok(
    ctx: StrategyContext, sym: str, params: Params, close: float, atr_val: float
) -> bool:
    """Reject a short when the bar is over-extended below its short EMA.

    ``stretch = (ema - close) / atr`` — how many ATRs price sits *below* its own
    EMA. A reading above ``max_stretch_atr`` means the decline is already
    stretched, so shorting here is fading into exhaustion / the relative bottom.
    Returns ``True`` (allow) when the filter is disabled or data is unavailable.
    """
    if params.max_stretch_atr <= 0:
        return True
    if np.isnan(atr_val) or atr_val <= 0:
        return True
    ema = _last(ctx.ta.ema(sym, _STRETCH_EMA))
    if np.isnan(ema):
        return True
    stretch = (ema - close) / atr_val
    return stretch <= params.max_stretch_atr


def _vix_series(ctx: StrategyContext) -> tuple[np.ndarray, np.ndarray] | None:
    """VIX prints as ``(dates, values)`` on a dense nightly grid, cached per run.

    Forward-filled on load, so the print resolved at a cursor is at most the
    previous session's close — lookahead-free at any base interval.
    """
    cached = ctx.shared.get(_VIX_CACHE_KEY)
    if cached is False:
        return None  # sentinel: asset absent, never reload
    if cached is not None:
        return cast(tuple[np.ndarray, np.ndarray], cached)
    s = load_daily("vix")
    if s.empty:
        ctx.shared[_VIX_CACHE_KEY] = False  # sentinel: absent, don't reload
        return None
    pair = (
        np.asarray(s.index.values, dtype="datetime64[ns]"),
        s.to_numpy(dtype=float),
    )
    ctx.shared[_VIX_CACHE_KEY] = pair
    return pair


def _vix_ratio(ctx: StrategyContext, params: Params) -> float | None:
    """Latest VIX divided by its own trailing multi-year baseline.

    The baseline is the mean of every print in the ``vix_baseline_years`` span
    ending at the cursor, so ``ratio`` is an ABSOLUTE vol read that is still
    comparable across decades: 1.0 is a normal tape, 1.5 an elevated one, 0.7 a
    dead one. That is what a percentile rank cannot express — after a crash the
    trailing window is mostly elevated prints, so the rank drops back to ~0.5
    while the level is still 1.5x normal.

    ``None`` when the gate is disabled, the series is absent, the cursor
    precedes coverage, or the baseline window is too thin — a missing read is
    not a vol signal and must not block a trade.
    """
    if params.vix_max_ratio <= 0:
        return None
    pair = _vix_series(ctx)
    if pair is None:
        return None
    dates, values = pair
    j = int(np.searchsorted(dates, np.datetime64(ctx.candle.timestamp))) - 1
    if j < 0 or j >= values.size or not np.isfinite(values[j]):
        return None
    span = int(_VIX_BASELINE_YEARS * 252)
    if span <= 0:
        return None
    hist = values[max(0, j - span) : j]
    hist = hist[np.isfinite(hist)]
    if hist.size < 2:
        return None
    baseline = float(np.mean(hist))
    if baseline <= 0:
        return None
    return float(values[j]) / baseline


def _vix_ok(ctx: StrategyContext, params: Params) -> bool:
    """VIX vol-regime gate: allow a NEW short only inside the configured band.

    Returns ``True`` (allow) whenever the gate is disabled or the read is
    unavailable — the gate narrows an existing edge, it never creates one.
    """
    ratio = _vix_ratio(ctx, params)
    if ratio is None:
        return True
    if params.vix_max_ratio > 0 and ratio > params.vix_max_ratio:
        return False
    return True


def _size(ctx: StrategyContext, params: Params, price: float, atr_val: float) -> float:
    """0..1 capital fraction to deploy, ATR-risk sized on current cash."""
    if np.isnan(atr_val) or atr_val <= 0 or price <= 0:
        return 0.0
    cash = ctx.state.portfolio.cash
    if cash <= 0:
        return 0.0
    qty = risk_sized_qty(
        equity=cash,
        price=price,
        stop_dist=STOP_ATR_MULT * atr_val,
        risk_pct=params.risk_pct,
    )
    if qty <= 0:
        return 0.0
    size = qty * price / ctx.state.portfolio.initial_capital
    return min(max(size, 0.0), 1.0)


# ---------------------------------------------------------------------------
# DSL strategy
# ---------------------------------------------------------------------------


@strategy(stateful=True)
def on_candle(ctx: StrategyContext):
    params: Params = ctx.params
    shared = ctx.shared
    shared.setdefault(_POS_KEY, {})

    _refresh_vp(ctx, params)

    # Warmup bars fill strategy state only — no fill can happen there, so the
    # DSL raises on any entry emitted during them.
    if ctx.phase == "warmup":
        return
    vix_ok = _vix_ok(ctx, params)

    # ---- Advances / exits first --------------------------------------------
    for sym in ctx.symbols:
        o = ctx.ohlcv(sym)
        _feed_entropy(ctx, shared, sym, o)  # regime tracker every bar
        pos = ctx.position(sym)
        if pos is None:
            continue
        _manage_open(ctx, shared, params, sym, o, pos)

    # ---- Entries ------------------------------------------------------------
    for sym in ctx.symbols:
        if ctx.position(sym) is not None:
            continue
        _enter(ctx, sym, params, vix_ok)


# ── entry (VP breakdown gated by the bear regime) ────────────────────────────


def _enter(ctx: StrategyContext, sym: str, params: Params, vix_ok: bool) -> None:
    """Open a short when the VP breakdown and every regime gate agree."""
    snap = _snap(ctx, sym)
    if snap is None or not getattr(snap, "ready", False):
        return
    o = ctx.ohlcv(sym)
    n = len(o.close)
    if n < max(_VP_WARMUP, params.trend_lookback + 1, _VOL_PERIOD + 1):
        return

    close = float(o.close[-1])
    high = float(o.high[-1])
    low = float(o.low[-1])
    val = getattr(snap, "val", None)
    if not isinstance(val, float):
        return

    atr_val = _last(ctx.ta.atr(sym, ATR_PERIOD))
    if close <= 0 or np.isnan(atr_val) or atr_val <= 0:
        return

    # Adaptive-entropy regime gate: never fade a confirmed bull.
    if not bear_gate_ok(entropy_result(ctx, ctx.shared, sym), MIN_TREND):
        return

    # VIX vol-regime gate: only harvest breakdowns inside the configured band.
    if not vix_ok:
        return

    # Stretch filter: reject entries already too far below their own short EMA.
    if not _stretch_ok(ctx, sym, params, close, atr_val):
        return

    avg_vol = _avg_volume(ctx, sym, params)
    if np.isnan(avg_vol) or avg_vol <= 0:
        return
    vol_expanding = float(o.volume[-1]) >= params.vol_expand_mult * avg_vol
    strength = _bar_position_close(close, high, low)

    short_breakdown = (
        close < val  # close (not a wick) breaks the value area low
        and strength <= 1.0 - params.wick_check  # held near the lows
        and vol_expanding  # order-flow confirmation
        and not _trend_up(ctx, sym, params)  # concurrency: not against trend
    )
    if not short_breakdown:
        return

    size = _size(ctx, params, close, atr_val)
    if size <= 0:
        return

    # Aggregate gross-exposure cap: skip if this entry would push total |gross|
    # notional past ``max_gross_exposure`` of live equity. ``_size`` returns the
    # candidate as a fraction of *initial capital*, so rescale onto the equity
    # base first. A value >= 1.0 disables the cap.
    equity = ctx.current_equity()
    init = ctx.state.portfolio.initial_capital
    candidate = size * init / equity if equity > 0 else 0.0
    if gross_gate(equity, ctx.state.portfolio.positions, MAX_GROSS_EXPOSURE, candidate):
        return

    # Initial stop for sizing is one ATR distance above the entry; the dynamic
    # trailing stop is established from here and ratcheted down each bar.
    stop_price = close + STOP_ATR_MULT * atr_val
    ctx.short(
        sym,
        size=size,
        reason=f"[breakdown] short close {close:.2f} < VAL {val:.2f} in bear regime",
    )
    ctx.shared[_POS_KEY][sym] = {"entry": close, "trail": stop_price, "pid": ""}


# ── position management (dynamic ATR trailing stop) ──────────────────────────


def _manage_open(
    ctx: StrategyContext,
    shared: dict,
    params: Params,
    sym: str,
    o,
    pos: Position,
) -> None:
    """Manage an open short: value-area revert first, then the ATR trail.

    (1) value-area revert (close back above VAL) is the institutional tag stop,
    and (2) a dynamic ATR trail re-anchored every bar on the highest high of the
    last ``trail_lookback`` bars minus ``trail_atr_mult * ATR``, only ever moved
    *down* for a short.
    """
    rec = shared[_POS_KEY].get(sym)
    atr = _last(ctx.ta.atr(sym, ATR_PERIOD))
    close = float(o.close[-1])

    if rec is not None and not rec.get("pid"):
        rec["pid"] = pos.position_id

    # (1) Value-area revert (institutional tag stop): close back inside the
    # fair-value range invalidates the breakdown.
    snap = _snap(ctx, sym)
    val = getattr(snap, "val", None) if snap is not None else None
    if isinstance(val, float) and close > val:
        _exit(ctx, shared, sym, "reverted into value area")
        return

    # (2) Dynamic ATR trailing stop, only ever ratcheted down.
    if rec is not None and np.isfinite(atr):
        new_trail = trail_stop(
            o.high.to_array(), atr, params.trail_lookback, params.trail_atr_mult
        )
        if np.isfinite(new_trail) and new_trail < rec["trail"]:
            rec["trail"] = new_trail
        if close >= rec["trail"]:
            _exit(ctx, shared, sym, "dynamic ATR trail hit")
            return


def _exit(ctx: StrategyContext, shared: dict, sym: str, reason: str) -> None:
    shared[_POS_KEY].pop(sym, None)
    ctx.close(sym, reason=f"[breakdown] {reason}")
