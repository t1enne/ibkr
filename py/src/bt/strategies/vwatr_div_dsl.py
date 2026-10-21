"""VWATR divergence strategy -- long only, one unconditional breakout entry.

ONE module, ONE parameter grammar, TWO execution venues. Every signal, state and
sizing read is a DAILY read (``_state_interval``); the base bar
(``config.bars[0]``) only decides WHEN a signal fills. The config's ``bars`` is
the ONLY switch:

=============  ===================  ====================================
``bars``       signal read          what the run is
=============  ===================  ====================================
``["1d"]``     base feed (``None``) daily: reads and fills are both daily
``["1h","1d"]`` ``"1d"``             intraday: daily rules, base-bar fill
=============  ===================  ====================================

Entry: rising VWATR expansion on a breakout above the prior ``breakout_look``
highs, taken only while the universe is in a breadth leg up. Within that gate
the trigger is unconditional by design -- entry-side features carry no
measurable edge here, so name selection lives in sizing instead.

Exit, in order:

1. VWATR slope divergence at a new high -- the primary exit, and load-bearing
   (turning it off lowers return AND worsens drawdown and tails). A STATE rule,
   so it stays a next-open ``ctx.close``.
2. ``vwatr_mult`` chandelier trail off the running best close, armed via
   ``ctx.set_stops`` as an ABSOLUTE level (ratchet-only): fires intrabar on the
   next bar's low, gap-adjusted. Width is load-bearing. ``tp`` unused.

Intraday, the once-per-daily-bucket guard: a daily rule holds for the rest of
its session, so ``_new_daily_bucket`` gates the ENTRY and the ``decel`` EXIT to
the first base bar of a COMPLETED daily bucket. The trail is the one exception,
and only in its FIRING: its LEVEL is recomputed from the daily read but
re-armed every base bar (``set_stops`` is tighten-only, so the arm is
idempotent and gating it would only leave the level stale for a session).

Lookahead-free on both venues: HTF reads expose COMPLETED daily buckets only,
and the VIX read resolves the greatest grid date at or before the cursor.

Sizing: ``risk_pct`` of LIVE equity times ``_amp_score`` and the structural
``_fred_band_score``. The ``_BOOK_SLOTS`` cap throttles per-name size before the
engine's silent cash-exhaustion cohort scale, the binding constraint (AGENTS.md
§ Fills).

Data contract: ``ctx.ta.vwatr`` / ``vwatr_baseline`` use a PLAIN rolling-mean
ATR, not Wilder -- ``ctx.ta.atr`` would silently change every stop distance. A
``1d`` read needs ``vwatr_period + vwatr_base_win`` completed buckets, so
``warmup`` must cover that many TRADING days.

Single kill switch: ``decel_ratio=0`` turns the divergence exit OFF.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import cast

import numpy as np

from src.bt.size.pure import risk_sized_qty
from src.indicators.macro._shared import load_daily
from src.bt.state import ActionType
from src.bt.strategies.dsl import StrategyContext, strategy
from src.bt.strategies.types import (
    Marker,
    Overlay,
    Panel,
    PlotSpec,
    StrategyParams,
    ms,
    sparse,
)

STRATEGY_TYPE = "vwatr_div_dsl"
_STATE_KEY = "vwatr_div_state"

#: Signal read interval. ``_state_interval`` returns ``None`` (the base feed)
#: when ``bars[0]`` is itself daily, else this completed-bucket feed.
_SIGNAL_INTERVAL = "1d"
_BASE_INTERVAL_1D = "1d"


@dataclass(frozen=True)
class _State:
    """Per-symbol cross-call state (``ctx.shared``, per run)."""

    best: float | None = None
    #: Visible daily-bucket count at the last base bar -- the once-per-bucket
    #: guard's baseline. Only consulted when the read is an HTF feed.
    daily_count: int | None = None


@dataclass(frozen=True)
class Params(StrategyParams):
    # -- VWATR: the signal, the stop and their units --
    vwatr_period: int = 14  # DAILY bars
    vwatr_base_win: int = 60  # DAILY bars
    vwatr_mult: float = 3.0  # chandelier trail width AND the sizing stop
    slope_bars: int = 2  # DAILY bars: the expansion gate and the exit slope
    breakout_look: int = 20  # DAILY bars: close > the prior 20 daily highs
    # -- sizing: flat risk budget times ONE trend amplifier --
    risk_pct: float = 0.004
    risk_amp: float = 0.0  # 0 = flat; see _amp_score (non-conserved re-lever)
    risk_topk: int = 0  # >0 = also concentrate onto the top-K by dollar vol
    # -- rolling universe selection (0 = OFF: trade every configured symbol) --
    select_top_n: int = 0  # >0 -> trade only the rolling top-N survivors
    select_bars: int = 126  # re-selection period in DAILY bars (~6 months)
    # -- exit --
    decel_ratio: float = 0.1  # VWATR slope-divergence exit; 0 turns it OFF


@strategy(stateful=True)
def on_candle(ctx: StrategyContext) -> None:
    if ctx.phase == "warmup":
        return
    p: Params = ctx.params
    for sym in ctx.symbols:
        if sym.startswith("GATE_"):
            continue  # synthetic macro series: fed for the gate, never traded
        _process_symbol(ctx, p, sym)


def _state_interval(ctx: StrategyContext) -> str | None:
    """Signal read interval for this run; ``None`` means the base feed.

    ``None`` is the daily venue (every bar is its own bucket, so the guard is
    moot); anything else is an intraday base where the signal is the completed
    ``1d`` bucket and the base bar only times the fill.
    """
    return None if ctx.interval == _BASE_INTERVAL_1D else _SIGNAL_INTERVAL


def _is_htf(ctx: StrategyContext) -> bool:
    """True when the SIGNAL reads a higher-timeframe feed (base bar = fill only)."""
    return _state_interval(ctx) is not None


def _slope_decelerating(
    vwatr_arr: np.ndarray, n: int, slope_bars: int, decel_ratio: float
) -> bool:
    """VWATR slope at t vs the slope one ``slope_bars`` window earlier."""
    if n <= 2 * slope_bars:
        return False
    i = n - 1
    v0 = float(vwatr_arr[i])
    v1 = float(vwatr_arr[i - slope_bars])
    v2 = float(vwatr_arr[i - 2 * slope_bars])
    if not (np.isfinite(v0) and np.isfinite(v1) and np.isfinite(v2)):
        return False
    slope_now = v0 - v1
    slope_prev = v1 - v2
    # only while expansion positive; falling VWATR -> trail's exit
    if slope_prev <= 0:
        return False
    return slope_now < decel_ratio * slope_prev


#: FRED (VIX) derisking -- STRUCTURAL, no params. Shrinks the low-vol bucket
#: only: rising vol is not the risk on this population.
#:
#: This is the ONE read taken at the BASE-bar cursor rather than on a completed
#: daily bucket. ``load_daily`` forward-fills VIX onto a nightly grid, so the
#: print resolved at the cursor is at most the previous session's close -- the
#: value a daily-bar run reads for that session -- and it steps at the first
#: base bar of each day, the same bucket the entry is evaluated on.
_FRED_ASSET: str = "vix"
_FRED_WIN: int = 250
_RS_BAND_LO: float = 0.45
#: Low-vol size factor (structural). Lower is the Sharpe end of the frontier,
#: higher the Ann end.
_RS_BAND_FACTOR: float = 0.80
_FRED_CACHE_KEY = "vwatr_div_fred_series"


def _fred_series(ctx: StrategyContext) -> tuple[np.ndarray, np.ndarray] | None:
    """VIX prints as ``(dates, values)``, cached in ``ctx.shared`` per run.

    Dense nightly grid, so a cursor lookup is one searchsorted: no per-bar disk
    read, and no way to see a print dated after the cursor.
    """
    cache: dict[str, tuple[np.ndarray, np.ndarray] | None] = ctx.shared.setdefault(
        _FRED_CACHE_KEY, {}
    )
    if _FRED_ASSET not in cache:
        s = load_daily(_FRED_ASSET)
        if s.empty:
            cache[_FRED_ASSET] = None
            return None
        dates = np.asarray(s.index.values, dtype="datetime64[ns]")
        cache[_FRED_ASSET] = (dates, s.to_numpy(dtype=float))
    return cache[_FRED_ASSET]


def _percentile_rank(series: np.ndarray, i: int, window: int) -> float:
    """Fraction of the trailing window (current excluded) strictly below
    ``series[i]``; 0.5 neutral on out-of-bounds / NaN / thin window."""
    if i < 0 or i >= series.size:
        return 0.5
    current = series[i]
    if not np.isfinite(current):
        return 0.5
    hist = series[max(0, i - window) : i]
    hist = hist[np.isfinite(hist)]
    if hist.size < 2:
        return 0.5
    return float(np.mean(hist < current))


def _fred_band_score(ctx: StrategyContext) -> float:
    """``_RS_BAND_FACTOR`` in the low-vol bucket, neutral otherwise.

    Neutral too when the series is missing or the cursor precedes it -- a
    missing print is not a vol signal. Lookahead-free at any cursor resolution.
    """
    series = _fred_series(ctx)
    if series is None:
        return 1.0
    dates, values = series
    j = int(np.searchsorted(dates, np.datetime64(ctx.candle.timestamp))) - 1
    if j < 0 or not np.isfinite(values[j]):
        return 1.0
    pct = _percentile_rank(values, j, _FRED_WIN)
    return _RS_BAND_FACTOR if pct < _RS_BAND_LO else 1.0


#: Trend amplifier constants. Slow SMA slope (short horizons carry no rank IC)
#: and a breadth "leg up" gate: enough of the universe above its own SMA. Both
#: DAILY bars -- a 50-bar window on an intraday base would read a 1-week signal.
_TREND_LOOK: int = 50
_TREND_GATE_LO: float = 0.5
_TREND_CACHE_KEY = "vwatr_div_trend"
_COHORT_CACHE_KEY = "vwatr_div_cohort"
#: Book-slot cap: per-name risk scales by ``_BOOK_SLOTS / book`` once the book
#: (open names + this bar's cohort) exceeds it, so a dense tape throttles size
#: BEFORE the engine silently cohort-scales.
_BOOK_SLOTS: int = 10


def _new_daily_bucket(
    ctx: StrategyContext, holder: dict[str, _State], sym: str, n_daily: int
) -> bool:
    """True when ``sym`` has one MORE visible daily bucket than the last base
    bar -- i.e. this base bar is the bucket's first.

    The intraday once-per-bucket guard. Always True on the daily venue, where
    every bar is its own bucket.
    """
    if not _is_htf(ctx):
        return True
    state = holder.get(sym) or _State()
    last = state.daily_count
    if last is None:  # first base bar this run processed for ``sym``
        holder[sym] = dataclasses.replace(state, daily_count=n_daily)
        return False
    holder[sym] = dataclasses.replace(state, daily_count=max(last, n_daily))
    return n_daily > last


def _daily_reads(
    ctx: StrategyContext, p: Params, sym: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """``(close, vwatr, baseline)`` on the signal feed, or None.

    Cursor-truncated. Every signal, stop and sizing number comes from these
    arrays, so they share one interval.
    """
    o = ctx.ohlcv(sym, _state_interval(ctx))
    if o is None or len(o.close) == 0:
        return None
    close = o.close.to_array()
    vwatr = ctx.ta.vwatr(sym, p.vwatr_period, interval=_state_interval(ctx)).to_array()
    base = ctx.ta.vwatr_baseline(
        sym, p.vwatr_period, p.vwatr_base_win, interval=_state_interval(ctx)
    ).to_array()
    if close.size == 0 or vwatr.size < close.size or base.size < close.size:
        return None
    return close, vwatr, base


def _breakout_count(ctx: StrategyContext, p: Params) -> int:
    """Number of flat symbols whose entry rule is true on this bar.

    Cached per timestamp; bounded; read-only. Approximates the cohort the engine
    will execute in one bar cycle so size can be damped first. With selection
    ON it counts only the selected names the engine can actually open.
    """
    cache: dict[object, int] = ctx.shared.setdefault(_COHORT_CACHE_KEY, {})
    ts = ctx.candle.timestamp
    if ts in cache:
        return cache[ts]
    tradable = _selection(ctx, p) if p.select_top_n > 0 else None
    count = 0
    for sym in ctx.symbols:
        if sym.startswith("GATE_") or ctx.quantity(sym) != 0:
            continue
        if tradable is not None and sym not in tradable:
            continue
        o = ctx.ohlcv(sym, _state_interval(ctx))
        if o is None:
            continue
        close = o.close.to_array()
        n = close.size
        if n < p.breakout_look + 1:
            continue
        px = float(close[-1])
        v = ctx.ta.vwatr(sym, p.vwatr_period, interval=_state_interval(ctx)).to_array()
        if not np.isfinite(px) or px <= 0 or v.size < n:
            continue
        v_now = float(v[n - 1])
        v_prev = float(v[n - 1 - p.slope_bars])
        if not (np.isfinite(v_now) and np.isfinite(v_prev)) or not (v_now > v_prev):
            continue
        prior = close[n - 1 - p.breakout_look : n - 1]
        if prior.size < p.breakout_look or not np.all(np.isfinite(prior)):
            continue
        if px > float(np.max(prior)):
            count += 1
    if len(cache) > 8:
        cache.clear()
    cache[ts] = count
    return count


def _trend_cross_section(
    ctx: StrategyContext, p: Params
) -> tuple[dict[str, float], bool]:
    """Per-bar map of daily slope percentile + the leg-up gate, computed once.

    Slope is the ``_TREND_LOOK``-session SMA vs the SMA one horizon earlier, as
    a fraction of price; ``pct`` is its cross-sectional rank (0..1). Cached per
    timestamp in ``ctx.shared`` because ``on_candle`` visits one symbol a call.
    """
    cache: dict[object, tuple[dict[str, float], bool]] = ctx.shared.setdefault(
        _TREND_CACHE_KEY, {}
    )
    ts = ctx.candle.timestamp
    if ts in cache:
        return cache[ts]
    slopes: dict[str, float] = {}
    above: dict[str, bool] = {}
    for sym in ctx.symbols:
        if sym.startswith("GATE_"):
            continue
        o = ctx.ohlcv(sym, _state_interval(ctx))
        if o is None or len(o.close) < 2 * _TREND_LOOK:
            continue
        close = o.close.to_array()
        now = float(np.mean(close[-_TREND_LOOK:]))
        prev = float(np.mean(close[-2 * _TREND_LOOK : -_TREND_LOOK]))
        if not (np.isfinite(now) and np.isfinite(prev)) or prev <= 0:
            continue
        slopes[sym] = (now - prev) / prev
        above[sym] = float(close[-1]) > now
    pcts: dict[str, float] = {}
    for sym, v in slopes.items():
        below = sum(1 for u in slopes.values() if u < v)
        ties = sum(1 for u in slopes.values() if u == v) - 1
        pcts[sym] = (
            (below + 0.5 * ties) / max(1, len(slopes) - 1) if len(slopes) > 1 else 0.5
        )
    # Breadth is the share of the readable universe above its own SMA. No
    # minimum-size guard -- at one symbol the ratio is 0 or 1 over a denominator
    # of one, which is the rule, not a degenerate case to correct.
    leg_up = (
        (sum(1 for v in above.values() if v) / len(above)) >= _TREND_GATE_LO
        if above
        else False
    )
    out = (pcts, leg_up)
    if len(cache) > 8:  # keep the per-run cache bounded
        cache.clear()
    cache[ts] = out
    return out


def _amp_score(p: Params, ctx: StrategyContext, sym: str) -> float:
    """Size multiplier: trend re-lever inside a leg up, times optional
    dollar-volume top-K concentration.

    1. ``risk_amp``: ``1 + amp*(2*trend_pct - 1)``, gated to the leg-up. ``0``
       or gate shut -> flat.
    2. ``risk_topk > 0``: top-K names by trailing dollar volume draw
       ``x(1+_DV_CONC)``, the rest take a compensating cut, so the cohort's
       total budget is unchanged.

    Both neutral when off. A missing trend read is not a weak trend.
    """
    if p.risk_amp == 0 and p.risk_topk == 0:
        return 1.0
    score = 1.0
    if p.risk_amp != 0:
        pcts, leg_up = _trend_cross_section(ctx, p)
        if leg_up and sym in pcts:
            score *= 1.0 + p.risk_amp * (2.0 * pcts[sym] - 1.0)
    if p.risk_topk > 0:
        score *= _dv_topk_score(ctx, p, sym)
    return score


#: Dollar-volume concentration: cross-sectional percentile of trailing 60d
#: dollar volume. A SIZE/participation signal -- heavy dollar flow is where a
#: breakout is actually being paid for -- and conserved by ``_dv_topk_score``,
#: so total deployment (and cash-exhaustion scaling) is unchanged.
_DV_LOOK: int = 60
_DV_CACHE_KEY = "vwatr_div_dv"
#: Conserved top-K concentration strength (structural). Boosting more leaks
#: cash-exhaustion scaling.
_DV_CONC: float = 1.0


def _dv_cross_section(ctx: StrategyContext) -> dict[str, float]:
    """Trailing-60d dollar-volume percentile within the universe, computed once
    per timestamp and cached in ``ctx.shared`` (bounded)."""
    cache: dict[object, dict[str, float]] = ctx.shared.setdefault(_DV_CACHE_KEY, {})
    ts = ctx.candle.timestamp
    if ts in cache:
        return cache[ts]
    dv: dict[str, float] = {}
    for sym in ctx.symbols:
        if sym.startswith("GATE_"):
            continue
        o = ctx.ohlcv(sym, _state_interval(ctx))
        if o is None:
            continue
        close = o.close.to_array()
        volume = o.volume.to_array()
        if close.size < _DV_LOOK or volume.size < _DV_LOOK:
            continue
        d = float(np.mean(close[-_DV_LOOK:] * volume[-_DV_LOOK:]))
        if np.isfinite(d) and d > 0:
            dv[sym] = d
    pcts: dict[str, float] = {}
    for sym, v in dv.items():
        below = sum(1 for u in dv.values() if u < v)
        ties = sum(1 for u in dv.values() if u == v) - 1
        pcts[sym] = (below + 0.5 * ties) / max(1, len(dv) - 1) if len(dv) > 1 else 0.5
    if len(cache) > 8:  # keep the per-run cache bounded
        cache.clear()
    cache[ts] = pcts
    return pcts


def _dv_topk_score(ctx: StrategyContext, p: Params, sym: str) -> float:
    """Conserved top-K concentration on trailing dollar volume.

    Top ``risk_topk`` names get ``x(1+conc)``, the rest a compensating cut, so
    the cohort budget still sums to ~n. Neutral (1.0) when the symbol has no dv
    read or the cohort is empty.
    """
    picks = _selection(ctx, p) if p.select_top_n > 0 else None
    pcts = _dv_cross_section(ctx)
    if picks is not None:
        cohort = sorted((s for s in picks if s in pcts), key=lambda s: pcts[s])
    else:
        cohort = sorted((s for s, v in pcts.items() if v > 0), key=lambda s: pcts[s])
    n = len(cohort)
    if not cohort:
        return 1.0
    pos = cohort.index(sym) if sym in cohort else -1
    if pos < 0:
        return 1.0
    m = n - pos  # 1..n, larger = more $ flow
    k = min(p.risk_topk, n)
    if m <= k:
        return 1.0 + _DV_CONC
    rest = n - k
    if rest <= 0:
        return 1.0
    return 1.0 - _DV_CONC * (k / rest)


#: Rolling selection: top-``select_top_n`` names by combined rank of trailing
#: dollar volume and realized vol. DAILY reads -- the feature describes a name
#: over months, so the selected set matches on either venue. Recomputed every
#: ``select_bars`` bars, held in ``ctx.shared``. Deselected names are NOT
#: force-closed: the trail and the divergence exit still own open positions.
_SELECT_CACHE_KEY = "vwatr_div_selection"
_SELECT_LOOK: int = 60


def _pct_rank(vals: dict[str, float], sym: str) -> float:
    """Fraction of the cross-section strictly below ``vals[sym]`` (0..1)."""
    v = vals[sym]
    below = sum(1 for u in vals.values() if u < v)
    ties = sum(1 for u in vals.values() if u == v) - 1
    return (below + 0.5 * ties) / max(1, len(vals) - 1)


def _selection(ctx: StrategyContext, p: Params) -> frozenset[str]:
    """Tradable set for the current bar: the last rebalance's picks.

    Re-selects after ``select_bars`` signal-feed bars (counted off the reference
    symbol, so exact in that interval's bars, not wall days). Names without
    ``_SELECT_LOOK`` bars of history are never picked.
    """
    holder: dict[str, object] = ctx.shared.setdefault(_SELECT_CACHE_KEY, {})
    ref = next((s for s in ctx.symbols if not s.startswith("GATE_")), None)
    frame = ctx.ohlcv(ref, _state_interval(ctx)) if ref is not None else None
    bars = 0 if frame is None else len(frame.close)
    last_bars = holder.get("bars")
    if (
        isinstance(last_bars, int)
        and bars - last_bars < p.select_bars
        and "picks" in holder
    ):
        return cast("frozenset[str]", holder["picks"])

    dv: dict[str, float] = {}
    vol: dict[str, float] = {}
    for sym in ctx.symbols:
        if sym.startswith("GATE_"):
            continue
        o = ctx.ohlcv(sym, _state_interval(ctx))
        if o is None:
            continue
        close = o.close.to_array()
        volume = o.volume.to_array()
        if close.size < _SELECT_LOOK + 1 or volume.size < _SELECT_LOOK:
            continue
        logret = np.diff(np.log(close[-(_SELECT_LOOK + 1) :]))
        rvol = float(np.std(logret, ddof=1))
        dollar = float(np.mean(close[-_SELECT_LOOK:] * volume[-_SELECT_LOOK:]))
        if np.isfinite(rvol) and np.isfinite(dollar) and dollar > 0:
            vol[sym] = rvol
            dv[sym] = dollar
    common = sorted(set(dv) & set(vol))
    if len(common) <= p.select_top_n:
        picks = frozenset(common)
    else:
        ranked = sorted(
            common, key=lambda s: (-(_pct_rank(dv, s) + _pct_rank(vol, s)), s)
        )
        picks = frozenset(ranked[: p.select_top_n])
    holder["bars"] = bars
    holder["picks"] = picks
    return picks


def _exit_inputs(
    ctx: StrategyContext, p: Params, sym: str, state: _State
) -> tuple[np.ndarray, int, float | None, bool]:
    """Signal-feed VWATR array, its length, the running high-water mark, and
    whether the latest signal close set a new high.

    The mark is the last SIGNAL close, never the base-bar price: on an intraday
    base ratcheting off an hourly close runs it ahead of the daily mark the
    trail and the divergence test are written against.
    """
    reads = _daily_reads(ctx, p, sym)
    if reads is None:
        return np.empty(0), 0, None, False
    close_d, v_d, _ = reads
    nd = v_d.size
    if close_d.size == 0:
        return v_d, nd, None, False
    close_now = float(close_d[-1])
    if not np.isfinite(close_now) or close_now <= 0:
        return v_d, nd, None, False
    best = state.best
    if best is None or not np.isfinite(best):
        return v_d, nd, close_now, True
    at_new_high = close_now >= best
    return v_d, nd, max(best, close_now), at_new_high


def _arm_trail(
    ctx: StrategyContext, p: Params, sym: str, state: _State, best: float, px: float
) -> _State:
    """Arm the signal-derived chandelier level for the NEXT bar.

    ``best`` and the VWATR distance are both daily quantities, so the level is
    the source's. ``set_stops`` is tighten-only, so re-arming an unchanged level
    is idempotent -- free on every base bar, keeps a live stop between reads.
    """
    reads = _daily_reads(ctx, p, sym)
    if reads is None:
        return dataclasses.replace(state, best=best)
    _, v_d, _ = reads
    nd = v_d.size
    if nd == 0:
        return dataclasses.replace(state, best=best)
    vwatr_now = float(v_d[nd - 1])
    if not np.isfinite(vwatr_now) or vwatr_now <= 0:
        return dataclasses.replace(state, best=best)
    stop_distance = p.vwatr_mult * vwatr_now
    ctx.set_stops(
        sym,
        sl=best - stop_distance,
        reason=(
            f"trail {sym}: arm {best - stop_distance:.2f} "
            f"−{stop_distance:.2f} ({stop_distance / px:.2%}) "
            f"off best {best:.2f}"
        ),
    )
    return dataclasses.replace(state, best=best)


def _exit_daily(
    ctx: StrategyContext, p: Params, sym: str, state: _State, px: float
) -> _State:
    """Exit path on the signal read: divergence close, else the trail.

    Called only on bucket bars (every bar on the daily venue), so the rule is
    read where a daily-bar run would read it. ``decel`` is a STATE rule gated on
    a new high -> next-open close. The trail is a PRICE level, re-armed here
    and by ``_rearm_trail`` between reads so it still fires intrabar.
    """
    v_d, nd, best, at_new_high = _exit_inputs(ctx, p, sym, state)
    if best is None:
        return state

    # 1) slope divergence at a new high — a STATE rule, so a next-open close.
    # ``at_new_high`` is the signal-feed analogue of ``px >= best``; testing the
    # base close instead fires into a partly-formed bucket.
    if (
        p.decel_ratio > 0
        and at_new_high
        and _slope_decelerating(v_d, nd, p.slope_bars, p.decel_ratio)
    ):
        ctx.close(
            sym,
            reason=(
                f"decel {sym}: {best:.2f} at best, slope < {p.decel_ratio:.2f}× prior"
            ),
        )
        return dataclasses.replace(state, best=None)

    # 2) chandelier trail: ARM the level for the NEXT bar rather than closing at
    # the next open. ``set_stops`` does NOT clear ``best`` — the trail lives with
    # the position. ``tp`` intentionally unused: no take-profit.
    return _arm_trail(ctx, p, sym, state, best, px)


def _rearm_trail(
    ctx: StrategyContext, p: Params, sym: str, state: _State, px: float
) -> _State:
    """Refresh the high-water mark + trail arm on a NON-bucket base bar.

    The divergence exit is deliberately absent -- it is a state rule and only
    ``_exit_daily`` (bucket bars) may fire it.
    """
    _, nd, best, _ = _exit_inputs(ctx, p, sym, state)
    if best is None or nd == 0:
        return state
    return _arm_trail(ctx, p, sym, state, best, px)


def _entry(
    ctx: StrategyContext, p: Params, sym: str, px: float, holder: dict[str, _State]
) -> None:
    """Evaluate and emit the entry on a bucket bar (no-op when it doesn't fire).

    ``px`` is the base-bar price: the fill is timed on it, the break condition
    itself is the signal feed's last close.
    """
    reads = _daily_reads(ctx, p, sym)
    if reads is None:
        return
    close_d, v_d, base_d = reads
    nd = close_d.size
    if nd <= 2 * p.slope_bars:
        return
    base_now = float(base_d[nd - 1])
    if not np.isfinite(base_now) or base_now <= 0:
        return

    vwatr_d = float(v_d[nd - 1])
    vwatr_d_prev = float(v_d[nd - 1 - p.slope_bars])
    if not (np.isfinite(vwatr_d) and np.isfinite(vwatr_d_prev)):
        return
    if not (vwatr_d > vwatr_d_prev):
        return

    # Breakout: the last close above the max of the prior ``breakout_look``
    # closes -- one rule, same bar count, whatever the fill venue.
    prior = close_d[nd - 1 - p.breakout_look : nd - 1]
    if prior.size < p.breakout_look or not np.all(np.isfinite(prior)):
        return
    prior_high = float(np.max(prior))
    if not (float(close_d[nd - 1]) > prior_high):
        return

    # Breadth gate: only take breakouts while the universe is in a leg up
    # (>= _TREND_GATE_LO above their own SMA). Structural, no param -- the
    # same ``leg_up`` the trend amp uses, so a bear tape stops new risk.
    _, leg_up = _trend_cross_section(ctx, p)
    if not leg_up:
        return

    # Rolling universe selection: only names picked at the last rebalance may
    # open. Off by default, so the tuned configs are byte-for-byte unchanged.
    if p.select_top_n > 0 and sym not in _selection(ctx, p):
        return

    # The stop distance is the signal VWATR: the same risk unit the source
    # sizes against, so ``risk_pct`` keeps its meaning whatever the fill venue.
    stop_price = p.vwatr_mult * vwatr_d
    # Size off LIVE EQUITY: the risk budget re-levers with the book, which is
    # what keeps deployment (and therefore Ann) up as equity compounds.
    equity = ctx.current_equity()
    score = _fred_band_score(ctx) * _amp_score(p, ctx, sym)
    cohort = _breakout_count(ctx, p)
    book = cohort + sum(1 for s in ctx.symbols if ctx.quantity(s) != 0)
    if book > _BOOK_SLOTS:
        score *= _BOOK_SLOTS / book
    risk_eff = p.risk_pct * score
    qty = risk_sized_qty(
        equity=equity, price=px, stop_dist=stop_price, risk_pct=risk_eff
    )
    if qty <= 0:
        return
    size = min(qty * px / equity, 1.0) if equity > 0 else 0.0
    if size <= 0:
        return

    reason = (
        f"{sym}: close {float(close_d[nd - 1]):.2f} > {prior_high:.2f} "
        f"({p.breakout_look}b), "
        f"vwatr {vwatr_d:.3f} {vwatr_d / base_now:.2f}× base {base_now:.3f}, "
        f"↑ {vwatr_d_prev:.3f}/{p.slope_bars}b, "
        f"risk {risk_eff:.1%} ×{score:.2f}, stop {stop_price:.2f}, "
        f"fill @ {px:.2f}"
    )

    # Seed the high-water mark from the signal close that triggered the entry,
    # not the fill price: ``best`` is a signal-feed quantity everywhere else on
    # this strategy (see ``_exit_inputs``).
    state = holder.get(sym) or _State()
    holder[sym] = dataclasses.replace(state, best=float(close_d[nd - 1]))
    # Arming the initial stop in the SAME call via ``sl`` (ABSOLUTE PRICE):
    # ``stop_price`` is the DISTANCE at entry − vwatr_mult*VWATR, so subtract it
    # from the price the fill is timed off instead of passing it as a fraction.
    ctx.long(
        sym,
        size=size,
        size_mode="equity",
        reason=reason,
        sl=px - stop_price,
    )


def _process_symbol(ctx: StrategyContext, p: Params, sym: str) -> None:
    holder: dict[str, _State] = ctx.shared.setdefault(_STATE_KEY, {})
    state = holder.get(sym) or _State()
    holder[sym] = state

    reads = _daily_reads(ctx, p, sym)
    if reads is None:
        return
    nd = reads[0].size

    # The once-per-bucket guard (see the module docstring). Consumes the signal
    # read length unconditionally, so the baseline is adopted on the first bar
    # even while flat and never re-baselines later. Always True on the daily
    # venue, where the read IS the base feed.
    new_bucket = _new_daily_bucket(ctx, holder, sym, nd)
    state = holder[sym]

    base = ctx.ohlcv(sym)
    if base is None or len(base.close) == 0:
        return
    px = float(base.close[-1])
    if not np.isfinite(px) or px <= 0:
        return

    # ---- exit while long ----
    if ctx.quantity(sym) != 0:
        if not new_bucket:
            # Between daily reads the base bar owns one action only: keep the
            # daily-derived trail level armed. It is a PRICE stop, not a state
            # rule, so re-arming it at base cadence is idempotent
            # (``set_stops`` ratchets tighten-only) and keeps the position
            # covered until the next bucket closes.
            holder[sym] = _rearm_trail(ctx, p, sym, state, px)
            return
        holder[sym] = _exit_daily(ctx, p, sym, state, px)
        return

    if not new_bucket:
        return  # entry is a DAILY decision: once per completed bucket

    _entry(ctx, p, sym, px, holder)


def _shift(values: np.ndarray, bars: int) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if bars < len(values):
        out[bars:] = values[: len(values) - bars]
    return out


def _as_series(index, values: np.ndarray):
    import pandas as pd

    return pd.Series(values, index=index)


def plot(ctx: StrategyContext, params: Params) -> PlotSpec:
    """Post-run chart spec (cursor at terminal bar; ctx.ta series are the
    full visible history).

    The price pane is the BASE (execution) feed — that is where the fill lands.
    The signal series is reindexed onto the base index with a forward fill; on
    an intraday base with completed daily buckets that means a daily value only
    paints from the bar AFTER its session closed.
    """
    sym = ctx.candle.symbol
    df = ctx.state.candles.get((sym, ctx.interval))
    if df is None or len(df) < params.vwatr_period + params.vwatr_base_win:
        return PlotSpec()

    close = ctx.ta.close(sym).to_array()
    sig = ctx.ohlcv(sym, _state_interval(ctx))
    sig_close = np.full(close.shape, np.nan, dtype=np.float64)
    sig_slope = np.full(close.shape, np.nan, dtype=np.float64)
    if sig is not None and len(sig.close) > 2 * params.slope_bars:
        sc = sig.close.to_array()
        v_s = ctx.ta.vwatr(
            sym, params.vwatr_period, interval=_state_interval(ctx)
        ).to_array()
        s_slope = v_s - _shift(v_s, params.slope_bars)
        # Each signal bucket paints from the next base bar onwards (its session
        # has closed by then), so offset the mapped values by one base bar.
        sig_close[1 : 1 + sc.size] = sc[: len(sig_close) - 1]
        sig_slope[1 : 1 + s_slope.size] = s_slope[: len(sig_slope) - 1]

    stop_band = params.vwatr_mult * sig_close
    upper = sig_close + stop_band
    lower = sig_close - stop_band

    look = params.breakout_look
    channel = np.full(close.shape, np.nan, dtype=np.float64)
    if len(sig_close) > look:
        from numpy.lib.stride_tricks import sliding_window_view

        channel[look:] = sliding_window_view(sig_close, look).max(axis=1)[:-1]

    markers: list[Marker] = []
    for trade in ctx.state.portfolio.trades:
        if trade.symbol != sym or trade.position != ActionType.long:
            continue
        markers.append(
            Marker(
                ts=ms(trade.entry_time),
                price=float(trade.entry_price),
                kind="pivot_low",
            )
        )

    return PlotSpec(
        overlays=(
            Overlay(sparse(_as_series(df.index, sig_close)), "close", "line", "price"),
            Overlay(
                sparse(_as_series(df.index, channel)),
                f"breakout_high ({look})",
                "line",
                "price",
            ),
            Overlay(sparse(_as_series(df.index, upper)), "trail_hi", "line", "price"),
            Overlay(sparse(_as_series(df.index, lower)), "trail_lo", "line", "price"),
        ),
        # slope pair only: vwatr/close = noise; baseline div failed. The source
        # interval is the rule's; the base bar only times the fill.
        panels=(
            Panel(
                sparse(_as_series(df.index, sig_slope)),
                f"vwatr slope ({params.slope_bars}b, entry + exit state)",
                (0.0,),
            ),
        ),
        markers=tuple(markers),
    )
