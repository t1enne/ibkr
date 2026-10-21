"""VWATR divergence, DAILY signal on an INTRADAY execution grid.

Sibling of :mod:`src.bt.strategies.vwatr_div_dsl`. Same signals, intraday
fill timing: every rule is computed on **completed daily buckets** and only the
FILL is timed on the base (intraday) bar. The rule set, the parameter units and
the stop are the source strategy's; nothing here re-tunes a signal.

Interval assignment (the whole point of the variant):

======================  =========  =============================================
read                    interval   why
======================  =========  =============================================
breakout lookback       **1d**     the source rule: close > the prior N daily highs
VWATR expansion gate    **1d**     the source entry gate (VWATR > VWATR slope_bars ago)
VWATR baseline          **1d**     the expansion reference the state lives on
VWATR slope exit        **1d**     a daily expansion STATE rule, read on a closed day
chandelier trail        **1d**     daily VWATR distance off the running best close
sizing stop             **1d**     the stop the daily signal is sized against
trend cross-section     1d         universe breadth + slope rank (a regime read)
dollar-volume rank      1d         participation, not intraday noise
rolling universe select 1d         a name is liquid over months, not hours
VIX vol-regime factor   daily      the source's portfolio derisk; DAILY series,
                                   read at/before the base bar (see below)
current price            base       breakout trigger, high-water mark, stop arming
================================================================================

Sizing carries the source's portfolio vol-regime factor (``_fred_band_score``):
structural, no param, it shrinks the low-vol VIX bucket to ``_RS_BAND_FACTOR``.
It is the ONE read here taken at the base-bar cursor rather than on a completed
daily bucket, and deliberately so: the FRED series is a daily grid indexed at
midnight and forward-filled, so the print resolved at an hourly cursor is at
most the previous session's close — the same value a daily-bar run reads for
that session — and it steps at the first base bar of each new day, which is the
same bar the entry is evaluated on. Gating it to the last completed daily bucket
would hand the hourly variant a print one session older than the source's and
is not what removes lookahead (``searchsorted`` already cannot return a print
dated after the cursor). See the ``_FRED_ASSET`` block for the measurement.

Daily echo: this is a daily system — every rule holds on 1d, and every position
it decides to open IS the position the source strategy holds, with sizing and
stop fixed against the same daily inputs. The base bar is used for one thing
only: choosing the fill. An entry whose daily rule goes true is emitted once, on
the first base bar after the bucket it read closed, and fills at the NEXT base
bar's open — the following base bar, which is still inside the same session.
The signal, the size and the stop are the daily strategy's; only the fill's
timing and price differ.

The fill lands LATER than the source's, not earlier: the source shifts from its
daily close to the next daily open, while this variant shifts one base bar
forward and fills intraday. Its price is therefore not the source's — it is the
next base bar's open, which can sit far from the source's next-daily-open price
and is not systematically cheaper or more favourable. A daily level the entry
day itself does not hold moves an entry in TIME, not in signal: the position is
still taken on the same daily rule, just on a different bar and at a different
price. That price difference, not any change in the signal, is where the two
runs' results diverge, and it can help or hurt. Both behaviours follow from the
same rule on the same bars; they are not a different signal.

What this variant actually changes, mechanically, is its relationship to the
source's POSITION, not its execution: it is un-locked from the source. After a
``decel`` exit this variant is flat, so it can take the next qualifying entry
signal that the still-holding source must skip while its own position is open.
That is a property of the design (independently managed positions), not an
edge. It is the source of the higher trade count — more trades, not better
trades.

Every rule is therefore evaluated once per completed daily bucket and only the
fill is timed on the base bar. The one base-cadence action left is the trail
re-arm, and it writes the same daily-derived level repeatedly rather than
introducing a base-frequency rule.

The timing choice is also the ONLY execution decision this variant makes, and
it is structural rather than parameterised — no config field selects it, so
there is nothing honest to sweep here (see the param table in the run report).

Once-per-daily-bucket guard (ENTRY and the ``decel`` EXIT are once-per-bucket
rules — only the trail's re-arm runs at base cadence): the engine fires
``on_candle`` on base bars, so the daily rule stays true for every base bar of
the day it fired on and a naive port would open a fresh lot once per base bar
within that session. The guard is therefore structural, not heuristic — an
entry is evaluated ONLY on the first base bar of a completed daily bucket:

1. the daily read's own visible count (``len(ctx.ta.<ind>(sym, interval="1d"))``)
   must have INCREMENTED since the last base bar (``_new_daily_bucket``);
2. the fill lands on the NEXT base bar, still inside that daily bucket, so a
   lot exists before the following base bar and (1) blocks a second open.

Because (2) guarantees a position between the entry's signal bar and its fill,
the invariant holds regardless of how long that fill takes — it never turns
into a second daily-bucket evaluation. The increment can only happen at the
first base bar of a new day (a completed daily bucket is visible one day late,
and the engine never calls ``on_candle`` on a 1d candle), never mid-session.
Both halves are verified against the engine's own store cursor: the daily
visible count is constant across a session and advances by exactly one on the
next session's first base bar.

The exits are behind the SAME guard, for the same reason the entry is: both are
DAILY decisions. ``decel`` is a state rule read off daily VWATR, and the trail
level is a daily distance off a daily high-water mark, so evaluating either on
an hourly bar reads the day's rule at the day's first hour instead of at its
close — firing the divergence exit into a partially-formed bucket, and
ratcheting the high-water mark off a price the source strategy never sees.
Gating them to the bucket makes the 1h grid select only the fill bar; the daily
rule is otherwise identical to the source's.

The trail is the one exception to that gate, and only in its FIRING: the level
is still recomputed from the daily read (so the number is the source's), but it
is re-armed on every base bar. ``ctx.set_stops`` ratchets tighten-only and fires
intrabar from the next bar, so re-arming an unchanged daily level is idempotent
and loses nothing, while re-arming at base cadence keeps the position covered
between daily shots. Gating the arm to the bucket too would leave the level
stale for a full session and stretch the arm-to-fire gap, which is why the
re-arm is continuous (see ``_TRAIL_REARM_CONTINUOUS``).

Data contract: the ``1d`` reads need a daily warm-up measured in TRADING days.
``vwatr_baseline`` on ``1d`` is NaN until ``vwatr_period + vwatr_base_win``
completed daily buckets exist, and the last bucket is always hidden (completed
buckets only — the forming daily bar is invisible), so the config's ``warmup``
must cover at least that many trading days or the run trades on NaNs.

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

STRATEGY_TYPE = "vwatr_div_htf_dsl"
_STATE_KEY = "vwatr_div_htf_state"

#: The interval every SIGNAL and STATE read uses. One constant, one spelling:
#: the base bar (``config.bars[0]``) only times the fill, this carries the rule.
_STATE_INTERVAL = "1d"


@dataclass(frozen=True)
class _State:
    """Per-symbol cross-call state (``ctx.shared``, per run)."""

    best: float | None = None
    #: The run's first observed visible daily-bucket count, used as the
    #: increment baseline. Adopting it at the first base bar keeps state
    #: consistent with a run SEEDED with an open book (a live cycle), which
    #: starts flat inside its window.
    daily_count: int | None = None


@dataclass(frozen=True)
class Params(StrategyParams):
    # -- VWATR: the signal, the stop and their units are the source's --
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


#: FRED (VIX) derisking -- STRUCTURAL, no params. Percentile of the latest VIX
#: print against its prior window; shrink the low-vol bucket only. Rising vol is
#: NOT the risk on this population, and narrowing to the low band measured
#: better OOS than the older mid-band shrink.
#:
#: INTERVAL: the series is DAILY, and the print visible at the cursor is the one
#: dated at or before this BASE (hourly) bar. That is not a daily STATE read, so
#: it is not behind ``_new_daily_bucket``: ``load_daily`` forward-fills VIX onto
#: a daily grid indexed at midnight, so the print at or before an hourly cursor
#: is at most the previous session's close -- the value a daily-bar run reads for
#: the same session. It also moves at the first base bar of each new day, i.e.
#: the same bucket where the entry is evaluated, so the gate never lags a day
#: behind the signal. Gating it to the last COMPLETED daily bucket instead would
#: subtract a full session of information the source has (a case-A/B run measured
#: that), and it would not remove lookahead -- see ``_fred_band_score``.
_FRED_ASSET: str = "vix"
_FRED_WIN: int = 250
_RS_BAND_LO: float = 0.45
#: Low-vol size factor. Lower is the Sharpe end of the frontier, higher the
#: Ann end; the shipped value takes the Ann, since Ann is the binding
#: constraint at the chosen risk/amp.
_RS_BAND_FACTOR: float = 0.80
_FRED_CACHE_KEY = "vwatr_div_htf_fred_series"


def _fred_series(ctx: StrategyContext) -> tuple[np.ndarray, np.ndarray] | None:
    """VIX prints as ``(dates, values)``, cached in ``ctx.shared`` per run.

    ``load_daily`` forward-fills blanks on a daily grid, so the array is dense
    and a cursor lookup is one searchsorted -- no per-bar disk read, and no way
    to see a print dated after the cursor.
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
    """Portfolio-level VIX factor: ``_RS_BAND_FACTOR`` in the low-vol bucket,
    neutral otherwise. Neutral too when the series is missing or the cursor
    precedes it -- a missing print is not a vol signal.

    Lookahead-free at ANY cursor resolution: ``searchsorted(dates, ts) - 1``
    takes the greatest grid date ``<= ctx.candle.timestamp``, and the grid is
    the dense forward-filled one, so the print it resolves was released no later
    than that bar (a timestamp beyond the file's last print resolves to the last
    old print, never to a newer one). An hourly cursor therefore sees exactly
    what a daily-bar run sees at the close of the session it falls in.
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


#: Trend amplifier constants. The horizon is a slow SMA slope because short
#: horizons carry no measurable rank IC against realized trades while the long
#: slope repeats its sign out of sample. The gate is a "leg up": breadth, i.e.
#: enough of the tradable universe above its own SMA. Both are DAILY bars: the
#: regime breadth is a daily property, and the trend STATE must be the source
#: strategy's -- a 50-bar window here would fold ~7 base bars per session and
#: read a 1-week signal instead of the 50-session one this constant names.
_TREND_LOOK: int = 50
_TREND_GATE_LO: float = 0.5
_TREND_CACHE_KEY = "vwatr_div_htf_trend"
_COHORT_CACHE_KEY = "vwatr_div_htf_cohort"
#: Book-slot cap: per-name risk is scaled by ``_BOOK_SLOTS / book`` once the book
#: (open names + this bar's cohort) exceeds it, so a dense trend tape throttles
#: size BEFORE the engine silently cohort-scales. Lets risk_pct/amp rise without
#: cash-exhaustion scaling. Plateau over 8-12 slots; 10 is the flat middle.
_BOOK_SLOTS: int = 10

#: Structural (no config field selects it): re-arm the daily-derived trail on
#: every base bar, not only on completed-bucket bars. The trail's LEVEL is a
#: daily quantity and is always recomputed from the daily read; its FIRING is a
#: price stop that ``ctx.set_stops`` ratchets tighten-only and fires intrabar.
#: Re-arming an unchanged level is therefore a no-op, while gating the arm to
#: the bucket would leave the level stale for a session and stretch the
#: arm-to-fire gap the source does not have. Set to False to gate the trail to
#: the bucket along with the ``decel`` exit.
_TRAIL_REARM_CONTINUOUS: bool = True


def _new_daily_bucket(holder: dict[str, _State], sym: str, n_daily: int) -> bool:
    """True when ``sym`` has one MORE visible completed daily bucket than the
    last base bar — i.e. this base bar is the bucket's first.

    The gate that keeps a daily rule from re-firing on every base bar of the
    day it fired on: the entry is evaluated only when this returns True. The
    comparison count is the call's own daily read length, which is the same
    cursor-truncated number the entry copies, so a run never has to choose a
    reference symbol or an interval grid.
    """
    state = holder.get(sym) or _State()
    last = state.daily_count
    if last is None:  # first base bar this run processed for ``sym``
        holder[sym] = dataclasses.replace(state, daily_count=n_daily)
        return False
    holder[sym] = dataclasses.replace(state, daily_count=max(last, n_daily))
    return n_daily > last


def _breakout_count(ctx: StrategyContext, p: Params) -> int:
    """Number of flat symbols whose DAILY entry rule is true on this bar.

    Cached per timestamp; bounded. Approximates the cohort the engine will
    execute in one bar cycle, so size can be damped before the engine silently
    scales it — a cohort is one day's breakouts at most, since a daily signal
    fires once per daily bucket. Idempotent: it only reads.
    """
    cache: dict[object, int] = ctx.shared.setdefault(_COHORT_CACHE_KEY, {})
    ts = ctx.candle.timestamp
    if ts in cache:
        return cache[ts]
    # The engine only executes signals for symbols that clear the rolling
    # universe selection, so the cohort the book-slot cap must absorb is the
    # daily-breakout count WITHIN that selected set -- not the whole configured
    # universe. Counting all names overstates the cohort and keeps the throttle
    # (<_BOOK_SLOTS) permanently engaged, halving deployment on broad configs.
    # Missing names are deselected, so they open nothing and must not inflate
    # the damp.
    tradable = _selection(ctx, p) if p.select_top_n > 0 else None
    count = 0
    for sym in ctx.symbols:
        if sym.startswith("GATE_") or ctx.quantity(sym) != 0:
            continue
        if tradable is not None and sym not in tradable:
            continue
        o = ctx.ohlcv(sym, _STATE_INTERVAL)
        if o is None:
            continue
        close = o.close.to_array()
        nd = len(close)
        if nd < p.breakout_look + 1:
            continue
        px = float(close[-1])
        v = ctx.ta.vwatr(sym, p.vwatr_period, interval=_STATE_INTERVAL).to_array()
        if not np.isfinite(px) or px <= 0 or v.size < nd:
            continue
        v_now = float(v[nd - 1])
        v_prev = float(v[nd - 1 - p.slope_bars])
        if not (np.isfinite(v_now) and np.isfinite(v_prev)) or not (v_now > v_prev):
            continue
        prior = close[nd - 1 - p.breakout_look : nd - 1]
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
    """Per-bar map of DAILY slope percentile + the leg-up gate, computed ONCE.

    Slope = the ``_TREND_LOOK``-session SMA vs the SMA one horizon earlier, as a
    fraction of price. ``pct`` is the symbol's rank among the tradable universe
    at this bar (0..1), i.e. the cross-sectional selection score. ``leg_up`` is
    the breadth gate. Results are cached under the current timestamp in
    ``ctx.shared`` because ``on_candle`` invokes one symbol per call.
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
        o = ctx.ohlcv(sym, _STATE_INTERVAL)
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
    # The source rule, unchanged: breadth is the share of the readable universe
    # above its own SMA. No minimum-size guard -- at one symbol the ratio is 0 or
    # 1 over a denominator of one, which is the source's own behaviour and part
    # of the signal being ported, not a degenerate case to correct.
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
    """One amplifier knob: non-conserved trend re-lever inside a leg up, plus
    an optional conserved dollar-volume top-K concentration.

    The two effects multiply and each is load-bearing:

    1. Re-lever (``risk_amp``): ``1 + amp*(2*trend_pct - 1)`` inside a leg-up
       breadth gate. Boosting ALL names in a strong tape by their daily trend
       rank is what lifts return (the top names also carry the PnL); gating it
       to the leg-up is what keeps it from blowing up in a bear. ``amp=0`` or
       gate shut -> 1.0 (flat).
    2. Concentration (``risk_topk > 0``, ``_DV_CONC`` strength): the top-K
       names by trailing-60d dollar volume get ``x(1+conc)`` and the rest a
       compensating cut so the cohort budget is unchanged. Dollar volume is the
       OOS-stable predictor of which names carry the realized PnL (per-trade
       RETURN is unpredictable at ~0 IC); this re-deploys the same budget onto
       them. ``risk_topk=0`` -> no concentration.

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


#: Dollar-volume concentration. Cross-sectional percentile of trailing 60d
#: dollar volume. Unlike trend slope it is a SIZE/participation signal: names
#: with heavy recent dollar flow are the ones that carry the realized PnL in
#: this long-only breakout book (per-trade-PnL rank IC ~ +0.09 is the single
#: most OOS-stable predictor measured; per-trade RETURN is unpredictable at
#: ~0 IC across every trailing stat). Sized onto live equity, a high-dv name
#: is where a breakout is actually being paid for -- that's the size target.
#: top-K names draw more, the rest a compensating cut, so total deployment
#: (and so cash-exhaustion scaling) is unchanged. Consumed by _dv_topk_score.
_DV_LOOK: int = 60
_DV_CACHE_KEY = "vwatr_div_htf_dv"
#: Conserved top-K concentration strength (structural). Boosting more leaks
#: cash-exhaustion scaling; the shipped value was A/B'd on the broad book.
_DV_CONC: float = 1.0


def _dv_cross_section(ctx: StrategyContext) -> dict[str, float]:
    """Per-symbol trailing-60d dollar-volume percentile within the universe,
    computed ONCE per timestamp and cached in ``ctx.shared`` (bounded)."""
    cache: dict[object, dict[str, float]] = ctx.shared.setdefault(_DV_CACHE_KEY, {})
    ts = ctx.candle.timestamp
    if ts in cache:
        return cache[ts]
    dv: dict[str, float] = {}
    for sym in ctx.symbols:
        if sym.startswith("GATE_"):
            continue
        o = ctx.ohlcv(sym, _STATE_INTERVAL)
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
    """Conserved top-K concentration on trailing-60d dollar volume.

    The top-``risk_topk`` most-liquid names of the cohort get ``x(1+conc)``
    and the rest a compensating cut ``x(1 - conc*k/rest)`` so the cohort's
    total risk budget sums to ~n (same deployment, no added cash-exhaustion
    scaling). Dollar volume is the OOS-stable predictor of which names carry
    the realized PnL, so the budget is re-deployed onto them. Neutral (1.0)
    when the symbol has no dv read or the cohort is empty.
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


#: Rolling selection: the top-``select_top_n`` names by the combined rank of
#: trailing dollar volume and realized vol -- the two features that separate a
#: live, actively traded high-vol book from the rest of a research universe.
#: Both are DAILY reads: the feature describes a name over months, and keeping
#: each name's window the same length as the source's keeps the selected set
#: identical. Selection is recomputed every ``select_bars`` completed DAILY
#: buckets and held in ``ctx.shared`` (per-run by construction). Deselected
#: names are NOT force-closed: the trail and the divergence exit still own every
#: open position.
_SELECT_CACHE_KEY = "vwatr_div_htf_selection"
_SELECT_LOOK: int = 60


def _pct_rank(vals: dict[str, float], sym: str) -> float:
    """Fraction of the cross-section strictly below ``vals[sym]`` (0..1)."""
    v = vals[sym]
    below = sum(1 for u in vals.values() if u < v)
    ties = sum(1 for u in vals.values() if u == v) - 1
    return (below + 0.5 * ties) / max(1, len(vals) - 1)


def _selection(ctx: StrategyContext, p: Params) -> frozenset[str]:
    """Tradable set for the current bar: the last rebalance's picks.

    Re-selects when ``select_bars`` DAILY buckets have elapsed since the last
    rebalance (counted from the reference symbol's completed daily buckets, the
    same clock the selection features are read on). Names without
    ``_SELECT_LOOK`` daily bars of history are never picked -- a fresh listing is
    not a liquidity signal.
    """
    holder: dict[str, object] = ctx.shared.setdefault(_SELECT_CACHE_KEY, {})
    ref = next((s for s in ctx.symbols if not s.startswith("GATE_")), None)
    frame = ctx.ohlcv(ref, _STATE_INTERVAL) if ref is not None else None
    days = 0 if frame is None else len(frame.close)
    last_days = holder.get("days")
    if (
        isinstance(last_days, int)
        and days - last_days < p.select_bars
        and "picks" in holder
    ):
        return cast("frozenset[str]", holder["picks"])

    dv: dict[str, float] = {}
    vol: dict[str, float] = {}
    for sym in ctx.symbols:
        if sym.startswith("GATE_"):
            continue
        o = ctx.ohlcv(sym, _STATE_INTERVAL)
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
    holder["days"] = days
    holder["picks"] = picks
    return picks


def _daily_reads(
    ctx: StrategyContext, p: Params, sym: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """``(close, vwatr, baseline)`` on completed DAILY buckets, or None.

    One prefetched compute per series per run (``ctx.ta`` memoises), read
    cursor-truncated. Every signal, stop and sizing number on this strategy
    comes from these arrays, so they share one spelling and one interval.
    """
    o = ctx.ohlcv(sym, _STATE_INTERVAL)
    if o is None or len(o.close) == 0:
        return None
    close = o.close.to_array()
    vwatr = ctx.ta.vwatr(sym, p.vwatr_period, interval=_STATE_INTERVAL).to_array()
    base = ctx.ta.vwatr_baseline(
        sym, p.vwatr_period, p.vwatr_base_win, interval=_STATE_INTERVAL
    ).to_array()
    if close.size == 0 or vwatr.size < close.size or base.size < close.size:
        return None
    return close, vwatr, base


def _exit_daily(
    ctx: StrategyContext, p: Params, sym: str, state: _State, px: float
) -> _State:
    """The source exit path, on the daily read: divergence close or daily trail.

    Called only on the first base bar of a completed daily bucket (the same gate
    the entry uses), so the daily rule is read where a daily-bar run would read
    it — at the bucket close — and the base bar's only remaining job is to time
    the fill.

    ``decel`` is a STATE rule (the daily VWATR expansion decelerating against the
    prior window) gated on a new DAILY high, and it closes at the next base open.
    The chandelier trail is a PRICE level: its value is recomputed here from the
    daily read, and ``_arm_trail`` keeps it (re-)armed between daily reads so it
    still fires intrabar.
    """
    v_d, nd, best, at_new_high = _exit_inputs(ctx, p, sym, state)
    if best is None:
        return state

    # 1) DAILY slope divergence at a new DAILY high — a STATE rule (the daily
    # VWATR expansion decelerating against the prior window), not a price level,
    # so it stays a next-open close. ``at_new_high`` is the daily analogue of the
    # source's ``px >= best``: the last completed daily close is at least the
    # prior running best. Testing the base close instead is what fired this exit
    # on the day's FIRST base bar, into a partly-formed bucket.
    if (
        p.decel_ratio > 0
        and at_new_high
        and _slope_decelerating(v_d, nd, p.slope_bars, p.decel_ratio)
    ):
        ctx.close(
            sym,
            reason=(
                f"decel {sym}: {best:.2f} at 1d best, "
                f"1d slope < {p.decel_ratio:.2f}× prior"
            ),
        )
        return dataclasses.replace(state, best=None)

    # 2) chandelier trail: ARM the daily level (ratchet-only) for the NEXT base
    # bar instead of closing at the next open. A ``set_stops`` does NOT clear
    # ``best`` — the trail lives with the position; the armed level fires
    # intrabar on the next base bar's low, gap-adjusted. ``tp`` intentionally
    # unused: no take-profit on this strategy.
    return _arm_trail(ctx, p, sym, state, best, px)


def _arm_trail(
    ctx: StrategyContext, p: Params, sym: str, state: _State, best: float, px: float
) -> _State:
    """Arm the daily-derived chandelier level for the NEXT base bar.

    ``best`` and the VWATR distance are both DAILY quantities, so the level is
    the one the source strategy would carry. ``ctx.set_stops`` is tighten-only,
    so re-arming the same level is idempotent: calling this on every base bar
    costs nothing and keeps a live price stop between daily reads.
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
            f"off 1d best {best:.2f}"
        ),
    )
    return dataclasses.replace(state, best=best)


def _rearm_trail(
    ctx: StrategyContext, p: Params, sym: str, state: _State, px: float
) -> _State:
    """Refresh the daily high-water mark + trail arm on a NON-bucket base bar.

    The divergence exit is deliberately absent: it is a DAILY state rule and
    only ``_exit_daily`` (bucket bars) may fire it. This path exists so the
    trail — a PRICE level — stays armed and current between daily reads.
    """
    _, nd, best, _ = _exit_inputs(ctx, p, sym, state)
    if best is None or nd == 0:
        return state
    return _arm_trail(ctx, p, sym, state, best, px)


def _exit_inputs(
    ctx: StrategyContext, p: Params, sym: str, state: _State
) -> tuple[np.ndarray, int, float | None, bool]:
    """Daily VWATR array + its length + the running DAILY high-water mark
    (advanced to the last completed daily close) + whether that close set a new
    high.

    The high-water mark is the last COMPLETED daily close, not the base-bar
    ``px``: ``best`` names a daily quantity (the source strategy's running best
    daily close), and ratcheting it off an hourly close runs it ahead of the
    daily high-water mark the trail and the divergence test are written against.
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


def _process_symbol(ctx: StrategyContext, p: Params, sym: str) -> None:
    holder: dict[str, _State] = ctx.shared.setdefault(_STATE_KEY, {})
    state = holder.get(sym) or _State()
    holder[sym] = state

    reads = _daily_reads(ctx, p, sym)
    if reads is None:
        return
    close_d, v_d, base_d = reads
    nd = close_d.size

    # The once-per-daily-bucket guard (see the module docstring). Consumes the
    # daily read length unconditionally, so the baseline is adopted on the first
    # base bar even while flat and never re-baselines later.
    new_bucket = _new_daily_bucket(holder, sym, nd)
    state = holder[sym]

    n_base = len(ctx.ohlcv(sym).close)
    px = float(ctx.ohlcv(sym).close[-1])
    if n_base == 0 or not np.isfinite(px) or px <= 0:
        return

    # ---- exit while long ----
    if ctx.quantity(sym) != 0:
        if not new_bucket:
            # Between daily reads the base bar owns one action only: keep the
            # daily-derived trail level armed. It is a PRICE stop, not a state
            # rule, so re-arming it at base cadence is idempotent
            # (``set_stops`` ratchets tighten-only) and keeps the position
            # covered until the next bucket closes. The ``decel`` divergence
            # exit is a DAILY state rule and waits for the bucket.
            if _TRAIL_REARM_CONTINUOUS:
                holder[sym] = _rearm_trail(ctx, p, sym, state, px)
            return
        holder[sym] = _exit_daily(ctx, p, sym, state, px)
        return

    if not new_bucket:
        return  # entry is a DAILY decision: once per completed bucket

    # ---- entry: rising DAILY expansion + daily structural breakout ----
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

    # Daily breakout: the last completed session's close above the max of the
    # prior ``breakout_look`` daily closes -- the source rule, same bar count.
    prior = close_d[nd - 1 - p.breakout_look : nd - 1]
    if prior.size < p.breakout_look or not np.all(np.isfinite(prior)):
        return
    prior_high = float(np.max(prior))
    if not (float(close_d[nd - 1]) > prior_high):
        return

    # Breadth gate: only take breakouts while the universe is in a leg up
    # (>= _TREND_GATE_LO above their own 50d SMA). Structural, no param -- the
    # same ``leg_up`` the trend amp uses, so a bear tape stops new risk.
    _, leg_up = _trend_cross_section(ctx, p)
    if not leg_up:
        return

    # Rolling universe selection: only names picked at the last rebalance may
    # open. Off by default, so the tuned configs are byte-for-byte unchanged.
    if p.select_top_n > 0 and sym not in _selection(ctx, p):
        return

    # The stop distance is the DAILY VWATR: same risk unit the source strategy
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
        f"{sym}: 1d close {float(close_d[nd - 1]):.2f} > {prior_high:.2f} "
        f"({p.breakout_look}d), "
        f"1d vwatr {vwatr_d:.3f} {vwatr_d / base_now:.2f}× base {base_now:.3f}, "
        f"↑ {vwatr_d_prev:.3f}/{p.slope_bars}d, "
        f"risk {risk_eff:.1%} ×{score:.2f}, stop {stop_price:.2f}, "
        f"fill @ base {px:.2f}"
    )

    # Seed the high-water mark from the DAILY close that triggered the entry, not
    # the base-bar fill price: ``best`` is a daily quantity everywhere else on
    # this strategy (see ``_exit_inputs``), and a base-bar seed would run it
    # ahead of the daily mark before the first bucket closes.
    holder[sym] = dataclasses.replace(state, best=float(close_d[nd - 1]))
    # Arming the initial stop in the SAME call via ``sl`` (ABSOLUTE PRICE):
    # ``stop_price`` is the DISTANCE at entry − vwatr_mult*VWATR, so subtract it
    # from the base-bar price the fill is timed off instead of passing it as a
    # fraction.
    ctx.long(
        sym,
        size=size,
        size_mode="equity",
        reason=reason,
        sl=px - stop_price,
    )


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

    The price pane is the BASE (execution) feed — that is where the fill lands
    and where the trail's high-water mark lives. The daily signal series is
    reindexed onto the base index with a forward fill, which on the
    completed-bucket feed means a daily value only paints from the bar AFTER its
    session closed.
    """
    sym = ctx.candle.symbol
    df = ctx.state.candles.get((sym, ctx.interval))
    if df is None or len(df) < params.vwatr_period + params.vwatr_base_win:
        return PlotSpec()

    close = ctx.ta.close(sym).to_array()
    daily = ctx.ohlcv(sym, _STATE_INTERVAL)
    daily_close = np.full(close.shape, np.nan, dtype=np.float64)
    daily_slope = np.full(close.shape, np.nan, dtype=np.float64)
    if daily is not None and len(daily.close) > 2 * params.slope_bars:
        dc = daily.close.to_array()
        v_d = ctx.ta.vwatr(
            sym, params.vwatr_period, interval=_STATE_INTERVAL
        ).to_array()
        d_slope = v_d - _shift(v_d, params.slope_bars)
        # Each daily bucket paints from the next base bar onwards (its session
        # has closed by then), so offset the mapped values by one base bar.
        daily_close[1 : 1 + dc.size] = dc[: len(daily_close) - 1]
        daily_slope[1 : 1 + d_slope.size] = d_slope[: len(daily_slope) - 1]

    stop_band = params.vwatr_mult * daily_close
    upper = daily_close + stop_band
    lower = daily_close - stop_band

    look = params.breakout_look
    channel = np.full(close.shape, np.nan, dtype=np.float64)
    if len(daily_close) > look:
        from numpy.lib.stride_tricks import sliding_window_view

        channel[look:] = sliding_window_view(daily_close, look).max(axis=1)[:-1]

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
            Overlay(
                sparse(_as_series(df.index, daily_close)), "1d close", "line", "price"
            ),
            Overlay(
                sparse(_as_series(df.index, channel)),
                f"breakout_high ({look}d)",
                "line",
                "price",
            ),
            Overlay(sparse(_as_series(df.index, upper)), "trail_hi", "line", "price"),
            Overlay(sparse(_as_series(df.index, lower)), "trail_lo", "line", "price"),
        ),
        # slope pair only: vwatr/close = noise; baseline div failed. The signal
        # interval is the daily one; the base bar only times the fill.
        panels=(
            Panel(
                sparse(_as_series(df.index, daily_slope)),
                f"1d vwatr slope ({params.slope_bars}d, entry + exit state)",
                (0.0,),
            ),
        ),
        markers=tuple(markers),
    )
