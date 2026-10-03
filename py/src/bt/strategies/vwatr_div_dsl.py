"""VWATR divergence strategy -- long only, one unconditional breakout entry.

Entry: rising VWATR expansion on a breakout above the recent high, taken only
while the tradable universe is in a breadth leg up (``_trend_cross_section``).
Within that gate the trigger is unconditional by design -- entry-side features
carry no measurable edge on this population, so name selection lives in sizing
instead. The gate cuts the 2022-style drawdown depth and lifts Sharpe/Ann
out of sample (mean OOS 1.37 -> 1.46, min -1.11 -> -0.91); it does not shorten
recovery time -- a 575d underwater span is the bear's, not the rule's.

Exit, in order:

1. VWATR slope divergence while price is at a new high. This is the primary
   exit and it is load-bearing: turning it off lowers return AND worsens
   drawdown and tail risk -- it is not a free early out.
2. ``vwatr_mult`` chandelier trail off the running best close. Width is
   load-bearing too -- widening it loses Sharpe monotonically.

Sizing: flat ``risk_pct`` of LIVE equity times the amplifiers, so the risk
budget re-levers as the book compounds. Sizing off a frozen capital base
instead cuts cash-exhaustion fill scaling but caps compounding, and therefore
caps return. A structural book-slot cap (``_BOOK_SLOTS``) throttles per-name
size once open names plus the bar's cohort exceed 10, so a dense trend tape
self-scales instead of leaving it to the engine's silent cohort scale.

Amplifiers, both neutral by default:

- Cross-sectional trend score (``_trend_risk_score``): only a slow SMA slope
  pays. Short-lookback slopes screen at ~zero rank IC, and an EMA variant
  loses in every tested cell -- EMA vs EMA over overlapping windows compares
  decayed points, while SMA compares two disjoint windows.
- Portfolio vol-regime factor (``_fred_band_score``): shrink the low-vol bucket
  only. Rising vol is not the risk on this population, and the older mid-band
  shrink cut the second-best bucket while leaving the weakest at full size.

The binding constraint is cash-exhaustion fill scaling (see AGENTS.md § Fills),
not Sharpe: every extra unit of return at fixed per-name risk is bought with
silently scaled opens, so the honest point is the highest-return node whose
Scaled count is still zero.

Data contract: ``ctx.ta.vwatr`` / ``ctx.ta.vwatr_baseline`` are shared TA using
a PLAIN rolling-mean ATR (not Wilder) -- swapping in ``ctx.ta.atr`` silently
changes every stop distance.

Cut for evidence; do not re-add without a sweep: ``risk_symvol_amp`` (negative
rank IC), the FRED + earnings + clamp sizing layer (inert once the cap
collapses), ``norm_mode``, ``baseline_gate``, ``risk_trend_kind``. The exit path
and risk/amp surface are plateaus, not spikes -- re-sweep before changing a
default, and A/B any re-add rather than trusting prose.

Single kill switch: ``decel_ratio=0`` turns the divergence exit OFF.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

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


@dataclass(frozen=True)
class _State:
    best: float | None = None


@dataclass(frozen=True)
class Params(StrategyParams):
    # -- VWATR --
    vwatr_period: int = 14
    vwatr_base_win: int = 60
    vwatr_mult: float = 3.0  # chandelier trail width AND the sizing stop
    slope_bars: int = 2
    breakout_look: int = 20
    # -- sizing: flat risk budget times ONE trend amplifier --
    risk_pct: float = 0.004
    risk_trend_amp: float = 0.0  # 0 = flat sizing; see _trend_risk_score
    # -- exit --
    decel_ratio: float = 0.1  # VWATR slope-divergence exit; 0 turns it OFF


@strategy(bars="1d", stateful=True)
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
_FRED_ASSET: str = "vix"
_FRED_WIN: int = 250
_RS_BAND_LO: float = 0.45
#: Low-vol size factor. Lower is the Sharpe end of the frontier, higher the
#: Ann end; the shipped value takes the Ann, since Ann is the binding
#: constraint at the chosen risk/amp.
_RS_BAND_FACTOR: float = 0.80
_FRED_CACHE_KEY = "vwatr_div_fred_series"


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
#: enough of the tradable universe above its own SMA.
_TREND_LOOK: int = 50
_TREND_GATE_LO: float = 0.5
_TREND_CACHE_KEY = "vwatr_div_trend"
_COHORT_CACHE_KEY = "vwatr_div_cohort"
#: Book-slot cap: per-name risk is scaled by ``_BOOK_SLOTS / book`` once the book
#: (open names + this bar's cohort) exceeds it, so a dense trend tape throttles
#: size BEFORE the engine silently cohort-scales. Lets risk_pct/amp rise without
#: cash-exhaustion scaling. Plateau over 8-12 slots; 10 is the flat middle.
_BOOK_SLOTS: int = 10


def _breakout_count(ctx: StrategyContext, p: Params) -> int:
    """Number of flat symbols that fire an entry on THIS bar (cohort size).

    Cached per timestamp; bounded. Approximates the cohort the engine will
    execute in one bar cycle, so size can be damped before the engine silently
    scales it.
    """
    cache: dict[object, int] = ctx.shared.setdefault(_COHORT_CACHE_KEY, {})
    ts = ctx.candle.timestamp
    if ts in cache:
        return cache[ts]
    need = max(p.breakout_look + 1, 2 * p.slope_bars + 1)
    count = 0
    for sym in ctx.symbols:
        if sym.startswith("GATE_") or ctx.quantity(sym) != 0:
            continue
        o = ctx.ohlcv(sym)
        if o is None:
            continue
        close = o.close.to_array()
        m = len(close)
        if m < need:
            continue
        px = float(close[-1])
        v = ctx.ta.vwatr(sym, p.vwatr_period).to_array()
        if m == 0 or not np.isfinite(px) or px <= 0:
            continue
        v_now = float(v[-1])
        v_prev = float(v[m - 1 - p.slope_bars])
        if not (np.isfinite(v_now) and np.isfinite(v_prev)) or not (v_now > v_prev):
            continue
        prior = close[m - 1 - p.breakout_look : m - 1]
        if len(prior) < p.breakout_look or not np.all(np.isfinite(prior)):
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
    """Per-bar map of slope percentile + the leg-up gate, computed ONCE.

    Slope = 50d SMA vs the 50d SMA one horizon earlier, as a fraction of price.
    ``pct`` is the symbol's rank among the tradable universe at this bar (0..1),
    i.e. the cross-sectional selection score. ``leg_up`` is the breadth gate.
    Results are cached under the current timestamp in ``ctx.shared`` because
    ``on_candle`` invokes one symbol per call.
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
        o = ctx.ohlcv(sym)
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


def _trend_risk_score(p: Params, ctx: StrategyContext, sym: str) -> float:
    """Cross-sectional trend amp: ``1 + amp*(2*pct-1)`` inside a leg up, else 1.

    Neutral when the amp is off, the gate is shut, or the symbol has no slope
    at this bar -- a missing trend read is not a weak trend.
    """
    if p.risk_trend_amp == 0:
        return 1.0
    pcts, leg_up = _trend_cross_section(ctx, p)
    if not leg_up or sym not in pcts:
        return 1.0
    return 1.0 + p.risk_trend_amp * (2.0 * pcts[sym] - 1.0)


def _process_symbol(ctx: StrategyContext, p: Params, sym: str) -> None:
    holder: dict[str, _State] = ctx.shared.setdefault(_STATE_KEY, {})
    state = holder.get(sym) or _State()
    holder[sym] = state

    def put(**kw: object) -> None:
        nonlocal state
        state = dataclasses.replace(state, **kw)
        holder[sym] = state

    o = ctx.ohlcv(sym)
    if o is None or len(o.close) == 0:
        return

    close = o.close.to_array()
    n = len(close)

    px = float(close[-1])
    if not np.isfinite(px) or px <= 0:
        return

    vwatr_arr = ctx.ta.vwatr(sym, p.vwatr_period).to_array()
    vwatr = float(vwatr_arr[-1]) if n else float("nan")
    if not np.isfinite(vwatr) or vwatr <= 0:
        return

    base_now = float(ctx.ta.vwatr_baseline(sym, p.vwatr_period, p.vwatr_base_win)[-1])

    # ---- exit while long ----
    if ctx.quantity(sym) != 0:
        best = state.best
        if best is None or not np.isfinite(best):
            put(best=px)
            return
        if px > best:
            put(best=px)
            best = px

        reason: str | None = None

        # 1) slope divergence under a new high
        if (
            p.decel_ratio > 0
            and px >= best
            and _slope_decelerating(vwatr_arr, n, p.slope_bars, p.decel_ratio)
        ):
            reason = (
                f"decel {sym}: {px:.2f} at best, slope < {p.decel_ratio:.2f}× prior"
            )
        if reason is not None:
            put(best=None)
            ctx.close(sym, reason=reason)
            return

        # base chandelier trail
        stop_distance = p.vwatr_mult * vwatr
        if px <= best - stop_distance:
            put(best=None)
            ctx.close(
                sym,
                reason=(
                    f"trail {sym}: {px:.2f} −{stop_distance:.2f} "
                    f"({stop_distance / px:.2%}) off best {best:.2f}"
                ),
            )
        return

    # ---- entry: rising expansion + structural breakout ----
    if not np.isfinite(base_now) or base_now <= 0:
        return

    if n <= p.slope_bars:
        return
    vwatr_prev = float(vwatr_arr[n - 1 - p.slope_bars])
    if not np.isfinite(vwatr_prev):
        return
    if not (vwatr > vwatr_prev):
        return

    prior = close[n - 1 - p.breakout_look : n - 1]
    if len(prior) < p.breakout_look or not np.all(np.isfinite(prior)):
        return
    prior_high = float(np.max(prior))
    if not (px > prior_high):
        return

    # Breadth gate: only take breakouts while the universe is in a leg up
    # (>= _TREND_GATE_LO above their own SMA). Structural, no param -- the
    # same ``leg_up`` the trend amp uses, so a bear tape stops new risk.
    _, leg_up = _trend_cross_section(ctx, p)
    if not leg_up:
        return

    stop_price = p.vwatr_mult * vwatr
    # Size off LIVE EQUITY: the risk budget re-levers with the book, which is
    # what keeps deployment (and therefore Ann) up as equity compounds.
    equity = ctx.current_equity()
    score = _fred_band_score(ctx) * _trend_risk_score(p, ctx, sym)
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
        f"{sym}: {px:.2f} > {prior_high:.2f}, "
        f"vwatr {vwatr:.3f} {vwatr / base_now:.2f}× base {base_now:.3f}, "
        f"↑ {vwatr_prev:.3f}/{p.slope_bars}b, "
        f"risk {risk_eff:.1%} ×{score:.2f}, stop {stop_price:.2f}"
    )

    put(best=px)
    ctx.long(sym, size=size, size_mode="equity", reason=reason)


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
    full visible history)."""
    sym = ctx.candle.symbol
    df = ctx.state.candles.get((sym, ctx.interval))
    if df is None or len(df) < params.vwatr_period + params.vwatr_base_win:
        return PlotSpec()

    close = ctx.ta.close(sym).to_array()
    vwatr_arr = ctx.ta.vwatr(sym, params.vwatr_period).to_array()
    slope = vwatr_arr - _shift(vwatr_arr, params.slope_bars)

    stop_band = params.vwatr_mult * vwatr_arr
    upper = close + stop_band
    lower = close - stop_band

    look = params.breakout_look
    channel = np.full(close.shape, np.nan, dtype=np.float64)
    if len(close) > look:
        from numpy.lib.stride_tricks import sliding_window_view

        channel[look:] = sliding_window_view(close, look).max(axis=1)[:-1]

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
                sparse(_as_series(df.index, channel)), "breakout_high", "line", "price"
            ),
            Overlay(sparse(_as_series(df.index, upper)), "trail_hi", "line", "price"),
            Overlay(sparse(_as_series(df.index, lower)), "trail_lo", "line", "price"),
        ),
        # slope pair only: vwatr/close = noise; baseline div failed.
        panels=(
            Panel(
                sparse(_as_series(df.index, slope)),
                f"vwatr slope ({params.slope_bars}b)",
                (0.0,),
            ),
        ),
        markers=tuple(markers),
    )
