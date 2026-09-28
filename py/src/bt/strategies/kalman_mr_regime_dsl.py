"""Benchmark-gated Kalman mean reversion — DSL strategy.

Fades single-symbol over/under-extension relative to a smoothed Kalman fair
value, but only while a **benchmark** index is in an explicitly RANGING regime.
The regime gate is the whole ballgame: Kalman residual fades only mean-revert
in sideways markets. In a trend the residual is not mean-reverting and this
strategy bleeds — so the benchmark gate both *selects* the tradable range
periods and *kills* the book the moment the benchmark leaves RANGE.

Three layers:

  Layer 1 — Benchmark regime gate (defense in depth).
      + Trend gate: two modes, sweepable via ``trend_gate``.
          - ``reversion`` (default): the benchmark has crossed its rolling
            mean at least ``min_crossings`` times over ``reversion_window`` —
            a genuinely mean-reverting range, not a quiet one-way grind. This
            blocks the 2024-style slow-trend leak.
          - ``slope``: benchmark's normalized OLS log-slope over
            ``range_lookback`` is small (|cumulative move| < ``range_max_move``).
      + Vol gate: benchmark range-vol below its ``range_vol_cap_percentile``
        historical edge (a ranging market is calm; panic-spike volatility is
        not a fadeable range).
      Only when **both** agree RANGE does the book open. Leaving RANGE or an
      expanding-vol warning flattens every position (regime invalidation).

  Layer 2 — Kalman mean reversion (strategy-owned, single-symbol online).
      One :class:`src.indicators.kalman.strategy.OnlineLevel` constant-velocity
      Kalman filter per tradeable symbol is held in ``ctx.shared``, fed the
      latest close **once per candle**, and exposes a standardized
      one-step-ahead residual ``z_stat = (price - predicted)/sigma`` — a fully
      **online** signal (state-only, no rolling window over a fixed history).
      Fade-short when ``z_stat > z_entry``, fade-long when ``z_stat < -z_entry``.

  Layer 3 — Risk (every trade planned before entry).
      + Stop: ``atr_mult * ATR`` (2.0) against entry — implemented manually
        because the engine's SL/TP fields are informational, not enforced.
        Never widened for hope.
      + Sizing: ``allocation`` (default) fans ``symbol_alloc`` (0.20) of
        **initial** capital per symbol — full deployment across a 5-symbol
        basket (5 x 0.20 = 1.0), no leverage (capped at available cash);
        ``risk`` stays risk-targeted (``risk_pct`` of equity / ATR stop).
        Sweepable via ``sizing_mode``.
      + Exits: convergence (|z| < z_exit), ATR stop, or regime invalidation.
        All explicit — nothing called "hmm".

  Layer 3b — Volume confirmation (sweepable via ``volume_mode``). ``off``
      ignores volume; ``conform`` only fades into *drying* volume (<
      ``vol_mult`` x avg) — a range fade, not a breakout; ``reject`` blocks
      only climax/expanding-volume fades (never fade a breakout on rising
      volume). Experiments are driven via ``bt sweep`` on these knobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

from src.indicators.kalman.strategy import OnlineLevel
from src.bt.size.pure import risk_sized_qty
from src.bt.strategies.dsl import strategy, StrategyContext
from src.bt.strategies.series import SeriesView
from src.bt.strategies.types import StrategyParams

STRATEGY_TYPE = "kalman_mr_regime_dsl"

# Keys for per-run state in ctx.shared (minted fresh per run/window).
_FILTERS_KEY = "kalman_filters"  # dict[sym -> OnlineLevel]
_TS_KEY = "kalman_ts"  # dict[sym -> float] current z_stat (once/candle)
_COOLDOWN_KEY = "cooldowns"

# Size modes: ``allocation`` fans a fixed fraction of capital per symbol (full
# deployment, no leverage); ``risk`` sizes from a per-trade risk % and an ATR
# stop. Sweepable via ``sizing_mode``.
SizingMode = Literal["allocation", "risk"]
# Volume confirmation modes (Layer 3b): ``off`` ignores volume; ``conform``
# only fades into drying volume (a range fade, not a breakout); ``reject``
# blocks only climax/expanding-volume fades (fading a breakout).
VolumeMode = Literal["off", "conform", "reject"]


@dataclass(frozen=True)
class Params(StrategyParams):
    benchmark: str = "SPY"
    # -- layer 1: regime gate --
    range_lookback: int = 90  # bars for the benchmark OLS slope
    range_max_move: float = 0.09  # max cum. log move over lookback => RANGE
    range_vol_cap_percentile: float = 0.80  # reject vol this hi or above
    vol_lookback: int = 250
    # - trend-leak fix knob: ``slope`` (favoured) vs ``reversion`` -
    trend_gate: Literal["slope", "reversion"] = "slope"
    reversion_window: int = 60  # SMA width for the reversion test
    min_crossings: int = 2  # mean crossings required over the window
    # -- layer 2: kalman mispricing --
    z_entry: float = 2.0
    z_exit: float = 0.5
    # -- layer 3: risk / sizing --
    sizing_mode: SizingMode = "allocation"
    symbol_alloc: float = 0.20  # fraction of initial capital per symbol
    atr_period: int = 14
    atr_mult: float = 2.0  # stop = atr_mult * ATR against entry
    risk_pct: float = 0.005
    # -- layer 3b: volume confirmation (off favoured; filters hurt edge) --
    volume_mode: VolumeMode = "off"
    vol_period: int = 20
    vol_mult: float = 1.4  # threshold multiple vs avg volume
    # -- misc --
    warmup_bars: int = 150
    cooldown_bars: int = 5
    fades_short: bool = True
    fades_long: bool = True
    # -- kalman hyper-params (online constant-velocity level filter) --
    process_noise: float = 1e-4
    measurement_noise: float = 1e-3


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def _last(v: object) -> float:
    if isinstance(v, SeriesView):
        return float(v[-1])
    # Callers pass numeric scalars (or float-valued SeriesViews handled above).
    return float(cast(float, v))


def _bench_slope_move(ctx: StrategyContext, params: Params) -> float | None:
    """Cumulative log move of the benchmark over ``range_lookback`` bars.

    A small |value| means the benchmark isn't trending (ranging); the sign is
    direction. Computed via OLS log-slope over the lookback, so it's robust to
    within-window jaggedness (a market that goes up-then-down still reads ~0).
    Returns None when the lookback exceeds history (insufficient/warmup).
    """
    o = ctx.ohlcv(params.benchmark)
    n = len(o.close)
    if n < params.range_lookback + 1:
        return None
    lb = params.range_lookback
    y = np.log(np.array([float(o.close[n - lb + i]) for i in range(lb)], dtype=float))
    x = np.arange(lb, dtype=float)
    xc = x - x.mean()
    denom = float(xc @ xc)
    if denom <= 1e-12:
        return None
    slope = float(xc @ (y - y.mean()) / denom)  # per-bar log slope
    return slope * lb  # cumulative log move over the lookback


def _bench_vol_percentile(ctx: StrategyContext, params: Params) -> float:
    """Current benchmark true-range/price percentile over ``vol_lookback``.

    High values (>1e0) mean vol is elevated vs its own history. We only fade
    ranges, so vol above ``range_vol_cap_percentile`` blocks the gate.
    """
    o = ctx.ohlcv(params.benchmark)
    n = len(o.close)
    if n < 30 + 2:
        return 1.0
    lookback = min(params.vol_lookback, n - 2)
    target = _true_range(
        float(o.high[-1]), float(o.low[-1]), float(o.close[-2])
    ) / float(o.close[-1])
    hist: list[float] = []
    for i in range(n - lookback, n - 1):
        tr = _true_range(float(o.high[i]), float(o.low[i]), float(o.close[i - 1]))
        if o.close[i] > 0:
            hist.append(tr / float(o.close[i]))
    if len(hist) < 30:
        return 1.0
    return float(np.mean(np.array(hist) <= target))


def _bench_reverts(ctx: StrategyContext, params: Params) -> bool:
    """True when the benchmark is mean-reverting, not a quiet one-way grind.

    The 2024-style leak is a low-vol *trend* the plain slope gate admits: price
    drifts one way without oscillating. A genuine range crosses its rolling
    mean repeatedly. Require at least ``min_crossings`` sign-flips of
    (close - SMA) within the last ``reversion_window`` bars.
    """
    o = ctx.ohlcv(params.benchmark)
    n = len(o.close)
    w = params.reversion_window
    if n < w + 2:
        return False  # warmup
    ma = float(np.mean([float(o.close[i]) for i in range(n - w, n)]))
    crossings = 0
    prev = float(o.close[n - w]) - ma
    for i in range(n - w + 1, n):
        cur = float(o.close[i]) - ma
        if (prev <= 0 < cur) or (prev >= 0 > cur):
            crossings += 1
        prev = cur
    return crossings >= params.min_crossings


def _true_range(h: float, lo: float, prev_close: float) -> float:
    return max(h - lo, abs(h - prev_close), abs(lo - prev_close))


def _regime_ok(ctx: StrategyContext, params: Params) -> bool:
    """Layer 1 — benchmark in explicit RANGE (trend gate AND vol-cap gate)."""
    if params.trend_gate == "slope":
        move = _bench_slope_move(ctx, params)
        if move is None:
            return False
        if abs(move) > params.range_max_move:
            return False
    else:  # "reversion" — the default, fixes the quiet-grind leak
        if not _bench_reverts(ctx, params):
            return False
    if _bench_vol_percentile(ctx, params) > params.range_vol_cap_percentile:
        return False  # elevated vol — not a calm range
    return True


def _volume_ok(ctx: StrategyContext, sym: str, params: Params) -> bool:
    """Layer 3b — volume confirmation for a range fade.

    ``conform``: only fade when volume is drying (< ``vol_mult`` x avg) — a
    range fade, not a breakout (breakouts on rising volume are the opposite of
    what we trade). ``reject``: block only climax/expanding-volume fades
    (volume > ``vol_mult`` x avg), the dangerous breakout-fade. ``off``: no
    volume filter.
    """
    mode = params.volume_mode
    if mode == "off":
        return True
    o = ctx.ohlcv(sym)
    n = len(o.close)
    if n < params.vol_period + 1:
        return False
    vol = float(o.volume[-1])
    avg = float(
        np.mean([float(o.volume[i]) for i in range(n - params.vol_period, n - 1)])
    )
    if avg <= 0:
        return True  # degenerate volume series — don't block on bad data
    ratio = vol / avg
    if mode == "conform":
        return ratio < params.vol_mult
    return ratio <= params.vol_mult  # reject: block only climactic volume spikes


def _refresh_ts(ctx: StrategyContext, params: Params) -> None:
    """Feed each tradeable symbol's online Kalman once per candle.

    Stores the resulting ``z_stat`` (the standardized one-step-ahead residual,
    fully online) in ``ctx.shared`` so exit and entry logic read the same value
    for the same bar — feeding the filter twice would double-advance its state.
    """
    filters = ctx.shared.setdefault(_FILTERS_KEY, {})
    ts: dict[str, float] = {}
    for sym in ctx.symbols:
        if sym == params.benchmark:
            continue
        f = filters.get(sym)
        if not isinstance(f, OnlineLevel):
            continue
        o = ctx.ohlcv(sym)
        if len(o.close) < 2:
            continue
        price = _last(o.close)
        res = f.observe(price)
        if res.ready and res.z_stat is not None:
            ts[sym] = res.z_stat
    ctx.shared[_TS_KEY] = ts


def _t(ctx: StrategyContext, sym: str) -> float | None:
    return ctx.shared.get(_TS_KEY, {}).get(sym)


def _filters(ctx: StrategyContext, params: Params) -> None:
    """Lazily mint one online constant-velocity Kalman per tradeable symbol."""
    store = ctx.shared.setdefault(_FILTERS_KEY, {})
    for sym in ctx.symbols:
        if sym == params.benchmark:
            continue
        if not isinstance(store.get(sym), OnlineLevel):
            store[sym] = OnlineLevel(
                process_noise=params.process_noise,
                measurement_noise=params.measurement_noise,
                warmup_bars=params.warmup_bars,
            )


def _atr_stop_hit(
    ctx: StrategyContext, sym: str, is_long: bool, params: Params
) -> bool:
    """Manual ATR stop — engine SL/TP fields are informational, not enforced."""
    pos = ctx.position(sym)
    if pos is None:
        return False
    atr_val = _last(ctx.ta.atr(sym, params.atr_period))
    if np.isnan(atr_val) or atr_val <= 0:
        return False
    stop_dist = params.atr_mult * atr_val
    o = ctx.ohlcv(sym)
    if is_long:
        return float(o.low[-1]) <= pos.entry_price - stop_dist
    return float(o.high[-1]) >= pos.entry_price + stop_dist


@strategy(bars="1d", stateful=True)
def on_candle(ctx: StrategyContext):
    params: Params = ctx.params
    shared = ctx.shared
    shared.setdefault(_COOLDOWN_KEY, {})

    _filters(ctx, params)
    _refresh_ts(ctx, params)
    # Warmup bars fill strategy state only — no fill can happen there, so the
    # DSL raises on any entry emitted during them.
    if ctx.phase == "warmup":
        return
    regime_ok = _regime_ok(ctx, params)

    # ---- Exits first (always run, even outside the gate) --------------------
    for sym in ctx.symbols:
        if sym == params.benchmark:
            continue
        pos = ctx.position(sym)
        if pos is None:
            continue
        is_long = pos.qty >= 0
        reasons: list[str] = []

        if not regime_ok:
            reasons.append("[regime] benchmark left RANGE -> flatten")
        reason = _exit(ctx, sym, is_long, params)
        if reason:
            reasons.append(reason)
        if not reasons:
            continue

        ctx.close(sym, reason=" | ".join(reasons))
        shared[_COOLDOWN_KEY][sym] = params.cooldown_bars

    if not regime_ok:
        return

    # ---- Entry logic --------------------------------------------------------
    for sym in ctx.symbols:
        if sym == params.benchmark:
            continue
        cd = shared[_COOLDOWN_KEY].get(sym, 0)
        if cd > 0:
            shared[_COOLDOWN_KEY][sym] = cd - 1
            continue
        _enter(ctx, sym, params)


def _exit(ctx: StrategyContext, sym: str, is_long: bool, params: Params) -> str | None:
    """Converged or ATR-stopped? Returns a close reason, else None."""
    # ATR stop (strategy-owned; engine SL fields are not enforced)
    if _atr_stop_hit(ctx, sym, is_long, params):
        return "[stop] ATR stop hit"

    # Convergence: the fade mean-reverted past z_exit
    t = _t(ctx, sym)
    if t is None:
        return None
    if is_long and t <= params.z_exit:
        return f"[converge] long fade done t={t:.2f}"
    if not is_long and t >= -params.z_exit:
        return f"[converge] short fade done t={t:.2f}"
    return None


def _enter(ctx: StrategyContext, sym: str, params: Params) -> None:
    t = _t(ctx, sym)
    if t is None or abs(t) <= params.z_entry:
        return

    o = ctx.ohlcv(sym)
    if len(o.close) < params.warmup_bars:
        return
    price = _last(o.close)
    atr_val = _last(ctx.ta.atr(sym, params.atr_period))
    if price <= 0 or np.isnan(atr_val) or atr_val <= 0:
        return

    if t > params.z_entry:  # overextended above benchmark -> fade short
        if not params.fades_short:
            return
        side = "short"
    elif t < -params.z_entry:  # under-extended below benchmark -> fade long
        if not params.fades_long:
            return
        side = "long"
    else:
        return

    if ctx.position(sym) is not None:
        return

    # Volume confirmation (Layer 3b) — block breakout-fades.
    if not _volume_ok(ctx, sym, params):
        return

    # ---- sizing (mechanical, last step) ---------------------------------
    size = _size_fraction(ctx, sym, params, price, atr_val)
    if size <= 0:
        return

    if side == "long":
        ctx.long(
            sym,
            size=size,
            reason=f"[fade] long undervalued t={t:.2f} vol({params.volume_mode})",
        )
    else:
        ctx.short(
            sym,
            size=size,
            reason=f"[fade] short overvalued t={t:.2f} vol({params.volume_mode})",
        )


def _size_fraction(
    ctx: StrategyContext, sym: str, params: Params, price: float, atr_val: float
) -> float:
    """Return the 0..1 capital fraction to deploy, per ``sizing_mode``.

    ``allocation`` (default): a fixed ``symbol_alloc`` fraction of *initial*
    capital per symbol — full deployment across the basket (5 x 0.20 = 1.0),
    no leverage. ``risk``: back-solve the fraction from per-trade ``risk_pct``
    and the ``atr_mult``-ATR stop, matching the original risk-scaled sizing.
    """
    if params.sizing_mode == "allocation":
        # Fixed ``symbol_alloc`` of *initial* capital per symbol, but never
        # more than available cash (no leverage).
        target = params.symbol_alloc * ctx.state.portfolio.initial_capital
        cash = ctx.state.portfolio.cash
        notional = min(target, max(cash, 0.0))
        if notional <= 0 or price <= 0:
            return 0.0
        return min(max(notional / ctx.state.portfolio.initial_capital, 0.0), 1.0)
    if np.isnan(atr_val) or atr_val <= 0:
        return 0.0
    stop_dist = params.atr_mult * atr_val
    cash = ctx.state.portfolio.cash
    if cash <= 0:
        return 0.0
    qty = risk_sized_qty(
        equity=cash, price=price, stop_dist=stop_dist, risk_pct=params.risk_pct
    )
    if qty <= 0:
        return 0.0
    size = qty * price / ctx.state.portfolio.initial_capital
    return min(max(size, 0.0), 1.0)
