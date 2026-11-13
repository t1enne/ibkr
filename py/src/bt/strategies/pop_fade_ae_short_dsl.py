"""Hourly short-side mean reversion: fade an overextended counter-trend rally.

SHORT-only. A close stretched at least ``min_stretch_atr`` ATRs above its own
short EMA, on expanding volume and held near the bar high, is a liquidity pop
away from the mean. Taken only while the symbol's daily trend is down, the trade
is the reversion of that pop — not a continuation of it.

Exits are a hard stop and a time stop, nothing else. The measured short-signed
excess for this setup keeps building out to roughly ``max_hold_bars`` bars, so a
mean-reversion take-profit closes in one or two bars and hands the drift back to
commission; a chandelier trail was tested and is worse still (it fires on the
entry bar and cuts every winner).

The regime gate is the **adaptive-entropy** quantised trend on the daily feed:
enter only while that trend is down (the daily close sits below the
entropy-widened lower band, so the trigger width adapts to how structured the
tape is, and the previous side is held while price is inside the bands). A static
daily EMA on the same feed was the alternative and is worse on the frame that
matters for a short-only book — the book is paid in bear tape and must not bleed
in bull tape. ``entropy_max`` adds an independent STRUCTURE filter (low
normalized entropy = directional tape) on top of that direction test; its
measured surface is not monotone, so the default is a selected cell rather than a
fitted ridge, and it must be re-measured whenever the universe or window changes.
The VIX rank, ADX and VWATR gates are *bull-tape* filters: each was measured by
ablation and each, when removed, re-opens a bull-month bleed.

History worth keeping: the predecessor faded a Value Area breakdown behind a
``trend >= -1`` entropy gate — a no-op, since ``trend`` only ever takes -1/0/1.
Breakdown chasing has negative path asymmetry (more adverse than favourable
excursion at every horizon, so it pays to be stopped out); a stretched pop has
the opposite shape. The value-area entry added nothing once stretch was present,
so it is dropped rather than kept as dead weight.

``size_mult_k`` scales the risk-sized qty by signal strength and defaults to 0
(off). The measured direction is negative — the widest pops are the trades that
pay least — so a positive value fits against the evidence. The multiplier is
linear in stretch above ``min_stretch_atr`` (causal: no in-run quantile or
median) and clipped to a structural floor and ``size_mult_max``, which is 1.0 by
default so the knob can never lever past the risk-sized quantity.

Lookahead-free: every indicator read is cursor-truncated and the daily regime
read exposes completed daily buckets only (Pine ``lookahead=false``). State lives
in ``ctx.shared`` via ``@strategy(stateful=True)``, minted per run/window, so
there is no cross-window bleed and sweeps are split-safe across worker processes.
The entropy model is fed only unseen daily bars, and it REFUSES a non-finite one
(``NonFiniteBarError``) instead of skipping it: its EMA/ATR are running scalars,
so a junk bar would freeze ``trend``, and a gate that never fires looks exactly
like a gate that always passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict, cast

import numpy as np

from src.bt.size.pure import risk_sized_qty
from src.indicators.adaptive_entropy import (
    AdaptiveEntropyConfig,
    AdaptiveEntropyResult,
    OnlineAdaptiveEntropy,
)
from src.bt.strategies.dsl import strategy, StrategyContext
from src.bt.strategies.series import SeriesView
from src.bt.strategies.types import StrategyParams
from src.bt.strategies.utils import gross_exposure, gross_gate

STRATEGY_TYPE = "pop_fade_ae_short_dsl"

# Per-run state in ctx.shared (minted fresh per run/window).
_POS_KEY = "positions"  # dict[sym -> {"stop": float, "bars": int}]
_GROSS_KEY = "gross_pending"  # notional fraction already committed THIS bar
_ENT_KEY = "entropy"  # dict[sym -> _EntropyState] online adaptive-entropy state


class _EntropyState(TypedDict):
    """Per-symbol online adaptive-entropy bookkeeping in ``ctx.shared``."""

    model: OnlineAdaptiveEntropy
    fed: int
    result: AdaptiveEntropyResult | None


@dataclass(frozen=True)
class Params(StrategyParams):
    """Fitted knobs. Everything else in this module is a structural constant."""

    # -- the signal: an overextended pop above the mean --
    min_stretch_atr: float = 1.0  # pop must sit >= this many ATRs above the EMA
    vol_expand_mult: float = 1.3  # pop volume >= this x the 20-bar mean
    # -- regime gates (0 / False = off) --
    entropy_max: float = 0.85  # daily normalized entropy must be <= this;
    # low entropy = directional tape, high = chop. The direction test is the
    # entropy quantised trend itself (_regime_check), not a knob.
    vix_rank_min: float = 0.0  # VIX must rank >= this over its trailing window
    adx_max: float = 0.0  # 1h ADX must be <= this (no strong directional run)
    vwatr_ratio_min: float = 0.0  # VWATR / its baseline must be >= this
    require_earnings_decline: bool = False  # YoY net income must be falling
    # -- exit / risk --
    stop_atr_mult: float = 2.0  # hard stop, ATRs above the entry
    max_hold_bars: int = 48  # time stop, at the measured peak of the excess
    risk_pct: float = 0.01  # equity risked per trade (stop-distance sized)
    # -- sizing multiplier on signal strength (0 = off) --
    size_mult_k: float = 0.0  # size change per ATR of stretch above the
    # threshold; negative de-sizes the widest pops, which is the measured
    # direction. 0.0 disables the multiplier entirely.
    size_mult_max: float = 1.0  # cap on the multiplier (1.0 = never lever up)


# ---------------------------------------------------------------------------
# structural constants — fixed, not fitted. Sweeping these is a curve-fit knob.
# ---------------------------------------------------------------------------

_STRETCH_EMA = 20  # EMA the stretch read and the reversion target use
_VOL_PERIOD = 20  # volume baseline lookback
_CLOSE_STRENGTH = 0.5  # close must sit in the top half of the pop bar
ATR_PERIOD = 14  # ATR period for stretch, sizing and the stop
MAX_GROSS_EXPOSURE = 0.5  # aggregate |gross| cap vs live equity
_SIZE_MULT_FLOOR = 0.25  # a de-sized trade stays a trade
_VIX_RANK_WINDOW = 250  # VIX prints behind the regime rank
_VIX_MIN_PRINTS = 60  # refuse to gate on a thin VIX history
_ADX_PERIOD = 14  # 1h directional-strength read
_VWATR_PERIOD = 14  # VWATR period for the expansion read
_VWATR_BASE_WINDOW = 60  # VWATR baseline window
_YOY_MIN_DAYS = 330  # fiscal-period match window for the earnings YoY read
_YOY_MAX_DAYS = 400


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
        return float("nan")  # no range -> no close position: fail the gate closed
    return (close - low) / rng


def _avg_volume(ctx: StrategyContext, sym: str) -> float:
    """Mean volume over the prior ``_VOL_PERIOD`` bars (excl. the current one)."""
    vol = ctx.ohlcv(sym).volume.to_array()
    if len(vol) < _VOL_PERIOD + 1:
        return float("nan")
    return float(np.mean(vol[-(_VOL_PERIOD + 1) : -1]))


def _entropy_state(ctx: StrategyContext, params: Params, sym: str) -> _EntropyState:
    """Per-symbol online adaptive-entropy model, fed DAILY bars exactly once.

    The online model is stateful, so it lives in ``ctx.shared`` (minted per
    run/window) and consumes the ``1d`` feed — the regime clock. ``fed`` makes
    the feed gap-safe: a symbol that skips a timestamp is never fed a duplicate
    or stale bar. Indicator hyper-parameters are the library defaults
    (structural, not fitted).

    A non-finite daily bar raises out of the model (``NonFiniteBarError``);
    there is deliberately no skip here, because skipping is what turns a
    poisoned running EMA/ATR into a silently frozen ``trend``.
    """
    store: dict[str, _EntropyState] = ctx.shared.setdefault(_ENT_KEY, {})
    st: _EntropyState | None = store.get(sym)
    if st is None:
        st = {
            "model": OnlineAdaptiveEntropy(AdaptiveEntropyConfig()),
            "fed": 0,
            "result": None,
        }
        store[sym] = st
    o = ctx.ohlcv(sym, "1d")
    n_total = len(o.close)
    fed = st["fed"]
    if n_total > fed:
        close_arr = o.close.to_array()
        high_arr = o.high.to_array()
        low_arr = o.low.to_array()
        for i in range(fed, n_total):
            st["result"] = st["model"].observe(close_arr[i], high_arr[i], low_arr[i])
        st["fed"] = n_total
    return st


def _regime_check(
    ctx: StrategyContext, params: Params, sym: str
) -> AdaptiveEntropyResult | None:
    """Daily adaptive-entropy admission for the short leg, or ``None``.

    Two reads on the same ``1d`` feed, both required:

    * DIRECTION — ``trend == -1``: the daily close sits below the
      entropy-widened lower band. The band width scales with measured trend
      strength, so the trigger adapts to how structured the tape is instead of
      using one fixed EMA distance, and it holds its previous side while price
      is inside the bands.
    * STRUCTURE — ``normalized_entropy <= entropy_max`` when that knob is on;
      low entropy is a directional tape, high entropy is chop.

    Returning the snapshot rather than a bool lets the caller put the measured
    band position in the entry reason without a second model read.

    Fails closed: an unwarm model is not permission to short, which is exactly
    the failure the predecessor's ``trend >= -1`` gate hid.
    """
    st = _entropy_state(ctx, params, sym)
    result = st["result"]
    if result is None or not st["model"].ready:
        return None
    if params.entropy_max > 0 and result.entropy > params.entropy_max:
        return None
    if result.trend != -1:
        return None
    return result


def _band_depth(result: AdaptiveEntropyResult) -> float:
    """How far the daily close sits below the bearish trigger band, in ATRs.

    Zero at the trigger and rising as the close pushes toward (and through) the
    outer envelope, so it is comparable across symbols and regimes. Only
    meaningful once ``trend == -1``; the caller gates on that first.
    """
    if result.atr <= 0:
        return 0.0
    return max((result.inner_lower - result.close) / result.atr, 0.0)


def _vix_regime_ok(ctx: StrategyContext, params: Params) -> bool:
    """Macro volatility gate: only short while VIX ranks high in its own history.

    A short book is paid for in risk-off tape; in a calm grind up, fading pops
    is shorting a bid. ``rank`` compares the latest visible print against the
    trailing window, so it is level-free across decades. A thin or absent VIX
    history does not block (the gate narrows an existing edge).
    """
    if params.vix_rank_min <= 0:
        return True
    vix = ctx.macro.vix
    if len(vix) < _VIX_MIN_PRINTS:
        return True
    try:
        return bool(vix.rank(_VIX_RANK_WINDOW) >= params.vix_rank_min)
    except IndexError, ValueError:
        return True


def _trend_gates_ok(ctx: StrategyContext, params: Params, sym: str) -> bool:
    """Technical gates: not in a strong directional run, and vol expanding.

    ADX caps the fade to non-trending tape; the VWATR expansion read requires
    the pop to be a genuine volatility event rather than drift.
    """
    if params.adx_max > 0:
        adx = _last(ctx.ta.adx(sym, _ADX_PERIOD))
        if np.isfinite(adx) and adx > params.adx_max:
            return False
    if params.vwatr_ratio_min > 0:
        vw = _last(ctx.ta.vwatr(sym, _VWATR_PERIOD))
        base = _last(ctx.ta.vwatr_baseline(sym, _VWATR_PERIOD, _VWATR_BASE_WINDOW))
        if not (np.isfinite(vw) and np.isfinite(base) and base > 0):
            return False
        if vw / base < params.vwatr_ratio_min:
            return False
    return True


def _earnings_declining(ctx: StrategyContext, sym: str) -> bool | None:
    """``True``/``False`` on YoY net income, ``None`` when no clean pair exists.

    Matches the newest visible fiscal period against the one ending ~a year
    earlier (span-length based, so quarterly and annual filings both resolve).
    ``None`` means unreadable, and the gate then passes rather than blocking.
    """
    series = ctx.fundamentals.income(sym).net_income
    values = series.values()
    spans = series.spans()
    if len(values) < 2:
        return None
    latest = values[-1]
    if latest is None:
        return None
    latest_end = spans[-1][1]
    for (_, end), value in zip(reversed(spans[:-1]), reversed(values[:-1])):
        if not (_YOY_MIN_DAYS <= (latest_end - end).days <= _YOY_MAX_DAYS):
            continue
        if value is None:
            return None
        return float(latest) < float(value)
    return None


def _fundamental_gate_ok(ctx: StrategyContext, params: Params, sym: str) -> bool:
    """Fundamental gate: short only names whose earnings are actually falling."""
    if not params.require_earnings_decline:
        return True
    try:
        declining = _earnings_declining(ctx, sym)
    except KeyError:
        return True
    return True if declining is None else declining


def _size(
    ctx: StrategyContext, params: Params, price: float, stop_dist: float
) -> float:
    """0..1 initial-capital fraction to deploy, sized on live equity at risk."""
    if stop_dist <= 0 or price <= 0:
        return 0.0
    equity = ctx.current_equity()
    if equity <= 0:
        return 0.0
    qty = risk_sized_qty(
        equity=equity, price=price, stop_dist=stop_dist, risk_pct=params.risk_pct
    )
    if qty <= 0:
        return 0.0
    size = qty * price / ctx.state.portfolio.initial_capital
    return min(max(size, 0.0), 1.0)


def _size_mult(params: Params, stretch: float) -> float:
    """Scale the risk-sized qty by measured signal strength (1.0 = unchanged).

    Linear in how far the pop sits above ``min_stretch_atr``, so the read is
    causal — nothing here looks at the run's own trade distribution, which
    would be a lookahead the moment the config was swept.

    Clipped to ``[_SIZE_MULT_FLOOR, size_mult_max]``: the floor keeps a
    de-sized trade a trade, and the cap is 1.0 by default so enabling the knob
    can never lever past the risk-sized quantity. Scaling the size scales the
    risk, because ``risk_sized_qty`` sets ``qty * stop_dist = risk_pct *
    equity``.
    """
    if params.size_mult_k == 0.0:
        return 1.0
    raw = 1.0 + params.size_mult_k * (stretch - params.min_stretch_atr)
    return min(max(raw, _SIZE_MULT_FLOOR), params.size_mult_max)


# ---------------------------------------------------------------------------
# DSL strategy
# ---------------------------------------------------------------------------


@strategy(stateful=True)
def on_candle(ctx: StrategyContext):
    params: Params = ctx.params
    shared = ctx.shared
    shared.setdefault(_POS_KEY, {})
    # Same-bar cohort accounting: the gross cap must see the entries already
    # emitted earlier in this timestamp, or a wide universe over-subscribes cash
    # (every candidate is checked against a book that has not been debited yet).
    shared[_GROSS_KEY] = 0.0

    # Warmup bars fill strategy state only — no fill can happen there, so the
    # DSL raises on any entry emitted during them.
    if ctx.phase == "warmup":
        return

    # ---- exits first -------------------------------------------------------
    for sym in ctx.symbols:
        if ctx.position(sym) is not None:
            _manage_open(ctx, shared, params, sym)

    # ---- entries -----------------------------------------------------------
    for sym in ctx.symbols:
        if ctx.position(sym) is not None:
            continue
        _enter(ctx, shared, params, sym)


# ── entry: fade an overextended pop, only in a daily downtrend ───────────────


def _enter(ctx: StrategyContext, shared: dict, params: Params, sym: str) -> None:
    """Open a short when the pop is overextended and the daily regime agrees."""
    o = ctx.ohlcv(sym)
    if len(o.close) < max(_STRETCH_EMA, _VOL_PERIOD) + 1:
        return
    close = float(o.close[-1])
    atr = _last(ctx.ta.atr(sym, ATR_PERIOD))
    if close <= 0 or not np.isfinite(atr) or atr <= 0:
        return
    ema = _last(ctx.ta.ema(sym, _STRETCH_EMA))
    if not np.isfinite(ema):
        return

    # The signal: price has popped off its own mean by more than the threshold.
    stretch = (close - ema) / atr
    if stretch < params.min_stretch_atr:
        return

    # The pop must be a real push, not a wick, and must carry order flow.
    strength = _bar_position_close(close, float(o.high[-1]), float(o.low[-1]))
    if not (strength >= _CLOSE_STRENGTH):
        return
    avg_vol = _avg_volume(ctx, sym)
    if not np.isfinite(avg_vol) or avg_vol <= 0:
        return
    if float(o.volume[-1]) < params.vol_expand_mult * avg_vol:
        return

    # The regime gates: fading a pop in a daily uptrend is shorting a melt-up,
    # and doing it in calm tape is shorting a bid.
    regime = _regime_check(ctx, params, sym)
    if regime is None:
        return
    if not _vix_regime_ok(ctx, params):
        return
    if not _trend_gates_ok(ctx, params, sym):
        return
    if not _fundamental_gate_ok(ctx, params, sym):
        return

    stop_dist = params.stop_atr_mult * atr
    size = _size(ctx, params, close, stop_dist)
    mult = _size_mult(params, stretch)
    size *= mult
    if size <= 0:
        return

    # Aggregate gross-exposure cap. ``_size`` returns the candidate as a
    # fraction of *initial capital*, so rescale onto the live-equity base first.
    equity = ctx.current_equity()
    init = ctx.state.portfolio.initial_capital
    candidate = size * init / equity if equity > 0 else 0.0
    pending = float(shared.get(_GROSS_KEY, 0.0))
    if (
        gross_gate(equity, ctx.state.portfolio.positions, MAX_GROSS_EXPOSURE, candidate)
        or gross_exposure(equity, ctx.state.portfolio.positions) + pending + candidate
        >= MAX_GROSS_EXPOSURE
    ):
        return
    shared[_GROSS_KEY] = pending + candidate

    stop_price = close + stop_dist
    ctx.short(
        sym,
        size=size,
        reason=(
            f"[pop-fade] close {close:.2f} is {stretch:.2f} ATR above "
            f"EMA{_STRETCH_EMA} {ema:.2f}, daily entropy trend down "
            f"(norm_entropy {regime.entropy:.2f}, band {_band_depth(regime):.2f} ATR), "
            f"size x{mult:.2f}"
        ),
    )
    shared[_POS_KEY][sym] = {"stop": stop_price, "bars": 0}


# ── position management: mean revert, hard stop, time stop ───────────────────


def _manage_open(ctx: StrategyContext, shared: dict, params: Params, sym: str) -> None:
    """Manage an open short: the hard stop, then the time stop.

    There is deliberately no mean-reversion take-profit and no trailing exit.
    The measured short-signed excess for this setup peaks at roughly
    ``max_hold_bars`` bars and is still building before then; an exit at the
    short EMA closes in one or two bars and surrenders that drift to
    commission. A chandelier trail was tested and is worse on both counts (it
    fires almost on the entry bar and cuts every winner short).

    (1) the hard stop TRIGGERS on a close at or above the level (not intrabar —
    an intrabar breach that recovers does not exit), and is emitted with
    ``guard_price`` so the fill is the worse of the stop and the next open
    rather than a naive next-open fill;
    (2) a reversion that has not happened within ``max_hold_bars`` bars is wrong
    and is closed at the bar.

    The stop is anchored on the signal bar's close, while the fill lands at the
    next open; the realised risk therefore differs from ``stop_atr_mult`` by the
    overnight gap (materially so for a signal on the session's last bar).
    """
    rec = shared[_POS_KEY].get(sym)
    o = ctx.ohlcv(sym)
    close = float(o.close[-1])
    atr = _last(ctx.ta.atr(sym, ATR_PERIOD))

    if rec is None:
        # A lot the strategy did not open (e.g. a live-seeded book): adopt it by
        # seeding the stop from the current ATR rather than trading it naked.
        if not np.isfinite(atr) or atr <= 0:
            return
        rec = {"stop": close + params.stop_atr_mult * atr, "bars": 0}
        shared[_POS_KEY][sym] = rec

    rec["bars"] = int(rec["bars"]) + 1

    # (2) Hard stop, fired intrabar at the stop level (worse-of next open).
    stop = float(rec["stop"])
    if close >= stop:
        _exit(ctx, shared, sym, "hard stop hit", guard_price=stop)
        return

    # (3) Time stop — a reversion this slow is not the trade that was taken.
    if int(rec["bars"]) >= params.max_hold_bars:
        _exit(ctx, shared, sym, "time stop")


def _exit(
    ctx: StrategyContext,
    shared: dict,
    sym: str,
    reason: str,
    guard_price: float | None = None,
) -> None:
    shared[_POS_KEY].pop(sym, None)
    ctx.close(sym, reason=f"[pop-fade] {reason}", guard_price=guard_price)
