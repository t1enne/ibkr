"""Adaptive-entropy-gated VP breakdown — dynamic ATR trail (DSL).

SHORT-only order-flow breakdown, built on the Volume Profile
(``vp_breakout_dsl``'s short leg) and gated everywhere by the
**adaptive-entropy regime gate**.

Two layers of AE gating keep the book from bleeding when the regime changes:

  * **Per-symbol entry gate** — a short is entered only when the symbol's
    quantised ``trend`` is at or above ``min_trend`` (``-1`` = bear-only,
    ``0`` = flat/bear).
  * **Market regime filter** (``market_filter``) — a dedicated feed-only AE
    tracker on ``market_symbol`` (the broad market) acts as a macro risk
    switch, and every market-regime call is **double-confirmed** by two
    independent layers: (1) the market AE ``trend`` (
    ``+1`` bullish), and (2) the market close vs its slow
    ``market_confirm_window`` SMA. The AE layer catches short-term momentum;
    the SMA layer is a slow, non-flipping trend filter, so a single noisy AE
    bar can't mislabel the regime. When both layers agree the market is a
    **confirmed bull**, the strategy BLOCKS new shorts AND FORCE-CLOSES open
    shorts. Optionally (``require_market_bear``) new shorts are additionally
    gated on both layers agreeing the market is a **confirmed bear** (AE
    ``-1`` and close below SMA), so the short edge is only harvested in a
    robust down regime.

Entry signal — a **VP breakdown** (mirrors ``vp_breakout_dsl._enter``'s short
leg, reused here for the same order-flow thesis but gated by entropy instead of
the Kaufman efficiency ratio):

  * ``close < VAL`` — the bar closes below the rolling Value Area Low (close,
    not a wick; ``strength`` near the bar's low),
  * **expanding volume** — the bar's volume ``>= vol_expand_mult *`` the prior
    ``vol_period`` average (order-flow confirmation out of the fair-value
    range),
  * **not in an uptrend** — price on the wrong side of its own rolling mean
    (concurrency filter).

The Value Area comes from an :class:`src.indicators.volume_profile.strategy.
OnlineVP` per symbol in ``ctx.shared``, fed once per candle on the *prior* bars
(no same-bar lookahead — ``OnlineVP.observe`` excludes the current bar).

Exits use a **dynamic ATR stop**: a chandelier-style trailing stop re-anchored
every bar on ``highest(high, trail_lookback) - trail_atr_mult * ATR`` and only
ever moved in our favour (down for a short); it widens with volatility and
tightens against price as the trade works. A revert back **into the value area**
(close > VAL) fires an immediate exit regardless (the institutional tag stop).

All cross-candle state (per-symbol ``OnlineAdaptiveEntropy`` + ``OnlineVP``
instances, cooldowns, per-bar snapshots, dynamic-trail bookkeeping) lives in
``ctx.shared`` via ``@strategy(stateful=True)`` — a fresh holder per run/window,
so there is no cross-window bleed and the strategy is safe across concurrent
split/sweep/optimize workers. Pure helpers take cursor-truncated arrays /
SeriesViews so there is no lookahead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from src.bt.state.types import Position
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
_COOLDOWN_KEY = "cooldowns"
_POS_KEY = "positions"  # dict[sym -> dynamic-trail bookkeeping]
_MARKET_KEY = "market_trend"  # int: market AE quantised trend (-1/0/1), or None
_MARKET_TRACKER_KEY = "market_tracker"  # OnlineAdaptiveEntropy on the market symbol
_MARKET_BULL_RUN_KEY = "market_bull_run"  # int: consecutive market-bull bars
_MARKET_SMA_KEY = "market_above_sma"  # bool: market close > slow SMA (layer-2 confirm)


@dataclass(frozen=True)
class Params(StrategyParams):
    # -- online volume profile (short leg of vp_breakout_dsl) --
    vp_window: int = 100
    num_bins: int = 50
    value_area_pct: float = 0.70
    vp_warmup: int = 60
    # -- VP breakdown confirmation --
    vol_period: int = 20
    vol_expand_mult: float = 1.3  # breakdown volume >= this x avg
    wick_check: float = 0.5  # close must sit in the lower fraction of the bar
    trend_lookback: int = 50  # concurrency: price vs its own rolling mean
    # -- adaptive-entropy regime gate --
    # Per-symbol entry gate: a short is allowed when the symbol's quantised
    # ``trend >= min_trend`` (``-1`` = bear-only ``trend == -1``, ``0`` =
    # non-bull flat/bear). The *market*-level regime filter below is the
    # primary loss-prevention switch.
    entropy_lookback: int = 25
    entropy_num_bins: int = 10
    min_trend: int = 0
    # -- market regime filter (risk-off when the market turns bullish) --
    # When ``market_filter`` is enabled, a dedicated adaptive-entropy tracker on
    # ``market_symbol`` (feed-only, never traded) acts as a macro risk switch.
    # The bull risk-off condition is **confirmed by two independent layers** so a
    # single noisy AE bar can't whip it off:
    #   * layer 1 — the market AE ``trend == +1`` for ``market_confirm_bars``
    #     consecutive bars, AND
    #   * layer 2 — the market close sits above its ``market_confirm_window``-
    #     bar SMA (a slow, long-duration trend filter that doesn't flip intraday).
    # Both must agree the market is in a sustained up regime before the strategy
    # blocks NEW shorts AND force-closes OPEN shorts. Symmetrically, shorts are
    # only entered when the market is NOT bull-confirmed (a ``trend``-level
    # bear/neutral read, per ``min_trend`` on the symbol).
    market_filter: bool = True
    market_symbol: str = "SPY"
    market_confirm_bars: int = 3  # layer-1: consecutive market-bull AE bars
    market_confirm_window: int = 100  # layer-2: market close > SMA(period)
    # When ``require_market_bear`` is True, NEW shorts additionally require the
    # market to be in a *confirmed bear* — the two AE/SMA confirmation layers
    # agreeing on the down side (market AE ``trend == -1`` AND close below its
    # slow SMA). This is the mirror of the bull risk-off gate: it only lets the
    # book carry shorts while the broad market is robustly bearish, so the short
    # edge is harvested where it exists instead of fired into a rising tape.
    require_market_bear: bool = False
    # -- dynamic ATR trailing stop --
    atr_period: int = 14
    stop_atr_mult: float = 2.0
    trail_lookback: int = 10
    trail_atr_mult: float = 2.0
    # -- risk / sizing (ATR-risk, 0..1 size) --
    risk_pct: float = 0.01
    # Aggregate |gross notional| cap, fraction of initial capital. Values >= 1.0
    # disable the cap (the ``_enter`` gate requires < 1.0 to fire).
    max_gross_exposure: float = 0.5
    cooldown_bars: int = 5


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
    if n < params.vol_period + 1:
        return float("nan")
    return float(np.mean(vol[-params.vol_period : -1]))


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
    snapshot is cached for ``bear_gate_ok`` to read synchronously.
    """
    trackers = shared.setdefault(_ENT_KEY, {})
    tr = trackers.get(sym)
    if tr is None:
        cfg = AdaptiveEntropyConfig(
            lookback=ctx.params.entropy_lookback,
            num_bins=ctx.params.entropy_num_bins,
        )
        tr = OnlineAdaptiveEntropy(cfg)
        trackers[sym] = tr
    result = tr.observe(float(o.close[-1]), float(o.high[-1]), float(o.low[-1]))
    shared.setdefault(_ENT_RESULT_KEY, {})[sym] = result


def entropy_result(ctx: StrategyContext, shared: dict, sym: str) -> object:
    """The cached adaptive-entropy snapshot for ``sym`` after feeding this bar."""
    return shared.get(_ENT_RESULT_KEY, {}).get(sym)


def bear_gate_ok(result, min_trend: int) -> bool:
    """Adaptive-entropy regime gate: allow the short above a trend floor.

    A short is permitted when the quantised trend is at or above ``min_trend``
    (e.g. ``min_trend=-1`` = bear-only, ``min_trend=0`` = non-bull). A value of
    ``1`` blocks both flat and bear (effectively off). During warmup the running
    ``trend == 0`` default passes any ``min_trend <= 0``.
    """
    return bool(getattr(result, "trend", 0) >= min_trend)


def _feed_market(ctx: StrategyContext, shared: dict, params: Params) -> None:
    """Feed a dedicated market-level adaptive-entropy tracker from a benchmark.

    The tracker runs on ``params.market_symbol`` (feed-only — the symbol is
    never traded by this strategy) so its quantised ``trend`` is a macro regime
    gauge independent of any single short candidate. The latest ``trend`` is
    cached under ``_MARKET_KEY`` for both the entry gate and the open-position
    risk-off close to read synchronously.

    The market symbol must be present in the config's ``symbols`` (so its OHLCV
    is in the TaContext feed); it is skipped for trading in the strategy loop.
    """
    if ctx.params.market_filter is False:
        shared.pop(_MARKET_KEY, None)
        return
    try:
        o = ctx.ohlcv(params.market_symbol)
    except KeyError:
        # Market symbol absent from the feed -> no market gate (fail-open rather
        # than silently disabling the whole strategy).
        shared.pop(_MARKET_KEY, None)
        return
    if len(o.close) == 0:
        shared.pop(_MARKET_KEY, None)
        return
    tr = shared.get(_MARKET_TRACKER_KEY)
    if not isinstance(tr, OnlineAdaptiveEntropy):
        tr = OnlineAdaptiveEntropy(
            AdaptiveEntropyConfig(
                lookback=params.entropy_lookback,
                num_bins=params.entropy_num_bins,
            )
        )
        shared[_MARKET_TRACKER_KEY] = tr
    result = tr.observe(float(o.close[-1]), float(o.high[-1]), float(o.low[-1]))
    trend = getattr(result, "trend", 0)
    shared[_MARKET_KEY] = trend
    # Maintain a consecutive-bull-bar counter so risk-off requires a *sustained*
    # up regime, not a single bullish AE bar (which is ~half of every year).
    run = int(shared.get(_MARKET_BULL_RUN_KEY, 0))
    shared[_MARKET_BULL_RUN_KEY] = run + 1 if trend == 1 else 0

    # Layer-2 confirmation: is the market close above its slow SMA? This is a
    # long-duration trend filter that does NOT flip intraday, so it anchors the
    # AE layer-1 signal to the true macro direction.
    above = False
    if len(o.close) >= params.market_confirm_window:
        sma = ctx.ta.sma(params.market_symbol, params.market_confirm_window)
        above = float(o.close[-1]) > float(sma[-1])
    shared[_MARKET_SMA_KEY] = above


def market_risk_on(shared: dict, params: Params) -> bool:
    """True when shorts may be carried / opened by the market regime filter.

    Risk-off (``False``) only fires when **both** confirmation layers agree the
    market is in a sustained up regime: (1) the market AE ``trend == +1`` for at
    least ``market_confirm_bars`` consecutive bars, and (2) the market close is
    above its ``market_confirm_window`` SMA. Requiring layer-2 avoids trusting a
    noisy short-term AE trend alone (which is ~50/50 every year); the slow SMA
    disambiguates a real bull market from a bear-year bounce. When the filter is
    disabled, the market symbol is absent, or the layers don't agree on a
    sustained bull, this returns ``True`` and the strategy trades as normal.
    """
    if not params.market_filter:
        return True
    run = int(shared.get(_MARKET_BULL_RUN_KEY, 0))
    if run < params.market_confirm_bars:
        return True  # layer-1 not sustained -> not risk-off
    if not bool(shared.get(_MARKET_SMA_KEY, False)):
        return True  # layer-2 not confirmed (below slow SMA) -> not risk-off
    return False


def market_bear_ok(shared: dict, params: Params) -> bool:
    """True when the market is in a *confirmed bear* regime (both layers agree).

    Layer-1: the market AE quantised trend is bearish (``trend == -1``).
    Layer-2: the market close is below its slow ``market_confirm_window`` SMA.
    Both must agree before a NEW short is permitted (when
    ``require_market_bear`` is enabled). If the filter is off, the market symbol
    is absent, or the layers don't both read bear, this returns ``True`` so the
    market gate never blocks by default.
    """
    if not params.market_filter or not params.require_market_bear:
        return True
    if int(shared.get(_MARKET_KEY, 0)) != -1:
        return False  # layer-1: market not confirmed-bear
    # Layer-2: ``_MARKET_SMA_KEY`` stores *above* SMA; below-SMA (False) is the
    # bear-side confirmation.
    return not bool(shared.get(_MARKET_SMA_KEY, False))


def _refresh_vp(ctx: StrategyContext, params: Params) -> None:
    """Feed each symbol's online VP once per candle; store the snapshots."""
    profs = ctx.shared.setdefault(_VP_KEY, {})
    snaps: dict[str, object] = {}
    for sym in ctx.symbols:
        prof = profs.get(sym)
        if not isinstance(prof, OnlineVP):
            prof = OnlineVP(
                num_bins=params.num_bins,
                value_area_pct=params.value_area_pct,
                window=params.vp_window,
                warmup_bars=params.vp_warmup,
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
        stop_dist=params.stop_atr_mult * atr_val,
        risk_pct=params.risk_pct,
    )
    if qty <= 0:
        return 0.0
    size = qty * price / ctx.state.portfolio.initial_capital
    return min(max(size, 0.0), 1.0)


# ---------------------------------------------------------------------------
# DSL strategy
# ---------------------------------------------------------------------------


@strategy(bars="1d", stateful=True)
def on_candle(ctx: StrategyContext):
    params: Params = ctx.params
    shared = ctx.shared
    shared.setdefault(_COOLDOWN_KEY, {})
    shared.setdefault(_POS_KEY, {})

    _refresh_vp(ctx, params)

    # Market regime gauge (feed-only; never traded).
    _feed_market(ctx, shared, params)
    risk_off = not market_risk_on(shared, params)

    # ---- Advances / exits first --------------------------------------------
    for sym in ctx.symbols:
        if sym == params.market_symbol:
            continue  # market gauge symbol is never traded
        o = ctx.ohlcv(sym)
        _feed_entropy(ctx, shared, sym, o)  # regime tracker every bar

        pos = ctx.position(sym)
        if pos is None:
            continue
        # Risk-off flatten: if the broad market turned into a confirmed bull,
        # force-close the short now before it bleeds into the rising tape.
        if risk_off:
            _exit(ctx, shared, params, sym, "market regime flipped bull (risk-off)")
            continue
        _manage_open(ctx, shared, params, sym, o, pos)

    # ---- Entries ------------------------------------------------------------
    for sym in ctx.symbols:
        if sym == params.market_symbol:
            continue  # never open a position on the market gauge symbol
        cd = shared[_COOLDOWN_KEY].get(sym, 0)
        if cd > 0:
            shared[_COOLDOWN_KEY][sym] = cd - 1
            continue
        if ctx.position(sym) is not None:
            continue
        if risk_off:
            continue  # no new shorts while the market is confirmed-bull
        _enter(ctx, sym, params)


# ── entry (VP breakdown gated by bear regime) ────────────────────────────────


def _enter(ctx: StrategyContext, sym: str, params: Params) -> None:
    snap = _snap(ctx, sym)
    if snap is None or not getattr(snap, "ready", False):
        return
    o = ctx.ohlcv(sym)
    n = len(o.close)
    if n < max(params.vp_warmup, params.trend_lookback + 1, params.vol_period + 1):
        return

    close = float(o.close[-1])
    high = float(o.high[-1])
    low = float(o.low[-1])
    val = getattr(snap, "val", None)
    if not isinstance(val, float):
        return

    atr_val = _last(ctx.ta.atr(sym, params.atr_period))
    if close <= 0 or np.isnan(atr_val) or atr_val <= 0:
        return

    # Adaptive-entropy regime gate: never fade a confirmed bull (or tighter).
    ent = entropy_result(ctx, ctx.shared, sym)
    if not bear_gate_ok(ent, params.min_trend):
        return

    # Market-level confirmed-bear gate: when enabled, shorts require BOTH AE
    # (market trend == -1) and slow-SMA (close below SMA) to agree the broad
    # market is bearish — only harvest the short edge in a real down regime.
    if not market_bear_ok(ctx.shared, params):
        return

    avg_vol = _avg_volume(ctx, sym, params)
    if np.isnan(avg_vol) or avg_vol <= 0:
        return
    vol_expanding = float(o.volume[-1]) >= params.vol_expand_mult * avg_vol
    strength = _bar_position_close(close, high, low)

    short_breakdown = (
        close < val  # close (not a wick) breaks the value area top
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
    if gross_gate(
        equity, ctx.state.portfolio.positions, params.max_gross_exposure, candidate
    ):
        return

    # Initial stop for sizing is one ATR distance above the entry; the dynamic
    # trailing stop is established from here and ratcheted down each bar.
    stop_price = close + params.stop_atr_mult * atr_val
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
    """Manage an open short: market risk-off + value-area revert + ATR trail.

    ``on_candle`` force-closes shorts when the market AE regime turns bull
    (risk-off) before this runs. Here: (1) value-area revert (close back above
    VAL) is the institutional tag stop, and (2) a dynamic ATR trail re-anchored
    every bar on the highest high of the last ``trail_lookback`` bars minus
    ``trail_atr_mult * ATR``, only ever moved *down* for a short.
    """
    rec = shared[_POS_KEY].get(sym)
    atr = _last(ctx.ta.atr(sym, params.atr_period))
    close = float(o.close[-1])

    if rec is not None and not rec.get("pid"):
        rec["pid"] = pos.position_id

    # (1) Value-area revert (institutional tag stop): close back inside the
    # fair-value range invalidates the breakdown.
    snap = _snap(ctx, sym)
    val = getattr(snap, "val", None) if snap is not None else None
    if isinstance(val, float) and close > val:
        _exit(ctx, shared, params, sym, "reverted into value area")
        return

    # (2) Dynamic ATR trailing stop, only ever ratcheted down.
    if rec is not None and np.isfinite(atr):
        new_trail = trail_stop(
            o.high.to_array(), atr, params.trail_lookback, params.trail_atr_mult
        )
        if np.isfinite(new_trail) and new_trail < rec["trail"]:
            rec["trail"] = new_trail
        if close >= rec["trail"]:
            _exit(ctx, shared, params, sym, "dynamic ATR trail hit")
            return


def _exit(
    ctx: StrategyContext,
    shared: dict,
    params: Params,
    sym: str,
    reason: str,
) -> None:
    shared[_POS_KEY].pop(sym, None)
    shared[_COOLDOWN_KEY][sym] = params.cooldown_bars
    ctx.close(sym, reason=f"[breakdown] {reason}")
