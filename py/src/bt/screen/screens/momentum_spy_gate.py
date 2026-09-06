"""Momentum compression-breakout screen — SPY adaptive-entropy gate.

Scoring-layer mirror of the ``momentum_compression_breakout_dsl`` strategy for
manual trading: each symbol is scored, at its latest bar, on whether it is
staging a "flywheel after a big run" breakout:

  1. **Big move (setup)** — the stock gained ~35–100% over the trailing 3
     months, measured against the recent high (so a post-run pullback doesn't
     invalidate the setup). Not parabolic: capped at ``max_gain``.
  2. **Pullback + compression** — price coils with a *very small candle body*
     (a small fraction of ATR) on *low volume* while the close hovers at/inside
     the [SMA10, SMA20] band for several bars. A compression box (highest high
     / lowest low over the window) is built for the trigger.
  3. **Breakout** — score rises only when the close breaks above the box high
     with the 10 SMA above the 20 SMA (uptrend intact).
  4. **Regime gate** — the whole book is gated by an adaptive-entropy bull
     regime on ``benchmark`` (an index like SPY): a technically-valid breakout
     in a wrong regime scores 0 so a manual trader isn't alerted to chase a
     breakdown/choppy tape.

Returns a 0..1 **signal strength** (NOT a fill instruction). The score blends
the quality of the big-move setup, how decisively price pierced the box, and
how tightly the coil compressed. Diagnostic ``model_features`` expose the
breakout depth, ATR, hover streak and big-move gain.

Because a screen scores a single snapshot per call (no cross-call state), the
strategy's persistent per-symbol compression box + cooldown machinery is
re-expressed here as a fresh ``comp_window`` coil detection at the current bar.

This module is the canonical momentum home: it absorbed the former ``momentum``
screen (removed with the no-alpha divergent screens) and pins the validated
backtest pass ``strats/pass/momentum_compression_breakout_ae_gate_SPY.json`` —
the adaptive-entropy observer gates on **SPY** rather than QQQ, with that pass's
tuned knobs as defaults (``body_atr_ratio 0.30``, ``min_hover_bars 4``,
``min_gain 0.35``, ``regime_min_strength 0.12``, ``ae_lookback 40``). ``Params``
fields can still be overridden via ``resolve_screen_params`` to dial any knob —
including swapping ``benchmark`` back to any other index for a book-level gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.bt.screen.metrics import with_common_metrics
from src.bt.screen.types import (
    ScreenParams,
    ScreenResult,
    ScreenState,
    TrendRegime,
    VolRegime,
)

SCREEN_TYPE = "momentum_spy_gate"

#: Fresh-breakout pierce in ATR units above the box high at which the extra
#: dispatch term saturates (1.0) — beyond it adds no additional score.
_PIERCE_SATURATE = 1.5
#: Ceiling for still-coiling (armed, not yet broken) setups, so a coil never
#: outranks an actual breakout but still surfaces above flat for watchlisting.
_COIL_CEILING = 0.65
#: Score multiplier applied when the AE market-regime gate is not bullish, so a
#: bearish-regime setup ranks visibly lower but is still surfaced (never hidden).
_REGIME_SUPPRESS = 0.5


@dataclass(frozen=True)
class Params(ScreenParams):
    """SPY-gated momentum compression-breakout knobs.

    Defaults match the validated backtest pass
    ``strats/pass/momentum_compression_breakout_ae_gate_SPY.json``: AE regime
    gate observes SPY, with that pass's tuned compression/regime thresholds.
    ``from_dict`` (via ``StrategyParams``) extracts declared fields and ignores
    backtest-only extras (sizing/cooldown knobs), so a pass dict loads cleanly.
    """

    # -- big-move setup ----------------------------------------------------
    big_lookback: int = 63  # ~3 months of trading days
    min_gain: float = 0.35  # must be up at least 35% over the lookback
    max_gain: float = 1.00  # ...but not parabolic (cap at 100%)
    # -- compression -------------------------------------------------------
    ma_fast: int = 10
    ma_slow: int = 20
    comp_window: int = 15  # look back for the compression box high/low
    body_atr_ratio: float = 0.30  # candle body must be <= this * ATR
    vol_period: int = 20
    vol_mult: float = 0.8  # compression volume <= this * avg volume
    min_hover_bars: int = 4  # consecutive bars close hovering in the MA band
    hover_tol: float = (
        0.002  # close may fall within this fraction of ATR of the band edge
    )
    decay_bars: int = 10  # an armed box stays live for this many bars
    # -- breakout + risk ---------------------------------------------------
    atr_period: int = 14
    warmup_bars: int = 80
    # -- long-trend filter -------------------------------------------------
    # Require the latest close to sit ABOVE its SMA to be surfaced at all — a
    # stock parked below its key moving average is mid-pullback/weak and is not
    # a compression-breakout candidate. ``0`` disables the filter.
    trend_ma: int = 50  # SMA window for the close-above-MA gate (0 = off)
    # -- adaptive-entropy market-regime gate ---------------------------------
    # The AE indicator computed on ``benchmark`` (an index like SPY, embedded
    # by the CLI when ``benchmark`` is set) acts as an entry gate: longs only
    # score while the index is in an AE-confirmed bull regime. The benchmark is
    # never itself scored for a breakout. Uppercased for frame lookup.
    benchmark: str = "SPY"  # AE regime observer index (validated-pass default)
    regime_trend_min: int = 1  # require AE trend >= this (1 = bullish) to enter
    regime_min_strength: float = 0.12  # min AE trend strength to enter
    ae_lookback: int = 40  # AE entropy lookback
    ae_num_bins: int = 10  # AE log-return histogram bins

    @property
    def regime_symbol(self) -> str:
        """Uppercase alias for the benchmark used as the AE regime observer."""
        return self.benchmark.upper()


@dataclass(frozen=True)
class _Setup:
    """Armed compression-box state at the latest bar, packaged for scoring.

    ``breaked`` distinguishes a fresh breakout (close cleared the box high with
    the uptrend intact) from a still-coiling setup waiting on a trigger.
    ``gain`` is the big-move fraction over ``big_lookback`` (verified to sit
    within [min_gain, max_gain] when the box latched); ``pierce`` the breakout
    depth in ATR units above the box high (0 for a coiling setup);
    ``body_frac`` the ``body_atr_ratio`` * ATR-normalised body of the pulse bar
    (lower = tighter) ``bars_ago`` the bars elapsed since the coil last
    compressed (freshness); ``atr`` is the coil-era ATR used to scale the
    setup.
    """

    gain: float
    pierce: float  # (close - box_high) / ATR (0 when still coiling)
    body_frac: float  # pulse |close-open| / (body_atr_ratio * ATR)
    hover_streak: int
    bars_ago: int  # bars since the box last compressed
    box_low: float  # box lower bound (highest/high box floor)
    box_high: float
    atr: float
    breaked: bool  # True: close cleared the box high (actionable breakout)


# ---------------------------------------------------------------------------
# pure helpers (vectorized over numpy arrays / pandas Series; no module state)
# ---------------------------------------------------------------------------


def _ta_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int
) -> np.ndarray:
    """Lazy entry to ``ta.atr`` (Wilder) — avoids pulling ``src.indicators``
    package init at ``src.bt`` bootstrap/collection time."""
    from src.indicators.ta import atr

    return atr(high, low, close, window).to_numpy(dtype=float)


def _ta_sma(closes: pd.Series, window: int) -> np.ndarray:
    """Lazy entry to ``ta.sma``."""
    from src.indicators.ta import sma

    return sma(closes, window).to_numpy(dtype=float)


def _big_move_gain(closes: np.ndarray, params: Params) -> float | None:
    """Gain fraction over the trailing ``big_lookback`` bars.

    ``None`` when the series is too short, prices aren't strictly positive, or
    the range falls outside [min_gain, max_gain] (setup not present). Mirrors
    the strategy's peak-over-trough measure so a coil inside the window keeps
    the measured range intact.
    """
    n = len(closes)
    if n < params.big_lookback:
        return None
    window = closes[-params.big_lookback :]
    low = float(np.min(window))
    if low <= 0:
        return None
    high = float(np.max(window))
    gain = high / low - 1.0
    if not (params.min_gain <= gain <= params.max_gain):
        return None
    return gain


def _avg_volume(volumes: np.ndarray, params: Params) -> float:
    """Mean volume over the prior ``vol_period`` bars (excl. current bar)."""
    n = len(volumes)
    if n < params.vol_period + 1:
        return float("nan")
    return float(np.mean(volumes[-params.vol_period : -1]))


def _hovering_between_mas(
    close: float, sma_fast: float, sma_slow: float, atr_val: float, tol: float
) -> bool:
    """Close sits at/inside the [SMA_fast, SMA_slow] band (ATR-tolerant)."""
    if sma_fast != sma_fast or sma_slow != sma_slow or not atr_val or np.isnan(atr_val):
        return False
    lo, hi = min(sma_fast, sma_slow), max(sma_fast, sma_slow)
    return (lo - tol * atr_val) <= close <= (hi + tol * atr_val)


def _hovering_streak(
    closes: np.ndarray,
    sma_fast: np.ndarray,
    sma_slow: np.ndarray,
    atr_series: np.ndarray,
    params: Params,
) -> int:
    """Trailing bars whose close hovers in the fast/slow MA band."""
    n = len(closes)
    if n < params.ma_slow + params.min_hover_bars:
        return 0
    streak = 0
    for i in range(params.min_hover_bars + 8):
        if i >= n:
            break
        atr_back = float(atr_series[-(i + 1)])
        if not atr_back or np.isnan(atr_back):
            break
        if not _hovering_between_mas(
            float(closes[-(i + 1)]),
            float(sma_fast[-(i + 1)]),
            float(sma_slow[-(i + 1)]),
            atr_back,
            params.hover_tol,
        ):
            break
        streak += 1
    return streak


def _coil_conditions_at(
    closes: np.ndarray,
    opens: np.ndarray,
    volumes: np.ndarray,
    sma_f: np.ndarray,
    sma_s: np.ndarray,
    atr_series: np.ndarray,
    i: int,
    params: Params,
) -> tuple[bool, int]:
    """Compression (tiny body + low volume) plus MA-hovering at index ``i``.

    Returns ``(is_compression, hover_streak)``; ``is_compression`` already
    requires ``hover_streak >= min_hover_bars``. Mirrors the strategy's
    ``_coil_conditions`` evaluated at a single bar.
    """
    n = len(closes)
    atr_val = float(atr_series[i])
    if n < params.ma_slow + 2 or np.isnan(atr_val) or atr_val <= 0:
        return False, 0
    body = abs(float(closes[i]) - float(opens[i]))
    small_body = body <= params.body_atr_ratio * atr_val
    volumes_i = volumes[: i + 1]
    avg_vol = _avg_volume(volumes_i, params)
    vol_ok = avg_vol == avg_vol and float(volumes[i]) <= params.vol_mult * avg_vol
    if not (small_body and vol_ok):
        return False, 0
    streak = _hovering_streak(
        closes[: i + 1], sma_f[: i + 1], sma_s[: i + 1], atr_series[: i + 1], params
    )
    return streak >= params.min_hover_bars, streak


def _above_trend_ma(closes: np.ndarray, df: pd.DataFrame, params: Params) -> bool:
    """True when the latest close sits above its ``trend_ma`` SMA.

    ``params.trend_ma == 0`` disables the filter (always passes). Returns False
    when the series is shorter than the window or the SMA is undefined, i.e. a
    below-/un-defined trend fails the long-trend gate (conservative).
    """
    if params.trend_ma <= 0:
        return True
    if len(closes) < params.trend_ma:
        return False
    trend_sma = _ta_sma(df["close"], params.trend_ma)
    tv = float(trend_sma[-1]) if len(trend_sma) == len(closes) else float("nan")
    if tv != tv:
        return False
    return float(closes[-1]) > tv


def _latest_breakout(df: pd.DataFrame, params: Params) -> _Setup | None:
    """Replay the strategy's stateful coil/box machine; report the latest bar's
    live setup, else None.

    The strategy arms a compression box during the coil and keeps it live for
    ``decay_bars`` because the breakout *almost always happens on a later bar*
    than the tightest compression candle; a naive same-bar check (compression ⊗
    breakout) almost never fires. This replays ``_update_box`` across the whole
    frame so a box armed earlier can trigger a close-above-box-high breakout on
    the final bar — exactly like the backtest.

    Returns the final bar's ``_Setup`` whenever an armed (setup-ok, fresh)
    box is live, distinguishing ``breaked=True`` (close cleared the box high
    with the fast SMA above the slow SMA) from a still-coiling setup.
    """
    closes = df["close"].to_numpy(dtype=float)
    n = len(closes)
    if n < params.warmup_bars:
        return None
    if "volume" not in df.columns:
        return None
    opens = df["open"].to_numpy(dtype=float) if "open" in df.columns else closes
    volumes = df["volume"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float) if "high" in df.columns else closes
    lows = df["low"].to_numpy(dtype=float) if "low" in df.columns else closes

    atr_series = _ta_atr(df["high"], df["low"], df["close"], params.atr_period)
    sma_f = _ta_sma(df["close"], params.ma_fast)
    sma_s = _ta_sma(df["close"], params.ma_slow)
    if len(atr_series) != n or len(sma_f) != n or len(sma_s) != n:
        return None

    box_high: float | None = None
    box_low: float | None = None
    box_atr: float | None = None
    bar_counts: int = 0
    setup_ok: bool = False
    last_coil_body_frac: float = 1.0
    last_coil_streak: int = 0
    gain: float = 0.0

    for i in range(params.warmup_bars, n):
        # ---- _update_box -------------------------------------------------
        coil, streak = _coil_conditions_at(
            closes, opens, volumes, sma_f, sma_s, atr_series, i, params
        )
        if coil:
            lo = float(np.min(lows[i - params.comp_window + 1 : i + 1]))
            hi = float(np.max(highs[i - params.comp_window + 1 : i + 1]))
            if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
                box_high = box_low = box_atr = None
                bar_counts = 0
                setup_ok = False
            else:
                gain_ok = _big_move_gain(closes[: i + 1], params)
                if gain_ok is not None:
                    setup_ok = True
                    gain = gain_ok
                # else: keep prior latched setup_ok / gain
                box_high, box_low = hi, lo
                box_atr = float(atr_series[i])
                bar_counts = 0
                atr_i = float(atr_series[i])
                last_coil_body_frac = abs(float(closes[i]) - float(opens[i])) / (
                    params.body_atr_ratio * atr_i
                )
                last_coil_streak = streak
        elif box_high is not None and box_low is not None:
            # Not a compressive bar, but still coiling inside the band.
            if box_low <= closes[i] <= box_high:
                pass  # box stays armed
            else:
                box_high = box_low = box_atr = None
                bar_counts = 0
                setup_ok = False
        else:
            setup_ok = False

        bar_counts = bar_counts if coil else bar_counts + 1

    # ---- evaluate the latest bar against a live armed box ----------------
    live = box_high is not None and setup_ok and bar_counts <= params.decay_bars
    if not live:
        return None
    box_atr = box_atr if box_atr is not None else float(atr_series[-1])
    if np.isnan(box_atr) or box_atr <= 0:
        return None
    close = float(closes[-1])

    # Long-trend filter: a stock below its key SMA (default 50) is mid-pullback/
    # weak and not a compression-breakout candidate. Skip it entirely.
    if not _above_trend_ma(closes, df, params):
        return None

    sf, ss = float(sma_f[-1]), float(sma_s[-1])
    breaked = bool(close > box_high and sf == sf and ss == ss and sf > ss)
    pierce = (close - box_high) / box_atr if breaked else 0.0
    return _Setup(
        gain=gain if gain > 0 else (_big_move_gain(closes, params) or 0.0),
        pierce=pierce,
        body_frac=last_coil_body_frac,
        hover_streak=last_coil_streak,
        bars_ago=bar_counts,
        box_low=box_low if box_low is not None else 0.0,
        box_high=box_high,
        atr=box_atr,
        breaked=breaked,
    )


def _regime_allows_entry(state: ScreenState, params: Params) -> bool:
    """AE gate over ``regime_symbol``'s frame at the current bar.

    Uses the *batch* adaptive-entropy indicator (stateless) on the observer
    symbol's full frame, requiring its latest trend to clear
    ``regime_trend_min`` (and ``regime_min_strength`` when enabled). Borrows the
    strategy's convention: the observer symbol gates the whole book but is
    never itself a breakout candidate.
    """
    frame = state.frame(params.regime_symbol)
    if frame is None or frame.empty or "close" not in frame.columns:
        return True  # regime symbol absent from the feed -> un-gated
    closes = frame["close"]
    if len(closes) < params.ae_lookback + 1:
        return False

    from src.indicators.adaptive_entropy.pure import adaptive_entropy  # noqa: E402
    from src.indicators.adaptive_entropy.types import AdaptiveEntropyConfig  # noqa: E402

    res = adaptive_entropy(
        closes,
        frame["high"] if "high" in frame.columns else closes,
        frame["low"] if "low" in frame.columns else closes,
        AdaptiveEntropyConfig(lookback=params.ae_lookback, num_bins=params.ae_num_bins),
    )
    trend = res["trend"].iloc[-1]
    strength = res["trend_strength"].iloc[-1]
    if pd.isna(trend) or int(trend) < params.regime_trend_min:
        return False
    if params.regime_min_strength > 0.0 and (
        pd.isna(strength) or float(strength) < params.regime_min_strength
    ):
        return False
    return True


def _arm_score(setup: _Setup, params: Params) -> float:
    """0..1 armed-setup strength shared by the coiling and breakout scores.

    ``momentum_quality`` rewards a gain centered between ``min_gain`` and
    ``max_gain`` (too small under-achieves, too large risks a blow-off top).
    ``tightness`` is 1 when the coil body is ~0 and falls to 0 at the
    ``body_atr_ratio`` cap. ``freshness`` decays linearly from 1 at the coil
    to 0 at ``decay_bars``. Combine, clamp to [0, 1].
    """
    midsum = params.min_gain + params.max_gain
    span = max(1e-9, params.max_gain - params.min_gain)
    momentum_quality = max(0.0, 1.0 - abs(setup.gain - 0.5 * midsum) / (0.5 * span))
    tightness = min(1.0, max(0.0, 1.0 - setup.body_frac))
    freshness = min(1.0, max(0.0, 1.0 - setup.bars_ago / max(1e-9, params.decay_bars)))
    score = 0.50 * momentum_quality + 0.25 * tightness + 0.25 * freshness
    return float(min(1.0, max(0.0, score)))


def _setup_score(setup: _Setup, params: Params) -> float:
    """0..1 score for a live setup; a fresh breakout outranks a coil.

    A **breakout** (``breaked``) adds a ``depth_quality`` term (how far the
    close pierced the box, saturating at ``_PIERCE_SATURATE`` ATR) to the armed
    baseline. A still-**coiling** setup — same big-move, tightness and
    freshness — scores the baseline capped at ``_COIL_CEILING`` so it ranks
    above flat for watchlisting without ever outranking a real breakout.
    """
    base = _arm_score(setup, params)
    if not setup.breaked:
        return base * _COIL_CEILING
    depth_quality = min(1.0, max(0.0, setup.pierce / _PIERCE_SATURATE))
    return float(min(1.0, base + 0.30 * depth_quality))


def _flat(ts: pd.Timestamp, symbol: str, frame: pd.DataFrame) -> ScreenResult:
    return ScreenResult(
        symbol=symbol,
        timestamp=ts,
        score=0.0,
        action="flat",
        signals=("flat",),
        model_features=with_common_metrics({}, frame),
    )


# ---------------------------------------------------------------------------
# diagnostic regime helpers (used by the runner's ``build_state``)
# ---------------------------------------------------------------------------


def _trend_label(
    closes: pd.Series | None, fast: int, slow: int, range_threshold_pct: float
) -> TrendRegime | None:
    """Classify ``BULL`` / ``BEAR`` / ``RANGE`` via fast/slow EMA crossover.

    Diagnostic helper retained for ``build_state`` regime labeling — not part
    of the compression-breakout score/action.
    """
    if closes is None or len(closes) < slow:
        return None
    from src.indicators.ta import ema

    fast_ema = ema(closes, fast).iloc[-1]
    slow_ema = ema(closes, slow).iloc[-1]
    if pd.isna(fast_ema) or pd.isna(slow_ema) or slow_ema <= 0:
        return None
    spread = abs(fast_ema - slow_ema) / slow_ema
    if spread <= range_threshold_pct:
        return "RANGE"
    return "BULL" if fast_ema > slow_ema else "BEAR"


def _vol_label(closes: pd.Series | None, lookback: int = 20) -> VolRegime | None:
    """Diagnostic vol label from ``ta.volatility`` vs its historical median."""
    if closes is None or len(closes) < lookback * 2:
        return None
    from src.indicators.ta import volatility

    vols = volatility(closes, window=lookback, annualized=False).dropna()
    if len(vols) < 2:
        return None
    latest = float(vols.iloc[-1])
    if latest <= 0:
        return None
    med = float(vols.median())
    if med <= 0:
        return None
    ratio = latest / med
    if ratio > 1.25:
        return "HIGH_VOL"
    if ratio < 0.75:
        return "LOW_VOL"
    return "MED_VOL"


# ---------------------------------------------------------------------------
# the screen
# ---------------------------------------------------------------------------


def on_state(state: ScreenState, params: Params) -> tuple[ScreenResult, ...]:
    """Score every symbol's latest bar for a compression-breakout setup.

    Two tiers are surfaced (so the manual trader sees what's building even on a
    quiet or bearish-regime day):

      * **coiling** — an armed, setup-ok, fresh box that hasn't triggered yet
        (watchlist candidate; ``action="long"``), scored by ``_COIL_CEILING``.
      * **breakout** — close cleared the box high with the uptrend intact; the
        actual trigger (``action="long"`` only when the AE regime gate is
        bullish, else ``"flat"`` with a regime note).

    The AE regime gate scales the coiling score (a bearish index caps the
    watchlist) and blocks breakout *action*, but never hides a live setup
    entirely — hiding coiling setups would make the screen flat on most days.
    """
    results: list[ScreenResult] = []
    gate = _regime_allows_entry(state, params)
    regime_tag = "bull" if gate else "regime-bearish"
    for symbol, df in state.frames:
        if symbol == params.regime_symbol:
            # Observer/benchmark symbol: never a breakout candidate itself.
            continue
        if df is None or df.empty or "close" not in df.columns:
            continue
        if len(df) < params.warmup_bars:
            results.append(_flat(state.ts, symbol, df))
            continue

        setup = _latest_breakout(df, params)
        if setup is None:
            results.append(_flat(state.ts, symbol, df))
            continue

        score = _setup_score(setup, params)
        if setup.breaked:
            # Real trigger: reflect the regime in action, never hide the signal.
            action = "long" if gate else "flat"
            signals = (
                f"[breakout] close box high {setup.box_high:.2f} pierce "
                f"{setup.pierce:.2f} ATR",
                f"big-move gain {setup.gain:.0%}",
                f"coil streak {setup.hover_streak} bars",
                f"regime {regime_tag}",
            )
            if not gate:
                score *= _REGIME_SUPPRESS
        else:
            # Watchlist: always surface, but cap/scale in a bearish regime.
            action = "long"
            if not gate:
                score *= _REGIME_SUPPRESS
            signals = (
                f"[coiling armed] box {setup.box_low:.2f}-{setup.box_high:.2f} "
                f"({setup.bars_ago}b since coil)",
                f"big-move gain {setup.gain:.0%}",
                f"coil streak {setup.hover_streak} bars",
                f"regime {regime_tag}",
            )
        if score <= 0:
            results.append(_flat(state.ts, symbol, df))
            continue
        results.append(
            ScreenResult(
                symbol=symbol,
                timestamp=state.ts,
                score=score,
                action=action,
                signals=signals,
                model_features=with_common_metrics(
                    {
                        "compress_gain": setup.gain,
                        "pierce_atr": setup.pierce,
                        "coil_atr": setup.atr,
                        "coil_hover_bars": float(setup.hover_streak),
                        "coil_body_frac": setup.body_frac,
                        "box_high": setup.box_high,
                    },
                    df,
                ),
            )
        )
    return tuple(results)
