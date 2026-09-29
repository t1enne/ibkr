"""VWATR expansion strategy -- long only.

VWATR = smooth(TR * volume) / smooth(volume): a true-range average weighted by
each bar's own volume. Same unit as ATR. All windows come from cursor-truncated
ctx.ohlcv, so no bar can leak; VWATR is not read from ctx.ta (no accessor for
the weighted form). Entry, two conditions:
  1. rising VWATR:  VWATR[t] > VWATR[t - slope_bars]
  2. breakout:      close > max(close[t - breakout_look : t])

Slope is measured on VWATR in absolute price units, not the ratio (VWATR /
baseline): the ratio form swings the mean trade 1.10 / 3.43 / 2.25 across
slope_bars 2/3/5 (fitting noise), the absolute form moves little (4.60 / 4.33 /
3.60). No level gate on the ratio -- the [expand, mania] band is occupied ~38%
of the time and cannot time anything.

VWATR does three jobs: gate (above); size (stop = vwatr_mult * VWATR, shares
via risk_sized_qty so each name risks the same equity fraction whatever its
volatility); exit (chandelier trail off the running best close, distance
recomputed each bar from current VWATR so it tightens as expansion dies).

CLOSED NEGATIVE RESULTS -- do not repeat these searches. Coil precondition -- FAILED. VWATR compressed into a low tight band then
breaking its ceiling. Fatal defect: on MSTR it was a strict SUBSET operation
(18 admitted, 138 rejected, zero new entries) -- it could only delete good
trades. coil_tight = max/min over the window measured window *span*, not
current compression; 1.6 was the ~25th percentile of its own distribution.
20-combo sweep (vwatr_mult x coil_win, 8 syms): every coil-on row lost, best 6
combos all coil_win=0; DD per trade rose (0.59 -> 1.53). A replacement (floor
<= p25 of its own 120-bar history, 5-bar breakout, second path) fixed the
subset defect (242 added entries, added trades beat base OOS +2.49% vs -4.31%)
but still failed: CI95 [-1.44, +6.68], P(<=0)=0.051, Sharpe 0.21 -> 0.15.

Market regime gate -- FAILED. SPY above SMA(200) AND 20d realized vol below its
252d median. 4-cell test (8 syms): Sharpe base 0.21 / gate 0.21 / coil 0.15 /
gate+coil 0.25. Gate delta bootstrap CI spans zero in every period. DD per
trade worse (-1.16 -> -1.42) -- the maxDD gain was arithmetic from dropping 34%
of trades (permutation p98 on maxDD, p3.5 on DD/tr). Delta sign flips with
trail mult (+0.55 -> -2.09pp at 2 -> 6). No plateau: 24-cell sweep swings
+3.69% -> -4.93% on vol_pct 0.5-0.7 alone. Does not flip the OOS sign. Prior
art: momentum_compression_breakout_dsl.py documents the same
window-vs-daily-gate failure for a SPY 200SMA gate.

Root cause -- the entry is not the problem. 18 high-vol syms, 2024-2026, n=231
trades: mean +0.30%, median -6.02%, win 38.1%. Corr of every entry feature with
trade return: VWATR/baseline -0.074, rank120 -0.028, TR contraction +0.003,
Donchian width -0.022, volume ratio -0.012 (CI ~ +-0.13; all straddle zero). No
entry-side filter can fix this. The many-small-losses distribution is an EXIT
signature -- the vwatr_mult trail is hit constantly. The exit is the untested
lever. vwatr_mult=2.0 is contested: wider is monotonically better across 10
names (5.5% -> 23.0% at 2.0x -> 4.0x) but monotonically worse on MSTR 2024.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from src.bt.size.pure import risk_sized_qty
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

STRATEGY_TYPE = "vwatr_dsl"
_STATE_KEY = "vwatr_state"

_EPS = 1e-12


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling mean; entries before the window fills are NaN."""
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if window < 1 or len(values) < window:
        return out
    from numpy.lib.stride_tricks import sliding_window_view

    out[window - 1 :] = sliding_window_view(values, window).mean(axis=1)
    return out


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Wilder true range; first bar NaN (no previous close)."""
    n = len(close)
    tr = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return tr
    prev_close = close[:-1]
    hl = high[1:] - low[1:]
    hc = np.abs(high[1:] - prev_close)
    lc = np.abs(low[1:] - prev_close)
    tr[1:] = np.maximum(np.maximum(hl, hc), lc)
    return tr


def _vwatr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    period: int,
) -> np.ndarray:
    """Volume-weighted ATR: ``smooth(TR * volume) / smooth(volume)``."""
    tr = _true_range(high, low, close)
    num = _rolling_mean(tr * volume, period)
    den = _rolling_mean(volume, period)
    out = np.full(close.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > _EPS)
    out[ok] = num[ok] / den[ok]
    return out


@dataclass(frozen=True)
class _State:
    best: float | None = None  # running best (highest) close since entry


@dataclass(frozen=True)
class Params(StrategyParams):
    # -- VWATR (the only indicator) --
    vwatr_period: int = 14  # smoothing period of TR*vol and vol
    vwatr_base_win: int = 60  # window for the VWATR baseline mean
    vwatr_mult: float = 2.0  # stop distance as a multiple of current VWATR
    # -- expansion direction --
    slope_bars: int = 2  # VWATR[t] must exceed VWATR[t - slope_bars]
    # -- structure --
    breakout_look: int = 20  # close must top the last N closes (entry trigger)
    # -- risk --
    risk_pct: float = 0.02  # equity fraction risked per trade (VWATR-sized)


@strategy(bars="1d", stateful=True)
def on_candle(ctx: StrategyContext) -> None:
    if ctx.phase == "warmup":
        return
    p: Params = ctx.params
    for sym in ctx.symbols:
        _process_symbol(ctx, p, sym)


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
    high = o.high.to_array()
    low = o.low.to_array()
    volume = o.volume.to_array()
    n = len(close)
    if n == 0 or not (len(high) == len(low) == len(volume) == n):
        return

    px = float(close[-1])
    if not np.isfinite(px) or px <= 0:
        return

    vwatr_arr = _vwatr(high, low, close, volume, p.vwatr_period)
    vwatr = float(vwatr_arr[-1]) if n else float("nan")
    if not np.isfinite(vwatr) or vwatr <= 0:
        return

    # ---- exit while long: VWATR chandelier trail off the running best ----
    # Distance uses *current* VWATR, not the entry value, so it widens with an
    # expanding range and tightens as expansion dies.
    if ctx.quantity(sym) != 0:
        best = state.best
        if best is None or not np.isfinite(best):
            put(best=px)
            return
        if px > best:
            put(best=px)
            best = px
        stop_distance = p.vwatr_mult * vwatr
        if px <= best - stop_distance:
            put(best=None)
            ctx.close(
                sym,
                reason=(
                    f"[vwatr] trail {sym}: close {px:.2f} "
                    f"{stop_distance:.2f} ({stop_distance / px:.2%}) off best {best:.2f}"
                ),
            )
        return

    # ---- entry: rising expansion + structural breakout ----
    baseline = _rolling_mean(vwatr_arr, p.vwatr_base_win)
    base_now = float(baseline[-1])
    if not np.isfinite(base_now) or base_now <= 0:
        return

    # (1) rising expansion: absolute VWATR, not the ratio. See module docstring.
    if n <= p.slope_bars:
        return
    vwatr_prev = float(vwatr_arr[n - 1 - p.slope_bars])
    if not np.isfinite(vwatr_prev):
        return
    if not (vwatr > vwatr_prev):
        return

    # (2) structural breakout -- VWATR supplies no direction.
    prior = close[n - 1 - p.breakout_look : n - 1]
    if len(prior) < p.breakout_look or not np.all(np.isfinite(prior)):
        return
    prior_high = float(np.max(prior))
    if not (px > prior_high):
        return

    stop_price = p.vwatr_mult * vwatr
    equity = ctx.current_equity()
    qty = risk_sized_qty(
        equity=equity, price=px, stop_dist=stop_price, risk_pct=p.risk_pct
    )
    if qty <= 0:
        return
    size = min(qty * px / equity, 1.0) if equity > 0 else 0.0
    if size <= 0:
        return

    reason = (
        f"[vwatr] entry {sym}: close {px:.2f} > {prior_high:.2f}, "
        f"vwatr {vwatr:.3f} ({vwatr / base_now:.2f}x base {base_now:.3f}), "
        f"rising vs {vwatr_prev:.3f} over {p.slope_bars}b, "
        f"risk {p.risk_pct:.1%} qty {qty:.2f} stop {stop_price:.2f}"
    )

    put(best=px)
    ctx.long(sym, size=size, size_mode="equity", reason=reason)


# ---------------------------------------------------------------------------
# plot -- VWATR slope / ratio panels, trail band, and trade markers, so the
# read behind each entry is auditable. VWATR is recomputed with the same
# helpers on_candle uses; ctx.ta has no volume-weighted accessor. Markers come
# from state.portfolio.trades, so a marker cannot disagree with a fill.
# ---------------------------------------------------------------------------


def _daily_frame(ctx: StrategyContext, sym: str):
    """Cursor-truncated OHLCV DataFrame for ``sym`` at the signal interval."""
    return ctx.state.candles.get((sym, ctx.interval))


def plot(ctx: StrategyContext, params: Params) -> PlotSpec:
    """Post-run chart spec: slope + ratio panels, trail band, trade markers.

    Called once per symbol with the cursor at the terminal bar, so every series
    is the full visible history. Returns an empty spec (rather than raising) for
    a symbol whose frame is missing or too short to form the gate -- the output
    layer logs-and-drops a raising ``plot``, but returning empty is cheaper and
    leaves the chart honest about having nothing to show.
    """
    sym = ctx.candle.symbol
    df = _daily_frame(ctx, sym)
    if df is None or len(df) < params.vwatr_period + params.vwatr_base_win:
        return PlotSpec()

    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    volume = df["volume"].to_numpy(dtype=np.float64)

    vwatr_arr = _vwatr(high, low, close, volume, params.vwatr_period)
    baseline = _rolling_mean(vwatr_arr, params.vwatr_base_win)

    ratio = _safe_div(vwatr_arr, baseline)
    # VWATR slope over slope_bars: >0 is the entry condition.
    slope = vwatr_arr - _shift(vwatr_arr, params.slope_bars)

    # --- price pane: trail distance band, and breakout channel (shifted 1) ---
    stop_band = params.vwatr_mult * vwatr_arr
    upper = close + stop_band
    lower = close - stop_band

    look = params.breakout_look
    channel = np.full(close.shape, np.nan, dtype=np.float64)
    if len(close) > look:
        from numpy.lib.stride_tricks import sliding_window_view

        channel[look:] = sliding_window_view(close, look).max(axis=1)[:-1]

    series = {
        "stop_hi": sparse(_as_series(df.index, upper)),
        "stop_lo": sparse(_as_series(df.index, lower)),
        "breakout": sparse(_as_series(df.index, channel)),
        "ratio": sparse(_as_series(df.index, ratio)),
        "slope": sparse(_as_series(df.index, slope)),
        "vwatr": sparse(_as_series(df.index, vwatr_arr)),
    }

    return PlotSpec(
        overlays=(
            Overlay(series["breakout"], "breakout_high", "line", "price"),
            Overlay(series["stop_hi"], "trail_hi", "line", "price"),
            Overlay(series["stop_lo"], "trail_lo", "line", "price"),
        ),
        panels=(
            Panel(series["slope"], f"vwatr slope ({params.slope_bars}b)", (0.0,)),
            Panel(series["vwatr"], "vwatr (price units)", ()),
            Panel(series["ratio"], "vwatr / baseline (diagnostic)", ()),
        ),
        markers=_trade_markers(ctx, sym),
    )


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Elementwise ``num / den`` with non-finite / tiny denominators -> NaN."""
    out = np.full(num.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > _EPS)
    out[ok] = num[ok] / den[ok]
    return out


def _shift(values: np.ndarray, bars: int) -> np.ndarray:
    """``values`` shifted forward by ``bars`` (head filled with NaN)."""
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if bars < len(values):
        out[bars:] = values[: len(values) - bars]
    return out


def _as_series(index, values: np.ndarray):
    """numpy array + DatetimeIndex -> pandas Series (for ``sparse``)."""
    import pandas as pd

    return pd.Series(values, index=index)


def _trade_markers(ctx: StrategyContext, sym: str) -> tuple[Marker, ...]:
    """Entry markers from the run's own trade log (``pivot_low`` glyph).

    Read off ``state.portfolio.trades`` rather than re-derived: a marker is a
    claim about where an entry *actually happened*, and the trade log is the
    only source that cannot disagree with the fills.
    """
    out: list[Marker] = []
    for trade in ctx.state.portfolio.trades:
        if trade.symbol != sym or trade.position != ActionType.long:
            continue
        out.append(
            Marker(
                ts=ms(trade.entry_time),
                price=float(trade.entry_price),
                kind="pivot_low",
            )
        )
    return tuple(out)
