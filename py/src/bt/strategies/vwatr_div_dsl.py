(
    """VWATR divergence strategy -- long only. Adds THREE divergence exits over
vwatr_dsl; no entry filter (entry features correlate ~0 with trade return).
`ctx.ta.vwatr`/`vwatr_baseline` are shared TA; ``atr_ratio`` uses PLAIN
rolling-mean ATR (not Wilder) -- swapping in ``ctx.ta.atr`` changes every
number.

LEDGER (read before touching params).
CONFIG: strats/pass/vwatr_div_exp8_6y_risk0.08.json (8 high-vol names, 1d, 6y)
-- SR 1.12, +2223% vs SPY +118%, DD -50%, kurt 12.5.
- `risk_pct`: 6y response is UNIMODAL interior peak; shipped value sits on the
  plateau -- treat as plateau, never argmax. `decel_ratio`: MONOTONE toward
  boundary = degenerate. `bt split --folds 2|3` passes; `bt optimize` fails on
  thin early IS (picks grid boundary).
- TAIL CONCENTRATION: median trade << mean; a few winners carry gross profit.
- SYMBOL DEPENDENCE: worst drop-one omission drops SR to ~SPY (ENPH among the
  LEAST damaging -- earlier drafts wrong).
- REGIME: P&L mostly from the last third of the window -- late-window bull.
- NO GATE EXISTS (pre-registered search, 45 series, 0/45 p<0.05;
  strats/wip/vwatr_div_gate_research/). Hole is FLAT, not losing -- the
  missing mass is the FAT RIGHT TAIL. Gate filters entries; cannot create an
  absent melt-up. CALENDAR PLACEBO: hole is mid-sample, so slow level series
  separate by construction.
- EXITS: slope divergence = real mechanism, degenerate param (decel boundary);
  price-unit + baseline divergence FAILED; wider trail WORSE.
- FRED SIZING (`risk_scale="fred"`, vix dir, 250d, base 0.02, 20 names): mean
  OOS 0.67 (0.48/0.76/0.75), fold-1 hole 0.48 -- first variant to move the
  2022-23 hole (none/composite same-base: 0.48 mean / 0.22 min; fred adds
  +0.19 OOS at identical base). Blends / other series failed (t10y2y 0.37;
  diluted vix 0.45-0.48). VIX mean-reverts: dir -1 shrinks in vol, grows in
  calm -- unlike t10y2y (era level, date-bet). Composite module reverted;
  `_percentile_rank` stays local. Below 8-name pass (OOS 1.06). `hyspread`
  FRED data starts 2023-09-30; `vix` (1990+) / `t10y2y` (1976+) cover hole.

HYGIENE: single switch -- `decel_ratio=0` turns the divergence exit OFF
(`div_exit` removed, was collinear). Grid over `strategy_params` only.
`bt split --folds 2|3`; thin early IS breaks optimizers.
"""
    ""
)

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

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

# Guard for the volume-weighted denominator (also used by ``plot``'s ratio
# panels): a window whose volume sums to ~0 has no meaningful VWATR.
_EPS = 1e-12


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full(num.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > _EPS)
    out[ok] = num[ok] / den[ok]
    return out


@dataclass(frozen=True)
class _State:
    best: float | None = None


@dataclass(frozen=True)
class Params(StrategyParams):
    # -- VWATR --
    vwatr_period: int = 14
    vwatr_base_win: int = 60
    vwatr_mult: float = 2.0
    slope_bars: int = 2
    breakout_look: int = 20
    risk_pct: float = 0.02
    # -- FRED-only dynamic sizing --
    risk_scale: Literal["none", "fred"] = "none"  # "fred" = risk_pct x macro score
    risk_fred: str = ""  # FRED asset (load_daily name), e.g. "vix"; "" = off
    risk_fred_dir: Literal[-1, 1] = 1  # +1 size up with level, -1 size down
    risk_fred_mode: Literal["dir", "band"] = "dir"  # monotone | research band
    rs_fred_win: int = 250  # FRED percentile window (daily); VIX pct250
    # -- divergence layer --
    decel_ratio: float = 0.0  # slope-div exit: frac of prior slope; 0 = OFF
    norm_mode: str = "none"  # price-unit div exit: none | pct_ratio | atr_ratio
    baseline_gate: str = "none"  # entry gate: none | above | below


def _macro_gate_open(ctx: StrategyContext, specs: tuple[str, ...]) -> bool:
    """True when every spec "SYM:WINDOW:THRESHOLD" passes: close z-scored vs
    trailing WINDOW (cursor-safe); short history blocks conservatively."""
    for spec in specs:
        try:
            sym, win_s, thr_s = spec.split(":")
            win, thr = int(win_s), float(thr_s)
        except ValueError:
            return False
        o = ctx.ohlcv(sym)
        if o is None or len(o.close) < win + 1:
            return False
        arr = o.close.to_array()
        window = arr[-win:]
        if not np.all(np.isfinite(window)):
            return False
        mu = float(np.mean(window))
        sd = float(np.std(window, ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            return False
        z = (float(arr[-1]) - mu) / sd
        if not (z >= thr):
            return False
    return True


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


def _norm_diverging(
    vwatr_arr: np.ndarray,
    atr_arr: np.ndarray,
    close: np.ndarray,
    n: int,
    norm_mode: str,
) -> bool:
    """VWATR rising in $ while falling as % of price; delta-vs-delta not level."""
    if n < 3:
        return False
    if norm_mode == "pct_ratio":
        norm = _safe_div(vwatr_arr, close)
    elif norm_mode == "atr_ratio":
        norm = _safe_div(vwatr_arr, atr_arr)
    else:
        return False
    d_v = vwatr_arr[n - 1] - vwatr_arr[n - 2]
    d_n = norm[n - 1] - norm[n - 2]
    if not (np.isfinite(d_v) and np.isfinite(d_n)):
        return False
    return d_v > 0 and d_n < 0  # $ expanding, % contracting


# VIX research band (research_vix.py): pct250 Q3 [0.45,0.8] negative in BOTH
# split halves; Q1/Q4 stable-positive. Shrink only inside the band.
_RS_BAND_LO: float = 0.45
_RS_BAND_HI: float = 0.80
_RS_BAND_FACTOR: float = 0.65
_RS_DIR_AMP: float = 1.0  # dir-mode amp, engine-validated (OOS 0.67)
_FRED_CACHE_KEY = "vwatr_div_fred_series"


def _fred_series(
    ctx: StrategyContext, name: str
) -> tuple[np.ndarray, np.ndarray] | None:
    """Daily forward-filled FRED asset as ``(dates, values)``, cached in
    ``ctx.shared`` (per-run, worker-safe; loaded once per process)."""
    cache: dict[str, tuple[np.ndarray, np.ndarray] | None] = ctx.shared.setdefault(
        _FRED_CACHE_KEY, {}
    )
    if name not in cache:
        s = load_daily(name)
        if s.empty:
            cache[name] = None
            return None
        dates = np.asarray(s.index.values, dtype="datetime64[ns]")
        cache[name] = (dates, s.to_numpy(dtype=float))
    return cache[name]


def _percentile_rank(series: np.ndarray, i: int, window: int) -> float:
    """Fraction of the trailing window (current index excluded) strictly
    below ``series[i]``; 0.5 (neutral) on out-of-bounds / NaN / thin window."""
    if i < 0 or i >= series.size:
        return 0.5
    current = series[i]
    if not np.isfinite(current):
        return 0.5
    finite_window = series[max(0, i - window) : i]
    finite_window = finite_window[np.isfinite(finite_window)]
    if finite_window.size < 2:
        return 0.5
    return float(np.mean(finite_window < current))


def _fred_risk_score(p: Params, ctx: StrategyContext, ts: pd.Timestamp) -> float:
    """FRED-only risk multiplier: risk_pct x this, NO composite blend.

    Percentile of the macro series' latest print at ``ts`` against its own
    PRIOR daily prints (current excluded, ``load_daily`` lookahead-free),
    then either monotone scaling (``risk_fred_dir``, structural amp
    ``_RS_DIR_AMP``) or the non-monotone research band
    (``risk_fred_mode="band"``). Missing asset / out-of-span -> neutral 1.0.
    Clamped to [0.5, 1.5].
    """
    if not p.risk_fred:
        return 1.0
    series = _fred_series(ctx, p.risk_fred)
    if series is None:
        return 1.0
    dates, values = series
    j = int(np.searchsorted(dates, np.datetime64(ts))) - 1
    if j < 0 or not np.isfinite(values[j]):
        return 1.0
    pct = _percentile_rank(values, j, p.rs_fred_win)
    if p.risk_fred_mode == "band":
        score = _RS_BAND_FACTOR if _RS_BAND_LO <= pct <= _RS_BAND_HI else 1.0
    else:
        score = 1.0 + (pct - 0.5) * p.risk_fred_dir * _RS_DIR_AMP
    return min(1.5, max(0.5, score))


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
                f"[vwatr-div] slope decel {sym}: close {px:.2f} at best, "
                f"slope below {p.decel_ratio:.2f}x prior"
            )
        # 2) price-unit divergence
        elif p.norm_mode in ("pct_ratio", "atr_ratio"):
            # plain rolling-mean ATR (not Wilder) for atr_ratio -- see ledger
            atr_arr = (
                ctx.ta.plain_atr(sym, p.vwatr_period).to_array()
                if p.norm_mode == "atr_ratio"
                else vwatr_arr
            )
            if _norm_diverging(vwatr_arr, atr_arr, close, n, p.norm_mode):
                reason = (
                    f"[vwatr-div] {p.norm_mode} diverge {sym}: close {px:.2f} "
                    f"at best, vwatr +{vwatr_arr[n - 1] - vwatr_arr[n - 2]:.4f} "
                    f"but ratio falling"
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
                    f"[vwatr] trail {sym}: close {px:.2f} "
                    f"{stop_distance:.2f} ({stop_distance / px:.2%}) off best {best:.2f}"
                ),
            )
        return

    # ---- entry: rising expansion + structural breakout ----
    if not np.isfinite(base_now) or base_now <= 0:
        return

    # 3) baseline divergence gate
    if p.baseline_gate == "above" and not (vwatr > base_now):
        return
    if p.baseline_gate == "below" and not (vwatr < base_now):
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

    stop_price = p.vwatr_mult * vwatr
    equity = ctx.current_equity()
    if p.risk_scale == "fred":
        score = _fred_risk_score(p, ctx, ctx.candle.timestamp)
        risk_eff = p.risk_pct * score
    else:
        score = 1.0
        risk_eff = p.risk_pct
    qty = risk_sized_qty(
        equity=equity, price=px, stop_dist=stop_price, risk_pct=risk_eff
    )
    if qty <= 0:
        return
    size = min(qty * px / equity, 1.0) if equity > 0 else 0.0
    if size <= 0:
        return

    risk_note = (
        f"risk {risk_eff:.1%} (x{score:.2f}) qty {qty:.2f}"
        if p.risk_scale == "fred"
        else f"risk {risk_eff:.1%} qty {qty:.2f}"
    )
    reason = (
        f"[vwatr-div] entry {sym}: close {px:.2f} > {prior_high:.2f}, "
        f"vwatr {vwatr:.3f} ({vwatr / base_now:.2f}x base {base_now:.3f}), "
        f"rising vs {vwatr_prev:.3f} over {p.slope_bars}b, "
        f"{risk_note} stop {stop_price:.2f}"
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
    """Post-run chart spec. The cursor is at the terminal bar here, so every
    ``ctx.ta`` series is the full visible history (not a truncation).
    """
    sym = ctx.candle.symbol
    df = ctx.state.candles.get((sym, ctx.interval))
    if df is None or len(df) < params.vwatr_period + params.vwatr_base_win:
        return PlotSpec()

    close = ctx.ta.close(sym).to_array()
    vwatr_arr = ctx.ta.vwatr(sym, params.vwatr_period).to_array()
    slope = vwatr_arr - _shift(vwatr_arr, params.slope_bars)
    slope_prev = _shift(slope, params.slope_bars)

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
        # slope pair only: vwatr/close = noise, baseline div failed (ledger)
        panels=(
            Panel(
                sparse(_as_series(df.index, slope)),
                f"vwatr slope ({params.slope_bars}b)",
                (0.0,),
            ),
            Panel(
                sparse(_as_series(df.index, slope_prev)), "vwatr slope (prior)", (0.0,)
            ),
        ),
        markers=tuple(markers),
    )
