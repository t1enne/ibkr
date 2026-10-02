"""
VWATR divergence strategy -- long only. Three divergence exits over
vwatr_dsl; no entry filter (entry features correlate ~0 with trade return).
`ctx.ta.vwatr`/`vwatr_baseline` are shared TA; ``atr_ratio`` uses PLAIN
rolling-mean ATR (not Wilder) -- swapping in ``ctx.ta.atr`` changes every
number.

KEY FACTS (condensed ledger).
CONFIG: strats/pass/vwatr_div_exp8_6y_risk0.08.json -- SR 1.12, +2223% vs
SPY +118%, DD -50%, kurt 12.5.
- risk_pct interior plateau, never argmax; decel_ratio boundary degenerate;
  thin early IS breaks optimizers (split 2|3 ok).
- No gate exists: hole is FLAT, not losing -- the missing mass is the fat
  right tail; a gate cannot create an absent melt-up.
- Slope divergence is the real exit; price-unit + baseline divergences failed.
- FRED sizing (vix dir) moved the 2022-23 hole; composite (fred x earn/
  revenue, clamp [0.5,1.5]) restores it -- SR 1.22, OOS 1.09. fred is
  load-bearing, revenue additive; earn alone loses.
- EXP30 (30 names, risk 0.01) ships: SR 1.35, DD -33%, kurt 2.46, OOS 1.31
  -- diversification + per-name risk cut, not signal.
- Trade-count cohorts: highest cohort wins OOS; less trades is an in-sample
  trap.
- Over-subscribed runs (cash-constrained cohorts) silently scale opens; the
  events cluster on crowded up-legs, not in deep drawdowns, so the distortion
  is a level tax, not a tail overlay -- results there are advisory.
- Trailing de-risk controls (cash/DD/vol throttles) and concurrency caps only
  trade away Sharpe/return; they do not add edge.
- Full-window universe trimming cuts drawdown and over-subscription but is
  universe fitting; the non-fit alternative is the rolling PIT fundamental
  selector (`fund_min_pct` = revenue-YoY + operating-margin rank).
- `max_positions`, `dd_damp_*`, `fund_min_pct` default OFF; 1d-only candidate.

HYGIENE: single switch -- `decel_ratio=0` turns the divergence exit OFF.
"""

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
from src.bt.strategies.fundamentals_context import SeriesPIT
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

# Guard for the volume-weighted denominator: ~0-volume window has no meaningful VWATR.
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
    # -- dynamic sizing: macro (fred) and/or earnings-momentum (earn) --
    risk_scale: Literal["none", "fred", "earn", "composite"] = "none"
    # "fred" = risk_pct x macro score (VIX percentile, portfolio-level);
    # "composite" = same x fred score x earn score, re-clamped [0.5, 1.5]
    risk_fred: str = ""  # FRED asset (load_daily name), e.g. "vix"; "" = off
    risk_fred_dir: Literal[-1, 1] = 1  # +1 size up with level, -1 size down
    risk_fred_mode: Literal["dir", "band"] = "dir"  # monotone | research band
    rs_fred_win: int = 250  # FRED percentile window (daily); VIX pct250
    # "earn" = risk_pct x fundamentals-momentum score (per-symbol, both
    # directions; metric picks the voice -- revenue fires on loss-makers)
    risk_earn_metric: Literal[
        "eps_diluted", "net_income", "revenue", "operating_cash_flow"
    ] = "eps_diluted"
    # dynamic-score clamp band (earn + composite re-clamp); hi < lo degenerates
    # to a constant scale (sweep diagnostic only).
    risk_clamp_lo: float = 0.5
    risk_clamp_hi: float = 2.0
    # -- divergence layer --
    decel_ratio: float = 0.0  # slope-div exit: frac of prior slope; 0 = OFF
    norm_mode: str = "none"  # price-unit div exit: none | pct_ratio | atr_ratio
    baseline_gate: str = "none"  # entry gate: none | above | below


def _macro_gate_open(ctx: StrategyContext, specs: tuple[str, ...]) -> bool:
    """True when every spec "SYM:WINDOW:THRESHOLD" passes (cursor-safe); short
    history blocks conservatively."""
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


# VIX band (research_vix.py): pct250 Q3 [0.45,0.8] negative in both halves;
# shrink only inside the band.
_RS_BAND_LO: float = 0.45
_RS_BAND_HI: float = 0.80
_RS_BAND_FACTOR: float = 0.65
_RS_DIR_AMP: float = 1.0  # dir-mode amp, engine-validated (OOS 0.67)
_FRED_CACHE_KEY = "vwatr_div_fred_series"


def _fred_series(
    ctx: StrategyContext, name: str
) -> tuple[np.ndarray, np.ndarray] | None:
    """Daily forward-filled FRED asset as ``(dates, values)``, cached in
    ``ctx.shared`` (per-run, worker-safe)."""
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
    """Fraction of the trailing window (current excluded) strictly below
    ``series[i]``; 0.5 neutral on out-of-bounds / NaN / thin window."""
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

    Percentile of the series' latest print against its PRIOR prints, then
    monotone scaling (``risk_fred_dir``) or the non-monotone band
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


#: Same-quarter YoY match window (days): 364-366 typical, fiscal filers drift;
#: 330-400 admits the same fiscal quarter, never a different-length span.
_QUARTER_YOY_LO = 330
_QUARTER_YOY_HI = 400


def _earn_yoy_growth(series: SeriesPIT) -> float | None:
    """YoY growth of the latest visible earnings vs its same-quarter filing.

    Matches ``period_end`` ~1 year apart so cumulative-YTD spans never mix
    quarters; visibility is cursor-borne. ``None`` on short series,
    non-positive baseline, or no same-quarter match yet.
    """
    spans = series.spans()
    values = series.values()
    if len(values) < 2:
        return None
    latest_end = spans[-1][1]
    latest = values[-1]
    for i in range(len(values) - 2, -1, -1):
        days = (latest_end - spans[i][1]).days
        if _QUARTER_YOY_LO <= days <= _QUARTER_YOY_HI:
            prev = values[i]
            if not (np.isfinite(prev) and np.isfinite(latest)) or prev <= 0:
                return None
            return float((latest - prev) / prev)
    return None


#: ``risk_earn_metric`` -> (statement accessor, field). Net income/EPS silent
#: on loss-makers; revenue fires on every filer.
_EARN_METRIC_FIELD: dict[str, tuple[str, str]] = {
    "eps_diluted": ("income", "eps_diluted"),
    "net_income": ("income", "net_income"),
    "revenue": ("income", "revenue"),
    "operating_cash_flow": ("cashflow", "operating_cash_flow"),
}


def _earn_risk_score(p: Params, ctx: StrategyContext, sym: str) -> float:
    """Per-symbol earnings-momentum risk multiplier: risk_pct x this.

    Same-quarter YoY growth, clamped to ``[risk_clamp_lo, risk_clamp_hi]``;
    neutral 1.0 when no fundamentals store or no matchable filing (missing
    fundamental is not a weak fundamental). Unknown metric is a config
    error, raised here rather than KeyError-ing on a later bar.
    """
    if p.risk_earn_metric not in _EARN_METRIC_FIELD:
        raise ValueError(
            f"unknown risk_earn_metric {p.risk_earn_metric!r}; pick one of "
            f"{', '.join(sorted(_EARN_METRIC_FIELD))}"
        )
    statement, field = _EARN_METRIC_FIELD[p.risk_earn_metric]
    accessor = getattr(ctx.fundamentals, statement)(sym)
    series: SeriesPIT = getattr(accessor, field)
    growth = _earn_yoy_growth(series)
    if growth is None:
        return 1.0
    return min(p.risk_clamp_hi, max(p.risk_clamp_lo, 1.0 + growth))


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
            # plain rolling-mean ATR (not Wilder) for atr_ratio
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
    elif p.risk_scale == "earn":
        score = _earn_risk_score(p, ctx, sym)
        risk_eff = p.risk_pct * score
    elif p.risk_scale == "composite":
        score = _fred_risk_score(p, ctx, ctx.candle.timestamp) * _earn_risk_score(
            p, ctx, sym
        )
        score = min(p.risk_clamp_hi, max(p.risk_clamp_lo, score))
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
        if p.risk_scale in ("fred", "earn", "composite")
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
    """Post-run chart spec (cursor at terminal bar; ctx.ta series are the
    full visible history)."""
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
        # slope pair only: vwatr/close = noise; baseline div failed.
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
