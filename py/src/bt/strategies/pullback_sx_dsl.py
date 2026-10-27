"""Pullback long, cross-sectional: buy the strongest name on a fresh low.

Entry: a fresh ``_FRESH_LOOK``-bar marginal low on drying volume, admitted only
inside the name's own uptrend. The universe is re-ranked every ``_SELECT_BARS``
and only the top ``select_top_n`` by dollar-volume and realized-vol rank may
open -- the wide admission of the single-name ancestor ran near-full capital
utilization and programmed thousands of cash-scaled fills.

Exits cut losers and leave winners alone: a hard ``entry_stop_atr`` stop set at
entry, plus a structural force-close under the prior ``_EXIT_LOOK``-bar low once
the dip thesis is invalidated. There is no trailing stop -- it measured as the
module's single largest drag.

Size reads live equity, split across ``book_slots`` and damped by the number of
names the rule fires on this bar, so a filling book does not exhaust cash before
the engine cohort scales. ``vix_size_gain`` then tilts each entry by the VIX
print's trailing percentile: positive sizes UP on a high-VIX tape.

Structure measured and rejected: a VIX or SPY-ROC admission gate (entries are
dried-up fresh lows, so they cluster in stressed tape -- the top VIX quintile is
a small share of trades and most of the P&L, so gating it removes the edge); a
cross-sectional trend-rank amplifier; a trailing stop; a concurrent-name cap; a
non-zero ROC floor; and a quiet-name ranking.

Cross-call state lives in ``ctx.shared`` (fresh per run, safe across worker
processes). Every read is cursor-truncated -- no lookahead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from src.bt.size.pure import risk_sized_qty
from src.bt.strategies.dsl import StrategyContext, strategy
from src.bt.strategies.types import StrategyParams
from src.indicators.macro._shared import load_daily

STRATEGY_TYPE = "pullback_sx_dsl"

# -- structural constants: hard-coded, never fitted ---------------------------
#: Entry: the close must undercut the minimum of the prior ``_FRESH_LOOK`` bars.
_FRESH_LOOK: int = 5
#: Volume ceiling, as a fraction of the prior ``_LEG_LOOK``-bar mean.
_VOL_DRY: float = 1.0
#: The name's own trailing ROC over this window must be non-negative.
_ROC_LOOK: int = 120
#: Volume baseline window for the drying test.
_LEG_LOOK: int = 40
#: ATR periods for the entry stop and the sizing denominator.
_ATR_PERIOD: int = 7
#: Forced exit: close under the prior ``_EXIT_LOOK``-bar low. Disabling it blows
#: max drawdown out and widening it degrades monotonically -- real risk work.
_EXIT_LOOK: int = 10
#: Universe re-ranking period, and the trailing stats window it ranks on.
_SELECT_BARS: int = 126
_SELECT_LOOK: int = 60
#: VIX percentile window (one trading year) and the multiplier's clamp.
_VIX_WIN: int = 250
_VIX_LO: float = 0.2
_VIX_HI: float = 3.0

_SELECT_CACHE_KEY = "pullback_sx_selection"
_VIX_CACHE_KEY = "pullback_sx_vix"
_COHORT_CACHE_KEY = "pullback_sx_cohort"


@dataclass(frozen=True)
class Params(StrategyParams):
    """The knobs that earned their place; everything else is a constant above."""

    #: Risk per open, read off live equity. Only meaningful against
    #: ``book_slots`` -- the per-name budget is ``risk_pct / book_slots``.
    risk_pct: float = 0.06
    #: Per-name budget divisor, and the floor on the sizing denominator.
    book_slots: float = 12.0
    #: Hard stop distance in ATR, set at entry.
    entry_stop_atr: float = 1.5
    #: Tradable set size: top N by dollar-volume + realized-vol rank. 0 = off.
    select_top_n: int = 15
    #: Sign of the realized-vol leg in that rank; ``+1`` prefers the loudest.
    #: Dollar-volume always prefers the more traded name -- the book needs
    #: turnover to fill, so liquidity is a constraint, not a preference.
    select_vol_sign: int = 1
    #: VIX size tilt: ``1 + gain * (2*pct - 1)`` on the print's trailing
    #: percentile, clamped to ``[_VIX_LO, _VIX_HI]``. 0 = neutral (no VIX read).
    vix_size_gain: float = 2.0


@strategy(stateful=True)
def on_candle(ctx: StrategyContext) -> None:
    if ctx.phase == "warmup":
        return
    p: Params = ctx.params
    for sym in ctx.symbols:
        _process_symbol(ctx, p, sym)


# -- small helpers ------------------------------------------------------------


def _traded(symbols: tuple[str, ...]) -> list[str]:
    """Symbols this module may trade (``GATE_`` macro series are read-only)."""
    return [s for s in symbols if not s.startswith("GATE_")]


def _bars(ctx: StrategyContext, sym: str) -> int:
    """Cursor-truncated bar count for ``sym``."""
    return len(ctx.ta.close(sym))


def _roc(o, look: int) -> float:
    """Trailing ``look``-bar rate of change of the close (fraction), or NaN."""
    arr = o.close.to_array()
    if arr.size <= look:
        return float("nan")
    ref = float(arr[-1 - look])
    cur = float(arr[-1])
    if not (np.isfinite(ref) and np.isfinite(cur)) or ref <= 0:
        return float("nan")
    return cur / ref - 1.0


def _pct_rank(vals: dict[str, float], sym: str) -> float:
    v = vals[sym]
    below = sum(1 for u in vals.values() if u < v)
    ties = sum(1 for u in vals.values() if u == v) - 1
    return (below + 0.5 * ties) / max(1, len(vals) - 1)


# -- VIX size tilt ------------------------------------------------------------


def _vix_series(ctx: StrategyContext) -> tuple[np.ndarray, np.ndarray] | None:
    """OBSERVED VIX prints as ``(dates, values)``, cached in ``ctx.shared``.

    ``load_daily`` reindexes to a calendar-daily grid with ``ffill``, repeating a
    print across weekends and holidays. Collapsing each run of equal values
    recovers the true print grid; ranking against the padded grid would weight
    the repeated weekend copies instead of real sessions.
    """
    cache: dict[str, tuple[np.ndarray, np.ndarray] | None] = ctx.shared.setdefault(
        _VIX_CACHE_KEY, {}
    )
    if "vix" not in cache:
        s = load_daily("vix")
        if s.empty:
            cache["vix"] = None
        else:
            vals = s.to_numpy(dtype=float)
            keep = np.ones(vals.size, dtype=bool)
            keep[1:] = vals[1:] != vals[:-1]
            cache["vix"] = (
                np.asarray(s.index.values, dtype="datetime64[ns]")[keep],
                vals[keep],
            )
    return cache["vix"]


def _vix_pct(ctx: StrategyContext) -> float | None:
    """The VIX print's trailing percentile (0..1), or None when unreadable."""
    series = _vix_series(ctx)
    if series is None:
        return None
    dates, values = series
    j = int(np.searchsorted(dates, np.datetime64(ctx.candle.timestamp))) - 1
    if j < 0 or not np.isfinite(values[j]):
        return None
    hist = values[max(0, j - _VIX_WIN) : j]
    hist = hist[np.isfinite(hist)]
    if hist.size < 2:
        return None
    return float(np.mean(hist < values[j]))


def _vix_factor(ctx: StrategyContext, p: Params) -> float:
    """Continuous size multiplier on the VIX percentile; 1.0 when unreadable.

    Nothing is gated on this: it only scales an entry that already passed every
    rule.
    """
    if p.vix_size_gain == 0.0:
        return 1.0
    pct = _vix_pct(ctx)
    if pct is None:
        return 1.0
    return float(np.clip(1.0 + p.vix_size_gain * (2.0 * pct - 1.0), _VIX_LO, _VIX_HI))


# -- rolling cross-sectional selection ----------------------------------------


def _selection(ctx: StrategyContext, p: Params) -> frozenset[str] | None:
    """Tradable set for this bar: the last rebalance's top-``select_top_n``.

    ``None`` when selection is off. Deselected names are not force-closed -- the
    entry stop and the forced exit still own an open position.
    """
    if p.select_top_n <= 0:
        return None
    holder: dict[str, object] = ctx.shared.setdefault(_SELECT_CACHE_KEY, {})
    bars = _bars(ctx, _traded(ctx.symbols)[0])
    last_bars = holder.get("bars")
    if (
        isinstance(last_bars, int)
        and bars - last_bars < _SELECT_BARS
        and "picks" in holder
    ):
        return cast("frozenset[str]", holder["picks"])

    dv: dict[str, float] = {}
    vol: dict[str, float] = {}
    for sym in _traded(ctx.symbols):
        o = ctx.ohlcv(sym)
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
    common = sorted(dv.keys() & vol.keys())
    if len(common) <= p.select_top_n:
        picks = frozenset(common)
    else:
        ranked = sorted(
            common,
            key=lambda s: (
                -(_pct_rank(dv, s) + p.select_vol_sign * _pct_rank(vol, s)),
                s,
            ),
        )
        picks = frozenset(ranked[: p.select_top_n])
    holder["bars"] = bars
    holder["picks"] = picks
    return picks


# -- entry rule ---------------------------------------------------------------


def _fires_long(ctx: StrategyContext, sym: str) -> bool:
    """A fresh marginal low over ``_FRESH_LOOK`` on drying volume, in an uptrend."""
    o = ctx.ohlcv(sym)
    closes = o.close.to_array()
    n = closes.size
    if n <= _LEG_LOOK + _FRESH_LOOK:
        return False
    roc = _roc(o, _ROC_LOOK)
    if not np.isfinite(roc) or roc < 0.0:
        return False
    fresh = closes[n - 1 - _FRESH_LOOK : n - 1]
    if fresh.size == 0 or not np.all(np.isfinite(fresh)):
        return False
    if not (closes[n - 1] < float(np.nanmin(fresh))):
        return False
    vols = o.volume.to_array()
    leg = vols[max(0, n - _LEG_LOOK - 1) : n - 1]
    if leg.size == 0 or not np.all(np.isfinite(leg)):
        return False
    mean_leg = float(np.nanmean(leg))
    cur_vol = float(vols[n - 1])
    if mean_leg <= 0 or not np.isfinite(cur_vol) or cur_vol <= 0:
        return False
    return cur_vol <= _VOL_DRY * mean_leg


def _cohort_count(ctx: StrategyContext, p: Params) -> int:
    """Flat, selected names whose entry rule fires on this bar.

    Lets size be damped before the engine cash-scales the cohort.
    """
    cache: dict[object, int] = ctx.shared.setdefault(_COHORT_CACHE_KEY, {})
    ts = ctx.candle.timestamp
    if ts in cache:
        return cache[ts]
    picks = _selection(ctx, p)
    count = 0
    for sym in _traded(ctx.symbols):
        if ctx.quantity(sym) != 0:
            continue
        if picks is not None and sym not in picks:
            continue
        if _fires_long(ctx, sym):
            count += 1
    if len(cache) > 8:
        cache.clear()
    cache[ts] = count
    return count


# -- risk: forced exit --------------------------------------------------------


def _forced_exit(ctx: StrategyContext, sym: str) -> None:
    """Close on a fresh ``_EXIT_LOOK``-bar low -- the dip thesis is invalidated."""
    o = ctx.ohlcv(sym)
    closes = o.close.to_array()
    if closes.size < _EXIT_LOOK + 1:
        return
    prior = closes[-1 - _EXIT_LOOK : -1]
    if not np.all(np.isfinite(prior)):
        return
    close = float(closes[-1])
    floor = float(np.min(prior))
    if not np.isfinite(close) or close >= floor:
        return
    ctx.close(
        sym,
        reason=(
            f"[sxg] broken {sym}: close {close:.2f} under {_EXIT_LOOK}b low {floor:.2f}"
        ),
    )


# -- sizing + admission -------------------------------------------------------


def _enter(ctx: StrategyContext, p: Params, sym: str) -> None:
    o = ctx.ohlcv(sym)
    close = float(o.close.to_array()[-1])
    atr = ctx.ta.atr(sym, period=_ATR_PERIOD).last()
    if not (np.isfinite(atr) and atr > 0 and close > 0):
        return
    equity = ctx.current_equity()
    if equity <= 0:
        return

    stop_dist = p.entry_stop_atr * atr
    open_names = sum(1 for s in _traded(ctx.symbols) if ctx.quantity(s) != 0)
    # An empty cohort is not a one-name book: keep ``book_slots`` as the floor so
    # a lone entry cannot inherit the whole per-name budget.
    book = max(p.book_slots, _cohort_count(ctx, p) + open_names, 1.0)
    damp = min(p.book_slots, book) / book
    risk_eff = (p.risk_pct / p.book_slots) * damp * _vix_factor(ctx, p)
    qty = risk_sized_qty(
        equity=equity, price=close, stop_dist=stop_dist, risk_pct=risk_eff
    )
    if qty <= 0:
        return
    size = min(qty * close / equity, 1.0)
    if size <= 0:
        return
    # Clamp to cash here so an over-subscribed cohort shows up as this honest
    # size rather than an unexplained one.
    cash_pct = ctx.state.portfolio.cash / equity
    if qty * close > ctx.state.portfolio.cash:
        size = min(size, max(0.0, cash_pct))

    ctx.long(
        sym,
        size=size,
        size_mode="equity",
        sl=close - stop_dist,
        reason=(
            f"[sxg] long {sym}: fresh {_FRESH_LOOK}b low {close:.2f} on dried vol, "
            f"ROC{_ROC_LOOK} {_roc(o, _ROC_LOOK):+.2%}, "
            f"risk {risk_eff:.2%} x{damp:.2f}, "
            f"cash {cash_pct:.0%}, "
            f"stop {stop_dist:.2f}"
        ),
    )


def _process_symbol(ctx: StrategyContext, p: Params, sym: str) -> None:
    if sym.startswith("GATE_"):
        return
    o = ctx.ohlcv(sym)
    if len(o.close) == 0:
        return
    close = float(o.close.to_array()[-1])
    if not np.isfinite(close) or close <= 0:
        return

    if ctx.quantity(sym) != 0:
        _forced_exit(ctx, sym)
        return

    picks = _selection(ctx, p)
    if picks is not None and sym not in picks:
        return
    if not _fires_long(ctx, sym):
        return
    _enter(ctx, p, sym)
