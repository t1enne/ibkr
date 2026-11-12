"""MFI divergence reversal, long+short.

Entry thesis: **price/flow divergence** -- price prints a lower low while MFI
prints a higher low (bullish absorption: sellers are being absorbed), or the
mirror (price higher high, MFI lower high = distribution).

Why this shape (measured, not assumed):

  * The original *level* filter ("MFI oversold trough in the window, now
    lifting") is net **value-destroying** on top of the raw reversal core: an
    MFI oversold trough appears on ~65% of bars, and requiring a *deeper*
    trough lowered forward returns. Removed.
  * MFI **divergence** is non-redundant with price and was robust across a 2D
    (``div_look`` x ``div_min_gap``) plateau -- the level filter was not.
  * Benchmark overlays (price-vs-EMA and bench-momentum gates) were swept and
    are **net-harmful** in every setting tested (best gated Sharpe well below
    ungated). Removed -- no regime gate.

Trend handling is the ``sym_roc_look`` / ``sym_roc_min`` sign gate: a name
net-rising admits longs (a divergence low in an uptrend is a dip), a name
net-falling admits shorts. ``sym_roc_look=0`` disables the sign gate. The
filter's *value is regime-dependent* (helps in sustained trend, neutral in
chop) -- pass the appropriate look per config rather than assuming one default.

Per-side chandelier ATR trailing exits and structural entry stops. Per-symbol
state in ``ctx.shared``; every read is cursor-truncated (no lookahead).

ONE module, TWO execution venues; ``config.bars`` is the only switch:
``["1d"]`` reads the base feed (daily venue, reads and fills both daily);
``["1h","1d"]`` reads completed daily buckets and uses the base bar only to time
the fill. Params keep their DAILY-bar units either way. On the intraday venue an
entry is a daily decision emitted once per completed bucket and fills at the next
base bar's open, since a signal-bar-close fill is not achievable live. The trail
is why the hourly grid exists: between buckets the daily rule cannot be
re-evaluated, so the trail level is armed as a PRICE STOP and the risk layer
checks it on every base bar, pricing gaps through the level.

Measured layers that did NOT survive on either payoff shape, so they are absent
by design: volume participation at the divergence extreme (flat to inverted), and
a VIX-percentile size amplifier (helped nothing at a clipped or a wide trail).
Do not re-add them without new evidence.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from src.bt.strategies.dsl import strategy, StrategyContext
from src.bt.strategies.ta_context import OhlcvView
from src.bt.strategies.types import StrategyParams
from src.bt.strategies.utils import gross_gate

STRATEGY_TYPE = "mfi_divergence_dsl"
_STATE_KEY = "mfi_divergence_state"

#: Signal read interval. ``["1d"]`` reads the base feed (daily venue, fill on
#: the signal bar); ``["1h","1d"]`` reads COMPLETED daily buckets and uses the
#: base bar only to time the fill.
_SIGNAL_INTERVAL = "1d"
_BASE_INTERVAL_1D = "1d"

#: Per-symbol state writer: replaces the call-scoped state with ``**kw`` applied.
_Put = Callable[..., None]


@dataclass(frozen=True)
class _State:
    """Per-symbol cross-call state (``ctx.shared``, minted per run)."""

    # running best (lowest) close since a long entry; None while flat/long-off
    best_long: float | None = None
    # running worst (highest) close since a short entry; None while flat/short-off
    best_short: float | None = None
    # True while the previous bar held an open position; used to stamp the bar
    # at which this symbol went flat and start the re-entry cooldown.
    was_in_position: bool = False
    # bar index (len(close) of the bar) at which the position was closed; -1
    # means never / no cooldown armed. Entry is blocked while
    # ``n - last_exit_bar < cooldown_bars``.
    last_exit_bar: int = -1
    #: Visible signal-feed length at the last base bar -- the once-per-bucket
    #: guard's baseline. Only consulted when the read is an HTF feed.
    daily_count: int | None = None


@dataclass(frozen=True)
class Params(StrategyParams):
    """Strategy parameters. Bar counts are in DAILY bars on both venues."""

    # -- MFI / divergence --
    mfi_period: int = 14
    div_look: int = 30  # divergence swing window (split prior|recent halves)
    div_min_gap: float = 8.0  # required MFI higher-low lead over the prior swing
    # Required *price* extension between the two swings, as a fraction (0.05 =
    # recent swing extreme must overshoot the prior by 5%). This is the
    # load-bearing quality gate: an mfi-gap-only trigger fires on ~every bar
    # (measured mean forward return ~0), while demanding the price itself thrust
    # past the prior extreme makes the divergence non-trivial (mean fwd10 rises
    # monotonically with this floor). 0.0 disables it.
    div_min_price_ext: float = 0.0
    # -- side admission --
    allow_longs: bool = True
    allow_shorts: bool = True
    # Bars to stay flat after a position on the same symbol closes (any reason:
    # trail, stop, end) before re-entry is allowed. 0 disables the cooldown.
    # Suppresses re-shorting/re-longing the same persistent signal right after
    # an exit (measured: the stale-signal re-entry loop on V-reversal names was
    # the single largest per-name loss driver).
    cooldown_bars: int = 0
    # -- per-symbol trend sign gate (0 look = off) --
    sym_roc_look: int = 0  # bars for the symbol's own ROC
    sym_roc_min: float = 0.0  # |ROC| (fraction) needed to admit a side
    # -- moving-average regime gates --
    # Longs are admitted only when price is *above* its MA, shorts only when
    # below -- i.e. the divergence is traded with the prevailing trend, not
    # against it.
    # ``spy_ma_gate``: gate on the benchmark (SPY) vs its own MA (market regime).
    # ``symbol_ma_gate``: gate on the traded symbol vs its own MA.
    # ``need_ma_alignment``: require *both* the SPY and the symbol MA to agree
    # with the trade side (implies both individual gates regardless of their
    # flags). SPY and symbol lengths are swept independently via
    # ``spy_ma_period`` / ``symbol_ma_period``.
    spy_ma_gate: bool = False
    symbol_ma_gate: bool = False
    need_ma_alignment: bool = False
    spy_ma_period: int = 200
    symbol_ma_period: int = 200
    benchmark: str = "SPY"
    # -- risk / management --
    atr_period: int = 14
    trail_atr_mult: float = 1.2  # ATRs off the running extreme that bank trade
    entry_stop_atr: float = 3.0  # initial stop, ATRs beyond entry
    risk_pct: float = 0.4
    warmup_bars: int = 60
    # -- drawdown control --
    # Per-name notional is ``risk_pct`` of live equity, so a correlated cluster
    # can dominate the book (measured: DD -49% on 6y at risk_pct=0.4).
    # ``risk_pct`` is the primary DD lever; this caps aggregate |gross| notional
    # as a fraction of **live equity** (1.0 = off).
    max_gross_exposure: float = (
        1.0  # aggregate |notional| cap, fraction of live equity (1.0 = off)
    )


def _state_interval(ctx: StrategyContext) -> str | None:
    """Signal read interval; ``None`` means the base feed is itself daily."""
    return None if ctx.interval == _BASE_INTERVAL_1D else _SIGNAL_INTERVAL


def _is_htf(ctx: StrategyContext) -> bool:
    """True when the base bar only times the fill (signal is a daily bucket)."""
    return _state_interval(ctx) is not None


def _new_daily_bucket(
    ctx: StrategyContext,
    holder: dict[str, _State],
    sym: str,
    n_daily: int,
) -> bool:
    """True on the FIRST base bar of a newly completed daily bucket.

    A daily rule stays true for the rest of its session, so without this guard
    an intraday base would re-emit the same daily entry on every hourly bar.
    Always True on the daily venue, where each bar is its own bucket.
    """
    if not _is_htf(ctx):
        return True
    state = holder.get(sym) or _State()
    last = state.daily_count
    if last is None:  # first base bar this run processed for ``sym``
        holder[sym] = replace(state, daily_count=n_daily)
        return False
    holder[sym] = replace(state, daily_count=max(last, n_daily))
    return n_daily > last


@strategy(stateful=True)
def on_candle(ctx: StrategyContext) -> None:
    # Warmup bars fill strategy state only — no fill can happen there, so the
    # DSL raises on any entry emitted during them. Accumulators above/below
    # still receive warmup bars because the engine feeds the store regardless.
    if ctx.phase == "warmup":
        return
    p: Params = ctx.params
    for sym in ctx.symbols:
        _process_symbol(ctx, p, sym)


def _process_symbol(ctx: StrategyContext, p: Params, sym: str) -> None:
    """Advance one symbol on this bar: manage an open lot or seek an entry."""
    holder: dict[str, _State] = ctx.shared.setdefault(_STATE_KEY, {})
    state = holder.get(sym)
    if state is None:
        state = _State()
        holder[sym] = state

    def put(**kw: object) -> None:
        holder[sym] = replace(state, **kw)

    si = _state_interval(ctx)
    o = ctx.ohlcv(sym, si)
    if o is None or len(o.close) <= p.warmup_bars:
        return

    # The once-per-bucket guard is evaluated on the SIGNAL feed length, and it
    # is consumed for every symbol on every base bar so the baseline is adopted
    # on the first bar of the run and never re-baselines later.
    new_bucket = _new_daily_bucket(ctx, holder, sym, len(o.close))
    state = holder[sym]
    if not new_bucket:
        # Between daily reads the base bar owns NO action: the entry stop and
        # the trail level armed on the previous bucket already cover the
        # position, and re-deciding off a partly-formed session would be a
        # different rule. Also keeps the intraday venue O(1) per base bar.
        return

    closes = o.close.to_array()
    n = len(closes)
    if n == 0:
        return
    close = float(closes[-1])

    mfi_vals = ctx.ta.mfi(sym, period=p.mfi_period, interval=si).to_array()
    if n != len(mfi_vals):
        return

    atr = ctx.ta.atr(sym, period=p.atr_period, interval=si).last()
    if not np.isfinite(atr):
        atr = 0.0

    qty = ctx.quantity(sym)
    if qty != 0:
        _manage_open(ctx, p, sym, state, put, close, atr, qty > 0)
        return
    _maybe_enter(ctx, p, sym, state, put, o, mfi_vals, n, si, close, atr)


def _manage_open(
    ctx: StrategyContext,
    p: Params,
    sym: str,
    state: _State,
    put: _Put,
    close: float,
    atr: float,
    is_long: bool,
) -> None:
    """Trail an open lot on its active side; emit the exit once breached."""
    put(was_in_position=True)
    level = _trail_level(p, state, put, close, atr, is_long)
    if level is not None:
        _fire_trail(ctx, sym, level, is_long)


def _trail_level(
    p: Params,
    state: _State,
    put: _Put,
    close: float,
    atr: float,
    is_long: bool,
) -> float | None:
    """Advance the high-water mark; return the trail level once it is breached.

    ``None`` means no exit: the mark moved further in the trade's favour, or ATR
    is unusable. The caller decides HOW that level becomes an exit (see
    ``_fire_trail``), because the two venues emit it differently.
    """
    best = state.best_long if is_long else state.best_short
    if best is None or not (best > 0):
        put(best_long=close) if is_long else put(best_short=close)
        return None
    sign = 1.0 if is_long else -1.0
    if (close - best) * sign > 1e-9:  # a new extreme in the trade's favour
        put(best_long=close) if is_long else put(best_short=close)
        return None
    if atr > 0 and (best - close) * sign >= p.trail_atr_mult * atr:
        put(best_long=None) if is_long else put(best_short=None)
        return best - sign * p.trail_atr_mult * atr
    return None


def _fire_trail(ctx: StrategyContext, sym: str, level: float, is_long: bool) -> None:
    """Emit the trail exit on the venue in use.

    Daily venue: the trail decision and the fill are the same bar, so a close
    signal is exact. Intraday venue: the daily read cannot be re-evaluated
    between buckets, so the level is armed as a PRICE STOP instead and the
    engine's risk layer checks it on every base bar -- the whole point of having
    the hourly grid. That also gets gap-through pricing from the risk path
    instead of a plain next-open close.
    """
    if _is_htf(ctx):
        if is_long:
            ctx.set_stops(sym, sl=level)
        else:
            ctx.set_stops(sym, tp=level)
        return
    side = "long" if is_long else "short"
    ctx.close(
        sym,
        reason=(
            f"[mfi_div] trail tp {side} {sym}: close crossed the trail level "
            f"{level:.2f}"
        ),
    )


def _maybe_enter(
    ctx: StrategyContext,
    p: Params,
    sym: str,
    state: _State,
    put: _Put,
    o: OhlcvView,
    mfi_vals: np.ndarray,
    n: int,
    si: str | None,
    close: float,
    atr: float,
) -> None:
    """Emit an entry when an admissible side's divergence trigger fires."""
    if state.was_in_position:
        # The close happened since the last bar (trail, stop, or end): stamp
        # this bar so the cooldown window starts here.
        put(was_in_position=False, last_exit_bar=n)
    long_ok, short_ok = _admissible_sides(ctx, p, sym, si)
    if _cooling(n, state.last_exit_bar, p.cooldown_bars):
        return
    if long_ok and _divergence(o, p, mfi_vals, n, is_long=True):
        _enter(ctx, p, sym, close, atr, is_long=True)
        return
    if short_ok and _divergence(o, p, mfi_vals, n, is_long=False):
        _enter(ctx, p, sym, close, atr, is_long=False)


def _admissible_sides(
    ctx: StrategyContext, p: Params, sym: str, si: str | None
) -> tuple[bool, bool]:
    """``(long_ok, short_ok)`` after the trend-sign and MA-regime gates."""
    sym_roc = _roc(ctx, sym, p.sym_roc_look, si) if p.sym_roc_look > 0 else 0.0
    if p.sym_roc_look > 0 and not np.isfinite(sym_roc):
        return False, False
    long_ok = p.allow_longs and ((p.sym_roc_look == 0) or (sym_roc >= p.sym_roc_min))
    short_ok = p.allow_shorts and ((p.sym_roc_look == 0) or (sym_roc <= -p.sym_roc_min))
    spy_up = _above_ma(ctx, p.benchmark, p.spy_ma_period, si)
    sym_up = _above_ma(ctx, sym, p.symbol_ma_period, si)
    return _ma_side_ok(
        long_ok,
        short_ok,
        spy_up,
        sym_up,
        p.spy_ma_gate,
        p.symbol_ma_gate,
        p.need_ma_alignment,
    )


def _roc(ctx: StrategyContext, sym: str, look: int, interval: str | None) -> float:
    """Trailing ``look``-bar rate of change of ``sym``'s close (fraction).

    Cursor-truncated. NaN when history is short or the reference close is
    non-finite, so every caller fails closed.
    """
    if look <= 0:
        return float("nan")
    o = ctx.ohlcv(sym, interval)
    if o is None or len(o.close) <= look:
        return float("nan")
    arr = o.close.to_array()
    ref = float(arr[-1 - look])
    cur = float(arr[-1])
    if not (np.isfinite(ref) and np.isfinite(cur)) or ref <= 0:
        return float("nan")
    return cur / ref - 1.0


def _cooling(now: int, last_exit_bar: int, cooldown_bars: int) -> bool:
    """True while a re-entry on this symbol must be blocked.

    ``now`` is the current bar index (``len(close)``) and ``last_exit_bar`` the
    bar at which the last position closed (``-1`` = never held one). Blocks
    while fewer than ``cooldown_bars`` bars have elapsed since the exit; a
    non-positive ``cooldown_bars`` disables the gate entirely.
    """
    if cooldown_bars <= 0 or last_exit_bar < 0:
        return False
    return now - last_exit_bar < cooldown_bars


def _above_ma(
    ctx: StrategyContext, sym: str, period: int, interval: str | None
) -> bool | None:
    """True when ``sym``'s latest close is above its ``period``-bar SMA.

    Returns ``None`` when history is too short or either value is non-finite,
    so every caller fails closed (a gate must not admit a side it cannot check).
    """
    if period <= 0:
        return None
    o = ctx.ohlcv(sym, interval)
    if o is None or len(o.close) < period:
        return None
    cur = float(o.close.to_array()[-1])
    ma = ctx.ta.sma(sym, period=period, interval=interval).last()
    if not (np.isfinite(cur) and np.isfinite(ma)) or ma <= 0:
        return None
    return cur > ma


def _ma_side_ok(
    long_ok: bool,
    short_ok: bool,
    spy_up: bool | None,
    sym_up: bool | None,
    spy_ma_gate: bool,
    symbol_ma_gate: bool,
    need_ma_alignment: bool,
) -> tuple[bool, bool]:
    """Apply the MA regime gates to the admissible sides (pure).

    ``spy_up`` / ``sym_up`` are the ``_above_ma`` results (``None`` = unknown,
    fails closed). ``need_ma_alignment`` overrides both individual flags and
    requires benchmark and symbol to trend with the side.
    """
    if need_ma_alignment:
        return (
            long_ok and spy_up is True and sym_up is True,
            short_ok and spy_up is False and sym_up is False,
        )
    if spy_ma_gate:
        long_ok = long_ok and spy_up is True
        short_ok = short_ok and spy_up is False
    if symbol_ma_gate:
        long_ok = long_ok and sym_up is True
        short_ok = short_ok and sym_up is False
    return long_ok, short_ok


def _divergence(
    o: OhlcvView, p: Params, mfi_vals: np.ndarray, n: int, is_long: bool
) -> bool:
    """Price/flow divergence trigger for one side (pure).

    Long: over the last ``div_look`` bars the recent half's price low is *below*
    the prior half's while its MFI low is *above* it by ``div_min_gap`` --
    sellers are being absorbed. Short mirrors it on highs. Either way
    ``div_min_price_ext`` makes the price itself thrust past the prior extreme,
    so a divergence against an essentially flat one does not fire.
    """
    halves = _divergence_halves(o, p, mfi_vals, n, is_long)
    if halves is None:
        return False
    thrust, flow_lead, prior_ext = halves
    if not (thrust > 0.0 and flow_lead > p.div_min_gap):
        return False
    if p.div_min_price_ext > 0 and prior_ext > 0:
        if thrust / abs(prior_ext) < p.div_min_price_ext:
            return False
    return True


def _divergence_halves(
    o: OhlcvView, p: Params, mfi_vals: np.ndarray, n: int, is_long: bool
) -> tuple[float, float, float] | None:
    """``(thrust, flow_lead, prior price extreme)``, or None when unmeasurable.

    ``thrust`` is how far the recent half's price extreme runs past the prior
    half's and ``flow_lead`` how far MFI refuses to follow it -- the thesis in
    two numbers. Long reads extremes as lows, short as highs. None on short
    history, a misaligned series, or any non-finite value in the windows, so the
    caller fails closed rather than trading an unmeasurable window.
    """
    look = p.div_look
    if n < p.warmup_bars + 2 or look < 2:
        return None
    closes = o.close.to_array()
    if len(closes) != n or len(mfi_vals) != n:
        return None
    half = look // 2
    prior_px = closes[n - look : n - half]
    rec_px = closes[n - half : n]
    prior_mo = mfi_vals[n - look : n - half]
    rec_mo = mfi_vals[n - half : n]
    if not (prior_px.size and rec_px.size):
        return None
    if not (
        np.all(np.isfinite(prior_px))
        and np.all(np.isfinite(rec_px))
        and np.all(np.isfinite(prior_mo))
        and np.all(np.isfinite(rec_mo))
    ):
        return None
    if is_long:
        prior_ext, rec_ext = float(np.min(prior_px)), float(np.min(rec_px))
        prior_flow, rec_flow = float(np.min(prior_mo)), float(np.min(rec_mo))
        return prior_ext - rec_ext, rec_flow - prior_flow, prior_ext
    prior_ext, rec_ext = float(np.max(prior_px)), float(np.max(rec_px))
    prior_flow, rec_flow = float(np.max(prior_mo)), float(np.max(rec_mo))
    return rec_ext - prior_ext, prior_flow - rec_flow, prior_ext


def _enter(
    ctx: StrategyContext,
    p: Params,
    sym: str,
    close: float,
    atr: float,
    is_long: bool,
) -> None:
    """Size and emit the entry, with its structural stop."""
    if not (atr > 0 and close > 0 and ctx.state.portfolio.cash > 0):
        return
    # -- drawdown control: aggregate gross-notional cap vs live equity --
    # The candidate lot is ``risk_pct`` of live equity, already equity-relative.
    if gross_gate(
        ctx.current_equity(),
        ctx.state.portfolio.positions,
        p.max_gross_exposure,
        p.risk_pct,
    ):
        return
    stop_dist = p.entry_stop_atr * atr
    if is_long:
        ctx.long(
            sym,
            size=p.risk_pct,
            size_mode="equity",
            sl=close - stop_dist,
            reason=(
                f"[mfi_div] long {sym}: bullish MFI divergence {close:.2f} "
                f"(gap={p.div_min_gap:.1f})"
            ),
        )
    else:
        ctx.short(
            sym,
            size=p.risk_pct,
            size_mode="equity",
            sl=close + stop_dist,
            reason=(
                f"[mfi_div] short {sym}: bearish MFI divergence {close:.2f} "
                f"(gap={p.div_min_gap:.1f})"
            ),
        )
