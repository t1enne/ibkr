"""XBI-gated laggard-basket mean reversion (DSL, daily, long-only).

Built from a cross-sectional research scan over the 25-name biotech core
(``universes/biotech.json`` ex SANA/AZN/XBI), daily closes, trading 2020-01 ->
2023-12 with OOS held to 2024+. The scan found the only sign-stable biotech
edge: cross-sectional *relative* reversion of the most-lagging names, which
survives where single-name time-series reversal dies (idiosyncratic catalyst
gaps, kurtosis ~300-800, destroy single-name stops but average out of a basket).

    LONG worst-20% (trailing 63d return rank), equal-weight, ~22d hold, gated
    XBI > its 60-day SMA:
        +1.18%/cycle  ann ~+13.5%  IR 0.68  win 55%  worst cycle -14%, n~840
    A/B (isolating the gate, full 2019-2026 window):
        gated XBI>60d : IR 0.68  worst -14%   (cycle avg +1.18%)
        no gate       : IR 0.53  worst -24%
        gated XBI<60d : IR 0.44  worst -24%
    The XBI>60d filter is a DRAWDOWN-REDUCER, not a return-driver: it barely
    moves mean return but cuts the worst cycle -24% -> -14% by keeping the
    book flat through confirmed bear-biotech episodes (no knife-catching).

Why this survives on biotech: the losers-recover effect compounds at the
*relative* basket level. A worst-20% basket of laggards cannot be wiped by any
single name's binary event; each name's idiosyncratic gap is averaged out.
This is the same mechanism as the repo's established `sector_mean_reversion`
family that wins elsewhere, validated on biotech.

Each refresh (once per ``hold_days`` trading bars, on any warm candle) the
strategy:

  * gates the regime: XBI close vs its trailing ``gate_span`` SMA. Book forms /
    refreshes only while the gate is ON (default: XBI above its SMA). When the
    gate turns OFF the open book is flattened (no knife-catching).
  * ranks the panel cross-sectionally by trailing ``lookback``-day log-return.
  * opens equal-weight LONG positions in the ``tail_n`` WORST-return members
    (ex XBI, which is the gate/benchmark and never traded here).

All cross-call bookkeeping (refresh cadence, open legs) lives in ``ctx.shared``
via ``@strategy(stateful=True)`` (fresh per run/window); pure helpers operate
only on cursor-truncated OHLCV views. Fills occur at the next bar's open, so a
signal close never leaks into its own bar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.bt.strategies.dsl import strategy, StrategyContext
from src.bt.strategies.types import StrategyParams

STRATEGY_TYPE = "xbi_gated_laggard_dsl"

_CTX_KEY = "xbi_gated_laggard_ctx"


@dataclass(frozen=True)
class Params(StrategyParams):
    # -- reversion lookback / hold --------------------------------------------
    lookback: int = 63  # trailing daily bars over which the laggard rank is measured
    hold_days: int = 22  # bars between refresh (rebalance cadence)
    # -- regime gate -----------------------------------------------------------
    gate_symbol: str = "XBI"  # gate index; never traded
    gate_span: int = 60  # gate SMA lookback (bars)
    gate_above: bool = True  # gate ON when gate_symbol close > SMA
    # -- portfolio -------------------------------------------------------------
    tail_n: int = 5  # WORST-return members to hold long (of the ~25-name panel)
    position_size: float = 0.18  # per-name 0..1 equity fraction (full idio deploy)
    # -- gating / history ------------------------------------------------------
    warmup_bars: int = 84  # visible bars (gate_span + lookback margin) pre-first-book
    min_total_daily_history: int = 130  # member needs this many daily bars to rank


@dataclass
class _Ctx:
    """Cross-call bookkeeping for one run/window (held in ``ctx.shared``)."""

    bars_to_refresh: int = 0
    initialized: bool = False


# ---------------------------------------------------------------------------
# pure helpers (typed; cursor-truncated OHLCV only)
# ---------------------------------------------------------------------------


def _closes(ctx: StrategyContext, sym: str) -> np.ndarray | None:
    """Visible (cursor-truncated) daily close array for ``sym``, else None."""
    try:
        return ctx.ohlcv(sym).close.to_array()
    except KeyError:
        return None


def _is_gate(sym: str, gate: str) -> bool:
    """Gate index check (case-insensitive; the gate is never traded)."""
    return sym.lower() == gate.lower()


def _tail_log_ret(closes: np.ndarray, window: int) -> float:
    """``ln(c[-1] / c[-1-window])`` from the visible tail; NaN if too short."""
    if closes.size <= window:
        return float("nan")
    a, b = closes[-1], closes[-(window + 1)]
    if a <= 0 or b <= 0 or not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return float(np.log(a / b))


def _gate_on(ctx: StrategyContext, params: Params) -> bool | None:
    """Three-state XBI gate: True (on), False (off), None (not yet evaluable).

    On when the gate symbol's close is on the ``gate_above`` side of its
    trailing ``gate_span`` SMA. None when the gate symbol is absent from the
    store or its SMA is not yet defined (no book yet).
    """
    gate_c = _closes(ctx, params.gate_symbol)
    if gate_c is None or gate_c.size < params.gate_span + 1:
        return None
    close = float(gate_c[-1])
    sma = float(gate_c[-params.gate_span :].mean())
    if not (np.isfinite(close) and np.isfinite(sma) and sma > 0):
        return None
    above = close > sma
    return bool(above if params.gate_above else not above)


def _select_worst_laggards(
    retmap: dict[str, float], exclude: set[str], tail_n: int
) -> list[str]:
    """Names of the ``tail_n`` worst log-return members, gate excluded.

    Pure and context-free so it is directly testable; ``retmap`` maps symbol
    -> finite trailing lookback log-return. Ascending-sorts and keeps the worst
    ``tail_n`` (never the gate, already in ``exclude``).
    """
    ordered = sorted((s for s in retmap if s not in exclude), key=retmap.__getitem__)
    n = min(tail_n, len(ordered))
    if n < 1:
        return []
    return ordered[:n]


def _laggard_targets(ctx: StrategyContext, params: Params) -> list[str]:
    """Worst-``tail_n`` panel members by trailing ``lookback`` log-return.

    The gate index is never a tradeable member. Members lacking the minimum
    daily history drop out. Returns [] when too few names rank (never opens a
    degenerate basket).
    """
    scored: dict[str, float] = {}
    for sym in ctx.symbols:
        if _is_gate(sym, params.gate_symbol):
            continue
        c = _closes(ctx, sym)
        if c is None or c.size < params.min_total_daily_history:
            continue
        r = _tail_log_ret(c, params.lookback)
        if np.isfinite(r):
            scored[sym] = r
    return _select_worst_laggards(scored, {params.gate_symbol.lower()}, params.tail_n)


# ---------------------------------------------------------------------------
# DSL strategy
# ---------------------------------------------------------------------------


@strategy(stateful=True)
def on_candle(ctx: StrategyContext):
    # Warmup bars fill strategy state only — no fill can happen there, so the
    # DSL raises on any entry emitted during them. Accumulators above/below
    # still receive warmup bars because the engine feeds the store regardless.
    if ctx.phase == "warmup":
        return
    params: Params = ctx.params
    shared = ctx.shared
    st: _Ctx = shared.setdefault(_CTX_KEY, _Ctx())

    # Hold cadence: refresh only every ``hold_days`` bars once initialized.
    if st.initialized and st.bars_to_refresh > 0:
        st.bars_to_refresh -= 1
        if st.bars_to_refresh > 0:
            return

    # Warmup: the gate SMA + the laggard lookback must both be defined.
    gate_c = _closes(ctx, params.gate_symbol)
    if gate_c is None or gate_c.size < params.warmup_bars:
        return

    # Regime gate decides whether a basket may exist at all this refresh.
    on = _gate_on(ctx, params)
    if on is False:
        # Gate off -> flat (hard drawdown reducer), no knife-catching.
        _flatten_long(ctx, params, "[xbi-gld] gate off flat")
        st.bars_to_refresh = params.hold_days
        st.initialized = True
        return
    if on is None:
        # Gate not yet evaluable -> no book, defer.
        return

    _refresh(ctx, st, params)


def _flatten_long(ctx: StrategyContext, params: Params, reason: str) -> None:
    """Close every open long in the panel (gate-off / rotation cleanup)."""
    for sym in ctx.symbols:
        if _is_gate(sym, params.gate_symbol):
            continue
        if ctx.quantity(sym) > 1e-6:
            ctx.close(sym, reason=reason)


def _refresh(ctx: StrategyContext, st: _Ctx, params: Params) -> None:
    """Hold the worst-laggard basket for the next ``hold_days`` window.

    Rotates out names no longer in the worst tail, opens fresh ones that have
    entered it. Fills occur at the next bar's open.
    """
    targets = set(_laggard_targets(ctx, params))

    # (a) rotate out members no longer in the worst-laggard tail
    for sym in ctx.symbols:
        if _is_gate(sym, params.gate_symbol):
            continue
        if ctx.quantity(sym) > 1e-6 and sym not in targets:
            ctx.close(sym, reason=f"[xbi-gld] rotate out {sym}")

    # (b) open fresh laggards, equal-weight
    current = {sym for sym in ctx.symbols if ctx.quantity(sym) > 1e-6}
    for sym in targets:
        if sym in current:
            continue
        ctx.long(
            sym,
            size=params.position_size,
            reason=f"[xbi-gld] long laggard {sym}",
            size_mode="equity",
        )

    st.bars_to_refresh = params.hold_days
    st.initialized = True
