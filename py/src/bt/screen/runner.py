"""Screen runner — orchestrates frames -> ScreenState -> screen -> ranked output.

I/O (loading frames, computing trend/vol) stays at the edges; the runner wires
pure pieces together and has no side effects other than reading the frames it is
given.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from src.bt.regime.tf_consensus import weighted_align
from src.bt.screen.screens import init_screen, resolve_screen_params
from src.bt.screen.screens.momentum_spy_gate import _trend_label, _vol_label
from src.bt.screen.types import (
    Action,
    ScreenResult,
    ScreenState,
    TrendRegime,
    VolRegime,
)


@dataclass(frozen=True)
class DivergenceParams:
    """Knobs for the cross-interval TF-alignment ranker.

    ``lower_tf_weight`` biases alignment toward the execution timeframe (the
    lowest supplied interval). Values >1.0 mean a bullish lowest TF counts for
    more than the same label on a higher TF. ``alignment_threshold`` is the
    minimum weighted alignment score (0..1) needed to call a non-flat action.
    """

    lower_tf_weight: float = 1.5
    alignment_threshold: float = 0.5


def rank_divergence(
    states: dict[str, ScreenState],
    params: DivergenceParams | None = None,
    *,
    top: int | None = None,
) -> tuple[ScreenResult, ...]:
    """Rank symbols by trend alignment across timeframes (TF divergence).

    Consumes the per-interval ``ScreenState`` objects from ``state_per_interval`` and
    merges them into ONE result per symbol. ``long_align``/``short_align`` are
    computed from the weighted share of intervals that label the symbol
    ``BULL``/``BEAR`` (lowest TF weighted via ``lower_tf_weight``); ``score`` is
    the max share, so fully-aligned consensus ranks high and net-flat rows sort
    to the bottom. A ``BULL`` on one interval coexisting with a ``BEAR`` on
    another sets ``divergent=True`` and is surfaced as a warning signal — the
    cross-TF conflict, not a hidden edge.
    """
    p = params or DivergenceParams()
    results: list[ScreenResult] = []
    symbols = set().union(*(set(st.trend) for st in states.values()))
    latest_ts = max(
        (st.ts for st in states.values()), default=pd.Timestamp(datetime.min)
    )
    assert isinstance(latest_ts, pd.Timestamp)
    for symbol in symbols:
        ivs = list(states)
        labels = [st.trend.get(symbol) for st in states.values()]
        long_align, short_align, divergent = weighted_align(
            labels, ivs, p.lower_tf_weight
        )
        align = max(long_align, short_align)
        direction: Action
        if align < p.alignment_threshold:
            direction = "flat"
            signals = tuple(
                f"{iv} {st.trend.get(symbol) or 'NA'}" for iv, st in states.items()
            )
        else:
            direction = "long" if long_align >= short_align else "short"
            signals = tuple(
                f"{iv} {st.trend.get(symbol) or 'NA'}" for iv, st in states.items()
            ) + (("DIVERGENT",) if divergent else ())
        results.append(
            ScreenResult(
                symbol=symbol,
                timestamp=latest_ts,
                score=align,
                action=direction,
                signals=signals,
                model_features={
                    "long_align": long_align,
                    "short_align": short_align,
                    "divergent": 1.0 if divergent else 0.0,
                },
            )
        )
    ranked = tuple(sorted(results, key=lambda r: r.score, reverse=True))
    return ranked[:top] if top is not None else ranked


def build_state(
    ts: pd.Timestamp,
    frames: tuple[tuple[str, pd.DataFrame], ...],
    *,
    labels_by_symbol: dict[str, tuple[TrendRegime | None, VolRegime | None]]
    | None = None,
    trend_fast: int = 50,
    trend_slow: int = 200,
    range_threshold_pct: float = 0.005,
) -> ScreenState:
    """Build a ScreenState.

    ``labels_by_symbol`` (optional) lets a caller precompute (trend, vol) labels
    (e.g. from an external model). When absent, trend/vol are derived from the
    frames via the same SMA gates used by the momentum screen, so a caller that
    already ran a detector can inject labels and skip recomputation.
    """
    trend: dict[str, TrendRegime | None] = {}
    vol: dict[str, VolRegime | None] = {}
    for symbol, df in frames:
        if labels_by_symbol is not None and symbol in labels_by_symbol:
            t, v = labels_by_symbol[symbol]
            trend[symbol], vol[symbol] = t, v
            continue
        closes = df["close"] if ("close" in df.columns and not df.empty) else None
        trend[symbol] = (
            _trend_label(closes, trend_fast, trend_slow, range_threshold_pct)
            if closes is not None
            else None
        )
        vol[symbol] = _vol_label(closes) if closes is not None else None
    return ScreenState(
        ts=ts,
        frames=frames,
        trend=trend,
        vol=vol,
    )


def run_screen(
    state: ScreenState,
    screen_name: str,
    params: dict | None = None,
) -> tuple[ScreenResult, ...]:
    """Run one named screen over a state, returning ranked results."""
    mod = init_screen(screen_name)
    resolved = resolve_screen_params(screen_name, params or {})
    results = mod.on_state(state, resolved)
    return tuple(sorted(results, key=lambda r: r.score, reverse=True))


def rank(
    results: tuple[ScreenResult, ...], *, top: int | None = None
) -> tuple[ScreenResult, ...]:
    """Return results sorted by score desc, optionally capped to ``top``."""
    ranked = tuple(sorted(results, key=lambda r: r.score, reverse=True))
    if top is not None:
        return ranked[:top]
    return ranked


def screen_over_history(
    frames: tuple[tuple[str, pd.DataFrame], ...],
    screen_name: str,
    params: dict | None = None,
) -> dict[pd.Timestamp, tuple[ScreenResult, ...]]:
    """Run a screen at every timestamp, cursor-safe (no look-ahead).

    For each timestamp, each symbol's frame is truncated to rows <= ts before
    scoring, mirroring the backtest CandleStore cursor so the screen sees the
    same data the strategy saw. Returns a map of ts -> ranked results.

    A screen that fires where the matching backtest strategy was flat is a
    look-ahead symptom; this walk is the honest reconciliation surface.
    """
    if not frames:
        return {}
    # Union of all timestamps, ascending.
    timestamps = sorted(set().union(*[set(f.index) for _, f in frames if not f.empty]))
    out: dict[pd.Timestamp, tuple[ScreenResult, ...]] = {}
    for ts in timestamps:
        truncated = tuple((sym, f[f.index <= ts]) for sym, f in frames if not f.empty)
        state = build_state(ts, truncated)
        out[ts] = run_screen(state, screen_name, params)
    return out
