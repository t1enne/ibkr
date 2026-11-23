"""Cross-sectional long/short momentum over a caller-supplied symbol list.

Membership-agnostic by construction: the strategy ranks exactly the symbols in
``config.symbols`` and reads no holdings file, no date and no universe. The
caller decides which names the panel represents, so point-in-time membership
filtering belongs upstream in the config, never in here.

The book is two equal-weight legs of a trailing-return cross-section::

    long  = the ``k`` highest ``lookback_bars`` returns
    short = the ``k`` lowest  ``lookback_bars`` returns

re-formed every ``rebalance_bars`` bars and held in between, so cohorts never
overlap. At a rebalance every held name that is not already on the wanted side
is closed, then every wanted name that is not already on its side is opened
(``ctx.long`` / ``ctx.short``, absolute share counts). A name that merely flips
legs is closed and re-opened in the same decision: the engine settles closes
before opens within one cohort, so the flip is one bar's rotation, not two.

Readiness is per name, not per panel. A name ranks only when it has more than
``lookback_bars`` visible bars, a finite trailing return, and a newest visible
bar equal to the evaluation bar. A name failing any of those is simply absent
from the ranking -- never zero-filled, never a crash -- so a name that stops
printing leaves the book at the next rebalance instead of being held on a frozen
price. Fewer than ``2k`` rankable names cannot be split into two disjoint legs,
so that rebalance emits nothing at all and the current book is held untouched.

``size`` is a fraction of INITIAL capital per position, so gross exposure is
``2k * size``. This engine funds a short out of cash on the same mirror basis as
a long, so ``2k * size`` above one has the bar's cohort silently scaled down --
keep it under ``1 / (2k)`` and the book stays inside the capital base.

There are no stops and no take-profit: the rebalance IS the exit, which is what
keeps the cohorts non-overlapping. Cross-call state (the rebalance bar clock and
the last target book) lives in ``ctx.shared`` via ``@strategy(stateful=True)``;
holdings are read back from the portfolio (``ctx.quantity``) so the recorded
book can never gate a close.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.bt.strategies.dsl import StrategyContext, strategy
from src.bt.strategies.types import StrategyParams

STRATEGY_TYPE = "ark_ls_momentum_dsl"
_STATE_KEY = "ark_ls_momentum_state"

Side = Literal["long", "short"]


@dataclass(frozen=True)
class Params(StrategyParams):
    # -- signal: one trailing return, ranked cross-sectionally -----------------
    lookback_bars: int = 126  # trailing bars the momentum score is measured over
    # -- book: two equal legs of k names, re-formed every rebalance_bars bars --
    k: int = 3
    rebalance_bars: int = 21
    # -- sizing: fraction of initial capital per position (2k * size <= 1) -----
    size: float = 0.1


@dataclass(frozen=True)
class _State:
    """Cross-call state for one run/window (held in ``ctx.shared``)."""

    bars: int = 0  # trade-phase bars processed — the rebalance clock
    next_bar: int = 0  # bar index the next rebalance is due on
    book: tuple[str, ...] = ()  # symbols the last rebalance picked (diagnostics)


@strategy(stateful=True)
def on_candle(ctx: StrategyContext) -> None:
    # Warmup bars fill strategy state only — an entry emitted there could never
    # fill, so the DSL raises. The rebalance clock therefore starts at
    # ``trading_start``, which is also the first bar with a full lookback behind
    # it when ``warmup`` covers the lookback.
    if ctx.phase == "warmup":
        return
    p: Params = ctx.params
    assert p.lookback_bars >= 1, p.lookback_bars
    assert p.rebalance_bars >= 1, p.rebalance_bars
    assert p.k >= 0, p.k
    assert 0.0 < p.size <= 1.0, p.size

    state = _state(ctx)
    if state.bars >= state.next_bar:
        book = _rebalance(ctx, p, frozenset(state.book))
        state = dataclasses.replace(
            state,
            next_bar=state.bars + p.rebalance_bars,
            book=tuple(sorted(book)),
        )
    _save(ctx, dataclasses.replace(state, bars=state.bars + 1))


def _state(ctx: StrategyContext) -> _State:
    """The run's :class:`_State`, minted on first use."""
    holder = ctx.shared.setdefault(_STATE_KEY, _State())
    assert isinstance(holder, _State), type(holder).__name__
    return holder


def _save(ctx: StrategyContext, state: _State) -> None:
    """Write a derived :class:`_State` back to the per-run book."""
    ctx.shared[_STATE_KEY] = state


def _rebalance(
    ctx: StrategyContext, p: Params, previous: frozenset[str]
) -> dict[str, Side]:
    """Rotate the book onto this rebalance's target and return that target."""
    scores = _score_panel(ctx, p)
    book = _select(scores, p.k)
    # Positive AND negative space: the book is either empty (cross-section too
    # thin to split) or exactly two disjoint legs of k.
    assert len(book) <= 2 * p.k, len(book)
    if book:
        assert sum(1 for s in book.values() if s == "long") == p.k
        assert sum(1 for s in book.values() if s == "short") == p.k

    # (a) Close anything held on a side the book no longer wants — including a
    # name that left the ranking because its quote stopped advancing.
    for sym in ctx.symbols:
        held = _side(ctx.quantity(sym))
        if held is None or book.get(sym) == held:
            continue
        ctx.close(sym, reason=_close_reason(sym, scores.get(sym), previous))

    # (b) Open the target book. A held name that is already right is untouched,
    # so a rebuild pays friction on the rotation only.
    for sym in sorted(book):
        side = book[sym]
        if _side(ctx.quantity(sym)) == side:
            continue
        reason = f"[ark-ls] {side} {sym}: trailing return {scores[sym]:+.2%}"
        if side == "long":
            ctx.long(sym, size=p.size, reason=reason)
        else:
            ctx.short(sym, size=p.size, reason=reason)
    return book


def _score_panel(ctx: StrategyContext, p: Params) -> dict[str, float]:
    """Trailing return per rankable name on this bar.

    An unrankable name (no bar at the evaluation bar, too little history, a
    non-finite window) is absent from the map rather than present with a filler
    score, so it can neither be selected nor displace a rankable name.
    """
    scores: dict[str, float] = {}
    for sym in ctx.symbols:
        closes = _live_closes(ctx, sym)
        if closes is None:
            continue
        score = _tail_return(closes, p.lookback_bars)
        if np.isfinite(score):
            scores[sym] = score
    return scores


def _live_closes(ctx: StrategyContext, sym: str) -> np.ndarray | None:
    """Cursor-truncated closes for ``sym`` when its newest bar IS this bar.

    ``None`` when the name has no bars at all or its newest visible bar predates
    the evaluation bar. The staleness test is what stops a name whose feed went
    quiet from sitting in the book forever on a frozen price: it stops being
    rankable, so the next rebalance sells it.
    """
    frame = ctx.state.candles.get((sym, ctx.interval))
    if frame is None or len(frame.index) == 0:
        return None
    assert frame.index.is_monotonic_increasing
    if frame.index[-1] != ctx.candle.timestamp:
        return None
    return frame["close"].to_numpy(dtype=float)


def _tail_return(closes: np.ndarray, lookback_bars: int) -> float:
    """``close[-1] / close[-1 - lookback_bars] - 1``; NaN when unmeasurable."""
    assert lookback_bars >= 1, lookback_bars
    if closes.size <= lookback_bars:
        return float("nan")
    now = float(closes[-1])
    then = float(closes[-1 - lookback_bars])
    if not (np.isfinite(now) and np.isfinite(then)):
        return float("nan")
    if now <= 0.0 or then <= 0.0:
        return float("nan")
    return now / then - 1.0


def _select(scores: dict[str, float], k: int) -> dict[str, Side]:
    """Top-``k`` long, bottom-``k`` short; ``{}`` when the split is impossible.

    Deterministic: strongest score first with a symbol tie-break, so the same
    cross-section always yields the same book.
    """
    assert k >= 0, k
    if k == 0 or len(scores) < 2 * k:
        return {}
    ranked = sorted(scores, key=lambda s: (-scores[s], s))
    assert len(ranked) >= 2 * k
    book: dict[str, Side] = {}
    for sym in ranked[:k]:
        book[sym] = "long"
    for sym in ranked[-k:]:
        assert sym not in book, sym
        book[sym] = "short"
    return book


def _side(qty: float) -> Side | None:
    """Signed net quantity -> the side it represents; ``None`` when flat."""
    if qty > 0.0:
        return "long"
    if qty < 0.0:
        return "short"
    return None


def _close_reason(sym: str, score: float | None, previous: frozenset[str]) -> str:
    """Rotation reason for the trade log (never a gate — the portfolio decides)."""
    tag = "rotated out" if sym in previous else "not in book"
    where = "not ranked" if score is None else f"{score:+.2%}"
    return f"[ark-ls] {tag} {sym}: trailing return {where}"
