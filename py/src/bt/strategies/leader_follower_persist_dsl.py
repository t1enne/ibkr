"""Leader-follower catch-up on a sector panel, gap-proportional (DSL, 1d).

The economic claim: a member's dislocation leads its peers, and laggards catch
up. The signal layer is ``_leader_follower_common``'s (beta-scaled catch-up gap,
self-relative impulse gate, FCF-sign quality gate, VIX direction).

What this module adds on top of that shared layer:

* **Leader persistence** -- an impulse only becomes a signal on the SECOND
  consecutive bar the same name leads, so a one-bar spike elects nobody.
* **Gap-proportional sizing** -- a baseline gross is redistributed in proportion
  to each name's catch-up gap, letting the gap (not a flat fraction) decide how
  much each leg carries.
* **Rank amplification** (`amp`) -- each leg's weight is multiplied by
  `1 + amp * rank_score`, `rank_score` being the name's ordinal position in the
  RISK-NORMALISED gap ranking, `|gap| / (sigma * sqrt(k))` (top = 1, last = 0).
  `amp = 0` is the flat-gross book; `amp > 0` raises gross AND tilts it toward
  the names that owe the most catch-up per unit of their own volatility, so it
  doubles as the leverage dial of an otherwise under-invested book. Ranking on
  the normalised gap rather than raw |gap| is what keeps the tilt from simply
  preferring the loudest follower: at matched leverage it held return and cut
  drawdown against the raw-gap order.
* **Close-checked stop** -- the ``SL_ATR`` level is checked on the close and the
  lot closed explicitly, instead of an intrabar engine stop that could retire a
  lot on the same bar a clock exit names it.
* **Harvest, then clock** -- shed half the lot into ``harvest_atr`` of strength,
  and flatten on ``hold_bars`` if the catch-up has not arrived.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.bt.strategies._leader_follower_common import (
    ATR_PERIOD,
    HISTORY_MIN,
    SL_ATR,
    VOL_WINDOW,
    BaseParams,
    Book,
    Lot,
    catch_up_gaps,
    fcf_positive,
    gate_threshold,
    leader_rank,
    log_returns,
    panel,
    vix_stress,
)
from src.bt.strategies.dsl import StrategyContext, strategy

STRATEGY_TYPE = "leader_follower_persist_dsl"

_CTX_KEY = "leader_follower_persist_ctx"
_TAG = "lfq"
HARVEST_FRACTION = 0.5  # share of the lot shed into strength


@dataclass(frozen=True)
class Params(BaseParams):
    """Signal-layer knobs plus this module's own hold and harvest levels."""

    vix_rank: float = 0.70  # VIX rank at/above which the book turns short
    hold_bars: int = 12  # hard time exit
    harvest_atr: float = 2.5  # partial exit once up this many entry ATRs
    amp: float = 0.0  # rank amplification of weight (and of gross); 0 = flat


@strategy(stateful=True)
def on_candle(ctx: StrategyContext):
    if ctx.phase == "warmup":
        return
    params: Params = ctx.params
    book: Book = ctx.shared.setdefault(_CTX_KEY, Book())
    book.bar += 1
    book.closed.clear()

    k = params.impulse_bars
    closes = panel(ctx, HISTORY_MIN)
    sig = _signal(closes, k, params.gate_rank, book)

    _manage(ctx, book, params, sig)
    if sig is None:
        return

    side = _side(ctx, params, sig)
    if side is None:
        return
    picks = _select(ctx, sig.gaps, params.top_n, side)
    if not picks:
        return

    rows: list[tuple[str, float, float, float]] = []  # (sym, gap, atr, px)
    for sym, gap in picks:
        atr = ctx.ta.atr(sym, ATR_PERIOD)[-1]
        px = ctx.price(sym)
        if not np.isfinite(atr) or atr <= 0 or not np.isfinite(px):
            continue
        rows.append((sym, gap, float(atr), float(px)))
    if not rows:
        return
    for sym, gap, atr, px, weight in _weights(rows, params, closes, k):
        if abs(ctx.quantity(sym)) > 1e-6 or sym in book.closed:
            continue
        _open(ctx, book, sym, side, atr, px, weight, sig.leader, gap)


# ---------------------------------------------------------------------------
# signal
# ---------------------------------------------------------------------------


@dataclass
class _Signal:
    leader: str
    up_strong: bool
    down_strong: bool
    gaps: dict[str, float]  # positive = lagging (long candidates)


def _signal(
    closes: dict[str, np.ndarray], k: int, gate_rank: float, book: Book
) -> _Signal | None:
    """Persisted leader election -> gated impulse -> catch-up gaps.

    The elected name is remembered every bar, so persistence is measured
    against the previous bar's leader even when that bar produced no signal.
    """
    picked = leader_rank(closes, k)
    if picked is None:
        return None
    sym, t = picked
    repeat = sym == book.prev_leader
    book.prev_leader = sym
    if not repeat:
        return None
    thresh = gate_threshold(closes[sym], k, gate_rank)
    if not np.isfinite(thresh):
        return None
    return _Signal(
        leader=sym,
        up_strong=bool(t > thresh),
        down_strong=bool(-t > thresh),
        gaps=catch_up_gaps(closes, sym, t, k),
    )


def _side(ctx: StrategyContext, params: Params, sig: _Signal) -> str | None:
    """VIX trailing-rank regime decides long vs short; disagreement = no trade."""
    stress = vix_stress(ctx, params.vix_rank)
    if stress is None:
        return None
    if stress:
        return "short" if sig.down_strong else None
    return "long" if sig.up_strong else None


def _select(
    ctx: StrategyContext, gaps: dict[str, float], top_n: int, side: str
) -> list[tuple[str, float]]:
    """Gap-ranked names whose FCF-sign side agrees with ``side``."""
    want_pos = side == "long"
    ordered = sorted(
        (s for s in gaps if (gaps[s] > 0) == want_pos),
        key=gaps.__getitem__,
        reverse=want_pos,
    )
    picked: list[tuple[str, float]] = []
    for sym in ordered:
        ok = fcf_positive(ctx, sym)
        if ok is None or ok is not want_pos:
            continue
        picked.append((sym, gaps[sym]))
        if len(picked) == top_n:
            break
    return picked


def _weights(
    rows: list[tuple[str, float, float, float]],
    params: Params,
    closes: dict[str, np.ndarray],
    k: int,
) -> list[tuple[str, float, float, float, float]]:
    """``(sym, gap, atr, px, weight)``, gap-proportional gross times rank amplification.

    The base book holds gross at ``size * n`` split by |gap|, so the gap decides
    the mix; a bar whose gaps are all zero falls back to a flat ``size``. Each
    weight is then multiplied by ``1 + amp * rank_score``, ``rank_score`` being
    the leg's ordinal position in the risk-normalised gap ranking, so ``amp``
    both raises gross and shifts it toward the preferred names. ``amp = 0``
    leaves the base book untouched.
    """
    n = len(rows)
    total = sum(abs(gap) for _s, gap, _a, _p in rows)
    gross = params.size * n
    base = (
        [params.size] * n
        if total <= 0
        else [gross * abs(gap) / total for _s, gap, _a, _p in rows]
    )
    scores = _rank_scores(rows, closes, k)
    return [
        (sym, gap, atr, px, w * (1.0 + params.amp * score))
        for (sym, gap, atr, px), w, score in zip(rows, base, scores, strict=True)
    ]


# ---------------------------------------------------------------------------
# ranking
# ---------------------------------------------------------------------------


def _ordinal_scores(raw: list[float]) -> list[float]:
    """Ordinal rank in [0, 1], 1 = highest raw score; NaN sorts to the bottom."""
    n = len(raw)
    span = max(n - 1, 1)
    order = sorted(range(n), key=lambda i: (np.nan_to_num(raw[i], nan=-np.inf), i))
    scores = [0.0] * n
    for pos, i in enumerate(order):
        scores[i] = pos / span
    return scores


def _sigma(own_r: np.ndarray) -> float:
    """Trailing ``VOL_WINDOW`` daily sigma of a name's own log returns."""
    if own_r.size < VOL_WINDOW:
        return float("nan")
    return float(np.std(own_r[-VOL_WINDOW:]))


def _rank_scores(
    rows: list[tuple[str, float, float, float]],
    closes: dict[str, np.ndarray],
    k: int,
) -> list[float]:
    """Ordinal [0, 1] score per row from the RISK-NORMALISED catch-up gap.

    Raw score is ``|gap| / (sigma * sqrt(k))`` -- how many of the name's own
    ``k``-bar sigmas the catch-up still owes -- so a big gap in a quiet name
    outranks the same gap in a noisy one. Normalising by each name's own sigma
    is what makes the tilt a risk tilt rather than a volatility tilt: the
    dispersion gap already carries risk, and ranking on raw |gap| prefers the
    loudest names. Ordinal reduction keeps ``amp`` comparable for any ``n``
    (scores average 0.5), and every read is at the signal bar.
    """
    raw: list[float] = []
    for sym, gap, _atr, _px in rows:
        c = closes.get(sym)
        sig = _sigma(log_returns(c)) if c is not None else float("nan")
        raw.append(
            abs(gap) / (sig * np.sqrt(k))
            if np.isfinite(sig) and sig > 0
            else float("nan")
        )
    return _ordinal_scores(raw)


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------


def _open(
    ctx: StrategyContext,
    book: Book,
    sym: str,
    side: str,
    atr: float,
    px: float,
    weight: float,
    leader: str,
    gap: float,
) -> None:
    """Open one follower leg; the stop is a level, not an engine order."""
    if weight <= 0:
        return
    stop = px - SL_ATR * atr if side == "long" else px + SL_ATR * atr
    reason = f"[lfq] {side} {sym} vs {leader} gap={gap:+.3f}"
    if side == "long":
        ctx.long(sym, size=weight, tag=_TAG, reason=reason, size_mode="equity")
    else:
        ctx.short(sym, size=weight, tag=_TAG, reason=reason, size_mode="equity")
    book.lots[sym] = Lot(side=side, entered_bar=book.bar, atr=atr, stop=stop)


def _manage(
    ctx: StrategyContext, book: Book, params: Params, sig: _Signal | None
) -> None:
    """Invalidate, stop, harvest and clock every open follower leg."""
    up_strong = sig.up_strong if sig is not None else False
    down_strong = sig.down_strong if sig is not None else False
    leader = sig.leader if sig is not None else ""
    for sym in list(book.lots):
        lot = book.lots[sym]
        if abs(ctx.quantity(sym)) <= 1e-6:
            book.lots.pop(sym, None)
            continue
        px = ctx.price(sym)
        reversed_against = down_strong if lot.side == "long" else up_strong
        if reversed_against:
            ctx.close(sym, reason=f"[lfq] leader {leader} impulse reversed")
        elif _stop_hit(lot, px):
            ctx.close(sym, reason=f"[lfq] stop {SL_ATR:.1f}xATR")
        elif book.bar - lot.entered_bar >= params.hold_bars:
            ctx.close(sym, reason=f"[lfq] clock {params.hold_bars} bars")
        elif not lot.harvested and _in_profit(ctx, sym, lot, params.harvest_atr):
            ctx.partial_close(
                sym,
                HARVEST_FRACTION,
                tag=_TAG,
                reason=f"[lfq] harvest {HARVEST_FRACTION:.0%} into strength",
            )
            lot.harvested = True
            continue
        if abs(ctx.quantity(sym)) <= 1e-6:
            book.lots.pop(sym, None)
            book.closed.add(sym)


def _in_profit(ctx: StrategyContext, sym: str, lot: Lot, harvest_atr: float) -> bool:
    """Lot has gained ``harvest_atr`` entry ATRs from its average entry."""
    entry = ctx.avg_entry(sym)
    if entry is None:
        return False
    px = ctx.price(sym)
    edge = harvest_atr * lot.atr
    return px >= entry + edge if lot.side == "long" else px <= entry - edge


def _stop_hit(lot: Lot, px: float) -> bool:
    """Close-checked stop: one code path (close fills) owns lot lifecycles."""
    if lot.stop <= 0:
        return False
    return px <= lot.stop if lot.side == "long" else px >= lot.stop
