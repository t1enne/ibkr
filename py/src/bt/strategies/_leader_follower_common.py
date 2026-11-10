"""Shared signal layer for the ``leader_follower_*`` family.

The estimator, in order: vol-normalized impulse per member -> leader election
(largest signed impulse) -> self-relative impulse gate -> beta-scaled catch-up
gap per follower -> FCF-sign quality gate -> VIX regime. The strategy module
reads these helpers and owns only the inventory/exit layer, so any metric is
attributable to that layer and not to a different signal.

Cross-call state lives in ``Book`` under ``ctx.shared`` (minted fresh per
run/window by the engine), never at module scope.

Lookahead-free: every read is ``ctx.ohlcv`` / ``ctx.macro.vix``, which the
engine cursor-truncates at the signal bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np

from src.bt.strategies.dsl import StrategyContext
from src.bt.strategies.types import StrategyParams

# -- structural constants (define the estimator; not params) ----------------
VOL_WINDOW = 20  # bars for the 1-bar sigma that normalizes impulse
BETA_WINDOW = 60  # bars for a follower's beta to the leader
RANK_WINDOW = 250  # trailing bars in the leader's own impulse percentile
HISTORY_MIN = 260  # bars a member needs before it may rank
MIN_BETA = 0.20  # a follower must share the leader's factor at all
ATR_PERIOD = 14
VIX_WINDOW = 250  # trailing macro rows in the VIX regime rank
FCF_PERIODS = 4  # visible filings whose signs set a name's quality side
SL_ATR = 2.5  # entry stop, in entry-bar ATRs


@dataclass(frozen=True)
class BaseParams(StrategyParams):
    """Signal-layer knobs every family member reads."""

    impulse_bars: int = 5  # horizon of the impulse and the catch-up gap
    top_n: int = 4  # concurrent followers per side
    gate_rank: float = 0.80  # leader |t| must beat this percentile of its history
    vix_rank: float = 0.80  # VIX rank at/above which the book turns short
    size: float = 0.06  # 0..1 live-equity fraction per name
    hold_bars: int = 8  # hard time exit


@dataclass
class Lot:
    """One follower leg's inventory state."""

    side: str = "long"
    entered_bar: int = 0
    atr: float = 0.0
    stop: float = 0.0  # close-checked stop level (0 = unarmed)
    harvested: bool = False


@dataclass
class Book:
    """Per-run bookkeeping held in ``ctx.shared``."""

    bar: int = 0
    prev_leader: str = ""  # leader-election persistence check
    lots: dict[str, Lot] = field(default_factory=dict)
    closed: set[str] = field(default_factory=set)  # closed on THIS bar


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def log_returns(closes: np.ndarray) -> np.ndarray:
    """Bar log-returns of a close array (length n-1)."""
    if closes.size < 2:
        return np.empty(0, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(closes[1:] / closes[:-1])
    return np.where(np.isfinite(r), r, 0.0)


def impulse_t(closes: np.ndarray, k: int) -> float:
    """Vol-normalized ``k``-bar impulse: t-stat of the window's mean return."""
    if k < 1 or closes.size <= k + VOL_WINDOW:
        return float("nan")
    r = log_returns(closes)
    sigma = float(np.std(r[-VOL_WINDOW:]))
    if not np.isfinite(sigma) or sigma <= 0:
        return float("nan")
    return float(np.sum(r[-k:])) / (sigma * np.sqrt(k))


def impulse_t_series(closes: np.ndarray, k: int) -> np.ndarray:
    """Rolling ``impulse_t`` over the visible series (self-relative gate)."""
    n = closes.size
    out = np.full(n, np.nan, dtype=np.float64)
    if k < 1 or n < VOL_WINDOW + k + 2:
        return out
    r = log_returns(closes)
    csum = np.concatenate(([0.0], np.cumsum(r)))
    csq = np.concatenate(([0.0], np.cumsum(r * r)))
    for i in range(VOL_WINDOW + k, n):
        m = VOL_WINDOW
        s1 = csum[i] - csum[i - m]
        s2 = csq[i] - csq[i - m]
        var = s2 / m - (s1 / m) ** 2
        if not np.isfinite(var) or var <= 0:
            continue
        out[i] = (csum[i] - csum[i - k]) / (np.sqrt(var) * np.sqrt(k))
    return out


def trailing_percentile(values: np.ndarray, q: float) -> float:
    """``q``-quantile of the finite tail of ``values`` (NaN if too short)."""
    finite = values[np.isfinite(values)]
    if finite.size < RANK_WINDOW // 2:
        return float("nan")
    return float(np.quantile(np.abs(finite[-RANK_WINDOW:]), q))


def gate_threshold(closes: np.ndarray, k: int, gate_rank: float) -> float:
    """The name's own trailing ``gate_rank`` percentile of |impulse|."""
    return trailing_percentile(impulse_t_series(closes, k), gate_rank)


def beta(follower_r: np.ndarray, leader_r: np.ndarray) -> float:
    """OLS slope of follower on leader over the last ``BETA_WINDOW`` returns."""
    n = min(follower_r.size, leader_r.size, BETA_WINDOW)
    if n < 20:
        return float("nan")
    lead = leader_r[-n:]
    var = float(np.var(lead))
    if not np.isfinite(var) or var <= 0:
        return float("nan")
    return float(np.cov(follower_r[-n:], lead, bias=True)[0, 1] / var)


# ---------------------------------------------------------------------------
# panel access (cursor-truncated)
# ---------------------------------------------------------------------------


def closes_of(ctx: StrategyContext, sym: str) -> np.ndarray | None:
    try:
        return ctx.ohlcv(sym).close.to_array()
    except KeyError:
        return None


def panel(ctx: StrategyContext, min_bars: int) -> dict[str, np.ndarray]:
    """Closes per tradeable member with at least ``min_bars`` visible bars."""
    out: dict[str, np.ndarray] = {}
    for sym in ctx.symbols:
        c = closes_of(ctx, sym)
        if c is None or c.size < min_bars:
            continue
        out[sym] = c
    return out


def leader_rank(panel: dict[str, np.ndarray], k: int) -> tuple[str, float] | None:
    """``(symbol, impulse_t)`` of the bar's largest signed ``k``-bar impulse."""
    best: tuple[str, float] | None = None
    for sym, closes in panel.items():
        t = impulse_t(closes, k)
        if not np.isfinite(t):
            continue
        if best is None or t > best[1]:
            best = (sym, t)
    return best


def catch_up_gaps(
    panel: dict[str, np.ndarray], leader: str, leader_t: float, k: int
) -> dict[str, float]:
    """``beta*leader_ret - own_ret`` per follower (positive = lagging)."""
    leader_r = log_returns(panel[leader])
    if leader_r.size < VOL_WINDOW + k:
        return {}
    sigma = float(np.std(leader_r[-VOL_WINDOW:]))
    if not np.isfinite(sigma) or sigma <= 0:
        return {}
    leader_ret = leader_t * sigma * np.sqrt(k)

    gaps: dict[str, float] = {}
    for sym, closes in panel.items():
        if sym == leader:
            continue
        own_r = log_returns(closes)
        if own_r.size < k + 1:
            continue
        b = beta(own_r, leader_r)
        if not np.isfinite(b) or b < MIN_BETA:
            continue
        gap = b * leader_ret - float(np.sum(own_r[-k:]))
        if np.isfinite(gap):
            gaps[sym] = gap
    return gaps


# ---------------------------------------------------------------------------
# regime and quality gates
# ---------------------------------------------------------------------------


def vix_stress(ctx: StrategyContext, vix_rank: float) -> bool | None:
    """``None`` when the VIX series is too short; else high rank AND rising."""
    vix = ctx.macro.vix
    try:
        rank = vix.rank(VIX_WINDOW)
        now: float = cast(float, vix[-1])
        prev: float = cast(float, vix[-2])
        rising = now >= prev
    except IndexError:
        return None
    return bool(rank >= vix_rank and rising)


def fcf_positive(ctx: StrategyContext, sym: str) -> bool | None:
    """FCF-sign majority over the last ``FCF_PERIODS`` filings (``None`` = tied)."""
    try:
        series = ctx.fundamentals.cashflow(sym).free_cash_flow
    except KeyError:
        return None
    n = len(series)
    if n < 1:
        return None
    vals = cast(list[float], series[-min(FCF_PERIODS, n) :])
    pos = sum(1 for v in vals if v > 0)
    neg = sum(1 for v in vals if v < 0)
    if pos == neg:
        return None
    return pos > neg
