"""Fundamentals pair — dollar-neutral long/short inside pre-declared sector pairs.

A market-neutral cross-section: the universe is a list of *pairs* (long leg +
short leg) that sit in the same sector, and the strategy is long the
fundamentally stronger leg and short the weaker one of **each** pair. Because
every position is matched by a same-sector opposite leg of equal dollar size,
portfolio beta is ~0 by construction rather than by a fitted hedge ratio — the
edge is the *ranking inside a pair*, not a market call.

Signal (per pair, fully cross-sectional — no per-symbol independent trades):

  1. Read each leg's as-first-stated fiscal series through ``ctx.fundamentals``
     (cursor-safe: a filing published after the current bar is invisible, so
     this cannot look ahead).
  2. Restrict to **annual (10-K) periods** and take the most recent
     ``lookahead_years + 1`` of them. SEC reports cumulative-YTD facts, so a
     4th-quarter 10-Q spans 12 months while a 1st-quarter 10-Q spans 3; mixing
     them compares a year to a quarter. The annual leg is the only place a
     period-over-period comparison is like-for-like, so the filter is a
     correctness requirement, not a convenience.
  3. Score each leg on the slope of a chosen fundamental over those years,
     normalised by the leg's own scale (``equity`` for a level, ``equity`` for a
     flow — return-on-equity style), so a mega-cap and a mid-cap are comparable.
  4. Long the higher-scoring leg, short the lower. Equal dollar size per leg,
     re-checked every ``hold_days`` so the pair can flip when the ranking flips.

Deliberately *not* here: a fitted hedge ratio, a market/beta model, or a
sector-ETF short. Sector ETFs file no 10-K/10-Q at all, so they have no
fundamentals to read and cannot participate in a fundamentals ranking — pairing
stock-against-stock is what keeps every leg measurable by the same signal.

Params
------
``metric``      which fundamental slopes feed the score
                (``"net_income"`` | ``"equity"`` | ``"operating_cash_flow"``).
``lookahead_years``  annual periods of history required before a pair is traded.
``hold_days``   bars between re-ranking (a flip is a full close + reopen).
``size``        fraction of capital per leg (0..1).
``long_pairs``  pairs traded as declared; ``short_pairs`` are inverted (use to
                separate "the ranking works" from "the ranking works only long").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from src.bt.strategies.dsl import StrategyContext, strategy
from src.bt.strategies.fundamentals_context import Fundamentals, SeriesPIT
from src.bt.strategies.types import StrategyParams

STRATEGY_TYPE = "fundamentals_pair_dsl"

#: Minimum days in a period to count as annual. A 10-K spans ~365 days while a
#: quarterly 10-Q spans ~90 (or ~270 for a cumulative 9-month YTD fact), so the
#: floor separates a year from any quarter without depending on the form string.
_ANNUAL_MIN_DAYS = 300

#: ``metric`` -> (statement accessor, field) read from ``ctx.fundamentals``.
#:
#: Every entry must be a **flow or a per-leg amount**, never ``equity``: equity
#: is the score's own denominator (see :func:`_score`), so scoring equity/equity
#: would pin every leg to ~1.0, tie every pair, and silently open nothing.
_METRIC_FIELD: dict[str, tuple[str, str]] = {
    "net_income": ("income", "net_income"),
    "operating_cash_flow": ("cashflow", "operating_cash_flow"),
    "revenue": ("income", "revenue"),
    "free_cash_flow": ("cashflow", "free_cash_flow"),
}

Metric = Literal["net_income", "operating_cash_flow", "revenue", "free_cash_flow"]


@dataclass(frozen=True)
class Params(StrategyParams):
    """Typed strategy params (see the module docstring)."""

    #: Pairs as ``LONG:SHORT`` — the declared sector twin, long leg first.
    #: Declared ``list[str]`` rather than ``tuple[str, ...]``: JSON has no tuple,
    #: so a config supplies a list and ``from_dict`` passes it through verbatim.
    pairs: list[str] = field(default_factory=list)
    metric: str = "net_income"
    lookahead_years: int = 3
    hold_days: int = 21
    size: float = 0.05
    long_pairs: bool = True
    short_pairs: bool = True


def _annual_values(series: SeriesPIT, years: int) -> list[float]:
    """Most recent ``years`` *annual* values of ``series``, oldest first.

    Filters on span length rather than on ``forms()``: the form string tells the
    filer's report type, but the length of ``period_start -> period_end`` is what
    actually decides whether the number is a full year. A shorter series than
    requested yields fewer entries, which the caller treats as "not tradable".
    """
    spans = series.spans()
    values = series.values()
    # ``SeriesPIT`` exposes values and spans as two parallel axes (spans are
    # needed for the length filter below). Materialising the pairing first keeps
    # the filter readable and lets the equal-length invariant be asserted once.
    paired = list(zip(values, spans))
    assert len(paired) == len(values), "values and spans must share a length"
    annual = [
        float(value)
        for value, (start, end) in paired
        if (end - start).days >= _ANNUAL_MIN_DAYS
    ]
    return annual[-years:]


def _scale(series: SeriesPIT) -> float | None:
    """Latest balance-sheet ``equity`` as the normaliser for a leg's metric.

    Levels and flows both divide by equity, giving a return-on-equity-shaped
    number that is comparable across a mega-cap and a mid-cap. ``None`` when the
    leg has no equity reading, which makes the leg unscoreable (never zero —
    a missing fundamental is not a weak fundamental).
    """
    value = series.last()
    if value is None or value <= 0:
        return None
    return float(value)


def _score(series: SeriesPIT, years: int, equity: float) -> float | None:
    """Fundamental trend score: equity-normalised latest value, plus its drift.

    Two components, both scaled by equity so they are unit-free:

    * **level** — latest annual value / equity. A leg earning more per unit of
      book scores higher (profitability, not size).
    * **drift** — (latest - oldest) / equity over the same window. Rewards a leg
      whose fundamentals are *improving*, which is the part a static value screen
      misses.

    ``None`` when the window is too short (fewer than 2 annual points) or the
    series is degenerate, so the caller can skip the pair instead of inventing a
    neutral score.
    """
    values = _annual_values(series, years)
    if len(values) < 2 or equity <= 0:
        return None
    drift = (values[-1] - values[0]) / equity
    level = values[-1] / equity
    return level + drift


def _leg_score(
    fundamentals: Fundamentals, symbol: str, metric: str, years: int
) -> float | None:
    """Score one leg of a pair, or None when it is not scoreable yet.

    Reads through the *bound* ``Fundamentals`` store, so every series is
    truncated to filings the strategy has already seen. An unregistered metric
    is a *config* error rather than a runtime condition that resolves later, so
    it raises here from a cold start instead of KeyError-ing on some later bar.
    """
    if metric not in _METRIC_FIELD:
        raise ValueError(
            f"unknown metric {metric!r}; pick one of {', '.join(sorted(_METRIC_FIELD))}"
        )
    statement, field = _METRIC_FIELD[metric]
    accessor = getattr(fundamentals, statement)(symbol)
    series: SeriesPIT = getattr(accessor, field)
    equity_series: SeriesPIT = fundamentals.balance(symbol).equity
    equity = _scale(equity_series)
    if equity is None:
        return None
    return _score(series, years, equity)


def _pair_legs(params: Params) -> list[tuple[str, str]]:
    """``[(long, short)]`` from ``params.pairs`` (``"NVDA:LRCX"`` strings).

    ``short_pairs=False`` inverts the declared leg order, letting a run test
    whether the ranking carries information on the short side or only long.
    """
    legs: list[tuple[str, str]] = []
    for pair in params.pairs:
        left, right = pair.split(":")
        legs.append((left, right) if params.short_pairs else (right, left))
    return legs


@strategy(bars="1d", stateful=True)
def on_candle(ctx: StrategyContext) -> None:
    """Re-rank every pair on a fixed cadence; hold the resulting long/short book.

    Fires on the last symbol's bar each timestamp, so all 20 legs are present in
    the store and a cross-sectional read is complete. Ranking runs only every
    ``hold_days`` bars (a fundamentals signal does not change daily, and
    re-ranking per bar would trade noise plus commission); between re-ranks the
    existing book is held untouched.
    """
    params: Params = ctx.params
    bars = int(ctx.shared.setdefault("bars", 0))
    ctx.shared["bars"] = bars + 1

    legs = _pair_legs(params)
    if bars % max(params.hold_days, 1) != 0:
        return

    # Warmup: a filing-based signal has nothing to say before the first 10-Ks
    # land, and `ctx.fundamentals` is only populated once the engine cursor has
    # advanced. Skipping is correct; guessing a direction would not be.
    desired_long: set[str] = set()
    desired_short: set[str] = set()
    for left, right in legs:
        left_score = _leg_score(
            ctx.fundamentals, left, params.metric, params.lookahead_years
        )
        right_score = _leg_score(
            ctx.fundamentals, right, params.metric, params.lookahead_years
        )
        if left_score is None or right_score is None or left_score == right_score:
            continue
        winner, loser = (left, right) if left_score > right_score else (right, left)
        desired_long.add(winner)
        desired_short.add(loser)

    if not desired_long and not desired_short:
        return

    long_book = desired_long if params.long_pairs else set()
    short_book = desired_short if params.short_pairs else set()

    # Flatten anything the new ranking no longer wants, then open the flip legs.
    # ``quantity`` is the *signed net* across lots, so the sign test below is a
    # real direction check; a pair that flips must close before reopening, and
    # the engine drains a close ahead of a same-bar open for that symbol.
    for symbol in ctx.symbols:
        if symbol in long_book or symbol in short_book:
            continue
        if ctx.quantity(symbol) != 0:
            ctx.close(symbol, reason=f"{params.metric} rank flip")

    for symbol in sorted(long_book):
        net = ctx.quantity(symbol)
        if net < 0:
            ctx.close(symbol, reason="flip to long")
            net = 0.0
        if net == 0:
            ctx.long(
                symbol,
                size=params.size,
                size_mode="equity",
                reason=f"{params.metric} top of pair",
            )

    for symbol in sorted(short_book):
        net = ctx.quantity(symbol)
        if net > 0:
            ctx.close(symbol, reason="flip to short")
            net = 0.0
        if net == 0:
            ctx.short(
                symbol,
                size=params.size,
                size_mode="equity",
                reason=f"{params.metric} bottom of pair",
            )
