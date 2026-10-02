"""Warmup span parsing: a human calendar duration -> a bar count.

Two distinct units, never mixed:

- ``warmup`` (config string, e.g. ``"1y"``) — a **calendar** duration. It
  controls how much feed the engine walks before ``trading_start``.
- ``warmup_bars`` (engine/strategy integer) — how many **bars** that duration
  is worth at the config's base bar size. Strategies compare this against their
  own bar-count readiness gates (``warmup_bars: 60``-``150`` in the strategy
  params), so both speak the same unit.

Why the conversion is interval-aware: trading bars are not calendar days. A
1d bar is one *trading* day (~5/7 of a calendar day), a 1h bar is one trading
hour (~6.5/24 of a trading day). Counting calendar days as bars would
under-count and reintroduce the cold-warmup bug this module exists to
prevent, so the conversion is deliberately conservative in the
over-estimating direction — a slightly longer warmup is harmless, a short
one is the bug.

The strategy's OWN bar-count gate remains the thing that guarantees enough
accumulated state. ``warmup`` only decides how much feed is walked; it cannot
by itself make an accumulator "ready".
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Final, cast

import pandas as pd

# Trading days per calendar week (5 of 7). The only calendar constant here:
# exchanges are closed on weekends, so a 7-day span holds ~5 trading days.
TRADING_DAYS_PER_WEEK: Final[float] = 5.0 / 7.0

# Trading hours per trading day (regular US session, 13:00-19:30 UTC = 6.5h).
# Used only for intraday bar sizes.
TRADING_HOURS_PER_DAY: Final[float] = 6.5

# Bars per trading day for each supported base interval. An intraday bar size
# not listed falls back to ``_INTRADAY_HOURS`` parsing ("1h" -> 6.5 bars/day).
_DAILY_INTERVALS: Final[frozenset[str]] = frozenset({"1d", "1D", "d", "D"})

_DURATION_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>y|yr|yrs|year|years|"
    r"m|mo|mon|month|months|w|wk|week|weeks|d|day|days|h|hour|hours)\s*$",
    re.IGNORECASE,
)

# Calendar days per duration unit. Months use 31 (the longest month) so a
# month-span is over- rather than under-counted; a "6m" warmup must never be
# shorter than 6 calendar months of bars.
_UNIT_DAYS: Final[dict[str, float]] = {
    "y": 365.0,
    "yr": 365.0,
    "yrs": 365.0,
    "year": 365.0,
    "years": 365.0,
    "m": 31.0,
    "mo": 31.0,
    "mon": 31.0,
    "month": 31.0,
    "months": 31.0,
    "w": 7.0,
    "wk": 7.0,
    "week": 7.0,
    "weeks": 7.0,
    "d": 1.0,
    "day": 1.0,
    "days": 1.0,
    "h": 1.0 / 24.0,
    "hour": 1.0 / 24.0,
    "hours": 1.0 / 24.0,
}


class WarmupParseError(ValueError):
    """Raised when a ``warmup`` string is not a supported calendar duration."""


@dataclass(frozen=True, slots=True)
class WarmupSpan:
    """A parsed ``warmup`` config value, in both units.

    ``duration`` is the original human string (``"1y"``); ``days`` is the
    calendar span; ``bars`` is that span expressed at the config's base
    interval. ``parse_warmup`` returns this so a caller that needs the actual
    date subtraction (``warmup_load_start``) reads ``days``, while the engine
    and strategies read ``bars``.
    """

    duration: str
    days: float
    bars: int


def bars_per_trading_day(bar: str) -> float:
    """Bars per *trading* day for ``bar`` — the interval-aware conversion rate.

    ``"1d"`` -> 1.0 (one daily bar is one trading day). ``"1h"`` -> 6.5 (the
    nominal regular session; 1h is the native granularity and is not resampled).
    Coarser intraday sizes are resampled **session-anchored** (see
    ``src.data.resample``), so a day yields ``ceil(session_hours / hours)`` whole
    buckets — ``"4h"`` -> 2, not the nominal 6.5/4 = 1.625. Any other intraday
    size is parsed from its hour count, so an unsupported ``"7h"`` still
    converts rather than silently degrading to 1 bar/day.
    """
    if bar in _DAILY_INTERVALS:
        return 1.0
    hours = _parse_interval_hours(bar)
    if hours <= 1.0:
        return TRADING_HOURS_PER_DAY / hours
    return float(math.ceil(TRADING_HOURS_PER_DAY / hours))


def parse_warmup_bars(warmup: str, bar: str) -> int:
    """Calendar duration string -> bar count at the config's base interval.

    >>> parse_warmup_bars("1y", "1d")
    260
    >>> parse_warmup_bars("90d", "1d")
    64
    >>> parse_warmup_bars("1y", "1h")
    1690

    Deliberately over-estimates: calendar days are scaled by trading-days-per-
    week (5/7) and intraday spans by trading-hours-per-day, and months are
    taken as 31 days. Under-counting a warmup walks too little history, which
    is exactly the cold-start bug this replaces.
    """
    return parse_warmup(warmup, bar).bars


def parse_warmup(warmup: str, bar: str) -> WarmupSpan:
    """Parse a ``warmup`` config string into calendar days + bar count.

    Raises:
        WarmupParseError: when ``warmup`` is not a supported duration string,
            or ``bar`` is not a recognisable bar size.
    """
    days = parse_warmup_days(warmup)
    try:
        rate = bars_per_trading_day(bar)
    except ValueError as exc:
        raise WarmupParseError(
            f"unrecognised bar size {bar!r} for warmup {warmup!r}; expected an "
            "interval like '1d', '1h' or '4h'"
        ) from exc
    # Over-estimate to the next whole bar so a fractional result never
    # truncates below the requested span.
    return WarmupSpan(
        duration=warmup,
        days=days,
        bars=_ceil_bars(days, rate),
    )


def parse_warmup_days(warmup: str) -> float:
    """Calendar days a ``warmup`` duration string denotes.

    Weeks = 7 days, months = 31 days, years = 365 days, hours = 1/24 day.
    """
    match = _DURATION_RE.match(warmup)
    if match is None:
        raise WarmupParseError(
            f"invalid warmup {warmup!r}; expected a calendar duration string "
            "such as '90d', '6m' or '1y'"
        )
    value = float(match.group("value"))
    unit = match.group("unit").lower()
    return value * _UNIT_DAYS[unit]


def warmup_start(trading_start: pd.Timestamp, warmup: str) -> pd.Timestamp:
    """``trading_start`` minus the warmup span — the feed's head for a window.

    The single place the calendar subtraction happens, so no caller has to
    re-derive it (and the ``Timestamp | NaTType`` that pandas' arithmetic
    types as is narrowed here once, not at every call site).
    """
    return cast(
        pd.Timestamp, trading_start - pd.Timedelta(days=parse_warmup_days(warmup))
    )


def _ceil_bars(days: float, bars_per_day: float) -> int:
    """Trading days in ``days`` -> whole bars, rounded up (never shorten)."""
    trading_days = days * TRADING_DAYS_PER_WEEK
    return int(-(-(trading_days * bars_per_day) // 1))


def _parse_interval_hours(bar: str) -> float:
    """Hours per bar for an intraday interval string (``"1h"`` -> 1.0)."""
    match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*(h|hour|hours|m|min|minutes)\s*$", bar)
    if match is None:
        raise ValueError(f"unrecognised bar size {bar!r}")
    value = float(match.group(1))
    if value <= 0:
        raise ValueError(f"bar size {bar!r} must be > 0")
    unit = match.group(2).lower()
    return value if unit.startswith("h") else value / 60.0


__all__ = [
    "WarmupParseError",
    "WarmupSpan",
    "TRADING_DAYS_PER_WEEK",
    "TRADING_HOURS_PER_DAY",
    "bars_per_trading_day",
    "parse_warmup",
    "parse_warmup_bars",
    "parse_warmup_days",
    "warmup_start",
]
