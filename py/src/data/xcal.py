"""Trading calendars per market, via ``exchange_calendars``.

Lazy-loaded — costs nothing if unused. The calendar answers one question in this
repo: does this date owe bars? A holiday must not be reported as a data gap, and
a universe that spans venues needs each venue's own holiday set rather than the
New York one.

A market is named by IBKR's exchange code (the ``symbol.market`` column, or a
universe's ``exchange``). The mapping to ``exchange_calendars`` codes lives with
the rest of the venue vocabulary in :mod:`src.data.venue`, which this module
re-exports so existing imports keep working. An unmapped market falls back to
:data:`DEFAULT_CALENDAR` — the historical US behaviour, never a guess.
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import TYPE_CHECKING

from src.data.venue import DEFAULT_CALENDAR, MARKET_CALENDARS, calendar_code

if TYPE_CHECKING:
    from exchange_calendars import ExchangeCalendar

__all__ = [
    "DEFAULT_CALENDAR",
    "MARKET_CALENDARS",
    "calendar_code",
    "get_xcal",
    "is_non_trading_day",
]


@lru_cache(maxsize=None)
def get_xcal(market: str | None = None) -> ExchangeCalendar:
    """Return the cached calendar for *market*, loading it on first call.

    One calendar per market, so a mixed universe never pays a second load for a
    market it has already seen.
    """
    import exchange_calendars as xcals

    return xcals.get_calendar(calendar_code(market))


def is_non_trading_day(d: datetime, market: str | None = None) -> bool:
    """True when *d* is a weekend or a holiday on *market*'s calendar.

    Falls back to a weekday-only check for dates outside a calendar's range
    (``exchange_calendars`` typically covers a few years forward).
    """
    if d.weekday() >= 5:
        return True
    try:
        return not get_xcal(market).is_session(d.strftime("%Y-%m-%d"))
    except Exception:
        return False
