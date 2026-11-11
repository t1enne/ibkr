"""Venue vocabulary and series keys — one name for "which market is this?".

A **venue** is IBKR's exchange code (``BVME``, ``IBIS``, ``SMART``…). A **series
key** is the string a strategy names one price history by, and it is what
``symbol.ticker`` and ``candle.ticker`` hold. One rule derives it:

* a US or synthetic venue keeps the ticker bare (``AAPL``, ``GATE_VIX``);
* every other venue appends its suffix (``ENI.MI``).

An unmapped venue is a hard error, never an invented suffix: a guessed suffix
silently allocates a second series for the same listing, and the bars then hide
under the wrong calendar and currency.

The suffix map is NOT invertible — ``BVME`` and ``MIL`` both mean Milan, and a US
ticker may itself contain a dot (``BRK.B``). A key is therefore never parsed to
recover a venue it was STORED under: the ``symbol`` row is the authority on where
a series lists, and :func:`venue_of` exists for display only. The one job the
suffix does have is choosing which venue to ASK when a key is unresolved
(:func:`venue_for_key`), which is what lets one universe list several venues.

:data:`MARKET_CALENDARS` lives here because a calendar is chosen by the same
venue code; it is the one mapping from IBKR's vocabulary to
``exchange_calendars`` names.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

#: Venues whose series keys are BARE: the US listing set. ``SMART`` is IBKR's
#: routed US default and must be here — it is what most US rows record.
#: Deliberately a superset of :data:`src.data.ibkr.lookup.US_EXCHANGES`, so a
#: venue the gateway can return can always derive a key.
US_VENUES: frozenset[str] = frozenset(
    {
        "SMART",
        "XNYS",
        "NYSE",
        "NASDAQ",
        "ARCA",
        "AMEX",
        "BATS",
        "IEX",
        "BATY",
        "ARCAEDGE",
        "EDGEA",
        "NYSEAMERICAN",
        "NASDAQNM",
    }
)

#: Vendor-side synthetic series: bare keys, never routable to an exchange.
SYNTHETIC: frozenset[str] = frozenset({"GATE", "VALUE"})

#: Non-US venue -> series-key suffix. Non-injective on purpose: one exchange can
#: be named several ways (``BVME``/``MIL``), and several venues share a suffix
#: family (``IBIS``/``FWB`` -> ``DE``) because the key names a listing, not a
#: microstructure.
VENUE_SUFFIXES: Mapping[str, str] = MappingProxyType(
    {
        "BVME": "MI",
        "MIL": "MI",
        "SBF": "PA",
        "IBIS": "DE",
        "FWB": "DE",
        "AEB": "AS",
        "LSE": "L",
        "BM": "MC",
        "VSE": "VI",
        "EBS": "SW",
        "MEXI": "MX",
    }
)

#: Calendar used for an unknown market — the historical, US-only default.
DEFAULT_CALENDAR = "XNYS"

#: IBKR exchange code -> ``exchange_calendars`` code. ``SMART`` is IBKR's routed
#: US default and is what a US contract's own ``exchange`` field reports, so it
#: maps to New York rather than being treated as its own venue.
MARKET_CALENDARS: Mapping[str, str] = MappingProxyType(
    {
        "SMART": "XNYS",
        "XNYS": "XNYS",
        "NYSE": "XNYS",
        "NASDAQ": "XNYS",
        "ARCA": "XNYS",
        "AMEX": "XNYS",
        "BATS": "XNYS",
        "IEX": "XNYS",
        "BVME": "XMIL",
        "MIL": "XMIL",
        "IBIS": "XETR",
        "FWB": "XFRA",
        "SBF": "XPAR",
        "AEB": "XAMS",
        "EBS": "XSWX",
        "LSE": "XLON",
        "BM": "XMAD",
        "VSE": "XWBO",
        "MEXI": "XMEX",
    }
)

#: Every suffix this module can produce — the only evidence a key carries about
#: its venue, and ambiguous wherever two venues share a suffix.
_SUFFIXES: frozenset[str] = frozenset(VENUE_SUFFIXES.values())

#: Suffix -> the ONE venue that suffix canonically names, used to choose a venue
#: to RESOLVE a suffixed key against. :data:`VENUE_SUFFIXES` is non-injective
#: (``BVME``/``MIL`` both spell ``MI``), so this inverse picks one canonical
#: spelling rather than pretending the key names both. The pick is a resolution
#: default, not a fact recovered from the key: a stored ``symbol`` row still
#: overrides it (see :func:`venue_for_key`).
SUFFIX_VENUES: Mapping[str, str] = MappingProxyType(
    {
        "MI": "BVME",
        "PA": "SBF",
        "DE": "IBIS",
        "AS": "AEB",
        "L": "LSE",
        "MC": "BM",
        "VI": "VSE",
        "SW": "EBS",
        "MX": "MEXI",
    }
)


class UnmappedVenueError(ValueError):
    """A venue with no series-key rule.

    Raised at the write path and at universe validation rather than defaulting to
    a bare key: a listing recorded under the wrong key is invisible to every
    later check, because nothing downstream can tell it from a real one.
    """


def _code(venue: str | None) -> str:
    """A venue code as stored: trimmed and upper-case (empty when unset)."""
    return (venue or "").strip().upper()


def is_bare_venue(venue: str | None) -> bool:
    """True when *venue*'s series keys carry no suffix.

    An unset venue means the historical US path, so it is bare too.
    """
    code = _code(venue)
    return code == "" or code in US_VENUES or code in SYNTHETIC


def suffix(venue: str | None) -> str:
    """The series-key suffix for a non-US venue.

    Raises :class:`UnmappedVenueError` when the venue has no rule yet — the
    operator's next move is to add it here, deliberately.
    """
    try:
        return VENUE_SUFFIXES[_code(venue)]
    except KeyError:
        raise UnmappedVenueError(
            f"no series-key suffix for venue {venue!r}: add it to "
            "src/data/venue.py VENUE_SUFFIXES before storing a listing there"
        ) from None


def series_key(ticker: str, venue: str | None) -> str:
    """The series key for *ticker* listed on *venue*.

    Idempotent: a ticker that already carries a KNOWN venue suffix is returned
    unchanged, so ``series_key("ENI.MI", "BVME")`` and
    ``series_key("ENI", "BVME")`` are one key, and a suffixed ticker overrides
    the venue a universe declared (that is what makes a pan-European universe a
    plain list). Both halves are upper-cased, which is how the DB stores them, so
    a key compares directly with ``symbol.ticker``.

    The venue is validated even when the ticker overrides it: an unmapped venue
    is a configuration error in either spelling.

    Raises :class:`UnmappedVenueError` for a non-US venue with no suffix rule and
    ``ValueError`` for an empty ticker.
    """
    upper = ticker.strip().upper()
    if not upper:
        raise ValueError("a series key needs a ticker")
    if is_bare_venue(venue):
        return upper
    suffix_for_venue = suffix(venue)  # validates the venue even if overridden below
    if venue_of(upper) is not None:
        return upper
    return f"{upper}.{suffix_for_venue}"


def try_series_key(ticker: str, venue: str | None) -> str | None:
    """*series_key*, or ``None`` when the venue is unmapped (a probe, not a write).

    The preflight uses this to REPORT an unmapped market instead of raising on the
    first row it meets.
    """
    try:
        return series_key(ticker, venue)
    except UnmappedVenueError:
        return None


def require_keyable(venue: str | None) -> None:
    """Raise unless *venue* can derive a series key.

    A caller checks this before I/O: an unmapped venue found after a gateway
    round-trip has already spent the request to learn a local configuration
    error.

    Raises:
        UnmappedVenueError: the venue has no series-key suffix rule.
    """
    if not is_bare_venue(venue):
        suffix(venue)


def venue_of(key: str) -> str | None:
    """DISPLAY-ONLY: the suffix part of a series key, or ``None`` without one.

    Never route, resolve or validate with this. The map is non-injective
    (``ENI.MI`` may have listed on ``BVME`` or ``MIL``) and a US ticker's dot is
    not a suffix (``BRK.B``), so a key genuinely does not say which venue it came
    from. Read the ``symbol`` row instead.
    """
    _, dot, tail = key.rpartition(".")
    return tail.upper() if dot and tail.upper() in _SUFFIXES else None


def bare_ticker(key: str) -> str:
    """*key* without a KNOWN venue suffix — the string a gateway search wants.

    Best effort in the same sense as :func:`venue_of`: a dot that is not one of
    this module's suffixes stays put, so a US name such as ``BRK.B`` survives.
    """
    head, dot, tail = key.rpartition(".")
    return head if dot and head and tail.upper() in _SUFFIXES else key


def venue_for_key(key: str, exchange: str | None = None) -> str | None:
    """The venue to RESOLVE *key* against: its suffix's venue, else *exchange*.

    This is the mixed-venue resolution rule. A key carrying a known suffix is
    resolved against the venue that suffix canonically names (``ENI.PA`` ->
    ``SBF``), so one universe may list ``ENI.MI`` and ``ENI.PA`` and each
    resolves its own listing. An unsuffixed key has no venue of its own, so it
    falls back to the universe's *exchange* (``None`` = the US path).

    A suffix only CHOOSES the question; it never answers it. The ``symbol`` row
    remains the authority on where a series lists, so an alias-stored listing
    (``MIL`` for ``ENI.MI``) is still found by its own row.

    Note this is the one place a key's suffix is allowed to pick a venue: it is
    a resolution default, not a claim about where the series came from.
    """
    tail = venue_of(key)
    if tail is None:
        return exchange
    return SUFFIX_VENUES[tail]


def calendar_code(market: str | None) -> str:
    """The ``exchange_calendars`` code for an IBKR market code (US by default)."""
    return MARKET_CALENDARS.get(_code(market), DEFAULT_CALENDAR)


__all__ = [
    "DEFAULT_CALENDAR",
    "MARKET_CALENDARS",
    "SUFFIX_VENUES",
    "SYNTHETIC",
    "US_VENUES",
    "VENUE_SUFFIXES",
    "UnmappedVenueError",
    "bare_ticker",
    "calendar_code",
    "is_bare_venue",
    "require_keyable",
    "series_key",
    "suffix",
    "try_series_key",
    "venue_for_key",
    "venue_of",
]
