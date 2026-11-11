"""Series keys and the venue vocabulary.

The key rule is the whole multi-venue model in one function: a US or synthetic
listing stays bare, every other venue appends its suffix, and a venue with no
rule REFUSES rather than inventing one. The alias tests pin that the map is
deliberately non-invertible — a key must never be asked where it came from.
"""

from __future__ import annotations

import pytest

from src.data.ibkr.lookup import US_EXCHANGES
from src.data.venue import (
    MARKET_CALENDARS,
    SUFFIX_VENUES,
    US_VENUES,
    VENUE_SUFFIXES,
    UnmappedVenueError,
    bare_ticker,
    calendar_code,
    series_key,
    try_series_key,
    venue_for_key,
    venue_of,
)


def test_a_us_listing_keeps_its_ticker_bare():
    assert series_key("AAPL", "SMART") == "AAPL"
    assert series_key("AAPL", "NASDAQ") == "AAPL"
    assert series_key("AAPL", None) == "AAPL"


def test_a_synthetic_series_keeps_its_ticker_bare():
    assert series_key("GATE_VIX", "GATE") == "GATE_VIX"
    assert series_key("JJC", "VALUE") == "JJC"


def test_a_foreign_listing_carries_its_venue_suffix():
    assert series_key("ENI", "BVME") == "ENI.MI"
    assert series_key("AI", "SBF") == "AI.PA"
    assert series_key("SAP", "IBIS") == "SAP.DE"


def test_deriving_a_key_twice_changes_nothing():
    # Resolution re-derives the key of a ticker a config already suffixed; a
    # second suffix would create a third series for one listing.
    for ticker, venue in (("ENI.MI", "BVME"), ("AI.PA", "SBF"), ("AAPL", "SMART")):
        assert series_key(series_key(ticker, venue), venue) == series_key(ticker, venue)


def test_a_suffixed_ticker_overrides_the_universe_venue():
    # A pan-European universe is one plain list: the suffix names the listing.
    assert series_key("ENI.PA", "BVME") == "ENI.PA"
    assert series_key("AI.PA", "BVME") == "AI.PA"


def test_an_unmapped_venue_is_refused_not_guessed():
    with pytest.raises(UnmappedVenueError, match="TSX"):
        series_key("SHOP", "TSX")
    assert try_series_key("SHOP", "TSX") is None


def test_an_empty_ticker_is_refused():
    with pytest.raises(ValueError, match="ticker"):
        series_key("   ", "SMART")


def test_venue_aliases_share_a_suffix_so_the_key_cannot_name_the_venue():
    assert series_key("ENI", "BVME") == series_key("ENI", "MIL") == "ENI.MI"
    assert venue_of("ENI.MI") == "MI"
    assert venue_of("ENI.MI") == venue_of(series_key("ENI", "MIL"))


def test_a_us_dotted_ticker_is_not_a_suffixed_key():
    assert series_key("BRK.B", "SMART") == "BRK.B"
    assert venue_of("BRK.B") is None
    assert bare_ticker("BRK.B") == "BRK.B"


def test_bare_ticker_drops_only_a_known_suffix():
    assert bare_ticker("ENI.MI") == "ENI"
    assert bare_ticker("AI.PA") == "AI"
    assert bare_ticker("ENI") == "ENI"


def test_every_suffixed_venue_has_its_own_calendar():
    # A suffixed series measured against New York reports its own holidays as
    # data gaps, so a venue with a key rule must also name a calendar.
    assert set(VENUE_SUFFIXES) <= set(MARKET_CALENDARS)


def test_the_bare_set_covers_every_venue_the_gateway_calls_us():
    # A venue the secdef search can return as US must derive a bare key, or the
    # write path would reject a contract the lookup just resolved.
    assert US_EXCHANGES <= US_VENUES


def test_calendars_follow_the_venue_not_the_ticker():
    assert calendar_code("BVME") == "XMIL"
    assert calendar_code("MEXI") == "XMEX"
    assert calendar_code("bvme") == "XMIL"
    assert calendar_code(None) == "XNYS"


def test_the_canonical_inverse_covers_every_suffix_it_can_produce():
    # Resolution asks ONE venue per suffix, so a suffix with no canonical spelling
    # would silently fall back to the universe's exchange (a US search for an
    # Italian key). Every producible suffix must name a venue that derives it.
    assert set(SUFFIX_VENUES) == set(VENUE_SUFFIXES.values())
    for key_suffix, venue in SUFFIX_VENUES.items():
        assert VENUE_SUFFIXES[venue] == key_suffix


def test_a_suffixed_key_resolves_against_its_suffix_venue():
    # The mixed-venue rule: the suffix picks the venue to ASK, the universe
    # exchange only answers for a key that names no venue of its own.
    assert venue_for_key("ENI.PA", "BVME") == "SBF"
    assert venue_for_key("SAP.DE", "BVME") == "IBIS"
    assert venue_for_key("ENI.MI", "SMART") == "BVME"


def test_an_unsuffixed_key_resolves_against_the_universe_venue():
    assert venue_for_key("ENI", "BVME") == "BVME"
    assert venue_for_key("AAPL", None) is None
    assert venue_for_key("AAPL", "SMART") == "SMART"


def test_a_us_dotted_ticker_is_not_read_as_a_venue():
    # BRK.B carries a dot but no venue suffix, so it keeps the US path.
    assert venue_for_key("BRK.B", "SMART") == "SMART"
    assert venue_for_key("BRK.B", None) is None
