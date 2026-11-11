"""Venue filtering of gateway secdef candidates.

The gateway serves two row shapes for the same stock, and only knowing one of
them silently drops every name-search hit — which is how a non-US market stops
being resolvable by ticker.
"""

from __future__ import annotations

from typing import Any, cast

from ib_rest_api_client.models import SecdefSearchResponseItem

from src.data.ibkr.lookup import listing_exchange, select_listings


def _item(**kwargs: Any) -> SecdefSearchResponseItem:
    return cast(SecdefSearchResponseItem, type("Item", (), kwargs)())


def test_symbol_search_row_names_the_venue_in_description():
    item = _item(description="BVME", company_header="ENI SPA - BVME")

    assert listing_exchange(item) == "BVME"


def test_name_search_row_names_the_venue_in_the_header_suffix():
    assert (
        listing_exchange(_item(description=None, company_header="ENI SPA (BVME)"))
        == "BVME"
    )
    assert (
        listing_exchange(_item(description=None, company_header="ENI SPA - BVME"))
        == "BVME"
    )


def test_a_row_with_no_venue_at_all_yields_none():
    assert listing_exchange(_item(description=None, company_header=None)) is None


def test_venue_filter_keeps_only_that_listing():
    us = _item(description="NYSE", company_header="GENERALI - NYSE", sections=None)
    italian = _item(description="BVME", company_header="GENERALI - BVME", sections=None)

    assert select_listings([us, italian], "bvme") == (italian,)


def test_venue_filter_is_empty_when_nothing_matches():
    us = _item(description="NYSE", company_header="GENERALI - NYSE", sections=None)

    assert select_listings([us], "BVME") == ()


def test_no_venue_keeps_the_historical_us_filter():
    us = _item(description="NYSE", company_header="APPLE INC - NYSE", sections=None)
    italian = _item(description="BVME", company_header="ENI SPA - BVME", sections=None)

    assert select_listings([us, italian]) == (us,)


def test_a_named_foreign_venue_is_never_taken_for_a_us_listing():
    # The gateway names these venues explicitly; only a venue-less row may fall
    # back to its sections. Otherwise a US lookup answers with a foreign listing.
    value = _item(
        description="VALUE", company_header="ENI SPA - VALUE", sections=["STK"]
    )

    assert select_listings([value]) == ()
