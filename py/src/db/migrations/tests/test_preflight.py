"""The preflight's refusals, on rows alone.

``symbol`` data is not guaranteed clean, so the checks are pure functions over
rows with the venue vocabulary INJECTED: the migration passes its own frozen
copy while the operator-facing report passes the live map
(:func:`src.data.venue.try_series_key`). Both run through this one algorithm, and
each refusal is asserted with the exact row it must name.
"""

from __future__ import annotations

from src.data.venue import try_series_key
from src.db.migrations.preflight import Listing, inspect


def test_a_clean_table_reports_nothing():
    rows = (
        Listing(1, "AAPL", "SMART"),
        Listing(2, "ENI.MI", "BVME"),
        Listing(3, "EA.MX", "MEXI"),
        Listing(4, "GATE_VIX", "GATE"),
    )

    assert inspect(rows, try_series_key).ok


def test_a_listing_stored_under_a_name_its_venue_does_not_derive_is_flagged():
    report = inspect((Listing(7, "EA", "MEXI"),), try_series_key)

    assert not report.ok
    (finding,) = report.miskeyed
    assert (finding.conid, finding.ticker, finding.market) == (7, "EA", "MEXI")
    assert finding.derived == "EA.MX"


def test_two_listings_deriving_one_key_collide():
    report = inspect(
        (Listing(1, "ENI", "BVME"), Listing(2, "ENI.MI", "BVME")), try_series_key
    )

    (collision,) = report.collisions
    assert collision.key == "ENI.MI"
    assert collision.conids == (1, 2)


def test_an_unmapped_venue_is_reported_rather_than_derived():
    report = inspect((Listing(5, "SHOP", "TSX"),), try_series_key)

    assert not report.ok
    (finding,) = report.unmapped
    assert (finding.conid, finding.ticker, finding.market) == (5, "SHOP", "TSX")


def test_every_finding_is_reported_on_one_actionable_line():
    report = inspect(
        (
            Listing(1, "EA", "MEXI"),
            Listing(2, "SHOP", "TSX"),
            Listing(3, "AAPL", "SMART"),
            Listing(4, "AAPL", "SMART"),
        ),
        try_series_key,
    )

    lines = report.lines()

    assert len(lines) == 3
    assert any("unmapped venue TSX: conid 2 SHOP" in line for line in lines)
    assert any("mis-keyed" in line and "EA.MX" in line for line in lines)
    assert any("key collision AAPL" in line and "3, 4" in line for line in lines)
