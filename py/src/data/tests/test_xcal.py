"""Calendar selection per market.

The regression these pin: an Italian listing measured against the New York
calendar reports every Italian holiday as a data gap.
"""

from __future__ import annotations

from datetime import datetime

from src.data.xcal import calendar_code, is_non_trading_day


def test_known_market_maps_to_its_calendar():
    assert calendar_code("BVME") == "XMIL"
    assert calendar_code("bvme") == "XMIL"
    assert calendar_code("IBIS") == "XETR"


def test_us_and_unknown_markets_fall_back_to_new_york():
    assert calendar_code(None) == "XNYS"
    assert calendar_code("SMART") == "XNYS"
    assert calendar_code("SOMEWHERE-ELSE") == "XNYS"


def test_weekend_is_non_trading_on_every_market():
    saturday = datetime(2026, 8, 15)
    assert is_non_trading_day(saturday, "BVME")
    assert is_non_trading_day(saturday)


def test_italian_holiday_is_a_trading_day_in_new_york():
    # Easter Monday 2026: Borsa Italiana closed, NYSE open.
    easter_monday = datetime(2026, 4, 6)
    assert is_non_trading_day(easter_monday, "BVME")
    assert not is_non_trading_day(easter_monday)


def test_labour_day_is_a_borsa_italiana_holiday():
    labour_day = datetime(2026, 5, 1)
    assert is_non_trading_day(labour_day, "BVME")
    assert not is_non_trading_day(labour_day)
