"""Tests for the warmup duration parser."""

from __future__ import annotations

import pytest

from src.bt.warmup import (
    TRADING_DAYS_PER_WEEK,
    WarmupParseError,
    bars_per_trading_day,
    parse_warmup,
    parse_warmup_bars,
    parse_warmup_days,
)


def test_parse_warmup_days_units() -> None:
    assert parse_warmup_days("90d") == 90.0
    assert parse_warmup_days("1w") == 7.0
    assert parse_warmup_days("1y") == 365.0
    assert parse_warmup_days("6m") == 186.0  # 6 * 31 — longest month, never short
    assert parse_warmup_days("12h") == pytest.approx(0.5)


def test_parse_warmup_days_spellings_and_whitespace() -> None:
    for spelling in ("1y", "1yr", "1year", "1 year", " 1 years "):
        assert parse_warmup_days(spelling) == 365.0
    for spelling in ("6m", "6mo", "6month", "6 months"):
        assert parse_warmup_days(spelling) == 186.0
    for spelling in ("90d", "90day", "90 days"):
        assert parse_warmup_days(spelling) == 90.0


@pytest.mark.parametrize(
    "bad",
    # one representative per garbage class: empty, whitespace, alpha,
    # sign-prefixed, and a bare number — all hit the same parse-fail branch.
    ["", "  ", "abc", "-1y", "90"],
)
def test_parse_warmup_days_rejects_garbage(bad: str) -> None:
    """A garbage/unknown duration must raise a clear error, never default."""
    with pytest.raises(WarmupParseError, match="expected a calendar duration"):
        parse_warmup_days(bad)


def test_parse_warmup_days_zero_is_legal() -> None:
    """Zero is a legal no-op warmup; a sign is not accepted at all."""
    assert parse_warmup_days("0d") == 0.0
    assert parse_warmup_bars("0d", "1d") == 0
    with pytest.raises(WarmupParseError, match="expected a calendar duration"):
        parse_warmup_days("-1d")


def test_bars_per_trading_day_interval_aware() -> None:
    assert bars_per_trading_day("1d") == 1.0
    assert bars_per_trading_day("1h") == 6.5
    # Session-anchored resampling (src.data.resample) tiles the session into
    # whole buckets: ceil(6.5 / hours), so "4h" is 2 bars/day, not 6.5/4.
    assert bars_per_trading_day("4h") == 2.0
    assert bars_per_trading_day("2h") == 4.0
    with pytest.raises(ValueError, match="unrecognised bar"):
        bars_per_trading_day("nonsense")


def test_parse_warmup_bars_reference_values() -> None:
    """The three documented reference conversions."""
    assert parse_warmup_bars("1y", "1d") == 261
    assert parse_warmup_bars("90d", "1d") == 65
    assert parse_warmup_bars("1y", "1h") == 1695
    # 365d -> ~261 trading days -> x2 bars/day = 522
    assert parse_warmup_bars("1y", "4h") == 522


def test_parse_warmup_bars_counts_trading_days_not_calendar_days() -> None:
    """The assumption under test: bars scale by 5/7 of calendar days for 1d."""
    for days, expect in ((7, 5), (14, 10), (70, 50), (365, 261)):
        assert parse_warmup_bars(f"{days}d", "1d") == expect
    # An intraday warmup is strictly longer in bars than the daily one.
    assert parse_warmup_bars("1y", "1h") > parse_warmup_bars("1y", "1d")


def test_parse_warmup_bars_never_under_counts_calendar_span() -> None:
    """Guard the direction of the rounding: a warmup must cover its span.

    For a 1d bar the bar count must be at least the number of trading days in
    the span (never fewer) — under-counting reintroduces the cold-warmup bug.
    """
    for days in (1, 3, 30, 90, 186, 365):
        trading_days = days * TRADING_DAYS_PER_WEEK
        assert parse_warmup_bars(f"{days}d", "1d") >= int(trading_days)


def test_parse_warmup_bars_rejects_bad_bar() -> None:
    with pytest.raises(WarmupParseError, match="unrecognised bar size"):
        parse_warmup_bars("1y", "eons")


def test_parse_warmup_span_carries_both_units() -> None:
    span = parse_warmup("6m", "1d")
    assert span.duration == "6m"
    assert span.days == 186.0
    assert span.bars == 133
