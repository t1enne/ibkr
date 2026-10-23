"""Test gap calculation logic — pure functions in candles.py.

The forward edge is judged against "today", so every test that can reach it
pins that clock via the ``_today_utc`` seam instead of trusting the machine's
date — otherwise the suite passes or fails depending on when it is run.
"""

from __future__ import annotations

import importlib
from datetime import date, datetime, timezone

import pytest

from src.data.ibkr.candles import calculate_gaps, _merge_and_sort_gaps

_candles = importlib.import_module("src.data.ibkr.candles")


@pytest.fixture
def today(monkeypatch):
    """Pin the forward-edge clock; call with the date the test pretends it is."""

    def _pin(day: date) -> None:
        monkeypatch.setattr(_candles, "_today_utc", lambda: day)

    return _pin


_JAN1 = 1767225600000
_JUN1 = 1780272000000


class TestCalculateGaps:
    def test_no_existing_data(self):
        gaps = calculate_gaps(datetime(2026, 1, 1), datetime(2026, 6, 1), None, None)
        assert len(gaps) == 1
        assert gaps[0] == (datetime(2026, 1, 1), datetime(2026, 6, 1))

    def test_full_coverage(self):
        assert (
            calculate_gaps(datetime(2026, 1, 1), datetime(2026, 6, 1), _JAN1, _JUN1)
            == []
        )

    def test_backward_gap(self):
        gaps = calculate_gaps(datetime(2025, 12, 1), datetime(2026, 6, 1), _JAN1, _JUN1)
        assert len(gaps) == 1
        assert gaps[0][0] == datetime(2025, 12, 1)

    def test_forward_gap(self, today):
        today(date(2026, 7, 1))
        gaps = calculate_gaps(datetime(2026, 1, 1), datetime(2026, 7, 1), _JAN1, _JUN1)
        assert len(gaps) == 1
        assert gaps[0][1] == datetime(2026, 7, 1)

    def test_forward_gap_weekend_filtered(self, today):
        today(date(2026, 6, 7))
        jun5_fri_19 = datetime(2026, 6, 5, 19, 0)
        jun7_sun_10 = datetime(2026, 6, 7, 10, 0)
        newest_ts = int(jun5_fri_19.replace(tzinfo=timezone.utc).timestamp() * 1000)
        gaps = calculate_gaps(datetime(2026, 1, 1), jun7_sun_10, _JAN1, newest_ts)
        assert len(gaps) == 0

    def test_forward_gap_already_current(self, today):
        """Yesterday's close, run today: nothing is owed, whatever the bar hour."""
        today(date(2026, 6, 9))
        for hour, minute in ((15, 30), (16, 0), (19, 0)):
            newest = datetime(2026, 6, 8, hour, minute)
            ts = int(newest.replace(tzinfo=timezone.utc).timestamp() * 1000)
            assert (
                calculate_gaps(datetime(2026, 1, 1), datetime(2026, 6, 9), _JAN1, ts)
                == []
            )


class TestMergeAndSortGaps:
    def test_merge_overlapping(self):
        gaps = [
            (datetime(2026, 1, 1), datetime(2026, 1, 10)),
            (datetime(2026, 1, 5), datetime(2026, 1, 15)),
        ]
        merged = _merge_and_sort_gaps(gaps)
        assert len(merged) == 1
        assert merged[0] == (datetime(2026, 1, 1), datetime(2026, 1, 15))

    def test_disjoint_kept_separate(self):
        g1 = (datetime(2026, 1, 1), datetime(2026, 1, 5))
        g2 = (datetime(2026, 1, 10), datetime(2026, 1, 15))
        merged = _merge_and_sort_gaps([g2, g1])  # unsorted
        assert len(merged) == 2
        assert merged[0] == g1
        assert merged[1] == g2
