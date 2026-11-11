"""Market-aware gap detection: the venue's holidays are not missing data.

Two bars straddling the Easter break. Borsa Italiana was shut on Good Friday and
Easter Monday; New York traded the Monday — so the same frame is complete for the
Milan listing and a real gap for the US one.
"""

from __future__ import annotations

import pandas as pd

from src.data._shared import find_gaps_48h


def _frame(stamps: list[str]) -> pd.DataFrame:
    index = pd.DatetimeIndex([pd.Timestamp(s) for s in stamps])
    return pd.DataFrame({"close": [1.0] * len(index)}, index=index)


def test_easter_break_is_not_a_gap_for_a_milan_listing():
    df = _frame(["2026-04-02 15:00", "2026-04-07 07:00"])

    assert find_gaps_48h(df, "BVME") == []


def test_the_same_break_is_a_gap_for_a_new_york_listing():
    df = _frame(["2026-04-02 15:00", "2026-04-07 07:00"])

    assert len(find_gaps_48h(df)) == 1


def test_a_real_hole_is_reported_whatever_the_market():
    df = _frame(["2026-04-02 15:00", "2026-04-17 07:00"])

    assert len(find_gaps_48h(df, "BVME")) == 1
    assert len(find_gaps_48h(df)) == 1
