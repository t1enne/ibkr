"""Pure-function tests for the fundamentals pair strategy's scoring helpers.

The engine-level wiring is covered by ``test_fundamentals_wiring``; what is worth
pinning here is the *scoring* arithmetic and, above all, the annual-period filter
— mixing a 10-K year with a cumulative 9-month 10-Q would silently compare a year
to three quarters, and that is the failure mode most likely to look like a signal.
"""

from __future__ import annotations

import pandas as pd

from src.bt.strategies.fundamentals_context import SeriesPIT
from src.bt.strategies.fundamentals_pair_dsl import (
    _METRIC_FIELD,
    Params,
    _annual_values,
    _pair_legs,
    _score,
)
from src.data.fundamentals.schema import Form, FundamentalRow
from src.utils import parse_timestamp


def ts(value: str) -> pd.Timestamp:
    """``pd.Timestamp`` narrowed past its ``NaTType`` union (repo convention)."""
    return parse_timestamp(value)


def _row(
    value: float, start: str, end: str, filed: str, form: Form = "10-K"
) -> FundamentalRow:
    return FundamentalRow(
        ticker="TEST",
        statement="income",
        field="net_income",
        value=value,
        period_start=ts(start),
        period_end=ts(end),
        filed=ts(filed),
        form=form,
    )


ANNUAL = [
    _row(100.0, "2020-01-01", "2020-12-31", "2021-02-15"),
    _row(120.0, "2021-01-01", "2021-12-31", "2022-02-15"),
    _row(150.0, "2022-01-01", "2022-12-31", "2023-02-15"),
]
QUARTERS = [
    _row(30.0, "2023-01-01", "2023-03-31", "2023-05-01", form="10-Q"),
    _row(70.0, "2023-01-01", "2023-09-30", "2023-11-01", form="10-Q"),
]


def _series(rows: list[FundamentalRow]) -> SeriesPIT:
    ordered = sorted(rows, key=lambda r: (r.period_end, r.filed))
    return SeriesPIT(
        period=tuple(r.period_end for r in ordered),
        window=tuple(ordered),
    )


def test_annual_values_excludes_cumulative_quarters() -> None:
    """A 9-month YTD 10-Q fact is not a year, however large its value."""
    values = _annual_values(_series(ANNUAL + QUARTERS), years=10)
    assert values == [100.0, 120.0, 150.0]


def test_annual_values_keeps_most_recent_years_in_order() -> None:
    """Oldest-first order, truncated to the newest ``years`` on the right."""
    assert _annual_values(_series(ANNUAL), years=2) == [120.0, 150.0]


def test_annual_values_returns_fewer_when_history_is_short() -> None:
    """A short series yields what exists; the caller treats that as tradable-or-not."""
    assert _annual_values(_series(ANNUAL[:1]), years=3) == [100.0]
    assert _annual_values(_series([]), years=3) == []


def test_score_is_equity_normalised_level_plus_drift() -> None:
    """Score = latest/equity + (latest - oldest)/equity, so it is unit-free."""
    # equity 1000: level 0.15, drift (150-100)/1000 = 0.05 -> 0.20
    assert _score(_series(ANNUAL), years=3, equity=1000.0) == 0.2


def test_score_none_when_window_too_short_or_equity_degenerate() -> None:
    """Not-scoreable is None, never 0.0 — a missing fundamental is not a weak one."""
    assert _score(_series(ANNUAL[:1]), years=3, equity=1000.0) is None
    assert _score(_series(ANNUAL), years=3, equity=0.0) is None
    assert _score(_series(ANNUAL), years=3, equity=-5.0) is None


def test_score_ranks_the_improving_leg_above_the_flat_one() -> None:
    """The whole premise: a leg growing per unit of book outranks a stagnant one."""
    flat = [_row(100.0, "2020-01-01", "2020-12-31", "2021-02-15")] * 1 + [
        _row(100.0, "2022-01-01", "2022-12-31", "2023-02-15")
    ]
    rising = [_row(100.0, "2020-01-01", "2020-12-31", "2021-02-15")] + [
        _row(200.0, "2022-01-01", "2022-12-31", "2023-02-15")
    ]
    flat_score = _score(_series(flat), years=2, equity=1000.0)
    rising_score = _score(_series(rising), years=2, equity=1000.0)
    assert flat_score is not None and rising_score is not None
    assert rising_score > flat_score


def test_metric_table_never_uses_equity_as_the_scored_field() -> None:
    """``equity`` is the score's denominator; scoring it would tie every pair.

    Pinned as a test because the degeneracy is silent: all scores collapse to
    ~1.0, every pair compares equal, and the strategy simply opens no trades.
    """
    assert "equity" not in _METRIC_FIELD
    assert all(field != "equity" for _, field in _METRIC_FIELD.values())


def test_pair_legs_inverts_declared_order_when_short_pairs_is_false() -> None:
    """Inverting separates "the ranking works" from "it works only on the long side"."""
    forward = _pair_legs(Params(pairs=["NVDA:LRCX"]))
    inverted = _pair_legs(Params(pairs=["NVDA:LRCX"], short_pairs=False))
    assert forward == [("NVDA", "LRCX")]
    assert inverted == [("LRCX", "NVDA")]


def test_params_accept_a_json_list_for_pairs() -> None:
    """JSON has no tuple, so ``pairs`` must survive a plain list from the config."""
    params = Params.from_dict({"pairs": ["NVDA:LRCX", "MU:WDC"], "metric": "revenue"})
    assert params.pairs == ["NVDA:LRCX", "MU:WDC"]
    assert params.metric == "revenue"
