"""PIT tests: as-first-stated dedupe, cursor safety, snapshot roundtrip, latest.

These cover the load-bearing invariants of the fundamentals store. Each one
fails on a real regression (a restatement overwriting history, a series leaking
a future filing, ``latest`` serving a stale period) rather than restating the
implementation.
"""

from __future__ import annotations

import pandas as pd

from src.bt.strategies.fundamentals_context import Fundamentals
from src.data.fundamentals.query import (
    as_first_stated,
    rows_to_snapshot,
    snapshot_to_rows,
)
from src.data.fundamentals.schema import (
    Form,
    Statement,
    BalanceSheet,
    CashFlow,
    FundamentalRow,
    Income,
)
from src.utils import parse_timestamp


def ts(value: str) -> pd.Timestamp:
    """``pd.Timestamp`` narrowed past its ``NaTType`` union (repo convention)."""
    return parse_timestamp(value)


# --- fixtures --------------------------------------------------------------
def _row(
    field: str,
    value: float,
    period_start: str,
    period_end: str,
    filed: str,
    *,
    statement: Statement = "income",
    ticker: str = "DEMO",
    form: Form = "10-Q",
) -> FundamentalRow:
    return FundamentalRow(
        ticker=ticker,
        statement=statement,
        field=field,
        value=value,
        period_start=parse_timestamp(period_start),
        period_end=parse_timestamp(period_end),
        filed=parse_timestamp(filed),
        form=form,
    )


Q1 = _row("net_income", 10.0, "2023-01-01", "2023-03-31", "2023-05-01")
Q2 = _row("net_income", 20.0, "2023-04-01", "2023-06-30", "2023-08-01")
# A 10-K filed after Q3 states its own Q3 quarter and restates Q1.
Q3 = _row("net_income", 30.0, "2023-07-01", "2023-09-30", "2023-11-06")
Q1_RESTATED = _row(
    "net_income", 99.0, "2023-01-01", "2023-03-31", "2024-02-01", form="10-K"
)


# --- as-first-stated -------------------------------------------------------


def test_as_first_stated_drops_the_restatement() -> None:
    """A restated old period never rewrites the curve."""
    assert as_first_stated([Q1, Q1_RESTATED]) == (Q1,)
    # Order of the input must not matter (the dedupe is by min ``filed``).
    assert as_first_stated([Q1_RESTATED, Q1]) == (Q1,)


def test_as_first_stated_keeps_distinct_periods_and_fields() -> None:
    rows = [Q1, Q2, Q3, _row("revenue", 1.0, "2023-01-01", "2023-03-31", "2023-05-01")]
    kept = as_first_stated(rows)
    assert {r.period_end for r in kept} == {r.period_end for r in rows}
    assert len(kept) == 4  # revenue shares Q1's span but is a different field


def test_as_first_stated_keeps_distinct_statements_for_one_span() -> None:
    """Same span on two statements (income vs balance) are independent facts."""
    balance = _row(
        "assets", 5.0, "2023-01-01", "2023-03-31", "2023-05-01", statement="balance"
    )
    assert len(as_first_stated([Q1, balance])) == 2


# --- series cursor safety --------------------------------------------------


def test_series_excludes_filings_after_the_cursor() -> None:
    """No filing with ``filed > cursor`` is ever visible."""
    cursor: list[pd.Timestamp | None] = [ts("2024-01-15")]
    fund = Fundamentals.build(
        {"DEMO": [Q1, Q2, Q3, Q1_RESTATED]}, cursor=lambda: cursor[0]
    )
    ni = fund.income("DEMO").net_income

    # At 2024-01-15 the Q1 restatement (filed 2024-02-01) is not public yet, and
    # Q1 still reads as first stated (10.0), not 99.0.
    assert ni.visible == 3
    assert ni[-3:] == [10.0, 20.0, 30.0]
    assert ni[-1] == 30.0
    assert len(ni) == 3

    # The restatement is filed on 2024-02-01, but it restates a period that was
    # first stated in Nov 2023 — as-first-stated means it never joins the curve,
    # at any cursor.
    cursor[0] = ts("2024-02-01")
    assert ni.visible == 3
    assert ni[-3:] == [10.0, 20.0, 30.0]


def test_series_is_empty_before_the_first_filing() -> None:
    cursor: list[pd.Timestamp | None] = [ts("2023-01-01")]
    fund = Fundamentals.build({"DEMO": [Q1, Q2]}, cursor=lambda: cursor[0])
    ni = fund.income("DEMO").net_income
    assert len(ni) == 0
    assert ni.last() is None
    assert ni.spans() == ()


def test_series_is_empty_before_the_engine_advances_its_cursor() -> None:
    """A cursor-bound but unadvanced store must not leak history (pre-run)."""
    engine_cursor: list[pd.Timestamp | None] = [None]
    fund = Fundamentals.build({"DEMO": [Q1, Q2]}, cursor=lambda: engine_cursor[0])
    assert len(fund.income("DEMO").net_income) == 0
    assert fund.latest("DEMO", "income") is None
    # Once the engine advances past a filing, only what is published shows up.
    engine_cursor[0] = ts("2023-05-01")
    assert fund.income("DEMO").net_income[:] == [10.0]


def test_series_without_any_cursor_is_fully_visible() -> None:
    """A store built with no cursor reader is the notebook/test path (not a run)."""
    fund = Fundamentals.build({"DEMO": [Q1, Q2]})
    assert fund.income("DEMO").net_income[:] == [10.0, 20.0]


def test_series_binds_the_live_cursor_not_a_snapshot() -> None:
    """Advancing the engine cursor is reflected on reads (no cached visibility)."""
    cursor: list[pd.Timestamp | None] = [ts("2023-06-01")]
    fund = Fundamentals.build({"DEMO": [Q1, Q2, Q3]}, cursor=lambda: cursor[0])
    ni = fund.income("DEMO").net_income
    assert ni[-1] == 10.0
    cursor[0] = ts("2023-09-01")
    assert ni[-1] == 20.0
    cursor[0] = None  # engine cursor reset -> nothing observable
    assert len(ni) == 0


def test_series_slice_and_scalar_index_semantics() -> None:
    cursor = lambda: ts("2024-01-15")  # noqa: E731
    fund = Fundamentals.build({"DEMO": [Q1, Q2, Q3]}, cursor=cursor)
    ni = fund.income("DEMO").net_income
    assert ni[-1] == 30.0
    assert ni[0] == 10.0
    assert ni[-2:] == [20.0, 30.0]
    assert ni[:2] == [10.0, 20.0]
    assert ni[:] == [10.0, 20.0, 30.0]
    # Absent periods are absent, never NaN-padded.
    for bad in (3, -4):
        try:
            ni[bad]
        except IndexError:
            continue
        raise AssertionError(f"expected IndexError for index {bad}")


def test_spans_and_forms_describe_the_fiscal_axis() -> None:
    cursor = lambda: ts("2024-06-01")  # noqa: E731
    fund = Fundamentals.build({"DEMO": [Q1, Q2, Q3, Q1_RESTATED]}, cursor=cursor)
    series = fund.income("DEMO").net_income
    assert series.spans()[:2] == (
        (ts("2023-01-01"), ts("2023-03-31")),
        (ts("2023-04-01"), ts("2023-06-30")),
    )
    assert series.forms() == ("10-Q", "10-Q", "10-Q")
    # The 10-K only supplies the restatement of an already-stated period, so the
    # curve keeps three spans — no restated period is appended.
    assert len(series.spans()) == 3


def test_unknown_symbol_and_field_are_loud() -> None:
    fund = Fundamentals.build({"DEMO": [Q1]}, cursor=lambda: ts("2024-06-01"))
    assert len(fund.income("NOPE").net_income) == 0
    try:
        fund.income("DEMO").net_income_typo
    except AttributeError as exc:
        assert "net_income_typo" in str(exc)
    else:
        raise AssertionError("expected AttributeError for an unknown field")


# --- snapshot roundtrip ----------------------------------------------------


def test_snapshot_roundtrip_through_rows() -> None:
    """snapshot -> rows -> snapshot is identity, one row per non-None field."""
    snapshot = Income(revenue=100.0, net_income=12.5, eps_diluted=1.25)
    rows = snapshot_to_rows(
        snapshot, "demo", "2024-01-01", "2024-03-31", "2024-05-01", "10-Q"
    )

    assert len(rows) == 3  # None fields are omitted, not zero-filled
    assert all(r.ticker == "DEMO" for r in rows)
    assert rows_to_snapshot(rows, "DEMO", ts("2024-03-31")) == snapshot


def test_snapshot_roundtrip_for_all_statements() -> None:
    for snapshot in (
        Income(net_income=1.0),
        BalanceSheet(assets=10.0, equity=4.0),
        CashFlow(operating_cash_flow=3.0, capex=-1.0, free_cash_flow=2.0),
    ):
        rows = snapshot_to_rows(
            snapshot, "DEMO", "2024-01-01", "2024-03-31", "2024-05-01", "10-K"
        )
        rebuilt = rows_to_snapshot(rows, "DEMO", ts("2024-03-31"))
        assert rebuilt == snapshot


def test_snapshot_uses_the_as_first_stated_value_by_default() -> None:
    """The default (series) view reads the earliest filing of each field."""
    period = ts("2023-03-31")
    first = rows_to_snapshot([Q1, Q1_RESTATED], "DEMO", period)
    restated = rows_to_snapshot([Q1, Q1_RESTATED], "DEMO", period, as_first=False)
    assert isinstance(first, Income) and first.net_income == 10.0
    assert isinstance(restated, Income) and restated.net_income == 99.0


def test_snapshot_is_none_for_an_unknown_period() -> None:
    assert rows_to_snapshot([Q1], "DEMO", ts("1999-12-31")) is None


def test_snapshot_of_mixed_statements_needs_an_explicit_statement() -> None:
    """A period with income and balance rows is two reconstructions, not one."""
    balance = _row(
        "assets", 5.0, "2023-01-01", "2023-03-31", "2023-05-01", statement="balance"
    )
    period = ts("2023-03-31")
    assert rows_to_snapshot([Q1, balance], "DEMO", period) is None
    income = rows_to_snapshot([Q1, balance], "DEMO", period, statement="income")
    assert isinstance(income, Income) and income.net_income == 10.0


# --- latest (newest-filed / read-at-cursor) --------------------------------


def test_latest_returns_the_newest_filed_restated_snapshot() -> None:
    """``latest`` is the restated view while the curve stays as-first-stated."""
    cursor: list[pd.Timestamp | None] = [ts("2024-06-01")]
    fund = Fundamentals.build(
        {"DEMO": [Q1, Q2, Q3, Q1_RESTATED]}, cursor=lambda: cursor[0]
    )

    latest = fund.latest("DEMO", "income")
    assert isinstance(latest, Income)
    # Newest period is Q3 (2023-09-30), and it is not the restated Q1.
    assert latest.net_income == 30.0
    # The curve keeps the as-first-stated Q1 value.
    assert fund.income("DEMO").net_income[0] == 10.0


def test_latest_restates_the_newest_period_when_that_period_was_restated() -> None:
    """The restated value is reachable via ``latest`` even though the curve hides it."""
    cursor: list[pd.Timestamp | None] = [ts("2024-06-01")]
    fund = Fundamentals.build({"DEMO": [Q1, Q1_RESTATED]}, cursor=lambda: cursor[0])
    latest = fund.latest("DEMO", "income")
    assert isinstance(latest, Income) and latest.net_income == 99.0
    assert fund.income("DEMO").net_income[-1] == 10.0  # curve unchanged


def test_latest_is_cursor_bounded() -> None:
    """At a cursor before the restatement was filed, ``latest`` is the original."""
    cursor: list[pd.Timestamp | None] = [ts("2024-01-15")]
    fund = Fundamentals.build({"DEMO": [Q1, Q1_RESTATED]}, cursor=lambda: cursor[0])
    latest = fund.latest("DEMO", "income")
    assert isinstance(latest, Income) and latest.net_income == 10.0
    # Cursor before anything was filed -> nothing readable at all.
    cursor[0] = ts("2023-01-01")
    assert fund.latest("DEMO", "income") is None


def test_latest_is_per_statement_and_none_when_uncovered() -> None:
    cursor = lambda: ts("2024-06-01")  # noqa: E731
    fund = Fundamentals.build({"DEMO": [Q1]}, cursor=cursor)
    assert fund.latest("DEMO", "balance") is None
    assert fund.latest("NOPE", "income") is None


def test_build_applies_as_first_stated_per_symbol() -> None:
    fund = Fundamentals.build(
        {"DEMO": [Q1_RESTATED, Q1], "OTHER": [Q2]},
        cursor=lambda: ts("2024-06-01"),
    )
    assert fund.symbols == ("DEMO", "OTHER")
    assert fund.income("DEMO").net_income[:] == [10.0]


# --- sparse-filing axis: ``filed`` need not ascend with ``period_end`` -------


def test_visibility_is_per_row_when_filed_disagrees_with_period_order() -> None:
    """Real SEC data files an older period *after* a newer one.

    A sparse series of cumulative facts does not keep ``filed`` in ``period_end``
    order: an FY fact covering an old year is routinely published later than the
    interim 10-Qs of the years after it. When that happens the visible periods
    are a *subsequence* of the window, so a prefix slice would both hide a
    legitimate period and expose a filing-future one.
    """
    old_year = _row(
        "net_income", 1.0, "2022-01-01", "2022-12-31", "2025-03-01", form="10-K"
    )
    newer_quarters = [
        _row("net_income", 2.0, "2023-01-01", "2023-03-31", "2023-05-01"),
        _row("net_income", 3.0, "2023-04-01", "2023-06-30", "2023-08-01"),
    ]
    cursor: list[pd.Timestamp | None] = [ts("2024-01-01")]
    fund = Fundamentals.build(
        {"DEMO": [old_year, *newer_quarters]}, cursor=lambda: cursor[0]
    )
    ni = fund.income("DEMO").net_income

    # The two 2023 quarters are public; the 2022 10-K is not filed until 2025.
    assert len(ni) == 2
    assert ni[:] == [2.0, 3.0]
    assert ni.last() == 3.0
    assert [end.year for _, end in ni.spans()] == [2023, 2023]
    assert ni.forms() == ("10-Q", "10-Q")

    # Once the late annual filing lands it joins the curve *behind* the quarters
    # in period order, not appended at the end.
    cursor[0] = ts("2025-03-01")
    assert ni[:] == [1.0, 2.0, 3.0]
    assert ni[-1] == 3.0
    assert ni.last() == 3.0
    assert ni.forms() == ("10-K", "10-Q", "10-Q")


def test_snapshot_defaults_to_newest_visible_not_last_period() -> None:
    """``snapshot()`` with no period must not pick a not-yet-filed period.

    The period-ordered window ends with whatever period is *latest by period
    date*, which may be a filing still in the strategy's future. Defaulting to
    it would leak: the snapshot's numeric values would come from ``period``
    regardless of the cursor.
    """
    late = _row(
        "net_income", 999.0, "2024-01-01", "2024-12-31", "2026-01-01", form="10-K"
    )
    early = _row(
        "net_income", 10.0, "2023-01-01", "2023-12-31", "2024-02-01", form="10-K"
    )
    cursor: list[pd.Timestamp | None] = [ts("2024-06-01")]
    fund = Fundamentals.build({"DEMO": [early, late]}, cursor=lambda: cursor[0])
    snapshot = fund.income("DEMO").snapshot()
    assert isinstance(snapshot, Income)
    assert snapshot.net_income == 10.0

    # With nothing published yet there is no defensible "newest" -> None.
    cursor[0] = ts("2020-01-01")
    assert fund.income("DEMO").snapshot() is None
