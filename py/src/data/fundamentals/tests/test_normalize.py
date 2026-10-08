"""Tests for SEC payload -> sparse rows (pure mapping)."""

from __future__ import annotations

import pandas as pd


from src.data.fundamentals.normalize import TAG_MAP, sec_payload_to_rows
from src.utils import parse_timestamp


def ts(value: str) -> pd.Timestamp:
    """``pd.Timestamp`` narrowed past its ``NaTType`` union (repo convention)."""
    return parse_timestamp(value)


def _fact(
    tag: str,
    value: float | str,
    start: str,
    end: str,
    filed: str,
    form: str = "10-Q",
) -> tuple[str, dict]:
    return tag, {
        "units": {
            "USD": [
                {"start": start, "end": end, "val": value, "form": form, "filed": filed}
            ]
        }
    }


def _payload(*facts: tuple[str, dict]) -> dict:
    return {"facts": {"us-gaap": {tag: body for tag, body in facts}}}


def test_maps_income_balance_and_cashflow_fields() -> None:
    payload = _payload(
        _fact(
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            1000.0,
            "2024-01-01",
            "2024-03-31",
            "2024-05-01",
        ),
        _fact("NetIncomeLoss", 120.0, "2024-01-01", "2024-03-31", "2024-05-01"),
        _fact("Assets", 5000.0, "2024-03-31", "2024-03-31", "2024-05-01"),
        _fact(
            "NetCashProvidedByUsedInOperatingActivities",
            200.0,
            "2024-01-01",
            "2024-03-31",
            "2024-05-01",
        ),
    )
    rows = sec_payload_to_rows("aapl", payload)

    by_key = {(r.statement, r.field): r for r in rows}
    assert by_key[("income", "revenue")].value == 1000.0
    assert by_key[("income", "net_income")].value == 120.0
    assert by_key[("balance", "assets")].value == 5000.0
    assert by_key[("cashflow", "operating_cash_flow")].value == 200.0
    # Ticker is normalized to the stored (uppercase) convention.
    assert all(r.ticker == "AAPL" for r in rows)
    # The fiscal span and the PIT anchor survive the mapping intact.
    revenue = by_key[("income", "revenue")]
    assert revenue.period_start == ts("2024-01-01")
    assert revenue.period_end == ts("2024-03-31")
    assert revenue.filed == ts("2024-05-01")
    assert revenue.form == "10-Q"


def test_first_matching_tag_wins_and_does_not_double_count() -> None:
    """A filer tagging both a specific and a coarse revenue tag yields one row."""
    payload = _payload(
        _fact(
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            1000.0,
            "2024-01-01",
            "2024-03-31",
            "2024-05-01",
        ),
        _fact("Revenues", 9999.0, "2024-01-01", "2024-03-31", "2024-05-01"),
    )
    revenue = [r for r in sec_payload_to_rows("AAPL", payload) if r.field == "revenue"]
    assert len(revenue) == 1
    assert revenue[0].value == 1000.0


def test_restated_period_emits_both_filings() -> None:
    """Both the original and the restatement are kept — the PIT distinction is data."""
    tag = "NetIncomeLoss"
    body = {
        "units": {
            "USD": [
                {
                    "start": "2023-07-01",
                    "end": "2023-09-30",
                    "val": 10.0,
                    "form": "10-Q",
                    "filed": "2023-11-06",
                },
                {
                    "start": "2023-07-01",
                    "end": "2023-09-30",
                    "val": 30.0,
                    "form": "10-K",
                    "filed": "2024-02-01",
                },
            ]
        }
    }
    rows = [
        r
        for r in sec_payload_to_rows("DEMO", _payload((tag, body)))
        if r.field == "net_income"
    ]
    assert sorted(r.value for r in rows) == [10.0, 30.0]
    assert sorted(r.filed for r in rows) == [
        ts("2023-11-06"),
        ts("2024-02-01"),
    ]


def test_unusable_facts_are_dropped_not_zeroed() -> None:
    """Null-date sentinel, non-numeric val, unknown form, missing filed -> no row."""
    body = {
        "units": {
            "USD": [
                {
                    "start": "2024-01-01",
                    "end": "1999-12-31",
                    "val": 5.0,
                    "form": "10-Q",
                    "filed": "2024-05-01",
                },
                {
                    "start": "2024-01-01",
                    "end": "2024-03-31",
                    "val": "n/a",
                    "form": "10-Q",
                    "filed": "2024-05-01",
                },
                {
                    "start": "2024-01-01",
                    "end": "2024-03-31",
                    "val": 5.0,
                    "form": "S-1",
                    "filed": "2024-05-01",
                },
                {
                    "start": "2024-01-01",
                    "end": "2024-03-31",
                    "val": 5.0,
                    "form": "10-Q",
                },  # no filed
                {
                    "end": "2024-03-31",
                    "val": 5.0,
                    "form": "10-Q",
                },  # no filed, instant
            ]
        }
    }
    assert sec_payload_to_rows("AAPL", _payload(("Assets", body))) == []


def test_balance_instants_are_kept_with_collapsed_span() -> None:
    """A balance-sheet fact is an instant: ``end`` without ``start`` is valid.

    SEC reports stocks (assets/equity) as point-in-time facts, so rejecting a
    missing ``start`` silently dropped **every** balance row. The admitted row
    collapses its span to one day, which is what lets a strategy tell a stock
    (``spans()`` start == end) from a flow (a real fiscal duration) — and it is
    what makes the column non-empty in the first place.
    """
    body = {
        "units": {
            "USD": [
                {
                    "end": "2024-03-31",
                    "val": 5.0,
                    "form": "10-Q",
                    "filed": "2024-05-01",
                }
            ]
        }
    }
    rows = sec_payload_to_rows("AAPL", _payload(("Assets", body)))
    assert len(rows) == 1
    row = rows[0]
    assert row.statement == "balance"
    assert row.field == "assets"
    assert row.value == 5.0
    assert row.period_start == row.period_end == parse_timestamp("2024-03-31")
    assert row.filed == parse_timestamp("2024-05-01")


def test_duplicate_fact_repeats_dedupe() -> None:
    """SEC repeats a fact per unit/context — the same datum must not become two rows."""
    body = {
        "units": {
            "USD": [
                {
                    "start": "2024-01-01",
                    "end": "2024-03-31",
                    "val": 7.0,
                    "form": "10-Q",
                    "filed": "2024-05-01",
                }
            ]
            * 3
        }
    }
    rows = [
        r
        for r in sec_payload_to_rows("AAPL", _payload(("Assets", body)))
        if r.field == "assets"
    ]
    assert len(rows) == 1


def test_free_cash_flow_derived_and_anchored_to_later_filing() -> None:
    """FCF = ocf - |capex|, visible only once the later of the two inputs was filed."""
    ocf = {
        "units": {
            "USD": [
                {
                    "start": "2024-01-01",
                    "end": "2024-03-31",
                    "val": 200.0,
                    "form": "10-Q",
                    "filed": "2024-05-01",
                }
            ]
        }
    }
    capex = {
        "units": {
            "USD": [
                {
                    "start": "2024-01-01",
                    "end": "2024-03-31",
                    "val": -80.0,
                    "form": "10-Q",
                    "filed": "2024-06-15",
                }
            ]
        }
    }
    payload = _payload(
        ("NetCashProvidedByUsedInOperatingActivities", ocf),
        ("PaymentsToAcquirePropertyPlantAndEquipment", capex),
    )
    fcf = [
        r for r in sec_payload_to_rows("AAPL", payload) if r.field == "free_cash_flow"
    ]
    assert len(fcf) == 1
    assert fcf[0].value == 120.0
    assert fcf[0].filed == ts("2024-06-15")
    # Capex keeps its reported (negative) sign on the row itself.
    raw_capex = [r for r in sec_payload_to_rows("AAPL", payload) if r.field == "capex"]
    assert raw_capex[0].value == -80.0


def test_no_fcf_when_a_component_is_missing() -> None:
    """A half-computed FCF would read as a real number — emit nothing instead."""
    ocf = {
        "units": {
            "USD": [
                {
                    "start": "2024-01-01",
                    "end": "2024-03-31",
                    "val": 200.0,
                    "form": "10-Q",
                    "filed": "2024-05-01",
                }
            ]
        }
    }
    payload = _payload(("NetCashProvidedByUsedInOperatingActivities", ocf))
    assert [
        r for r in sec_payload_to_rows("AAPL", payload) if r.field == "free_cash_flow"
    ] == []


def test_tag_map_targets_are_valid_statement_fields() -> None:
    """Every mapped (statement, field) must exist on that statement's dataclass."""
    from src.data.fundamentals.schema import field_names

    for statement, field in TAG_MAP:
        assert field in field_names(statement), (
            f"{statement}.{field} is not a dataclass field"
        )
