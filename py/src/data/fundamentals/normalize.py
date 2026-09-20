"""SEC companyfacts JSON -> sparse canonical rows (pure).

Mirrors the lumibot ``sec.py`` approach: a fixed US-GAAP tag map per normalized
field (first tag present wins), read out of the ``companyfacts`` payload that
SEC EDGAR serves per CIK. Nothing here touches HTTP or the DB — the payload is
passed in, rows come out, which makes every mapping decision testable offline.

Only tags the filer actually reported are emitted: real filings have holes
(a balance-sheet line that never appears in a 10-Q, a segment-tagged revenue),
and those must show up as absent series, never as a fabricated 0.0.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.fundamentals.schema import Form, FundamentalRow, Statement
from src.utils import parse_timestamp

# ---------------------------------------------------------------------------
# tag maps (US-GAAP, first match wins)
# ---------------------------------------------------------------------------

#: ``(statement, field) -> candidate XBRL tags``, most specific first. A tag
#: later in the list is a coarser fallback (e.g. ``Revenues`` when a filer never
#: tags ``RevenueFromContractWithCustomerExcludingAssessedTax``).
TAG_MAP: dict[tuple[Statement, str], tuple[str, ...]] = {
    # -- income -------------------------------------------------------------
    ("income", "revenue"): (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ),
    ("income", "cost_of_revenue"): (
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
        "CostOfServices",
    ),
    ("income", "gross_profit"): ("GrossProfit",),
    ("income", "operating_income"): (
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    ),
    ("income", "net_income"): (
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ),
    ("income", "eps_basic"): (
        "EarningsPerShareBasic",
        "EarningsPerShareBasicAndDiluted",
    ),
    ("income", "eps_diluted"): (
        "EarningsPerShareDiluted",
        "EarningsPerShareBasicAndDiluted",
    ),
    # -- balance sheet ------------------------------------------------------
    ("balance", "assets"): ("Assets",),
    ("balance", "current_assets"): ("AssetsCurrent",),
    ("balance", "cash"): (
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "CashAndDueFromBanks",
    ),
    ("balance", "liabilities"): ("Liabilities",),
    ("balance", "current_liabilities"): ("LiabilitiesCurrent",),
    ("balance", "debt"): (
        "LongTermDebt",
        "LongTermDebtNoncurrent",
        "DebtLongtermAndShorttermCombinedAmount",
    ),
    ("balance", "equity"): (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    ("balance", "shares_outstanding"): (
        "CommonStockSharesOutstanding",
        "EntityCommonStockSharesOutstanding",
    ),
    # -- cash flow ----------------------------------------------------------
    ("cashflow", "operating_cash_flow"): (
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ),
    ("cashflow", "capex"): (
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ),
    ("cashflow", "investing_cash_flow"): (
        "NetCashProvidedByUsedInInvestingActivities",
        "NetCashProvidedByUsedInInvestingActivitiesContinuingOperations",
    ),
    ("cashflow", "dividends_paid"): (
        "PaymentsOfDividendsCommonStock",
        "PaymentsOfDividends",
    ),
    ("cashflow", "buybacks"): (
        "PaymentsForRepurchaseOfCommonStock",
        "PaymentsForRepurchaseOfEquity",
    ),
}

#: SEC form in the payload -> our {@link Form}. Forms outside this set (S-1,
#: 13F, ...) carry no fiscal facts we want and are dropped rather than coerced.
_FORM_MAP: dict[str, Form] = {
    "10-K": "10-K",
    "10-Q": "10-Q",
    "20-F": "20-F",
    "40-F": "40-F",
    "8-K": "8-K",
}

#: Annual-report forms: the span/period disambiguation a consumer needs.
_ANNUAL_FORMS: frozenset[str] = frozenset({"10-K", "20-F", "40-F"})

#: SEC's placeholder end date for facts with no real period close.
_NULL_END_PREFIX = "1999-"


def _finite(raw: object) -> float | None:
    """Numeric ``val`` of a fact, or None when non-numeric/non-finite.

    XBRL ``val`` may arrive as a string (SEC ships some facts quoted), and a
    ``NaN``/``inf`` is not a reported value — drop both rather than coerce.
    """
    if raw is None or isinstance(raw, bool):
        return None
    if not isinstance(raw, (int, float, str)):
        return None
    try:
        value = float(raw)
    except TypeError, ValueError:
        return None
    return value if value == value and abs(value) != float("inf") else None


def _facts_for_tag(
    payload: dict[str, Any], tag: str
) -> list[tuple[float, dict[str, Any]]]:
    """Every finite reported fact for ``tag`` across all units (USD, USD/shares)."""
    tag_facts = payload.get("facts", {}).get("us-gaap", {}).get(tag)
    if not tag_facts:
        return []
    out: list[tuple[float, dict[str, Any]]] = []
    for facts in tag_facts.get("units", {}).values():
        for fact in facts:
            value = _finite(fact.get("val"))
            if value is not None:
                out.append((value, fact))
    return out


def _fact_period(
    fact: dict[str, Any],
    *,
    instant_ok: bool = False,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, Form] | None:
    """``(period_start, period_end, filed, form)`` for a fact, or None if unusable.

    Requires ``end`` (the fiscal close) plus ``filed`` (the PIT anchor); a fact
    missing either cannot be placed on the fiscal axis. ``end`` of
    ``1999-12-31`` is SEC's null-date sentinel.

    Balance-sheet facts are **instants**: they carry ``end`` but no ``start``
    (a point-in-time stock, unlike a duration flow). ``instant_ok`` admits them
    and collapses the span to a single day — ``period_start == period_end`` —
    which is what makes ``SeriesPIT.spans()`` able to distinguish a stock from a
    flow. Without the flag an instant is rejected, matching the original
    duration-only contract for income/cashflow tags.
    """
    start, end, filed = fact.get("start"), fact.get("end"), fact.get("filed")
    if not end or not filed or str(end).startswith(_NULL_END_PREFIX):
        return None
    if not start:
        if not instant_ok:
            return None
        start = end
    form = _FORM_MAP.get(str(fact.get("form", "")))
    if form is None:
        return None
    return (
        parse_timestamp(str(start)),
        parse_timestamp(str(end)),
        parse_timestamp(str(filed)),
        form,
    )


#: Statements whose facts are instants (stocks) rather than spans (flows).
#: A balance sheet is a snapshot at a fiscal close, so SEC reports ``end``
#: without ``start``; income and cashflow items are durations over a period.
_INSTANT_STATEMENTS: frozenset[Statement] = frozenset({"balance"})


def _rows_for_tag(
    ticker: str,
    payload: dict[str, Any],
    tag: str,
    statement: Statement,
    field: str,
) -> list[FundamentalRow]:
    """Rows for one tag, deduped on ``(start, end, filed)``.

    SEC repeats a fact once per unit/context; those repeats state the same
    datum and must not become duplicate rows.
    """
    seen: set[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = set()
    rows: list[FundamentalRow] = []
    for value, fact in _facts_for_tag(payload, tag):
        period = _fact_period(fact, instant_ok=statement in _INSTANT_STATEMENTS)
        if period is None:
            continue
        start, end, filed, form = period
        key = (start, end, filed)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            FundamentalRow(
                ticker=ticker,
                statement=statement,
                field=field,
                value=value,
                period_start=start,
                period_end=end,
                filed=filed,
                form=form,
            )
        )
    return rows


def sec_payload_to_rows(ticker: str, payload: dict[str, Any]) -> list[FundamentalRow]:
    """Map a companyfacts payload to sparse rows for ``ticker``.

    For every ``(statement, field)`` in :data:`TAG_MAP` the first tag with any
    usable facts supplies that field's rows; later tags are ignored, so a filer
    reporting both a specific and a coarse revenue tag does not double-count the
    same period. Free cash flow is derived per period (``ocf - |capex|``) rather
    than shipped as a separate tag, since SEC has no FCF concept.

    Returns rows in tag-map order (stable, and independent of payload key order
    for the statement/field pairs we read).
    """
    symbol = ticker.upper()
    rows: list[FundamentalRow] = []
    for (statement, field), tags in TAG_MAP.items():
        rows.extend(_first_tag_rows(symbol, payload, tags, statement, field))
    rows.extend(_free_cash_flow_rows(symbol, rows))
    return rows


def _first_tag_rows(
    ticker: str,
    payload: dict[str, Any],
    tags: tuple[str, ...],
    statement: Statement,
    field: str,
) -> list[FundamentalRow]:
    """Rows for the first tag in ``tags`` that reports any usable fact."""
    for tag in tags:
        rows = _rows_for_tag(ticker, payload, tag, statement, field)
        if rows:
            return rows
    return []


def _free_cash_flow_rows(
    ticker: str,
    rows: list[FundamentalRow],
) -> list[FundamentalRow]:
    """Derived FCF rows: ``operating_cash_flow - |capex|`` per fiscal period.

    Only periods where both components were filed for the same span produce a
    row — a half-computed FCF would silently misread as a real number. Derived
    rows inherit the *later* of the two component filings: the number did not
    exist publicly until both inputs were published.
    """
    ocf = {
        (r.period_start, r.period_end): r
        for r in rows
        if r.statement == "cashflow" and r.field == "operating_cash_flow"
    }
    capex = {
        (r.period_start, r.period_end): r
        for r in rows
        if r.statement == "cashflow" and r.field == "capex"
    }
    out: list[FundamentalRow] = []
    for key, ocf_row in ocf.items():
        capex_row = capex.get(key)
        if capex_row is None:
            continue
        out.append(
            FundamentalRow(
                ticker=ticker,
                statement="cashflow",
                field="free_cash_flow",
                value=ocf_row.value - abs(capex_row.value),
                period_start=key[0],
                period_end=key[1],
                filed=max(ocf_row.filed, capex_row.filed),
                form=ocf_row.form,
            )
        )
    return out


def is_annual(form: Form) -> bool:
    """True for annual-report forms (10-K / 20-F / 40-F)."""
    return form in _ANNUAL_FORMS


__all__ = ["TAG_MAP", "sec_payload_to_rows", "is_annual"]
