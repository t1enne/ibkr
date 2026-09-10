"""SEC EDGAR fundamentals — ingest, sparse PIT storage, and series assembly.

Public surface:

* :func:`download_fundamentals` / :func:`cik_for_ticker` — SEC HTTP (cached).
* :func:`sec_payload_to_rows` — payload -> sparse canonical rows (pure).
* :func:`insert_fundamentals` / :func:`bootstrap` — local DB write path.
* :func:`load_stated` / :func:`as_first_stated` / :func:`build_series` — reads.

The DSL read surface lives at
:mod:`src.bt.strategies.fundamentals_context` (``ctx.fundamentals``).
"""

from src.data.fundamentals.normalize import TAG_MAP, sec_payload_to_rows
from src.data.fundamentals.query import (
    as_first_stated,
    build_series,
    load_stated,
    rows_to_snapshot,
    snapshot_to_rows,
)
from src.data.fundamentals.schema import (
    BalanceSheet,
    CashFlow,
    Form,
    FundamentalRow,
    FundamentalSchema,
    Income,
    Statement,
    StatementSnapshot,
    bootstrap,
    insert_fundamentals,
    statement_of,
)
from src.data.fundamentals.sec_client import (
    cik_for_ticker,
    download_fundamentals,
    fetch_companyfacts,
)

__all__ = [
    "TAG_MAP",
    "sec_payload_to_rows",
    "as_first_stated",
    "build_series",
    "load_stated",
    "rows_to_snapshot",
    "snapshot_to_rows",
    "BalanceSheet",
    "CashFlow",
    "Form",
    "FundamentalRow",
    "FundamentalSchema",
    "Income",
    "Statement",
    "StatementSnapshot",
    "bootstrap",
    "insert_fundamentals",
    "statement_of",
    "cik_for_ticker",
    "download_fundamentals",
    "fetch_companyfacts",
]
