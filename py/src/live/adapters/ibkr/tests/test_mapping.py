"""Mapping: raw IBKR JSON -> typed records, with malformed rows warned not fatal."""

from __future__ import annotations

import pandas as pd

from src.exec.types import OrderSide
from src.live.adapters.ibkr.mapping import (
    parse_executions,
    parse_positions,
    parse_summary,
)


def test_parse_positions_reads_signed_qty_and_symbol() -> None:
    positions, warnings = parse_positions(
        [
            {
                "acctId": "DU1",
                "conid": 265598,
                "contractDesc": "AAPL",
                "position": -12.0,
                "avgCost": 180.5,
                "currency": "USD",
            }
        ]
    )
    assert warnings == ()
    assert len(positions) == 1
    assert positions[0].symbol == "AAPL"
    assert positions[0].qty == -12.0  # net-per-instrument, signed
    assert positions[0].avg_cost == 180.5


def test_parse_positions_skips_unidentifiable_and_zero() -> None:
    positions, warnings = parse_positions(
        [{"position": 3}, {"conid": 1, "contractDesc": "MSFT", "position": 0}]
    )
    assert positions == ()
    assert len(warnings) == 2
    assert "no conid or symbol" in warnings[0]
    assert "net-zero" in warnings[1]


def test_parse_positions_tolerates_string_numbers() -> None:
    positions, _ = parse_positions(
        [
            {
                "conid": "265598",
                "contractDesc": "AAPL",
                "position": "5",
                "avgCost": "10.25",
            }
        ]
    )
    assert positions[0].qty == 5.0
    assert positions[0].avg_cost == 10.25


def test_parse_positions_keeps_mkt_price() -> None:
    positions, _ = parse_positions(
        [{"conid": 1, "contractDesc": "AAPL", "position": 5, "mktPrice": 123.5}]
    )
    assert positions[0].mkt_price == 123.5


def test_parse_executions_canonicalises_float_order_id() -> None:
    """A float-shaped order_id and its int form mint the SAME lot handle."""
    records, _ = parse_executions(
        [
            {
                "execution_id": "e1",
                "order_id": "97932.0",
                "conid": 265598,
                "side": "B",
                "size": 1,
                "trade_time_r": 1,
            },
            {
                "execution_id": "e2",
                "order_id": 97932,
                "conid": 265598,
                "side": "B",
                "size": 1,
                "trade_time_r": 1,
            },
        ]
    )
    assert [e.order_id for e in records] == ["97932", "97932"]


def test_parse_executions_keeps_non_numeric_order_id() -> None:
    records, _ = parse_executions(
        [
            {
                "execution_id": "e",
                "order_id": "o1",
                "conid": 265598,
                "side": "B",
                "size": 1,
                "trade_time_r": 1,
            }
        ]
    )
    assert records[0].order_id == "o1"


def test_parse_summary_amounts_and_fallback() -> None:
    summary, warnings = parse_summary(
        {
            "netliquidation": {"amount": 51000.0, "currency": "USD"},
            "availablefunds": {"amount": 4000.0},
        }
    )
    assert summary.net_liquidation == 51000.0
    assert summary.total_cash == 4000.0  # falls back to availablefunds
    assert summary.currency == "USD"
    assert warnings == ()


def test_parse_summary_missing_fields_warn_and_zero() -> None:
    summary, warnings = parse_summary({})
    assert (summary.net_liquidation, summary.total_cash) == (0.0, 0.0)
    assert len(warnings) == 2


def test_parse_executions_happy_and_side_spellings() -> None:
    records, warnings = parse_executions(
        [
            {
                "execution_id": "e1",
                "order_id": "o1",
                "conid": 265598,
                "order_ref": "abc12345-deadbeef-20240101T0900-000",
                "symbol": "AAPL",
                "side": "BOT",
                "size": "10",
                "price": "100.5",
                "commission": "1",
                "trade_time_r": 1704187800000,
            },
            {
                "execution_id": "e2",
                "order_id": "o1",
                "conid": 265598,
                "order_ref": "abc12345-deadbeef-20240101T0900-000",
                "symbol": "AAPL",
                "side": "SLD",
                "size": 4,
                "price": 110,
                "trade_time_r": 1704191400000,
            },
        ]
    )
    assert warnings == ()
    assert [e.side for e in records] == [OrderSide.BUY, OrderSide.SELL]
    assert records[0].qty == 10.0 and records[0].price == 100.5
    assert records[0].ts == pd.Timestamp(1704187800000, unit="ms", tz="UTC")


def test_parse_executions_skips_malformed_rows() -> None:
    records, warnings = parse_executions(
        [
            {
                "execution_id": "a",
                "side": "B",
                "size": 1,
                "trade_time_r": 1,
            },  # no order_id
            {
                "execution_id": "b",
                "order_id": "o",
                "conid": 265598,
                "side": "?",
                "size": 1,
                "trade_time_r": 1,
            },
            {
                "execution_id": "c",
                "order_id": "o",
                "conid": 265598,
                "side": "B",
                "size": 0,
                "trade_time_r": 1,
            },
            {
                "execution_id": "d",
                "order_id": "o",
                "conid": 265598,
                "side": "B",
                "size": 1,
                "trade_time_r": 0,
            },
        ]
    )
    assert records == ()
    assert len(warnings) == 4
