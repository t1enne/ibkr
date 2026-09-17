"""Tests for the pure dashboard render helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from src.bt.dashboard.render import (
    build_chart,
    fmt_metric,
    metrics_table,
    price_frame,
    trades_for_frame,
    trades_table,
)


def _payload() -> dict[str, Any]:
    return {
        "metrics": {"total_return": 0.25, "sharpe_ratio": 1.23456},
        "symbols": {
            "AAPL": [
                {
                    "interval": "1d",
                    "bars": [
                        [1704067200000, 1.0, 2.0, 0.5, 1.5, 100.0],
                        [1704153600000, 1.5, 3.0, 1.0, 2.5, 200.0],
                    ],
                }
            ]
        },
        "trades": [
            {
                "symbol": "AAPL",
                "position": "long",
                "qty": 10,
                "entry_time": 1704067200000,
                "entry_price": 1.2,
                "exit_time": 1704153600000,
                "exit_price": 2.4,
                "pnl": 12.0,
                "commission": 1.0,
                "slippage": 0.5,
                "close_reason": "tp",
                "status": "closed",
                "reason": "test",
                "interval": "1d",
            },
            {
                "symbol": "MSFT",
                "position": "short",
                "qty": 5,
                "entry_time": 1704067200000,
                "entry_price": 5.0,
                "exit_time": None,
                "exit_price": None,
                "interval": "1d",
            },
        ],
    }


def test_fmt_metric_percent():
    assert fmt_metric("total_return", 0.25) == "25.00%"
    assert fmt_metric("max_drawdown", -0.1) == "-10.00%"


def test_fmt_metric_plain_and_non_numeric():
    assert fmt_metric("sharpe_ratio", 1.5) == "1.500"
    assert fmt_metric("status", "ok") == "ok"
    assert fmt_metric("flag", True) == "True"


def test_metrics_table_orders_and_formats():
    pairs = metrics_table(_payload())
    labels = [k for k, _ in pairs]
    assert labels[0] == "total_return"
    values = dict(pairs)
    assert values["total_return"] == "25.00%"
    assert values["sharpe_ratio"] == "1.235"


def test_trades_table_present_columns():
    df = trades_table(_payload())
    assert len(df) == 2
    assert "exit_price" in df.columns
    assert df.iloc[0]["symbol"] == "AAPL"


def test_trades_table_empty_payload():
    df = trades_table({"trades": []})
    assert df.empty
    assert list(df.columns) == []


def test_price_frame_indexed_and_sorted():
    frame = _payload()["symbols"]["AAPL"][0]
    price = price_frame(frame)
    assert list(price.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(price.index, pd.DatetimeIndex)
    assert price.index.is_monotonic_increasing


def test_trades_for_frame_filters_symbol_and_interval():
    trades = trades_for_frame(_payload(), "AAPL", "1d")
    assert len(trades) == 1
    assert trades.iloc[0]["position"] == "long"
    assert trades_for_frame(_payload(), "AAPL", "1h").empty


def test_trades_for_frame_empty_payload():
    trades = trades_for_frame({"trades": []}, "AAPL", "1d")
    assert trades.empty
    assert "entry_price" in trades.columns


def test_build_chart_has_candles_and_markers():
    payload = _payload()
    frame = payload["symbols"]["AAPL"][0]
    price = price_frame(frame)
    trades = trades_for_frame(payload, "AAPL", "1d")
    fig = build_chart(price, trades)
    names = [t.name for t in fig.data]
    assert names[0] == "OHLC"
    assert "open long" in names
    assert fig.layout.xaxis.rangeslider.visible is False


def test_build_chart_empty_trades():
    frame = _payload()["symbols"]["AAPL"][0]
    price = price_frame(frame)
    empty = trades_for_frame({"trades": []}, "AAPL", "1d")
    fig = build_chart(price, empty)
    assert len(fig.data) == 1  # candles only, no markers
    assert fig.data[0].name == "OHLC"


def test_build_chart_no_pandas_import_regression():
    # candle x-axis must be datetimes, not raw ints
    frame = _payload()["symbols"]["AAPL"][0]
    price = price_frame(frame)
    fig = build_chart(price, trades_for_frame({"trades": []}, "AAPL", "1d"))
    xs = pd.to_datetime(list(fig.data[0].x))
    assert list(xs) == list(price.index)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
