"""Tests for the output-boundary renderers (JSON/JSONL from BacktestResults)."""

import pandas as pd

from src.bt.output import (
    trade_json,
    equity_points,
    render_result_json,
    render_result_jsonl,
    render_plot_json,
    _frame_interval,
)
from src.bt.state import (
    ActionType,
    PortfolioResult,
    Trade,
    TradeStatus,
    TradeExitReason,
)


def _ts(s: str) -> pd.Timestamp:
    """Parse a datestring into a non-NaT Timestamp for typing clarity."""
    t = pd.Timestamp(s)
    assert isinstance(t, pd.Timestamp) and not pd.isna(t)
    return t


def _trade(status=TradeStatus.closed) -> Trade:
    return Trade(
        entry_time=_ts("2021-01-05 09:30:00"),
        entry_price=100.5,
        exit_time=_ts("2021-01-06 10:00:00"),
        exit_price=105.0,
        last_price=105.0,
        symbol="AAPL",
        position=ActionType.long,
        qty=10.0,
        stop_loss=95.0,
        take_profit=110.0,
        pnl=45.0,
        commission=0.5,
        slippage=0.1,
        reason="breakout",
        status=status,
        close_reason=TradeExitReason.tp,
        position_id="AAPL_123",
    )


def _pf():
    equity = pd.Series(
        [100000.0, 101000.0, 100500.0],
        index=pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]),
    )
    return PortfolioResult(
        total_return=0.005,
        sharpe_ratio=1.2,
        trades=(_trade(),),
        equity_curve=equity,
        annual_return=0.05,
    )


def test_trade_json_flattens_enums_and_value():
    j = trade_json(_trade())
    assert j["position"] == "long"  # ActionType enum -> plain value
    assert j["status"] == "closed"  # TradeStatus enum -> plain value
    assert j["close_reason"] == "tp"  # TradeExitReason enum -> plain value
    assert j["symbol"] == "AAPL"
    assert isinstance(j["pnl"], float)
    assert j["position_id"] == "AAPL_123"


def test_trade_json_unclosed_exit_is_none():
    t = _trade()
    t.exit_time = None
    t.exit_price = None
    j = trade_json(t)
    assert j["exit_time"] is None
    assert j["exit_price"] is None


def test_equity_points_dates_and_floats():
    pts = equity_points(_pf().equity_curve)
    assert len(pts) == 3
    assert pts[0]["ts"] == "2021-01-01 00:00:00"
    assert abs(pts[0]["equity"] - 100000.0) < 1e-9


def test_render_result_json_shape():
    r = render_result_json(type("R", (), {"pf": _pf(), "benchmark_curves": {}})())
    assert set(r) == {
        "metrics",
        "total_trades",
        "trades",
        "equity_curve",
        "benchmark_curves",
    }
    assert r["metrics"]["total_return"] == 0.005
    assert r["total_trades"] == 1
    assert len(r["trades"]) == 1
    # benchmark_curves keyed by symbol -> list of points
    assert r["benchmark_curves"] == {}


def test_render_result_json_trades_gated_by_flag():
    obj = type("R", (), {"pf": _pf(), "benchmark_curves": {}})()
    lean = render_result_json(obj, include_trades=False)
    assert "trades" not in lean
    assert lean["total_trades"] == 1  # count survives the omission
    assert set(lean) == {"metrics", "total_trades", "equity_curve", "benchmark_curves"}


def test_render_result_jsonl_ends_with_summary_record():
    obj = type("R", (), {"pf": _pf(), "benchmark_curves": {}})()
    lines = render_result_jsonl(obj)
    # 3 equity points + 1 summary
    assert len(lines) == 4
    assert set(lines[0]) == {"ts", "equity"}
    assert set(lines[-1]) == {"metrics", "total_trades", "trades"}
    assert lines[-1]["total_trades"] == 1


def test_render_result_jsonl_trades_gated_by_flag():
    obj = type("R", (), {"pf": _pf(), "benchmark_curves": {}})()
    last = render_result_jsonl(obj, include_trades=False)[-1]
    assert "trades" not in last
    assert last["total_trades"] == 1


def _candle_df(index, closes):
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [1000.0] * len(closes),
        },
        index=pd.to_datetime(index),
    )


def _fake_store():
    """CandleStore stand-in exposing Mapping keys + item lookup."""
    idx = ["2021-01-05 09:30:00", "2021-01-06 10:00:00", "2021-01-07 11:00:00"]
    closes = [100.0, 101.0, 103.0]
    return {
        ("AAPL", "1h"): _candle_df(idx, closes),
        ("AAPL", "4h"): _candle_df(["2021-01-05 10:00:00"], [99.0]),
    }


def test_render_plot_json_shape_and_interval_resolution():
    class FakeStore:
        def __init__(self, store):
            self._store = store

        def keys(self):
            return self._store.keys()

        def __getitem__(self, key):
            return self._store[key]

    data = FakeStore(_fake_store())
    obj = type("R", (), {"pf": _pf(), "data": data, "benchmark_curves": {}})()
    r = render_plot_json(obj)
    assert set(r) == {
        "metrics",
        "total_trades",
        "symbols",
        "trades",
        "equity_curve",
        "benchmark_curves",
    }
    assert r["total_trades"] == 1
    # two frames stored for AAPL (1h + 4h) come back as a list
    assert len(r["symbols"]["AAPL"]) == 2
    one_h = next(f for f in r["symbols"]["AAPL"] if f["interval"] == "1h")
    assert len(one_h["bars"]) == 3
    bar = one_h["bars"][1]
    assert len(bar) == 6 and bar[1] == 101.0 and bar[4] == 101.0
    # entry time on an index tick -> that frame's interval
    assert r["trades"][0]["interval"] == "1h"


def test_render_plot_json_trades_gated_by_flag():
    class FakeStore:
        def __init__(self, store):
            self._store = store

        def keys(self):
            return self._store.keys()

        def __getitem__(self, key):
            return self._store[key]

    obj = type(
        "R",
        (),
        {"pf": _pf(), "data": FakeStore(_fake_store()), "benchmark_curves": {}},
    )()
    lean = render_plot_json(obj, include_trades=False)
    assert "trades" not in lean
    assert lean["total_trades"] == 1
    assert "symbols" in lean  # chart geometry still present


def test_frame_interval_falls_back_to_first_frame():
    data = _fake_store()
    # a ts in no frame's index -> pick the symbol's first frame
    off = pd.Timestamp("2021-06-01")
    assert _frame_interval(type("R", (), {"data": data})(), "AAPL", off) == "1h"
