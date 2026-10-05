"""Tests for the screen -> LiveSignal bridge (pure mapping, stubbed driver)."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

import src.live.signals as signals
from src.bt.engine.candle_store import CandleStore
from src.bt.screen import ScreenRow, ScreenRun
from src.bt.state import BacktestState, PortfolioState
from src.bt.types import StrategyConfig

BASE_IV = "1d"


def _ts(value: str) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(value))


def _store(frames: dict[str, pd.DataFrame]) -> CandleStore:
    """Build a real CandleStore (numpy column arrays) from per-symbol frames."""
    rows = {}
    for sym, df in frames.items():
        n = len(df)
        rows[(sym, BASE_IV)] = {
            "timestamp": df.index.to_numpy(dtype="datetime64[ms]"),
            "open": df["open"].to_numpy(dtype=float),
            "high": df["high"].to_numpy(dtype=float),
            "low": df["low"].to_numpy(dtype=float),
            "close": df["close"].to_numpy(dtype=float),
            "volume": df["volume"].to_numpy(dtype=float),
            "_len": np.array([n], dtype=np.int64),
        }
    return CandleStore(rows)


def _frame(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    c = np.array(closes, dtype=float)
    return pd.DataFrame(
        {"open": c, "high": c, "low": c, "close": c, "volume": np.ones_like(c)},
        index=idx,
    )


def _state(candles: CandleStore) -> BacktestState:
    portfolio = PortfolioState(
        cash=0.0, positions={}, trades=(), equity_curve=(), initial_capital=0.0
    )
    return BacktestState(
        portfolio=portfolio,
        timestamp=_ts("2024-06-03"),
        pending_signals={},
        risk_events=(),
        candles=candles,
    )


def _config() -> StrategyConfig:
    return StrategyConfig(
        name="t",
        strategy_type="momentum",
        symbols=["AAA", "BBB", "CCC", "DDD"],
        initial_capital=100_000.0,
        commission=0.5,
        warmup="1y",
        trading_start="2023-01-01",
        trading_end="2024-06-03",
        bars=[BASE_IV],
        strategy_params={},
    )


def _stub(monkeypatch, rows: tuple[ScreenRow, ...], state: BacktestState) -> None:
    monkeypatch.setattr(
        signals,
        "run_screen_from_strategy",
        lambda *a, **k: ScreenRun(rows=rows, state=state, config=_config()),
    )
    monkeypatch.setattr(signals, "load_strategy", lambda *a, **k: _config())


def test_live_signals_maps_actionable_rows(monkeypatch) -> None:
    sig_ts = _ts("2024-06-01")
    ts = _ts("2024-06-03")
    rows = (
        ScreenRow(
            symbol="AAA",
            action="long",
            score=1.0,
            signals=("ema_cross",),
            ts=ts,
            sig_ts=sig_ts,
        ),
        ScreenRow(
            symbol="BBB",
            action="short",
            score=0.8,
            signals=("mfi",),
            ts=ts,
            sig_ts=sig_ts,
        ),
        ScreenRow(
            symbol="CCC", action="flat", score=0.0, signals=(), ts=ts, sig_ts=None
        ),
        ScreenRow(
            symbol="DDD", action="long", score=1.0, signals=("x",), ts=ts, sig_ts=sig_ts
        ),
    )
    candles = _store({"AAA": _frame([1.0, 2.0, 42.5]), "BBB": _frame([10.0, 9.0])})
    _stub(monkeypatch, rows, _state(candles))

    out = signals.live_signals("ignored.json")

    # flat (CCC) dropped; DDD dropped (no frame).
    assert tuple(s.symbol for s in out) == ("AAA", "BBB")

    aaa = out[0]
    assert aaa.action == "long"
    assert aaa.score == 1.0
    assert aaa.reasons == ("ema_cross",)
    assert aaa.signal_ts == sig_ts
    assert aaa.price == 42.5  # last close of AAA's base frame
    assert aaa.qty == 0.0

    bbb = out[1]
    assert bbb.action == "short"
    assert bbb.price == 9.0
    assert bbb.qty == 0.0


def test_live_signals_forwards_max_age_days_as_none(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def _capture(config_path: str, max_age_days: int | None = None):
        seen["max_age_days"] = max_age_days
        return ScreenRun(rows=(), state=_state(_store({})), config=_config())

    monkeypatch.setattr(signals, "run_screen_from_strategy", _capture)
    monkeypatch.setattr(signals, "load_strategy", lambda *a, **k: _config())

    # The driver is ALWAYS asked for every row (its own age filter drops the
    # flat rows a close is reconstructed from); the filter is local.
    assert signals.live_signals("ignored.json", max_age_days=7) == ()
    assert seen["max_age_days"] is None


def test_live_signals_reconstructs_close_from_flat_with_sig_ts(monkeypatch) -> None:
    ts = _ts("2024-06-03")
    sig_ts = _ts("2024-06-02")
    rows = (
        ScreenRow(
            symbol="AAA",
            action="flat",
            score=0.0,
            signals=("close",),
            ts=ts,
            sig_ts=sig_ts,
        ),
        # flat with no sig_ts -> never signalled -> HOLD (dropped).
        ScreenRow(
            symbol="BBB", action="flat", score=0.0, signals=(), ts=ts, sig_ts=None
        ),
    )
    candles = _store({"AAA": _frame([1.0, 2.0]), "BBB": _frame([1.0, 2.0])})
    _stub(monkeypatch, rows, _state(candles))

    out = signals.live_signals("ignored.json")

    assert tuple((s.symbol, s.action) for s in out) == (("AAA", "close"),)
    assert out[0].signal_ts == sig_ts
    assert out[0].reasons == ("close",)
    assert out[0].price == 2.0


def test_live_signals_maps_driver_close_action(monkeypatch) -> None:
    """The driver emits an explicit ``close`` action (ctx.close); keep it."""
    ts = _ts("2024-06-03")
    sig_ts = _ts("2024-06-02")
    rows = (
        ScreenRow(
            symbol="AAA",
            action="close",
            score=0.8,
            signals=("trail exit",),
            ts=ts,
            sig_ts=sig_ts,
        ),
    )
    candles = _store({"AAA": _frame([1.0, 2.0])})
    _stub(monkeypatch, rows, _state(candles))

    out = signals.live_signals("ignored.json")

    assert tuple((s.symbol, s.action) for s in out) == (("AAA", "close"),)
    assert out[0].price == 2.0


def test_live_signals_local_age_filter_drops_stale(monkeypatch) -> None:
    ts = _ts("2024-06-03")
    fresh = ScreenRow(
        symbol="AAA",
        action="long",
        score=1.0,
        signals=("x",),
        ts=ts,
        sig_ts=_ts("2024-06-02"),
    )
    stale = ScreenRow(
        symbol="BBB",
        action="long",
        score=0.8,
        signals=("x",),
        ts=ts,
        sig_ts=_ts("2024-05-01"),
    )
    candles = _store({"AAA": _frame([1.0]), "BBB": _frame([1.0])})
    _stub(monkeypatch, (fresh, stale), _state(candles))

    out = signals.live_signals("ignored.json", max_age_days=5)
    assert tuple(s.symbol for s in out) == ("AAA",)


def test_live_signals_zero_max_age_disables_local_filter(monkeypatch) -> None:
    ts = _ts("2024-06-03")
    stale = ScreenRow(
        symbol="BBB",
        action="long",
        score=0.8,
        signals=("x",),
        ts=ts,
        sig_ts=_ts("2024-05-01"),
    )
    candles = _store({"BBB": _frame([1.0])})
    _stub(monkeypatch, (stale,), _state(candles))

    assert tuple(s.symbol for s in signals.live_signals("ignored.json", 0)) == ("BBB",)


def test_live_signals_empty_when_no_rows(monkeypatch) -> None:
    _stub(monkeypatch, (), _state(_store({})))

    assert signals.live_signals("ignored.json") == ()
