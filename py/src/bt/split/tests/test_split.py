"""Tests for walk-forward / single-anchor split logic."""

from __future__ import annotations

import pandas as pd
import pytest
import src.bt.split as split_mod
import src.bt.window as window_mod

from src.bt.split import (
    anchor_split,
)
from src.bt.state import PortfolioResult
from src.bt.types import StrategyConfig
from src.utils import parse_timestamp


def _cfg(
    trading_start: str = "2015-01-02",
    trading_end: str = "2025-12-31",
    warmup: str = "0d",
) -> StrategyConfig:
    return StrategyConfig(
        name="t",
        strategy_type="sector_mean_reversion_trail",
        symbols=["XLB", "XLV", "XLY", "XLU", "SPY"],
        initial_capital=50000,
        commission=0.1,
        warmup=warmup,
        trading_start=trading_start,
        trading_end=trading_end,
        bars=["1d"],
        strategy_params={
            "momentum_lookback": 30,
            "position_size": 0.45,
            "stop_loss": 0.15,
            "take_profit": 0.3,
        },
    )


def _fake_result(sharpe: float) -> PortfolioResult:
    import pandas as pd

    eq = pd.Series([1000.0, 1100.0])
    from src.bt.state import Trade, TradeStatus, ActionType

    win = Trade(
        entry_time=parse_timestamp("2020-01-02"),
        entry_price=10.0,
        exit_time=parse_timestamp("2020-02-01"),
        exit_price=11.0,
        last_price=11.0,
        symbol="XLB",
        position=ActionType.long,
        qty=1.0,
        stop_loss=9.0,
        take_profit=12.0,
        pnl=1.0,
        status=TradeStatus.closed,
    )
    return PortfolioResult(
        total_return=0.1,
        sharpe_ratio=sharpe,
        trades=(win,),
        equity_curve=eq,
        annual_return=0.1,
        max_drawdown=-0.05,
        calmar_ratio=2.0,
    )


def test_anchor_split_is_end_past_trading_end_raises() -> None:
    cfg = _cfg(trading_end="2020-12-31")
    with pytest.raises(ValueError):
        anchor_split(cfg, parse_timestamp("2020-12-31"))  # equal to end
    with pytest.raises(ValueError):
        anchor_split(cfg, parse_timestamp("2021-01-01"))  # past end


def test_anchor_split_is_end_before_start_raises() -> None:
    cfg = _cfg()
    with pytest.raises(ValueError):
        anchor_split(cfg, parse_timestamp("2015-01-01"))


def _fold(i: int) -> split_mod.TestFold:
    offset = pd.offsets.DateOffset(years=i)
    return split_mod.TestFold(
        index=i,
        is_start=parse_timestamp("2015-01-02"),
        is_end=parse_timestamp("2020-01-01") + offset,
        oos_start=parse_timestamp("2021-01-01") + offset,
        oos_end=parse_timestamp("2022-01-01") + offset,
    )


def _candle_df(
    start: str, end: str, freq: str = "D", symbols: tuple[str, ...] = ("A",)
) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq=freq)
    syms = {
        s: pd.DataFrame({"open": range(1, len(idx) + 1)}, index=idx) for s in symbols
    }
    return pd.concat(list(syms.values()), axis=1, keys=list(symbols))


def test_window_df_keeps_head_drops_future() -> None:
    """Slicing must keep the warmup head but truncate past trading_end so the
    engine can't process future data (the look-ahead close/model/update leak)."""
    df = _candle_df("2020-01-01", "2020-01-10")
    sliced = window_mod.window_df(df, parse_timestamp("2020-01-05"))
    assert list(sliced.index) == list(
        pd.date_range("2020-01-01", "2020-01-05", freq="D")
    )


def test_window_df_keeps_model_warmup_head() -> None:
    """The pre-window head must survive slicing so the model warms up on prior
    history (trading_start is NOT the slice boundary)."""
    df = _candle_df("2015-01-01", "2020-01-10")
    sliced = window_mod.window_df(df, parse_timestamp("2020-01-05"))
    assert sliced.index[0] == pd.Timestamp("2015-01-01")  # warmup head intact
    assert sliced.index[-1] == pd.Timestamp("2020-01-05")
    assert len(sliced.index) == (len(df.index) - 5)


def test_window_df_multi_symbol_columns_preserved() -> None:
    """Multi-symbol MultiIndex columns must be unchanged after slicing."""
    df = _candle_df("2020-01-01", "2020-01-10", symbols=("A", "B"))
    sliced = window_mod.window_df(df, parse_timestamp("2020-01-05"))
    assert sliced.columns.equals(df.columns)
    assert list(sliced.index) == list(
        pd.date_range("2020-01-01", "2020-01-05", freq="D")
    )
