"""The ``broker`` field on a strategy config: default, known values, rejection."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from src.bt import load_strategy
from src.bt.types import StrategyConfig

_BASE_JSON = {
    "name": "t",
    "strategy_type": "momentum_compression_breakout_dsl",
    "symbols": ["AAPL"],
    "initial_capital": 10000.0,
    "commission": 0.5,
    "warmup": "0d",
    "trading_start": "2025-01-01",
    "trading_end": "2025-12-31",
    "bars": ["1d"],
    "strategy_params": {},
}


def _base() -> StrategyConfig:
    return StrategyConfig(
        name="t",
        strategy_type="momentum_compression_breakout_dsl",
        symbols=["AAPL"],
        initial_capital=10000.0,
        commission=0.5,
        warmup="0d",
        trading_start="2025-01-01",
        trading_end="2025-12-31",
        bars=["1d"],
        strategy_params={},
    )


def test_broker_defaults_to_sim() -> None:
    assert _base().broker == "sim"


def test_load_strategy_parses_a_named_broker(tmp_path) -> None:
    path = tmp_path / "s.json"
    path.write_text(json.dumps({**_BASE_JSON, "broker": "ibkr"}))
    assert load_strategy(str(path)).broker == "ibkr"


def test_unknown_broker_fails_loudly() -> None:
    with pytest.raises(ValueError):
        replace(_base(), broker="robinhood")
