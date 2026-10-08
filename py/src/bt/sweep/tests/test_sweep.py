"""Tests for the param sweep module (deep merge, grid expansion, ranking)."""

from src.bt.sweep import grid_combos
from src.bt.types import StrategyConfig


def _cfg(**strategy_params) -> StrategyConfig:
    return StrategyConfig(
        name="test",
        strategy_type="dummy",
        symbols=["A", "B"],
        initial_capital=100000.0,
        commission=0.05,
        warmup="0d",
        trading_start="2020-01-02",
        trading_end="2021-01-01",
        bars=["1d"],
        strategy_params={
            "position_size": 0.95,
            "stop_loss": 0.5,
            "take_profit": 0.8,
            "base": 1,
            **strategy_params,
        },
        benchmark_symbols=["A"],
    )


def test_grid_combos_empty():
    assert grid_combos({}) == [{}]


def test_grid_combos_cartesian_across_levels():
    merge = {
        "position_size": [0.8, 0.9],
        "strategy_params": {"sma_slow": [100, 200]},
    }
    combos = grid_combos(merge)
    assert len(combos) == 4
    assert {c["position_size"] for c in combos} == {0.8, 0.9}
    assert {c["strategy_params"]["sma_slow"] for c in combos} == {100, 200}
    # nested overrides preserved together per combo
    assert all(
        c["position_size"] == c0 and c["strategy_params"]["sma_slow"] == c1
        for c, (c0, c1) in zip(
            combos,
            [(0.8, 100), (0.8, 200), (0.9, 100), (0.9, 200)],
        )
    )
