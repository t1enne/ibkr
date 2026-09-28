"""Tests for the param sweep module (deep merge, grid expansion, ranking)."""

from src.bt.sweep import build_config, grid_combos
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


def test_grid_combos_scalar_only_single():
    assert grid_combos({"position_size": 0.8}) == [{"position_size": 0.8}]


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


def test_grid_combos_deep_merge_scalar_and_sweep():
    merge = {
        "stop_loss": 0.0,  # scalar -> fixed override
        "strategy_params": {"ma_slow": [9, 14]},  # list -> sweep
    }
    combos = grid_combos(merge)
    assert len(combos) == 2
    # scalar constant across all combos
    assert all(c["stop_loss"] == 0.0 for c in combos)


def test_build_config_deep_merges_top_level():
    cfg = _cfg()
    out = build_config(cfg, {"initial_capital": 5000.0})
    assert out.initial_capital == 5000.0
    # untouched config fields preserved (sizing/SL/TP now live in strategy_params)
    assert out.strategy_params["position_size"] == cfg.strategy_params["position_size"]
    assert out is not cfg


def test_build_config_merges_strategy_params():
    cfg = _cfg()
    out = build_config(cfg, {"strategy_params": {"sma_slow": 150, "base": 99}})
    assert out.strategy_params["sma_slow"] == 150
    assert out.strategy_params["base"] == 99
    # untouched strategy params preserved
    assert out.strategy_params != cfg.strategy_params
    assert out.symbols == cfg.symbols


def test_build_config_mixed_levels():
    cfg = _cfg()
    out = build_config(cfg, {"initial_capital": 5000.0, "strategy_params": {"x": 5}})
    assert out.initial_capital == 5000.0
    assert out.strategy_params["x"] == 5
    assert out.strategy_params["base"] == 1
    assert out.strategy_params["position_size"] == cfg.strategy_params["position_size"]


def test_build_config_target_weights_list_roundtrips():
    # A list-valued config field dumps/rebuilds as a list (was a tuple via params).
    cfg = _cfg()
    out = build_config(cfg, {"strategy_params": {"target_weights": [0.6, 0.4]}})
    assert out.strategy_params["target_weights"] == [0.6, 0.4]
