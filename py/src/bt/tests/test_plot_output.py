"""``render_plot_json`` plot-spec splicing: regression, nesting, failure guard.

The regression guarantee is the load-bearing assertion: a strategy without a
``plot`` (or ``plot=False``) yields a payload with **no** ``plot`` key anywhere,
byte-identical to the ``plot=False`` path. The remainder covers a spec landing
under the right symbol/frame, JSON round-trip, and a raising ``plot`` being
dropped without failing the payload.
"""

from __future__ import annotations

import json
from types import ModuleType, SimpleNamespace

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

import pytest

from src.bt.engine.backtest import Backtest, run_backtest
from src.bt.engine.handlers import default_execution_handler, default_risk_handler
from src.bt.engine.utils import candle_generator
from src.bt.output import render_plot_json
from src.bt.strategies.dsl import StrategyContext, strategy
from src.bt.strategies.ta_context import init_ta
from src.bt.strategies.types import Marker, Panel, PlotSpec
from src.bt.types import StrategyConfig

if TYPE_CHECKING:
    pass


SYMBOL = "AAPL"
BARS = 120


def _feed(periods: int = BARS) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=periods, freq="D")
    closes = [100.0 + float(np.sin(i / 3.0)) * 5 for i in range(periods)]
    data: dict[tuple[str, str], list[float]] = {
        (SYMBOL, "open"): closes,
        (SYMBOL, "high"): [c + 1 for c in closes],
        (SYMBOL, "low"): [c - 1 for c in closes],
        (SYMBOL, "close"): closes,
        (SYMBOL, "volume"): [1000.0] * periods,
    }
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _cfg(strategy_type: str) -> StrategyConfig:
    return StrategyConfig(
        name="plot-output",
        strategy_type=strategy_type,
        symbols=[SYMBOL],
        initial_capital=10000.0,
        commission=0.0,
        warmup="0d",
        trading_start="2024-01-02",
        trading_end="2024-06-01",
        bars=["1d"],
        strategy_params={},
        benchmark_symbols=[],
    )


@pytest.fixture
def _register_module(monkeypatch: pytest.MonkeyPatch):
    """Register a temp strategy module for one test.

    ``resolve_params`` is monkeypatched in **both** modules that bind it
    (``src.bt.strategies`` and the engine that imported it) so the run resolves
    the temp module's params regardless of import/collection ordering. ``plot``
    itself reaches the payload through ``plot_fn_for`` (which reads the module
    off ``sys.modules`` via the same patched lookup).
    """
    import sys

    import src.bt.engine.backtest as engine_mod
    import src.bt.strategies as strat_pkg

    temp: dict[str, ModuleType] = {}

    def _resolve(name: str, raw: dict) -> object:
        mod = temp.get(name)
        if mod is None:
            return real_resolve(name, raw)
        params_cls = getattr(mod, "Params", None)
        return params_cls.from_dict(raw) if params_cls is not None else raw

    def _init(name: str) -> object:
        mod = temp.get(name)
        return mod if mod is not None else real_init(name)

    real_resolve = strat_pkg.resolve_params
    real_init = strat_pkg.init_strat

    def register(name: str, plot_fn: object | None) -> ModuleType:
        mod = ModuleType(name)
        setattr(mod, "STRATEGY_TYPE", name)
        if plot_fn is not None:
            setattr(mod, "plot", plot_fn)
        sys.modules[name] = mod

        def on_candle(ctx: StrategyContext) -> None:  # noqa: ARG001
            return None

        on_candle.__module__ = name
        setattr(mod, "on_candle", strategy(bars="1d")(on_candle))
        temp[name] = mod
        monkeypatch.setattr(strat_pkg, "resolve_params", _resolve)
        monkeypatch.setattr(strat_pkg, "init_strat", _init)
        monkeypatch.setattr(engine_mod, "resolve_params", _resolve)
        return mod

    return register


def _run(adapter: object, strategy_type: str) -> object:
    data = _feed()
    bt = Backtest(_cfg(strategy_type))
    gen = candle_generator(data, bt.config)
    ta = init_ta(data, bt.config.symbols, bt.config.bars[0])
    results, _ = run_backtest(
        bt,
        gen,
        default_execution_handler(),
        default_risk_handler(),
        strategy_mod=SimpleNamespace(on_candle=adapter),
        ta=ta,
    )
    return results


def _noop_adapter():
    @strategy(bars="1d")
    def on_candle(ctx: StrategyContext) -> None:  # noqa: ARG001
        return None

    return on_candle


def _spec_plot(ctx: StrategyContext, params: object) -> PlotSpec:  # noqa: ARG001
    ts = int(pd.Timestamp(ctx.candle.timestamp).value // 1_000_000)
    return PlotSpec(
        panels=(Panel(series=((ts, 50.0),), name="MFI(14)", hlines=(20.0, 80.0)),),
        markers=(Marker(ts=ts, price=ctx.candle.close, kind="pivot_low"),),
    )


def _raising_plot(ctx: StrategyContext, params: object) -> PlotSpec:  # noqa: ARG001
    raise ValueError("boom")


# --- regression: no plot fn -> no "plot" key --------------------------------


def test_payload_without_plot_fn_matches_plot_false() -> None:
    """THE regression test: an existing strategy's payload gains no ``plot`` key."""
    results = _run(_noop_adapter(), "mfi_divergence_dsl")  # registered, no plot
    with_spec = render_plot_json(results)
    without = render_plot_json(results, plot=False)
    assert with_spec == without
    assert "plot" not in json.dumps(with_spec)


def test_plot_false_suppresses_an_existing_plot(_register_module) -> None:
    mod = _register_module("_plot_output_suppressed", _spec_plot)
    results = _run(mod.on_candle, "_plot_output_suppressed")
    assert "plot" not in json.dumps(render_plot_json(results, plot=False))
    assert "plot" in json.dumps(render_plot_json(results))


# --- spec lands under the right frame and round-trips -----------------------


def test_plot_spec_nested_under_signal_interval_frame(_register_module) -> None:
    mod = _register_module("_plot_output_nested", _spec_plot)
    results = _run(mod.on_candle, "_plot_output_nested")
    payload = render_plot_json(results)
    frame = next(f for f in payload["symbols"][SYMBOL] if f["interval"] == "1d")
    assert "plot" in frame
    assert frame["plot"]["panels"][0]["name"] == "MFI(14)"
    reparsed = json.loads(json.dumps(payload))
    assert reparsed["symbols"][SYMBOL][0]["plot"]["markers"][0]["kind"] == "pivot_low"


def test_plot_failure_is_dropped_not_fatal(_register_module) -> None:
    mod = _register_module("_plot_output_raising", _raising_plot)
    results = _run(mod.on_candle, "_plot_output_raising")
    payload = render_plot_json(results)
    assert payload["metrics"] and payload["symbols"]
    for frames in payload["symbols"].values():
        assert all("plot" not in f for f in frames)
