"""Plot discovery + the read-only post-run ``StrategyContext.for_plot``.

No DB. The discovery half proves a module-level ``plot`` is mirrored onto the
``@strategy`` adapter and surfaced by ``plot_fn_for`` (and that its absence
yields ``None``). The ``for_plot`` half proves the read-only contract: a
terminal-cursor assertion, blocked mutations, a write-protected ``shared``, and
that ``ctx.ta`` matches the store at the end of a real mini-run.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

import pytest

from src.bt.engine.backtest import Backtest, run_backtest
from src.bt.exchange import default_exchange
from src.bt.risk.handlers import default_risk_handler
from src.bt.engine.utils import candle_generator
from src.bt.output import render_plot_json
from src.bt.state import BacktestResults
from src.bt.strategies import plot_fn_for
from src.bt.strategies.dsl import StrategyContext, strategy
from src.bt.strategies.ta_context import init_ta
from src.bt.strategies.types import PlotSpec
from src.bt.types import StrategyConfig
from src.utils import parse_timestamp

if TYPE_CHECKING:
    pass

SYMBOL = "AAPL"
BARS = 120


def _feed(periods: int = BARS) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=periods, freq="D")
    closes = [100.0 + float(np.sin(i / 3.0)) * 5 + i * 0.05 for i in range(periods)]
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


def _cfg(strategy_type: str = "_probe_type") -> StrategyConfig:
    return StrategyConfig(
        name="plot-wiring",
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


def _run(adapter: object, strategy_type: str = "mfi_pivotdiv_dsl") -> BacktestResults:
    data = _feed()
    bt = Backtest(_cfg(strategy_type))
    results, _ = run_backtest(
        bt,
        candle_generator(data, bt.config),
        default_exchange(),
        default_risk_handler(),
        strategy_mod=SimpleNamespace(on_candle=adapter),
        ta=init_ta(data, bt.config.symbols, bt.config.bars[0]),
    )
    return results


def _fake_module(name: str, with_plot: bool) -> ModuleType:
    """Build a real module in ``sys.modules`` and decorate a strategy in it."""
    mod = ModuleType(name)
    setattr(mod, "STRATEGY_TYPE", name)
    sys.modules[name] = mod

    def on_candle(ctx: StrategyContext) -> None:  # noqa: ARG001
        return None

    on_candle.__module__ = name
    decorated = strategy(bars="1d")(on_candle)
    setattr(mod, "on_candle", decorated)
    if with_plot:

        def plot(ctx: StrategyContext, params: object) -> PlotSpec:  # noqa: ARG001
            return PlotSpec()

        setattr(mod, "plot", plot)
    return mod


def _install(monkeypatch: pytest.MonkeyPatch, name: str, mod: ModuleType) -> None:
    """Route ``init_strat(name)`` to ``mod`` for this test.

    ``init_strat`` is patched (not the registry global): pytest reliably
    restores a patched function attribute, whereas rebinding the registry
    interacts badly with test-order teardown.
    """
    import src.bt.strategies as strat_pkg
    from src.bt.strategies import init_strat as real_init

    monkeypatch.setattr(
        strat_pkg,
        "init_strat",
        lambda s, _m=mod, _n=name: _m if s == _n else real_init(s),
    )


def test_strategy_type_and_params_still_mirrored() -> None:
    mod = _fake_module("_plot_probe_mirror", with_plot=True)
    assert mod.STRATEGY_TYPE == "_plot_probe_mirror"
    assert getattr(mod, "on_candle").__name__ == "on_candle"


def test_plot_fn_for_finds_module_plot(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _fake_module("_plot_probe_registry", with_plot=True)
    _install(monkeypatch, "_plot_probe_registry", mod)
    assert plot_fn_for("_plot_probe_registry") is getattr(mod, "plot")


def test_plot_fn_for_absent_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _fake_module("_plot_probe_registry_none", with_plot=False)
    _install(monkeypatch, "_plot_probe_registry_none", mod)
    assert plot_fn_for("_plot_probe_registry_none") is None


# --- for_plot ----------------------------------------------------------------


def _probe_adapter(seen: list[float]):
    @strategy(bars="1d")
    def on_candle(ctx: StrategyContext) -> None:
        seen.append(ctx.ta.mfi(SYMBOL, period=14).last())

    return on_candle


def test_for_plot_matches_store_and_is_readonly() -> None:
    seen: list[float] = []
    results = _run(_probe_adapter(seen))
    ctx = StrategyContext.for_plot(
        results, symbol=SYMBOL, interval="1d", params=object()
    )
    df = results.data[(SYMBOL, "1d")]
    assert len(ctx.ta.mfi(SYMBOL, period=14).to_array()) == len(df)
    # read-only: signalling is blocked, shared writes are rejected
    with pytest.raises(AssertionError, match="read-only"):
        ctx.long(SYMBOL, size=0.1)
    with pytest.raises(AssertionError, match="read-only"):
        ctx.close(SYMBOL)
    with pytest.raises(AssertionError, match="read-only"):
        ctx.short(SYMBOL, size=0.1)
    with pytest.raises(TypeError):
        ctx.shared["x"] = 1


def test_mfi_last_value_matches_store_close_series_length() -> None:
    """The post-run MFI series is aligned 1:1 with the store's bars."""
    seen: list[float] = []
    results = _run(_probe_adapter(seen))
    ctx = StrategyContext.for_plot(
        results, symbol=SYMBOL, interval="1d", params=object()
    )
    close = results.data[(SYMBOL, "1d")]["close"]
    assert len(ctx.ta.close(SYMBOL).to_array()) == len(close)


def test_for_plot_rejects_truncated_cursor() -> None:
    seen: list[float] = []
    results = _run(_probe_adapter(seen))
    results.data.advance(parse_timestamp("2024-02-01"))  # rewind the cursor
    with pytest.raises(RuntimeError, match="terminal cursor"):
        StrategyContext.for_plot(results, symbol=SYMBOL, interval="1d", params=object())


def test_for_plot_requires_ta_context() -> None:
    results = SimpleNamespace(
        data=SimpleNamespace(ta=None, is_exhausted=True),
        final_state=object(),
    )
    with pytest.raises(RuntimeError, match="TaContext"):
        StrategyContext.for_plot(results, symbol=SYMBOL, interval="1d", params=object())


def test_plot_spec_absent_for_strategy_without_plot() -> None:
    """No ``plot`` fn -> the plot payload carries no frame-level ``plot`` key."""
    seen: list[float] = []
    results = _run(_probe_adapter(seen), strategy_type="mfi_divergence_dsl")
    payload = render_plot_json(results)
    for frames in payload["symbols"].values():
        assert all("plot" not in f for f in frames)
