"""Tests for run_sweep's data-loading grouping (Option B).

Verifies that sweeping top-level, data-affecting fields (``symbols`` and the
trading/training window) loads candles once per distinct (symbols, bar) set
over the *union* of spans, and slices each combo to its own window — instead
of the old behavior of loading once from the first combo only.
"""

import pandas as pd
import pytest

from src.bt.sweep import run_sweep, grid_combos
from src.bt.types import StrategyConfig


def _fix_cfg() -> StrategyConfig:
    return StrategyConfig(
        name="test",
        strategy_type="momentum_compression_breakout_dsl",
        symbols=["A"],
        initial_capital=100000.0,
        commission=0.05,
        training_start="2020-01-01",
        training_end="2020-06-01",
        trading_start="2020-06-01",
        trading_end="2020-12-31",
        bars=["1d"],
        strategy_params={"fast": 9, "slow": 21},
        benchmark_symbols=[],
    )


def _fake_feed(symbols, start, end, bar, freq="D"):
    """Build a MultiIndex-column OHLCV frame covering [start, end]."""
    idx = pd.date_range(start, end, freq=freq)
    frames = {}
    for s in symbols:
        n = len(idx)
        df = pd.DataFrame(
            {
                "open": [10.0] * n,
                "high": [11.0] * n,
                "low": [9.0] * n,
                "close": [10.5] * n,
                "volume": [1000.0] * n,
            },
            index=idx,
        )
        frames[s] = df
    return pd.concat(frames, axis=1, keys=symbols, sort=False)


def test_sweep_symbols_groups_loads_by_symbol_set(monkeypatch):
    """Combos with different symbol sets each get their own load (once each)."""
    cfg = _fix_cfg()
    merge = {"symbols": [["A"], ["A", "B"]]}  # sweep symbol list

    combos = grid_combos(merge)
    assert len(combos) == 2

    calls: list[tuple[list[str], pd.Timestamp, pd.Timestamp, str]] = []

    def fake_load_candles(symbols, start, end, bar, **kwargs):
        calls.append((symbols, start, end, bar))
        return _fake_feed(symbols, start, end, bar)

    def fake_run(bt, data, strat_mod=None, benchmark_curves=None):
        # Echo the sliced feed back as a "pf" so callers see it.
        from types import SimpleNamespace

        return SimpleNamespace(
            pf=SimpleNamespace(
                annual_return=0.01,
                sharpe_ratio=0.1,
                max_drawdown=0.0,
                _feed=data,
            ),
        )

    # run_sweep does `from src.bt.data_feed import load_candles` and
    # `from src.bt.engine.backtest import run` INSIDE the function, so patch
    # the source modules — the fresh import resolves them at call time.
    monkeypatch.setattr("src.bt.data_feed.load_candles", fake_load_candles)
    monkeypatch.setattr("src.bt.engine.backtest.run", fake_run)

    results = run_sweep(cfg, merge, sort_metric="annual_return")

    assert len(calls) == 2, f"expected one load per distinct symbol set: {calls}"
    syms_seen = {tuple(sorted(c[0])) for c in calls}
    assert syms_seen == {("A",), ("A", "B")}

    # Each combo ran against a feed containing exactly its symbols.
    assert len(results) == 2
    for r in results:
        # pf is a SimpleNamespace stub in this test, so ``_feed`` is dynamic.
        feed_symbols = {s for s, *_ in getattr(r.pf, "_feed").columns}
        assert feed_symbols == set(r.overrides["symbols"])


def test_sweep_window_slices_each_combo_to_its_own_span(monkeypatch):
    """Combos with different trading windows load once over the union span and
    each run's feed is trimmed to that combo's trading_end (no lookahead)."""
    cfg = _fix_cfg()
    # Sweep trading_start/trading_end so each combo has a distinct window.
    merge = {
        "trading_start": ["2020-06-01", "2020-08-01"],
        "trading_end": ["2020-10-01", "2020-12-31"],
    }

    combos = grid_combos(merge)
    assert len(combos) == 4

    load_calls: list[tuple[list[str], pd.Timestamp, pd.Timestamp, str]] = []
    run_ends: list[pd.Timestamp] = []

    def fake_load_candles(symbols, start, end, bar, **kwargs):
        load_calls.append((symbols, start, end, bar))
        # Same symbols throughout, load over the wide span.
        return _fake_feed(symbols, start, end, bar, freq="D")

    def fake_run(bt, data, strat_mod=None, benchmark_curves=None):
        from types import SimpleNamespace

        run_ends.append(data.index.max())
        return SimpleNamespace(
            pf=SimpleNamespace(annual_return=0.01, sharpe_ratio=0.1, max_drawdown=0.0),
            data=data,
        )

    monkeypatch.setattr("src.bt.data_feed.load_candles", fake_load_candles)
    monkeypatch.setattr("src.bt.engine.backtest.run", fake_run)

    run_sweep(cfg, merge, sort_metric="annual_return")

    # Single load over the union span (start = min training_start = 2020-01-01,
    # end = max trading_end = 2020-12-31).
    assert len(load_calls) == 1
    assert load_calls[0][0] == ["A"]
    load_start, load_end = load_calls[0][1], load_calls[0][2]
    assert load_start == pd.Timestamp("2020-01-01")
    assert load_end == pd.Timestamp("2020-12-31")

    # Each combo ran against a feed trimmed exactly to its own trading_end.
    expected_ends = {
        pd.Timestamp("2020-10-01"),
        pd.Timestamp("2020-12-31"),
    }
    assert set(run_ends) == expected_ends


def _pool_feed(symbols, start, end, bar, freq="D", **kwargs):
    """A deterministic synthetic MultiIndex-column OHLCV feed for a real run."""
    import numpy as np

    idx = pd.date_range(start, end, freq=freq)
    n = len(idx)
    np.random.seed(7)
    close = 10.0 + np.cumsum(np.random.randn(n) * 0.1)
    frames = {}
    for s in symbols:
        frames[s] = pd.DataFrame(
            {
                "open": close - 0.01,
                "high": close + 0.02,
                "low": close - 0.02,
                "close": close,
                "volume": 1000.0,
            },
            index=idx,
        )
    return pd.concat(frames, axis=1, keys=symbols, sort=False).sort_index()


@pytest.mark.slow
# Spawns a real process pool (~2.5s forkserver startup); skipped by default
# via the `not slow` marker filter (run with `pytest -m slow`).
def test_sweep_pooled_matches_sequential_engine(monkeypatch):
    """The process-pool path runs the real engine and matches sequential.

    Verifies that StrategyConfig, the candle feed and PortfolioResult pickle
    across the process boundary and produce byte-identical metrics. One pool
    spawn (workers=2) proves picklability + per-worker feed caching.
    """
    from src.bt.types import StrategyConfig

    cfg = StrategyConfig(
        name="test",
        strategy_type="momentum_compression_breakout_dsl",
        symbols=["A"],
        initial_capital=100000.0,
        commission=0.05,
        training_start="2019-01-01",
        training_end="2020-01-01",
        trading_start="2020-01-02",
        trading_end="2020-12-31",
        bars=["1d"],
        strategy_params={"fast": 5, "slow": 21},
        benchmark_symbols=[],
    )

    monkeypatch.setattr("src.bt.data_feed.load_candles", _pool_feed)

    merge = {"strategy_params": {"fast": [3, 5, 9], "slow": [15, 21]}}

    streamed: list[int] = []
    seq = run_sweep(cfg, merge, sort_metric="sharpe_ratio", workers=1)
    par = run_sweep(
        cfg,
        merge,
        sort_metric="sharpe_ratio",
        workers=2,
        on_result=lambda i, total, overrides, pf: streamed.append(i),
    )

    assert len(seq) == len(par) == 6
    # Pooled `on_result` streams in input (combo) order, like sequential.
    assert streamed == list(range(6))
    by_params = {frozenset(r.overrides.items()): r for r in par}
    for r in seq:
        key = frozenset(r.overrides.items())
        assert key in by_params
        other = by_params[key]
        # Metrics must be identical regardless of execution path.
        for field in (
            "total_return",
            "annual_return",
            "sharpe_ratio",
            "max_drawdown",
            "calmar_ratio",
        ):
            a, b = getattr(r.pf, field), getattr(other.pf, field)
            assert a == b or abs(a - b) < 1e-12, (field, a, b)
