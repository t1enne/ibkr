"""Warmup-phase engine tests.

Covers the warmup contract end to end: warmup bars fill strategy state but
produce no trades, a short trading window still trades immediately with warm
state, fold windows each get their own warmup, and warmup bars never reach
metrics/equity/benchmarks. Also pins the throwing contract for signal emission
during warmup.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from src.bt import load_strategy
from src.bt.engine.backtest import Backtest, run
from src.bt.strategies.dsl import StrategyContext, WarmupTradeError, strategy
from src.bt.types import StrategyConfig
from src.utils import parse_timestamp

#: Sentinel for "omit this kwarg" in ``_build_config``.
_MISSING = object()


@dataclass
class _FixtureMod:
    """Module-shaped holder the engine consumes (``.on_candle`` = adapter)."""

    on_candle: object


def _daily_df(symbols: list[str], n: int, start: str = "2024-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="D")
    data: dict = {}
    for sym in symbols:
        data.update(
            {
                (sym, "open"): [100.0] * n,
                (sym, "high"): [105.0] * n,
                (sym, "low"): [95.0] * n,
                (sym, "close"): [102.0] * n,
                (sym, "volume"): [1000.0] * n,
            }
        )
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _cfg(
    warmup: str = "0d",
    trading_start: str = "2024-01-01",
    trading_end: str = "2024-12-31",
) -> StrategyConfig:
    return StrategyConfig(
        name="t",
        strategy_type="momentum_compression_breakout_dsl",
        symbols=["AAPL"],
        initial_capital=10000.0,
        commission=0.5,
        warmup=warmup,
        trading_start=trading_start,
        trading_end=trading_end,
        bars=["1d"],
        strategy_params={},
        benchmark_symbols=[],
    )


# --- fixtures -------------------------------------------------------------


@strategy(bars="1d", stateful=True)
def _count_phases(ctx: StrategyContext):
    """Record every dispatch's phase and close, then always try to go long."""
    seen = ctx.shared.setdefault("seen", [])
    seen.append((ctx.phase, ctx.candle.timestamp))
    ctx.shared["n"] = ctx.shared.get("n", 0) + 1
    ctx.long(ctx.candle.symbol, size=0.05, reason="always")


@strategy(bars="1d", stateful=True)
def _warm_then_trade(ctx: StrategyContext):
    """Accumulate during warmup (no emission), enter ONCE when trading starts.

    Deliberately does NOT emit during warmup — the strategy guards on phase, so
    it accumulates state and only enters once trading starts. This is the
    pattern every strategy in the repo now follows. Enters a single lot so
    ``_finalize``'s end-of-run close is the only exit.
    """
    ctx.shared.setdefault("bars", 0)
    ctx.shared["bars"] += 1
    if ctx.phase == "warmup":
        return
    if ctx.shared.get("entered"):
        return
    ctx.shared["entered"] = True
    ctx.long(
        ctx.candle.symbol,
        size=0.5,
        reason=f"{ctx.shared['bars']} warm bars",
    )


def test_warmup_bars_feed_state_but_produce_no_trades() -> None:
    """Warmup dispatches accumulate state; no trade may appear before trading_start."""
    cfg = _cfg(warmup="30d", trading_start="2024-02-15")
    bt = Backtest(cfg)
    # 60 daily bars from Jan 1; trading starts Feb 15 (bar ~46).
    data = _daily_df(["AAPL"], 60)
    results = run(bt, data, _FixtureMod(_warm_then_trade))
    trades = results.pf.trades
    assert len(trades) >= 1
    # Every trade is inside the trading window — none leaked from warmup.
    for t in trades:
        assert t.entry_time >= bt.window.test_start


def test_warmup_zero_is_a_pure_noop() -> None:
    """A zero warmup must behave exactly like the pre-warmup engine: the first
    bar is already a trading bar."""
    cfg = _cfg(warmup="0d", trading_start="2024-01-01")
    bt = Backtest(cfg)
    assert bt.window.warmup_bars == 0
    data = _daily_df(["AAPL"], 5)
    results = run(bt, data, _FixtureMod(_warm_then_trade))
    # First bar trades immediately.
    assert len(results.pf.trades) >= 1
    assert results.pf.trades[0].entry_time == pd.Timestamp("2024-01-01")


def test_short_trading_window_trades_immediately_with_warm_state() -> None:
    """A tiny trading window still trades on its first bar because state is warm."""
    cfg = _cfg(warmup="60d", trading_start="2024-02-20", trading_end="2024-02-21")
    bt = Backtest(cfg)
    data = _daily_df(["AAPL"], 60)
    results = run(bt, data, _FixtureMod(_warm_then_trade))
    assert len(results.pf.trades) >= 1
    assert results.pf.trades[0].entry_time == pd.Timestamp("2024-02-20")


def test_state_persists_from_warmup_into_trading_window() -> None:
    """The counter accumulated during warmup must survive into the trading window
    (same run, same ctx.shared holder) — that is what 'warm' means."""
    cfg = _cfg(warmup="20d", trading_start="2024-01-21")
    bt = Backtest(cfg)
    data = _daily_df(["AAPL"], 40)
    results = run(bt, data, _FixtureMod(_warm_then_trade))
    assert results.pf.trades, "expected a trade on the first trading bar"
    assert "warm bars" in str(results.pf.trades[0].reason)
    warm_bars = int(str(results.pf.trades[0].reason).split()[0])
    assert warm_bars > 1, f"state did not persist from warmup (bars={warm_bars})"


def test_equity_curve_excludes_warmup_bars() -> None:
    """Metrics/equity must cover the trading window only."""
    cfg = _cfg(warmup="30d", trading_start="2024-02-15")
    bt = Backtest(cfg)
    data = _daily_df(["AAPL"], 60)
    results = run(bt, data, _FixtureMod(_warm_then_trade))
    equity = results.pf.equity_curve
    assert len(equity) > 0, "expected equity points inside the trading window"
    # Warmup bars never contributed a mark.
    assert equity.index.min() >= bt.window.test_start
    assert equity.index.max() <= bt.window.test_end


def test_finalize_closes_at_last_trading_bar() -> None:
    """_finalize must never stamp the end-of-run close at a warmup bar.

    The close lands on the last bar the engine processed (which may run past
    ``trading_end`` when the feed does — pre-existing behavior, unchanged here);
    the warmup invariant is that it is never BEFORE ``trading_start``.
    """
    cfg = _cfg(warmup="30d", trading_start="2024-02-15", trading_end="2024-02-25")
    bt = Backtest(cfg)
    data = _daily_df(["AAPL"], 60)
    results = run(bt, data, _FixtureMod(_warm_then_trade))
    assert results.pf.trades
    for t in results.pf.trades:
        assert t.entry_time >= bt.window.test_start
        assert t.exit_time is not None
        assert t.exit_time >= bt.window.test_start


def test_emitting_during_warmup_raises() -> None:
    """A strategy that emits on a warmup bar must fail loudly, naming symbol+bar."""
    cfg = _cfg(warmup="30d", trading_start="2024-02-15")
    bt = Backtest(cfg)
    data = _daily_df(["AAPL"], 40)
    with pytest.raises(WarmupTradeError) as exc:
        run(bt, data, _FixtureMod(_count_phases))
    msg = str(exc.value)
    assert "AAPL" in msg
    assert "2024-01" in msg
    assert "warmup" in msg


def test_wide_warmup_report_records_every_phase() -> None:
    """Sanity: the phase flag is present and correct on every dispatch.

    Uses a docile strategy (accumulate only) so no emission can throw; asserts
    the engine marked warmup bars as 'warmup' and trading bars as 'trade'.
    """
    cfg = _cfg(warmup="10d", trading_start="2024-01-11", trading_end="2024-01-20")
    bt = Backtest(cfg)
    data = _daily_df(["AAPL"], 20)
    phases: list[tuple[str, pd.Timestamp]] = []

    @strategy(bars="1d", stateful=True)
    def _record(ctx: StrategyContext):
        phases.append((ctx.phase, ctx.candle.timestamp))

    run(bt, data, _FixtureMod(_record))
    warm = [ts for ph, ts in phases if ph == "warmup"]
    trade = [ts for ph, ts in phases if ph == "trade"]
    assert warm, "expected warmup dispatches"
    assert trade, "expected trading dispatches"
    assert max(warm) < bt.window.test_start <= min(trade)


def test_short_history_symbol_warms_with_what_it_has() -> None:
    """A symbol whose feed is shorter than the warmup span must not crash or
    prevent trading in the trading window."""
    cfg = _cfg(warmup="365d", trading_start="2024-01-03", trading_end="2024-01-10")
    bt = Backtest(cfg)
    data = _daily_df(["AAPL"], 10)  # only 10 bars, far shorter than the warmup
    results = run(bt, data, _FixtureMod(_warm_then_trade))
    assert results.pf.trades, "trading window should still trade"


def test_warmup_bars_derived_from_calendar_span() -> None:
    """The engine's warmup_bars is the calendar span at the base interval."""
    cfg = _cfg(warmup="1y")
    bt_d = Backtest(cfg)
    hourly = _cfg(warmup="1y")
    hourly = StrategyConfig(
        name="t",
        strategy_type=hourly.strategy_type,
        symbols=hourly.symbols,
        initial_capital=hourly.initial_capital,
        commission=hourly.commission,
        warmup=hourly.warmup,
        trading_start=hourly.trading_start,
        trading_end=hourly.trading_end,
        bars=["1h"],
        strategy_params={},
        benchmark_symbols=[],
    )
    bt_h = Backtest(hourly)
    assert bt_d.window.warmup_bars == 261
    assert bt_h.window.warmup_bars > bt_d.window.warmup_bars


def test_removed_training_field_fails_loudly(tmp_path: Path) -> None:
    """No back-compat: a config still carrying the old field raises on load.

    Loads through ``load_strategy`` — the real surface a stale config file hits
    — so the failure is ``StrategyConfig(**data)`` rejecting the removed key.
    """
    cfg_path = tmp_path / "stale.json"
    cfg_path.write_text(
        json.dumps(
            {
                "name": "t",
                "strategy_type": "momentum_compression_breakout_dsl",
                "symbols": ["AAPL"],
                "initial_capital": 10000.0,
                "commission": 0.5,
                "training_start": "2023-01-01",
                "training_end": "2023-12-31",
                "trading_start": "2024-01-01",
                "trading_end": "2024-12-31",
                "bars": ["1d"],
                "strategy_params": {},
            }
        )
    )
    with pytest.raises(TypeError, match="training_start"):
        load_strategy(str(cfg_path))


def test_missing_warmup_fails_loudly(tmp_path: Path) -> None:
    """A config without ``warmup`` is a hard error, not a silent default."""
    cfg_path = tmp_path / "no_warmup.json"
    cfg_path.write_text(
        json.dumps(
            {
                "name": "t",
                "strategy_type": "momentum_compression_breakout_dsl",
                "symbols": ["AAPL"],
                "initial_capital": 10000.0,
                "commission": 0.5,
                "trading_start": "2024-01-01",
                "trading_end": "2024-12-31",
                "bars": ["1d"],
                "strategy_params": {},
            }
        )
    )
    with pytest.raises(TypeError, match="warmup"):
        load_strategy(str(cfg_path))


def test_warmup_load_start_subtracts_calendar_span() -> None:
    from src.bt import warmup_load_start

    cfg = _cfg(warmup="90d")
    start = warmup_load_start(cfg, parse_timestamp("2024-06-01"))
    assert start == pd.Timestamp("2024-03-03")


def test_each_fold_window_gets_its_own_warmup() -> None:
    """OVERRIDE 4: a fold's window is fed ``[window_start - warmup, window_end]``,
    so an OOS fold that starts mid-feed still warms on prior bars.

    Runs the same strategy through ``run_window`` twice with different
    ``trading_start`` values and asserts each recorded warmup dispatches that
    are strictly before its own trading_start — i.e. the warmup is re-derived
    per window, not shared from the first.
    """
    from src.bt.window import run_window

    warmups: dict[pd.Timestamp, int] = {}

    @strategy(bars="1d", stateful=True)
    def _count_warmup(ctx: StrategyContext):
        warmups[ctx.candle.timestamp] = 0 if ctx.phase != "warmup" else 1

    cfg = _cfg(warmup="20d", trading_start="2024-01-05", trading_end="2024-03-01")
    data = _daily_df(["AAPL"], 90)
    mod = _FixtureMod(_count_warmup)

    for start_iso in ("2024-01-11", "2024-02-11"):
        start = parse_timestamp(start_iso)
        run_window(cfg, mod, data, None, start, parse_timestamp("2024-03-01"))
        before = [ts for ts, w in warmups.items() if w == 1]
        assert before, f"no warmup dispatches before {start}"
        assert max(before) < start, "warmup dispatch landed inside the trading window"
