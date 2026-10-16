"""Tests for backtest engine — critical paths only."""

from dataclasses import dataclass, replace
from typing import cast

import pandas as pd
from src.bt.engine.backtest import Backtest, run_backtest, candle_generator, run
from src.bt.exchange import default_exchange
from src.bt.risk.handlers import default_risk_handler
from src.bt.state import (
    ActionType,
    BacktestState,
    Position,
    create_initial_backtest_state,
)
from src.bt.types import StrategyConfig
from src.bt.strategies.dsl import StrategyContext, strategy


@dataclass
class _FixtureMod:
    """Mirrors a real strategy module's engine-facing surface: ``.on_candle``
    is the ``@strategy`` adapter (carrying ``ctx_fn``). Strat passed straight to
    ``run``/``run_backtest`` as this holder, exactly as ``init_strat`` does for
    production modules."""

    on_candle: object


def _make_multi_idx_df(symbols: list[str], n: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    data = {}
    for sym in symbols:
        data.update(
            {
                (sym, "open"): [100] * n,
                (sym, "high"): [105] * n,
                (sym, "low"): [95] * n,
                (sym, "close"): [102] * n,
                (sym, "volume"): [1000] * n,
            }
        )
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


@strategy(bars="1d", stateful=True)
def _fixture_dsl_on_candle(ctx: StrategyContext):
    """Real ``@strategy`` fixture — DSL replacement for the removed plain
    ``_stub_strategy``. Fires one long on the 3rd on_candle dispatch using a
    stateful ``ctx.shared`` counter. Does NOT read ``ctx.params`` (the engine
    resolves params off ``config.strategy_type``'s registered type, not this
    fixture's), so observer/cash tests sidestep that mismatch entirely."""
    assert ctx.shared is not None  # stateful adapter binds a per-run holder
    n = ctx.shared.setdefault("_n", 0)
    ctx.shared["_n"] = n + 1
    if n == 2:  # 3rd dispatch (0-indexed counter)
        ctx.long(ctx.candle.symbol, size=0.05, reason="test long")


def _fixture_dsl():
    """DSL strategy_mod for the engine: the ``@strategy`` adapter wrapped in the
    module-shaped holder the engine consumes (``.on_candle`` = adapter)."""
    return _FixtureMod(_fixture_dsl_on_candle)


def _kit(cfg=None):
    """A ready-to-run ``Backtest`` over a 1-daily-bar AAPL feed."""
    cfg = cfg or _cfg(["AAPL"])
    return Backtest(cfg), _daily_df(["AAPL"])


def _daily_df(symbols: list[str], n: int = 5) -> pd.DataFrame:
    """MultiIndex-column frame over ``n`` daily bars aligned to a 1d config."""
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    data: dict = {}
    for sym in symbols:
        data.update(
            {
                (sym, "open"): [100] * n,
                (sym, "high"): [105] * n,
                (sym, "low"): [95] * n,
                (sym, "close"): [102] * n,
                (sym, "volume"): [1000] * n,
            }
        )
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _cfg(symbols):
    """1d config tied to a registered strategy type with a 2025 trading window
    over a 5-day daily feed (an injected strategy_mod overrides the type)."""
    return StrategyConfig(
        name="test",
        strategy_type="momentum_compression_breakout_dsl",  # registered
        symbols=symbols,
        initial_capital=10000.0,
        commission=0.5,
        warmup="0d",
        trading_start="2025-01-01",
        trading_end="2025-12-31",
        bars=["1d"],
        strategy_params={},
    )


def test_run_backtest_rejects_plain_strategy_mod():
    """A hand-rolled (non-@strategy) module handed to run_backtest raises;
    None stays a legal no-strategy loop."""
    from pytest import raises

    cfg = _cfg(["AAPL"])
    bt = Backtest(cfg)
    gen = candle_generator(_daily_df(["AAPL"]), bt.config)
    cl = type("_Mod", (object,), {"on_candle": lambda self, s, c, p: []})()
    with raises(TypeError):
        run_backtest(
            bt,
            gen,
            default_exchange(),
            default_risk_handler(),
            strategy_mod=cl,
        )


# --- initial_state seeding -------------------------------------------------


@strategy(bars="1d")
def _exit_when_holding(ctx: StrategyContext):
    """A held-lot exit: ``ctx.close`` is a NO-OP on a flat book, so this emits
    only while a lot is open — the rule a screen's short window cannot reach
    unless the book is seeded."""
    ctx.close(ctx.candle.symbol, reason="exit")


def _seeded_state(qty: float = 10.0) -> BacktestState:
    """A state already holding one AAPL lot, as a live book would be."""
    state = create_initial_backtest_state(
        symbols=["AAPL"],
        initial_capital=10000.0,
        start_timestamp=cast(pd.Timestamp, pd.Timestamp("2025-01-01")),
    )
    lot = Position(
        symbol="AAPL",
        qty=qty,
        entry_price=100.0,
        entry_time=cast(pd.Timestamp, pd.Timestamp("2024-06-03")),
        stop_loss=None,
        take_profit=None,
        last_price=102.0,
        type=ActionType.long,
        position_id="AAPL_old",
    )
    portfolio = replace(state.portfolio, positions={"AAPL": (lot,)})
    return replace(state, portfolio=portfolio)


def test_run_seeded_state_lets_a_held_lot_exit():
    # The regression: without the seed the strategy never sees the lot it holds,
    # so its exit rule never fires and the position is left open forever.
    cfg = _cfg(["AAPL"])
    observed: list = []
    run(
        Backtest(cfg),
        _daily_df(["AAPL"]),
        _FixtureMod(_exit_when_holding),
        signal_observer=observed.append,
        initial_state=_seeded_state(),
    )
    assert [s.action for s in observed] == [ActionType.close]
    assert observed[0].position_id == "AAPL_old"


def test_run_without_a_seed_never_sees_the_held_lot():
    # The shape of the bug: a flat-seeded run over the same feed emits nothing,
    # so nothing downstream can learn the lot must be closed.
    cfg = _cfg(["AAPL"])
    observed: list = []
    run(
        Backtest(cfg),
        _daily_df(["AAPL"]),
        _FixtureMod(_exit_when_holding),
        signal_observer=observed.append,
    )
    assert observed == []


# --- candle_generator bar count correctness --------------------------------


def _hourly_df(symbols: list[str], days: int = 1) -> pd.DataFrame:
    hours_per_day = [f"{h:02d}:00" for h in range(14, 22)]
    stamps = []
    for d in range(days):
        for h in hours_per_day:
            stamps.append(f"2025-01-{6 + d:02d} {h}")
    idx = pd.to_datetime(stamps)
    n = len(idx)
    data: dict = {}
    for sym in symbols:
        data.update(
            {
                (sym, "open"): [100] * n,
                (sym, "high"): [105] * n,
                (sym, "low"): [95] * n,
                (sym, "close"): [102] * n,
                (sym, "volume"): [1000] * n,
            }
        )
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _cfg_1h(symbols):
    return StrategyConfig(
        name="test",
        strategy_type="momentum_compression_breakout_dsl",
        symbols=symbols,
        initial_capital=10000.0,
        commission=0.5,
        warmup="0d",
        trading_start="2025-01-06",
        trading_end="2025-12-31",
        bars=["1h"],
        strategy_params={},
    )


def _cfg_1h_4h(symbols):
    return StrategyConfig(
        name="test",
        strategy_type="momentum_compression_breakout_dsl",
        symbols=symbols,
        initial_capital=10000.0,
        commission=0.5,
        warmup="0d",
        trading_start="2025-01-06",
        trading_end="2025-12-31",
        bars=["1h", "4h"],
        strategy_params={},
    )


def test_candle_generator_1h_matches_input_row_count():
    df = _hourly_df(["SPY"], days=3)
    bt = Backtest(_cfg_1h(["SPY"]))
    base = [c for c in candle_generator(df, bt.config) if c.interval == "1h"]
    assert len(base) == len(df)


def test_candle_generator_4h_emits_two_per_full_day():
    n_days = 5
    df = _hourly_df(["SPY"], days=n_days)
    bt = Backtest(_cfg_1h_4h(["SPY"]))
    candles = list(candle_generator(df, bt.config))
    htf = [c for c in candles if c.interval == "4h"]
    assert len(htf) == n_days * 2
