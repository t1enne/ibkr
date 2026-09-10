"""End-to-end fundamentals wire-up: engine -> CandleStore -> ctx.fundamentals.

Runs the real engine over a tiny 1-symbol daily feed with a decorated strategy
that reads fundamentals, which is what actually proves the plumbing:

* the store reaches the DSL through ``state.candles`` (loose ``Any`` on the
  engine side, narrowed by isinstance in the DSL),
* the series the strategy sees is cursor-bounded by the *engine's* bar cursor,
  so a filing published after a bar is invisible while that bar is decided,
* ``ctx.fundamentals`` fails loudly when no store was loaded.

Fundamentals are attached by handing ``run_backtest`` the store (the seam
``run`` uses in production to load it); ``run`` itself would go to the DB, and a
wiring test has no business depending on local DB contents.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

import pytest

from src.bt.engine.backtest import Backtest, run_backtest
from src.bt.engine.handlers import default_execution_handler, default_risk_handler
from src.bt.engine.utils import candle_generator
from src.bt.state import Candle, TradeSignal
from src.bt.strategies.dsl import StrategyContext, strategy
from src.bt.strategies.fundamentals_context import Fundamentals
from src.bt.strategies.ta_context import init_ta
from src.bt.types import StrategyConfig
from src.data.fundamentals.schema import FundamentalRow, Income
from src.utils import parse_timestamp


def ts(value: str) -> pd.Timestamp:
    """``pd.Timestamp`` narrowed past its ``NaTType`` union (repo convention)."""
    return parse_timestamp(value)


SYMBOL = "AAPL"
FIRST_BAR = "2024-01-02"
BARS = 6


def _feed(periods: int = BARS) -> pd.DataFrame:
    idx = pd.date_range(FIRST_BAR, periods=periods, freq="D")
    closes = [100.0 + i for i in range(periods)]
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


def _cfg() -> StrategyConfig:
    return StrategyConfig(
        name="fundamentals-wiring",
        # Registered type is required by resolve_params; the injected adapter
        # drives the actual behaviour.
        strategy_type="momentum_compression_breakout_dsl",
        symbols=[SYMBOL],
        initial_capital=10000.0,
        commission=0.0,
        training_start="2024-01-01",
        training_end="2024-01-02",
        trading_start=FIRST_BAR,
        trading_end="2024-01-07",
        bars=["1d"],
        strategy_params={},
        benchmark_symbols=[],
    )


def _income(value: float, filed: str, period_end: str = "2023-12-31") -> FundamentalRow:
    return FundamentalRow(
        ticker=SYMBOL,
        statement="income",
        field="net_income",
        value=value,
        period_start=ts("2023-10-01"),
        period_end=parse_timestamp(period_end),
        filed=parse_timestamp(filed),
        form="10-Q",
    )


def _run_with(adapter: object, fund: Fundamentals | None) -> None:
    """Run the engine with ``adapter`` as the strategy module's ``on_candle``.

    The strategy itself is the observation hook: it appends what it reads, so
    the recorded values are exactly what the DSL served inside the hot loop.
    ``SimpleNamespace`` is the smallest object the engine's module contract
    (``<mod>.on_candle``) accepts.
    """
    data = _feed()
    bt = Backtest(_cfg())
    run_backtest(
        bt,
        candle_generator(data, bt.config),
        default_execution_handler(),
        default_risk_handler(),
        strategy_mod=SimpleNamespace(on_candle=adapter),
        ta=init_ta(data, bt.config.symbols, bt.config.bars[0]),
        fundamentals=fund,
    )


def _probe(seen: list[tuple[str, int, float]]):
    @strategy(bars="1d")
    def on_candle(ctx: StrategyContext) -> None:
        series = ctx.fundamentals.income(SYMBOL).net_income
        seen.append(
            (ctx.candle.timestamp.date().isoformat(), len(series), series.last() or 0.0)
        )
        ctx.long(SYMBOL, size=0.1)

    return on_candle


def test_engine_attaches_and_dsl_reads_cursor_bounded_fundamentals() -> None:
    """A filing is visible only once the bar cursor has reached its filing date."""
    seen: list[tuple[str, int, float]] = []
    fund = Fundamentals.build(
        {
            SYMBOL: [
                _income(10.0, "2024-01-03", period_end="2023-09-30"),
                _income(20.0, "2024-01-06", period_end="2023-12-31"),
            ]
        }
    )

    _run_with(_probe(seen), fund)

    assert len(seen) == BARS
    by_date = {date: (n, v) for date, n, v in seen}
    assert by_date["2024-01-02"] == (0, 0.0)  # nothing filed yet
    assert by_date["2024-01-03"] == (1, 10.0)  # filing lands on the cursor bar
    assert by_date["2024-01-05"] == (1, 10.0)
    assert by_date["2024-01-06"] == (2, 20.0)
    assert by_date["2024-01-07"] == (2, 20.0)


def test_visibility_is_re_read_per_bar_not_cached_at_first_access() -> None:
    """The series cache must not freeze the cursor at its first read."""
    seen: list[tuple[str, int, float]] = []
    fund = Fundamentals.build(
        {
            SYMBOL: [
                _income(1.0, FIRST_BAR, period_end="2023-09-30"),
                _income(2.0, "2024-01-04", period_end="2023-12-31"),
            ]
        }
    )

    _run_with(_probe(seen), fund)

    assert [n for _, n, _ in seen] == [1, 1, 2, 2, 2, 2]
    assert [v for _, _, v in seen] == [1.0, 1.0, 2.0, 2.0, 2.0, 2.0]


def test_latest_is_cursor_bounded_inside_the_engine() -> None:
    """``latest`` (newest-filed) honors the engine cursor as well."""
    seen: list[float] = []
    fund = Fundamentals.build(
        {
            SYMBOL: [
                _income(10.0, FIRST_BAR, period_end="2023-09-30"),
                # Same period, later filing: a restatement. The curve keeps 10.0,
                # but ``latest`` must pick this up only once the cursor passes it.
                _income(99.0, "2024-01-05", period_end="2023-09-30"),
            ]
        }
    )

    @strategy(bars="1d")
    def on_candle(ctx: StrategyContext) -> None:
        latest = ctx.fundamentals.latest(SYMBOL, "income")
        assert isinstance(latest, Income)
        seen.append(float(latest.net_income or 0.0))
        ctx.long(SYMBOL, size=0.1)

    _run_with(on_candle, fund)

    # The restatement is filed on the 4th bar (2024-01-05), so bars 1-3 see the
    # original and everything after sees the restated figure.
    assert seen[:3] == [10.0, 10.0, 10.0]
    assert seen[3:] == [99.0, 99.0, 99.0]


def test_ctx_fundamentals_raises_without_a_loaded_store() -> None:
    """Reading fundamentals on a store-less run fails loudly, not as empty data."""

    @strategy(bars="1d")
    def on_candle(ctx: StrategyContext) -> None:
        ctx.fundamentals.income(SYMBOL)

    with pytest.raises(RuntimeError, match="ctx.fundamentals requires"):
        _run_with(on_candle, None)


def test_dsl_narrows_a_mistyped_store_loudly() -> None:
    """A non-Fundamentals object on the store is a wiring bug, not empty data."""
    from src.bt.strategies.ta_context import TaContext

    @strategy(bars="1d")
    def on_candle(ctx: StrategyContext) -> None:
        ctx.fundamentals.income(SYMBOL)

    class _Store:
        ta = TaContext({}, (SYMBOL,), "1d")
        strategy_state = None
        fundamentals = "not-a-store"

    class _State:
        candles = _Store()

    candle = Candle(
        timestamp=parse_timestamp(FIRST_BAR),
        symbol=SYMBOL,
        open=1.0,
        high=1.0,
        low=1.0,
        close=1.0,
        volume=1.0,
    )
    with pytest.raises(RuntimeError, match="is not a Fundamentals store"):
        on_candle(_State(), candle, {})  # type: ignore[arg-type]


def test_engine_does_not_import_the_concrete_dsl_context() -> None:
    """The engine holds fundamentals loosely — no dependency on the strategy layer."""
    import src.bt.engine.candle_store as csmod

    source = open(csmod.__file__).read()
    assert "fundamentals_context" not in source
    assert "attach_fundamentals" in source


def test_signals_still_flow_with_fundamentals_attached() -> None:
    """Attaching fundamentals must not disturb normal signal emission."""
    fund = Fundamentals.build({SYMBOL: [_income(10.0, FIRST_BAR)]})

    seen: list[int] = []

    def _on_candle(ctx: StrategyContext) -> None:
        seen.append(len(ctx.fundamentals.income(SYMBOL).net_income))
        ctx.long(SYMBOL, size=0.1)

    on_candle = strategy(bars="1d")(_on_candle)
    data = _feed()
    bt = Backtest(_cfg())

    results, _ = run_backtest(
        bt,
        candle_generator(data, bt.config),
        default_execution_handler(),
        default_risk_handler(),
        strategy_mod=SimpleNamespace(on_candle=on_candle),
        ta=init_ta(data, bt.config.symbols, bt.config.bars[0]),
        fundamentals=fund,
    )
    assert seen == [1] * BARS
    assert results.final_state.portfolio.trades != ()
    assert TradeSignal is not None
