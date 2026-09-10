"""Functional backtest module.

Candle processing pipeline (per-candle stages, in order):
  _append_candle       – stash every candle (base + HTF) in accumulator
  _execute_pending     – fill signals queued for current symbol from prior candles
  _generate_signals    – run strategy on the last symbol per timestamp,
                          bucket returned signals by symbol into pending dict
  _execute_pending     – same-bar: fill same-symbol signals (skip fill_at_next_open)
  _check_risk          – evaluate stop-loss / take-profit
  _mark_to_market      – update position prices and equity curve

Pipeline invariants:
  - Signals execute before risk on the same bar. A rebalance emitted in
    Stage 4 is filled in Stage 6 before Stage 7 risk check runs, so
    risk events always fire against the post-rebalance position state
    (no stale position_id crashes).
  - Strategies emitting multiple signals for the same symbol in one batch
    must avoid races (e.g. close+reopen): close signals fill at next bar's
    open (Stage 4), open/rebalance fill same-bar (Stage 6), so they never
    collide on the same pass.
  - Signals are bucketed by symbol into a dict. _execute_pending reads
    directly from the current symbol's bucket — no O(N) scan over all
    pending signals.

Usage:
    from src.bt.engine.backtest import Backtest, candle_generator, run_backtest
    from src.bt.engine.handlers import default_execution_handler, default_risk_handler

    bt = Backtest(config)
    gen = candle_generator(df, config.symbols)
    results, state = run_backtest(bt, gen, exec_handler, risk_handler)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.bt.engine.candle_store import CandleStore, CandleRows
from src.bt.engine.utils import candle_generator, merge_bt_state

from dataclasses import dataclass, field, replace
from typing import Generator, Tuple, Optional, Any, Callable, Mapping

import numpy as np
import pandas as pd

from src.bt.metrics import calculate_portfolio_result
from src.bt.strategies import resolve_params
from src.bt.strategies.ta_context import TaContext

if TYPE_CHECKING:
    from src.bt.types import StrategyConfig

from src.bt.state import (
    ActionType,
    BacktestState,
    Candle,
    TradeSignal,
    FillEvent,
    ExecutionParams,
    RiskConfig,
    create_initial_backtest_state,
    create_execution_params,
    create_risk_config,
    TradeExitReason,
)
from src.bt.size.pure import SizingParams, equity_of, sized_signal
from src.bt.types import StrategyConfig, EngineWindow, BacktestResults
from src.bt.engine.handlers import ExecutionHandler, RiskHandler
from src.utils import parse_timestamp


@dataclass
class Backtest:
    """Backtest configuration container.

    This is a pure dataclass - no methods, no state. It just holds config.
    All backtest logic is in the standalone run_backtest function.
    """

    config: StrategyConfig
    window: EngineWindow = field(init=False)
    execution_params: ExecutionParams = field(init=False)
    risk_config: RiskConfig = field(init=False)
    sizing: SizingParams = field(init=False)

    def __post_init__(self):
        self.window = EngineWindow(
            train_start=parse_timestamp(self.config.training_start),
            train_end=parse_timestamp(self.config.training_end),
            test_start=parse_timestamp(self.config.trading_start),
            test_end=parse_timestamp(self.config.trading_end),
        )
        self.execution_params = create_execution_params(
            fixed_commission=self.config.commission
        )
        # SL/TP is strategy-owned (set per-trade on TradeSignal from
        # strategy_params). No config-level fallback: zero pct means the risk
        # module never derives SL/TP — a strategy-set level is the only source.
        self.risk_config = create_risk_config(stop_loss_pct=0.0, take_profit_pct=0.0)
        # Shared position-sizing config, driven by strategy_params
        # (sizing_mode / size / max_symbol_allocation). Applied by
        # the engine to signals whose qty <= 0. Risk-targeted sizing is
        # strategy-owned (see size.risk_sized_qty) and never routed here.
        self.sizing = SizingParams.from_dict(self.config.strategy_params)


def run_backtest(
    bt: Backtest,
    candle_gen: Generator[Candle, None, None],
    exec_handler: ExecutionHandler,
    risk_handler: RiskHandler,
    initial_state: Optional[BacktestState] = None,
    strategy_mod: Any = None,
    benchmark_curves: Optional[Mapping[str, pd.Series]] = None,
    ta: Optional[TaContext] = None,
    strategy_state: Optional[dict] = None,
    fundamentals: Any = None,
    signal_observer: Optional[Callable] = None,
) -> Tuple[BacktestResults, BacktestState]:
    """Run backtest with the given candle generator and handlers.

    This is a pure function - given the same inputs, it always returns
    the same results.

    Args:
        bt: Backtest config
        candle_gen: Generator yielding Candles (OHLCV bars)
        exec_handler: Execution handler with execute_signal, execute_risk_event, apply_fill
        risk_handler: Risk handler with check_risk
        initial_state: Optional initial state (default: create from config)
        strategy_mod: Optional strategy module — must be a ``@strategy`` adapter;
            plain ``on_candle`` modules are rejected by ``_assert_dsl_strategy``.
            ``None`` is a legal no-strategy (data-only) run.
        benchmark_curves: Optional pre-sliced benchmark curves to reuse across
            windows (avoids a per-window benchmark DB reload). When omitted,
            benchmarks are loaded+drawn here from config.
        ta: Optional prefetched ``TaContext`` for DSL strategies (built once from
            the full feed in ``run``; attached to the CandleStore so decorated
            strategies read it cursor-safely via ``state.candles.ta``).
        strategy_state: Optional per-run cross-candle state holder for stateful
            DSL strategies (minted fresh by ``run`` per window; attached to the
            CandleStore so no module-level singleton is shared).
        fundamentals: Optional prefetched ``Fundamentals`` store (as-first-stated
            fiscal series loaded once from the local DB). Typed ``Any`` like
            ``ta`` — the engine layer stays decoupled from the strategy layer and
            the DSL narrows it via isinstance.

    Returns:
        Tuple of (BacktestResults, final BacktestState)
    """
    config = bt.config
    _assert_dsl_strategy(strategy_mod, allow_none=True)
    symbols = config.symbols
    strategy_fn = strategy_mod.on_candle if strategy_mod else None
    last_symbol = symbols[-1] if symbols else None

    # Resolve typed params once if strategy defines them; inject the top-level

    resolved_params = resolve_params(
        config.strategy_type,
        config.strategy_params,
    )

    def get_initial_state():
        start_date = parse_timestamp(config.trading_start)
        return create_initial_backtest_state(
            symbols=symbols,
            initial_capital=config.initial_capital,
            start_timestamp=start_date,
            rolling_window_size=config.rolling_window_size,
        )

    state = initial_state or get_initial_state()
    rows: CandleRows = {}
    # Engine-owned equity accumulator (avoids the O(n) tuple rebuild per candle
    # in ``update_prices``). Seeded with the initial equity point; frozen to a
    # tuple on the final PortfolioState at ``_finalize``.
    eq_buffer: list = list(state.portfolio.equity_curve)

    # Create CandleStore once — wraps rows by reference, mutates in-place.
    # Strategies access it as state.candles (Mapping interface) + .latest()/.count().
    store = CandleStore(rows)
    # Bind the prefetched TaContext (DSL) to share the store's cursor. Reading
    # ``store.ta`` from a decorated strategy is always lookahead-safe because
    # the TaContext slices against this same cursor.
    if ta is not None:
        store.attach_ta(ta)
        ta.bind(store)
    if strategy_state is not None:
        store.attach_strategy_state(strategy_state)
    if fundamentals is not None:
        # Bind after construction so the store's cursor reader is wired to the
        # same cursor the TaContext uses — publication-time and bar-time
        # visibility come from one clock.
        store.attach_fundamentals(fundamentals)
    state = merge_bt_state(state, dict(candles=store))

    for candle in candle_gen:
        can_trade = bt.window.test_start <= candle.timestamp <= bt.window.test_end
        is_base = not candle.interval or candle.interval == config.bars[0]

        # Stage 1: stash EVERY candle (base + HTF) into the same accumulator
        rows, state = _append_candle(rows, state, candle, config.bars[0])

        # HTF-only candles: accumulate and skip rest of pipeline
        if not is_base:
            continue

        # Stage 3: execute pending signals (from prior candles)
        state = _execute_pending(
            state, candle, exec_handler, config, bt.execution_params, bt.sizing
        )

        # Stage 5: generate new signals (only on last symbol per timestamp)
        state = _generate_signals(
            state,
            candle,
            resolved_params,
            strategy_fn,
            last_symbol,
            can_trade,
            rows,
            signal_observer=signal_observer,
        )

        # Stage 6: execute signals generated this tick (skip fill_at_next_open
        # signals — they fill at next bar's open via Stage 4)
        state = _execute_pending(
            state,
            candle,
            exec_handler,
            config,
            bt.execution_params,
            bt.sizing,
            skip_next_open=True,
        )

        # Stage 7: check stop-loss / take-profit
        state = _check_risk(
            state,
            candle,
            exec_handler,
            risk_handler,
            bt.execution_params,
            bt.risk_config,
        )

        # Stage 8: mark to market
        state = _mark_to_market(state, candle, eq_buffer)

    # Finalize: close positions, build results
    state = _finalize(state, bt.execution_params, equity_points=eq_buffer)

    # Build equity series, deduplicating by timestamp (equity curve
    # accumulates one point per candle = N points per timestamp).
    # Take the last equity value per unique timestamp.
    raw_equity = pd.DataFrame(
        [(p.timestamp, p.equity) for p in state.portfolio.equity_curve],
        columns=["ts", "equity"],
    )
    equity_series = raw_equity.groupby("ts")["equity"].last()
    # Slice to trading window for metrics
    equity_series = equity_series[
        (equity_series.index >= bt.window.test_start)
        & (equity_series.index <= bt.window.test_end)
    ]

    if benchmark_curves is not None:
        bm_curves = dict(benchmark_curves)
    else:
        bm_curves = _get_bench_curves(config, bt)
    # Use first benchmark curve for alpha/beta (typically the primary equity index).
    # If no benchmark symbols configured, alpha/beta will be 0.0/1.0.
    first_bm = next(iter(bm_curves.values()), None) if bm_curves else None
    pf_result = calculate_portfolio_result(
        equity_series,
        state.portfolio.trades,
        state.portfolio.initial_capital,
        benchmark_curve=first_bm,
        equity_points=state.portfolio.equity_curve,
    )

    return (
        BacktestResults(
            pf=pf_result,
            data=state.candles,
            final_state=state,
            benchmark_curves=bm_curves,
        ),
        state,
    )


def build_benchmark_curves(
    bm_df: pd.DataFrame,
    config: StrategyConfig,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> dict[str, pd.Series]:
    """Build benchmark curves from an already-loaded benchmark DataFrame.

    Pure and window-parameterised so a caller can load benchmark candles once
    and slice per window (avoids a DB reload every window).
    """
    bm_curves: dict[str, pd.Series] = {}
    closes: dict[str, pd.Series] = {}
    for bm_sym in config.benchmark_symbols:
        try:
            bm_close = bm_df.xs(bm_sym, axis=1, level=0)["close"]
            # Slice to trading window for comparison
            bm_close = bm_close[
                (bm_close.index >= test_start) & (bm_close.index <= test_end)
            ]
            if len(bm_close) < 2:
                continue
            # Normalize to same initial capital as strategy
            bm_eq = bm_close / bm_close.iloc[0] * config.initial_capital
            bm_curves[bm_sym] = bm_eq
            closes[bm_sym] = bm_eq
        except KeyError:
            pass

    # Composite 50/50 buy-and-hold rebalanced: half capital in each symbol
    if len(closes) == 2:
        sym_a, sym_b = list(closes.keys())
        aligned = pd.concat(
            [closes[sym_a].rename("a"), closes[sym_b].rename("b")],
            axis=1,
        ).dropna()
        if len(aligned) > 1:
            ret_a = aligned["a"].pct_change().fillna(0.0)
            ret_b = aligned["b"].pct_change().fillna(0.0)
            avg_ret = (ret_a + ret_b) / 2.0
            cum = (1.0 + avg_ret).cumprod()
            bm_curves["50/50"] = cum * config.initial_capital
    return bm_curves


def _get_bench_curves(config: StrategyConfig, bt: Backtest):
    from src.bt.data_feed import load_candles

    if not config.benchmark_symbols:
        return {}

    try:
        bm_df = load_candles(
            config.benchmark_symbols,
            bt.window.train_start,
            bt.window.test_end,
            config.bars[0],
        )
        return build_benchmark_curves(
            bm_df, config, bt.window.test_start, bt.window.test_end
        )
    except Exception:
        return {}


def _bucket_signals(
    signals: tuple[TradeSignal, ...],
) -> dict[str, tuple[TradeSignal, ...]]:
    """Bucket flat signal list by symbol."""
    buckets: dict[str, list[TradeSignal]] = {}
    for s in signals:
        buckets.setdefault(s.symbol, []).append(s)
    return {sym: tuple(v) for sym, v in buckets.items()}


def _execute_pending(
    state: BacktestState,
    candle: Candle,
    exec_handler: ExecutionHandler,
    config: StrategyConfig,
    exec_params: ExecutionParams,
    sizing: SizingParams,
    skip_next_open: bool = False,
) -> BacktestState:
    """Stage 4/6: Execute pending signals for the current symbol.

    Reads from state.pending_signals[symbol] directly — no filtering needed.
    When skip_next_open is True (Stage 6, same-bar), signals with
    fill_at_next_open=True are deferred to the next bar's Stage 4 call.

    Signals whose qty <= 0 are sized by the shared sizing layer (equity/cash/
    fixed base, size, per-symbol cap + cash clamp) before
    execution; explicitly-sized signals pass through unchanged.
    """
    symbol = candle.symbol
    queued = state.pending_signals.get(symbol, ())
    if not queued:
        return state

    portfolio = state.portfolio
    deferred: list[TradeSignal] = []
    for signal in queued:
        if skip_next_open and signal.fill_at_next_open:
            deferred.append(signal)
            continue
        equity = equity_of(portfolio)
        # Rebalancing reduces (partial cover) carry an explicit signed delta and
        # closes route by position_id -- neither is a fresh open, so neither is
        # sized by the shared sizing layer. Only long/short opens with qty <= 0
        # are engine-sized (sized_signal turns qty<=0 opens into share counts).
        if signal.action in (ActionType.long, ActionType.short):
            signal = sized_signal(signal, equity, portfolio.cash, candle, sizing)
        fill = exec_handler.execute_signal(signal, candle, exec_params)
        portfolio = exec_handler.apply_fill(portfolio, fill)

    new_pending = dict(state.pending_signals)
    if deferred:
        new_pending[symbol] = tuple(deferred)
    else:
        new_pending.pop(symbol, None)

    return merge_bt_state(state, dict(portfolio=portfolio, pending_signals=new_pending))


def _generate_signals(
    state: BacktestState,
    candle: Candle,
    resolved_params: object,
    strategy_fn: Optional[Callable],
    last_symbol: Optional[str],
    can_trade: bool,
    rows: CandleRows,
    signal_observer: Optional[Callable] = None,
) -> BacktestState:
    """Run strategy on last symbol per timestamp, bucket signals by symbol.

    When ``signal_observer`` is set, it is invoked once per freshly-generated
    ``TradeSignal`` (before bucketing/finalize) so a caller — e.g. the screen
    driver — captures the strategy's current-bar intent that ``_finalize``
    would otherwise discard. ``None`` (default) keeps behavior byte-for-byte
    identical to earlier builds.
    """
    if not (can_trade and strategy_fn and candle.symbol == last_symbol):
        return state

    # Advance cursor so CandleStore only sees data up to this timestamp
    state.candles.advance(candle.timestamp)

    new_signals = strategy_fn(state, candle, resolved_params)
    if not new_signals:
        return state

    if signal_observer is not None:
        for sig in new_signals:
            signal_observer(sig)

    # Merge into existing pending dict — signals for same symbol accumulate
    pending = dict(state.pending_signals)
    for sym, sigs in _bucket_signals(tuple(new_signals)).items():
        existing = pending.get(sym, ())
        pending[sym] = existing + sigs

    return merge_bt_state(state, dict(pending_signals=pending))


def _check_risk(
    state: BacktestState,
    candle: Candle,
    exec_handler: ExecutionHandler,
    risk_handler: RiskHandler,
    exec_params: ExecutionParams,
    risk_config: RiskConfig,
) -> BacktestState:
    """Stage 8: Check stop-loss / take-profit and execute risk closes.

    risk_handler.check_risk returns (events, updated_portfolio). The updated
    portfolio carries persisted SL/TP levels (initialised or trailed) even
    when no risk event fires.
    """
    risk_events, portfolio = risk_handler.check_risk(
        state.portfolio, candle, risk_config
    )

    for event in risk_events:
        fill = exec_handler.execute_risk_event(event, candle, exec_params)
        portfolio = exec_handler.apply_fill(portfolio, fill)

    return merge_bt_state(state, dict(portfolio=portfolio, risk_events=risk_events))


def _mark_to_market(
    state: BacktestState,
    candle: Candle,
    eq_buffer: list,
) -> BacktestState:
    """Stage 9: Update position prices and append equity point.

    Uses ``mark_to_market_list`` to append into the engine-owned ``eq_buffer``
    (O(1)) instead of rebuilding the O(n) immutable tuple every candle.
    """
    from src.bt.portfolio.pure import mark_to_market_list

    portfolio = mark_to_market_list(state.portfolio, candle, eq_buffer)
    return merge_bt_state(state, dict(portfolio=portfolio, timestamp=candle.timestamp))


def _append_candle(
    rows: CandleRows,
    state: BacktestState,
    candle: Candle,
    base_interval: str,
) -> Tuple[CandleRows, BacktestState]:
    """Stash candle row into numpy column arrays keyed by (symbol, interval).

    All candles — base and HTF — land in the same accumulator.
    """
    key = (candle.symbol, candle.interval or base_interval)
    if key not in rows:
        rows[key] = {
            "timestamp": np.empty(256, dtype="datetime64[ms]"),
            "open": np.empty(256, dtype=np.float64),
            "high": np.empty(256, dtype=np.float64),
            "low": np.empty(256, dtype=np.float64),
            "close": np.empty(256, dtype=np.float64),
            "volume": np.empty(256, dtype=np.float64),
            "_len": np.array([0], dtype=np.int64),
        }

    cols = rows[key]
    n = int(cols["_len"][0])

    if n >= len(cols["timestamp"]):
        new_cap = n * 2
        for col_name in ("timestamp", "open", "high", "low", "close", "volume"):
            new_arr = np.empty(new_cap, dtype=cols[col_name].dtype)
            new_arr[:n] = cols[col_name]
            cols[col_name] = new_arr

    cols["timestamp"][n] = np.datetime64(candle.timestamp.to_datetime64())
    cols["open"][n] = candle.open
    cols["high"][n] = candle.high
    cols["low"][n] = candle.low
    cols["close"][n] = candle.close
    cols["volume"][n] = candle.volume
    cols["_len"][0] = n + 1

    return rows, state


def _finalize(
    state: BacktestState,
    exec_params: ExecutionParams,
    equity_points: list | tuple | None = None,
) -> BacktestState:
    """Close all positions at end of backtest.

    ``equity_points`` (when provided) is the engine-buffered equity curve that
    replaces the empty per-candle curve so the final ``PortfolioState`` carries
    the full, frozen tuple.
    """
    portfolio = state.portfolio

    for symbol, positions_tuple in list(portfolio.positions.items()):
        for position in positions_tuple:
            close_signal = TradeSignal(
                action=ActionType.close,
                symbol=symbol,
                timestamp=state.timestamp or pd.Timestamp.now(),
                price=position.last_price,
                reason=TradeExitReason.end,
                position_id=position.position_id,
            )

            fill = FillEvent(
                signal=close_signal,
                filled_qty=abs(position.qty),
                executed_price=position.last_price,
                commission=exec_params.fixed_commission,
                slippage=0.0,
                timestamp=close_signal.timestamp,
            )

            from src.bt.portfolio.pure import apply_fill

            portfolio = apply_fill(portfolio, fill)

    # Freeze the engine-buffered equity curve onto the final portfolio.
    if equity_points is not None:
        portfolio = replace(portfolio, equity_curve=tuple(equity_points))

    return merge_bt_state(
        state,
        dict(
            portfolio=portfolio,
            pending_signals={},
            risk_events=(),
        ),
    )


def _assert_dsl_strategy(strategy_mod: Any, *, allow_none: bool = True) -> None:
    """Require a strategy handed to the engine be a ``@strategy`` product.

    ``allow_none=False`` (used by ``run()``, whose ``strat_mod`` is a required
    positional with no None meaning) rejects a missing/absent module outright;
    ``allow_none=True`` (used by ``run_backtest``, where ``strategy_mod=None``
    is a legal no-strategy loop) only checks non-None values.

    ``@strategy`` decorates a plain decision fn into a ``_StrategyAdapter`` whose
    ``__call__`` exposes ``ctx_fn`` (the decorated source fn). Only such adapters
    are backed by the cursor-safe ``TaContext`` prefetch the DSL needs; a raw
    ``on_candle(state, candle, params)`` module or hand-rolled class cannot be
    surfaced safely (no ``ctx.ta``, no per-run state holder). Raise rather than
    silently degrade — the DSL is the only authoring surface.
    """
    if strategy_mod is None:
        return  # no-strategy loop remains legal (data-only runs)
    on_candle = getattr(strategy_mod, "on_candle", None)
    if on_candle is None or getattr(on_candle, "ctx_fn", None) is None:
        what = type(strategy_mod).__name__
        raise TypeError(
            f"{what}.on_candle is not a @strategy adapter; the engine only runs "
            "decorated strategies. Wrap it with `from src.bt.strategies.dsl "
            "import strategy; @strategy(...)`."
        )


def run(
    bt: Backtest,
    data: pd.DataFrame,
    strat_mod,
    benchmark_curves: Optional[Mapping[str, pd.Series]] = None,
    signal_observer: Optional[Callable] = None,
    fundamentals: Any = None,
) -> BacktestResults:
    """Convenience function for running backtest with defaults.

    This creates default handlers and runs the backtest.  Use run_backtest()
    for full control.

    ``fundamentals`` is an optional prefetched ``Fundamentals`` store (see
    ``load_fundamentals``); it is attached to the CandleStore so decorated
    strategies reach it via ``ctx.fundamentals``. Typed ``Any`` on purpose:
    the engine never imports the concrete DSL context.
    """
    from src.bt.engine.handlers import default_execution_handler, default_risk_handler

    # Enforce the DSL-only contract: ``strat_mod`` must be a ``@strategy``
    # adapter (required positional — every live caller passes ``init_strat``).
    _assert_dsl_strategy(strat_mod, allow_none=False)

    gen = candle_generator(data, bt.config)
    exec_handler = default_execution_handler()
    risk_handler = default_risk_handler()

    # Every strategy here is a ``@strategy`` adapter, so a TaContext is always
    # built and minted per-run (never a module singleton): the DSL computes
    # every indicator it reads ONCE over the full feed, then does O(1)
    # cursor-truncated reads per candle -- no per-candle recompute, no
    # per-access DataFrame rebuild. A stateful adapter additionally gets a
    # fresh cross-candle state holder for every run/window so concurrent
    # split/sweep/optimize workers can't share or race on strategy state.
    on_candle = strat_mod.on_candle
    from src.bt.strategies.ta_context import init_ta

    ta = init_ta(data, bt.config.symbols, bt.config.bars[0])
    strategy_state = {} if getattr(on_candle, "stateful", False) else None
    if fundamentals is None:
        # Load the symbol set's as-first-stated fiscal series once per run. A
        # missing table / uncovered symbol yields empty series rather than
        # failing the run, so fundamentals stay an additive input.
        from src.bt.strategies.fundamentals_context import load_fundamentals

        fundamentals = load_fundamentals(bt.config.symbols)

    results, _ = run_backtest(
        bt,
        gen,
        exec_handler,
        risk_handler,
        strategy_mod=strat_mod,
        benchmark_curves=benchmark_curves,
        ta=ta,
        strategy_state=strategy_state,
        fundamentals=fundamentals,
        signal_observer=signal_observer,
    )
    return results
