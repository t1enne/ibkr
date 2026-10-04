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
  - A cash shortfall SCALES a cohort, it does not reject — only a lone open
    hits the legacy reject-or-full path. Scaling is silent: risk-sized
    entries land at scale × plan, RR vs plan shifts (per-share R unchanged).
    Grouping, not arithmetic, is the order-sensitive lever, so runs with
    rejections are advisory across symbol permutations (see
    ``portfolio.pure.apply_fills``).

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
from src.bt.warmup import parse_warmup_bars

if TYPE_CHECKING:
    from src.bt.types import StrategyConfig

from src.bt.state import (
    ActionType,
    BacktestState,
    Candle,
    EquityPoint,
    TradeSignal,
    FillEvent,
    ExecutionParams,
    PortfolioState,
    RiskConfig,
    create_initial_backtest_state,
    create_execution_params,
    create_risk_config,
    TradeExitReason,
)
from src.bt.size.pure import SizingParams, equity_of, sized_signal
from src.bt.portfolio.pure import (
    FillRejection,
    ScaleRecord,
    apply_fill,
    apply_fills,
)
from src.bt.execution.pure import execute_signal
from src.bt.types import (
    StrategyConfig,
    EngineWindow,
    BacktestResults,
    commission_model_from_config,
)
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
            warmup_bars=parse_warmup_bars(self.config.warmup, self.config.bars[0]),
            test_start=parse_timestamp(self.config.trading_start),
            test_end=parse_timestamp(self.config.trading_end),
        )
        self.execution_params = create_execution_params(
            spread_bps=self.config.spread_bps,
            slippage_bps=self.config.slippage_bps,
            fixed_commission=self.config.commission,
            commission_model=commission_model_from_config(self.config),
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
    _assert_benchmark_symbols_last(config)
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
    # Engine-owned sinks: fills dropped for insufficient cash land in
    # ``rejections`` and cohorts PARTIALLY scaled for cash land in ``scales``
    # (both pure records). They are surfaced as final ``PortfolioResult`` counts
    # (Rejected / Scaled), never as run-time stderr noise.
    rejections: list = []
    scales: list = []

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

    # Timestamp-grouped drain: candles are buffered by timestamp and the whole
    # bar is flushed through Stages 4-8 together, so Stage 4/6 settle as ONE
    # atomic cohort (see ``_flush_bar`` / ``apply_fills``). Buffering is what
    # makes a cohort COMPLETE before its first fill — the allocation is then a
    # single pure call, and ``config.symbols`` order drops out of the result by
    # construction rather than by memoisation. HTF candles never reach the
    # pipeline; they are stashed in Stage 1 and the bar flush skips them.
    bar: list[Candle] = []
    bar_ts: Optional[pd.Timestamp] = None
    for candle in candle_gen:
        if bar_ts is not None and candle.timestamp != bar_ts:
            state = _flush_bar(
                bar,
                state,
                exec_handler,
                risk_handler,
                config,
                bt,
                resolved_params,
                strategy_fn,
                last_symbol,
                rows,
                eq_buffer,
                rejections,
                scales,
                signal_observer,
            )
            bar = []
        bar_ts = candle.timestamp
        # Phase for this bar: bars strictly before ``test_start`` are warmup —
        # the strategy IS invoked (accumulators/indicators fill, cursor
        # advances) but trading is suppressed and emitting a signal is a hard
        # error in the DSL. Trading (and only trading) happens inside the
        # test window. ``warmup_bars == 0`` makes every bar a trading bar, so a
        # zero warmup is a pure no-op.
        state.candles.set_phase(
            "warmup" if candle.timestamp < bt.window.test_start else "trade"
        )
        # Stage 1: stash EVERY candle (base + HTF) into the same accumulator
        rows, state = _append_candle(rows, state, candle, config.bars[0])
        if not candle.interval or candle.interval == config.bars[0]:
            bar.append(candle)

    if bar:
        state = _flush_bar(
            bar,
            state,
            exec_handler,
            risk_handler,
            config,
            bt,
            resolved_params,
            strategy_fn,
            last_symbol,
            rows,
            eq_buffer,
            rejections,
            scales,
            signal_observer,
        )

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
        scaled_trades=sum(len(record.members) for record in scales),
        rejected_trades=len(rejections),
    )

    return (
        BacktestResults(
            pf=pf_result,
            data=state.candles,
            final_state=state,
            benchmark_curves=bm_curves,
            config=config,
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
    from src.bt.warmup import warmup_start

    if not config.benchmark_symbols:
        return {}

    try:
        bm_df = load_candles(
            config.benchmark_symbols,
            warmup_start(bt.window.test_start, config.warmup),
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


def _close_fill_qty(signal: TradeSignal, portfolio: PortfolioState) -> float:
    """Share count a close signal will actually flatten.

    Close signals from the DSL carry ``qty=0`` (the portfolio flattens the
    targeted lot); stamping the real lot size lets ``execute_signal`` scale the
    spread/slippage dollar cost by the shares actually traded.
    """
    lots = portfolio.positions.get(signal.symbol, ())
    if signal.position_id is not None:
        for pos in lots:
            if pos.position_id == signal.position_id:
                return abs(pos.qty)
    return sum(abs(pos.qty) for pos in lots)


def _execute_cohort(
    state: BacktestState,
    cohort: list[tuple[TradeSignal, Candle]],
    exec_handler: ExecutionHandler,
    exec_params: ExecutionParams,
    sizing: SizingParams,
    skip_next_open: bool,
    scale_cohorts: bool = True,
) -> tuple[BacktestState, tuple[FillRejection, ...], tuple[ScaleRecord, ...]]:
    """Build one phase's fills and settle them atomically in ONE pure call.

    ``cohort`` is ``(signal, candle)`` pairs for every symbol that has a pending
    fill in this phase of the bar, in ``config.symbols`` order. Each pair is
    sized (qty <= 0 opens only) against the SAME bar-start equity and priced with
    ``execute_signal`` against its own candle, then the whole batch is handed to
    ``apply_fills`` — which scales opens by one shared cash factor and applies
    non-opens first. Because the batch is complete before the first fill,
    ``config.symbols`` order cannot reach the result.

    Only the symbols present in ``cohort`` have their buckets drained; every
    other symbol keeps its pending signals (a symbol with no bar this timestamp
    fills at its next one). Returns the new state and the drained symbols.
    """
    if not cohort:
        return state, (), ()
    equity = equity_of(state.portfolio)
    fills: list[FillEvent] = []
    drained: list[str] = []
    for signal, candle in cohort:
        if skip_next_open and signal.fill_at_next_open:
            continue
        if signal.action in (ActionType.long, ActionType.short):
            signal = sized_signal(signal, equity, state.portfolio.cash, candle, sizing)
        elif signal.action == ActionType.close and signal.qty <= 0:
            signal = replace(signal, qty=_close_fill_qty(signal, state.portfolio))
        fills.append(exec_handler.execute_signal(signal, candle, exec_params))
        drained.append(signal.symbol)
    portfolio, rejections, scales = apply_fills(
        state.portfolio,
        tuple(fills),
        scale_cohorts=scale_cohorts,
        commission_model=exec_params.commission_model,
    )
    pending = dict(state.pending_signals)
    for sym in drained:
        pending.pop(sym, None)
    state = merge_bt_state(state, dict(portfolio=portfolio, pending_signals=pending))
    return state, tuple(rejections), tuple(scales)


def _mark_bar(state: BacktestState, bar: list[Candle]) -> BacktestState:
    """Set each bar symbol's open lots to that symbol's close. No equity point.

    Pure ``replace`` over the positions dict. Used at evaluation time so every
    symbol's mark is current regardless of its place in ``config.symbols``;
    Stage 8 re-marks (idempotent) and records the equity point.
    """
    positions = dict(state.portfolio.positions)
    changed = False
    for candle in bar:
        lots = positions.get(candle.symbol)
        if not lots:
            continue
        positions[candle.symbol] = tuple(
            replace(p, last_price=candle.close) for p in lots
        )
        changed = True
    if not changed:
        return state
    return merge_bt_state(
        state, dict(portfolio=replace(state.portfolio, positions=positions))
    )


def _flush_bar(
    bar: list[Candle],
    state: BacktestState,
    exec_handler: ExecutionHandler,
    risk_handler: RiskHandler,
    config: StrategyConfig,
    bt: Backtest,
    resolved_params: object,
    strategy_fn: Optional[Callable],
    last_symbol: Optional[str],
    rows: CandleRows,
    eq_buffer: list,
    rejections: list,
    scales: list,
    signal_observer: Optional[Callable],
) -> BacktestState:
    """Run one timestamp's base candles through Stages 3-8, cohort-atomic.

    Stage 3 (next-open fills of PRIOR signals) and Stage 6 (same-bar fills of
    THIS bar's signals) each settle as one ``apply_fills`` call over every symbol
    of the bar; Stage 5 runs only on the last symbol (the evaluation clock).
    Stage 7/8 stay per symbol, in ``config.symbols`` order, exactly as before —
    only the fill settlement became atomic.

    The Stage 3 cohort is complete before its first fill because
    ``fill_at_next_open`` signals are all still pending when the bar opens, which
    is what makes atomic settlement possible at all. Warmup bars run Stage 5
    only (state warms, no fills, no risk, no marking).
    """
    ts = bar[0].timestamp
    in_warmup = ts < bt.window.test_start
    can_trade = bt.window.test_start <= ts <= bt.window.test_end

    if in_warmup:
        last = bar[-1]
        if last.symbol == last_symbol:
            state = _generate_signals(
                state, last, resolved_params, strategy_fn, last_symbol, True, rows
            )
        return state

    cohort4 = [
        (sig, candle)
        for candle in bar
        for sig in state.pending_signals.get(candle.symbol, ())
    ]
    state, rejected, scaled = _execute_cohort(
        state,
        cohort4,
        exec_handler,
        bt.execution_params,
        bt.sizing,
        False,
        bt.config.cohort_scaling,
    )
    rejections.extend(rejected)
    scales.extend(scaled)

    # Mark EVERY symbol of the bar at THIS bar before the strategy runs. Stage 8
    # marks positions only AFTER Stage 5, so at evaluation time the symbols the
    # bar has not yet marked are stale by one bar. Marking only the evaluation
    # clock (symbols[-1]) left its PEERS stale instead, so ``current_equity()`` —
    # and with it ``size_mode="equity"`` sizing — depended on which symbol
    # happened to be last. Marking the whole bar makes every symbol equally
    # current, so dispatch equity is independent of ``config.symbols`` order.
    # Stage 8's later mark is idempotent (same close) and still appends the point.
    state = _mark_bar(state, bar)

    state = _generate_signals(
        state,
        bar[-1],
        resolved_params,
        strategy_fn,
        last_symbol,
        can_trade,
        rows,
        signal_observer=signal_observer,
    )

    cohort6 = [
        (sig, candle)
        for candle in bar
        for sig in state.pending_signals.get(candle.symbol, ())
    ]
    if cohort6:
        state, rejected, scaled = _execute_cohort(
            state,
            cohort6,
            exec_handler,
            bt.execution_params,
            bt.sizing,
            True,
            bt.config.cohort_scaling,
        )
        rejections.extend(rejected)
        scales.extend(scaled)

    for candle in bar:
        state = _check_risk(
            state,
            candle,
            exec_handler,
            risk_handler,
            bt.execution_params,
            bt.risk_config,
        )
        state = _mark_to_market(state, candle, eq_buffer)
    return state


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

    # Mark the LAST symbol (the evaluation clock) at THIS bar before the
    # strategy runs. Stage 8 marks the clock's positions only AFTER Stage 5, so
    # without this the clock symbol's mark is stale by one bar at evaluation
    # time — a tail-symbol-specific asymmetry (the same class as the truncated
    # evaluation clock): ``current_equity()`` then differs purely because the
    # clock symbol happens to be last, and ``size_mode="equity"`` sizing is
    # order-dependent. Marking it here makes every symbol equally current, so
    # equity at dispatch is independent of ``config.symbols`` order. Stage 8's
    # later mark is idempotent (same price) and still appends the equity point.
    clock_lots = state.portfolio.positions.get(candle.symbol)
    if clock_lots:
        marked = tuple(replace(p, last_price=candle.close) for p in clock_lots)
        positions = dict(state.portfolio.positions)
        positions[candle.symbol] = marked
        state = merge_bt_state(
            state, dict(portfolio=replace(state.portfolio, positions=positions))
        )

    new_signals = strategy_fn(state, candle, resolved_params)
    if not new_signals:
        return state

    if signal_observer is not None:
        for sig in new_signals:
            signal_observer(sig)

    # Bucket by symbol. Allocation is NOT stamped here: it needs the cohort's
    # fills (priced at the next bar) and the pre-execution book, which only exist
    # at the bar flush. ``apply_fills`` (in the portfolio layer) settles the
    # whole cohort in one order-invariant call, so generation stays a pure
    # bucket+merge step.
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
                position_side=position.type,
                qty=abs(position.qty),
                fill_at_next_open=False,
            )

            # Route the end-of-run flatten through the SAME friction path as
            # every other fill, so a closed book pays the spread/slippage it
            # really would and records the costs the report reconciles against.
            fill = execute_signal(
                close_signal,
                Candle(
                    timestamp=close_signal.timestamp,
                    symbol=symbol,
                    open=position.last_price,
                    high=position.last_price,
                    low=position.last_price,
                    close=position.last_price,
                    volume=0.0,
                ),
                exec_params,
            )

            portfolio = apply_fill(portfolio, fill)

    # Freeze the engine-buffered equity curve onto the final portfolio.
    if equity_points is not None:
        # The end-of-run flatten pays real friction (it routes through
        # ``execute_signal``), so the book's post-close cash is NOT the last
        # bar's mark-to-market equity. Stamp one final point AFTER the flatten
        # settles so the frozen curve's last value equals the post-close book
        # — otherwise the trades include the end-of-run friction while the
        # equity-based Net (curve[-1] - curve[0]) misses it, and the report's
        # reconciliation identity (Net = Gross - Commission - Spread/Slip)
        # breaks by exactly that cost.
        equity_points = list(equity_points)
        ts = state.timestamp or pd.Timestamp.now()
        equity_points.append(
            EquityPoint(
                timestamp=ts,
                equity=portfolio.cash,
                cash=portfolio.cash,
                positions_value=0.0,
            )
        )
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


def _assert_benchmark_symbols_last(config: StrategyConfig) -> None:
    """Assert every tradable benchmark symbol sits at the tail of ``symbols``.

    The engine fires ``on_candle`` once per timestamp, only on
    ``config.symbols[-1]``. That symbol's bar is therefore the run's
    *evaluation clock*: the engine only visits timestamps the last symbol has
    a bar for. A benchmark that gates entries (regime/trend gate) reads its
    own series from the candle store, so if it is ALSO a configured symbol but
    is placed anywhere before the tail, the run silently evaluates on the tail
    symbol's calendar instead — a later-IPO or sparser tail symbol drops every
    timestamp the gating benchmark lacks, and the same config yields different
    results purely from symbol order.

    Benchmarks are observers, never tail-competing trade targets, so this is a
    hard config error: reject instead of degrading silently.

    This is the benchmark-specific case of the general rule below (see
    ``_assert_evaluation_clock_covers_window``): ``symbols[-1]`` IS the run's
    evaluation clock, and nothing but a full-history symbol may occupy it.
    """
    symbols = config.symbols
    if not symbols:
        return
    benchmarks_in_feed = [b for b in config.benchmark_symbols if b in symbols]
    if not benchmarks_in_feed:
        return
    tail = symbols[len(symbols) - len(benchmarks_in_feed) :]
    if tail == benchmarks_in_feed:
        return
    offenders = [f"{b!r} at index {symbols.index(b)}" for b in benchmarks_in_feed]
    raise AssertionError(
        "benchmark symbols present in config.symbols must be the LAST entries "
        f"(in benchmark_symbols order); got {symbols!r} with benchmark(s) "
        f"{', '.join(offenders)}. on_candle fires only on symbols[-1], so the "
        "evaluation clock must be a benchmark, not a tradable tail symbol."
    )


def _assert_evaluation_clock_covers_window(
    data: pd.DataFrame, config: StrategyConfig
) -> None:
    """Assert the evaluation clock's first bar precedes the trading window.

    ``symbols[-1]`` IS the run's evaluation clock (see
    ``_assert_benchmark_symbols_last``): the engine fires ``on_candle`` only
    on that symbol's candle, so every timestamp the tail symbol lacks a bar
    for is a timestamp the WHOLE strategy is never evaluated on — for every
    symbol, not just the late one. A tail symbol whose history starts inside
    the trading window (later IPO, or data synced from a later date) therefore
    silently truncates the run, and merely reordering ``symbols`` yields a
    different — usually far better — result from identical data and params.

    The check is exact, not tolerance-based: the clock must have a bar at or
    before ``trading_start`` so the strategy is live for the whole window.
    Bars before ``trading_start`` never fire (warmup has ``can_trade=False``),
    so a tail that starts anywhere in ``[feed_start, trading_start]`` is fine —
    the pre-existing multi-day spread between symbols' first bars (e.g.
    2019-11-06 vs 2019-11-15) is normal and must NOT trip this. Only a tail
    whose FIRST bar lands after the window opens is rejected: that is the case
    that silently degrades, and it is unambiguous given the loaded feed.

    Raise rather than degrade: the user accepted throwing until the engine can
    pick a per-timestamp clock itself.
    """
    symbols = config.symbols
    if len(symbols) < 2:  # nothing to be late *relative to*; single/empty ok
        return
    tail = symbols[-1]
    try:
        block = data.xs(tail, axis=1)
    except KeyError:
        return  # symbol absent from this feed: not a clock-truncation case
    own = block.dropna(subset=["close"])
    trading_start = parse_timestamp(config.trading_start)
    if len(own) == 0:
        raise AssertionError(
            f"evaluation clock {tail!r} (symbols[-1]) has no candles in the "
            f"loaded feed at all; on_candle fires only on it, so the strategy "
            f"would never be evaluated. Remedy: reorder config.symbols so a "
            f"full-history symbol is last, or sync {tail!r}'s history."
        )
    first = own.index.min()
    if first <= trading_start:
        return
    starts = {
        s: data.xs(s, axis=1).dropna(subset=["close"]).index.min()
        for s in symbols
        if s != tail
    }
    earliest = min(starts.values())
    earliest_syms = [s for s, ts in starts.items() if ts == earliest]
    raise AssertionError(
        f"evaluation clock {tail!r} (symbols[-1]) starts {first} — AFTER "
        f"trading_start {trading_start} — while {earliest_syms!r} start "
        f"{earliest}. on_candle fires only on symbols[-1], so every timestamp "
        f"before {first} is never evaluated for ANY symbol: the run is silently "
        f"truncated to the tail symbol's calendar. Remedy: reorder "
        f"config.symbols so a full-history symbol is last, or sync {tail!r}'s "
        f"history back to trading_start."
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
    # Reject a config whose symbols[-1] (the evaluation clock) starts inside the
    # trading window — the engine fires only on it, so the run would be silently
    # truncated to the tail symbol's calendar (see the assertion's docstring).
    # Checked here (not in run_backtest) because only ``run`` holds the feed.
    _assert_evaluation_clock_covers_window(data, bt.config)

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
