# HANDOFF — Unify Screens With the Backtest Pipeline

Implements: the screen command ("bt screen") must run a **backtest-like per-candle
pipeline that reuses strategy modules as-is**, instead of the bespoke stateless
screen-scoring layer. Full architectural analysis concluded; this is the
locked, implementable spec. All facts below verified against the repo.

## Decisions (LOCKED — do not reopen)

- Approach **B** = real engine run + **engine emit-capture hook (#1)**.
- Delete `rs` and `momentum_spy_gate` → the entire `src/bt/screen/screens/`
  bespoke scoring layer goes away. Momentum surface lives on via its real
  strategy `momentum_compression_breakout_dsl`.
- Rewrite `screen` command completely. Reuse DSL strategy modules unchanged.
- **No backwards compat** on old screen CLI/semantics/TF-merge.
- Single base interval (`config.bars[0]`), like a backtest. **Drop** the
  multi-interval `rank_divergence` TF-consensus merge.
- **Coiling/"watchlist armed" tier is NOT built** (YAGNI — a DSL strategy
  cannot express "about to trigger"; surface only what it actually emits).

## 1. What exists today (verified)

**Backtest** — `src/bt/engine/backtest.py`:
- `Backtest(config)` builds `window` (EngineWindow from `training_start/end`,
  `trading_start/end`), `execution_params`, `risk_config` (SL/TP 0.0), `sizing`.
- `candle_generator(df, config)` (`src/bt/engine/utils.py:35`) yields Candles
  from a MultiIndex `(symbol, timestamp)` OHLCV feed (base + HTF interleaved).
- `run_backtest(bt, candle_gen, exec_handler, risk_handler, *, initial_state,
  strategy_mod, benchmark_curves, ta, strategy_state) -> (BacktestResults,
  BacktestState)` is the pure per-candle core. **`_finalize` runs inside
  `run_backtest` (~line 131)** and force-closes every position, zeroes
  `pending_signals`, freezes equity.
- `run()` is a thin wrapper that mints a fresh `TaContext` (for DSL) and
  per-run `strategy_state` dict, then calls `run_backtest`.

**The killer constraints** (why a stock run cannot be a screen):
1. A `long`/`short` emitted on the **final data bar never fills**
   (`fill_at_next_open` deferred to a bar that doesn't exist) and is then
   **discarded by `_finalize`**. That unfilled emission is the fresh
   manual-trade signal a screen must show.
2. `_finalize` flattens `portfolio.positions` → `final_state` book is always
   empty. No "currently set up" view survives.
3. Only the closed `trades` ledger survives — cannot express "just fired now".

**Strategy surface** — DSL (`src/bt/strategies/`, `@strategy(bars=..., stateful=...)`
decorator in `dsl.py`) compiles a user `def on_candle(ctx)` into an engine hook
`on_candle(state, candle, params) -> list[TradeSignal]`. Many are stateful
(`ctx.shared`), need the engine's prefetched `TaContext` + fresh per-run
`strategy_state`. Emits `ActionType{long, short, close, rebalance}` with human
`reason` strings (e.g. `"[breakout] long close 123.4 > VAH ..."`). Close paths
only fire while a lot is open. `init_strat(strategy_type)` auto-discovers
(`src/bt/strategies/__init__.py`); `resolve_params(strategy_type,
config.strategy_params)` instantiates typed `Params`.

**Feed** — `src/bt/data_feed/__init__.py::load_candles(symbols, start, end,
bar)` returns the MultiIndex OHLCV frame; `run_backtest_results`
(`src/bt/__init__.py`) drives it as `load_candles(config.symbols,
bt.window.train_start, bt.window.test_end, config.bars[0])`. DB path: sqlite
`../data/db.sqlite` relative to repo parent, ~1.8M rows, native 1h granularity,
other bars resampled on read. Core tickers back to 2019-11; SPY to 2004.

**Screen (to be rewritten)** — `src/bt/cmds/screen.py` currently: `screen NAME
--symbols/-U --interval -i (repeatable, default 1d) --from --to --params --top`
→ loads frames once per interval via `screen/adapter.state_per_interval`, builds
static `ScreenState` per interval scoring only the latest bar, optional
`rank_divergence` cross-TF merge, renders ranked table
(`table.render_from_dicts`, `COMMON_COLS = ema_50, ema_100, atr_14, rsi_14,
hi_52w, lo_52w`). `src/bt/screen/{adapter,runner,metrics}.py` + `screens/` +
`types.py` are the bespoke layer to delete. `ScreenResult{symbol,timestamp,
score,action,signals,model_features}` is the row shape but tied to `ScreenState`.

## 2. The fix (minimal, behavior-neutral engine hook)

`run_backtest` is fully reusable except its sink discards current-bar intent.
Add ONE optional keyword to `run_backtest` and thread it into the
`_generate_signals` stage (`src/bt/engine/backtest.py`, `_generate_signals`
fires `new_signals = strategy_fn(state, candle, resolved_params)` on the last
symbol per timestamp, pre-fill and pre-finalize). Invoke the observer once per
fresh `TradeSignal`. When the observer is `None` (the default), `run_backtest`
behaves byte-for-byte as today — **zero regression surface**. `run()` stays
unchanged (the screen driver calls `run_backtest` directly).

## 3. Files — created / modified / deleted

**MODIFIED — `src/bt/engine/backtest.py`**
Add `signal_observer: SignalObserver | None = None` param to `run_backtest`;
thread to `_generate_signals`; call once per fresh signal right where
`new_signals` is produced (before bucketing). No other engine change.

**CREATED — `src/bt/screen/run_strategy.py`** (new driver, replaces bespoke layer)
**REWRITTEN — `src/bt/cmds/screen.py`**
**DELETED — `src/bt/screen/screens/` (whole), `src/bt/screen/adapter.py`,
`src/bt/screen/runner.py`, bespoke parts of `src/bt/screen/metrics.py`,
`ScreenState`-tied `types.py`. Re-home any still-used per-symbol common-metric
helpers (`with_common_metrics`, `common_metrics`) into `run_strategy.py`.**
**DELETED — old screen tests**: `src/bt/screen/tests/test_adapter.py`,
`src/bt/screen/tests/test_momentum_spy_gate.py`.
**REVIEW** `scripts/run_screens.py` for removal/adjustment (references
`screen_adapter`/discovery; verify it still exists before assuming — earlier
`ls scripts/` did not list it).

## 4. Types (all new; keep output row stable for table consumers)

```python
# in src/bt/engine/backtest.py
SignalObserver = Callable[[TradeSignal], None]  # saw fresh strategy signal, pre-fill/pre-finalize

# in src/bt/screen/run_strategy.py
from typing import Literal
Action = Literal["long", "short", "flat"]
SignalsCollected = tuple[TradeSignal, ...]  # chronological observer feed, one run

@dataclass(frozen=True)
class ScreenRow:
    symbol: str
    action: Action          # derived (see posture below)
    score: float            # >0 iff actionable; 1.0 fresh open, <1.0 retained
    signals: tuple[str, ...]# human reasons from TradeSignal.reason strings
    ts: pd.Timestamp

@dataclass(frozen=True)
class Posture:              # intent ledger, latest-wins per symbol
    base_score_open: float = 1.0   # fresh open signal on newest bar
    base_score_held: float = 0.8   # prior intent, none newer this window
    include_flat: bool = True      # emit explicit flat rows for universe
```

**Posture rule** (from observer feed, per symbol): a fresh `long`/`short` sets
that symbol toward the given action at `base_score_open`; a `close`
clears/reverts to flat; a `rebalance` keeps prior side. Track only the LAST
distinct per-symbol instruction in the run; if the newest found signal shares
the newest-scored bar's timestamp → treat as fresh-open, else retained. This is
pure **intent**, never real fills — screens do not trade, so no one-bar lag and
no `_finalize` flatten issue.

## 5. Function signatures

```python
# engine/backtest.py
def run_backtest(  # ONLY ADDITION: signal_observer param + one call site
    bt: Backtest,
    candle_gen: Generator[Candle, None, None],
    exec_handler: ExecutionHandler,
    risk_handler: RiskHandler,
    *,
    initial_state: Optional[BacktestState] = None,
    strategy_mod: Any = None,
    benchmark_curves: Optional[Mapping[str, pd.Series]] = None,
    ta: Optional[TaContext] = None,
    strategy_state: Optional[dict] = None,
    signal_observer: SignalObserver | None = None,
) -> Tuple[BacktestResults, BacktestState]
# role unchanged; when signal_observer set, invoke once per fresh TradeSignal.
```

```python
# src/bt/screen/run_strategy.py
class SignalCollector:
    """Collects observed signals; presents final posture per symbol."""
    def __init__(self, symbols: tuple[str, ...]) -> None: ...
    def on_signal(self, sig: TradeSignal) -> None: ...       # observer target
    def postures(self) -> dict[str, TradeSignal]: ...        # symbol -> latest intent

def run_screen_from_strategy(
    config_path: str,
    posture: Posture = Posture(),
) -> tuple[tuple[ScreenRow, ...], BacktestState]
# Load StrategyConfig, feed, Backtest, handlers, TaContext, fresh strategy_state;
# drive run_backtest with a SignalCollector; return projected rows + final state.

def _load_feed(config: StrategyConfig):          # data_feed.load_candles(...) -> df
def _latest_ts(config: StrategyConfig): ...      # latest data timestamp (scoring bar)
def _project(collector, symbols, posture): tuple[ScreenRow,...]
def _score(sig: TradeSignal | None, fresh: bool, posture: Posture) -> float
```

```python
# cmds/screen.py (REWRITTEN)
@click.command("screen")
@click.argument("strategy_file")                 # same config a `bt run` consumes
def screen(strategy_file: str) -> None:
    """Score a universe by running its strategy through the real engine."""
    # -> run_strategy.run_screen_from_strategy(config_path)
    # -> render rows via table.render_from_dicts (keep TABLE_COLS output shape)
```

## 6. Call graph (Production)

```ts
cmds.screen.screen(strategy_file: str): None
  → run_strategy.run_screen_from_strategy(config_path: str, posture: Posture): tuple[tuple[ScreenRow,...], BacktestState]
    → load_strategy(path: str): StrategyConfig                          --> src.bt.load_strategy
    → _load_feed(config: StrategyConfig): pd.DataFrame                  --> data_feed.load_candles(symbols, train_start, test_end, bars[0])
    → engine.Backtest(config)                                           --> engine/backtest.py
    → handlers.default_execution_handler(): ExecutionHandler            --> engine/handlers.py
    → handlers.default_risk_handler(): RiskHandler
    → ta.init_ta(df, config.symbols, config.bars[0]): TaContext         --> strategies/ta_context.py (only when DSL)
    → init_strat(config.strategy_type): strategy module                 --> strategies/__init__.py
    → utils.candle_generator(df, config): Generator[Candle]             --> engine/utils.py:35
    → engine.run_backtest(bt, gen, exec, risk, strategy_mod=..., ta=..., strategy_state={},
                          signal_observer=collector.on_signal): (BacktestResults, BacktestState)
      → engine._generate_signals(state, candle, resolved_params, strategy_fn, last_symbol,
                                 can_trade, rows): BacktestState
        → collector.on_signal / collector postures                        // FRESH signal captured here, pre-finalize
      → engine._finalize(state, exec_params, equity_points): BacktestState // book flattened; rows already captured
    → _project(collector, config.symbols, posture): tuple[ScreenRow, ...]
  → table.render_from_dicts(...) : rows to stdout                         --> src.bt.table
```

Tests flow:

```ts
screen/tests/test_run_strategy.py
  → build synthetic MultiIndex OHLCV feed (fixture) + Backtest config
  → _load_feed(...) : feed (or inject frame)
  → engine run with a stub/simple strategy
  → SignalCollector.on_signal(...) : captured
  → _project(...) : assert row action/score per posture rule
```

## 7. Verification (runnable, assertion-based)

1. `run_backtest(..., signal_observer=None)` returns results identical to the
   current implementation (no behavioral drift). Run an existing engine test
   unchanged.
2. `_project` with a stub observer feed: symbol got `long` on newest bar →
   `action=="long"`, `score==posture.base_score_open`; a `close` after → flat.
3. DSL momentum strategy over a small fixture that provably triggers once →
   exactly one fresh `long` row scored 1.0; a no-trigger window → all flat.
4. `bt screen strats/pass/momentum_compression_breakout_ae_gate_SPY.json`
   prints a non-empty ranked table (real feed present in local DB).
5. Observer fires only on fresh `_generate_signals` output — never doubled by
   `_execute_pending` re-execution (which reads pending buckets, not new emits).

## 8. YAGNI (NOT built)

- Coiling/"armed watchlist" DSL intent (`ctx.watch`) — blocked by a real
  limitation; revisit only if the momentum manual work demands discovery of a
  building-but-not-triggered setup.
- `screen --walk` over history; a fresh per-bar walker — later if needed.
- Real filled-position column / PnL in screen rows (screens never trade).
- Multi-interval TF-consensus merge — dropped by decision.

## Ground rules (repo AGENTS.md)

- Python 3.14+, `uv`, fully-typed code (`ty` typechecker + `ruff format`), no
  `Any` w/o comment, frozen dataclasses for state, pure functions, `make check`
  (lint + format + typecheck + tests) must pass. Every new computation gets a
  test. Functions ≤ 50 LOC, classes ≤ 150 LOC. Vectorize hot paths.
