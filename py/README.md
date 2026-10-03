# IBKR PY — Composable Quantitative Trading Toolkit

Modular CLI toolkit for quantitative trading: data synchronization, indicator computation, statistical models, and a functional backtesting engine. Configure strategies in JSON, run from the terminal, iterate fast.

## Quickstart

```bash
uv sync
uv run ibkr bt run strats/pass/<config>.json
uv run ibkr bt run strats/pass/<config>.json --trades   # full trade list (default off)
make run bt run strats/pass/<config>.json   # same, via Make shortcut
```

Configs live classified under `strats/{pass,wip,fail}/` (see `SKILL.md`);
the module behind each is `strategy_type` in the JSON.

> **Agents: read [SKILL.md](SKILL.md) before writing or changing any strategy.**
> It is the strategy-authoring contract. Do not infer authoring conventions
> from this README or from `src/bt/strategies/` alone.

`make run` with no args shows available commands:

```bash
make run bt -- --help              # Click help for backtesting subgroup
```

**Make eats its own flags.** To pass CLI options through to `ibkr` (`--format`, `--help`, `-F`)
you must separate them with `--`:

```bash
make run bt run strats/pass/<config>.json -- --format jsonl
```

Run `make run <args>` for zero extra typing; use `uv run ibkr <args>` directly to avoid the `--` syntax.

## Agent Rules (read before touching strategies)

Hard rules for AI agents. The usual failure is not a broken backtest — it's an
agent burning a session rebuilding machinery the toolkit already has.

1. **Read [SKILL.md](SKILL.md) first.** Any strategy authoring, editing, tuning
   or evaluation starts there. Skipping it causes rules 2–5.
2. **Never test a strategy.** Strategies, `strats/*.json`, research and sweeps
   are exempt from `AGENTS.md`'s test rule — no `test_<strategy>.py`, no
   fixtures. Tests cover *engine* code only. Validate strategies by **running** them.
3. **Use `bt run` / `bt sweep` / `bt split` / `bt optimize` — no throwaway
   harnesses.** A bespoke candle-load + grid-loop + `run()` script is forbidden;
   that is `bt sweep` (`bt optimize` if folds must also validate OOS). Custom
   scripts only beyond this surface — say so first.
4. **Never mangle output.** No `grep`/`head` when reporting, no re-deriving
   metrics in scratch code. Read the full output; the risk story (kurtosis,
   skewness, stability, per-symbol bleed) is in the tail. Name any trim.
5. **`--workers` is throughput, not correctness.** Raise it for many combos or
   folds; keep `1` for trivial runs.

## Architecture

Pure functions + immutable state. No classes, no side effects in the hot path.

### Pipeline (per candle)

```
append_candle → execute_pending → generate_signals
     → execute_pending (same-bar) → check_risk → mark_to_market
```

- Signals fill at next bar's open (fill_at_next_open) or same bar (immediate).
- Risk checks (stop-loss / take-profit) run after same-bar fills, so they always see the post-rebalance position.
- Higher-timeframe candles accumulate alongside base candles; HTF-only candles skip the pipeline.

### Composition

Strategies are authored on the **strategy DSL** (`@strategy()`-decorated functions
of a `StrategyContext`; see `src/bt/strategies/dsl.py`). Decoration produces an
adapter that plugs into auto-discovery — every module exposing `STRATEGY_TYPE`
is registered. The DSL owns cursor-safe indicator prefetch, per-symbol signal
bucketing, and (for `stateful` strategies) a per-run `ctx.shared` holder. The
engine core still takes a callable `on_candle(state, candle, params)` — the DSL
adapter satisfies that shape, and test doubles / programmatic `run()` callers
may too — but authoring surface is DSL-only. Handlers (`ExecutionHandler`,
`RiskHandler`) are dataclasses wrapping injectable functions.

### Multi-Symbol & Multi-Interval Data Flow

#### Per-timestamp iteration

The candle generator yields every symbol at every timestamp, then interleaves any
higher-timeframe (HTF) candles sharing the same timestamp. Base-interval candles
run through the full pipeline; HTF-only candles are merely accumulated and skip
the rest.

With `symbols: ["AAPL", "GOOGL", "MSFT"]` and `bars: ["1h", "4h"]` each
1h timestamp produces:

```
AAPL:1h  → full pipeline (append, signals, risk, mtm)
GOOGL:1h → full pipeline
MSFT:1h  → full pipeline
AAPL:4h  → append only (if 4h boundary)
GOOGL:4h → append only (if 4h boundary)
MSFT:4h  → append only (if 4h boundary)
```

#### `on_candle` invocation rule

`on_candle` fires **only on the last symbol per timestamp** (i.e. `candle.symbol
== symbols[-1]`). The `candle` parameter carries that symbol's OHLCV + interval.

**Why:** This guarantees `state.candles` has accumulated every symbol's data
up to the current timestamp, so the strategy can read cross-sectional (multi-symbol)
decisions in one shot. If the engine fired `on_candle` on every symbol,
the first symbol's invocation would see incomplete data for later symbols.

#### `state.candles` — the CandleStore

All candles — base and HTF — accumulate in `state.candles`, a `Mapping[(str, str),
DataFrame]` keyed by `(symbol, interval)`. It is wrapped in a `CandleStore` that
supports lookahead-safe access:

```python
# Get DataFrame up to current timestamp (safe — no future data)
df = state.candles[("AAPL", "1h")]
df = state.candles.get(("AAPL", "4h"))           # None if not present

# O(1) fast path — latest close or bar count (no DataFrame allocation)
close = state.candles.latest("AAPL", "1h")       # float | None
n     = state.candles.count("AAPL", "4h")         # int
```

Before `on_candle` fires, the engine calls `state.candles.advance(current_ts)`,
which sets a cursor. All DataFrame-building access (`__getitem__`, `get`,
iteration) truncates rows beyond the cursor. Fast-path methods (`latest`,
`count`) ignore the cursor and return the absolute latest value.

#### `candle.interval` — which bar is this?

`ctx.candle.interval` is `"1h"` for base bars, `"4h"` for HTF bars, etc. The
engine fires the strategy only on base-interval candles, so ``ctx.candle.interval``
is the signal interval at every call. Multi-interval strategies that also read
HTF structure reach the accumulated store via ``ctx.state.candles`` (cursor-
truncated):

```python
@strategy(bars="1h")
def on_candle(ctx):
    # HTF trend read straight from the store (ctx.ta serves only the base interval)
    htf_df = ctx.state.candles.get((ctx.candle.symbol, "4h"))
    if htf_df is not None and len(htf_df) > 0:
        htf_close = htf_df["close"].iloc[-1]
    ...
```

#### Multi-symbol strategy pattern

The engine has already accumulated every symbol up to the current timestamp
when the strategy runs (fires on the last symbol), so cross-sectional reads are
complete. Read any symbol from ``ctx.state.candles``, then emit signals per
target symbol with ``ctx.long/close``. Returned signals are bucketed by
``signal.symbol`` — the engine drains each symbol's queue when its candle
iteration reaches the execution stage. Prefer a typed ``Params`` dataclass
(fields reachable as attributes on ``ctx.params``; see ``StrategyParams``):

```python
from dataclasses import dataclass
from src.bt.strategies.types import StrategyParams
from src.bt.strategies.dsl import strategy

STRATEGY_TYPE = "momentum_top"

@dataclass(frozen=True)
class Params(StrategyParams):
    warmup: int = 20
    top_n: int = 1
    size: float = 0.05

@strategy(bars="1h")
def on_candle(ctx):
    params: Params = ctx.params
    closes = {}
    for sym in ctx.symbols:
        df = ctx.state.candles.get((sym, ctx.interval))
        if df is not None and len(df) >= params.warmup:
            closes[sym] = df["close"].iloc[-1]
    for sym in sorted(closes, key=lambda s: closes[s])[: params.top_n]:
        ctx.long(sym, size=params.size)   # engine routes this signal to sym's bucket
```

Signals carry `signal.symbol` — the engine routes each signal to its
symbol's bucket in `state.pending_signals` (a `dict[str, tuple[TradeSignal,
...]]`). Execution is O(1) per symbol, no scan over unrelated signals.

```
src/
├── data/             ← IBKR market data sync, DB, resampling
├── indicators/       ← TA, Kalman filters, HMM regime detection
├── bt/               ← Backtesting engine
│   ├── engine/       ← Backtest loop (pure functional), candle store
│   ├── strategies/   ← Strategy implementations (auto-discovered)
│   ├── state/        ← Immutable dataclasses + factories
│   ├── types.py      ← StrategyConfig, EngineWindow, Protocol types
│   ├── portfolio/    ← Position sizing, fill application, mark-to-market
│   ├── execution/    ← Signal → fill with slippage/spread
│   ├── risk/         ← Stop-loss/take-profit checks
│   ├── regime/       ← Regime detection (HMM, trend, volatility)
│   ├── metrics.py    ← Sharpe, Sortino, Calmar, drawdowns
│   └── data_feed/    ← Load/sync OHLCV candles
├── utils.py          ← Shared utilities
└── main.py           ← CLI entry point (Click)
```

## CLI Reference

```bash
# Backtesting
uv run ibkr bt run <strategy.json> [--format jsonl] [--trades]   # --trades = full trade list (off by default)
uv run ibkr bt split <strategy.json> --folds 4          # IS/OOS walk-forward
uv run ibkr bt split <strategy.json> --is-end 2020-12-31  # single anchor split
uv run ibkr bt sweep <strategy.json> '{...grid...}'     # hyperparameter sweep
uv run ibkr bt optimize <strategy.json> '{...grid...}' --folds 4  # per-fold IS tune → OOS validate
uv run ibkr bt screen <strategy.json>                   # current-bar intent (opens AND closes)
uv run ibkr bt screen <strategy.json> --trades          # + executed-trade table (default off)
# All three parallelize units across worker processes with:
#   --workers N   (default 1 = sequential)

# Inspect what's on disk (recap: date range, rows, gaps >48h) — no Gateway needed
uv run ibkr data query AAPL
uv run ibkr data query --universe universes/nsdq.json

# Download / refresh candles (see `data dl` runbook below)
uv run ibkr data dl AAPL MSFT --from 2019-01-01

# Pipe workflows
uv run ibkr data query AAPL --from 2024-01-01 | uv run ibkr bt run strategy.json
```

### `data dl` — download / refresh OHLCV

```bash
uv run ibkr data dl AAPL MSFT --from 2019-01-01   # backfill + refresh tail
uv run ibkr data dl --universe universes/nsdq.json --from 2019-01-01
```

Idempotent (`on_conflict_ignore`); safe to call as-is. `--from` sets the history
floor — earlier = deeper backfill plus the trailing tail in one call. Its
`0 fetch gaps`/`up to date` tail is *post-download* output and can print even on
a successful fill; always confirm with:

```bash
uv run ibkr data query AAPL        # max date should advance
```

Needs an authenticated IBKR Gateway at `https://localhost:5000/v1/api`:

```bash
uv run python -c "import httpx;print(httpx.get('https://localhost:5000/v1/api/iserver/auth/status',verify=False,timeout=8).json().get('authenticated'))"  # True = ready
uv run python scripts/login_ibkr.py   # else login (.env holds creds)
```

`data query`/`preview` need no login — only `data dl` touches the live Gateway.

### `bt sweep` — hyperparameter sweep

Sweeps a param grid over a strategy and ranks every combo by a chosen metric
(`--sort-by`, default `annual_return`). `PARAM_GRID` is a partial-config JSON
that deep-merges into the strategy config; any value that is a **list** is
swept (cartesian product over all sweepable leaves), scalars override once.

```bash
uv run ibkr bt sweep strat.json '{"strategy_params":{"position_size":[0.8,0.95],"drift_tolerance":[0.01,0.05]}}'
uv run ibkr bt sweep strat.json '{...grid...}' --sort-by sharpe_ratio --limit 5 --format json
```

Candle data loads once per distinct (symbol set, bar) and is window-sliced per
combo — no per-combo reload. Ranked by `--sort-by`; use `--limit` to show only
the top N.

Every report — `bt run|sweep|split|optimize`, text and `-F json` — draws from
one canonical metric set (Sharpe, Ann, MaxDD, Kurt, Skew, Win, Trades,
Scaled), so the columns and the JSON keys can never drift. The sweep table's
`params` column is one `k=v` per line, so a wide swept grid grows row height
instead of blowing the line width. `--format json` emits the same per-run
metric dict — including `kurtosis`, `skewness`, `scaled_trades` and
`win_rate` — under each result's `metrics`.

### `bt split` — in-sample vs out-of-sample validation

Evaluates a strategy's **fixed** parameter set across IS/OOS windows. Two modes:

- `--is-end <date>` — single anchor split: IS=`[trading_start, is_end]`, OOS=`[is_end+1d, trading_end]`.
- `--folds <n>` — expansion-window walk-forward: IS always starts at `trading_start` and grows, producing `n` non-empty folds.

```bash
uv run ibkr bt split strats/pass/<config>.json --folds 4
uv run ibkr bt split strats/pass/<config>.json --is-end 2020-12-31 --format json
```

Options: `--min-is-years` (walk-forward first-fold history floor, default 5.0),
`--train-start` (warmup override), `--format text|json`. Does **not** re-tune
params per fold — it answers _"given these locked params, how does performance
hold up out-of-sample?"_ The wide table pairs IS/OOS columns for every
canonical metric (incl. Kurt, Skew, Scaled); `-F json` emits the same per-run
metric dict per fold's `is`/`oos`.

Run folds in parallel across worker processes with `--workers N`. Each fold
(IS+OOS window pair) is independent; the shared candle + benchmark feeds pickle
once per worker, not per fold.

### `bt optimize` — walk-forward parameter optimization

Bridges `bt sweep` and `bt split`. Per fold: sweep the param grid **on the
fold's in-sample window**, pick the best combo (by `--sort-by`, default
`sharpe_ratio`), lock it, and run the **out-of-sample** window with those
params. OOS metrics are never optimized against.

```bash
uv run ibkr bt optimize strats/pass/<config>.json \
  '{"strategy_params":{"atr_mult":[1.5,2.0,2.5]}}' --folds 4
uv run ibkr bt optimize strat.json '{...grid...}' --is-end 2020-12-31 --format json
```

Param grid shape matches `bt sweep` (list-valued leaves swept, scalars
override once). **Honest about overfitting:** tuning per fold curve-fits the IS
window, and the OOS result prices that cost. If mean OOS Sharpe holds up across
folds, the edge is likely real; if IS is strong but OOS collapses, the grid is
fitting noise. Reports per-fold chosen params + IS/OOS metrics (same canonical
columns as `bt split`, incl. Scaled), and an aggregate of mean/min OOS Sharpe.
`-F json` emits the canonical per-run metric dict for both `is` and `oos`.

Run folds in parallel across worker processes with `--workers N` — each fold
tunes its own IS window and validates its OOS independently (combos inside a
fold stay sequential).

### Parallelism (`--workers`)

`bt sweep`, `bt split` and `bt optimize` all accept `--workers N` (default `1` =
sequential) to parallelize their independent units of work — grid combos
(sweep) or folds (split, optimize) — across a **process pool** (`multiprocessing`
/ `concurrent.futures`), not threads. Backtests are CPU-bound, and because
their engine is pure over immutable inputs, separate worker processes get
isolation for free. The shared candle feed is pickled once per worker, and
deterministic streaming order is preserved for `--format text`. Effective
workers are capped at the unit count, so `--workers` larger than the number of
combos/folds wastes nothing.

#### When to use `--workers`

`--workers` is a throughput knob, not a correctness one — every count returns
**identical** results. The question is only whether a process pool is *faster*.
There is a **fixed cost of ~2s per pooled run** (spawning a `forkserver` + worker
processes on Python 3.14), so the pool wins only when the units it parallelizes
are cheap by comparison. Roughly: if total compute is under a few seconds, keep
`--workers 1`; the pool starts paying off around many-combo sweeps and
many-fold splits over heavy feeds.

- **`bt sweep` — biggest win.** Parallelize the cartesian grid of combos. On a
  198-combo sweep this measured **≈3x faster with `--workers 8`** (41s → 14s).
  Reach for `--workers` whenever the grid has dozens of combos or the feed is
  large.
- **`bt split` — only with many folds / heavy feeds.** It parallelizes across
  folds, and each fold is just an IS+OOS pair. With a handful of folds on a
  small daily feed the ~2s spawn cost exceeds the compute, so `--workers 4`
  was **slower** (3s → 6s CLI wall-clock) than the default `1`. `--workers`
  helps here only for high `--folds` (e.g. 10+) or large intraday feeds.
- **`bt optimize` — same as split.** Fold-level parallelism; inside a fold the
  IS sweep stays sequential. Use `--workers` for many folds + a large grid.
  A single optimize is heavier than a split (grid × folds), so it benefits at
  lower fold counts than split.

**Rule of thumb:** default `--workers 1`; raise it when a single unrunnable run
would take more than a few seconds (many combos, many folds, or heavy
per-candle data). When in doubt, time `--workers 1` first — if it is already
fast (<~2s), more workers will only add overhead.

All commands usable via `make run <subcommand> <args>`.

## Toolchain

| Tool                        | Purpose            |
| --------------------------- | ------------------ |
| Python 3.14+                | Runtime            |
| `uv`                        | Package management |
| `ty`                        | Type checking      |
| `ruff format`               | Formatting         |
| `pytest` + `pytest-asyncio` | Testing            |
| `make`                      | Script runner      |

```bash
make check       # lint + format + typecheck + tests
make test        # all tests
make test-fast   # quick tests
```

## Key Types

```python
# StrategyConfig — loaded from JSON via load_strategy()
@dataclass
class StrategyConfig:
    name: str; strategy_type: str; symbols: list[str]
    initial_capital: float; commission: float
    training_start: str; training_end: str
    trading_start: str; trading_end: str
    bars: list[str]; strategy_params: dict
    rolling_window_size: int | None
    benchmark_symbols: list[str]

# BacktestResults — returned by run()
@dataclass(frozen=True)
class BacktestResults:
    pf: PortfolioResult         # all metrics + trades + equity curve
    data: dict                  # candles dict keyed by symbol
    final_state: BacktestState  # full state at end of backtest
    benchmark_curves: dict[str, pd.Series]
```

### Programmatic API

```python
from src.bt import load_strategy, Backtest, run, get_backtest_results_analysis
from src.bt.data_feed import load_candles
from src.bt.strategies import init_strat

config = load_strategy("strats/pass/momentum_compression_breakout_ae_gate_SPY.json")
bt = Backtest(config)
df = load_candles(config.symbols, bt.window.train_start, bt.window.test_end, config.bars[0])
results = run(bt, df, strat_mod=init_strat(config.strategy_type))
print(get_backtest_results_analysis(results.pf))
```

## IBKR REST API Client

The `ib-rest-api-client` package is a local dependency (`../ib-rest-api-client`), generated from `openapi.spec.json` (IBKR's OpenAPI spec).

### Setup

Run the Gateway login script before any data sync or API call:

```bash
export IBKR_USERNAME=...
export IBKR_PASSWORD=...
export TRADING_MODE=paper          # or "live"

uv run python scripts/login_ibkr.py
```

This logs into IBKR Client Portal Gateway via Playwright. The Gateway must be running (see `../client-portal/`).

### Regenerating the Client

After an API spec update:

```bash
cd ../ib-rest-api-client
uvx openapi-python-client generate --path ../py/openapi.spec.json --output-dir .
cd ../py && uv sync
```

## Resources

- **Strategy authoring & backtesting workflow**: [SKILL.md](SKILL.md) — read before any strategy work
- **Coding standards for contributors**: [AGENTS.md](AGENTS.md)
