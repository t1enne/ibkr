---
name: backtester
description: Backtest quantitative trading strategies independently — create JSON configs, write custom strategies, run backtests, and interpret results. Use when asked to "backtest a trading strategy", "test a pairs trade", "run a backtest", "create a strategy", or "evaluate a trading idea".
allowed-tools: Bash(uv:*), Bash(cd:*), Bash(find:*), Bash(cat:*), Bash(ls:*), Bash(grep:*), Read, Write, Edit
---

# IBKR Backtesting — Standalone Agent Skill

Full backtesting agent for the IBKR PY quantitative trading toolkit. Design, implement, and run strategy backtests end-to-end.

## Hard rules (read first)

1. **Never test a strategy.** Strategies, `strats/*.json`, research scripts and
   sweeps are exempt from `AGENTS.md`'s test rule. No `test_<strategy>.py`, no
   strategy fixtures. Tests cover _engine_ code only (indicators, metrics,
   sizing, risk, portfolio). Validate strategies by **running** them.
2. **`bt sweep` / `bt split` / `bt optimize` before any script.** Grid, IS/OOS
   and walk-forward questions all have a subcommand. A hand-rolled
   candle-load + grid-loop + `run()` harness is forbidden; that is `bt sweep`
   (`bt optimize` if folds must also validate OOS). Custom scripts only when
   the task exceeds this surface — say so first.
3. **Never filter or re-derive output.** No `grep`/`head`/`tail` over a report,
   no recomputing a metric in scratch code. Risk lives in the tail (kurtosis,
   skewness, stability, per-symbol draws, worst-DD). If you cut rows, name them.
4. **Don't add config fields or modules** to work around a missing capability —
   check the DSL surface and the `bt` subcommands first.

## Workflow

The **strategy development lifecycle** below is mandatory. Do not skip stages,
and do not grow scope before the current stage is signed off.

### Stage 0 — Minimum parameters

Use **as few params as possible**. Before adding any param, stop and ask:
_can the signal generation be rewritten so this param is unnecessary?_

A param is a fitted degree of freedom. Every one you add trades robustness for
`sweep` surface and invites curve-fit. Prefer:

- Rewriting the signal condition (structural change) over thresholding a new knob.
- Hard-coded structural constants (e.g. "close above 200d SMA") over tunable `**params`.
- Deriving a value from data (ATR-scaled, percentile-ranked) over an absolute number.

Hard-coding a structural constant is allowed and encouraged; adding a param to
absorb a bad signal is not. State the justification when you do add a param.

### Stage 1 — Develop on 1 symbol, ≤1 year

Develop and iterate with **`symbols` = one instrument** and a trading window of
**at most 1 year**. Fast iteration, and a small sample hides less.

When the run completes, **spawn a specialized subagent to review each trade (`quant` or `trader`, one
by one**, against the entry/exit logic. Ask for: does every entry match the
stated rule, does every exit fire for the stated reason, any lookahead or
off-by-one, any bar where the rule should have fired and did not. Trade-by-trade
review is the point of the small window — a 1-year single-symbol log fits.

Report the trades + the subagent's verdict; **do not proceed until the user
approves the entries and exits.**

### Stage 2 — Expand only after approval

On explicit user approval of the entries/exits, begin expanding. Grow **one
axis at a time, exponentially** — do not jump to the final universe/window:

1. **Universe:** 1 → 2 → 4 → 8 → … symbols. Re-check `bt run` after each step.
2. **Window:** 1y → 2y → 4y → full available history. Re-check after each step.

Exponential growth means the doubling step where the edge breaks is the answer
you were looking for — a metric that collapses at 4 symbols or 2 years is
evidence, not a setback. Use `bt split --folds` to separate IS from OOS as the
window grows; never tune on the expanded window without an OOS check.

Only after the expanded run survives do you move the config to its earned
bucket under `strats/<pass|wip|fail>/`.

### Commands per stage

1. **Understand the request** — what symbols, what kind of strategy, what timeframe?
2. **Pick or write a strategy module** — reuse an existing `STRATEGY_TYPE` if one fits, otherwise write a new one (DSL by default).
3. **Write the JSON config** — create `strats/<pass|wip|fail>/<name>.json`.
4. **Run the backtest** — `uv run ibkr bt run strats/<pass|wip|fail>/<name>.json`.
5. **Interpret and report** — summarize equity curve, metrics, drawdowns, trade log.

## Project Location

```bash
cd /home/nasrt/Documents/code/dev/ibkr/py
```

## Prerequisites

- Python 3.14+
- `uv` package manager
- Project dependencies: `uv sync`
- **IBKR REST API client** at `../ib-rest-api-client/` (local dependency). If missing, generate it:
  ```bash
  cd /home/nasrt/Documents/code/dev/ibkr/ib-rest-api-client
  uvx openapi-python-client generate --path ../py/openapi.spec.json --output-dir .
  cd ../py && uv sync
  ```
- **Gateway session** — login required before data sync or API calls:
  ```bash
  export IBKR_USERNAME=... IBKR_PASSWORD=... TRADING_MODE=paper
  uv run python scripts/login_ibkr.py
  ```
  Gateway must be running (see `../client-portal/`).

## Backtest a Strategy in 3 Steps

### Step 1: Pick or Write a Strategy Module

Strategies live in `src/bt/strategies/`. Authoring is **DSL-only** — decorate a
function of a `StrategyContext` with `@strategy(...)`; the module is
auto-discovered (any module exposing `STRATEGY_TYPE` is registered; nothing to
wire). The DSL owns candle iteration, cursor-safe indicator prefetch, and
signal construction, so you write _what_ to do, not _how_ data reaches you:

```python
# src/bt/strategies/my_strategy.py
from src.bt.strategies.dsl import strategy

STRATEGY_TYPE = "my_strategy"

@strategy(bars="1d")
def on_candle(ctx):
    fast = ctx.ta.ema("AAPL", 9)
    slow = ctx.ta.ema("AAPL", 21)
    if ctx.cross_over(fast, slow):
        ctx.long("AAPL", size=0.1, sl=0.04, tp=0.08, reason="ema cross up")
    elif ctx.cross_under(fast, slow):
        ctx.close("AAPL", reason="ema cross down")
```

Surface (all cursor-safe, no future bars):

- `ctx.ta` — prefetched indicators: `ema`, `sma`, `atr`, `rsi`, `adx`, `highest`,
  `lowest`, `sum`, `close`, `ohlcv`, `field` — e.g. `ctx.ta.rsi("AAPL", 14)` returns a `SeriesView`.
  Read the current bar with `view[-1]` / `.last()`; count back with negative indexing
  (`view[-2]` = one bar ago); never out of range — the view is cursor-truncated, so it
  structurally cannot leak a future bar.
- `ctx.long / ctx.short(sym, size=, sl=, tp=, size_mode="capital"|"equity")` — open a fresh lot,
  `size` = 0..1 fraction of the capital base. `sl`/`tp` are fractional percentages (0.04 = 4%).
- `ctx.close(sym)` — close all lots; `ctx.partial_close(sym, qty, lot=, tag=)` — shed a fraction of one.
- `ctx.ohlcv(sym)`, `ctx.price(sym)`, `ctx.position(sym)`, `ctx.quantity(sym)`, `ctx.avg_entry(sym)`.
- `ctx.cross_over / cross_under`, `ctx.change`, `ctx.barssince`, `ctx.nz` (Pine built-ins).
- `ctx.shared` — cross-call state, only with `@strategy(stateful=True)`; a fresh dict per run/window
  (never module globals; clear between `split`/`sweep` windows by construction).
- `ctx.state` — raw `BacktestState` for power needs (portfolio/candles lookup is not forbidden,
  just bypassed — prefer the shortcut methods).

Cross-timeframe reads (base bar vs HTF) go through `ctx.ta.<indicator>(sym, interval=...)`
or `state.candles.get((sym, interval))` — both cursor-truncated.

Register a strategy by dropping the file in `src/bt/strategies/` — the import
path is `src.bt.strategies.<module_name>`. Optional typed `Params` dataclass
(subclass of `StrategyParams`, see `types.py`) and the engine instantiates it
from `strategy_params` instead of a dict.

### Step 2: Write the JSON Config

Create `strats/<pass|wip|fail>/<name>.json` (configs are dropped in the
classification bucket they earn — see `strats/README.md`):

```json
{
  "name": "strategy-name",
  "training_start": "2024-01-01",
  "training_end": "2024-01-02",
  "trading_start": "2024-01-02",
  "trading_end": "2025-01-01",
  "commission": 0.1,
  "initial_capital": 10000,
  "strategy_type": "my_strategy", // must match a discovered STRATEGY_TYPE
  "bars": ["1h", "4h"],
  "strategy_params": {
    "fast": 9,
    "slow": 30
  },
  "symbols": ["COIN", "AAPL"]
}
```

**Config field reference:**

- `training_start`/`training_end` — warmup window (data loaded for indicator/model warmup before trading)
- `trading_start`/`trading_end` — actual backtest window
- `bars` — list of bar sizes; `bars[0]` is the base signal interval, additional
  entries (e.g. `["1h", "4h"]`) are higher-timeframe bars injected alongside
  (lookahead-safe). There is no separate `htf` field.
- `commission` — fixed commission per trade
- `strategy_params` — arbitrary dict forwarded verbatim to `on_candle(state, candle, params)`;
  position sizing, stop-loss/take-profit, and any model hyper-params live here
  (strategy-owned, per-trade). There is **no** top-level `model_updater`/`model_params` —
  cross-candle model state is owned by the strategy itself

**Available tickers** — see `/home/nasrt/Documents/code/dev/ibkr/py/universes/*.json` for symbols with local data.

### Step 3: Run the Backtest

```bash
cd /home/nasrt/Documents/code/dev/ibkr/py
uv run ibkr bt run strats/<bucket>/<name>.json --trades   # full trade list (off by default)
make run bt run strats/<bucket>/<name>.json   # same, via Make shortcut
```

> **Note:** Make intercepts its own flags (`--help`, `--format`). To pass them through to `ibkr`, use `-- ` separator: `make run bt run strat.json -- --format jsonl`.

**For programmatic use** (if you need structured output):

```python
import asyncio
from src.bt import load_strategy, Backtest, run
from src.bt.data_feed import load_candles
from src.bt.strategies import init_strat

config = load_strategy("strats/wip/<config>.json")
bt = Backtest(config)
df = load_candles(config.symbols, bt.window.train_start, bt.window.test_end, config.bars[0])
strat_mod = init_strat(config.strategy_type)
results = run(bt, df, strat_mod=strat_mod)
# results.pf → PortfolioResult (total_return, sharpe_ratio, trades, equity_curve, ...)
# results.data → dict[str, pd.DataFrame] of candles
# results.final_state → BacktestState
```

## Interpreting Results

The `bt run` report contains:

- **Drawdown periods** — worst 5 drawdowns with dates and duration
- **Trade log** — every trade with entry/exit times, prices, PnL, direction, reason, SL/TP levels. **Omitted by default** (big runs can hold thousands): pass `--trades` to print/serialize the full list. Off, every format still reports the count — text keeps the `Trades` header as a one-line note, `json`/`jsonl`/`plot` carry a top-level `total_trades`.
- **Statistics** — win rate, total trades, starting capital, total P&L, backtest duration
- **Metrics table** — annual return, volatility, Sharpe, Calmar, Sortino, Omega, max drawdown, stability, skewness, kurtosis, alpha, beta, plus **Scaled Trades** (fills dropped for cash exhaustion). `sweep`/`split`/`optimize` reports use the shared canonical set (Sharpe, Ann, MaxDD, Kurt, Skew, Win, Trades, Scaled) and their `-F json` emits the identical per-run metric dict.

**Read the whole report before reporting.** A healthy Sharpe can hide fat tails
(kurtosis/skewness), regime dependence (stability), or one symbol's bleed. Never
summarize from the headline number; never trim except explicitly.

**Scope discipline:** a report from a 1-symbol / ≤1-year run is an _entry/exit
review artifact_, not evidence of an edge. Do not present it as a strategy
verdict, and do not expand scope without the Stage 2 approval gate above.

| Question                              | Command                                   |
| ------------------------------------- | ----------------------------------------- |
| Does this config work?                | `bt run <config>`                         |
| Best params over the whole window?    | `bt sweep <config> '{grid}'`              |
| Are locked params curve-fit?          | `bt split <config> --folds N`             |
| Tune per fold, validate OOS honestly? | `bt optimize <config> '{grid}' --folds N` |

Sweep = search. Split = sanity check. Optimize = both, chained honestly.

## Design Heuristics

When building or modifying strategies, follow these patterns from the codebase.
The parameter-minimization rule from Stage 0 is a hard rule, not a heuristic.

### Keep `on_candle()` simple

One function, no classes. Read state, compute indicators, return signals. The engine handles the rest.

### Volume filtering

Volume confirmation is a common guard. Cross-check volume against price action
before emitting an entry signal.

### Ranging/squeeze detection

Use EMA convergence + ATR contraction to detect a squeeze before a guarded
breakout entry.

### Statefulness within strategy

Write strategies that need cross-call state on the **stateful DSL**
(`@strategy(stateful=True)`): hold it in `ctx.shared`, a fresh dict the engine
mints **per run/window**, so cross-window bleed is impossible and strategies are
safe to run concurrently (including across `sweep`/`split`/`optimize` worker
processes). Model objects (e.g. `OnlinePairs`, `OnlineRegime`) live in
`ctx.shared` and are fed per candle; there is no engine `model_updater`/
`ModelState` channel.

### Always handle "no position" and "in position" paths

```python
position = state.portfolio.positions.get(symbol)
if not position:
    # entry logic
else:
    # exit logic
```

### Multi-timeframe reads

Prefer `state.candles.get((sym, freq))` (or DSL `ctx.ta` with an explicit
`interval=`) for HTF bars — both are cursor-truncated and lookahead-safe.
There is no `state.model_state` channel; cross-candle model state is owned by
the strategy (`ctx.shared`).

## Testing

**Never write tests for strategies** — even ones with nontrivial computation.
Test only **engine** code (indicators, metrics, sizing, risk, portfolio).
Strategy validation = run the backtest and read the full report.

```bash
make test                                     # all tests
make test-fast                                # quick tests (no header)
uv run pytest src/bt/engine/tests/ -v
uv run pytest src/bt/portfolio/tests/ -v
uv run pytest src/bt/risk/tests/ -v
```

## Common Gotchas

- **Don't invent a workflow.** Check for an existing `bt` subcommand first.
- **Don't launder output.** A re-derived metric or `grep`-ed report is not evidence.
- **Data availability**: when an agent needs candles that are missing/stale, just run `data dl` (see the runbook below) — `data query` only _reads_ the local DB and never fetches.
- **Two sources, one command.** `data dl` fetches candles via the IBKR Gateway **and** company fundamentals from SEC EDGAR (no Gateway, disk-cached) — one symbol list, both passes, always. US macro = `scripts/fetch_macro_fred.py` (`FRED_API_KEY` is configured, writes `assets/*.csv`) is separate and not fixed by `data dl`.
- **Bar size**: strategies expect the bar size in config to match available data. Most data is `1h`.
- **HTF lookahead**: `state.candles.get((sym, freq))` and the DSL `ctx.ta`
  `interval=` reads are both safe (cursor-truncated).
- **Multiple symbols**: the engine iterates all symbols per timestamp. `on_candle` fires only on the last symbol per timestamp (so `state.candles` has all symbols' data). Signals for any symbol are valid — engine routes fills by `signal.symbol`. Pending signals for non-current symbols fill when that symbol's own `_execute_pending` stage runs (same bar cycle, later in the timestamp iteration).
- **Model-backed strategies** (e.g. a Kalman pairs filter): the model is
  strategy-owned — an `OnlinePairs`-style filter object lives in `ctx.shared`
  (stateful DSL), fed per candle from `state.candles`. No engine `model_updater`
  is involved.

## `data dl` — one-shot candle + fundamentals download (when data is missing/stale)

When a backtest has no data or a symbol's daily read looks stale, just attempt:

```bash
uv run ibkr data dl AAPL MSFT --from 2019-01-01        # backfill + refresh
uv run ibkr data dl --universe universes/nsdq.json --from 2019-01-01
```

One command, two independent sources: candles from the IBKR Gateway (bounded
by `--from`/`--to`) **and** SEC EDGAR fundamentals (direct SEC HTTP, no
Gateway). The symbol list is resolved once and feeds both passes; the SEC pass
always runs.

Idempotent and gap-based; `--from` earlier = deeper history. Don't loop chunks.
Its `0 fetch gaps`/`up to date` tail can print even on success, so confirm the
file actually grew:

```bash
uv run ibkr data query AAPL        # max date should advance
```

Needs an authenticated Gateway at `https://localhost:5000/v1/api`:

```bash
uv run python -c "import httpx;print(httpx.get('https://localhost:5000/v1/api/iserver/auth/status',verify=False,timeout=8).json().get('authenticated'))"  # True = ready
uv run python scripts/login_ibkr.py   # else login
```

## SEC EDGAR fundamentals (part of `data dl`, no Gateway needed)

The same command stores SEC EDGAR (XBRL) filings in the local DB as **sparse
fiscal rows** — one row per `(ticker, statement, field, period)` filing (~5
rows per symbol per year, not a daily grid), plus a `fundamentals:` recap block:

```bash
uv run ibkr data dl AAPL MSFT --from 2019-01-01        # candles + fundamentals
uv run ibkr data dl AAPL --from 2019-01-01 --refresh-fundamentals   # ignore SEC cache
```

Fundamentals options (they only affect the SEC pass):

- `--fundamentals-from`, `--fundamentals-to` — bound the **filing** window (the
  `filed` date, not the fiscal period: a 10-K filed in 2024 restates 2022's
  period, and excluding it by period would drop exactly the point-in-time facts
  we keep). Unset by default — the candle `--from` is deliberately **not**
  imposed, so as-first-stated history keeps its earliest filings.
- `--refresh-fundamentals` — bypass the on-disk payload cache and re-fetch.
- `--fundamentals-cache` — payload cache dir (default `../data/fundamentals_cache`).

Idempotent. The per-symbol recap prints rows written (or `up to date` when `0`),
fiscal periods landed, and the filed span — re-run after new filings appear.
Symbols SEC has no CIK for (ETF, non-US registrant) report `0 rows` without
aborting the batch.

**Reading fundamentals in a strategy** — no CLI read path; use the cursor-safe
`ctx.fundamentals` surface inside `on_candle`:

```python
series = ctx.fundamentals.income("AAPL").net_income   # fiscal-period SeriesPIT, not a bar series
latest = series[-1]       # most recent period whose filing the strategy has already seen
                          # as-first-stated: a restatement never rewrites a prior period,
                          # rows filed after the cursor are invisible (no lookahead)
snap = ctx.fundamentals.latest("AAPL", "income")  # newest-filed (restated) statement instead
```

Reference implementation: `vwatr_div_dsl.py` `risk_scale="earn"` — same-quarter
YoY earnings growth (`series = ctx.fundamentals.income(sym).net_income`; match
the period ~365 days earlier, 330–400-day window) scaling position size,
clamped to [0.5, 1.5].

## FRED macro data & macro indicators (optional strategy data)

US macro series live as two-column `assets/<name>.csv` files (`date`, `<name>`;
**blank cells = FRED no-print dates**) consumed by `src.indicators.macro`.

**Fetch / refresh:**

```bash
export FRED_API_KEY=...     # free key from fred.stlouisfed.org
uv run python scripts/fetch_macro_fred.py               # all defaults
uv run python scripts/fetch_macro_fred.py --out assets --series GDPC1 INDPRO
```

Default set: `gdpc1`/`gdp` (GDP), `payems`/`unrate` (employment), `indpro`/`tcu`
(production), `bopgstb` (trade balance), `dgs2`/`t10y2y` (yields), `hyspread`
(HY OAS), `vix` (VIXCLS). `cpi` is World Bank data (annual inflation chained
into a 1.0-based daily level — **not** from FRED, different loader). All series
are forward-filled onto a daily grid on load, so every value is the latest
known print.

**Read in a strategy — lookahead-free factory:**

```python
from src.indicators.macro import init_macro_indicator

vix = init_macro_indicator("vix")     # loads assets/vix.csv once at init
level = vix(ts)                       # latest VIX print with release date <= ts
```

`init_macro_indicator("<name>")` returns `f(ts) -> float | None` — a single
`Series.asof`, structurally unable to leak a future observation. Out-of-span or
missing asset → `None` (treat as neutral). Bind **once** per run.

Hot path (percentile / rolling reads): cache the loaded arrays in `ctx.shared`
and index with `searchsorted` — the pattern behind `vwatr_div_dsl.py`
`risk_scale="fred"` (VIX percentile vs prior 250 prints, scaled to [0.5, 1.5]):

```python
from src.indicators.macro._shared import load_daily

@strategy(bars="1d", stateful=True)
def on_candle(ctx):
    cache = ctx.shared.setdefault("macros", {})
    if "vix" not in cache:
        s = load_daily("vix")
        cache["vix"] = (
            np.asarray(s.index.values, dtype="datetime64[ns]"),
            s.to_numpy(dtype=float),
        )
    dates, values = cache["vix"]
    j = int(np.searchsorted(dates, np.datetime64(ctx.candle.timestamp))) - 1
    pct = float(np.mean(values[max(0, j - 250) : j] < values[j]))  # prior prints only
```

`deflated_log_prices(nominal, cpi)` is a separate pure vectorised helper
(`ln(nominal) - ln(cpi)` real prices), not part of the cursor-indicator API.

## Module Reference

All CLI groups under the `py` root command — also callable via `make run <subcommand> <args>`:

| Group  | Commands                                      | Description                                                                  |
| ------ | --------------------------------------------- | ---------------------------------------------------------------------------- |
| `data` | `dl`, `query`, `preview`                    | Sync/download OHLCV from IBKR **and** SEC EDGAR fundamentals (one `dl`), query local DB |
| `bt`   | `run`, `sweep`, `split`, `optimize`, `screen` | Backtesting engine, hyperparam sweep, IS/OOS validation, walk-forward tuning, live-intent screening |
| `live` | `run`, `abandon`                              | One-shot reconcile cycle against the IBKR account (paper or live); see README § Live Trading |

### `bt sweep` — hyperparameter sweep

Sweeps a `PARAM_GRID` over a strategy and ranks every combo by a chosen metric
(`--sort-by`, default `annual_return`). Any list-valued leaf in the grid is
swept (cartesian); scalars override once. Shown `--limit` top N if given.

```bash
uv run ibkr bt sweep strat.json '{"strategy_params":{"position_size":[0.8,0.95]}}'
uv run ibkr bt sweep strat.json '{...grid...}' --sort-by sharpe_ratio --limit 5 --format json
```

Candle data loads once per distinct (symbol set, bar) and is window-sliced per
combo — no per-combo reload. Every report (text and `-F json`) uses one
canonical metric set — Sharpe, Ann, MaxDD, Kurt, Skew, Win, Trades, Scaled —
so columns and JSON keys never drift; the `params` column is one `k=v` per
line, so a wide grid grows row height rather than line width. `-F json` emits
the same per-run metric dict (incl. `kurtosis`, `skewness`, `scaled_trades`)
under each result's `metrics`.

### `bt split` — IS/OOS walk-forward validation

Evaluates a strategy's **fixed** params across in-sample/out-of-sample windows
(does **not** re-tune per fold). Two modes:

- `--is-end <date>` — single anchor split: IS=`[trading_start, is_end]`.
- `--folds <n>` — expansion-window walk-forward (IS grows from `trading_start`).

```bash
uv run ibkr bt split strats/<bucket>/<config>.json --folds 4
uv run ibkr bt split strats/<bucket>/<config>.json --is-end 2020-12-31 --format json
```

Reports per-fold IS/OOS cells for every canonical metric (Sharpe, Ann, MaxDD,
Kurt, Skew, Win, Trades, Scaled), plus a summary of mean/min OOS Sharpe and
OOS→IS degradation. `-F json` emits the same per-run metric dict per fold's
`is`/`oos`. Useful to check whether a strategy's edge survives out-of-sample
rather than being curve-fit to the training window.

Run folds in parallel with `--workers N` — each IS/OOS window pair is an
independent unit of work.

### `bt optimize` — per-fold IS tune → OOS validate

Bridges `bt sweep` (tune params, whole window) and `bt split` (locked params).
Per fold it sweeps a param grid **on the in-sample window**, locks the best
combo, and validates it on the out-of-sample window.

```bash
uv run ibkr bt optimize strats/<bucket>/<config>.json \
  '{"strategy_params":{"atr_mult":[1.5,2.0,2.5]}}' --folds 4
uv run ibkr bt optimize strats/<bucket>/<config>.json \
  '{"strategy_params":{"ma_slow":[50,100,200]}}' --is-end 2020-12-31 --format json
```

Honest about overfitting: per-fold tuning curve-fits the IS window, and the OOS
result prices that cost. If mean OOS Sharpe holds up across folds the edge is
likely real; if IS is strong but OOS collapses, the grid is fitting noise.
Reports perf-fold chosen params + IS/OOS canonical metrics (same columns as
`bt split`, incl. Scaled), plus mean/min OOS Sharpe.

**Neither `sweep`+`split` nor `optimize` is a clean test — both leak.**

- `sweep`+`split` leaks _selection into OOS_: sweep picks params on the whole
  window, then split cuts folds out of data those params already saw.
- `optimize` leaks _IS selection into OOS_: OOS is honest w.r.t. the chosen
  params, but degenerate IS optima (knife-edge param, overfit tail) carry
  forward. Low fold count makes one bad pick poison that fold's OOS.

Which is worse depends on split width and param count. Never quote either
number alone. Read the OOS **distribution**, not the mean:

1. Inspect perfold chosen params. High fold-to-fold param variance = IS
   selection is noise; the OOS number is meaningless regardless of OOS
   discipline. This is the tell.
2. Report OOS min and spread across folds, not just the mean.
3. For a genuinely clean test, nest: tune on IS, select on a middle segment,
   OOS untouched until the end.

`split` measures stability of a _fixed_ config. `optimize` measures a _search_.
Testing a search → `optimize`, with the param-variance check in step 1.

**A run with rejected or scaled fills is not a solid result.** A cash
shortfall SCALES a multi-open cohort (silent — entries land at `scale ×
plan`), only a lone open rejects. Risk-sized entries then realize less $
risk than authored. Cohort membership is order-sensitive (clock, data gaps,
Stage 4 vs 6), so WHICH entries fill full / scaled / never depends on symbol
list composition. Each report carries a `Scaled` metric — the count of fills
the engine dropped for genuine cash exhaustion — so a non-zero value is
visible next to Sharpe/Kurt instead of buried in a stderr warning.
`[bt] WARNING: N fill(s) rejected` or chronic over-subscription = metrics not
comparable across symbol permutations or config edits. Fix sizing before
quoting numbers.

Run folds in parallel with `--workers N` — each fold tunes its IS and
validates its OOS independently (combos inside a fold stay sequential).

### `bt screen` — current-bar intent from a real strategy

Runs a strategy's own `on_candle` through the real engine over a trailing
warm-up window ending at the newest data bar, then surfaces **intent**, not
fills: each symbol's latest emitted posture as an action + score. Actions are
`long`/`short` (open or reorient), `close` (explicit exit) and `flat` (no
signal — text only). Ranked by score desc.

The intent table is an order ticket, not a metric sheet: signal-time price,
qty, stop-loss, take-profit, and the bar it fired on. Output is an intent rank
only — a high score means "the condition fired", never "expected profit".

```bash
uv run ibkr bt screen strats/pass/<config>.json            # ranked intent table
uv run ibkr bt screen strats/pass/<config>.json -F json    # live-consumer payload
uv run ibkr bt screen strats/pass/<config>.json --trades   # + executed-trade table
```

`-F json` emits only actionable (`long`/`short`/`close`) rows under `signals`
(with `position_id`/`tag`), so a live layer can act without replaying the
engine; a no-signal `flat` row is never emitted as an instruction to flatten.

`--trades` (default **off**) additionally surfaces the run's executed trades,
from the engine's real fills (open → close). Text mode prints a second table
after the intent table (symbol, position, qty, entry/exit time and price, pnl,
close_reason; blank for an unclosed trade). JSON mode adds a `trades` key — one
dict per trade (`trade_json`): the full fill record incl. `stop_loss`,
`take_profit`, `commission`, `slippage`, `status`, `close_reason`, `reason`.
The run flattens its book at the final bar, so the last fills are `_finalize`,
**not** strategy intent — read them as the engine's realized book, not as
signals to act on. Off by default so payloads and byte-for-byte output are
unchanged for existing intent consumers.

### Live consumption (`ibkr live run`)

A live cycle loads the **same strategy JSON** and runs the strategy's own
`on_candle` through the screen bridge for the current bar, then reconciles that
resulting posture against the live book. There is no backtest replay: the
strategy is the only source of intent, and everything downstream (sizing, cash
boundedness, order identity) belongs to `src/live`.

When authoring, that means the live edge acts on **posture only** — `long`,
`short`, `flat`, plus an explicit `qty` when the strategy sizes itself. It places
market DAY orders, and an intent carrying stop-loss or take-profit is **refused**
rather than sent naked, so a stop-dependent strategy is not live-ready until that
changes. `scope` (defaulting to the strategy name) is the ownership key for the
live book and the order-id prefix — two configs sharing a name share one book,
its cash and its fills.

### Parallelism (`--workers`)

`bt sweep`, `bt split` and `bt optimize` each accept `--workers N` (default `1` =
sequential) to parallelize their independent units of work — grid combos
(sweep) or folds (split, optimize) — over a **process pool** via
`concurrent.futures.ProcessPoolExecutor`, not threads (`src/bt/parallel.py`).
The engine is pure over immutable inputs, so separate worker processes get
isolation for free; the shared candle feed pickles once per worker, and
streaming output order is preserved.

**When to use `--workers`** (every count returns identical results; it is a
throughput knob, not a correctness one):

- There is a **fixed ~2s spawn cost** per pooled run (forkserver + workers on
  Py 3.14). The pool only helps when the units it parallelizes are expensive
  by comparison — so keep `--workers 1` for trivial runs (a few combos/folds on
  a small daily feed), where the spawn cost exceeds any savings.
- **`bt sweep`: biggest win** — parallelizes the cartesian grid. A 198-combo
  sweep measured **≈3x faster with `--workers 8`** (41s → 14s). Use it whenever
  the grid has dozens of combos or a large feed.
- **`bt split`: only with many folds / heavy feeds** — each fold is just an
  IS+OOS pair, and the spawn overhead dwarfs a few folds (a 4-fold daily split
  was _slower_ pooled: 3s → 6s). Reach for `--workers` at `--folds 10+` or on
  large intraday feeds.
- **`bt optimize`: same as split** — fold-level parallelism (the in-fold IS
  sweep stays sequential). It is heavier than split (grid × folds), so it
  pays off at lower fold counts.

Effective workers are capped at the unit count, so `--workers` much larger
than the number of combos/folds wastes nothing. Rule of thumb: raise `--workers`
only once a single run would take more than a few seconds.
