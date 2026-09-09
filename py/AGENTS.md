# AGENTS.md — IBKR PY

> AI coding agent instructions for this repository.
> Read this before generating any code.

## Alpha research

Read SKILL.md

## Development Workflow

When implementing a feature or fix:

1. **Understand** — read relevant strategy code, types, and tests. Don't guess.
2. **Plan** — state approach before writing. If unclear, ask.
3. **Implement** — minimum code that works. Pure functions, immutable state, full type annotations.
4. **Test** — every new computation gets a test. Strategies, scripts and research doesn't need a test. Run `make test` before declaring done.
5. **Verify** — `make check` must pass (lint + format + typecheck + tests).

### Running things

**Read full `bt run` / sweep output — never filter it through `grep`/`head` when
reporting.** Metrics that live in the tail/body (kurtosis, skewness, stability,
per-symbol draws, worst-DD periods) carry the actual risk story; a summary
line like Sharpe can look healthy while kurtosis or a single symbol's bleed
tells the real tale. Only trim the output when you are very sure the removed
rows add no signal, and say when you cut it.

```bash
uv run ibkr bt run strats/trend.json       # CLI entry point
uv run ibkr data query SPY                 # Query SPY data from the local DB
make run bt run strats/trend.json          # Make shortcut
make check                                 # lint + format + typecheck + test
make test                                  # all tests
make test-fast                             # quick tests

# Sweep/split/optimize parallelize their independent units (grid combos or
# folds) over a process pool. Add --workers N (>1) to speed up CPU-bound runs:
uv run ibkr bt sweep strats/trend.json '{...grid...}' --workers 8
uv run ibkr bt split strats/trend.json --folds 4 --workers 4
uv run ibkr bt optimize strats/trend.json '{...grid...}' --folds 4 --workers 4
# WHEN to use --workers: it is a throughput knob, never a correctness one (all
# counts return identical results). There is a ~2s fixed spawn cost per pooled
# run, so keep --workers 1 for trivial workloads (a few combos/folds on a small
# feed) — the spawn overhead then exceeds the savings. Prefer --workers when a
# run would take more than a few seconds: many combos (sweep benefits most;
# a ~200-combo sweep measured ~3x faster at --workers 8), many folds (split /
# optimize, esp. --folds 10+), or heavy intraday feeds. Effective workers are
# capped at the unit count, so oversized --workers is harmless.
```

## Running Screens (no CLI — run via Python)

Screens are scoring layers that return 0..1 `ScreenResult`
(`score` + `action` long/short/flat + reasons + `model_features`), NEVER fills.
No `ibkr screen` CLI.

- **Code:** `src/bt/screen/` (`types.py`, `runner.py`, `adapter.py`, `screens/*.py`).
- **Discovery:** any `screens/*.py` with `SCREEN_TYPE` + `on_state(state, params)` — auto-`_discover()`ed, no registry.
- **The 6:** `momentum`, `macd_divergence`, `mfi_divergence`, `obv_divergence`, `rsi_divergence` (fresh signals) + `rs` (relative strength — cross-sectional ranking; fires by construction, so DON'T count it as corroboration).
- **Benchmarks:** `rs` needs its benchmark in-state (raises otherwise); `momentum` gates on `QQQ`. Pass `benchmarks=['QQQ','SPY']`.

### Run all screens over a universe (ready-made)

```bash
uv run python scripts/run_screens.py   # nsdq + QQQ/SPY, latest 1d bar, convergence view
```

Loads `universes/nsdq.json`, runs every discovered screen, separates
**absolute-screen convergence** (2+ fresh-signal screens, same direction) from
the `rs` overlay. Extend this script before writing a new one.

### Wire-up

```python
from src.bt.screen.screens import init_screen, resolve_screen_params
from src.bt.screen.adapter import state_per_interval   # or state_from_feed

daily = state_per_interval(symbols, start, end, ["1d","4h"], benchmarks=["QQQ","SPY"])["1d"]
results = init_screen("mfi_divergence").on_state(daily, resolve_screen_params("mfi_divergence", {}))
```

For a cursor-safe walk across history use `screen_over_history(...)`.



~All ~300 tickers plus ~1.8M hourly candles already live in the local candle DB.
Before assuming data is missing, check it here first — most "is data present?"
questions are answered by one query.

### The database

- **Path:** SQLite at `../data/db.sqlite` — **relative to the repo's parent**
  (`/home/nasrt/Documents/code/dev/ibkr/data/db.sqlite`), NOT inside this `py/`
  dir. ~166 MB, `journal_mode=wal`.
  - `src/data/db.py::_DEFAULT_DB_PATH` resolves it file-relative
  (correct — use this).
  - `src/data/types.py::db_path` resolves it via
  `os.getcwd()/../data/...`, so it depends on the CWD being the repo's parent.
  Prefer `query_candles`/`get_connection` from `src.data.db` over the peewee
  instance.
- `sqlite3` CLI may not be installed. Query with Python instead:
  `python -c "import sqlite3; c=sqlite3.connect('../data/db.sqlite')"`, or use
  the CLI below.

### Schema

- **`symbol`**: `conid` (PK), `ticker`, `name`, `market`, `currency`. ~309 rows.
- **`candle`**: `id` (PK autoincrement), `ticker`, `conid`, `timestamp` (ms epoch),
  `open`, `high`, `low`, `close`, `volume`. Indexed on `(ticker, timestamp)` —
  always filter by `ticker`, never join through `symbol.conid` (that join is
  unindexed and ~20x slower). `ticker` is stored UPPERCASE.
- Two migration tables (`kysely_migration`) — schema versioning, ignore.

### Quick verification

```bash
uv run ibkr data query SPY                    # recap: date range, rows, gaps>48h
uv run ibkr data query AAPL --bar 1d          # resampled agg
uv run ibkr data query --universe universes/nsdq.json   # whole universe recap
# Note: --universe/-U takes a FILE PATH (e.g. universes/nsdq.json), not a bare
# universe name. `ibkr data query/dl/preview` all go through
# load_universe_config() which opens the string as a path verbatim.
# Raw CLI: `ibkr data` subcommands are dl / preview / query (see src/data/cli.py)
```

### What's actually in there (snapshot as of 2026-08-15)

- **201 symbols with candle data**, ~**1,822,306** rows total.
- **Native granularity is 1h** for every symbol (median gap = 3600000 ms). Other
  bars (1d, 4h, …) are **resampled on read** from 1h by `src.data/resample.py`,
  never stored separately.
- **Deepest history**: SPY from **2004-01-23** (40,882 rows); many core tickers
  (AAPL, MSFT, NVDA, GOOGL, etc.) go back to **2019-11** (~11.8k rows, ~12k for
  META). Long-history tickers (~20k rows, back to **2014**) include SPY, QQQ,
  GDX, GL.D, SLV, UNG, USO, UUP, XLE/XLF/XLK/XLU/XLV/XLY/XLB, SHV, DBA, REET.
- **Full-core group** (~11.77k rows, 2019-11-15 → 2026-08-07): AAPL, ADBE, ADI,
  AMAT, AMD, AMGN, AMZN, AVGO, BKNG, BKR, CCEP, CDNS, CMCSA, COST, CRWD, CSCO,
  CSX, CTAS, DDOG, DXCM, EXC, FANG, FAST, FTNT, GEHC-partial, GILD, GOOG/GOOGL,
  HON, IDXX, INTC, INTU, ISRG, KDP, KHC, KLAC, LIN, LITE, LRCX, MAR, MCHP,
  MDLZ, MELI, MNST, MPWR, MRVL, MSFT, MSTR, MU, NFLX, NVDA, NXPI, ODFL, ORLY,
  PANW, PAYX, PCAR, PDD, PEP, PYPL, QCOM, REGN, ROP, ROST, SBUX, SHOP, SNPS,
  STX, TER, TMUS, TRI, TSLA, TTWO, TXN, VRTX, WBD, WDAY, WDC, WMT, XEL.
- **Early-window group** since their IPO / later synced (fewer rows than the
  core): ABNB (2020-12), APP (2021-04), ARM (2023-09), BTC (2024-07), CEG
  (2022-01), COIN (2023-11), DASH (2020-12), PLTR (2020-09), RKLB (2020-11),
  GTLB (2021-12), ALAB (2024-03), NBIS (2024-10), CRWV (2025-03), SNDK
  (2025-02), HONA (2026-06), SPCX (2026-06), FER (2024-05).
- **Short-2000-row group (~2024-12-26 → 2026-02-20)**, mostly ETFs/bonds recently
  synced with a 2000-row cap: AGG, BIL, BND, BNDX, BSV, DFAC, DGRO, EFA, IBIT,
  IEMG, IJH, ITOT, IVE, IVV, IVW, IWB, IWD, IWF, IWR, IXUS, JEPI, IWM (11k),
  MBB, MUB, QQQM, RSP, SCHD, SCHF, SCHG, SCHX, SGOV, SMH (2742), SPDW, SPYG,
  SPYM, VB, VCSH, VEA, VEU, VGIT, VGT, VIG, VNQ-err VNQ full, VO, VOO, VTI,
  VTV, VUG, VV, VXUS, VYM. Several end earlier (2026-02-20 vs the 08-07 core) —
  these are the most likely to need a refresh via `ibkr data dl`.

Don't re-derive this list from the DB per task; trust the snapshot above. If you
need a fresh one: `SELECT ticker, COUNT(*), MIN(timestamp), MAX(timestamp) FROM
candle GROUP BY ticker ORDER BY ticker`.

## Language & Toolchain

- **Python 3.14+** (required)
- **Package manager:** `uv` (not pip)
- **Type checker:** `ty`
- **Formatter:** `ruff format`
- **Test runner:** `pytest` with `pytest-asyncio`
- **Script Runner:** `make`
- **CLI binary:** `ibkr` (installed via `uv sync` from `pyproject.toml` entry point)

## Core Principles

### 1. Type Safety (NON-NEGOTIABLE)

Every function, method, and dataclass MUST have complete type annotations.
No `Any` unless truly unavoidable — and even then, comment why.

```python
# ✅ Good — fully typed
def calculate_zscore(
    prices_a: pd.Series,
    prices_b: pd.Series,
    window: int = 75,
) -> pd.Series: ...

# ❌ Bad — missing types
def calculate_zscore(prices_a, prices_b, window=75): ...

# ❌ Bad — Any escape hatch
def process(data: Any) -> Any: ...
```

#### Rules

- **Use `Protocol` for dependency injection** — the codebase uses `ExecutionFn`, `RiskCheckFn`, `PositionSizerFn`, `DataLoaderFn` for engine-handler seams. Follow this pattern. Never pass raw `Callable` when a Protocol exists or should exist. (Strategy *authoring* has its own seam: a ``@strategy``-produced callable of ``StrategyContext`` — not a raw ``StrategyFn`` Protocol; see §4.)
- **Use `@dataclass(frozen=True)` for state.** Immutable state makes backtesting deterministic and testable. See `Tick`, `PortfolioState`, `BacktestState`, `FillEvent`, etc.
- **Use `Literal` for enums of strings.** Prefer `Literal["long", "short", "close"]` over bare `str`.
- **Use `TypedDict`** for structured dicts when a dataclass would be overkill.
- **No `object` or bare `dict`** as parameter/return types.
- **`TYPE_CHECKING` guard** for import-only types to avoid circular imports.
- **Top level imports only** No lazy imports.

```python
# ✅ Pattern: TYPE_CHECKING for type-only imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.bt.types import PlotConfig
```

### 2. Performance

This is a quantitative finance backtesting system — hot paths matter.

```python
# ✅ Vectorized — entire Series at once
z = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()

# ❌ Loop over rows
z = []
for i in range(len(spread)):
    window = spread[i-window_size:i]
    z.append((spread[i] - window.mean()) / window.std())
```

#### Rules

- **Vectorize with pandas/numpy.** A loop over ticks is acceptable in the backtest engine's main loop (necessary for event sequencing). A loop over data points inside a computation is not.
- **Avoid `pd.DataFrame.append` / `pd.concat` in loops.** See `_append_candle` — this is a known hotspot. Pre-allocate or batch.
- **Use `tuple` not `list`** for immutable sequences (see `BacktestState.risk_events: Tuple[Any, ...]`).
- **Dataclass `field(default_factory=list)`** is fine — but `Tuple[...]` is preferred for state types.
- **Lazy imports** for heavy modules (plotting, HMM) inside functions, not at module top-level.
- **Profile before optimizing.** Don't guess.
- **Avoid** writing expensive tests. `make check` should run under 10s. So no HMM, ML or other long-running computations in tests.
- **Before running** expensive tasks, verify it runs and outputs the expected data. No running a 300s task to discover that it doesn't print anything, or we truncated the needed output by misusing `head` or `tail`.

### 3. Testing

Tests live alongside code in `tests/` subdirectories, named `test_*.py`.

```bash
uv run pytest                                  # all tests
uv run pytest src/bt/engine/tests/ -v          # specific module
```

#### Rules

- **Every new feature/computation gets a test.** Only strategies, scripts and research are excluded.
- **Use `respx`** for mocking HTTP calls (IBKR API layer).
- **Use `pytest-mock`** for general mocking.
- **Test pure functions first** — they're the easiest and most valuable to test.
- **Test edge cases:** empty DataFrames, single-row windows, NaN handling, boundary timestamps.
- **Test determinism:** given the same inputs, backtest results must be identical. Use fixed seeds and fixed data.
- **Test files go in the module's `tests/` directory**, e.g., `src/bt/risk/tests/test_risk.py`.

```python
# ✅ Good test — exercises a pure function, tests edge cases
def test_zscore_empty_series():
    result = calculate_zscore(pd.Series([], dtype=float))
    assert len(result) == 0

def test_zscore_constant_series():
    s = pd.Series([5.0] * 100)
    result = calculate_zscore(s, window=20)
    assert result.dropna().abs().max() < 1e-10  # ~zero z-score
```

### 4. Functional Patterns

The codebase is built around **immutable state + pure functions**.

```python
# ✅ Pure function — returns new state
def apply_fill(portfolio: PortfolioState, fill: FillEvent) -> PortfolioState:
    new_positions = {**portfolio.positions, fill.signal.symbol: new_position}
    return replace(portfolio, positions=new_positions, cash=new_cash)

# ❌ Mutation
def apply_fill(portfolio: PortfolioState, fill: FillEvent) -> None:
    portfolio.positions[fill.signal.symbol] = new_position  # mutates!
```

#### Rules

- **`replace()` for state updates.** Use `dataclasses.replace()` when modifying frozen dataclasses.
- **Return new state, never mutate.** Every function in the pipeline takes state in, returns new state out.
- **Compose functions,** don't chain methods. The backtest engine composes `strategy_fn → exec_handler → risk_handler`.
- **Protocol-based injection** over class inheritance. Engine-handler seams are Protocols (`ExecutionFn`, `RiskCheckFn`, …); strategy *authoring* is the stateful DSL (a ``@strategy``-decorated pure function of ``StrategyContext``) — the only supported authoring surface. See §9 for how the engine feeds a decorated strategy's ``on_candle``.
- **No side effects in pure functions.** I/O (DB, HTTP, file) belongs at the edges.
- **Use `merge_bt_state`** for partial state updates — it's the established pattern.

##### Mutable strategy state: hold it in the stateful DSL's `ctx.shared`

Strategies that carry cross-call state (cooldowns, trails, cache dicts, sets,
bar counters, model objects like `OnlinePairs`/`OnlineRegime`) must hold it in
`ctx.shared`, using the **stateful DSL** (`@strategy(stateful=True)`). The
engine mints a **fresh `ctx.shared` dict per run/window**, so cross-window
bleed is impossible by construction:

```python
@strategy(bars="1d", stateful=True)
def on_candle(ctx: StrategyContext):
    cooldowns = ctx.shared.setdefault("cooldowns", {})
    trails = ctx.shared.setdefault("trails", {})
    ...
```

- All cross-call state lives under `ctx.shared` — nothing at module scope.
  This is what makes strategies safe to run concurrently across
  `sweep`/`split`/`optimize` **worker processes** without global races.
- Do **not** write module-level `GLOBAL` dicts with a hand-rolled
  `reset_global()`; per-run `ctx.shared` removes the need to reset, and
  discarding module globals avoids silent cross-fold bleed entirely.
- The DSL attaches a no-op `reset_global()` back-compat shim; the runner
  calls it defensively between units, but you should not rely on it — state is
  already per-run.

#### Size constraints (from README)

- Functions: **≤ 50 LOC**
- Classes: **≤ 150 LOC**
- If you're exceeding these, extract or decompose.

### 5. Code Organization

```
src/bt/
├── state/types.py      # All immutable state dataclasses
├── types.py            # Protocols, StrategyConfig, enums
├── engine/             # Backtest loop (pure functional)
├── strategies/              # Strategy implementations
├── models/             # Z-score, regime, market data models
├── portfolio/pure.py   # Pure functions for position/PnL
├── execution/pure.py   # Signal → fill execution
├── risk/pure.py        # Stop-loss / take-profit checks
├── indicators.py       # Technical indicators (pure functions)
└── metrics.py          # Performance metrics (pure functions)
```

#### Module naming

- `pure.py` = stateless functions. No classes, no side effects.
- `types.py` = type definitions, Protocols, dataclasses.
- `__init__.py` = public API exports and wiring/DI.

### 6. Negative-Space Programming

Use assertions instead of early returns where it improves clarity.

```python
# ✅ Assert preconditions (DSL surface)
def on_candle(ctx: StrategyContext):
    close = ctx.ta.close(ctx.candle.symbol)[-1]
    assert close is not None, "price must be available before signal generation"
    assert ctx.candle.symbol in ctx.params.symbols, f"Unexpected symbol: {ctx.candle.symbol}"
    ...
```

Assertions document invariants. They're also free runtime checks during tests.

### 7. Configuration & Strategy JSON

Strategies are defined in JSON files loaded via `load_strategy()` → `StrategyConfig`.

- Add new fields to `StrategyConfig` dataclass, not as loose dict entries.
- Validate config early. The codebase currently lacks schema validation — when adding validation, use dataclass field constraints or a schema library, not ad-hoc checks scattered across the codebase.

### 8. Async I/O

- The IBKR data sync layer uses `httpx` with `asyncio`.
- The backtest engine itself is synchronous (pure computation). Async only at the I/O boundary.
- Use `pytest-asyncio` for testing async code.
- Use `respx` for mocking HTTP in async tests.

### 9. Engine Data Flow to `on_candle`

How the engine feeds data to a decorated strategy's ``on_candle(ctx)``. The DSL
adapter exposes the engine state via ``ctx``: ``ctx.state`` (the BacktestState / CandleStore
below), ``ctx.candle`` (the current Candle), and ``ctx.params`` (typed ``Params`` subclass or
raw dict). Read the engine-internal names below with that mapping.

#### `on_candle` fires once per timestamp

The engine calls the strategy's ``on_candle`` **only when ``ctx.candle.symbol``
is the last symbol** in `config.symbols`. With `["AAPL", "GOOGL", "MSFT"]`,
the generator yields → AAPL → GOOGL → MSFT per timestamp before moving to the
next timestamp. ``on_candle`` fires on MSFT.

**Why:** At that point `state.candles` contains all symbols' data up to the
current timestamp. If the engine fired on every symbol, the first symbol's
invocation would see incomplete data (later symbols haven't been appended yet).

#### Parameter reference

- **`state: BacktestState`** — full snapshot (portfolio, pending_signals, candles)
- **`candle: Candle`** — the OHLCV bar for the last symbol at current timestamp. Has `.symbol`, `.interval` (`"1h"`, `"4h"`, etc.), `.open`, `.high`, `.low`, `.close`, `.volume`
- **`params`** — typed dataclass (`StrategyParams` subclass) resolved by `resolve_params(config.strategy_type, config.strategy_params)`; or raw `dict` if no typed params registered

#### CandleStore: `state.candles`

The primary data interface. A `Mapping[(str, str), DataFrame]` keyed by
`(symbol, interval)`. Backed by a cursor that ensures lookahead safety:

```python
# DataFrame access (cursor-truncated — safe, no future data)
df = state.candles[("AAPL", "1h")]              # KeyError if missing
df = state.candles.get(("AAPL", "4h"))          # None if missing

# O(1) fast path — absolute latest, no DataFrame allocation
close = state.candles.latest("AAPL", "1h")      # float | None
n     = state.candles.count("AAPL", "4h")        # int
```

**Cursor semantics:** The engine calls `state.candles.advance(ts)` before each
`on_candle` invocation. `__getitem__` and `get` build DataFrames truncated to
rows ≤ cursor. `latest()` and `count()` ignore the cursor (absolute latest).

#### Strategy-owned state: `ctx.shared`

There is **no** `ModelState` and no engine-level `model_updater`. Cross-candle
state is fully strategy-owned via the stateful DSL's `ctx.shared` —
per-run dict minted fresh by the engine for every split/sweep window, so
cross-window bleed is impossible and strategies are safe across concurrent
worker processes. Read/write `ctx.shared["key"]`. Model objects (e.g.
`src.indicators.kalman.strategy.OnlinePairs`, `src.indicators.hmm.strategy.OnlineRegime`)
live in `ctx.shared` and are fed per candle; there is no hidden engine channel.
See section 4 for the full convention.

#### HTF (higher-timeframe) access pattern

HTF candles accumulate in `state.candles` keyed by their interval string (reachable
in DSL as `ctx.state.candles`). The candle generator interleaves HTF candles
(e.g. `"4h"`) at boundaries after all base candles for that timestamp, and the
engine fires the strategy only on base-interval candles — HTF-only candles are
merely accumulated and never trigger signal generation. A strategy reads HTF
structure from the store (``ctx.ta`` serves only the base/signal interval):

```python
@strategy(bars="1h")
def on_candle(ctx):
    htf_df = ctx.state.candles.get((ctx.candle.symbol, "4h"))
    if htf_df is not None and len(htf_df) >= 2:
        htf_trend = htf_df["close"].iloc[-1] > htf_df["close"].iloc[-2]
    ...
```

HTF-only candles (where `candle.interval != base_interval`) **skip the
pipeline** — they are appended to the accumulator but never trigger
`on_candle`, signal execution, or risk checks.

#### Multi-symbol strategy pattern

Read cross-sectional data from `ctx.state.candles` and emit signals for any symbol.
Returned signals are bucketed by `signal.symbol` into `state.pending_signals`
(a `dict[str, tuple[TradeSignal, ...]]`). The engine drains each symbol's bucket
when that symbol's candle iteration reaches Stage 4.

Key rules:

- Signals for any symbol are valid — the engine dispatches fills by `signal.symbol`.
- The engine fires only on the last symbol per timestamp, so a cross-sectional read
  via `ctx.state.candles` is always complete for every configured symbol. (No manual
  interval gating needed — HTF-only candles never trigger the strategy.)
- `_execute_pending` (Stage 4/6) reads directly from `state.pending_signals[symbol]`.
  No O(N) scan over all pending signals — routing is explicit and O(1).
- **Same-bar execution:** Signals emitted during `on_candle` for non-current
  symbols will fill in the same bar cycle — the corresponding symbol's
  `_execute_pending` stage runs immediately before its `_generate_signals`.

#### `qty` in TradeSignal — two sizing modes

Authoring is the DSL, so sizing flows through the DSL + engine `SizingParams`
(config-level `position_size` was removed — sizing/sl/tp are strategy-owned via
`strategy_params` and per-trade `ctx.long(..., size=, sl=, tp=)`).

- **`qty = 0` (engine-sized):** `ctx.long(sym)` with no `size` emits `qty=0`;
  the engine's shared sizing layer derives the share count from `SizingParams`
  (`equity`/`cash`/`fixed` base + `size` fraction, per-symbol + cash caps). A
  `close` always flattens the targeted lot.
- **`qty > 0` (explicit):** `ctx.long(sym, size=0.1)` converts the 0–1 fraction
  of the capital base (`size_mode="capital"` = initial capital, `"equity"` = live
  MTM equity) to an absolute share count before the signal is emitted. Either way,
  `qty` on the emitted `TradeSignal` is an absolute share count, not a fraction —
  the engine never rescales a DSL-emitted `qty`.

## Quick Checklist Before Committing Code

- [ ] Run `make check` for formatting, lintin, typechecking and testing
- [ ] All functions/classes fully type-annotated (no `Any` without comment)
- [ ] State changes return new objects (no mutation)
- [ ] Hot-path computation is vectorized (numpy/pandas, not Python loops)
- [ ] New logic has tests covering edge cases
- [ ] Functions ≤ 50 LOC, classes ≤ 150 LOC (or extracted)
- [ ] Protocols used for injection, not inheritance
- [ ] No dead code or commented-out blocks left behind
