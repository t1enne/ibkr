# AGENTS.md — IBKR PY

> AI coding agent instructions for this repository.
> Read this before generating any code.

## Alpha research

Read SKILL.md **before** authoring, editing, tuning or evaluating any strategy —
it is the authoring contract (DSL surface, config fields, `bt` subcommands).
Hard rules (full text at the top of `SKILL.md`):

- **Never test a strategy.** Strategy modules, `strats/*.json`, research scripts
  and sweeps are exempt from the test rule below — validate by **running** them.
- **Use `bt run`/`sweep`/`split`/`optimize` — never a throwaway script.** A
  hand-rolled candle-load + grid-loop + `run()` harness is `bt sweep` (or
  `bt optimize` for per-fold IS tune → OOS validate). Custom scripts only beyond
  this surface — say so first.
- **Never filter or re-derive a run's output.** No `grep`/`head`/`tail` over the
  report, no recomputing metrics. The risk story (kurtosis, skewness, stability,
  per-symbol draws, worst-DD) is in the tail. Name any trim.

### Strategy development lifecycle (mandatory)

Full text in `SKILL.md` § Workflow. In order:

1. **Minimum parameters.** Use as few params as possible. Before adding one, ask
   whether the signal generation can be rewritten so the param is unnecessary.
   Every param is a fitted degree of freedom.
2. **Develop on 1 symbol, ≤1 year.** Keep the trading window to max 1 year and a
   single symbol. Then spawn a specialized subagent to **review every trade, one
   by one**, against the stated entry/exit rules. Report trades + verdict; do not
   proceed until the **user approves the entries and exits**.
3. **Expand exponentially after approval.** Grow one axis at a time — symbols
   1 → 2 → 4 → 8 → …, window 1y → 2y → 4y → full. Re-run `bt run` after each
   doubling step and use `bt split --folds` to separate IS from OOS. The step
   where the edge breaks is the finding. Classify into `strats/<pass|wip|fail>/`
   only after the expanded run survives.

## Development Workflow

When implementing a feature or fix:

1. **Understand** — read relevant strategy code, types, and tests. Don't guess.
2. **Plan** — state approach before writing. If unclear, ask.
3. **Implement** — minimum code that works. Pure functions, immutable state, full type annotations.
4. **Test** — every new computation gets a test, **except** strategies, strategy configs, scripts and research — those are validated by _running_ them, never by unit tests (see § Alpha research). Run `make test` before declaring done.
5. **Verify** — `make check` must pass (lint + format + typecheck + tests).

### Running things

**Read full `bt run` / sweep output — never filter it through `grep`/`head` when
reporting.** Metrics that live in the tail/body (kurtosis, skewness, stability,
per-symbol draws, worst-DD periods) carry the actual risk story; a summary
line like Sharpe can look healthy while kurtosis or a single symbol's bleed
tells the real tale. Only trim the output when you are very sure the removed
rows add no signal, and say when you cut it.

Every report — `bt run|sweep|split|optimize`, text and `-F json` — draws its
metric columns from one canonical set (Sharpe, Ann, MaxDD, Kurt, Skew, Win,
Trades, Scaled, Rejected) defined in `src/bt/report_metrics.py`, so text
columns and JSON keys cannot drift. `Scaled` is the count of opening fills
whose qty was reduced by a shared cohort cash scale (`PortfolioResult.scaled_trades`,
from `portfolio.pure.ScaleRecord`); `Rejected` is the count of fills dropped
for genuine cash exhaustion (`PortfolioResult.rejected_trades`). Either being
non-zero means the run is advisory across symbol permutations (see Fills
below). The `bt run` trade list is opt-in (`--trades`): off, every format
still reports the count (text note / top-level `total_trades`).

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

### Fills: rejected or scaled = results not solid

A cash shortfall SCALES a multi-open cohort (entries land at `scale × plan`),
only a lone open rejects. Both are surfaced in the FINAL metrics (`Scaled`,
`Rejected`) — no mid-run stderr warning. Cohort grouping is order-sensitive
(clock, data gaps, Stage 4 vs 6), so with rejections or over-subscription
active, results vary with `config.symbols` order — advisory, never quoted
across permutations. See `src/bt/portfolio/pure.py` + `src/bt/engine/backtest.py`
docstrings.

## Running Screens

Screens are scoring layers that return 0..1 `ScreenResult`
(`score` + `action` long/short/flat + reasons + `model_features`), NEVER fills.
The old `ibkr screen` CLI layer was deleted — screen scoring has no CLI. But
there IS a live-intent CLI over a **real strategy**: `ibkr bt screen
<strategy.json>` runs the strategy's own `on_candle` through the engine and
surfaces its current-bar intent (opens AND closes), never fills. Append
`--trades` (default off) to also print the run's executed trades — the engine's
real fills, the final-bar flatten included (see SKILL.md § `bt screen`).

- **Code:** `src/bt/screen/` (`types.py`, `runner.py`, `adapter.py`, `screens/*.py`).
- **Discovery:** any `screens/*.py` with `SCREEN_TYPE` + `on_state(state, params)` — auto-`_discover()`ed, no registry.
- **The 6:** `momentum`, `macd_divergence`, `mfi_divergence`, `obv_divergence`, `rsi_divergence` (fresh signals) + `rs` (relative strength — cross-sectional ranking; fires by construction, so DON'T count it as corroboration).
- **Benchmarks:** `rs` needs its benchmark in-state (raises otherwise); `momentum` gates on `QQQ`. Pass `benchmarks=['QQQ','SPY']`.

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

Two SQLite files, both under the repo's `../data/` directory (`py/`'s sibling),
all resolved by **`src/db/path.py`** — one path truth per file, file-relative (never
`os.getcwd()`, which silently reads the wrong file from another directory).

- **`data/db.sqlite`** — candles, `symbol`, `fundamental`: bulk, regenerable
  research data. `IBKR_DB_PATH` overrides it. `resolve_db_path()` /
  `DEFAULT_DB_PATH`.
- **`data/live.db`** — the durable live book (`live_*`). `IBKR_LIVE_DB_PATH`
  overrides it. `resolve_live_db_path()` / `LIVE_DB_PATH`. It is a SEPARATE file
  because the book is state no download can rebuild: a bloated or corrupt research
  write must not be able to take it down with it (todo D18).

```bash
uv run ibkr db status              # every migration + applied/pending, both files
uv run ibkr db migrate --dry-run   # what WOULD run; writes nothing
uv run ibkr db migrate             # apply pending (both files)
uv run ibkr db migrate --down --yes  # unwind (refuses an irreversible target)
```

- **`src/db/` is a leaf layer:** `path`, `connection`, `models`, `introspect`
  import NOTHING from `src.*`. Only `src/db/migrations/versions/**` may reach back
  into `src.live` / `src.data`. `src/db/__init__.py` must NOT re-export
  `migrations.versions.*` — that would drag `src.live` into every `import src.db`.
- **`sqlite3` CLI may not be installed.** Query with Python instead:
  `python -c "import sqlite3; c=sqlite3.connect('../data/db.sqlite')"`, or use
  the CLI above.

### Migrations

- **`src/db/migrations/`**: `types` (frozen `Migration` + `Registry`),
  `bookkeeping` (our own `peewee_migration` table — NOT kysely's),
  `runner` (`run_pending` / `run_down` / `status`), `helpers`,
  `versions/` (the explicit ordered tuples `LIVE_MIGRATIONS` / `DATA_MIGRATIONS`).
- **Order is the tuple order** in `versions/__init__.py`. There is no filename scan
  and no discovery — a rename cannot silently reorder history.
- **Two registries, never one.** `ibkr db migrate` runs both (one per file); the
  ledger's `_ready_schema` runs **LIVE only**, so the first write of a cycle never
  replays a slow multi-million-row candle migration.
- **Concurrency:** a batch's `up()`s + bookkeeping writes share ONE
  `BEGIN IMMEDIATE` transaction, so racing runners serialize on the SQLite write
  lock. There is no separate lock table — the write lock IS the mutex.
- **`down()` only where a genuine, lossless inverse exists.** Live migrations
  rename and never drop, so `live_0001_baseline.down` is absent and `run_down`
  refuses loudly (`IrreversibleMigrationError`) rather than partially unwinding.
  `data_0001_baseline.down` is a real no-op (it owns no rows).
- **The baseline is an idempotent absorber, and its guards are load-bearing.**
  `live_0001_baseline.up` is the old `ledger_migration.migrate()` body lifted
  verbatim: recognition guards distinguish three `live_position` shapes, re-keys
  early-return when the key is current, column adds are `IF NOT EXISTS`-style.
  An operator's DB has no `peewee_migration` row, so this runs against whatever
  shape that file is in. Do not "clean up" the guards to assume a version.

### Schema

- **`symbol`**: `conid` (PK), `ticker`, `name`, `market`, `currency`.
- **`candle`**: `id` (PK autoincrement), `ticker`, `conid`, `timestamp` (ms epoch),
  `open`, `high`, `low`, `close`, `volume`. Indexed on `(ticker, timestamp)` —
  always filter by `ticker`, never join through `symbol.conid` (that join is
  unindexed and ~20x slower). `ticker` is stored UPPERCASE.
- **`peewee_migration`**: `name` (PK), `applied_at`. OUR bookkeeping, separate from
  the TS-created `kysely_migration` (which the TypeScript data pipeline owns and
  may rewrite — sharing it would couple two tools' rollout state).
- **The `live_*` tables** live in `data/live.db`, not here (see § Where state lives).

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

## Live trading (`src/live`)

One-shot reconcile cycles over the **same strategy JSON** the backtester loads.
There is no daemon — you schedule it. Open work: `todo.md`.

```bash
uv run ibkr live run strats/pass/<cfg>.json --adapter sim --dry-run  # places/writes nothing
uv run ibkr live run strats/pass/<cfg>.json --adapter ibkr --allow-live
uv run ibkr live abandon --scope <scope> --symbol AAPL --action long --yes
```

- `--allow-live` is required to read a `live` account; a `paper` config pointed
  at a live account is refused.
- **Adapter resolution:** CLI `--adapter` (unset by default) > config `adapter` >
  legacy config `broker` > `ibkr`. Both adapters (`ibkr`, `sim`) are STATELESS —
  the fetched book travels in as a parameter and the settled results come back
  out; the durable book lives in sqlite (`live_position`), never on the adapter.
- **Scope grammar:** `scope = <adapter>_<config_name>_<config_hash>`, where
  `<config_hash>` is a short digest of the **strategy-intent fields only**
  (strategy type, symbols, params, bars, warm-up, sizing — `src/live/scope.py`).
  Friction/capital/broker/bookkeeping edits do NOT re-mint it. **Operator risk:**
  a strategy-intent edit while exposed re-mints the scope, and the OLD scope's
  book is no longer reconciled by the new one → the new scope can enter the same
  position again, doubling exposure. Edit strategy intent while FLAT.
- **One book table:** `live_position`, keyed `(scope, position_id)`, with
  `source ∈ {account, executions}`. `account` is the human/broker-editable exposure
  surface (sim lots); `executions` is the engine-owned fold of our own fills, used
  as the divergence oracle. A human edit surfaces as `DIVERGENCE` (exit 3) — never
  a silent re-size onto the account's number.
- **Per-scope lease:** `<db>.<scope_tag>.cycle.lock` (an OS advisory `flock`, held
  for the cycle). Per scope, not per DB, so two adapters (or two configs) run
  CONCURRENTLY while the same scope cannot overlap.
- **Exit codes are the machine contract:** `0` clean, `1` config/stale/gateway
  (`ClickException`), `2` usage, `3` **unsafe cycle**. Alert on non-zero.
  `--allow-unsafe` is for report-consuming callers only — never put it in a
  scheduled run.
- `--dry-run` must place nothing, write nothing and take no lease. Tests assert
  zero DDL on a dry run; keep them passing.
- `ibkr live abandon` clears OUR durable record only. It does not cancel anything
  at the broker, so verify broker-side before passing `--yes`, or the next cycle
  places a duplicate.

### Invariants — do not weaken these

1. **Never POST after a failed working-orders read.** Fail closed: a duplicate
   order is unbounded exposure, a skipped cycle is recoverable next bar.
2. **The order ref is bar-free** — `scope_tag-token-attempt`, from
   `(scope, symbol, action, position_id)` plus a re-send counter, never the clock
   or the bar. A durable id must not move because a sibling in the same batch
   appeared, and a re-send must never reuse an attempt.
3. **Ambiguity is never reported as "not placed".** Absence from the gateway's
   working-orders read proves nothing: that endpoint lists working orders **and**
   orders filled/cancelled in the current session. Unknown is reported as
   `unresolved`, never as `rejected`.
4. **`live_order_intent` is the only owner of OPEN state.** Never re-mint an
   order whose predecessor is still OPEN or still working; never downgrade a
   `WORKING` record to `UNRESOLVED` (it carries the provenance a later cycle
   needs).
5. **Adoption matches our own `order_ref` by exact `scope_tag` prefix.** Never
   symbol+side — the account is shared with other scopes and with a human.

Also: migrations **rename, never drop**, and re-keys preserve rows; the close and
open guards fail **closed** on every unknown, and the open guard may be excused
only by our own already-confirmed reducing fills (a same-cycle flip) — do not
widen that; stop-loss/take-profit and LMT are **refused**, never silently sent
naked and never implied in the report; a partially filled entry is not topped up.

### Gateway contract

`/iserver/account/orders` echoes our client order id as **`order_ref`** (not
`cOID`), and an order we did not place carries no `order_ref` at all. The repo's
`openapi.spec.json` is **stale** for this endpoint and must not be edited to
match. **Any test touching a gateway response shape parses a captured fixture**
from `src/live/adapters/ibkr/tests/fixtures/` — a hand-built mock invented the
wrong field name and hid the bug through several review passes.

### Testing the live path

- Live code is async + HTTP: `pytest-asyncio` with `respx`.
- A mock that stands for broker state must **move** when our order lands. A
  constant account position held across a close hid a real refusal bug.
- The sim adapter settles cohorts with the backtest's own rule
  (`src/bt/portfolio/pure.py`) and the same shared cash scale. Do not fork the
  rule for live — if live must differ, change the shared rule deliberately and
  say so.
- Strategies, configs and research scripts are validated by RUNNING them, not by
  unit tests (see § Alpha research) — the live edge itself is not exempt.
- A test that passes both before and after your change is a characterisation
  test, not a regression test. Say which one you wrote.

#### `src/live` tests: critical paths only

A `src/live` test earns its place only when it pins one of these behaviours:

1. **Order-safety invariants** (see §Live trading): never POST after a failed
   working-orders read; the order ref is bar-free; ambiguity is reported
   `unresolved`, never as "not placed"; an OPEN intent is never re-minted;
   adoption matches our own `order_ref` by exact `scope_tag` prefix.
2. **Exit-code contract**: clean cycle `0`, config/stale/gateway `1`, usage `2`,
   unsafe `3`; `--allow-unsafe` exits `0` AND names the suppressed outcomes on
   stderr.
3. **Money/book accounting**: a fill advances the scope's book and cash through
   the ledger; sim and ibkr settle through the same shared cohort-scale rule.
4. **Divergence**: the fill-derived book vs the account book — agreement is
   silent, disagreement is `DIVERGENCE` and unsafe.
5. **Identity/scope**: `config_hash` covers strategy-intent fields only; two
   adapters never share a scope.
6. **Durability**: migrations preserve rows and never drop; `--dry-run` writes
   nothing and takes no lease; two cycles on one scope cannot overlap.
7. **Reconcile posture**: flat→long, add, reduce, close, invert.

**Not eligible — delete on sight:** rendering/formatting assertions (table
layout, padding, text wording, column ordering); shape assertions (dataclass
field lists, dict keys, schema columns, reprs); mock-only wiring tests that fail
only if a call disappears (no behaviour branch); direct tests of private helpers
already covered through a public entry point; anything that would still pass if
the behaviour it claims to protect were broken.

**Mocks:** only at the true edges — HTTP (`respx` over the captured fixtures in
`src/live/adapters/ibkr/tests/fixtures/`) and the clock. A mock standing for
broker state must move when our order lands. For everything else run against a
temp sqlite via `IBKR_DB_PATH`.

### Where state lives

The live book has its **OWN sqlite file**, `../data/live.db` (overridable via
`IBKR_LIVE_DB_PATH`, or `db_path` on `SqliteLedger`), resolved by
`src/db/path.py::resolve_live_db_path`. It is split from the candle file because
the book is durable state no download can rebuild (todo D18). Tables:
`live_strategy`, `live_cash`, `live_position` (the ONE book table — both roles,
told apart by `source`), `live_execution`, `live_order_intent`, `live_scope_alias`
(a re-keyed legacy scope reads as its new name), plus the `*_legacy` copies past
migrations preserve (e.g. `live_position_legacy`, `live_sim_lot_legacy`) rather
than drop. The cycle lease is an OS advisory lock on `<db>.<scope_tag>.cycle.lock`
— sound on a single host with a local filesystem only.

**The split was a one-off, already performed.** The book lives in `data/live.db`
and was populated by a verified row-for-row copy whose tooling
(`ibkr db adopt-live`) has since been removed as spent. The stale `live_*` tables
that copy left behind in `data/db.sqlite` were then DROPPED by
`data_0002_drop_migrated_live_tables` — the one sanctioned exception to
rename-never-drop, gated on a guard that refuses unless the live file exists, is
migrated, and holds at least as many rows per table. `data/db.sqlite` now holds
only `symbol` / `candle` / `fundamental` (plus the 1-row `kysely_*` lineage
tables).

Getting this wrong is the dangerous direction: a ledger pointed at an empty or
wrong file reads a held position as FLAT and re-enters it (double exposure).
Verify `ibkr db status` and the live file's `live_position` rows before trusting a
flat book. `data/db.sqlite.pre-adopt.bak` and `data/live.db.pre-drop.bak` are the
pre-step snapshots.

`src/db/connection.py` holds one process-global peewee handle per file (`db`,
`live_db`). The `live_*` **models** deliberately do NOT bind to one: peewee binds
at CLASS level and tests run two ledgers on two paths in one process, so they keep
the per-instance `SqliteDatabase(path)` + `bind_ctx(LIVE_MODELS)` pattern (see
`src/live/models.py`). Do not "simplify" that to a process-wide handle.

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

- **Use `Protocol` for dependency injection** — the codebase uses `ExecutionFn`, `RiskCheckFn`, `PositionSizerFn`, `DataLoaderFn` for engine-handler seams. Follow this pattern. Never pass raw `Callable` when a Protocol exists or should exist. (Strategy _authoring_ has its own seam: a `@strategy`-produced callable of `StrategyContext` — not a raw `StrategyFn` Protocol; see §4.)
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
- **Protocol-based injection** over class inheritance. Engine-handler seams are Protocols (`ExecutionFn`, `RiskCheckFn`, …); strategy _authoring_ is the stateful DSL (a `@strategy`-decorated pure function of `StrategyContext`) — the only supported authoring surface. See §9 for how the engine feeds a decorated strategy's `on_candle`.
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

#### Docstrings (hard rule)

- **≤ 50 lines** per docstring. Over that → delete, not expand.
- **No numbers, no stats that drift.** Symbol counts, row counts, Sharpe/return
  figures, version strings, dates, "N tickers", "~X%", benchmark numbers — all
  rot the moment data/config changes and mislead readers. State the behavior,
  not the measurement.
- **High level only.** One-line purpose + contract (inputs, outputs, invariants,
  failure modes). Leave derivation, history, thresholds and worked examples to
  code and tests.
- Docstrings document _intent and guarantees_, never current results. If a fact
  is bound to change, it does not belong in a docstring.

### 5. Code Organization

```
src/bt/
├── state/types.py      # All immutable state dataclasses
├── types.py            # Protocols, StrategyConfig, enums
├── engine/             # Backtest loop (pure functional)
├── strategies/              # Strategy implementations
├── models/             # Z-score, regime, market data models
├── portfolio/pure.py   # Pure functions for position/PnL
├── exchange/sim.py     # SimExchange — signal → fill execution (Broker adapter)
├── risk/pure.py        # Stop-loss / take-profit checks
├── indicators.py       # Technical indicators (pure functions)
└── metrics.py          # Performance metrics (pure functions)

src/live/
├── identity.py         # IntentKey / IntentRecord / order identity (bar-free cOID)
├── reconcile.py        # pure: signals + book → OrderIntent[] (posture diff)
├── engine.py           # the cycle: lease → resync → reconcile → place → record
├── broker.py           # LiveBroker / PortfolioSource Protocol seams + sim broker
├── ports.py            # small injected seams (e.g. BookExposure)
├── lease.py            # exclusive cycle lock
├── ledger.py           # SqliteLedger seam over the stores below
├── ledger_base.py      # schema/template/DDL/migration plumbing
├── ledger_sim.py       # sim-lot book
├── ledger_migration.py # preserve-don't-drop migrations and re-keys
├── cli.py              # `ibkr live run` / `abandon`, report rendering, exit codes
└── adapters/ibkr/      # the real edge: broker, orders, trades, mapping, authz
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
- The live cycle (`src/live/engine.py`) is async end to end: it awaits the portfolio
  read, the broker's working-orders read and every placement. Reconcile is pure and
  synchronous — keep I/O out of it so it stays directly testable.
- Use `pytest-asyncio` for testing async code.
- Use `respx` for mocking HTTP in async tests.

### 9. Engine Data Flow to `on_candle`

How the engine feeds data to a decorated strategy's `on_candle(ctx)`. The DSL
adapter exposes the engine state via `ctx`: `ctx.state` (the BacktestState / CandleStore
below), `ctx.candle` (the current Candle), and `ctx.params` (typed `Params` subclass or
raw dict). Read the engine-internal names below with that mapping.

#### `on_candle` fires once per timestamp

The engine calls the strategy's `on_candle` **only when `ctx.candle.symbol`
is the last symbol** in `config.symbols`. With `["AAPL", "GOOGL", "MSFT"]`,
the generator yields → AAPL → GOOGL → MSFT per timestamp before moving to the
next timestamp. `on_candle` fires on MSFT.

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
structure from the store (`ctx.ta` serves only the base/signal interval):

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
- [ ] Docstrings ≤ 50 lines, high level, zero driftable numbers/stats
- [ ] Protocols used for injection, not inheritance
- [ ] No dead code or commented-out blocks left behind
