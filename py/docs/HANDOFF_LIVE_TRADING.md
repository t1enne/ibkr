# Handoff: Live Trading Engine

**Architect:** pi (architect mode)
**Implementer:** you
**Status:** Ready to implement — **open decisions in §Open Decisions. Do not pick silently; wire them from the config or ask.**

This doc is the single source of truth for building `src/live/`. It **reuses the
existing data + screen + portfolio/execution layers** and adds only a thin
**batch reconcile cycle** on top. There is no tick→bar loop and no socket feed.

---

## 0. TL;DR — the live command is a 5-stage batch cycle

```
cron: ibkr data dl          →  candles land in the local SQLite
        │
        ▼
ibkr live run <config.json>  (one-shot cycle; cron it)
  1. sync     — assumed done by cron (step 0). Live reads the local DB.
  2. fetch pf — PortfolioSource.fetch() → PortfolioSnapshot (shared PortfolioState)
  3. signals  — run the strategy-as-screen → actionable LiveSignals
  4. reconcile— pure diff: target posture vs live book → OrderIntents
  5. execute  — Broker.place_cohort(intents), settled atomically (SIMULATED for v1)
  6. record   — Ledger: mark opened lots / mark-closed filled closes
```

- **New package `src/live/`** — 8 modules + tests + 1 edit (`main.py`).
- **One portfolio interface, both sides.** Live **reuses the backtest
  `PortfolioState` and `Position`** (`src/bt/state`) — there is no parallel
  `LivePortfolio`. `apply_fill` / `apply_fills` / `equity_of` therefore apply
  unchanged to a live book, and `reconcile` depends only on a minimal
  `PortfolioView` Protocol that `PortfolioState` satisfies structurally.
- **Reuses (do NOT reimplement):** `query_candles` (`src.data.db`),
  `run_screen_from_strategy` / `render_screen_json` / `ScreenRow`
  (`src.bt.screen`), `load_strategy`, `SizingParams` (`src/bt/size/pure.py`),
  `execute_signal` / `apply_fill` / `apply_fills`
  (`src/bt/{execution,portfolio}/pure.py`), `ActionType`, `TradeSignal`,
  `FillEvent`, `Position`, `PortfolioState`.
- **The screen is the signal source.** No separate signal vocabulary: the live
  layer consumes the same actionable intent the `bt screen -F json` command
  emits (`long` / `short` / `close`).
- **Reconcile is pure and testable.** All broker/PF I/O stays at the edges.
- **A strategy→lot ownership ledger** (`src/live/ledger.py`, SQLite). Live-only:
  records which lots a strategy opened, keyed by config hash, and **mark-closes**
  lots when exited. The broker stays the source of truth for cash/qty; the
  ledger only answers "which open lots are mine, and what is their broker id?"
  On close the row is **mark-closed** (status + `closed_at`), not deleted;
  closed rows are pruned by age later.
- **v1 is mock-pf + simulated broker.** Real IBKR portfolio fetch and real
  order routing are separate follow-ons (§YAGNI).

---

## 1. Assumptions & Open Decisions

### Assumptions already locked

- **Market data arrives by cron**, not a live socket. Live reads the local DB
  (`../data/db.sqlite`) exactly as the backtest does. "Is the feed fresh?" is a
  data-freshness check, not a connection state.
- **One cycle per invocation.** The command runs once and exits (idempotent,
  cron-friendly, replayable). Long-running daemons are out of scope.
- **Async only at I/O boundaries** (portfolio fetch, order placement, ledger
  writes). The reconcile core is synchronous and pure.
- **The live book is a `PortfolioState`.** The broker/PF adapter normalises its
  response into the same frozen dataclass the backtest engine carries, so the
  two portfolios share one interface by construction. `trades`/`equity_curve`
  may be empty on a live read; the read-only consumers (`reconcile`, sizing)
  never touch them.
- **`position_id` IS the broker lot id.** A live `Position.position_id` is
  whatever the broker calls that lot (a real IBKR lot/conid handle, or the
  `SimulatedBroker`'s synthetic id). No local pid is minted for live — the
  broker id is the canonical handle, so a reconcile close targets directly.
- **Ledger is live-only; the backtest persists nothing.** The backtest is
  deterministic and in-memory. Only `src/live` reads/writes the ledger tables.
- **Config hash is the strategy scope.** `strategy_id = sha256(canonical
config JSON)`. Re-running a mutated config gets a new id, so lots from an
  older config revision are never mis-adopted (no cross-contamination).

### ❗ OPEN DECISIONS (confirm with the user; defaults noted)

1. **Position sizing.** A screen row may carry `qty` (absolute, already
   converted by the DSL). When it is `0.0` the live layer must size itself.
   _Default:_ size from config via `SizingParams` (`size_mode` + `size` +
   `max_symbol_allocation`); **hard-error on an unsized open** rather than
   guessing a default.
2. **Reconcile on position _value_ or _side_.** _Default:_ **side only** —
   `long` while already long is a HOLD (no resize/rebalance). Resizing live
   books from a batch signal is a separate feature.
3. **Missing-signal semantics.** _Default:_ **absence = HOLD**, never flatten.
   A symbol the screen did not emit is left untouched. Only an explicit
   `close` (or a side flip) closes a live position.
4. **Real orders vs simulated.** _Default:_ **simulated** — `SimulatedBroker`
   logs `OrderIntent`s and returns synthetic `FillEvent`s. Real routing = a
   later `IBKRBroker` implementing the same Protocol.
5. **Trading-hours / freshness gate.** _Default:_ **fail the cycle** if the DB's
   newest bar for the traded universe is older than one base interval, so a
   stale-data cron failure cannot silently retrade yesterday's intent.
   (Optionally gate the `place()` calls on `src.data.xcal.is_non_trading_day`.)

> Only the affected module changes per decision. The envelope
> (`PortfolioSource`, `Broker`, `reconcile`, `run_cycle`) is unchanged.

### ✅ RESOLVED (folded into this doc)

- **Persistence scope key:** config hash (`strategy_id`).
- **Close semantics:** **mark-closed** (status + `closed_at`; row retained,
  pruned by age later) — never a hard delete on close.
- **Backtest persistence:** **none** — ledger is live-only.
- **Lot identity:** `position_id` **is the broker lot id** directly (no local
  pid, no separate `broker_lot_id` column).

---

## 2. Types & Module Ownership

### New modules

| Module                         | Owns                                                            |
| ------------------------------ | --------------------------------------------------------------- |
| `src/live/types.py`            | Domain dataclasses + `PortfolioView` **re-export** + `LiveConfig` |
| `src/live/result.py`           | `Ok`/`Err` `Result` building block (~15 lines, no dep)          |
| `src/live/portfolio_source.py` | `PortfolioSource` Protocol + `MockPortfolioSource`              |
| `src/live/signals.py`          | Bridge: screen run → `tuple[LiveSignal, ...]`                   |
| `src/live/reconcile.py`        | **Pure**: `LiveSignals` × `PortfolioView` → `OrderIntents`      |
| `src/live/broker.py`           | `Broker` Protocol + `SimulatedBroker`                           |
| `src/live/ledger.py`           | SQLite strategy→lot ledger (`Ledger` Protocol + `SqliteLedger`) |
| `src/live/engine.py`           | `run_cycle` orchestration + `CycleReport`                       |
| `src/live/cli.py`              | `ibkr live run <config.json>`                                   |

### Modified

| Module      | Change                                           |
| ----------- | ------------------------------------------------ |
| `src/bt/**` | **none** — screen ready-output is consumed as-is |
| `main.py`   | register `live_group`                            |

### Priority / dependency order

1. `result.py` → 2. `types.py` → 3. `portfolio_source.py` → 4. `signals.py` →
2. `reconcile.py` → 6. `broker.py` → 7. `ledger.py` → 8. `engine.py` →
3. `cli.py` / `main.py`.

---

### All types (verbatim contract)

```python
# src/live/result.py
from typing import Generic, TypeVar
T = TypeVar("T"); E = TypeVar("E")

@dataclass(frozen=True)
class Ok(Generic[T, E]):
    value: T

@dataclass(frozen=True)
class Err(Generic[T, E]):
    error: E

Result = Ok[T, E] | Err[T, E]
```

```python
# src/live/types.py
from typing import Literal, Protocol
import pandas as pd
from src.bt.state import ActionType, PortfolioState, Position

SignalAction = Literal["long", "short", "close"]

# The ``PortfolioView`` Protocol lives in ``src/bt/state/types.py`` — beside
# ``PortfolioState``, the concrete type it abstracts — and is re-exported here so
# the shared bt layer never imports the live package. Same members (`cash`,
# `positions`, `initial_capital`), read-only by design; mutation goes through
# ``apply_fill``.
from src.bt.state import PortfolioView as PortfolioView

@dataclass(frozen=True)
class PortfolioSnapshot:
    """A live broker read: the shared ``PortfolioState`` + when it was read.

    Live deliberately reuses ``PortfolioState``/``Position`` rather than a
    parallel ``LivePortfolio``: one portfolio interface end to end, so
    ``apply_fill`` / ``apply_fills`` / ``equity_of`` apply unchanged to a live
    book. ``as_of`` is the only live-only addition. ``trades`` /
    ``equity_curve`` may be empty on a live read.
    """
    portfolio: PortfolioState
    as_of: pd.Timestamp

@dataclass(frozen=True)
class LiveSignal:
    """Actionable intent from the screen (mirrors ScreenRow, actionable only)."""
    symbol: str
    action: SignalAction
    score: float
    reasons: tuple[str, ...]
    signal_ts: pd.Timestamp | None
    price: float                 # ref price (last close of the decision bar)
    qty: float                   # absolute shares; 0.0 = unsized (size from config)
    stop_loss: float | None = None
    take_profit: float | None = None
    position_id: str | None = None
    tag: str = ""

@dataclass(frozen=True)
class OrderIntent:
    """A single order to submit, after reconciliation. Never a batch."""
    symbol: str
    action: ActionType           # long / short / close
    qty: float                   # absolute shares, always > 0
    ref_price: float
    reason: str                  # e.g. "open long (flat->long)" / "close all lots"
    position_id: str | None = None   # target lot for a close; None = whole symbol
    stop_loss: float | None = None
    take_profit: float | None = None
    tag: str = ""

@dataclass(frozen=True)
class LiveConfig:
    strategy_type: str
    symbols: tuple[str, ...]
    initial_capital: float
    strategy_params: dict            # the screen's strategy params (verbatim)
    bars: tuple[str, ...]            # bars[0] = signal interval
    warmup: str                      # screen warm-up window, e.g. "1y"
    # sizing (used only when a LiveSignal.qty == 0.0)
    size_mode: Literal["equity", "cash", "fixed"] = "equity"
    size: float = 0.0
    max_symbol_allocation: float = 1.0
    portfolio_path: str = ""         # MockPortfolioSource fixture path
    mode: Literal["paper", "live"] = "paper"
```

```python
# src/live/portfolio_source.py
class PortfolioSource(Protocol):
    async def fetch(self) -> Result[PortfolioSnapshot, FeedError]: ...

@dataclass(frozen=True)
class FeedError:
    kind: Literal["auth", "rate_limit", "transport", "bad_fixture", "stale_data"]
    message: str
    symbol: str | None = None
```

```python
# src/live/broker.py
@dataclass(frozen=True)
class OrderResult:
    intent: OrderIntent
    fill: FillEvent | None       # None when rejected (simulated or broker)
    ok: bool
    message: str = ""
    position_id: str | None = None   # broker lot id assigned on an OPEN; None on close

class Broker(Protocol):
    def seed(self, portfolio: PortfolioState) -> None: ...
    async def place(self, intent: OrderIntent) -> Result[OrderResult, FeedError]: ...
    async def place_cohort(
        self, intents: tuple[OrderIntent, ...]
    ) -> Result[tuple[OrderResult, ...], FeedError]: ...
    async def close(self) -> Result[None, FeedError]: ...
```

```python
# src/live/ledger.py
from typing import Literal, Protocol

PositionStatus = Literal["open", "closed"]

@dataclass(frozen=True)
class PositionRecord:
    """One strategy-owned live lot. ``position_id`` IS the broker lot id."""
    strategy_id: str                 # sha256(canonical config JSON)
    position_id: str                 # broker lot id (canonical handle)
    symbol: str
    side: str                        # "long" | "short"
    qty: float                       # absolute shares
    entry_price: float
    entry_time: pd.Timestamp
    stop_loss: float | None
    take_profit: float | None
    tag: str
    status: PositionStatus
    opened_at: pd.Timestamp
    closed_at: pd.Timestamp | None = None

class Ledger(Protocol):
    """Strategy→lot ownership record. Broker is the cash/qty source of truth;
    this only says which open lots belong to this strategy and their handle."""
    def ensure_strategy(self, strategy_id: str, name: str, mode: str) -> None: ...
    def record_open(self, rec: PositionRecord) -> None: ...
    def mark_closed(self, strategy_id: str, position_id: str,
                    closed_at: pd.Timestamp) -> None: ...
    def open_positions(self, strategy_id: str) -> tuple[PositionRecord, ...]: ...
    def prune_closed(self, before: pd.Timestamp) -> int: ...  # rows deleted
    def touch_cycle(self, strategy_id: str, at: pd.Timestamp) -> None: ...
```

```python
# src/live/engine.py
@dataclass(frozen=True)
class CycleReport:
    as_of: pd.Timestamp
    signals: tuple[LiveSignal, ...]
    intents: tuple[OrderIntent, ...]
    results: tuple[OrderResult, ...]
    portfolio_before: PortfolioState   # the shared backtest type
```

---

## 3. Functions (signatures + one-line contract)

### Signals bridge — `src/live/signals.py`

```python
def live_signals(config_path: str, max_age_days: int) -> tuple[LiveSignal, ...]
    """Run the strategy-as-screen and project its actionable rows.

    Calls ``run_screen_from_strategy(config_path, max_age_days=...)`` (the same
    driver ``ibkr bt screen`` uses) and maps every emitted ``ScreenRow`` whose
    action is in ``ACTIONABLE`` (``long``/``short``/``close``) to a
    ``LiveSignal``. ``flat`` rows are never produced: absence of a signal means
    HOLD downstream (see reconcile). Pure mapping, no I/O of its own beyond the
    screen run.
    """
```

Note: this is **in-process** (no shelling out). The screen's `-F json`
document is the _external_ contract for observability / piping — the live
command consumes the driver's typed `ScreenRun`, not JSON text.

### Reconciliation — `src/live/reconcile.py` (the heart; pure)

```python
Side = Literal["long", "short", "flat"]

def current_side(portfolio: PortfolioView, symbol: str) -> Side
    """Net the symbol's lots. ``Position.qty`` is positive; the side lives on
    ``Position.type`` (ActionType.long/short), so: sum +qty for long lots and
    -qty for short lots; > 0 -> long, < 0 -> short, 0 -> flat."""

def target_side(sig: LiveSignal) -> Side
    """long -> long, short -> short, close -> flat."""

def reconcile(
    signals: tuple[LiveSignal, ...],
    portfolio: PortfolioView,
    config: LiveConfig,
) -> tuple[OrderIntent, ...]
    """POSTURE DIFF -> orders. Deterministic, closes-before-opens, symbols in
    ``config.symbols`` order. See the decision table below."""

def size_qty(
    price: float, portfolio: PortfolioView, config: LiveConfig,
) -> float
    """Shares for an unsized open. Build ``SizingParams.from_dict(...)`` from the
    config and call ``compute_qty`` in ``src/bt/size/pure.py``. Equity via the
    shared ``equity_of(portfolio)`` — it takes a ``PortfolioState``, and a live
    book IS one, so no live-specific equity code. Never re-derive sizing."""
```

**Reconciliation decision table (make this exact — it is the whole feature):**

| target        | current        | result                                             |
| ------------- | -------------- | -------------------------------------------------- |
| `flat`        | `flat`         | **no-op**                                          |
| `flat`        | `long`/`short` | **CLOSE** every lot (or just `signal.position_id`) |
| `long`        | `flat`         | **OPEN long** (`qty = sig.qty or size_qty(...)`)   |
| `short`       | `flat`         | **OPEN short**                                     |
| `long`        | `long`         | **HOLD** — no resize in v1 (open decision 2)       |
| `short`       | `short`        | **HOLD**                                           |
| `long`        | `short`        | **CLOSE** short lot(s), then **OPEN long**         |
| `short`       | `long`         | **CLOSE** long lot(s), then **OPEN short**         |
| _(no signal)_ | any            | **HOLD** — absence is not an exit                  |

Rules:

- **Closes precede opens** (mirrors `apply_fills`' non-opens-first rule, so a
  close's freed cash is available to an open in the same cycle).
- **`_close_position` requires a `position_id`.** Every close `OrderIntent`
  therefore carries one, targeting exactly one live lot. A `close` with
  `signal.position_id` emits one intent for that lot; a bare `close` (or a side
  flip) emits **one intent per lot**, each with that lot's `position_id`.
- `qty` on an OPEN is `sig.qty` if `> 0`, else `size_qty(...)`; if that is
  `<= 0` the whole cycle **raises** (`ValueError("unsized open …")`) — never
  place an order sized by accident.
- Symbols not in `config.symbols` are asserted out.

### Sources & broker

```python
# src/live/portfolio_source.py
class MockPortfolioSource:
    """Reads a JSON fixture. Async to match the Protocol; no network."""
    def __init__(self, path: str) -> None: ...
    async def fetch(self) -> Result[PortfolioSnapshot, FeedError]: ...
def load_mock_portfolio(raw: Mapping[str, object], as_of: pd.Timestamp) -> PortfolioSnapshot
    """Pure: fixture dict -> a real ``PortfolioState`` (cash, symbol -> tuple
    of ``Position``\ s, ``initial_capital``, empty trades/equity_curve) inside a
    ``PortfolioSnapshot``. Validates symbols/qty/timestamps/``type``."""

# src/live/broker.py
class SimulatedBroker:
    """No real routing. Prices each intent per-order, then settles the whole
    cycle through the SAME atomic backtest primitive ``apply_fills``, so paper and
    backtest book accounting are identical — an over-subscribed multi-open cycle
    SCALES by one shared cash factor, exactly as a backtest cohort does."""
    def __init__(
        self, portfolio: PortfolioState, params: ExecutionParams,
        log: Callable[[str], None],
        handler: ExecutionHandler | None = None,  # default: default_execution_handler()
    ) -> None: ...
    async def place(self, intent: OrderIntent) -> Result[OrderResult, FeedError]: ...
    async def place_cohort(
        self, intents: tuple[OrderIntent, ...]
    ) -> Result[tuple[OrderResult, ...], FeedError]: ...
    async def close(self) -> Result[None, FeedError]: ...
    def portfolio(self) -> PortfolioState: ...

def intent_to_signal(intent: OrderIntent, ts: pd.Timestamp) -> TradeSignal
    """Pure: OrderIntent -> TradeSignal so ``execute_signal``/``apply_fills``
    (the backtest primitives) price and settle the simulated fill identically.
    Delegates to ``trade_signal`` — the ONE probe-signal factory, beside the ONE
    ``ref_candle`` bar factory; ``reconcile`` imports both instead of rebuilding
    its own. A close intent carries ``position_id`` (required by
    ``_close_position``)."""
```

### Ledger — `src/live/ledger.py` (live-only persistence)

```python
SCHEMA = """
CREATE TABLE IF NOT EXISTS live_strategy (
  strategy_id  TEXT PRIMARY KEY,   -- sha256(canonical config JSON)
  name         TEXT NOT NULL,
  mode         TEXT NOT NULL,       -- 'paper' | 'live'
  created_at   INTEGER NOT NULL,
  last_cycle_at INTEGER
);
CREATE TABLE IF NOT EXISTS live_position (
  strategy_id  TEXT NOT NULL,
  position_id  TEXT NOT NULL,       -- broker lot id (canonical handle)
  symbol       TEXT NOT NULL,
  side         TEXT NOT NULL,
  qty          REAL NOT NULL,
  entry_price  REAL NOT NULL,
  entry_time   INTEGER NOT NULL,
  stop_loss    REAL, take_profit REAL, tag TEXT NOT NULL DEFAULT '',
  status       TEXT NOT NULL,       -- 'open' | 'closed'
  opened_at    INTEGER NOT NULL,
  closed_at    INTEGER,
  PRIMARY KEY (strategy_id, position_id)
);
CREATE INDEX IF NOT EXISTS ix_live_pos_open
  ON live_position(strategy_id, status);
"""

class SqliteLedger:
    """sqlite3-backed Ledger over the SAME ``../data/db.sqlite`` candle DB.
    Creates its two tables idempotently on first use; uses
    ``src.data.db.get_connection`` (never a second db_path)."""
    def __init__(self, db_path: str | Path | None = None) -> None: ...
    # Ledger Protocol methods (sync — tiny local writes; the async boundary is
    # the broker/source, not SQLite)

def config_hash(config: Mapping[str, object]) -> str
    """Pure: sha256 of canonical JSON (sorted keys, no whitespace) -> hex.
    The strategy scope key; a mutated config hashes differently by design."""

from_position(rec: PositionRecord, pos: Position) -> PositionRecord
    """Pure: lift a live ``Position`` into a ledger record (symbol/side/qty/
    entry/sl/tp/tag/position_id)."""
```

**Write points (only two):**

- **After a successful OPEN** (`OrderResult.ok` and `position_id` set) →
  `record_open(PositionRecord(status="open", ...))`.
- **After a successful CLOSE** (`intent.action is close` and `ok`) →
  `mark_closed(strategy_id, intent.position_id, at)` — row retained, `status`
  flips to `closed`, `closed_at` stamped. A failed/rejected close marks nothing,
  so the lot stays open and the next cycle retries.

**Cleanup:** `prune_closed(before)` deletes `status='closed'` rows older than
`before` (default: run-end housekeeping in the CLI, cutoff e.g. 90d). Open rows
are never pruned. Pruning is the only DELETE in the module.

### Engine — `src/live/engine.py`

```python
def build_report(
    portfolio: PortfolioState,
    signals: tuple[LiveSignal, ...],
    intents: tuple[OrderIntent, ...],
    results: tuple[OrderResult, ...],
) -> CycleReport
    """Pure: assemble the cycle report."""

async def run_cycle(
    config: LiveConfig,
    *,
    source: PortfolioSource,
    broker: Broker,
    ledger: Ledger,
    strategy_id: str,
    max_age_days: int = 5,
) -> CycleReport
    """One full batch pass: fetch pf → screen → reconcile → place → record.
    Not a loop. Caller (CLI / cron) drives cadence.
    Steps: fetch → assert data fresh (§1 decision 5) → live_signals →
    reconcile → for intent: await broker.place → on ok, record_open /
    mark_closed per intent.action → ledger.touch_cycle (all skipped when
    ``dry_run``).
    The snapshot's ``PortfolioState`` feeds ``reconcile`` directly — no adapter
    between the live and backtest portfolio. Ownership is scoped by
    ``strategy_id`` so a cycle never closes lots opened by another config."""
```

### CLI — `src/live/cli.py`

```python
@click.group(name="live")
def live_group(): ...

@click.command("run")
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--dry-run", is_flag=True, help="Reconcile + report; place nothing.")
@click.option("--max-age", "-a", type=int, default=5, show_default=True)
@click.option("--format", "-F", "fmt",
              type=click.Choice(["text", "json"]), default="text")
def live_run(config_path, dry_run, max_age, fmt) -> None:
    """Run ONE live cycle (cron-friendly). --dry-run reconciles without placing."""

def load_live_config(path: str) -> LiveConfig
    """Parse + validate the live JSON -> LiveConfig (bars/symbols/sizing)."""
def render_report(report: CycleReport, fmt: str) -> str
    """text table (signals + intents + results) or a JSON document (use
    ``_json_default`` from ``src/bt/cmds/_shared.py``)."""
```

`main.py` edit:

```python
from src.live.cli import live_group
...
main.add_command(live_group)
```

---

## 4. Call Graph (implement as specified)

### Production (one cycle)

```ts
main (click)
  → live_run(config_path, dry_run, max_age, fmt)                       --> src/live/cli.py
    → load_live_config(path): LiveConfig                              --> src/live/cli.py
    → asyncio.run(run_cycle(config, source, broker, max_age))         --> src/live/engine.py
      → MockPortfolioSource(config.portfolio_path).fetch(): Result    --> src/live/portfolio_source.py
        → load_mock_portfolio(raw, as_of): PortfolioSnapshot           (pure)
          → PortfolioState(cash, positions, initial_capital)          --> src/bt/state
      → assert_data_fresh(config)                          --> src/data.db (query_candles)
      → live_signals(config_path, max_age): tuple[LiveSignal,...]     --> src/live/signals.py
        → run_screen_from_strategy(config_path, max_age_days=max_age) --> src/bt/screen
        → map ScreenRun.rows (long/short/close) -> LiveSignal         (pure)
      → reconcile(signals, snapshot.portfolio, config): OrderIntents  --> src/live/reconcile.py
        → current_side(portfolio, symbol) / target_side(sig)          (pure)
        → size_qty(price, portfolio, config)                          --> src/bt/size/pure.py
          → equity_of(portfolio: PortfolioView)                       --> src/bt/size/pure.py
      → broker.place_cohort(intents): Result[tuple[OrderResult,...],..] --> src/live/broker.py
        → intent_to_signal / trade_signal / ref_candle               (pure, one home)
        → execute_signal(signal, ref_candle, params): FillEvent       --> src/bt/execution/pure.py
        → apply_fills(portfolio, fills): (PortfolioState, rejections) --> src/bt/portfolio/pure.py
      → sizing: sized_signal(signal, equity_of(view), cash, bar, params) --> src/bt/size/pure.py
        → equity_of(portfolio: PortfolioView) / compute_qty           --> src/bt/size/pure.py
      → on OrderResult.ok: ledger.record_open / ledger.mark_closed    --> src/live/ledger.py
        → get_connection(): sqlite3.Connection                        --> src/data/db.py
      → if not dry_run: ledger.touch_cycle(strategy_id, now)
      → broker.close()
      → build_report(...): CycleReport                                (pure)
    → render_report(report, fmt): str  → stdout
    → ledger.prune_closed(now - 90d)  (housekeeping, non-fatal)
```

### Tests

```ts
reconcile tests
  → reconcile(signals, portfolio, config): tuple[OrderIntent,...]     --> tests/test_reconcile.py
    - flat→long opens; long→close closes; long→long holds
    - side flip: close-then-open ordering asserted by index
    - position_id close targets one lot; absent closes all lots
    - no-signal symbol produces zero intents (HOLD)
    - same-side signal HOLDs; empty owned closes nothing; a foreign-only book
      HOLDs the opposite open instead of doubling gross exposure
    - unsized open raises ValueError
    - determinism: shuffled lot order -> same intents
signals tests
  → live_signals(...) with a stubbed ScreenRun                        --> tests/test_signals.py
    - only long/short/close mapped; flat dropped
    - field-for-field carry (qty/price/sl/tp/position_id/tag)
portfolio_source tests
  → load_mock_portfolio(raw, as_of): PortfolioSnapshot                --> tests/test_portfolio_source.py
    - builds a real PortfolioState; Position.type side; empty positions
broker tests
  → SimulatedBroker.place_cohort(intents): Result[tuple[OrderResult,...]]
    - synthetic fill price = ref_price ± spread/slippage
    - apply_fills updates the held PortfolioState (parity with backtest)
    - multi-open cohort SCALES like a backtest cohort (no reject-tail);
      a lone open is bit-identical to a bare apply_fill
    - a flip open sized by reconcile survives settlement
engine tests
  → run_cycle with fake source + fake broker: CycleReport             --> tests/test_engine.py
    - end-to-end: mock pf + signals -> placed intents, order + count
    - open fill -> record_open; close fill -> mark_closed
    - dry run places nothing AND writes nothing (no record, no cycle stamp)
    - a foreign lot survives a cycle untouched; a changed config value
      re-scopes ownership so a prior revision's lot is never closed
ledger tests
  → SqliteLedger over a temp db: record_open/mark_closed/prune          --> tests/test_ledger.py
    - config_hash stable across key order; differs on value change
    - open_positions returns only status='open'
    - mark_closed stamps closed_at, retains row; prune_closed deletes it
    - unknown strategy_id -> empty; no cross-strategy leakage
```

---

## 5. Verification (plain `assert`, no framework)

Drop as `src/live/tests/test_live.py`, run `uv run pytest src/live/tests/`.

Helper (shared): `pf(cash, *lots)` -> a real `PortfolioState` with
`Position(symbol, qty>0, entry_price, ..., type=ActionType.long/short)`.

1. **Reconcile — open**
   ```python
   def test_flat_to_long_opens():
       book = pf(100_000.0)
       sig = LiveSignal("AAPL", "long", 1.0, ("breakout",), TS, price=100.0, qty=10.0)
       (o,) = reconcile((sig,), book, CFG)
       assert o.action is ActionType.long and o.qty == 10.0
   ```
2. **Reconcile — close (carries the lot's position_id)**
   ```python
   def test_long_to_close_closes():
       book = pf(100_000.0, lot("AAPL", 10.0, 90.0, ActionType.long, pid="L1"))
       sig = LiveSignal("AAPL", "close", 1.0, ("exit",), TS, price=100.0)
       (o,) = reconcile((sig,), book, CFG)
       assert o.action is ActionType.close and o.qty == 10.0
       assert o.position_id == "L1"   # _close_position requires this
   ```
3. **Reconcile — no signal holds**
   ```python
   def test_absent_signal_holds():
       book = pf(100_000.0, lot("AAPL", 10.0, 90.0, ActionType.long, pid="L1"))
       assert reconcile((), book, CFG) == ()
   ```
4. **Flip ordering**
   ```python
   def test_flip_closes_before_open():
       book = pf(100_000.0, lot("AAPL", 5.0, 100.0, ActionType.short, pid="S1"))
       sig = LiveSignal("AAPL", "long", 1.0, (), TS, price=100.0, qty=3.0)
       a, b = reconcile((sig,), book, CFG)
       assert a.action is ActionType.close and b.action is ActionType.long
   ```
5. **Unsized open raises**
   ```python
   def test_unsized_open_raises():
       with pytest.raises(ValueError):
           reconcile((LiveSignal("AAPL","long",1.0,(),TS,price=100.0,qty=0.0),),
                     pf(0.0), CFG)
   ```
6. **Mock PF parse** —
   `load_mock_portfolio({"cash":1.0,"positions":[{"symbol":"AAPL","qty":2,"type":"short","entry_price":5,"position_id":"L1"}]}, TS)`
   returns a `PortfolioSnapshot` whose `.portfolio` is a real `PortfolioState`
   holding one `Position(qty=2.0, type=ActionType.short)`.
7. **Simulated fill parity** — `SimulatedBroker.place` on a `long` intent whose
   `ref_price==100.0` with default `ExecutionParams` produces the SAME fill an
   equivalent `TradeSignal` gets from `execute_signal` (adverse spread/slippage),
   and folding it through `apply_fill` yields the expected `PortfolioState`
   (`cash` down by `qty*price + commission`, one new lot).
8. **Screen→signal mapping** — a `ScreenRow(action="close", qty=7.0)` maps to
   `LiveSignal(action="close", qty=7.0)`; a `flat` row maps to nothing.
9. **Data-freshness guard** — an empty/stale DB for the universe makes
   `run_cycle` return an `Err`/raise instead of placing orders.
10. **Interface-shared check** — `reconcile`/`size_qty` accept a `PortfolioState`
    built by `create_initial_backtest_state(...)` unchanged (structural
    `PortfolioView` conformance).
11. **Ledger mark-closed, not delete** — `mark_closed` leaves the row present
    (assert `open_positions` excludes it but the row still exists with
    `status='closed'` and a non-null `closed_at`).
12. **Config-hash scoping** — `config_hash` is order-insensitive over dict keys;
    two runs with the same config reuse one `strategy_id`; a value change yields
    a different id (no lot adoption).

---

## 6. Integration notes (read before implementing)

- **Screen output is authoritative.** `ScreenRow.action` is already
  `Literal["long","short","close","flat"]` and `-F json` emits only the
  actionable three. Live must respect that split: `flat` never becomes an order
  and never appears in the JSON document.
- **Sizing ownership.** The DSL converts a 0–1 fraction to absolute shares on
  the emitted signal when it can (`qty > 0`). The live layer sizes only when
  `qty == 0.0`, and only via `SizingParams` (§Open Decision 1). Do not
  re-implement sizing in `reconcile.py`.
- **Ref price.** Use the signal's `price` (the decision bar close). Real intraday
  execution pricing is a broker concern; simulated fills must reuse
  `execute_signal` so paper matches the backtest.
- **Data freshness is checked, not trusted.** A cron `data dl` failure must not
  let the cycle trade on a stale tail. Compare the market-data newest bar to
  now; one base interval of slack (§Open Decision 5).
- **Prices stored uppercase / bars resampled on read.** `query_candles` filters
  `ticker = UPPER(...)` and resamples 1h → 1d on read (`src/data/resample.py`).
- **Immutable state.** Reconcile returns new tuples; never mutate a
  `PortfolioState`/`LiveConfig`. Book updates go through `apply_fill` only.
- **One portfolio interface.** Live imports `PortfolioState`/`Position` from
  `src/bt/state` and reads them through the `PortfolioView` Protocol. Never
  define a parallel live-portfolio type — that is the divergence this doc
  exists to prevent. If a live-only field is ever needed, add it to the
  Protocol, not a fork.
- **Ledger is live-only and owned by `src/live`.** The backtest never touches
  it; the DB stays a pure input for `bt`. Ledger writes are post-execution:
  record an open only after the broker confirms a lot id; mark a close only
  after the broker confirms the close. A rejected order records nothing, so the
  next cycle recomputes the same intent.
- **The broker id IS `position_id`.** No local pid minting for live — the
  `SimulatedBroker` returns a synthetic lot id (mirror the backtest scheme,
  e.g. `f"{symbol}_{ts.timestamp()}"`), and a real `IBKRBroker` returns the
  broker's handle. `reconcile` then closes lots by that id.
- **Ownership scoping.** Every ledger read/write is scoped to `strategy_id`
  (config hash). A cycle must never close a lot it did not open — the broker
  account may hold lots from other configs. `reconcile` sees the whole broker
  book (for mark-to-market), but close intents are emitted only for lots in
  `ledger.open_positions(strategy_id)`.
- **JSON at the edge.** Use `_json_default` (`src/bt/cmds/_shared.py`) for
  pandas/Enum values — same encoder the other CLI commands use.

---

## 7. AGENTS.md compliance checklist

- [ ] Full type annotations on every function/dataclass. No `Any` without a
      `# comment:` reason.
- [ ] `@dataclass(frozen=True)` for all state; `Protocol` for injection
      (`PortfolioSource`, `Broker`, `Ledger`).
- [ ] No mutation — pure `reconcile`/`signals`/`build_report`/`config_hash`;
      side effects only in `portfolio_source.py` / `broker.py` / `ledger.py` /
      `cli.py`.
- [ ] Functions ≤ 50 LOC, classes ≤ 150 LOC.
- [ ] `Result[Ok, Err]` at I/O boundaries instead of exceptions for expected
      errors (auth/transport/stale data).
- [ ] Reuse `src.data.db` (`get_connection` — one sqlite path), `src.bt.screen`,
      `src.bt.execution/pure`, `src.bt.portfolio/pure`, `src.bt.size/pure` —
      never re-derive. Ledger tables live in the SAME db, created idempotently.
- [ ] New logic has assertion-based tests (§5). `make check` passes, < 10s.
- [ ] No new dependencies.
- [ ] The screen tests (`src/bt/screen/tests/`) stay green — live consumes their
      output shape.

---

## 8. YAGNI (explicitly NOT built)

Real IBKR order routing; real IBKR portfolio fetch; raw `/ws` socket feed;
intraday bar aggregation (data comes pre-aggregated from cron); position
resizing/rebalancing; **restart recovery from the ledger** (the broker is the
source of truth and is re-fetched each cycle — the ledger is an ownership/audit
record, not a resumable portfolio); multi-broker adapters; order-book depth;
real-time dashboards; latency metrics; new dependencies. Each is a separate
follow-on.

---

## 9. Open decisions to confirm with the user before finalizing

1. **Sizing** — explicit `sig.qty` required, else `SizingParams` from config,
   else hard-error? (default: yes)
2. **Reconcile granularity** — side-only HOLD (default) vs value resizing?
3. **Missing signal** — HOLD (default) vs flatten?
4. **Execution** — simulated (default) vs real IBKR routing in v1?
5. **Freshness gate** — fail-on-stale (default) vs trade-anyway?

Confirm these and the plan is fully implementable as written.

---

## 10. Implementation addendum (doc-vs-reality, recorded after implementation)

Where the built code differs from this doc, and the defaults actually wired:

- **No `ScreenRun`.** The driver returns `(rows, final_state)`, not an object.
- **`ScreenRow`** carries only `symbol`, `action` (`∈ long|short|flat`),
  `score`, `signals`, `ts`, `sig_ts` — there is no `"close"` in the vocabulary.
- **The driver's `max_age_days` filter deletes the flat rows** close must be
  derived from, so live calls it with `max_age_days=None` and filters locally.
  Closes are reconstructed from `flat` rows that carry a `sig_ts` (a flat row
  with no `sig_ts` never signalled -> HOLD).
- **`SizingParams.sizing_mode`, not `size_mode`** (the live config accepts flat
  `size_mode`/`sizing` as the public spelling and maps it).
- **`equity_of` takes the `PortfolioView` Protocol** (widened in the
  shared-execution pass), so `reconcile` reuses the ONE equity implementation
  instead of re-deriving `cash + calculate_positions_value(positions)`.
- **`execute_signal` takes a `Candle`** (the broker builds a synthetic
  ref-price bar: all OHLCV = `intent.ref_price`).
- **`Broker` gained `seed(portfolio)`** — the engine aligns the simulated book
  with the freshly fetched snapshot before reconciling/settling it.
- **`PortfolioView` uses `@property` members** and now lives in
  `src/bt/state/types.py` (re-exported by `src/live/types.py`), so the shared bt
  layer never imports the live package; a frozen `PortfolioState` still conforms
  structurally with no cast.
- **Live-only top-level config keys** (`portfolio_path`, `mode`, sizing) are
  projected through a **temp strategy-only file** for the screen bridge, because
  `load_strategy` = `StrategyConfig(**data)` rejects unknown keys. `strategy_id`
  stays the hash of the ORIGINAL raw config (temp path never affects scope).
- **`SimulatedBroker` settles through the shared `apply_fills`** (via
  `place_cohort`); the cash guard lives inside it as `describe_open_rejection`, so
  a dropped open is never recorded as a phantom fill and genuine exhaustion is
  reported from the shared `FillRejection` records.

Implemented defaults (the §9 open decisions, as built):

1. **Sizing** — explicit `qty` else `SizingParams` from config else `ValueError`.
2. **Reconcile** — side-only HOLD (a side flip closes then reopens).
3. **Missing signal** — HOLD.
4. **Execution** — `SimulatedBroker` only (no real IBKR routing).
5. **Freshness** — fail the cycle on a stale feed.

---

## 11. Known gaps (post-implementation review, 2026-10-02)

Defects found by an independent review of `origin/main..HEAD` and their status.
Everything below is **open** unless marked fixed; each gap names the module and
what would close it. The open list was renumbered by the 2026-10-03
shared-execution pass, which moved the cohort-parity and close-proceeds gaps into
the fixed section below.

### Fixed during review

- **Sizing double-count on a close+open cycle** (`src/live/reconcile.py`,
  originally `_sizing_view`). Close proceeds were added to `cash` while the
  closing lots stayed in `positions`, so `calculate_positions_value` counted them
  again and equity was inflated by the closed notional — a flip sized up to
  `(E+V)/E` too large. Fixed once, then superseded by the shared-execution pass
  below: `_settled_book` prices the closes with `execute_signal` and settles them
  through `apply_fills`, so the closing lot is dropped by the very code path that
  produces the proceeds. Regression test pins the exact qty
  (`test_flip_sizes_open_against_freed_cash == 4.9965`, was 5.0, was 10.0).
- **`--dry-run` was not write-free**: `ledger.touch_cycle` still stamped
  `last_cycle_at`. Now guarded by `if not dry_run` (`src/live/engine.py`).
- **Freshness gate used raw symbols** while `query_candles` filters
  `ticker = UPPER(...)`; a lowercase config symbol produced a bogus
  `StaleDataError("no data for universe")`. Symbols are uppercased
  (`src/live/engine.py::_newest_ms`).
- **Freshness clock skew**: `pd.Timestamp.now()` is naive local while the
  candle `timestamp` is UTC epoch-ms; age was off by the host's UTC offset. A
  single UTC base is now used (`src/live/engine.py::_utc`); `now` injection
  accepts naive or aware.
- **Foreign-only book on a side flip** opened the opposite side without closing
  anything (gross exposure doubled, target posture never reached). Now that
  open is skipped (HOLD) and the asymmetry is documented in `reconcile`.

### Fixed after the review — shared-execution pass (2026-10-03)

The accidental divergences between the bt execution/sizing core and the live
edge. Live reuses the shared unit wherever one exists; the ownership ledger stays
the only intended divergence. Verification: `make check` green at 678 passed.

- **Cohort settlement (was gap 2).** `run_cycle` settled intents one at a time
  through `apply_fill`, so an over-subscribed multi-open cycle REJECTED its tail
  while a backtest cohort SCALES. `Broker` gained `place_cohort`; the simulated
  book prices orders per-order (`execute_signal`) and settles the cycle ONCE
  through the shared `apply_fills`, so the settled book equals the backtest's on
  the same fills. `place` is the single-order wrapper (cohort of one, which hits
  `apply_fills`' lone-open guard, so a lone open is numerically unchanged).
  Trade-off: a cohort-level transport `Err` now skips the whole cycle rather than
  one intent — the next cycle recomputes the same intents.
- **Duplicated equity rebuild.** `equity_of` accepts the `PortfolioView`
  Protocol, so `reconcile` calls it instead of re-deriving
  `cash + calculate_positions_value(positions)`.
- **Duplicated close-proceeds math (was gap 5).** `_SizingView`'s
  `Σ qty * ref_price` approximation and its hand-built positions-dropping loop
  are deleted. `_settled_book` derives the cycle's close fills, settles them
  through `apply_fills`, then sizes opens against that real book — one
  implementation of close proceeds, shared with settlement. The flip test's qty
  therefore moved from the approximation to the real proceeds (5.0 -> 4.9965).
- **Duplicated sizing rule.** `_open_intent` calls the shared `sized_signal` for
  the whole "explicit qty else compute it" rule; an open that still sizes to
  `<= 0` still raises `ValueError` rather than placing an accidental order.
- **One ref-bar / probe-signal factory.** `ref_candle` and `trade_signal` live in
  `src/live/broker.py`; `intent_to_signal` delegates to the latter and
  `reconcile` imports both instead of rebuilding its own pair.
- **Layering.** `PortfolioView` moved to `src/bt/state/types.py` — the shared bt
  layer no longer type-imports the live package (verified: zero `src.live`
  imports under `src/bt`).
- **Execution seam (partial).** `SimulatedBroker` takes an injectable
  `ExecutionHandler` (default `default_execution_handler()`) for
  `execute_signal`. `apply_fills` stays a direct import — `ExecutionHandler` has
  no such field and bt's engine imports it directly too.
- **Epoch-0 probe artifacts.** `_settled_book` re-emits a clean view
  (`trades=()`, `equity_curve=()`), so no `pd.Timestamp(0)` trade can leak
  downstream.

### Open gaps (ordered by exposure)

1. **Freshness slack is a day count, not one base interval.** §1.5 asks for one
   base interval of slack; the CLI defaults to `--max-age 5` days, so a multi-day
   `ibkr data dl` outage on a `1h` feed still passes the gate and trades a stale
   tail. Close it by deriving the budget from `config.bars[0]` (resampled
   interval x a small multiple) or by asserting the newest bar is the latest
   expected session bar. `--max-age 0` disables the gate. **Worse than the doc
   says:** the gate is a universe-wide `MAX(timestamp)`, so one fresh symbol
   clears the whole universe, and the local per-row filter cannot catch the
   difference — a symbol frozen for weeks has a posture age of ~0 against its own
   last bar and trades at that stale close. Close it by requiring EVERY
   `config.symbols` member to be fresh (`GROUP BY ticker`) and by measuring the
   budget against the screen's decision bar (`row.ts`), not the DB max.
2. **Sizing clamp vs fill price (shared layer).** `compute_qty` clamps against
   `ref_price`, but the broker fills at
   `ref * (1 + spread_bps + slippage_bps) + commission` — the 1.5x adverse
   multiplier never applies to a live fill, because the synthetic ref bar has
   `open == close` so `calculate_adverse_selection` returns False; a full-fraction
   open (`size ≈ 1.0`, cash clamp binding) is then rejected by the guard and the
   cycle trades nothing, with no explicit error. Property of
   `src/bt/size/pure.py`, surfaced by live. Close it by clamping on a padded
   price estimate or by sizing against available cash net of the cost model.
3. **Close reconstruction is a heuristic.** A close is inferred from a `flat`
   row that carries a `sig_ts`. A strategy that fired nothing on the newest bar
   yields a `sig_ts` older than `--max-age` -> the row is dropped -> HOLD, so a
   genuinely-desired exit on a stale decision bar is not taken (consistent with
   default #3). `--max-age 0` disables the local filter and lets stale closes
   fire.
4. **Posture vs ownership ambiguity (partly mitigated).** `current_side` nets
   the WHOLE broker book while closes are scoped by `owned` (ledger ids). A
   foreign lot on the same symbol can suppress our open (`target == current` ->
   HOLD). The opposite-side-with-no-owned-close case now HOLDs, but the
   same-side suppression remains. Deliberate; unresolved.
5. **`record_open` upserts**, so re-recording the same broker pid resurrects a
   previously closed row and clears `closed_at`. Deliberate (a broker pid is
   expected unique per lot), but it means a broker that recycles lot ids would
   corrupt the audit trail.
6. **No end-to-end CLI test.** `live_run` -> real screen needs the candle DB and
   a full strategy run; tests cover the seam (config parse, temp projection,
   render, cycle with fake source/broker) but not a real `ibkr live run`.
   Manual smoke only: `ibkr live --help`, `ibkr live run --help`.
7. **Freshness/infra gaps outside the module.** The worktree needed the
   `../ib-rest-api-client` path dependency and a `uv`-usable `.venv`; resolved
   outside the committed tree. Pre-existing untracked `data/data` symlink left
   alone.
8. **Params beyond the doc** (testability only, all keyword-only with
    defaults, recorded here): `run_cycle` gained `dry_run`, `db_path`, `now`,
    `signal_source`; `reconcile` gained `owned`.

### Open gaps — independent architecture review (2026-10-03)

Found by a read-only architecture review of the live cycle (sync / screen /
reconcile). Ranked by expected loss; each names the smallest change that closes
it. All are **open**.

9. **Live cannot honour the strategy's exit semantics (largest loss, silent).**
   `_to_live_signal` drops `sl`/`tp` (`signals.py`), no module under `src/live`
   imports `check_risk`/`RiskConfig`, and bt enforces the levels every candle
   (Stage 8). A bracket-bounded strategy is unbounded live. Sharper than that:
   the SCREEN is the same hole, because `_resolve_posture` replays only emitted
   `TradeSignal`s and an engine-side risk exit never is one — so a position the
   screen's own engine already stopped out still reports its original side, and
   live would keep holding it. Close it by carrying `sl`/`tp` onto `LiveSignal`
   and running one `check_risk(book, decision_bar, risk_config)` pass per cycle;
   until then, guard in the bridge: fail loudly when captured signals carry
   levels so nobody runs a bracketed strategy live believing it is bracketed.
10. **Explicit strategy qty is dropped, so live silently re-sizes.** `ScreenRow`
    has no qty field and `LiveSignal.qty` is hard-coded `0.0`, so `sized_signal`
    always applies the live config's `size_mode`/`size`. A risk-sized
    (`sizing_mode: "risk"`) or fixed-size strategy is re-sized to a fraction of
    equity. Not forced — the collector holds the full `TradeSignal`.
11. **`partial_close` / `rebalance` are silently discarded.** `_side_of` returns
    `None` for `rebalance`, so a trim never reaches posture and the live position
    keeps full size until a full close fires (live strategies use it:
    `shannons_demon_dsl`, `pf_equal_weight`).
12. **No idempotency key, no lock, and the sim book never persists.** Every cycle
    re-seeds the broker from the static fixture, `broker.portfolio()` has no
    production caller, `live_strategy.last_cycle_at` is written and never read,
    and there is no `flock`/`BEGIN IMMEDIATE` — so overlapping cron runs both
    place, and N runs on an unchanged fixture pile up unclosable ledger rows.
    Close it with a per-`strategy_id` lock, a decision-bar dedupe, and writing the
    settled book back at cycle end.
13. **Config-hash scoping orphans the live book, with no orphan detection.**
    Every parameter edit yields a new `strategy_id`, instantly making prior lots
    foreign: the new revision opens on top of them or HOLDs forever, and
    `prune_closed` only deletes CLOSED rows. The same hole covers a crash between
    `broker.place_cohort` and `_record`. Close it with an orphan check at cycle
    start (book lots vs `open_positions()` across all scopes) that warns or
    flattens.
14. **`mode: "live"` silently runs the simulated broker.** `cfg.mode` is only
    persisted; the CLI always builds `SimulatedBroker`, so a config declaring
    live mode exits 0 with paper fills. Refuse the mode until `IBKRBroker` exists.
15. **A scaled-to-zero open is recorded as a phantom lot.** When `apply_fills`
    floors an open to 0 shares nothing is recorded as a `FillRejection`, so
    `_result` falls back to the pre-scale qty with a predicted pid and `_record`
    writes a row for a lot that is not in the book — unclosable, since
    `_close_error` rejects every retry. Narrow (`qty*scale < 1e-4`), 2-line fix:
    treat "lot not found after settlement" as `ok=False`.
16. **Two handoff docs.** The committed root `HANDOFF_LIVE_TRADING.md` is a
    superseded tick-driven design (`feed.py`, `baragg.py`, `/ws` poller); the live
    one is `docs/HANDOFF_LIVE_TRADING.md`. Delete or redirect the root file.

Unverified assumption behind the ownership model: that a real broker read can
supply per-LOT ids. IBKR positions are net per instrument (conid identifies the
instrument), so either the adapter mints lot ids — making the adapter, not the
broker, the lot-id source of truth — or lot identity has to move into the ledger.
Check the real positions payload before `IBKRBroker`.

Deliberately NOT defects: batch-not-loop, side-only HOLD, absence = HOLD,
ownership scoping, `Broker.seed`, and `fill_at_next_open=False` (forced — a batch
has no next bar to price; note it means live paper fills are not trade-for-trade
price-comparable to a `bt run` of the same strategy).

### Deliberately unchanged

- Other `pd.Timestamp.now()` sites (`broker.py`, `cli.py`,
  `portfolio_source.py`) stay naive-local: they are never compared against the
  UTC epoch-ms candle DB, so there is no offset-skew defect there.
- `make check` coverage: `Makefile` `SRC_DIRS` now includes `src/live`, so lint,
  format, typecheck and `test-fast` all cover the package (commit `3f8908b`).
