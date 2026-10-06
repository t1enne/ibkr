# Plan — Unified execution core, IBKR as first real broker (rev 4.1)

Rev 4.1 supersedes rev 4's §3 reconciliation model. Rev 4 tried to derive each
strategy's book from the ACCOUNT net minus foreign executions; that is wrong for the
case this system actually runs — **two or more strategies on one account**. An account
net cannot be split between strategies, and with no in-window execution of ours it reads
a manual position as ours (closable). Rev 4.1 attributes by OUR OWN cOIDs and keeps the
book in sqlite.

- **Attribution is per scope, from our executions.** One `scope` = one strategy; a
  strategy's book is exactly the executions whose `order_ref` resolves to its scope.
- **sqlite is the store; the trades endpoint is a confirmation channel.** `/iserver/account/trades`
  is a 7-day window, so a lot opened last month exists because the local book says so.
- **Positions the account holds that no scope owns are ignored by design** — a human may
  hold anything; we neither reconcile nor trade it.

Rev 4's two surviving changes (from rev 3) still hold, and both were forced by live
paper evidence in phase 3 (see §3/§4 and §6 `Phase 3.5`):

- **Lot identity is `(scope, conid)`, not `order_id`.** A close is a new order with a new
  `order_id`, so rev 3's "one lot per `order_id`" could never see a lot close.
- **Ownership scope is a stable `scope`, not `strategy_id[:8]`.** A config *edit* changes
  the config hash, so rev 3 silently orphaned every open lot on a tuning change.

Everything else in rev 3 stands, including the phase-split and the surviving open
decisions.

## Shipped status vs this plan (2026-10-05)

Phases shipped and verified (all on `feat/live`), in order:

| Phase | Commit | Status |
| --- | --- | --- |
| 0 shared order domain | `a3d4e2b` | done — matcher/refs/friction table-tested |
| 1 backtest on `SimExchange` | `e7bfb9a` | done — golden parity 48/48 byte-identical |
| 1.5 wire the seam, one `Fill.price` | `2e63c34` | done |
| 2 IBKR read path | `67f38b7` | done — cash matched live; positions mismatch is the rev-4.1 ignition |
| 2.5 fixes + compose keepalive | `eae15a3` | done |
| 3 IBKR MKT placement | `2167c01` | done — open/close paper fills carry our cOID |
| 3.5 per-scope sqlite book | `e3a6feb` `b599b92` | done — two scopes on one account proved live |
| peewee `SqliteLedger` | `5bab39b` | done — no behavior change; lazy DDL |
| `db` test marker | `6efdf05` | done — `test-fast` excludes sqlite tests |

**Deviations from the plan text, each deliberate, with the reason:**

- **D1 The engine's fill seam is a bt-local `FillSurface`**, not a third port in
  `src/exec/ports.py`. The seam is bt-shaped (`TradeSignal`/`Candle`/`FillEvent`/
  `PortfolioState`), so it lives in `src/bt/exchange/ports.py`; `src/exec/ports.py`
  keeps exactly the broker-agnostic `Broker` + `Exchange`. Both are satisfied by
  `SimExchange` / the pure matcher (conformance-tested, `ty`-checked).
- **D2 Two broker protocols exist and are not converged**: the shared sync `Broker`
  (`src/exec/ports.py`) and the live async `LiveBroker` (`src/live/broker.py`). Rev 3's
  "ONE `Broker` port" is effectively two. This is an explicit phase-4 decision (see §6),
  deferred because it is only load-bearing once LMT/`OrderRequest` must reach the live
  edge.
- **D3 The keepalive lives in docker-compose**, as the user directed: the `ib-gateway`
  healthcheck does `curl -fsk /tickle | grep -q '"authenticated":true'` every 30s — the
  GET tickle IS the keepalive, and a logged-out session now reads UNHEALTHY instead of
  healthy. The in-app `ensure_ready` stays auth-status-only and never auto-logs-in.
- **D4 The sqlite book is peewee ORM** (5 peewee models, `CompositeKey` for
  `(scope, conid)` / `(scope, execution_id)` / `(strategy_id, position_id)`, explicit
  `Meta.table_name`), with **lazy DDL**: tables created only on a real write, so a
  `--dry-run` writes no schema.
- **D5 Sim close-scoping was RESTORED, not deleted.** Rev 4.1's "delete the engine's
  `owns_book`/`owned=None` hack" is superseded: the sim/mock source keeps a ledger
  ownership filter (`owns_book` + `live_sim_lot` — a mock-fixture lot the strategy never
  opened must not be closed), while the ibkr source needs none because its book is ours
  by construction. Two different ownership surfaces, each correct for its source.
- **D6 New safety invariant (not in rev 3/4.1): never send a REDUCING order for a conid
  the account is flat on.** Adds to plan §3's invariants; unit-tested
  (`test_close_on_a_flat_account_is_refused_before_submitting`).
- **D7 `status_to_fill` reads `cum_fill`/`average_price`, never `size`** (a filled order
  reports `size 0.0` because that field is remaining). Rev 3/4.1's "must read `cum_fill`"
  is thereby already resolved.
- **D8 The phase-2 proof matched CASH live but not POSITIONS**: the replay had no lots
  for the two manual holdings because the trades window had none of our executions. This
  exact observation is why rev 4.1 §3 ignores positions no scope owns.
- **D9 Ref scheme shipped as `{slug(scope)}-{cycle_ts:%Y%m%dT%H%M%S}-{seq}`** with
  identity-keyed `seq` (cOID `alpha-20261005T182108-10201` observed live).
- **D10 Project tooling:** sqlite-touching tests are marked `db` and excluded from
  `make test-fast`; `make test` runs the full suite.

Rev 3 reflects review direction: (a) the backtest adopts the live architecture —
one `Broker` port, MKT/LMT orders — instead of live mimicking the backtest;
(b) reconciliation comes from IBKR trade history, not a local ledger's claims;
(c) phase 1 stays small.

Rev 2 (`docs/PLAN_LIVE_HEXAGONAL.md`) is superseded. Pros/cons for the surviving
questions stay in `docs/OPEN_QUESTIONS_PROS_CONS.md`.

## 0. Direction taken

| Rev 2 problem | Why it existed | Rev 3 resolution |
| --- | --- | --- |
| Fill parity broken (cohort scaling, aggregated positions) | `apply_fills` cohort scaling and instant fills are properties of the backtest's own execution model; live could only imitate them | Both sides implement ONE `Broker` port. Backtest gets a `SimExchange` adapter; live gets `IbkrBroker`. Cohort scaling becomes a property of `SimExchange`, not of the shared core |
| No LMT anywhere | Backtest fills instantly at next open with friction — an order type has no meaning | `OrderType.MKT\|LMT` joins the shared domain; a pure candle matcher decides LMT fills in the backtest, IBKR decides them live |
| Lot truth in a local ledger, drift checks, grouping, `closed_position_ids` | Assumed the broker read is the only external truth | `/iserver/account/trades` returns per-execution `order_ref` (the cOID we set), `order_id`, `size`, `price`, `commission`, `trade_time_r` — lots are **replayed from executions**, so no drift class exists |
| 14 open questions, some inventing complexity | Each adapter invented its own book/identity model | Order metadata (SL/TP/tag) stays in a small local table; qty, entry price, commission and open/closed state come from executions. Q1/2/3/12/13 collapse |

## 1. The shared execution architecture

```
TradeSignal (strategy output, unchanged)
    │  engine maps signal -> OrderRequest, one per lot
    ▼
OrderRequest  {symbol, side, qty, order_type: MKT|LMT, limit_price, tif, order_ref, tag}
    │
    ▼
Broker port  (submit / cancel / fills / close)
    ├── SimExchange   (bt)   — candle matcher + friction + cohort cash scale
    └── IbkrBroker    (live) — IBKR routing, real fills
```

Consequences:

- `ExecutionHandler` (`src/bt/engine/handlers.py`, callables typed `Any`) is deleted.
  The engine composes a `Broker`; every seam is a typed Protocol.
- `src/execution/pure.py`'s `execute_signal` / `apply_friction` / `commission_for_fill`
  move under the shared core and become `SimExchange` internals — the matcher, not a
  free function the engine calls directly.
- Live and backtest share the order vocabulary, the `order_ref` scheme, the fill
  type, and the Ledger's metadata contract. They differ only in the adapter.
- A strategy JSON names its broker (`"broker": "sim" | "ibkr"`, default `"sim"`), so
  one file describes a strategy for both research and live.

### What stays backtest-only

`apply_fills` cohort cash scaling, `Scaled`/`Rejected` reporting, `fill_at_next_open`,
`fill_guard_price` gap-through math — all remain, but behind `SimExchange`. IBKR relies
on broker-side buying power. The report columns stay; they describe the sim exchange.

## 2. Shared core layout

```
src/exec/                     NEW — shared by bt and live, pure
  types.py       OrderType(MKT|LMT), TimeInForce(DAY|IOC), OrderSide, OrderRequest,
                 OrderAck, OrderState, Fill, RejectReason
  ports.py       Broker Protocol, Exchange Protocol
  matching.py    match_bar(order, candles) -> Fill | None   (pure candle matcher)
  refs.py        order_ref(strategy_id, cycle_ts, seq) -> str   (pure, deterministic)
  friction.py    apply_friction, commission_for_fill  (moved from src/bt/execution/pure.py)

src/bt/exchange/              NEW — the backtest broker adapter
  sim.py         SimExchange(Broker)  — matcher + friction + cohort scale, legacy-identical
  cohort.py      apply_fills cohort scaling (moved out of portfolio/pure.py when safe)

src/live/                     (rev 2 layout, minus the book/ledger-of-truth complexity)
  ports.py       PortfolioSource, Ledger (metadata only), SignalSource, MarketData, Gateway
  orders.py      TradeSignal <-> OrderRequest mapping, side resolution
  reconcile.py   keep — pure posture diff
  engine.py      run_cycle, I/O via ports only
  app.py         CompositionRoot + build_cycle + adapter registry
  cli.py         parse -> build_cycle -> run -> render
  adapters/
    sim/         MockPortfolioSource, SimLedger
    sqlite/      SqliteLedger (order metadata), SqliteMarketData
    screen.py    SignalSource impl
    ibkr/        client, orders, trades, mapping, portfolio_source, broker, gateway, authz
```

`src/exec/refs.py` is the single definition of the order-ref scheme, used by
`SimExchange` (as a stable id in tests) and `IbkrBroker` (as `cOID`), so both sides
label orders identically.

### The matcher (where MKT/LMT parity is decided)

```python
def match_bar(order: OrderRequest, bars: pd.DataFrame) -> Fill | None
```

- MKT: fills at the next bar's open (`tif=DAY` at that bar). Same rule as today.
- LMT buy: fills only if `low <= limit`; price = `min(limit, open)` (a gapped-through
  open is not improved on). LMT sell: `high >= limit`; price = `max(limit, open)`.
- No fill: `None`, and the backtest reports it as an unfilled order rather than
  pretending an instant fill. `tif=DAY` drops it at that bar; carry-over is phase 4.
- Friction (`spread_bps`, `slippage_bps`, adverse multiplier) is NOT the matcher's
  job: `match_bar` returns the frictionless base price, and the ADAPTER applies
  friction + commission. `SimExchange.match_bar(order, bars, *, spread_bps,
  slippage_bps, commission_model)` therefore requires those args (no silent
  default) and returns a `Fill` whose `price` is ALWAYS the executed price with
  `commission`/`spread`/`slippage` costs populated — one meaning for `Fill.price`
  across the matcher and the `execute_signal` path. It reuses the same adverse
  rule (`calculate_adverse_selection`, `adverse_multiplier=1.5`) so the two paths
  cannot disagree. Because it adds required friction args, `SimExchange.match_bar`
  is deliberately NOT structurally the pure `Exchange` port.

This is the only place where a candle-based decision can honestly differ from a real
matching engine, so it is pure, table-tested, and shared with the live dry-run path.

## 3. Reconciliation from the broker's own book

Truth is layered, and each layer answers exactly one question:

| Layer | Source | Answers |
| --- | --- | --- |
| Quantity + state | `GET /portfolio/{acct}/positions/{page}` — `conid`, `position`, `avgCost`, `mktPrice`, `currency` | what we hold, and whether we are flat |
| Price/cost/attribution | `GET /iserver/account/trades` (7-day window) — per execution `execution_id`, `conid`, `order_id`, `order_ref`, `side`, `size`, `price`, `commission`, `net_amount`, `trade_time_r`, `position` | at what price and cost, and which part of the book was ours |
| In-cycle lifecycle | `GET /iserver/account/order/status/{orderId}` — `order_status`, `cum_fill` (**filled**), `average_price`, `total_size`, `size` (**remaining**, not filled) | did this cycle's order fill |
| Metadata | local `LivePosition` table | `stop_loss`, `take_profit`, `tag`, `scope`, `order_ref`. Never qty, price or state |

```python
@dataclass(frozen=True)
class Execution:
    execution_id: str; order_id: str; order_ref: str; conid: int
    symbol: str; side: OrderSide; qty: float; price: float
    commission: float; ts: pd.Timestamp

def reconcile(
    scope: str,
    executions: tuple[Execution, ...],   # the trades window, already filtered to us
    book: StrategyBook,                   # the durable rows for this scope
) -> tuple[StrategyBook, tuple[str, ...]]   # advanced book + warnings
```

- **Attribution is the cOID scope, nothing else.** `order_ref` resolves to a scope
  (§4); a strategy's book is exactly the executions whose ref is its own. Two strategies
  on one account never see each other's fills, and neither reads the account net.
- **The account summary and positions are informational only.** N strategies share one
  account's cash and one positions endpoint, so neither can be a per-strategy truth. Cash
  and equity for a scope are derived from that scope's own executions (its config
  `initial_capital` advanced by its fills and commissions). `/portfolio/{acct}/summary`
  is displayed for context, never used as a book input.
- **sqlite is the store** (`data/db.sqlite`, alongside the existing ledger tables): a row
  per `(scope, conid)` carrying qty, side, `entry_price`, `opened_at`, `closed_at`, the
  metadata columns (`stop_loss`, `take_profit`, `tag`, `order_ref`) and a watermark; plus a
  row per applied `execution_id` so applying the window twice is a no-op. The 7-day window
  is a confirmation channel that advances the book; it is never the reason a lot exists or
  stops existing.
- **Applying an unseen execution advances the book**: open / add / reduce / close per side
  and sign, `entry_price` = VWAP of the opening executions of the current open interval.
  qty reaching 0 stamps `closed_at`, so a round trip leaves ONE row closed — never two
  open lots. A same-sign re-entry after flat reopens the row with a new `opened_at` (and
  reports it), and a flip is reported rather than merged.
- **An execution we cannot apply** (a reduction with nothing to reduce, a conid whose
  symbol does not match the stored row) is a warning carrying the raw execution — never a
  silent drop, and never a reason to place an order.
- **A position no scope owns is ignored by design.** The account may hold anything a human
  did: it is not our book, not a mismatch, not an error, and never traded. Nothing is
  absorbed into a scope by accident, and nothing needs a "foreign" entity.
- **A manual trade on a symbol we hold is not absorbed** (its ref is not ours). If it
  closes our lot, the local book and the account simply disagree; that is reported for
  context and changes no intent, because we act on our own book only.

**Stated limits, not hidden ones.** `/iserver/account/trades` covers 7 days: the window
confirms what we can see, and the local book carries what we cannot. A cross-currency book
(an EUR account holding USD stock) is carried exactly as the broker reports it, with no FX
conversion, until a phase takes that on. Cash and equity are per scope and no longer read
from the account summary.

This deletes from rev 2: `join_book` and its six drift rules, close grouping,
`OrderResult.closed_position_ids`, the cost-tolerance knob, and the tiered repair policy.
It also deletes rev 3's `order_id`-keyed lot and `strategy_id[:8]` prefix scope, rev 4's
`account_net − foreign_net` attribution (an account net cannot be split across
strategies), and rev 4's EXTERNAL-entity requirement.

## 4. Order identity and re-run safety

```python
def order_ref(scope: str, cycle_ts: pd.Timestamp, seq: int) -> str
# f"{slug(scope)}-{cycle_ts:%Y%m%dT%H%M%S}-{seq:03d}"
```

`scope` is a **stable strategy identity that survives a config edit**: the live
config's `scope` key, defaulting to `StrategyConfig.name`. It is also the attribution key
(§3): the ref's `slug(scope)` prefix is how a strategy recognises its own executions, so
it must be present on every ref and unique per strategy on an account. Ownership is
decided on the whole slug, never on a slice of a hash. Two consequences of rev 3's
`strategy_id[:8]` scope, both observed in the phase-3 paper run:

- Two strategies whose config hashes share 8 characters share each other's lots —
  proven by construction (a deliberate prefix collision was needed to make the close
  cycle see the open cycle's lot at all).
- **Any parameter edit changes the config hash, so the new revision stops owning the old
  revision's open lots.** In the cron flow a tuning change silently orphans a live
  position into the "foreign, reported, never traded" bucket, where it can never be
  closed. The phase-3 ledger row was left `open` for exactly this reason.

The config hash survives ONLY as an audit column (`strategy_id` on ledger rows: which
revision placed what). It is never an ownership filter.

`slug(scope)` keeps the ref short and safe as a broker client id, and the prefix is what
makes attribution possible on a shared account. `cycle_ts` is second-granular so two
different cycles can never share a timestamp, and `seq` is the intent index within the
cycle with deterministic reconcile order (config symbol order, closes first). Re-running
the same cycle re-sends the same refs and IBKR dedupes; a different cycle gets different
refs, so a legitimate re-entry is never blocked. No hashing of price or qty, so a size
change does not silently defeat the dedupe. **Open, must be decided in phase 3.5:** a
re-run whose intent SET differs from the original (the close filled, so the open moved
to seq 0) can mint a ref the broker already saw and dedupe a genuinely new order away;
the fix is to key `seq` on the intent's stable identity (symbol + action + lot), not on
its position in the batch.

## 5. Gateway lifecycle

Unchanged from rev 2, and phase-split: `src/data/ibkr/gateway.py` +
`src/data/ibkr/login.py` hold the moved logic; `IbkrGateway` (port `Gateway`) is the
live adapter; `sync_market_data.py` and `scripts/login_ibkr.py` become thin shims over
the same implementation.

Mode guard (`authz.py`) also stands: `paper` config against a live account hard-fails;
`mode: "live"` with a live account requires `--allow-live`; a dry run places nothing.

## 6. Phases

Each phase ships alone. Later phases are not prerequisites for earlier ones.

### Phase 0 — shared order domain (no behavior change)

- `src/exec/{types,ports,matching,refs,friction}.py` + tests. Nothing imports it yet.
- Proof: matcher table tests (MKT open fill, LMT touched, LMT gap-through, LMT missed,
  friction direction); `order_ref` determinism; `make check`.

### Phase 1 — backtest runs on the shared core

- `src/bt/exchange/sim.py`: `SimExchange` implements `Broker`, wrapping the matcher,
  friction and the existing cohort scaling.
- Engine composes `SimExchange` instead of `ExecutionHandler`; `handlers.py` deleted.
- `StrategyConfig.broker: str = "sim"` (new optional field, default preserves behavior).
- Proof: **golden-report parity**. Capture `bt run` text+json for every `strats/**/*.json`
  before the change, re-run after, diff. Any metric drift is a bug, not a refactor
  (AGENTS: read full output, name any trim). Plus `bt split`/`sweep` spot checks.
- Non-goals: no LMT strategy JSON yet, no report column changes.

### Phase 2 — IBKR read path

- `src/data/ibkr/client.py` (`IbkrClient`, respx-tested: 401→auth, 429/503→rate_limit).
- Gateway extraction + `IbkrGateway`; `--adapter ibkr`; `ensure_ready` before the cycle.
- `adapters/ibkr/{mapping,trades,portfolio_source}.py`: positions + summary + execution
  replay → `PortfolioState`.
- Proof: `ibkr live run cfg.json --dry-run --adapter ibkr` matches the Gateway UI book
  and places nothing; ledger metadata table populated only in a real cycle.
- Non-goals: no order placement, no LMT, no stops.

### Phase 3 — IBKR order placement (MKT first) — shipped

- `adapters/ibkr/{orders,broker,authz}.py`: one order per intent (opens and closes),
  reply-confirmation loop, `wait_filled` polling, `status_to_fill`.
- Shipped: `orders.py` (pure ticket build + reply classification + `status_to_fill`),
  `IbkrBroker` (`LiveBroker`: seed without touching `apply_fills`, reply loop bounded at
  5, `wait_filled`, dry-run refusal at the broker itself), `FeedError` widened with
  `rejected`/`unfilled`/`timeout`, ledger metadata only.
- Live paper evidence (DUM452563): open `order_id 1469916425` @333.18 and close
  `1469916426` @333.08, both carrying our cOID; a foreign `order_ref` was reported and
  never traded.
- Two bugs the respx ladder could not catch, found only live: `signals._action_of`
  silently dropped every `close` row (no close cycle could ever trade), and the submit
  body needs the `{"orders": [<ticket>]}` envelope or the gateway answers
  `400 Missing orders`.
- Unmet as specified: "the replay reconstructs one lot open→closed". The reply-loop and
  partial/timeout paths are unit-tested only — live submits returned an `order_id`
  immediately. See Phase 3.5 and §3.
- Non-goals: no resting stops, no cancel/modify, no bracket/OCA.

### Phase 3.5 — per-strategy book in sqlite (rev 4.1 §3/§4) — shipped

The correctness follow-up phase 3 proved is needed. No new broker surface, and no order
may be placed from a reconciliation mismatch. SHIPPED (`e3a6feb`, `b599b92`) and
verified live: two scopes on one account each reconcile only their own book (cOIDs
`alpha-…`/`beta-…`), an open→close round trip leaves ONE closed `(scope, conid)` row per
scope, `GS2C`/`GOOG` are ignored with no warnings, and a re-run of the closes is a no-op
(0 intents, cash stable, no new rows). The sqlite store is peewee (deviation D4); lazy
DDL is tested. Bullets below that were shipped are marked ✓; the rest are open for
phase 4.

- **sqlite book, per scope.** `data/db.sqlite` carries a row per `(scope, conid)` (qty,
  side, `entry_price`, `opened_at`, `closed_at`, `stop_loss`, `take_profit`, `tag`,
  `order_ref`, watermark) plus a row per applied `execution_id`. The existing
  `live_position` table changes primary key and gains `conid`; `live_strategy` keeps
  `strategy_id` (the config hash) as an AUDIT column only.
- **`reconcile(scope, executions, book)`** replaces the `order_id`-keyed replay in
  `adapters/ibkr/trades.py`: apply unseen executions to the durable rows (open/add/
  reduce/close/reopen-as-reported), VWAP the open interval, stamp `closed_at` at zero,
  warn on anything unapplicable. A round trip must leave ONE closed row, not two open
  lots.
- **Stop deriving the book from the account.** `portfolio_source.py` keeps the summary
  for display only; `cross_check` and `signed_net` book-building go away, so a position
  no scope owns is neither an error nor an input. `mapping.parse_executions` must parse
  `conid` (and the per-execution `position` for diagnostics).
- **Thread `scope`** from the live config (`LiveConfig.scope`, `StrategyConfig.scope`
  defaulting to `name`) through `cli.py`, `engine.py`, `ledger.py`; `cli._ref_prefix`
  deleted. ✓ Note (deviation D5): the engine's `owns_book`/`owned=None` was NOT deleted
  — the sim source keeps a ledger ownership filter (`live_sim_lot`), only the ibkr
  source passes `owned=None` because its book is ours by construction.
- **Per-scope cash/equity** derived from the scope's own executions (config
  `initial_capital` advanced by its fills/commissions). The account summary is no longer
  a book input; state that in the report so N-strategies-one-account is honest.
- **Fix the ref scheme** per §4: `slug(scope)` + second-granular `cycle_ts` + a `seq`
  keyed on intent identity, with a test for the shifted-batch case. ✓
- **Placement hygiene:** whole-share quantities (round and reject/flag fractional before
  submit — `compute_qty` currently returns 4 dp), and read `/iserver/account/orders`
  before re-sending after an ambiguous submit timeout, so a working order is seen instead
  of duplicated. An LMT/working order is out of scope (phase 4) but the re-send guard is
  not. ✓ (plus deviation D6's close-flat refusal, which is part of this hygiene)
- **Decide `Broker` vs `LiveBroker`** convergence before phase 4 LMT: the shared
  `OrderRequest` vocabulary needs `limit_price`/`tif` on the live edge.
- Proof, on the paper account: two scopes on one account each reconcile only their own
  book; an open→close round trip shows one closed row; a manual/foreign position is
  ignored without a warning storm; a parameter edit keeps the same scope's book; a
  `--dry-run` writes no DDL; `make check` green.

### Phase 4 — deferred (deliberately not phase 1)

LMT carry-over and reprice, cancel/modify, resting per-lot `STP` orders (the rev 2 Q14
decision), alerting for a halted cycle, `/iserver/account/trades` windowing beyond 7 days,
multi-account allocation, IBKR market data.

## 7. Surviving open decisions

1. **LMT in phase 3 or phase 4?** Shipping LMT live means owning unfilled orders across
   cycles. Default: domain-ready in phase 0, live LMT in phase 4 unless a strategy needs it.
2. **`StrategyConfig.broker`** — SHIPPED AS: both the config field (`Literal["sim","ibkr"]`,
   default `"sim"`, unknown value raises) AND the `--adapter` CLI flag exist;
   `resolve_broker` reads the `broker` key once (top-level, then `strategy_params`) so the
   field and the flag cannot disagree, and `mode: "live"` must name its adapter.
3. **`CommissionModel` for live.** Broker-reported commission from executions is exact;
   the sim's model is an approximation. Default: record the broker's number in live and
   make live and backtest reports state which source produced their costs.
4. **`mode` doubles as gateway login toggle and live guard** (`LiveConfig.mode`). Default: reuse it.
5. **Who starts the gateway.** Default: `ensure_ready` in the cycle, flag to disable for cron.
6. **`--allow-live` naming and no env var.** Default: as stated.
7. **`adapter: "auto"`** — SHIPPED AS: there is NO `auto` token. `--adapter` accepts only `sim`/`ibkr`; `mode: "live"` must NAME its adapter (`--adapter` or a `broker` key) and a live run with neither hard-errors rather than defaulting to `sim`. `resolve_broker` reads the `broker` key once (top-level, then `strategy_params`) so the adapter choice and the "was it named?" flag cannot disagree.
8. **New `FeedError` kinds.** SHIPPED AS: `rejected`, `unfilled`, `timeout` added;
   `position_drift` is gone with the ledger-of-truth.
9. **Golden-report parity artifact.** Where does the phase-1 baseline get stored so the
   diff is reproducible (a `tests/fixtures/bt_reports/` snapshot vs a scratch dir)?
   Default: scratch dir in phase 1, committed snapshot only if it stays stable.
10. **Ownership scope key.** Resolved in rev 4.1 §4: the live config's `scope` (default
    `StrategyConfig.name`), not the config hash and never a hash prefix. It is both the
    durable book key and the cOID prefix a strategy uses to recognise its own fills on a
    shared account. A hash-keyed scope orphans live lots on every parameter edit.
11. **Lot identity.** Resolved in rev 4.1 §3: one row per `(scope, conid)`, closed when
    the scope's own qty returns to zero. Per-`order_id` lots cannot represent closure at
    all.
12. **Positions we do not own.** Resolved in rev 4.1 §3: IGNORED BY DESIGN. The account
    may hold anything a human did; a position no scope owns is not our book, not a
    mismatch, never traded. No "foreign" entity, no EXTERNAL reporting.
13. **Per-strategy cash/equity on a shared account.** Resolved in rev 4.1 §3: derived
    from each scope's own executions and its config `initial_capital`. The account summary
    is display-only, because N strategies share one account's cash.
14. **Share granularity.** Resolved in rev 4.1 §6 phase 3.5: orders are whole shares;
    a fractional quantity is rounded and flagged before submit, not sent to the broker.
