# Plan — Unified execution core, IBKR as first real broker (rev 4)

Rev 4 changes exactly two things from rev 3, both forced by live paper evidence from
phase 3 (see §3 and §4 for the observations, and §6 `Phase 3.5` for the follow-up work):

- **Lot identity is `(scope, conid)`, not `order_id`.** A close is a new order with a new
  `order_id`, so rev 3's "one lot per `order_id`" could never see a lot close.
- **Ownership scope is a stable `scope`, not `strategy_id[:8]`.** A config *edit* changes
  the config hash, so rev 3 silently orphaned every open lot on a tuning change.

Everything else in rev 3 stands, including the phase-split and the six surviving
open decisions.

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
    executions: tuple[Execution, ...],
    positions: tuple[BrokerPosition, ...],
    *,
    scope: str,
) -> BrokerSnapshot
```

**One lot per `(scope, conid)` — not per `order_id`.** Rev 3's rule could not express
closure, and the live paper run proved it: the open filled as `order_id 1469916425` and
the close as `1469916426`, so the replay produced *two open lots* on a flat account and
the ledger row for the original stayed `open`. In `(scope, conid)` terms the same
account reads as one lot that opened and then closed, whatever orders were involved.

- `account_net(conid)` comes from the positions read. Fallback, when a position is not
  listed: the last in-window execution's `position`, which the API reports as the
  account's net **after** that execution (observed live: `1 → 2 → 1 → 0` across four
  AAPL fills).
- `foreign_net(conid)` = sum of signed `size` over executions whose `order_ref` is not
  ours; `our_qty(conid) = account_net − foreign_net`. Cross-check it against the sum of
  our own signed `size` and report any mismatch as a warning carrying both numbers and
  the executions — never as a silent abort.
- `our_qty != 0` → one lot: `qty = |our_qty|`, side from the sign, `entry_price` = VWAP of
  our opening executions on that conid, `status` open iff `our_qty != 0`. A sign flip
  inside the window is reported, not silently merged into one lot.
- `our_qty == 0` while `account_net != 0` → the position is **EXTERNAL**: reported loudly
  (symbol, conid, account net, our executions) and never traded. That is the honest
  treatment of a manual holding; rev 3 made the whole position invisible instead.
- Cash from `/portfolio/{acct}/summary`, with the account currency carried rather than
  assumed.

**Stated limits, not hidden ones.** `/iserver/account/trades` covers 7 days, so a lot
held longer has no in-window opening executions: its entry price falls back to the
broker's `avgCost` (the only number available) while attribution above still holds.
Windowing beyond 7 days is phase 4. A cross-currency book (an EUR account holding USD
stock) is carried exactly as the broker reports it, with no FX conversion, until a phase
takes that on.

This deletes from rev 2: `join_book` and its six drift rules, close grouping,
`OrderResult.closed_position_ids`, the cost-tolerance knob, and the tiered repair policy.
It also deletes rev 3's `order_id`-keyed lot, its `strategy_id[:8]` prefix scope, and the
phase-3 constraint paragraph those two implied.

## 4. Order identity and re-run safety

```python
def order_ref(scope: str, cycle_ts: pd.Timestamp, seq: int) -> str
# f"{slug(scope)}-{cycle_ts:%Y%m%dT%H%M}-{seq:03d}"
```

`scope` is a **stable strategy identity that survives a config edit**: the live
config's `scope` key, defaulting to `StrategyConfig.name`. Ownership is decided on the
whole scope, never on a slice of a hash. Two consequences of rev 3's
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

`slug(scope)` keeps the ref short and safe as a broker client id. `seq` is the intent
index within the cycle, and reconcile order is deterministic (config symbol order, closes
first). Re-running the same cycle re-sends the same refs and IBKR dedupes; the next bar
gets fresh refs, so a legitimate re-entry is never blocked. No hashing of price or qty, so
a size change does not silently defeat the dedupe.

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

### Phase 3.5 — reconciliation model (rev 4 §3/§4)

The correctness follow-up phase 3 proved is needed. No new broker surface, no new orders
until the book model is right.

- `adapters/ibkr/trades.py`: replace the `order_id`-keyed replay with
  `reconcile(executions, positions, *, scope) -> BrokerSnapshot` — one lot per
  `(scope, conid)`, the attribution rule (`our_qty = account_net − foreign_net`), the
  `avgCost` fallback, and EXTERNAL reporting for a position we do not own.
- `refs.order_ref(scope, cycle_ts, seq)`; thread `scope` from the live config through
  `cli.py`, `engine.py` and `ledger.py`; key the ledger by `(scope, conid)` and keep
  `strategy_id` as an audit column only.
- `status_to_fill` must read `cum_fill`/`average_price`: a filled order reports
  `size 0.0` because that field is the remaining size (observed live on four orders).
- Proof, on the paper account: an open→close cycle reads as ONE lot going open→closed
  from the broker's net; a manual/foreign holding appears as EXTERNAL and is never
  traded; a parameter edit no longer orphans the lot; `make check` green.

### Phase 4 — deferred (deliberately not phase 1)

LMT carry-over and reprice, cancel/modify, resting per-lot `STP` orders (the rev 2 Q14
decision), alerting for a halted cycle, `/iserver/account/trades` windowing beyond 7 days,
multi-account allocation, IBKR market data.

## 7. Surviving open decisions

1. **LMT in phase 3 or phase 4?** Shipping LMT live means owning unfilled orders across
   cycles. Default: domain-ready in phase 0, live LMT in phase 4 unless a strategy needs it.
2. **`StrategyConfig.broker`** — new optional field on a config the research surface also
   reads. Default `"sim"`. Alternative: keep broker selection purely a CLI flag.
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
10. **Ownership scope key.** Resolved in rev 4 §4: the live config's `scope` (default
    `StrategyConfig.name`), not the config hash and never a hash prefix. A hash-keyed
    scope orphans live lots on every parameter edit.
11. **Lot identity.** Resolved in rev 4 §3: one lot per `(scope, conid)`, closed when the
    broker's net returns to zero. Per-`order_id` lots cannot represent closure at all.
12. **Positions we do not own.** Resolved in rev 4 §3: `our_qty = account_net −
    foreign_net`; a nonzero net with `our_qty == 0` is EXTERNAL — reported loudly and
    never traded, which is what makes a manual holding visible instead of invisible.
