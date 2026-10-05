# Plan — Unified execution core, IBKR as first real broker (rev 3)

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

## 3. Reconciliation from trade history

`GET /iserver/account/trades` (7-day window) returns per execution: `execution_id`,
`order_id`, `order_ref`, `symbol`, `side`, `size`, `price`, `commission`, `net_amount`,
`trade_time_r`, `account_code`.

```python
@dataclass(frozen=True)
class Execution:
    execution_id: str; order_id: str; order_ref: str
    symbol: str; side: OrderSide; qty: float; price: float
    commission: float; ts: pd.Timestamp

def replay(executions: tuple[Execution, ...], *, ref_prefix: str) -> BrokerSnapshot
```

- Ours = `order_ref.startswith(ref_prefix)` (prefix = `strategy_id[:8]`). Foreign orders
  are excluded from the strategy book and reported, never traded.
- One lot per `order_id`: qty = net of its executions, `entry_price` = VWAP of the
  opening executions, status open/closed from the net.
- Cash from `/portfolio/{acct}/summary`; net qty cross-checked against
  `/portfolio/{acct}/positions/{page}`. A cross-check mismatch is a warning with the
  execution detail attached, not a silent abort — the trades are the finer record.
- `LivePosition` (local table) keeps ONLY strategy-owned metadata: `order_ref`,
  `stop_loss`, `take_profit`, `tag`, `strategy_id`. It is metadata, never the source of
  qty, price or open/closed state.

This deletes from rev 2: `join_book` and its six drift rules, close grouping,
`OrderResult.closed_position_ids`, the cost-tolerance knob, and the tiered repair policy.

## 4. Order identity and re-run safety

```python
def order_ref(strategy_id: str, cycle_ts: pd.Timestamp, seq: int) -> str
# f"{strategy_id[:8]}-{blake2b(strategy_id, 4).hex()}-{cycle_ts:%Y%m%dT%H%M}-{seq:03d}"
```

The human-readable 8-char prefix stays first (so `startswith(strategy_id[:8])`
ownership scoping in §3 still holds), but it is followed by a 4-byte digest of the
FULL `strategy_id`: two strategies whose ids share the first 8 chars would
therefore no longer mint identical refs (which a real broker would silently
dedupe as a re-send).

`seq` is the intent index within the cycle, and reconcile order is deterministic
(config symbol order, closes first). Re-running the same cycle re-sends the same refs and
IBKR dedupes; the next bar gets fresh refs, so a legitimate re-entry is never blocked.
No hashing of price or qty, so a size change does not silently defeat the dedupe.

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

### Phase 3 — IBKR order placement (MKT first)

- `adapters/ibkr/{orders,broker,authz}.py`: one order per intent (opens and closes),
  reply-confirmation loop, `wait_filled` polling, `status_to_fill`.
- Step 7-style test ladder, then the manual paper checklist from rev 2 (scaled-in symbol
  closes correctly; manual TWS trade is reported, not silently absorbed).
- LMT is available in the domain from phase 0 and can ship here once carry-over policy
  exists; default remains MKT.
- Non-goals: no resting stops, no cancel/modify, no bracket/OCA.

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
7. **`adapter: "auto"`** for `mode: "paper"` only; `mode: "live"` must name its adapter. Default: as stated.
8. **New `FeedError` kinds.** Now only `rejected`, `unfilled`, `timeout` needed —
   `position_drift` is gone with the ledger-of-truth. Confirm the widened literal.
9. **Golden-report parity artifact.** Where does the phase-1 baseline get stored so the
   diff is reproducible (a `tests/fixtures/bt_reports/` snapshot vs a scratch dir)?
   Default: scratch dir in phase 1, committed snapshot only if it stays stable.
