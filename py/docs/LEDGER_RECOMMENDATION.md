# Ledger Recommendation

**Decision record.** Companion to [`HANDOFF_LIVE_TRADING.md`](HANDOFF_LIVE_TRADING.md).
Answers: *should the backtester also use the strategy→lot ledger (with an
in-memory implementation)?*

---

## Verdict

**No — the ledger stays live-only.** The backtest keeps its in-memory
`PortfolioState`; the SQLite `Ledger` (`src/live/ledger.py`) and no in-memory
twin in `src/bt`.

### Why the backtest does not need a ledger

- **`PortfolioState.positions` already IS the ledger**, in-memory, per run.
  `positions: dict[str, tuple[Position, ...]]` is the strategy→lot record, and
  `Position.position_id` is the lot handle. A `Ledger` mirror would duplicate it.
- **No ownership ambiguity.** A backtest is one config = one book. There is no
  broker account shared with other strategies, so "which lots are mine?" is
  always "all of them". The live ledger exists precisely because a broker
  account can hold lots opened by other configs.
- **Runs are isolated.** `sweep` / `split` / `optimize` fork fully independent
  processes with fresh state; nothing survives a run to be reconciled later.
- **The surfaces don't match.** The engine's lifecycle is
  `apply_fill(portfolio, fill) -> PortfolioState` — a pure state transition. The
  `Ledger` surface (`ensure_strategy` / `record_open` / `mark_closed` /
  `prune_closed`) is an append/status side-record. Forcing the engine through a
  ledger duplicates state and fights the pure-functional design.

### Where the shared `Ledger` Protocol *does* pay

1. **`InMemoryLedger` as a live test fake.** Implement it in
   `src/live/tests/`, conforming to the `Ledger` Protocol, so `run_cycle` tests
   exercise the full record-open / mark-close path with no SQLite. This is a
   test double, **not** a `src/bt` component — it never enters the backtest.
2. **Future multi-strategy backtests.** If a single book ever runs several
   strategies (a portfolio-of-strategies), ownership scoping becomes necessary
   and `reconcile` already speaks the Protocol. Revisit then; not before.

---

## Real latent bug: `position_id` collision

Found while assessing the ledger — worth fixing independently of it.

- `_open_position` auto-generates the handle as
  `f"{signal.symbol}_{fill.timestamp.timestamp()}"` (`src/bt/portfolio/pure.py`).
- The DSL's open path (`ctx.long` / `ctx.short` → `_emit`) leaves
  `position_id=None`, so **every open relies on that auto-id**.
- Two lots opened on the **same symbol at the same fill bar** (they drain in one
  `apply_fills` cohort, so `fill.timestamp` is identical) mint the **same id**.
  `_close_position` then matches the first lot only; the second is unreachable by
  `position_id` (multi-lot strategies that close via `position_id` mis-target).

### Fix: a shared, unique lot-id allocator

- Make the auto-handle **unique per open**, not merely per `(symbol, ts)`.
- **Deterministic and pure** — derive the monotonic component from append-only
  state already in the book (e.g. `len(portfolio.trades)`, which grows by exactly
  one per open and is never truncated during a run). **Do not** use a module-level
  counter: it breaks determinism and is unsafe across `sweep`/`split`/`optimize`
  worker processes.
- One helper (`next_position_id(symbol, ts, seq)`) used by **both** the engine's
  `_open_position` and the live `SimulatedBroker`, so backtest and paper share
  one handle scheme and `reconcile` can close the right lot.

Invariant to preserve: `position_id` is the canonical handle for every operation
on an open lot (`close`, `rebalance`, `resolve_lot`), and it must be unique among
an open symbol's lots at any bar.

---

## Action items

- [ ] **Fix the collision** — extract the id allocator; wire the engine's
      `_open_position` (and the live `SimulatedBroker` once it lands). Tests:
      two opens, same symbol, same bar → two distinct `position_id`s; a
      `close` targets exactly one.
- [ ] **Live tests** — add `InMemoryLedger` (Protocol-conforming) as the fake
      for `run_cycle` ledger interactions; no SQLite in unit tests.
- [ ] **Revisit** — only if multi-strategy single-book backtests arrive, promote
      the ledger from live-only to a shared concern.
