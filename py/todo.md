# TODO — live IBKR path

Working list of open items for `src/live/` and `src/live/adapters/ibkr/`.

Read `AGENTS.md` (repo rules) and `SKILL.md` (alpha research) before changing
anything here. The review pass that produced most of this list deliberately
argued from the code alone, not from a design document. Keep that posture when
re-verifying.

Contract note: `openapi.spec.json` is **stale** for `/iserver/account/orders` (it
lists 35 fields with neither `cOID` nor `order_ref`; the live response has 31
including `order_ref`). The captured fixtures in
`src/live/adapters/ibkr/tests/fixtures/` are the contract. Never re-introduce a
hand-built mock for a contract-shaped response.

---

## P0 — open defects

### [ ] D20 — an unsized-open config loses the whole cycle's opens (recon, NOT fixed)

- **Where:** `src/live/reconcile.py:300-303` (`_open_intent` raises
  `ValueError("unsized open ...")` when the sizer yields `qty <= 0`);
  `src/live/cli.py:217-220` (the `except ValueError` that turns it into a
  `ClickException`).
- **Repro:** a strategy whose sizing lives ONLY in `strategy_params` (no
  top-level `size`, no per-trade `ctx.long(..., size=)`) emitting a bare
  `ctx.long(sym)` (signal `qty=0`) — `SizingParams.size` defaults `0.0`
  (`src/bt/size/pure.py:50`), `_size_based_qty` returns `0.0`
  (`src/bt/size/pure.py:129-131`), so `sized.qty == 0` and `_open_intent` raises.
  Reproduced with a `cup_handle_dsl` config stripped of `size`:
  `ValueError: unsized open AAPL: signal qty=0.0, sized qty=0.0
  (size_mode='equity', size=0.0)`.
- **Effect:** the cycle's ENTIRE open set is lost — no intent is emitted for any
  opening symbol, and the CLI reports `Error: unsized open ...` at exit **1**
  (the task brief said exit 0 with an empty-`intents` report; the reproduction
  shows exit 1 with a bare `ClickException`, which cron also sees as failure —
  but the opens are still silently absent from the report body, so a config with
  one unsized symbol starves every symbol that cycle). Out of scope for the
  adapter rewrite — record, do not fix.
- **Fix sketch (later):** surface the unsized open as a typed cycle/placement
  error naming the symbol, or fail config validation when a qty=0-emitting
  strategy has no resolvable `size`.

### [ ] D10 — per-symbol staleness is ungated: one fresh symbol masks a dead one

- **Where:** `src/live/engine.py:190-195` (`assert_data_fresh`), `:234` (default
  `max_age_days=5`, `:263` call), `src/live/signals.py:93-95` (`_is_fresh`).
- **Problem:** the freshness gate compares the **universe's NEWEST** bar to the
  wall clock — `_newest_ms` is `SELECT MAX(timestamp) … WHERE ticker IN
  (universe)`, so *any* one fresh symbol satisfies it. `_is_fresh` then compares
  the row's data bar (`row.ts`) to its **posture** bar (`row.sig_ts`) — both
  derived from data, never from `now`. Neither check is per-symbol wall-clock. A
  symbol whose feed died is traded on ancient bars for as long as any other
  symbol in the universe stays fresh. The DB's known stale tail (ETFs ending
  2026-02-20 vs core names ending 2026-08-07) is exactly that population.
- **Fix sketch:** gate each symbol on `now − its own newest bar`, not the universe
  `MAX`; a per-symbol bar older than its bar cadence is a hard stale error for
  that symbol (drop the row, or fail the cycle).
- **Test:** a universe with one fresh and one dead symbol is refused (or the dead
  row is dropped), not silently traded.

### [ ] D11 — no cancel/modify path anywhere; the only stop-out is manual TWS

- **Where:** `src/live/adapters/ibkr/__init__.py:6` ("LMT carry-over,
  cancel/modify, resting stops and brackets stay phase 4");
  `src/live/cli.py:375-381` (`ibkr live abandon` "clears our durable record only
  and does NOT cancel the broker order").
- **Problem:** zero cancel in non-test `src/live` (grep confirms). A working order
  is only ever removed by DAY expiry. If a placement wedges, the operator's only
  recourse is manual TWS, and `abandon` warns that re-minting the key over a
  still-live broker order **duplicates** it. Same phase-4 root as the stop-loss
  gap (P2 `Live stop-loss / take-profit`) and the `abandon` verb item under
  "Never reviewed independently".
- **Fix sketch:** a broker cancel by `order_ref`/`order_id`, wired into the
  stuck-cycle path and `abandon`; resting stops/brackets follow (phase 4).
- **Test:** a wedged working order is cancelled at the broker before the key is
  re-minted.

### [ ] D12 — no portfolio-level risk limits and no flatten-all command

- **Where:** `src/live/reconcile.py:8` ("Absence of a signal is HOLD — never
  flatten"); nothing in `src/live` matches `max_loss` / `kill` / `daily_loss` /
  `max_positions` (grep confirms); the per-cohort cash scale is the only bound.
- **Problem:** there is no kill switch: no max gross/net exposure cap, no
  max-positions cap, no flatten-all. A strategy going HOLD on every symbol leaves
  the whole book on by design, and drawdown is bounded only per-cohort, never at
  the portfolio level. Hard dependency: without D13 (no persisted equity) a
  daily-loss kill cannot even be computed.
- **Fix sketch:** a portfolio exposure/position cap checked before the cycle, and
  a `flatten` verb emitting closing orders for the scope's live lots.
- **Test:** a book over the exposure cap opens nothing; `flatten` emits one close
  per live lot and reaches flat.

---

## P1 — smaller correctness / hygiene

### [ ] D6 — fabricated `0.0` on an unreadable `cum_fill`

`_unresolved_result` does `filled_qty = fill.qty if fill is not None else 0.0`,
but `_filled` → `status_to_fill` returns `None` both for `cum_fill <= 0` **and**
for an absent/unreadable `cum_fill`. The terminal branch already reports `None`
(`_no_readable_fill`); the timeout branch asserts `0.0` on the same input. Mirror
`_no_readable_fill`.

### [ ] D7 — the shortfall denominator is pre-whole-share rounding on IBKR

`_shortfall` diffs against `intent.qty`, but the ticket carries
`whole_quantity(intent)` (`orders.py`). A sized ask of `10.4` filled as `10`
reports `short=0.4` — a rounding artifact rendered as a partial. Carry the
ticket's whole quantity as the denominator.

### [ ] D8 — sim and IBKR disagree on whether a cohort scale is a shortfall

`SimulatedBroker._result` reports the post-scale settled qty against the
pre-scale `intent.qty`, so a scaled **paper** cohort prints `partial=…`; the IBKR
path rewrites `intent.qty` before `place` (`scale_open_cohort`), so a scaled
**live** cohort prints nothing. Pick one. If the scale must be visible live,
surface it structurally (it is already logged to stderr).

### [ ] D9 — stale docstrings on the exposure guard

`_open_exposure_guard` says "pending/open intent records are read as part of the
book", but `net_exposure` reads only `live_position`; intent records are never
consulted. Also confirm `_resolve`'s note that the orders endpoint lists
filled/cancelled session orders is still accurate (it is, per the captured
fixture) and that no other comment still describes a superseded mechanism.

### [ ] `''`-for-NULL should be structural, not by convention

`LivePosition`/intent `position_id` is `TextField(default="")` — nullable. Every
writer currently normalises, but a single future writer storing `None` silently
reintroduces duplicate-PK rows in SQLite. Make it `null=False, default=""`.

### [ ] `size_mode` is declared twice, the `FeedKind` bug shape

`src/live/types.py` has `size_mode: Literal["equity", "cash", "fixed"]` while
`src/live/cli.py` has `_SIZE_MODES = frozenset({...})`. Same divergence risk L3
fixed for `FeedKind` — derive the set from the `Literal` via `get_args`.

### [ ] `owns_book` polarity is inverted at its most consequential use

`IbkrPortfolioSource.owns_book = False` means "the book is already ours, apply no
scope filter" (`engine.py` passes `owned=None`), and `MockPortfolioSource = True`
means "scope closes to the sim-owned ids". Rename (e.g. `scope_closes_to_sim_lots`
/ a `CloseScoping` enum) or document it loudly at the IBKR site — it decides
whether a close is emitted at all.

### [ ] attempt monotonicity is bounded by the prune window

`prune` deletes terminal rows and the next `open_attempt` restarts at 0, so a
90-day-old cOID can be re-minted. Safe against IBKR's dedupe windows, but the
invariant as written ("attempt is monotonic per key") is not what is implemented.
Either state the bound or keep the counter monotonic independently of retention.

### [ ] D13 — live NAV / equity is never persisted, so there is no drawdown or daily-loss monitor

- **Where:** `src/live/portfolio_source.py:74` and `src/live/reconcile.py:198,221`
  each set `equity_curve=()`.
- **Problem:** the live book carries cash and positions but no equity series;
  scope equity is never persisted. So no drawdown, no daily-loss monitor, no
  "am I up" from our own data — the account summary is display-only
  (`src/live/adapters/ibkr/portfolio_source.py:12`). This is the enabling gap
  behind D12's kill switch.
- **Fix sketch:** persist a per-scope equity mark per cycle (own cash + positions
  valued at the cycle's reference prices) and compute drawdown from it.

### [ ] D14 — data ingestion is not wired to the live path; the freshness default is far too loose

- **Where:** `src/live/engine.py:168-207` (`assert_data_fresh` only READS the DB);
  `src/live/cli.py:110` (`--max-age` default `5` days).
- **Problem:** nothing in the live path downloads bars. If cron does not run
  `ibkr data dl` first, the cycle trades whatever is in the DB (or errors). The
  `5`-day default is far looser than any bar cadence — a daily strategy trading a
  5-day-old bar passes the gate. Undocumented operational dependency, and it blunts
  D10's gate further.
- **Fix sketch:** document the `data dl` prerequisite in the run/cron unit (see P2
  `No scheduling artifact`) and set `--max-age` to the strategy's bar cadence.

### [ ] D15 — scope cash is derived from our own executions; a missed fill is unreconciled and two scopes can over-allocate

- **Where:** `src/live/ledger.py:12-13` ("per-scope cash is derived from the
  scope's own executions, never read from the account summary");
  `src/live/adapters/ibkr/portfolio_source.py:12`.
- **Problem:** cash and equity are the scope's `initial_capital` advanced by its
  own fills. A fill we never reconcile (7-day window lapse, contract drift —
  `ledger.py:3`) leaves cash permanently wrong with no account-side cross-check.
  Two scopes on one shared account each size against their own book; there is no
  cross-scope budget, so together they can over-allocate the account's real cash.
- **Fix sketch:** a periodic account-vs-sum-of-scopes warning (per-scope need not
  match, but the SUM must not exceed the account).

### [ ] D16 — shorting has no locate/borrow handling (shorts may silently never open)

- **Where:** `src/live/reconcile.py:279` can emit `ActionType.short`.
- **Problem:** IBKR equity shorts need a locate/borrow and margin; a refusal
  decodes as `rejected`, which is excluded from `_UNSAFE_OUTCOMES` (D3), so the
  cycle exits 0 and the short silently never opens. Overlaps D3.
- **Fix sketch:** surface a hard-to-borrow / locate refusal as its own unsafe
  outcome; fail the open loudly, or pre-screen borrowability.

### [ ] D17 — MKT DAY only, and no execution-quality measure in the report

- **Where:** `src/live/adapters/ibkr/orders.py:28,171-174,197,215` (MKT only,
  `tif` always `DAY`); fill price is never diffed against the expected/ref price
  anywhere in the live report. `slippage_bps` (`src/live/types.py:210`) is a
  sizing input, not a measurement.
- **Problem:** with MKT DAY there is no limit discipline and no feedback on
  execution quality — live fills are never compared to the reference price, so
  realised slippage is invisible and cannot inform whether LMT carry-over (D11 /
  phase 4) is warranted.
- **Fix sketch:** report realised slippage = (fill price − ref price)/ref price
  per fill.

### [x] D18 — CLOSED: the live book has its own sqlite file (`data/live.db`)

The live book no longer shares `../data/ibkr.db` with candles/research. Paths
are one truth per file in the leaf `src/db/path.py` (`resolve_db_path` /
`resolve_live_db_path`, both file-relative, both env-overridable); the ledger
defaults to the live file. The DDL-extraction and migration framework landed with
it: `src/db/` (path/connection/models/introspect + `migrations/`), our own
`peewee_migration` bookkeeping table (not kysely's), ordered `LIVE_MIGRATIONS` /
`DATA_MIGRATIONS` registries, and `ibkr db migrate|status`.

**The orphaned source tables are gone.** After the copy, the stale `live_*` tables
left in `../data/ibkr.db` were dropped by
`data_0002_drop_migrated_live_tables` — the one sanctioned exception to
rename-never-drop, gated on a guard that refuses unless the live file exists, is
migrated, and holds at least as many rows per table (the folded `live_position` /
`live_sim_lot` pair is counted against `live_position`). `ibkr.db` now holds only
`symbol` / `candle` / `fundamental` plus the 1-row `kysely_*` lineage tables. A
ledger pointed at the wrong file reads a FLAT book — verify `live_position` before
trusting it. See AGENTS.md § The database / § Where state lives.

### [ ] D19 — a scaled live cohort prints no shortfall (silent undersizing)

- **Where:** `src/live/reconcile.py:157` (a partial entry stands, never topped
  up); `src/live/adapters/ibkr/orders.py:438` (`scale_open_cohort` rewrites
  `intent.qty` via `replace`), so the ticket carries the scaled qty and the
  shortfall denominator is the already-scaled qty (see D7 for the rounding cousin).
- **Problem:** a cohort rescaled down by the shared-cash scale prints **no**
  shortfall at all, because the intent was rewritten before the fill was compared
  to it. This is D8's live half; the partial-entry permanence is the P2
  `Partial-open top-up` decision and the missing ask column is noted there.
- **Fix sketch:** report the scale and the pre-scale target structurally (as D8
  argues for the sim side), so a live undersize is visible.

### [ ] VERIFY — currency is carried but ignored in the cash math (non-USD names)

- **Where:** `src/live/adapters/ibkr/mapping.py:36,47` (`currency` on the position
  and summary rows), `:138,164` (read from the payload); grep shows `currency`
  appears ONLY in `mapping.py` — no cash/qty math consults it.
- **Check:** some configured universes include non-USD names
  (`strats/pass/momentum_compression_breakout_ae_gate_SPY.json:40,79` — CCEP,
  NBIS). If a non-USD fill is booked as USD, the scope's cash and every derived
  size are wrong. Confirm whether IBKR reports these in USD (they can trade USD on
  a US venue) before treating this as a defect — a claim only once checked.

### [ ] VERIFY — gateway credential source

- **Where:** no `os.environ`/`getenv` anywhere in `src/live` (grep confirms);
  credentials load in `src/data/ibkr/login.py:27-113` (`IBKR_USERNAME` /
  `IBKR_PASSWORD` via `.env`, `os.environ`), and the gateway URL from
  `src/data/ibkr/client.py:85`.
- **Check:** taken together this reads as resolved — credentials come from the
  environment / a gitignored `.env` (`.gitignore:16`), and NO `strats/*.json`
  carries password/token/secret/credential (grep confirms). Confirm the exact
  loader path the live entrypoint uses and that no committed file names them,
  before go-live. Enter as a claim only once checked.

---

## P2 — deferred decisions (need a call, not just code)

### [ ] Live stop-loss / take-profit — designed, not implemented

Plan: `docs/PLAN_LIVE_STOPS.md` (rev 1). Today a stop-carrying strategy is
refused every cycle **at exit 0** (D3's shape), so many DSL strategies are
untradeable live. The plan ships a resting `STP` (`GTC`) child plus a per-cycle
breach backstop, behind `LiveConfig.allow_stops` (default OFF), in five slices;
it also fixes the missing level persistence on the IBKR path and the cancel
needs-confirmation rule. Slice 1 (vocabulary + migration, zero behaviour change)
needs no gateway.

### [ ] Partial-open top-up — revisit only on measurement

Decision taken: the posture diff compares **sides**, never sizes, so a partial
entry stands and the residual is never chased. Rationale: chasing re-sizes on
every equity/price tick, and under-filling errs toward **less** exposure than
intended (the safe direction for risk-sized strategies). What shipped instead is
reporting: `OrderResult.filled_qty` / `FeedError.filled_qty` → `filled` +
`shortfall` in the JSON and `partial=x/y short=z` in the text line.

**You cannot measure the partial rate today:** `IntentRecord` persists state,
attempt, `order_ref`, `order_id`, `decision_ts`, `tif`, `stuck_cycles` — but not
the ask (`total_size`). Add that column (additive, preserve-don't-drop) and count
terminal partials over N live cycles.

**Trigger to revisit:** if terminal partials are more than a few percent of
opens, implement the top-up: for an OPEN whose record is terminal and whose held
qty is short of the freshly sized target by more than a threshold, emit an open
for the residual under the same key (new attempt). Note the interaction with
`_open_cash_guard` and the cohort scale before doing it.

### [ ] Cycle lease is unsound on a shared/network filesystem

`lease.py` uses `flock` on `<ledger-db>.cycle.lock`. Right on one host (the kernel
drops it on `SIGKILL`, a stale file is not a lease, no TTL needed). Unsound when:
NFS/CIFS with `nolock` (two hosts both "hold" it), an NFS server restart inside
the lock-grace window, a container on another host, or the lock file deleted and
re-created (new inode → new lock domain). Either use a distributed lock (a lease
row with a heartbeat, or `BEGIN EXCLUSIVE` on a lease table) or state
"single host / local filesystem" in the docstring and enforce nothing.

### [ ] A dry run can read a torn book

`dry_run` deliberately takes no lease (a read must not block live trading), so it
can read a book another cycle is mid-way through writing. One docstring line,
or a read-only snapshot.

### [ ] No scheduling artifact, and the exit code is undocumented for cron

There is no cron/systemd unit in the repo (`scripts/` holds `streamline_cycle.py`,
`login_ibkr.py`, `fetch_macro_fred.py`, `get_ticker_range.py`). Add a documented
cron example that relies on the exit codes: **0** ok, **1** config/stale/gateway
(`ClickException`), **2** usage, **3** unsafe cycle. Say plainly that
`--allow-unsafe` must NOT appear in it.

### [ ] Capture a filled-order fixture

`gateway_trades.json` holds only a pre-upgrade-scheme `order_ref`
(`511350df-7f3f2b38-20261005T1749-000`). No captured fixture shows a CURRENT
scheme ref echoed through `/iserver/account/trades` after a real fill, nor the
filled case in `/iserver/account/orders` (only `PreSubmitted`/`Cancelled`).
One MKT round trip on Paper closes that gap.

---

## Verification hygiene (learned the hard way)

- **Never build a contract mock by hand.** The `cOID` bug survived three review
  waves because every test invented a field the gateway does not send. Parse the
  captured fixtures.
- **Beware constant mocks in ordering tests.** `test_close_is_sequenced_before_open_in_the_cohort`
  held a constant account position across a close, which is exactly why D1 (the
  flip refusal) was invisible. If a mock represents broker state, it must MOVE
  when our order lands.
- **A test that passes before and after is a characterisation test, not a
  regression test.** Say which it is when reporting.
- **Probe scripts get a target check.** A probe wrote a cancel against every
  listed order instead of only ref-matching ones, and cancelled a pre-existing GOOG
  order that was not ours. Filter by our own `order_ref` before acting, and print
  what will be acted on.
- **`bt`/live outputs are read whole.** Metrics that matter live in the tail
  (`AGENTS.md`); state any trim.

---

## Done — do not re-litigate

- **D3 CLOSED — a refused CLOSE is a book divergence, so it exits 3, not 0.** The
  close-vs-account mismatch reports `kind="divergence"`/`OrderOutcome.DIVERGENCE`
  (`_close_guard`, `src/live/adapters/ibkr/broker.py`) and the structural cohort
  drops (`_refuse_opens`/`_apply_scale`) report the distinct `unfunded` outcome,
  both added to `_UNSAFE_OUTCOMES`. Pinned by `test_a_refused_close_exits_three`,
  `test_an_all_opens_dropped_cohort_exits_three`,
  `test_a_plain_refused_open_still_exits_zero` (`src/live/tests/test_cli.py`) and
  `test_a_refused_close_reports_the_unsafe_divergence_outcome` and
  `test_an_all_opens_dropped_cohort_reports_the_unsafe_unfunded_outcome`
  (`src/live/adapters/ibkr/tests/test_ibkr_broker.py`).
- **D2 CLOSED — `--allow-unsafe` names what it suppressed.** An unsafe cycle under
  `--allow-unsafe` now exits 0 AND writes a stderr note naming the suppressed
  outcomes, derived from the report (never a fixed string). Pinned by
  `test_allow_unsafe_suppresses_the_nonzero_exit` and
  `test_allow_unsafe_emits_a_note_naming_the_suppressed_outcome`
  (`src/live/tests/test_cli.py`).

- **Order identity** is bar-free (`identity.py`): `cOID =
  scope_tag-token-attempt`, durability in `live_order_intent`, five invariants
  (no POST after a failed working-orders read; bar-free prefix; ambiguity never
  reads as "not placed"; the intent table owns OPEN state; adoption by exact
  scope prefix only). Superseded helpers deleted.
- **Gateway contract** settled empirically: `/iserver/account/orders` echoes our
  id as **`order_ref`**; foreign orders carry no `order_ref` at all; the endpoint
  lists filled/cancelled session orders (so terminal rows are filtered).
- **Exit code** for unsafe cycles (3) with `--allow-unsafe`; `resync_error`
  surfaced; outcome vocabulary never defaults to `REJECTED` for an ambiguous POST.
- **Partial fills** are reported as a shortfall (see P2 for the deferred top-up).
- **Cohort cash** mirrors the backtest `_scale_opens` (commission reserve,
  friction price); a failed funding close drops its opens; opens bound their
  whole-share notional.
- **Open exposure divergence guard** exists with the in-cycle delta so a flip's
  own close is excused (`f72d720`) — keep that case covered.
- **Ledger split** (`ledger.py` → `ledger_base` / `ledger_migration`, with the
  sim-lot store later folded back into `ledger.py`) verified refactor-only,
  migrations atomic and rename-never-drop.
- **Removed:** `ports.Gateway`, `reconcile.size_qty`, the vestigial imports fixed
  by the `src/timestamps.py` leaf (killed the `src/exec` collection cycle).
- **Closed finding history** (all fixed): C1, H1, H1b, H3, H4, H5, M1–M8,
  L1–L10, D1, D4, D5.

## Never reviewed independently

Worth a pass when someone has budget: the `ibkr live abandon` verb and its
`--yes` gating, `WEDGED_CYCLES = 3` tuning, `_EXPOSURE_TOLERANCE = 1e-6` and
`_OPEN_CASH_TOLERANCE = 0.02` against real fills, and the pruning/retention
policy in `ledger_*`.
