# Plan — live stop-loss / take-profit (rev 1)

Status: **design only, nothing implemented.** Author: architect pass, 2026-10-07.
Read `AGENTS.md` (invariants, `src/live` layout, size/type rules) and
`todo.md` (open live defects) first. Companion plans: `PLAN_LIVE_HEXAGONAL.md`,
`PLAN_LMT_ORDERS.md`. Pros/cons history belongs in
`OPEN_QUESTIONS_PROS_CONS.md`.

## 0. Why

Many DSL strategies set `sl=`/`tp=` on `ctx.long(...)`/`ctx.short(...)` (grep
`sl=` / `tp=` across `src/bt/strategies/` — e.g. `mfi_pivotdiv_dsl.py:316`,
`cup_handle_dsl`, `pullback_freshlow_dsl`, `trend_pullback_atr_trail_dsl`).
The IBKR live edge **refuses** any intent carrying a level
(`adapters/ibkr/orders.py::validate_order`, ~:176, `UnsupportedStopOrder`), so
those strategies are untradeable live. The decision: honour them, do not refuse.

The refusal exists for a good reason (module docstring, same file): this adapter
places no resting order, so sending the entry naked would silently drop the
strategy's risk levels — and `AGENTS.md` § Live trading states "stop-loss /
take-profit and LMT are refused, never silently sent naked and never implied in
the report". **This plan deliberately changes that invariant** (§9) and must say
what replaces it.

## 1. Findings that constrain the design

| # | Finding | Consequence |
| --- | --- | --- |
| F1 | `sl`/`tp` are minted **once, at entry**, by `dsl.py::_emit` → `sl_tp_from_pct` (absolute prices). No strategy re-emits a moved stop: `ctx.long(sl=)` on an already-long symbol reconciles to HOLD. Trailing exits are `ctx.close`. | A stop-REPLACE path is unreachable from the DSL today. Design it correctly, do not depend on it. |
| F2 | The refusal exits **0**, not 1: `validate_order` → `_prepare_leg` → `feed_error("rejected")` → `OrderResult(REJECTED)`, and `REJECTED` is not in `_UNSAFE_OUTCOMES` (`engine.py`). | A stop-carrying strategy is refused **every cycle at exit 0**; cron never alerts. Same shape as `todo.md` D3. This plan must fix it, not inherit it. |
| F3 | `live_position.stop_loss/take_profit` are **never written on the IBKR path**: `trades._open_row` sets them `None` and `_apply` preserves via `replace`; the execution stream carries no level. | Levels must be persisted by us (`record_levels`). |
| F4 | `singleOrderSubmissionRequest` (`openapi.spec.json:37848` — authoritative for submit, unlike the stale `/orders` list): requires `conid/orderType/quantity/side/tif`; `tif` enum includes `GTC`; `auxPrice` is "used … such as stop orders"; `parentId` = "child order in a bracket … equal to the cOID of the parent"; `isSingleGroup` = "orders in the containing array treated as an OCA group". | STP/child/bracket placement is expressible. GTC is available. |
| F5 | Cancel = `DELETE /iserver/account/{accountId}/order/{orderId}`, response `orderCancelSuccess`, whose own spec text says it "does not report whether the cancellation can or will ultimately be enacted". | A cancel must be CONFIRMED by a subsequent working-orders read; fail closed otherwise. |
| F6 | A just-filled lot is not yet in the seeded book, so `order_side`/`position_side_of` cannot resolve a protective child's side there. | The child carries its own `position_side` (§3). |

## 2. Decisions

### D1 — Enforcement: hybrid. Resting `STP` child for the stop + per-cycle backstop for both levels

On a confirmed entry fill the adapter places **one** sibling `STP` child
(`tif=GTC`) sized to the **confirmed filled** quantity. `tp` gets **no** resting
order — it is enforced by the per-cycle breach backstop, reported as a price
miss when it fires between cycles.

Rejected:
- **pure resting** — a resting order can fail to exist (refusal, lost session,
  partial fill, unconfirmable cancel) and nothing would notice; that is exactly
  "silently sent naked";
- **pure per-cycle** — the whole inter-cycle gap is unprotected while the
  backtest triggers intrabar (`bt/risk/pure.py::check_risk` on `tick.low/high`),
  so it is neither equivalent nor "honoured";
- **`tif=DAY`** — dies at every session boundary; with ~one cycle per session the
  position is naked every session, silently;
- **bracket/OCA pair** (entry + sl + tp via `parentId`/`isSingleGroup`) as
  slice 1 — right mechanism in principle, but needs a multi-leg reply loop, qty
  re-derivation after `scale_open_cohort`, and OCA semantics that are
  **unverifiable offline**. Deferred with the two-sibling rule below.

**Two-sibling hazard, avoided by construction.** Two resting reduce orders on one
lot means one firing leaves the other live, which then opens the opposite side.
With **sl only** resting there is no sibling. Any resting `tp` child ships only
together with OCA (`isSingleGroup`), after a Paper capture.

**Naked window.** Entry fill → child POST, inside one cycle (seconds). If the
child POST fails, the outcome is `UNPROTECTED` → exit 3. Never silent.

### D2 — Identity: extend the existing key space with a `leg` discriminator

`IntentKey` gains `leg: IntentLeg = "entry"`. `token()` folds the leg **only when
non-entry**, so every existing `order_ref` is byte-identical and in-flight
adoption survives:

```python
identity = f"{symbol}|{action}|{position_id or ''}"            # leg == "entry" (unchanged)
identity = f"{symbol}|{action}|{position_id or ''}|{leg}"      # protective legs only
```

Why not a separate key space: `live_order_intent` (PK = the identity columns),
`OPEN_STATES`, `stuck_cycles`, `abandon`, `match_working` and `resync` all work
unchanged for any key. A plain `close` and its `stop` would otherwise share a
prefix (same scope/symbol/close/conid) and `match_working` (max-attempt ref for a
prefix) would alias them. The leg splits them.

- **REPLACE** (moved trail / qty change): cancel-confirmed first, then
  `open_attempt` on the **same key** → attempt+1 → new cOID, so IBKR's cOID
  dedupe cannot swallow it and the old attempt is gone before the new exists. An
  unconfirmable cancel means **do not submit the replacement** → `UNRESOLVED`,
  exit 3. (Unreachable from today's DSL per F1; kept correct anyway.)
- **`abandon`** gains `--leg` (default `entry`). For a protective leg the warning
  escalates: clearing our record while a GTC stop rests broker-side means the
  next cycle mints a **second** stop → over-cover → possible flip. The message
  names the `order_id` and demands a broker-side cancel first.
- **A stop fill books as a `close` of the parent lot** — same conid, same sign;
  `trades.reconcile` reduces the row to 0 and stamps `closed_at`. No change to
  `trades.py`. `is_ours` still matches (`scope_tag-` prefix preserved). A stored
  `tif='GTC'` makes `_day_expired` return False, so resync never wrongly expires
  a resting stop.
- **Levels for the backstop** live on `live_position.stop_loss/take_profit`,
  written by us when an entry is confirmed filled, preserved across replays by
  `replace` in `_apply`, and already mapped onto `Position` by
  `IbkrPortfolioSource._to_position`.

### D3 — Close guard / same-cycle flip: no new allowance, no widening

- The stop fill is replayed by `fetch()` **before** `resync`/`reconcile`, so the
  lot is already closed when reconcile diffs posture → reconcile emits no close.
- If the fill is not yet visible, account net (flat) ≠ booked (+qty) + delta (0),
  so `_open_exposure_guard` **refuses**. Intended: a fill we cannot see may be
  live. Unchanged.
- `_confirmed_reduce_delta` credits only results from **this cycle's** placement
  loop; a broker-initiated stop fill is never credited. Unchanged.
- **`_cancel_protection` runs before every entry-leg close** (and before every
  synthetic breach close). That is the disambiguator: with the resting stop
  cancelled and confirmed absent, a synthetic close cannot race a live stop; if
  the stop already fired, the close guard refuses on a flat account. An
  unconfirmable cancel refuses the close (exit 3) — a live stop plus a close
  would over-reduce.
- "Intended to open but the stop fired": replay closes the lot → `cur=flat`,
  `tgt=long` → an ordinary re-entry. Not a special case.

### D4 — Order type: add `STP`; entries stay MKT, LMT entries stay refused

`src/exec/types.py`: `OrderType.STP = "STP"`. No `STP_LMT` (adds a "triggered but
limit not hit, therefore naked" mode; gaps already diverge from the backtest).

`validate_order` becomes **leg-driven**:

| intent | rule |
| --- | --- |
| `leg=="entry"`, `MKT` | allowed (unchanged) |
| `leg=="entry"`, `LMT` | **refused** `UnsupportedOrderType` (carry-over still unbuilt) |
| `leg=="entry"` carrying a level | **refused** unless the levels are being placed as children (adapter `allow_stops`); the entry ticket itself **never** carries a level |
| `leg=="stop"` | requires `order_type is STP`, `stop_loss` finite > 0, `tif GTC`, reduce side |
| `leg=="take_profit"` | requires `order_type is LMT`, `take_profit` finite > 0 (unused until OCA ships) |

Ticket bodies (one builder, table-driven):

```python
entry:       {"conid": int, "side": "BUY"|"SELL", "quantity": float, "orderType": "MKT", "tif": "DAY", "cOID": ref}
stop:        {"conid": int, "side": reduce_side, "quantity": float, "orderType": "STP", "price": sl, "tif": "GTC", "cOID": ref}
take_profit: {"conid": int, "side": reduce_side, "quantity": float, "orderType": "LMT", "price": tp, "tif": "GTC", "cOID": ref}
```

**Unverified without a live gateway — each pinned by a required Paper capture
(§8), never guessed silently:**

- whether an `STP` trigger goes in `price` or `auxPrice` (the spec names
  `auxPrice` for stop orders; IBKR's own API uses `price` for `STP` and
  `auxPrice` for `STP LMT`). The plan pins `_STOP_TRIGGER_FIELD = "price"`,
  **explicitly marked unverified**;
- whether `tif="GTC"` is accepted for `STP` on this build;
- the working-orders row for a stop (`orderType`, where the trigger appears,
  whether `order_ref` is echoed) — needed for adopt/replace;
- the `DELETE` cancel body and post-cancel absence;
- whether an `STP` trip draws an `o354`/price-band **reply** (must still abort)
  or the ordinary confirm;
- `outsideRTH` default behaviour (the design leaves it absent).

### D5 — Sim parity

The sim has no resting orders and no clock between cycles, so it **cannot**
represent the resting-stop path — say so in the module docstring. It **can**
exercise offline end to end: intent → child minting, level persistence on the
held book, and the shared pure breach predicate. `SimulatedBroker`:

- places the synthetic stop child as an `OrderResult` (PLACED, message naming the
  level) and forces `stop_loss`/`take_profit` onto the settled lot;
- on the next cycle, a held book beyond the level drives the same
  `breached_levels` → synthetic close.

Backtest agreement for entries is unchanged (same `settle_cohort`/`apply_fills`).

### D6 — Report / exit-code contract

New `OrderOutcome.UNPROTECTED`, added to `_UNSAFE_OUTCOMES`.

| event | row | exit |
| --- | --- | --- |
| entry filled, stop placed GTC | `order AAPL close PLACED leg=stop qty=100 @ 92.10 GTC cOID=… order_id=…` | 0 |
| stop adopted at cycle start | `adopted … leg=stop` (`ADOPTED`) | 0 |
| stop fired | resync `FILLED`, `close AAPL qty=100 @ 91.80 … leg=stop` | 0 |
| stop **refused** by gateway | `order AAPL close UNPROTECTED leg=stop: stop 92.10 NOT resting (o…) — position is NAKED` | **3** |
| stop submit ambiguous | `UNRESOLVED` | 3 |
| SL breached, no working stop | `UNPROTECTED` row **+** synthetic close `PLACED` | **3** |
| TP breached between cycles | close `PLACED`, `tp 105.00 breached (mark 106.2) between cycles; no resting tp; closing at market now` | 0 |
| lot closed deliberately, stop cancelled OK | close `PLACED` + stop `UNFILLED` `stop cancelled: lot 265598 closed` | 0 |
| close refused: cancel unconfirmable | close `UNRESOLVED` `refused close: stop 92.10 could not be cancelled-confirmed` | **3** |
| `allow_stops=False`, intent carries a level | `REJECTED` `UnsupportedStopOrder` — and this must now be `UNPROTECTED`/exit 3, **not** today's exit 0 (F2) | **3** |

Renderer: the intent `leg` already flows through `asdict`/`_result_dict`; add
`leg=` to the text line.

### D7 — Migrations (rename, never drop)

- `live_order_intent`: **add column `leg`** and **widen the PK to
  `(scope, symbol, action, position_id, leg)`**. A PK change means a rebuild:
  `_rekey_intents_for_leg` detects the 4-column PK →
  `ALTER … RENAME TO live_order_intent_legacy_<n>` → `create_tables` rebuilds →
  `_restore_intents` copies with literal `'entry'`. Runs **after** the existing
  `_rekey_order_intents` (token-keyed legacy) inside `migrate()`. Old rows read
  `leg='entry'` and **their `order_ref`s are unchanged** (D2 token rule), so
  in-flight adoption survives.
- `live_position`: `stop_loss`/`take_profit`/`tag` already exist → **no migration**.
- `tif` already exists; a stop row stores `GTC`, which `_day_expired` correctly
  never expires.
- No new table, no `price` column: a REPLACE re-derives the trigger from the
  lot's persisted level.

## 3. Types

```python
# src/exec/types.py
class OrderType(Enum):
    MKT = "MKT"
    LMT = "LMT"
    STP = "STP"          # NEW

# src/live/types.py
IntentLeg = Literal["entry", "stop", "take_profit"]
PROTECTIVE_LEGS: frozenset[IntentLeg] = frozenset({"stop", "take_profit"})

@dataclass(frozen=True)
class OrderIntent:
    ...                                       # unchanged fields
    leg: IntentLeg = "entry"                  # NEW: which leg this is
    position_side: ActionType | None = None   # NEW: the LOT side, set on a protective child
                                              # minted from a just-filled entry (the replayed book
                                              # cannot show that lot yet, F6). None elsewhere.

@dataclass(frozen=True)
class LiveConfig:
    ...                                       # unchanged fields
    allow_stops: bool = False                 # NEW: opt-in; False reproduces today's refusal exactly

@dataclass(frozen=True)
class LevelBreach:
    symbol: str
    position_id: str                          # lot conid as str
    kind: Literal["stop", "take_profit"]
    qty: float
    level: float
    mark: float

# src/live/ports.py
class LevelWriter(Protocol):
    def record_levels(
        self, scope: str, conid: int, stop_loss: float | None,
        take_profit: float | None, tag: str,
    ) -> None: ...
```

## 4. Signatures

```python
# src/live/identity.py
IntentKey.leg: IntentLeg = "entry"                              # NEW field, defaulted
def token(self) -> str: ...                                     # folds leg only when != "entry"
def intent_key(scope: str, intent: OrderIntent) -> IntentKey: ...

# src/live/protect.py  (NEW: pure functions, no I/O)
def protective_children(
    entry: OrderIntent, *, conid: int, filled_qty: float, tif: str = "GTC"
) -> tuple[OrderIntent, ...]:
    """Protective child intents for a just-filled entry lot (sl only; tp is
    backstop-enforced). Empty when the entry carries no level. Reduce side is the
    opposite of entry.action; position_side is set; qty is the CONFIRMED fill."""

def breached_levels(
    lots: Sequence[Position], marks: Mapping[str, float] | None = None
) -> tuple[LevelBreach, ...]:
    """Open lots whose persisted sl/tp the current mark has crossed (pure). Marks
    default to each lot's last_price. sl: long low<=level / short high>=level
    (the mark is the only intrabar proxy available between cycles)."""

def breach_closes(
    breaches: tuple[LevelBreach, ...], transition: str
) -> list[OrderIntent]:
    """Synthetic entry-leg close intents for breached lots (pure, one per lot)."""

# src/live/adapters/ibkr/orders.py
def validate_order(intent: OrderIntent, *, allow_stops: bool) -> None: ...   # leg-driven, D4
def build_ticket(intent: OrderIntent, *, conid: int, side: OrderSide,
                 order_ref: str) -> Ticket: ...                              # leg-driven bodies
def protective_ref(key: IntentKey, attempt: int) -> str: ...                 # == order_ref (naming only)

# src/data/ibkr/client.py
async def cancel_order(self, account: str, order_id: str) -> dict[str, Any]: ...
    """DELETE iserver/account/{account}/order/{order_id}; same error mapping as get/post."""

# src/live/ledger.py  (BookStore mixin — implements LevelWriter)
def record_levels(self, scope: str, conid: int, stop_loss: float | None,
                  take_profit: float | None, tag: str) -> None: ...

# src/live/adapters/ibkr/broker.py   (protective logic via the new mixin, see §6)
def __init__(..., levels: LevelWriter | None = None, allow_stops: bool = False): ...
async def place_cohort(self, intents: tuple[OrderIntent, ...]) -> Result[...]: ...   # extended
async def _protect(self, entry: OrderIntent, result: OrderResult, conid: int,
                   side: OrderSide) -> tuple[OrderResult, ...]: ...
async def _cancel_protection(self, symbol: str, position_id: str) -> Result[tuple[OrderResult, ...], FeedError]: ...
async def cancel_order(self, order_id: str) -> Result[None, FeedError]: ...
async def _check_levels(self, working: Mapping[str, WorkingOrder]) -> tuple[OrderResult, ...]: ...

# src/live/broker.py  (LiveBroker Protocol + sim)
async def cancel(self, intent: OrderIntent) -> Result[OrderResult, FeedError]: ...   # NEW Protocol member
# SimulatedBroker: sets levels on its held book + shares the breach predicate

# src/live/reconcile.py
def reconcile(signals, portfolio, config, owned=None) -> tuple[OrderIntent, ...]: ...   # unchanged signature
    # NEW step: + breach_closes(breached_levels(...)) when config.allow_stops, closes-first
```

## 5. Call graph — one cycle

Production (IBKR):

```text
run_cycle(config, *, source, broker, ledger, strategy_id, scope, ...) -> CycleReport
  → source.fetch() -> Result[PortfolioSnapshot, FeedError]        # trades replay: a fired stop already closed the lot
  → broker.seed(portfolio) -> None
  → _resync(broker) -> tuple[tuple[OrderResult, ...], FeedError | None]
    → broker.resync() -> Result[tuple[OrderResult, ...], FeedError]
      → IbkrBroker._working_index() -> Result[dict[str, WorkingOrder], FeedError]
      → IntentStore.load_open(scope) -> tuple[IntentRecord, ...]
      → match_working(orders, prefix, prefer) -> WorkingOrder | None
      → IbkrBroker._check_levels(working) -> tuple[OrderResult, ...]
        → breached_levels(lots, marks) -> tuple[LevelBreach, ...]        # reports UNPROTECTED (never places)
      → IbkrBroker._resync_status(record, now) -> bool
      → IbkrBroker._persist_working(key, working, decision_ts) -> None
      → IntentStore.close(key, state, order_id, now) -> None
  → reconcile(signals, portfolio, config, owned) -> tuple[OrderIntent, ...]
    → _plan_symbol(portfolio, sig, owned) → _close_intents(symbol, sig, lots, transition)
    → _settled_book(portfolio, closes, config) -> PortfolioView
    → _open_intent(sig, view, config, cur, tgt) -> OrderIntent
    → breached_levels(...) -> tuple[LevelBreach, ...]                    # NEW
      → breach_closes(breaches, transition) -> list[OrderIntent]         # NEW
  → _place_all(broker, intents) -> Result[tuple[OrderResult, ...], FeedError]
    → broker.place_cohort(intents) -> Result[...]
      → placement_order(intents) -> tuple[OrderIntent, ...]              # closes (incl. synthetic) first
      → IbkrBroker._place(intent, *, in_cycle_delta) -> Result[OrderResult, FeedError]
        → _prepare_leg(intent) -> Result[tuple[OrderSide, int], FeedError]
          → order_side(intent, position_side) -> OrderSide               # prefers intent.position_side
          → _conid(symbol) -> int
        → _cancel_protection(symbol, position_id) -> Result[tuple[OrderResult, ...], FeedError]   # NEW
          → cancel_order(order_id) -> Result[None, FeedError]            # NEW client DELETE
          → _working_index() -> Result[...]                              # confirm absence (fail closed)
        → _pre_guard(intent, side, conid, in_cycle_delta) -> FeedError | None
        → _working_index() -> Result[dict[str, WorkingOrder], FeedError]
        → _resolve(intent, key, index) -> Resolution
        → _guard_lost_predecessor(intent, key) -> Result[OrderResult, FeedError] | None
        → _submit_new(account, intent, key, side, conid) -> Result[OrderResult, FeedError]
          → PendingIntents.open_attempt(key, decision_ts, now) -> IntentRecord
          → build_ticket(intent, conid, side, order_ref) -> Ticket        # MKT / STP / LMT by leg
          → _submit(account, ticket, key, attempt, decision_ts) -> Result[str, FeedError]
          → wait_filled(order_id, intent, ticket) -> _WaitOutcome
      → IbkrBroker._protect(entry, result, conid, side) -> tuple[OrderResult, ...]   # NEW; only when result.ok
        → protective_children(entry, conid, filled_qty, tif) -> tuple[OrderIntent, ...]   # STOP MINTED HERE
        → IbkrBroker._place(child, in_cycle_delta=0.0)                    # reuses the whole guard chain
        → LevelWriter.record_levels(scope, conid, stop_loss, take_profit, tag) -> None
        # child refused/ambiguous ⇒ OrderOutcome.UNPROTECTED ⇒ exit 3
```

Tests:

```text
reconcile(signals, portfolio, config, owned) -> tuple[OrderIntent, ...]
  → breached_levels(lots, marks) -> tuple[LevelBreach, ...]              # fixture lot: last_price past sl
    → breach_closes(breaches, transition) -> list[OrderIntent]           # asserts the close is entry-leg
protective_children(entry, conid, filled_qty, tif) -> tuple[OrderIntent, ...]
  → build_ticket(child, conid, side, order_ref) -> Ticket                # STOP body: STP/price/GTC
validate_order(intent, allow_stops=False) -> None                        # raises UnsupportedStopOrder
IbkrBroker._cancel_protection(symbol, position_id) -> Result[...]        # respx: DELETE + absence read
IbkrBroker._check_levels(working) -> tuple[OrderResult, ...]             # UNPROTECTED when no stop key
```

## 6. File-by-file change list

| file | kind | ΔLOC |
| --- | --- | --- |
| `src/exec/types.py` | edit | +1 (`STP`) |
| `src/live/types.py` | edit | +45 (`IntentLeg`, `OrderIntent.leg/position_side`, `LiveConfig.allow_stops`, `LevelBreach`) |
| `src/live/identity.py` | edit | +30 (`IntentKey.leg`, leg-folding token, `intent_key`) |
| `src/live/protect.py` | **NEW** (pure) | ~90 (`protective_children`, `breached_levels`, `breach_closes`) |
| `src/live/reconcile.py` | edit | +40 (breach step, closes-first) |
| `src/live/ports.py` | edit | +12 (`LevelWriter`) |
| `src/live/ledger.py` | edit | +45 (`leg` column/PK, `record_levels`) |
| `src/live/ledger_migration.py` | edit | +35 (`_rekey_intents_for_leg`, restore with `leg='entry'`) |
| `src/live/engine.py` | edit | +2 (`UNPROTECTED` in `_UNSAFE_OUTCOMES`; see D6/F2) |
| `src/live/broker.py` | edit | +60 (Protocol `cancel`; sim child + levels + breach) |
| `src/live/adapters/ibkr/orders.py` | edit | +110 (leg-driven `validate_order`/`build_ticket`) |
| `src/live/adapters/ibkr/protect.py` | **NEW** (mixin) | ~180 (`_protect`, `_cancel_protection`, `cancel_order`, `_check_levels`) |
| `src/live/adapters/ibkr/broker.py` | edit | +30 (ctor `levels`/`allow_stops`, mixin wiring) |
| `src/data/ibkr/client.py` | edit | +25 (`_delete`, `cancel_order`) |
| `src/live/cli.py` | edit | +30 (`allow_stops`, `abandon --leg`, `leg=` in text, wire `levels=ledger`) |
| `AGENTS.md` | edit | invariant rewrite (§9) |

`IbkrBroker` is already near the 150-LOC class budget, so the protective logic
goes in a **class-level mixin** `src/live/adapters/ibkr/protect.py` — the same
precedent as the `ledger.py` mixins — rather than growing that class.

## 7. Ordered slices (each independently verifiable)

1. **S1 — vocabulary + persistence, zero behaviour change.** `OrderType.STP`,
   `IntentLeg`/`leg`, `IntentKey.leg` + leg-folding token, the `leg` column and
   PK rebuild migration, `LiveConfig.allow_stops=False`, `LevelWriter` /
   `record_levels`. *Verify:* golden test that `order_ref` for a fixed key equals
   today's literal; migration test on a pre-leg sqlite copy; existing suite green.
2. **S2 — pure layer.** `src/live/protect.py`, leg-driven `validate_order` /
   `build_ticket`, the reconcile breach step (gated on `allow_stops`). *Verify:*
   table tests on bodies; `allow_stops=False` still raises `UnsupportedStopOrder`;
   breach → close intent pure test.
3. **S3 — sim end to end, offline.** `SimulatedBroker` places the child, sets
   levels, enforces the shared predicate. *Verify:* `ibkr live run <cfg>
   --adapter sim` (two fixtures) and `--dry-run`; assert the child row, the level
   round-trip, the breach close.
4. **S4 — IBKR placement, flag-gated, default OFF.** The protect mixin,
   `_cancel_protection`, `cancel_order`, `_check_levels`, `UNPROTECTED`.
   *Verify:* respx tests over **captured fixtures** (§8); engine exit-3 rows.
5. **S5 — Paper capture → local run → flip the default → rewrite the invariant.**
   Requires a live Paper session.

## 8. Test plan

| step | tests | kind | fixture / gate |
| --- | --- | --- | --- |
| S1 | `order_ref` golden equality; `IntentKey.token` leg-fold; migration: pre-leg table → rebuilt with `leg='entry'`, rows preserved, refs unchanged | REGRESSION (golden) + CHARACTERISATION (migration) | none — `tmp_path` sqlite seeded with the pre-leg DDL |
| S2 | `validate_order` per leg; `build_ticket` bodies (MKT/STP/LMT); `protective_children` (reduce side, confirmed qty, empty without a level); `breached_levels` long/short, sl/tp, mark == level; `breach_closes` closes-first | REGRESSION | none |
| S3 | sim places the child; level persisted on the held book; next-cycle breach → synthetic close; `--dry-run` writes nothing (zero DDL) | REGRESSION | mock portfolio fixture (two revisions) |
| S4 | `parse_working_order` on a captured **stop** row; `cancel_order` DELETE shape + post-cancel absence; `_cancel_protection` fail-closed; close refused when the cancel is unconfirmable; `_check_levels` → `UNPROTECTED`; exit codes 0/3 | REGRESSION | **new captures** (below) |
| S5 | full Paper round trip | live only | Paper account |

**Captures required (new — the existing fixtures are the contract model to copy):**

- `gateway_stop_working_orders.json` — a real `STP` we submitted, from
  `GET /iserver/account/orders`, proving: `order_ref` echoed, `orderType` /
  `origOrderType` spelling, **where the trigger price appears**, the `tif` value
  for GTC, `status`.
- `gateway_stop_submit_reply.json` — the `POST /orders` response for an `STP`,
  proving whether it returns an `order_id` directly or an `o354`-style confirm
  prompt (which must still abort).
- `gateway_cancel_order.json` — the `DELETE /order/{id}` 200 body, plus the same
  order **absent** from a subsequent `GET /orders` (proves the confirmation rule).
- Existing `gateway_trades.json` should be **extended** with a current-scheme ref
  echoed after a real fill (already an open item in `todo.md`) so stop fills
  replay/attribute correctly.

Validated **only by running:** `ibkr live run <cfg> --adapter sim` and
`--adapter sim --dry-run` (S3); `--adapter ibkr --dry-run` against a real gateway
(the read/mint path); and S5's real Paper cycle.

## 9. AGENTS.md edits (S5, on ship)

In § Live trading invariants, replace:

> "…the open guard may be excused only by our own already-confirmed reducing
> fills (a same-cycle flip) — do not widen that; stop-loss/take-profit and LMT are
> **refused**, never silently sent naked and never implied in the report; a
> partially filled entry is not topped up."

with:

> "…the open guard may be excused only by our own already-confirmed reducing
> fills (a same-cycle flip) — do not widen that. **Stop-loss is honoured by a
> resting `STP` child (`GTC`) placed only after the entry is confirmed filled,
> sized to the confirmed fill, and cancelled-and-confirmed before any close of
> its lot. A take-profit is enforced at the next cycle, never by a resting order
> (a second resting reduce order on one lot can flip the account). An entry
> ticket never carries a level; an LMT entry is still refused. Any level not
> proven resting — a refused/ambiguous child, or a breach with no working stop —
> is reported `unprotected` and exits 3, never implied.** A partially filled
> entry is still not topped up."

Add a numbered invariant: **"A protective child is cancelled-and-confirmed
before any close of its lot; an unconfirmable cancel refuses the close."**

Add to § Gateway contract: the `STP` trigger field (`price` vs `auxPrice`) and
GTC-on-STP are **unverified pending the Paper capture**; the capture fixtures are
the contract.

Add to § Live trading commands: `ibkr live abandon --leg <entry|stop|take_profit>`
with its escalated warning, plus a `cancel-stops` verb.

Add to § Where state lives: the `leg` column / widened intent PK.

Add the module list note for `src/live/protect.py` and
`src/live/adapters/ibkr/protect.py`.

## 10. Risks

**Verifiable offline:** identity/migration safety (refs unchanged); ticket bodies
against the spec; the pure breach predicate; the sim end to end;
cancel-before-close ordering; report/exit codes; the level round-trip.

**Requires a live gateway session:** the `STP` trigger field (`price` vs
`auxPrice`); `GTC` acceptance on `STP`; the working-orders shape for a stop; the
`DELETE` body and post-cancel absence; whether an `STP` draws a reply prompt;
`outsideRTH` semantics; and — inherently — that a live stop fills during RTH gaps
at a worse price than the backtest's level fill (a permanent, disclosed
divergence). Also unverified: whether the gateway accepts an `STP` whose qty came
from the fractional-rounding floor (it should: whole shares).

**Rollback.** `LiveConfig.allow_stops=False` is the default and reproduces
today's refusal byte-for-byte, so S1–S4 ship dark. Turning the flag back off
stops *new* children but **does not cancel already-resting `GTC` stops** — so the
rollback procedure is: flip the flag **and** run
`ibkr live cancel-stops --scope <s> --yes` (a ref-filtered, our-`order_ref`-only
cancel; print what will be acted on before acting). The adapter-level `dry_run`
flag remains defence in depth.

## 11. YAGNI — not built

`STP LMT`; `trailingAmt`/`trailingType`; a resting take-profit plus the OCA
bracket (`isSingleGroup`/`parentId`); a `price` column on `live_order_intent`; a
stop REPLACE path from the DSL (unreachable today, F1); LMT entry carry-over
(`PLAN_LMT_ORDERS.md`); top-up of a partial entry (`todo.md` P2).
