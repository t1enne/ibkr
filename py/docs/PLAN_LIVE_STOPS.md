# Plan — live stop-loss / take-profit

Status: **design, nothing implemented.**
Read `AGENTS.md` (invariants, `src/live` layout, size/type rules) and `todo.md`
(D3, D17) first. Companion plans: `PLAN_LIVE_HEXAGONAL.md`,
`PLAN_LMT_ORDERS.md`. Pros/cons history → `OPEN_QUESTIONS_PROS_CONS.md`.

> **Refactor impact (live-ledger refactor) — this plan needs a schema pass before
> it is actionable.** The plan predates the refactor and its level-home premise is
> now invalid: `live_position` (and `live_cash`) no longer exist, and the book is
> DERIVED from `live_execution` via `book_rows_from_executions`. There is no
> `live_position.stop_loss/take_profit` row for a level to land on, and
> `ledger._lot_fields` is gone. The design choice "the level's home is the
> POSITION row" (F4, §2 D8) must be re-decided against the three-table schema
> (`live_strategy` / `live_execution` / `live_order_intent`) — e.g. a level column
> on `live_execution` (additive, preserve-don't-drop) or a derived level read.
> Every `engine.py` reference below now points at `src/live/pipeline.py` (the thin
> `run_cycle` composer) or the stage it names. Do not implement this plan as
> written until the level-home question is settled.

## 0. Why

Level-carrying strategies are untradeable live: `adapters/ibkr/orders.py::
validate_order` raises `UnsupportedStopOrder` for any intent with
`stop_loss`/`take_profit`. The refusal is principled — this adapter places no
resting order, so a naked entry would silently drop the strategy's risk levels —
and `AGENTS.md` § Live trading encodes it. Decision: **honour the levels**,
always, with no opt-in flag.

Two organising choices shape everything below:

- **A stop is not a position-intent.** The level was to live on the position row
  (the pre-refactor `live_position.stop_loss/take_profit` — see the refactor note
  above, this home is now gone) and the resting order lives at the broker. A
  protective stop gets **no `live_order_intent` row** (§2 D2). The broker's
  working-orders read is the authority for what rests, and it is
  already invariant 1 that no POST happens after a failed read.
- **The DSL keeps both arming surfaces** — `ctx.long(sl=)`/`ctx.short(tp=)` at
  entry (fractions, made absolute at emit) and `ctx.set_stops(sym, sl=, tp=)`
  after entry (absolute prices, per lot, ratchet-only, `ActionType.stop_update`).
  Both funnel into the same protective mint; live mints the child from whichever
  the strategy used.

## 1. Findings that constrain the design

| #   | Finding                                                                                                                                                                                                                                                                                                                       | Consequence                                                                                                                                                                                                              |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| F1  | Levels arrive both at entry (fraction → absolute) and post-entry (`ctx.set_stops`, absolute).                                                                                                                                                                                                                                 | Level maintenance is a live requirement, not a nicety.                                                                                                                                                                   |
| F2  | The refusal exits **0**, not 1: `validate_order` → `_prepare_leg` → `feed_error("rejected")` → `OrderResult(REJECTED)`, absent from `_UNSAFE_OUTCOMES`.                                                                                                                                                                       | A level-carrying strategy is refused every cycle at exit 0; cron never alerts. Same shape as `todo.md` D3. Fix here, do not inherit.                                                                                     |
| F3  | Stop fills are `close`s of the parent lot (same conid, same sign), so `trades.reconcile` already reduces the row and stamps `closed_at`.                                                                                                                                                                                      | No `trades.py` change; a fired stop needs no intent row to be booked.                                                                                                                                                    |
| F4  | (Pre-refactor) `live_position.stop_loss/take_profit` were never written on the IBKR path: `trades._open_row` set them `None` and `ledger._lot_fields` hard-coded `"stop_loss": None, "take_profit": None` on every fold. **The `live_position` table and `_lot_fields` no longer exist** — this finding must be re-cast against the derived book (see the refactor note).                                                                                                                     | Was: the level's home is the POSITION row (no separate record, no new write seam). Now: the level home must be re-decided against `live_execution` (the entries' intent already carries the levels, so the fix is to let the fill/lot write carry them).                              |
| F5  | The live bridge **drops** level updates: `signals.ACTIONABLE = {long,short,close}`, the screen's `_side_of` returns `None` for `stop_update`, and `ScreenRow` is one row per symbol, latest posture only. `reconcile` additionally indexes its input as `by_symbol = {sig.symbol: sig …}` — last wins, one signal per symbol. | A level channel is required (§2 D5); it cannot ride the posture channel without forking `reconcile`'s input. The collector already _sees_ the signals, nothing downstream does.                                          |
| F6  | Live lots are per-conid (`portfolio_source` and the fill's `position_id=str(row.conid)`) — IBKR nets per contract. Backtest `position_id`s are not portable.                                                                                                                                                                                | Level work is **symbol-scoped**; N per-lot strategy updates collapse onto the symbol's one live lot.                                                                                                                     |
| F7  | `apply_stop_updates` is **ratchet-only** (long: sl `max`, tp `min`; short mirrored).                                                                                                                                                                                                                                          | Mirror it live — and it doubles as the safe live rule: never widen a working stop.                                                                                                                                       |
| F8  | `DELETE /iserver/account/{accountId}/order/{orderId}` returns `orderCancelSuccess`, whose own spec text says it does not report whether the cancel "can or will ultimately be enacted".                                                                                                                                       | Every cancel is CONFIRMED by a follow-up working-orders read; fail closed.                                                                                                                                               |
| F9  | A just-filled lot is not yet in the seeded book, so a protective child's side cannot be resolved from it.                                                                                                                                                                                                                     | The child carries its own `position_side`.                                                                                                                                                                               |
| F10 | `singleOrderSubmissionRequest` (`openapi.spec.json:37848`, authoritative for submit): `STP`, `tif=GTC`, `auxPrice`, `parentId`, `isSingleGroup` all expressible.                                                                                                                                                              | A resting STP child is placeable. Bracket/OCA is expressible but unverifiable offline (§8).                                                                                                                              |
| F11 | Without an intent row there is no attempt counter, so a ref built the entry way (`scope-token-attempt`) is IDENTICAL for a stop and its replacement — same key ⇒ same token ⇒ same cOID ⇒ IBKR's own dedupe swallows the replacement after the original was cancelled, leaving the lot unprotected.                           | The protective ref must be derived from `(lot, level)` so a tightened level mints a new ref while a re-send of the same level keeps the old one (§2 D2). This is the one amendment to invariant 2, protective legs only. |

## 2. Decisions

### D1 — Enforcement: resting `STP` child for `sl` + per-cycle backstop for both

On a confirmed entry fill the adapter places **one** sibling `STP` child
(`tif=GTC`), sized to the **confirmed filled** qty. `tp` gets no resting order —
it is enforced by the per-cycle breach backstop (reported as a price miss when it
fires between cycles). A level tightening cancels and re-places the child (D2).

Rejected: **pure resting** (a child can fail to exist — refusal, lost session,
partial fill, unconfirmable cancel — and nothing notices: exactly "silently
naked"); **pure per-cycle** (the inter-cycle gap is unprotected while the
backtest fires intrabar on `tick.low/high` — not equivalent, not "honoured");
**`tif=DAY`** (naked at every session boundary, silently); **bracket/OCA** (right
in principle, but needs a multi-leg reply loop, qty re-derivation after
`scale_open_cohort`, and is unverifiable offline — deferred).

**Two-sibling hazard.** Two resting reduce orders on one lot means the first to
fire leaves the other live, which then opens the opposite side. With `sl` only
resting there is no sibling; a resting `tp` ships only with `isSingleGroup` OCA,
after a Paper capture.

**Naked window.** Entry fill → child POST, inside one cycle. Child POST fails →
`UNPROTECTED`, exit 3. Never silent.

### D2 — A stop has no intent row: the position holds the level, the broker holds the order

Nothing about a protective order is persisted locally beyond the lot's level:

- **Levels** were to live on `live_position.stop_loss/take_profit` — position
  fields, never a record of their own. **That table is gone** (refactor note):
  the level home must move to the derived book. They ride on the `ExecutionRecord`
  the pure planner already builds (`stop_loss`/`take_profit`/`tag` sourced from
  `result.intent`) and were to be stamped by the lot write, `ledger._lot_fields`,
  which hard-coded them to `None` (F4) and no longer exists. A fill-less re-arm
  would touch the same fields. `trades._apply` / `book_rows_from_executions`
  preserve them across replays; `IbkrPortfolioSource._to_position` already maps
  them onto `Position`.
- **No `live_order_intent` row for a stop.** Consequences, all deliberate:
  - no `leg` discriminator and **no PK migration** — the intent table is
    untouched;
  - no `attempt` counter, so the ref rule changes (below);
  - `abandon --leg` disappears: our only stop handle is broker-side, so
    abandoning a stop IS `cancel-stops` (ref-prefix filtered, our `order_ref`s
    only);
  - no stop `UNRESOLVED` state. Ambiguity is not _lost_ — it is resolved by the
    next working-orders read. The placing cycle must then report `UNPROTECTED`
    (exit 3), never "placed";
  - no local provenance for a stop placement (the cycle report and the gateway
    are the audit) and no `stuck_cycles` for it. The alarm becomes repetition:
    every cycle that cannot confirm a stop for a levelled lot exits 3.
- **The broker is the authority for what rests.** The re-mint guard becomes: no
  protective POST without a successful working-orders read (invariant 1), and
  never place a stop for a lot+kind that already has one. A read that cannot
  answer places nothing and reports `UNPROTECTED`.
- **REPLACE (level tightened):** read working orders → locate ours (our
  `order_ref` prefix, matching conid, kind) → `DELETE` its `order_id` → confirm
  absence with a second read (F8) → POST the new child. An unconfirmable cancel
  means **do not submit** and report `UNPROTECTED`, exit 3: a live stop plus a
  replacement is over-cover, and cancelling without replacing leaves the lot
  naked — neither may be silent.
- **The protective ref (the one amendment to invariant 2).** A stop's ref is
  derived from the LOT and the LEVEL, not from a counter:
  `scope_tag` + a token over `(symbol, "stop", conid, lot-anchored time, level)`.
  Same lot and same level ⇒ the same ref ⇒ IBKR's dedupe protects a genuine
  double-send; a tightened level ⇒ a new ref ⇒ the replacement is not swallowed
  (F11). The lot-anchored time is the lot's own open/fill time — never the
  decision bar or the wall clock, so a sibling in the batch cannot move it. The
  exact token inputs are pinned at implementation; the requirement is _distinct
  per (lot, level), stable for a re-send of the same (lot, level)_.
- **Kind from the wire.** Which resting order is the stop-loss vs a future
  take-profit limit is read off the working row (`orderType`, and the trigger
  field once captured, §5) — never assumed from our own book.

### D3 — Close guard / same-cycle flip: no new allowance

A fired stop is replayed by `fetch()` **before** `resync`/`reconcile`, so the lot
is already closed when reconcile diffs posture and emits no close. If the fill is
not yet visible, account net (flat) ≠ booked (+qty) + delta (0) and
`_open_exposure_guard` **refuses** — intended: a fill we cannot see may be live.
`_confirmed_reduce_delta` credits only this cycle's placement loop; a
broker-initiated fill is never credited. Unchanged.

**`_cancel_protection` runs before every entry-leg close** (and every synthetic
breach close). That is the disambiguator: with the stop cancelled and confirmed
absent, a synthetic close cannot race a live stop; if the stop already fired, the
close guard refuses on a flat account. An unconfirmable cancel refuses the close
(exit 3) — a live stop plus a close would over-reduce.

"Intended to open but the stop fired": replay closes the lot → `cur=flat`,
`tgt=long` → an ordinary re-entry, not a special case.

### D4 — Order types: add `STP`; entries stay MKT; LMT entries still refused

`src/exec/types.py`: `OrderType.STP = "STP"`. No `STP_LMT` (adds a "triggered but
limit not hit, therefore naked" mode). `validate_order` becomes **kind-driven**:
an `OrderIntent` carries its `order_type` and a `protective: bool` marker, and
the rules are

| intent                 | rule                                                                                                                                               |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| entry, `MKT`           | allowed (unchanged)                                                                                                                                |
| entry, `LMT`           | refused `UnsupportedOrderType` (carry-over unbuilt)                                                                                                |
| entry carrying a level | allowed: the entry ticket itself **never** carries a level — the levels are placed as protective children after the entry is confirmed filled (D1) |
| protective, `STP`      | requires `stop_loss` finite > 0, `tif` `"GTC"`, reduce side                                                                                        |
| protective, `LMT`      | requires `take_profit` finite > 0 (unused until OCA ships)                                                                                         |

Ticket bodies, one table-driven builder (the tif is chosen by the kind — a stop
is GTC because it must outlive the session, and with no intent row there is no
rollover rule to satisfy):

```python
entry:        {"conid": int, "side": "BUY"|"SELL", "quantity": float, "orderType": "MKT", "tif": "DAY", "cOID": ref}
stop:         {"conid": int, "side": reduce_side, "quantity": float, "orderType": "STP", "price": sl, "tif": "GTC", "cOID": ref}
take_profit:  {"conid": int, "side": reduce_side, "quantity": float, "orderType": "LMT", "price": tp, "tif": "GTC", "cOID": ref}
```

Pinned unverified, each by a required Paper capture (§5), never guessed silently:
the `STP` trigger field (`price` vs `auxPrice` — the plan pins
`_STOP_TRIGGER_FIELD = "price"`, marked unverified); `GTC` acceptance on `STP`;
the working-orders row for a stop (its `order_ref`, `orderType`, trigger field,
`tif`, status — this is what D2's "kind from the wire" and adoption both read);
the `DELETE` body and post-cancel absence; whether an `STP` trip draws an
`o354`/price-band reply (which must still abort); `outsideRTH` (left absent).

### D5 — Level updates: a separate pure step, not a second signal kind

`stop_update` never reaches the live layer today (F5, F6). Plumb it as its own
intent producer rather than another `LiveSignal` kind.

1. `src/live/protect.py` gains `level_intents(config_path, book, levels) ->
tuple[OrderIntent, ...]` (pure): for each symbol with both a level update and
   a live lot, diff the update against the lot's persisted level and emit a
   protective `STP` intent when (a) the lot is unlevelled and the update arms it,
   (b) the lot has a level but nothing rests for it, or (c) the new level
   **tightens** and a stop rests (REPLACE, D2). A level on a flat symbol is
   dropped silently, mirroring `apply_stop_updates`' skip on no-match. The
   ratchet gate (F7) lives here, so a wider level from a replayed bar can never
   widen a live stop; `tp` follows the same rule (long: `min`).
2. `SignalSource` — `run_cycle`'s EXISTING injected seam (`src/live/pipeline.py`,
   stage `screen_signals`: `signal_source: SignalSource = live_signals`) — gains a
   second method returning the strategy's latest `stop_update` per symbol.
   `SignalCollector` already records every generated signal, so this exposes what
   is recorded; no pipeline change, and the `_flush_bar` drain already keeps them
   out of fills.
3. `execute_cohort` concatenates: `intents = (*reconcile(signals, book, config,
owned), *level_intents(...))` (the `diff_intents`/`execute_cohort` stage in
   `src/live/pipeline.py`). Level intents bypass reconcile entirely.

Why not a `SignalAction["stops"]` + `reconcile` step: `reconcile` consumes one
signal per symbol (`by_symbol`, F5) and a stops signal would shadow that symbol's
posture signal; it also reads pre-settlement posture, so a symbol that also
closes in the same cycle would be armed and closed in one cohort. Both problems
vanish when the level step reads the live book directly.

Invariants for the step (state in its docstring):

- Levels are **symbol-scoped** (F6); the strategy's `position_id` is not portable.
- It targets lots that **already exist**. A lot filled in this same cycle is
  `_protect`'s job (D1), so the entry/level race never arises here.
- The level is the strategy's own advisory price off its simulated book, like
  every other live intent field — never derived from the live entry price.
- The protective intent's `action` stays `close`: that is what the reducing-order
  predicates read (`order_side`, `whole_quantity`'s floor, `placement_order`,
  `plan_result_writes`). Nothing is keyed on it, and a stop has no ledger row, so
  a plain close and its stop cannot collide (D2).

### D6 — Sim parity

The sim has no resting orders and no clock between cycles, so it **cannot**
represent the resting-stop path — say so in the docstring. It **can** exercise
offline end to end: level persistence on the held book, the shared pure breach
predicate, protective minting, and the `level_intents` diff.
`SimulatedBroker` places the synthetic child as an `OrderResult` (PLACED, message
naming the level), persists the levels on its settled lot, reports the child from
its own working read (the sim store IS the broker here, so "trust the broker"
means trusting that store), and drives the same `breached_levels` → synthetic
close. Backtest agreement for entries is unchanged (same
`settle_cohort`/`apply_fills`).

### D7 — Report / exit-code contract

New `OrderOutcome.UNPROTECTED`, added to `_UNSAFE_OUTCOMES`.

| event                                      | row                                                                                                    | exit  |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------ | ----- |
| entry filled, stop placed GTC              | `order AAPL stop PLACED qty=100 @ 92.10 GTC cOID=… order_id=…`                                         | 0     |
| stop found resting at cycle start          | `stop AAPL 92.10 resting (o…) cOID=…`                                                                  | 0     |
| stop fired                                 | resync `FILLED`, `close AAPL qty=100 @ 91.80 … stop`                                                   | 0     |
| level tightened                            | `stop AAPL 94.00 PLACED (replaces 92.10, cancelled o…, cOID=…)`                                        | 0     |
| level widened                              | no intent (ratchet) — cycle silent                                                                     | 0     |
| stop **refused** by gateway                | `UNPROTECTED: stop 92.10 NOT resting (…) — position is NAKED`                                          | **3** |
| stop submit ambiguous                      | `UNPROTECTED: stop 92.10 unconfirmed; not claimed as resting`                                          | **3** |
| lot carries a level, nothing rests         | `UNPROTECTED` **+** synthetic close `PLACED`                                                           | **3** |
| TP breached between cycles                 | close `PLACED`, `tp 105.00 breached (mark 106.2) between cycles; no resting tp; closing at market now` | 0     |
| lot closed deliberately, stop cancelled OK | close `PLACED` + `stop 92.10 cancelled (o…)`                                                           | 0     |
| close refused: cancel unconfirmable        | close `UNRESOLVED` `refused close: stop 92.10 could not be cancelled-confirmed`                        | **3** |

The `UnsupportedStopOrder` refusal is deleted, not re-pointed: `validate_order`
is kind-driven and always admits the protective bodies, so a level-carrying
strategy can no longer be refused every cycle at exit 0 (F2). What replaces it is
the `UNPROTECTED` row — a level that is not proven resting.

### D8 — No migration

Written pre-refactor: the plan assumed nothing in the schema changed, because
`live_position.stop_loss/take_profit` already existed and merely started being
populated (F4). **That premise is void** — `live_position` is gone and the level
home is unresolved (refactor note). Whatever home the re-decision picks (e.g. an
additive level column on `live_execution`) WILL need a live-schema change;
`live_order_intent` stays untouched, since a stop has no row (D2). No `leg`
column, no PK rebuild, no `tif` field, no `price` column: the resting trigger is
read back from the broker.

## 3. Types and signatures (new/changed only)

```python
# src/exec/types.py
class OrderType(Enum):
    MKT = "MKT"; LMT = "LMT"; STP = "STP"      # STP NEW

# src/live/types.py
@dataclass(frozen=True)
class OrderIntent:
    ...                                        # unchanged fields
    protective: bool = False                   # NEW: a reduce order protecting a lot (never keyed,
                                               # never persisted; D2). Sets the stop/tp ticket shape.
    position_side: ActionType | None = None    # NEW: the LOT side, set on a child minted from a
                                               # just-filled entry (F9). None elsewhere.

@dataclass(frozen=True)
class LevelBreach:                             # NEW
    symbol: str; position_id: str; kind: Literal["stop", "take_profit"]
    qty: float; level: float; mark: float

# src/live/pure_plan.py  (EXISTING planner — no new write type, no extra branch)
ExecutionRecord.stop_loss / take_profit / tag     # NEW fields, sourced from result.intent

# src/live/ledger.py  (level field/column stamped on the fill; `_lot_fields` deleted)
# ...stamps those fields onto the derived lot (home re-decided — see refactor note)

# src/live/protect.py  (NEW — pure, no I/O)
def protective_children(entry: OrderIntent, *, conid: int, filled_qty: float,
                        tif: str = "GTC") -> tuple[OrderIntent, ...]: ...
    # sl only (tp is backstop-enforced); empty when the entry carries no level;
    # reduce side = opposite of entry.action; position_side set; qty = CONFIRMED fill.
def breached_levels(lots: Sequence[Position],
                    marks: Mapping[str, float] | None = None) -> tuple[LevelBreach, ...]: ...
    # mark is the only intrabar proxy available between cycles:
    # long low<=level / short high>=level; marks default to each lot's last_price.
def breach_closes(breaches: tuple[LevelBreach, ...], transition: str) -> list[OrderIntent]: ...
def level_intents(config_path: str, book: PortfolioView,
                  levels: Sequence[TradeSignal]) -> tuple[OrderIntent, ...]: ...
    # the level step (D5): symbol-scoped, ratchet-gated, protective STP; empty when the
    # symbol is flat, unchanged or wider. Consumes the SignalSource method below.
def protective_ref(scope_tag: str, *, symbol: str, conid: int, anchor: pd.Timestamp,
                   level: float) -> str: ...
    # the ref rule (D2, F11): distinct per (lot, level), stable for a re-send of one.
def resting_stop(working: Sequence[WorkingOrder], *, scope_tag: str, conid: int,
                 kind: Literal["stop", "take_profit"]) -> WorkingOrder | None: ...
    # ours-by-ref-prefix + conid + kind, read off the wire (D2). Pure: no book, no ledger.

# src/live/pipeline.py  (was engine.py)
class SignalSource(Protocol):                   # EXISTING seam, gains one method
    def __call__(self, config_path: str, max_age_days: int | None = None) -> tuple[LiveSignal, ...]: ...
    def levels(self, config_path: str) -> tuple[TradeSignal, ...]: ...   # NEW: latest stop_update per symbol

# src/live/adapters/ibkr/orders.py
def validate_order(intent: OrderIntent) -> None: ...   # kind-driven, D4; no opt-in flag
def build_ticket(intent: OrderIntent, *, conid: int, side: OrderSide, order_ref: str) -> Ticket: ...

# src/live/adapters/ibkr/protect.py  (NEW — class-level mixin; IbkrBroker is far over the 150-LOC budget)
async def _protect(entry, result, conid, side) -> tuple[OrderResult, ...]: ...
async def _cancel_protection(symbol, position_id) -> Result[tuple[OrderResult, ...], FeedError]: ...
async def _check_levels(working) -> tuple[OrderResult, ...]: ...   # reports UNPROTECTED, never places

# src/data/ibkr/client.py
async def cancel_order(self, account: str, order_id: str) -> dict[str, Any]: ...
    # DELETE iserver/account/{account}/order/{order_id}; same error mapping as get/post.

# src/live/adapter.py  (LiveAdapter Protocol; sim adapter mirrors it)
async def cancel_order(self, order_id: str) -> Result[None, FeedError]: ...   # NEW member
```

## 4. Ordered slices (each independently verifiable)

1. **S1 — vocabulary + level persistence, zero behaviour change.** `OrderType.STP`,
   `OrderIntent.protective`/`position_side`, the level fields on `ExecutionRecord`
   - the fill/lot write (levels start landing on the derived lot; `_lot_fields` is
     deleted, so the home is re-decided — refactor note), and the
     `SignalSource.levels` method **declared but unconsumed** (additive schema if a
     new level column is needed; see D8).
     _Verify:_ a level round-trips through a recorded fill onto the derived lot and
     is preserved across a replay; `live_order_intent` gains no rows for protective
     intents; suite green.
2. **S2 — pure layer + level step.** `src/live/protect.py` (`protective_children`,
   `breached_levels`, `breach_closes`, `level_intents`, `protective_ref`,
   `resting_stop`); kind-driven `validate_order`/`build_ticket`; the `_place_all`
   concat. _Verify:_ tests on the wire ticket bodies (trigger field, `orderType`,
   `tif`); an entry ticket still carries no level; `level_intents` tighten vs
   widen vs flat vs no-lot; `protective_ref` distinct per level and stable for a
   re-send; `resting_stop` picks ours and ignores a foreign scope's.
3. **S3 — sim end to end, offline.** `SimulatedBroker` places the child, persists
   levels, exposes the child from its own working read, enforces the shared
   predicate, applies the REPLACE diff. _Verify:_ `ibkr live run <cfg> --adapter
sim` (two fixtures) and `--dry-run`; assert the child row, the level
   round-trip, the tighten/replace, the breach close.
4. **S4 — IBKR placement.** The protect mixin, `_cancel_protection`,
   `cancel_order`, `_check_levels`, `UNPROTECTED`. _Verify:_ respx over
   **captured fixtures** (§5); pipeline exit-3 rows.
5. **S5 — Paper capture → local run → rewrite the invariants (§6).** Requires a
   live Paper session.

## 5. Test plan

| slice | tests                                                                                                                                                                                                                                                                                                                                                | kind       | gate                                  |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ------------------------------------- |
| S1    | level round-trip through the lot write; preserved on replay; no intent row for a protective intent                                                                                                                                                                                                                                                   | REGRESSION | `tmp_path` sqlite                     |
| S2    | `validate_order` per kind; `build_ticket` bodies (MKT/STP/LMT); `protective_children` (side, confirmed qty, empty without a level); `breached_levels` long/short, sl/tp, mark == level; `level_intents` tighten/widen/flat/no-lot; `protective_ref` per-level distinct and same-level stable; `resting_stop` ours-only; `breach_closes` closes-first | REGRESSION | none                                  |
| S3    | sim places the child; level persisted; next-cycle breach → synthetic close; REPLACE diff; `--dry-run` writes nothing (zero DDL)                                                                                                                                                                                                                      | REGRESSION | mock portfolio fixture, two snapshots |
| S4    | `parse_working_order` on a captured **stop** row; `cancel_order` DELETE shape + post-cancel absence; `_cancel_protection` fail-closed; close refused on an unconfirmable cancel; `_check_levels` → `UNPROTECTED`; exits 0/3                                                                                                                          | REGRESSION | new captures below                    |
| S5    | full Paper round trip                                                                                                                                                                                                                                                                                                                                | live only  | Paper account                         |

**Captures required** (new; the existing fixtures are the contract model to copy):

- `gateway_stop_working_orders.json` — a real `STP` we submitted, from `GET
/iserver/account/orders`: our `order_ref` echoed, `orderType`/`origOrderType`
  spelling, **where the trigger price appears**, the `tif` value for GTC,
  `status`. This fixture is now load-bearing twice over — it is how we know a
  stop rests at all (D2) and how we tell the kind from the wire.
- `gateway_stop_submit_reply.json` — the `POST /orders` reply for an `STP`:
  direct `order_id` or an `o354`-style confirm prompt (which must still abort).
- `gateway_cancel_order.json` — the `DELETE /order/{id}` 200 body, plus the same
  order **absent** from a subsequent `GET /orders` (proves the confirmation rule).
- Extend `gateway_trades.json` with a current-scheme ref echoed after a real fill
  (already an open item in `todo.md`) so stop fills replay/attribute correctly.

Validated **only by running:** `--adapter sim` and `--adapter sim --dry-run` (S3);
`--adapter ibkr --dry-run` against a real gateway (read/mint path); and S5's Paper
cycle.

## 6. AGENTS.md edits (S5, on ship)

In § Live trading invariants, replace the stop/LMT clause with:

> …the open guard may be excused only by our own already-confirmed reducing fills
> (a same-cycle flip) — do not widen that. **Stop-loss is honoured by a resting
> `STP` child (`GTC`) placed only after the entry is confirmed filled, sized to
> the confirmed fill, re-triggered on a strategy level tightening (never widened),
> and cancelled-and-confirmed before any close of its lot. A take-profit is
> enforced at the next cycle, never by a resting order (a second resting reduce
> order on one lot can flip the account). An entry ticket never carries a level;
> an LMT entry is still refused. Any level not proven resting — a refused or
> ambiguous child, or a breach with no working stop — is reported `unprotected`
> and exits 3, never implied.** A partially filled entry is still not topped up.

Amend invariant 2 (the order ref) for protective legs: **a stop's ref is derived
from `(lot, level)` — distinct per level so a replacement is never swallowed by
the broker's cOID dedupe, stable for a re-send of the same level — never from a
decision bar or the wall clock.** Amending invariant 4 (the only owner of OPEN
state): **a protective stop is the one order whose OPEN state is the BROKER's,
not `live_order_intent`'s — it has no row. It may be placed only after a
successful working-orders read, never for a lot+kind that already rests, and its
cancel is confirmed by a re-read before any replacement or close.**

Add a numbered invariant: **"A protective child is cancelled-and-confirmed before
any close of its lot; an unconfirmable cancel refuses the close."**

Add to § Gateway contract: the `STP` trigger field (`price` vs `auxPrice`) and
GTC-on-STP are **unverified pending the Paper capture**; the capture fixtures are
the contract, and the working-orders row is the sole authority for a stop's
existence and kind.

Add to § Live trading commands: `ibkr live cancel-stops --scope <s> --yes`
(ref-prefix filtered, our `order_ref`s only; print what will be acted on before
acting) — which is now also how a stop is abandoned, since there is no intent row
to clear.
Add to § Where state lives: a protective stop has **no** row; the lot's level is
the local trace, and the broker is the order's owner.

## 7. Risks and rollback

**Verifiable offline:** the protective ref rule; ticket bodies against the spec;
the pure breach predicate and the ratchet gate; `resting_stop` attribution; the
level round-trip; the sim end to end; cancel-before-close ordering; report/exit
codes.

**Needs a live gateway session:** the `STP` trigger field; `GTC` on `STP`; the
stop's working-orders shape (now doubling as D2's existence-and-kind check); the
`DELETE` body and post-cancel absence; whether an `STP` draws a reply prompt;
`outsideRTH`; and — inherently — that a live stop fills in an RTH gap at a worse
price than the backtest's level fill (permanent, disclosed). Also unverified: an
`STP` whose qty came from the fractional-rounding floor.

**Accepted, disclosed losses** from D2: no local record of a stop placement (the
report and the gateway are the audit), and no `stuck_cycles` counter to surface a
wedged stop — the signal is repetition, since every cycle that cannot confirm a
stop exits 3.

**Rollback.** There is no flag to flip. To stand the edge down: `ibkr live
cancel-stops --scope <s> --yes` (which clears the resting stops, since nothing
local remembers them) **and** revoke the placement path (S4 revert).
`cancel-stops` alone is not a rollback — the next cycle would arm them again. The
adapter-level `dry_run` flag stays as defence in depth.

## 8. YAGNI — not built

`STP LMT`; `trailingAmt`/`trailingType`; a resting take-profit plus the OCA
bracket (`isSingleGroup`/`parentId`); a `price` column anywhere — the resting
trigger is read back from the broker, never stored; LMT entry carry-over
(`PLAN_LMT_ORDERS.md`); top-up of a partial entry (`todo.md` P2); per-lot live
level identity (F6 makes it degenerate — one net lot per symbol).
