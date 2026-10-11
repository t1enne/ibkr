# Spec — broker-authoritative open state + resting SL/TP brackets

Status: **spec, ratified against a live gateway probe on account `DUM452563` (paper).
Nothing in `src/live` implements it yet.**

Goal: the broker is the **single source of truth** for every order it holds — an
open entry **and** a resting protective bracket. No local record decides whether
an order exists or may be placed again.

Two changes, in this order (each independently verifiable; Step 1 must land
first because it removes the record the later steps would otherwise have been
built on):

1. **Step 1 — broker-authoritative open state:** delete the durable open-order
   store (`live_order_intent`). Entries are re-minted off the union of two live
   reads, never off a remembered row.
2. **Steps 2–6 — resting SL/TP:** both legs of an OCA bracket rest at the
   broker, placed with the entry, level-read from the wire.

## 0. Wire contract (confirmed against a live gateway, paper)

Observed facts, not assumptions. These bodies/replies are the contract model for
tests (`src/live/adapters/ibkr/tests/fixtures/`).

- **Submit shape.** `POST /iserver/account/{acct}/orders` with
  `{"orders": [ticket…]}` — the array is wrapped in an `orders` key. A bare
  array is `400 Missing order parameters`.
- **STP trigger field is `price`, not `auxPrice`.** `auxPrice` alone on an `STP`
  returns `"error": "Invalid order price fields"`. Pure `STP` carries
  `price: <trigger>`; a `stop_price` field appears on the returned working row
  (mirror of the submitted `price`).
- **Submit is two-step (o354-style challenge).** An `STP` submission returns
  `200` with a challenge array
  `[{"id":"…","message":["You are about to submit a stop order…"],"messageIds":["o10331"],"isSuppressible":true}]`.
  The order does **not** rest until `POST /iserver/reply/{id}`
  with `{"confirmed": true}`. The confirmed reply carries `order_id`,
  `local_order_id` (= our `cOID`) and `oca_group_id`.
- **Confirm failure ⇒ order never rests.** A `503`/failure on the confirm leaves
  the order `status: "Inactive"` / `order_ccp_status: "Pending Submit"`. It is
  **not** submitted, appears in the working-orders read, and `DELETE` on it
  returns `400` (`order is inactive`). Treat any unconfirmed/ambiguous state as
  not-resting (`UNPROTECTED`), never "placed".
- **OCA works.** Two tickets in one `{"orders":[…]}` array, each with
  `"isSingleGroup": true`, rest as one bracket. Confirmed: an `STP` + an `LMT`
  on the same conid, same reduce side, same qty, both `Submitted`, sharing
  `oca_group_id` (e.g. `"oco-241885626"`).
- **Working-orders read contract** (`GET /iserver/account/orders`) — the row an
  order exposes:
  - `order_ref` — echoes our `cOID` verbatim (how we attribute ours).
  - `orderId` — the broker's id, needed for `DELETE`.
  - `stop_price` (STP trigger), `price` (LMT limit) — floats-as-strings.
  - `orderType`/`origOrderType`: `"Stop"/"STOP"` vs `"Limit"/"LIMIT"` — the
    SL-vs-TP discriminator; nothing is assumed from local state.
  - `timeInForce`: `"GTC"`.
  - `status` / `order_ccp_status`: `"Submitted"` means resting.
  - `remainingQuantity` / `totalSize`, `side`, `conidex`, `ticker`,
    `filledQuantity`.
- **`oca_group_id` is a reply-only field.** It is `null` on the working-orders
  read, so bracket membership is not re-readable. Identity is reconstructed from
  `order_ref` prefix + conid + side.
- **Cancel.** `DELETE /iserver/account/{acct}/order/{orderId}` on a resting
  order; the confirming evidence is a follow-up working-orders read showing the
  `orderId` absent.

---

# Step 1 — broker-authoritative open state: remove `live_order_intent`

**Must land first.** It strips the durable record, so every later step builds on
"the read is the truth" rather than "the table remembers".

## 1.1 The dedupe argument (why no durable row is needed)

Reconcile reads **orders + positions unconditionally**. A market entry we placed
satisfies at least one read at the instant we look:

| order state                                     | visible in                                              |
| ----------------------------------------------- | ------------------------------------------------------- |
| working / routing (`PreSubmitted`, `Submitted`) | orders read (working section)                           |
| filled this session                             | orders read (it also lists session-filled/cancelled)    |
| filled, prior/any session                       | positions read                                          |
| partially filled                                | orders read (`remainingQuantity`) + positions (partial) |
| expired DAY / cancelled / rejected              | neither read — the order is dead                        |

So the union `{orders read, positions read}` covers every live order we
placed. An **intent absent from both reads does not exist (or is dead) ⇒
re-mint is safe.** The failed-confirm case from the probe (`Inactive/Pending
Submit`) is still _inside_ the orders read — even the stuck case is visible, not
silent.

The one residual is **not** double-submit; it is fail-open:

> an orders/positions read that **errors** mid-cycle makes absence unproven.
> Concluding "not there" could re-ALIVE a live order. So: read error or
> ambiguity ⇒ **exit 3, never re-mint.** A skipped cycle is recoverable; a
> duplicate is not.

This is invariant 1 carried over unchanged.

## 1.2 Entry discipline that keeps the argument true

1. **Entries are `MKT` + `tif=DAY` only.** A resting `LMT`/`STP` entry could sit
   filled-in-neither-read across a session boundary and vanish from both — the
   exact hole the union cannot cover. Keep `LMT` entries refused.
2. **Reads are mandatory and unconditional** — orders and positions every cycle,
   before any POST (invariant 1).
3. **Read failure ⇒ fail closed** (exit 3), never "not there".

## 1.3 What is removed

- The `live_order_intent` table and its `IntentStore` (`src/live/ledger.py`).
- `PendingIntents` seam, `IntentRecord`, `IntentKey`, `ref_prefix`, `attempt_of`
  (the attempt counter embedding), `decision_ts` audit, adoption reconciliation
  (INV-4's adopt path).
- `ibkr live abandon` (nothing to clear — a wedged _market_ entry is resolved by
  the next read).
- `stuck_cycles` / `WEDGED_CYCLES` alarm (replaced by exit-code repetition).
- `UNRESOLVED` as a _persisted_ state (kept only as a per-cycle report entry).

## 1.4 The ref rule becomes pure

`cOID = scope_tag(scope) + token(symbol, action, position_id)`, no attempt tail.
A legitimate re-send reuses the **same** ref; IBKR's own `cOID` dedupe absorbs
it. Distinct orders are already distinct by key. This is a strict simplification
of the bar-free scheme — the token stays a pure function of the key, so adoption
(now just attribution) survives bars, days and reruns without a row.

## 1.5 What stays

- `live_execution` (the append-only fill fold) and the derived book — the fills
  are still the durable position record; _*only the open-order* intent_ store is
  removed.
- `live_strategy` (ownership, audit hash, cash seed).
- The cycle lease, `fetch()` replay-then-reconcile, divergence oracle.

## 1.6 Migration

One live-schema change: drop `live_order_intent`. This is **irreversible** —
the rows are order-intent history and cannot be reconstructed from the broker
(which is the whole point of removing them). Per the rename-never-drop rule's
guard pattern, `down()` is absent and `run_down` refuses loudly. The `up()` runs
only while the live file is migrated and the cycle lease is clear — no cycle may
be mid-flight during the drop.

## 1.7 Verify (Step 1)

- `make check` green after the delete; no test references `live_order_intent`.
- A reconcile trusts reads alone: flat book + no working order + no position ⇒
  a re-minted entry proceeds; a working order with our prefix ⇒ no re-mint;
  orders/positions read failure ⇒ exit 3 with no POST.
- Enter → crash mid-cycle → next cycle: the fill replays from `fetch()` and the
  position/live_execution reflect it; no double entry.
- Guarded drop migration applies; `--dry-run` still records zero DDL.

---

# Step 2 — protective vocabulary + wire fixtures

Zero behaviour change. Add `OrderType.STP`, `OrderIntent.protective` /
`position_side`, `BracketIntent`; `confirm()` and `cancel_order()` on the client;
captured fixtures for submit-challenge, confirmed reply, resting bracket read,
cancel. Nothing calls it yet. _Verify:_ suite green; fixtures parse.

```python
# src/exec/types.py
class OrderType(Enum):
    MKT = "MKT"; LMT = "LMT"; STP = "STP"        # STP NEW

# src/live/types.py
@dataclass(frozen=True)
class OrderIntent:
    ...                        # unchanged fields
    protective: bool = False      # NEW: reduce child protecting a lot; never keyed, never persisted.
    position_side: ActionType | None = None  # NEW: lot side on a child minted from a confirmed entry.

@dataclass(frozen=True)
class BracketIntent:             # NEW: one OCA pair.
    symbol: str; position_id: str
    sl: float | None; tp: float | None
    qty: float; side: ActionType
    refs: tuple[str, str]        # (sl_cOID, tp_cOID)
```

---

# Step 3 — pure bracket layer

`src/live/protect.py` and kind-driven order building. _Verify:_ ticket bodies
(`price` on STP, `price` on LMT, `isSingleGroup`, `GTC`); refs distinct per
level, stable for a re-send; `resting_bracket` picks ours and ignores a foreign
scope's.

```python
def bracket_children(entry: OrderIntent, *, conid: int, filled_qty: float,
                     tif: str = "GTC") -> list[OrderIntent]:
    # the pair; empty when the entry carries no level; reduce side = opposite of
    # entry.action; each order_type STP/LMT by kind; position_side set; qty = confirmed fill.
def protective_ref(scope_tag: str, *, symbol: str, conid: int,
                   anchor: pd.Timestamp, kind: Literal["stop","take_profit"],
                   level: float) -> str:
    # lot + level; distinct per level, stable for a re-send of the same level;
    # lot-anchored time (never the decision bar / wall clock).
def resting_bracket(working: Sequence[WorkingOrder], *, scope_tag: str,
                    conid: int) -> tuple[WorkingOrder, WorkingOrder] | None:
    # ours by ref-prefix + conid + side; kind read off orderType; None if either leg missing.
def level_bracket_intents(config_path, book, levels) -> list[OrderIntent]:
    # arm / tighten→REPLACE / ratchet-drop / flat-drop. Pure; no I/O.

# src/live/adapters/ibkr/orders.py
def validate_order(intent: OrderIntent) -> None:    # kind-driven; admits protective bodies.
def build_ticket(intent, *, conid, side, order_ref) -> Ticket:  # Step 0 ticket bodies.
```

---

# Step 4 — sim end-to-end (offline)

The sim store **is** the broker: it submits the pair, persists the OCA group,
exposes both legs from its own working read, cancels a leg when the sibling
triggers, and applies the REPLACE diff. _Verify:_ `ibkr live run <cfg>
--adapter sim` (two fixtures) and `--dry-run`; the pair, the level round-trip
through the sim broker, tighten/replace, sibling cancel on trigger.
Simulation cannot represent the resting-stop clock behaviour — say so in the
docstring.

---

# Step 5 — IBKR placement

`src/live/adapters/ibkr/protect.py` (mixin; the broker class is far over the
size budget). _Verify:_ respx over captured fixtures; pipeline exit-3 rows.

```python
async def _place_bracket(entry, filled_qty, conid, side) -> tuple[OrderResult, ...]:
    # submit pair ({'orders':[...], isSingleGroup}) → o354 confirm → re-read Submitted.
async def _replace_bracket(symbol, position_id, pair) -> ...:
    # cancel both legs → confirm absent → resubmit pair. Never one leg.
async def _cancel_bracket(symbol, position_id) -> ...:
    # cancel + confirm absent; before any close of the lot.
async def _verify_resting(working) -> tuple[OrderResult, ...]:
    # report UNPROTECTED, never place.

# src/data/ibkr/client.py — NEW
async def confirm(self, reply_id: str) -> Any: ...          # POST /iserver/reply/{id}
async def cancel_order(self, account: str, order_id: str) -> Any: ...  # DELETE, same error mapping.

# src/live/adapter.py (LiveAdapter protocol; sim mirrors it)
async def cancel_order(self, order_id: str) -> Result[None, FeedError]: ...  # NEW
```

New outcome `OrderOutcome.UNPROTECTED`, added to the unsafe set (exit 3). A
level that is not proven resting is never implied.

---

# Step 6 — Paper round trip

Full live cycle against Paper; leaves brackets in place for inspection then
cancels them.

---

# Design of the bracket (applies Steps 2–6)

## 4.1 Both legs rest, as one OCA bracket

```python
sl:  {"conid": int, "side": reduce_side, "quantity": float, "orderType": "STP",
      "price": sl, "tif": "GTC", "cOID": ref_sl, "isSingleGroup": true}
tp:  {"conid": int, "side": reduce_side, "quantity": float, "orderType": "LMT",
      "price": tp, "tif": "GTC", "cOID": ref_tp, "isSingleGroup": true}
```

- **No sibling flip:** OCA cancels the sibling when either leg fills — both can
  coexist on one lot without either opening the opposite side. Confirmed.
- **Placed inside the entry cycle**, immediately after the entry is confirmed
  filled. Gap = entry-fill → children-submit-and-confirm, within one cycle.
  Failure ⇒ `UNPROTECTED`, exit 3 — never silent.
- **Both legs GTC.** They outlive the session; with no local rollover rule there
  is nothing to re-arm.
- **Qty = entry's whole-share plan at submit.** OCA tickets are fixed at submit,
  so a **partial entry** leaves over-cover equal to (planned − filled), no top-up;
  disclosed in the report, never silently re-sized.

## 4.2 Broker is the single source of truth for levels

- No stop/take-profit column, no level stamp, no intent row, no OPTION. The
  broker's working row holds the level.
- Each cycle reads working orders first and attributes our bracket by
  `order_ref` prefix + conid + side. Level = `stop_price`/`price`; kind =
  `orderType`.
- **Missing ⇒ naked, loudly** — a lot whose bracket should rest but reads empty
  or unsubmitted ⇒ `UNPROTECTED` + synthetic market close, exit 3.
- **Re-mint guard** = no protective POST without a successful working-orders
  read, never place a bracket for a lot that already rests one.

## 4.3 Ratchet (never widen), from the wire

Compare the strategy's new level against the resting `stop_price`/`price`:
unlevelled + armed → place pair; resting + tightened → **REPLACE**; resting +
unchanged/wider → silent; flat symbol with a level → dropped silently.
REPLACE = cancel both legs → confirm absent via re-read → submit new pair → confirm.
Never one leg. An unconfirmable cancel ⇒ do not submit, `UNPROTECTED`, exit 3.

## 4.4 Confirm is part of placement

A bracket must pass the o354 challenge and then be **re-read as `Submitted`**
before it may be reported resting. Bounded retry; on exhaustion `UNPROTECTED`,
exit 3. (Invariant 1 extended: the confirm loop is gated by a read; nothing is
claimed resting without proof.)

---

# 5. List of invariants (must not be weakened)

1. Never POST after a failed **orders read**; the o354 confirm loop is also gated
   by a read. An orders/positions read that errors ⇒ exit 3, never "not there".
2. The entry ref is pure `(scope, symbol, action, position_id)` — no attempt
   counter; a re-send reuses it and IBKR's `cOID` dedupe absorbs it. A protective
   leg's ref derives from `(lot, level)` — distinct per level, stable per
   re-send — so a REPLACE is never swallowed as a duplicate.
3. Ambiguity is never reported as "not placed": unknown is `UNPROTECTED`, not
   `rejected`.
4. **There is no local owner of OPEN state.** The union of the broker's orders
   read + positions read is the authority. `live_execution` remains the durable
   fill book. Never place a protective bracket for a lot that already rests one.
5. Attribution matches our own `order_ref` by exact prefix. Ref-prefix + conid +
   side is the bracket's identity — never symbol+side alone, never a stored
   group id.
6. A protective child is cancelled-and-confirmed before any close of its lot; an
   unconfirmable cancel refuses the close (exit 3).

# 6. Test plan

| step | coverage                                                                                                                                                                                                                                                      | kind       | gate                        |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | --------------------------- |
| 1    | no `live_order_intent` refs; reconcile trusts reads alone; crash-replay no double entry; read error ⇒ exit 3 no POST; guarded drop applies; `--dry-run` zero DDL                                                                                              | REGRESSION | temp sqlite                 |
| 2    | fixtures parse; STP/LMT bodies; OCA group idle; no behaviour change                                                                                                                                                                                           | REGRESSION | none                        |
| 3    | `validate_order` per kind; ticket bodies; `bracket_children` (pair, side, confirmed qty, empty without a level); `protective_ref` distinct-per-level/stable-per-resend; `resting_bracket` ours-only-or-None; `level_bracket_intents` arm/replace/ratchet/flat | REGRESSION | none                        |
| 4    | sim places pair; sibling cancel on trigger; REPLACE; `--dry-run` zero DDL                                                                                                                                                                                     | REGRESSION | temp sqlite + two snapshots |
| 5    | submit challenge → confirm → `Submitted` re-read; confirm-lost → `UNPROTECTED`; cancel pair → confirm absent → resubmit; close refused when cancel unconfirmable; `_verify_resting` → `UNPROTECTED`; exits 0/3                                                | REGRESSION | captured fixtures           |
| 6    | full Paper round trip                                                                                                                                                                                                                                         | live only  | Paper account               |

**Captures required** (new, paper):

- `gateway_stop_submit_challenge.json` — o354 array (`id`, `messageIds:["o10331"]`).
- `gateway_stop_submit_confirmed.json` — confirm reply (`order_id`, `local_order_id`, `oca_group_id`).
- `gateway_bracket_working.json` — both legs resting, with `order_ref`, `stop_price`/`price`, `orderType` Stop/Limit, `timeInForce` GTC, `status`/`order_ccp_status` Submitted, qty.
- `gateway_cancel.json` — DELETE reply plus the order absent from a second read.

Validated only by running: `--adapter sim` (Step 4), `--adapter ibkr --dry-run`
(Step 5 read/mint path), and Step 6's Paper cycle.

# 7. Risks and rollback

**Verified offline (spec-carried):** STP trigger is `price`; OCA pair rests as
one group; the working row echoes `order_ref`/`stop_price`/`price`/`orderType`/
`timeInForce`; the o354 challenge; the unconfirmed-inactive row. **Needs Paper:**
bracket stop filling in an RTH gap at a worse price than the backtest level
(permanent, disclosed); partial-fill over-cover; `oca_group_id` reconstruction
limits; whether an `STP` replacement prompt re-fires on REPLACE.

**Accepted losses (Step 1):** no durable open-order record (the reads and the
fill book are the audit); re-mint is decided by read absence; a wedged entry
surfaces only as repeated exit-3; the drop of `live_order_intent` is
irreversible.

**Rollback.** Step 1's drop is one-way — revert means re-adding the table and
the IntentStore seam (rows unrecoverable, and the invariant reverts to "only
owner"). For brackets: `ibkr live cancel-brackets --scope <s> --yes` (cancels
both legs; prints what it will act on first) **and** revert the placement path
(Step 5). `cancel-brackets` alone is not a rollback — the next cycle re-arms.
The adapter-level dry-run stays as defence in depth.

# 8. Not built (YAGNI)

`STP LMT`; `trailingAmt`/`trailingType`; a local `price`/trigger column anywhere;
abandon-by-leg (rolling a bracket back = `cancel-brackets`); per-lot
multi-position identity (one net lot per conid makes the pair the unit); a
durable `oca_group_id` record (it is reply-only); the old attempt counter
(Step 1 removes the need).
