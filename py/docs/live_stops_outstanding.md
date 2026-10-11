# Live stops — outstanding work

Status of `docs/SPEC_LIVE_STOPS.md` implementation after the independent review.
Done: step 1 (drop `live_order_intent`), steps 2–3 (vocabulary + pure bracket
layer), step 4 (sim end-to-end), step 5 (IBKR placement outline). The reviewer's
go/no-go verdict: **sim path GO offline; live IBKR entry-fill NO-GO until the
Paper confirm capture; re-arm/backstop channel NO-GO (unwired).**

Everything below is still open. Order roughly by ship-blocking.

## 1. Step 6 — Paper capture of the confirm-reply shape [BLOCKS live]

- The only live evidence (`gateway_stop_submit_confirmed.json`) wraps the confirm
  success under `{"reply": {...}}`. `classify_reply` now unwraps a top-level
  `"reply"` mapping, so the code works for BOTH the wrapped and a flat
  `{"order_id": ...}` reply — but which one the real gateway returns is still a
  guess until a Paper cycle confirms it.
- Run the full Step 6 Paper round trip (`--adapter ibkr`, paper account) and
  lock the actual reply body for an `STP` submit + confirm.

## 2. Per-cycle naked backstop + D5 level-channel are un-wired [M2, blocks live]

- `level_bracket_intents` and `_route_protective`/`_replace_bracket`/`_cancel_pair_before_close`
  subsist but nothing PRODUCES a `protective` intent: no `stop_update` signal
  channel (spec D5) and no `BracketIntent` source reach `pipeline.py`.
- Consequence: only the entry-fill bracket (`_protect_after_fill`) runs. A
  bracket that is rejected, cancelled externally, or fires-partially (sibling
  cancelled) AFTER the entry cycle leaves the lot **unprotected with a clean
  exit 0** — a HOLD cycle never re-checks it.
- `_verify_resting` exists but has **no caller**. Wire it (or an equivalent
  re-arm pass) every cycle for every level-carrying lot; report `UNPROTECTED`
  + exit 3 when the pair is missing/not `Submitted`.
- This is spec §4.2 "missing ⇒ naked, loudly" and plan D1.

## 3. `position_side` means two different things [m1]

- Spec Step 2 / `order_side` (and `sim fire_stops._stop_intent`) treat
  `position_side` as the LOT side; the protective emitters (`protect._child`,
  `_wire_side`, sim `wire_side`) treat it as the REDUCE side (the lot's
  opposite).
- Safe today only because each path builds its own intents and the wired
  entry-fill path never feeds a child through `order_side`. Any future reuse
  (synthetic close of a child, the backstop, a re-arm routed through reconcile)
  silently INVERTS the wire BUY/SELL.
- Fix: one meaning (spec says lot side). Rename the emitters' field to
  `reduce_side` and align all consumers + docstrings, or make `order_side` take
  the semantic it is given.

## 4. Lone-leg bracket is not a "pair" anywhere [m2]

- `resting_bracket` returns `None` unless BOTH a stop and a take leg read
  `Submitted`, so a legitimate lone-`sl` bracket (vwatr_div_dsl arms only `sl`)
  is never "resting":
  - `_verify_resting` would report it NAKED forever;
  - `_replace_bracket` would re-place it (duplicate);
  - `_cancel_pair_before_close` would cancel nothing before a close → a live
    reduce order keeps leaning on a lot being closed.
- The lone-leg success path is now crash-free (M1 fixed in
  `_re_read_resting`), but the pair-vs-single policy above is undecided.
- Fix: decide single-leg vs pair explicitly per level-set and make
  `resting_bracket`/`_verify_resting`/cancel handle both.

## 5. `position_side` / level carry on the IBKR position read

- `src/live/adapters/ibkr/portfolio_source.py` maps `stop_loss`/`take_profit`
  from a broker row; `trades.py` `_open_row` still hard-codes them `None`. The
  derived book therefore loses a lot's advisory level across cycles — the very
  source the ratchet (never widen) compares against.
- Plan F4: the level must ride the fill/lot write so `level_bracket_intents`
  ("lot's advisory level") is real, not always `None`.

## 6. F2 residual: a level-carrying strategy is refused per-order, and the kind
  gate is now admitted — verify the refusal never regresses to exit 0

- `validate_order` now admits protective children (kind-driven). Confirm there
  is NO remaining path where a level-carrying intent is dropped as
  `OrderOutcome.REJECTED` (safe, exit 0) rather than `UNPROTECTED` (unsafe, exit
  3). The old `UnsupportedStopOrder` refusal was exactly that (plan F2).

## 7. Sim parity checkpoints (Step 4 is GO; keep it in lockstep)

- `src/live/adapters/sim/protect.py` models place-pair / fire / sibling-cancel /
  REPLACE. Keep it feeding the SAME pure `protect.py` builders (`bracket_children`,
  `child_ref`, `_reduce_side`) so the sim cannot drift from the gate logic the
  ibkr edge ships.
- The sim broker never reads an `orders[]` resting queue outside its own bracket
  model — OK as the broker, but do not let it certify a wire shape only a Paper
  capture can.

## Notes

- The confirm unwrap (`classify_reply`), lone-leg `_re_read_resting` fix, and
  cancel-before-synthetic-close are DONE (reviewer C1/M1/n1). Re-review targets:
  1, 2, 3, 4 above.
- Every item here is a follow-up to `cd8e327` / `docs/SPEC_LIVE_STOPS.md`.