# Live stops — outstanding work

Status of `docs/SPEC_LIVE_STOPS.md` implementation after the independent review.
Done: step 1 (drop `live_order_intent`), steps 2–3 (vocabulary + pure bracket
layer), step 4 (sim end-to-end), step 5 (IBKR placement outline), items 2, 3, 4
and 8 below. The reviewer's go/no-go verdict:
**sim path GO offline; live IBKR entry-fill NO-GO until the Paper confirm
capture; re-arm/backstop channel NO-GO (unwired).**

The re-arm/backstop channel is now WIRED (`attest_protection`, item 2 below) —
what still blocks live is the Paper confirm capture (item 1).

## 1. Step 6 — Paper capture of the confirm-reply shape [BLOCKS live]

- The only live evidence (`gateway_stop_submit_confirmed.json`) wraps the confirm
  success under `{"reply": {...}}`. `classify_reply` now unwraps a top-level
  `"reply"` mapping, so the code works for BOTH the wrapped and a flat
  `{"order_id": ...}` reply — but which one the real gateway returns is still a
  guess until a Paper cycle confirms it.
- Run the full Step 6 Paper round trip (`--adapter ibkr`, paper account) and
  lock the actual reply body for an `STP` submit + confirm.

## 2. [DONE] Per-cycle naked backstop is WIRED (spec §4.2 "missing ⇒ naked, loudly", plan D1)

Was: only the entry-fill bracket (`_protect_after_fill`) ran; a bracket rejected,
cancelled externally, or fired-partially AFTER the entry cycle left the lot
unprotected with a clean exit 0 — a HOLD cycle never re-checked it, and
`_verify_resting` had no caller.

Fixed: `LiveAdapter.attest_protection(book)` is the per-cycle backstop — it
reads the working orders ONCE (via the ibkr edge's `_working_index()`; the sim
store on the sim edge) and, for every OPEN lot the scope's own book carries,
reports `UNPROTECTED` unless a `Submitted` protective leg of ours rests
(`unprotected_targets`/`resting_legs`, the SAME shared pure rule on both
edges). The pipeline runs it after `read_portfolio` (before resync, so a sim
stop firing in-resync is protection working, not a naked lot) and the rows flow
into the cycle's results, so a naked lot exits 3 — never a clean exit 0.

The pass ATTESTs only (never POSTs), and a failed working-orders read reports
`UNPROTECTED` for every target — ambiguity is never clean, and no POST follows it
(invariant 1). Ownership stays scope-clean: only conids in OUR book are checked.

D5's `stop_update` signal channel (nothing PRODUCES a `protective` intent) is
still open as a separate follow-up; the backstop it was bundled with is done.

## 3. [DONE] `position_side` means one thing (the LOT side) [m1]

Was: spec Step 2 / `order_side` / sim `fire_stops._stop_intent` read
`position_side` as the LOT side while the protective emitters (`protect._child`,
ibkr `_wire_side`, sim `wire_side`) read it as the REDUCE side. Fixed by making
all of them mean the lot side: `_child` now mirrors `entry.action`, both edges
derive the wire reduce side by inverting it (`ibkr/protect._wire_side`, sim
`protect.wire_side`), and the now-unused `_reduce_side` builder is gone. Wire
behaviour is unchanged on every wired path (a long-lot child still SELLs);
what disappears is the inversion any future consumer would have hit.

## 4. [DONE] Lone-leg bracket is protection — a pair is only for pair-flows

Was: `resting_bracket` returned `None` unless BOTH a stop and a take leg read
`Submitted`, so a legitimate lone-`sl` bracket (vwatr_div_dsl arms only `sl`)
was never "resting": `_verify_resting` reported it NAKED forever, `_replace_bracket`
re-placed it (duplicate), and `_cancel_pair_before_close` cancelled nothing
before a close.

Fixed: `resting_legs(working, *, scope_tag, conid) -> tuple[WorkingOrder, ...]`
— I/O-free, ours-by-prefix + conid + `Submitted` + known kind off `orderType`,
all legs agreeing on the wire side, 0..n legs. `resting_bracket` stays the
pair-only reader (built on the same rule) for flows that need a full pair.
Every single-leg caller now acts on whatever actually rests: `_verify_resting`
is replaced by the item-2 backstop, `_replace_bracket`/`_cancel_pair_before_close`
cancel any count (via `_cancel_legs`), `_submit_children` re-reports a conid
whose resting legs already cover every child kind (never a duplicate POST), the
sim `apply_bracket` replaces a lone leg, and `resting_rows` stays pair-only with
`sim_resting_legs` for the any-count reader.

Also found and fixed en route: `_cancel_legs` and `cancel_order` confirmed
absence against the working index keyed by `order_ref` (always "absent") — they
now check the broker's actual `orderId`s, so an unconfirmable cancel genuinely
refuses (invariant 6).

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
  `child_ref`, `resting_legs`, `unprotected_targets`) so the sim cannot drift
  from the gate logic the ibkr edge ships.
- The sim broker never reads an `orders[]` resting queue outside its own bracket
  model — OK as the broker, but do not let it certify a wire shape only a Paper
  capture can.

## 8. [DONE-AS-WARNING] The IBKR divergence oracle is tautological

`divergence.guard_divergence` cannot fire on the ibkr path: the pipeline passes
`pf.portfolio`, and `portfolio_source.build_snapshot` builds that book from OUR
own scope-tagged sqlite rows, so it is compared against `book_from_executions`
of the same rows. It caught (via `ibkr live pf`) a doubled PLTR lot
(`ours=82 account=41`, one spurious `import-<conid>-<side>` seed row) that a
cycle reported as clean, exit 0.

Fixed as a WARNING: `build_snapshot` now compares our book against the
account's OWN `/positions` net for every conid we hold and emits a warning on a
disagreement (a conid absent from the read reads as flat). The warning rides the
`PortfolioSnapshot` into `CycleReport.warnings` and is rendered (text +
`-F json`), and is logged. It is deliberately NOT a divergence/exit 3: a shared
account can legally carry a conid another scope or a human moved.

Also done here: `live run -F json` now carries `divergences` + `warnings`, and
`unsafe_outcomes` names each divergence, so a machine consumer sees WHY a cycle
exited 3 and any advisory finding.

## Notes

- The confirm unwrap (`classify_reply`), lone-leg `_re_read_resting` fix, and
  cancel-before-synthetic-close are DONE (reviewer C1/M1/n1). Items 2, 3, 4
  and 8 are DONE. Re-review targets: 1 (Paper confirm capture), 5, 6, 7 above.
- Item 2's backstop rides the new `LiveAdapter.attest_protection` seam; the sim
  edge feeds the SAME shared pure builders (`resting_legs`,
  `unprotected_targets`), so the rule cannot fork (item 7's parity checkpoints
  hold for the backstop too).
- Every item here is a follow-up to `cd8e327` / `docs/SPEC_LIVE_STOPS.md`.