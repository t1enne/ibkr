# Plan — LMT orders, gaps and unfilled outcomes (rev 1)

Split out of `docs/PLAN_LIVE_HEXAGONAL.md` §7.1 (2026-10-05). The hexagonal plan
ships MKT only; everything that has to exist before a limit order can rest across
bars lives here. Pros/cons history stays in `docs/OPEN_QUESTIONS_PROS_CONS.md`
(lean: "MKT for the liquid universe, `LMT` opt-in per config once carry-over
exists. Do not ship `LMT` without a cancel/reprice policy.").

## 0. What already exists (shipped with the shared core, phase 0)

| Piece | Where | State |
| --- | --- | --- |
| `OrderType.MKT \| LMT`, `limit_price`, `TimeInForce.DAY \| IOC` | `src/exec/types.py` | shipped |
| `match_price` / `match_bar` (pure candle matcher) | `src/exec/matching.py` | shipped, table-tested |
| `SimExchange.submit` / `cancel` / `match_bar` (staged order surface) | `src/bt/exchange/sim.py` | shipped, table-tested |
| `FeedError.kind` `unfilled` | `src/live/types.py` | shipped |

What is NOT shipped, and is the actual subject of this plan:

- **No bt run path ever emits an LMT.** `src/bt/engine/backtest.py` builds MKT
  signals only; `SimExchange.submit` stages orders but nothing drains a resting
  queue across bars. The LMT surface is exercised by tables, not by `bt run`.
- **No unfilled outcome is reported.** `bt run|sweep|split|optimize` draw their
  columns from `src/bt/report_metrics.py` (Sharpe, Ann, MaxDD, Kurt, Skew, Win,
  Trades, Scaled, Rejected) — there is no `Unfilled`. An intent that never fills
  currently has nowhere to appear, so "the backtest reports it as an unfilled
  order" (hexagonal plan §2) is aspirational text, not behavior.
- **No live LMT.** The IBKR adapter refuses a non-MKT intent, and the live edge
  has no cancel/modify, no open-order book, and no reprice path.

## 1. The gap problem

`match_price` today (buy):

```python
return min(limit, open_) if low <= limit else None
```

A bar whose open gapped *through* the limit in our favour therefore fills at the
**open**, which is better than the limit. This was a live contradiction: the code
is the realistic side, and `src/exec/matching.py`'s docstring and hexagonal plan
§2 both said the opposite ("a bar that gapped below the limit is not improved
on"). **Resolved (L0, prose+test only): the R realistic convention wins** — the
prose was rewritten to state the shipped rule for BOTH sides (buy: `min(limit,
open)`; sell: `max(limit, open)`), and the matcher table now pins the shipped
truth (an exact-touch bar `low == limit` for a buy DOES fill, and an LMT with
`limit_price=None` never fills). No behavior change.

The honest choices, per side (buy shown; sell mirrors):

| Convention | Rule | Effect |
| --- | --- | --- |
| **R realistic** (what the code does) | `min(limit, open)` on touch | Matches a real matching engine. A gap through the limit silently `limit − open` better than planned, so the run is flattered exactly where gaps are largest |
| **C conservative** | fill at `limit` whenever `low <= limit` | Never overstates the edge; understates it. Diverges from live by the gap subsidy, in the safe direction |
| **T strict-touch** | require `low < limit` (strict) | Removes the same-price-touch ambiguity about whether the queue reached us — orthogonal to R/C, can be combined |

**Recommendation:** ship **R** (one shared core means one fill semantic, and the
live side will really fill at the open), and make the subsidy *visible* rather
than assumed: report, per run, how many fills were gap-improved and the total
`Σ(limit − fill)`. If the edge only survives with the subsidy, the run says so
instead of hiding it. Keep **T** as a table-tested edge case, not a knob.
A conservative-mode flag is phase L4 at the earliest — every knob is a fitted
degree of freedom (AGENTS § Alpha research).

Related gap math that must not be double-counted:

- `fill_guard_price` / gap-through handling already exists for the next-open MKT
  path; an LMT that fills *at* the open and a MKT that fills at the open must not
  apply the guard twice.
- Friction stays the adapter's job (`SimExchange.match_bar` applies spread /
  slippage / commission); the matcher returns the frictionless base price. An
  unfilled order pays **no** friction and no commission.

## 2. The unfilled problem

An intent that does not fill is a real outcome, and three things must not happen
silently: the trade not happening, the strategy's state advancing as if it had,
and the cash cohort pretending the order is gone.

**L1 — bar-scoped (`tif=DAY`), no carry-over.** The order lives in exactly one
bar. Unfilled → the order dies there and is *reported*, never re-queued. Requires:

- a resting-order drain in the engine loop: orders submitted at the fill bar are
  matched at that bar; a `None` produces an `Unfilled` record (order ref, symbol,
  side, qty, limit, bar ts, reason `no_touch`), and the order is dropped;
- `Unfilled` added to the canonical metric set in `src/bt/report_metrics.py`, so
  text columns and JSON keys cannot drift, plus the detail rows under `--trades`;
- strategy state must not advance on an unfilled intent: the cooldown / trail /
  `ctx.shared` bookkeeping that a `long` would have advanced stays untouched, and
  the next bar may re-emit the same intent. A re-emitted intent is fine here
  because a dropped DAY order has no identity left at the broker; `order_ref`
  uniqueness is a *live* constraint (§4), not a backtest one;
- cohort cash: an unfilled order consumes no cash and is not a `Scaled`/`Rejected`
  member. Because the cohort set now depends on which orders filled, `Scaled`
  becomes load-bearing in a new way — the AGENTS "results are advisory across
  symbol permutations" caveat widens rather than narrows.

**L2 — resting orders across bars.** This is what breaks the "one cycle = one
bar" assumption, and is the reason the split is separate from the MKT plan.
Needs, each an explicit decision:

1. **Expiry.** `GTC` forever, `n`-bars, or until the strategy's own signal flips.
   Default proposal: an explicit `expire_bars` per strategy, no unbounded rests;
   a `GTC` that can outlive the run's window is a research trap.
2. **Matching priority inside a bar.** Resting orders (older intent) match before
   same-bar signals (new intent), so an add and a fresh open cannot both consume
   the same cash in an undetermined order.
3. **Late-fill semantics.** The strategy learns of a fill several bars after the
   signal; SL/TP set on the original intent must be attached at *fill* time and
   re-based on the fill price, not on the signal bar.
4. **Strategy state on a late fill.** The intent was emitted N bars ago; a
   cooldown/trail that already advanced must be reconciled with a fill that now
   arrives. Cheapest correct rule: an unfilled intent advances nothing (L1), so a
   late fill is the only thing that advances state, at the fill bar.
5. **Partial fills.** The sim decides all-or-nothing (a single bar's OHLC cannot
   honestly split a quantity). Live will return partials (`cum_fill` < size); the
   book must apply the partial and keep the remainder resting. State the
   asymmetry in the report rather than pretending the sim models it.

## 3. Reporting changes (bt)

- `Unfilled` joins the canonical set in `src/bt/report_metrics.py`.
- Gap subsidy counters from §1 (`fills_gap_improved`, `Σ(limit − fill)`).
- `--trades` detail gains unfilled rows with the reason and the bar.
- Non-goal: any change to MKT fills, friction, or the `Scaled`/`Rejected`
  definitions. `bt run` on MKT-only configs must stay byte-identical — the
  phase-1 golden-parity guarantee is re-verified at every L-step.

## 4. Live side (blocked on the hexagonal plan's D2)

Order matters; none of this can ship before the decision it depends on.

1. **`Broker` / `LiveBroker` convergence** (hexagonal plan deviation D2, an
   explicit phase-4 decision). `OrderRequest.limit_price` / `tif` must reach the
   live edge before an LMT can be sent.
2. **Refs vs reprice.** `order_ref(scope, cycle_ts, seq)` is idempotent per cycle
   by design, and IBKR dedupes on `cOID`. A *reprice* is a genuinely new order for
   the same intent, so a reprice must mint a new ref (new `cycle_ts` or a new
   `seq` keyed on intent identity) — a reprice that reuses the ref is deduped into
   silence. The cOID scheme needs a stated reprice rule before live LMT.
3. **Open-order book.** Resting orders must be durable (which refs are open, at
   what limit, for which scope/lot) so a cycle can cancel or reprice what an
   earlier cycle placed. This is metadata, not qty/price truth: the broker's
   `/iserver/account/orders` remains the authority on what is working.
4. **Cancel / modify.** IBKR cancel + modify endpoints, dry-run refusal, and the
   ambiguous-submit path already reading `/iserver/account/orders` so a working
   order is *seen* rather than duplicated.
5. **Unfilled accounting.** A DAY LMT that never fills is cancelled at the day
   boundary and reported; a partial fill is booked from `cum_fill` (never
   `size`, which is remaining).
6. **Stops while an entry rests.** A resting entry has no lot yet, so a resting
   `STP` cannot be attached to it — resting stops stay with the MKT path until
   L2 late-fill semantics exist.

## 5. Phases

Each ships alone; MKT behavior must not move.

- **L0 — resolve the gap contradiction.** SHIPPED (prose + tests only). The
  wrong prose in `src/exec/matching.py` was rewritten to state the shipped
  convention (R); hexagonal plan §2 already carried no contradictory parenthetical
  (it points here), so nothing to fix there. The matcher table pins the shipped
  truth: an exact-touch bar (`low == limit` for a buy) DOES fill (the matcher is
  non-strict), and an LMT with `limit_price=None` never fills. No behavior change.
- **L1 — unfilled is first-class, DAY-scoped.** Engine drains staged orders at
  the fill bar, drops unfilled ones, records `Unfilled` + the gap subsidy; report
  metric + `--trades` rows. Proof: MKT-only configs byte-identical (golden
  parity), an LMT config reports the unfilled count with the reason.
- **L2 — resting orders.** Expiry, priority, late-fill SL/TP rebasing, partial
  asymmetry documented. Proof: `bt run` then `bt split --folds` on the resting
  config; the fold where the edge breaks is the finding.
- **L3 — live LMT.** After D2. Submit, cancel/modify, open-order book, dry-run
  path, per-scope attribution on the existing `scope`/`cOID` machinery.
- **L4 — reprice policy** and (only if a strategy demands it) a conservative-gap
  research flag.

## 6. Open decisions

1. **Gap convention R vs C** (§1). Recommendation: R + visible subsidy counter.
2. **Expiry representation** for a resting order: `expire_bars` param vs a
   `TimeInForce.GTC` with a run-level cap. Recommendation: `expire_bars`, so no
   order can outlive the run window.
3. **Strict touch** as a mandatory non-fill case or a flag. Recommendation: case
   only, never a knob.
4. **Where `Unfilled` sits relative to `Trades`** in the report — a count next to
   `Trades`, or a separate `Intents` column so `Trades + Unfilled = Intents` reads
   as an identity. Recommendation: the identity, it is self-checking.
5. **Partial fills in the sim**: keep all-or-nothing and say so, or model a
   volume-capped partial. Recommendation: keep all-or-nothing; a 1h OHLC bar
   cannot honestly split a fill.
