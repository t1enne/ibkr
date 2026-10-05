# Live execution — pros/cons for the open decisions

> **Status after rev 3** (`docs/PLAN_LIVE_HEXAGONAL.md`): the reasoning below still
> holds for the surviving questions; the rest are closed by the shared-execution
> redesign. Q1 → simplified (`order_ref`, no hashing). Q2 → resolved (commission from
> executions). Q3 → resolved (lots replay from executions). Q12 → resolved (no local
> ledger-of-truth, so no drift class). Q13 → resolved (one order per intent, no close
> grouping). Q14 → deferred to phase 4. Q4/5/6/7/8/9/10/11 remain open as written.

Companion to `docs/PLAN_LIVE_HEXAGONAL.md` §8. No decision is taken here; each
question lists its options with both sides and a lean.

## Q1 — cOID scheme (idempotency key)

| Option | Pros | Cons |
| --- | --- | --- |
| Deterministic from `strategy_id + symbol + action + qty + round(ref_price,4)` | Re-run safe: a crash between place and record re-sends the same key and IBKR dedupes. Retry after a `transport`/`timeout` `Err` cannot double-order. Cron is idempotent by construction. | Blocks a legitimate same-bar re-entry on the same symbol at the same size. Also fragile: a re-sized qty (equity moved between cycles) produces a NEW key, so the "dedupe" silently fails exactly when the book has changed. |
| Cycle nonce (`strategy_id + cycle_ts + seq`) | Always places the intended order. Re-entry never blocked. Key is unique per attempt so no false dedupe. | Re-run after a crash re-places an order that likely already landed. Requires reconciliation against `GET /iserver/account/orders` before placing, or you accept duplicate orders. |
| Bar-stamped + per-lot sequence (`strategy_id + bar_ts + symbol + lot_seq`) | Within one bar/one lot index the key is stable (safe re-run, safe retry). Next bar gets a fresh key, so a genuine re-entry works. Closes key on sorted `position_id`s, which is stable. | Needs a bar timestamp and a deterministic lot index in the intent — the live cycle knows its bar, but reconcile must pass it through. Still blocks two same-bar entries at the same lot index. |

Lean: bar-stamped + per-lot sequence. Deterministic-per-attempt keys are a false
comfort once sizing is equity-dependent.

## Q2 — fill price and commission source

| Option | Pros | Cons |
| --- | --- | --- |
| `orderStatus.avg_price`, commission `0.0` | One call, already needed for the fill gate. No extra parse surface. | Commission `0.0` makes live PnL optimistic versus the backtest's commission model, so live and backtest numbers are not comparable. `avg_price` is rounded, so per-lot cost basis drifts after grouping. |
| Parse `/iserver/account/trades` executions | Real commission and real per-execution price. Reconciles against TWS. | Second endpoint, own pagination and 7-day window. Execution-to-lot attribution is heuristic (order id, then time). More respx fixtures, one more failure mode in the cycle. |

Lean: keep `avg_price` for the fill, but take commission from `/iserver/account/trades`
when the call succeeds, and record `0.0` with a flag when it does not.

## Q4 — `LiveConfig.mode` doubles as gateway login toggle and live guard

| Option | Pros | Cons |
| --- | --- | --- |
| One field | Cannot drift: the guard and the login click the same switch. Nothing new in the config contract. | Cannot test a live-configured strategy against a paper account without editing the file. Couples a deployment concern (which session to open) to a strategy-declared risk posture. |
| Separate `login_mode` | Live config can be exercised on paper. Posture and session are independent concerns. | Two knobs can contradict. `login_mode=paper` with `mode=live` is ambiguous and dangerous unless the account-prefix guard is the real authority (it is). More config surface to validate and document. |

Lean: one field. The account-id prefix guard is the authority either way, so a second
knob adds surface without adding safety.

## Q5 — who brings the gateway up

| Option | Pros | Cons |
| --- | --- | --- |
| `run_cycle` calls `ensure_ready()` (config flag, default on) | One command does everything — matches the request. Cron-safe: a dead container is repaired in place. No stale-session surprise at order time. | Trading path acquires an infra side effect (`docker compose up`, login automation, up to 180s wait). Needs docker and playwright reachable from the trading host. Slow failure inside the cycle. Hard to unit-test; the manual checklist carries it. |
| Cron/`sync_market_data.py` owns it; live only keeps alive | Trading cycle stays fast and side-effect-light. One infra owner. Easier to test. | Two places must be correct. A failed login surfaces only when the first order is attempted. Contradicts the request as stated. |

Lean: flag with default on, so the single command works, and cron can set it off.
Never let `ensure_ready` mutate anything account-side — container and session only.

## Q6 — default order type

| Option | Pros | Cons |
| --- | --- | --- |
| `MKT` | Fill guaranteed, so the cycle ends with a settled book and no carry-over state. No unfilled-order bookkeeping. Simple tests. | Unbounded slippage. A wide-spread or thin name can fill far from `ref_price`, and the fill is what the ledger records. Worst case is an illiquid open at a gap. |
| `LMT` at `ref_price` ± spread | Price bounded, so slippage is a known constant. Aligns with the backtest's spread/slippage model. | May not fill. Needs reprice/carry-over policy, partial-fill handling, and a cancel path. Re-placing next cycle collides with the cOID dedupe. Orders can linger, so the "one cycle" model leaks state. |

Lean: `MKT` for the liquid universe, `LMT` opt-in per config once carry-over exists.
Do not ship `LMT` without a cancel/reprice policy.

## Q7 — live acknowledgement flag

| Option | Pros | Cons |
| --- | --- | --- |
| `--allow-live` on the command | Explicit per invocation and visible in the process list. Nothing sticky in the environment. Failing closed when forgotten is the safe direction. | Cron scripts bake it in, so it stops being a conscious act. No protection against a copy-pasted command. |
| Environment variable | Keeps the command line clean. | Invisible in process listings and audit. Sticky in `.env`, so the guard silently stays open. Can be set by an unrelated tool in the same shell. |

Lean: `--allow-live`, no environment variable.

## Q8 — `core/` subpackage

| Option | Pros | Cons |
| --- | --- | --- |
| Flat `src/live/` (no `core/`) | No rename churn in imports, tests, or docs. Files stay where they already are. | Purity is grep-enforceable only; nothing physically stops `engine.py` importing an adapter. Read order is less obvious to a newcomer. |
| `src/live/core/` | Boundary is visible in the import layout and reviewable in a diff. Clear read order: core, ports, adapters, app. | Rename churn across every test and import for five files. One more package level for little content. Module paths in the handoff doc drift. |

Lean: flat now. Introduce `core/` if and when a second reviewer needs the boundary.

## Q9 — account id resolution

| Option | Pros | Cons |
| --- | --- | --- |
| `config.account_id`, else first of `GET /portfolio/accounts` | Works with zero config for the single-account case. Explicit when several accounts exist. | Order of `GET /portfolio/accounts` is unspecified, so "first" can silently change. A multi-account deployment can route to the wrong book. |
| `config.account_id` required | No ambiguity, no surprise ordering. Multi-account is safe by construction. | Friction on every config, and duplicates information the gateway already knows. |

Lean: config first, but hard-fail when the listing returns more than one account and
the config is silent — never pick silently.

## Q10 — `adapter: "auto"` rule

| Option | Pros | Cons |
| --- | --- | --- |
| `"sim"` when `portfolio_path` is set, else `"ibkr"` | Existing sim configs keep working unchanged. Zero config for the common case. | Implicit. A sim-only fixture key decides which broker receives real orders — adding `portfolio_path` to a live file silently switches the broker. Surprising failure mode. |
| Explicit `adapter` required | No ambiguity, and the money path is always stated. | Breaks every existing config. Small friction per new file. |

Lean: explicit required for `mode: "live"`; `auto` allowed only for `mode: "paper"`.

## Q11 — widen the `FeedError.kind` literal

| Option | Pros | Cons |
| --- | --- | --- |
| Add `rejected` / `unfilled` / `timeout` / `position_drift` / `unknown_position` | Typed causes, exhaustive matching, callers branch on the real reason. `timeout` is distinguishable from `transport` for retry policy. | Contract change in `types.py`, so every matcher and test updates. Broker vocabulary (`rejected`, `unfilled`) leaks into a shared live type. |
| One `broker_error` kind plus `message` | No churn; the literal stays small. | Forces string matching or substring tests at every call site. Loses exhaustiveness, so a new cause is invisible to the type checker. |

Lean: widen. `position_drift` especially must be its own kind — it is the one error the
operator has to act on.

## Q12 — drift handling on a configured symbol

| Option | Pros | Cons |
| --- | --- | --- |
| Abort the cycle | Safe: no trade on a book that cannot be explained. Forces a human look at exactly the case that matters. No phantom closes. | One manual TWS trade halts all trading for the whole strategy, not just that symbol. With no alert path yet, cron fails silently forever. |
| Warn and trade the explained symbols only | Unrelated manual activity does not block the rest of the book. | Silent degradation. A symbol can stay excluded indefinitely with no signal. Partially-explained book while closes still fire elsewhere. |
| `--reconcile-close` to repair | Self-healing for lots closed outside the engine. Clears a halt without manual SQL. | Destructive assumption: it needs a price for the close. If the position is actually still open, the ledger lies and a later close targets nothing. Must never be the default. |
| Tiered: abort on average-cost drift; auto-repair only when lot sums are consistent with `net_qty` | Each case gets the response its evidence supports. Cost drift (unknown entry) aborts; quantity drift with a matching subset (a lot closed by hand) resolves. | Two code paths and a more complex test table. Tiering rules must be documented or they read as arbitrary. |

Lean: tiered, with abort as the fallback and `--reconcile-close` explicit and never
default. Add an alert path before trusting any of it in cron.

## Q13 — grouped close results

| Option | Pros | Cons |
| --- | --- | --- |
| `OrderResult.closed_position_ids` (one result per order) | One order maps to one result, mirroring broker truth. Ledger rows still all get closed. Order-level fields (message, id) are unambiguous. | New field on a shared type, and the engine must loop ids while the sim adapter returns per-lot results — an asymmetry between adapters. |
| One `OrderResult` per lot | Uniform with the sim adapter, so the engine loop is unchanged. | N results share one fill, so naive aggregation double-counts the fill in trade counts and metrics. Which lot the fill "belongs" to is arbitrary. |

Lean: the field, plus one `OrderResult` per lot carrying `ok=True` and the shared
`FillEvent` only on the first, so both adapters look the same to the engine.

## Q14 — resting per-lot protective stops

| Option | Pros | Cons |
| --- | --- | --- |
| Resting `STP` order per lot on open | Protection survives the gap between cycles. Closest match to backtest stop semantics, so live and backtest results stay comparable. Survives a dead cron. | Cancel-on-close bookkeeping, including a cancel that races the close order. Orphan stops if a close fails. Interaction with partial fills. Stop orders must be reconciled against TWS at cycle start, or the book has undocumented resting state. Lifecycle mocks in tests. |
| SL/TP evaluated at cycle boundaries only | Simple, no resting state, matches the one-cycle-per-invocation architecture. Nothing to reconcile. | An intraday gap can blow through the stop with no protection. Backtest results fill intrabar stops that live cannot, so live systematically diverges from the tested strategy — a correctness gap in the research contract, not just convenience. |
| Boundary evaluation now, resting stops as an explicit later step | Ships the simple version, with the divergence written down as a known limitation rather than discovered later. | The limitation is real: until the stop step lands, a strategy whose edge depends on the stop is not faithfully live-traded. |

Lean: boundary evaluation for v1, but only after deciding whether that divergence is
acceptable for the strategies actually scheduled — for a stop-driven strategy it is not.

## Safety framing

Q12 and Q14 are the safety-critical pair, and both fail in the direction that costs
money while looking healthy. A silent halt (Q12 abort with no alert) and an unhedged
stop (Q14 boundary-only) both read as "cycle succeeded". Whichever way they are
resolved, the resolution needs an alert path and a documented divergence, not just a
default in a config file.
