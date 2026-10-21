---
name: downsample
description: Migrate a working daily-bar (1d) strategy to an hourly (1h) EXECUTION grid while keeping entries and exits in their original daily-bar units.
---

# Downsample: 1d signal → 1h execution grid

## What this skill is (and the one thing it is NOT)

Porting a daily strategy to hourly is **not** "reinterpret every bar-count param
in hourly bars". That is the anti-pattern this skill exists to prevent. It is the
same error class as scaling an ATR multiple by bars-per-day, and it silently
rewrites the signal:

- `breakout_look: 20` on 1d means **"the month's high"** (20 daily bars ≈ 1
  month). Read as 20 hourly bars it becomes **"the 3-session high"**. Entry rule
  changed. A port that then swept `breakout_look ∈ [5,10,20,40,80,120,160]` and
  kept `20` because it scored best did not migrate anything — it **re-fit the
  signal and called it convergence**.

The rule: **keep every signal param in its original DAILY-bar units, compute the
signal on `1d`, and use the `1h` grid only to time the fill.** The daily rule is
the strategy; the hourly bar is the execution venue.

## Contract

- User points you at a strategy JSON config (path like `strats/pass/foo.json`).
- The original config is never modified — every step creates a NEW config under
  `strats/wip/`.
- The original strategy module is never modified. The port writes a NEW sibling
  module (see Step 2) — never an edit of the original.
- At each step, compare the canonical metric set (Sharpe, Ann, MaxDD, Kurt, Skew,
  Win, Trades, Scaled, Rejected) between the two runs. Read the FULL report —
  never `grep`/`head`/`tail` it (AGENTS.md hard rule 3).
- Nothing is "converged" until the **port checklist** below has been walked
  against the original, param-units included.

## The port checklist (do this BEFORE tuning anything)

A port that "converges" while missing a gate is not converged. Diff the ORIGINAL
module against the port line by line and enumerate every one of these. Write the
list down; do not do it from memory.

- [ ] **Every entry gate.** Breakout/compression/regime/breadth conditions — all
      present, same bar count, same units.
- [ ] **Every exit.** Divergence/slope exits, take-profit, time stops. A port
      that drops an exit is a different strategy.
- [ ] **Every sizing factor.** Risk fraction, amplifiers, concentration,
      portfolio-level derisking (VIX/vol-regime), book-slot caps. These are the
      most commonly dropped because they are often structural constants, not
      config fields.
- [ ] **Param UNITS.** For each bar-count param, state which interval it is read
      on. If the port reinterprets a daily window as an hourly one, that is a
      signal change — stop and fix it, do not sweep it back.
- [ ] **Bucket gating.** Every DAILY rule (entry AND exit AND any daily-derived
      level) must be gated to once per completed daily bucket. See Step 3.
- [ ] **Warm-up.** A `1d` signal needs daily-bar warm-up. If the port reads
      `1d` series, `warmup` must cover enough TRADING days for the slowest daily
      indicator, plus one bucket for completion lag.

A missing gate is the **most likely port defect** and the hardest to see: the
run still produces metrics and can look healthy. A real port silently dropped the
daily strategy's VIX-derisking block (the `_fred_*` factor; in
`vwatr_div_dsl.py` that is `_fred_series` + `_percentile_rank` +
`_fred_band_score`, ~60 lines of structural sizing) — a load-bearing gate, not a
config field, so no config diff surfaced it. It went unnoticed until a code diff.
Restoring it moved Sharpe 2.06 → 2.10 and MaxDD −10.24% → −8.78%. **Diff the
gate before you tune the knob.**

## Step 0 — Baseline the daily run

Run the original daily config over its full window/symbols. Record every metric.
Record `Scaled` and `Rejected` too: non-zero means the run is advisory across
symbol permutations (AGENTS.md § Fills), which you must carry into every
comparison.

## Step 1 — A single-symbol, short-window daily sanity run

Create a config that inherits the parent's `strategy_type`, `strategy_params`,
`commission`, `initial_capital` and `bars` (`["1d"]`). Override:

- `symbols`: **one** symbol from the parent (SKILL.md Stage 1).
- `trading_start`, `trading_end`: **≤ 1 year** (SKILL.md Stage 1 cap).
- `warmup`: at least as many **calendar days** as the parent — the engine
  converts warm-up to bars internally, so `"300d"` stays `"300d"` at any bar size.
- `name`: `<parent_name>_s1`.

Run `uv run ibkr bt run <config>`. It should reproduce the parent's behaviour on
that slice (same trade timing, roughly the same trades). If it does not, you have
not understood the parent yet — stop and read the module, do not proceed.

## Step 2 — The interval-faithful HTF port (the PRIMARY path)

This is the normal, expected port. Not a fallback.

- `bars`: `["1h", "1d"]` — `1h` is the base fill grid, `1d` is the signal feed.
- `strategy_type`: **the SAME module as the daily original.** Resolve the signal
  interval from the run, never from a second file:
  `_state_interval(ctx) -> None if ctx.interval == "1d" else "1d"`. `None`
  means the base feed IS the signal feed (daily venue); `"1d"` means an
  intraday base that only times the fill. Reference:
  `src/bt/strategies/vwatr_div_dsl.py`, which serves BOTH venues.
- `strategy_params`: **the original daily values, unchanged.** Params keep their
  daily-bar units; no `×7`.
- `warmup`: daily-bar warm-up large enough for the slowest `1d` indicator plus
  one completion-lag bucket.

In the module, every signal/state read goes through that resolver:

```python
si = _state_interval(ctx)        # None (base feed) or "1d"
close = ctx.ohlcv(sym, si).close
vwatr = ctx.ta.vwatr(sym, params.vwatr_period, interval=si)
```

The base (1h) bar is used for exactly one thing: **selecting the fill**. A daily
rule that goes true is emitted once, and fills at the next base bar's open.

A second module is a fork of the rule set and will drift from it — do not write
one. The parity check is two `bt run`s on the same code: one `["1d"]`, one
`["1h","1d"]`.

### Why the venue switch carries the approval

Re-pointing a strategy at `["1h","1d"]` changes the fills it takes, and AGENTS.md
hard rule 1 + SKILL.md's trade-by-trade review gate mean the result is
**reviewed trade by trade before it is trusted**. An intraday run is not
"config-only" in effect, so it cannot be waved through. Frame it as the expected
path and get the review; do not treat the extra work as a reason to fall back to
a bar-count-scaled `["1h"]`-only config.

### When bar-count scaling is even legitimate

Only for a param that is genuinely a **fill-grid** quantity: a cooldown measured
in bars between fills, or a decay applied per bar. It is NOT legitimate for a
signal window. If you cannot say which of the two a param is, it is a signal
window — leave it in daily units.

### If you do take a `["1h"]`-only route

It is a different strategy, not a port. Any bar-count param you scale changes the
entry/exit rule; any param you sweep to "recover" the metrics is re-fitting. Say
so explicitly in the report and treat the result as a new strategy.

## Step 3 — Once-per-bucket gating (entry AND exit)

A daily rule evaluated on every hourly bar re-fires ~7×/day. The daily decision
must be gated to once per **COMPLETED** daily bucket — and this applies to the
**EXIT path too**, not just entry.

**The real migration's worst bug:** entry was gated, the exit was not. The daily
`decel` divergence exit fired at the day's FIRST hourly bar instead of at the
daily close — exiting ~6h early and truncating winners. Measured signature:
**win rate UP 14.5pp while mean win DOWN 57%** (more, smaller wins + the tails
cut off). After gating the exit: Sharpe 1.66 → 1.98, kurtosis 22.4 → 5.1, trade
count 1059 → 792 (matching the daily original's 791 — the correct signature of a
port that stopped firing extra intraday exits). (Those "before" figures are the
migration's recorded measurements; they are the failure's fingerprint, cited as
evidence for the rule.)

### The corrected design

Reference: `src/bt/strategies/vwatr_div_dsl.py` (both venues, one module).

- **`_new_daily_bucket(ctx, holder, sym, n_daily)`** — the gate. Returns True
  unconditionally on the daily venue (`_state_interval(ctx) is None`), else
  compares the call's
  own `len(ctx.ta.<ind>(sym, interval="1d"))` against the last base bar's. True
  only on the first base bar of a newly-completed daily bucket. Because the
  engine fires `on_candle` on base bars and never on `1d` candles, the visible
  daily count is constant within a session and increments by exactly one on the
  next session's first base bar.
- **`_process_symbol`** — consumes the bucket flag once. The daily ENTRY is
  evaluated only when `new_bucket`; between buckets, entry is skipped entirely.
  A position exactly between its signal bar and its fill is what blocks a second
  open in the same bucket.
- **`_exit_daily`** — the daily EXIT path, called only on bucket bars. The
  `decel` divergence exit is a STATE rule read off daily VWATR, so it fires at
  the bucket close where a daily-bar run would read it.
- **`_arm_trail`** — recomputes the chandelier level from the DAILY read and
  calls `ctx.set_stops`. `set_stops` ratchets tighten-only, so re-arming an
  unchanged level is idempotent.
- **`_rearm_trail`** — the NON-bucket path. Deliberately omits the divergence
  exit and only refreshes the trail.

### The trail nuance: daily LEVEL, intrabar FIRING

The trail's **level** is a daily quantity (a daily VWATR distance off a daily
high-water mark) and is always recomputed from the daily read — the number is the
original's. Its **FIRING** is a price stop, not a state rule, and stays intrabar
(`ctx.set_stops` fires on the next base bar's low, gap-adjusted). So the trail is
re-armed on EVERY base bar from the daily-derived level. Gating the arm to the
bucket too would leave the level stale for a full session and stretch the
arm-to-fire gap the original does not have. The `decel` exit is the opposite: a
state rule, bucket-gated.

## Step 4 — HTF reads are lookahead-safe (why the gate works)

`ctx.ta.<ind>(sym, period, interval="1d")` and `ctx.ohlcv(sym, "1d")` expose
**COMPLETED BUCKETS ONLY** (Pine `lookahead=false`): a forming daily bucket is
invisible. Verify against `src/bt/strategies/ta_context.py` (HTF reads are
cursor-truncated `SeriesView`s) and `src/bt/engine/candle_store.py` —
`_visible()` applies the one-bucket completion lag
(`raw - 1 if raw > 0 else 0`) to every truncating read, and `latest`/`count`
apply the same lag to HTF intervals.

Two consequences:
1. The once-per-bucket guard works: the visible daily count only advances when a
   bucket has actually CLOSED, never mid-session.
2. Every HTF read lags one bucket by construction; a symbol gapped at the cursor
   may lag one more (the safe direction).

## Step 5 — Expand symbols, then window

Only after the entries AND exits have survived trade-by-trade review (SKILL.md
Stage 1 → user approval):

- Symbols 1 → 2 → 4 → 8 → … (all parent symbols last). Re-run `bt run` after
  each doubling and compare against the daily run on the SAME symbol set.
- Window 1y → 2y → 4y → full. Re-run after each doubling.

The doubling step where the edge breaks is the finding. A more-trades result is
expected on hourly bars; **better metrics are not automatically a better port** —
check the port checklist again at each step (a size/cohort difference can lift
Ann without any signal change).

## Step 6 — Validate with `bt split`, not `bt run` alone

`bt run` on the full window is a point estimate. **Require `bt split --folds 4`
at the final step** and read the OOS distribution, not just the mean. A port is
validated only when OOS holds.

```bash
uv run ibkr bt split strats/wip/<name>_hourly.json --folds 4 --workers 3
```

Report, for BOTH the daily original and the port: per-fold OOS Sharpe, mean and
min OOS Sharpe, IS→OOS decay, and `Scaled`/`Rejected`. `Scaled`/`Rejected`
non-zero = conditional on symbol order; never quote across permutations.

The migration's final result — hourly beats daily (Sharpe 2.24 vs 1.99, Ann
35.5% vs 35.6%, mean OOS Sharpe 1.78 vs 1.51, Min OOS −0.58 vs −0.72) — is
credible **only** because `bt split --folds 4` held OOS. The old skill's "use
`bt run` only" rule nearly led to banking a fitted point.

## Sizing knobs behave differently than you expect

- **`risk_pct` is a pure leverage dial — until cash exhaustion bites.** Keeping
  it identical preserves daily risk-per-trade, which is right, but **Ann is then
  not comparable unless you state what is held fixed.** Verified on the shipped
  hourly config (`bt sweep` over `risk_pct`):

  | `risk_pct` | Sharpe | Ann | MaxDD | Trades | Scaled | Rejected |
  |---|---|---|---|---|---|---|
  | 0.003 | 2.24 | 20.7% | −6.9% | 792 | 0 | 0 |
  | 0.005 | 2.24 | 35.5% | −11.0% | 792 | 0 | 0 |
  | 0.0075 | 2.08 | 47.0% | −16.0% | 791 | **64** | **9** |

  Sharpe is flat across the 2.5× range while Ann scales, **but** the top rung
  trips cohort cash scaling (64 scaled, 9 rejected) and Sharpe drops. "Identical
  `risk_pct`" is right for risk-per-trade; it does not make Ann comparable
  across a port whose sizing scale differs.
- **`risk_topk` (dollar-volume concentration) FLIPPED SIGN across a strategy
  change.** It hurt (Sharpe 1.94 → 1.67) before a VIX gate existed and helped
  after (median 1.78 → 1.99). An amplifier/concentration interaction is not
  portable: **re-test it in the port, never inherit it.**

## Failure signatures (recognise these)

- **(a) Win rate UP while mean win DOWN** → an exit is firing EARLY. The daily
  exit is being evaluated on a non-closed bucket (or an hourly close instead of a
  daily one). Gate the exit to the bucket. (Measured: +14.5pp win, −57% mean win
  — see Step 3.)
- **(b) "No trades"** → a param-unit mismatch (a window read on the wrong
  interval is far too long to ever trigger) or a MISSING gate (the entry
  condition can never be satisfied). Check the port checklist, not the sweep grid.
- **(c) Attractive MaxDD with collapsing capital utilization** → under-risking,
  not skill. Compare `Capital Utilization` against the daily original before
  believing a better drawdown. The same failure as deleting a gate and calling it
  convergence.
- **(d) Convergence that required sweeping a SIGNAL param** → re-fitting, not
  migration. A legitimately ported signal param is not tuned to make metrics
  match; the matching is supposed to come from the fill-venue change alone.

## ATR multiples: scale by `~sqrt(N)` — and if you took the HTF path, don't scale at all

**Two rules, pick by path.**

- **HTF path (Step 2): do NOT scale bar-count params, and do NOT scale ATR
  multiples either.** Every signal read is on `1d`, so every param (window and
  multiple alike) is a daily quantity and stays the daily value. `vwatr_mult`
  stays `3.0` because it multiplies a **daily** VWATR.
- **`["1h"]`-only route: ATR MULTIPLES scale by `~sqrt(N)`, NOT ×N, NOT ×1.**
  An ATR multiple is a per-bar price range expressed in the current bar's
  volatility. For a near-random walk, a range scales with `sqrt(time)`:

  ```
  ATR_hourly ~= ATR_daily / sqrt(N)      N = bars per session
  ```

  A multiple of ATR is a fixed price distance in the *current bar's*
  volatility, so the conversion is **`daily_value × sqrt(N)`**. For `N = 7`,
  `sqrt(7) ≈ 2.65`; **measured on real data the ratio is ~3.0** (2.49–3.28 across
  MSTR / PLTR / RKLB, hourly vs daily ATR, same window). Rule: **`daily_value ×
  ~3`**, then **confirm empirically and re-validate on every larger window** —
  the conversion is strategy- and feed-specific, not a constant to assume.

  For an ATR multiple, neither "keep identical" nor "×7" is right. An ATR
  **period** IS a window length, so on this route it scales by the bar count
  (`atr_period: 14 → 98`).

  Name every ATR-multiple param in the config and convert each. Missing one
  leaves that gate or stop too tight, which reads as "the signal never fires" —
  and you will then blame the wrong parameter.

### Anti-pattern: the ×7 over-scale (documented failure)

ATR is a per-bar *range*, and ranges scale with `sqrt(time)`. Multiplying an ATR
multiple by the **bar count** treats a range as if it scaled linearly with time —
it over-widens stops and under-risks.

Observed on `entropy_vp_breakdown`, 6-symbol / 2022-H1 slice (daily target:
14 trades, Sharpe 2.62, Ann 35.32%, Capital Utilization 15.6%):

| ATR conv. | trades | Sharpe | Ann%   | Cap Util% | Win%   |
| --------- | ------ | ------ | ------ | --------- | ------ |
| ×1 literal (broken) | 1  | -1.12 | -4.29  | 1.4  | 0.00   |
| ×2.65 (sqrt 7)      | 16 | 0.17  | -1.14  | 3.5  | 43.75  |
| **×3 (measured)**   | 19 | 1.05  | **6.13** | 3.4 | 57.89 |
| ×7 (over-scaled)    | 36 | 2.12  | 2.44   | **0.4** | 63.89 |

**The ×7 row is fake convergence:** utilization collapses to 0.4% against the
daily 15.6% (~2.8× under-risked), so the attractive MaxDD is an artifact of
barely risking anything. Trade count rises and Win rises, but Ann falls.

Measured ATR ratios that settle it (hourly ATR ÷ daily ATR, same window):
MSTR 3.20, PLTR 3.28, RKLB 2.49 — i.e. ~3, consistent with `sqrt(7) = 2.65`,
nowhere near 7.

### Rules for the `["1h"]`-only route

- Do **not** "fix" a dead hourly run by setting an ATR-multiple gate to `0`
  (disabling it). That deletes entry logic and calls the deletion convergence.
- Do **not** accept a variant whose improvement comes with collapsing capital
  utilization — check Capital Utilization against the daily run.
- Scale the multiple, then sweep it to confirm the scaled value is **binding** (a
  value past which the metric is byte-identical is a disabled gate wearing a
  number).
- Slice-fitting is a starting point only: re-validate on each larger window and
  report drift rather than silently re-fitting.

### Finding hours-per-session

The US session here stores **7** native 1h bars (15:30–21:00), so `N = 7`, not
the 6.5 a nominal session length implies. Verify it on the feed before scaling:

```bash
python3 -c "import sqlite3,datetime as d; c=sqlite3.connect('../data/ibkr.db');\
t0=int(d.datetime(2022,1,3).timestamp()*1000); t1=int(d.datetime(2022,1,8).timestamp()*1000);\
s=set(r[0] for r in c.execute('SELECT timestamp FROM candle WHERE ticker=\"AAPL\" AND timestamp BETWEEN ? AND ?',(t0,t1)));\
import collections; dd=collections.Counter(d.datetime.fromtimestamp(x/1000).date() for x in s);\
print(sorted(dd.values()))"
```

## Reporting

At every step report:

1. Config path and the strategy module (original vs sibling port).
2. Window and symbol count.
3. The canonical metric set: Sharpe, Ann, MaxDD, Kurt, Skew, Win, Trades,
   **Scaled, Rejected** — for both the daily original and the port.
4. The port-checklist diff (gates/ exits / sizing factors / param units / bucket
   gating), with anything missing called out.
5. Any adjustments made since the last step.
6. Converged or diverged, by how much (%) — and whether convergence required
   sweeping a SIGNAL param (failure signature (d)).
7. At the final step, the `bt split --folds 4` per-fold OOS Sharpe + mean/min +
   IS→OOS decay.

## Rules

- Never modify the original strategy module; the port is a NEW sibling module.
- Never modify the original config; every step writes a new config under
  `strats/wip/`.
- Never test a strategy — validate by RUNNING it. No `test_<strategy>.py`.
- Read the full output, never `grep`/`head`/`tail` it.
- Use `bt run` for comparison runs; **require `bt split --folds 4` at the final
  step.** `bt sweep`/`bt optimize` only for a named, justified search — never to
  make a ported signal param "match".
- Keep signal params in their original DAILY units. The `1h` grid is the fill
  venue, not a re-parameterization.
- Diff every gate, exit and sizing factor BEFORE tuning anything.
