---
name: downsample
description: Systematic process to migrate a working 1d / daily-bar strategy to 1h / hourly, keeping entries and exits as close as possible.
---

# Downsample: 1d → 1h strategy migration

## Contract

- User points you at a strategy JSON config (path like `strats/pass/foo.json`).
- You never modify the original config — every step creates a NEW config file
  under `strats/wip/<name>_hourly.json`.
- The original strategy's code is never modified. You only change config fields
  (bars, symbols, dates, strategy_params) to adapt to hourly bars.
- At each step, compare metric outputs (Sharpe, Ann, MaxDD, Kurt, Skew, Win,
  Trades, total_return) between the two runs. Iterate parameter adjustments
  until the hourly run's metrics are close to the original daily run's over the
  same window + symbol set.

## Step 0 — Baseline the daily run

Run the original daily config over the full original window/symbols. Record
every metric. This is your target.

## Step 1 — Create a 6m / 5-stock slice

Create a new config inheriting the parent's `strategy_type`, `strategy_params`,
`commission`, `initial_capital` and `bars` (`["1d"]`).

Override:

- `symbols`: pick **5** symbols from the parent. Prefer the most-traded / most
  profitable ones (check `Trades` count per symbol if the parent has a trade
  list, or use the first 5).
- `trading_start`, `trading_end`: exactly **6 months** of trading.
- `warmup`: keep at least as many **calendar days** as the parent — for 1h bars
  the engine converts warmup to bars internally; a warmup of `"250d"` stays
  `"250d"` regardless of bar size.
- `name`: `<parent_name>_hourly_s1` (s1 = stage 1).

Run: `uv run ibkr bt run <new_config>` — confirm metrics land close to the
parent's on the same slice window. Save the report.

### Parameter adjustment for Step 1 (1d candles on slice)

If metrics already match, move on. If they drift (different symbol set), adjust
only `strategy_params` values — never the strategy code. Iterate `bt run` until
Sharpe, Ann, MaxDD are within ~15% of the parent on that slice.

## Step 2 — Change bars to 1h, re-param

Copy the Step 1 config and change:

- `bars`: `["1h"]` (single hourly bar, no HTF)
- `name`: `<parent_name>_hourly_s2`

All bar-count params (lookback windows, MA periods, ATR periods, cooldown bars,
decay bars, etc.) must be scaled from daily to hourly:

### Scaling rule

**Hours per trading day.** First find the real bar count per session, then use
it. The US session here stores **7** native 1h bars (15:30–21:00), so the
effective divisor is **7**, not the 6.5 a nominal session length implies. Verify
it on the feed before scaling:

```bash
python3 -c "import sqlite3,datetime as d; c=sqlite3.connect('../data/ibkr.db');\
t0=int(d.datetime(2022,1,3).timestamp()*1000); t1=int(d.datetime(2022,1,8).timestamp()*1000);\
s=set(r[0] for r in c.execute('SELECT timestamp FROM candle WHERE ticker=\"AAPL\" AND timestamp BETWEEN ? AND ?',(t0,t1)));\
import collections; dd=collections.Counter(d.datetime.fromtimestamp(x/1000).date() for x in s);\
print(sorted(dd.values()))"
```

**Bar-count parameters** scale by hours-per-day (7):

| daily param                                                              | hourly value                    |
| ------------------------------------------------------------------------ | ------------------------------- |
| `big_lookback: 63`                                                       | `63 * 7 = 441`                  |
| `ma_fast: 10`                                                            | `10 * 7 = 70`                   |
| `ma_slow: 20`                                                            | `20 * 7 = 140`                  |
| `atr_period: 14`                                                         | `14 * 7 = 98`                   |
| `cooldown_bars: 5`                                                       | `5 * 7 = 35`                    |
| `decay_bars: 10`                                                         | `10 * 7 = 70`                   |
| `vol_period: 20`                                                         | `20 * 7 = 140`                  |
| `comp_window: 15`                                                        | `15 * 7 = 105`                  |
| `ae_lookback: 40`                                                        | `40 * 7 = 280`                  |
| `min_hover_bars: 4`                                                      | `4 * 7 = 28`                    |
| `risk_pct`: **keep identical** (not bar-dependent — it's $risk / equity) |
| `size`: **keep identical**                                               |
| `max_gain`, `min_gain`: **keep identical**                               |

Round all scaled values to the nearest integer. Run and compare metrics.

### ATR MULTIPLES scale by ~sqrt(bars-per-day) — NOT "keep identical", NOT ×bars

**This table row used to read `atr_mult: keep identical`. That was wrong.** An
ATR multiple is NOT bar-invariant, and leaving it at the daily value silently
kills the hourly run. But the fix is **NOT ×bars-per-day** either — see the
anti-pattern below.

ATR is a **per-bar price range**. For a near-random walk, a range scales with
the square root of elapsed time, so:

```
ATR_hourly  ~=  ATR_daily / sqrt(N)        N = bars per session
```

A multiple of ATR is a fixed price distance expressed in the *current bar's*
volatility, so the conversion is **`daily_value × sqrt(N)`**.

For `N = 7`: `sqrt(7) ≈ 2.65`. **Measured on real data the ratio is ~3.0**
(2.49 – 3.28 across MSTR / PLTR / RKLB, hourly vs daily ATR, same window).

Rule: **`daily_value × ~3`** (target `sqrt(N)`, rounded toward the measured
ratio). Then **confirm empirically and re-validate on every larger window** —
the conversion is strategy- and feed-specific, not a constant to assume.

| daily value | literal ❌ | ×7 ❌ (over-scaled) | **~×3 ✅** | what it is              |
| ----------- | --------- | ------------------- | --------- | ----------------------- |
| `2.0`       | 2.0       | 14.0                | **6.0**   | `max_stretch_atr`       |
| `1.5`       | 1.5       | 10.5                | **4.5**   | `stop_atr_mult`         |
| `3.0`       | 3.0       | 21.0                | **9.0**   | `trail_atr_mult`        |
| `4.0`       | 4.0       | 28.0                | **12.0**  | `tp_atr_mult` (if used) |

Name every ATR-multiple param in the config and scale each one. In
`entropy_vp_breakdown_dsl` that is `max_stretch_atr`, `stop_atr_mult`,
`trail_atr_mult` (and `tp_atr_mult` when non-zero). Missing one leaves that gate
or stop too tight, which reads as "the signal never fires" — you will then blame
the wrong parameter.

#### Anti-pattern: the ×7 over-scale (documented failure)

ATR is a per-bar *range*, and ranges scale with `sqrt(time)`. Multiplying an ATR
multiple by the **bar count** treats a range as if it scaled linearly with time —
it over-widens stops and under-risks.

Observed on `entropy_vp_breakdown`, 6-symbol / 2022-H1 slice (daily target:
14 trades, Sharpe 2.62, Ann 35.32%, Cap Util 15.6%):

| ATR conv. | trades | Sharpe | Ann%   | Cap Util% | Win%   |
| --------- | ------ | ------ | ------ | --------- | ------ |
| ×1 literal (broken) | 1  | -1.12 | -4.29  | 1.4  | 0.00   |
| ×2.65 (sqrt 7)      | 16 | 0.17  | -1.14  | 3.5  | 43.75  |
| **×3 (measured)**   | 19 | 1.05  | **6.13** | 3.4 | 57.89 |
| ×7 (over-scaled)    | 36 | 2.12  | 2.44   | **0.4** | 63.89 |

**The ×7 row is fake convergence:** utilization collapses to 0.4% against the
daily 15.6% (~2.8× under-risked), so the attractive MaxDD is an artifact of
barely risking anything — the same failure mode as disabling a gate and calling
the deletion convergence. Trade count rises and Win rises, but Ann falls.

Measured ATR ratios that settle it (hourly ATR ÷ daily ATR, same window):
MSTR 3.20, PLTR 3.28, RKLB 2.49 — i.e. ~3, consistent with `sqrt(7) = 2.65`,
nowhere near 7.

#### Rules

- Do **not** "fix" a dead hourly run by setting an ATR-multiple gate to `0`
  (disabling it). That deletes entry logic and calls the deletion convergence.
- Do **not** accept a variant whose improvement comes with a collapsing capital
  utilization — check `Cap Util` against the daily run before believing a better
  MaxDD. Fake convergence via under-risking looks like success on the metric
  sheet.
- Scale the multiple, then sweep it to confirm the scaled value is **binding**
  (a value past which the metric is byte-identical is a disabled gate wearing a
  number, not a tuned parameter).
- Slice-fitting is a starting point only: re-validate the conversion on each
  larger window (S3, S4) and report drift rather than silently re-fitting.
- ATR-**period** params still scale by **bar count** (e.g. `atr_period: 14 → 98`)
  — that is a *window length*, not a multiple. Only the multiples take the
  `sqrt(N)` rule.

### If metrics diverge — iterate

After the first 1h run, compare trade timing. The hourly run will fire at
different intraday moments. Common adjustments:

- **Increase `min_hover_bars`** if too many false breakouts: the hourly
  compression is naturally noisier.
- **Increase `cooldown_bars`** proportionally (already done in the scale step)
  — but may need further tuning (whiplash on hourly is worse).
- **Increase `decay_bars`** if boxes expire too fast: hourly compression coils
  last ~same wall-clock time, so the bar count scaling should already hold.
- **Keep `risk_pct` identical** — it governs $ risk per trade, not bar count.
- **Re-check every ATR multiple** (see the `sqrt(N)` section above) — a missed
  one shows up as "no trades", not as an obviously wrong stop. Do not scale it
  by the bar count; ranges scale by `sqrt(time)`.

Iterate: adjust one param at a time, re-run, compare metrics. Target: Sharpe
and Ann within ~15% of the daily slice run.

### Alternative strategy — dual-bar (HTF) configuration

If a single `["1h"]` bar fails to converge after 3 iteration rounds, switch:

- `bars`: `["1h", "1d"]` — base is 1h, daily is a higher-timeframe accumulator
- Leave the original bar-count `strategy_params` unchanged — the strategy's
  indicator calls (all through `ctx.ta`) remain on the base bar (1h), but you
  can use `ctx.state.candles.get((sym, "1d"))` or `ctx.ta.<indicator>(sym,
interval="1d")` to read daily-resampled series for the setup conditions.
  **Requires modifying strategy code** — you must write a new strategy module
  that reads `1d` data from the HTF candle store for the big-move / compression
  detection and uses `1h` only for the breakout trigger.
- This is the higher-effort path. Only take it if pure 1h fails.

**Because the AGENTS.md hard rule says "never test a strategy" and strategy
code changes need trade-by-trade review, the HTF path is permitted only after
the user explicitly approves it.** Tell the user why pure 1h failed and what
the HTF approach entails.

## Step 3 — Expand to 5x2 symbols, 1 year

Create a new config inheriting the winning Step 2 config. Override:

- `symbols`: **10 symbols** (5 from Step 1 + 5 more from the parent)
- `trading_start`, `trading_end`: exactly **1 year** of trading
- `bars`, `strategy_params`: same as the winning Step 2 config
- `name`: `<parent_name>_hourly_s3`

Run 1d baseline and 1h config on this same 10-symbol / 1y window
**separately** (two configs, two runs). Compare metrics. Iterate parameter
tweaks (same adjustment rules as Step 2) until Sharpe and Ann are within ~15%
of each other.

## Step 4 — Full run (all symbols, full window)

Create the final config:

- `symbols`: **all symbols** from the parent config
- `trading_start`, `trading_end`: same as parent config
- `bars`, `strategy_params`: final values from Step 3
- `name`: `<parent_name>_hourly`

Run once and report all metrics vs the original daily run. Note:

- **More trades** is expected (hourly bar count → more signal checks).
- **Sharpe may differ** because intraday fills diverge from open-only daily fills.
- **MaxDD may be tighter** (hourly stop-out at a better price than daily open).
- **Scaled / Rejected** fills: if the hourly run has more, the issue is sizing,
  not signal — hourly generates more simultaneous signals.

## Reporting

At every step, report:

1. Config path
2. Window and symbol count
3. Metrics table (Sharpe, Ann, MaxDD, Kurt, Skew, Win, Trades)
4. Any adjustments made since the last step
5. Whether metrics converged or diverged — and by how much (%)

## Rules

- Never modify strategy source code except as the explicit HTF dual-bar fallback
  (Step 2 alt), which requires user approval.
- Never skip a step — S1 → S2 → S3 → S4.
- Never test a strategy — validate by running.
- Use `bt run` (not sweep/split/optimize) for all comparison runs.
- Read the full output, never grep/head it.
