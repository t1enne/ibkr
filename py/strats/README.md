# `strats/` — classification criteria

Three buckets, one rule: a config is filed only after the evidence for its tier
exists. The filesystem (`pass/`, `wip/`, `fail/`) is canonical; this file states
what a bucket _means_, not what is in it.

Tier is about **evidence**, not sentiment. A config moves up only by accumulating
the gates below, and moves down whenever a gate regresses against the current
engine.

## Evidence gates

Every config, regardless of tier, is judged on the same set. All are required
for PASS; a missing or failing gate caps a config at WIP.

| gate                       | requirement                                                                                                                                      |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Benchmark-relative**     | Beats buy-and-hold of the natural benchmark over the _same_ window. Raw return is not evidence.                                                  |
| **Cost-adjusted**          | Metrics assume commission + slippage. Dollar-per-trade cost is stated, and the edge survives it.                                                 |
| **Sample size**            | Enough closed trades to support the claim. Below the confidence floor, only a WIP claim is allowed.                                              |
| **Out-of-sample**          | Walk-forward / fold split separates IS tune from OOS validation. OOS mean and OOS **min** both reported.                                         |
| **Regime stability**       | Per-year (or per-regime-block) breakdown, not just the full-window aggregate. Concentration in one stretch of history is a hazard, not a result. |
| **Point-in-time universe** | Symbol lists must survive a survivorship-free rebuild. A curated list of today's survivors is a bias, not a universe.                            |
| **Fill integrity**         | Zero cash-exhausted fills. Scaled or rejected fills make a run advisory — its metrics move with symbol order and cannot be quoted as solid.      |
| **Risk shape**             | Kurtosis and skewness reported alongside Sharpe. A Sharpe carried by a fat tail is not the same object as a Sharpe carried by breadth.           |

Additional honesty requirements, cheap to state and easy to skip:

- **A/B one variable at a time.** An "improvement" that changes several knobs
  measures nothing.
- **No-op the control.** Any new guard or knob must reproduce the baseline
  byte-identically at its neutral setting, proving the comparison is real.
- **Sub-SE deltas are noise.** A difference inside one standard error of the
  metric is not a finding; record the flat result.
- **Advisory is not quotable.** A run whose fills were constrained by cash is
  reported as advisory and never compared across symbol permutations.
- **Documented negative is a deliverable.** The finding is kept even when the
  probe config is retired — "keep the finding, bin the probe".

## PASS

All evidence gates hold, with margin:

- Beats the benchmark over the same window after costs.
- OOS min is positive, not merely OOS mean.
- Trade count supports the claim.
- No fill constraint in the resident run.
- No single symbol, single year or single regime is load-bearing.
- Universe survived a point-in-time rebuild and the edge persisted in thinned
  form.

A PASS config carries any standing caveat in the config's own record. A PASS
with an unexplained hazard is a WIP.

## WIP

Promising, but at least one gate is open. Canonical hazards:

- **Window too short / single-regime** — the claim rests on one year or one
  market regime.
- **Trade count below the confidence floor** — direction may be right, sample is
  not.
- **Trails the benchmark** — positive but not beating hold.
- **Cash-marginal** — scaled or rejected fills active; advisory.
- **OOS min negative** — mean positive, worst fold negative.
- **Universe not point-in-time** — curated winners only.
- **Risk shape unmeasured** — tails not characterised.

WIP is the default state of anything new. A config enters `wip/` the moment it
has a stated hypothesis and a run that does not contradict it; it leaves only by
satisfying every gate or by failing to.

## FAIL

Any of:

- **Negative expectancy** after costs.
- **Underperforms buy-and-hold** by a wide margin over the same window.
- **No statistical confidence** and no path to it — sample cannot be grown by
  extending the window or the panel.
- **Contradicted by its own control** — the claimed mechanism does not survive
  the neutral-setting or single-variable test.
- **Structurally incompatible with its target panel** — the signal's assumption
  (continuity, time-series structure, gap profile) is violated by the data class.

FAIL is a finding worth keeping: the record explains _why_ the mechanism does not
work, so it is not re-probed. A retired config's module may be removed; the
negative stays written down.

## Documented negatives

A negative recorded during another config's investigation belongs here as
prose, not as a file. Requirements:

- Name the variable that was isolated.
- State the direction and magnitude of the change, in the same units as the
  baseline.
- State the conclusion the negative licenses — what should _not_ be re-probed.

A negative without those three is an opinion.

### `mfi_pivotdiv_dsl` — short leg, four admission gates (negative)

Baseline `pass/mfi_pivotdiv_long.json` (112 names, 2020-01 → 2026-09, `1d`):
Sharpe 1.22, Ann 13.84%, MaxDD -14.00%, 1403 trades, kurt 14.3, Scaled 0 /
Rejected 0 on every arm below. One variable at a time; the neutral gate setting
reproduced the baseline byte-identically, so the comparison is real.

Shorts standalone (`allow_shorts` on, `allow_longs` off) are a loss engine:
Sharpe **-0.66**, Ann -7.60%, MaxDD -48.7%, 1855 trades. Combined with the long
book they take Sharpe **1.22 → 0.10** and total return 137.5% → 3.1%.

| short admission gate (structural windows) | Sharpe | Ann | MaxDD | trades | kurt |
|---|---|---|---|---|---|
| off (ungated) | -0.66 | -7.60% | -48.7% | 1855 | 4.7 |
| VIX rank ≥ 0.70 | -0.69 | -5.0% | -35.2% | 700 | 12.5 |
| VIX rank ≥ 0.90 | -0.21 | -1.2% | -18.2% | 363 | 28.1 |
| VIX rank ≥ 0.98 | 0.27 | 1.0% | -4.8% | 164 | 55.5 |
| bear tape (benchmark < 200d SMA) | -0.35 | -2.42% | -20.6% | 353 | 18.5 |
| bear tape, confirmed (200d falling) | -0.45 | -2.96% | -25.7% | 254 | — |
| own HMM = high-vol state | -0.34 | -1.3% | -14.4% | 379 | 13.0 |

VIX admission improves monotonically as it tightens and crosses zero only at
≥0.98, where 164 trades carry kurt 55 — a crash tail, not an edge. Back in the
combined book the best gated arm (VIX ≥ 0.98) prints Sharpe 1.25 vs 1.22, i.e.
**+0.03, inside one standard error**, while Ann falls 13.84% → 13.7% and MaxDD
worsens -14.00% → -15.3%.

**Bear-tape split (long bull years, short 2022) — also rejected.** Both premises
failed on the data:
the bear-tape short leg is not a 2022 leg (per-year: 2020 **-4,739**, 2021 no
trades, 2022 **+2,501**, 2023 -2,145, 2025 -2,195), and the long leg's edge is
*inside* the weak tape it was meant to skip — ungated 2022 is breakeven (-205),
gating longs to bull tape turns it into **-8,969** and 2023 from +13,325 into
+159 (Sharpe 1.22 → 0.60, MaxDD -20.18%). Best split arm (longs full, shorts
bear-tape only) is 0.90 Sharpe / -13.36% MaxDD; the only thing any split buys is
drawdown, which is the same frontier as shrinking `risk_pct` — a risk-appetite
choice, not an edge.

Conclusion: no short leg and no regime knob on this mechanism. This is the
fourth confirmation of the family finding (`mfi_divergence_dsl` shorts 0.53 vs
1.17 long-only). Do not re-probe a short-admission gate; the short side would
need a different **exit** rule to be worth revisiting. Arms are reproducible by
flipping `allow_shorts` / `allow_longs` on the filed config; the probes were
retired.

## Naming and layout

```
strats/
├── pass/    # every gate closed
├── wip/     # one or more gates open
└── fail/    # a gate structurally cannot close
```

Bucket membership is a claim about evidence, so moving a config between buckets
is a reviewed change, not housekeeping. Deleting a config is fine once its
