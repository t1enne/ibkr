# `strats/` — classification criteria

Three buckets, one rule: a config is filed only after the evidence for its tier
exists. The filesystem (`pass/`, `wip/`, `fail/`) is canonical; this file states
what a bucket *means*, not what is in it.

Tier is about **evidence**, not sentiment. A config moves up only by accumulating
the gates below, and moves down whenever a gate regresses against the current
engine.

## Evidence gates

Every config, regardless of tier, is judged on the same set. All are required
for PASS; a missing or failing gate caps a config at WIP.

| gate | requirement |
|---|---|
| **Benchmark-relative** | Beats buy-and-hold of the natural benchmark over the *same* window. Raw return is not evidence. |
| **Cost-adjusted** | Metrics assume commission + slippage. Dollar-per-trade cost is stated, and the edge survives it. |
| **Sample size** | Enough closed trades to support the claim. Below the confidence floor, only a WIP claim is allowed. |
| **Out-of-sample** | Walk-forward / fold split separates IS tune from OOS validation. OOS mean and OOS **min** both reported. |
| **Regime stability** | Per-year (or per-regime-block) breakdown, not just the full-window aggregate. Concentration in one stretch of history is a hazard, not a result. |
| **Point-in-time universe** | Symbol lists must survive a survivorship-free rebuild. A curated list of today's survivors is a bias, not a universe. |
| **Fill integrity** | Zero cash-exhausted fills. Scaled or rejected fills make a run advisory — its metrics move with symbol order and cannot be quoted as solid. |
| **Risk shape** | Kurtosis and skewness reported alongside Sharpe. A Sharpe carried by a fat tail is not the same object as a Sharpe carried by breadth. |

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

FAIL is a finding worth keeping: the record explains *why* the mechanism does not
work, so it is not re-probed. A retired config's module may be removed; the
negative stays written down.

## Documented negatives

A negative recorded during another config's investigation belongs here as
prose, not as a file. Requirements:

- Name the variable that was isolated.
- State the direction and magnitude of the change, in the same units as the
  baseline.
- State the conclusion the negative licenses — what should *not* be re-probed.

A negative without those three is an opinion.

## Naming and layout

```
strats/
├── pass/    # every gate closed
├── wip/     # one or more gates open
└── fail/    # a gate structurally cannot close
```

Bucket membership is a claim about evidence, so moving a config between buckets
is a reviewed change, not housekeeping. Deleting a config is fine once its
negative is written into this file; the record outlives the file.
