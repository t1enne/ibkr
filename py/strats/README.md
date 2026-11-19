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

### `kalman_pairs_dsl` — XLE/TLT is not a pair (negative)

Isolated variable: the **symbol pair**. Same config, same window, same params; the
only thing that changed between the two rows below is the engine (B1 hold path
reachable, see the code-defect note under the third entry).

`fail/kalman_pairs_dsl.json` (XLE/TLT, 2018-11-12 → 2024-12-31, `1d`). Pre-fix: 138
trades, gross **+2,295.50**, spread+slippage 5,465.00 + commission 13.80 = friction
**5,478.80**, net **-3,183.29**, Sharpe -0.09, MaxDD -11.93%, kurt 70.98. Cost per
round trip (69 of them): gross $33.3 vs friction $79.4 → **-$46/round trip**. Post-fix,
hold path reachable: 120 trades, gross **-3,344.17**, friction 4,728.48 + 12.00 =
**4,740.48**, net **-8,084.65**, Sharpe -0.08, MaxDD -21.10%, kurt 113.91, holds 2-10
bars (was a 1-bar floor), Scaled 18 → **advisory**. Split post-fix: mean OOS **+0.20**,
min **-0.50**, OOS folds -0.50 / -0.06 / +0.25 / +1.11, IS folds +0.10 / -0.33 / -0.26 /
-0.19 (3 of 4 negative) — and the positive mean sits inside the advisory band, so it is
not quotable either way.

Why the pair was never a pair: log-OLS β of `log(XLE) ~ log(TLT)` = **-1.311**, and the
same-filter diagnostic shows the fade paying **-6.6 bps at h=1** against +75.5 bps at
h=5 and +358.3 bps at h=20 (t≈1.9 on a sd-1756 2020 tail).

Conclusion: the pre-fix gross-positive number was an artifact of a bar-flat round-trip
loop — once the spread is actually held, gross is **negative** (-3,344.17), i.e. there is
no reversion edge to capture. Do not re-probe XLE/TLT as a cointegrated pair; β = -1.31
makes it an energy-vs-long-bonds macro spread, and no exit rule can fix a sign-wrong
hedge.

### `kalman_pairs_goog_googl` — Kalman convergence is not P&L convergence (negative)

Isolated variable: the **pair**, selected on stability alone (β CV 0.010, 60% of rolling
2y windows cointegrated, eg_p 0.002, residual sd 67 bps, half-life 19.9 td) rather than on
p-value. Four declared params, all round numbers, no sweeps; every filter hyper-parameter
left at its default.

`wip/kalman_pairs_goog_googl.json`, one year (2024-01-01 → 2024-12-31): 20 trades / 10
round trips, gross **+313.90**, commission 2.00 + spread+slippage 475.88, net **-163.98**,
Sharpe **-0.80**, kurt 13.10, Scaled 0 / Rejected 0 (quotable). Per round trip: gross
**+$31.4** vs friction **$47.6**, i.e. **12.6 bps gross vs 19.1 bps of friction**. Six of
ten round trips lost money *after* converging in z — the exit is measured against a
drifting Kalman level. Split (`--folds 4`): all four **IS** folds negative (-0.64, -0.95,
-0.59, -0.41), mean OOS **-1.08**, min **-2.08**, 3/4 OOS negative. `z_entry 2.0` is a
real threshold post-B3 (4.0% of bars, 96.0th percentile), while `z_exit_stop 3.5` is
unreachable (max observed |z| 2.72).

Conclusion: even a genuinely cointegrated dual-class pair does not clear friction at
daily bars. The binding constraint is **residual amplitude vs cost**, not filter quality —
67 bps of spread does not survive four fills. Do not re-probe daily Kalman spread-fade
unless the selected pair's residual sd is an order of magnitude above the friction floor.

### `kalman_mr_regime_dsl` — single-name residual fade is sign-wrong (negative)

Isolated variable: **`z_stat` scaling** (absolute price² units → scale-free). The
threshold then becomes a threshold: 2.0 sits at the 96th percentile instead of passing on
95-99% of bars.

Pre-fix (337 trades, Sharpe 0.53, net +15,534.00): `z_entry 2.0` was inert against a
z_stat with sd 75-245 per symbol and |z|>2 on 95.4-99.2% of bars, so entry was fully
delegated to the regime gate. Post-fix (2021-12-31 → 2025-12-31): **46 trades** over four
years, gross **-5,050.94** before friction (friction 890.49 + 4.60, net **-5,946.03**),
Sharpe **-0.68**, MaxDD -9.01%, kurt 32.12, Scaled 0. Split: mean OOS **-1.18**, min
**-1.54**, OOS folds -1.26 / -1.09 / -0.84 / -1.54 (all four negative). Exit census moved
from 236 long-side exits + 1 stop in 337 to 18 short fades / 17 long fades / 10 regime
flattens / 1 ATR stop (on a short).

Conclusion: a constant-velocity Kalman residual is a **momentum** proxy; fading its upper
tail loses at every horizon tested. The pre-fix +15.5k was a regime-gated long/short book
wearing a Kalman label, not a Kalman strategy. Do not re-probe residual fade on single
names.

### Code defects behind the numbers above (fixed, not evidence changes)

Four CODE defects, all fixed and reviewed OK before any of the post-fix figures were
produced — they are engine repairs, not revisions to the evidence, and both sides of each
before/after pair are negatives, so every verdict stands either way: **B1**
`kalman_pairs_dsl._exit_if_due` had no hold path (every pair closed on the next bar) —
`src/bt/strategies/kalman_pairs_dsl.py`; **B2** `is_long` was inferred from the unsigned
`Position.qty`, so it was always true and shorts got the long-side stop —
`src/bt/strategies/kalman_mr_regime_dsl.py`; **B3** Kalman noise was absolute (price²), so
`z_stat` was not scale-free and the batch/online filters disagreed —
`src/indicators/kalman/strategy.py` and `src/indicators/kalman/pure.py`; **B4** the HMM
detector's warmup guard ignored feature warmup, so the fit raised on 251-272 bars —
`src/bt/regime/detectors.py`. Do not re-derive these from a fresh run; the pinning tests
fail on the pre-fix code and pass on the current code.

### `hmm_regime_filter` — a frozen HMM regime gate vs its own SMA control (negative)

Isolated variable: **the HMM regime gate** — `use_regime` on vs off — in front of an otherwise
identical SMA 10/30 entry. Same symbol (SPY, `1d`), same feed, same costs (4.5 bps/side +
$0.05/fill), same size (`0.99` equity, next-open fills). The control is the same config with
`use_regime=false`, so the HMM is never constructed and the label is the only thing that differs.
The fit is frozen (`fit_once=True`, one fit strictly before `trading_start`), so no trading bar is
labelled from its own future. The configs are filed in `fail/`, the control beside them because
the control is the evidence.

**Stage 1 — SPY, 2010, one year, n=5:** filter **+0.97%** vs SMA control **−0.02%** vs SPY B&H
**+10.99%**; max DD **−12.64% / −13.81% / −16.05%**. The HMM's marginal **1.17 pp** of DD cost
**10.18 pp of CAGR**, and 74% of the filter's net was the year-end mark (rule-only **−1,813.37**).

**Stage 2 — SPY, 2008-01-02 → 2012-12-31, 1258 bars, n=8, one frozen fit 2004-02 → 2007-10:**

| metric | HMM filter | SMA 10/30 control | SPY B&H |
|---|---|---|---|
| total return | **+2.70%** | +19.35% | −3.39% |
| CAGR | **0.54%** | 3.61% | −0.69% |
| max DD | **−8.41%** | −16.78% | −53.04% |
| Sharpe | **0.12** | 0.34 | −0.03 |
| Calmar | **0.06** | 0.22 | — |
| profit factor | **1.36** | 1.54 | — |

Rule-only (excluding the window-end mark): filter **+4,524.76** on n=7.

**Where the drawdown goes.** Of the 44.63 pp of DD reduction from B&H, the SMA base supplies
**36.26 pp (81.2%)**; the HMM's marginal **8.37 pp (18.8%)** — the −16.78% → −8.41% step — costs
**3.07 pp of CAGR**.

**The gate's mechanism did not do the crisis work.** 502 of 510 would-be entry bars blocked
(**98.4%**), one state-1 spell of 502 bars (2008-01-02 → 2009-12-28), **zero trades in 2008 and
zero in 2009**. Between 2008-09-05 and 2008-12-15 the crash core had **0 trend-up bars**, so both
books were already flat while SPY fell. Net cost of the two-year risk-off spell ≈ **8,542.80**
(8.5% of starting capital): avoided −5,656.09 of control P&L, forgone +14,198.89. Split: filter
mean OOS Sharpe **−0.10** (min **−0.76**) vs control **0.44** (min **−0.55**).

Conclusion: the reference's claim — no longs opened in the high-vol regime, so the crisis is dodged
— transfers only as a **mechanical side effect** of a frozen pre-crisis fit labelling the whole
period high-vol (label 1 on 82.3% of the window's bars), not as an economic edge; and the marginal
DD it buys is beaten by not having the HMM at all (control +19.35%, Calmar 0.22 vs filter +2.70%,
0.06). Do not re-probe **a frozen-fit HMM regime gate in front of an SMA trend entry, on SPY
daily**. A rolling/retrained variant is a DIFFERENT, unprobed test — nothing here measures it.

**The code is kept regardless of the negative.** The underlying implementation was fixed in this
pass and is worth keeping on its own terms: the online per-bar label now equals batch
prefix-Viterbi (**308/308 bars**; it disagreed on **42%** before) at lower cost, and the HMM warmup
`ValueError` band is gone.

### `hmm_gas_allocation` — Nystrup et al. (2017) GAS regime allocator does not reproduce (negative)

Isolated variable: **the regime forecast → mean-variance weight mechanism itself** — the paper's
single-risky-asset variant, SPY against 0% cash, 2005-02-14 → 2026-11-10, **5469 decision bars**. The
paper's two-asset SAA is out of reach: **no bond proxy older than 2016-12 exists in the DB**, so the
31% the static control puts in a bond leg is 0%-interest cash here. Cited as Nystrup, Madsen &
Lindstrøm (2017), *Quantitative Finance* **18(1) 83-95**.

**Claim table on the final build** (all three arms `fail/hmm_gas_spy*.json`; Scaled 0 / Rejected 0 on
every arm, so the weight path is complete and permutable):

| id | claim | this build | status |
|---|---|---|---|
| **F1** | MPC Sharpe > buy-and-hold, net of costs and the 1-day delay | **0.314 vs 0.463 AR/vol (-0.149)** | **FAILED**, sign reversed |
| **F2** | maxDD < buy-and-hold | **-24.02% vs -56.49%, ratio 0.425** against the paper's own **0.456** | **MET** on the stated criterion |
| **F3** | beats rule-based RBAA on Sharpe | RBAA arm is structurally inert: 1 lot, 100% of bars at w≈1.000, max filtered `change_prob` **0.5890** over 21.7 y against a **0.9998** threshold | **UNTESTABLE** |
| **F4** | annual turnover ≤ ~2 with rho = 0.02 | **2.289/yr** fill-level (2.243 excluding the forced final liquidation; intent 2.304) | **FAILED** — 1.15x the gate, 1.96x the paper's **1.17** |
| **F5** | filtered state prob must move, no multi-year pin | no pin — longest single-label stretch **152 bars** — but **389 argmax flips = 17.9/yr** against the paper's **~0.9** | **MET** on its wording |
| **F6** | window must contain the drawdowns it claims | 2005-02-14 → 2026-11-10 contains 2008 and 2020, and the arm de-risks in both | **in scope** |

**The mechanism finding — the actual deliverable.** Not "just holds less": mean held weight **0.741**
with maxDD **-24.02%**, against a static 69/31 control at **0.690** and **-42.95%** — 19 pp less
drawdown on 5 pp *more* exposure. A constant-weight counterfactual at the arm's own 0.741 predicts
**-45.5%** (linear method calibrated x1.012 against the engine-run static arm), so average exposure
explains none of the 24.02%: **~21.5 pp is genuine time variation**. But that variation **lags**: the
forecast mean is **0.655**-correlated with the **trailing** 60-day market return and **-0.024** with the
**forward** 20-day, and the de-risks land **12-22% into each drawdown** — GFC first w<0.1 **+101 days**
into the drawdown at **-15.6%** already taken, COVID **+9 days** at **-12.5%**, 2022 **+161 days** at
**-21.8%**. Only **22** round-trip lots, one lot 31.9% of |PnL|, two years 58.8%.

Conclusion: the paper's forward-looking regime-probability mechanism is **not supported** — what
survives is a lagging, momentum-like de-risking overlay whose only advantage (the 0.425 DD ratio) rests
on **four episodes inside one 0.8-SE-wide confidence interval**, paid for with 0.15 Sharpe. Do not
re-probe a GAS-HMM mean-variance overlay on a single equity index expecting it to anticipate drawdowns.
The static and RBAA arms travel with the paper arm into `fail/` because they *are* the evidence: the
static arm is the exposure control that kills "it just holds less", the RBAA arm is the control that
cannot fire.

**Why the negative is trustworthy — the estimator moved three times under correction.** Crisis `sd_hi`
went **102 → 224.7 → 244.0 bp** (241.2 on this build's own bar set, both below the 250 bp band floor);
a **backwards window weighting** held the **oldest** bar **2.71x** the newest against the docstring's
`w_n = f**(t-n)`; an **under-scaled Newton step** (realised-outer-product curvature, then a step damped
by **1/164**) froze the recursion; and a **parameter box that clipped `mu`** at ±100 bp/day pinned the
high-vol drift instead of estimating it. Each correction was re-measured, not tuned. The final build
takes **two declared deviations** from the paper: the **conditional-expectation information matrix**
(the paper's matrix-inversion-lemma recursion implies the realised outer product) and **A = 1** on the
emission blocks instead of **1/N_eff**.

### `kalman_chan_pairs_dsl` — Chan (2013) Example 3.3 reproduces mechanically, not economically (negative)

Isolated variable: **the economics of the rule set on a substituted pair**. Window, filter
hyper-parameters (`delta`, `Ve`, `k`), friction and fill model are held at the faithful
configuration; what moves between rows is only the **exit convention** (band vs zero) and the
**price domain** (absolute vs log). Three rule variants × two windows = six configs, all filed.

**Reference.** Chan, *Algorithmic Trading: Winning Strategies and Their Rationale* (Wiley
2013), ch. 3, Example 3.3 — Kalman-filter pairs trade, EWA/EWC. **Provenance caveat, stated
honestly:** the chapter text could not be opened (no scan on archive.org or Open Library, the
Wiley full text is paywalled), so the rules were reconstructed from three secondary sources
that read it; the rule set is itemised with per-item provenance in
`/tmp/kalman_ref/reference.md` §2.

**Not a literal reproduction.** EWA/EWC have **zero rows** in the local DB and TLT/IEI has no
2009-2016 history, so the pair was chosen **before any strategy existed**, on the reference's
own falsifiers, from a bounded economically-motivated candidate set: **GDX/SLV** (x = GDX
regressor, y = SLV observed leg). ADF passes, beta stable, non-degenerate trade sample.

**FINAL numbers** (post hedge fix, repo fill model, 2014-11-14 → 2026-11-11, net):

| arm | Ann | Sharpe | MaxDD | net P&L | round trips | Scaled / Rejected |
|---|---|---|---|---|---|---|
| faithful, band exit | **-0.74%** | -0.13 | -17.57% | **-8,501** | 460 | 0 / 0 |
| faithful, zero exit | **+0.42%** | 0.11 | -9.37% | **+5,189** | 406 | 0 / 0 |
| repaired, log domain, zero exit | **+0.67%** | 0.26 | -4.53% | **+8,274** | 17 | 0 / 0 |

Scaled 0 / Rejected 0 on every arm. 1-year faithful band: net **-605** on **75** round trips.

**Statistical power — the decisive qualifier.** The faithful-zero mean trade is **+16.74** USD
with SE **32.94** (t = **0.51**); SE(annual) is **±1.35%** on the 460/406-trade arms and
**±0.72%** on the 17-trade arm, so **no annual return is distinguishable from zero**. The
17-trade arm's +8,274 is **74% one trade and 67% one calendar year**; it licenses no edge claim.

**THE STRUCTURAL FINDING, which is the real deliverable.** Measured on the traded pair,
conditional on an entry, `|z_{t+1}| <= k` is satisfied **82.9%** of the time for long entries and
**84.1%** for short (band), and **49.1% / 52.6%** (zero exit) — because the update drags the
fitted level onto the price, so **the EXIT test, not the entry, sets the holding period**.
Median hold is **1 bar** at **34-39 round trips/yr**, and the **frictionless** band arm earns
**0.09 bps per round trip**: it cannot pay for its own turnover at any cost assumption. No
parameter, price domain or hedge convention changes that.

**Mechanics DID reproduce.** Trade counts matched an independent implementation exactly
(**460 / 406 / 17**), the filter matches reference items 2-8 including the `R = 0` first-bar
case, and the hedge is now `floor(beta*N)` **SHARES** — a notional hedge under-hedged by
**0.73x at the median** and was caught in review before filing. Two declared deviations:
next-open fills instead of same-bar close, and the sizing expressed as a capital fraction.

Conclusion: the published family is a **turnover machine whose edge is inside the noise on
substituted instruments**. Do not re-probe a **single-sigma innovation-band exit on daily
bars** expecting it to clear friction, and do not read the repaired log-domain variant's
**+0.67%** as a fix — at **17 round trips** it has no power.

The six arms travel as the evidence: `fail/kalman_chan_gdxslv_faithful.json`,
`fail/kalman_chan_gdxslv_faithful_zero.json`, `fail/kalman_chan_gdxslv_repaired.json`, and their
`_full` window twins beside them — plus the four frictionless `*_gross_verify.json` re-runs
(commission / spread / slippage = 0, all else identical), which are the control behind the
0.09 bps/RT figure and are filed with the arms for the same reason.

## Naming and layout

```
strats/
├── pass/    # every gate closed
├── wip/     # one or more gates open
└── fail/    # a gate structurally cannot close
```

Bucket membership is a claim about evidence, so moving a config between buckets
is a reviewed change, not housekeeping. Deleting a config is fine once its
