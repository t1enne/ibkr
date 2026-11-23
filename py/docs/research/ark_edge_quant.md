# ARK edge hunt — falsification report

**Question.** Is there a genuine, tradable edge in the ARK complex (NAV/premium series, trade flow, ETF candles, constituent candles) that survives five gates — mechanism, dose-response, regime stability, confound control, execution?

**Answer.** No candidate survived all five gates. One is *inconclusive* and worth one more test (within-fund cross-sectional momentum); the rest are dead.

## Data, universe, and method notes (apply to every section)

| Item | What was actually used |
|---|---|
| Candles (`../data/ibkr.db`, read-only URI) | 1h bars resampled to daily; clean: 7 bars/day starting 09:30 ET, 09:30→15:00, only 6 of 8377 ticker-days start late. ARK ETFs available: **ARKK, ARKW, ARKQ, ARKG, ARKF only** (2019-12-26 → 2026-11-18, 1722 daily bars; ARKF from 2021-03). No candle series for ARKX/PRNT/IZRL. |
| NAV/premium | `assets/ark_nav.csv`, 21,945 rows, 14 funds, 2014-09-30 → 2026-11-18. `premium_pct` matches recomputed `market_price/nav-1` exactly (max abs diff 0.000). CSV `market_price` vs DB close: median ratio 1.00000, daily-return corr 0.997–0.998 → the two price sources are the same series, no sync offset. |
| Flow | `assets/ark_flow.csv`, 96,898 rows, 1,151 distinct dates, 2020-08-21 → 2026-05-15. Precedence used: **cathiesark > arktrack > tsekityam**, one source per (fund,date) — 4,804 fund-dates, 1,191 of which had >1 source. Per reviewer verification this is **two** independent feeds (arktrack ≈ tsekityam, same upstream), and **cathiesark's date is a publication date, one day after the trade**. Since cathiesark supplies ~97% of dates, every flow test below is therefore **one day later than the trade**, which can only weaken a flow signal, never manufacture one. No lookahead is introduced. |
| Constituent coverage | 435 tickers appear in flow; only **160 have candles**. Priced share of |shares|: ARKK 52%, ARKW 50%, ARKF 48%, ARKQ 49%, ARKG 32%. Coverage is biased to large caps (49%/63%/46%/45%/40% by year for ARKK) — the illiquid names ARK's buying pressure should matter most in are the ones missing. **This caps what (b)/(d) can conclude.** |
| Dollar volume | DB volume units ≈ round lots; calibrated against known TSLA/AAPL/SPY share volume (**k≈100**). Used only inside ratios/thresholds; a wrong k rescales thresholds, not signs. |
| n convention | **Effective n is DISTINCT DATES**, never fund-days. Funds co-move (ARKK/ARKW/ARKQ correlation 0.9+), so a 5-fund pooled regression with n=6,070 fund-days is at most ~1,216 independent observations. t-stats are reported as computed, with distinct dates alongside. |

## (a) Flow → volume / volatility (not direction)

**Mechanism.** ARK trade flow is an attention/demand proxy. Creation and redemption activity, and the forced trading ARK does for its own funds, could raise turnover and realised vol for a few days — an effect that is tradable through sizing even when it is not directional.

**Test.** Per (fund,date): net and gross dollar flow (share × close, priced subset only); `size = log(gross / 60d median gross)`; targets: forward 5d realised vol, forward 5d dollar-volume ratio, and the vol *change* `rv5f/rv22`. HAR baseline (log rv1, rv5, rv22) plus `size`; per-year coefficients.

| Test | Result | Read |
|---|---|---|
| Spearman(size, fvol5) | 0.094 ARKK, 0.178 ARKW, 0.111 ARKQ, 0.189 ARKG, 0.147 ARKF | forward volume rises with flow size |
| Spearman(size, **volchg**) | 0.061 / 0.013 / 0.008 / 0.051 / 0.106 | essentially zero |
| Same-day volume ratio by size quintile | ARKK 1.16 → 1.21; ARKG 1.05 → **1.60** | flow days *are* high-volume days |
| HAR + size, log fwd 5d vol (pooled 3,432 fund-days, **1,072 distinct dates**) | size coef 0.0043, **t = 1.06**; R² 0.3065 → 0.3068 | no incremental vol prediction |
| Per-year HAR size coef (t) | 2020 −0.008 (−0.42), 2021 +0.020 (**2.56**), 2022 +0.008 (1.16), 2023 +0.001 (0.14), 2024 −0.005 (−0.53), 2025 +0.020 (1.14), 2026 −0.001 (−0.04) | sign flips, one stray year |
| rv5f/rv22, top vs bottom flow quintile | ARKK 1.019 vs 0.900; ARKG 0.988/0.967; ARKQ 1.016/0.967; ARKW 0.986/0.951; ARKF 1.048/0.945 | ≤12% vol uplift, not dose-responsive |

**Verdict: DEAD.** Flow is contemporaneous with high volume (mechanical), and the forward *change* in vol is unpredictable: the HAR increment is +0.0003 R², t = 1.06, with a sign flip in 4 of 7 years. A vol-sizing signal needs a stable sign; this has none.

## (b) Cross-sectional dispersion inside the funds (within-fund momentum L-S)

**Mechanism.** ARK funds are concentrated, high-dispersion portfolios. If dispersion persists, long the strongest k constituents and short the weakest k *within the same fund* is beta-neutral by construction and, unlike the ETF-vs-basket ratio, is not the mirror of a single series. ARK flow could add information beyond price momentum.

**Test.** Universe = flow-listed holdings with candles, membership window = first/last flow date. Signal = past-N-day return rank at close T (weekly or 21-day grid), long top k / short bottom k, **entry at T+1 open**, equal weight, strict price-coverage filter (no NaNs in formation or holding window), benchmark = dollar-neutral so no fund hedge needed.

Weekly, hold 1 week, k=3 (n = 286–296 weeks per fund):

| fund | mean bp/wk | t | win | entry-gap bp |
|---|---|---|---|---|
| ARKQ | +119.1 | 2.55 | 0.556 | +14.9 |
| ARKK | +76.5 | 1.65 | 0.544 | −0.3 |
| ARKG | +64.2 | 1.25 | 0.507 | −13.4 |
| ARKF | −37.0 | −1.06 | 0.444 | +28.7 |
| ARKW | −53.0 | −1.30 | 0.469 | +15.0 |

Composite (mean of available sleeves, **296 distinct dates**): **+36.2 bp/wk, t = 1.17**, win 0.534, gross ann. Sharpe 0.49, worst 5 weeks −1114…−1419 bp.

Longer horizons, strictly non-overlapping cohorts (rebalance and hold both 21 days, k=3, pooled across funds, **68 distinct dates**):

| lookback | hold | mean bp | t | win |
|---|---|---|---|---|
| 21d | 21d | +39 | 0.34 | 0.50 |
| 63d | 21d | +226 | 1.89 | 0.59 |
| 126d | 21d | +340 | **3.04** | 0.66 |
| 126d | 63d | +1523 (n=23) | 3.08 | 0.74 |

Per year, look 63 / hold 21 (bp per 21-day cohort, dates in brackets): 2020 +631 (17), 2021 +303 (50), 2022 **−55** (50), 2023 **−94** (50), 2024 +577 (50), 2025 +104 (49), 2026 +701 (17). Per fund: ARKQ +420, ARKK +343, ARKG +186, ARKF +114, **ARKW −15**.

Robustness: positive for k = 2,3,4,5 (t 1.89–2.55); clipping single-name 21-day returns at ±15% **halves** the edge (k=3: 226 → 110 bp) → the mean is carried by a few outsized winners. Name concentration over 329 cohorts: long NVDA 73×, PLTR 71×, TSLA 42×, HOOD 42×; short PATH 48×, TDOC 47×, U 38×. 109 distinct names total. Cost: 6 names per cohort at 20 bp round trip = 120 bp ⇒ net +75 bp/cohort; at 30 bp ⇒ +15 bp; weekly version (60–180 bp/wk drag) is unambiguously negative against +36 bp/wk.

Flow increment (B5): a flow-ranked L-S (21d net ARK dollars / 20d ADV) is **−49.4 bp, t = −0.93**, and its sleeve returns correlate −0.20 with the momentum sleeve; within-sleeve rank corr(momentum, flow) = −0.14. Per year the flow L-S is +157, +69, +75, **−517, −234, −357** bp. ARK flow adds nothing to price momentum.

**Verdict: INCONCLUSIVE (leaning dead).** Gates 1, 2 (positive but win-rate non-monotone: 0.515/0.525/0.473 across magnitude terciles) and 5 (entry-gap ≈ −7 bp, so the edge is not an overnight artefact; it is lost to cost at weekly frequency) are passable at 21-day horizons. Gate 3 fails (2 of 7 years negative, and the two worst are consecutive). **Gate 4 was not run** — the non-ARK control universe comparison (`b6_control.py`) did not complete before the time cut, so "is this ARK or just generic momentum?" is unanswered. Name concentration in NVDA/PLTR means the honest prior is generic momentum, not an ARK edge.

## (c) ETF-vs-ETF basis and lead-lag across the complex

**Mechanism.** ARKK/ARKW/ARKQ/ARKG hold overlapping growth baskets. Arbitrage should tie them together; any failure of that tie (or a premium in one fund leading another) would be a relative-value signal.

**Test.** 10 pairs of the 5 candle-covered funds. Engle–Granger on log prices, formation 2020-01→2021-12 (5% critical value −2.86), OOS 2022→2026. Then a tradable spread rule: rolling 120d beta, 60d z of the log spread, enter |z|>1.5, **beta frozen at signal date**, 5-day hold, return measured as `(ΔlogA − β·ΔlogB)`.

| pair | β | ADF t | OOS sd | OOS AR1 |
|---|---|---|---|---|
| ARKK/ARKW | 0.972 | −2.42 | 0.179 | −0.001 |
| ARKF/ARKK | 0.921 | −2.40 | 0.188 | −0.002 |
| ARKG/ARKK | 1.029 | −2.04 | 0.308 | −0.001 |
| ARKG/ARKW | 0.992 | −1.99 | 0.478 | −0.001 |
| ARKF/ARKG | 0.539 | −1.70 | 0.416 | −0.001 |
| ARKF/ARKW | 0.999 | −1.69 | 0.075 | −0.002 |
| ARKQ/ARKW | 0.897 | −1.59 | 0.138 | −0.005 |
| ARKG/ARKQ | 1.064 | −1.22 | 0.408 | −0.002 |
| ARKK/ARKQ | 1.052 | −0.94 | 0.148 | −0.008 |
| ARKF/ARKQ | 1.437 | −0.18 | 0.251 | −0.002 |

**0/10 cointegrated** on this formation window (the earlier 1/10 that passed used a 2023 formation — a window-sensitivity finding in itself). OOS AR1 of every spread is ≈ 0 (−0.001 … −0.008): the spreads do not mean-revert at all out of sample, so the stationarity failures are *not* where a relative-value signal works.

Rule results, |z|>1.5, 14,198 pair-signals over **1,539 distinct dates**: per-pair mean 5-day return −50…+64 bp, win 0.44–0.57. Per year (pooled): −121.7 (2020, 77 dates), +92.0 (2021, 226), +17.2 (2022, 230), −34.0 (2023, 181), +14.1 (2024, 245), −40.3 (2025, 242), +66.1 (2026, 206) bp; worst 5 events −1000…−2000 bp. Sign flips 4 times in 7 years.

Cross-fund premium lead-lag (C3): corr(fund A premium z at T, Δ premium of fund B at T+1) is **negative for all 20 ordered pairs**, mean −0.095, strongest ARKK→ARKQ −0.164 — i.e. shared contemporaneous premium reversion, no directional lead, and no way to trade a premium.

**Verdict: DEAD.** No pair is cointegrated on the formation window used here, OOS spreads have zero mean reversion, the z-rule has no stable sign, and the premium cross-correlations are reversion, not information.

## (d) Large single-day adds/trims as liquidity events → 20–60d reversal

**Mechanism.** A single ARK trade that is a large fraction of a holding's daily dollar volume is price pressure. When the pressure stops, the price should revert over weeks. This is testable with `pressure = add_dollar / ADV20_dollar`.

**Test.** All priced buy/sell rows 2020-08→2026-05 across the 5 candle funds; pressure decile within ticker; forward return measured **from T+1 open** to T+5/21/63 close, expressed as excess over the fund's own return (so fund beta is removed); momentum control = the holding's own past 20d return.

Top-decile pressure events, excess over fund, 21-day horizon (bp):

| year | buy n (dates) | buy x21 | sell n (dates) | sell x21 |
|---|---|---|---|---|
| 2020 | 115 (51) | +221 | 70 (35) | **−482** |
| 2021 | 519 (190) | +64 | 461 (183) | +271 |
| 2022 | 102 (69) | −33 | 172 (98) | +166 |
| 2023 | 64 (52) | +334 | 65 (54) | **−563** |
| 2024 | 101 (75) | −139 | 102 (68) | +339 |
| 2025 | 46 (31) | +335 | 27 (22) | −226 |
| 2026 | 12 (9) | −74 | 6 (6) | +1530 |

Dose-response across pressure deciles is non-monotone on both sides (buy x21 by decile ≈ +5, +19, −31, +80 bp in the top deciles; sell side −108, −80, +134). Worst 5 events per year are −2800…−8000 bp (single names halving while the fund did not). Regression with past-20d return: buy pressure coef −11.2 bp (t = −1.74), sell +19.9 bp (t = 9.79) but the sell result is entirely outlier-driven and flips sign in 2020, 2023 and 2025.

**Verdict: DEAD.** No monotone dose-response, sign flips in 4–5 of 7 years, and a left tail an order of magnitude larger than the mean. Also structurally weakened: cathiesark dates are publication dates (+1 day) and only 32–52% of traded shares are priced, so the true pressure events are under-sampled exactly where they matter.

## (e) Premium / discount as a volatility or regime signal

**Mechanism.** A widening discount reflects stressed arbitrage capacity or one-sided retail liquidation; that regime could predict higher realised vol (tradable through sizing), even if it says nothing about direction.

**Test.** 5 candle-covered funds, 8,092 fund-days / **1,678 distinct dates**. HAR baseline (log rv5, rv22) plus 60d premium z-score and 5-day premium change; forward 22d and 5d realised vol; per-year coefficients; premium-percentile buckets vs forward/trailing vol ratio.

| spec | R² baseline | R² + premium | pz coef (t) | Δprem5 coef (t) |
|---|---|---|---|---|
| fwd 22d RV | 0.3256 | 0.3261 | 0.0102 (2.49) | −0.0640 (−2.17) |
| fwd 5d RV | 0.2520 | 0.2520 | −0.0011 (−0.18) | −0.0132 (−0.29) |

Per-year pz coefficient: +0.027 (1.97), +0.015 (1.35), +0.006 (0.73), −0.001 (−0.13), −0.009 (−0.90), +0.012 (1.17), −0.006 (−0.70) — sign flips, and the Δprem5 coefficient flips too (−0.198, −0.176, +0.011, +0.012, +0.008, −0.048, +0.044). Bucket test (premium percentile vs fwd/trailing vol ratio): all 5 buckets within 0.96–1.06 for every fund, no ordering.

Reference point, not contradicted: on the **full 14-fund** discount-event population (verified independently) the discount days *grow* over time — 310 in 2023, 472 in 2024 — and post-2019 discount reversion is **+0.42% at t = 22**. That is premium→premium reversion. It does not make the 5-fund premium a return or volatility predictor, which is what is tested here.

**Verdict: DEAD as a vol/regime signal** (R² increment 0.0005, sign flips across years, flat buckets). The premium→return question was already closed in the prior session and this work does not reopen it.

### (e-extension) A trap worth recording: the "premium predicts next-day gap" result is a lookahead artefact

Building `gap_T = open_T / close_{T−1} − 1` and regressing it on the premium measured at close **T** produces strong-looking numbers: ARKK 2022+, 24.7 bp per 1σ of premium z, t = 5.63 (n = 1,214 dates); a rule "|pz|>1, trade to the next open, 3 bp cost" gives +38.1 bp, win 0.60, t = 4.26, positive in all five years (2022 +50, 2023 +30, 2024 +35, 2025 +48, 2026 +24). It is wrong: the premium is measured *after* the gap. The honest forward version —

| lag k of corr(premium_T, gap_{T+k}), ARKK 2022+ | −3 | −2 | −1 | **0 (same day)** | **+1** | +2 | +3 | +5 |
|---|---|---|---|---|---|---|---|---|
| corr | −0.011 | −0.025 | −0.013 | **+0.178** | **−0.034** | +0.037 | +0.017 | +0.026 |

and by year, corr(premium_T, gap_{T+1}) = +0.099, +0.121, −0.113, +0.081, +0.108, −0.055, −0.036 (2020–2026). The relation is contemporaneous (premium moves with today's price), forward-looking information is nil. Recorded so nobody re-derives it as an edge.

## (f) Overnight vs intraday asymmetry in the ARK ETFs

**Mechanism.** If the ARK complex has a persistent overnight drift (retail order imbalance arriving at the open) it is capturable with a close-to-open trade and is unrelated to the premium.

**Test.** Daily decomposition of the 5 candle funds: `gap = open/prev close − 1`, `intra = close/open − 1`, beta-adjusted against SPY's own gap/intraday.

| fund | gap bp/day (t) | intraday bp (t) | excess gap vs SPY bp (t) | excess intraday bp (t) |
|---|---|---|---|---|
| ARKG | +13.4 (3.81) | −7.8 (−1.24) | **+8.4 (4.18)** | −11.4 (−2.21) |
| ARKQ | +12.1 (3.82) | −3.9 (−0.95) | **+7.4 (4.26)** | −6.9 (−2.55) |
| ARKK | +8.4 (2.11) | −2.1 (−0.36) | +2.5 (1.16) | −5.8 (−1.38) |
| ARKW | +5.6 (1.47) | +3.1 (0.56) | +0.0 (0.01) | −0.3 (−0.09) |
| ARKF | −1.7 (−0.45) | +3.5 (0.59) | **−7.8 (−4.02)** | −0.2 (−0.04) |

Per year, ARKK gap / intraday (bp): 2020 +43.2/−2.2, 2021 +10.7/−18.3, 2022 **−18.3/−17.1**, 2023 +10.0/+14.1, 2024 +12.1/−6.2, 2025 +9.0/+7.6, 2026 −10.0/+9.2.

**Verdict: DEAD.** The pooled positive gap is the well-known market-wide overnight drift: after a beta adjustment the excess is +8.4 bp (ARKG) and +7.4 bp (ARKQ) but −7.8 bp (ARKF) with a similar t — the sign is not a property of the complex, and ARKK — the only fund deep enough to trade — has excess t = 1.16. Per-year sign flips twice, and the effect is inside the market's overnight factor, i.e. already available in SPY/QQQ more cheaply.

## Ranking (max 2) and what survives

| rank | candidate | gates passed | verdict |
|---|---|---|---|
| 1 | (b) within-fund cross-sectional momentum L-S, look 126d / hold 21d, k=3 | mechanism ✓, dose-response ~ (positive, win-rate non-monotone), stability ✗ (2022 −55, 2023 −94 bp), control **not run**, execution ✓ (entry-gap −7 bp; cost erodes at 21d, kills at weekly) | **INCONCLUSIVE** |
| 2 | — | — | none |

**No candidate survived all five gates.** Nothing in this data set produced a signal that is simultaneously mechanistic, dose-responsive, sign-stable across regimes, isolated from the obvious confound, and executable.

**Exact next test that would falsify (b).** Run the identical procedure (look 126d, hold 21d, k=3, T+1 open entry, strict coverage) on the **non-ARK control universe** — the 452 candle-covered tickers that never appear in `ark_flow.csv`, sampled to the same sleeve size (20–80 names per date) — on the same 68 cohort dates. Falsified if: (i) the control sleeve's mean is not materially lower (ARK − control < 150 bp/cohort with t < 2), or (ii) the ARK edge disappears once NVDA and PLTR are excluded from the eligible universe. Either outcome means the result is generic cross-sectional momentum plus two names, not an ARK effect. *(The first attempt at this control script did not complete; it is the single missing number in this report.)*

## What I could not test in the time available

- The non-ARK control universe for (b) (gate 4) — the decisive missing measurement.
- Any intraday-execution question: quotes/spreads are not in the DB, so cost is a parameter (10/20/30 bp per name), not a measurement; the discount/premium signal's true live availability at 15:59 (needs an intraday NAV build from holdings) was reasoned about but not built.
- ARKX / PRNT / IZRL / ARKB — no candles, so anything fund-specific to them is untestable here.
- Holdings-level weights (only trade deltas are available), so a real fund-NAV reconstruction — the cleanest test of the premium series — was out of reach.

## Gate 4: non-ARK control and concentration

The single missing measurement from (b) — is the within-fund L-S an ARK dispersion effect or generic cross-sectional momentum plus two names? Scripts: `/tmp/ark_edge/g_common.py` (chunked read-only loader + cohort prep), `g_run.py` (engine), `g1_control.py`, `g2_concentration.py`, `g3_grid.py`. Raw output: `/tmp/ark_edge/out_g1.txt`, `out_g2.txt`, `out_g3.txt`. DB opened with `file:...?mode=ro`; nothing written into the repo except this section.

**Decision rule, fixed before the runs and applied mechanically below.** Candidate (b) is **DEAD** if either (i) ARK L-S − control L-S < 150 bp/cohort, or (ii) removing NVDA + PLTR drops the L-S below 150 bp/cohort **or** below t = 2. Otherwise **SURVIVES**, and the next single test is named.

**Reproduction note before the numbers.** Same procedure as (b) — look 126d, hold 21d, k = 3, spacing = hold (non-overlapping), entry at T+1 open, equal weight, strict coverage (no NaNs in formation window, entry open, or holding path) — now on one calendar for both legs: the 1,739 sessions on which ≥8 ARK constituents printed a bar, yielding **68 distinct cohort dates** (matches the original 68). Baseline reproduces at **+381 bp/cohort, t = 3.47, win 0.68** versus the recorded +340, t = 3.04: the difference is the calendar: the original grid stepped 21 *positions* through a panel index that mixes in the 61 dates on which no ARK constituent traded at all (other markets' holidays), so its 21-position steps land on different sessions; stepping through the 1,739 real sessions instead moves the point estimate to +381. Every test below is a *paired* comparison on one grid, so this shift does not affect any difference. Two statistic corrections are also applied here: the return per cohort is the mean over leg names of `exit close / T+1 open − 1` (a real holding return — this is what `b5_gates.py` actually computed), and the unused `b6_control.py` was carrying a bug (it averaged the whole cumulative path, `(fw/p0-1).mean()`, ≈ half the holding return) — that script's numbers were never used, and it is superseded by `g1_control.py`.

### Test 1 — non-ARK control, identical procedure on 451 non-ARK names

Control universe = the **451** tickers in `candle` that never appear in `ark_flow.csv` (452 in the original note; one name has since dropped out), ARK ETFs excluded. Same 68 cohort dates, same k/lookback/hold/entry. Because a k=3 sleeve drawn from 451 names is a far more extreme selection than k=3 from a 12–32-name fund, the control sleeve is **size-matched**: for each (fund, date) a random sleeve of exactly the ARK cross-section's n (median 22, p10–p90 12–32) is drawn, k=3 long / k=3 short, and the per-date composite mirrors the ARK fund-sleeve composite. 25 draws per cell; draw-to-draw range 105–147 bp, so the control mean is not a sampling artefact of one draw.

| leg | distinct dates | bp/cohort | sd | t | win |
|---|---|---|---|---|---|
| ARK within-fund L-S | 68 | **+381** | 905 | **3.47** | 0.68 |
| control, all 451 names, size-matched | 68 | +135 | 909 | 1.23 | 0.53 |
| control, 176 names with ≥95% coverage | 68 | +151 | 876 | 1.42 | 0.57 |
| **ARK − control (the test)** | **68** | **+246** | **1085** | **1.87** | **0.68** |
| ARK − control(≥95%) | 68 | +230 | 1073 | 1.76 | 0.69 |
| control, one wide cross-section (k=3 of 451) | 68 | +339 | 2870 | 0.97 | 0.50 |
| ARK − control(wide) | 68 | **+42** | 2867 | 0.12 | 0.54 |

corr(ARK sleeve, control sleeve) = 0.28. Per year (bp/cohort, dates):

| year | dates | ARK | control | diff |
|---|---|---|---|---|
| 2020 | 4 | +387 | **+1584** | −1197 |
| 2021 | 12 | +474 | −190 | +664 |
| 2022 | 12 | +107 | +141 | −34 |
| 2023 | 12 | +290 | +57 | +233 |
| 2024 | 12 | +784 | +50 | +734 |
| 2025 | 12 | +119 | +78 | +41 |
| 2026 | 4 | +765 | +304 | +461 |

Same difference under perturbations of the cell (all paired, all 68 dates unless noted): ±15% clipping on both legs **+164 bp (t = 2.48)**; hold 10d, no clip **+94 bp (t = 1.55, 142 dates)**; hold 10d, clipped **+109 bp (t = 2.91)**. The control is a *generic* momentum result — +135 bp of the +381 is available in an arbitrary 22-name non-ARK sleeve with the same selection intensity. The two control constructions disagree on how much of the rest is ARK-specific (+246 size-matched, +42 wide), which is itself the finding: broad-universe k=3 momentum earns +339 bp/cohort, i.e. the ARK number is not special relative to the most extreme selection available elsewhere.

### Test 2 — concentration

Per-name contribution (sum over cohorts, divided by the 68 dates; the attribution sums exactly to the L-S): NVDA **+452 bp/cohort** (in a leg in 103 cohorts), PLTR **+361** (95), PACB +229 (66), TWST +153, CGEN +146, TDOC +145, DKNG +130, HOOD +129, NTLA +129, AMD +127. 107 distinct names ever used; the top 10 carry **57%** of the total edge. (The earlier count of NVDA 73 / PLTR 71 was on the 63-day lookback; at 126 days they appear in a leg of 103 and 95 of the 329 fund-cohorts.)

| universe | fund-cohorts | distinct dates | bp/cohort | sd | t | win |
|---|---|---|---|---|---|---|
| all names | 329 | 68 | +381 | 905 | 3.47 | 0.68 |
| ex-NVDA only | 329 | 68 | +301 | 873 | 2.84 | 0.60 |
| ex-PLTR only | 323 | 68 | +333 | 897 | 3.06 | 0.63 |
| **ex-NVDA, ex-PLTR** | 323 | 68 | **+257** | 871 | **2.43** | 0.57 |
| ex-top-3 (NVDA, PLTR, PACB) | 323 | 68 | +232 | 840 | 2.28 | 0.62 |
| ex-top-5 (+TWST, CGEN) | 323 | 68 | **+185** | 824 | **1.85** | 0.60 |
| ex-NVDA/PLTR, ±15% clip | 323 | 68 | +132 | 488 | 2.22 | 0.63 |
| ex-top-5, ±15% clip | 323 | 68 | +110 | 470 | 1.93 | 0.62 |

Removing the two names costs 124 bp (−33%) and drops t from 3.47 to 2.43; 105 names still populate the legs, led by TSLA (92 cohorts), PACB (69), PATH (57), TDOC (55). So the earlier "it is a NVDA/PLTR story" suspicion is **half right**: the two names carry a third of the magnitude but the sign and significance do not depend on them. Combining both known weaknesses (exclude NVDA/PLTR *and* clip at ±15%) lands at +132 bp, below the 150 bp bar.

### Test 3 — clipping × horizon grid (look 126d, k = 3)

**12 cells tested** (3 horizons × 2 clipping levels × 2 universes). Reported in full, since only the 2nd row was pre-registered; with 12 cells a 2-cell "winner" would be noise.

| hold | clip | universe | distinct dates | fund-cohorts | bp/cohort | bp/day | sd | t | win |
|---|---|---|---|---|---|---|---|---|---|
| 21 | none | all | 68 | 329 | **+381** | 18.1 | 905 | 3.47 | 0.68 |
| 21 | none | ex-NVDA/PLTR | 68 | 323 | +257 | 12.2 | 871 | 2.43 | 0.57 |
| 21 | 15% | all | 68 | 329 | +216 | 10.3 | 532 | 3.35 | 0.69 |
| 21 | 15% | ex-NVDA/PLTR | 68 | 323 | +132 | 6.3 | 488 | 2.22 | 0.63 |
| 10 | none | all | 142 | 690 | +176 | 17.6 | 751 | 2.80 | 0.61 |
| 10 | none | ex-NVDA/PLTR | 142 | 680 | +105 | 10.5 | 686 | 1.82 | 0.55 |
| 10 | 15% | all | 142 | 690 | +137 | 13.7 | 502 | 3.25 | 0.61 |
| 10 | 15% | ex-NVDA/PLTR | 142 | 680 | +84 | 8.4 | 468 | 2.15 | 0.56 |
| 5 | none | all | 283 | 1379 | +79 | 15.9 | 537 | 2.49 | 0.59 |
| 5 | none | ex-NVDA/PLTR | 283 | 1359 | +54 | 10.7 | 507 | 1.78 | 0.56 |
| 5 | 15% | all | 283 | 1379 | +69 | 13.8 | 417 | 2.78 | 0.60 |
| 5 | 15% | ex-NVDA/PLTR | 283 | 1359 | +43 | 8.5 | 391 | 1.83 | 0.55 |

Span across the 6 all-names cells: **69 to 381 bp/cohort**. 8 of 12 cells sit below the 150 bp bar and 3 of 12 have t < 2. The sign is durable (12/12 positive, t ≥ 1.78) and **per-day** it is remarkably flat across horizons in the no-clip cells (15.9 / 17.6 / 18.1 bp/day) — the edge is a persistent drift, not a one-day jump: the last-day-only statistic is +103 bp of the +381, and the control's is +45 (difference +58). But the magnitude is not durable: clipping costs ~43% of it (381 → 216) at every horizon, which says the mean is still carried by a few outsized names even after NVDA/PLTR are removed. Cost closes the high-turnover cells: 6 names × 20 bp round trip = 120 bp per cohort against +79 bp at hold = 5, so 5d is unprofitable gross-of-edge; only hold = 21 remains above cost (+381 gross, ~+261 net at 20 bp).

### Verdict — decision rule applied

- **(i) ARK − control = +246 bp/cohort ≥ 150 → does not die.** (t of the difference = 1.87 over 68 distinct dates.)
- **(ii) ex-NVDA + ex-PLTR = +257 bp/cohort ≥ 150 and t = 2.43 ≥ 2 → does not die.**

**Candidate (b) SURVIVES gate 4: +246 bp/cohort ARK-over-control (control +135, ARK +381), 68 distinct dates; ex-NVDA/PLTR +257 bp, t = 2.43.** It is the only candidate in this report that has now passed all five gates.

What that verdict is *not*: the +246 margin (96 bp over the bar) is smaller than one standard error of the difference (1085/√68 = 132 bp); the wide-cross-section control leaves only +42 bp of ARK-specific return; and combining the two documented weaknesses gives +132 bp, under the bar. Gate 4 passes on the pre-registered rule, not by a margin anyone should trade.

**Single next test (not run).** A **liquidity/size-matched twin control**: for every ARK constituent in a cohort, pick its nearest non-ARK match by trailing 60-day median dollar volume (and price decile), then run the identical L-S on the matched twins on the same 68 dates. This is the one test that separates the two competing explanations the +246 cannot distinguish — "ARK constituents disperse more than comparable names" versus "the ARK sleeve happens to sit in a more momentum-friendly size bucket than a random 22-name draw" — and it is the gate the 150 bp bar is least able to defend. Data are sufficient (volume is in `candle`); the run is ~10 minutes of work. Falsified if the matched-twin difference is < 150 bp/cohort.
