# ARK unstaged artifacts — reviewer report

**Scope.** Read-only review of `docs/research/ark_*.md`, `docs/review/ark_data_code_audit.md`,
`scripts/ark_pairs_screen.py`, `assets/ark_flow.csv`, `assets/ark_nav.csv`.
Docs were read as *claims*. Every number I repeat as verified was recomputed here from the CSVs
(stdlib `csv`) and `../data/ibkr.db` (`file:...?mode=ro`) with `python3 sqlite3`. No strategy was
run, no backtest executed, no reviewed file modified. This report is the only write.

**One line.** The work rules out the cheap, over-specified hypotheses and demonstrates that the
flow file cannot answer the interesting one. It does **not** rule out an ARK edge. The only live
candidate (within-fund L-S momentum) was selected in-sample, is survivor-filtered, and its own
control difference has a 95% CI containing zero.

---

## 1. Experiment ledger — what was tested, verdict, quoted numbers

- **§1 ETF-vs-constituent log-ratio, 34 pairs.** MR and sign-flipped momentum are exact mirrors:
  `corr(ann_rev, ann_mom) = -1.0000`; median Sharpe `-0.40` → `+0.42`; identical trade bars.
  Expectancy `≈ -round-trip cost`. **DEAD — tautology, not a test.**
- **§2 multi-pair ladder 2→4→8→16→34.** Best rung `+0.15%` vs 50/50 buy-and-hold `+42.53%`;
  excess kurtosis `12.0 → 6.0` (tail fixed, sign not). **DEAD.**
- **§3 classic cointegration (OLS β, ADF, half-life).** Formation 2023, OOS 2024: `13/34` pass at
  `α=0.10` (~3.4 expected by chance); OOS `-11.89%`, Sharpe `-0.93`; friction only `5%` of loss;
  no `|beta|` gate, `ARKK/TDOC β=-0.024` lost `-7,040`. **DEAD.**
- **§4 flow front-running, 6 years / 96,898 rows.** T+1 buy−sell excess `+0.21%` pooled; inverted
  dose-response in `3/7` years; monthly spread flips `6/13` months; trader's notional quintiles run
  the **opposite** way (monotone increasing, +0.201% top vs −0.002% bottom). **DEAD (sign unstable).**
- **§5 NAV premium decay.** Discount days `r_price +55.3bp` vs `r_nav -2.4bp`, paired `t=28.2`
  (n=1,059); excess over fund mean `+52.6bp`, `t=8.83`; post-2019 `+0.42%`, `t=22` (n=666, 500 of
  them IZRL/PRNT). **ALIVE statistically, DEAD economically (~8bp capturable/event, 10–20 events/yr).**
- **§6 within-fund L-S, look 126d / hold 21d / k=3.** `+381bp/cohort, t=3.47`, 329 fund-cohorts,
  **68 distinct dates**, win 0.68. Size-matched non-ARK control `+135` → **diff `+246`, t=1.87,
  95% CI `[-17,+509]`, SE 132**. Wide control `+339` (own SE 339, t 0.97) → diff `+42` (t 0.12).
  Clip ±15%: diff `+164 (t=2.48)`. Hold 10: `+94` (t=1.55, 142 dates). ex-NVDA/PLTR `+257 (t=2.43)`;
  +clip `+132` (below the 150bp bar). ex-top-5 `+185 (t=1.85)`; top-10 names = `57%` of edge.
  Per-day `15.9 / 17.6 / 18.1 bp` at hold 5/10/21. Grid: 12 cells, `69–381bp`, 8/12 below 150,
  12/12 positive. Weekly composite (higher n): `+36.2bp/wk, t=1.17`, gross Sharpe 0.49, 296 dates;
  ARKF `-37.0` (t −1.06), ARKW `-53.0` (t −1.30). Flow-ranked L-S `-49.4bp, t=-0.93`. **FRAGILE.**
- **(a) flow → volume/vol.** Spearman(size, fwd vol5) `0.094–0.189`; Spearman(size, volchg)
  `0.008–0.106`; HAR+size coef `0.0043`, **t=1.06**, R² `0.3065 → 0.3068`; per-year sign flips;
  quintile vol uplift ≤12%. **DEAD.**
- **(c) ETF-vs-ETF basis / lead-lag, 10 pairs.** `0/10` cointegrated on 2020-21 formation;
  OOS AR1 `-0.001…-0.008` (no reversion); z-rule `14,198` signals / `1,539` dates, per-pair mean
  `-50…+64bp`, win `0.44–0.57`, per-year sign flips 4×; worst events `-1000…-2000bp`; premium
  lead-lag negative for all 20 ordered pairs (mean `-0.095`). **DEAD.**
- **(d) large adds/trims → 20–60d reversal.** Top-decile buy x21 by year `+221/+64/-33/+334/-139/
  +335/-74bp`; sell `-482/+271/+166/-563/+339/-226/+1530bp`; deciles non-monotone; regression
  buy `-11.2bp (t=-1.74)`, sell `+19.9bp (t=9.79, outlier-driven)`; tails `-2800…-8000bp`.
  **DEAD.**
- **(e) premium as vol/regime signal.** Fwd 22d RV R² `0.3256 → 0.3261`; pz coef `0.0102 (t=2.49)`;
  Δprem5 `-0.0640 (t=-2.17)`; fwd 5d RV `t=-0.18 / -0.29`; per-year sign flips; buckets `0.96–1.06`.
  **DEAD.**
- **(e-ext) "premium predicts next-day gap" trap.** Same-day `corr(premium_T, gap_T)=+0.178`,
  `T+1 = -0.034`; ARKK same-day `24.7bp/1σ, t=5.63`; rule `+38.1bp, t=4.26` positive 5/5 years —
  all lookahead. **ARTEFACT, correctly recorded.**
- **(f) overnight vs intraday, 5 funds.** ARKG gap `+13.4bp (t=3.81)`, excess `+8.4 (t=4.18)`;
  ARKQ excess `+7.4 (t=4.26)`; **ARKF excess `-7.8 (t=-4.02)`**; ARKK excess `+2.5 (t=1.16)`;
  per-year ARKK flips twice. **DEAD — market-wide overnight factor, not ARK.**
- **Desk A1–A6 (trader doc).** A1 discount trade: `8bp/event`, 10–20 events/yr, ~`10bp/yr` portfolio,
  ARKF = 87/162 post-2019 events (stale NAV). A2 follow buys: `+20.6bp` gross vs `15–25bp` cost,
  first hour `+0.106%` of the `+0.206%`. A3 fade: quintiles **monotone increasing**, sign opposite
  to the quant. A4 sequences: `P(run≥2)=46.8%`, `P(≥3|≥2)=62.2%`, `P(≥4|≥3)=72.7%`, forward
  `15–20bp` vs `20bp+` cost. A5 ADSK/PTC: full-sample β `0.969` (claimed 0.48), `rho=0.9994`,
  HL `1,178 bars`, yearly β `-0.084…1.257`, rolling `-1.16…+2.36 (sd 0.50)`, 1-sd excursion
  `+7/+3bp` vs `8–12bp` cost. A6 AP business. **ALL PASS = REJECTED.**

---

## 2. Claim audit

### Blocking

1. **The headline "edge" is not the edge.** §6 quotes `+381bp, t=3.47` (gross ARK sleeve). The
   ARK-specific quantity is the difference: `+246bp, t=1.87`, CI `[-17,+509]`. At 0.73 SE over the
   bar, the correct label is "not distinguishable from generic momentum", not "survives". The
   conclusion says this in prose but the table leads with 381/3.47.
2. **No out-of-sample test anywhere for §6.** 68 cohort dates 2020-09→2026-04 are the same dates
   used to pick lookback (21/63/126), hold (21/63) and k (2–5) — the winner is the argmax of a
   12-cell grid plus a k-grid plus the weekly variant. "Pre-registered rule" covers the *gate*, not
   the *candidate*; the candidate was selected on the evaluation data. Only §3 and (c) got a real
   formation→OOS split, and both died. The one survivor never got one.
3. **Survivorship is undisclosed in the research docs.** I confirmed: all 160 flow tickers that
   have candles have a last bar **on/after 2026-11-02** (median 2026-11-19). Zero delisted names,
   zero acquired names, zero fund holdings that stopped trading. The §6 cross-section is "ARK names
   that still trade in 2026". The audit notes it; neither `ark_edge_quant.md` nor
   `ark_conclusion.md` does.
4. **`90.2% one-sided` does not reproduce at the stated 75% threshold.** My recomputation from the
   raw file: **79.7%** of line items, **51.0%** of fund-days at >75% one-sided. 90.2% is reached
   only at a ~62% one-sidedness threshold (threshold sweep: 0.60→91.3%, 0.67→85.8%, 0.75→79.7%).
   The substance (most flow is one-sided basket plumbing) survives; the quoted magnitude/threshold
   pair is wrong.
5. **`0.3% both-side triples` is wrong by 3.7x.** Reconfirmed `713/64,901 = 1.10%`. The audit's B3
   is correct; the trader doc's `0.3%` should be struck.
6. **Sample sizes count mirror duplicates.** Raw file: `31,284` duplicate `(date,fund,ticker,side)`
   keys across sources (`65,614` unique events in 96,898 rows). In the 2020-10→2021-10 window,
   `27,627` `tsekityam` keys are also `arktrack` keys — the desk's A2 "52,714 line items" double-counts
   the same trade in the only full-coverage year. "6 years, 96,898 rows" is really ~1 clean year
   (median 50 rows/fund-day in 2021) + 4.5 years of 2–3 rows/fund-day from a single noisy mirror.
7. **§5's execution claim contradicts the audit's own re-derivation.** Conclusion: "auction absorbs
   the reversion: excess gap `+7.6bp` vs intraday `+8.3bp`". Audit §6(iii) on the same event class:
   `n=28`, gap `+20.0bp (t=0.46)`, intraday `-31.3bp (t=-0.53)`, total `-11.3bp`. Neither
   construction is in the repo. Also, the "gross `20.6bps/day` vs `15–25bps` cost" line in §5 is
   A2's follow-the-buy number, not the discount trade's (`7.6+8.3 ≈ 15.9bp`). Mixed metrics.
8. **The dose-response sign is contradicted between docs.** Quant: "inverted in 3 of 7 years".
   Trader A3: "monotone *increasing*, not inverted" and attributes the inversion to mirror
   duplication. Conclusion §4 records only the quant version. With `31,284` duplicate keys, the
   trader's explanation is plausible; the record should carry both signs and the reason.
9. **§1–§3 have no surviving artifact.** The four strategy modules and every `ark_*.json` config
   (16–17 by the audit's count) are deleted, untracked, never committed. The 34-pair Sharpe/-11.89%/`-7,040` numbers cannot be
   re-derived from anything in the repo; `scripts/ark_pairs_screen.py` produces only the ADF/β/HL
   screen, not P&L. Treat §1–§3 as UNVERIFIED assertions (audit B1), not measured results.
10. **"`1–2%` of NAV per event" is ambiguous.** As position size (trader §9: "Size 1-2% of NAV per
    event") it is consistent with `~10bp/yr`. As edge it is 2–4x the measured `+52.6bp`. Reword.

### Should-fix

- **§3 "13/34 ≈ coin flip" understates the in-sample pass rate.** 13 vs 3.4 expected is 3.8x
  chance; 13/34 ADF p<0.10 on dependent pairs is *high*, not coin-flip. The OOS failure is the
  finding, not the pass count.
- **§6 "drift, not an event" is contradicted by the audit's own profile.** Last day carries
  `+103bp` of `+381` (27% of return in 1/21 of the period, 5.7x the mean daily); last 5 days 27%;
  offset 21 `+103`, 20 `-14`, 19 `-26`, 15 `+73`. The horizon-flat *average* does not make it drift.
- **§6 "22/22 claims reproduced" validates arithmetic, not inference.** The same script family has
  6 high-severity estimator defects: overlapping 5-day spacing with hold 21 inflates t ≈ √(21/5) ≈ 2.0
  (`b3_robust.py`, `b4_book.py`), iid SEs on overlapping forward windows inflate t ≈ √5–√22
  (`e_premium_vol.py`, `a3_vol_resid.py`), no date clustering on 21-day event windows
  (`d_events.py`). The 68-date re-run avoids the first; the rest still colour (a),(d),(e).
- **"8 of 12 grid cells below the 150bp bar" mixes scales.** 8 of 12 are *gross ARK* numbers; the
  150bp bar was defined on the *difference*. Only 3 cells have a control arm (164/94/109).
- **The conclusion omits the strongest null in §6's own family:** the weekly composite
  (`+36.2bp/wk, t=1.17`, 296 dates, 3 of 5 funds negative) and the hold-10 `t=1.55`. The summary
  table keeps only cells that clear or nearly clear the bar.
- **Multiplicity is never counted.** Six families, ≥34 pairs, 5 rungs, 10 ETF pairs, 3 lookbacks ×
  2 holds × k-grid + 12-cell grid, 7 yearly splits, 2 HAR specs. >100 verdict-bearing specs, no FDR,
  no trial count, no Bonferroni. The surviving t=3.47 is the max of a correlated grid on the same
  68 dates; the audit's own null simulation puts a single draw over 150bp at 13%.
- **Cost model is optimistic relative to the trader doc.** §6 uses 20bp round trip on the 6 sleeve
  names (the momentum extremes, i.e. the smaller names in these funds); trader §7 estimates 20–100bp
  spreads and 5–100%/yr borrow on exactly those names. Borrow is omitted entirely: on the doc's own
  6×20bp accounting, 21-day borrow on the 3 short names at 5–20%/yr costs ~1.25–5.0% of notional
  (125–500bp in the same units as the 381bp gross). At 50bp spreads the 6-name drag is 300bp >
  381bp gross. The `+261bp net` is a best case, not a base case.
- **`cathiesark` +1 day "can only weaken a signal, never manufacture one" is not guaranteed.** A
  one-day delay can flip a momentum effect into a reversal; it attenuates under a null but biases
  under an alternative. `cathiesark` resolves 86% of all fund-dates (audit H1) and is the only source
  after 2021-10, so the pooled `+0.21%` is mostly the same-day `arktrack` rows of 2020-21, not a
  6-year result.
- **"All 14" era counts mix a growing universe.** ARKT NAV starts 2025-10-01, ARKD 2026-01-02,
  ARKI 2026-04-01, ARKE 2026-07-01, ARKY 2026-08-19 — quarter-boundary starts are vendor
  truncation, not zero-discount days. Fixed-7-fund panel at `≤-0.25`: 2023 `273` → 2024 `343` →
  2025 `235` → 2026 `235` (partial), while the 5 liquid funds stay `7/32/16/20`. So "discount
  frequency grows" is real only in PRNT/IZRL/ARKX, i.e. the untradeable stale-NAV funds.
- **Candle bar count reconciles, with one caveat.** DB: ARKK 1,736 sessions, 1,722 with ≥5 bars
  (14 four-bar early closes dropped) — exactly the doc's 1,722 and its 2019-12-26 start. So the
  doc is internally consistent; the caveat is that the same nbar≥5 filter silently removes all 14
  early closes from every panel (audit §13), not that the count is wrong.

### Optional

- Quant coverage table `ARKK 52 / ARKW 50 / ARKF 48 / ARKQ 49 / ARKG 32%` vs my raw `56/51/31/43/34`
  (precedence panel `52/50/40/43/31`). ARKF/ARKQ differ materially; only ARKK/ARKW/ARKG reproduce.
  The qualitative claim (≈half of shares priced) holds.
- `notional/share` median `$93.2` (p10 19, p90 406) confirms dollars, and that `notional` is real
  only for arktrack (48,608/96,898 rows).

---

## 3. `scripts/ark_pairs_screen.py` review (184 lines)

**What it does.** Takes a strategy config, builds `Backtest(cfg)` *only* to read
`window.test_start`, loads candles via `load_candles(cfg.symbols, warmup_load_start(...),
test_end, cfg.bars[0])`, ffills both legs on the feed's union timestamp grid, fits
`ln(fund) ~ ln(cons)` on the **first 252 paired closes**, and prints ADF p, AR(1) half-life, α, β,
μ, σ per pair with a PASS/screened verdict. No P&L, no metrics, no grid. `Backtest` is a pure
dataclass, so the docstring's "no strategy-module import" is accurate.

**Repo-rule compliance (AGENTS.md § Alpha research: "Use `bt run`/`sweep`/`split`/`optimize` —
never a throwaway script").** Not a literal violation: there is no `run()` and no grid, so there is
nothing for `bt sweep` to sweep, and the docstring declares the exception ("NOT a backtest
harness"). But it is a hand-rolled config-load + candle-load + analytic loop — exactly the shape the
rule wants inside `bt` — and it is now **orphaned**: zero ARK strategy modules and zero
`ark_*.json` configs remain in `strats/`. `main()` requires a config with
`strategy_params["pairs"]`; none exists. As shipped it cannot be run. Fold the screen into a
`bt screen` subcommand or delete it with the report.

**Data reads.**

- **Path resolution: correct.** `load_candles → get_local_candles → get_connection →
  resolve_db_path` (`src/db/path.py`); no hardcoded path, env-overridable. Good.
- **`ffill` across a union index masks halts/delistings.** A leg that stops printing keeps its last
  close forever; the spread becomes stale-price arithmetic and ADF/HL are distorted in an
  uncontrolled direction. `dropna()` after `ffill` only removes *leading* NaNs, so it cannot detect
  a dead leg at all. Should inner-join on common sessions and require a bar on the formation end date.
- **Formation window is not pinned.** `warmup` is a *calendar* span (`"250d"` ≈ 172 trading bars)
  while the formation is 252 *bars*. With warmups in this repo of 200/250/300/400d, 3 of 4 put the
  first 252 daily closes **past** `test_start` — i.e. the screen is fitted on trading-window data
  for most configs. There is no assertion or print of `formation_end < test_start`. For 1h configs
  the opposite happens: 252 hourly bars end months before the test window, so the reported screen
  describes a different window than the one traded. Docstring's "same window/rule the strategy
  family used" is false in both cases.
- **No `|beta|` gate.** `Formation.tradeable` checks ADF p, half-life and `sd>0`; β is printed but
  never tested. This is the exact defect the conclusion (§3) blames for the classic book's death
  (`ARKK/TDOC β=-0.024`). The kept screen still has it.
- **No multiple-testing control.** 34 pairs at `α=0.10` → ~3.4 false PASSes expected; the output
  presents `PASS` with no FDR/Bonferroni guard and no note.
- **`from src.utils import _get_ols_fit_model` inside `fit_formation`** — lazy import of a private
  cross-module symbol, violating AGENTS.md:491 ("Top level imports only / No lazy imports"). Same
  class as audit M7, in the one file that was kept. Also aliases `_get_ols_fit_model` semantics:
  args `(y, x)` = `(fund, cons)` → `alpha=const, beta=slope`; that order is correct.
- Minor: header reads `df.index[0]` before any empty-frame guard; the "no history" branch depends on
  `load_candles` emitting an all-NaN column for an absent symbol (untested here, no config to run).

---

## 4. Data audit (independently recomputed)

| file | rows | dates | range | coverage | duplicates / defects |
|---|---|---|---|---|---|
| `ark_flow.csv` | **96,898** | 1,151 distinct | 2020-08-21 → 2026-05-15 | 8 funds (ARKK 23,048 / ARKG 21,870 / ARKW 17,474 / ARKQ 14,039 / ARKF 12,312 / ARKX 4,521 / PRNT 1,893 / IZRL 1,741); 435 tickers | **31,284 duplicate `(date,fund,ticker,side)` keys across mirrors; 0 within a source**; 1 zero-share row; 8 non-integer (float-delta) share counts; 0 negative shares; 0 weekend dates; `notional` filled only for arktrack (48,608) |
| `ark_nav.csv` | **21,945** | 3,053 distinct | 2014-09-30 → 2026-11-18 | 14 funds; flow covers only 8 (ARKB/ARKD/ARKE/ARKI/ARKT/ARKY absent) | 0 duplicate `(date,fund)`; 0 null/≤0 nav; `premium_pct` recomputes to `5e-5 pp`; range `-3.70 … +3.91`; 5 funds start on quarter boundaries (truncation) |
| `ibkr.db : candle` | **5,089,807** | 2004-01-23 → 2026-11-19 | ARK ETFs 2019-12-24 → 2026-11-19 | 616 tickers | ARK ETFs: ARKK/ARKW/ARKQ/ARKG/ARKF only; ARKX/PRNT/IZRL/ARKB/ARKT/ARKD/ARKI/ARKE/ARKY absent |

**Flow structure.** 4,804 fund-dates, 1,191 with >1 source. Median rows/fund-day: 2020 `35`,
2021 `50`, 2022 `3`, 2023 `3`, 2024 `3`, 2025 `2`, 2026 `2.5`. That is one clean year and a
4.5-year sliver — not six years of evidence. `shares` is unsigned (side carries sign), units are
shares; `notional/share` median `$93` confirms dollars for the arktrack subset.

**Candle coverage claim — verified, not trusted.** `435` flow tickers, `160` with candles,
`275` without (docs correct). Priced share of `|shares|` by fund (raw file): ARKK `55.9%`,
ARKW `50.9%`, ARKQ `43.4%`, ARKG `33.9%`, ARKF `31.3%`, ARKX `20.9%`, PRNT `30.2%`, IZRL `8.7%`.
The missing names are the illiquid ones ARK's pressure should matter in; every event study here is
a large-cap US study.

**Survivorship — confirmed.** All 160 priced flow tickers have a last bar `>= 2026-11-02`
(median `2026-11-19`); **0** have a last bar before 2026-06. There is not one delisted ticker — and therefore no
acquired holding that stopped trading — in the sample. The bias direction is ambiguous but the filter is structural and unstated in
the research docs.

**DB is mutable.** The audit already caught `52,714` vs `52,493` shifting mid-audit. Any row count
here is a snapshot; pin a hash or a dump before publishing.

---

## 5. Three next steps (ranked by information gain per unit of work)

1. **Matched-twin control (≈10 min, decisive for the only live candidate).** For each ARK
   constituent on each of the 68 cohort dates, match one non-ARK name on trailing-60d median dollar
   volume + price decile, run the identical 126d/21d/k=3 L-S on the twins, report ARK−twin `bp` and
   `t`. `<150bp` kills §6; `≥150bp with t≥2` is the first defensible ARK-specific number.
2. **Nested re-selection (≈15 min, tests the selection bias the docs never test).** Re-select the
   best (lookback, hold) cell using only cohort dates before 2024-07-01, then report that cell on
   the ~20 dates from 2024-07-01. If the re-selected cell is not 126/21, or its holdout mean is
   `<150bp`, §6 is the max of a grid, not an edge. State in advance that ~20 dates can only detect
   a large effect.
3. **Turn flow into active trades (≈1–2 h, the only way to give §4 a fair shot).** Download ARK's
   free daily holdings **levels**, subtract the pro-rata creation/redemption leg from each fund-day
   delta, keep only the residual active trades, dedupe the `27,627` arktrack∩tsekityam keys, and
   re-run A2/A3 with date-clustered t on unique events. Without this, every flow conclusion tests a
   mixture of signal and AP plumbing.

*(Not recommended next: another premium/cointegration variant. Those hypotheses got real OOS tests
and died; re-parameterising them is the cheapest way to generate a false positive.)*

---

## 6. What this work does and does not rule out

**Actually ruled out:** the ETF-vs-constituent ratio as a pairs trade (mirror degeneracy); classic
cointegration on these 34 pairs at 2023→2024; ETF-vs-ETF basis/lead-lag; flow→vol as specified;
premium as a directional or vol signal; ARK-specific overnight drift; flow-following after
realistic cost; the ADSK/PTC pair (β not identifiable).

**Not ruled out:** an ARK-specific dispersion premium (the control that would decide it was never
run); any flow signal measured on **active** trades rather than net basket deltas (the data cannot
see the difference yet); anything intraday, options-based (untested, trader §8 item 3), or
holdings-level. "No edge found" here means the cheap hypotheses failed — it is not evidence there
is no edge.

**Residual risks of this review:** §1–§3 P&L numbers are unverifiable (code deleted), so I can
only audit their internal consistency; my one-sided/dose-response recomputations use the
definitions stated above (threshold sweep given); no network, no intraday quotes, no borrow data;
the candle table is mutable and my counts are a snapshot.
