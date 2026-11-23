# ARK edge research — conclusion record

Status: **closed**, one item still open (within-fund momentum, §6). Five
hypotheses were measured to completion; four are dead, one is statistically
alive but economically dead. This file exists so nobody spends another session
on the dead ones.

Numbers below are the audited results. They are a record, not a live metric —
do not update them, and do not quote them as current behaviour.

## 1. ETF vs constituent log-ratio — mean reversion is a mirror, not an edge

Trading `ln(ARKx) - ln(constituent)` as a spread, 34 pairs, both the
mean-reversion and the sign-flipped (momentum) form:

- The two books are **exact mirrors**: `corr(ann_rev, ann_mom) = -1.0000`,
  identical trade bars, same 34 pairs.
- Median Sharpe `-0.40` → `+0.42` when the sign flips is arithmetic, not alpha.
  One direction of one trade counted twice.
- Expected value ≈ `-round-trip cost`: every "win" is the mirror of a loss.

Dead. There is no second signal hiding behind the sign.

## 2. Multi-pair book (2 → 4 → 8 → 16 → 34 pairs)

Laddering the pairs book to see whether diversification rescues it:

- Best rung `+0.15%` against a 50/50 buy-and-hold `+42.53%`.
- Diversification **did** fix the tail — excess kurtosis `12.0 → 6.0`.
- It never fixed the sign. The mean stayed at minus costs.

Dead. Adding pairs bought smoother losses, not returns.

## 3. Classic cointegration pairs (OLS hedge ratio, ADF + half-life screen)

Formation 2023, out-of-sample 2024, `13/34` pairs passed at `α = 0.10`
(~`3.4` expected by chance — i.e. the screen is close to a coin flip with a
loose threshold):

- 2024 OOS `-11.89%`, Sharpe `-0.93`.
- Friction is only `5%` of the loss, so this is not an execution problem.
- The screen had **no `|beta|` gate**. `ARKK/TDOC` with `β = -0.024` is a naked
  directional bet on the fund leg, not a hedge, and it lost `-7,040`.

Dead. Rejected at the source: the cointegration statistic did not survive the
year change, and the book's worst line was structurally unhedged.

## 4. ARK flow front-running (6 years, 96,898 delta rows)

T+1 buy-minus-sell excess was `+0.21%` pooled — but:

- Dose-response was **inverted in 3 of 7 years**.
- The monthly spread flipped sign **6 times in 13 months**.

Reviewer's correction, and it matters: `ark_flow.csv` is mostly
creation/redemption basket plumbing (`90.2%` one-sided fund-days), so these
deltas are **not** ARK's active trades. Even taken at face value the sign is
unstable.

Dead. Pooled mean is one regime outvoting the rest.

## 5. NAV premium decay — real, robust, and too small

The premium/discount reversion is the only statistically solid finding:

- Discount days: `r_price +55.3bp` vs `r_nav -2.4bp`, paired `t = 28.2`.
- Excess over each fund's own unconditional mean: `+52.6bp`, `t = 8.83`.
- Still present post-2019: `+0.42%` (`t = 22`).

Statistically alive, economically dead:

- ~`1–2%` of NAV per event, ~`10bps/yr` of portfolio return.
- The `09:30` auction absorbs the reversion: excess gap `+7.6bp` vs intraday
  `+8.3bp`; gross `20.6bps/day` against `15–25bps` round-trip cost.

Do not rebuild this as a strategy. It is a microstructure fact, not a trade.

## 6. WITHIN-FUND MOMENTUM, LONG/SHORT — FRAGILE (not dead, not tradeable)

Long top-k / short bottom-k ARK constituents by 126-day lookback, hold 21 days,
k=3, non-overlapping cohorts. **Every number below was reproduced independently
by a second reviewer agent (22/22 claims).**

| quantity | value |
|---|---|
| ARK L-S | **+381bp/cohort**, t=3.47, 68 distinct dates (329 fund-cohorts, 0 non-session dates) |
| non-ARK control, sleeve-size matched | +135bp → **diff +246bp, t(diff)=1.87, 95% CI [-17, +509]** |
| non-ARK control, wide cross-section (3 of 451) | +339bp → diff **+42bp** — but that control's own SE is 339bp (t=0.12), median cohort +15bp, and its 5 best cohorts are 111% of its total (MARA, CVNA, RIOT, CELH, MSTR) |
| ±15% clipping | diff +164bp (t=2.48) |
| hold 10 | +94bp over 142 dates |
| ex-NVDA+PLTR | +257bp (t=2.43) |
| ex-NVDA+PLTR **and** clipped | **+132bp — under the 150bp bar** |
| ex-top-5 | +185bp (t=1.85); top-10 names = 57% of the edge |
| per-day rate | 15.9 / 17.6 / 18.1 bp at hold 5 / 10 / 21 → drift, not an event |
| grid | 12 cells, span 69–381bp, **8 of 12 below the bar** (though 12/12 positive) |

The pre-registered rule was satisfied on the sleeve-matched control, so the
label is not DEAD. It is also not an edge: the margin is 0.73 SE, the CI
contains zero, the wide control *brackets* rather than refutes (+246 to +42 by
selection intensity — its t=0.12 is a failure to reject, not evidence of
equivalence), and the two named weaknesses combined land at +132bp.

**Reading:** this is most likely generic cross-sectional momentum that an
ARK-heavy basket inherits, not an ARK-specific effect. Cost kills the 5-day
variant outright (120bp drag against +79bp gross).

**The one test that would settle it (NOT RUN):** a liquidity/size-matched twin
control — match each ARK constituent to a non-ARK name on trailing-60d median
dollar volume plus price decile, then re-run. If the increment stays above
150bp, it is real; if it collapses, it was a size-bucket artifact.

**Confidence caveat from the code audit:** the flow-magnitude variables feeding
this family come from `panel.py:45-56`, which retains only ~40% of flow rows
(5-6 of 35-50 trades/day in 2020-21), and a premium-after-gap lookahead is still
live in G1/G2/G4. Neither touches the L-S price construction itself, which was
verified clean (entry T+1 open / exit close, bar-close vs official close
correlation 0.9996).


## 7. Data caveat — `ark_flow.csv`

Structure, before any reuse:

- It is **two** independent upstream feeds, not three. `arktrack` and
  `tsekityam` agree `99.5%` on side and `97.7%` exact.
- `cathiesark` is the only independent source, and its date column is a
  **publication** date — one day later than the trade date. Any join on it
  silently shifts a day.
- Source priority for a rebuild: reconcile `arktrack`/`tsekityam` first, treat
  `cathiesark` as a dated cross-check only.

Known defects to fix before reuse:

- a zero-share buy at line 97;
- 8 float-subtraction share counts;
- 4 `IZRL`/`ICCM` odd-lot lines.

## 8. What was removed (audit trail)

These four strategy modules and their `strats/wip/ark_*.json` configs were
deleted as measured failures — all untracked, none ever passed:

- `src/bt/strategies/ark_ratio_mr_dsl.py` — mirror-sign degeneracy (§1).
- `src/bt/strategies/ark_ratio_mom_dsl.py` — same book, sign flipped (§1).
- `src/bt/strategies/ark_ratio_book.py` — multi-pair ladder (§2).
- `src/bt/strategies/ark_pairs_classic_dsl.py` — no cointegration OOS, OOS
  negative (§3); its module docstring also breached the `AGENTS.md` limit
  (134 lines against a limit of 50, with 9 driftable-number lines).

`scripts/ark_pairs_screen.py` was **kept** and made self-contained: it is a
read-only cointegration screen with no dependency on the retired book.

## 9. If you are here to reopen something

- §1–§4 are closed on measured results, not on a hunch. Reopening needs a new
  data source, not a new parameter.
- §5 is closed on economics. Re-verify only if auction microstructure changes.
- §6 is the only place new work belongs, and only with the control gate.
