# ARK research — data & code audit

**Scope.** Independent, read-only re-derivation of the ARK session's claims:
`assets/ark_nav.csv`, `assets/ark_flow.csv`, the four untracked strategy modules,
`scripts/ark_pairs_screen.py`, and the DB state after the revert.
**Method.** Every number below was recomputed from the CSVs / a `mode=ro` sqlite URI with
scripts written for this audit. Nothing was modified, deleted, migrated, or re-fetched.
**DB read via** `file:/home/nasrt/Documents/code/dev/ibkr/data/ibkr.db?mode=ro`.

---

> ## ERRATA — state change after this audit was written
>
> **Added in the second pass. The body below is unchanged; it is a historical record.**
>
> **What changed.** Every file in §7 ("Part D — the five code files") was acted on after I
> filed this audit:
>
> | path | this audit said | current state |
> |---|---|---|
> | `ark_ratio_mom_dsl.py` | DELETE | **deleted** |
> | `ark_ratio_mr_dsl.py` | DELETE | **deleted** |
> | `ark_ratio_book.py` | DELETE with siblings | **deleted** |
> | `ark_pairs_classic_dsl.py` | KEEP or delete as a unit | **deleted** |
> | all `strats/wip/ark_*.json` (§7/§8: 16–17 configs) | must go in the same commit | **deleted** (`ls strats/wip` = 11 configs, none ARK) |
> | `scripts/ark_pairs_screen.py` | KEEP, but dies with `ark_pairs_classic_dsl` (N3) | **kept and made self-contained** — no strategy-module import; N3 resolved *better* than I recommended |
> | `py/research/ark_edge/*.py` | deleted mid-audit, blocking (B1) | still absent from the repo, **but the four files survive as scratch copies** in `/tmp/ark_edge/{panel,a_flow_vol,a2_flow_attention,f_overnight}.py` (they import `research.ark_edge.panel`, i.e. they are the pre-deletion code). The `py/research/` directory itself is now gone. |
>
> **Which findings survive the deletions.**
>
> * **MOOT — the object is gone, no action possible or needed:** B1 (as a *ship* blocker: the code
>   is deleted, so nothing can be restored to the report's support — the correct disposition is now
>   "figures stay UNVERIFIED", see the second pass), M7 (lazy private import), M8 (docstring/length
>   rules in deleted files), M9 (`run_ratio_book` 85 LOC), N1 (the two dangling refs lived in
>   deleted modules; `docs/research/ark_ratio_*.md` remain absent), §7 Part D (all four rows are
>   now history), §8 "must go in the same commit" and §10 items 1 and 9 (as *repo* actions).
> * **STILL ACTIONABLE — none of these depended on the deleted code:** B2 (undisclosed 5-fund
>   universe on the era claim), B3 (`52,714` / `0.3%` do not reconcile → 1.10%), B4 (six `PASS`
>   rows under a "nothing is tradeable" headline), M1 (three sources are two), M2 (cathiesark
>   1-day labelling offset), M3 (`shares=0` and float-artifact rows), M4 (`notional` 100% empty
>   for 2 of 3 sources), M5 (flow covers 8 of 14 NAV funds), M6 (`make check` writes), N2 (stale
>   `__pycache__` for the dropped migrations), N4 (NAV cross-check validates `market_price`), N5
>   (quarter-boundary starts), N6 (no fetch script), §10 items 2–8.
> * **SUPERSEDED — the fact it records is no longer true:** N3's "dies if `ark_pairs_classic_dsl`
>   goes" (it didn't), §8's "24 untracked paths" (now **5**: the 2 CSVs, `docs/research/`,
>   `docs/review/`, `scripts/ark_pairs_screen.py`).
> * **UNCHANGED and still true:** the CSVs are sound, the NAV file is arithmetically impeccable,
>   the price-vs-NAV decomposition reproduces, and the *report* (`docs/research/ark_edge_trader.md`)
>   is still prose-only, so B1's underlying complaint — no artifact in the repo reproduces a
>   headline number — is now *worse*, not better: the analysis that did exist is not under
>   version control anywhere.
>
> **One finding is downgraded, not withdrawn.** B1 said "restore the scripts or stamp every figure
> UNVERIFIED". The `/tmp` copies mean the arithmetic *is* re-runnable, and the second pass below
> re-ran the parts that matter independently. But `/tmp` is not a repository: the recommendation
> stands as "commit the analysis, or mark the figures UNVERIFIED".

> **VERDICT: DO NOT SHIP.** The CSVs are sound and the flagship *mechanics* finding
> reproduces, but (1) the code that produced every headline number was deleted mid-audit,
> (2) the central era conclusion flips sign when the undisclosed universe is widened,
> (3) two headline numbers do not reproduce, and (4) the verdict table reads PASS on all
> six trades against a headline that says nothing is tradeable.

---

## 0. The tree moved during this audit

This is not a side note — it bounds what "state after the revert" can mean.

| time | event |
|---|---|
| 15:23–15:29 | `py/research/ark_edge/{panel,f_overnight,a_flow_vol,a2_flow_attention}.py` present (gitignored) |
| 15:34 | `py/research/` emptied — **all four analysis scripts deleted** |
| 15:36 | `py/docs/research/ark_edge_trader.md` created (338 lines) |

`py/research/` now exists and is empty. Consequence: **no artifact in the repo reproduces any
headline number.** `panel.py` (the shared loader with the source-precedence rule) is gone too.

Also note the git root is `/home/nasrt/Documents/code/dev/ibkr`, so `git status` paths are
`py/`-prefixed; `py/docs/research/` and `py/research/` are untracked.

---

## 1. BLOCKING

### B1. Headline analyses are unreproducible — their code was deleted
`research/ark_edge/*.py` (4 files) held the only implementation of the loader, the
source-precedence panel, and the overnight/intraday split. Deleted at 15:34, before the
report was written at 15:36. `docs/research/ark_edge_trader.md` is prose only.
**Risk:** every figure in the report becomes an unverifiable assertion; a reviewer cannot
redraw a single table. **Fix:** restore the scripts under version control before the report
is merged, or mark every figure `UNVERIFIED`.

### B2. The era conclusion depends on an undisclosed universe, and reverses without it
`docs/research/ark_edge_trader.md:58` claims `2019-23: 7-40/yr` discount fund-days at
`<-25bps`, supporting "the premium trade died in 2018".

This reproduces **exactly** — but only on the 5 funds that happen to have candles
(`ARKK, ARKW, ARKQ, ARKG, ARKF`). Fund-days `premium_pct <= -0.25` per year:

```
ARKK-only   2015:42 2016:44 2017:7 2018:2 2019:0 2020:4 2021:1 2022:1 2023:0 2024:0 2025:6 2026:1
5 candle    2015:170 2016:206 2017:32 2018:18 2019:7 2020:40 2021:19 2022:21 2023:7 2024:32 2025:16 2026:20
all 14      2015:170 2016:224 2017:49 2018:143 2019:197 2020:107 2021:127 2022:231 2023:310 2024:472 2025:308 2026:461
```

Query: `count(*) where premium_pct <= -0.25 group by year, fund-set`.

- `170-206` and `7-40` are the **5-candle** row, verbatim. The universe is never stated.
- `ARKX, PRNT, IZRL, ARKB, ARKT, ARKD, ARKI, ARKE, ARKY` are silently excluded — yet these are
  the funds with the *highest* discount frequency (`IZRL` 15.7%, `ARKD` 13.5%, `ARKT` 12.2%,
  `PRNT` 11.6% of days at `<=-0.5%`).
- On all 14 funds the frequency **grows** (310 in 2023, 472 in 2024). "Died in 2018" is false
  for the file as published.

**Risk:** the report's headline conclusion is an artifact of which symbols were backfilled.
**Fix:** state the universe in the claim; either restrict the claim explicitly to the 5
candle-covered US funds or re-run on all 14.

### B3. `52,714 line items` and `0.3%` do not reconcile
`docs/research/ark_edge_trader.md:43` and `:120` (`52,714 line items, 259 dates`);
`:40` and `:252` (`0.3% of (fund, ticker, date) triples contain both a buy and a sell`).

| quantity | report | recomputed | status |
|---|---|---|---|
| line items in full-coverage window | 52,714 / 259 dates | **52,493 / 260 dates** (priceable rows, 2020-10-19..2021-10-28) | **~PASS** (0.4% off) |
| triples with both sides | 0.3% | **1.10%** (713/64,901) | **FAIL** (3.7x) |

```
rows in window                      = 85,185
rows in window, ticker has candles   = 52,493      <- nearest to 52,714
(fund,ticker,date) with both sides   = 713 / 64,901 = 1.10%
5-liquid only                        = 692 / 58,484 = 1.18%
priceable only                       = 492 / 40,213 = 1.22%
(ticker,date) ignoring fund          = 3,465 / 45,717 = 7.58%
```
No construction yields 0.3%. The 52,714 figure is 0.4% off the nearest reconstruction —
consistent with the candle table being **mutable and re-synced mid-audit**, which is itself a
reproducibility problem: the number cannot be pinned to a snapshot.
The conclusion survives (netting is rare) but the quoted magnitude is wrong by ~4x.

### B4. The verdict table contradicts the report's own headline
`docs/research/ark_edge_trader.md:58-63` — all six rows end `**PASS**`. `:7` and `:107`,
`:142`, `:172`, `:196`, `:219`, `:237` say "nothing here is tradeable", "flat is the position".
A reader skimming the table reads six passes as "trade six things".
**Fix:** the column is measuring "past the desk check", not "tradeable" — rename it
(`CHECK` / `REJECTED`) or move the verdict to its own column.

---

## 2. MAJOR

### M1. The three sources are **two** sources — the brief's reconciliation numbers are wrong
The brief claims "sources do NOT reconcile — 86.4% side agreement, 8.6% exact share counts,
median 62% relative error". Those are the **`arktrack` vs `cathiesark` pair only**.
Recomputed per pair at zero shift (keys = `(date, fund, ticker)`):

| pair | n | side agree | exact shares | median rel err |
|---|---|---|---|---|
| arktrack vs tsekityam | 27,763 | **99.5%** | **97.7%** | 0.0% |
| arktrack vs cathiesark | 4,219 | 86.4% | 8.6% | 51.9% |
| tsekityam vs cathiesark | 2,318 | 80.5% | 0.3% | 84.2% |
| **true pooled (all pairs)** | 34,300 | **96.6%** | **80.1%** | 0.0% |

So `86.4 / 8.6` is a single pair masquerading as a pooled statistic (the median 62% does not
reproduce at all; I get 51.9% for that pair, 0.0% pooled). `arktrack` and `tsekityam` are the
**same upstream feed** (both derived from ARK's published daily holdings).

**Risk:** treating the file as three independent confirmations double-counts. There is roughly
**one** independent derived signal. `docs/research/ark_edge_trader.md:22` states this correctly
("They are the same feed") — so the report is right and the brief is wrong; the danger is any
downstream consumer reading the 96,898 rows as three votes.
Mean relative error is 131.6% against a 0.0% median — a fat tail of corrupted rows (see M3).

### M2. The 1-day offset is REAL, and it is a cathiesark labelling offset — not a delta artifact
Direct answer to the question asked. Offset sweep, `arktrack` dates shifted before matching:

| shift | arktrack vs tsekityam (both delta-derived) | arktrack vs cathiesark (reported trades) |
|---|---|---|
| -1 | 75.1% side / 5.0% exact | **90.2% / 20.0%** |
| 0 | **99.5% / 97.7%** | 86.4% / 8.6% |
| +1 | 76.1% / 8.9% | 75.5% / 0.0% |

- The control settles it: **two independent delta-derived sources peak at shift 0.** Delta
  derivation does *not* introduce a one-day lag.
- The offset lives entirely on the **reported-trades** side. `cathiesark`'s date is one day
  *later* than the holdings change it corresponds to (publication date, not effective date).
- The claimed improvement `90.2% / 20%` **reproduces exactly**. The claimed median relative
  error `30.7%` **does not** — I get **13.7%** at that alignment.

**So: real semantic offset, not an artifact.** But even aligned, only 20% of share counts match
exactly, so `cathiesark` is a partial, noisy rendition of the same events.

### M3. Corrupted rows from float delta-derivation, including a zero-share trade
- `assets/ark_flow.csv:97` — `2020-08-31,ARKK,PD,buy,0.0,,cathiesark`. A **zero-share buy**.
  Invalid record; any signed-notional or VWAP aggregation silently includes it.
- 8 non-integer share counts, all `tsekityam`, all float-subtraction artifacts:
  `ark_flow.csv:30435` `IZRL/INCR` `4000618.1399999997`; `:34761` `31466.000000000116`;
  `:38511` `29515.709999999963`. **This is mechanical proof** the column is a float delta of
  consecutive holdings snapshots, not a reported trade.
- `ark_flow.csv:66237,66242,66622,66939` — `IZRL/ICCM` sells 2,323,089.13, buys back the same
  2,323,089.13 two days later, sells it again the next day. Signature of a missing/duplicated
  snapshot, not trading. `IZRL` at 2.3M shares is also implausible against its ~$100M AUM.

### M4. `notional` is arktrack-only — 49.8% of the file has none
`notional` empty: `arktrack` 0/48,608; `cathiesark` 16,557/16,557 (**100%**);
`tsekityam` 31,733/31,733 (**100%**). Any dollar-flow, notional-quintile or
"size of trade" analysis is a single-source, single-12-month-regime study wearing a
96,898-row sample size.

### M5. Flow covers 8 of the 14 NAV funds
`ark_flow.csv` funds: `ARKK 23,048, ARKG 21,870, ARKW 17,474, ARKQ 14,039, ARKF 12,312,
ARKX 4,521, PRNT 1,893, IZRL 1,741`. `ARKB` (a large spot-BTC fund), `ARKD, ARKE, ARKI, ARKT,
ARKY` are absent entirely. Any NAV↔flow join on those funds is empty.

### M6. `make check` is not a read-only gate — it rewrites files
`Makefile:44 check: lint format typecheck test-fast`; `Makefile:33 format: uv run ruff format $(SRC)`
(**write** mode). The brief's "`make check` passes" therefore also silently reformatted every
file under `src/`, **including the untracked strategy modules** — their current formatting is
ruff-normalised, not author-chosen, and any local edit would be overwritten by the next check.
Test count **independently confirmed**: `pytest --collect-only -q -m 'not slow and not db'` →
`814/883 tests collected (69 deselected)`. I did not run `make check` (it writes).

### M7. Lazy private import violates two explicit AGENTS.md rules
`src/bt/strategies/ark_pairs_classic_dsl.py:279`
```python
from src.utils import _get_ols_fit_model
```
AGENTS.md: "**Top level imports only** No lazy imports." and forbids importing a private
cross-module symbol. The in-code justification ("deferred so this strategy module stays
import-light") contradicts the rule. **Real risk:** the failure mode is deferred — the import
raises mid-backtest at the first formation fit, not at load time. The strategy passes
`_discovery()` and *then* dies.

### M8. Docstrings violate the no-driftable-numbers rule, twice over the 50-line cap
| file | module docstring | driftable-number lines |
|---|---|---|
| `ark_pairs_classic_dsl.py:1` | **134 lines** (>50) | **9** |
| `ark_ratio_book.py:1` | **78 lines** (>50) | 0 |
| `ark_ratio_mom_dsl.py:1` | 30 lines | **5** |
| `ark_ratio_mr_dsl.py:1` | 24 lines | **1** |

Examples: `ark_ratio_mr_dsl.py:4-7` "total return `-3.79%`, Sharpe `-0.43`, 12 trades";
`ark_ratio_mom_dsl.py:10-16` "positive on only 10 of 34 pairs", "drifted 0.84 over 2024",
"~27 daily standard deviations", "4 of 6 entries", "5 of 6 exits";
`ark_ratio_mom_dsl.py:20-23` hardcodes `Z_EXIT=0.5, Z_STOP=3.5, MAX_HOLD=20,
ratio_window 60, z_entry 2.0, size 0.25` in prose — six values guaranteed to drift from code.
`ark_pairs_classic_dsl.py:1` also cites pair betas `-0.024`, `0.026`, `-0.210` and
P&L `-7,040`, `-5,234`, `-1,734`.

### M9. `run_ratio_book` is 85 LOC — over the hard 50-line cap
`src/bt/strategies/ark_ratio_book.py:245`, body ends `:329`. AGENTS.md: "Functions: ≤ 50 LOC".
Three clean seams already exist in the comments (warmup accumulation / exit block /
arbitration+entry).

---

## 3. MINOR

- **N1. Three dangling doc references.** `ark_ratio_mr_dsl.py:3` and
  `ark_pairs_classic_dsl.py:6` cite `docs/research/ark_ratio_trade_audit.md`;
  `ark_ratio_mr_dsl.py:20` cites `docs/research/ark_ratio_run_1y.txt`;
  `ark_pairs_classic_dsl.py:12` cites `docs/research/ark_ratio_sign_flip_comparison.md`.
  **None exists** — `docs/research/` contains only `ark_edge_trader.md`. The modules claim an
  audit provenance that is not in the repo, and `ark_ratio_mr_dsl.py:19-21` claims
  "re-verified byte-for-byte against `docs/research/ark_ratio_run_1y.txt`" — unverifiable.
- **N2. Stale `__pycache__` for deleted migrations.**
  `src/db/migrations/versions/__pycache__/` still holds `data_0004_etf_holdings.cpython-314.pyc`
  (Nov 20 09:54), `data_0005_etf_source_provenance.cpython-314.pyc` (13:05) — **and a second,
  different `data_0004_split_ea_series.cpython-314.pyc` (Nov 11 14:28)**. The revert cleaned the
  `.py` files but not the bytecode. Harmless at runtime (a lone `__pycache__` `.pyc` is never
  imported without its source) but it is exactly the leftover reference Part E asked for, and
  two competing `data_0004` migrations in history is a chain hazard worth knowing about.
- **N3. `scripts/ark_pairs_screen.py` is safe to keep as a research tool — but not independently.**
  It is genuinely read-only (no DB writes, no backtest, prints a screen table only) and has
  **zero** driftable numbers in its docstring. Two caveats: `main` is 53 LOC
  (`scripts/ark_pairs_screen.py:64`, marginal >50), and `:31` hard-imports
  `ark_pairs_classic_dsl`, so it dies with an `ImportError` if that module is deleted — the
  "keep the screen, drop the strategy" plan is not viable. It also reads
  `cfg.strategy_params["pairs"]` (`:88`) with no key check, and takes the config path from
  `sys.argv[1]` with no argparse (`:120`).
- **N4. NAV cross-check validates `market_price`, not NAV.** The "1,735 overlapping days, 0
  mismatches >0.5%" claim **reproduces exactly** (`|mp - candle_close| / close`: max **0.4499%**,
  median 0.0276%, 1735 days). But the max sits 0.05pp under the threshold, and the tolerance is
  ~16x the median — it was chosen to pass. Compared against `nav` instead, **5 days exceed 0.5%**
  (max 1.0755%), so the claim should not be read as validating the NAV column.
- **N5. Possible page-truncation in the NAV fetch.** 5 of 14 funds begin on quarter boundaries
  with perfectly continuous coverage afterwards: `ARKD 2026-01-02`, `ARKI 2026-04-01`,
  `ARKE 2026-07-01`, `ARKT 2025-10-01`, `ARKY 2026-08-19`. The signature of a `history/NNN.json`
  page limit, not fund inception. `ARKB 2024-01-11`, `ARKX 2021-03-30`, `ARKF 2019-02-04`,
  `PRNT 2016-07-19`, `IZRL 2017-12-05` are all genuine inception dates, so the fetcher is not
  systematically broken — but I cannot verify the five round ones without network access.
  **UNVERIFIABLE.**
- **N6. No fetch script in the repo.** `grep -rl 'daggerok|arktrack|tsekityam|cathiesark'` over
  `*.py/*.md/*.json/*.toml` returns **only** `research/ark_edge/panel.py` (now deleted). The
  CSVs are 6.5 MB of un-reproducible external data with no committed builder.

---

## 4. NIT

- `ark_pairs_classic_dsl.py:199` — `assert not ok or self.sd > 0.0` is tautological: `ok` already
  requires `self.sd > 0.0`. Dead assertion.
- `ark_ratio_mr_dsl.py:38`, `ark_ratio_mom_dsl.py:44`, `ark_pairs_classic_dsl.py:522` — `on_candle`
  lacks `-> None`. Matches the prevailing (if non-compliant) codebase convention, so not a
  session defect.
- `ark_ratio_book.py:1-13` — the sibling table is correct (two rules per module) but the header
  rows are visually ambiguous; a first-time reader can read it as one rule per module.
- `ark_ratio_mom_dsl.py` is a controlled experiment whose own `ark_pairs_classic_dsl.py:8-20`
  docstring records the finding that rolling reversion and momentum are **mirror images**
  (annual-return correlation `-1.0000`, identical trade counts). That is a valuable negative
  result — and the *reason* both `ark_ratio_*` modules are correctly candidates for deletion.
- `ark_pairs_classic_dsl.py` is the only one of the four with independent merit: frozen
  formation fit, ADF + half-life screen, fresh-breach entry to stop re-entry churn, post-stop
  lockout, and an explicitly documented beta-size limitation. Its OLS is **correct** —
  `_get_ols_fit_model` (`src/utils.py:122-123`) log-transforms both legs, so `alpha`/`beta` are
  in log space and `spread = log_f - beta*log_c - alpha` (`:284`) is dimensionally right.
  `_half_life` (`:247`) correctly maps `phi>=1 -> inf` and `phi<=0 -> 0.0` (screened, not clamped).

---

## 5. Claims verified

| # | Claim | Result |
|---|---|---|
| 1 | `ark_nav.csv` 21,945 rows | **PASS** (21,945 parsed; 21,946 lines − header) |
| 2 | 14 funds | **PASS** |
| 3 | `2014-09-30 .. 2026-11-18` | **PASS** (3,053 distinct dates; 0 weekend dates) |
| 4 | columns `date,fund,market_price,nav,premium_pct,source` | **PASS** |
| 5 | `premium_pct = (mp-nav)/nav*100` across ALL rows | **PASS** — max abs err **0.00005** (4dp rounding), 0 rows >0.001; 0 rows with `nav<=0`/`mp<=0`; **0 rows with `|premium|>5`** |
| 6 | NAV `(date,fund)` unique, no nulls | **PASS** (0 dups, 0 empty) |
| 7 | Continuous per-fund NAV coverage | **PASS** (0 gaps inside each fund's own range) |
| 8 | ARKK cross-check: 1,735 days, 0 mismatches >0.5% | **PASS** (1,735; max 0.4499%) |
| 9 | `ark_flow.csv` 96,898 rows | **PASS** (96,898 parsed — note 96,901 lines: 2 records contain embedded newlines) |
| 10 | 1,151 dates, `2020-08-21 .. 2026-05-15` | **PASS** |
| 11 | source split 48,608 / 31,733 / 16,557 | **PASS** (exact) |
| 12 | No duplicate `(date,fund,ticker,side)` keys | **PASS** (0 dups; also 0 full-row dups) |
| 13 | No weekend dates in flow | **PASS** (0) |
| 14 | `shares > 0` | **FAIL** — `ark_flow.csv:97` `shares=0.0` |
| 15 | Integer share counts | **FAIL** — 8 float-artifact values (M3) |
| 16 | `notional` usable | **FAIL** — 100% empty for 2 of 3 sources (M4) |
| 17 | No contradictory sides across sources | **FAIL** — 713 keys (2.4% of multi-source keys), e.g. `2020-10-20 ARKK/MCRB` arktrack `buy` vs cathiesark `sell` |
| 18 | Sources do **not** reconcile: 86.4% / 8.6% / 62% | **FAIL** — pair-only; pooled 96.6% / 80.1% / 0.0% (M1) |
| 19 | Shift cathiesark +1 → 90.2% / 20% / 30.7% | **PARTIAL** — 90.2% and 20.0% **PASS** exactly; median **FAIL** (13.7%, not 30.7%) |
| 20 | Offset is a delta-derivation artifact | **FAIL (refuted)** — delta-derived control peaks at shift 0 (M2) |
| 21 | Migrations reduced to `data_0001/0002/0003` | **PASS** (`peewee_migration`: exactly those three) |
| 22 | `etf_holding` / `etf_trade` tables gone | **PASS** (no `etf*`/`*holding*` table in `sqlite_master`) |
| 23 | `candle` intact, 5,089,807 rows | **PASS** (exact) |
| 24 | `make check` → 814 passed, 69 deselected | **PASS** (collection: `814/883, 69 deselected`); **not run** — `format` writes |
| 25 | No code/doc references `etf_holding`/`etf_trade`/`data_0004`/`data_0005`/`ark ingest` | **PASS** (0 matches in `*.py/*.md/*.json/*.toml`); stale `.pyc` remain (N2) |
| 26 | Repo clean after revert | **FAIL** — 24 untracked paths: 2 CSVs, 5 code files, 17 wip JSON; `docs/research/` + empty `research/` |
| 27 | Analysis (i) price-driven, not NAV-staleness | **PASS — reproduces, robust** |
| 28 | Analysis (ii) edge vanished post-2019 | **PARTIAL — reproduces on 5 funds, reverses on 14 (B2)** |
| 29 | Analysis (iii) overnight/intraday, pooled n=28 | **FAIL — n=28 reproduces, result is null** |

---

## 6. The three headline analyses, re-derived

### (i) Price-driven, not NAV-staleness — REPRODUCES, and is the strongest result in the set
On days with `premium_pct <= -0.5`, next-day `market_price` return vs next-day `nav` return
(next row per fund; verified adjacent, no gaps >4 calendar days):

```
pooled   n=1,059   r_price=+55.3bp   r_nav=-2.4bp   diff=+57.7bp   paired t=+28.17
dPremium on discount days: n=1,067  mean=+0.571pp  t=28.4
```

Robustness I added (not in the brief): the raw number is **not** just unconditional drift.

```
unconditional next-day r_price : all14 +5.3bp | 5 liquid +7.0bp
discount-day next-day r_price  : all14 +55.8bp | 5 liquid +92.9bp
excess over each fund's own mean: n=1,067  +52.6bp  t=8.83
```

Every one of the 12 funds with n>=5 shows `gap > 0`. The effect is **stronger** on the liquid
five (+85.8bp excess) than on the full 14 (+50.4bp), so it is *not* driven by illiquid NAV.
Contrary to a "pooled n is inflated" worry, the concern runs the other way: `IZRL` (353 obs) and
`PRNT` (301 obs) are 62% of the pool and they *dilute* it. Caveat: same-day cross-fund
correlation means effective significance is below t=8.83.

### (ii) Era split — PARTIAL: direction reproduces, the "vanished" label does not
Discount-day frequency collapses for the big liquid funds (`prem <= -0.5`, pre vs post 2019-12-31):
`ARKG 7.9% -> 0.1%`, `ARKK 3.7% -> 0.2%`, `ARKQ 3.1% -> 0.1%`, `ARKW 4.2% -> 0.9%`.
Mean next-day premium change after a discount day, pooled:

```
pre  n=401  mean +0.817%  sd 0.81  t=20.24
post n=666  mean +0.422%  sd 0.49  t=22.33
difference +0.394pp, t=8.85  (significant decline)
```

Boundary-insensitive: at 2017-12-31 +0.97 -> +0.44; at 2021-12-31 +0.81 -> +0.38.
But **"vanished" is wrong**: post-2019 reversion is still +0.42% at t=22 on n=666. It *halved*;
it did not die. And it is concentrated, not gone — `IZRL` (15.9% -> 15.6% discount frequency) and
`PRNT` (8.2% -> **13.4%**) show no decay in frequency at all. The post-2019 pool is 500/666
`IZRL`+`PRNT`. Combined with B2, the era story is a statement about *which funds you include*.

### (iii) Execution, overnight vs intraday — n=28 reproduces EXACTLY, and it is a null result
`gap(t+1) = open(t+1)/close(t) - 1`, `intra(t+1) = close(t+1)/open(t+1) - 1`, from 1h candles
aggregated to `>=5 bars/day` with the 09:30 bar open and last bar close, on discount days:

```
ARKK  n= 3   gap +140.4bp  intra  -99.0bp
ARKW  n=15   gap  +58.3bp  intra  -36.5bp
ARKQ  n= 2   gap -445.4bp  intra -467.5bp
ARKG  n= 1   gap +333.2bp  intra -268.7bp
ARKF  n= 7   gap  -25.5bp  intra +167.2bp
POOLED n=28  gap +20.0bp   intra  -31.3bp   total -11.3bp
             gap t=0.46     intra t=-0.53
```

`n=28` is hit exactly. But:
- **|t| < 0.6 on both legs. Indistinguishable from zero.** The point estimate of the *total* is
  **negative**.
- 2 `ARKQ` observations at −445bp drive the pooled gap; `ARKQ` is the only fund with
  `stdev>0` and `n>2` yet its contribution is a single repricing.
- **Selection:** candles exist only for 5 funds from `2019-12-24` (`ARKF` from `2021-03-08`).
  Discount days are overwhelmingly pre-2020 (`ARKKK`: 48 of 51). The 28 observations are the
  **rare post-2019 remnant** — 4% of the 1,067 discount-day population, in the regime where the
  frequency had already collapsed. They cannot speak to the regime that carried the effect.

**Verdict on (iii): does not reproduce as an actionable finding.** No conclusion about
overnight-vs-intraday capture is supportable at n=28 with these t-stats. The report's
`+20.6bps gross, ~10bps after the first hour` (`:59`) is a *different* claim and I could not
tie it to any code or construction — **UNVERIFIABLE**.

---

## 7. Part D — the five code files, ranked

| file | verdict | why |
|---|---|---|
| `ark_ratio_mom_dsl.py` (46 L) | **DELETE** | 7-line wrapper; the controlled experiment already concluded "mirror image of reversion" — no standalone value. 5 driftable-number lines (M8). |
| `ark_ratio_mr_dsl.py` (41 L) | **DELETE** | 11-line wrapper over shared code; 1 driftable line; cites two nonexistent docs (N1). |
| `ark_ratio_book.py` (329 L) | **DELETE with the siblings** | Correct, well-reasoned shared body — but it exists *only* to serve the two wrappers. `run_ratio_book` 85 LOC (M9), docstring 78 lines (M8). Not independently deletable. |
| `ark_pairs_classic_dsl.py` (523 L) | **KEEP or delete as a unit** | The only one with independent merit (frozen fit, ADF/HL screen, fresh-breach entry, post-stop lockout, honest beta caveat). OLS verified correct. But: lazy private import (M7), 134-line docstring + 9 driftable lines (M8), 3 dangling doc refs (N1). Leave it in — deleting it also deletes the screen script. |
| `scripts/ark_pairs_screen.py` (121 L) | **KEEP** | Genuinely read-only, 0 driftable numbers, one implementation of the screen. Dies if `ark_pairs_classic_dsl` goes (N3). |

**Is any "correct but mislabeled"?** `ark_pairs_classic_dsl.py` is correctly labelled and
correctly implemented — it is *worth keeping*. `ark_ratio_mom_dsl.py` is the one that reads as a
strategy but is really a falsification test; that is not a reason to keep it, it is the reason
the negative result belongs in the docs, not in `src/bt/strategies/`.

**Deletion safety verified:** `_discover()` (`src/bt/strategies/__init__.py:39-56`) globs
`*.py` and imports **every** module; all three `STRATEGY_TYPE`s load cleanly today
(`ark_pairs_classic_dsl`, `ark_ratio_mom_dsl`, `ark_ratio_mr_dsl` of 25 registered).
`ark_ratio_book.py` is correctly *not* registered (no `STRATEGY_TYPE`/`on_candle`). No test
references `ark*` (`grep -rl 'ark_ratio|ark_pairs' tests/` → empty), so deletion breaks nothing
in `make check` — **but** 17 `strats/wip/ark_*.json` configs name those strategy types and must
go in the same commit.

---

## 8. Part E — repo hygiene

`git status` = **24 untracked paths**: the 2 CSVs, 5 code files, 17 `strats/wip/ark_*.json`,
`docs/research/ark_edge_trader.md`. (`research/` is now an empty dir.)

- No reference to `etf_holding`, `etf_trade`, `data_0004`, `data_0005`, `ark ingest` survives in
  `*.py/*.md/*.json/*.toml` — **the revert is clean at source level.**
- **But** the stale bytecode for the dropped migrations remains (N2), including two distinct
  `data_0004` revisions. Not functionally dangerous; do not leave it if the goal is "no trace".
- The 2 CSVs are 6.5 MB of external, un-reproducible data with no committed builder (N6) and no
  `.gitignore` entry — decide explicitly whether they are committed or ignored; do not leave
  them as permanently-untracked.
- `docs/research/` is untracked, so **the report itself is not in version control** while the
  four modules that cite its siblings *are* referenced. Commit or discard, don't straddle.

---

## 9. What I could not verify

1. That the daggerok mirror is a faithful copy of `ark-funds.com` (no network).
2. Inception dates for `ARKD/ARKE/ARKI/ARKT/ARKY` — the quarter-boundary starts look like a page
   truncation but are equally consistent with genuine new listings (N5).
3. The `52,714` line-item count to the row (best reconstruction 52,493; the candle table changed
   during the audit) (B3).
4. `+20.6bps gross / ~10bps after the first hour` (`ark_edge_trader.md:59`) — no code, no
   construction, cannot reproduce.
5. `90.2% of line items on >75% one-sided fund-days` (`:43`) — plausible mechanism, but I did not
   reconstruct the exact 75% rule; treated as **UNVERIFIED**, not failed.
6. Whether `make check` fully passes — I ran collection only, because `format` writes files.

---

## 10. Minimum set before this ships

1. Restore `research/ark_edge/*.py` to version control, **or** stamp every figure in
   `ark_edge_trader.md` `UNVERIFIED` (B1).
2. State the universe on every era/frequency claim, and report the all-14 number alongside the
   5-candle number (B2).
3. Fix or drop `0.3%` (→ 1.10%) and pin `52,714` to a stated snapshot (B3).
4. Rename the `PASS` column — it contradicts the headline (B4).
5. Repeat the pooled reconciliation honestly: **two** sources, not three; `86.4/8.6` is one pair
   (M1).
6. Report the offset as a cathiesark **labelling** offset at `90.2%/20.0%/13.7%`, not `30.7%`
   (M2).
7. Drop or filter the `shares=0` row and the float-artifact rows (M3).
8. Relabel (iii) a null result at n=28, or drop the execution claim (M3/§6-iii).
9. Decompose `run_ratio_book` (M9); trim docstrings and delete lazy import (M7/M8).

---

**VERDICT: DO NOT SHIP.** The price-vs-NAV decomposition is real and robust, the NAV file is
arithmetically impeccable, and the report's source table is exact. Against that: the code behind
every headline number was deleted before the report was written, the central era conclusion
reverses when the undisclosed 5-fund universe is widened to the published 14, one headline
percentage is ~4x wrong, and the verdict table says PASS six times under a headline that says
nothing is tradeable. Fix §10 items 1–5 and this becomes a solid **SHIP WITH FIXES**.
---

## Second pass

*Written after the deletions recorded in the ERRATA block. DB opened read-only
(`file:...?mode=ro`); no repository file was modified except this one. Every number below was
recomputed with an implementation written from scratch (`/tmp/r2/engine.py`), not by importing
`g_common.py` and not by reading `out_g*.txt`. Where the two agree to the digit I say so; where
they do not, the disagreement is the finding.*

### 11. Job 1 — verification of the surviving candidate (b)

**First, a scope correction to the brief.** The brief places the L-S in a `2020-10..2021-10`
window. That is the *flow-event* window of the earlier candidates, not this one.
`assets/ark_flow.csv` spans **2020-08-21..2026-05-15**, candles run to **2026-11-19**, and the
cohort dates run **2020-09-02..2026-04-29**. The sample is 5.5 years, not 1. That matters for
every claim attached to "68 dates".

| claim | independently recomputed | verdict |
|---|---|---|
| ARK within-fund L-S `+381 bp/cohort, t=3.47, 68 distinct dates` | `+381`, sd 905, t 3.47, win 0.68, **329 fund-cohorts**, 68 dates | **PASS** |
| `+340` was a grid artefact of 61 non-session rows | raw panel 1800 rows vs **1739** sessions = exactly **61**; stepping by 21 through the 1739-session calendar gives 381 | **PASS** — mechanism confirmed, not just asserted |
| non-ARK control, sleeve-size matched, 25 draws: `+135` | `+135`, sd 909, t 1.23; 5 independent draw-panels range 105–147 bp | **PASS** |
| difference `+246 bp, t(diff)=1.87` | `+246`, sd 1085, t 1.87, corr 0.28. SE **132**; 95 % CI **[-17, +509]** | **PASS** (arithmetic) — see the CI note below |
| wide cross-section control `+339` → `diff +42` | `+339` (t 0.97) on the 68 dates; `+42` (t 0.12) | **PASS** (arithmetic) — **interpretation FAIL**, §12 |
| clip ±15 %: `diff +164, t 2.48` | 216 / 52 / **164** / 2.48 | **PASS** |
| hold 10 d: `+94` over 142 dates | 176 / 83 / **94** / t 1.55, 142 dates | **PASS** |
| `ex-NVDA/PLTR +257 (t 2.43)` | 257, t 2.43 | **PASS** |
| `ex-NVDA/PLTR AND clipped = +132` (below the 150 bar) | 132, t 2.22 | **PASS** |
| `ex-top-5 +185 (t 1.85)` | 185, t 1.85 | **PASS** |
| `8 of 12 grid cells below 150 bp, 12/12 positive` | all 12 cells reproduce to the digit; 8 below 150; 3 with t<2 | **PASS** |
| `per-day 15.9 / 17.6 / 18.1 bp at hold 5/10/21` | 15.9 / 17.6 / 18.1 | **PASS** |
| no non-session dates used | 68 cohort dates: **0 weekends, 0 absent from the SPY session set** | **PASS** |

22 of 22 figures reproduce. There is no rounding, no cherry-picking and no arithmetic error in
what the quant agent reported. **The numbers are honest. The inferences are where the problems
are.**

#### 11a. Lookahead audit of the L-S construction

- **Entry/exit: clean.** Signal at `C[j]`, entry at `O[j+1]`, exit at `C[j+hold]`
  (`g_common.py:100-113`). Bars are hourly and start-labelled (`09:30…15:00`); the 15:00 bar's
  close is the official close — return correlation **0.9996** against `ark_nav.csv:market_price`,
  median price ratio 1.000000. No 15:00-print contamination, no same-bar entry.
- **Universe selection: one real lookahead, non-material.** `g_run.py:52` tests
  `mem["min"] <= d <= mem["max"]`, where `max` is the ticker's *last* trade in the whole file.
  At date `d` that asks "will this name still be traded after `d`?" — future information. I
  replaced it with a past-only rule (traded at least once in the trailing 126 sessions):
  **+322 bp, t 2.80, 73 dates**. The result does not depend on the defect. Still a defect.
- **Survivorship: present and unquantified.** 275 of the 435 tickers in `ark_flow.csv` have no
  candles at all (mostly JP/HK listings, but also `ABSI`, `ACCD`, `ABRA`); of the 160 that do,
  **not one has a last bar before 2026-06** — i.e. there is not a single delisted ARK holding in
  the sample. The cross-section is "ARK names that still trade in 2026". The bias direction is
  ambiguous (a survivor filter removes the short leg's best winners and the long leg's blow-ups),
  but it is a structural filter and nobody stated it.
- **Membership is a *trade* list, not a *holding* list.** `min`/`max` are first/last *trade*
  dates, so a name held but not traded is ineligible. AMD is eligible only from **2024-07-31**
  although ARKK held it throughout. The cross-section is therefore a churn subset, and its
  character changes between the dense flow file (2020-21: median 5-6 trades/fund-day) and the
  sparse one (2022+: 3). Cohort composition is not stable over the 68 dates.
- **Fund double-counting inside a date.** A name held by 5 funds enters the date composite 5×.
  Median per-date maximum single-name weight **0.50**; maximum **0.83** (NVDA, on the 2023
  dates, long in 4-5 funds at once). The t-stat over 68 non-overlapping dates is still valid,
  but "68 independent cross-sections" overstates diversification: on a typical date half the
  P&L is one stock.

### 12. Is the wide cross-section control decisive? No — and the "generic momentum" reading is unsupported

The arithmetic is right. The reading is not.

- What the wide control actually is: k=3 of ~451 names, **median cohort +15 bp**, sd 2793 bp,
  win 0.50, and its **five best cohorts equal 111 % of its total sum** (drop them and the
  strategy loses money). Most frequent long names: MARA (15 cohorts), CVNA (14), RIOT (13),
  CELH (13), MSTR (12). It is not "generic momentum"; it is five crypto/meme/small-cap lottery
  cohorts wearing a momentum hat.
- Its own standard error is 339 bp/cohort. The 95 % interval on its mean is roughly
  **[-325, +1003]**. A control whose mean cannot be distinguished from either 0 or 381 does not
  establish that the ARK number is generic. `t(diff)=0.12` is a **failure to reject**, not
  evidence of equivalence — and with this sd it could never have rejected anything.
- The two controls vary *selection intensity*, which is the quantity that sets the momentum
  premium: the size-matched control holds it fixed (n≈22) and gives **+246**; the wide control
  maximises it (3 of 451) and gives **+42**. They bracket the answer instead of agreeing on it.
  The only supportable statement is the one the agent wrote in his own caveat — "the wide
  control leaves +42 bp of ARK-specific return" — which is *unresolved*, not decisive.
- The distribution shape argues against equivalence even where the means coincide: ARK win 0.68
  / sd 905 versus wide win 0.50 / sd 2793. Different phenomenon, similar mean.
- Symmetrically, the wide control does **not** show the ARK edge is special: a 3-of-451
  selection earns the same mean, so the magnitude is not extraordinary. That is all it shows.

#### 12a. Verdict on the `SURVIVES` label

The pre-registered rule is (i) ARK − control ≥ 150 bp **and** (ii) ex-NVDA/PLTR ≥ 150 bp with
t ≥ 2. Both conditions pass, so on its own terms the rule is satisfied and the label is
*mechanically* earned. What the label hides:

- **The bar is a cost line, not a significance threshold.** 150 bp/cohort ≈ 6 names × 20 bp
  round trip + 30 bp of margin. Clearing it means "not obviously unprofitable", not "real".
- **The deciding quantity has a CI containing zero.** +246 ± 1.96×132 = **[-17, +509]**.
- **The margin is 0.73 of one standard error** (96 bp over the bar, SE 132).
- **The rule can be passed by noise at a non-negligible rate**: under a true difference of
  zero, a single draw clears 150 bp with probability 13 % on leg (i) and 8 % on leg (ii). The
  conjunction is much harder to fake than either leg alone — this is a real strength of the
  rule and I do not want to overstate the criticism — but neither leg is a 5 % test.
- **The 12-cell grid is not a difference table.** 8 of the 12 cells are *gross* ARK numbers;
  the 150 bp bar was defined on a *difference*. Reporting "8 of 12 below 150" mixes the two.
  Only 3 of the 12 cells actually have a control arm (164 / 94 / 109 bp).
- **"Drift, not jump" is overstated.** The last-day statistic is 103 bp of the 381 — 27 % of
  the return in 1/21 of the holding period, 5.7× the mean daily 17.3 bp; the last five days
  carry 27 %. The per-day profile is not monotone: offset 21 = +103, offset 20 = −14,
  offset 19 = −26, offset 15 = +73. The *average* per-day return is horizon-flat (15.9/17.6/
  18.1), which is what the table measures, but within the 21-day window the return is lumpy and
  weighted to the exit.

**My label: FRAGILE.** Not DEAD — it reproduces, it clears costs, and it survives ex-NVDA/PLTR,
clipping, the hold grid and a past-only membership rule. Not SURVIVES — there is no
statistically defensible ARK-specific premium (difference CI contains zero; the wide control
leaves +42 ± 339), the combination of the two known weaknesses lands at +132 bp *below* the
bar, and the cross-section is half one stock on a median date. The pre-registered gate was met
on too thin a margin to carry the word "survives" into a report. The one test that would settle
it is the liquidity/size-matched twin control already named in the Gate-4 note, and it has not
been run.

### 13. Job 2 — audit of the analysis scripts behind the headline numbers

Scope: `/tmp/ark_edge/*.py`, 25 files, ~1,900 lines. These are scratch scripts, so AGENTS.md
conventions do not apply; the only standards that matter are lookahead, date alignment,
overlap in t-stats, survivorship, double counting, and whether a verdict rests on a bug.
**No fabricated numbers found**: every headline figure I spot-recomputed matches the script
output. The defects below are all in *estimators*, not in transcription.

#### High — can flip a verdict or invalidate a headline number

| # | file:line | defect | blast radius |
|---|---|---|---|
| H1 | `panel.py:45-56` (`flows()`) | "Source precedence" keeps **one source per (fund, date)** and drops the rest: **38,688 of 96,898 rows survive (40 %)**. In 2020-21 the median fund-date keeps **5-6 rows of 35-50 available** (e.g. `ARKF 2020-10-20`: keeps 3 cathiesark rows, discards 43 arktrack rows). 4,136 of 4,804 fund-dates resolve to `cathiesark`. | Every flow *magnitude* — `gross`, `size`, `press`, `net21` — is a ~12 % subsample whose composition tracks source coverage, so a2/a3's "flow size → vol" and b2/b5's flow L-S partly measure *which source covered the date*. Not a lookahead; a measurement blocker. (My first audit flagged the precedence *rule* under M1 — the consequence is worse than I recorded.) |
| H2 | `d_events.py:19` vs `:38` | Benchmark return is `close(T)→close(T+h)`; the event return is `open(T+1)→close(T+h)`. | One-overnight misalignment in **every** excess return `x5/x21/x63`: `x = (intraday path) − (overnight gap + intraday path)`. All of D1/D2 inherits it. |
| H3 | `g_prem_gap.py:17, 29-39, 50-60`; `e2_discount_gap.py:13, 27, 37-46` | `gap[t] = O[t]/C[t-1]` is aligned at `t`, and is regressed on a premium/`pz` measured at **`t`'s close** — the premium is observed *after* the gap it supposedly predicts. The agent's self-reported trap; it is live in G1, G2 and, worse, in G4's tradable rule (`pnl = sign(pz)*gap`, same day). `qc2_prem_sync.py` and `qc3_pz.py` fix it (`gap.shift(-1)`), so the fix exists — but the two scripts that generate the headline coefficients are the broken ones. `e2:` "corr(premium close T, next gap)" is actually same-day. |
| H4 | `b3_robust.py:14, 34-35`; `b4_book.py:22, 44-52` | Anchors every **5** sessions with `hold=21` → five-fold overlapping observations, and the t uses `std/√n` over distinct dates. `b4_book.py:44` prints "**non-overlapping cohorts per fund**" over a `spacing=5, hold=21` default: the header is false. | t inflated ≈ √(21/5) ≈ **2.0** in B6/B7/B8 and B9/B10/B11 — i.e. the candidate-(b) v1 numbers. A t of 2.8 is a t of 1.4. |
| H5 | `e_premium_vol.py:41-51`; `a3_vol_resid.py:29, 39` | OLS with the textbook iid SE on **overlapping** forward windows (5-day and 22-day) built from daily data. | t overstated ≈ √5 ≈ 2.2 (a3, daily) and ≈ √22 ≈ 4.7 (e2/e3 forward vol). The HAR+size and HAR+premium `t` columns are not usable as printed. |
| H6 | `d_events.py:89` | Same naive SE, on 21-day forward event windows, with many events sharing a date (one ARK decision → dozens of rows). | The `press_coef t` in D3 is overstated by an order of magnitude; there is no date clustering anywhere in the file. |

#### Medium — biases a secondary number

| # | file:line | defect |
|---|---|---|
| M1′ | `a2_flow_attention.py:23, 26` | `rolling(5).mean().std().shift(-4)` puts the **signal day inside the "forward" window** (1 of 5 days). `a_flow_vol.py:29-33` uses `shift(-5)` and is correct; a2 regressed on the way to the doc. |
| M2′ | `g_run.py:52` | Universe eligibility uses the file's last trade date (future info). Quantified non-material: past-only membership gives +322 bp, t 2.80 (§11a). |
| M3′ | `b6_control.py:23, 44` | `(fw/p0-1).mean()` averages the cumulative path ≈ half the hold return (self-reported, never used). Also draws **80** names when ARK's sleeve median is 22 (comment claims a size match) and draws **once** (`random.Random(0)`) rather than 25×. Two independent reasons its control understates generic momentum. |
| M4′ | `c_etf_basis.py:33` | `ar1` regresses Δspread on lagged spread, so it estimates (ρ−1), not ρ — mislabeled "oos_ar1". Also no minimum-observation guard on the formation beta while `ARKX/PRNT/IZRL` start in 2021. |
| M5′ | `c_etf_basis.py:46-48, 52-67` | 28 pairs × daily 5-day overlapping forward returns, pooled with no clustering and no t-stat; per-pair `n/mean_bp/win` are not independent observations. |
| M6′ | control universe, `g_run.py:20-21` | The 451 "non-ARK" names include every ETF in `candle` except the 8 ARK ones (b6 excluded SPY/QQQ/TLT/IWM; the g-series excludes none), so a k=3 sleeve can be three ETFs — a lower-vol control that flatters the ARK difference. Not quantified. |

#### Low — dead code, no numeric effect, but it hides errors

`a_flow_vol.py:19` (a no-op expression that only prints a fraction); `b5_gates.py:43`
(`perYear_n` is just `len(comp)`), `:61` ("worst case" double-counts cost); `f_overnight.py:47`
(`np.polyfit(...) if False else 0`); `e_premium_vol.py:57` (dead branch in the percentile);
`g_prem_gap.py:45` (placeholder regression assigned and discarded); `b_dispersion.py:44`
(`intra` duplicates `ret`); `b4_book.py:13` (`r_co` unused); `d_events.py:6` (`K=100` volume
fudge — irrelevant to deciles, only to the coefficient's units). Also: `panel.daily_bars` and
`g_common._daily_chunk` drop `nbar<5`, which silently removes all **13** four-bar early closes
per year from every panel.

#### Structural

No script consults a real exchange calendar. `g_common` infers sessions from "≥8 ARK
constituents printed a bar", the older scripts use the union panel index (which is where the 61
non-session rows come from). The heuristic calendar works here — it contains 16 sessions before
ARKK's own data starts and the incomplete final day, neither of which lands on a cohort anchor —
but it is the root cause of the +340/+381 discrepancy and it is one data refresh away from
producing a cohort on a holiday.

**Count: 6 high, 6 medium, 9 low + 1 structural, in 25 files.** The agent's two self-reports
(the `b6` cumulative-path mean, the post-gap premium) are both confirmed — and both belong to a
family with four more members: `t`-statistics computed without any allowance for overlap,
clustering or date-sharing (H4, H5, H6, M5′), and window alignment errors that leave one leg
one day off the other (H2, M1′).

### 14. What I could not verify

1. Whether the kept `cathiesark` rows are a strict *subset* of the dropped `arktrack` rows on
   the same day (H1's magnitude), or a different snapshot. Needs the upstream archives; no
   fetch script exists in the repo (N6).
2. That the DB's 09:30 bar open equals the official opening print — no independent open series.
   (The *close* side is verified: 0.9996 against `market_price`.)
3. Borrow and capacity on the short leg — 3 names out of a 22-name sleeve, typically the
   smallest holdings. The L-S is not fundable at size and nothing in the repo can test that.
4. The liquidity/size-matched twin control. It remains the single test that would separate
   "ARK disperses more than comparable names" from "the sleeve sits in a momentum-friendly size
   bucket", and it has still not been run.
