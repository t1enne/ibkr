# ARK §6 control gate — decisive result

**VERDICT: KILL.**

The within-fund ARK long/short momentum increment does not survive a
size-and-price matched twin control. On the long era (50 non-overlapping
cohorts) ARK − matched twins is **+26bp of annual return and +17bp per cohort
(t = 0.05)** — an order of magnitude under the 150bp bar — and ARK is *worse*
than the wide non-ARK control by 1694bp/yr (−441bp/cohort, t = −1.42). ARK's own
level (+63bp/cohort, t = 0.26) is indistinguishable from zero and nowhere near
§6's claimed +381bp/cohort.

Read this plainly: §6's "ARK-specific" increment was an artifact of comparing an
ARK-heavy basket to a control that was not matched on size or price. Match it,
and what is left is noise around zero, with the wrong sign against the broad
stock panel.

---

## 1. Claim tested

§6 of `docs/research/ark_conclusion.md` claims an ARK-specific increment:
ARK constituents ranked by 126-day return, long top-3 / short bottom-3,
21-day non-overlapping holds, beat a non-ARK control by **+246bp/cohort**
(t = 1.87, CI [−17, +509]) and that control was never properly run — the
reviewer's words. The pre-registered bar for "the increment is real" is
**150bp/cohort of difference with t ≥ 2**.

This gate builds the control §6 says was never built: 1:1 twins matched on
trailing-60d median dollar-volume decile **and** price decile at era start, plus
the wide non-ARK panel as a reference. Three universes, two eras, one strategy
config — nothing but `config.symbols` changes.

## 2. Strategy, params, cost model (identical in all six gate runs)

| item | value |
|---|---|
| strategy | `src/bt/strategies/ark_ls_momentum_dsl.py` — **unmodified** |
| `lookback_bars` | 126 |
| `rebalance_bars` | 21 |
| `k` | 3 (3 longs, 3 shorts) |
| `size` | 0.1 of initial capital per position → 60% gross |
| `bars` | `["1d"]` (resampled from native 1h) |
| `warmup` | `300d` ≈ 214 trading bars > the 126-bar lookback |
| `commission` | 0.05 per trade |
| `initial_capital` | 100,000 |
| eras | 2021-01-01 → 2021-12-31 and 2022-01-01 → 2026-04-30 |

Cost model actually charged by the engine: 5bp spread + 2bp slippage per side
(≈14bp round trip) plus the flat 0.05 commission — i.e. the task's "≈20bp
round-trip" class of assumption. See §8: **borrow is not modelled at all** and
that is not a footnote here.

Execution convention (matters for every arm equally): the DSL emits
`fill_at_next_open=False`, so a rebalance fills at the signal bar's close. A
126-day momentum book filled at the close it just measured is optimistic by up
to one bar of drift; it flatters all three arms, so the *difference* is the least
contaminated quantity, but the absolute numbers are slightly generous.

## 3. Universe construction

Data: `assets/ark_flow.csv` (96,898 rows, 2020-08-21 → 2026-05-15, 435 distinct
tickers, 8 funds) for point-in-time membership, `../data/ibkr.db` (659 tickers,
5.57M 1h candles → daily bars aggregated in SQL) for prices. Scripts:
`scripts/ark_gate_universes.py` (universes), `scripts/ark_gate_configs.py`
(eras), `scripts/ark_gate_cohorts.py` (cohort series). Every count below is from
`universes/ark_gate_summary.json`.

**Screens, applied identically per era:**

1. *Membership (PIT).* A ticker is ARK-active in an era when its own flow window
   `[first_flow, last_flow]` intersects the era window. The stricter
   "member at era start only" reading was measured and rejected as too
   restrictive: it leaves **111** names (2021) and
   **110** (2022-2026) because it throws away every
   name ARK added later — including most of the 2022-2026 book. Documented
   consequence: the panel is static per era, so a name ARK bought mid-era is
   rankable from era start. That is a membership-timing lookahead, not a
   price/return lookahead, and it applies to the ARK sleeve only. It cannot
   manufacture the result we got (ARK loses), but it is real.
2. *Ticker-reuse gate.* A name is dropped when >50% of its own flow dates fall
   outside its candle span. Excluded (**7** names — the trap the brief
   named, `SI` included):

   | ticker | ARK flow span | DB candle span | flow dates inside span |
   |---|---|---|---|
   | BIDU | 2020-08-31 → 2025-12-08 | 2026-10-01 → 2026-11-19 | 0/212 |
   | XONE | 2020-09-25 → 2021-09-30 | 2022-09-16 → 2026-11-20 | 0/148 |
   | SI | 2020-10-20 → 2023-01-23 | 2025-07-31 → 2026-11-20 | 0/186 |
   | VOE | 2021-03-05 → 2021-09-30 | 2026-07-27 → 2026-11-19 | 0/32 |
   | TEVA | 2021-03-10 → 2021-09-30 | 2026-09-14 → 2026-11-19 | 0/24 |
   | DIS | 2021-05-14 → 2022-06-24 | 2026-07-27 → 2026-11-19 | 0/93 |
   | ETSY | 2021-06-02 → 2022-06-24 | 2026-08-20 → 2026-11-19 | 0/105 |

   Additionally excluded on the same >50% rule with partial overlap:
   `DSY` (8/199 inside), `KSPI` (8/162 inside).
   The 8 ARK funds themselves (`ARKK/ARKG/ARKQ/ARKW/ARKF/ARKX/PRNT/IZRL`) are
   never members, and no ETF/trust/ETN/note vehicle is allowed in either control
   (`189` DB tickers matched a vehicle-name token; a
   first pass with a narrower token set let `CPER`, `UNG`, `USO`, `VXX` and the
   `XLB..XLY` sector SPDRs into the control pool — those runs were discarded and
   are not reported).
3. *Tradability.* ≥100 daily bars inside the era, ≥60 bars of history before the
   era start (needed for the match), and no daily-bar gap > 4 days anywhere in
   `[era start − 300d warmup, era end]` — the criterion `src.bt.data_feed`'s own
   integrity guard enforces at load time (it aborts the whole run otherwise).
   Two consequences worth naming: it drops `OPEN`/`BNR`/`OKLO` (holes of 5–7
   days) from the ARK arm, and it is the reason the control pool is US-only —
   the DB's 44–46 Milan/Mexico listings have 5–35 day holiday/illiquidity holes
   and cannot be loaded at all. Venue policy is therefore `market='SMART'`,
   `currency='USD'`, which also keeps the pool like-for-like with the ARK sleeve
   (which is 100% US-listed).
4. *1:1 twin match.* Deciles are cut on the control pool's own distribution of
   (median trailing-60d dollar volume, price) at era start — no lookahead, all
   inputs are bars strictly before the era start. Inside each of the 100 cells,
   ARK names and pool names are paired alphabetically, `min(counts)` per cell.
   An ARK name whose cell has no free pool name is **dropped from both sleeves**
   (the brief's rule; it keeps the two sleeves the same size and the same
   selection intensity).

**Resulting sizes:**

| era | ARK flow-active in era | ARK after screens | ARK matched sleeve | ARK dropped (no twin) | twins | wide control |
|---|---|---|---|---|---|---|
| 2021 | 174 | 158 | **57** | 101 | **57** | **100** |
| 2022-2026 | 126 | 119 | **54** | 65 | **54** | **106** |

Two caveats that decide how much weight this result carries. Neither one
rescues the ARK case, and both are quantified below:

* **The matched ARK sleeve is a mid/small-cap ARK subset.** The dropped names
  are the ones whose dollar-volume decile has no non-ARK neighbour in the DB —
  in 2021 that is 101 of 158 names
  including `TSLA`, `NVDA`, `MSFT`, `AMZN`, `GOOGL`, `PLTR`, `NFLX`, `SHOP`,
  `ROKU`, `SPOT`, `PYPL`, `PINS`, `SNAP`, `TWLO`, `ZM` (full list in
  `ark_gate_summary.json → eras.<era>.ark_unmatched_names`). That is a genuine
  composition shift away from megacaps.
  **This is why the full screened ARK sleeve is also run** (§6, two
  supplementary runs, identical config): it contains all the megacaps. It does
  not rescue the case — 119 ARK names give +138bp/cohort in
  2022-2026 (t = 0.56) and −714bp/cohort in 2021.
* **The control pool is thin: 100 / 106 stocks.** The DB
  holds 659 tickers, of which 189 are fund/trust/ETF
  vehicles and 435 are ARK-flow tickers; the non-ARK US stock residue is all
  that is left. §6's wide control was run against 451 names — we cannot
  reproduce a pool that size from this database, so our wide control is a
  100–106 name panel.

## 4. The six gate runs — full, unfiltered output

`Scaled` and `Rejected` are **0 and 0 in all six gate runs** (and in both
supplementary runs) — no run is advisory on those grounds. One `bt split` fold
is advisory; see §7.

### 4.1 Compact metric rows (from the reports below, nothing recomputed)

### 2021

| universe | symbols | Sharpe | Ann | MaxDD | Kurt | Skew | Win | Trades | Scaled | Rejected |
|---|---|---|---|---|---|---|---|---|---|---|
| ARK sleeve (matched), era 2021 | 57 | -0.06 | -2.77% | -21.38% | 0.62 | +0.07 | 34.48% | 29 | 0 | 0 |
| matched non-ARK twins, era 2021 | 57 | -0.18 | -13.44% | -40.65% | 4.04 | +0.55 | 33.33% | 36 | 0 | 0 |
| wide non-ARK/non-ETF control, era 2021 | 100 | +0.41 | +8.37% | -37.32% | 3.61 | +0.54 | 45.71% | 35 | 0 | 0 |
| SUPPLEMENTARY: full screened ARK sleeve, era 2021 | 158 | -1.08 | -24.67% | -32.24% | 1.25 | -0.37 | 45.95% | 37 | 0 | 0 |

### 2022-2026

| universe | symbols | Sharpe | Ann | MaxDD | Kurt | Skew | Win | Trades | Scaled | Rejected |
|---|---|---|---|---|---|---|---|---|---|---|
| ARK sleeve (matched), era 2022-2026 | 54 | +0.16 | +0.98% | -37.95% | 13.64 | +0.67 | 50.37% | 135 | 0 | 0 |
| matched non-ARK twins, era 2022-2026 | 54 | +0.13 | +0.72% | -29.84% | 2.21 | -0.40 | 45.38% | 130 | 0 | 0 |
| wide non-ARK/non-ETF control, era 2022-2026 | 106 | +1.06 | +17.92% | -16.61% | 2.93 | +0.01 | 53.03% | 132 | 0 | 0 |
| SUPPLEMENTARY: full screened ARK sleeve, era 2022-2026 | 119 | +0.18 | +1.53% | -29.65% | 10.73 | -0.69 | 43.71% | 151 | 0 | 0 |

### 4.3 — ARK sleeve (matched), era 2021

`uv run ibkr bt run strats/wip/ark_gate_ark_2021.json`

```

Backtest Results
================================================================================

Worst Drawdown Periods
Net DD %   Peak Date  Valley Date  Recovery  Days
-------------------------------------------------
  21.38%  2021-02-17   2021-07-16         —   316

Period Drawdowns (peak-to-trough within period)
Period  Max DD              Peak            Valley
--------------------------------------------------
Month   13.03%  2021-02-09 00:00  2021-02-26 00:00

Trades
  29 trades — list omitted (pass --trades to print)

Trading Statistics
Metric                      Value
---------------------------------
Starting Capital       100,000.00
Total Trades                   29
Closed Trades                  29
Win Rate                   34.48%
Total P&L (Net)         -2,752.17
Gross P&L               -2,483.82
Commission Costs             2.90
Spread+Slippage Costs      265.45
Net P&L                 -2,752.17
Capital Utilization         64.2%

Return Drivers (per Symbol)
Symbol        P&L  Trades  Win Rate  % of Total
-----------------------------------------------
STNE     4,921.17       1      100%     -178.8%
CGEN     4,581.49       1      100%     -166.5%
HUYA     3,675.03       1      100%     -133.5%
SRPT     2,308.46       1      100%      -83.9%
HUBS     1,732.31       3       67%      -62.9%
TREE     1,699.51       1      100%      -61.8%
NET        362.38       1      100%      -13.2%
BEKE       144.18       1      100%       -5.2%
HDB       -227.60       1        0%        8.3%
CAMT      -412.04       2        0%       15.0%
U         -454.94       1        0%       16.5%
NNDM      -607.93       2        0%       22.1%
ALLT      -903.89       1        0%       32.8%
EVGN    -1,178.68       2        0%       42.8%
AUDC    -1,316.10       1        0%       47.8%
IONS    -1,372.81       1        0%       49.9%
GBTC    -1,998.07       2       50%       72.6%
ACIC    -3,296.85       2        0%      119.8%
NTLA    -3,544.10       2        0%      128.8%
DDD     -6,863.68       2        0%      249.4%

Data: 2021-01-01 → 2021-12-30
Duration: 252 periods · 11.9 months

Performance Metrics
Metric             Strategy     SPY
-----------------------------------
Total Return         -2.75%  29.08%
Annual Return        -2.77%  29.57%
Annual Volatility    18.53%  13.06%
Sharpe Ratio          -0.06    2.26
Max Drawdown        -21.38%  -5.41%
                                   
Calmar Ratio          -0.13       —
Sortino Ratio         -0.24       —
Omega Ratio            0.99       —
Stability              0.13       —
Skewness               0.07       —
Kurtosis               0.62       —
Scaled Trades             0       —
Rejected Trades           0       —
Alpha                     —   -0.09
Beta                      —    0.30
```
### 4.4 — matched non-ARK twins, era 2021

`uv run ibkr bt run strats/wip/ark_gate_twins_2021.json`

```

Backtest Results
================================================================================

Worst Drawdown Periods
Net DD %   Peak Date  Valley Date    Recovery  Days
---------------------------------------------------
  40.65%  2021-04-06   2021-12-20           —   268
  17.03%  2021-02-19   2021-03-08  2021-04-05    45
   9.71%  2021-01-19   2021-01-27  2021-02-08    20
   7.18%  2021-02-09   2021-02-10  2021-02-17     8

Period Drawdowns (peak-to-trough within period)
Period  Max DD              Peak            Valley
--------------------------------------------------
Month   19.65%  2021-04-05 00:00  2021-04-22 00:00

Trades
  36 trades — list omitted (pass --trades to print)

Trading Statistics
Metric                      Value
---------------------------------
Starting Capital       100,000.00
Total Trades                   36
Closed Trades                  36
Win Rate                   33.33%
Total P&L (Net)        -13,358.73
Gross P&L              -13,004.24
Commission Costs             3.60
Spread+Slippage Costs      350.89
Net P&L                -13,358.73
Capital Utilization         67.0%

Return Drivers (per Symbol)
Symbol        P&L  Trades  Win Rate  % of Total
-----------------------------------------------
MARA     4,539.99       3       33%      -34.0%
MRNA     3,697.83       1      100%      -27.7%
AXSM     3,186.01       1      100%      -23.8%
MSTR     1,439.76       1      100%      -10.8%
JAZZ       975.78       2      100%       -7.3%
BMRN       914.05       1      100%       -6.8%
SNY        294.20       1      100%       -2.2%
INTC        31.58       2       50%       -0.2%
IDXX      -249.20       1        0%        1.9%
ALKS      -372.03       1        0%        2.8%
MRK       -420.28       1        0%        3.1%
GILD      -713.00       1        0%        5.3%
ROP       -948.81       1        0%        7.1%
ASML    -1,022.10       1        0%        7.7%
MPWR    -1,208.49       1        0%        9.0%
NXE     -1,352.49       1        0%       10.1%
FANG    -1,534.56       1        0%       11.5%
CELH    -1,715.81       2       50%       12.8%
LTBR    -1,930.42       1        0%       14.5%
EXEL    -1,990.67       1        0%       14.9%
LULU    -2,319.15       2        0%       17.4%
BYDDF   -2,361.59       1        0%       17.7%
ENPH    -2,401.26       1        0%       18.0%
URGN    -2,483.73       4       50%       18.6%
FORM    -2,624.82       2        0%       19.6%
UEC     -2,789.53       1        0%       20.9%

Data: 2021-01-01 → 2021-12-30
Duration: 252 periods · 11.9 months

Performance Metrics
Metric             Strategy     SPY
-----------------------------------
Total Return        -13.36%  29.08%
Annual Return       -13.44%  29.57%
Annual Volatility    38.72%  13.06%
Sharpe Ratio          -0.18    2.26
Max Drawdown        -40.65%  -5.41%
                                   
Calmar Ratio          -0.33       —
Sortino Ratio         -0.53       —
Omega Ratio            0.97       —
Stability              0.54       —
Skewness               0.55       —
Kurtosis               4.04       —
Scaled Trades             0       —
Rejected Trades           0       —
Alpha                     —   -0.26
Beta                      —    0.86
```
### 4.5 — wide non-ARK/non-ETF control, era 2021

`uv run ibkr bt run strats/wip/ark_gate_wide_2021.json`

```

Backtest Results
================================================================================

Worst Drawdown Periods
Net DD %   Peak Date  Valley Date    Recovery  Days
---------------------------------------------------
  37.32%  2021-02-17   2021-05-13           —   316
  13.20%  2021-01-19   2021-01-27  2021-02-08    20

Period Drawdowns (peak-to-trough within period)
Period  Max DD              Peak            Valley
--------------------------------------------------
Month   23.58%  2021-04-05 00:00  2021-04-22 00:00

Trades
  35 trades — list omitted (pass --trades to print)

Trading Statistics
Metric                      Value
---------------------------------
Starting Capital       100,000.00
Total Trades                   35
Closed Trades                  35
Win Rate                   45.71%
Total P&L (Net)          8,314.61
Gross P&L                8,666.46
Commission Costs             3.50
Spread+Slippage Costs      348.34
Net P&L                  8,314.61
Capital Utilization         65.2%

Return Drivers (per Symbol)
Symbol        P&L  Trades  Win Rate  % of Total
-----------------------------------------------
RIOT     7,542.16       2       50%       90.7%
MARA     4,539.99       3       33%       54.6%
CELH     3,801.35       2      100%       45.7%
MRNA     3,697.83       1      100%       44.5%
AXSM     3,186.01       1      100%       38.3%
JAZZ       975.78       2      100%       11.7%
BMRN       914.05       1      100%       11.0%
AMBA       563.06       1      100%        6.8%
DXCM       557.88       1      100%        6.7%
ASML       533.10       1      100%        6.4%
XEL        269.48       1      100%        3.2%
SYNA      -160.65       1        0%       -1.9%
WDC       -344.86       1        0%       -4.1%
FTNT      -370.15       1        0%       -4.5%
ALKS      -372.03       1        0%       -4.5%
MRK       -420.28       1        0%       -5.1%
MSTR      -528.18       1        0%       -6.4%
URGN      -749.14       4       50%       -9.0%
GILD      -774.48       1        0%       -9.3%
BE        -963.85       1        0%      -11.6%
LULU      -999.89       1        0%      -12.0%
FORM    -1,120.26       1        0%      -13.5%
LTBR    -1,930.42       1        0%      -23.2%
FSLR    -2,413.97       1        0%      -29.0%
UEC     -3,281.81       1        0%      -39.5%
PLUG    -3,836.09       2       50%      -46.1%

Data: 2021-01-01 → 2021-12-30
Duration: 252 periods · 11.9 months

Performance Metrics
Metric             Strategy     SPY
-----------------------------------
Total Return          8.31%  29.08%
Annual Return         8.37%  29.57%
Annual Volatility    49.18%  13.06%
Sharpe Ratio           0.41    2.26
Max Drawdown        -37.32%  -5.41%
                                   
Calmar Ratio           0.22       —
Sortino Ratio          0.26       —
Omega Ratio            1.08       —
Stability              0.20       —
Skewness               0.54       —
Kurtosis               3.61       —
Scaled Trades             0       —
Rejected Trades           0       —
Alpha                     —   -0.08
Beta                      —    1.07
```
### 4.6 — SUPPLEMENTARY: full screened ARK sleeve, era 2021

`uv run ibkr bt run strats/wip/ark_gate_arkfull_2021.json`

```

Backtest Results
================================================================================

Worst Drawdown Periods
Net DD %   Peak Date  Valley Date  Recovery  Days
-------------------------------------------------
  32.24%  2021-01-26   2021-12-09         —   338

Period Drawdowns (peak-to-trough within period)
Period  Max DD              Peak            Valley
--------------------------------------------------
Month   13.52%  2021-02-02 00:00  2021-02-26 00:00

Trades
  37 trades — list omitted (pass --trades to print)

Trading Statistics
Metric                      Value
---------------------------------
Starting Capital       100,000.00
Total Trades                   37
Closed Trades                  37
Win Rate                   45.95%
Total P&L (Net)        -24,542.38
Gross P&L              -24,183.80
Commission Costs             3.70
Spread+Slippage Costs      354.88
Net P&L                -24,542.38
Capital Utilization         69.8%

Return Drivers (per Symbol)
Symbol         P&L  Trades  Win Rate  % of Total
------------------------------------------------
BILL      3,495.43       2       50%      -14.2%
PACB      3,387.14       1      100%      -13.8%
EVGN      3,348.02       1      100%      -13.6%
WKHS      3,189.86       2      100%      -13.0%
HUYA      3,075.83       1      100%      -12.5%
INMD      2,465.91       1      100%      -10.0%
SRPT      2,308.46       1      100%       -9.4%
PERI      2,206.88       1      100%       -9.0%
TEAM      1,692.25       1      100%       -6.9%
CLLS      1,026.97       1      100%       -4.2%
CGEN        614.82       1      100%       -2.5%
RDHL        320.62       1      100%       -1.3%
MTLS        306.76       1      100%       -1.2%
NNDM         81.17       1      100%       -0.3%
BEKE         49.50       1      100%       -0.2%
TSLA       -142.27       1        0%        0.6%
KMDA       -246.08       1        0%        1.0%
PYPD       -275.21       1        0%        1.1%
TREE       -381.41       1        0%        1.6%
STNE       -602.76       1        0%        2.5%
NVDA       -760.27       1        0%        3.1%
ARCT       -827.10       1        0%        3.4%
NTLA     -1,438.14       1        0%        5.9%
LSPD     -2,369.37       1        0%        9.7%
ACIC     -2,673.72       1        0%       10.9%
NET      -3,140.74       1        0%       12.8%
GBTC     -3,542.71       1        0%       14.4%
DDD      -3,887.52       1        0%       15.8%
IOVA     -4,301.63       1        0%       17.5%
VUZI     -7,580.95       2        0%       30.9%
MCRB     -8,367.01       2        0%       34.1%
API     -11,575.12       2       50%       47.2%

Data: 2021-01-01 → 2021-12-30
Duration: 252 periods · 11.9 months

Performance Metrics
Metric             Strategy     SPY
-----------------------------------
Total Return        -24.54%  29.08%
Annual Return       -24.67%  29.57%
Annual Volatility    23.56%  13.06%
Sharpe Ratio          -1.08    2.26
Max Drawdown        -32.24%  -5.41%
                                   
Calmar Ratio          -0.77       —
Sortino Ratio         -1.43       —
Omega Ratio            0.83       —
Stability              0.25       —
Skewness              -0.37       —
Kurtosis               1.25       —
Scaled Trades             0       —
Rejected Trades           0       —
Alpha                     —   -0.31
Beta                      —    0.42
```
### 4.7 — ARK sleeve (matched), era 2022-2026

`uv run ibkr bt run strats/wip/ark_gate_ark_2022_2026.json`

```

Backtest Results
================================================================================

Worst Drawdown Periods
Net DD %   Peak Date  Valley Date    Recovery  Days
---------------------------------------------------
  37.95%  2025-02-18   2026-02-12           —   435
  15.01%  2022-06-07   2022-07-21  2023-06-23   381
  13.97%  2024-06-21   2024-10-22  2024-11-20   152
   8.55%  2023-09-21   2023-12-28  2024-02-01   133
   8.28%  2024-04-12   2024-05-01  2024-05-22    40

Period Drawdowns (peak-to-trough within period)
Period  Max DD              Peak            Valley
--------------------------------------------------
Month   29.07%  2025-07-01 00:00  2025-07-21 00:00

Trades
  135 trades — list omitted (pass --trades to print)

Trading Statistics
Metric                      Value
---------------------------------
Starting Capital       100,000.00
Total Trades                  135
Closed Trades                 135
Win Rate                   50.37%
Total P&L (Net)          4,295.23
Gross P&L                5,593.69
Commission Costs            13.50
Spread+Slippage Costs    1,284.95
Net P&L                  4,295.23
Capital Utilization         61.2%

Return Drivers (per Symbol)
Symbol         P&L  Trades  Win Rate  % of Total
------------------------------------------------
PTON     11,780.49       8       62%      274.3%
GBTC     11,717.30       3      100%      272.8%
AQB      10,063.90       5       80%      234.3%
GENI      8,160.67       4       75%      190.0%
DNA       8,128.24       4       75%      189.2%
TXG       4,777.68       4      100%      111.2%
DKNG      4,384.46       2      100%      102.1%
SHOP      4,354.85       2       50%      101.4%
RBLX      3,201.87       3       67%       74.5%
SNAP      2,791.75       1      100%       65.0%
CDNA      2,734.87       4       50%       63.7%
EVGN      1,665.97       4       25%       38.8%
ADPT      1,523.47       5       40%       35.5%
AMGN      1,447.91       2       50%       33.7%
NFLX      1,298.58       3       67%       30.2%
BILL      1,167.24       2       50%       27.2%
ACHR      1,155.20       5       60%       26.9%
HEI       1,133.01       3      100%       26.4%
CGEN        755.88       4       50%       17.6%
ADBE        735.79       1      100%       17.1%
TDOC        733.22       2       50%       17.1%
LMT         297.81       2       50%        6.9%
DOCU        187.65       1      100%        4.4%
CLLS         85.86       3       67%        2.0%
LHX        -131.59       1        0%       -3.1%
OKTA       -180.54       2       50%       -4.2%
MELI       -253.64       1        0%       -5.9%
CRWD       -781.27       2       50%      -18.2%
AIR      -1,182.63       2       50%      -27.5%
PFE      -1,417.78       1        0%      -33.0%
ROKU     -1,606.48       1        0%      -37.4%
DE       -1,752.74       2        0%      -40.8%
FATE     -1,844.50       5       40%      -42.9%
BABA     -2,278.33       1        0%      -53.0%
ISRG     -2,597.75       1        0%      -60.5%
BFLY     -2,987.39       6       50%      -69.6%
EDIT     -3,145.06       5       40%      -73.2%
CERS     -4,183.86       3       33%      -97.4%
SNPS     -4,436.87       3        0%     -103.3%
BEAM     -4,478.71       1        0%     -104.3%
AMD      -4,713.58       3        0%     -109.7%
NTLA     -6,277.26       3       33%     -146.1%
ARCT     -7,994.31       6       33%     -186.1%
DDD     -11,228.16       4       25%     -261.4%
OPEN    -16,516.01       5       40%     -384.5%

Return Drivers (per Year)
Year         P&L  Trades  Win Rate  % of Total
----------------------------------------------
2022   -3,623.30      31       45%      -84.4%
2023   11,039.37      28       50%      257.0%
2024   15,091.70      28       64%      351.4%
2025  -36,037.11      31       35%     -839.0%
2026   17,824.57      17       65%      415.0%

Data: 2022-01-01 → 2026-04-29
Duration: 1085 periods · 4.3 years

Performance Metrics
Metric             Strategy      SPY
------------------------------------
Total Return          4.30%   48.95%
Annual Return         0.98%    9.67%
Annual Volatility    24.10%   17.48%
Sharpe Ratio           0.16     0.55
Max Drawdown        -37.95%  -25.37%
                                    
Calmar Ratio           0.03        —
Sortino Ratio          0.06        —
Omega Ratio            1.03        —
Stability              0.25        —
Skewness               0.67        —
Kurtosis              13.64        —
Scaled Trades             0        —
Rejected Trades           0        —
Alpha                     —     0.03
Beta                      —     0.05
```
### 4.8 — matched non-ARK twins, era 2022-2026

`uv run ibkr bt run strats/wip/ark_gate_twins_2022_2026.json`

```

Backtest Results
================================================================================

Worst Drawdown Periods
Net DD %   Peak Date  Valley Date    Recovery  Days
---------------------------------------------------
  29.84%  2023-12-29   2025-12-18           —   852
  15.66%  2022-07-15   2022-08-15  2023-03-27   255
  12.74%  2023-04-12   2023-05-23  2023-07-13    92
  10.83%  2022-03-11   2022-03-29  2022-05-18    68
  10.58%  2023-08-16   2023-10-03  2023-12-27   133

Period Drawdowns (peak-to-trough within period)
Period  Max DD              Peak            Valley
--------------------------------------------------
Month   15.13%  2025-06-05 00:00  2025-06-30 00:00

Trades
  130 trades — list omitted (pass --trades to print)

Trading Statistics
Metric                      Value
---------------------------------
Starting Capital       100,000.00
Total Trades                  130
Closed Trades                 130
Win Rate                   45.38%
Total P&L (Net)          3,133.47
Gross P&L                4,380.17
Commission Costs            13.00
Spread+Slippage Costs    1,233.71
Net P&L                  3,133.47
Capital Utilization         61.2%

Return Drivers (per Symbol)
Symbol         P&L  Trades  Win Rate  % of Total
------------------------------------------------
ELF      27,810.24       4       75%      887.5%
RIOT     12,311.00       6       67%      392.9%
UEC       5,782.34       6       83%      184.5%
CELH      5,554.42       4       75%      177.3%
AXON      4,214.81       4      100%      134.5%
NXE       3,388.29       1      100%      108.1%
PLUG      3,109.86       5       60%       99.2%
MRNA      2,738.88       6       50%       87.4%
MTCH      2,698.74       1      100%       86.1%
RMBS      2,622.26       3       67%       83.7%
ROP       1,990.76       1      100%       63.5%
LLY       1,274.99       2      100%       40.7%
CMCSA       964.35       1      100%       30.8%
EXEL        462.05       3       67%       14.7%
INTC        446.98       3       67%       14.3%
ADP        -126.93       1        0%       -4.1%
ARGX       -173.28       2       50%       -5.5%
BIIB       -242.57       2       50%       -7.7%
UNH        -380.13       2       50%      -12.1%
ASML       -444.87       1        0%      -14.2%
LULU       -626.45       3       33%      -20.0%
CRUS       -654.53       2       50%      -20.9%
BMRN       -667.39       1        0%      -21.3%
COST       -728.43       1        0%      -23.2%
AMAT     -1,002.05       2       50%      -32.0%
JAZZ     -1,035.66       2        0%      -33.1%
LRCX     -1,051.14       2        0%      -33.5%
BYDDF    -1,291.02       5       40%      -41.2%
ALKS     -1,367.91       1        0%      -43.7%
FORM     -1,550.21       2       50%      -49.5%
ALNY     -1,630.46       4       25%      -52.0%
AXSM     -2,965.97       6       33%      -94.7%
ONTO     -5,598.70       6       17%     -178.7%
MARA     -6,407.63       6       33%     -204.5%
ONON     -7,151.86       5       20%     -228.2%
SANA     -9,783.46       8       38%     -312.2%
URGN    -12,471.44       7       29%     -398.0%
LTBR    -14,884.41       9       11%     -475.0%

Return Drivers (per Year)
Year         P&L  Trades  Win Rate  % of Total
----------------------------------------------
2022    4,765.09      24       50%      152.1%
2023   -1,444.49      26       31%      -46.1%
2024   -2,081.10      30       43%      -66.4%
2025  -10,940.78      34       47%     -349.2%
2026   12,834.75      16       62%      409.6%

Data: 2022-01-01 → 2026-04-29
Duration: 1085 periods · 4.3 years

Performance Metrics
Metric             Strategy      SPY
------------------------------------
Total Return          3.13%   48.95%
Annual Return         0.72%    9.67%
Annual Volatility    17.48%   17.48%
Sharpe Ratio           0.13     0.55
Max Drawdown        -29.84%  -25.37%
                                    
Calmar Ratio           0.02        —
Sortino Ratio          0.06        —
Omega Ratio            1.02        —
Stability              0.33        —
Skewness              -0.40        —
Kurtosis               2.21        —
Scaled Trades             0        —
Rejected Trades           0        —
Alpha                     —     0.03
Beta                      —    -0.06
```
### 4.9 — wide non-ARK/non-ETF control, era 2022-2026

`uv run ibkr bt run strats/wip/ark_gate_wide_2022_2026.json`

```

Backtest Results
================================================================================

Worst Drawdown Periods
Net DD %   Peak Date  Valley Date    Recovery  Days
---------------------------------------------------
  16.61%  2022-07-15   2022-08-12  2023-03-01   229
  16.03%  2025-02-20   2025-08-25  2026-01-27   341
  10.48%  2023-10-11   2024-07-17  2024-10-31   386
   9.52%  2024-12-09   2025-01-08  2025-02-13    66
   8.55%  2022-03-15   2022-03-18  2022-05-05    51

Period Drawdowns (peak-to-trough within period)
Period  Max DD              Peak            Valley
--------------------------------------------------
Month   10.26%  2025-02-14 00:00  2025-02-27 00:00

Trades
  132 trades — list omitted (pass --trades to print)

Trading Statistics
Metric                      Value
---------------------------------
Starting Capital       100,000.00
Total Trades                  132
Closed Trades                 132
Win Rate                   53.03%
Total P&L (Net)        103,939.34
Gross P&L              105,232.26
Commission Costs            13.20
Spread+Slippage Costs    1,279.72
Net P&L                103,939.34
Capital Utilization         53.6%

Return Drivers (per Symbol)
Symbol        P&L  Trades  Win Rate  % of Total
-----------------------------------------------
APP     24,309.05       4       75%       23.4%
LITE    24,132.29       3       33%       23.2%
CVNA    23,735.41       4      100%       22.8%
ELF     18,383.92       4       75%       17.7%
BE      11,781.03       4       50%       11.3%
WDC      9,213.75       3      100%        8.9%
UEC      7,343.69       4      100%        7.1%
RIOT     7,189.18       4      100%        6.9%
FSLR     3,251.06       1      100%        3.1%
FORM     3,227.91       2       50%        3.1%
MRNA     3,040.27       4       75%        2.9%
ENPH     2,774.55       3       67%        2.7%
STX      2,696.43       1      100%        2.6%
PLUG     2,562.18       5       80%        2.5%
LULU     2,430.09       2      100%        2.3%
TRI      2,236.38       1      100%        2.2%
BROS     2,234.31       1      100%        2.1%
GRAB     1,938.34       1      100%        1.9%
ACLS     1,548.63       5       60%        1.5%
AXON     1,295.68       2       50%        1.2%
ALNY       783.63       1      100%        0.8%
CRUS       593.59       1      100%        0.6%
DECK       537.05       2       50%        0.5%
MU         508.59       1      100%        0.5%
NXE        310.03       1      100%        0.3%
SYNA       287.75       1      100%        0.3%
EXC        279.82       1      100%        0.3%
DXCM      -140.92       1        0%       -0.1%
URGN      -152.73       4       50%       -0.1%
BKR       -443.96       2       50%       -0.4%
JAZZ      -594.34       1        0%       -0.6%
MARA      -711.10       5       20%       -0.7%
LLY       -758.86       1        0%       -0.7%
RUN     -1,055.28       4       50%       -1.0%
SBUX    -1,287.24       1        0%       -1.2%
FANG    -1,335.35       2       50%       -1.3%
ALKS    -1,367.91       1        0%       -1.3%
AXSM    -1,535.05       4       25%       -1.5%
ONTO    -1,554.99       2        0%       -1.5%
MSTR    -1,592.78       5       20%       -1.5%
ONON    -1,605.20       1        0%       -1.5%
BYDDF   -1,616.37       1        0%       -1.6%
CELH    -1,620.97       6       50%       -1.6%
INTC    -1,659.13       1        0%       -1.6%
ON      -1,663.89       1        0%       -1.6%
PANW    -1,918.98       1        0%       -1.8%
UNH     -2,385.69       3       67%       -2.3%
AMBA    -3,412.50       2       50%       -3.3%
LSCC    -3,442.91       1        0%       -3.3%
NVO     -3,661.06       2        0%       -3.5%
SANA    -9,238.44       7       29%       -8.9%
LTBR    -9,929.65       7       14%       -9.6%

Return Drivers (per Year)
Year        P&L  Trades  Win Rate  % of Total
---------------------------------------------
2022  -3,096.47      29       45%       -3.0%
2023  35,558.07      24       54%       34.2%
2024  -3,711.57      27       44%       -3.6%
2025  21,255.93      38       58%       20.5%
2026  53,933.39      14       71%       51.9%

Data: 2022-01-01 → 2026-04-29
Duration: 1085 periods · 4.3 years

Performance Metrics
Metric             Strategy      SPY
------------------------------------
Total Return        103.94%   48.95%
Annual Return        17.92%    9.67%
Annual Volatility    16.96%   17.48%
Sharpe Ratio           1.06     0.55
Max Drawdown        -16.61%  -25.37%
                                    
Calmar Ratio           1.08        —
Sortino Ratio          1.49        —
Omega Ratio            1.20        —
Stability              0.83        —
Skewness               0.01        —
Kurtosis               2.93        —
Scaled Trades             0        —
Rejected Trades           0        —
Alpha                     —     0.19
Beta                      —     0.03
```
### 4.10 — SUPPLEMENTARY: full screened ARK sleeve, era 2022-2026

`uv run ibkr bt run strats/wip/ark_gate_arkfull_2022_2026.json`

```

Backtest Results
================================================================================

Worst Drawdown Periods
Net DD %   Peak Date  Valley Date    Recovery  Days
---------------------------------------------------
  29.65%  2022-07-19   2023-02-02  2024-08-21   764
  29.09%  2025-07-14   2025-07-21           —   289
  16.81%  2025-02-18   2025-03-10  2025-06-18   120
   9.03%  2025-01-10   2025-01-17  2025-02-10    31
   8.02%  2024-09-24   2024-10-22  2024-11-06    43

Period Drawdowns (peak-to-trough within period)
Period  Max DD              Peak            Valley
--------------------------------------------------
Month   28.85%  2025-07-02 00:00  2025-07-21 00:00

Trades
  151 trades — list omitted (pass --trades to print)

Trading Statistics
Metric                      Value
---------------------------------
Starting Capital       100,000.00
Total Trades                  151
Closed Trades                 151
Win Rate                   43.71%
Total P&L (Net)          6,806.22
Gross P&L                8,265.73
Commission Costs            15.10
Spread+Slippage Costs    1,444.41
Net P&L                  6,806.22
Capital Utilization         65.0%

Return Drivers (per Symbol)
Symbol         P&L  Trades  Win Rate  % of Total
------------------------------------------------
HOOD     17,786.06       4      100%      261.3%
RKLB     17,724.92       3       67%      260.4%
PTON     11,151.40       4      100%      163.8%
AQB       8,798.56       5       80%      129.3%
NVDA      7,871.54       2       50%      115.7%
PACB      7,191.44       3       67%      105.7%
GBTC      6,359.72       2      100%       93.4%
ACHR      6,263.30       2      100%       92.0%
GENI      5,836.48       3       67%       85.8%
EVGN      5,382.47       2       50%       79.1%
TER       5,324.47       1      100%       78.2%
DNA       4,832.48       3       33%       71.0%
PLTR      4,116.37       5       60%       60.5%
STNE      3,223.09       1      100%       47.4%
PSNL      2,946.38       5       80%       43.3%
META      2,741.47       1      100%       40.3%
SDGR      2,712.23       1      100%       39.8%
ESLT      2,582.09       2       50%       37.9%
TTD       2,190.84       2       50%       32.2%
CDXS      2,020.34       5       40%       29.7%
NFLX      1,835.05       1      100%       27.0%
PD        1,213.43       2       50%       17.8%
CRWD        725.40       1      100%       10.7%
VUZI        530.12       2       50%        7.8%
BILL        -61.79       1        0%       -0.9%
AVAV       -111.54       1        0%       -1.6%
IOVA       -138.21       2       50%       -2.0%
PDD        -165.48       4       50%       -2.4%
RBLX       -168.76       1        0%       -2.5%
EDIT       -242.77       3       67%       -3.6%
SNAP       -322.89       1        0%       -4.7%
TWST       -426.10       3       67%       -6.3%
LMT        -439.74       2        0%       -6.5%
CERS       -534.84       1        0%       -7.9%
TDOC       -559.51       1        0%       -8.2%
AIR        -612.20       2       50%       -9.0%
SHOP       -726.79       1        0%      -10.7%
TXG        -942.63       3       33%      -13.8%
VRTX       -966.35       3       33%      -14.2%
IRDM       -983.49       3       33%      -14.4%
IONS     -1,104.47       3       67%      -16.2%
CDNA     -1,202.54       3        0%      -17.7%
DDOG     -1,554.45       1        0%      -22.8%
VCYT     -1,660.47       1        0%      -24.4%
SE       -1,710.63       1        0%      -25.1%
CLLS     -2,023.83       1        0%      -29.7%
AMD      -2,056.88       1        0%      -30.2%
RXRX     -2,069.87       1        0%      -30.4%
ARCT     -2,670.97       2        0%      -39.2%
NRIX     -2,721.56       2       50%      -40.0%
COIN     -3,581.94       3       33%      -52.6%
QSI      -3,874.99       1        0%      -56.9%
BFLY     -4,053.35       7       43%      -59.6%
CGEN     -4,410.08       4       25%      -64.8%
TSLA     -4,513.30       2        0%      -66.3%
DDD      -5,509.59       2        0%      -80.9%
NTLA     -5,875.56       1        0%      -86.3%
JOBY     -6,007.34       3        0%      -88.3%
NIU      -7,039.10       3        0%     -103.4%
MASS     -7,068.29       4        0%     -103.9%
FATE    -10,359.07       5       60%     -152.2%
OPEN    -36,082.08       6        0%     -530.1%

Return Drivers (per Year)
Year        P&L  Trades  Win Rate  % of Total
---------------------------------------------
2022  -9,906.73      40       40%     -145.6%
2023  16,501.70      34       50%      242.5%
2024  -5,883.78      29       52%      -86.4%
2025   4,603.13      30       40%       67.6%
2026   1,491.91      18       33%       21.9%

Data: 2022-01-01 → 2026-04-29
Duration: 1085 periods · 4.3 years

Performance Metrics
Metric             Strategy      SPY
------------------------------------
Total Return          6.81%   48.95%
Annual Return         1.53%    9.67%
Annual Volatility    24.08%   17.48%
Sharpe Ratio           0.18     0.55
Max Drawdown        -29.65%  -25.37%
                                    
Calmar Ratio           0.05        —
Sortino Ratio          0.08        —
Omega Ratio            1.03        —
Stability              0.24        —
Skewness              -0.69        —
Kurtosis              10.73        —
Scaled Trades             0        —
Rejected Trades           0        —
Alpha                     —     0.04
Beta                      —     0.06
```
## 5. Per-cohort series and the two differences

`bt run` reports Sharpe, annual return and drawdown, but not a per-cohort
series, so the cohort numbers below are derived from the report's own
`equity_curve` (`bt run -F json`): a cohort is 21 consecutive bars of the trade
window starting at the first trade bar — the same clock the strategy rebalances
on — and the three arms are aligned on the dates they share, so every cohort is
the same window in every arm and the difference is paired. `n` is the cohort
count (11 for 2021, 50 for 2022-2026; the first partial window is dropped).

Two bases are shown because they are not the same number:

* **portfolio bp** = the cohort's return on equity (60% gross), what `bt`'s
  annual return compounds;
* **spread bp** = portfolio bp / 0.30 — long-leg-minus-short-leg on 100%
  notional legs, which is the basis §6's +381 / +135 / +246bp are stated in and
  therefore the basis the 150bp bar has to be read on.

### 2021 (11 cohorts)

| series | portfolio bp/cohort | t | spread bp/cohort | t |
|---|---|---|---|---|
| ARK sleeve (57) | −89.2 | −0.66 | **−297.2** | −0.66 |
| matched twins (57) | −206.5 | −0.82 | **−688.5** | −0.82 |
| wide control (100) | +38.5 | +0.11 | **+128.2** | +0.11 |
| ARK full sleeve (158) | −214.1 | −0.90 | **−713.7** | −0.90 |
| **ARK − twins** | +117.4 | **+0.36** | **+391.2** | **+0.36** |
| **ARK − wide** | −127.6 | **−0.30** | **−425.4** | **−0.30** |

### 2022-2026 (50 cohorts)

| series | portfolio bp/cohort | t | spread bp/cohort | t |
|---|---|---|---|---|
| ARK sleeve (54) | +18.9 | +0.26 | **+63.1** | +0.26 |
| matched twins (54) | +13.9 | +0.19 | **+46.2** | +0.19 |
| wide control (106) | +151.1 | +2.28 | **+503.7** | +2.28 |
| ARK full sleeve (119) | +41.5 | +0.56 | **+138.3** | +0.56 |
| **ARK − twins** | +5.1 | **+0.05** | **+16.9** | **+0.05** |
| **ARK − wide** | −132.2 | **−1.42** | **−440.7** | **−1.42** |
| ARK − ARK full sleeve | −22.6 | −0.32 | −75.2 | −0.32 |

### Annual-return differences (`bt`'s own numbers)

| era | ARK | twins | wide | ARK − twins | ARK − wide |
|---|---|---|---|---|---|
| 2021 | −2.77% | −13.44% | +8.37% | **+1067bp/yr** | **−1114bp/yr** |
| 2022-2026 | +0.98% | +0.72% | +17.92% | **+26bp/yr** | **−1694bp/yr** |

### What the two differences say

* **ARK − twins, long era: +17bp/cohort, t = 0.05.** The whole 150bp bar is
  nowhere near. On an annual basis the two arms are the same strategy: +0.98%
  vs +0.72%.
* **2021 ARK − twins: +391bp/cohort but t = 0.36 on 11 cohorts**, and the ARK
  sleeve's own level in that year is *negative* (−297bp). One year of
  rebalances cannot distinguish 391bp from zero, and the number is the twins
  sleeve losing (portfolio −206bp/cohort, vol 38.7% against ARK's 18.5%, worst
  lines `URGN`, `FORM`, `UEC`, `ENPH`, `BYDDF`), not ARK winning.
* **ARK − wide is negative in both eras** (−425bp, t = −0.30; −441bp,
  t = −1.42). The broad non-ARK stock panel is the *best* of the three books in
  the long era (+504bp/cohort, t = 2.28) — generic cross-sectional momentum on
  US stocks exists in this DB and ARK membership subtracts from it.
* **The full ARK sleeve (megacaps included) is not better**: +138bp/cohort
  (t = 0.56) in 2022-2026, −714bp in 2021. The composition shift caused by
  matched-pair dropping is not what produces the kill.

### Power, stated honestly

With a per-cohort difference sd of ~2200bp (spread basis) over 50 cohorts,
SE = 311bp, so a t of 2 needs a **+620bp** difference. This design therefore
cannot *certify* a 150bp increment — it can only kill one. The observed
difference is +17bp, i.e. not merely "under the bar" but indistinguishable from
zero, so the kill does not rest on the power limit. But do not read "t = 0.05"
as "we proved there is no edge of 100bp"; read it as "nothing here is large
enough to matter, and the point estimate is 10x under the bar".

## 6. Supplementary runs — the full screened ARK sleeve

The matched design drops ARK's megacaps (§3). These two runs use the identical
config on every ARK name that passed the screens (158 in 2021,
119 in 2022-2026), so they bracket the composition
objection. They are *not* part of the gate (no twin matching), they are the
sensitivity check. Their full output is in §4.6 and §4.10; headline: **−24.67% ann, Sharpe
−1.08 (2021)** and **+1.53% ann, Sharpe +0.18 (2022-2026)**, per-cohort
−714bp and +138bp. Adding megacaps makes ARK *worse*, not better.

## 7. `bt split --folds 4`, long era

Full output for both arms is reproduced below. Reading:

* ARK: mean OOS Sharpe **+0.02**, min −0.81, IS→OOS decay +0.44.
* twins: mean OOS Sharpe **−0.33**, min −0.85, IS→OOS decay +0.61.
* Both arms are unstable across folds and neither has a fold pattern that
  supports a stable edge. ARK's fold 4 OOS is −45.4% annual with max DD −48.7%.
* **Advisory flag:** ARK fold 4 OOS reports **Scaled = 18** (the other three
  folds and the full-period run report 0/0). Per `AGENTS.md`, non-zero `Scaled`
  means that window's results are order-sensitive and must not be quoted across
  symbol permutations — so that fold's −45% number is advisory, not evidence.
* Structural note: the strategy's rebalance clock is anchored to the window
  start, so a split fold is a *different book* (different phase, different
  entry composition), not a later segment of the same book. Fold-to-fold
  variation therefore mixes regime with book-phase luck. That is a property of
  the strategy as built, not of this gate.

### `uv run ibkr bt split strats/wip/ark_gate_ark_2022_2026.json --folds 4`

```
[fold 1]  IS 2022-01-01→2022-11-13 | OOS 2022-11-14→2023-09-25
  IS sharpe=0.02  ann=-1.73%
  OOS sharpe=0.16  ann=1.28%  maxdd=-13.14%
[fold 2]  IS 2022-01-01→2023-09-25 | OOS 2023-09-26→2024-08-06
  IS sharpe=0.61  ann=10.17%
  OOS sharpe=0.08  ann=-0.15%  maxdd=-12.36%
[fold 3]  IS 2022-01-01→2024-08-06 | OOS 2024-08-07→2025-06-18
  IS sharpe=0.60  ann=9.67%
  OOS sharpe=0.66  ann=13.30%  maxdd=-17.66%
[fold 4]  IS 2022-01-01→2025-06-18 | OOS 2025-06-19→2026-04-30
  IS sharpe=0.61  ann=9.82%
  OOS sharpe=-0.81  ann=-45.45%  maxdd=-48.65%

Split: ark_gate_ark_2022_2026
Fold  Phase  Window                 Sharpe     Ann   MaxDD  Kurt   Skew  Win  Trades  Scaled  Rejected
------------------------------------------------------------------------------------------------------
1     IS     2022-01-01→2022-11-13    0.02   -1.7%  -15.0%   4.4   0.52  47%      34       0         0
      OOS    2022-11-14→2023-09-25    0.16    1.3%  -13.1%   0.4  -0.24  56%      34       0         0
2     IS     2022-01-01→2023-09-25    0.61   10.2%  -15.0%   5.7   0.64  47%      58       0         0
      OOS    2023-09-26→2024-08-06    0.08   -0.2%  -12.4%   0.7  -0.05  45%      31       0         0
3     IS     2022-01-01→2024-08-06    0.60    9.7%  -15.0%   4.9   0.59  51%      81       0         0
      OOS    2024-08-07→2025-06-18    0.66   13.3%  -17.7%   1.8  -0.31  61%      28       0         0
4     IS     2022-01-01→2025-06-18    0.61    9.8%  -15.0%   4.2   0.36  54%     105       0         0
      OOS    2025-06-19→2026-04-30   -0.81  -45.4%  -48.7%  13.4  -0.88  49%      35      18         0

AGGREGATE: mean OOS Sharpe 0.02 · min OOS Sharpe -0.81 · IS→OOS decay +0.44 · 4 fold(s)
```
### `uv run ibkr bt split strats/wip/ark_gate_twins_2022_2026.json --folds 4`

```
[fold 1]  IS 2022-01-01→2022-11-13 | OOS 2022-11-14→2023-09-25
  IS sharpe=0.49  ann=6.86%
  OOS sharpe=0.54  ann=8.15%  maxdd=-15.38%
[fold 2]  IS 2022-01-01→2023-09-25 | OOS 2023-09-26→2024-08-06
  IS sharpe=0.44  ann=6.22%
  OOS sharpe=-0.85  ann=-17.17%  maxdd=-27.61%
[fold 3]  IS 2022-01-01→2024-08-06 | OOS 2024-08-07→2025-06-18
  IS sharpe=0.18  ann=1.63%
  OOS sharpe=-0.57  ann=-12.42%  maxdd=-16.18%
[fold 4]  IS 2022-01-01→2025-06-18 | OOS 2025-06-19→2026-04-30
  IS sharpe=0.01  ann=-1.17%
  OOS sharpe=-0.41  ann=-9.20%  maxdd=-18.89%

Split: ark_gate_twins_2022_2026
Fold  Phase  Window                 Sharpe     Ann   MaxDD  Kurt   Skew  Win  Trades  Scaled  Rejected
------------------------------------------------------------------------------------------------------
1     IS     2022-01-01→2022-11-13    0.49    6.9%  -15.7%   1.4  -0.64  48%      27       0         0
      OOS    2022-11-14→2023-09-25    0.54    8.1%  -15.4%   2.9   0.07  46%      28       0         0
2     IS     2022-01-01→2023-09-25    0.44    6.2%  -15.7%   2.0  -0.09  43%      51       0         0
      OOS    2023-09-26→2024-08-06   -0.85  -17.2%  -27.6%   4.9  -1.02  43%      28       0         0
3     IS     2022-01-01→2024-08-06    0.18    1.6%  -18.8%   1.6  -0.13  42%      74       0         0
      OOS    2024-08-07→2025-06-18   -0.57  -12.4%  -16.2%   4.6  -1.12  42%      31       0         0
4     IS     2022-01-01→2025-06-18    0.01   -1.2%  -22.3%   2.8  -0.47  43%     102       0         0
      OOS    2025-06-19→2026-04-30   -0.41   -9.2%  -18.9%   1.0  -0.49  47%      36       0         0

AGGREGATE: mean OOS Sharpe -0.33 · min OOS Sharpe -0.85 · IS→OOS decay +0.61 · 4 fold(s)
```
## 8. Cost reality — do not bury this

1. **Borrow is not modelled. At all.** This engine funds a short out of cash
   on the same mirror basis as a long and charges no financing. The short leg is
   30% of capital per cohort and holds 3 of the panel's *most-extended-down*
   names for 21 days — precisely the names that are hard or expensive to borrow
   (`PACB`, `VUZI`, `SDGR`, `QSI`, `PSNL` in this book). At 5%/yr a 21-day hold
   costs ≈42bp of the spread basis; at 20%/yr ≈167bp; hard-to-borrow at
   30–60%/yr is ≈250–500bp **per cohort**. The reviewer's estimate (125–500bp on
   21-day holds at 5–20%/yr) is in the right range; the top of it is the
   realistic case for this panel. **The 150bp gate bar is smaller than the
   borrow cost of the short leg.** Any certified +150bp would be eaten whole by
   financing; a −17bp difference is dead every way you cut it.
2. **20bp round trip is the optimistic end for this book.** The engine charges
   5bp spread + 2bp slippage per side plus a flat 0.05 commission. The names it
   trades are the 3 best / 3 worst of a mid/small-cap panel — the short leg in
   particular lives in the bottom dollar-volume decile by construction. Real
   spreads on 3–5 name extremes are wider, and a 21-day hold that must be
   unwound in one bar is a market-order exit.
3. **Same-bar-close fills.** The DSL fills at the signal bar's close
   (`fill_at_next_open=False`). A 126-day momentum book gets that bar's drift for
   free. It is applied identically to all arms, so the differences stand, but the
   absolute returns are optimistic.
4. **Short availability, recall risk, and the uptick rule are not modelled.**
   Neither is any borrow-fee feedback into sizing.

Direction of the error: every one of these makes the ARK arm *worse* relative to
zero, and they hit the ARK arm and the twins arm about equally (both are
mid/small-cap L/S books), and the wide arm is the least affected (large caps,
liquid). So costs cannot manufacture the ARK advantage — there isn't one to
manufacture, and financing pushes ARK further below the wide control.

## 9. What this does and does not settle

**Settled.** §6's control gap is closed: against a size-and-price matched twin
control, on 50 non-overlapping cohorts, the ARK increment is +17bp/cohort
(t = 0.05), and against a wide non-ARK panel ARK underperforms by ~1700bp/yr
(t = −1.42). §6's claimed +381bp/cohort ARK level also does not reproduce here:
this construction gives −297bp (2021) and +63bp (2022-2026) for the ARK sleeve
and −714bp / +138bp for the full screened ARK sleeve. Something in §6's
fund-cohort reconstruction (all 8 funds, T+1-open entries, a 378-name panel) is
doing work that a straightforward engine run on the DB's candle coverage does
not reproduce. That is a second, independent reason to stop quoting §6 as an
open opportunity.

**Not settled.** (a) The DB has no room for a bigger control pool, so the twin
match consumes 57%/51% of it and the twins overlap the wide control more than
an ideal design would. A deeper DB (more non-ARK mid/small caps) is the only fix
and it is a data acquisition task, not an analysis task. (b) Borrow cost is not
in the engine; even a *positive* certified increment would need financing
modelled before anyone called it tradeable. (c) The membership-timing lookahead
in §3.1 (static per-era panel) is small but real.

**Verdict per the pre-registered rule:** ARK − twins = **+17bp < 150bp** →
**KILL**. §6 is generic cross-sectional momentum at best; the honest reading of
the sign is that ARK membership *subtracts* from it.

## 10. Artifacts

| path | content |
|---|---|
| `universes/ark_gate_{{ark,twins,wide,arkfull}}_{{2021,2022_2026}}.json` | the 8 universes |
| `universes/ark_gate_summary.json` | every filter count, decile cuts, twin pairs, dropped names, ticker-reuse exclusions |
| `strats/wip/ark_gate_*.json` | the 8 configs (identical except `symbols` + era) |
| `scripts/ark_gate_universes.py` | universe builder (PIT membership, screens, decile match) |
| `scripts/ark_gate_configs.py` | config materialiser |
| `scripts/ark_gate_cohorts.py` | per-cohort series + t-stats from the JSON equity curves |
| `docs/research/ark_gate_runs/` | raw `bt run`/`bt split` stdout, the `-F json` payloads the cohort series are derived from, and the per-cohort CSVs |

Nothing was committed. `assets/ark_flow.csv` and `docs/research/ark_conclusion.md`
were not modified. The strategy module was not modified.
