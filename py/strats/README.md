# `strats/` classifications — pass / wip / fail

Each strategy config is classified per the backtest-first honesty protocol
(Sharpe, return vs **SPY buy-and-hold** over the same window, profit factor,
expectancy, $/trade cost, trade count, regime stability, point-in-time
universe check). The benchmark column is the buy-and-hold total return of the
gate/benchmark index over the same backtest window.

## PASS — money-consistent, enough samples, beats buy-and-hold

| config | bar | n | Sharpe | tot_ret | bench | why |
|---|---|---|---|---|---|---|
| `momentum_compression_breakout_ae_gate_SPY` | 1d | 349 | 1.40 | +491% | SPY +140% | Reproducible standout. Entropy(SPY) regime gate. **SPY-gate specific** — see caveats. |
| `cup_handle_dsl` | 1d | 122 | 1.38 | +187% | SPY +80% | Beats hold 2.3x, PF 2.82, +768/trade, 5yr. Re-validated. |
| `vwatr_div_exp8_6y_risk0.08` | 1d | 122 | 1.12 | +2223% | SPY +118% | VWATR slope-divergence exit + `risk_pct=0.08`. **BULL-REGIME RESULT** — see caveats. |

> **Read this before trusting either number.** Both winners carry two standing
> caveats established by A/B stress tests (findings recorded here with the
> historical session notes; probe configs were retired after the negatives were
documented):
> 1. **Survivorship-weighted universe.** Symbol lists are curated today's
>    winners. A point-in-time/survivorship-free rebuild cut the momentum
>    config's +491%→~+187%, SR 1.40→~1.0 (one-variable A/B, same engine). Any
>    live expectation should assume the *thinned* PIT numbers, not the
>    retrospective +491%.
> 2. **SPY-gate load-bearing.** A clean 2x2 (universe fixed, only
>    `regime_symbol` × `benchmark_symbols` varied) proved `regime_symbol=SPY`
>    is causal and the benchmark symbol is cosmetic:
>    gate=SPY → SR ~1.40 regardless of bench; gate=QQQ → SR ~0.77 regardless of
>    bench. Do **not** "fix" these to gate/bench the universe's own index —
>    that removes the edge. Deploy only with `regime_symbol=SPY`.

### `vwatr_div_exp8_6y_risk0.08` — performs strongly in BULL

8 high-vol names (LTBR SRPT MSTR PLUG ENPH MARA NTLA LULU), 1d, 6y
(2020-09 → 2026-09). +2223% vs SPY +118%, SR 1.12, DD −50%, kurtosis 12.5,
n=122, win 57.4%. Walk-forward folds=3: OOS SR 0.85 / 1.28 / 1.06 (mean 1.06,
min 0.85, IS→OOS decay +0.10).

**Standing caveat — read before deploying.** 87% of P&L comes from 2024–2026;
the first four years contribute 13%. 2020 is *worse* at `risk_pct=0.08` than at
0.02 (5.2k vs 12.1k): sizing amplifies the tail in both directions and the tail
only exists late in this window. **This is a late-window / bull-regime result,
not an all-weather edge.** The 2022–2023 improvement (hole −11.1k→+12.3k flat→
positive) is the absent right tail becoming survivable under larger size — not a
new bear edge.

What was actually fixed: `risk_pct` had been hardcoded at 0.02 and **never
swept**. The 6y response is unimodal with an interior peak (0.04/0.06/0.08/0.10
→ mean OOS 0.38/0.95/1.06/0.88), unlike the degenerate `decel_ratio` which was
monotone toward its boundary. Three bootstraps exclude zero at 0.08 (symbol
n=8 CI95 [985%,3235%]; year-block n=7 CI95 [820%,3811%]; jackknife t=2.65);
leave-one-out worst case still +1723%, so no single symbol is load-bearing.
Small-sample and correlated-name caveats apply — see the module docstring in
`src/bt/strategies/vwatr_div_dsl.py` for the full ledger, including the proof
that **no entry gate can work here** (45 external series, 0/45 carry signal
within the 2022-09→2024-08 hole).

## WIP — promising metrics, but a standing hazard blocks full PASS

| config | n | Sharpe | hazard |
|---|---|---|---|
| `entropy_vp_breakdown` | 72 | 1.77 | Best risk-adjusted but the config window is **ONE YEAR (2022)** → regime-fit risk. Extend window + walk-forward. |
| `bear_breakout_dsl_2022` | 64 | 1.47 | Highest PF (4.69) but a *bear-breakout* run mostly in bull/grind → edge likely rides 2022. Needs a bear-regime split. |
| `vp_breakout_dsl_regime` | 202 | 0.91 | Positive (+59%) but trails SPY and Sharpe<1; regime filter may cost alpha. Tune. |
| `rsi_divergence_dsl` | 25 | 0.91 | PF 4.22 but n=25 < 30 → no statistical confidence yet. |
| `xbi_gated_laggard_dsl` | 61 | 0.52 | Biotech XBI-gated worst-laggard basket. +21.6% vs XBI −5.6% full-window 2020–2023, but **4-fold walk-forward OOS Sharpe ~0.03** (fold-3 biotech bear −11%, SR −1.02). Cross-sectional green-regime only; gate reduces DD but not the bear. 2024+ OOS untouched. Needs regime-robust rebuilt or explicit green-biotech-only scope. |

## FAIL — negative, weak, or no statistical confidence

- **Rejected A/B (documented negative):** `momentum_compression_breakout_ae_gate_flushguard_SPY` — a "book-cooling" entry-cap rule could not fix the momentum strategy's worst-month MTM losses (those come from oversized OPEN single-name positions, not concurrent-entry breadth). Aggressive cooling made results worse (SR 1.40→0.92); mild cooling could not clear its own cost. No-op guard verified byte-identical to baseline. Keep the strategy module so this can be re-run.
- **Losing / negative expectancy:** `kalman_pairs_dsl` (−13.6%, PF 0.71).
- **Underperforms buy-and-hold badly:** `trend_pullback_atr_trail_dsl_L15_r15` (+73% vs SPY +234%), `shannons_demon_dsl` (+156% vs its own 50/50 SPY+GLD hold +189%), `kalman_mr_regime_dsl` (+9.9% vs SPY +44%).
- **No statistical confidence (n<30):** `ema_cross_dsl` (7), `macd_mfi_divergence` (8), `pf_equal_weight` (4), `rsi_bearish_divergence` (1).
- **Retired after honest fail — biotech 1h single-name reversions** (removed): `bio_post_catalyst_fade_dsl` (24-name biotech, short failed up-breakouts on 1h; −0.87%, SR −0.165, n=26, gap-killed) and `bio_failed_down_breakout_long_dsl` (long mirror; ≈0, SR 0.01, n=32). Both under n≥40, no OOS edge. Lesson: biotech 1h idiosyncratic catalyst gaps (kurtosis 300–800) destroy single-name time-series edges; the surviving structure is cross-sectional + basket-level.

## Documented negatives from the momentum-gate investigation

These A/B probes (QQQ-gate, biotech/XBI-gate, and the gate×benchmark split)
were standalone run configs, since retired after recording — they are NOT pass
candidates, and the findings survive here as documented negative transfers:
- Momentum AE-gate on a Nasdaq/QQQ bench: SR → ~0.77 (edge was SPY-gate).
- Momentum AE-gate on biotech/XBI: near-dead (SR ~0.25, PF ~1.24) — setup does
  not fit biotech's binary-event / gap profile.

They can be rebuilt from the caveats above if re-testing is ever needed.

## Run any resident config

Configs live by classification under `strats/{pass,wip,fail}/` (names drift as
strategies are moved/retired — this ledger tracks them, the filesystem is
canonical):

```bash
uv run ibkr bt run strats/pass/momentum_compression_breakout_ae_gate_SPY.json
uv run ibkr bt run strats/wip/cup_handle_dsl.json
uv run ibkr bt run strats/wip/entropy_vp_breakdown.json
# batch all configurations in one bucket to JSON/text:
uv run ibkr bt run <config> -F json
```
