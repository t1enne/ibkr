# ARK edge — desk verdict

Author: discretionary trader review. Read-only inspection of `py/assets/ark_nav.csv`,
`py/assets/ark_flow.csv`, `../data/ibkr.db` (1h candles, read-only URI). No strategy code written,
no data modified. Everything below is either measured from these files or labelled **estimate**.

**One line:** nothing here is tradeable as a strategy. The flow file is dominated by in-kind
creation/redemption legs, not conviction trades; the premium trade died in 2018 and its remaining
occurrences are stale-NAV artifacts in the funds with foreign holdings; the "pairs" result does not
survive a hedge-ratio stability check. Flat is the position.

---

## 0. Read this before the verdicts — the flow file does not measure what it claims

Three mirror sources, and they are not interchangeable:

| source | dates | rows/date (median) | span |
|---|---|---|---|
| arktrack | 258 | **196** | 2020-10-19 .. 2021-10-28 |
| tsekityam | 145 | **215** | 2021-03-05 .. 2021-10-01 |
| cathiesark | 1146 | **13** | 2020-08-21 .. 2026-05-15 |

- `arktrack` and `tsekityam` agree **exactly** (verified row-by-row on 2021-06-15: 65/65 identical
  keys and values). They are the same feed.
- `cathiesark` is neither a superset nor a subset and disagrees on values: 12.8% of multi-source
  keys differ in share count, ratio p10 = 0.14, p90 = 3.87 (JD 2021-06-15: 54,700 vs 47,000;
  KSPI: 4,393 vs 1,703). A 5-day share sum matches on only 11/34 (and 1/56) keys — so it is not a
  simple time shift either.
- **Full-coverage daily flow exists for ~12 months only: 2020-10 .. 2021-10, one regime (the 2021
  growth melt-up).** Everything after 2021-10 is `cathiesark` at 4-15 line items/day, falling to a
  median of **4 rows/day in 2025** — i.e. a non-random sliver of ARK's actual book.
- 275 of the 435 flow tickers have **no candles** in `ibkr.db` (only 160 priceable). ARKX, PRNT and
  IZRL have flow rows and zero price history. Any event study here is implicitly a large-cap US study.
- Timestamps/`notional` are only populated for `arktrack` (48,608 rows); `notional` is empty for the
  other two mirrors, so dollar flow has to be reconstructed as shares x close.

### The structural problem: the file is NET position deltas, and most of it is basket plumbing

- Only **0.3%** of `(fund, ticker, date)` triples contain both a buy and a sell. ARK certainly buys
  and sells the same name on the same day; the file nets them. What you would "follow" is a net
  position change, not a trade.
- **90.2%** of all 52,714 line items fall on fund-days that are >75% one-sided (42.3% of fund-days
  are >75% buys, 41.2% are >75% sells). That is the signature of in-kind creation/redemption: an AP
  delivers or takes a pro-rata basket and every holding moves together. A "big add" in this file is
  frequently **not a decision by ARK at all**.

This single fact explains the quant's inverted dose-response (biggest flow quintile = passive inflow
legs, and inflows chase performance). It also means A2/A3/A4 as *tested* were testing a mixture of
signal and plumbing; the mixture has no stable sign, which is what we find in section 3 and 4 below.

---

## 1. The six verdicts

| # | Trade | Size | Horizon | Who is on the other side, and why they're wrong | Edge decays to | Verdict |
|---|---|---|---|---|---|---|
| 1 | Buy ETF on discount, hold to convergence | 1-2% NAV per event, ~$2-5M | 1 day | Panic seller who needs out at any price after the close already printed | 2015-16: 170-206 fund-days/yr at <-25bps. 2019-23: 7-40/yr. Capturable part ~8bps/event | **PASS** |
| 2 | Follow disclosed buys at T+1 | n/a | 1 day | Nobody identifiable — you are the liquidity for ARK's own next-day rebalance | +20.6bps gross, ~10bps after the first hour; cost 15-25bps | **PASS** |
| 3 | Fade disclosed buys T+1..T+10 | n/a | 1-10 days | Same crowd, opposite side; sign is not stable | Sign flips with aggregation: my notional quintiles are monotone *increasing* (+0.20% top vs -0.00% bottom) | **PASS** |
| 4 | Front-run ARK's concentration sequences | n/a | 2-5 days | Other people reading the same free EOD file | Continuation is 47%/62%/73% detectable, forward returns still 15bps | **PASS** |
| 5 | ADSK/PTC intra-fund relative value | n/a | 6 bars claimed | Nobody — there is no enforcing mechanism | Beta not identifiable: 0.97 full-sample, -0.08..1.26 by year; hl 1,178 bars | **PASS** |
| 6 | Cash-and-carry / creation-redemption adjacent | n/a | n/a | The AP, who is not on the other side at all | Requires being the AP; basket is $1M+ in kind | **PASS** |

---

## 2. A1 — Buy the discount, hold to convergence

**Trade.** Fund trades below NAV at the close; buy, hold for the gap to close.

**Trigger.** `premium_pct <= -0.50` at the close of fund F in {ARKK, ARKW, ARKQ, ARKG}; exit at the
next close, or at premium >= 0, whichever first. No entry in ARKF/PRNT/IZRL (see below).

**What the data says.** Post-2019, |premium| > 25bps at the close, by fund (1,982 fund-days):

| fund | days < -25bps | days < -50bps | +days > +25bps |
|---|---|---|---|
| ARKK | 13 (0.7%) | 3 | 18 |
| ARKQ | 8 | 2 | 19 |
| ARKW | 41 (2.1%) | 15 | 12 |
| ARKG | 13 | 3 | 34 |
| ARKF | 87 (4.4%) | 14 | 165 |

Whole complex, days with any liquid fund < -25bps: 2014: 36, **2015: 170, 2016: 206**, 2017: 32,
2018: 18, **2019: 7**, 2020: 40, 2021: 19, 2022: 21, 2023: 7, 2024: 32, 2025: 16, 2026: 20.
The regime break is not subtle: ~20-40 events/yr now, most of them in the funds with the worst NAV.

**Entry is not where you think it is.** The close-based premium is only known after ~18:00 ET. Earliest
retail action is 09:30 ET next session, so the real question is where the reversion prints. On 221
event-days with 1h bars, decomposing against the other ARK funds the same night:

- **discount < -25bps (n=135):** mean premium -0.42%, of which excess **gap +7.6bps**, excess
  **intraday +8.3bps**. About **half of the fund-specific reversion is in the opening gap** — i.e.
  before you can trade. Peer-adjusted, only ~16bps of the 42bps is fund-specific at all; the rest is
  common NAV move you do not capture by owning the ETF instead of the basket.
- Per fund the discount events are thin and noisy: ARKK n=13 (+2.4/+0.1bps), ARKW n=41 (+0.5/+2.7),
  ARKF n=62 (+3.9/+3.9), ARKQ n=8 (+88.7/-14.5), ARKG n=11 (+2.4/+80.5). The big numbers sit on
  n=8-11 samples.

**The killer.** The funds with the most discount days are the funds whose NAV is least trustworthy.
ARKF holds HK-listed names (SE, JD, BEKE, 6060.HK, 9923.HK) whose closes are 8 hours stale at the US
close; a discount in ARKF is mostly the US tape moving after Asia closed, and the "convergence" is the
NAV catching up — not something you can capture by owning the ETF. Same signature in PRNT (mean
premium -0.149%, 66.9% of days in discount) and IZRL (-0.095%, 58.7%) — those are **permanent
structural discounts**, not convergence. Note that 87 of the 162 post-2019 discount events sit in ARKF.

**Verdict: PASS.** Real but ~8bps capturable per event, 10-20 events/yr, one-day hold, concentrated in
the least reliable NAV. Cost: 3-6bps spread in ARKW plus the open-auction slippage eats the reachable
half.

**What would change it:** intraday IOPV (see section 5) to separate stale-NAV from genuine dislocation.
That, and only that, would resurrect the 2015-16 version of this trade.

---

## 3. A2 — Follow the disclosed buys at T+1

**Trade.** ARK bought X during day t; buy at the open of t+1, exit at close of t+1.

**Measured, full-coverage window (52,714 line items, 259 dates), entry at t+1 open, excess vs ARKK:**

| horizon | buy cohort | sell cohort | long/short |
|---|---|---|---|
| t+1 | **+0.206%** | -0.012% | +0.218% |
| t+2 | +0.198% | -0.000% | +0.198% |
| t+5 | +0.145% | +0.022% | +0.122% |
| t+10 | +0.232% | +0.109% | +0.123% |

Confirms the quant's +0.21%. Two things it does not survive:

1. **Half the edge is in the first hour.** Intra-day decomposition of the t+1 excess: first bar
   +0.106%, rest of the session +0.100%. You either take the opening print (widest spread of the day,
   3-10x normal on these names) or wait out the first hour and keep 10bps.
2. **Cost.** Round-trip on a mid-cap at the open: 3-8bps spread + 1bp commission + opening-auction
   slippage = **15-25bps**. Gross 20.6bps, net ≈ 0. Turnover is 100%/day, so the book pays that
   every single session.

**Counterparty.** There isn't one you can name. ARK is not leaving money on the table for you; you are
buying the same names ARK is buying because both of you are reacting to the same free file, and the
marginal seller is whoever provided liquidity into ARK's own next-day continuation.

**Verdict: PASS.** Gross edge equals transaction cost, and the file's biggest "buys" are creation legs
(90.2% one-sided fund-days) rather than conviction — consistent with the inverted dose-response.

---

## 4. A3 — Fade the disclosed buys / A4 — front-run the sequences

### A3 (fade T+1..T+10)

I could not reproduce an inverted dose-response, and that is the point. Ranking the same events by
reconstructed notional (shares x close) into quintiles, t+1 excess vs ARKK:

| notional quintile | median $ | t+1 excess | t+5 excess |
|---|---|---|---|
| 1 | 5.4M | -0.002% | -0.060% |
| 2 | 21.4M | +0.071% | +0.066% |
| 3 | 54.9M | +0.130% | +0.009% |
| 4 | 140.1M | +0.086% | +0.236% |
| 5 | 589.8M | **+0.201%** | +0.167% |

Monotone *increasing*, not inverted. The quant's inversion almost certainly comes from the mirror
mixture: in the 2020-10..2021-10 overlap the duplicated `arktrack`/`tsekityam` rows beef up mid-size
events, and `cathiesark`'s 12.8% share-count disagreements scramble the ranking elsewhere. Two
researchers, two signs, same "data" — that is a data problem, not an edge.

Either way the fade is uninvestable for a reason that does not depend on the sign: **half of the
one-day move is in the opening gap**, so a short entered at t+1 open starts 10bps behind, and the
reversal you are trying to harvest is smaller than the borrow+spread on exactly the names ARK buys
(small-cap genomics/fintech, 20-100bps wide, 5-50% borrow).

**Verdict: PASS (no stable sign, and the sign is the whole trade).**

### A4 (sequence jumping)

The sequence *is* detectable, and cheaply — this is the one thing in the file that is real:

- 19,207 runs of same-(fund, ticker, side) consecutive sessions. `P(run >= 2) = 46.8%`,
  `P(>=3 | >=2) = 62.2%`, `P(>=4 | >=3) = 72.7%`.
- So on ~47% of disclosed line items you can say, at t+1 open, "they will do it again tomorrow" with
  62% confidence. Mechanically true, publicly visible, free.

It earns nothing. Forward excess vs ARKK by position in the run:

| run position | buy t+3 | buy t+5 | sell t+3 |
|---|---|---|---|
| 1 | +0.139% | +0.117% | +0.035% |
| 2 | +0.168% | +0.128% | -0.080% |
| 3 | **+0.002%** | **-0.129%** | -0.013% |
| 4+ | +0.183% | +0.228% | +0.115% |

No monotonicity, no sign flip you can trade, everything inside 15-20bps of gross against 20bps+ of
cost. The information "ARK will keep buying" is worth approximately zero, because everybody with the
same free file knows it and ARK's order size is small relative to the names' float.

**Verdict: PASS.** Detectable pattern, no payment for it.

---

## 5. A5 — ADSK/PTC intra-fund relative value

The quant's fit (p=0.0024, half-life 6 bars, beta 0.48) does not survive being applied out of sample.
Name-level facts on 12,218 common 1h bars, 2019-11 .. 2026-11:

- **Full-sample beta = 0.969**, not 0.48. Residual standard deviation 2,332bps; residual range
  -5,903bps .. +4,642bps.
- **AR(1) on the full-sample residual: rho = 0.9994, half-life = 1,178 bars (~7 months).** That is a
  random walk with a trend, not a cointegrated pair.
- **The hedge ratio is not identifiable.** Yearly betas: 2019 -0.084, 2020 0.928, 2021 0.253,
  2022 0.292, 2023 1.202, 2024 0.260, 2025 1.257, 2026 1.173. Rolling 250-bar beta ranges
  **-1.16 .. +2.36 (sd 0.50)** — the sign flips. A rolling re-fit "half-life of 6 bars" is a tautology:
  the residual is defined to mean-revert to a mean you just estimated.
- Pinning beta=0.48, the residual drifts from +14,365bps to +26,774bps over the sample — a
  12,000bps one-way move. That is a position, not a spread.
- Economics even if it were stable: 1-sd excursions (n=1,729 premium / 2,684 discount) pay **+7bps /
  +3bps** over 4 bars of *anything*. Two legs at 2-3bps each plus slippage = 8-12bps round-trip on
  1.48x gross. Negative before you add the 6-month-beta-decay risk.

**Verdict: PASS.** Not a pair. Ten years of a shared uptrend plus a free hedge ratio fitted to the
same window will always produce a small p-value.

---

## 6. A6 — Cash-and-carry / creation-redemption

What the AP does that you cannot: they hold the **creation basket**, they buy the basket and deliver
it in kind against newly created shares at NAV, and they do it in units of 10,000-50,000 ETF shares
(millions of dollars) with an in-kind tax-free exchange and a same-day hedge on the futures/tape.
They never touch the secondary-market premium the way you would. On the ARK complex that is exactly
the force that closed the pre-2019 discount — which is why the discount trade died: **the AP
arbitrage worked, and the 2015-16 edge was the shrinking of an inefficiency, not a repeatable trade.**

The retail imitation — short the ETF, buy the 100-name basket, wait for convergence — requires either
an AP relationship or 100 short locates. On the constituents ARK actually holds (small-cap genomics,
HK-listed fintech, pre-revenue space names), the locate either does not exist or costs 20-100%/yr.

**Verdict: PASS.** This is an AP's business, and they already took the money.

---

## 7. B — Market-structure reality

**When the data actually exists.**

- ARK publishes holdings/trade files on its site each business day **after the close** (the trade
  notification email lands ~16:15-18:00 ET; the site file updates during the evening). Retail earliest
  actionable time is therefore **09:30 ET the next session — a 15.5-17.5 hour lag**, and the file
  gives you the *net end-of-day holdings change*, with no execution price and no execution time.
- The mirrors are worse. `arktrack`/`tsekityam` snapshots disagree with `cathiesark` on 12.8% of
  overlapping share counts, both "same-day" — that is snapshot-timing noise, and it is the same noise
  that would corrupt a live sequence detector.
- Both-sides netting (0.3% of ticker-days) means a large buy and a large sell the same session look
  like a small buy.
- **The disclosed trade is not in the price by the time you can act — it *is* the price.** Half of the
  t+1 excess has already printed in the opening gap (+7.6bps of the discount reversion, +0.106% of the
  buy-signal edge). What is left is a coin flip against the spread.

**Spread / liquidity (estimate from tape knowledge, not derivable from these files — the `volume`
column in `ibkr.db` is unusable: ARKK shows a $4.8M median dollar volume vs TSLA $474M, a 100:1 ratio
where the real one is ~40:1 and both are ~50x too small).**

| instrument | ADV (est) | spread (est) | short borrow (est) |
|---|---|---|---|
| ARKK | $200-600M | 1-2bps (1c on ~$70) | 1-3%, easy |
| ARKW / ARKG / ARKQ / ARKF | $5-40M | 3-10bps | 2-5%, available |
| ARKX / ARKT / ARKD / ARKI | $1-5M | 10-30bps | hard-to-borrow |
| ARK's typical added constituent | $1-20M, often <$2M | 20-100bps | 5-100%, often none |

Consequence for sizing: the signal fires on ~200 line items/day but a realistic book can only trade
the top 20-30 by liquidity, and those are precisely the names where ARK's flow is smallest relative
to ADV. The tranche that works is the tranche that has no edge; the tranche with the edge is untradeable.

---

## 8. C — What would actually make this tradeable, and what it costs

Ranked by expected value per dollar spent. **The first one is free, and it is the only one that
changes a verdict.**

1. **ARK's full daily holdings LEVELS (not deltas) — $0.** ARK publishes full holdings and a
   historical archive. With levels you can decompose each day's delta into
   (a) pro-rata creation/redemption legs and (b) active trades — the split we can only *infer* here
   from the 90.2% one-sided statistic. That is the difference between front-running ARK and
   front-running an AP's basket delivery. Everything in A2/A3/A4 should be re-run on the residual
   after removing creation legs. **This is what I would buy first: nothing. I would download this.**
2. **Intraday IOPV / indicative NAV + consolidated tape — ~$200-2,000/mo** (Polygon/Databento tier,
   or ~$30/mo L1 through a broker API for the ETFs plus the top 100 constituents). Turns the premium
   series from one close-based observation per day into a measurable, tradeable dislocation and
   separates stale-NAV (ARKF/PRNT/IZRL) from genuine discount (ARKK/ARKW).
3. **Options on ARKK/ARKW — ~$200-500/mo** (weekly chains since 2020). Completely untested here and
   the only unexplored source of *forward-looking positioning* on this complex. A cheap, liquid,
   mechanically-driven underlying with a weekly expiry is a much better place to look for an edge
   than a 4-15 row/day holdings file.
4. **Real-time creation/redemption / ETF flow — $20-50k/yr** (Bloomberg ETF, FactSet). The only way
   to see inflows before the holdings file shows them as one-sided plumbing days. Almost certainly not
   worth it at retail scale.
5. **Short borrow / locate data — ~$10/mo IBKR API, but availability is the binding constraint** and
   it is exactly what kills the short leg of A3/A5.
6. **Intraday holdings** — does not exist for sale at any price to a retail algo. ARK's own daily file
   is the ceiling.

Untested with data we *do* have, if anyone wants one more pass: (i) the 2024 ARKW cluster (25 days
< -25bps in one year, against 0-1 in the surrounding years) — that is a single anomaly worth
explaining before it is traded; (ii) ARK's *exit* events (last day a name appears) versus *trims*;
(iii) the dose-response re-run net of creation legs.

---

## 9. D — Blunt judgement

**Run none of them.** Six verdicts, six passes, and the passes are not close: the gross edge in the
best of them (A2, +20.6bps/day) is exactly the round-trip cost, and the best risk-adjusted one (A1)
fires 10-20 times a year at ~8bps reachable.

**If I were forced to run one tomorrow:** A1, long-only, **ARKK and ARKW only** (never ARKF — stale
NAV), entry on `premium <= -50bps`, exit at the next close, hard no-trade if the ETF's top-10
constituents fell >2% that session (that is a real repricing, not a dislocation). Size **1-2% of NAV
per event, $2-5M for a $200-500M book, max one event live at a time.** Expected: ~10-15 events/yr,
30-40bps gross each, ~2/3 of them in the opening gap, so realistically ~150-250bps/yr on capital
that is deployed <5% of the time — about 10bps/yr of portfolio return for a compliance file and a
manual process. That is not a strategy, it is a hobby with an audit trail.

**What kills each one first:**

| trade | first thing that kills it |
|---|---|
| A1 | the 09:30 auction — half the reversion is in the gap, and the remaining 8bps is inside the spread |
| A2 | 100% daily turnover against a 20.6bps edge on names whose opening spread is 3-8bps |
| A3 | sign instability (my quintiles run opposite the prior screen) plus 20-100bps spreads on the shorts |
| A4 | the pattern is free and public; 47%/62%/73% continuation is worth 0bps |
| A5 | the hedge ratio. 250-bar beta range -1.16..+2.36 means the "spread" is a directional bet you cannot size |
| A6 | you are not an AP, and the AP already closed the gap |

**Portfolio-level honesty:** the reason every one of these dies is the same. ARK's disclosures are a
free, EOD, netted, plumbing-contaminated file, and the only force with a mandate to exploit the
resulting mispricings — the authorized participant — has already done it. The pre-2019 discount edge
was the *evidence* of that arbitrage working, and its disappearance is the proof that the edge is
gone rather than dormant. There is no residual for a retail algo here.
