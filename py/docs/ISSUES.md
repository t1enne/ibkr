## Issue 1 — `_scale` admits near-zero positive equity

**Where:** `src/bt/strategies/fundamentals_pair_dsl.py`, `_scale()`.

**What:** the guard is `value is None or value <= 0`, so equity values in
`(0, small)` pass. That value is the denominator of _both_ score components
(`level` and `drift`), so a leg whose book has been wiped toward zero produces an
astronomically large score, wins its pair unconditionally, and is opened at full
size.

**Why it matters:** the failure is silent and directional — it does not skip the
pair, it sizes the leg. A near-bankrupt filer is exactly the leg that should be
_hardest_ to score, and instead it becomes the strongest signal in the book. Any
pair list that includes a distressed name is exposed.

**Not exercised in the Stage-1 run:** OKTA equity 5.7–6.3B, ZS 0.8–1.4B.

**Options:** floor the denominator (e.g. require equity above a fraction of the
leg's own asset base or revenue), or treat a below-floor equity as unscoreable
(`None`) the way negative equity already is. The second keeps the existing
"missing fundamental is not a weak fundamental" contract and adds no param.

**Status:** open. Fix deliberately, not as a drive-by — the change alters which
pairs are tradable.

---

## Issue 2 — report labels entry on the signal bar, exit on the fill bar

**Where:** `src/bt/portfolio/pure.py` — `Trade.entry_time = signal.timestamp`
(~line 439) but `exit_time = fill.timestamp` (~line 767).

**What:** both legs actually fill at the next open, but the trade log prints the
entry as the _decision_ bar and the exit as the _fill_ bar. The Stage-1 log
therefore reads `2024-03-04 → 2024-03-05` for what is a single next-open
round-trip, which looks like a one-bar holding period that never existed.

**Why it matters:** cosmetic in P&L (both fill prices are next-open), but
misleading when auditing a trade list — the asymmetry invites a false
"off-by-one on entry" conclusion during trade-by-trade review, which is exactly
what the Stage-1 gate is for. It is engine-wide, not specific to this strategy.

**Options:** label both on the same convention (signal or fill) and say which.
Changing an engine field's meaning touches every report surface, so it needs its
own pass plus the report-metric tests.

**Status:** open, deliberately deferred. Not fixed as part of alpha research.
