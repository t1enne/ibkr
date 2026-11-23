"""Per-cohort series + difference t-stats from `bt run -F json` equity curves.

`bt` reports Sharpe / annual return / drawdown, but not a per-cohort series, so
the gate's cohort-level difference and its t-stat have to be derived from the
report's own equity curve (the published artifact, not a re-derived return).

A cohort is one engine rebalance period: ``rebalance_bars`` (21) consecutive
bars of the trade window, starting at the first trade bar -- the same clock the
strategy uses. Series are aligned on the dates the three universes share, so a
cohort is the SAME window in every arm and the difference is paired.

Two bases are printed:

* ``portfolio bp`` -- the cohort's return on equity (6 x 10% positions = 60%
  gross), which is what `bt`'s annual return compounds.
* ``spread bp`` -- portfolio bp / 0.30, i.e. the long-leg-minus-short-leg
  spread on 100% notional legs. This is the basis §6's +381bp / +135bp /
  +246bp figures are stated in, so it is the one the 150bp bar is read on.

Usage: uv run python scripts/ark_gate_cohorts.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path("/tmp/gate/runs")
ERAS = {"2021": 12, "2022_2026": 51}
REBALANCE_BARS = 21
GROSS_FRACTION = 0.30  # 3 longs x 10% (or 3 shorts x 10%) = one leg's notional


def curve(name: str) -> pd.Series:
    d = json.loads((RUNS / f"{name}.json").read_text())
    return pd.Series(
        [p["equity"] for p in d["equity_curve"]],
        index=pd.to_datetime([p["ts"] for p in d["equity_curve"]]),
    )


def cohorts(eq: pd.Series) -> np.ndarray:
    n = len(eq) // REBALANCE_BARS
    v = eq.to_numpy()
    return np.array(
        [
            v[(k + 1) * REBALANCE_BARS - 1] / v[k * REBALANCE_BARS - 1] - 1.0
            for k in range(1, n)
        ]
    )


def stats(x: np.ndarray) -> str:
    n = len(x)
    m, sd = float(np.mean(x)), float(np.std(x, ddof=1))
    t = m / (sd / np.sqrt(n))
    return f"n={n:3d} mean={m * 1e4:+8.1f}bp sd={sd * 1e4:7.1f}bp t={t:+5.2f}"


def main() -> None:
    for era in ("2021", "2022_2026"):
        arm = {
            u: curve(f"ark_gate_{u}_{era}") for u in ("ark", "twins", "wide", "arkfull")
        }
        idx = arm["ark"].index
        for s in arm.values():
            idx = idx.intersection(s.index)
        arm = {u: s.loc[idx] for u, s in arm.items()}
        coh = {u: cohorts(s) for u, s in arm.items()}
        n = min(len(v) for v in coh.values())
        coh = {u: v[:n] for u, v in coh.items()}
        print(
            f"\n===== era {era} — {n} cohorts of {REBALANCE_BARS} bars, {len(idx)} shared bars"
        )
        for u, v in coh.items():
            print(f"  {u:8s} portfolio {stats(v)}   spread {stats(v / GROSS_FRACTION)}")
        for u in ("twins", "wide", "arkfull"):
            d = coh["ark"] - coh[u]
            print(
                f"  ARK - {u:7s} portfolio {stats(d)}   spread {stats(d / GROSS_FRACTION)}"
            )
        pairs = pd.DataFrame(coh)
        print("  per-cohort portfolio bp (ark/twins/wide) first 12:")
        print((pairs[["ark", "twins", "wide"]].head(12) * 1e4).round(1).to_string())
        Path(RUNS / f"cohorts_{era}.csv").write_text(pairs.to_csv())


if __name__ == "__main__":
    main()
