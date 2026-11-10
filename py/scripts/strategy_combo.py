"""Blend N ``ibkr bt run -F json`` equity curves into one 50/50-style portfolio.

Why this exists: ``bt`` models ONE strategy per run (one ``StrategyConfig``, one
shared cash account), so a multi-strategy allocation has no report of its own.
This script is the outer combiner only — it takes each run's OWN reported equity
curve, forms daily returns, and reports the blend. It does not re-derive or
filter any single run's metrics, and it REPRINTS each input's reported metrics
alongside the blend so both are visible.

Method (stated, not implied):

* Each leg's curve is first resampled to one point per trading DAY (last value
  of the day), because a leg's curve carries its own base-bar grid (a ``1h``
  strategy reports hourly points). Without this step two legs on different base
  bars intersect to almost nothing.
* Each leg's daily return is ``equity_curve.pct_change()``, exactly as the run
  reported it.
* The blend at weight ``w`` is ``sum(w_i * r_i)`` — daily rebalanced to the
  target weights, each leg normalised to its own return series so a leg's
  absolute capital level does not matter.
* The blended row (SR/Ann/Vol/MaxDD/Tot) is THIS script's arithmetic, not a
  ``bt`` report. Annualisation uses 252 trading days.

Usage:
    uv run python scripts/strategy_combo.py long.json short.json [--weights 0.2,0.3,0.4,0.5]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_run(path: Path) -> tuple[str, dict[str, Any], pd.Series]:
    """``(label, reported_metrics, equity_curve)`` from a ``bt run -F json`` file."""
    payload: dict[str, Any] = json.loads(path.read_text())
    curve = payload["equity_curve"]
    points = curve if isinstance(curve, list) else curve["equity"]
    series = pd.Series(
        {pd.Timestamp(p["ts"]): float(p["equity"]) for p in points}
    ).sort_index()
    label = payload.get("config", {}).get("name") or path.stem
    return str(label), dict(payload.get("metrics", {})), series


def _to_daily(series: pd.Series) -> pd.Series:
    """One point per calendar day (last value), so legs on different base bars align.

    Empty days become NaN and the caller's ``dropna`` removes them, so the blend
    still intersects the two legs' real trading days.
    """
    return series.resample("1D").last().dropna()


def stats(returns: pd.Series) -> dict[str, float]:
    """Sharpe / annualised return / vol / max drawdown / total from a return series."""
    eq = (1.0 + returns).cumprod()
    ann = float(eq.iloc[-1] ** (252.0 / len(returns)) - 1.0)
    vol = float(returns.std() * np.sqrt(252.0))
    dd = float((eq / eq.cummax() - 1.0).min())
    sr = float(returns.mean() / returns.std() * np.sqrt(252.0)) if vol > 0 else 0.0
    return {
        "SR": sr,
        "Ann": ann,
        "Vol": vol,
        "MaxDD": dd,
        "Tot": float(eq.iloc[-1] - 1.0),
        "Kurt": float(returns.kurtosis()),
        "Skew": float(returns.skew()),
    }


def row(label: str, s: dict[str, float]) -> str:
    return (
        f"{label:<26s} SR {s['SR']:6.2f}  Ann {s['Ann']:8.2%}  "
        f"Vol {s['Vol']:7.2%}  MaxDD {s['MaxDD']:8.2%}  Tot {s['Tot']:9.1%}  "
        f"Kurt {s['Kurt']:5.1f}  Skew {s['Skew']:5.2f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--weights", default="0.2,0.3,0.4,0.5,0.6")
    args = ap.parse_args()
    if len(args.runs) != 2:
        raise SystemExit("this combiner takes exactly two runs (long, short)")

    (l_name, l_metrics, l_eq) = load_run(args.runs[0])
    (s_name, s_metrics, s_eq) = load_run(args.runs[1])
    l_eq, s_eq = _to_daily(l_eq), _to_daily(s_eq)
    frame = pd.concat({"long": l_eq, "short": s_eq}, axis=1, sort=True).dropna()
    rl = frame["long"].pct_change().dropna()
    rs = frame["short"].pct_change().dropna()

    print(
        f"common window: {frame.index.min().date()} -> {frame.index.max().date()}"
        f"  ({len(frame)} bars)"
    )
    print("\nreported by bt (each run's own report):")
    for name, m in ((l_name, l_metrics), (s_name, s_metrics)):
        print(
            f"  {name:<24s} SR {m.get('sharpe_ratio', float('nan')):6.2f}  "
            f"Ann {m.get('annual_return', float('nan')):8.2%}  "
            f"MaxDD {m.get('max_drawdown', float('nan')):8.2%}  "
            f"beta {m.get('beta', float('nan')):5.2f}"
        )

    print(
        "\ndaily-return correlation long/short: "
        f"{float(np.corrcoef(rl, rs)[0, 1]):+.3f}"
    )
    print("\ncombiner arithmetic (this script, daily rebalanced):")
    print(row("long 100%", stats(rl)))
    print(row("short 100%", stats(rs)))
    for w in (float(x) for x in args.weights.split(",")):
        print(row(f"{w:.0%} long / {1 - w:.0%} short", stats(w * rl + (1 - w) * rs)))

    print("\nper year (rho, each leg SR, 50/50 SR):")
    for y in sorted(set(rl.index.year)):
        a, b = rl[rl.index.year == y], rs[rs.index.year == y]
        rho = float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan")
        print(
            f"  {y}  rho {rho:+6.2f}   long {stats(a)['SR']:6.2f}   "
            f"short {stats(b)['SR']:6.2f}   50/50 {stats(0.5 * a + 0.5 * b)['SR']:6.2f}"
        )


if __name__ == "__main__":
    main()
