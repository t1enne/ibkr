"""Download US macro time series from FRED into ``assets/``.

Writes one two-column CSV per series (``date`` YYYY-MM-DD, ``<name>`` value)
matching the exact contract of ``assets/dgs10.csv`` — the source consumed by
the lookahead-free accessors in :mod:`src.indicators.macro`. FRED no-print dates
append as **blank** cells (``''``) exactly like the yield file, so the loaders
can forward-fill them onto a daily grid unchanged.

Usage:
    export FRED_API_KEY=...
    uv run python scripts/fetch_macro_fred.py
    uv run python scripts/fetch_macro_fred.py --out assets --series GDPC1 INDPRO

The default set (below) covers the four requested categories:

:GDP:                ``GDPC1`` (Real GDP, chained 2017$, quarterly),
                     ``GDP`` (Nominal GDP, quarterly)
:Employment:         ``PAYEMS`` (All employees, total nonfarm — monthly),
                     ``UNRATE`` (unemployment rate %, monthly)
:Manufacturing/:prod:``INDPRO`` (industrial production index, monthly),
                     ``TCU`` (capacity utilization %, monthly)
:Balance of trade:   ``BOPGSTB`` (trade balance, goods & services, $millions,
                     monthly)

All downloads carry an ``observation_start=1900-01-01`` so the assets files
pipeline the longest possible history (the loader keeps only what the backtest
grid needs). HTTP lives at this I/O boundary; ``load_*`` stays pure.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx
import pandas as pd

# When run as ``uv run python scripts/fetch_macro_fred.py``, sys.path[0] is the
# scripts/ dir rather than the project root. Insert the root so ``src`` imports
# resolve when future code reuses the loader from here.
_proj_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_proj_root))

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# Default set keyed by the on-disk series name (also the CSV value column).
DEFAULT_SERIES: dict[str, dict] = {
    "gdpc1": {"id": "GDPC1", "title": "Real Gross Domestic Product"},
    "gdp": {"id": "GDP", "title": "Nominal Gross Domestic Product"},
    "payems": {"id": "PAYEMS", "title": "All Employees, Total Nonfarm"},
    "unrate": {"id": "UNRATE", "title": "Unemployment Rate"},
    "indpro": {"id": "INDPRO", "title": "Industrial Production: Total Index"},
    "tcu": {"id": "TCU", "title": "Capacity Utilization: Total Index"},
    "bopgstb": {
        "id": "BOPGSTB",
        "title": "Trade Balance: Goods and Services, Balance of Payments",
    },
    "dgs2": {"id": "DGS2", "title": "Treasury Yield 2-Year"},
    "t10y2y": {
        "id": "T10Y2Y",
        "title": "Market Yield on U.S. Treasury Securities at 10-Year, 2-Year Constant Maturity,",
    },
    "hyspread": {
        "id": "BAMLH0A0HYM2",
        "title": "ICE BofA US High Yield Index Option-Adjusted Spread",
    },
    "vix": {
        "id": "VIXCLS",
        "title": "CBOE Volatility Index: VIX",
    },
}


def _api_key() -> str:
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError(
            "FRED_API_KEY is not set. Export it, e.g. `export FRED_API_KEY=...`."
        )
    return key


def fetch_observations(series_id: str, api_key: str) -> pd.DataFrame:
    """Pull FRED observations for ``series_id`` as a (date, value) frame.

    Returns a DataFrame with ``date`` (datetime) and ``value`` (float or NaN
    for FRED blank/no-print cells). The raw ``value`` strings are parsed so a
    missing observation becomes ``NaN`` — the loader forward-fills these.
    """
    resp = httpx.get(
        FRED_BASE,
        params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": "1900-01-01",
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    payload = resp.json()
    if "error_message" in payload:
        raise RuntimeError(f"FRED error for {series_id}: {payload['error_message']}")

    rows = [
        (obs.get("date"), obs.get("value")) for obs in payload.get("observations", [])
    ]
    df = pd.DataFrame(rows, columns=["date", "value"])
    df["date"] = pd.to_datetime(df["date"])
    # FRED emits "." as the value string for no-print cells -> NaN.
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def write_asset(df: pd.DataFrame, name: str, out_dir: Path) -> Path:
    """Write ``df`` to ``out_dir/<name>.csv`` in the dgs10 two-column contract.

    FRED may return duplicate dates (two releases on one day); keep the last,
    and drop fully-blank rows so the file is pure ``date,<name>`` pairs with
    occasional blank value cells for no-print dates.
    """
    df = df.drop_duplicates(subset="date", keep="last").sort_values("date")
    df = df.dropna(subset=["date"])
    path = out_dir / f"{name}.csv"
    # Write the value column with empty strings for no-print dates (dgs10-style).
    csv = df[["date", "value"]].copy()
    csv["date"] = csv["date"].dt.strftime("%Y-%m-%d")
    csv["value"] = csv["value"].where(csv["value"].notna(), "")
    csv.to_csv(path, index=False, header=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download US macro series from FRED into assets/ CSVs."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_proj_root / "assets",
        help="Output directory for <name>.csv files (default: assets/).",
    )
    parser.add_argument(
        "--series",
        nargs="+",
        default=list(DEFAULT_SERIES),
        help="Subset of series names to fetch (default: all).",
    )
    args = parser.parse_args()

    api_key = _api_key()
    args.out.mkdir(parents=True, exist_ok=True)

    for name in args.series:
        meta = DEFAULT_SERIES.get(name)
        if meta is None:
            print(f"  !! unknown series {name!r}; skipping")
            continue
        df = fetch_observations(meta["id"], api_key)
        path = write_asset(df, name, args.out)
        n = int(df["value"].notna().sum())
        print(
            f"  ✓ {name:8s}  <-  {meta['id']:5s}  {meta['title'][:50]:52s}"
            f"  [{n} observations] -> {path}"
        )


if __name__ == "__main__":
    main()
