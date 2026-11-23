"""Build the three universe families for the ARK §6 control gate.

Prep only — no backtest runs here. Reads ``assets/ark_flow.csv`` (point-in-time
ARK membership via per-ticker first/last flow date) and the candle DB (daily
bars aggregated in SQL from 1h), then writes six universe files under
``universes/`` plus ``ark_gate_summary.json`` with every filter count.

Universe rules (all applied identically per era):

* ARK sleeve — ticker appears in ``ark_flow.csv``; its flow-active window
  intersects the era; >=50% of its own flow dates fall inside its candle span
  (kills ticker reuse such as ``SI``); >=100 daily bars inside the era; 60 bars
  of history before the era start (for the match); no daily-bar gap > 4 days in
  [era start - 300d warmup, era end] (the loader's own integrity guard).
* Twin pool — US-listed (``market='SMART'``, ``currency='USD'``) DB names that
  never appear in ``ark_flow.csv``, are not ETF-like, and pass the same
  gap/tradability screens.
* Twins — exact (trailing-60d dollar-volume decile, price decile) cell match at
  era start, 1:1, alphabetically paired inside each cell. An ARK name whose
  cell has no free pool name is DROPPED FROM BOTH SLEEVES so the two sleeves
  keep the same size and the same selection intensity.
* Wide control — the whole twin pool (ETF + ARK excluded), same era screens.

Usage: uv run python scripts/ark_gate_universes.py
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT.parent / "data" / "ibkr.db"
FLOWS = ROOT / "assets" / "ark_flow.csv"
UNI = ROOT / "universes"
OUT = UNI / "ark_gate_summary.json"

ERAS: dict[str, tuple[str, str]] = {
    "2021": ("2021-01-01", "2021-12-31"),
    "2022_2026": ("2022-01-01", "2026-04-30"),
}
WARMUP = pd.Timedelta(days=300)
MAX_GAP_DAYS = 4  # mirrors src.bt.data_feed.DEFAULT_MAX_GAP (96h)
ARK_ETFS = {"ARKK", "ARKG", "ARKQ", "ARKW", "ARKF", "ARKX", "PRNT", "IZRL"}
# Fund/trust/note vehicle words. The first pass only matched "ETF"/issuer
# names and let CPER, UNG, USO, VXX and the XLB..XLY sector SPDRs into the
# control pool -- their stored names are truncated ("SS MATERIALS SELECT
# SECTOR", "US NATURAL GAS FUND LP", "IPATH SERIES B S&P 500 VIX"), so the
# expanded token set below is what actually keeps the control stock-only.
ETF_NAME_TOKENS = (
    "ETF",
    "ETP",
    "ETN",
    "FUND",
    "TRUST",
    "ISHARES",
    "SPDR",
    "INVESCO",
    "WISDOMTREE",
    "PROSHARES",
    "VANGUARD",
    "UCITS",
    "SELECT SECTOR",
    "INDEX",
    "VIX",
    "SERIES B",
)


@dataclass
class Ticker:
    """One DB ticker with the facts the universe rules need."""

    ticker: str
    name: str
    market: str
    currency: str
    days: pd.DataFrame
    flow_dates: list[pd.Timestamp] = field(default_factory=list)

    @property
    def first(self) -> pd.Timestamp:
        return self.days.index[0]

    @property
    def last(self) -> pd.Timestamp:
        return self.days.index[-1]


def load_daily() -> tuple[dict[str, Ticker], dict[str, str]]:
    """Per-ticker daily close + dollar volume straight from SQL (no 1h loads)."""
    con = sqlite3.connect(DB)
    meta = {
        str(t).upper(): (str(m), str(c), str(n or ""))
        for t, m, c, n in con.execute(
            "select ticker, market, currency, name from symbol"
        )
    }
    q = """
        select ticker, d, dv, close from (
          select ticker,
                 date(timestamp/1000, 'unixepoch') as d,
                 sum(close*volume) as dv,
                 close,
                 row_number() over (
                   partition by ticker, date(timestamp/1000, 'unixepoch')
                   order by timestamp desc) as rn
          from candle group by ticker, d, timestamp
        ) where rn = 1 order by ticker, d
    """
    df = pd.read_sql_query(q, con)
    con.close()
    df["ticker"] = df["ticker"].str.upper()
    df["d"] = pd.to_datetime(df["d"])
    out: dict[str, Ticker] = {}
    names: dict[str, str] = {}
    for tick, g in df.groupby("ticker", sort=False):
        m, c, nm = meta.get(str(tick), ("", "", ""))
        names[str(tick)] = nm
        out[str(tick)] = Ticker(
            ticker=str(tick),
            name=nm,
            market=m,
            currency=c,
            days=g.set_index("d").sort_index()[["dv", "close"]].astype(float),
        )
    return out, names


def load_flows() -> dict[str, list[pd.Timestamp]]:
    df = pd.read_csv(FLOWS, usecols=["date", "ticker"])
    df["ticker"] = df["ticker"].str.upper()
    df["date"] = pd.to_datetime(df["date"])
    return {str(t): sorted(set(g["date"])) for t, g in df.groupby("ticker", sort=False)}


def gap_clean(t: Ticker, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    """True when no consecutive daily bars in the window are > MAX_GAP_DAYS apart."""
    s = t.days[(t.days.index >= start) & (t.days.index <= end)]
    if len(s) < 2:
        return False
    deltas = np.diff(s.index.to_numpy()).astype("timedelta64[D]").astype(int)
    return bool(deltas.max() <= MAX_GAP_DAYS)


def era_days(t: Ticker, start: pd.Timestamp, end: pd.Timestamp) -> int:
    return int(len(t.days[(t.days.index >= start) & (t.days.index <= end)]))


def stats_at(t: Ticker, start: pd.Timestamp) -> tuple[float, float] | None:
    """(median trailing-60d dollar volume, last close) using bars before ``start``."""
    hist = t.days[t.days.index < start]
    if len(hist) < 60:
        return None
    return float(np.median(hist["dv"].iloc[-60:])), float(hist["close"].iloc[-1])


def decile(value: float, cuts: list[float]) -> int:
    return int(np.searchsorted(cuts, value, side="right"))


def main() -> None:
    daily, names = load_daily()
    flows = load_flows()
    etf_like = sorted(
        t
        for t, v in daily.items()
        if t in ARK_ETFS or any(tok in v.name.upper() for tok in ETF_NAME_TOKENS)
    )
    summary: dict[str, object] = {
        "db_tickers": len(daily),
        "flow_file_tickers": len(flows),
        "etf_like_excluded": len(etf_like),
        "excluded_ticker_reuse": [],
        "excluded_span_overlap": [],
        "eras": {},
    }

    # ---- ARK eligibility on flow/candle facts (era-independent part) --------
    ark_ok: dict[str, Ticker] = {}
    for tick, fdates in flows.items():
        t = daily.get(tick)
        if t is None or tick in ARK_ETFS:  # the funds themselves are never holdings
            continue
        inside = sum(1 for d in fdates if t.first <= d <= t.last)
        if inside / len(fdates) <= 0.5:
            entry = {
                "ticker": tick,
                "flow_span": [str(fdates[0].date()), str(fdates[-1].date())],
                "candle_span": [str(t.first.date()), str(t.last.date())],
                "flow_dates_inside_span": inside,
                "flow_dates": len(fdates),
            }
            if t.first > fdates[-1]:
                summary["excluded_ticker_reuse"].append(entry)  # type: ignore[union-attr]
            else:
                summary["excluded_span_overlap"].append(entry)  # type: ignore[union-attr]
            continue
        t.flow_dates = fdates
        ark_ok[tick] = t

    universes: dict[str, list[str]] = {}
    for era, (start_s, end_s) in ERAS.items():
        start, end = pd.Timestamp(start_s), pd.Timestamp(end_s)
        load_start = start - WARMUP

        # ARK arm -------------------------------------------------------------
        ark_member = [
            t
            for t, v in ark_ok.items()
            if v.flow_dates[0] <= end and v.flow_dates[-1] >= start
        ]
        ark_at_start = [
            t for t, v in ark_ok.items() if v.flow_dates[0] <= start <= v.flow_dates[-1]
        ]
        ark_screened = sorted(
            t
            for t in ark_member
            if era_days(ark_ok[t], start, end) >= 100
            and gap_clean(ark_ok[t], load_start, end)
            and stats_at(ark_ok[t], start) is not None
        )

        # Twin pool / wide control --------------------------------------------
        pool = sorted(
            t
            for t, v in daily.items()
            if v.market == "SMART"
            and v.currency == "USD"
            and t not in flows
            and t not in etf_like
            and era_days(v, start, end) >= 100
            and gap_clean(v, load_start, end)
            and stats_at(v, start) is not None
        )
        stats_pool = {t: stats_at(daily[t], start) for t in pool}
        stats_ark = {t: stats_at(ark_ok[t], start) for t in ark_screened}
        assert all(v is not None for v in stats_pool.values())
        assert all(v is not None for v in stats_ark.values())
        dv_cuts = [
            float(x)
            for x in np.quantile(
                [stats_pool[t][0] for t in pool], np.arange(1, 10) / 10
            )
        ]  # type: ignore[index]
        px_cuts = [
            float(x)
            for x in np.quantile(
                [stats_pool[t][1] for t in pool], np.arange(1, 10) / 10
            )
        ]  # type: ignore[index]

        cells: dict[tuple[int, int], list[str]] = {}
        for t in pool:
            cells.setdefault(
                (decile(stats_pool[t][0], dv_cuts), decile(stats_pool[t][1], px_cuts)),
                [],
            ).append(t)  # type: ignore[index]
        ark_cells: dict[tuple[int, int], list[str]] = {}
        for t in ark_screened:
            ark_cells.setdefault(
                (decile(stats_ark[t][0], dv_cuts), decile(stats_ark[t][1], px_cuts)), []
            ).append(t)  # type: ignore[index]

        matched_ark: list[str] = []
        matched_twins: list[str] = []
        unmatched: list[str] = []
        for cell, members in sorted(ark_cells.items()):
            cands = sorted(cells.get(cell, []))
            n = min(len(members), len(cands))
            matched_ark += sorted(members)[:n]
            matched_twins += cands[:n]
            unmatched += sorted(members)[n:]

        universes[f"ark_gate_ark_{era}"] = sorted(matched_ark)
        universes[f"ark_gate_twins_{era}"] = sorted(matched_twins)
        universes[f"ark_gate_wide_{era}"] = pool
        universes[f"ark_gate_arkfull_{era}"] = ark_screened
        summary["eras"][era] = {
            "window": [start_s, end_s],
            "ark_flow_active_in_era": len(ark_member),
            "ark_member_at_era_start": len(ark_at_start),
            "ark_screened": len(ark_screened),
            "ark_matched_sleeve": len(matched_ark),
            "ark_unmatched_dropped": len(unmatched),
            "ark_unmatched_names": sorted(unmatched),
            "ark_sleeve_members": sorted(matched_ark),
            "ark_full_sleeve": ark_screened,
            "twins": len(matched_twins),
            "twin_pairs": sorted(zip(sorted(matched_ark), matched_twins)),
            "pool_wide": len(pool),
            "dv_decile_cuts": dv_cuts,
            "px_decile_cuts": px_cuts,
        }

    UNI.mkdir(exist_ok=True)
    for name, syms in universes.items():
        (UNI / f"{name}.json").write_text(
            json.dumps({"name": name, "symbols": syms}, indent=2) + "\n"
        )
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    slim = {
        "db_tickers": summary["db_tickers"],
        "flow_file_tickers": summary["flow_file_tickers"],
        "etf_like_excluded": summary["etf_like_excluded"],
        "excluded_ticker_reuse": summary["excluded_ticker_reuse"],
        "excluded_span_overlap": summary["excluded_span_overlap"],
        "eras": {
            e: {
                k: v
                for k, v in d.items()
                if k
                not in (
                    "ark_sleeve_members",
                    "ark_full_sleeve",
                    "ark_unmatched_names",
                    "twin_pairs",
                    "dv_decile_cuts",
                    "px_decile_cuts",
                )
            }
            for e, d in summary["eras"].items()  # type: ignore[union-attr]
        },
    }
    leaked = sorted(
        t
        for k, syms in universes.items()
        if k.startswith("ark_gate_wide")
        for t in syms
        if any(tok in daily[t].name.upper() for tok in ETF_NAME_TOKENS)
    )
    assert not leaked, f"vehicle leaked into the control pool: {leaked}"
    print(json.dumps(slim, indent=2))


if __name__ == "__main__":
    main()
