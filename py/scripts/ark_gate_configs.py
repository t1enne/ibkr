"""Materialise the six gate configs: 3 universes x 2 eras, identical otherwise.

Only ``symbols`` (and the era window) differ; ``strategy_params``, commission,
capital, bars and warmup are byte-identical across all six so the comparison
isolates the universe.

Usage: uv run python scripts/ark_gate_universes.py && uv run python scripts/ark_gate_configs.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNI = ROOT / "universes"
OUT = ROOT / "strats" / "wip"

ERAS: dict[str, dict[str, str]] = {
    "2021": {"trading_start": "2021-01-01", "trading_end": "2021-12-31"},
    "2022_2026": {"trading_start": "2022-01-01", "trading_end": "2026-04-30"},
}
UNIVERSES: tuple[str, ...] = ("ark", "twins", "wide", "arkfull")
STRATEGY_PARAMS: dict[str, float | int] = {
    "lookback_bars": 126,
    "rebalance_bars": 21,
    "k": 3,
    "size": 0.1,
}


def main() -> None:
    for era, window in ERAS.items():
        for uni in UNIVERSES:
            symbols = json.loads((UNI / f"ark_gate_{uni}_{era}.json").read_text())[
                "symbols"
            ]
            name = f"ark_gate_{uni}_{era}"
            cfg = {
                "name": name,
                "warmup": "300d",
                "trading_start": window["trading_start"],
                "trading_end": window["trading_end"],
                "commission": 0.05,
                "initial_capital": 100000,
                "strategy_type": "ark_ls_momentum_dsl",
                "bars": ["1d"],
                "strategy_params": STRATEGY_PARAMS,
                "symbols": symbols,
            }
            (OUT / f"{name}.json").write_text(json.dumps(cfg, indent=2) + "\n")
            print(f"{name}: {len(symbols)} symbols")


if __name__ == "__main__":
    main()
