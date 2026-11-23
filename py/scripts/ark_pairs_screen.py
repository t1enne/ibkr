"""Read-only cointegration screen report for an ARK fund/constituent pairs book.

WHY this exists: the screen (OLS hedge ratio, ADF p-value, AR(1) half-life) is
the only thing a pairs report cannot show, so this prints it per pair. It fits
with the same window/rule the strategy family used, self-contained: no import
of any strategy module, so it survives that book being retired.

This is NOT a backtest harness: it computes no P&L, no metric, no strategy
result.

Alignment is carry-forward rather than a naive join — a leg that does not print
a bar on a session keeps its last close, and a pair starts on the first
timestamp where BOTH legs have a close.

Usage:
    uv run python scripts/ark_pairs_screen.py <config.json>
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

from src.bt import Backtest, load_strategy, warmup_load_start
from src.bt.data_feed import load_candles

FORMATION_BARS = 252
ADF_P_MAX = 0.10
HL_MIN = 1.0
HL_MAX = 60.0


@dataclass(frozen=True)
class Formation:
    """One pair's frozen formation fit; never updated after the window closes."""

    alpha: float
    beta: float
    mu: float
    sd: float
    adf_p: float
    half_life: float

    @property
    def tradeable(self) -> bool:
        """Cointegrated AND reverting inside the band AND z-scorable."""
        return bool(
            math.isfinite(self.adf_p)
            and self.adf_p < ADF_P_MAX
            and math.isfinite(self.half_life)
            and HL_MIN <= self.half_life <= HL_MAX
            and self.sd > 0.0
        )


def _half_life(spread: np.ndarray) -> float:
    """Mean-reversion half-life in bars from an AR(1) on the spread.

    ``phi >= 1`` does not revert (``inf``); ``phi <= 0`` is the alternating
    limit (``0.0``) and fails ``HL_MIN`` instead of being clamped into range.
    """
    if spread.shape[0] < 3:
        return math.inf
    phi = float(sm.OLS(spread[1:], sm.add_constant(spread[:-1])).fit().params[1])
    if phi >= 1.0:
        return math.inf
    if phi <= 0.0:
        return 0.0
    return -math.log(2.0) / math.log(phi)


def fit_formation(fund: Sequence[float], cons: Sequence[float]) -> Formation:
    """OLS hedge ratio on logs, then ADF p and AR(1) half-life of the spread."""
    if len(fund) != FORMATION_BARS or len(cons) != FORMATION_BARS:
        raise ValueError(
            f"formation needs {FORMATION_BARS} paired closes, got {len(fund)}/{len(cons)}"
        )
    # Repo OLS path: ln(fund) ~ ln(cons) with an intercept => params (alpha, beta).
    from src.utils import _get_ols_fit_model

    alpha, beta = (float(p) for p in _get_ols_fit_model(fund, cons).params)
    spread = (
        np.log(np.asarray(fund, dtype=np.float64))
        - beta * np.log(np.asarray(cons, dtype=np.float64))
        - alpha
    )
    return Formation(
        alpha=alpha,
        beta=beta,
        mu=float(np.mean(spread)),
        sd=float(np.std(spread, ddof=1)),
        adf_p=float(adfuller(spread, regression="c", autolag="AIC")[1]),
        half_life=_half_life(spread),
    )


def _pair_history(df: pd.DataFrame, fund: str, cons: str) -> pd.DataFrame:
    """Both legs' closes on the feed's own timestamp grid (carried forward)."""
    closes = pd.DataFrame(
        {
            "fund": df[(fund, "close")].astype(float),
            "cons": df[(cons, "close")].astype(float),
        }
    )
    return closes.ffill()


def _screen(history: pd.DataFrame) -> tuple[Formation | None, str]:
    """Fit on the first ``FORMATION_BARS`` paired closes, or report no fit."""
    usable = history.dropna()
    if len(usable) < FORMATION_BARS:
        return None, ""
    window = usable.iloc[:FORMATION_BARS]
    fit = fit_formation(window["fund"].to_list(), window["cons"].to_list())
    return fit, f"{window.index[0].date()}..{window.index[-1].date()}"


def main(config_path: str) -> int:
    cfg = load_strategy(config_path)
    bt = Backtest(cfg)
    df = load_candles(
        cfg.symbols,
        warmup_load_start(cfg, bt.window.test_start),
        bt.window.test_end,
        cfg.bars[0],
    )

    print(f"cointegration screen — {config_path}")
    print(
        f"formation bars {FORMATION_BARS} · ADF p < {ADF_P_MAX} · "
        f"half-life in [{HL_MIN:g}, {HL_MAX:g}] bars · "
        f"feed {df.index[0].date()}..{df.index[-1].date()}"
    )
    header = (
        f"{'#':>3} {'pair':<14} {'verdict':<11} {'adf p':>7} {'half-life':>9} "
        f"{'beta':>7} {'alpha':>8} {'mu':>8} {'sd':>7} {'formed':<23} bars"
    )
    print(header)
    print("-" * len(header))

    passed = rejected = unformed = 0
    for i, spec in enumerate(cfg.strategy_params["pairs"], start=1):
        pair = spec if isinstance(spec, str) else "/".join(spec)
        fund, _, cons = pair.partition("/")
        fit, formed = _screen(_pair_history(df, fund, cons))
        if fit is None:
            unformed += 1
            print(
                f"{i:>3} {pair:<14} {'no history':<11} {'-':>7} {'-':>9} "
                f"{'-':>7} {'-':>8} {'-':>8} {'-':>7} {'-':<23} <{FORMATION_BARS}"
            )
            continue
        verdict = "PASS" if fit.tradeable else "screened"
        if fit.tradeable:
            passed += 1
        else:
            rejected += 1
        print(
            f"{i:>3} {pair:<14} {verdict:<11} {fit.adf_p:>7.4f} "
            f"{fit.half_life:>9.1f} {fit.beta:>7.3f} {fit.alpha:>8.3f} "
            f"{fit.mu:>8.4f} {fit.sd:>7.4f} {formed:<23} {FORMATION_BARS}"
        )

    print("-" * len(header))
    print(
        f"tradeable {passed} · screened out {rejected} · "
        f"never formed {unformed} · total {passed + rejected + unformed}"
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: uv run python scripts/ark_pairs_screen.py <config.json>"
        )
    raise SystemExit(main(sys.argv[1]))
