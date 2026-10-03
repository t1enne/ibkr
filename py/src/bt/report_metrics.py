"""Canonical per-run metric formatting, shared by every ``bt`` report.

One source of truth — :data:`CANONICAL_METRICS` — drives both the text cells
(:func:`metric_cells`) and the JSON records (:func:`metric_dict`) so the sweep,
split and optimize reports (and their ``-F json`` outputs) can never drift.
The canonical set is Sharpe, Ann, MaxDD, Kurt, Skew, Win, Trades, Scaled,
Rejected (kurtosis + skewness tell the tail story; Scaled is the count of
partially-scaled entries, Rejected the count dropped for genuine cash
exhaustion — see AGENTS.md "Fills").
"""

from __future__ import annotations

from typing import Callable, Mapping, NamedTuple

from src.bt.metrics import trade_count, win_rate
from src.bt.types import PortfolioResult


class MetricSpec(NamedTuple):
    """One canonical metric: JSON key, text label, text formatter, int-ness."""

    key: str
    label: str
    fmt: Callable[[float], str]
    as_int: bool = False


# Order defines both the column order and the JSON dict key order.
CANONICAL_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("sharpe_ratio", "Sharpe", lambda v: f"{v:.2f}"),
    MetricSpec("annual_return", "Ann", lambda v: f"{v:.1%}"),
    MetricSpec("max_drawdown", "MaxDD", lambda v: f"{v:.1%}"),
    MetricSpec("kurtosis", "Kurt", lambda v: f"{v:.1f}"),
    MetricSpec("skewness", "Skew", lambda v: f"{v:.2f}"),
    MetricSpec("win_rate", "Win", lambda v: f"{v:.0%}"),
    MetricSpec("trade_count", "Trades", lambda v: str(int(v)), as_int=True),
    MetricSpec("scaled_trades", "Scaled", lambda v: str(int(v)), as_int=True),
    MetricSpec("rejected_trades", "Rejected", lambda v: str(int(v)), as_int=True),
)


def metric_value(pf: PortfolioResult, key: str) -> float:
    """Raw numeric value for ``key`` — ``win_rate``/``trade_count`` derived."""
    if key == "win_rate":
        return win_rate(pf)
    if key == "trade_count":
        return float(trade_count(pf))
    return float(getattr(pf, key))


def metric_labels() -> tuple[str, ...]:
    """Text column labels in canonical order."""
    return tuple(spec.label for spec in CANONICAL_METRICS)


def metric_cells(pf: PortfolioResult) -> tuple[str, ...]:
    """Formatted text cells for one run, in canonical order."""
    return tuple(spec.fmt(metric_value(pf, spec.key)) for spec in CANONICAL_METRICS)


def metric_cells_from_dict(metrics: Mapping[str, float]) -> tuple[str, ...]:
    """Text cells from an already-serialized :func:`metric_dict` mapping.

    Used where only the dict survives (e.g. an optimize fold's IS metrics);
    missing keys render as ``—``.
    """
    return tuple(
        spec.fmt(float(metrics[spec.key])) if spec.key in metrics else "—"
        for spec in CANONICAL_METRICS
    )


def metric_dict(pf: PortfolioResult) -> dict[str, float | int]:
    """JSON-ready per-run metric record, in canonical order.

    Same fields as :func:`metric_cells` — ints stay ints for the two counts.
    """
    out: dict[str, float | int] = {}
    for spec in CANONICAL_METRICS:
        value = metric_value(pf, spec.key)
        out[spec.key] = int(value) if spec.as_int else value
    return out


__all__ = [
    "MetricSpec",
    "CANONICAL_METRICS",
    "metric_value",
    "metric_labels",
    "metric_cells",
    "metric_cells_from_dict",
    "metric_dict",
]
