"""Rendering of backtest results into JSON-shaped output.

This is the output boundary. It consumes ``BacktestResults`` / ``PortfolioResult``
/ ``Trade`` objects directly and returns JSON-serializable structures — it does
NOT define a backtest_results→dict intermediate that flows through the engine.
Engine logic deals in ``BacktestResults``; text and JSON are both produced from
the same objects when output is actually emitted.

The helpers return plain dicts/points that already carry JSON-native scalars
(floats, strings); pandas Timestamps and Enums are normalized here, so callers
pass ``default=_json_default`` only as a safety net.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Callable, Iterable

import pandas as pd

logger = logging.getLogger(__name__)


def _ts_str(v: Any) -> str | None:
    """Format a timestamp-like value as an ISO string, else None."""
    if v is None:
        return None
    if isinstance(v, pd.Timestamp):
        return str(v)
    if hasattr(v, "strftime"):
        return v.strftime("%Y-%m-%d %H:%M:%S")
    return str(v)


def _metric_pairs(pf: Any) -> Iterable[tuple[str, float]]:
    """Yield (name, float(value)) for every scalar PortfolioResult metric."""
    from dataclasses import fields

    skip = {"trades", "equity_curve"}
    for f in fields(pf):
        if f.name in skip:
            continue
        val = getattr(pf, f.name)
        if isinstance(val, bool):  # never a PortfolioResult metric, but be safe
            continue
        if isinstance(val, (int, float)):
            yield f.name, float(val)
        else:
            yield f.name, str(val)


def trade_json(trade: Any) -> dict[str, Any]:
    """One Trade -> JSON-ready dict (scalars already flattened)."""
    return {
        "position_id": getattr(trade, "position_id", ""),
        "symbol": trade.symbol,
        "position": getattr(trade.position, "value", trade.position),
        "qty": float(trade.qty),
        "entry_time": _ts_str(trade.entry_time),
        "entry_price": float(trade.entry_price),
        "exit_time": _ts_str(trade.exit_time),
        "exit_price": float(trade.exit_price) if trade.exit_price is not None else None,
        "last_price": float(trade.last_price),
        "stop_loss": float(trade.stop_loss) if trade.stop_loss is not None else None,
        "take_profit": float(trade.take_profit)
        if trade.take_profit is not None
        else None,
        "pnl": float(trade.pnl),
        "commission": float(trade.commission),
        "slippage": float(trade.slippage),
        "status": getattr(trade.status, "value", trade.status),
        "close_reason": (
            getattr(trade.close_reason, "value", trade.close_reason)
            if trade.close_reason is not None
            else None
        ),
        "reason": str(trade.reason) if trade.reason is not None else None,
    }


def equity_points(equity_curve: pd.Series) -> list[dict[str, Any]]:
    """Equity curve -> [{"ts": iso, "equity": float}, ...]."""
    out: list[dict[str, Any]] = []
    for ts, val in equity_curve.items():
        out.append({"ts": _ts_str(ts) or str(ts), "equity": float(val)})
    return out


def benchmark_json(benchmark_curves: dict[str, pd.Series]) -> dict[str, Any]:
    """Benchmark curves -> {symbol: [{"ts": iso, "equity": float}, ...]}."""
    return {sym: equity_points(curve) for sym, curve in benchmark_curves.items()}


def render_result_json(results: Any, *, include_trades: bool = True) -> dict[str, Any]:
    """BacktestResults -> one JSON-ready dict.

    Always carries a top-level ``total_trades``; the ``trades`` list itself is
    emitted only when ``include_trades`` is True (default preserves the full
    payload for existing callers).
    """
    out: dict[str, Any] = {
        "metrics": dict(_metric_pairs(results.pf)),
        "total_trades": len(results.pf.trades),
    }
    if include_trades:
        out["trades"] = [trade_json(t) for t in results.pf.trades]
    out["equity_curve"] = equity_points(results.pf.equity_curve)
    out["benchmark_curves"] = benchmark_json(results.benchmark_curves)
    return out


def _frame_interval(results: Any, symbol: str, entry_ts: Any) -> str | None:
    """Resolve which candle frame a trade belongs to for the given symbol.

    A trade's entry/exit timestamps are exact index points in the signal
    interval's frame (the interval that triggered the fill). When multiple
    intervals exist for a symbol (base + HTF read via the CandleStore), pick
    the frame whose index actually contains the entry timestamp; otherwise
    fall back to the symbol's first frame.
    """
    data = results.data
    frames: list[tuple[str, pd.DataFrame]] = [
        (iv, data[(sym, iv)]) for (sym, iv) in data.keys() if sym == symbol
    ]
    if not frames:
        return None
    for iv, df in frames:
        if entry_ts in df.index:
            return iv
    return frames[0][0]


def render_plot_json(
    results: Any, *, plot: bool = True, include_trades: bool = True
) -> dict[str, Any]:
    """BacktestResults -> one JSON doc shaped for candlestick charting.

    Extends the base json shape (metrics + trades + equity) with, per symbol,
    the candle OHLCV series used by the dashboard so no DB re-query is needed
    to draw price markers. Each trade also gains its resolved ``interval``.

    When ``plot`` is True and the strategy defines a module-level ``plot``, the
    returned :class:`~src.bt.strategies.types.PlotSpec` is spliced into the
    owning symbol/frame under a ``plot`` key. Strategies without a ``plot``
    (or ``plot=False``) get a payload byte-identical to before this feature.
    """
    symbols: dict[str, list[dict[str, Any]]] = {}
    for sym, iv in results.data.keys():
        df = results.data[(sym, iv)]
        symbols.setdefault(sym, []).append(
            {
                "interval": iv,
                "bars": [
                    [
                        _ts_str(ts),
                        float(opn),
                        float(high),
                        float(low),
                        float(clse),
                        float(vol),
                    ]
                    for ts, opn, high, low, clse, vol in zip(
                        df.index,
                        df["open"],
                        df["high"],
                        df["low"],
                        df["close"],
                        df["volume"],
                    )
                ],
            }
        )
    if plot:
        _splice_plot_specs(results, symbols)
    out: dict[str, Any] = {
        "metrics": dict(_metric_pairs(results.pf)),
        "total_trades": len(results.pf.trades),
        "symbols": symbols,
    }
    if include_trades:
        trades = []
        for t in results.pf.trades:
            tj = trade_json(t)
            tj["interval"] = _frame_interval(results, t.symbol, t.entry_time)
            trades.append(tj)
        out["trades"] = trades
    out["equity_curve"] = equity_points(results.pf.equity_curve)
    out["benchmark_curves"] = benchmark_json(results.benchmark_curves)
    return out


def _splice_plot_specs(results: Any, symbols: dict[str, list[dict[str, Any]]]) -> None:
    """Attach each symbol's ``plot`` spec to its signal-interval frame in place.

    A strategy without a ``plot`` (or without a config) leaves the payload
    untouched -- the regression guarantee. A failure inside ``plot`` is logged
    and dropped: plotting must never fail an already-computed backtest.
    """
    from src.bt.strategies import plot_fn_for, resolve_params

    config = getattr(results, "config", None)
    strat_type = getattr(config, "strategy_type", None) if config is not None else None
    if strat_type is None:
        return
    fn = plot_fn_for(strat_type)
    if fn is None:
        return
    params = resolve_params(strat_type, getattr(config, "strategy_params", {}))
    bars = getattr(config, "bars", None)
    interval = bars[0] if bars else None
    for sym, frames in symbols.items():
        target = _plot_frame(frames, interval)
        if target is None:
            continue
        _call_plot(fn, results, sym, target["interval"], params, target)


def _plot_frame(
    frames: list[dict[str, Any]], interval: str | None
) -> dict[str, Any] | None:
    """The frame a plot belongs to: the signal interval, else the first frame."""
    if interval is not None:
        for frame in frames:
            if frame["interval"] == interval:
                return frame
    return frames[0] if frames else None


def _call_plot(
    fn: Callable[..., Any],
    results: Any,
    sym: str,
    interval: str,
    params: Any,
    frame: dict[str, Any],
) -> None:
    """Invoke ``plot`` under the guard and splice ``asdict(spec)`` into ``frame``."""
    from src.bt.strategies.dsl import StrategyContext

    try:
        ctx = StrategyContext.for_plot(
            results, symbol=sym, interval=interval, params=params
        )
        frame["plot"] = asdict(fn(ctx, params))
    except Exception as exc:  # plotting must never fail a computed backtest
        logger.warning("plot spec failed for %s (%s); omitting", sym, exc)


def render_result_jsonl(
    results: Any, *, include_trades: bool = True
) -> list[dict[str, Any]]:
    """BacktestResults -> JSONL-shaped list.

    One dict per equity-curve point, then a final ``metrics`` + ``total_trades``
    record (plus ``trades`` unless ``include_trades`` is False) for consumers
    that read to EOF.
    """
    points: list[dict[str, Any]] = [
        {"ts": _ts_str(ts) or str(ts), "equity": float(val)}
        for ts, val in results.pf.equity_curve.items()
    ]
    record: dict[str, Any] = {
        "metrics": dict(_metric_pairs(results.pf)),
        "total_trades": len(results.pf.trades),
    }
    if include_trades:
        record["trades"] = [trade_json(t) for t in results.pf.trades]
    points.append(record)
    return points


__all__ = [
    "trade_json",
    "equity_points",
    "benchmark_json",
    "render_result_json",
    "render_result_jsonl",
    "render_plot_json",
]
