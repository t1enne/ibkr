"""Pure rendering helpers for the backtest dashboard.

Kept free of Streamlit so the chart/table builders stay testable in isolation.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

# Marker glyphs:
#   open long  -> green up-triangle   (▲)     close long  -> red flag   (⚑)
#   open short -> red down-triangle   (▼)     close short -> green flag (⚑)
# Candle up/down theme is grey (light vs dark); markers stay colored.
_GREEN = "#1fae54"
_RED = "#e64545"
_GREY_LIGHT = "#cfd3d5"  # rising candles
_GREY_DARK = "#444b52"  # falling candles

_METRIC_KEYS: tuple[str, ...] = (
    "total_return",
    "annual_return",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "annual_volatility",
)

_PCT_KEYS = frozenset({"total_return", "annual_return", "max_drawdown"})

_TRADE_COLS: tuple[str, ...] = (
    "symbol",
    "position",
    "qty",
    "entry_time",
    "entry_price",
    "exit_time",
    "exit_price",
    "pnl",
    "commission",
    "slippage",
    "close_reason",
    "status",
    "reason",
    "interval",
)


def fmt_metric(key: str, value: Any) -> str:
    """Format a metric value for display (percent for return/drawdown keys)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if key in _PCT_KEYS:
            return f"{value:.2%}"
        return f"{value:.3f}"
    return str(value)


def metrics_table(payload: dict[str, Any]) -> list[tuple[str, str]]:
    """Ordered (label, formatted value) pairs for the headline metrics."""
    metrics = payload.get("metrics", {})
    return [(key, fmt_metric(key, metrics.get(key))) for key in _METRIC_KEYS]


def trades_table(payload: dict[str, Any]) -> pd.DataFrame:
    """All trades as a display DataFrame (unfiltered by symbol)."""
    rows = payload.get("trades", [])
    present = [c for c in _TRADE_COLS if rows and c in rows[0]]
    return pd.DataFrame(rows, columns=present)


def price_frame(frame: dict[str, Any]) -> pd.DataFrame:
    """Timestamp-indexed OHLCV frame from one symbol's bars."""
    price = pd.DataFrame(
        frame["bars"],
        columns=["ts", "open", "high", "low", "close", "volume"],
    )
    price["ts"] = pd.to_datetime(price["ts"])
    return price.sort_values("ts").set_index("ts")


def trades_for_frame(
    payload: dict[str, Any], symbol: str, interval: str
) -> pd.DataFrame:
    """Trades for one symbol/interval pairing."""
    cols = [
        "symbol",
        "position",
        "qty",
        "entry_time",
        "entry_price",
        "exit_time",
        "exit_price",
    ]
    rows = [
        t
        for t in payload.get("trades", [])
        if t.get("symbol") == symbol and t.get("interval") == interval
    ]
    return pd.DataFrame(rows, columns=cols)


def build_chart(
    price: pd.DataFrame, trades: pd.DataFrame, plot: dict[str, Any] | None = None
) -> go.Figure:
    """OHLC candles + open/close trade glyphs for one symbol/interval.

    ``plot`` is an optional declarative spec (as serialized by
    ``output.render_plot_json``): overlays, markers, legs and oscillator panels.
    ``None`` reproduces today's bare-``go.Figure`` layout exactly, so every
    existing call site and test is unchanged.
    """
    panels = (plot or {}).get("panels") or []
    if panels:
        return _build_with_panels(price, trades, plot, panels)
    fig = go.Figure()
    _add_candles(fig, price)
    _add_open_markers(fig, trades)
    _add_close_markers(fig, trades)
    _add_plot_artifacts(fig, price, plot)
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        height=560,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _build_with_panels(
    price: pd.DataFrame,
    trades: pd.DataFrame,
    plot: dict[str, Any] | None,
    panels: list[dict[str, Any]],
) -> go.Figure:
    """Price row + one sub-row per oscillator panel (shared x-axis)."""
    from plotly.subplots import make_subplots

    spec = plot or {}
    fig = make_subplots(
        rows=1 + len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.03
    )
    _add_candles(fig, price, row=1)
    _add_open_markers(fig, trades, row=1)
    _add_close_markers(fig, trades, row=1)
    _add_plot_artifacts(fig, price, spec, row=1)
    for i, panel in enumerate(panels, start=2):
        xs, ys = _series_xy(panel.get("series", []))
        fig.add_trace(
            go.Scatter(x=xs, y=ys, mode="lines", name=panel.get("name", "panel")),
            row=i,
            col=1,
        )
        for hline in panel.get("hlines", []):
            fig.add_hline(y=float(hline), row=i, col=1, line_dash="dot")
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        height=560,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _add_candles(fig: go.Figure, price: pd.DataFrame, row: int | None = None) -> None:
    """Grey OHLC candles (the price row of any layout)."""
    trace = go.Candlestick(
        x=price.index,
        open=price["open"],
        high=price["high"],
        low=price["low"],
        close=price["close"],
        name="OHLC",
        increasing_line_color=_GREY_LIGHT,
        decreasing_line_color=_GREY_DARK,
        increasing_fillcolor=_GREY_LIGHT,
        decreasing_fillcolor=_GREY_DARK,
    )
    if row is None:
        fig.add_trace(trace)
    else:
        fig.add_trace(trace, row=row, col=1)


def _series_xy(
    series: list[list[float]],
) -> tuple[pd.DatetimeIndex, list[float]]:
    """``[[ms_epoch, value], ...]`` -> (timestamps, values) for plotly."""
    xs = pd.to_datetime([int(ms_val) for ms_val, _ in series], unit="ms")
    ys = [float(val) for _, val in series]
    return xs, ys


def _add_plot_artifacts(
    fig: go.Figure,
    price: pd.DataFrame,
    plot: dict[str, Any] | None,
    row: int | None = None,
) -> None:
    """Overlays + pivot/divergence markers + legs from a serialized plot spec."""
    if not plot:
        return
    _add_overlays(fig, plot.get("overlays", []), price, row=row)
    _add_markers(fig, plot.get("markers", []), row=row)
    _add_legs(fig, plot.get("legs", []), row=row)


def _add_trace(
    fig: go.Figure, trace: go.Scatter | go.Bar, row: int | None = None
) -> None:
    """Add a trace to row 1, or the bare figure when no subplots exist."""
    if row is None:
        fig.add_trace(trace)
    else:
        fig.add_trace(trace, row=row, col=1)


def _overlay_range_ok(ys: list[float], price: pd.DataFrame, style: str) -> bool:
    """Guard an overlay whose range dwarfs price (an oscillator mislabelled).

    A line/dots overlay more than 5x the price span would collapse the y-axis,
    so it is skipped. Band/histogram overlays carry no y-scale intent and are
    always drawn.
    """
    if style in ("band", "histogram") or not ys:
        return True
    price_span = float(price["high"].max() - price["low"].min())
    span = max(ys) - min(ys)
    return price_span <= 0 or span <= 5 * price_span


def _add_overlays(
    fig: go.Figure,
    overlays: list[dict[str, Any]],
    price: pd.DataFrame,
    row: int | None = None,
) -> None:
    """Render price-pane overlays: lines/dots as scatter, band/hist as bars."""
    for ov in overlays:
        style = ov.get("style", "line")
        xs, ys = _series_xy(ov.get("series", []))
        if not _overlay_range_ok(ys, price, style):
            continue
        name = ov.get("name", "overlay")
        if style in ("band", "histogram"):
            trace: go.Scatter | go.Bar = go.Bar(x=xs, y=ys, name=name, opacity=0.3)
        else:
            trace = go.Scatter(
                x=xs,
                y=ys,
                mode="markers" if style == "dots" else "lines",
                marker=dict(size=3) if style == "dots" else None,
                name=name,
            )
        _add_trace(fig, trace, row=row)


def _add_markers(
    fig: go.Figure, markers: list[dict[str, Any]], row: int | None = None
) -> None:
    """Fold markers into one scatter trace per kind, reusing trade glyph style."""
    glyphs: dict[str, tuple[str, str, int]] = {
        "pivot_low": ("\u25b3", _GREY_DARK, 10),
        "pivot_high": ("\u25bd", _GREY_DARK, 10),
        "div_long": ("\u27cb", _GREEN, 22),
        "div_short": ("\u27cb", _RED, 22),
        "signal": ("\u25cf", _GREEN, 12),
    }
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for m in markers:
        by_kind.setdefault(m.get("kind", "signal"), []).append(m)
    for kind, items in by_kind.items():
        glyph, color, size = glyphs.get(kind, glyphs["signal"])
        trace = go.Scatter(
            x=[pd.Timestamp(int(m["ts"]), unit="ms") for m in items],
            y=[float(m["price"]) for m in items],
            mode="text",
            text=[glyph] * len(items),
            textfont=dict(size=size, color=color),
            textposition="middle center",
            name=kind,
        )
        _add_trace(fig, trace, row=row)


def _add_legs(
    fig: go.Figure, legs: list[dict[str, Any]], row: int | None = None
) -> None:
    """One line per divergence leg, green (long) / red (short)."""
    for leg in legs:
        color = _GREEN if leg.get("kind") == "div_long" else _RED
        trace = go.Scatter(
            x=[
                pd.Timestamp(int(leg["a_ts"]), unit="ms"),
                pd.Timestamp(int(leg["b_ts"]), unit="ms"),
            ],
            y=[float(leg["a_price"]), float(leg["b_price"])],
            mode="lines",
            line=dict(color=color, width=2),
            name=str(leg.get("kind", "leg")),
            showlegend=False,
        )
        _add_trace(fig, trace, row=row)


def _add_open_markers(
    fig: go.Figure, trades: pd.DataFrame, row: int | None = None
) -> None:
    """Long ▲ green, short ▼ red at entry."""
    glyphs = {
        "long": ("\u25b2", _GREEN),
        "short": ("\u25bc", _RED),
    }
    for pos, (glyph, color) in glyphs.items():
        sub = trades[trades["position"] == pos]
        if sub.empty:
            continue
        _add_trace(
            fig,
            go.Scatter(
                x=pd.to_datetime(sub["entry_time"]),
                y=sub["entry_price"],
                mode="text",
                text=[glyph] * len(sub),
                textfont=dict(size=17, color=color),
                textposition="middle center",
                name=f"open {pos}",
            ),
            row=row,
        )


def _add_close_markers(
    fig: go.Figure, trades: pd.DataFrame, row: int | None = None
) -> None:
    """Flags colored opposite the position at exit."""
    glyphs = {
        "long": ("\u2691", _RED),  # red flag on long close
        "short": ("\u2691", _GREEN),  # green flag on short close
    }
    for pos, (glyph, color) in glyphs.items():
        closed = trades[(trades["position"] == pos) & trades["exit_time"].notna()]
        if closed.empty:
            continue
        _add_trace(
            fig,
            go.Scatter(
                x=pd.to_datetime(closed["exit_time"]),
                y=closed["exit_price"],
                mode="text",
                text=[glyph] * len(closed),
                textfont=dict(size=24, color=color),
                textposition="top center",
                name=f"close {pos}",
            ),
            row=row,
        )
