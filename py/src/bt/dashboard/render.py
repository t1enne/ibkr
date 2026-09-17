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


def build_chart(price: pd.DataFrame, trades: pd.DataFrame) -> go.Figure:
    """OHLC candles + open/close trade glyphs for one symbol/interval."""
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
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
    )
    _add_open_markers(fig, trades)
    _add_close_markers(fig, trades)
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        height=560,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _add_open_markers(fig: go.Figure, trades: pd.DataFrame) -> None:
    """Long ▲ green, short ▼ red at entry."""
    glyphs = {
        "long": ("\u25b2", _GREEN),
        "short": ("\u25bc", _RED),
    }
    for pos, (glyph, color) in glyphs.items():
        sub = trades[trades["position"] == pos]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(sub["entry_time"]),
                y=sub["entry_price"],
                mode="text",
                text=[glyph] * len(sub),
                textfont=dict(size=17, color=color),
                textposition="middle center",
                name=f"open {pos}",
            )
        )


def _add_close_markers(fig: go.Figure, trades: pd.DataFrame) -> None:
    """Flags colored opposite the position at exit."""
    glyphs = {
        "long": ("\u2691", _RED),  # red flag on long close
        "short": ("\u2691", _GREEN),  # green flag on short close
    }
    for pos, (glyph, color) in glyphs.items():
        closed = trades[(trades["position"] == pos) & trades["exit_time"].notna()]
        if closed.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(closed["exit_time"]),
                y=closed["exit_price"],
                mode="text",
                text=[glyph] * len(closed),
                textfont=dict(size=24, color=color),
                textposition="top center",
                name=f"close {pos}",
            )
        )
