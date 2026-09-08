"""Streamlit dashboard over ``ibkr bt run <cfg> -F plot`` output.

Render one candlestick chart per symbol (picked via a select box) with A-style
trade markers, plus a full trades table.

Launcher:
    ibkr bt run strats/ema_cross_dsl.json -F plot -o run.json
    streamlit run scripts/bt_dashboard.py -- run.json

Run in repo root so ``src`` is importable (``make dash`` does this).
"""

from __future__ import annotations

import json
import sys
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Marker glyphs per request:
#   open long  -> green up-triangle   (▲)     close long  -> red flag   (⚑)
#   open short -> red down-triangle   (▼)     close short -> green flag (⚑)
# Candle up/down theme is grey (light vs dark) per request; markers stay colored.
_GREEN = "#1fae54"
_RED = "#e64545"
_GREY_LIGHT = "#cfd3d5"  # rising candles
_GREY_DARK = "#444b52"  # falling candles


def _fmt_metric(key: str, value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if key in ("total_return", "annual_return", "max_drawdown"):
            return f"{value:.2%}"
        return f"{value:.3f}"
    return str(value)


def load_payload(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def trades_table(payload: dict[str, Any]) -> pd.DataFrame:
    """All trades as a display DataFrame (unfiltered by symbol)."""
    cols = [
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
    ]
    rows = payload.get("trades", [])
    present = [c for c in cols if rows and c in rows[0]]
    return pd.DataFrame(rows, columns=present)


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
    # --- open markers: long ▲ green, short ▼ red ---------------------------
    open_glyphs = {
        "long": ("\u25b2", _GREEN, "middle center"),
        "short": ("\u25bc", _RED, "middle center"),
    }
    for pos, (glyph, color, textpos) in open_glyphs.items():
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
                textposition=textpos,
                name=f"open {pos}",
            )
        )
    # --- close markers: flags colored opposite the position ----------------
    close_glyphs = {
        "long": ("\u2691", _RED, "top center"),  # red flag on long close
        "short": ("\u2691", _GREEN, "top center"),  # green flag on short close
    }
    for pos, (glyph, color, textpos) in close_glyphs.items():
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
                textposition=textpos,
                name=f"close {pos}",
            )
        )
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        height=560,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def frame_for_symbol(payload: dict[str, Any], symbol: str) -> dict[str, Any]:
    frames = payload.get("symbols", {}).get(symbol, [])
    if not frames:
        st.warning(f"No candle frames recorded for {symbol}.")
        st.stop()
    return frames[0]


def render_metrics(payload: dict[str, Any]) -> None:
    metrics = payload.get("metrics", {})
    keys = [
        "total_return",
        "annual_return",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "annual_volatility",
    ]
    for col, key in zip(st.columns(len(keys)), keys):
        col.metric(key, _fmt_metric(key, metrics.get(key)))


def main(argv: list[str]) -> None:
    st.set_page_config(page_title="Backtest results", layout="wide")
    if not argv:
        st.error(
            "Pass the path to a `-F plot` json file:\n\n"
            "  ibkr bt run <cfg> -F plot -o run.json\n"
            "  streamlit run scripts/bt_dashboard.py -- run.json"
        )
        st.stop()

    payload = load_payload(argv[0])
    st.title("Backtest results")
    render_metrics(payload)

    # Symbol picker (reference point for the per-symbol candle chart).
    symbols = list(payload.get("symbols", {}))
    if not symbols:
        st.warning("Payload carries no symbol candles.")
        st.stop()
    symbol = st.selectbox("Symbol", symbols)

    frame = frame_for_symbol(payload, symbol)
    price = pd.DataFrame(
        frame["bars"],
        columns=["ts", "open", "high", "low", "close", "volume"],
    )
    price["ts"] = pd.to_datetime(price["ts"])
    price = price.sort_values("ts").set_index("ts")

    trades = payload.get("trades", [])
    cols = [
        "symbol",
        "position",
        "qty",
        "entry_time",
        "entry_price",
        "exit_time",
        "exit_price",
    ]
    trades = pd.DataFrame(
        [
            t
            for t in trades
            if t.get("symbol") == symbol and t.get("interval") == frame["interval"]
        ],
        columns=cols,
    )

    st.subheader(f"{symbol} — {frame.get('interval', '?')} candles")
    if price.empty:
        st.info("No candles for this symbol.")
    else:
        st.plotly_chart(build_chart(price, trades), width="stretch")

    all_trades = trades_table(payload)
    if all_trades.empty:
        st.info("No trades recorded.")
    else:
        st.subheader("All trades")
        st.dataframe(all_trades, width="stretch")


if __name__ == "__main__":
    main(sys.argv[1:])
