"""Streamlit app over ``ibkr bt run --plot`` payloads.

Launched by ``src.bt.dashboard.launch_dashboard`` via
``python -m streamlit run <this file> -- <payload.json>``. Streamlit needs a
real module path, so the render logic lives in ``render.py`` and the payload
path arrives as ``sys.argv[1]``.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import streamlit as st

from src.bt.dashboard.render import (
    build_chart,
    metrics_table,
    price_frame,
    trades_for_frame,
    trades_table,
)


def load_payload(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main(argv: list[str]) -> None:
    st.set_page_config(page_title="Backtest results", layout="wide")
    if not argv:
        st.error("No payload path given. Launch with:\n\n  ibkr bt run <cfg> --plot")
        st.stop()

    payload = load_payload(argv[0])
    st.title("Backtest results")

    pairs = metrics_table(payload)
    for col, (key, value) in zip(st.columns(len(pairs)), pairs):
        col.metric(key, value)

    symbols = list(payload.get("symbols", {}))
    if not symbols:
        st.warning("Payload carries no symbol candles.")
        st.stop()
    symbol = st.selectbox("Symbol", symbols)

    frames = payload.get("symbols", {}).get(symbol, [])
    if not frames:
        st.warning(f"No candle frames recorded for {symbol}.")
        st.stop()
    frame = frames[0]

    price = price_frame(frame)
    trades = trades_for_frame(payload, symbol, frame.get("interval", ""))

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
