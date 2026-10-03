"""Pure position-sizing layer.

Converts a ``TradeSignal`` whose ``qty <= 0`` ("compute it") into an absolute
share count. Supports three capital bases (``equity`` / ``cash`` / ``fixed``),
plus per-symbol and available-cash caps so multi-symbol strategies cannot
over-deploy.

Risk-targeted sizing (``qty`` from ``risk_pct`` and a stop distance / ATR) is
**strategy-owned** — it needs per-strategy stop semantics that a generic layer
cannot know. Strategies that want it use :func:`risk_sized_qty` directly and
emit an explicit ``qty`` (or a back-solved ``size``), rather than routing
through the engine's shared ``SizingParams``.

Everything here is a pure function: identical inputs always yield identical
outputs, no side effects, immutable inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal, Mapping, Any

from src.bt.state.types import PortfolioView, TradeSignal, Candle
from src.bt.portfolio.pure import calculate_positions_value

#: Allowed capital bases for size-based sizing.
SizingMode = Literal["equity", "cash", "fixed"]


@dataclass(frozen=True)
class SizingParams:
    """Config for the shared position-sizing layer.

    Sizes positions as a fixed fraction of a capital base. Risk-targeted
    sizing lives in the strategy (via :func:`risk_sized_qty`), not here.

    Attributes:
        sizing_mode: Capital base for size-based sizing. ``equity`` sizes off
            current total equity, ``cash`` off available cash, ``fixed`` treats
            ``size`` as a fixed nominal dollar amount per trade.
        size: Fixed-fraction (0-1) of ``sizing_mode`` base, or (in ``fixed``
            mode) the fixed dollar amount deployed per trade.
        max_symbol_allocation: Cap on a single symbol's position value as a
            fraction of equity (portfolio-weight semantics; per-symbol caps
            should sum to <= 1.0). Applies in every mode.
    """

    sizing_mode: SizingMode = "equity"
    size: float = 0.0
    max_symbol_allocation: float = 1.0

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SizingParams":
        """Build from a config dict, coercing to valid values and defaults."""
        mode = raw.get("sizing_mode", "equity")
        if mode not in ("equity", "cash", "fixed"):
            mode = "equity"
        return cls(
            sizing_mode=mode,
            size=float(raw.get("size", 0.0)),
            max_symbol_allocation=float(raw.get("max_symbol_allocation", 1.0)),
        )


def equity_of(portfolio: PortfolioView) -> float:
    """Total equity = cash + value of open positions.

    Accepts the minimal ``PortfolioView`` Protocol (``cash`` + ``positions``),
    not only a concrete ``PortfolioState``, so the live reconcile sizing path
    reuses the ONE equity implementation instead of re-deriving
    ``cash + calculate_positions_value``. ``PortfolioState`` satisfies the
    Protocol structurally, so every existing bt caller is unchanged.
    """
    return portfolio.cash + calculate_positions_value(portfolio.positions)


def risk_sized_qty(
    *,
    equity: float,
    price: float,
    stop_dist: float,
    risk_pct: float,
) -> float:
    """Risk-targeted share count: ``qty = equity * risk_pct / stop_dist``.

    Strategy-owned risk sizing — the caller supplies the stop distance
    (ATR * mult, price - stop, etc.), keeping risk semantics out of the shared
    sizing layer. Returns 0.0 for invalid inputs; never negative.
    """
    if equity <= 0 or price <= 0 or stop_dist <= 0 or risk_pct <= 0:
        return 0.0
    return round(equity * risk_pct / stop_dist, 4)


def compute_qty(
    *,
    equity: float,
    cash: float,
    price: float,
    params: SizingParams,
) -> float:
    """Compute an absolute share count for a position sized per ``params``.

    Rules:
      - size-based: ``qty = base*size/price`` (base from ``sizing_mode``).
      - Always clamped so ``qty*price <= cash`` and
        ``qty*price <= equity*max_symbol_allocation``.
      - Returns 0.0 on invalid inputs; never negative; rounded to 4 dp.
    """
    if price <= 0:
        return 0.0
    return _clamp(
        _size_based_qty(equity, cash, price, params),
        equity,
        cash,
        price,
        params,
    )


def _size_based_qty(
    equity: float,
    cash: float,
    price: float,
    params: SizingParams,
) -> float:
    """Fixed-fraction / equal-weight / fixed-dollar qty before caps."""
    if params.size <= 0:
        return 0.0
    if params.sizing_mode == "cash":
        base = cash
    elif params.sizing_mode == "fixed":
        # ``size`` is a fixed nominal dollar amount per trade.
        return params.size / price
    else:  # equity
        base = equity
    if base <= 0:
        return 0.0
    return base * params.size / price


def _clamp(
    qty: float,
    equity: float,
    cash: float,
    price: float,
    params: SizingParams,
) -> float:
    """Apply cash clamp and per-symbol allocation cap; round to 4 dp."""
    if qty <= 0 or price <= 0:
        return 0.0
    cap_share = math.inf
    if cash > 0:
        cap_share = min(cap_share, cash / price)
    if equity > 0 and params.max_symbol_allocation > 0:
        cap_share = min(cap_share, equity * params.max_symbol_allocation / price)
    return round(min(qty, cap_share), 4)


def sized_signal(
    signal: TradeSignal,
    equity: float,
    cash: float,
    candle: Candle,
    params: SizingParams,
) -> TradeSignal:
    """Fill ``signal.qty`` via ``compute_qty`` when the signal requests it.

    Signals with an explicit ``qty > 0`` are returned unchanged (the strategy
    already sized them — fixed-size / risk-solved orders). Signals with
    ``qty <= 0`` get an engine-sourced share count.
    """
    if signal.qty > 0:
        return signal
    price = signal.price if signal.price > 0 else candle.close
    qty = compute_qty(
        equity=equity,
        cash=cash,
        price=price,
        params=params,
    )
    return replace(signal, qty=qty)
