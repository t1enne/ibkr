"""Friction and commission helpers, shared by every fill path.

Moved out of ``src/bt/execution/pure.py`` unchanged (that module re-exports them)
so the backtest and live paths charge costs through ONE implementation. The
commission model itself (``CommissionModel``/``FixedCommission``/
``PerShareCommission``) and ``FrictionResult`` are defined in the shared order
core (``src/exec/types.py``) rather than the bt package, so this module never
imports ``src.bt`` (which would be a cycle).
"""

from __future__ import annotations

from src.exec.types import CommissionModel, FixedCommission, FrictionResult


def commission_for_fill(model: CommissionModel, qty: float, price: float) -> float:
    """Charge $ for one fill under ``model``.

    Flat models ignore qty/price. Per-share models charge ``per_share * |qty|``,
    raise to the per-fill floor, then cap at ``max_pct_of_value`` percent of the
    traded value (when set).
    """
    if isinstance(model, FixedCommission):
        return model.amount
    charge = abs(qty) * model.per_share
    if charge < model.min_per_fill:
        charge = model.min_per_fill
    if model.max_pct_of_value is not None:
        cap = abs(qty) * price * model.max_pct_of_value / 100.0
        charge = min(charge, cap)
    return charge


def apply_friction(
    base_price: float,
    *,
    is_buy: bool,
    spread_bps: float,
    slippage_bps: float,
    qty: float,
    adverse_multiplier: float = 1.0,
) -> FrictionResult:
    """Shift ``base_price`` by half-spread plus slippage; report qty-scaled $.

    A buyer pays above the mid, a seller receives below it, so the half-spread
    and slippage always lean against the fill. ``adverse_multiplier`` scales the
    slippage (both exec paths share this one knob). All recorded costs are
    dollar amounts scaled by ``qty``, never per-share fractions.
    """
    half_spread = base_price * (spread_bps / 2.0) / 10000.0
    slip = base_price * (slippage_bps * adverse_multiplier) / 10000.0
    if is_buy:
        executed_price = base_price + half_spread + slip
    else:
        executed_price = base_price - half_spread - slip
    return FrictionResult(
        executed_price=executed_price,
        spread_cost=abs(half_spread) * qty,
        slippage_cost=abs(slip) * qty,
    )
