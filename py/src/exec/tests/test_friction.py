"""Direction and scaling tests for the shared friction helpers."""

from __future__ import annotations

from src.bt.state.types import FixedCommission, PerShareCommission
from src.exec.friction import apply_friction, commission_for_fill


def test_buy_pays_above_the_mid() -> None:
    result = apply_friction(
        100.0, is_buy=True, spread_bps=10.0, slippage_bps=4.0, qty=0.0
    )
    assert result.executed_price > 100.0


def test_sell_receives_below_the_mid() -> None:
    result = apply_friction(
        100.0, is_buy=False, spread_bps=10.0, slippage_bps=4.0, qty=0.0
    )
    assert result.executed_price < 100.0


def test_friction_is_symmetric_around_the_mid() -> None:
    buy = apply_friction(100.0, is_buy=True, spread_bps=10.0, slippage_bps=4.0, qty=0.0)
    sell = apply_friction(
        100.0, is_buy=False, spread_bps=10.0, slippage_bps=4.0, qty=0.0
    )
    assert buy.executed_price - 100.0 == 100.0 - sell.executed_price


def test_costs_scale_with_qty() -> None:
    one = apply_friction(100.0, is_buy=True, spread_bps=10.0, slippage_bps=4.0, qty=1.0)
    ten = apply_friction(
        100.0, is_buy=True, spread_bps=10.0, slippage_bps=4.0, qty=10.0
    )
    assert ten.spread_cost == 10.0 * one.spread_cost
    assert ten.slippage_cost == 10.0 * one.slippage_cost


def test_adverse_multiplier_widens_slippage_only() -> None:
    plain = apply_friction(
        100.0, is_buy=True, spread_bps=10.0, slippage_bps=4.0, qty=1.0
    )
    adverse = apply_friction(
        100.0,
        is_buy=True,
        spread_bps=10.0,
        slippage_bps=4.0,
        qty=1.0,
        adverse_multiplier=1.5,
    )
    assert adverse.spread_cost == plain.spread_cost
    assert adverse.slippage_cost == 1.5 * plain.slippage_cost


def test_flat_commission_ignores_size() -> None:
    model = FixedCommission(0.5)
    assert commission_for_fill(model, 1000.0, 999.0) == 0.5


def test_per_share_commission_applies_floor_then_cap() -> None:
    model = PerShareCommission(per_share=0.005, min_per_fill=1.0)
    assert commission_for_fill(model, 100.0, 10.0) == 1.0  # floor
    capped = PerShareCommission(per_share=0.01, min_per_fill=0.0, max_pct_of_value=0.1)
    # 0.1 % of 10 shares * $10 = $0.10 cap.
    assert commission_for_fill(capped, 10.0, 10.0) == 0.1
