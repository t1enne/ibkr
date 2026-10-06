"""Tests for the live domain types (the shared edge vocabulary)."""

from __future__ import annotations

from typing import get_args

from src.bt.state import FixedCommission, PerShareCommission
from src.live.types import FeedKind, LiveConfig, exec_params_of, feed_error


def _cfg_friction(
    *,
    commission: float = 0.5,
    spread_bps: float = 5.0,
    slippage_bps: float = 2.0,
    commission_per_share: float | None = None,
    commission_min: float = 0.0,
    commission_max_pct: float | None = None,
) -> LiveConfig:
    return LiveConfig(
        strategy_type="x",
        symbols=("AAPL",),
        initial_capital=0.0,
        strategy_params={},
        bars=("1d",),
        warmup="1y",
        commission=commission,
        spread_bps=spread_bps,
        slippage_bps=slippage_bps,
        commission_per_share=commission_per_share,
        commission_min=commission_min,
        commission_max_pct=commission_max_pct,
    )


def test_feed_error_preserves_every_declared_kind() -> None:
    # Guards the L3 hazard: a kind added to the ``FeedKind`` Literal but not to the
    # accepted set would silently degrade to "transport" here. Because the set is
    # DERIVED from the Literal, this can never drift.
    for kind in get_args(FeedKind):
        assert feed_error(kind, "m").kind == kind


def test_feed_error_degrades_an_unknown_kind_to_transport() -> None:
    assert feed_error("not-a-real-kind", "m").kind == "transport"


def test_exec_params_of_mirrors_the_config_friction() -> None:
    # The ONE construction reconcile sizes with and the CLI's sim broker fills
    # with (finding L8): spread/slippage/flat-commission all come from the config.
    params = exec_params_of(
        _cfg_friction(commission=0.05, spread_bps=3.0, slippage_bps=1.0)
    )
    assert params.spread_bps == 3.0
    assert params.slippage_bps == 1.0
    assert params.commission_model == FixedCommission(0.05)


def test_exec_params_of_selects_per_share_commission_when_configured() -> None:
    params = exec_params_of(
        _cfg_friction(
            commission_per_share=0.005, commission_min=1.0, commission_max_pct=0.01
        )
    )
    assert params.commission_model == PerShareCommission(
        per_share=0.005, min_per_fill=1.0, max_pct_of_value=0.01
    )
