"""Behaviour tests for the pure scope identity module (see ``src.live.scope``).

Two properties are load-bearing: the intent hash is blind to friction/bookkeeping
edits, and it MOVES when the strategy itself changes. Both are exercise-able from a
directly built ``LiveConfig`` — no DB, no mocks.
"""

from __future__ import annotations

import dataclasses

from src.live.scope import (
    ScopeParts,
    parse_scope,
    scope_of,
)
from src.live.types import LiveConfig


def cfg(**overrides: object) -> LiveConfig:
    """A minimal complete config; ``overrides`` replace individual fields."""
    base = LiveConfig(
        strategy_type="vwap_reversion",
        symbols=("AAPL", "MSFT"),
        initial_capital=25_000.0,
        strategy_params={"z_entry": 2.0, "window": 75},
        bars=("5m", "1h"),
        warmup="1y",
        size_mode="equity",
        size=0.25,
        max_symbol_allocation=0.5,
    )
    return dataclasses.replace(base, **overrides)  # type: ignore[arg-type]


def test_scope_round_trips_through_parse() -> None:
    """``parse_scope`` inverts ``scope_of`` for every adapter."""
    for adapter in ("ibkr", "sim"):
        parts = ScopeParts(adapter=adapter, config_name="vwatr_hv", instance="a1b2c3d4")
        assert parse_scope(scope_of(parts)) == parts


def test_parse_scope_rejects_a_missing_instance() -> None:
    """A two-segment scope is not a scope."""
    assert parse_scope("vwatr_hv_gated") is None
