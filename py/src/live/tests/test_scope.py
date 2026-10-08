"""Behaviour tests for the pure scope identity module (see ``src.live.scope``).

Two properties are load-bearing: the intent hash is blind to friction/bookkeeping
edits, and it MOVES when the strategy itself changes. Both are exercise-able from a
directly built ``LiveConfig`` — no DB, no mocks.
"""

from __future__ import annotations

import dataclasses

from src.live.scope import (
    ScopeParts,
    config_hash,
    config_name_of,
    parse_scope,
    scope_of,
    slug_segment,
    strategy_intent_payload,
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


def test_same_intent_config_has_same_hash_and_scope() -> None:
    """Two configs with identical intent hash and scope identically."""
    assert config_hash(cfg()) == config_hash(cfg())
    parts = ScopeParts(
        adapter="ibkr", config_name=config_name_of(cfg()), instance="a1b2c3d4"
    )
    assert scope_of(parts) == scope_of(parts)


def test_friction_and_bookkeeping_edits_do_not_move_the_hash() -> None:
    """Every non-intent field is outside the hash's input."""
    baseline = config_hash(cfg())
    for field, value in (
        ("spread_bps", 99.0),
        ("commission", 9.99),
        ("initial_capital", 1_000_000.0),
        ("broker", "ibkr"),
        ("mode", "live"),
        ("adapter", "sim"),
        ("config_name", "renamed"),
    ):
        assert config_hash(cfg(**{field: value})) == baseline, field


def test_strategy_param_edit_moves_the_hash() -> None:
    """A changed param is a different strategy."""
    assert config_hash(
        cfg(strategy_params={"z_entry": 2.5, "window": 75})
    ) != config_hash(cfg())


def test_size_edit_moves_the_hash() -> None:
    """Sizing is intent: a different size is a different risk profile."""
    assert config_hash(cfg(size=0.5)) != config_hash(cfg())


def test_intent_payload_keys_are_exactly_the_intent() -> None:
    """The payload carries intent only — no friction leak can slip in."""
    assert set(strategy_intent_payload(cfg())) == {
        "strategy_type",
        "symbols",
        "strategy_params",
        "bars",
        "warmup",
        "size_mode",
        "size",
        "max_symbol_allocation",
    }


def test_adapter_is_part_of_the_scope_not_the_hash() -> None:
    """Paper and live are different books, same strategy identity."""
    config = cfg()
    ibkr = ScopeParts(adapter="ibkr", config_name="vwatr", instance="a1b2c3d4")
    sim = dataclasses.replace(ibkr, adapter="sim")
    assert scope_of(ibkr) != scope_of(sim)
    assert config_hash(config) == config_hash(config)


def test_scope_round_trips_through_parse() -> None:
    """``parse_scope`` inverts ``scope_of`` for every adapter."""
    for adapter in ("ibkr", "sim"):
        parts = ScopeParts(adapter=adapter, config_name="vwatr_hv", instance="a1b2c3d4")
        assert parse_scope(scope_of(parts)) == parts


def test_parse_scope_rejects_a_missing_instance() -> None:
    """A two-segment scope is not a scope."""
    assert parse_scope("vwatr_hv_gated") is None


def test_slug_segment_normalises_and_collapses() -> None:
    """Runs of disallowed characters collapse to one ``-``, then trim."""
    assert slug_segment("VWATR / HV!!Gated") == "vwatr-hv-gated"
    assert slug_segment("...") == ""
    assert config_name_of(cfg(config_name="Hype/Config")) == "hype-config"
    assert config_name_of(cfg(config_name="")) == "config"
