"""Live scope identity — parse, build and hash the ``<adapter>_<config>_<instance>`` key.

A live scope has two halves that must not drift apart:

- the **name** (``ScopeParts``) — adapter, config name, instance — which keys the
  per-scope book and prefixes the cOID, and
- the **strategy intent hash** (``config_hash``) — the capital-at-risk identity of
  the strategy *itself*.

Friction and bookkeeping fields (fees, mode, capital, paths) are deliberately
excluded from the hash: re-running the same strategy with a different commission
assumption is the same strategy, and changing those must not fork its identity.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Literal

from src.live.types import LiveConfig

#: Adapters a scope may name. Anything else fails to parse.
AdapterName = Literal["ibkr", "sim"]

#: One or more characters outside the allowed slug set, replaced by a single ``-``.
_DISALLOWED = re.compile(r"[^A-Za-z0-9_-]+")

#: Longest adapter token that can open a scope string ("ibkr").
_ADAPTERS = ("ibkr", "sim")


@dataclass(frozen=True)
class ScopeParts:
    """The three segments of a live scope key, parsed or built."""

    adapter: AdapterName
    config_name: str
    instance: str


def slug_segment(text: str) -> str:
    """Normalise ``text`` into a scope-safe segment.

    Keeps ``[A-Za-z0-9_-]``, collapses every other run to a single ``-``, trims
    leading/trailing ``-`` and lowercases. A result with nothing left is returned
    as ``""`` — the caller decides the fallback, so an empty segment never appears
    in a scope by accident.
    """
    return _DISALLOWED.sub("-", text).strip("-").lower()


def parse_scope(scope: str) -> ScopeParts | None:
    """Parse ``<adapter>_<config_name>_<instance>``, or ``None`` if the shape fails.

    The adapter must be ``ibkr`` or ``sim``, the config name a non-empty slug, and
    the instance a non-empty tail. The instance is the LAST segment, so a config
    name containing ``_`` stays whole and ``parse_scope`` inverts ``scope_of``
    exactly. The 8-hex token ``identity`` mints is NOT enforced: a hand-typed scope
    must stay parseable, so any non-empty tail is accepted.
    """
    adapter, sep, rest = scope.partition("_")
    config_name, sep2, instance = rest.rpartition("_")
    if not sep or not sep2 or adapter not in _ADAPTERS:
        return None
    if not config_name or not instance:
        return None
    return ScopeParts(adapter=adapter, config_name=config_name, instance=instance)


def scope_of(parts: ScopeParts) -> str:
    """The scope key string for ``parts``."""
    return f"{parts.adapter}_{parts.config_name}_{parts.instance}"


def config_name_of(cfg: LiveConfig) -> str:
    """The config's scope segment: its ``name`` slugged, or ``config`` when empty."""
    return slug_segment(cfg.config_name) or "config"


def strategy_intent_payload(cfg: LiveConfig) -> dict[str, object]:
    """The fields that define the strategy's intent — the hash's whole input.

    Only what the strategy *does*: type, universe, params, bars, warm-up and
    sizing. Friction and bookkeeping (fees, spread/slippage, mode, broker, scope,
    capital, portfolio path, config name) are excluded by construction, so editing
    them leaves the identity unchanged.
    """
    return {
        "strategy_type": cfg.strategy_type,
        "symbols": list(cfg.symbols),
        "strategy_params": cfg.strategy_params,
        "bars": list(cfg.bars),
        "warmup": cfg.warmup,
        "size_mode": cfg.size_mode,
        "size": cfg.size,
        "max_symbol_allocation": cfg.max_symbol_allocation,
    }


def config_hash(cfg: LiveConfig) -> str:
    """A short, stable digest of ``strategy_intent_payload(cfg)``.

    Canonical JSON (sorted keys, no spaces) so field order never changes the hash,
    with ``default=str`` so an exotic param value degrades instead of raising.
    """
    payload = json.dumps(
        strategy_intent_payload(cfg), sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]
