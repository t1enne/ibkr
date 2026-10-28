"""TARGET resolution: config path, name, scope or scope prefix (plan §2).

Resolution is **registry-row first**: the row written at spawn owns a bot's
identity, so a target addresses the scope the bot actually holds even after its
config was edited or removed. Only when no row matches does a target mint from
the config on disk — and never the other way round, because re-minting a live
bot's scope would address a book the bot does not reconcile.

An ambiguous prefix is an error listing the candidates, never a silent pick.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.bot.types import BotLiveness, BotRecord


class TargetError(ValueError):
    """A TARGET token matched nothing, or matched more than one bot."""


@dataclass(frozen=True)
class BotTarget:
    """A resolved target: identity from the registry row, else minted from config.

    ``from_registry`` is the provenance: True means a row owned the identity, so
    the scope is the bot's own and must never be re-derived from the config.
    """

    config_path: str
    scope: str
    from_registry: bool


def registry_rows(registry: BotLiveness) -> tuple[BotRecord, ...]:
    """Every row the registry holds, in the registry's own order.

    ``list_bots`` is part of the ``BotLiveness`` protocol, so ``--all`` reaches
    the whole registry through the same seam every per-scope probe uses.
    """
    return tuple(registry.list_bots())


def resolve_targets(
    registry: BotLiveness,
    tokens: Sequence[str],
    *,
    mint: Callable[[str], str],
) -> tuple[BotTarget, ...]:
    """Resolve every TARGET token, in the order given, without duplicates.

    *mint* is the ONE function that derives a scope from a config on disk
    (``load_scope``); it is called only when no registry row matches a token, so
    a read-only command can never re-mint a scope a live bot does not hold.
    """
    rows = registry_rows(registry)
    resolved: list[BotTarget] = []
    seen: set[str] = set()
    for token in tokens:
        target = _resolve_one(token, rows, mint)
        if target.scope in seen:
            continue
        seen.add(target.scope)
        resolved.append(target)
    return tuple(resolved)


def name_of(config_path: str) -> str:
    """The bot's NAME: the config file's basename without ``.json``."""
    return Path(config_path).name.removesuffix(".json")


def _resolve_one(
    token: str, rows: Sequence[BotRecord], mint: Callable[[str], str]
) -> BotTarget:
    """One token: a registry row when any matches, else minted from the config."""
    matched = _match(token, rows)
    if len(matched) == 1:
        row = matched[0]
        return BotTarget(row.config_path, row.scope, True)
    if len(matched) > 1:
        raise TargetError(_ambiguous(token, matched))
    if not Path(token).is_file():
        raise TargetError(f"no bot or config matches {token!r}")
    return BotTarget(token, mint(token), False)


def _match(token: str, rows: Sequence[BotRecord]) -> tuple[BotRecord, ...]:
    """Rows a token names, most specific rule first.

    Exact scope, exact config path, exact NAME, then a scope PREFIX. The first
    non-empty rule wins: a name that is also another bot's scope prefix must
    resolve to itself, not to both.
    """
    absolute = _absolute(token)
    rules: tuple[Callable[[BotRecord], bool], ...] = (
        lambda row: row.scope == token,
        lambda row: _absolute(row.config_path) == absolute,
        lambda row: name_of(row.config_path) == token,
        lambda row: row.scope.startswith(token),
    )
    for rule in rules:
        found = tuple(row for row in rows if rule(row))
        if found:
            return found
    return ()


def _absolute(path: str) -> str:
    """*path* normalised for comparison (never created, never resolved to a link)."""
    return os.path.abspath(path)


def _ambiguous(token: str, rows: Sequence[BotRecord]) -> str:
    """The usage-error text for an ambiguous token: it names every candidate."""
    candidates = ", ".join(sorted(row.scope for row in rows))
    return f"{token!r} is ambiguous; it matches: {candidates}"


__all__ = [
    "BotTarget",
    "TargetError",
    "name_of",
    "registry_rows",
    "resolve_targets",
]
