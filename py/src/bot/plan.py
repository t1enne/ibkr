"""The resolved plan for one target — what ``inspect`` shows (plan §4).

Pure and read-only by construction: every path here is DERIVED, never created
(``probe_live`` opens without ``O_CREAT``), and the scope comes from the
target, which took it from the registry row when one existed. So asking about a
bot can never re-mint its identity or write to its book.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.bot.journal import log_path_of
from src.bot.liveness import bot_lock_path, bot_pid_path, bot_stop_flag_path
from src.bot.loop import BotOptions, derive_cadence
from src.bot.targets import BotTarget
from src.bot.types import BotLiveness, BotState
from src.db.path import resolve_live_db_path
from src.live.options import AdapterName, LiveOptions
from src.live.run_once import load_strat_config
from src.live.scope import parse_scope
from src.live.types import LiveConfig


@dataclass(frozen=True)
class BotPlan:
    """Everything ``inspect`` shows and ``--dry-run`` prints; read-only."""

    target: BotTarget
    adapter: AdapterName
    interval_s: int
    max_age_days: int
    retries: int
    sync: bool
    db_path: Path
    lock_path: Path
    pid_path: Path
    stop_flag_path: Path
    log_path: Path
    state: BotState
    exit_code: int | None
    started_at: pd.Timestamp | None


def resolve_plan(
    registry: BotLiveness,
    target: BotTarget,
    live_options: LiveOptions,
    options: BotOptions,
) -> BotPlan:
    """The plan for *target*: identity, cadence and every path it touches.

    *options* is the merged ``[bot]`` knobs (``interval``/``max_age``/
    ``no_sync``/``retries``), and *live_options* the merged ``[live]`` ones, so
    the plan states exactly what a spawn WOULD use without computing anything
    twice. A target whose config no longer exists still resolves: the scope is
    the bot's own, and the cadence falls back to the config-bearing row.
    """
    cfg = _config_of(target)
    cadence = derive_cadence(
        cfg.bars[0] if cfg is not None else "1d",
        interval=options.interval,
        max_age=options.max_age,
    )
    row = registry.probe(target.scope)
    db_path = resolve_live_db_path()
    return BotPlan(
        target=target,
        adapter=_adapter_of(target, live_options),
        interval_s=cadence.interval_s,
        max_age_days=cadence.max_age_days,
        retries=options.retries,
        sync=not options.no_sync,
        db_path=db_path,
        lock_path=Path(bot_lock_path(db_path, target.scope)),
        pid_path=Path(bot_pid_path(db_path, target.scope)),
        stop_flag_path=Path(bot_stop_flag_path(db_path, target.scope)),
        log_path=log_path_of(target.scope),
        state="stale" if row is None else row.state,
        exit_code=None if row is None else row.exit_code,
        started_at=None if row is None else row.started_at,
    )


def _adapter_of(target: BotTarget, live_options: LiveOptions) -> AdapterName:
    """The adapter the plan's SCOPE actually names.

    A registry row's scope encodes its adapter, so deriving it there keeps the
    block self-consistent: ``inspect --adapter sim <an ibkr bot>`` must not print
    ``ibkr_…`` beside ``sim``. ``live_options`` decides only for a freshly-minted
    target, whose scope was derived FROM it. A row whose scope does not parse
    keeps the resolved option: it is still what a spawn would use.
    """
    if target.from_registry:
        parsed = parse_scope(target.scope)
        if parsed is not None:
            return parsed.adapter
    return live_options.adapter


def _config_of(target: BotTarget) -> LiveConfig | None:
    """The target's config, or ``None`` when the recorded path is gone.

    Loaded only to derive the cadence default; a vanished config must not stop
    an operator from inspecting (or stopping) a bot whose row still exists.
    """
    try:
        return load_strat_config(target.config_path)
    except OSError, ValueError:
        return None


__all__ = ["BotPlan", "resolve_plan"]
