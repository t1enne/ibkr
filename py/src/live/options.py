"""Live run options — the resolved knobs a live CLI invocation runs with.

Two layers, mirroring ``[bot]`` (``src.bot.cli._merge_bot_options``):
:class:`src.config.LiveConfig` is the raw, all-optional ``[live]`` table, and
:class:`LiveOptions` is what a command actually consumes — every field RESOLVED,
code defaults baked in. :func:`merge_live_options` is the ONE place those two
meet, so precedence is stated once instead of re-derived per call site:

    CLI flag  >  ``config.toml [live]``  >  code default

Names are validated here rather than in the leaf ``src.config``: that module
deliberately imports nothing from ``src.*``, so it cannot see the adapter or
format vocabularies, and a typo must raise rather than silently trade the wrong
side or emit a format nothing can parse.

``allow_unsafe`` is DELIBERATELY not a field. It forces exit 0 on a cycle the
operator cannot fully see, which is the one outcome a scheduler must never get
by default — so it stays a per-invocation flag and never a file setting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast, get_args

from src.config import LiveConfig
from src.live.scope import AdapterName

#: Every adapter name a scope may name — DERIVED from the ONE typing definition
#: (``src.live.scope.AdapterName``), never a second literal list that can drift.
_ADAPTER_NAMES: tuple[str, ...] = get_args(AdapterName)

#: Report formats a live command can emit.
FormatName = Literal["text", "json"]
_FORMAT_NAMES: tuple[str, ...] = get_args(FormatName)

#: Days before the newest data bar counts as stale, when nothing configures it.
DEFAULT_MAX_AGE_DAYS = 5


def adapter_names() -> tuple[str, ...]:
    """The adapter vocabulary, for a CLI ``click.Choice`` (never a second list)."""
    return _ADAPTER_NAMES


def format_names() -> tuple[str, ...]:
    """The report-format vocabulary, for a CLI ``click.Choice``."""
    return _FORMAT_NAMES


@dataclass(frozen=True)
class LiveOptions:
    """The resolved knobs one live command runs with. No ``None``: defaults baked.

    ``adapter`` names the backend the cycle reads and trades through;
    ``allow_oth`` lets the IBKR broker auto-confirm orders outside NYSE hours;
    ``max_age`` is the staleness gate in days; ``dry_run`` reconciles and reports
    without placing; ``format`` selects the report encoding.
    """

    adapter: AdapterName = "ibkr"
    allow_oth: bool = False
    max_age: int = DEFAULT_MAX_AGE_DAYS
    dry_run: bool = False
    format: FormatName = "text"


def merge_live_options(
    adapter: str | None,
    allow_oth: bool | None,
    cfg: LiveConfig,
    default: LiveOptions,
    *,
    max_age: int | None = None,
    dry_run: bool | None = None,
    fmt: str | None = None,
) -> LiveOptions:
    """CLI flag > ``config.toml [live]`` > code default, one level each.

    Every argument is a RAW CLI value where ``None`` means "flag absent"; the
    boolean flags are therefore already ``False``-vs-``None`` at the call site,
    which is why they take ``None`` here rather than a bare ``bool``. *default*
    carries the code defaults. A name outside its vocabulary raises at EITHER
    layer — a typo in a flag and a typo in the file are the same class of error.
    """
    return LiveOptions(
        adapter=_validate_adapter(_pick(adapter, cfg.adapter, default.adapter)),
        allow_oth=_pick(allow_oth, cfg.allow_oth, default.allow_oth),
        max_age=_pick(max_age, cfg.max_age, default.max_age),
        dry_run=_pick(dry_run, cfg.dry_run, default.dry_run),
        format=_validate_format(_pick(fmt, cfg.format, default.format)),
    )


def _pick[T](flag: T | None, configured: T | None, default: T) -> T:
    """First non-``None`` of flag, configured value, code default."""
    if flag is not None:
        return flag
    if configured is not None:
        return configured
    return default


def _validate_adapter(name: str) -> AdapterName:
    """``name`` as an ``AdapterName``, or raise naming the legal adapters."""
    if name not in _ADAPTER_NAMES:
        raise ValueError(
            f"adapter must be one of {sorted(_ADAPTER_NAMES)}, got {name!r}"
        )
    return cast("AdapterName", name)


def _validate_format(name: str) -> FormatName:
    """``name`` as a ``FormatName``, or raise naming the legal formats."""
    if name not in _FORMAT_NAMES:
        raise ValueError(f"format must be one of {sorted(_FORMAT_NAMES)}, got {name!r}")
    return cast("FormatName", name)


__all__ = [
    "DEFAULT_MAX_AGE_DAYS",
    "FormatName",
    "LiveOptions",
    "adapter_names",
    "format_names",
    "merge_live_options",
]
