"""The bot log line: one format, one parser, two renderers (plan §3).

The file is the truth: every line the bot writes carries its own UTC timestamp,
level and dotted event name, so ``bot logs`` reads a real time instead of the
file's mtime and ``tail -f`` can separate a transient from an escalation.
``scope`` and ``pid`` are ordinary fields on every line the bot writes, so
interleaved output stays self-identifying; a child traceback shares the file but
is not written by the bot.

Nothing here touches the filesystem — :mod:`src.bot.journal` appends, this
module only decides what the bytes mean. A legacy bare ``cycle exit N`` line
does not parse and is passed through verbatim, so a mid-migration deployment
(and a historical file) still reads.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

import pandas as pd

#: The level vocabulary. Ordered from noisiest to deadliest.
type Level = Literal["DEBUG", "INFO", "WARN", "ERROR"]

#: The output encoding every ``bot`` command group accepts.
type OutputFormat = Literal["text", "json"]

#: Size at which :func:`src.bot.journal.rotate` rolls one log file over.
LOG_MAX_BYTES_ENV = "IBKR_BOT_LOG_MAX_BYTES"
#: How many rotated generations are kept beside the live log.
LOG_KEEP_ENV = "IBKR_BOT_LOG_KEEP"
#: The minimum level a resident bot writes to its own file.
LOG_LEVEL_ENV = "IBKR_BOT_LOG_LEVEL"

#: Defaults as the plan states them; env vars override, ``0`` disables rotation.
DEFAULT_LOG_MAX_BYTES = 8 * 1024 * 1024
DEFAULT_LOG_KEEP = 3
DEFAULT_LOG_LEVEL: Level = "INFO"

_LEVELS: tuple[Level, ...] = ("DEBUG", "INFO", "WARN", "ERROR")

#: Longest event name a line may carry; anything else is a caller bug.
_MAX_EVENT = 64

#: A key is a field name: no space, no ``=``, no quote, no control byte.
_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")

#: ``<ts> <LEVEL> <event> k=v…`` — the level is a fixed vocabulary word, so a
#: legacy ``cycle exit 0`` line can never be mistaken for a structured one.
_LINE_RE = re.compile(
    r"^(?P<at>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})Z "
    r"(?P<level>DEBUG|INFO|WARN|ERROR) "
    r"(?P<event>[A-Za-z_][A-Za-z0-9_.-]*)"
    r"(?: (?P<fields>.*))?$"
)


@dataclass(frozen=True)
class LogLine:
    """One parsed log line: the only shape any consumer of a bot log sees."""

    at: pd.Timestamp
    level: Level
    event: str
    fields: Mapping[str, str]


def render_line(line: LogLine) -> str:
    """The canonical one-line form of *line* — what the writer appends."""
    parts = [
        f"{_stamp(line.at)} {level_of(line.level)} {event_of(line.event)}",
    ]
    for key, value in line.fields.items():
        parts.append(f"{_key_of(key)}={_quote(value)}")
    return " ".join(parts)


def parse_line(text: str) -> LogLine | None:
    """*text* as a :class:`LogLine`, or ``None`` when it is not one of ours.

    ``None`` is the ONE "not structured" answer: a legacy ``cycle exit N`` line,
    a child traceback and a bare gateway warning all read as ``None`` and are
    passed through by the reader rather than guessed at.
    """
    match = _LINE_RE.match(text.rstrip("\r\n"))
    if match is None:
        return None
    at = _stamp_of(match.group("at"))
    if at is None:
        return None
    return LogLine(
        at=at,
        level=cast("Level", match.group("level")),
        event=match.group("event"),
        fields=_fields_of(match.group("fields") or ""),
    )


def line_to_json(line: LogLine) -> dict[str, object]:
    """*line* as a JSON-ready object: the timestamp, then level/event/fields."""
    return {
        "at": _stamp(line.at),
        "level": line.level,
        "event": line.event,
        "fields": dict(line.fields),
    }


def resolve_log_level(raw: str | None) -> Level:
    """*raw* as a :class:`Level`; an empty/absent value means the default.

    A name outside the vocabulary raises — a typo in ``IBKR_BOT_LOG_LEVEL``
    must fail loudly rather than silently fall back to a level the operator did
    not pick.
    """
    if raw is None or not raw.strip():
        return DEFAULT_LOG_LEVEL
    name = raw.strip().upper()
    if name not in _LEVELS:
        raise ValueError(f"log level must be one of {list(_LEVELS)}, got {raw!r}")
    return name


def level_at_least(level: Level, minimum: Level) -> bool:
    """Whether *level* is at or above *minimum* (DEBUG is the lowest)."""
    return _LEVELS.index(level) >= _LEVELS.index(minimum)


def resolve_log_max_bytes(raw: str | None) -> int:
    """``IBKR_BOT_LOG_MAX_BYTES`` as a byte count; ``0`` disables rotation."""
    return _env_int(raw, DEFAULT_LOG_MAX_BYTES, LOG_MAX_BYTES_ENV)


def resolve_log_keep(raw: str | None) -> int:
    """``IBKR_BOT_LOG_KEEP`` as a generation count; ``0`` keeps none."""
    return _env_int(raw, DEFAULT_LOG_KEEP, LOG_KEEP_ENV)


def _env_int(raw: str | None, default: int, name: str) -> int:
    """*raw* as a non-negative int, or *default* when it is absent."""
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def _stamp(at: pd.Timestamp) -> str:
    """*at* as ``YYYY-MM-DDTHH:MM:SSZ`` — UTC, whole seconds, always ``Z``.

    Truncated, never rounded: a line can read up to a second early but can never
    claim a time the process had not reached.
    """
    if at.tzinfo is None:
        at = at.tz_localize("UTC")
    return at.tz_convert("UTC").floor("s").strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp_of(text: str) -> pd.Timestamp | None:
    """The UTC timestamp a rendered ``<ts>`` string holds, or ``None``.

    ``errors="coerce"`` is load-bearing: the format accepts digit-pair fields, so
    ``2026-99-99T00:00:00Z`` matches the line regex but is not a real instant.
    Coercing turns it into ``NaT`` here instead of raising, so one garbled line
    cannot take down ``logs`` for the whole file.
    """
    parsed = pd.to_datetime(
        f"{text}Z", utc=True, format="%Y-%m-%dT%H:%M:%SZ", errors="coerce"
    )
    if pd.isna(parsed):
        return None
    return parsed


def _quote(value: str) -> str:
    """*value* bare, or double-quoted (escaped) when a bare form would not parse.

    Empty renders as ``-`` (the documented absent sentinel), and an empty string
    is rendered that way too. A LITERAL ``-`` must therefore be quoted: bare it
    would parse back as absent, so the round trip would silently lose a value.
    """
    if value == "":
        return "-"
    if value == "-" or not _bare_ok(value):
        return json.dumps(value)
    return value


def _bare_ok(text: str) -> bool:
    """Whether *text* survives a bare round trip: no space, no quote, no escape."""
    if '"' in text or "\\" in text:
        return False
    return not any(char.isspace() for char in text)


def _key_of(key: str) -> str:
    """*key* after checking it is a field name (a caller asserts, not guesses)."""
    if not _KEY_RE.match(key):
        raise ValueError(f"log field name {key!r} is not a bare key")
    return key


def _fields_of(raw: str) -> dict[str, str]:
    """The ``k=v`` pairs of one line's field tail (tolerant: unparseable tail
    tokens become a ``rest`` field rather than losing the line)."""
    fields: dict[str, str] = {}
    rest: list[str] = []
    for token in _split_tokens(raw):
        key, sep, value = token.partition("=")
        if not sep or not _KEY_RE.match(key):
            rest.append(token)
            continue
        fields[key] = _unquote(value)
    if rest:
        fields["rest"] = " ".join(rest)
    return fields


def _split_tokens(raw: str) -> list[str]:
    """Field tokens, keeping a quoted value (spaces and escapes included) whole."""
    tokens: list[str] = []
    position = 0
    length = len(raw)
    while position < length:
        if raw[position] == " ":
            position += 1
            continue
        start = position
        quoted = False
        escaped = False
        while position < length:
            char = raw[position]
            if escaped:
                escaped = False
            elif char == "\\" and quoted:
                escaped = True
            elif char == '"':
                quoted = not quoted
            elif char == " " and not quoted:
                break
            position += 1
        tokens.append(raw[start:position])
    return tokens


def _unquote(value: str) -> str:
    """*value* with the quoting :func:`_quote` applied removed (``-`` = absent)."""
    if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return value
        return decoded if isinstance(decoded, str) else value
    return "" if value == "-" else value


def env_of(name: str) -> str | None:
    """``os.environ``'s value for *name* — read at CALL time so tests can repoint."""
    return os.environ.get(name)


def event_of(event: str) -> str:
    """*event* after asserting it is a dotted, additive-only name."""
    if not _KEY_RE.match(event) or len(event) > _MAX_EVENT:
        raise ValueError(f"log event name {event!r} is not a dotted name")
    return event


def level_of(level: str) -> Level:
    """*level* as a :class:`Level`, raising on a name outside the vocabulary."""
    if level not in _LEVELS:
        raise ValueError(f"log level must be one of {list(_LEVELS)}, got {level!r}")
    return level


__all__ = [
    "DEFAULT_LOG_KEEP",
    "DEFAULT_LOG_LEVEL",
    "DEFAULT_LOG_MAX_BYTES",
    "LOG_KEEP_ENV",
    "LOG_LEVEL_ENV",
    "LOG_MAX_BYTES_ENV",
    "Level",
    "LogLine",
    "OutputFormat",
    "env_of",
    "event_of",
    "level_at_least",
    "level_of",
    "line_to_json",
    "parse_line",
    "render_line",
    "resolve_log_keep",
    "resolve_log_level",
    "resolve_log_max_bytes",
]
