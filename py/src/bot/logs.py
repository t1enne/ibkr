"""Reading a bot's log — the file half of ``ibkr bot logs`` (plan §4).

``docker logs`` semantics against one plain-text file per scope: a tail of the
last N lines, optionally followed as the bot appends. This module only READS —
it creates nothing, locks nothing, spawns nothing — and every wait is a bounded
sleep, so following is interruptible and never a busy spin. The file carries no
per-line timestamps (``loop._log`` writes bare ``cycle exit N``), so nothing
here invents one.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from io import BufferedReader
from pathlib import Path

#: ``docker logs --tail`` default: an operator wants the recent tail, not a
#: months-old file. ``--tail all`` prints the whole thing.
DEFAULT_TAIL = 200

#: Seconds between reads while following — bounded, so Ctrl-C lands promptly.
FOLLOW_POLL_S = 0.25

#: Bytes read per poll; also the cap on one buffered partial line, so a wedged
#: writer cannot make the reader allocate without bound.
MAX_CHUNK_BYTES = 8192

#: One poll of the follow loop: returns ``""`` to keep going, ``None`` to stop
#: (the seam a test uses instead of sleeping forever).
TickFn = Callable[[], "str | None"]


@dataclass(frozen=True)
class LogTail:
    """What one target's log held at read time, plus the offset a follow resumes.

    ``missing`` is the ONE failure shape: a bot that never spawned or has not
    written yet. ``lines`` is then empty and ``path`` says where to look —
    nothing is fabricated.
    """

    config_path: str
    scope: str
    path: Path
    lines: tuple[str, ...]
    offset: int
    missing: bool


def tail_lines(path: Path, limit: int | None) -> tuple[list[str], int]:
    """The last *limit* lines of *path* and the end offset (``None`` = all).

    Only *limit* lines are ever held, so a huge file costs a scan, not a read.
    A missing file is ``([], 0)``; the caller decides how to report it. The
    trailing newline does not mint a phantom empty line.
    """
    assert limit is None or limit >= 0, "a negative tail is a caller bug"
    held: deque[str] = deque(maxlen=limit)
    whole: list[str] = []
    offset = 0
    with path.open("rb") as handle:
        for raw in handle:
            offset += len(raw)
            line = raw.decode("utf-8", "replace").rstrip("\n")
            # ``deque(maxlen=None)`` is EMPTY, not unbounded: ``all`` needs its
            # own accumulator rather than a sentinel for "no cap".
            if limit is None:
                whole.append(line)
            else:
                held.append(line)
    return (whole if limit is None else list(held)), offset


def read_tail(config_path: str, scope: str, path: Path, limit: int | None) -> LogTail:
    """One target's tail; a missing log is reported, never invented."""
    if not path.is_file():
        return LogTail(config_path, scope, path, (), 0, True)
    lines, offset = tail_lines(path, limit)
    return LogTail(config_path, scope, path, tuple(lines), offset, False)


def follow(path: Path, offset: int, tick: TickFn, first: str = "") -> Iterator[str]:
    """Yield appended lines past *offset*, one bounded poll per *tick*.

    ``tick`` is the whole wait: production sleeps :data:`FOLLOW_POLL_S` and
    returns ``""``; a test returns ``""`` at once and ``None`` to end the
    stream. *first* is a partial line the previous read already consumed, which
    is carried so no bytes are lost across the seam and none is printed twice.

    Rotation and truncation look alike — the file is shorter than our position,
    or gone — and both reopen from zero, which is where a rotated file's
    replacement (or a truncated one) starts.
    """
    assert offset >= 0, "an offset is a byte count"
    handle = path.open("rb") if path.is_file() else None
    if handle is not None:
        handle.seek(min(offset, _size(path)))
    pending = first
    try:
        while True:
            if handle is not None:
                text, pending = _read_chunk(handle, pending)
                yield from text.splitlines()
            if tick() is None:
                return
            handle, pending = _reopen(handle, path, pending)
    finally:
        if handle is not None:
            handle.close()


def _read_chunk(handle: BufferedReader, pending: str) -> tuple[str, str]:
    """Whole lines accumulated since the last read, plus the partial tail."""
    assert MAX_CHUNK_BYTES > 0, "a read must be bounded"
    data = handle.read(MAX_CHUNK_BYTES)
    if not data:
        return "", pending
    text = pending + data.decode("utf-8", "replace")
    whole, newline, residue = text.rpartition("\n")
    return (whole + newline if newline else "", residue)


def _reopen(
    handle: BufferedReader | None, path: Path, pending: str
) -> tuple[BufferedReader | None, str]:
    """Re-point at the file, from zero when it shrank or went away."""
    if not path.is_file():
        if handle is not None:
            handle.close()
        return None, ""
    position = handle.tell() if handle is not None else 0
    was_shrunk = position > _size(path)
    if handle is not None:
        handle.close()
    fresh = path.open("rb")
    fresh.seek(0 if was_shrunk else position)
    return fresh, "" if was_shrunk else pending


def _size(path: Path) -> int:
    """The file's byte size, or 0 when it is gone (a rotation is not an error)."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def sleep_tick() -> str:
    """Production tick: wait one bounded poll, then read again."""
    time.sleep(FOLLOW_POLL_S)
    return ""


__all__ = [
    "DEFAULT_TAIL",
    "FOLLOW_POLL_S",
    "LogTail",
    "follow",
    "read_tail",
    "sleep_tick",
    "tail_lines",
]
