"""Bot domain types — the supervisor/registry contract (plan §4).

:class:`BotRecord` is one scope's registry row; :class:`BotLiveness` is the
registry surface ``start`` tunes against; :class:`BotSpawner` is the detached
process seam tests fake. No trading and no database here: a bot's identity is
an OS lock plus a pid record.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import pandas as pd

#: A bot's reportable state: lock held = live; freed lock with a recorded exit
#: code = exited; freed lock without one = stale.
BotState = Literal["live", "exited", "stale"]


@dataclass(frozen=True)
class BotRecord:
    """One scope's registry row: identity plus the derived liveness state.

    ``state`` is DERIVED at read time from the scope lock (an OS flock the
    kernel drops on process exit), never stored, so a stale pid row can never
    read as ``live``.
    """

    scope: str
    config_path: str
    pid: int
    started_at: pd.Timestamp
    state: BotState
    exit_code: int | None


class BotLiveness(Protocol):
    """The registry surface the supervisor consumes (plan §4): liveness probes,
    the spawning claim, and the spawn side-path (log/stop-flag paths)."""

    def probe(self, scope: str) -> BotRecord | None:
        """The current row for *scope*, or ``None`` when the registry has none.

        ``live`` means the scope lock is currently HELD; ``exited``/``stale``
        mean it was released and the pid row records (or lacks) an exit code.
        """
        ...

    def claim(self, scope: str) -> AbstractContextManager[None]:
        """Hold the short-lived spawning claim around probe-and-spawn.

        Raises if another ``start`` is already mid-handshake for this scope, so
        two concurrent invocations cannot both spawn a second bot.
        """
        ...

    def log_path(self, scope: str) -> Path:
        """Where the bot's output is appended: ``<runtime>/<scope_tag>.log``."""
        ...

    def stop_flag_path(self, scope: str) -> Path:
        """The stop flag path ``bot stop`` writes for *scope*."""
        ...


class BotSpawner(Protocol):
    """Detaches a bot process; ``start`` waits for its ready signal after."""

    def spawn(self, argv: Sequence[str], log_path: Path) -> int:
        """Start *argv* detached with stdout/stderr into *log_path*; return pid.

        The argv is the hidden ``bot _run`` invocation; the ready pipe rides in
        as an inherited fd that the spawner must pass through unclosed.
        """
        ...


__all__ = [
    "BotLiveness",
    "BotRecord",
    "BotSpawner",
    "BotState",
]
