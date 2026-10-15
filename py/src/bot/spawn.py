"""Detached bot spawning with the ready-pipe handshake (plan §4.1).

``start`` builds the hidden ``bot _run`` child argv, points stdout/stderr at
the bot's log, and ``Popen``s it in a new session so the bot survives the
supervisor. The ready pipe rides in as an inherited fd so the parent can WAIT
for the child to claim its scope lock before reporting ``live``: the child
writes ``b"1"`` after claiming, ``b"0"`` when the claim failed.
"""

from __future__ import annotations

import os
import select
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

#: How long ``start`` waits for the child's ready byte before reporting
#: failure. Generous by design: the child boots the full CLI before it can
#: claim and signal.
READY_TIMEOUT_S = 60.0


class PopenSpawner:
    """Production spawner: a detached process, output appended to its log."""

    def spawn(self, argv: Sequence[str], log_path: Path) -> int:
        """Start *argv* detached; return its pid (the parent does NOT wait).

        stdout/stderr append to *log_path* (parent dirs created); the ready fd is
        inherited via ``pass_fds``; stdin is /dev/null.
        """
        ready_fd = ready_fd_of(argv)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as log:
            proc = subprocess.Popen(
                list(argv),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(ready_fd,) if ready_fd is not None else (),
            )
        return proc.pid


def child_argv(
    config_path: str,
    *,
    ready_fd: int | None,
    adapter: str,
    interval: int | None = None,
    max_age: int | None = None,
    no_sync: bool = False,
    retries: int | None = None,
) -> list[str]:
    """The hidden child invocation for one config.

    Uses our own interpreter + the repo's ``main.py`` by absolute path (the
    script's directory lands on ``sys.path``). ``adapter`` is the RESOLVED name
    so parent and child mint the identical scope. Cadence/retry knobs are
    appended only when set, so an unset default stays the CHILD's derivation
    from ``bars[0]``.
    """
    main_script = Path(__file__).resolve().parents[2] / "main.py"
    argv = [
        sys.executable,
        str(main_script),
        "bot",
        "_run",
        str(Path(config_path).resolve()),
    ]
    argv += ["--adapter", adapter]
    if ready_fd is not None:
        argv += ["--ready-fd", str(ready_fd)]
    if interval is not None:
        argv += ["--interval", str(interval)]
    if max_age is not None:
        argv += ["--max-age", str(max_age)]
    if no_sync:
        argv += ["--no-sync"]
    if retries is not None:
        argv += ["--retries", str(retries)]
    return argv


def _flag_value(argv: Sequence[str], name: str) -> str | None:
    """The value of the ``--name <value>`` pair in a child argv, if present."""
    for index, token in enumerate(argv):
        if token == name and index + 1 < len(argv):
            return argv[index + 1]
    return None


def ready_fd_of(argv: Sequence[str]) -> int | None:
    """The ready-pipe fd a child argv carries, or ``None`` when absent.

    ONE parser shared by the spawner (``pass_fds``), the child entry and test
    fakes, so all three agree on its shape.
    """
    value = _flag_value(argv, "--ready-fd")
    return int(value) if value is not None else None


def adapter_flag_of(argv: Sequence[str]) -> str | None:
    """The resolved adapter name a child argv carries (scope consistency)."""
    return _flag_value(argv, "--adapter")


def run_config_of(argv: Sequence[str]) -> str | None:
    """The config token the hidden ``_run`` entry was invoked with."""
    for index, token in enumerate(argv):
        if token == "_run" and index + 1 < len(argv):
            return argv[index + 1]
    return None


def wait_ready(read_fd: int, timeout_s: float = READY_TIMEOUT_S) -> bool:
    """Whether the child signalled ``b"1"`` on *read_fd* within the deadline.

    ``b"0"`` (claim failed) and EOF (dead child) both read as failure, so a
    child that dies before claiming cannot hang the parent past the deadline.
    """
    readable, _, _ = select.select([read_fd], [], [], timeout_s)
    if not readable:
        return False
    return os.read(read_fd, 1) == b"1"


__all__ = [
    "PopenSpawner",
    "READY_TIMEOUT_S",
    "adapter_flag_of",
    "child_argv",
    "ready_fd_of",
    "run_config_of",
    "wait_ready",
]
