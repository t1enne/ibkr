"""The flock+pid primitives behind bot identity (plan §4).

Liveness is an OS advisory ``flock`` on ``<db>.<scope_tag>.bot.lock`` held for
the bot's life: the kernel drops it on ANY exit — clean, crash or SIGKILL — so
"lock held" and "bot alive" cannot drift, unlike a bare pid file. The pid row
is a record and the stop flag lives beside it, never the liveness authority.
Identity files sit BESIDE the database (like the cycle lease) so every process
of one host resolves the SAME path. Single host / local filesystem only.
"""

from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from src.exec.refs import scope_tag


class BotAlreadyRunningError(RuntimeError):
    """Another live bot already holds this scope's lifetime lock."""


class BotStartInProgressError(RuntimeError):
    """Another ``bot start`` currently holds the spawning claim for this scope."""


class Flock(Protocol):
    """libc seam: exclusive advisory lock on an open fd (fake for unit tests)."""

    def acquire(self, fd: int, blocking: bool) -> bool:
        """Try the exclusive lock; ``True`` when this call now holds it.

        Non-blocking returns ``False`` when someone else holds it; blocking
        waits until it is free.
        """
        ...

    def release(self, fd: int) -> None:
        """Drop the exclusive lock this process holds on *fd*."""
        ...


class PosixFlock:
    """Production seam over ``fcntl.flock`` (POSIX advisory, one host only)."""

    def acquire(self, fd: int, blocking: bool) -> bool:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except OSError:
            return False
        return True

    def release(self, fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


_DEFAULT_FLOCK: Flock = PosixFlock()


def _identity_path(db: str | Path, scope: str, kind: str) -> str:
    """ONE derivation for a bot identity file: ``<db>.<scope_tag>.<kind>``.

    The tag comes from ``src.exec.refs.scope_tag`` — the same derivation the
    cycle lease uses — so no second derivation can drift the names apart.
    """
    return f"{db}.{scope_tag(scope)}.{kind}"


def bot_lock_path(db: str | Path, scope: str) -> str:
    """The lifetime lock file: ``<db>.<scope_tag>.bot.lock``."""
    return _identity_path(db, scope, "bot.lock")


def bot_spawn_lock_path(db: str | Path, scope: str) -> str:
    """The spawning claim file: ``<db>.<scope_tag>.bot.spawn.lock``."""
    return _identity_path(db, scope, "bot.spawn.lock")


def bot_pid_path(db: str | Path, scope: str) -> str:
    """The pid row file: ``<db>.<scope_tag>.bot.pid``."""
    return _identity_path(db, scope, "bot.pid")


def bot_stop_flag_path(db: str | Path, scope: str) -> str:
    """The stop flag file: ``<db>.<scope_tag>.bot.stop`` (``bot stop`` writes it)."""
    return _identity_path(db, scope, "bot.stop")


def probe_live(db: str | Path, scope: str, flock: Flock = _DEFAULT_FLOCK) -> bool:
    """The scope's lifetime lock held right now? (a bot is alive).

    Opens WITHOUT ``O_CREAT`` so a probe creates no file. A released lock file
    reads ``False`` — a stale file is harmless.
    """
    try:
        fd = os.open(bot_lock_path(db, scope), os.O_RDWR)
    except OSError:
        return False
    try:
        acquired = flock.acquire(fd, blocking=False)
        if acquired:
            flock.release(fd)
        return not acquired
    finally:
        os.close(fd)


def acquire_lifetime_fd(
    db: str | Path, scope: str, flock: Flock = _DEFAULT_FLOCK
) -> int:
    """Open and exclusively lock the scope, returning the held fd.

    Keep the fd open for the bot's life; the kernel drops the lock on any exit.
    Raises :class:`BotAlreadyRunningError` when another bot holds it.
    """
    path = bot_lock_path(db, scope)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    if not flock.acquire(fd, blocking=False):
        os.close(fd)
        raise BotAlreadyRunningError(f"another bot already holds the lease {path}")
    return fd


@contextmanager
def lifetime_lock(
    db: str | Path, scope: str, flock: Flock = _DEFAULT_FLOCK
) -> Iterator[None]:
    """Context-managed lifetime claim: the scope lock is held until exit."""
    fd = acquire_lifetime_fd(db, scope, flock)
    try:
        yield
    finally:
        flock.release(fd)
        os.close(fd)


@contextmanager
def spawn_claim(
    db: str | Path, scope: str, flock: Flock = _DEFAULT_FLOCK
) -> Iterator[None]:
    """The short-lived spawning claim around probe-and-spawn (§4.1).

    Raises :class:`BotStartInProgressError` while another ``start`` is
    mid-handshake, so two concurrent invocations cannot both spawn. Refuses
    rather than queues: the loser simply retries.
    """
    path = bot_spawn_lock_path(db, scope)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        if not flock.acquire(fd, blocking=False):
            raise BotStartInProgressError(f"another start is in progress for {path}")
        try:
            yield
        finally:
            flock.release(fd)
    finally:
        os.close(fd)


__all__ = [
    "BotAlreadyRunningError",
    "BotStartInProgressError",
    "Flock",
    "PosixFlock",
    "acquire_lifetime_fd",
    "bot_lock_path",
    "bot_pid_path",
    "bot_spawn_lock_path",
    "bot_stop_flag_path",
    "lifetime_lock",
    "probe_live",
    "spawn_claim",
]
