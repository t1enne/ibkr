"""Cross-process mutual exclusion for one live cycle (the concurrency lease).

A cron trigger and a human ``ibkr live run`` can overlap: both read the SAME
pre-order book (``load_book``) and both compute the same intents from it, so both
place. One cycle at a time PER SCOPE is enforced with an OS advisory lock
(``flock``) held for the cycle's duration. Per scope (not per database) because
two adapters — ``ibkr`` and ``sim`` — or two configs must be able to run
CONCURRENTLY: they own disjoint books and disjoint cOID prefixes, so sharing one
lock would block work that cannot conflict.

The kernel releases an ``flock`` when the fd's process exits — including a
crash or ``SIGKILL`` — so a dead run can NEVER wedge live trading. That is why
there is no TTL and no takeover rule: a stale *file* is harmless (the lock is the
open fd, not the file's existence), so a leftover lock file is never a held
lease. A second cycle that cannot take the lock REFUSES to start with
:class:`CycleInProgressError` rather than trading on a book another cycle is
mid-way through.

CAVEAT — single host / local filesystem only. ``flock`` is meaningful between
processes on one machine; on NFS or across hosts it is advisory at best, so run
one host per database and keep the DB on local disk.
"""

from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class CycleInProgressError(RuntimeError):
    """Another live cycle already holds the lease on this scope."""


def lease_path(db: str | Path, scope_tag: str) -> str:
    """The per-scope lock file path: ``<db>.<scope_tag>.cycle.lock``.

    Kept beside the database (not in a temp dir) so every process of one host
    that can see the DB resolves the SAME path — that, not the filename, is what
    makes the lock mutual.
    """
    return f"{db}.{scope_tag}.cycle.lock"


@contextmanager
def file_lease(db: str | Path, scope_tag: str) -> Iterator[None]:
    """Exclusive, non-blocking advisory lock for *scope_tag* on database *db*.

    Refuses (raises) rather than blocking: a queued cycle would place a decision
    computed against a book that has since moved.
    """
    lock_path = lease_path(db, scope_tag)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise CycleInProgressError(
                f"another live cycle holds the lease {lock_path}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


__all__ = ["CycleInProgressError", "file_lease", "lease_path"]
