"""Cross-process mutual exclusion for one live cycle (the concurrency lease).

A cron trigger and a human ``ibkr live run`` can overlap: both read the SAME
pre-order book (``load_book``) and both compute the same intents from it, so both
place. One cycle at a time per database is enforced with an OS advisory lock
(``flock``) held for the cycle's duration.

The kernel releases an ``flock`` when the fd's process exits — including a
crash or ``SIGKILL`` — so a dead run can NEVER wedge live trading. That is why
there is no TTL and no takeover rule: a stale *file* is harmless (the lock is the
open fd, not the file's existence), so a leftover lock file is never a held
lease. A second cycle that cannot take the lock REFUSES to start with
:class:`CycleInProgressError` rather than trading on a book another cycle is
mid-way through.
"""

from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager


class CycleInProgressError(RuntimeError):
    """Another live cycle already holds the lease on this database."""


@contextmanager
def file_lease(lock_path: str) -> Iterator[None]:
    """Exclusive, non-blocking advisory lock on *lock_path* for the block's scope.

    Refuses (raises) rather than blocking: a queued cycle would place a decision
    computed against a book that has since moved.
    """
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


__all__ = ["CycleInProgressError", "file_lease"]
