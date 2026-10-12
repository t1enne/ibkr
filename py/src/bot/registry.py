"""Bot registry: pid rows plus the liveness surface built on them (plan §4).

The registry has NO database: a row is one JSON file,
``<db>.<scope_tag>.bot.pid`` (scope, pid, started_at, config_path, exit_code),
written via tmp + atomic rename. Liveness is NOT the row — it is the scope lock
(:func:`liveness.probe_live`), so a dead bot's row reads ``exited``/``stale``
with no TTL. ``scope`` is stored IN the row because the filename tag is
one-way, and it is what lets :meth:`BotRegistry.list_bots` name every bot.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd

from src.exec.refs import scope_tag
from src.bot.liveness import (
    Flock,
    PosixFlock,
    acquire_lifetime_fd,
    bot_pid_path,
    bot_stop_flag_path,
    lifetime_lock,
    probe_live,
    spawn_claim,
)
from src.bot.types import BotRecord, BotState

#: Env override for the bot runtime dir (logs). Read at CALL time so tests can
#: repoint it in-process.
_RUNTIME_DIR_ENV = "IBKR_BOT_RUNTIME_DIR"

#: ``py/src/bot/registry.py`` -> ``py/src/bot`` -> ``py/src`` -> ``py``.
_DEFAULT_RUNTIME_DIR = Path(__file__).resolve().parents[2] / "runtime"


def runtime_dir() -> Path:
    """The bot runtime dir: ``IBKR_BOT_RUNTIME_DIR`` when set, else ``runtime/``.

    Resolved file-relative (never ``os.getcwd()``) and evaluated per call so a
    test can redirect it in-process.
    """
    return Path(os.environ.get(_RUNTIME_DIR_ENV) or _DEFAULT_RUNTIME_DIR)


@dataclass(frozen=True)
class PidRecord:
    """The durable identity row one bot owns: who, when, where, last exit."""

    scope: str
    pid: int
    started_at: pd.Timestamp
    config_path: str
    exit_code: int | None


def _row_to_json(record: PidRecord) -> str:
    """Canonical JSON for a row (sorted keys, machine-parseable)."""
    return json.dumps(
        {
            "scope": record.scope,
            "pid": record.pid,
            "started_at": record.started_at.isoformat(),
            "config_path": record.config_path,
            "exit_code": record.exit_code,
        },
        sort_keys=True,
    )


def _row_from_json(text: str) -> PidRecord | None:
    """Parse a row; ``None`` when the payload is missing or not the right shape.

    Lenient on purpose: ``status`` lists a registry an operator or a crash may
    have half-updated, and a bad row must never crash the supervisor.
    """
    try:
        parsed: object = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    raw: dict[str, object] = cast("dict[str, object]", parsed)
    try:
        started_at = pd.Timestamp(_required_str(raw, "started_at"))
        if not isinstance(started_at, pd.Timestamp):
            # A string like "NaT"/"nan" parses to NaT, which is NOT a Timestamp:
            # reject it so a malformed row never reaches ``isoformat()`` downstream.
            raise ValueError("started_at")
        return PidRecord(
            scope=_required_str(raw, "scope"),
            pid=_required_int(raw, "pid"),
            started_at=started_at,
            config_path=_required_str(raw, "config_path"),
            exit_code=_optional_int(raw, "exit_code"),
        )
    except KeyError, TypeError, ValueError:
        return None


def _required_str(raw: dict[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str):
        raise ValueError(key)
    return value


def _required_int(raw: dict[str, object], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(key)
    return value


def _optional_int(raw: dict[str, object], key: str) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    return _required_int(raw, key)


def _atomic_write(target: Path, text: str) -> None:
    """Write *text* via a sibling tmp + rename (readers never see a partial row)."""
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, target)


def read_pid_row(db: str | Path, scope: str) -> PidRecord | None:
    """The row for *scope*, or ``None`` when absent/unreadable/malformed."""
    try:
        return _row_from_json(Path(bot_pid_path(db, scope)).read_text())
    except OSError:
        return None


def write_pid_row(
    db: str | Path,
    scope: str,
    *,
    pid: int,
    started_at: pd.Timestamp,
    config_path: str,
) -> None:
    """Atomically write the scope's pid row.

    A restart flips ``exit_code`` back to ``None``: the new life starts clean,
    and a crash leaves the row stale rather than inheriting the old code.
    """
    record = PidRecord(
        scope=scope,
        pid=pid,
        started_at=started_at,
        config_path=config_path,
        exit_code=None,
    )
    _atomic_write(Path(bot_pid_path(db, scope)), _row_to_json(record))


def write_exit_code(
    db: str | Path,
    scope: str,
    code: int,
    *,
    pid: int,
    started_at: pd.Timestamp,
) -> bool:
    """Stamp *code* onto the row, but ONLY when the row is still THIS life's.

    A restart racing a dying bot must not let the old life clobber the new one:
    the caller proves identity with ``pid`` + ``started_at`` captured when it
    wrote its own row, and a mismatch is a no-op. Returns whether it stamped.
    """
    record = read_pid_row(db, scope)
    if record is None:
        return False
    if record.pid != pid or record.started_at != started_at:
        return False
    updated = PidRecord(
        scope=record.scope,
        pid=record.pid,
        started_at=record.started_at,
        config_path=record.config_path,
        exit_code=code,
    )
    _atomic_write(Path(bot_pid_path(db, scope)), _row_to_json(updated))
    return True


class BotRegistry:
    """Registry surface: probe/list/claim for the supervisor, record helpers
    for the child. One instance = one db + one runtime dir; the flock seam is
    injectable for fakes.
    """

    def __init__(
        self,
        db: str | Path,
        runtime: str | Path | None = None,
        flock: Flock | None = None,
    ) -> None:
        self._db = str(db)
        self._runtime = Path(runtime) if runtime is not None else runtime_dir()
        self._flock: Flock = flock if flock is not None else PosixFlock()

    def probe(self, scope: str) -> BotRecord | None:
        """The scope's current row; ``None`` when the registry holds none."""
        row = read_pid_row(self._db, scope)
        if row is None:
            return None
        return self._record(scope, row)

    def list_bots(self) -> list[BotRecord]:
        """Every bot the db dir holds a row for, sorted by scope.

        The scope comes from the row itself — the filename only carries the
        one-way scope tag — so a row always knows which scope it describes.
        """
        base = Path(self._db)
        found: list[BotRecord] = []
        for pid_file in sorted(base.parent.glob(f"{base.name}.*.bot.pid")):
            try:
                row = _row_from_json(pid_file.read_text())
            except OSError:
                continue
            if row is not None:
                found.append(self._record(row.scope, row))
        return found

    @contextmanager
    def claim(self, scope: str) -> Iterator[None]:
        """Hold the short-lived spawning claim (raises mid-flight starts)."""
        with spawn_claim(self._db, scope, self._flock):
            yield

    @contextmanager
    def hold_lifetime(self, scope: str) -> Iterator[None]:
        """Hold the scope lock for the bot's life (raises if a bot already does)."""
        with lifetime_lock(self._db, scope, self._flock):
            yield

    def acquire_lifetime(self, scope: str) -> int:
        """Lock the scope and return the held fd; keep it open for the life.

        The raw form lets a fake child in tests hold a claim past the call,
        exactly like the real child process does.
        """
        return acquire_lifetime_fd(self._db, scope, self._flock)

    def write_pid_record(
        self, scope: str, *, pid: int, started_at: pd.Timestamp, config_path: str
    ) -> None:
        """Atomically (re)write the scope's pid row."""
        write_pid_row(
            self._db, scope, pid=pid, started_at=started_at, config_path=config_path
        )

    def record_exit_code(
        self,
        scope: str,
        code: int,
        *,
        pid: int,
        started_at: pd.Timestamp,
    ) -> bool:
        """Stamp this life's exit code; a no-op when the row is a newer life's."""
        return write_exit_code(self._db, scope, code, pid=pid, started_at=started_at)

    def log_path(self, scope: str) -> Path:
        """The bot's log: ``<runtime>/<scope_tag>.log``."""
        return self._runtime / f"{scope_tag(scope)}.log"

    def stop_flag_path(self, scope: str) -> Path:
        """The stop flag ``bot stop`` writes; the loop polls this between sleeps."""
        return Path(bot_stop_flag_path(self._db, scope))

    def _record(self, scope: str, row: PidRecord) -> BotRecord:
        """Derive the reportable row: lock held = live, freed = exited/stale."""
        live = probe_live(self._db, scope, self._flock)
        state: BotState = (
            "live" if live else ("exited" if row.exit_code is not None else "stale")
        )
        return BotRecord(
            scope=scope,
            config_path=row.config_path,
            pid=row.pid,
            started_at=row.started_at,
            state=state,
            exit_code=row.exit_code,
        )


__all__ = [
    "BotRegistry",
    "PidRecord",
    "runtime_dir",
]
