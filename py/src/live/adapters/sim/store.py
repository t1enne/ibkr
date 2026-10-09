"""JSON-backed sim broker book — an operator-editable IBKR-shaped store.

The sim ACCOUNT book lives here, ONE JSON file keyed by scope. The engine reads
it to build the account side of the divergence oracle; ``place_cohort`` writes it
back so a placed fill lands in the same file an operator can hand-edit. Our OWN
record stays the sqlite fill fold (``live_execution`` / ``live_position
source='executions'``) — this file only ever mimics the broker's report.

The shape mirrors what IBKR's portfolio endpoints return, with ONE sim extension:
each position row carries ``position_id`` (the per-lot handle the fold keys on),
because IBKR nets by ``conid`` while the sim book is per-lot.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, cast

from src.db.path import resolve_live_db_path
from src.live.types import synthetic_conid

#: The env override naming the sim broker JSON file (tests set this).
_PATH_ENV = "IBKR_SIM_BROKER_PATH"


def resolve_sim_book_path(override: str | Path | None = None) -> Path:
    """The sim broker JSON path: *override*, else ``IBKR_SIM_BROKER_PATH``,
    else ``<live db dir>/sim_broker.json``."""
    if override is not None:
        return Path(override)
    env = os.environ.get(_PATH_ENV)
    if env:
        return Path(env)
    return resolve_live_db_path().parent / "sim_broker.json"


@dataclass(frozen=True)
class SimBook:
    """One scope's broker-shaped book, read from (or the empty default for) JSON."""

    account: str = ""
    summary: Mapping[str, object] = field(default_factory=dict)
    positions: tuple[Mapping[str, object], ...] = ()
    orders: tuple[Mapping[str, object], ...] = ()
    trades: tuple[Mapping[str, object], ...] = ()


@dataclass(frozen=True)
class SimBookStore:
    """The JSON file holding every scope's sim book, keyed ``{"scopes": {...}}``."""

    path: Path

    def read(self, scope: str) -> SimBook:
        """The scope's book, or the empty default when the file/scope/JSON is absent.

        NEVER creates the file: a missing or malformed file reads as an empty
        book (a never-written scope is a flat funded book downstream).
        """
        try:
            raw = json.loads(self.path.read_text())
        except OSError, ValueError:
            return SimBook()
        if not isinstance(raw, Mapping):
            return SimBook()
        scopes = raw.get("scopes")
        if not isinstance(scopes, Mapping):
            return SimBook()
        body = scopes.get(scope)
        if not isinstance(body, Mapping):
            return SimBook()
        return _book_of(body)

    def write(self, scope: str, book: SimBook) -> None:
        """Upsert *scope*'s book atomically (tmp file in the same dir, ``os.replace``)."""
        existing: Mapping[str, object] = {}
        try:
            raw = json.loads(self.path.read_text())
            if isinstance(raw, Mapping) and isinstance(raw.get("scopes"), Mapping):
                existing = cast("Mapping[str, object]", raw["scopes"])
        except OSError, ValueError:
            existing = {}
        scopes = dict(existing)
        scopes[scope] = _body_of(book)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps({"scopes": scopes}, indent=2))
        os.replace(tmp, self.path)


def _book_of(body: Mapping[str, object]) -> SimBook:
    """Narrow a scope body to a ``SimBook`` (best-effort, empty on any miss)."""
    return SimBook(
        account=str(body.get("account", "")),
        summary=_mapping(body.get("summary")),
        positions=_list(body.get("positions")),
        orders=_list(body.get("orders")),
        trades=_list(body.get("trades")),
    )


def _body_of(book: SimBook) -> dict[str, object]:
    """A ``SimBook`` as the scope-body mapping the file stores."""
    return {
        "account": book.account,
        "summary": dict(book.summary),
        "positions": [dict(row) for row in book.positions],
        "orders": [dict(row) for row in book.orders],
        "trades": [dict(row) for row in book.trades],
    }


def _mapping(value: object) -> Mapping[str, object]:
    """Narrow to a mapping (JSON objects only), else empty."""
    return cast("Mapping[str, object]", value) if isinstance(value, Mapping) else {}


def _list(value: object) -> tuple[Mapping[str, object], ...]:
    """Narrow to a tuple of mappings, else empty."""
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(
        cast("Mapping[str, object]", row) for row in value if isinstance(row, Mapping)
    )


__all__ = [
    "SimBook",
    "SimBookStore",
    "resolve_sim_book_path",
    "synthetic_conid",
]
