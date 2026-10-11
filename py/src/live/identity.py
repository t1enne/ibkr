"""Order identity — the bar-free cOID scheme, with NO durable open-state store.

The broker is the single source of truth for whether an order exists or may be
placed again: the union of the broker's orders read and positions read covers
every order we placed (working, session-filled, prior-session-filled, partial),
so an intent absent from both reads does not exist and re-minting is safe. There
is therefore no ``live_order_intent`` row, no attempt counter, no wedged-key
alarm: a pure (bar-free, drift-free) ref is all identity needs:

    cOID = f\"{scope_tag(scope)}-{token}\"

``token`` is the full 32-bit crc32 of ``symbol|action|position_id`` rendered as
eight hex digits — a pure function of the key, so the SAME intent always mints
the same ref (a re-send reuses it and IBKR's own ``cOID`` dedupe absorbs it).
The decision bar, the batch membership and the wall clock are NOT in the ref.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from enum import Enum

from src.bt.state import ActionType
from src.exec.refs import scope_tag
from src.live.types import OrderIntent


@dataclass(frozen=True)
class IntentKey:
    """The bar-free identity of an order: ``(scope, symbol, action, position_id)``.

    ``position_id`` is ``None`` for an open, the lot's conid for a close. Two
    keys are equal exactly when they name the same order; the same key always
    mints the same ``cOID`` (see :func:`order_ref`), so a re-send is absorbed by
    the broker's ``cOID`` dedupe while a distinct order is distinct by key.
    """

    scope: str
    symbol: str
    action: ActionType
    position_id: str | None

    def token(self) -> str:
        """Stable eight-hex-digit token: crc32 of ``symbol|action|position_id``.

        A pure function of the key, so it does not depend on the decision bar,
        the batch membership, or the wall clock. The full 32-bit crc32 is used
        (never a ``%`` reduction) so a durable id cannot shift with the batch.
        """
        identity = f"{self.symbol}|{self.action.value}|{self.position_id or ''}"
        return f"{zlib.crc32(identity.encode('utf-8')):08x}"


def intent_key(scope: str, intent: OrderIntent) -> IntentKey:
    """The bar-free key for *intent* under *scope* (pure)."""
    return IntentKey(
        scope=scope,
        symbol=intent.symbol,
        action=intent.action,
        position_id=intent.position_id,
    )


def ref_prefix(key: IntentKey) -> str:
    """The exact prefix every cOID for *key* carries: the pure ref itself.

    With no attempt tail the whole ref IS the prefix, so an exact
    ``startswith(ref_prefix(key))`` matches our order for the key and nothing
    else — a foreign order (another scope, or a human) can never be adopted.
    """
    return order_ref(key)


def order_ref(key: IntentKey) -> str:
    """The cOID for *key*: ``scope_tag-token`` (bar-free, no attempt tail).

    Same key always mints the same ref, so a duplicate re-send is absorbed by the
    broker's ``cOID`` dedupe rather than swallowed-then-replaced.
    """
    return f"{scope_tag(key.scope)}-{key.token()}"


def ref_is_ours(scope: str, coid: str) -> bool:
    """Whether *coid* was minted by *scope* (its scope segment equals the tag).

    A pure ref is ``scope_tag-token``; the token is the ONE dashless crc32 tail,
    so the scope segment is the ref with that single tail removed. Matching the
    whole ``scope_tag`` exactly keeps scope ``momentum`` from claiming
    ``momentum-v2``'s refs, and the non-collapsing ``scope_tag`` keeps two scopes
    that share a ``slug`` distinct.
    """
    return coid.rsplit("-", 1)[0] == scope_tag(scope)


@dataclass(frozen=True)
class WorkingOrder:
    """One order read from the broker's open-orders endpoint."""

    order_ref: str
    order_id: str
    conid: int
    symbol: str
    side: str
    status: str
    filled_qty: float


class OrderOutcome(Enum):
    """The disposition of a placement attempt, carried on ``OrderResult``.

    There is no durable open-state row, so an ``UNRESOLVED``/``WEDGED`` state is
    a per-cycle report marker resolved by the next read — the broker's rows and
    the account positions are the truth, never a remembered row.
    """

    PLACED = "placed"
    ADOPTED = "adopted"
    REJECTED = "rejected"
    UNFILLED = "unfilled"
    TIMEOUT = "timeout"
    UNRESOLVED = "unresolved"
    WEDGED = "wedged"  # retained for the report vocabulary; never emitted now
    #: An OPEN refused because the account net and the ledger's booked exposure on
    #: the conid disagree: a fill we cannot see may be live, or our book is ahead
    #: of the account. Distinct from a broker ``rejected`` and from an ``unresolved``.
    DIVERGENCE = "divergence"
    #: An OPEN dropped by a STRUCTURAL rule: the cohort cannot state a shared
    #: cash bound, or a scaled qty floors to 0. Unlike a ``REJECTED`` open —
    #: genuine cash exhaustion for THIS bar, re-minted next cycle at exit 0 — this
    #: recurs every cycle forever, so it is UNSAFE.
    UNFUNDED = "unfunded"


__all__ = [
    "IntentKey",
    "OrderOutcome",
    "WorkingOrder",
    "intent_key",
    "order_ref",
    "ref_is_ours",
    "ref_prefix",
]
