"""Order identity + pending-intent state — the ONE durable owner of open state.

The live path's identity scheme, replacing the decision-bar-anchored ``cOID``
the previous iterations used. A ref embeds a *bar-free* key token so adoption
survives bars, days and reruns:

    cOID = f"{scope_tag(scope)}-{key.token()}-{attempt:02x}"

``key`` is ``(scope, symbol, action, position_id)`` — ``position_id`` is ``None``
for an open and the lot's conid for a close. The token is the full 32-bit crc32
of ``symbol|action|position_id`` rendered as eight hex digits; it is a pure
function of the key, so the SAME intent always mints the same prefix. The
``attempt`` counter is what distinguishes a legitimate re-send (a close on a lot
re-closed after a partial fill) from a duplicate: it changes the cOID, so IBKR's
own dedupe cannot swallow the second, intended order.

The decision bar is NOT in the ref. It is recorded in the pending-intent table
(``IntentRecord.decision_ts``) for audit only, which is what makes duplicate
detection independent of the bar.

``PendingIntents`` is the durable seam: the only record of an unresolved intent.
Its implementation (``SqliteLedger``) owns state; the broker reads it before
minting anything. A close on the same key after a terminal state mints a NEW
attempt; an open intent that is still OPEN never re-mints.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Protocol

import pandas as pd

from src.bt.state import ActionType
from src.exec.refs import scope_tag
from src.live.types import OrderIntent


@dataclass(frozen=True)
class IntentKey:
    """The bar-free identity of an intent: ``(scope, symbol, action, position_id)``.

    ``position_id`` is ``None`` for an open, the lot's conid for a close. Two
    keys are equal exactly when they name the same trade to make; the attempt
    counter (not the key) separates a re-send.
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
    """The exact prefix every cOID for *key* carries: ``scope_tag-token-``.

    The scope is tagged with :func:`scope_tag` (broker-safe AND non-collapsing, so
    two scopes that share a ``slug`` do not share an owner) and the token is
    dashless, so ``trades.is_ours``'s ``rsplit("-", 2)[0]`` still yields the whole
    tag for attribution.
    """
    return f"{scope_tag(key.scope)}-{key.token()}-"


def order_ref(key: IntentKey, attempt: int) -> str:
    """The cOID for *key*'s *attempt*: ``scope_tag-token-attempt`` (bar-free)."""
    return f"{scope_tag(key.scope)}-{key.token()}-{attempt:02x}"


def ref_is_ours(scope: str, coid: str) -> bool:
    """Whether *coid* was minted by *scope* (its scope segment equals the tag).

    Mirrors ``trades.is_ours``: the token and attempt are dashless, so the scope
    segment is the ref minus its last two ``-``-separated parts. An exact match
    keeps scope ``momentum`` from claiming ``momentum-v2``'s refs, and the
    non-collapsing ``scope_tag`` keeps two scopes with one ``slug`` distinct.
    """
    return coid.rsplit("-", 2)[0] == scope_tag(scope)


def ref_matches_key(scope: str, key: IntentKey, coid: str) -> bool:
    """Whether *coid* was minted for *key* under *scope* (tag AND token match).

    The ref carries no bar, so this attributes a ref from ANY attempt of *key*
    to that key — the property the executions sweep relies on to catch a
    predecessor order whose durable row was lost.
    """
    if not ref_is_ours(scope, coid):
        return False
    parts = coid.rsplit("-", 2)
    return len(parts) == 3 and parts[1] == key.token()


def attempt_of(order_ref: str) -> int | None:
    """The attempt counter in *order_ref*'s dashless hex tail, or ``None``.

    ``None`` means the tail is not readable hex, which is distinct from a
    genuine attempt ``0``.
    """
    try:
        return int(order_ref.rsplit("-", 1)[-1], 16)
    except ValueError:
        return None


#: The time-in-force every live ticket carries today (an MKT order has no
#: resting life beyond its session). Persisted on the record so the day-roll
#: expiry inference names the rule rather than assuming it.
DEFAULT_TIF = "DAY"


class IntentState(Enum):
    """The lifecycle of one intent's durable record (persisted on every change)."""

    PENDING = "pending"  # open_attempt written, not yet submitted
    WORKING = "working"  # adopted or accepted by the broker, not yet settled
    UNRESOLVED = "unresolved"  # ambiguous submit/confirm; whether it is live is unknown
    FILLED = "filled"  # terminal, filled
    UNFILLED = "unfilled"  # terminal, cancelled/expired with no (or partial) fill
    REJECTED = "rejected"  # terminal, refused by the broker


#: The non-terminal states: an intent in one of these still needs reconciliation
#: before a new order for its key may be minted.
OPEN_STATES: frozenset[IntentState] = frozenset(
    {IntentState.PENDING, IntentState.WORKING, IntentState.UNRESOLVED}
)


@dataclass(frozen=True)
class IntentRecord:
    """The durable row for one intent key: state, attempt, ref and order id."""

    key: IntentKey
    state: IntentState
    attempt: int
    order_ref: str
    order_id: str | None
    decision_ts: pd.Timestamp | None
    #: The time-in-force the order was placed with. Persisted so the DAY-rollover
    #: expiry rule is explicit and stays correct if GTC is ever introduced (only a
    #: DAY order cannot survive its session). Defaulted for older rows.
    tif: str = DEFAULT_TIF
    #: Consecutive cycle-start resyncs that left this OPEN record unresolved (no
    #: working order, status unreadable or still non-terminal). A wedged key is
    #: surfaced in the report once this reaches ``WEDGED_CYCLES``.
    stuck_cycles: int = 0


@dataclass(frozen=True)
class WorkingOrder:
    """One working order read from the broker's open-orders endpoint."""

    order_ref: str
    order_id: str
    conid: int
    symbol: str
    side: str
    status: str
    filled_qty: float


@dataclass(frozen=True)
class Resolution:
    """What a pre-flight decided for one intent: adopt, submit, or skip (no submit).

    ``skip`` is a refusal, never a submit: the caller reports it ``unresolved``
    (a known order id with no working order is not proof the order is gone).
    """

    kind: Literal["adopt", "submit", "skip"]
    order_id: str | None
    reason: str


class OrderOutcome(Enum):
    """The disposition of a placement attempt, carried on ``OrderResult``."""

    PLACED = "placed"
    ADOPTED = "adopted"
    REJECTED = "rejected"
    UNFILLED = "unfilled"
    TIMEOUT = "timeout"
    UNRESOLVED = "unresolved"
    #: An OPEN record nothing could settle across ``WEDGED_CYCLES`` resyncs: a
    #: distinct, LOUD marker so an operator acts instead of reading one more
    #: ``unresolved`` line.
    WEDGED = "wedged"
    #: An OPEN refused because the account net and the ledger's booked exposure on
    #: the conid disagree (``FeedError.kind == "divergence"``): a fill we cannot
    #: see may be live, or our book is ahead of the account. Distinct from a
    #: broker ``rejected`` and from an order ``unresolved``.
    DIVERGENCE = "divergence"


#: Consecutive resyncs an OPEN record may stay unresolved before it is called
#: WEDGED and surfaced distinctly. Small enough to surface within a session, large
#: enough to absorb a transient gateway 503 / a single missed cron cycle.
WEDGED_CYCLES = 3


class PendingIntents(Protocol):
    """Durable store for intent records — the only owner of OPEN state.

    ``open_attempt`` mints (or bumps) an attempt and persists PENDING; ``close``
    stamps a terminal or unresolved state; ``save`` upserts an arbitrary record
    (used to persist WORKING after an adopt). ``prune`` drops closed rows older
    than a cutoff and returns the count.
    """

    def load(self, key: IntentKey) -> IntentRecord | None: ...

    def load_open(self, scope: str) -> tuple[IntentRecord, ...]: ...

    def save(self, record: IntentRecord) -> None: ...

    def open_attempt(
        self, key: IntentKey, decision_ts: pd.Timestamp | None, now: pd.Timestamp
    ) -> IntentRecord: ...

    def close(
        self,
        key: IntentKey,
        state: IntentState,
        order_id: str | None,
        now: pd.Timestamp,
    ) -> None: ...

    def prune(self, before: pd.Timestamp) -> int: ...


__all__ = [
    "DEFAULT_TIF",
    "OPEN_STATES",
    "WEDGED_CYCLES",
    "IntentKey",
    "IntentRecord",
    "IntentState",
    "OrderOutcome",
    "PendingIntents",
    "Resolution",
    "WorkingOrder",
    "attempt_of",
    "intent_key",
    "order_ref",
    "ref_is_ours",
    "ref_matches_key",
    "ref_prefix",
]
