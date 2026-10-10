"""The ``live_*`` peewee models — the durable live book's schema.

Split out of :mod:`src.live.ledger` so the migration versions can build the live
schema without importing the ledger. The baseline simply runs
``create_tables(LIVE_MODELS)``; there is no legacy shape to absorb.

These models stay bound to :data:`src.live.ledger_base._TEMPLATE_DB` (a
``SqliteDatabase(None)`` placeholder) and are rebound per INSTANCE via
``bind_ctx``. That is deliberate and must not be "simplified" to a process-global
handle: peewee binds a model at CLASS level, and the tests genuinely run two
ledgers on two different paths in ONE process. A process-wide binding would make
those two books retarget each other.
"""

from __future__ import annotations

from peewee import (
    CompositeKey,
    FloatField,
    IntegerField,
    TextField,
)

from src.live.ledger_base import _Base


class LiveStrategy(_Base):
    """One row per scope: ownership, audit and the cash seed, merged.

    ``scope`` is the primary key — it is the ownership key, the cash seed's key
    and the lease key all at once. ``strategy_id`` (the config hash) is an AUDIT
    column only, never an ownership filter: a strategy-intent edit re-mints it,
    so keying on it would orphan a live lot.
    """

    scope = TextField(primary_key=True)
    #: The latest config hash that wrote this scope (audit only, not a key).
    strategy_id = TextField(null=False, default="")
    #: Scope segments, additive: which adapter/config wrote this scope.
    adapter = TextField(null=False, default="")
    config_name = TextField(null=False, default="")
    instance = TextField(null=False, default="")
    name = TextField(null=False, default="")
    #: The sizing base, set on the scope's FIRST write and never reset after.
    initial_capital = FloatField(null=False, default=0.0)
    created_at = IntegerField(null=False, default=0)
    last_cycle_at = IntegerField(null=True)
    updated_at = IntegerField(null=False, default=0)

    class Meta:
        table_name = "live_strategy"


class LiveExecution(_Base):
    """One applied fill (the append-only fold source, keyed by ``execution_id``).

    Minimal and immutable: the fold needs the lot id, the symbol, the signed side,
    the size, the price, the cash it moved and when. It carries NO ``conid`` and
    NO ``commission`` — the ``ExecutionRecord`` projection derives what it needs
    and nothing reads those columns back.
    """

    scope = TextField()
    execution_id = TextField()
    #: The lot this fill belongs to (``str(conid)`` for IBKR; a sim lot's id).
    position_id = TextField(default="")
    #: Carried on the row itself, so reading the fold needs no ``live_position``.
    symbol = TextField(default="")
    side = TextField()
    qty = FloatField()
    price = FloatField()
    cash_delta = FloatField()
    ts = IntegerField()

    class Meta:
        table_name = "live_execution"
        primary_key = CompositeKey("scope", "execution_id")


class LiveOrderIntent(_Base):
    """The durable owner of OPEN order state, keyed by the IDENTITY columns.

    The primary key is ``(scope, symbol, action, position_id)`` — the intent
    identity itself (``position_id`` is ``''`` for an open) — NOT the crc32
    ``token``: a crc32 collision can never alias two distinct identities' rows.
    The ``token`` stays as a plain column, the bar-free cOID prefix.
    """

    scope = TextField()
    token = TextField()
    symbol = TextField()
    action = TextField()
    position_id = TextField(default="")
    state = TextField()
    attempt = IntegerField()
    order_ref = TextField()
    order_id = TextField(null=True)
    decision_ts = IntegerField(null=True)
    #: The time-in-force the order was placed with (see ``identity.DEFAULT_TIF``).
    tif = TextField(default="DAY")
    #: Consecutive resyncs this OPEN record stayed unresolved (wedged-key alarm).
    stuck_cycles = IntegerField(default=0)
    updated_at = IntegerField()

    class Meta:
        table_name = "live_order_intent"
        primary_key = CompositeKey("scope", "symbol", "action", "position_id")


#: Every live model, in one place: the ``bind_ctx`` group AND the list
#: ``create_tables`` builds.
LIVE_MODELS: tuple[type[_Base], ...] = (
    LiveStrategy,
    LiveExecution,
    LiveOrderIntent,
)

#: The live TABLES, in the order the migration runner reports them.
LIVE_TABLES: tuple[str, ...] = tuple(m._meta.table_name for m in LIVE_MODELS)

__all__ = [
    "LIVE_MODELS",
    "LIVE_TABLES",
    "LiveExecution",
    "LiveOrderIntent",
    "LiveStrategy",
]
