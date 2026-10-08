"""The six ``live_*`` peewee models — the durable live book's schema.

Split out of :mod:`src.live.ledger` so the migration versions can build the live
schema without importing the ledger (which imports the versions' registry back).
BREAKS THE CYCLE: ``ledger`` -> ``ledger_migration``/``versions`` -> ``models``,
and nothing in ``models`` imports ``ledger``.

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

#: ``live_position.source`` for the sim lot book (the human/broker-editable surface).
SOURCE_ACCOUNT = "account"
#: ``live_position.source`` for the fill fold (the engine-owned, append-only truth).
SOURCE_EXECUTIONS = "executions"


class LiveStrategy(_Base):
    """Which config revision wrote this scope (an AUDIT row, never an ownership key)."""

    strategy_id = TextField(primary_key=True)
    scope = TextField(null=False, default="")
    #: Scope segments (plan §4.1), additive: which adapter/config wrote this scope.
    adapter = TextField(null=False, default="")
    config_name = TextField(null=False, default="")
    instance = TextField(null=False, default="")
    name = TextField()
    mode = TextField()
    created_at = IntegerField()
    last_cycle_at = IntegerField(null=True)

    class Meta:
        table_name = "live_strategy"


class LivePosition(_Base):
    """The ONE book table: both roles, told apart by ``source``.

    ``position_id`` is TEXT because the two roles name lots differently: an IBKR
    lot is ``str(conid)`` (what the IBKR portfolio source mints, so the re-key
    keeps every existing row's identity), a sim lot the broker's synthetic
    ``SYM_<ts>_<seq>`` id.
    """

    scope = TextField()
    position_id = TextField()
    symbol = TextField()
    side = TextField()
    qty = FloatField()
    entry_price = FloatField()
    stop_loss = FloatField(null=True)
    take_profit = FloatField(null=True)
    tag = TextField(default="")
    order_ref = TextField(default="")
    opened_at = IntegerField(null=True)
    closed_at = IntegerField(null=True)
    #: The entry/exit legs' fees (sim fill detail; NULL when unknown).
    entry_commission = FloatField(null=True)
    exit_price = FloatField(null=True)
    exit_commission = FloatField(null=True)
    source = TextField(null=False, default=SOURCE_ACCOUNT)

    class Meta:
        table_name = "live_position"
        primary_key = CompositeKey("scope", "position_id")
        indexes = ((("scope", "closed_at"), False),)


class LiveExecution(_Base):
    """One applied fill (the append-only fold source, keyed by ``execution_id``)."""

    scope = TextField()
    execution_id = TextField()
    #: The lot this fill belongs to (``str(conid)`` for IBKR; a sim lot's id).
    position_id = TextField(default="")
    side = TextField()
    qty = FloatField()
    price = FloatField()
    commission = FloatField()
    cash_delta = FloatField()
    ts = IntegerField()

    class Meta:
        table_name = "live_execution"
        primary_key = CompositeKey("scope", "execution_id")


class LiveCash(_Base):
    """Each scope's ``initial_capital`` (the sizing base, never the account summary)."""

    scope = TextField(primary_key=True)
    initial_capital = FloatField()
    updated_at = IntegerField()

    class Meta:
        table_name = "live_cash"


class LiveScopeAlias(_Base):
    """Migration audit: the scope a legacy book moved to (plan §4.4 step 4).

    Written once per re-keyed scope so a read can follow a bare legacy scope to
    the charged one; the rows themselves are re-keyed in every live table, never
    dropped.
    """

    legacy_scope = TextField(primary_key=True)
    new_scope = TextField()

    class Meta:
        table_name = "live_scope_alias"


class LiveOrderIntent(_Base):
    """The durable owner of OPEN order state, keyed by the IDENTITY columns.

    The primary key is ``(scope, symbol, action, position_id)`` — the intent
    identity itself (``position_id`` is ``''`` for an open) — NOT the crc32
    ``token``: a crc32 collision can never alias two distinct identities' rows
    (D7). The ``token`` stays as a plain column, the bar-free cOID prefix.

    ``live_sim_lot`` is NOT here: it was folded into :class:`LivePosition`
    (``source='account'``) by ``live_0001_baseline`` and survives only as the kept
    copy ``live_sim_lot_legacy``.
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
    LivePosition,
    LiveExecution,
    LiveCash,
    LiveScopeAlias,
    LiveOrderIntent,
)

#: The live TABLES, in the order the migration runner reports them.
LIVE_TABLES: tuple[str, ...] = tuple(m._meta.table_name for m in LIVE_MODELS)

__all__ = [
    "LIVE_MODELS",
    "LIVE_TABLES",
    "SOURCE_ACCOUNT",
    "SOURCE_EXECUTIONS",
    "LiveCash",
    "LiveExecution",
    "LiveOrderIntent",
    "LivePosition",
    "LiveScopeAlias",
    "LiveStrategy",
]
