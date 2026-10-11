"""Pure bracket layer — protective SL/TP children, refs, attribution, level diff.

No I/O: every function here is a pure mapping, table-tested against the spec's
wire contract (SPEC_LIVE_STOPS §3). The resting order is the BROKER's; nothing
here persists a level, stamps a column or keeps an intent row.

- :func:`bracket_children` — the OCA pair for an entry (empty without a level;
  reduce side is the entry's opposite, qty is the confirmed fill). The entry
  ticket never carries a level (spec §Step 3).
- :func:`protective_ref` — distinct per ``(lot, level)``, stable for a re-send
  of the same pair, anchored on the lot's OWN open/fill time (never the decision
  bar, never the clock).
- :func:`resting_bracket` — ours-by-prefix + conid + side on the wire, kind read
  off ``orderType``; ``None`` when either leg is missing or foreign.
- :func:`level_bracket_intents` — arm / tighten->REPLACE / ratchet-never-widen /
  flat-drops-silently against the book's advisory levels. The WIRE comparison
  (resting level vs the new one) belongs to the edge (Step 5): this layer
  refuses only a level that would WIDEN the book's advisory one.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Callable, Sequence
from typing import Literal, cast

import pandas as pd

from src.bt.state import ActionType, PortfolioView, Position, TradeSignal
from src.exec.types import OrderType
from src.live.identity import WorkingOrder
from src.live.types import FLAT_QTY_EPS, OrderIntent, synthetic_conid

#: Absolute ``stop_loss``/``take_profit`` must be strictly above zero.
_MIN_LEVEL = 0.0

#: Kind names (the wire's SL-vs-TP discriminator is ``orderType``; the kinds here
#: name the leg). ONE vocabulary shared by the ref rule, the child builder and
#: the wire reader.
StopKind = "stop"
TakeKind = "take_profit"
LevelKind = Literal["stop", "take_profit"]
_LEVEL_KINDS: frozenset[str] = frozenset({StopKind, TakeKind})


def _finite_positive(level: float | None) -> bool:
    """Whether a level is a usable absolute price (finite, strictly > 0)."""
    return level is not None and math.isfinite(level) and level > _MIN_LEVEL


def bracket_children(
    entry: OrderIntent,
    *,
    conid: int,
    filled_qty: float,
    tif: str = "GTC",
) -> tuple[OrderIntent, ...]:
    """The OCA pair protecting *entry*; empty when the entry carries no level.

    One ``STP`` child for ``stop_loss`` and one ``LMT`` for ``take_profit``,
    both in the reduce direction of the lot (the entry action's opposite), both
    sized to the CONFIRMED fill (never the planned qty, never topped up), each
    carrying its own ref minted from the lot's own time. ``position_side`` is
    the entry's side. An entry with neither level yields ``()`` — nothing to
    protect.
    """
    stop, take = entry.stop_loss, entry.take_profit
    if stop is None and take is None:
        return ()
    assert stop is None or _finite_positive(stop), (
        f"{entry.symbol}: stop_loss {stop!r} must be finite and > 0 to rest a stop"
    )
    assert take is None or _finite_positive(take), (
        f"{entry.symbol}: take_profit {take!r} must be finite and > 0 to rest a limit"
    )
    assert entry.action in (ActionType.long, ActionType.short), (
        f"{entry.symbol}: a protective child requires an entry action, got "
        f"{entry.action.value!r}"
    )
    assert filled_qty > FLAT_QTY_EPS, (
        f"{entry.symbol}: a protective child needs a confirmed positive fill, got "
        f"{filled_qty!r}"
    )
    anchored = _lot_time(entry)
    reduce_side = _reduce_side(entry.action)
    children: list[OrderIntent] = []
    if stop is not None:
        children.append(
            _child(
                entry,
                kind=StopKind,
                order_type=OrderType.STP,
                price=stop,
                tif=tif,
                conid=conid,
                filled_qty=filled_qty,
                reduce_side=reduce_side,
                anchor=anchored,
            )
        )
    if take is not None:
        children.append(
            _child(
                entry,
                kind=TakeKind,
                order_type=OrderType.LMT,
                price=take,
                tif=tif,
                conid=conid,
                filled_qty=filled_qty,
                reduce_side=reduce_side,
                anchor=anchored,
            )
        )
    return tuple(children)


def _child(
    entry: OrderIntent,
    *,
    kind: LevelKind,
    order_type: OrderType,
    price: float,
    tif: str,
    conid: int,
    filled_qty: float,
    reduce_side: ActionType,
    anchor: pd.Timestamp,
) -> OrderIntent:
    """ONE protective child: reduce side, confirmed qty, kind-shaped ref."""
    pid = entry.position_id if entry.position_id is not None else str(conid)
    return OrderIntent(
        symbol=entry.symbol,
        action=ActionType.close,
        qty=filled_qty,
        ref_price=price,
        reason=f"protect {entry.symbol} {kind}",
        position_id=pid,
        stop_loss=price if kind == StopKind else None,
        take_profit=price if kind == TakeKind else None,
        tag=entry.tag,
        decision_ts=entry.decision_ts,
        order_type=order_type,
        protective=True,
        position_side=reduce_side,
    )


def _reduce_side(entry_action: ActionType) -> ActionType:
    """The side a reduce leg of an entry trades: a long exits via a short, a
    short via a long."""
    return (
        ActionType.short if entry_action is ActionType.long else ActionType.long
    )


def _lot_time(entry: OrderIntent) -> pd.Timestamp:
    """The lot-anchored time of *entry*: its own decision/fill time, or epoch.

    The ref rule needs the lot's OWN open/fill time — never the decision bar of
    a sibling and never the wall clock — so a re-send reuses the ref. An entry
    with no timestamp anchors at the epoch, which is stable and deterministic.
    """
    ts = entry.decision_ts
    if ts is None:
        return cast("pd.Timestamp", pd.Timestamp(0))
    return ts


def protective_ref(
    scope_tag: str,
    *,
    symbol: str,
    conid: int,
    anchor: pd.Timestamp,
    kind: LevelKind,
    level: float,
) -> str:
    """The cOID of one protective leg: distinct per ``(lot, level)``, stable per re-send.

    The token is the full 32-bit crc32 of the lot+level identity rendered as
    eight hex digits — a pure function of ``(symbol, conid, lot-anchored time,
    kind, level)``, so a re-send of the SAME level mints the SAME ref (IBKR's
    ``cOID`` dedupe absorbs it) while a tightened level mints a NEW ref (a
    REPLACE is never swallowed as a duplicate). The lot-anchored ``anchor`` is
    the lot's own open/fill time — never the decision bar or wall clock.
    """
    assert kind in _LEVEL_KINDS, f"unknown protective kind {kind!r}"
    assert _finite_positive(level), f"level {level!r} must be finite and > 0"
    identity = f"{symbol}|{conid}|{anchor.value}|{kind}|{level}"
    token = f"{zlib.crc32(identity.encode('utf-8')):08x}"
    return f"{scope_tag}-{token}"


def child_ref(
    scope_tag: str,
    child: OrderIntent,
    *,
    conid: int,
    anchor: pd.Timestamp,
) -> str:
    """The ref of ONE protective *child* (the shared minting for both edges).

    The child's kind and level decide the token; the ``anchor`` is the lot's OWN
    open/fill time — for a just-opened lot the ENTRY FILL's timestamp, for a
    re-arm the lot's ``entry_time`` read from the live book (both are stable per
    lot, never the decision bar or the wall clock).
    """
    if child.order_type is OrderType.STP:
        kind: LevelKind = StopKind
        level = child.stop_loss
    else:
        kind = TakeKind
        level = child.take_profit
    assert level is not None
    return protective_ref(
        scope_tag,
        symbol=child.symbol,
        conid=conid,
        anchor=anchor,
        kind=kind,
        level=level,
    )


def resting_bracket(
    working: Sequence[WorkingOrder],
    *,
    scope_tag: str,
    conid: int,
) -> tuple[WorkingOrder, WorkingOrder] | None:
    """Our resting OCA pair on *conid*: ``(stop, take_profit)`` or ``None``.

    Ours means the ref carries our ``scope_tag`` prefix and the row is on
    ``conid``. Kind is read off the wire's ``orderType`` (set by the parser),
    never assumed from our book, so an STP resting under another scope's
    bracket is foreign. Both legs must agree on the reduce side (the wire's
    ``side``), or the "pair" is two strangers that happen to share a prefix.
    ``None`` when either leg is missing or foreign — a lot whose bracket is
    incomplete is NOT covered (the caller reports ``UNPROTECTED``, never
    "placed").
    """
    # Only rows that are actually RESTING count (spec §0: ``status``/
    # ``order_ccp_status`` "Submitted" means resting) — an ``Inactive``/
    # ``Pending Submit`` leg from a failed confirm is NOT protection.
    ours = [o for o in working if _is_ours(o.order_ref, scope_tag) and o.conid == conid]
    resting = [o for o in ours if o.status.strip().upper() == "SUBMITTED"]
    stops = [o for o in resting if kind_of(o) == StopKind]
    takes = [o for o in resting if kind_of(o) == TakeKind]
    if not stops or not takes:
        return None
    stop, take = stops[0], takes[0]
    if stop.side != take.side:
        return None
    return (stop, take)


def kind_of(order: WorkingOrder) -> str:
    """The leg kind from the wire's ``orderType`` (``Stop``/``STOP`` vs ``Limit``).

    ``resting_bracket`` relies on this being set by the parse layer; an unset
    string is a foreign/unparseable row and is never a kind.
    """
    upper = order.order_type.strip().upper()
    if upper in ("STOP", "STP"):
        return StopKind
    if upper in ("LIMIT", "LMT"):
        return TakeKind
    return ""


def _is_ours(order_ref: str, scope_tag: str) -> bool:
    """Exact-prefix attribution: our refs are ``scope_tag-token``; the tag itself
    carries a crc tail, so a foreign scope sharing the slug cannot match."""
    return order_ref.startswith(f"{scope_tag}-")


# -- level diff (arm / tighten / ratchet / flat-drop) ------------------------


def level_bracket_intents(
    config_path: str,
    book: PortfolioView,
    levels: Sequence[TradeSignal],
) -> tuple[OrderIntent, ...]:
    """The bracket-level step: arm / tighten->REPLACE / ratchet / flat-drop. Pure.

    ``book`` is the LIVE book (the strategy's own advisory levels never come from
    the live entry price); ``levels`` is the strategy's newest ``stop_update``
    signal per symbol. For each symbol with both a level update and a live lot:

    - unlevelled lot + armed level -> the pair's children (place);
    - new level strictly tighter than the book's advisory one -> the pair's
      children for the NEW level (a level-tighten REPLACE — the edge cancels
      both legs and resubmits);
    - new level equal or wider -> nothing (silent; the ratchet never widens);
    - a level on a flat symbol -> nothing (dropped silently).

    The WIRE comparison (resting level vs the new one) happens at the edge
    (Step 5) under invariant 1; this layer refuses only a WIDENING against the
    book's advisory level, so the pure diff never re-arms a wider level.
    """
    if not levels:
        return ()
    out: list[OrderIntent] = []
    for signal in levels:
        if signal.action is not ActionType.stop_update:
            continue
        lot = _live_lot(book, signal.symbol)
        if lot is None:
            continue  # flat symbol: the level is dropped silently
        if not _any_armed(signal):
            continue
        if not _tightens_or_arms(signal, lot):
            continue  # equal/wider: ratchet, silent
        entry = _level_entry(signal, lot)
        conid = _lot_conid(signal.symbol, lot)
        out.extend(
            bracket_children(
                entry, conid=conid, filled_qty=lot.qty, tif="GTC"
            )
        )
    return tuple(out)


def _live_lot(book: PortfolioView, symbol: str) -> Position | None:
    """The symbol's first OPEN lot (``None`` when flat)."""
    lots = [p for p in book.positions.get(symbol, ()) if p.qty > FLAT_QTY_EPS]
    return lots[0] if lots else None


def _any_armed(signal: TradeSignal) -> bool:
    return _finite_positive(signal.stop_loss) or _finite_positive(signal.take_profit)


def _tightens_or_arms(signal: TradeSignal, lot: Position) -> bool:
    """Whether the update tightens (or arms) either leg vs the lot's advisory level.

    The ratchet: a long's stop only moves UP and its target only DOWN; a short
    mirrors. ``old=None`` (unlevelled) takes the new level outright; ``new=None``
    keeps the old. Equal levels stay (no emission). A level that would WIDEN a
    rested/advisory level is refused here.
    """
    is_long = lot.type is ActionType.long
    if _changes_leg(lot.stop_loss, signal.stop_loss, tighter=max if is_long else min):
        return True
    if _changes_leg(
        lot.take_profit, signal.take_profit, tighter=min if is_long else max
    ):
        return True
    return False


def _changes_leg(
    old: float | None, new: float | None, tighter: Callable[[float, float], float]
) -> bool:
    """Whether ``new`` actually moves the leg (arm or tighten), never a widen.

    The ratchet result must equal the NEW level for a change to count: if the
    tighter of the two is the OLD one, the update is a widening and is refused.
    """
    if new is None:
        return False
    if old is None:
        return _finite_positive(new)
    ratcheted = tighter(old, new)
    return _finite_positive(new) and ratcheted == new and ratcheted != old


def _level_entry(signal: TradeSignal, lot: Position) -> OrderIntent:
    """The synthetic entry carrying the new levels (refs anchor on the lot's time)."""
    return OrderIntent(
        symbol=signal.symbol,
        action=lot.type,
        qty=lot.qty,
        ref_price=(
            signal.price
            if math.isfinite(signal.price) and signal.price > 0
            else lot.last_price
            if math.isfinite(lot.last_price) and lot.last_price > 0
            else 0.0
        ),
        reason="level update",
        position_id=lot.position_id,
        stop_loss=signal.stop_loss,
        take_profit=signal.take_profit,
        tag=signal.tag,
        decision_ts=_lot_time_of(lot),
    )


def _lot_time_of(lot: Position) -> pd.Timestamp:
    """The lot's own open time (never the clock/decision bar) for its refs."""
    return (
        lot.entry_time
        if lot.entry_time is not None
        else cast("pd.Timestamp", pd.Timestamp(0))
    )


def _lot_conid(symbol: str, lot: Position) -> int:
    """The conid handle for the lot's ref token: int-able id, else the synthetic one.

    A sim lot's ``position_id`` is not a conid, so its refs key on the stable
    ``synthetic_conid(symbol)`` — the SAME conid the sim book itself uses, so
    attribution through the sim broker's working read agrees.
    """
    try:
        return int(str(lot.position_id))
    except (TypeError, ValueError):
        return synthetic_conid(symbol)


__all__ = [
    "bracket_children",
    "child_ref",
    "kind_of",
    "level_bracket_intents",
    "protective_ref",
    "resting_bracket",
]