"""Critical-path tests for ``ibkr live rm`` and the ``live pf`` scope filter.

Both are order-safety surfaces, so this file pins behaviour that would be
DANGEROUS if broken, never rendering:

* ``delete_scope`` removes BOTH of a scope's own rows and leaves every other
  scope's book intact (a scope-scoped delete, not a table wipe).
* ``live rm`` refuses a scope whose book holds an open lot without
  ``--force-flat`` — deleting the fold does not close the broker's position, it
  FORGETS it, and the next cycle re-enters (double exposure).
* ``live rm`` refuses a scope the bot registry reports LIVE.
* ``--yes`` is what actually deletes; without it nothing is written.
* ``scope_holds_positions`` decides the default ``pf`` scope list, so a retired
  scope that holds nothing drops out while one holding an open lot stays in.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd
import pytest
from click.testing import CliRunner, Result

from src.bot.liveness import bot_lock_path
from src.bot.registry import BotRegistry, write_pid_row
from src.live import cli as live_cli
from src.live.ledger import SqliteLedger
from src.live.types import Execution, OrderSide, StrategyBook
from src.live.pf import scope_holds_positions
from src.live.tests.ledgers import live_ledger

_TS: pd.Timestamp = cast(pd.Timestamp, pd.Timestamp("2024-06-03T15:00:00Z"))
_OPEN_SCOPE = "ibkr_open_aaaa1111"
_FLAT_SCOPE = "ibkr_flat_bbbb2222"


def _fill(execution_id: str, *, opening: bool, lot: int = 265598) -> Execution:
    """One broker fill; ``opening`` buys the lot, otherwise sells it flat.

    ``lot`` is the conid the fill folds onto. The book groups fills by
    ``(symbol, position_id)``, so the entry and exit of a round trip MUST share
    one lot id for the group to net to zero — a distinct id per fill would leave
    two open lots instead of a flat book.

    ``qty`` is stored ABSOLUTE: the fold derives the sign from ``side``
    (``_signed_qty``), so a negative qty on a SELL would double-negate back into
    a BUY and the round trip would never net out.
    """
    qty = 10.0
    return Execution(
        execution_id=execution_id,
        order_id=f"o{execution_id}",
        order_ref="",
        conid=lot,
        symbol="AAPL",
        side=OrderSide.BUY if opening else OrderSide.SELL,
        qty=qty,
        price=100.0,
        commission=1.0,
        ts=_TS,
    )


def _seed(ledger: SqliteLedger) -> None:
    """Two scopes: one holding an OPEN lot, one whose fills net to zero."""
    ledger.ensure_strategy("audit1", _OPEN_SCOPE, "open_strat", 50000.0)
    opening = _fill("e1", opening=True)
    ledger.save_book(
        _OPEN_SCOPE,
        StrategyBook(applied=frozenset({"e1"})),
        (opening,),
        50000.0,
    )
    ledger.ensure_strategy("audit2", _FLAT_SCOPE, "flat_strat", 50000.0)
    ledger.save_book(
        _FLAT_SCOPE,
        StrategyBook(applied=frozenset({"e2", "e3"})),
        (_fill("e2", opening=True), _fill("e3", opening=False)),
        50000.0,
    )


@pytest.fixture
def book(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SqliteLedger:
    """A migrated live book on a temp file, wired into the CLI's constructors.

    ``SqliteLedger``/``BotRegistry`` are patched at the CLI's own seams (the
    module-level names it calls) so the command under test runs the REAL code
    against a temp file instead of the operator's ``data/live.db``.
    """
    path = tmp_path / "live.db"
    ledger = live_ledger(path)
    _seed(ledger)
    monkeypatch.setattr(live_cli, "SqliteLedger", lambda: ledger)
    monkeypatch.setattr(live_cli, "resolve_live_db_path", lambda: path)
    monkeypatch.setattr(live_cli, "BotRegistry", lambda _db: BotRegistry(_db))
    return ledger


def _rm(*args: str) -> Result:
    return CliRunner().invoke(live_cli.live_rm, list(args))


# --- (a) the ledger delete: scope-scoped, both tables -------------------------


def test_delete_scope_removes_its_fills_and_audit_rows_only(
    book: SqliteLedger,
) -> None:
    """A flat scope goes entirely; the other scope's book is untouched."""
    strategies, executions = book.delete_scope(_FLAT_SCOPE)
    assert (strategies, executions) == (1, 2)
    assert book.executions_count_of(_FLAT_SCOPE) == 0
    assert book.strategies_of(_FLAT_SCOPE) == ()
    # The sibling scope keeps its audit row AND its open fill.
    assert len(book.strategies_of(_OPEN_SCOPE)) == 1
    assert book.executions_count_of(_OPEN_SCOPE) == 1


def test_scopes_of_store_stops_reporting_a_deleted_scope(book: SqliteLedger) -> None:
    book.delete_scope(_FLAT_SCOPE)
    assert book.scopes_of_store() == (_OPEN_SCOPE,)


# --- (b) the two refusal guards ----------------------------------------------


def test_rm_refuses_a_scope_whose_book_holds_an_open_lot(book: SqliteLedger) -> None:
    """Guard 2: forgetting an open position would re-enter it next cycle."""
    result = _rm("--yes", _OPEN_SCOPE)
    assert result.exit_code == 1
    assert "REFUSED" in result.stderr
    assert book.executions_count_of(_OPEN_SCOPE) == 1  # NOT deleted


def test_rm_force_flat_deletes_the_same_scope(book: SqliteLedger) -> None:
    """The escape hatch is real, and it is the ONLY way past guard 2."""
    result = _rm("--yes", "--force-flat", _OPEN_SCOPE)
    assert result.exit_code == 0
    assert book.executions_count_of(_OPEN_SCOPE) == 0


def test_rm_refuses_a_live_bot(
    book: SqliteLedger, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guard 1: a live bot's row must survive, or nothing can stop it by name."""
    db = tmp_path / "live.db"
    # ``probe_live`` opens the lock file WITHOUT ``O_CREAT``, so the file must
    # exist for the seam to be consulted at all.
    Path(bot_lock_path(db, _FLAT_SCOPE)).touch()
    write_pid_row(db, _FLAT_SCOPE, pid=1, started_at=_TS, config_path="x.json")
    registry = BotRegistry(db, flock=_LiveFlock())
    monkeypatch.setattr(live_cli, "BotRegistry", lambda _p: registry)
    result = _rm("--yes", _FLAT_SCOPE)
    assert result.exit_code == 1
    assert "LIVE" in result.stderr
    assert registry.probe(_FLAT_SCOPE) is not None


class _LiveFlock:
    """A flock seam that reports every lock as HELD (a bot owns the scope).

    ``acquire`` returning ``False`` is what ``probe_live`` reads as held, so the
    registry derives ``state="live"`` without any real process.
    """

    def acquire(self, fd: int, blocking: bool) -> bool:
        return False

    def release(self, fd: int) -> None:
        return None


# --- (c) --yes is what writes ------------------------------------------------


def test_rm_without_yes_deletes_nothing(book: SqliteLedger) -> None:
    """A dry run reports the scope and writes no row."""
    result = _rm(_FLAT_SCOPE)
    assert result.exit_code == 0
    assert "would delete" in result.output
    assert book.executions_count_of(_FLAT_SCOPE) == 2


def test_rm_ignores_an_unknown_scope_without_failing(
    book: SqliteLedger,
) -> None:
    """A typo deletes nothing and is not an error — the operator got their wish."""
    result = _rm("--yes", "no_such_scope")
    assert result.exit_code == 0
    assert "unknown scope" in result.output


# --- (d) the pf scope filter -------------------------------------------------


def test_scope_holds_positions_separates_open_from_round_tripped(
    book: SqliteLedger,
) -> None:
    """The default ``pf`` list is decided by this predicate."""
    from src.live.pf import read_store

    assert scope_holds_positions(read_store(book, _OPEN_SCOPE)) is True
    assert scope_holds_positions(read_store(book, _FLAT_SCOPE)) is False
