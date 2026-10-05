"""Mode guard — which account a config may read, and what it must be told.

The plan §5 rule, in one place: **a paper config against a live account
hard-fails**, and ``mode: "live"`` against a live account requires the explicit
``--allow-live`` flag (no env var). Phase 2 is read-only, so the guard's whole job
is to stop a research config from *reading* a funded book by accident — the
placing side of the same rule lands in phase 3.

Account type is read from the account id, which IBKR makes unambiguous: paper
accounts begin with ``DU`` (older ``DF``), live accounts with ``U``. That is the
only signal the gateway's ``/iserver/auth/status`` gives us about the account, so
it is what we key on, and an id that looks like neither is treated as live
(fail-closed).
"""

from __future__ import annotations

from typing import Literal

from src.live.result import Err, Ok, Result
from src.live.types import FeedError

#: Account-id prefixes IBKR uses for a paper (simulated) account.
_PAPER_PREFIXES = ("DU", "DF")


def is_paper_account(account: str) -> bool:
    """True when *account* names a simulated (paper) account."""
    return account.strip().upper().startswith(_PAPER_PREFIXES)


def authorize(
    *,
    mode: Literal["paper", "live"],
    account: str,
    allow_live: bool,
    dry_run: bool,
) -> Result[bool, FeedError]:
    """Decide whether *mode* may read *account*; ``Ok`` carries "is live".

    Returns ``Ok(True)`` for a live account it permits and ``Ok(False)`` for a
    paper one. ``dry_run`` never relaxes the mode rule — a dry run still reads a
    real book — it only records that nothing will be placed. Both refusals are
    ``auth`` ``FeedError``s naming the fix.
    """
    paper = is_paper_account(account)
    if paper:
        return Ok(False)
    if mode == "paper":
        return Err(
            FeedError(
                kind="auth",
                message=(
                    f"mode 'paper' refuses live account {account}: "
                    f"set mode to 'live' (and pass --allow-live) or point at a paper account"
                ),
            )
        )
    if not allow_live:
        return Err(
            FeedError(
                kind="auth",
                message=(
                    f"live account {account} requires --allow-live "
                    f"(dry_run={dry_run} does not waive it)"
                ),
            )
        )
    return Ok(True)
