"""``bot stop`` — request a clean stop, then await exit (plan §2/§4).

Stopping is a REQUEST, never a kill: the loop checks the stop flag only at its
sleep point, so the bot finishes the cycle it is in and exits on its own — no
live process is ever signalled. The wait is bounded; a scope still locked at the
deadline is reported STILL LIVE and escalated by hand.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from src.bot.types import BotLiveness

#: How long ``stop`` awaits a bot's exit before reporting it still live.
#: Generous: a bot only notices the flag after the cycle it is mid-way through.
STOP_WAIT_S = 120.0

#: Seconds between liveness probes while awaiting an exit.
_POLL_SECONDS = 0.2


@dataclass(frozen=True)
class StopOutcome:
    """One config's stop result.

    ``stopped`` means the request is delivered (gone, not live, or a
    ``--no-wait`` request); ``still_live`` is the ONE failure — a waited scope
    still locked at the deadline, needing manual escalation.
    """

    scope: str
    stopped: bool
    still_live: bool
    message: str


def wait_for_exit(registry: BotLiveness, scope: str, timeout_s: float) -> bool:
    """Whether *scope*'s lock is released within *timeout_s*.

    Polls the liveness probe — the OS lock, never a signal — so a bot that is
    mid-cycle is awaited, not interrupted. Returns ``False`` on the deadline.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        live = registry.probe(scope)
        if live is None or live.state != "live":
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(_POLL_SECONDS)


def stop_one(
    registry: BotLiveness,
    scope: str,
    *,
    wait: bool,
    timeout_s: float = STOP_WAIT_S,
) -> StopOutcome:
    """Write *scope*'s stop flag and await exit when *wait* is set.

    Takes NO scope lock: the bot holds it for life, and grabbing it here would
    block behind the very process this call is trying to stop. Never signals.
    """
    registry.stop_flag_path(scope).write_text("")
    if not wait:
        return StopOutcome(scope, True, False, "stop requested")
    if wait_for_exit(registry, scope, timeout_s):
        return StopOutcome(scope, True, False, "stopped")
    return StopOutcome(scope, False, True, "stop requested; still live")


__all__ = [
    "STOP_WAIT_S",
    "StopOutcome",
    "stop_one",
    "wait_for_exit",
]
