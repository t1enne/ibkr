"""Gateway lifecycle: is the Client Portal Gateway up, authenticated, and alive?

This is the live adapter for the `Gateway` port (``src/live/ports.py``). It wraps
an :class:`~src.data.ibkr.client.IbkrClient` and answers two questions with typed
values, never exceptions: *may I read the book right now?* (``is_ready``) and
*make it so, or tell me why not* (``ensure_ready``).

``ensure_ready`` performs the session check + keepalive: a ``/tickle`` POST keeps
the session from idling out, and ``/iserver/auth/status`` decides readiness. It
deliberately does NOT start the docker container or drive a browser — that is
``sync_market_data`` / ``login``'s job, and a cycle that finds the gateway down
should fail fast with an ``auth``/``transport`` reason rather than open a socket
it cannot open. A caller that *can* log in (cron) injects a ``login`` callable and
``ensure_ready`` re-probes after it.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, cast

from src.data.ibkr.client import ErrorKind, IbkrClient, IbkrError
from src.live.result import Err, Ok, Result
from src.live.types import FeedError

#: The kinds ``FeedError`` can carry that a gateway failure maps onto.
_KIND_MAP = {"auth": "auth", "rate_limit": "rate_limit", "transport": "transport"}


def feed_error(error: IbkrError) -> FeedError:
    """Map a client :class:`IbkrError` onto the live ``FeedError`` vocabulary."""
    kind = cast("ErrorKind", _KIND_MAP.get(error.kind, "transport"))
    return FeedError(kind=kind, message=str(error))


def _authenticated(status: dict[str, Any]) -> bool:
    """Read the auth flag from an ``/iserver/auth/status`` body, tolerating shape."""
    return bool(status.get("authenticated"))


class IbkrGateway:
    """``Gateway`` adapter: readiness probe + keepalive over one ``IbkrClient``."""

    def __init__(self, client: IbkrClient | None = None) -> None:
        self._client = client if client is not None else IbkrClient()

    @property
    def client(self) -> IbkrClient:
        """The underlying client (the CLI reuses it for the portfolio source)."""
        return self._client

    async def aclose(self) -> None:
        """Release the client's connection pool."""
        await self._client.aclose()

    async def is_ready(self) -> Result[bool, FeedError]:
        """Probe readiness: authenticated session, else ``Ok(False)``.

        A transport/auth failure to *reach* the gateway is an ``Err`` (we cannot
        tell whether it is up); an answered-but-unauthenticated session is a
        clean ``Ok(False)`` — the gateway is up, the session just needs login.
        """
        try:
            status = await self._client.auth_status()
        except IbkrError as exc:
            return Err(feed_error(exc))
        return Ok(_authenticated(status))

    async def ensure_ready(
        self,
        *,
        login: Callable[[], Awaitable[None]] | None = None,
    ) -> Result[None, FeedError]:
        """Keepalive + auth check; optionally log in, then re-probe.

        Order: ``/tickle`` first (a session that is about to idle out gets poked
        before we read its status), then ``/iserver/auth/status``. When the
        session is not authenticated and a ``login`` callable is supplied it is
        awaited and the status re-checked once. Returns ``Ok(None)`` only when
        the session is authenticated; every other path is a typed ``Err``.
        """
        try:
            await self._client.tickle()
            status = await self._client.auth_status()
        except IbkrError as exc:
            return Err(feed_error(exc))
        if _authenticated(status):
            return Ok(None)
        if login is None:
            return Err(
                FeedError(
                    kind="auth",
                    message="gateway session not authenticated (no login provided)",
                )
            )
        try:
            await login()
            status = await self._client.auth_status()
        except IbkrError as exc:
            return Err(feed_error(exc))
        if _authenticated(status):
            return Ok(None)
        return Err(
            FeedError(
                kind="auth", message="login completed but session is not authenticated"
            )
        )
