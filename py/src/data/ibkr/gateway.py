"""Gateway lifecycle: is the Client Portal Gateway up, authenticated, and alive?

This is the CLI's readiness/keepalive adapter, used directly by ``ibkr live
run`` (``src/live/cli.py``). It wraps an :class:`~src.data.ibkr.client.IbkrClient`
and answers two questions with typed values, never exceptions: *may I read the
book right now?* (``is_ready``) and *make it so, or tell me why not*
(``ensure_ready``).

``ensure_ready`` performs the session check + keepalive: a ``GET /tickle`` keeps
the session from idling out, and ``/iserver/auth/status`` decides readiness. It
deliberately does NOT start the docker container or drive a browser — that is
``sync_market_data`` / ``login``'s job (the compose-side keepalive lives in
``docker-compose.yml``'s healthcheck; ``data.ibkr.sync`` drives login). A cycle
that finds the gateway down fails fast with an ``auth``/``transport`` reason
rather than opening a socket it cannot open.

GET, not POST: this gateway build serves ``/tickle`` and ``/iserver/auth/status``
over GET. ``openapi.spec.json`` declares them POST-only, but a POST answers 411
from the fronting server, so the spec is unenforced and GET is correct here.
"""

from __future__ import annotations

from typing import Any

from src.data.ibkr.client import IbkrClient, IbkrError, is_authenticated
from src.live.result import Err, Ok, Result
from src.live.types import FeedError, feed_error


def _authenticated(status: dict[str, Any]) -> bool:
    """Read the auth flag from a status body, tolerating both gateway shapes."""
    return is_authenticated(status)


class IbkrGateway:
    """Readiness probe + keepalive over one ``IbkrClient``."""

    def __init__(self, client: IbkrClient | None = None) -> None:
        self._client = client if client is not None else IbkrClient()

    @property
    def client(self) -> IbkrClient:
        """The underlying client (the CLI reuses it for the portfolio source)."""
        return self._client

    async def aclose(self) -> None:
        """Release the client's connection pool (a cycle's writer is done with it)."""
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
            return Err(feed_error(exc.kind, str(exc)))
        return Ok(_authenticated(status))

    async def ensure_ready(self) -> Result[None, FeedError]:
        """Keepalive + auth check.

        Order: ``GET /tickle`` first (a session about to idle out gets poked
        before we read its status), then ``/iserver/auth/status``. Returns
        ``Ok(None)`` only when the session is authenticated; a logged-out session
        is a typed ``auth`` failure (never a generic transport one).
        """
        try:
            await self._client.tickle()
            status = await self._client.auth_status()
        except IbkrError as exc:
            return Err(feed_error(exc.kind, str(exc)))
        if _authenticated(status):
            return Ok(None)
        return Err(
            FeedError(
                kind="auth",
                message="gateway session not authenticated (login via `ibkr login`/cron)",
            )
        )
