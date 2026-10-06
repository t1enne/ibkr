"""The ONE HTTP seam to the IBKR Client Portal Gateway.

Everything that talks to ``https://localhost:5000/v1/api/`` goes through an
:class:`IbkrClient`: one base url, one ``verify=False`` policy (the gateway ships
a self-signed cert), one place that turns a transport/HTTP failure into a typed
:class:`IbkrError`. Before this module, ``shared.py`` held a module-level
``httpx.AsyncClient`` and ``candles.py`` held a second module-level
``ib_rest_api_client.Client`` — two clients, two policies. They now both come
from here.

Failure is a value at the *port* boundary (``Result[.., FeedError]``); at this
module's boundary a failed call raises :class:`IbkrError` so a caller cannot
forget to check. The mapping is exactly::

    401                    -> kind="auth"
    429, 503               -> kind="rate_limit"
    timeout, connect error -> kind="transport"
    any other status       -> kind="transport"

Base url / account come from the repo's env conventions (``IBKR_GATEWAY_URL``,
``IBKR_ACCOUNT``) with the gateway's documented defaults.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Literal, cast

import httpx
from ib_rest_api_client import Client as _RestClient

#: The gateway's documented local endpoint (see docker-compose.yml / .env.example).
DEFAULT_BASE_URL = "https://localhost:5000/v1/api/"
DEFAULT_TIMEOUT_S = 10.0

#: Failure kinds this seam distinguishes. Kept deliberately small — the live
#: ``FeedError`` vocabulary is mapped from these at the port boundary.
ErrorKind = Literal["auth", "rate_limit", "transport"]

#: HTTP statuses that mean "slow down", not "you are broken".
_RATE_LIMIT_STATUSES = frozenset({429, 503})

#: Positions-per-page the gateway serves; a shorter page ends the walk.
_POSITIONS_PAGE_SIZE = 30

#: Hosts for which the self-signed-cert ``verify=False`` policy is legitimate.
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def is_authenticated(body: object) -> bool:
    """True only when a session body reports an authenticated session.

    Accepts BOTH shapes the gateway serves:

    - flat ``{"authenticated": true}`` — ``/iserver/auth/status`` in this build;
    - nested ``{"iserver": {"authStatus": {"authenticated": true}}}`` — ``/tickle``.

    Strict ``is True`` is deliberate: a string ``"false"`` (or ``"true"``) must
    NOT read as authenticated, so a truthiness check is wrong here.
    """
    if not isinstance(body, Mapping):
        return False
    if body.get("authenticated") is True:
        return True
    iserver = body.get("iserver")
    if isinstance(iserver, Mapping):
        status = iserver.get("authStatus")
        if isinstance(status, Mapping):
            return status.get("authenticated") is True
    return False


class IbkrError(Exception):
    """A typed gateway failure: ``kind`` says what went wrong, ``message`` why."""

    def __init__(self, kind: ErrorKind, message: str, endpoint: str = "") -> None:
        super().__init__(message)
        self.kind = kind
        self.endpoint = endpoint


def _default_base_url() -> str:
    """The configured gateway url (``IBKR_GATEWAY_URL``), else the local default."""
    return os.environ.get("IBKR_GATEWAY_URL", DEFAULT_BASE_URL)


def _default_account() -> str | None:
    """The configured account id (``IBKR_ACCOUNT``), else ``None`` (gateway picks)."""
    value = os.environ.get("IBKR_ACCOUNT")
    return value or None


class IbkrClient:
    """Async read client for the gateway. Owns the httpx + generated REST clients.

    ``http`` is the raw ``httpx.AsyncClient`` (used by the prose endpoints below);
    ``rest`` is the generated ``ib_rest_api_client.Client`` the candle fetcher
    drives (``get_iserver_marketdata_history``). Both are configured from ONE
    base url / verify policy, so a change here moves both.
    """

    def __init__(
        self,
        base_url: str | None = None,
        account: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        verify: bool = False,
    ) -> None:
        self.base_url = base_url or _default_base_url()
        # ``verify=False`` exists ONLY because the local gateway ships a self-signed
        # cert. Make that a code fact, not a silent default: a non-localhost url
        # must not be talked to with TLS verification off.
        if not verify:
            host = httpx.URL(self.base_url).host
            assert host in _LOCAL_HOSTS, (
                f"verify=False is localhost-only; refusing {host!r} "
                f"(set verify=True for a remote gateway)"
            )
        self.account = account if account is not None else _default_account()
        self._verify = verify
        self._http = httpx.AsyncClient(
            base_url=self.base_url, timeout=timeout, verify=verify
        )
        self._rest = _RestClient(base_url=self.base_url, verify_ssl=verify)

    @property
    def http(self) -> httpx.AsyncClient:
        """The shared httpx client (already base-url + TLS configured)."""
        return self._http

    @property
    def rest(self) -> _RestClient:
        """The generated REST client, on the SAME url/verify policy."""
        return self._rest

    async def aclose(self) -> None:
        """Close the underlying pool. The generated client holds no socket."""
        await self._http.aclose()

    # -- transport ---------------------------------------------------------

    async def get(self, endpoint: str) -> Any:
        """Public GET: parsed JSON or :class:`IbkrError` (one error policy)."""
        return await self._get(endpoint)

    async def post(self, endpoint: str, json: object | None = None) -> Any:
        """Public POST: parsed JSON or :class:`IbkrError` (one error policy)."""
        return await self._post(endpoint, json)

    async def _get(self, endpoint: str) -> Any:
        """GET *endpoint* and return the parsed JSON, raising :class:`IbkrError`.

        The single place a transport/HTTP failure becomes typed. A 2xx with a
        non-JSON body is also a transport failure (the gateway answered
        something we cannot read).
        """
        try:
            response = await self._http.get(endpoint)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise IbkrError("transport", f"{endpoint}: {exc}", endpoint) from exc
        if response.status_code == 401:
            raise IbkrError("auth", f"{endpoint}: 401 unauthenticated", endpoint)
        if response.status_code in _RATE_LIMIT_STATUSES:
            raise IbkrError(
                "rate_limit", f"{endpoint}: {response.status_code}", endpoint
            )
        if response.status_code >= 400:
            raise IbkrError(
                "transport", f"{endpoint}: {response.status_code}", endpoint
            )
        try:
            return response.json()
        except ValueError as exc:
            raise IbkrError(
                "transport", f"{endpoint}: unparseable body", endpoint
            ) from exc

    async def _post(self, endpoint: str, json: object | None = None) -> Any:
        """POST *endpoint* (reply-confirmation style); same error mapping as GET."""
        try:
            response = await self._http.post(endpoint, json=json)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise IbkrError("transport", f"{endpoint}: {exc}", endpoint) from exc
        if response.status_code == 401:
            raise IbkrError("auth", f"{endpoint}: 401 unauthenticated", endpoint)
        if response.status_code in _RATE_LIMIT_STATUSES:
            raise IbkrError(
                "rate_limit", f"{endpoint}: {response.status_code}", endpoint
            )
        if response.status_code >= 400:
            raise IbkrError(
                "transport", f"{endpoint}: {response.status_code}", endpoint
            )
        try:
            return response.json()
        except ValueError as exc:
            raise IbkrError(
                "transport", f"{endpoint}: unparseable body", endpoint
            ) from exc

    # -- read endpoints this phase needs ----------------------------------

    async def auth_status(self) -> dict[str, Any]:
        """``/iserver/auth/status`` — connected/authenticated/competing flags.

        GET in this build (spec says POST-only; the fronting server answers a
        POST with 411, so the spec is unenforced). The body is the flat
        ``{"authenticated": ...}`` shape :func:`is_authenticated` reads.
        """
        return cast("dict[str, Any]", await self._get("iserver/auth/status"))

    async def tickle(self) -> dict[str, Any]:
        """``/tickle`` — keepalive; also the canonical session probe.

        This build serves it over GET (a POST returns 411 from the fronting
        server even though ``openapi.spec.json`` declares the endpoint
        POST-only — that declaration is NOT enforced here). GET both keeps the
        session alive and returns the session/``iserver.authStatus`` shape.
        """
        return cast("dict[str, Any]", await self._get("tickle"))

    async def accounts(self) -> list[dict[str, Any] | str]:
        """``/iserver/accounts`` — the accounts this session may read.

        The gateway returns bare account-id strings (some builds wrap them in a
        dict), so the element type is the union rather than a guess.
        """
        body = await self._get("iserver/accounts")
        return cast("list[dict[str, Any] | str]", _as_list(body, "accounts"))

    async def resolve_account(self) -> str:
        """The configured account, else the session's first; a typed failure if none.

        Cached on the instance so a second call is free. Never invents an id: an
        empty account list is an ``auth`` failure, not a guess.
        """
        if self.account:
            return self.account
        for entry in await self.accounts():
            # The gateway returns account ids as bare strings; some builds wrap
            # them in a dict. Accept both, prefer the explicit key.
            if isinstance(entry, str):
                acct = entry
            else:
                acct = entry.get("accountId") or entry.get("account")
            if isinstance(acct, str) and acct:
                self.account = acct
                return acct
        raise IbkrError("auth", "no account available from the gateway", "accounts")

    async def portfolio_summary(self, account: str | None = None) -> dict[str, Any]:
        """``/portfolio/{acct}/summary`` — cash and net liquidation for the book."""
        account = self._require_account(account)
        return cast("dict[str, Any]", await self._get(f"portfolio/{account}/summary"))

    async def positions(
        self, account: str | None = None, page: int = 0
    ) -> list[dict[str, Any]]:
        """``/portfolio/{acct}/positions/{page}`` — one page of net positions."""
        account = self._require_account(account)
        body = await self._get(f"portfolio/{account}/positions/{page}")
        return cast("list[dict[str, Any]]", _as_list(body, "positions"))

    async def positions_all(self, account: str | None = None) -> list[dict[str, Any]]:
        """Every page of net positions: walk until a SHORT page ends the book.

        Reading page 0 only silently truncates a book of more than one page, so
        paginate until a page carries fewer than the expected page size (IBKR
        pages ~30 rows). A hard page cap guards against a gateway that never
        returns a short page.
        """
        account = self._require_account(account)
        out: list[dict[str, Any]] = []
        for page in range(1000):
            rows = await self.positions(account, page)
            out.extend(rows)
            if len(rows) < _POSITIONS_PAGE_SIZE:
                break
        return out

    async def trades(self) -> list[dict[str, Any]]:
        """``/iserver/account/trades`` — per-execution trade history (7d window)."""
        body = await self._get("iserver/account/trades")
        return cast("list[dict[str, Any]]", _as_list(body, "trades"))

    async def open_orders(self) -> list[dict[str, Any]]:
        """``/iserver/account/orders`` — the account's currently working orders.

        Read before re-sending after an ambiguous submit so a working order is
        seen rather than duplicated (plan §6 phase 3.5 placement hygiene).
        """
        body = await self._get("iserver/account/orders")
        return cast("list[dict[str, Any]]", _as_list(body, "orders"))

    # -- helpers -----------------------------------------------------------

    def _require_account(self, account: str | None) -> str:
        """The explicit account, the configured one, or a typed failure.

        Never guesses across multiple accounts: an unset account is an
        ``auth`` failure (the caller must name which book it reads).
        """
        resolved = account or self.account
        if not resolved:
            raise IbkrError(
                "auth", "no account configured (set IBKR_ACCOUNT)", "portfolio"
            )
        return resolved


def _as_list(body: object, field: str) -> list[object]:
    """Coerce an endpoint body to a list, or raise on an unreadable shape.

    Accepts the two shapes the gateway serves — a bare list, or ``{field: [...]}``
    under a wrapper. Anything else (a shape drift, an ``error`` body, a string)
    raises :class:`IbkrError` rather than returning ``[]``: an empty list reads as
    "no trades", which would silently drop the fills that gate duplicate opens.
    """
    if isinstance(body, list):
        return list(body)
    if isinstance(body, dict):
        inner = cast("dict[str, object]", body).get(field)
        if isinstance(inner, list):
            return list(inner)
    raise IbkrError(
        "transport",
        f"unexpected {field} body shape (want a list or {{{field!r}: [...]}}): "
        f"{body!r}",
        field,
    )


_default: IbkrClient | None = None


def default_client() -> IbkrClient:
    """The process-wide client ``shared.py``/``candles.py`` route through."""
    global _default
    if _default is None:
        _default = IbkrClient()
    return _default
