"""Origin / Host guard for the HTTP API outside ``/api/r18/*``.

The WebUI/Admin token layer is gone, so a browser page on another origin could
send a simple (body-less or ``text/plain``) request to a state-changing route
(CSRF) or, after a DNS rebinding, read a GET route.  This pure ASGI middleware
refuses both, for every ``/api`` path EXCEPT ``/api/r18/*`` (the approval API
keeps its own, stricter guard, ``R18RequestGuard``, which this module neither
reuses for decisions nor changes).

Rules (decided over the RAW ``Host`` / ``Origin`` / ``Sec-Fetch-Site`` only;
``X-Forwarded-*`` and ``Forwarded`` are never read, so a forwarded value can
neither grant nor deny anything):

1. **Host (DNS rebinding), every method.**  Exactly one ``Host``; its
   authority is an IP literal (v4, or bracketed v6), ``localhost``,
   ``testserver`` (Starlette's TestClient default) or a name in
   ``api.allowed_hosts``; an optional port is 1-65535 digits without leading
   zero.  Userinfo, paths, trailing dots, whitespace, non-ASCII and unknown
   names are refused.  A rebinding page is served under the attacker's own
   domain, so its ``Host`` is never one of these.
2. **Origin, every method.**  At most one ``Origin``; when present it is
   ``http(s)://<authority>`` and either that authority equals the request's own
   (already accepted) ``Host``, or the origin is listed in
   ``api.allowed_origins`` / ``api.cors_origins``.  ``null``, userinfo, paths
   and every other scheme are refused.  An absent ``Origin`` (curl, the smoke
   script, the Next.js admin proxy, a same-origin GET) is allowed.
3. **Fetch metadata, non-safe methods** (anything but GET / HEAD / OPTIONS):
   ``Sec-Fetch-Site: cross-site`` is refused (``same-site`` is judged by rule 2).

The guard sits OUTSIDE the app-wide ``CORSMiddleware`` so a refused request
gets no CORS headers; nothing here adds a permissive CORS header, so a GET
response is never readable cross-origin unless the origin is already in
``api.cors_origins``.  A refusal is a closed 403 body (no request content is
echoed), logged at most once per refusal class per minute.  The check is total:
an unexpected error refuses the request (fails closed) and never reaches a route.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, MutableMapping
from dataclasses import dataclass
from enum import Enum
import ipaddress
import logging
import re
import threading
import time
from typing import Any, Final

from fastapi.responses import JSONResponse
from starlette.routing import get_route_path

from suzka.config.schema import ApiSettings, parse_approval_origin

LOGGER = logging.getLogger(__name__)
API_PREFIX: Final = "/api"
# Owned by the R18 approval API's own guard (not imported: an architecture test
# keeps every other module from naming that module).
R18_PREFIX: Final = "/api/r18"
SAFE_METHODS: Final = frozenset({"GET", "HEAD", "OPTIONS"})
# Starlette's TestClient default Host (every test client uses it).
DEFAULT_HOSTS: Final = frozenset({"localhost", "testserver"})
LOG_INTERVAL_SECONDS: Final = 60.0
_MAX_AUTHORITY: Final = 260
_V6: Final = re.compile(r"\[([0-9a-f:.]+)\](?::([1-9][0-9]{0,4}))?")
_NAME: Final = re.compile(r"([0-9a-z.-]+)(?::([1-9][0-9]{0,4}))?")
_ORIGIN: Final = re.compile(r"(https?)://([^/?#@\s]+)")


def is_r18_path(path: str) -> bool:
    return path == R18_PREFIX or path.startswith(R18_PREFIX + "/")


def _headers(raw: Iterable[tuple[bytes, bytes]]) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for name, value in raw:
        found.setdefault(name.decode("latin-1").lower(), []).append(value.decode("latin-1"))
    return found


class Refusal(str, Enum):
    """Why a request was refused: closed, and free of request content."""

    HOST = "api_host_not_accepted"
    ORIGIN = "api_origin_not_accepted"
    FETCH_SITE = "api_cross_site_request_refused"


def is_guarded_path(path: str) -> bool:
    """Whether this guard covers ``path`` (``/api`` outside ``/api/r18``)."""

    return (path == API_PREFIX or path.startswith(API_PREFIX + "/")) and not is_r18_path(path)


@dataclass(frozen=True, slots=True)
class OriginGuardConfig:
    """Parsed configuration (immutable)."""

    hosts: frozenset[str]
    origins: frozenset[str]

    @classmethod
    def from_settings(cls, api: ApiSettings) -> OriginGuardConfig:
        origins: set[str] = set()
        for item in (*api.allowed_origins, *api.cors_origins):
            try:
                parse_approval_origin(item)
            except ValueError:
                continue  # e.g. "*": it grants this guard nothing
            origins.add(item)
        return cls(
            DEFAULT_HOSTS | frozenset(api.allowed_hosts),
            frozenset(origins),
        )


def parse_authority(value: str, hosts: frozenset[str]) -> str | None:
    """The normalised ``host[:port]`` of an acceptable authority, else None."""

    if type(value) is not str or not value.isascii() or not 0 < len(value) <= _MAX_AUTHORITY:
        return None
    text = value.lower()
    v6 = _V6.fullmatch(text)
    if v6 is not None:
        host: str | None = None
        try:
            host = f"[{ipaddress.IPv6Address(v6.group(1)).compressed}]"
        except ValueError:
            return None
        port = v6.group(2)
    else:
        name = _NAME.fullmatch(text)
        if name is None:
            return None
        host, port = name.group(1), name.group(2)
        try:
            ipaddress.IPv4Address(host)
        except ValueError:
            if host not in hosts:
                return None
    if port is not None and int(port) > 65535:
        return None
    return host if port is None else f"{host}:{port}"


def check_request(
    config: OriginGuardConfig,
    method: str,
    raw_headers: Iterable[tuple[bytes, bytes]],
) -> Refusal | None:
    """Rules 1-3 over the RAW headers; None when accepted.  Pure."""

    headers = _headers(raw_headers)
    hosts = headers.get("host", [])
    own = parse_authority(hosts[0], config.hosts) if len(hosts) == 1 else None
    if own is None:
        return Refusal.HOST
    origins = headers.get("origin", [])
    if len(origins) > 1:
        return Refusal.ORIGIN
    if origins:
        origin = origins[0]
        if origin not in config.origins:
            matched = _ORIGIN.fullmatch(origin)
            # The scheme is not compared: a TLS-terminating proxy keeps Host.
            if matched is None or parse_authority(matched.group(2), config.hosts) != own:
                return Refusal.ORIGIN
    if method not in SAFE_METHODS:
        site = headers.get("sec-fetch-site", [])
        if any(item.strip().lower() == "cross-site" for item in site):
            return Refusal.FETCH_SITE
    return None


class _RefusalLog:
    """At most one log line per refusal class per interval."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._last: dict[Refusal, float] = {}
        self._lock = threading.Lock()

    def note(self, refusal: Refusal) -> None:
        now = self._clock()
        with self._lock:
            last = self._last.get(refusal)
            if last is not None and now - last < LOG_INTERVAL_SECONDS:
                return
            self._last[refusal] = now
        LOGGER.warning("api request refused: %s", refusal.value)


Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]


class ApiOriginGuard:
    """Pure ASGI middleware: the Origin/Host guard for ``/api`` outside R18."""

    def __init__(
        self,
        app: Callable[[Scope, Receive, Send], Awaitable[None]],
        config: OriginGuardConfig,
    ) -> None:
        self._app = app
        self._config = config
        self._log = _RefusalLog()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        kind = scope["type"]
        if kind not in ("http", "websocket"):
            await self._app(scope, receive, send)  # lifespan
            return
        # Guard both spellings of the path (``root_path`` stripped or not); a
        # path either spelling of which is under /api/r18 belongs to the R18
        # guard, exactly as that guard decides it.
        raw_path = str(scope.get("path", ""))
        route_path = str(get_route_path(scope))
        if is_r18_path(raw_path) or is_r18_path(route_path):
            await self._app(scope, receive, send)
            return
        if not (is_guarded_path(raw_path) or is_guarded_path(route_path)):
            await self._app(scope, receive, send)
            return
        try:
            refusal = check_request(
                self._config, str(scope.get("method", "")), scope.get("headers", ())
            )
        except Exception:  # noqa: BLE001 - fail closed, never raise into the app
            refusal = Refusal.HOST
        if refusal is not None:
            try:
                self._log.note(refusal)
            except Exception:  # noqa: BLE001 - logging must not break a refusal
                pass
            if kind == "websocket":
                # Closed before accept: a handshake is a cross-site read too.
                await send({"type": "websocket.close", "code": 1008})
                return
            response = JSONResponse(
                {"status": "refused", "reason": refusal.value},
                status_code=403,
                headers={"Cache-Control": "no-store"},
            )
            await response(scope, receive, send)
            return
        await self._app(scope, receive, send)
