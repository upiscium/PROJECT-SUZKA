"""R18 IP-17c: the approval API inside SUZKA (#411 D1 r7 §4, §5, §6, §9).

Routes (mounted only when ``r18.enabled``; otherwise nothing here exists and
every ``/api/r18/*`` path is a 404):

* ``GET  /api/r18/approve``                          the static approval page
* ``GET  /api/r18/approve.js``                       its script (``script-src 'self'``)
* ``GET  /api/r18/approvals``                        open Intents: ledger facts, no event
* ``POST /api/r18/approvals/{intent_id}/review``     the challenge (in memory), no event
* ``POST /api/r18/approvals/{intent_id}/confirm``    ``{nonce, code}``: issue + CLAIM
* ``GET  /api/r18/approvals/{intent_id}/status``     ledger facts, no event

**This module is the only caller of the Approval owner's confirm / claim /
challenge-issue submit methods** (an architecture test pins it): no driver,
chat or model path reaches them.  No request carries an "approved" field: the
authority is a request accepted by the guards, a live nonce and the typed code
(§4, §9; HD-12).  Anyone accepted by the guards is L1 (§9).

Request guards (every ``/api/r18/*``), applied by ``R18RequestGuard``, a pure
ASGI middleware mounted OUTSIDE the app-wide ``CORSMiddleware`` so that an
``OPTIONS`` preflight is refused before CORS could answer it (§4 rule 2):

1. **Transport.**  The API listens on loopback only (startup refuses a
   trusted-network origin otherwise), so every TCP peer is local.  A request is
   accepted only as (a) *direct localhost*: ``Host`` is a localhost origin's
   host:port and NONE of ``Forwarded`` / ``X-Forwarded-For`` /
   ``X-Forwarded-Host`` / ``X-Forwarded-Proto`` is present; or (b)
   *Caddy-forwarded*: ``Host`` is a trusted-network origin's host:port and the
   LAST ``X-Forwarded-For`` entry is in ``approval_trusted_networks``.  The
   guard reads the RAW headers: uvicorn rewrites ``request.client`` from
   ``X-Forwarded-For`` for loopback peers, so that value is never used.
2. **Data requests** (everything except the static page GETs) carry
   ``X-SUZKA-R18: 1``; ``Origin`` must be in ``approval_origins`` when present
   and is required on POSTs; ``Sec-Fetch-Site`` must be ``same-origin`` when
   present; POSTs are ``application/json`` with a ``Content-Length`` of at most
   16 KiB.
3. **Limits.**  A hard request rate limit; reviews per minute; and one
   approval event (confirm) in flight in total (a further one is a 429), so an
   API flood cannot delay cognition events (§8a).

``/health`` and every route outside ``/api/r18`` are untouched.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Awaitable, Callable, Iterable, MutableMapping, Sequence
from dataclasses import dataclass
from enum import Enum
import ipaddress
import json
from pathlib import Path
import re
import threading
import time
from typing import Any, Final

from fastapi import APIRouter, FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response

from suzka.config.schema import (
    R18Settings,
    approval_host_header,
    is_localhost_origin,
    parse_approval_origin,
)
from suzka.r18.approval_owner import (
    ApprovalChallengeOwnerV1,
    ApprovalFacts,
    ApprovalStatus,
    FactsResult,
    R18ApprovalReason,
    ReviewResult,
    ReviewStatus,
)
from suzka.r16.history_records import MAX_CHALLENGES_PER_INTENT

API_PREFIX: Final = "/api/r18"
PAGE_PATH: Final = "/api/r18/approve"
SCRIPT_PATH: Final = "/api/r18/approve.js"
STATIC_PATHS: Final = frozenset({PAGE_PATH, SCRIPT_PATH})
DATA_HEADER: Final = "x-suzka-r18"
MAX_BODY_BYTES: Final = 16 * 1024
# Hard limits (§4): every accepted data request, and confirms.  Reviews use
# ``R18Settings.approval_reviews_per_minute``.
MAX_DATA_REQUESTS_PER_MINUTE: Final = 120
MAX_CONFIRMS_PER_MINUTE: Final = 30
PAGE_HEADERS: Final = {
    "Content-Security-Policy": "frame-ancestors 'none'; script-src 'self'",
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Cache-Control": "no-store",
    "Referrer-Policy": "no-referrer",
}
_FORWARDING_HEADERS: Final = (
    "forwarded",
    "x-forwarded-for",
    "x-forwarded-host",
    "x-forwarded-proto",
)
_INTENT_ID: Final = re.compile(r"\Aintent-[0-9a-f]{64}\Z")
_PAGE_DIR: Final = Path(__file__).with_name("approval_page")
_Network = ipaddress.IPv4Network | ipaddress.IPv6Network


class Refusal(str, Enum):
    """Why a request was refused: closed, and free of request content."""

    OPTIONS = "options_refused"
    METHOD = "method_not_allowed"
    HOST = "host_not_accepted"
    FORWARDING = "forwarding_headers_not_accepted"
    NETWORK = "client_network_not_accepted"
    DATA_HEADER = "missing_data_header"
    ORIGIN = "origin_not_accepted"
    FETCH_SITE = "fetch_site_not_same_origin"
    CONTENT_TYPE = "content_type_not_json"
    BODY_SIZE = "body_missing_or_too_large"
    RATE = "rate_limited"


_STATUS: Final = {
    Refusal.OPTIONS: 405,
    Refusal.METHOD: 405,
    Refusal.BODY_SIZE: 413,
    Refusal.RATE: 429,
}


@dataclass(frozen=True, slots=True)
class GuardConfig:
    """The request guards' parsed configuration (immutable)."""

    localhost_hosts: frozenset[str]
    network_hosts: frozenset[str]
    origins: frozenset[str]
    networks: tuple[_Network, ...]
    reviews_per_minute: int

    @classmethod
    def from_settings(cls, r18: R18Settings) -> GuardConfig:
        localhost: set[str] = set()
        network: set[str] = set()
        for origin in r18.approval_origins:
            parse_approval_origin(origin)
            target = localhost if is_localhost_origin(origin) else network
            target.add(approval_host_header(origin).lower())
        return cls(
            frozenset(localhost),
            frozenset(network),
            frozenset(r18.approval_origins),
            tuple(ipaddress.ip_network(item) for item in r18.approval_trusted_networks),
            r18.approval_reviews_per_minute,
        )


def is_r18_path(path: str) -> bool:
    return path == API_PREFIX or path.startswith(API_PREFIX + "/")


def _headers(raw: Iterable[tuple[bytes, bytes]]) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for name, value in raw:
        found.setdefault(name.decode("latin-1").lower(), []).append(value.decode("latin-1"))
    return found


def _client_in(networks: Sequence[_Network], forwarded: list[str]) -> bool:
    """The last ``X-Forwarded-For`` entry (Caddy sets it from the real peer and
    ignores a client-supplied value) is inside the trusted networks."""

    entries = [part.strip() for line in forwarded for part in line.split(",")]
    if not entries or not entries[-1]:
        return False
    try:
        address = ipaddress.ip_address(entries[-1])
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return any(address in network for network in networks if address.version == network.version)


def check_request(
    config: GuardConfig,
    method: str,
    path: str,
    raw_headers: Iterable[tuple[bytes, bytes]],
) -> Refusal | None:
    """Rules 1 and 2 over the RAW headers; None when accepted.  Pure."""

    if method == "OPTIONS":
        # Refused before the CORS middleware could answer a preflight.
        return Refusal.OPTIONS
    if method not in ("GET", "POST"):
        return Refusal.METHOD
    headers = _headers(raw_headers)
    hosts = headers.get("host", [])
    if len(hosts) != 1:
        return Refusal.HOST
    host = hosts[0].lower()
    # 1. the transport.
    if host in config.localhost_hosts:
        if any(name in headers for name in _FORWARDING_HEADERS):
            return Refusal.FORWARDING
    elif host in config.network_hosts:
        if not _client_in(config.networks, headers.get("x-forwarded-for", [])):
            return Refusal.NETWORK
    else:
        return Refusal.HOST
    if method == "GET" and path in STATIC_PATHS:
        return None
    # 2. data requests.
    if headers.get(DATA_HEADER) != ["1"]:
        return Refusal.DATA_HEADER
    origins = headers.get("origin", [])
    if len(origins) > 1 or (origins and origins[0] not in config.origins):
        return Refusal.ORIGIN
    if method == "POST" and not origins:
        return Refusal.ORIGIN
    fetch_site = headers.get("sec-fetch-site", [])
    if fetch_site and fetch_site != ["same-origin"]:
        return Refusal.FETCH_SITE
    if method == "POST":
        types = headers.get("content-type", [])
        if len(types) != 1 or types[0].split(";")[0].strip().lower() != "application/json":
            return Refusal.CONTENT_TYPE
        lengths = headers.get("content-length", [])
        if len(lengths) != 1 or not lengths[0].isdecimal() or int(lengths[0]) > MAX_BODY_BYTES:
            return Refusal.BODY_SIZE
    return None


class _Window:
    """A sliding one-minute counter."""

    def __init__(self, limit: int, clock: Callable[[], float]) -> None:
        self._limit = limit
        self._clock = clock
        self._stamps: deque[float] = deque()
        self._lock = threading.Lock()

    def allow(self) -> bool:
        now = self._clock()
        with self._lock:
            while self._stamps and now - self._stamps[0] >= 60.0:
                self._stamps.popleft()
            if len(self._stamps) >= self._limit:
                return False
            self._stamps.append(now)
            return True


@dataclass(slots=True)
class ApprovalLimits:
    """The approval API's hard limits (§4)."""

    data: _Window
    reviews: _Window
    confirms: _Window
    # One approval event in flight in total.
    event_lock: threading.Lock

    @classmethod
    def create(
        cls, config: GuardConfig, clock: Callable[[], float] = time.monotonic
    ) -> ApprovalLimits:
        return cls(
            _Window(MAX_DATA_REQUESTS_PER_MINUTE, clock),
            _Window(config.reviews_per_minute, clock),
            _Window(MAX_CONFIRMS_PER_MINUTE, clock),
            threading.Lock(),
        )


Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]


class R18RequestGuard:
    """Pure ASGI middleware: the request guards for ``/api/r18/*``.

    Mounted after (so outside) the app-wide ``CORSMiddleware``.  A refusal is
    answered here with no CORS headers and nothing reaches a route.
    """

    def __init__(self, app: Callable[[Scope, Receive, Send], Awaitable[None]], config: GuardConfig, limits: ApprovalLimits) -> None:
        self._app = app
        self._config = config
        self._limits = limits

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not is_r18_path(str(scope.get("path", ""))):
            await self._app(scope, receive, send)
            return
        refusal = check_request(
            self._config,
            str(scope.get("method", "")),
            str(scope["path"]),
            scope.get("headers", ()),
        )
        if refusal is None and not self._limits.data.allow():
            refusal = Refusal.RATE
        if refusal is not None:
            response = JSONResponse(
                {"status": "refused", "reason": refusal.value},
                status_code=_STATUS.get(refusal, 403),
                headers={"Cache-Control": "no-store"},
            )
            await response(scope, receive, send)
            return
        await self._app(scope, receive, send)


# ------------------------------------------------------------------ routes


def _no_store(body: dict[str, object], status: int = 200) -> JSONResponse:
    return JSONResponse(body, status_code=status, headers={"Cache-Control": "no-store"})


def _hold(reason: R18ApprovalReason) -> JSONResponse:
    status = 429 if reason is R18ApprovalReason.REVIEW_BUSY else 409
    return _no_store({"status": "hold", "reason": reason.value}, status)


def _owner(request: Request) -> ApprovalChallengeOwnerV1 | None:
    owner = getattr(request.app.state, "approval_owner", None)
    if type(owner) is not ApprovalChallengeOwnerV1 or not owner.enabled:
        return None
    return owner


def _fact(item: ApprovalFacts) -> dict[str, object]:
    return {
        "intent_id": item.intent_id,
        "ledger_state": item.ledger_state,
        "decision_id": item.decision_id,
        "window_end_utc": item.window_end_utc,
        "attempts_used": item.attempts_used,
        "max_attempts": MAX_CHALLENGES_PER_INTENT,
        "live_challenge": item.live_challenge,
        "challenge_expires_utc": item.challenge_expires_utc,
        "commitment_id": item.commitment_id,
        "commitment_deadline_utc": item.commitment_deadline_utc,
        "target_instance_id": item.target_instance_id,
    }


def _facts_response(result: FactsResult) -> JSONResponse:
    if result.reason is not None:
        return _hold(result.reason)
    return _no_store({"status": "ok", "approvals": [_fact(item) for item in result.items]})


def _review_body(result: ReviewResult) -> dict[str, object] | None:
    challenge = result.challenge
    intent = result.intent
    if challenge is None or intent is None:
        return None
    return {
        "status": "review",
        "source": result.status.value,
        "intent_id": result.intent_id,
        # Canonical bytes: the page recomputes every digest and the code.
        "challenge": challenge.canonical_bytes().decode("utf-8"),
        "challenge_digest": challenge.challenge_digest,
        "intent": intent.canonical_bytes().decode("utf-8"),
        "forecast_digest": challenge.expectation_digest,
        "attempt": result.attempt,
        "max_attempts": MAX_CHALLENGES_PER_INTENT,
        "window_end_utc": result.window_end_utc,
        "expires_utc": challenge.expires_utc,
    }


async def _json_body(request: Request) -> dict[str, object] | None:
    try:
        body = await request.body()
    except Exception:
        return None
    if len(body) > MAX_BODY_BYTES:
        return None
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def load_page() -> tuple[bytes, bytes]:
    """The static page and its script, read once at mount time.  A missing
    file is a packaging error and fails startup."""

    return (
        (_PAGE_DIR / "approve.html").read_bytes(),
        (_PAGE_DIR / "approve.js").read_bytes(),
    )


def build_router(limits: ApprovalLimits) -> APIRouter:
    html, script = load_page()
    router = APIRouter(prefix=API_PREFIX, tags=["r18"])

    @router.get("/approve")
    def page() -> Response:
        return Response(html, media_type="text/html; charset=utf-8", headers=PAGE_HEADERS)

    @router.get("/approve.js")
    def page_script() -> Response:
        return Response(
            script, media_type="application/javascript; charset=utf-8", headers=PAGE_HEADERS
        )

    @router.get("/approvals")
    async def list_approvals(request: Request) -> JSONResponse:
        owner = _owner(request)
        if owner is None:
            return _no_store({"status": "unavailable"}, 503)
        return _facts_response(await run_in_threadpool(owner.open_approvals))

    @router.get("/approvals/{intent_id}/status")
    async def approval_status(intent_id: str, request: Request) -> JSONResponse:
        owner = _owner(request)
        if owner is None:
            return _no_store({"status": "unavailable"}, 503)
        if not _INTENT_ID.fullmatch(intent_id):
            return _no_store({"status": "unknown_intent"}, 404)
        return _facts_response(await run_in_threadpool(owner.status, intent_id))

    @router.post("/approvals/{intent_id}/review")
    async def review(intent_id: str, request: Request) -> JSONResponse:
        owner = _owner(request)
        if owner is None:
            return _no_store({"status": "unavailable"}, 503)
        if not _INTENT_ID.fullmatch(intent_id):
            return _no_store({"status": "unknown_intent"}, 404)
        if not limits.reviews.allow():
            return _no_store({"status": "refused", "reason": Refusal.RATE.value}, 429)
        result = await run_in_threadpool(owner.review, intent_id)
        if result.status is ReviewStatus.HOLD:
            assert result.reason is not None
            return _hold(result.reason)
        body = _review_body(result)
        if body is None:
            return _hold(R18ApprovalReason.HISTORY_UNAVAILABLE)
        return _no_store(body)

    @router.post("/approvals/{intent_id}/confirm")
    async def confirm(intent_id: str, request: Request) -> JSONResponse:
        owner = _owner(request)
        if owner is None:
            return _no_store({"status": "unavailable"}, 503)
        if not _INTENT_ID.fullmatch(intent_id):
            return _no_store({"status": "unknown_intent"}, 404)
        body = await _json_body(request)
        nonce = None if body is None else body.get("nonce")
        code = None if body is None else body.get("code")
        if (
            body is None
            or set(body) != {"nonce", "code"}
            or type(nonce) is not str
            or type(code) is not str
            or not 0 < len(nonce) <= 128
            or not 0 < len(code) <= 32
        ):
            return _no_store({"status": "bad_request"}, 400)
        if not limits.confirms.allow():
            return _no_store({"status": "refused", "reason": Refusal.RATE.value}, 429)
        # One approval event in flight in total: a further confirm is a 429.
        if not limits.event_lock.acquire(blocking=False):
            return _no_store({"status": "refused", "reason": Refusal.RATE.value}, 429)
        try:
            result = await run_in_threadpool(owner.confirm, intent_id, nonce, code)
        finally:
            limits.event_lock.release()
        if result.status is ApprovalStatus.CLAIMED and result.claim is not None:
            return _no_store(
                {
                    "status": "claimed",
                    "intent_id": intent_id,
                    "claim_id": result.claim.claim_id,
                    "attempt": result.attempt,
                }
            )
        reason = result.reason or R18ApprovalReason.HISTORY_UNAVAILABLE
        return _hold(reason)

    return router


def mount_approval_api(app: FastAPI, r18: R18Settings) -> ApprovalLimits:
    """Mount the routes and the request guard (``r18.enabled`` only).

    ``add_middleware`` puts the guard OUTSIDE every middleware added before it,
    so call this after the app-wide ``CORSMiddleware`` is added.
    """

    config = GuardConfig.from_settings(r18)
    limits = ApprovalLimits.create(config)
    app.include_router(build_router(limits))
    app.add_middleware(R18RequestGuard, config=config, limits=limits)
    app.state.approval_guard_config = config
    app.state.approval_limits = limits
    return limits


__all__ = [
    "API_PREFIX",
    "GuardConfig",
    "R18RequestGuard",
    "Refusal",
    "build_router",
    "check_request",
    "is_r18_path",
    "mount_approval_api",
]
