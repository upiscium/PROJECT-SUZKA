"""R19 IP-21c: the Inbox routes on the existing HTTP server (IP-21 D1 r3 §5, §8, A-1, A-7).

Routes (mounted **only when** ``r19.enabled``; otherwise nothing here exists and
every ``/api/r19/inbox/*`` path is a 404):

* ``GET  /api/r19/inbox/messages?context_id=<id>[&limit=<1..50>]``  the closed
  projection of one Context's messages, newest first; no event, never a write;
* ``POST /api/r19/inbox/messages/{message_id}/ack``
  ``{"context_id", "status": "ACKNOWLEDGED" | "REJECTED", "client_session_id"?}``.

Authentication, CORS and network exposure are **exactly those of ``/api/chat``**
(the existing server's; this module adds no guard, no token, no origin list),
and there is **no ACK command in chat text**: chat never acknowledges anything.
There is no REPLY route (a REPLY is an ordinary chat turn, D1 §6), no "mark
read", no delete and no list-all-Contexts route.

**Scope amendment A-14 / HD-21-9.**  The only R17-R19 rows any response here is
built from are ``outbox_messages`` and ``outbox_acks``, through ``inbox.py``
(``derive_inbox_view`` / ``InboxView.public_value``); ``message_id`` is a row
reference.  No other row, id or digest is returned or cited.  An architecture
test pins it.

Safety stop never blocks the subject (D1 §12-3): a refusal, a failure or a full
Journal here answers this request only.  Nothing in chat, a Decision or a Wake
waits on an Inbox route.

Request guards (D1 §5-4 step 1, A-8): ``application/json`` and a body of at most
4 KiB for POST, a global per-minute cap for the Inbox routes (120), and at most
one ``R19_INBOX_ACK`` event in flight in the process (a second is a 429), so an
API flood cannot queue events ahead of cognition.  An ACK that writes nothing
(replay, conflict, wrong Context, expired, unknown) is answered from the
pre-check read and submits no event.  The routes never call
``resolve_chat_context``: no Context is created, selected or made current.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import logging
import re
import threading
import time
from typing import Any, Final

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from suzka.identifiers import validate_identifier
from suzka.r19.ack_owner import AckOutcome, InboxAckOwnerV1
from suzka.r19.contracts import AckStatus
from suzka.r19.inbox import (
    MAX_LIST_LIMIT,
    MIN_LIST_LIMIT,
    AckEligibility,
    AckHold,
    AckResult,
    MessageState,
    is_message_id,
)
from suzka.api.runtime_execution import execute_with_event
from suzka.runtime import (
    AgentEventSource,
    AgentEventType,
    EventJournalAdmissionBand,
)

LOGGER = logging.getLogger(__name__)
API_PREFIX: Final = "/api/r19/inbox"
MAX_BODY_BYTES: Final = 4 * 1024
MAX_REQUESTS_PER_MINUTE: Final = 120
# ACK events that reach submission (a write is expected): far below the request cap.
MAX_ACK_EVENTS_PER_MINUTE: Final = 30
_RATE_WINDOW_S: Final = 60.0
_GET_QUERY: Final = frozenset({"context_id", "limit"})
_POST_KEYS: Final = frozenset({"context_id", "status", "client_session_id"})
_DIGITS: Final = re.compile(r"\A[0-9]{1,3}\Z")


class _Window:
    """A sliding one-minute cap (the IP-17c style: a constant, not a setting)."""

    def __init__(self, limit: int, *, clock: Any = time.monotonic) -> None:
        self._limit = limit
        self._clock = clock
        self._stamps: deque[float] = deque()
        self._lock = threading.Lock()

    def allow(self) -> bool:
        now = self._clock()
        with self._lock:
            while self._stamps and now - self._stamps[0] >= _RATE_WINDOW_S:
                self._stamps.popleft()
            if len(self._stamps) >= self._limit:
                return False
            self._stamps.append(now)
            return True


@dataclass(slots=True)
class InboxLimits:
    """The Inbox routes' process-wide limits."""

    requests: _Window = field(
        default_factory=lambda: _Window(MAX_REQUESTS_PER_MINUTE)
    )
    # At most one ``R19_INBOX_ACK`` event in flight in the process.
    event_lock: threading.Lock = field(default_factory=threading.Lock)
    # A separate small cap on the ACK events that reach submission.
    ack_events: _Window = field(default_factory=lambda: _Window(MAX_ACK_EVENTS_PER_MINUTE))


class _Slot:
    """A once-only release token for the single-flight lock.  The worker thread
    is the normal releaser; the coroutine frees the slot only if its await
    failed (cancelled, no thread) **before** the worker started, and then the
    worker, should it ever run, refuses to submit.  Releasing twice is a no-op."""

    def __init__(self, lock: threading.Lock) -> None:
        self._lock = lock
        self._mutex = threading.Lock()
        self._started = False
        self._released = False

    def start(self) -> bool:
        with self._mutex:
            if self._released:
                return False
            self._started = True
            return True

    def release(self) -> None:
        with self._mutex:
            if not self._released:
                self._released = True
                self._lock.release()

    def abandon(self) -> None:
        """The coroutine's release: only while the worker has not started."""

        with self._mutex:
            if not self._started and not self._released:
                self._released = True
                self._lock.release()


def _reply(body: dict[str, object], status: int = 200) -> JSONResponse:
    return JSONResponse(body, status_code=status, headers={"Cache-Control": "no-store"})


def _bad_request() -> JSONResponse:
    return _reply({"status": "BAD_REQUEST"}, 400)


def _rate() -> JSONResponse:
    return _reply({"status": "RATE_LIMITED"}, 429)


def _unavailable(reason: str = AckHold.HISTORY_UNAVAILABLE.value) -> JSONResponse:
    return _reply({"status": reason}, 503)


def _hold_response(reason: AckHold) -> JSONResponse:
    """A closed reason, no request content.  Transient holds (an ACK in flight,
    a Context selector that conflicts) are 409; the rest are 503."""

    if reason in (AckHold.IN_FLIGHT, AckHold.CONTEXT_CONFLICT):
        return _reply({"status": reason.value}, 409)
    return _unavailable(reason.value)


def _identifier(value: object) -> str | None:
    if type(value) is not str:
        return None
    try:
        return validate_identifier(value)
    except Exception:
        return None


def _owner(request: Request) -> InboxAckOwnerV1 | None:
    owner = getattr(request.app.state, "inbox_ack_owner", None)
    if type(owner) is not InboxAckOwnerV1 or not owner.enabled:
        return None
    return owner


class _TooLarge(Exception):
    pass


async def _json_body(request: Request) -> dict[str, object] | None:
    """The body, bounded while reading (never buffer more than the limit)."""

    chunks: list[bytes] = []
    total = 0
    try:
        async for chunk in request.stream():
            total += len(chunk)
            if total > MAX_BODY_BYTES:
                raise _TooLarge
            chunks.append(chunk)
    except _TooLarge:
        raise
    except Exception:
        return None
    try:
        value = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _journal_full(request: Request) -> bool:
    try:
        status = request.app.state.event_journal.admission_status()
        return status.band is EventJournalAdmissionBand.HARD
    except Exception:
        return False


def build_router(limits: InboxLimits) -> APIRouter:
    router = APIRouter(prefix=API_PREFIX, tags=["r19-inbox"])

    @router.get("/messages")
    async def list_messages(request: Request) -> JSONResponse:
        if not limits.requests.allow():
            return _rate()
        params = request.query_params
        if set(params.keys()) - _GET_QUERY or len(params.multi_items()) != len(
            set(params.keys())
        ):
            return _bad_request()
        context_id = _identifier(params.get("context_id"))
        if context_id is None:
            return _bad_request()
        limit = MAX_LIST_LIMIT
        if "limit" in params:
            raw = params["limit"]
            if _DIGITS.fullmatch(raw) is None or not (
                MIN_LIST_LIMIT <= int(raw) <= MAX_LIST_LIMIT
            ):
                return _bad_request()
            limit = int(raw)
        owner = _owner(request)
        if owner is None:
            return _unavailable()
        # Always 200 for a readable store: an unknown, empty or closed Context
        # looks the same (no existence oracle, no registry read, D1 §5-3).
        read = await run_in_threadpool(owner.read_inbox, context_id)
        if read.view is None:
            assert read.hold is not None
            return _unavailable(read.hold.value)
        return _reply(read.view.public_value(limit))

    @router.post("/messages/{message_id}/ack")
    async def acknowledge(message_id: str, request: Request) -> JSONResponse:
        if not limits.requests.allow():
            return _rate()
        content_type = request.headers.get("content-type", "")
        if content_type.split(";", 1)[0].strip().lower() != "application/json":
            return _reply({"status": "UNSUPPORTED_MEDIA_TYPE"}, 415)
        declared = request.headers.get("content-length")
        if declared is not None:
            if not (declared.isascii() and declared.isdigit()):
                return _bad_request()
            if int(declared) > MAX_BODY_BYTES:
                return _reply({"status": "BODY_TOO_LARGE"}, 413)
        try:
            body = await _json_body(request)
        except _TooLarge:
            return _reply({"status": "BODY_TOO_LARGE"}, 413)
        if body is None:
            return _bad_request()
        context_id = _identifier(body.get("context_id"))
        raw_status = body.get("status")
        session = body.get("client_session_id")
        try:
            status = AckStatus(raw_status) if type(raw_status) is str else None
        except ValueError:
            status = None
        if (
            set(body) - _POST_KEYS
            or context_id is None
            or status is None
            or not is_message_id(message_id)
            or (session is not None and _identifier(session) is None)
        ):
            return _bad_request()
        owner = _owner(request)
        if owner is None:
            return _unavailable()
        # The read-only pre-check: only a WRITE submits an event (D1 §5-4).
        pre = await run_in_threadpool(
            owner.check_ack,
            message_id,
            context_id,
            status,
            session if type(session) is str else None,
        )
        if pre.check is None:
            assert pre.hold is not None
            return _hold_response(pre.hold)
        check = pre.check
        if check.eligibility is AckEligibility.NOT_FOUND:
            return _reply({"status": "NOT_FOUND"}, 404)
        if check.eligibility is AckEligibility.EXPIRED:
            return _reply({"status": "EXPIRED"}, 410)
        if check.eligibility is AckEligibility.CONFLICT:
            return _reply({"status": "CONFLICT"}, 409)
        if check.eligibility is AckEligibility.EXISTING:
            return _reply({"result": AckResult.EXISTING.value, "state": status.value})
        if check.eligibility is AckEligibility.HOLD:
            assert check.hold is not None
            return _hold_response(check.hold)
        # One ACK event in flight in the process: a further one is a 429.
        if not limits.event_lock.acquire(blocking=False):
            return _rate()
        slot = _Slot(limits.event_lock)
        if not limits.ack_events.allow():
            slot.release()
            return _rate()
        # The worker thread releases the slot when the event is done; the
        # coroutine frees it only if the worker never started (once-only token).
        try:
            handler = owner.ack_handler(
                message_id,
                context_id,
                status,
                session if type(session) is str else None,
            )
        except BaseException:
            slot.release()
            raise
        try:
            outcome = await run_in_threadpool(_submit, request, handler, slot)
        except BaseException:
            slot.abandon()
            raise
        if isinstance(outcome, JSONResponse):
            return outcome
        return _outcome_response(outcome.result, outcome.hold, status)

    return router


def _submit(request: Request, handler: Any, slot: _Slot) -> Any:
    """Submit the one ACK event; a runtime refusal is a closed 503, never an
    exception text (a full Journal never blocks chat; it only refuses this)."""

    if not slot.start():
        # The request was abandoned before this worker began: submit nothing.
        return _unavailable("RUNTIME_UNAVAILABLE")
    try:
        return _run_event(request, handler)
    finally:
        slot.release()


def _run_event(request: Request, handler: Any) -> Any:
    runtime = request.app.state.agent_runtime
    try:
        # ``.result()`` inside has no timeout by design: the runtime always
        # settles an admitted event, and the slot must stay held until it does.
        event = execute_with_event(
            runtime,
            AgentEventType.R19_INBOX_ACK,
            AgentEventSource.API_R19_INBOX_ACK,
            handler,
        )
    except HTTPException as error:
        if error.status_code == 503:
            return _unavailable("JOURNAL_FULL" if _journal_full(request) else "RUNTIME_UNAVAILABLE")
        LOGGER.warning("R19 inbox ack event failed (%d)", error.status_code)
        return _reply({"status": "ACK_FAILED"}, 500)
    except Exception:
        LOGGER.warning("R19 inbox ack event failed", exc_info=True)
        return _reply({"status": "ACK_FAILED"}, 500)
    value = event.value
    if type(value) is not AckOutcome:
        LOGGER.warning("R19 inbox ack event returned an unexpected value")
        return _reply({"status": "ACK_FAILED"}, 500)
    return value


def _outcome_response(
    result: AckResult, hold: AckHold | None, status: AckStatus
) -> JSONResponse:
    if result is AckResult.COMMITTED:
        return _reply({"result": "COMMITTED", "state": _state_of(status)}, 201)
    if result is AckResult.EXISTING:
        return _reply({"result": "EXISTING", "state": _state_of(status)})
    if result is AckResult.CONFLICT:
        return _reply({"status": "CONFLICT"}, 409)
    if result is AckResult.EXPIRED:
        return _reply({"status": "EXPIRED"}, 410)
    if result is AckResult.NOT_FOUND:
        return _reply({"status": "NOT_FOUND"}, 404)
    assert hold is not None
    return _hold_response(hold)


def _state_of(status: AckStatus) -> str:
    return (
        MessageState.ACKNOWLEDGED.value
        if status is AckStatus.ACKNOWLEDGED
        else MessageState.REJECTED.value
    )


def mount_inbox_api(app: FastAPI) -> InboxLimits:
    """Mount the Inbox routes (``r19.enabled`` only; the caller checks it)."""

    limits = InboxLimits()
    app.include_router(build_router(limits))
    app.state.inbox_limits = limits
    return limits


__all__ = [
    "API_PREFIX",
    "InboxLimits",
    "build_router",
    "mount_inbox_api",
]
