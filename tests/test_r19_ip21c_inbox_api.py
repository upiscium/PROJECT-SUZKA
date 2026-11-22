"""IP-21c: the Inbox GET / ACK routes and the ACK owner on the real app
(IP-21 D1 r3 §5, §6, §8, §10-1 F11, §10-2 F13, §10-3 IP-21c).

There is no emitter in this unit (IP-21b runs in parallel): a message is written
by ``_message`` with the real runtime and participant, from a real FAILED_TERMINAL
Wake (the trigger requires the adopted terminal).  Time is the forward-only fake
of ``r19_wake_fixtures``; the ACK owner's live clock is set to it as well.
"""

from __future__ import annotations

import ast
import inspect
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
from hashlib import sha256

from datetime import timedelta

from fastapi.testclient import TestClient
import pytest

import r19_wake_fixtures as fx
from suzka.r16.history_records import HistoryOperationKind, RecordKind
from suzka.r16.history_store import HistoryStoreCapacityError
from suzka.r19 import ack_owner, inbox, inbox_api
from suzka.r19.ack_owner import AckOutcome, InboxAckOwnerV1
from suzka.r19.claim_owner import WakeCommitRefused, _Hold
from suzka.r19.contracts import (
    AckStatus,
    InboxAckV1,
    OutboxMessageV1,
    TerminalReason,
)
from suzka.r19.inbox import AckHold, AckResult
from suzka.r19.ledger import WakeHold
from suzka.runtime import AgentEventSource, AgentEventType
from suzka.runtime.agent_runtime import AgentRuntimeAdmissionBlocked
from suzka.runtime.event_journal import EventLifecycle
from test_r16_u2a_source_owner import _app
from test_r16_u3a_plan_preview import _qualified_goal

K = HistoryOperationKind
BASE = "/api/r19/inbox"
HEADERS = {"content-type": "application/json"}
SECRET = "SENTINEL-HIDDEN-THOUGHT-7f3a"


def _ack_body(context_id: str, status: str = "ACKNOWLEDGED", **extra: object) -> dict[str, object]:
    return {"context_id": context_id, "status": status, **extra}


@pytest.fixture
def world(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    with TestClient(_app(fx.settings(tmp_path / "w"))) as client:
        state = client.app.state
        clock = fx.Time(state, monkeypatch)
        state.inbox_ack_owner._clock = clock.live
        _x, gid = _qualified_goal(state)
        yield SimpleNamespace(
            client=client, state=state, time=clock, gid=gid, monkeypatch=monkeypatch
        )


def _message(  # type: ignore[no-untyped-def]
    w, *, ttl: int = 3600, another: bool = False, created: int = 0, expire_now: bool = True
):  # type: ignore[no-untyped-def]
    """One FAILED_TERMINAL cycle and its Outbox message (the emitter's role).
    ``another``: a Wake of a different cause event (one Wake per cause)."""

    state, clock = w.state, w.time
    changes: dict[str, object] = {}
    if another:
        changes = {
            "cause_event_id": str(uuid4()),
            "evidence_fingerprint": sha256(uuid4().bytes).hexdigest(),
        }
    if expire_now:
        # Already expired: the terminal needs no clock advance (the chat route,
        # which reads the real clock, stays usable afterwards).
        wake = fx.materialize(state, w.gid, clock, not_before=-1200, expires=-600, **changes)
    else:
        wake = fx.materialize(state, w.gid, clock, not_before=-600, expires=300, **changes)
        clock.advance(301)
    done = fx.terminal(state, wake, TerminalReason.EXPIRED_WAKE)
    message = OutboxMessageV1.for_failed_cycle(
        wake,
        done.terminal,
        created_utc=clock.stamp(created),
        expires_utc=clock.stamp(created + ttl),
    )
    fx.write_record(
        state,
        K.R19_OUTBOX_EMIT,
        RecordKind.OUTBOX_MESSAGE,
        lambda _event: message,
        AgentEventType.R19_WAKE_COMPLETE,
        AgentEventSource.RUNTIME_R19_WAKE_COMPLETE,
    )
    return message


def _ack_rows(state) -> list:  # type: ignore[no-untyped-def]
    return fx.operations(state, K.R19_INBOX_ACK)


def _acks(state) -> list[InboxAckV1]:  # type: ignore[no-untyped-def]
    return fx.adopted(state, InboxAckV1)


def _journal_len(state) -> int:  # type: ignore[no-untyped-def]
    return len(state.event_journal.inspect().records)


class _AckSubmits:
    """Counts ``R19_INBOX_ACK`` submissions to the runtime (a replay must not)."""

    def __init__(self, state, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        self.count = 0
        runtime = state.agent_runtime
        real = runtime.submit
        outer = self

        def spy(event_type, source, handler, *args, **kwargs):  # type: ignore[no-untyped-def]
            if event_type is AgentEventType.R19_INBOX_ACK:
                outer.count += 1
            return real(event_type, source, handler, *args, **kwargs)

        monkeypatch.setattr(runtime, "submit", spy)


class _Prepares:
    def __init__(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        from suzka.r16.history_participant import SubjectHistoryRecordParticipant

        self.count = 0
        real = SubjectHistoryRecordParticipant.prepare
        outer = self

        def spy(self_, binding):  # type: ignore[no-untyped-def]
            outer.count += 1
            return real(self_, binding)

        monkeypatch.setattr(SubjectHistoryRecordParticipant, "prepare", spy)


def _context(w) -> str:  # type: ignore[no-untyped-def]
    return w.state.main_loop.context_registry.current_context_id


def _ack(w, message, status: str = "ACKNOWLEDGED", **extra):  # type: ignore[no-untyped-def]
    body = _ack_body(message.context_id, status, **extra)
    return w.client.post(f"{BASE}/messages/{message.message_id}/ack", json=body)


def _get(w, context_id: str, query: str = ""):  # type: ignore[no-untyped-def]
    return w.client.get(f"{BASE}/messages?context_id={context_id}{query}")


# ------------------------------------------------------------- mount / 404


def test_routes_are_not_mounted_with_r19_off(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with TestClient(_app(fx.settings(tmp_path / "off", r19=False))) as client:
        state = client.app.state
        assert all(
            not str(getattr(route, "path", "")).startswith("/api/r19")
            for route in client.app.routes
        )
        mid = "outbox-" + "a" * 64
        assert client.get(f"{BASE}/messages?context_id=x").status_code == 404
        assert client.get(BASE).status_code == 404
        assert (
            client.post(f"{BASE}/messages/{mid}/ack", json=_ack_body("x")).status_code == 404
        )
        # The owner exists in every build but HOLDs before reading anything.
        assert state.inbox_ack_owner is not None and not state.inbox_ack_owner.enabled
        owner = state.inbox_ack_owner

        def boom(*_a, **_k):  # type: ignore[no-untyped-def]
            raise AssertionError("read before the R19_DISABLED hold")

        owner._classify = boom  # type: ignore[method-assign]
        handler = owner.ack_handler(mid, "x", AckStatus.ACKNOWLEDGED, None)
        value = (
            state.agent_runtime.submit(
                AgentEventType.R19_INBOX_ACK, AgentEventSource.API_R19_INBOX_ACK, handler
            )
            .result(timeout=30)
            .value
        )
        assert value == AckOutcome(AckResult.HOLD, AckHold.R19_DISABLED)
        assert owner.read_inbox("x").hold is AckHold.R19_DISABLED
        assert not _ack_rows(state)


def test_only_the_two_routes_exist_under_the_prefix(world) -> None:  # type: ignore[no-untyped-def]
    found = {
        (method, route.path)  # type: ignore[attr-defined]
        for route in world.client.app.routes
        if str(getattr(route, "path", "")).startswith("/api/r19")
        for method in route.methods  # type: ignore[attr-defined]
    }
    assert found == {
        ("GET", f"{BASE}/messages"),
        ("POST", f"{BASE}/messages/{{message_id}}/ack"),
    }
    # No REPLY, mark-read, delete or list-all-Contexts route.
    for path, method in ((f"{BASE}/reply", "POST"), (f"{BASE}/messages", "DELETE"), (BASE, "GET")):
        assert world.client.request(method, path).status_code in (404, 405)


# ------------------------------------------------------------------- GET


def test_get_is_200_and_empty_for_every_unreadable_context(world) -> None:  # type: ignore[no-untyped-def]
    message = _message(world)
    before = _journal_len(world.state)
    empty = None
    for context in ("no-such-context", "x", world.state.main_loop.context_registry.current_context_id):
        response = _get(world, context)
        assert response.status_code == 200
        body = response.json()
        if context != message.context_id:
            empty = body["messages"] if empty is None else empty
            assert body["messages"] == []
    # No event, no write, no registry mutation by a GET.
    assert _journal_len(world.state) == before
    assert set(response.json()) == {"messages", "as_of_utc"}


def test_get_returns_the_closed_projection_only(world) -> None:  # type: ignore[no-untyped-def]
    message = _message(world)
    response = _get(world, message.context_id)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert set(body) == {"messages", "as_of_utc"}
    (item,) = body["messages"]
    assert set(item) == {
        "message_id", "template", "reason_code", "text",
        "created_utc", "expires_utc", "state", "acked_utc",
    }
    assert item["message_id"] == message.message_id
    assert (item["template"], item["reason_code"]) == ("CYCLE_FAILED", "FAILED")
    assert item["text"] == inbox.TEMPLATE_TEXT[(message.template, message.reason_code)]
    assert item["state"] == "PENDING" and item["acked_utc"] is None
    # Nothing that names a cycle, a Wake, a digest, the incarnation or the Context.
    raw = response.text
    for forbidden in (
        message.source_id, message.result_digest, message.dedupe_key,
        message.incarnation_id, message.context_id, message.message_digest,
    ):
        assert forbidden not in raw
    assert "context_id" not in raw and "source_id" not in raw


def test_get_validation_and_limit(world) -> None:  # type: ignore[no-untyped-def]
    first = _message(world)
    second = _message(world, another=True)
    context = first.context_id
    both = _get(world, context).json()["messages"]
    assert {item["message_id"] for item in both} == {first.message_id, second.message_id}
    assert [item["created_utc"] for item in both] == sorted(
        (item["created_utc"] for item in both), reverse=True
    )
    assert len(_get(world, context, "&limit=1").json()["messages"]) == 1
    for bad in ("&limit=0", "&limit=51", "&limit=x", "&limit=-1", "&limit=1&limit=2", "&extra=1"):
        assert _get(world, context, bad).status_code == 400
    assert world.client.get(f"{BASE}/messages").status_code == 400
    assert world.client.get(f"{BASE}/messages?context_id=bad id!").status_code == 400
    assert world.client.get(f"{BASE}/messages?context_id=").status_code == 400


def test_get_read_errors_are_503_with_no_body_data(world) -> None:  # type: ignore[no-untyped-def]
    message = _message(world)
    owner = world.state.inbox_ack_owner

    def blocked():  # type: ignore[no-untyped-def]
        raise _Hold(WakeHold.HISTORY_UNAVAILABLE)

    world.monkeypatch.setattr(owner, "_classify", blocked)
    response = _get(world, message.context_id)
    assert response.status_code == 503
    assert response.json() == {"status": "HISTORY_UNAVAILABLE"}
    # An unexpected store error is the same closed answer.
    def broken():  # type: ignore[no-untyped-def]
        raise TypeError("an adopted record is not what its kind names")

    world.monkeypatch.setattr(owner, "_classify", broken)
    response = _get(world, message.context_id)
    assert (response.status_code, response.json()) == (503, {"status": "HISTORY_UNAVAILABLE"})
    # ... and the pre-check of an ACK.
    assert _ack(world, message).status_code == 503


def test_get_never_selects_or_creates_a_context(world) -> None:  # type: ignore[no-untyped-def]
    from suzka.runtime import chat_context

    message = _message(world)
    registry = world.state.main_loop.context_registry
    before = registry.state

    def forbidden(*_a, **_k):  # type: ignore[no-untyped-def]
        raise AssertionError("the Inbox must not resolve a chat Context")

    world.monkeypatch.setattr(chat_context, "resolve_chat_context", forbidden)
    assert _get(world, message.context_id).status_code == 200
    assert _get(world, "never-seen-context").status_code == 200
    assert _ack(world, message).status_code == 201
    assert _ack(world, message, client_session_id="other-session").status_code in (200, 409)
    assert registry.state == before
    assert registry.current_context_id == before.current_context_id


# -------------------------------------------------------------------- ACK


def test_ack_of_a_pending_message_writes_one_row_and_one_event(world) -> None:  # type: ignore[no-untyped-def]
    state = world.state
    message = _message(world)
    submits = _AckSubmits(state, world.monkeypatch)
    response = _ack(world, message)
    assert (response.status_code, response.json()) == (
        201,
        {"result": "COMMITTED", "state": "ACKNOWLEDGED"},
    )
    assert submits.count == 1
    (row,) = _acks(state)
    assert row.status is AckStatus.ACKNOWLEDGED
    assert (row.message_id, row.context_id) == (message.message_id, message.context_id)
    assert row.session_selector == "context-only"
    assert row.canonical_value()["inbox_assurance"] == "SUZKA_API_CLIENT_V1"
    assert len(_ack_rows(state)) == 1
    (item,) = _get(world, message.context_id).json()["messages"]
    assert item["state"] == "ACKNOWLEDGED" and item["acked_utc"] == row.occurred_utc
    # Exactly one R19_INBOX_ACK event reached the Journal and none failed.
    records = state.event_journal.inspect().records
    assert [
        r for r in records
        if r.event_type is AgentEventType.R19_INBOX_ACK and r.lifecycle is EventLifecycle.ACCEPTED
    ].__len__() == 1
    assert not any(r.lifecycle is EventLifecycle.FAILED for r in records)


def test_reject_is_recorded_and_the_session_literal_is_kept(world) -> None:  # type: ignore[no-untyped-def]
    message = _message(world)
    # The default Context has no source session: a client_session_id conflicts.
    refused = _ack(world, message, "REJECTED", client_session_id="some-session")
    assert (refused.status_code, refused.json()) == (409, {"status": "CONTEXT_CONFLICT"})
    assert not _acks(world.state)
    done = _ack(world, message, "REJECTED")
    assert (done.status_code, done.json()["state"]) == (201, "REJECTED")
    (row,) = _acks(world.state)
    assert row.status is AckStatus.REJECTED and row.session_selector == "context-only"


def test_a_session_context_records_only_the_literal_selector(world) -> None:  # type: ignore[no-untyped-def]
    """A client_session_id equal to the Context's own session is recorded as given."""

    from suzka.runtime.context import ContextType

    registry = world.state.main_loop.context_registry
    registry.create("inbox.session.ctx", ContextType.CONVERSATION, "chat", source_session_id="sess-1")
    # A Wake of that Context: the message is bound to it.
    message = _message_in(world, "inbox.session.ctx")
    wrong = _ack(world, message, client_session_id="sess-2")
    assert (wrong.status_code, wrong.json()) == (409, {"status": "CONTEXT_CONFLICT"})
    assert not _acks(world.state)
    right = _ack(world, message, client_session_id="sess-1")
    assert right.status_code == 201
    assert _acks(world.state)[0].session_selector == "sess-1"


def _message_in(w, context_id: str, *, ttl: int = 3600):  # type: ignore[no-untyped-def]
    state, clock = w.state, w.time
    wake = fx.materialize(
        state, w.gid, clock, not_before=-600, expires=300, context_id=context_id
    )
    clock.advance(301)
    done = fx.terminal(state, wake, TerminalReason.EXPIRED_WAKE)
    message = OutboxMessageV1.for_failed_cycle(
        wake, done.terminal, created_utc=clock.stamp(), expires_utc=clock.stamp(ttl)
    )
    fx.write_record(
        state,
        K.R19_OUTBOX_EMIT,
        RecordKind.OUTBOX_MESSAGE,
        lambda _event: message,
        AgentEventType.R19_WAKE_COMPLETE,
        AgentEventSource.RUNTIME_R19_WAKE_COMPLETE,
    )
    return message


def test_replayed_ack_is_existing_with_no_event_no_row(world) -> None:  # type: ignore[no-untyped-def]
    state = world.state
    message = _message(world)
    assert _ack(world, message).status_code == 201
    stored = _acks(state)[0].canonical_bytes()
    submits = _AckSubmits(state, world.monkeypatch)
    rows, journal = len(_ack_rows(state)), _journal_len(state)
    for extra in ({}, {"client_session_id": "different-selector"}):
        replay = _ack(world, message, **extra)
        assert (replay.status_code, replay.json()) == (
            200,
            {"result": "EXISTING", "state": "ACKNOWLEDGED"},
        )
    assert submits.count == 0
    assert len(_ack_rows(state)) == rows and _journal_len(state) == journal
    assert _acks(state)[0].canonical_bytes() == stored  # the selector is never rewritten


@pytest.mark.parametrize(("first", "second"), [("ACKNOWLEDGED", "REJECTED"), ("REJECTED", "ACKNOWLEDGED")])
def test_a_contradicting_ack_is_a_conflict_and_the_original_stays(world, first, second) -> None:  # type: ignore[no-untyped-def]
    state = world.state
    message = _message(world)
    assert _ack(world, message, first).status_code == 201
    stored = _acks(state)[0].canonical_bytes()
    submits = _AckSubmits(state, world.monkeypatch)
    journal = _journal_len(state)
    other = _ack(world, message, second)
    assert (other.status_code, other.json()) == (409, {"status": "CONFLICT"})
    assert submits.count == 0 and _journal_len(state) == journal
    assert _acks(state)[0].canonical_bytes() == stored
    assert len(_ack_rows(state)) == 1


def test_wrong_context_and_unknown_message_are_the_same_404(world) -> None:  # type: ignore[no-untyped-def]
    state = world.state
    message = _message(world)
    submits = _AckSubmits(state, world.monkeypatch)
    journal = _journal_len(state)
    wrong = world.client.post(
        f"{BASE}/messages/{message.message_id}/ack", json=_ack_body("some-other-context")
    )
    unknown = world.client.post(
        f"{BASE}/messages/outbox-{'0' * 64}/ack", json=_ack_body(message.context_id)
    )
    assert (wrong.status_code, wrong.content) == (unknown.status_code, unknown.content)
    assert wrong.status_code == 404 and wrong.json() == {"status": "NOT_FOUND"}
    assert submits.count == 0 and _journal_len(state) == journal and not _acks(state)
    # A message id that is not even well formed is a 400, not a lookup.
    for bad in ("outbox-message-" + "a" * 64, "outbox-" + "A" * 64, "outbox-abc", "x"):
        assert (
            world.client.post(f"{BASE}/messages/{bad}/ack", json=_ack_body("x")).status_code == 400
        )


def test_an_expired_message_cannot_be_acked(world) -> None:  # type: ignore[no-untyped-def]
    state = world.state
    message = _message(world, ttl=120)
    world.time.advance(121)
    submits = _AckSubmits(state, world.monkeypatch)
    journal = _journal_len(state)
    response = _ack(world, message)
    assert (response.status_code, response.json()) == (410, {"status": "EXPIRED"})
    assert submits.count == 0 and _journal_len(state) == journal and not _acks(state)
    (item,) = _get(world, message.context_id).json()["messages"]
    assert item["state"] == "EXPIRED"
    # An ACKed message that later expires still answers its replay (the row is immutable).


def test_a_replay_after_expiry_still_answers_existing(world) -> None:  # type: ignore[no-untyped-def]
    message = _message(world, ttl=120)
    assert _ack(world, message).status_code == 201
    world.time.advance(500)
    assert _ack(world, message).status_code == 200
    assert _ack(world, message, "REJECTED").status_code == 409


def test_a_closed_context_is_refused_by_the_handler_and_a_missing_registry_skips_it(world) -> None:  # type: ignore[no-untyped-def]
    """The handler reads the registry (reachable on the runtime thread); a closed
    or suspended Context answers like an unknown message.  With no registry the
    check is skipped (A-5) and the ACK can pass."""

    state = world.state
    message = _message(world)
    registry = state.main_loop.context_registry
    registry.suspend(message.context_id)
    submits = _AckSubmits(state, world.monkeypatch)
    refused = _ack(world, message)
    assert (refused.status_code, refused.json()) == (404, {"status": "NOT_FOUND"})
    assert submits.count == 1 and not _acks(state)  # the handler, not the route, refused
    registry.resume(message.context_id)
    registry.close(message.context_id)
    assert _ack(world, message).status_code == 404
    assert not _acks(state)
    # GET does no eligibility check: the closed Context's list is still returned.
    assert len(_get(world, message.context_id).json()["messages"]) == 1
    # No registry reachable: the active check is skipped; the trigger's own
    # Context equality still holds.
    world.monkeypatch.setattr(state.inbox_ack_owner, "_registry", lambda: None)
    assert _ack(world, message).status_code == 201
    assert len(_acks(state)) == 1


def test_a_registry_that_fails_is_a_typed_hold_never_a_fail_stop(world) -> None:  # type: ignore[no-untyped-def]
    state = world.state
    message = _message(world)

    def broken():  # type: ignore[no-untyped-def]
        raise RuntimeError("registry exploded")

    world.monkeypatch.setattr(state.inbox_ack_owner, "_registry", broken)
    response = _ack(world, message)
    assert (response.status_code, response.json()) == (503, {"status": "HISTORY_UNAVAILABLE"})
    assert not _acks(state)
    world.monkeypatch.undo()
    # The runtime is still serving and chat is unaffected.
    assert state.agent_runtime.status.name == "ACCEPTING"
    assert world.client.get("/health").status_code == 200


# ----------------------------------------------------------------- guards


def test_post_request_guards(world) -> None:  # type: ignore[no-untyped-def]
    state = world.state
    message = _message(world)
    url = f"{BASE}/messages/{message.message_id}/ack"
    submits = _AckSubmits(state, world.monkeypatch)
    body = json.dumps(_ack_body(message.context_id))
    assert world.client.post(url, content=body, headers={"content-type": "text/plain"}).status_code == 415
    assert world.client.post(url, content=body).status_code == 415
    assert world.client.post(url, content="{", headers=HEADERS).status_code == 400
    assert world.client.post(url, content="[]", headers=HEADERS).status_code == 400
    big = json.dumps(_ack_body(message.context_id, pad="x" * 5000))
    assert world.client.post(url, content=big, headers=HEADERS).status_code == 413
    # Unknown keys, a wrong status and a non-string selector are refused unread.
    for bad in (
        {**_ack_body(message.context_id), "extra": 1},
        _ack_body(message.context_id, "APPROVED"),
        _ack_body(message.context_id, "acknowledged"),
        {"status": "ACKNOWLEDGED"},
        {"context_id": message.context_id},
        _ack_body(message.context_id, client_session_id=5),
        _ack_body(message.context_id, client_session_id="bad id!"),
        _ack_body(5),
    ):
        assert world.client.post(url, json=bad).status_code == 400, bad
    assert submits.count == 0 and not _acks(state)
    # JSON with a charset is accepted.
    ok = world.client.post(
        url, content=body, headers={"content-type": "application/json; charset=utf-8"}
    )
    assert ok.status_code == 201


def test_rate_cap_and_one_ack_in_flight(world) -> None:  # type: ignore[no-untyped-def]
    state = world.state
    message = _message(world)
    limits = state.inbox_limits
    submits = _AckSubmits(state, world.monkeypatch)
    # A second ACK event while one is in flight is a 429 and submits nothing.
    assert limits.event_lock.acquire(blocking=False)
    try:
        busy = _ack(world, message)
        assert (busy.status_code, busy.json()) == (429, {"status": "RATE_LIMITED"})
        # A replay needs no event, so it is not refused by the in-flight rule.
        assert _get(world, message.context_id).status_code == 200
    finally:
        limits.event_lock.release()
    assert submits.count == 0 and not _acks(state)
    # The global per-minute cap.
    world.monkeypatch.setattr(limits, "requests", inbox_api._Window(2))
    assert _get(world, message.context_id).status_code == 200
    assert _get(world, message.context_id).status_code == 200
    assert _get(world, message.context_id).status_code == 429
    assert _ack(world, message).status_code == 429
    assert limits.event_lock.acquire(blocking=False)  # the lock is released after each ACK
    limits.event_lock.release()


def test_the_window_is_a_sliding_minute() -> None:
    now = [0.0]
    window = inbox_api._Window(2, clock=lambda: now[0])
    assert window.allow() and window.allow() and not window.allow()
    now[0] = 59.9
    assert not window.allow()
    now[0] = 60.0
    assert window.allow()


def test_hard_band_refuses_the_ack_only(world) -> None:  # type: ignore[no-untyped-def]
    state = world.state
    message = _message(world)
    rows = len(_ack_rows(state))

    def blocked(*_a, **_k):  # type: ignore[no-untyped-def]
        raise AgentRuntimeAdmissionBlocked("journal hard band")

    runtime = state.agent_runtime
    real = runtime.submit

    def submit(event_type, source, handler, *a, **k):  # type: ignore[no-untyped-def]
        if event_type is AgentEventType.R19_INBOX_ACK:
            return blocked()
        return real(event_type, source, handler, *a, **k)

    world.monkeypatch.setattr(runtime, "submit", submit)
    response = _ack(world, message)
    assert response.status_code == 503
    assert response.json()["status"] in {"JOURNAL_FULL", "RUNTIME_UNAVAILABLE"}
    assert len(_ack_rows(state)) == rows and not _acks(state)
    # Chat (the subject) and GET are unaffected.
    assert _get(world, message.context_id).status_code == 200
    chat = world.client.post("/api/chat", json={"message": "still here"})
    assert chat.status_code == 200
    # The in-flight slot was released.
    assert state.inbox_limits.event_lock.acquire(blocking=False)
    state.inbox_limits.event_lock.release()


def test_a_failing_ack_event_is_a_closed_500_and_releases_the_slot(world) -> None:  # type: ignore[no-untyped-def]
    state = world.state
    message = _message(world)
    owner = state.inbox_ack_owner

    def raising(*_a, **_k):  # type: ignore[no-untyped-def]
        raise WakeCommitRefused("an invariant")

    world.monkeypatch.setattr(owner, "_current_event", raising)
    response = _ack(world, message)
    assert response.status_code == 500 and response.json() == {"status": "ACK_FAILED"}
    assert "invariant" not in response.text
    assert state.inbox_limits.event_lock.acquire(blocking=False)
    state.inbox_limits.event_lock.release()


# ---------------------------------------- the handler's own mirrored checks


def _run(w, message, status: AckStatus = AckStatus.ACKNOWLEDGED, session=None):  # type: ignore[no-untyped-def]
    """Submit the ACK event directly: the handler repeats the route's checks (a
    route's read can be stale)."""

    state = w.state
    handler = state.inbox_ack_owner.ack_handler(
        message.message_id, message.context_id, status, session
    )
    return (
        state.agent_runtime.submit(
            AgentEventType.R19_INBOX_ACK, AgentEventSource.API_R19_INBOX_ACK, handler
        )
        .result(timeout=30)
        .value
    )


def test_the_handler_repeats_every_precheck_before_prepare(world) -> None:  # type: ignore[no-untyped-def]
    """One test per mirrored condition: a miss is a typed value with no PREPARE."""

    state, mp = world.state, world.monkeypatch
    message = _message(world)
    prepares = _Prepares(mp)
    owner = state.inbox_ack_owner
    # Unknown message / wrong Context (the trigger's binding).
    other = SimpleNamespace(message_id="outbox-" + "1" * 64, context_id=message.context_id)
    assert _run(world, other) == AckOutcome(AckResult.NOT_FOUND)
    foreign = SimpleNamespace(message_id=message.message_id, context_id="another-context")
    assert _run(world, foreign) == AckOutcome(AckResult.NOT_FOUND)
    # A live ``ack:`` reservation (PREPARED, not yet adopted).
    real_live = owner._live_keys
    mp.setattr(owner, "_live_keys", lambda *keys: frozenset(keys))
    assert _run(world, message) == AckOutcome(AckResult.HOLD, AckHold.IN_FLIGHT)
    mp.setattr(owner, "_live_keys", real_live)
    # Headroom.
    def full(_size):  # type: ignore[no-untyped-def]
        raise HistoryStoreCapacityError("full")

    real_headroom = state.subject_history_store.check_headroom
    mp.setattr(state.subject_history_store, "check_headroom", full)
    assert _run(world, message) == AckOutcome(AckResult.HOLD, AckHold.HISTORY_CAPACITY)
    mp.setattr(state.subject_history_store, "check_headroom", real_headroom)
    # The store's own switch.
    mp.setattr(state.subject_history_store, "_r19_enabled", False)
    assert _run(world, message) == AckOutcome(AckResult.HOLD, AckHold.R19_DISABLED)
    mp.setattr(state.subject_history_store, "_r19_enabled", True)
    # Collisions: a live key (IN_FLIGHT) or an occupied slot (COLLISION).
    real_collisions = state.subject_history_store.prepare_collisions
    mp.setattr(
        state.subject_history_store,
        "prepare_collisions",
        lambda op: SimpleNamespace(live_keys=("k",), decision_slot_occupied=False),
    )
    assert _run(world, message) == AckOutcome(AckResult.HOLD, AckHold.IN_FLIGHT)
    mp.setattr(
        state.subject_history_store,
        "prepare_collisions",
        lambda op: SimpleNamespace(live_keys=(), decision_slot_occupied=True),
    )
    assert _run(world, message) == AckOutcome(AckResult.HOLD, AckHold.COLLISION)
    mp.setattr(state.subject_history_store, "prepare_collisions", real_collisions)
    # The Context selector.
    assert _run(world, message, session="not-the-session") == AckOutcome(
        AckResult.HOLD, AckHold.CONTEXT_CONFLICT
    )
    # A foreign incarnation (the trigger binds the incarnation as well).
    real_classify = owner._classify

    def other_incarnation():  # type: ignore[no-untyped-def]
        classified, inspection = real_classify()
        return SimpleNamespace(records=classified.records, incarnation="f" * 64), inspection

    mp.setattr(owner, "_classify", other_incarnation)
    assert _run(world, message) == AckOutcome(AckResult.NOT_FOUND)
    mp.setattr(owner, "_classify", real_classify)
    assert prepares.count == 0 and not _acks(state) and not _ack_rows(state)
    # Every refusal was a value: the runtime never failed.
    assert not any(
        r.lifecycle is EventLifecycle.FAILED for r in state.event_journal.inspect().records
    )
    # With every mirror satisfied the same call prepares exactly once.
    assert _run(world, message) == AckOutcome(AckResult.COMMITTED)
    assert prepares.count == 1 and len(_acks(state)) == 1
    # A stale route read: the handler answers the replay and the conflict itself.
    assert _run(world, message) == AckOutcome(AckResult.EXISTING)
    assert _run(world, message, AckStatus.REJECTED) == AckOutcome(AckResult.CONFLICT)
    assert prepares.count == 1 and len(_ack_rows(state)) == 1


def test_the_handler_checks_expiry_and_the_clock(world) -> None:  # type: ignore[no-untyped-def]
    state = world.state
    expiring = _message(world, ttl=120)
    future = _message(world, another=True, created=3000, ttl=600)
    assert _run(world, future) == AckOutcome(AckResult.HOLD, AckHold.CLOCK_UNVERIFIABLE)
    fresh_message = _message(world, another=True)
    world.time.advance(121)
    assert _run(world, expiring) == AckOutcome(AckResult.EXPIRED)
    assert not _acks(state)
    # The event clock is below the Journal's high-water mark (a rolled-back clock
    # cannot ACK a message that has really expired): refused, nothing written.
    ahead = world.time.now() + timedelta(days=1)
    world.monkeypatch.setattr(ack_owner, "wake_clock_floor", lambda *a, **k: ahead)
    assert _run(world, fresh_message) == AckOutcome(AckResult.HOLD, AckHold.CLOCK_UNVERIFIABLE)
    assert _ack(world, fresh_message).status_code == 503
    assert not _acks(state)


def test_two_acks_in_a_row_do_not_share_pending_state(world) -> None:  # type: ignore[no-untyped-def]
    state = world.state
    first = _message(world)
    second = _message(world, another=True)
    assert _run(world, first) == AckOutcome(AckResult.COMMITTED)
    assert state.inbox_ack_owner._pending is None and state.inbox_ack_owner._inflight is None
    assert _run(world, second) == AckOutcome(AckResult.COMMITTED)
    assert len(_acks(state)) == 2


def test_a_handler_that_fails_midway_clears_its_pending(world) -> None:  # type: ignore[no-untyped-def]
    state, mp = world.state, world.monkeypatch
    message = _message(world)
    owner = state.inbox_ack_owner

    def boom(*_a, **_k):  # type: ignore[no-untyped-def]
        raise RuntimeError("mid-handler")

    mp.setattr(owner, "_prepare", boom)
    assert _run(world, message) == AckOutcome(AckResult.HOLD, AckHold.HISTORY_UNAVAILABLE)
    assert owner._pending is None and owner._inflight is None
    mp.undo()
    owner._clock = world.time.live
    assert _run(world, message) == AckOutcome(AckResult.COMMITTED)


def test_a_handler_runs_only_inside_its_own_event(world) -> None:  # type: ignore[no-untyped-def]
    state = world.state
    message = _message(world)
    handler = state.inbox_ack_owner.ack_handler(
        message.message_id, message.context_id, AckStatus.ACKNOWLEDGED, None
    )
    # Run under another event type: refused before anything is read.
    future = state.agent_runtime.submit(
        AgentEventType.CHAT, AgentEventSource.API_CHAT, handler
    )
    with pytest.raises(Exception):
        future.result(timeout=30)
    assert not _acks(state)
    with pytest.raises(ValueError):
        state.inbox_ack_owner.ack_handler("bad", "x", AckStatus.ACKNOWLEDGED, None)


# ----------------------------------------------- A-12: no cognition reach


def _state_hash(state) -> str:  # type: ignore[no-untyped-def]
    """Every committed AgentState domain, minus the snapshot's own event counter
    and save time (those move with any event)."""

    value = state.agent_state_store.load().model_dump(mode="json")
    for own in ("last_processed_event_sequence", "saved_at"):
        value.pop(own)
    return json.dumps(value, sort_keys=True, default=str)


def _cognition(state):  # type: ignore[no-untyped-def]
    loop = state.main_loop
    driver = getattr(state, "r16_driver", None)
    return (
        # The whole committed AgentState (every domain) but its own event counter.
        _state_hash(state),
        loop.export_goal_state(),
        loop.export_motivation_state(),
        loop.export_commitment_state(),
        loop._value_system.snapshot(),
        loop.context_registry.state,
        None if driver is None else tuple(sorted(getattr(driver, "_tickets", {}))),
        len(fx.operations(state, K.R16_CYCLE_DECISION)),
    )


def test_an_ack_does_not_reach_cognition(world) -> None:  # type: ignore[no-untyped-def]
    """AgentState (goals, motivation, commitments, values, Contexts), the driver's
    tickets and the Decision chain are unchanged by an ACK; neither the driver
    callbacks nor the model are touched; and execute_with_event hands the event
    to nobody else."""

    state, mp = world.state, world.monkeypatch
    message = _message(world)
    driver = getattr(state, "r16_driver", None)
    called: list[str] = []
    if driver is not None:
        for name in ("notify_chat_terminal", "activate", "notify_cause_terminal"):
            if hasattr(driver, name):
                mp.setattr(driver, name, lambda *a, _n=name, **k: called.append(_n))
    calls_before = state.model_provider.calls
    before = _cognition(state)
    assert _ack(world, message).status_code == 201
    assert _ack(world, message).status_code == 200
    assert _cognition(state) == before
    assert state.model_provider.calls == calls_before
    assert called == []
    # Sensitivity: an ordinary chat turn does move the same fingerprint.
    assert world.client.post("/api/chat", json={"message": "hello"}).status_code == 200
    assert _state_hash(state) != before[0]


# ------------------------------------------------- REPLY (differential)


def _effects(state):  # type: ignore[no-untyped-def]
    """Structure-only effects of a turn: counts, never times or ids."""

    loop = state.main_loop
    goals = loop.export_goal_state()
    return {
        "goal_state": type(goals).__name__,
        "decisions": len(fx.operations(state, K.R16_CYCLE_DECISION)),
        "wakes": len(fx.operations(state, K.R19_WAKE_MATERIALIZE)),
        "outbox": len(fx.operations(state, K.R19_OUTBOX_EMIT)),
        "acks": len(fx.operations(state, K.R19_INBOX_ACK)),
        "contexts": len(loop.context_registry.state.frames),
    }


def _turn(client, state, text: str, **selectors):  # type: ignore[no-untyped-def]
    before = _effects(state)
    response = client.post("/api/chat", json={"message": text, **selectors})
    assert response.status_code == 200, response.text
    after = _effects(state)
    return {key: after[key] - before[key] if type(after[key]) is int else after[key] for key in after}


@pytest.mark.parametrize("selectors", [{}, {"client_session_id": "other-reply-session"}])
def test_a_reply_is_an_ordinary_chat_turn(tmp_path, monkeypatch, selectors) -> None:  # type: ignore[no-untyped-def]
    """Differential (D1 §6, §10-1): the effects of a chat turn quoting a
    ``message_id`` equal those of the same body when no such message exists, and
    no outbox / ack row is added; the message stays PENDING (also when the turn
    is typed into another Context)."""

    quoted = None
    with TestClient(_app(fx.settings(tmp_path / "with"))) as client:
        state = client.app.state
        fx.Time(state, monkeypatch)
        _x, gid = _qualified_goal(state)
        w = SimpleNamespace(client=client, state=state, time=fx.Time(state, monkeypatch), gid=gid)
        state.inbox_ack_owner._clock = w.time.live
        message = _message(w)
        quoted = message.message_id
        rows = (len(fx.operations(state, K.R19_OUTBOX_EMIT)), len(_ack_rows(state)))
        with_effects = _turn(client, state, f"re {quoted}: I will do it", **selectors)
        assert (len(fx.operations(state, K.R19_OUTBOX_EMIT)), len(_ack_rows(state))) == rows
        assert with_effects["outbox"] == 0 and with_effects["acks"] == 0
        (item,) = _get(w, message.context_id).json()["messages"]
        assert item["state"] == "PENDING"
    monkeypatch.undo()
    with TestClient(_app(fx.settings(tmp_path / "without"))) as client:
        state = client.app.state
        fx.Time(state, monkeypatch)
        _qualified_goal(state)
        # Same body, but no such message exists.
        assert not fx.operations(state, K.R19_OUTBOX_EMIT)
        without_effects = _turn(client, state, f"re {quoted}: I will do it", **selectors)
    assert with_effects == without_effects


# ------------------------------------------------------------------- F13


def test_a_sentinel_secret_reaches_no_message_response_or_log(world, caplog) -> None:  # type: ignore[no-untyped-def]
    state = world.state
    # A turn and a goal text carrying a secret and a "hidden thought".
    assert world.client.post("/api/chat", json={"message": f"my password is {SECRET}"}).status_code == 200
    message = _message(world)
    assert SECRET.encode() not in message.canonical_bytes()
    with caplog.at_level(logging.DEBUG):
        body = _get(world, message.context_id).text
        ack = _ack(world, message).text
        _ack(world, message)
    assert SECRET not in body and SECRET not in ack
    assert SECRET not in caplog.text
    for row in fx.operations(state, K.R19_OUTBOX_EMIT) + _ack_rows(state):
        assert SECRET.encode() not in row.pending_bytes


def test_every_response_string_is_closed() -> None:
    """The GET body schema: only the projection's fields; every string is a
    code-owned value, a timestamp or the message id (D1 §10-2)."""

    assert set(inbox.InboxItem.__dataclass_fields__) == {
        "message_id", "template", "reason_code", "text",
        "created_utc", "expires_utc", "state", "acked_utc",
    }
    for body in (
        {"status": status}
        for status in (
            "BAD_REQUEST", "RATE_LIMITED", "NOT_FOUND", "EXPIRED", "CONFLICT",
            "ACK_FAILED", "UNSUPPORTED_MEDIA_TYPE", "BODY_TOO_LARGE",
            *(hold.value for hold in AckHold),
        )
    ):
        assert set(body) == {"status"}


# ----------------------------------------------------------- architecture


def _src(module) -> str:  # type: ignore[no-untyped-def]
    return inspect.getsource(module)


def test_only_the_inbox_modules_read_outbox_and_ack_records() -> None:
    """D1 §5-2 / A-14: no route returns an R17-R19 row other than outbox / ack."""

    root = Path(inspect.getfile(inbox)).resolve().parents[1]
    offenders = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if "RecordKind.OUTBOX_MESSAGE" in path.read_text()
        or "RecordKind.INBOX_ACK" in path.read_text()
    }
    # Adapters never name the kinds themselves: the route goes through ``inbox``.
    assert "r19/inbox_api.py" not in offenders
    assert not {item for item in offenders if item.startswith("api/")}
    assert {"r19/inbox.py", "r19/ack_owner.py"} <= offenders
    for forbidden in ("api/context_read_model.py", "api/routes/debug.py"):
        text = (root / forbidden).read_text()
        assert "suzka.r19" not in text and "OutboxMessageV1" not in text
    # Every Explanation / response builder and route imports none of the Outbox types.
    for path in sorted((root / "api").rglob("*.py")):
        if path.name == "server.py":
            continue
        text = path.read_text()
        assert "suzka.r19.inbox" not in text and "InboxAckV1" not in text, path.name


def test_the_route_module_only_uses_the_inbox_projection() -> None:
    tree = ast.parse(_src(inbox_api))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    # No store, reader, record-kind, Journal or other R17-R19 module: a route
    # can return nothing but what the owner and ``inbox`` hand it.
    for module in imported:
        assert not module.startswith(
            ("suzka.r16.history", "suzka.r17", "suzka.r18", "suzka.r19.ledger",
             "suzka.r19.materializer", "suzka.r19.claim_owner", "suzka.r19.review_owner",
             "suzka.r19.scheduler", "suzka.runtime.event_journal")
        ), module
    source = _src(inbox_api)

    def code_names(module) -> set[str]:  # type: ignore[no-untyped-def]
        found: set[str] = set()
        for node in ast.walk(ast.parse(_src(module))):
            if isinstance(node, ast.Name):
                found.add(node.id)
            elif isinstance(node, ast.Attribute):
                found.add(node.attr)
            elif isinstance(node, ast.ImportFrom):
                found.update(alias.name for alias in node.names)
        return found

    forbidden = {"adopted_records", "RecordKind", "resolve_chat_context", "set_current",
                 "SubjectHistoryReader", "get_agent_runtime", "create", "close"}
    assert not code_names(inbox_api) & forbidden
    assert not code_names(ack_owner) & {"resolve_chat_context", "set_current", "create", "close"}
    assert "registry.get(" in _src(ack_owner)
    # The routes add no guard of their own: no token, no origin list (A-1).
    for name in ("require_admin", "Origin", "approval_origins", "X-Forwarded"):
        assert name not in source


def test_only_the_route_module_submits_the_ack_event() -> None:
    root = Path(inspect.getfile(inbox)).resolve().parents[1]
    users = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if "API_R19_INBOX_ACK" in path.read_text()
    }
    # The enum itself, the owner (binds its own event) and the route.
    assert users == {"runtime/agent_runtime.py", "r19/ack_owner.py", "r19/inbox_api.py"}
    chat = (root / "api/routes/chat.py").read_text() + (root / "r16/commands.py").read_text()
    assert "ack" not in chat.lower().replace("back", "").replace("track", "").replace("pack", "")


def test_ack_owner_writes_only_ack_rows() -> None:
    source = _src(ack_owner)
    assert "HistoryOperationKind.R19_INBOX_ACK" in source
    for other in ("R19_OUTBOX_EMIT", "R19_WAKE_CLAIM", "R19_WAKE_TERMINAL", "R16_CYCLE"):
        assert other not in source
    assert issubclass(InboxAckOwnerV1, ack_owner._WakeOwnerBase)
    assert InboxAckOwnerV1._EVENT_TYPES == frozenset({AgentEventType.R19_INBOX_ACK})
    # A closed vocabulary of reasons only.
    assert {item.name for item in ack_owner._HOLD_MAP.values()} <= {item.name for item in AckHold}


def test_neither_module_schedules_anything() -> None:
    # The route never schedules: no thread, timer or sleep in either module.
    for module in (inbox_api, ack_owner):
        source = _src(module)
        for name in ("Timer(", "Thread(", "sleep("):
            assert name not in source
