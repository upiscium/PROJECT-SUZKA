"""IP-6b (#383 / #370 R20-DRV D0-D2): the R16 production driver.

Integration tests drive the real app through ``/api/chat``; unit tests use
fakes to pin timing (E deferral, held commands, retries, drops).
"""

from __future__ import annotations

from concurrent.futures import Future
import time
from typing import Any

from fastapi.testclient import TestClient
import pytest

from suzka.motivation.goal import GoalLifecycle
from suzka.r16.driver import (
    IDLE_WAIT_LIMIT_SECONDS,
    R16CommandKind,
    R16Driver,
    R16DriverReason,
    R16TicketStatus,
    parse_command,
)
from suzka.r16.outcomes import R16OwnerOutcome, R16OwnerReason, R16OwnerStatus
from suzka.runtime.agent_runtime import (
    AgentEvent,
    AgentEventOutcome,
    AgentEventSource,
    AgentEventType,
    AgentRuntimeQueueFull,
)
from suzka.runtime.event_journal import EventLifecycle
from suzka.runtime.main_loop import ChatResult

from test_r16_ip4c_past_retention import app  # noqa: F401
from test_r16_u2a_source_owner import _app

TOKEN = "ip6b-admin-token"
ADMIN = {"X-SUZKA-Admin-Token": TOKEN}
EXP = "experience-00000000-0000-4000-8000-000000000001"


@pytest.fixture(autouse=True)
def _admin_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUZKA_P1B_TEST_TOKEN", TOKEN)


def _until(predicate, timeout: float = 30.0):  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("condition not reached")


def _chat(client: TestClient, text: str, **selectors: str) -> dict[str, Any]:
    response = client.post(
        "/api/chat", json={"message": text, "attachments": [], **selectors}
    )
    assert response.status_code == 200, response.text
    return response.json()


def _settled(client: TestClient, ticket_id: str) -> dict[str, Any]:
    def view() -> dict[str, Any] | None:
        body = client.get(f"/api/chat/commands/{ticket_id}").json()
        return None if body["status"] == "pending" else body

    return _until(view)


def _activate(client: TestClient) -> None:
    assert client.post("/api/r16/driver/activate", headers=ADMIN).status_code == 200


def _interest(client: TestClient, experience_id: str) -> None:
    state = client.app.state

    def present() -> bool:
        return any(
            record.target.reference == experience_id
            for record in state.main_loop.export_motivation_state().records
        )

    _until(present)


# --- parsing ---------------------------------------------------------------------


def test_commands_are_recognized_by_exact_fullmatch_only() -> None:
    assert parse_command(f"REVIEW_EXPERIENCE:{EXP}") == (
        R16CommandKind.REVIEW_EXPERIENCE,
        EXP,
    )
    assert parse_command(f"COMMIT_REVIEW_EXPERIENCE:1:{EXP}") == (
        R16CommandKind.COMMIT_REVIEW_EXPERIENCE,
        EXP,
    )
    for text in (
        f" REVIEW_EXPERIENCE:{EXP}",
        f"REVIEW_EXPERIENCE:{EXP} ",
        f"review_experience:{EXP}",
        f"COMMIT_REVIEW_EXPERIENCE:{EXP}",
        f"COMMIT_REVIEW_EXPERIENCE:2:{EXP}",
        "REVIEW_EXPERIENCE:experience-not-a-uuid",
        f"please REVIEW_EXPERIENCE:{EXP}",
    ):
        assert parse_command(text) is None


# --- the real app ---------------------------------------------------------------------


def test_an_inactive_driver_does_nothing(app) -> None:  # noqa: F811
    with TestClient(app) as client:
        state = client.app.state
        driver = state.r16_driver
        assert isinstance(driver, R16Driver) and not driver.active
        x = _chat(client, "an experience before activation")
        command = _chat(client, f"REVIEW_EXPERIENCE:{x['experience_id']}")
        assert command["command_ticket"] is None
        assert command["incarnation"] is None
        time.sleep(0.5)
        assert driver.counters["submitted"] == 0
        assert state.main_loop.export_motivation_state().records == ()


def test_review_command_is_chained_to_adoption_and_one_preview(app) -> None:  # noqa: F811
    with TestClient(app) as client:
        state = client.app.state
        _activate(client)
        x = _chat(client, "something wonderful happened today")
        _interest(client, x["experience_id"])
        command = _chat(client, f"REVIEW_EXPERIENCE:{x['experience_id']}")
        ticket = command["command_ticket"]
        assert ticket["kind"] == "REVIEW_EXPERIENCE"
        assert ticket["status"] == "pending"
        assert command["incarnation"] == state.r16_driver.incarnation
        assert _settled(client, ticket["ticket_id"]) == {
            "status": "adopted",
            "reason": None,
        }
        goals = state.main_loop.export_goal_state().records
        assert [goal.lifecycle for goal in goals] == [GoalLifecycle.ADOPTED]

        def previewed() -> int:
            return sum(
                1
                for row in state.event_journal.records
                if row.event_type is AgentEventType.PLAN_PREVIEW_REVIEW
                and row.lifecycle is EventLifecycle.COMPLETED
            )

        assert _until(previewed) == 1
        # A resend is held by the owner, with its closed reason.
        again = _chat(client, f"REVIEW_EXPERIENCE:{x['experience_id']}")
        assert _settled(client, again["command_ticket"]["ticket_id"]) == {
            "status": "hold",
            "reason": "goal_already_exists",
        }
        time.sleep(0.5)
        assert previewed() == 1


def test_commit_command_is_accepted_and_previews_the_adopted_goal(app) -> None:  # noqa: F811
    with TestClient(app) as client:
        state = client.app.state
        _activate(client)
        x = _chat(client, "something wonderful happened today")
        _interest(client, x["experience_id"])
        review = _chat(client, f"REVIEW_EXPERIENCE:{x['experience_id']}")
        assert _settled(client, review["command_ticket"]["ticket_id"])["status"] == (
            "adopted"
        )
        commit = _chat(client, f"COMMIT_REVIEW_EXPERIENCE:1:{x['experience_id']}")
        assert _settled(client, commit["command_ticket"]["ticket_id"]) == {
            "status": "accepted",
            "reason": None,
        }

        def previews() -> int:
            return sum(
                1
                for row in state.event_journal.records
                if row.event_type is AgentEventType.PLAN_PREVIEW_REVIEW
                and row.lifecycle is EventLifecycle.COMPLETED
            )

        _until(lambda: previews() >= 2)
        time.sleep(0.5)
        assert previews() == 2


def test_unobserved_or_other_context_targets_are_indistinguishable(app) -> None:  # noqa: F811
    with TestClient(app) as client:
        _activate(client)
        other = _chat(client, "another context's turn", client_session_id="other")
        reasons = []
        for target in (EXP, other["experience_id"]):
            command = _chat(client, f"REVIEW_EXPERIENCE:{target}")
            reasons.append(_settled(client, command["command_ticket"]["ticket_id"]))
        assert reasons == [
            {"status": "unavailable", "reason": "source_not_observed"},
            {"status": "unavailable", "reason": "source_not_observed"},
        ]


def test_tickets_are_capabilities_and_do_not_survive_restart(app) -> None:  # noqa: F811
    settings = app.state.settings
    with TestClient(app) as client:
        _activate(client)
        command = _chat(client, f"REVIEW_EXPERIENCE:{EXP}")
        ticket_id = command["command_ticket"]["ticket_id"]
        _settled(client, ticket_id)
        incarnation = client.app.state.r16_driver.incarnation
        unknown = f"{incarnation}.{'A' * 22}"
        assert client.get(f"/api/chat/commands/{unknown}").status_code == 404
        assert client.get("/api/chat/commands/not-a-ticket").status_code == 404
    with TestClient(_app(settings)) as client:
        driver = client.app.state.r16_driver
        assert driver.active  # the activation marker persists
        assert driver.incarnation != incarnation
        body = client.get(f"/api/chat/commands/{ticket_id}").json()
        assert body == {"status": "unknown_after_restart", "reason": None}
    # Shutdown joined the driver before the runtime stopped.
    assert driver._thread is None


def test_neutral_turns_are_not_submitted_to_u2a(app, monkeypatch) -> None:  # noqa: F811
    import dataclasses

    from suzka.experience.records import ExperienceAppraisalEvidence

    original = ExperienceAppraisalEvidence.from_result.__func__  # type: ignore[attr-defined]

    def neutral(cls, result):  # type: ignore[no-untyped-def]
        return dataclasses.replace(
            original(cls, result),
            novelty=None,
            novelty_valid=False,
            goal_progress=None,
            threat=None,
        )

    monkeypatch.setattr(
        ExperienceAppraisalEvidence, "from_result", classmethod(neutral)
    )
    with TestClient(app) as client:
        _activate(client)
        driver = client.app.state.r16_driver
        _chat(client, "an entirely neutral line")
        _until(lambda: driver.counters["skipped_neutral"] == 1)
        assert driver.counters["submitted"] == 0


# --- fakes for timing ---------------------------------------------------------------


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _FakeRuntime:
    def __init__(self) -> None:
        self.pending = 0

    def pending_count(self) -> int:
        return self.pending


class _FakeOwner:
    """Records submissions; futures are resolved by the test."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.futures: list[Future[Any]] = []
        self.raise_next: list[BaseException] = []

    def _record(self, name: str, argument: object) -> Future[Any]:
        if self.raise_next:
            raise self.raise_next.pop(0)
        self.calls.append((name, argument))
        future: Future[Any] = Future()
        self.futures.append(future)
        return future

    def _submit(self, ref: object) -> Future[Any]:
        return self._record("u2a", ref)

    def _submit_proposal(self, selection: object) -> Future[Any]:
        return self._record("proposal", selection)

    def _submit_review(self, subject: str) -> Future[Any]:
        return self._record("review", subject)

    def _submit_preview(self, goal_id: str) -> Future[Any]:
        return self._record("preview", goal_id)


def _event(sequence: int, source=AgentEventSource.API_CHAT) -> AgentEvent:  # type: ignore[no-untyped-def]
    import uuid

    return AgentEvent(
        str(uuid.uuid4()),
        AgentEventType.CHAT,
        source,
        __import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        sequence,
    )


def _turn(
    sequence: int,
    experience_id: str | None,
    *,
    eligible: bool = True,
    context: str = "context-a",
) -> AgentEventOutcome[ChatResult]:
    return AgentEventOutcome(
        _event(sequence),
        ChatResult(
            episode_id=f"episode-{sequence}",
            response="",
            loss=None,
            valence=0.0,
            arousal=0.0,
            optimal_loss=0.0,
            model_id="m",
            adapter_id=None,
            experience_id=experience_id,
            eligible_for_u2a=eligible,
            context_id=context,
        ),
    )


def _resolve(future: Future[Any], sequence: int, value: object) -> None:
    future.set_result(AgentEventOutcome(_event(sequence), value))


@pytest.fixture
def fakes():  # type: ignore[no-untyped-def]
    runtime, source, goal, commitment, preview = (
        _FakeRuntime(),
        _FakeOwner(),
        _FakeOwner(),
        _FakeOwner(),
        _FakeOwner(),
    )
    clock = _Clock()
    driver = R16Driver(
        runtime,  # type: ignore[arg-type]
        source,  # type: ignore[arg-type]
        goal,  # type: ignore[arg-type]
        commitment,  # type: ignore[arg-type]
        preview,  # type: ignore[arg-type]
        active=True,
        clock=clock,
    )
    driver.start()
    yield driver, runtime, source, goal, commitment, preview, clock
    driver.stop()


X1 = "experience-00000000-0000-4000-8000-0000000000a1"


def test_u2a_is_deferred_while_busy_then_submitted(fakes) -> None:  # type: ignore[no-untyped-def]
    driver, runtime, source, *_rest, clock = fakes
    runtime.pending = 2
    driver.notify_chat_terminal(_turn(5, X1), "hello")
    time.sleep(0.5)
    assert source.calls == []
    clock.now += IDLE_WAIT_LIMIT_SECONDS
    _until(lambda: len(source.calls) == 1)


def test_command_waits_for_u2a_and_is_released_by_sequence(fakes) -> None:  # type: ignore[no-untyped-def]
    driver, runtime, source, goal, *_rest = fakes
    runtime.pending = 1
    driver.notify_chat_terminal(_turn(5, X1), "hello")
    ticket = driver.notify_chat_terminal(_turn(6, "experience-00000000-0000-4000-8000-0000000000a2"), f"REVIEW_EXPERIENCE:{X1}")
    assert ticket is not None
    # The command stops the deferral: U2a is submitted now; the command waits.
    _until(lambda: len(source.calls) == 1)
    assert goal.calls == []
    # U2a admitted after P: the command cannot qualify, so the user resends.
    _resolve(source.futures[0], 7, "motivation_evidence_applied")
    view = _until(
        lambda: (v := driver.ticket(ticket.ticket_id)).status is not R16TicketStatus.PENDING
        and v
    )
    assert view.status is R16TicketStatus.HOLD
    assert view.reason is R16DriverReason.SOURCE_NOT_YET_EVALUATED


def test_command_after_u2a_with_earlier_sequence_is_submitted(fakes) -> None:  # type: ignore[no-untyped-def]
    driver, runtime, source, goal, *_rest = fakes
    driver.notify_chat_terminal(_turn(5, X1), "hello")
    _until(lambda: len(source.calls) == 1)
    ticket = driver.notify_chat_terminal(
        _turn(8, "experience-00000000-0000-4000-8000-0000000000a3"),
        f"REVIEW_EXPERIENCE:{X1}",
    )
    _resolve(source.futures[0], 6, "motivation_evidence_applied")
    _until(lambda: len(goal.calls) == 1)
    _resolve(goal.futures[0], 9, R16OwnerOutcome(R16OwnerStatus.PROPOSED, "goal-1"))
    _until(lambda: len(goal.calls) == 2)
    assert goal.calls[1] == ("review", "goal-1")
    _resolve(goal.futures[1], 10, R16OwnerOutcome.hold(R16OwnerReason.CONFLICTING_GOAL, "goal-1"))
    view = _until(
        lambda: (v := driver.ticket(ticket.ticket_id)).status is not R16TicketStatus.PENDING
        and v
    )
    assert (view.status, view.reason) == (
        R16TicketStatus.HOLD,
        R16OwnerReason.CONFLICTING_GOAL,
    )


def test_queue_full_submissions_are_retried_then_give_up(fakes) -> None:  # type: ignore[no-untyped-def]
    driver, runtime, source, goal, _c, _p, clock = fakes
    event = _event(1)
    goal.raise_next = [AgentRuntimeQueueFull(event)] * 4
    driver.notify_chat_terminal(_turn(5, X1, eligible=False), "hello")
    ticket = driver.notify_chat_terminal(
        _turn(6, "experience-00000000-0000-4000-8000-0000000000a4"),
        f"REVIEW_EXPERIENCE:{X1}",
    )

    def settled():  # type: ignore[no-untyped-def]
        clock.now += 5  # every pending retry becomes due at the next poll
        view = driver.ticket(ticket.ticket_id)
        return view if view.status is not R16TicketStatus.PENDING else None

    view = _until(settled)
    assert (view.status, view.reason) == (
        R16TicketStatus.UNAVAILABLE,
        R16DriverReason.RUNTIME_UNAVAILABLE,
    )
    assert goal.calls == []
    assert goal.raise_next == []  # first try and three retries


def test_results_are_never_dropped_even_when_inputs_are_full(fakes, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Review P1-1: a full input queue cannot lose a submitted stage's result,
    so a held command never wedges."""
    import queue as queue_module

    driver, runtime, source, goal, *_rest = fakes
    driver.notify_chat_terminal(_turn(5, X1), "hello")
    _until(lambda: len(source.calls) == 1)
    ticket = driver.notify_chat_terminal(
        _turn(8, "experience-00000000-0000-4000-8000-0000000000a6"),
        f"REVIEW_EXPERIENCE:{X1}",
    )
    time.sleep(0.3)

    def full(_item: object) -> None:
        raise queue_module.Full

    monkeypatch.setattr(driver._queue, "put_nowait", full)
    _resolve(source.futures[0], 6, "motivation_evidence_applied")
    _until(lambda: len(goal.calls) == 1)
    _resolve(goal.futures[0], 9, R16OwnerOutcome.hold(R16OwnerReason.GOAL_ALREADY_EXISTS))
    view = _until(
        lambda: (v := driver.ticket(ticket.ticket_id)).status is not R16TicketStatus.PENDING
        and v
    )
    assert view.reason is R16OwnerReason.GOAL_ALREADY_EXISTS


def test_a_stopped_runtime_stops_the_driver(fakes) -> None:  # type: ignore[no-untyped-def]
    from suzka.runtime.agent_runtime import AgentRuntimeStopped

    driver, runtime, source, goal, *_rest = fakes
    driver.notify_chat_terminal(_turn(5, X1), "hello")
    _until(lambda: len(source.calls) == 1)
    ticket = driver.notify_chat_terminal(
        _turn(8, "experience-00000000-0000-4000-8000-0000000000a7"),
        f"REVIEW_EXPERIENCE:{X1}",
    )
    time.sleep(0.3)
    source.futures[0].set_exception(AgentRuntimeStopped(_event(6)))
    _until(lambda: not driver.active)
    view = driver.ticket(ticket.ticket_id)
    assert (view.status, view.reason) == (
        R16TicketStatus.UNAVAILABLE,
        R16DriverReason.RUNTIME_UNAVAILABLE,
    )
    # After a stop, no new ticket can be issued and left pending.
    assert driver.notify_chat_terminal(_turn(9, X1), f"REVIEW_EXPERIENCE:{X1}") is None


def test_callbacks_only_append_without_io(fakes, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import suzka.r16.driver as driver_module

    driver, *_rest = fakes
    callback = driver._callback(driver_module._Stage.PLAN_PREVIEW, "goal-1")
    calls: list[str] = []
    monkeypatch.setattr(
        driver_module.LOGGER, "warning", lambda *a, **k: calls.append("log")
    )
    monkeypatch.setattr(driver, "_post", lambda *_a: calls.append("post"))
    future: Future[Any] = Future()
    future.set_result(AgentEventOutcome(_event(3), None))
    callback(future)  # inline, as for an already finished future
    assert calls == []


def test_debug_chat_and_inactive_drivers_get_no_ticket(fakes) -> None:  # type: ignore[no-untyped-def]
    driver, *_rest = fakes
    debug = AgentEventOutcome(
        _event(4, AgentEventSource.API_CHAT_DEBUG), _turn(4, X1).value
    )
    assert driver.notify_chat_terminal(debug, f"REVIEW_EXPERIENCE:{X1}") is None
    driver._active = False
    assert driver.notify_chat_terminal(_turn(5, X1), f"REVIEW_EXPERIENCE:{X1}") is None


def test_stop_marks_pending_tickets_unavailable(fakes) -> None:  # type: ignore[no-untyped-def]
    driver, runtime, *_rest = fakes
    runtime.pending = 5
    driver.notify_chat_terminal(_turn(5, X1), "hello")
    ticket = driver.notify_chat_terminal(
        _turn(6, "experience-00000000-0000-4000-8000-0000000000a5"),
        f"REVIEW_EXPERIENCE:{X1}",
    )
    time.sleep(0.3)
    driver.stop()
    view = driver.ticket(ticket.ticket_id)
    assert view.status is R16TicketStatus.UNAVAILABLE
