"""A Context that names a participant never reaches U2a (eve finding, #383).

U2a admits no event-time participant identity yet, so before this fix every
turn sent with ``interlocutor_key`` failed its MOTIVATION_SOURCE_REVIEW event
("DB1/Context source ambiguous") and commands on it HOLD as an opaque
``source_unavailable``. The driver now skips U2a there and holds commands with
``participant_scope_unsupported``.
"""

from __future__ import annotations

import time
from dataclasses import replace

from fastapi.testclient import TestClient

from suzka.r16.driver import IDLE_WAIT_LIMIT_SECONDS
from suzka.runtime.agent_runtime import AgentEventType
from suzka.runtime.event_journal import EventLifecycle

from test_r16_ip4c_past_retention import app  # noqa: F401
from test_r16_ip6b_driver import (  # noqa: F401
    _activate,
    _admin_token,
    _chat,
    _interest,
    _settled,
    _turn,
    _until,
    fakes,
)


def _u2a_rows(state, lifecycle: EventLifecycle) -> int:  # type: ignore[no-untyped-def]
    return sum(
        1
        for row in state.event_journal.records
        if row.event_type is AgentEventType.MOTIVATION_SOURCE_REVIEW
        and row.lifecycle is lifecycle
    )


def test_participant_context_skips_u2a_and_holds_commands(app) -> None:  # noqa: F811
    with TestClient(app) as client:
        state = client.app.state
        _activate(client)
        selectors = {"client_session_id": "with-participant", "interlocutor_key": "alice"}
        x = _chat(client, "something wonderful happened today", **selectors)
        assert x["experience_id"] is not None
        for text in (
            f"REVIEW_EXPERIENCE:{x['experience_id']}",
            f"COMMIT_REVIEW_EXPERIENCE:1:{x['experience_id']}",
        ):
            command = _chat(client, text, **selectors)
            assert _settled(client, command["command_ticket"]["ticket_id"]) == {
                "status": "hold",
                "reason": "participant_scope_unsupported",
            }
        time.sleep(0.5)
        driver = state.r16_driver
        assert driver.counters["skipped_participant_scope"] == 1
        assert driver.counters["submitted"] == 0
        assert _u2a_rows(state, EventLifecycle.ACCEPTED) == 0
        assert _u2a_rows(state, EventLifecycle.FAILED) == 0
        assert state.main_loop.export_motivation_state().records == ()


def test_participant_bound_after_the_target_holds_the_command(app) -> None:  # noqa: F811
    # X is observed before the Context names anyone; the command turn binds a
    # participant to the same Context, which U2a would now refuse.
    with TestClient(app) as client:
        state = client.app.state
        _activate(client)
        x = _chat(client, "something wonderful happened today", client_session_id="late")
        _interest(client, x["experience_id"])
        command = _chat(
            client,
            f"REVIEW_EXPERIENCE:{x['experience_id']}",
            client_session_id="late",
            interlocutor_key="bob",
        )
        assert _settled(client, command["command_ticket"]["ticket_id"]) == {
            "status": "hold",
            "reason": "participant_scope_unsupported",
        }
        assert state.main_loop.export_goal_state().records == ()


def test_anonymous_context_is_unchanged(app) -> None:  # noqa: F811
    with TestClient(app) as client:
        state = client.app.state
        _activate(client)
        x = _chat(client, "something wonderful happened today", client_session_id="anon")
        _interest(client, x["experience_id"])
        command = _chat(
            client, f"REVIEW_EXPERIENCE:{x['experience_id']}", client_session_id="anon"
        )
        assert _settled(client, command["command_ticket"]["ticket_id"])["status"] == (
            "adopted"
        )
        assert state.r16_driver.counters["skipped_participant_scope"] == 0
        assert _u2a_rows(state, EventLifecycle.FAILED) == 0


NONEXISTENT = "experience-00000000-0000-4000-8000-000000000001"


def test_scoped_target_in_another_context_matches_a_nonexistent_one(app) -> None:  # noqa: F811
    # Review P2-2: the scope HOLD must stay behind the same-Context check, or
    # it would tell another Context that X exists (#370 D2 P2-2).
    with TestClient(app) as client:
        _activate(client)
        x = _chat(
            client,
            "something wonderful happened today",
            client_session_id="a",
            interlocutor_key="alice",
        )
        for selectors in (
            {"client_session_id": "b"},
            {"client_session_id": "c", "interlocutor_key": "carol"},
        ):
            answers = [
                _settled(
                    client,
                    _chat(client, f"REVIEW_EXPERIENCE:{target}", **selectors)[
                        "command_ticket"
                    ]["ticket_id"],
                )
                for target in (x["experience_id"], NONEXISTENT)
            ]
            assert answers == [
                {"status": "unavailable", "reason": "source_not_observed"},
            ] * 2


# --- the driver alone (review P2-1) ----------------------------------------------

X1 = "experience-00000000-0000-4000-8000-0000000000a1"
X2 = "experience-00000000-0000-4000-8000-0000000000a2"


def _scoped_turn(sequence: int, experience_id: str, *, context: str = "context-a"):  # type: ignore[no-untyped-def]
    outcome = _turn(sequence, experience_id, eligible=False, context=context)
    return replace(outcome, value=replace(outcome.value, participant_scoped=True))


def test_binding_a_participant_drops_a_waiting_u2a_in_that_context(fakes) -> None:  # type: ignore[no-untyped-def]  # noqa: F811
    driver, runtime, source, goal, *_rest, clock = fakes
    runtime.pending = 2  # busy: U2a for X1 waits
    driver.notify_chat_terminal(_turn(5, X1), "hello")
    other = "experience-00000000-0000-4000-8000-0000000000b1"
    driver.notify_chat_terminal(_turn(6, other, context="context-b"), "hello")
    # The next turn in context-a binds a participant.
    driver.notify_chat_terminal(_scoped_turn(7, X2), "hello")
    ticket = driver.notify_chat_terminal(
        _scoped_turn(8, "experience-00000000-0000-4000-8000-0000000000a3"),
        f"REVIEW_EXPERIENCE:{X1}",
    )
    assert ticket is not None

    def settled():  # type: ignore[no-untyped-def]
        view = driver.ticket(ticket.ticket_id)
        return None if view.status.value == "pending" else view

    view = _until(settled)
    assert (view.status.value, view.reason.value) == (
        "hold",
        "participant_scope_unsupported",
    )
    clock.now += IDLE_WAIT_LIMIT_SECONDS
    # Only context-b's U2a is ever submitted.
    _until(lambda: len(source.calls) == 1)
    time.sleep(0.3)
    assert len(source.calls) == 1
    assert source.calls[0][1].event_sequence == 6
    assert goal.calls == []
    # X1 (dropped while waiting) and X2 (scoped itself); a command turn is
    # never a U2a candidate.
    assert driver.counters["skipped_participant_scope"] == 2
