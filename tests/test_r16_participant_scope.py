"""A Context that names a participant never reaches U2a (eve finding, #383).

U2a admits no event-time participant identity yet, so before this fix every
turn sent with ``interlocutor_key`` failed its MOTIVATION_SOURCE_REVIEW event
("DB1/Context source ambiguous") and commands on it HOLD as an opaque
``source_unavailable``. The driver now skips U2a there and holds commands with
``participant_scope_unsupported``.
"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from suzka.runtime.agent_runtime import AgentEventType
from suzka.runtime.event_journal import EventLifecycle

from test_r16_ip4c_past_retention import app  # noqa: F401
from test_r16_ip6b_driver import _activate, _admin_token, _chat, _interest, _settled  # noqa: F401


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
