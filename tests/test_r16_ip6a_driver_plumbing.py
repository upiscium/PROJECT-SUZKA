"""IP-6a (#383 / #370 D0-D2): plumbing for the R16 driver.

``pending_count``, ``execute_with_event``, the CHAT outcome facts
(``experience_id``, ``eligible_for_u2a``, ``context_id``), and the unique
operator activation ``R16_DRIVER_ACTIVATION``, which COMPACTION carries
once it is archived (C9, Human decision 2026-11-13).
"""

from __future__ import annotations

from threading import Event

from fastapi.testclient import TestClient
import pytest

from suzka.api.runtime_execution import execute_with_event
from suzka.memory.experience_participant import experience_id_for_event
from suzka.r16.activation import driver_activation
from suzka.runtime.agent_runtime import AgentEventSource, AgentEventType
from suzka.runtime.event_journal import EventLifecycle

from test_r16_ip4c_past_retention import _archive, app  # noqa: F401
from test_r16_u2a_source_owner import _app

TOKEN = "ip6a-admin-token"
ADMIN = {"X-SUZKA-Admin-Token": TOKEN}


@pytest.fixture(autouse=True)
def _admin_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUZKA_P1B_TEST_TOKEN", TOKEN)


def _chat(client: TestClient, text: str) -> dict:  # type: ignore[type-arg]
    response = client.post("/api/chat", json={"message": text, "attachments": []})
    assert response.status_code == 200, response.text
    return response.json()


def test_pending_count_is_read_only_and_counts_waiting_events(app) -> None:  # noqa: F811
    with TestClient(app) as client:
        runtime = client.app.state.agent_runtime
        assert runtime.pending_count() == 0
        release = Event()
        started = Event()

        def blocking() -> str:
            started.set()
            release.wait(timeout=30)
            return "done"

        first = runtime.submit(
            AgentEventType.R16_DRIVER_ACTIVATION,
            AgentEventSource.API_R16_DRIVER_ACTIVATE,
            blocking,
        )
        assert started.wait(timeout=30)
        second = runtime.submit(
            AgentEventType.R16_DRIVER_ACTIVATION,
            AgentEventSource.API_R16_DRIVER_ACTIVATE,
            lambda: "next",
        )
        assert runtime.pending_count() == 1
        assert runtime.pending_count() == 1
        release.set()
        assert first.result(timeout=30).value == "done"
        assert second.result(timeout=30).value == "next"
        assert runtime.pending_count() == 0


def test_execute_with_event_returns_the_admitted_event(app) -> None:  # noqa: F811
    with TestClient(app) as client:
        state = client.app.state
        outcome = execute_with_event(
            state.agent_runtime,
            AgentEventType.CHAT,
            AgentEventSource.API_CHAT,
            lambda: state.main_loop.chat("hello through execute_with_event"),
        )
        assert outcome.event.event_type is AgentEventType.CHAT
        assert type(outcome.event.processing_sequence) is int
        assert outcome.value.experience_id == experience_id_for_event(
            outcome.event.event_id, outcome.event.processing_sequence
        )


def test_chat_outcome_facts_match_the_committed_experience(app) -> None:  # noqa: F811
    with TestClient(app) as client:
        state = client.app.state
        for text in ("a neutral line", "something wonderful and surprising!"):
            outcome = execute_with_event(
                state.agent_runtime,
                AgentEventType.CHAT,
                AgentEventSource.API_CHAT,
                lambda text=text: state.main_loop.chat(text),
            )
            result = outcome.value
            record = state.experience_store.load_committed_current(
                result.experience_id
            ).record
            appraisal = record.appraisal
            neutral = (
                not appraisal.novelty_valid
                and appraisal.goal_progress is None
                and appraisal.threat is None
            )
            # The same pre-selection condition U2a re-checks (#370 D1 D).
            assert result.eligible_for_u2a is (not neutral)
            assert result.context_id == record.context_id
        response = _chat(client, "visible through the API")
        assert response["experience_id"].startswith("experience-")


def test_activation_is_admin_only_unique_and_durable(app) -> None:  # noqa: F811
    settings = app.state.settings
    with TestClient(app) as client:
        state = client.app.state
        assert driver_activation(state.event_journal.inspect()) is None
        assert client.post("/api/r16/driver/activate").status_code in {401, 403}
        first = client.post("/api/r16/driver/activate", headers=ADMIN)
        assert first.status_code == 200, first.text
        body = first.json()
        assert body["active"] is True
        record = driver_activation(state.event_journal.inspect())
        assert record is not None
        assert record.lifecycle is EventLifecycle.COMPLETED
        assert record.processing_sequence == body["activation_sequence"]
        second = client.post("/api/r16/driver/activate", headers=ADMIN)
        assert second.status_code == 409
        assert driver_activation(state.event_journal.inspect()) == record
        assert state.agent_runtime.status.value == "accepting"

    with TestClient(_app(settings)) as client:
        assert driver_activation(client.app.state.event_journal.inspect()) == record
        assert (
            client.post("/api/r16/driver/activate", headers=ADMIN).status_code == 409
        )


def test_activation_survives_compaction_as_a_carried_record(app) -> None:  # noqa: F811
    """C9: once the activation is archived, COMPACTION carries it."""
    settings = app.state.settings
    with TestClient(app) as client:
        state = client.app.state
        assert client.post("/api/r16/driver/activate", headers=ADMIN).status_code == 200
        record = driver_activation(state.event_journal.inspect())
        assert record is not None
        from suzka.r16.source_owner import _MotivationSourceRef

        _archive(
            state,
            _MotivationSourceRef(record.event_id, record.processing_sequence, "episode-x"),
        )
        inspection = state.event_journal.inspect()
        assert not any(row.event_id == record.event_id for row in inspection.records)
        assert record in inspection.carried_records
        assert driver_activation(inspection) == record
        # A later compaction keeps carrying it from the one in effect.
        for number in range(12):
            _chat(client, f"after activation {number}")
        later = state.event_journal.inspect()
        assert later.latest_compaction is not None
        assert driver_activation(later) == record
        assert client.post("/api/r16/driver/activate", headers=ADMIN).status_code == 409

    with TestClient(_app(settings)) as client:
        assert driver_activation(client.app.state.event_journal.inspect()) == record


def test_a_compaction_without_activation_keeps_its_bytes(app) -> None:  # noqa: F811
    with TestClient(app) as client:
        state = client.app.state
        outcome = execute_with_event(
            state.agent_runtime,
            AgentEventType.CHAT,
            AgentEventSource.API_CHAT,
            lambda: state.main_loop.chat("an old turn"),
        )
        from suzka.r16.source_owner import _MotivationSourceRef

        _archive(
            state,
            _MotivationSourceRef(
                outcome.event.event_id, outcome.event.processing_sequence, "episode-x"
            ),
        )
        compaction = state.event_journal.inspect().latest_compaction
        assert compaction is not None
        assert compaction.driver_activation_record is None
        assert "driver_activation_record" not in compaction.model_dump(mode="json")

