"""R15 U2 TEST-ONLY event-time Context/participant matrix.

Real isolated Chat, R07/R12/DB1 and immutable in-event Context capture.
NO production U2 subject adoption, person proof, trust/trait setters or WAL
continuation for this synthetic observation.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from experiments.r15_p1.u2_event_time_preflight import (
    EventTimeCapture,
    EventTimeParticipantView,
    EventTimeStatus,
    assess_single_opaque_participant,
)
from suzka.memory.experience_participant import experience_id_for_event
import suzka.runtime.main_loop as main_loop_module
from suzka.runtime.context import ContextType

from test_r15_p1_owner_bridge import scenario


def _test_view(
    client: TestClient,
    captured: EventTimeCapture,
    episode_id: str,
) -> EventTimeParticipantView:
    assert len(captured.observations) == 1
    owner = client.app.state
    return assess_single_opaque_participant(
        captured.observations[0],
        episode_id,
        owner.event_journal,
        owner.experience_store,
        owner.memory_system,
        owner.main_loop.context_registry,
    )


def _instrument(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    runtime = client.app.state.agent_runtime
    captured = EventTimeCapture(
        runtime,
        main_loop_module.resolve_chat_context,
    )
    monkeypatch.setattr(main_loop_module, "resolve_chat_context", captured)
    return captured


def _chat(
    client: TestClient,
    *,
    key: str | None,
    session: str | None = None,
    context: str | None = None,
) -> str:
    request: dict[str, object] = {
        "message": "実装検証用の一回の合成会話。信頼や人物同一性は設定しない。",
        "attachments": [],
    }
    if key is not None:
        request["interlocutor_key"] = key
    if session is not None:
        request["client_session_id"] = session
    if context is not None:
        request["context_id"] = context
    response = client.post("/api/chat", json=request)
    assert response.status_code == 200, response.text[:200]
    return response.json()["episode_id"]


def test_actual_in_event_one_opaque_ref_observed_with_real_terminal_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _, _, _, _ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        with monkeypatch.context() as patch:
            capture = _instrument(client, patch)
            episode_id = _chat(client, key="opaque-reference-a", session="session-a")
        view = _test_view(client, capture, episode_id)
        assert view.status is EventTimeStatus.OPAQUE_OBSERVED
        assert view.participant_ref == "opaque-reference-a"
        assert view.event_context_revision is not None
        assert view.person_verified is False
        assert view.relationship_admitted is False
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0
        assert (
            client.app.state.agent_state_store.load().relationship_state["revision"]
            == 0
        )


def test_a_participant_added_after_e_cannot_retroactively_qualify_e(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _, _, _, _ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        with monkeypatch.context() as patch:
            capture = _instrument(client, patch)
            episode_id = _chat(client, key=None, session="initially-anonymous")
        assert capture.observations[0].frame.participant_refs == ()
        registry = client.app.state.main_loop.context_registry
        context_id = capture.observations[0].frame.context_id
        registry.add_participant_ref(context_id, "late-opaque-ref")
        assert registry.get(context_id).participant_refs == ("late-opaque-ref",)
        view = _test_view(client, capture, episode_id)
        assert view.status is EventTimeStatus.HOLD
        assert view.participant_ref is None
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0


def test_one_preexisting_ref_becomes_ambiguous_after_second_ref_is_added(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _, _, _, _ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        with monkeypatch.context() as patch:
            capture = _instrument(client, patch)
            episode_id = _chat(client, key="opaque-reference-a")
        assert (
            _test_view(client, capture, episode_id).status
            is EventTimeStatus.OPAQUE_OBSERVED
        )
        registry = client.app.state.main_loop.context_registry
        registry.add_participant_ref(
            capture.observations[0].frame.context_id, "opaque-reference-b"
        )
        assert _test_view(client, capture, episode_id).status is EventTimeStatus.HOLD


def test_multiple_participants_at_e_are_not_one_interlocutor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _, _, _, _ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        registry = client.app.state.main_loop.context_registry
        frame = registry.create(
            "multi-opaque-context",
            ContextType.CONVERSATION,
            "chat",
            participant_refs=("ref-a", "ref-b"),
        )
        with monkeypatch.context() as patch:
            capture = _instrument(client, patch)
            episode_id = _chat(client, key=None, context=frame.context_id)
        assert capture.observations[0].frame.participant_refs == ("ref-a", "ref-b")
        assert _test_view(client, capture, episode_id).status is EventTimeStatus.HOLD


def test_forged_event_ref_or_retired_experience_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _, _, _, _ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        with monkeypatch.context() as patch:
            capture = _instrument(client, patch)
            episode_id = _chat(client, key="opaque-reference-a")
        normal = capture.observations[0]
        assert (
            _test_view(client, capture, episode_id).status
            is EventTimeStatus.OPAQUE_OBSERVED
        )
        capture.observations[0] = replace(normal, event_id="forged-event")
        assert _test_view(client, capture, episode_id).status is EventTimeStatus.HOLD
        capture.observations[0] = normal

        experience = client.app.state.experience_store
        experience_id = experience_id_for_event(
            normal.event_id,
            normal.event_sequence,
        )
        record_file = experience.record_path(experience_id, 0)
        original = record_file.read_bytes()
        record_file.unlink()
        try:
            assert (
                _test_view(client, capture, episode_id).status is EventTimeStatus.HOLD
            )
        finally:
            record_file.write_bytes(original)
            record_file.chmod(0o600)
        assert (
            _test_view(client, capture, episode_id).status
            is EventTimeStatus.OPAQUE_OBSERVED
        )


def test_same_claimed_identity_does_not_merge_two_opaque_interlocutors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _, _, _, _ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        registry = client.app.state.main_loop.context_registry
        registry.upsert_interlocutor_binding(
            "opaque-ref-a",
            identity_key="same-claimed-person",
        )
        registry.upsert_interlocutor_binding(
            "opaque-ref-b",
            identity_key="same-claimed-person",
        )
        with monkeypatch.context() as patch:
            capture = _instrument(client, patch)
            episode_a = _chat(client, key="opaque-ref-a", session="session-a")
        first = _test_view(client, capture, episode_a)
        assert first.status is EventTimeStatus.OPAQUE_OBSERVED
        assert first.participant_ref == "opaque-ref-a"

        # A distinct Chat in another session cannot convert the first
        # reference to an independently verified same human identity.
        with monkeypatch.context() as patch:
            second_capture = _instrument(client, patch)
            episode_b = _chat(client, key="opaque-ref-b", session="session-b")
        second = _test_view(client, second_capture, episode_b)
        assert second.status is EventTimeStatus.OPAQUE_OBSERVED
        assert second.participant_ref == "opaque-ref-b"
        assert first.participant_ref != second.participant_ref
        assert not first.person_verified and not second.person_verified
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0


def test_test_only_frame_does_not_survive_as_durable_u2_source_witness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _, _, _, _ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        with monkeypatch.context() as patch:
            capture = _instrument(client, patch)
            episode_id = _chat(client, key="opaque-ref")
        assert (
            _test_view(client, capture, episode_id).status
            is EventTimeStatus.OPAQUE_OBSERVED
        )

    # The real source files contain the context ID, not this captured
    # test-only participant value or its event-time frame bytes.
    assert capture.observations and capture.observations[0].frame.participant_refs
    assert client.app.state.r15_state_port.relationship.snapshot().revision == 0
    assert not hasattr(client.app.state.experience_store, "get_event_time_participant")
