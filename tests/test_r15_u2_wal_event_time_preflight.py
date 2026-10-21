"""R15 U2 test-only: WAL-retained event-time opaque participant provenance.

This is NOT a production Relationship admission, a person identity proof,
or a new Source issuer. All experiments use isolated disposable R07/R12/DB1/WAL.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from suzka.api.server import create_app
from suzka.learning import AdapterRegistry
from suzka.memory import DualMemorySystem
from suzka.memory.experience_participant import experience_id_for_event
from suzka.runtime.agent_runtime import AgentEventSource, AgentEventType
from suzka.runtime.event_journal import EventLifecycle
from suzka.runtime.state_wal import TransitionRecord

from test_r15_p1_owner_bridge import SequencedProvider, scenario


@dataclass(frozen=True)
class _ObservedEventTime:
    """Historical technical proof only, not an identity or Relationship record."""

    participant_refs: tuple[str, ...]
    context_id: str
    context_revision: int
    wal_record_hash: str
    person_verified: bool = False
    relationship_admitted: bool = False


def _chat(client: TestClient, session: str, participant: str | None = None) -> str:
    request = {
        "message": "一回の合成会話。本人確認・信頼・関係性は判定しない。",
        "attachments": [],
        "client_session_id": session,
    }
    if participant is not None:
        request["interlocutor_key"] = participant
    response = client.post("/api/chat", json=request)
    assert response.status_code == 200, response.text[:200]
    return response.json()["episode_id"]


def _event_time_from_existing_wal(
    client: TestClient, episode_id: str
) -> _ObservedEventTime:
    """Test fixture: rejoin an actual terminal R07/R12/DB1 event to v10 WAL.

    Caller-provided episode_id has NO authority: the committed owners must
    independently supply one exact matching terminal event and Context.
    """
    owners = client.app.state
    journal = owners.event_journal.inspect()
    assert not journal.external_reconciliation_required
    assert not journal.open_transactions
    assert not journal.reconciliation_required_transactions
    candidates = []
    for row in journal.records:
        if (
            row.lifecycle is not EventLifecycle.COMPLETED
            or row.event_type is not AgentEventType.CHAT
            or row.source is not AgentEventSource.API_CHAT
        ):
            continue
        entry = owners.experience_store.load_committed_current(
            experience_id_for_event(row.event_id, row.processing_sequence)
        )
        if entry.record.source_episode_id == episode_id:
            candidates.append((row, entry.record))
    assert len(candidates) == 1, "episode is not uniquely committed"
    event, experience = candidates[0]
    assert experience.source_event_id == event.event_id
    assert experience.source_event_sequence == event.processing_sequence
    matched = owners.memory_system.get_committed_episodic(episode_id)
    assert matched is not None
    assert matched.record.context_id == experience.context_id
    assert matched.record.source_channel == "chat"
    assert matched.record.source_session_id is not None
    assert (
        sum(
            row.lifecycle is EventLifecycle.TRANSACTION_COMPLETED
            and row.event_id == event.event_id
            and row.processing_sequence == event.processing_sequence
            for row in journal.records
        )
        == 1
    )

    inspection = owners.state_wal.inspect()
    assert inspection.exists
    assert inspection.active_manifest is not None
    assert not inspection.active_manifest.external_reconciliation_required
    retained = [
        row
        for row in inspection.records
        if isinstance(row, TransitionRecord)
        and str(row.event_id) == event.event_id
        and row.processing_sequence == event.processing_sequence
        and row.event_type == "chat"
        and row.event_source == "api.chat"
    ]
    assert len(retained) == 1, "historical event-time WAL record not retained"
    row = retained[0]
    assert row.candidate_snapshot.schema_version == 10
    assert (
        row.candidate_snapshot.last_processed_event_sequence
        == event.processing_sequence
    )
    frames = tuple(
        frame
        for frame in row.candidate_snapshot.context_state.frames
        if frame.context_id == experience.context_id
    )
    assert len(frames) == 1
    frame = frames[0]
    assert frame.status == "active"
    assert frame.source_channel == matched.record.source_channel
    assert frame.source_session_id == matched.record.source_session_id
    return _ObservedEventTime(
        frame.participant_refs,
        frame.context_id,
        row.candidate_snapshot.context_state.revision,
        row.record_hash,
    )


def test_exact_wal_chat_keeps_one_opaque_ref_at_event_time(
    tmp_path: Path,
) -> None:
    app, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        episode = _chat(client, "u2-wal-positive", "opaque-ref-a")
        observation = _event_time_from_existing_wal(client, episode)
        assert observation.participant_refs == ("opaque-ref-a",)
        assert observation.context_revision > 0
        assert len(observation.wal_record_hash) == 64
        assert not observation.person_verified
        assert not observation.relationship_admitted
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0


def test_late_token_cannot_be_attributed_to_earlier_anonymous_event(
    tmp_path: Path,
) -> None:
    app, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        original = _chat(client, "u2-wal-late")
        before = _event_time_from_existing_wal(client, original)
        assert before.participant_refs == ()
        later = _chat(client, "u2-wal-late", "opaque-late")
        now = _event_time_from_existing_wal(client, later)
        preserved = _event_time_from_existing_wal(client, original)
        assert preserved == before
        assert now.participant_refs == ("opaque-late",)
        assert now.context_id == before.context_id
        assert now.context_revision > before.context_revision
        assert client.app.state.main_loop.context_registry.get(
            before.context_id
        ).participant_refs == ("opaque-late",)
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0


def test_restart_preserves_event_time_ref_without_r09_r12_schema_change(
    tmp_path: Path,
) -> None:
    app, *_ = scenario(tmp_path, no_reveal=True)
    settings = app.state.settings
    with TestClient(app) as client:
        episode = _chat(client, "u2-wal-restart", "opaque-restart-ref")
        observed = _event_time_from_existing_wal(client, episode)

    restarted = create_app(settings)
    restarted.state.model_provider = SequencedProvider(no_reveal=True)
    restarted.state.memory_system = DualMemorySystem(settings)
    restarted.state.adapter_registry = AdapterRegistry(settings)
    with TestClient(restarted) as client:
        recovered = _event_time_from_existing_wal(client, episode)
        assert recovered == observed
        assert not recovered.person_verified
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0


def test_wrong_episode_id_cannot_yield_positive_wal_attribution(
    tmp_path: Path,
) -> None:
    app, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        actual = _chat(client, "u2-wal-negative", "opaque-one")
        assert _event_time_from_existing_wal(client, actual).participant_refs == (
            "opaque-one",
        )
        with pytest.raises(AssertionError, match="not uniquely committed"):
            _event_time_from_existing_wal(client, "fabricated-episode")
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0
