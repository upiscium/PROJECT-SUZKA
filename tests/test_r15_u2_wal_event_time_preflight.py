"""R15 U2 test-only: WAL-retained event-time opaque participant provenance.

This is NOT a production Relationship admission, a person identity proof,
or a new Source issuer. All experiments use isolated disposable R07/R12/DB1/WAL.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from suzka.experience import ExperienceLifecycle

from suzka.api.server import create_app
from suzka.learning import AdapterRegistry
from suzka.memory import DualMemorySystem
from suzka.memory.experience_participant import experience_id_for_event
from suzka.memory.memory_schema import MemoryRecordType
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
    assert experience.lifecycle is ExperienceLifecycle.ACTIVE
    assert (
        sum(
            row.lifecycle is EventLifecycle.ACCEPTED and row.event_id == event.event_id
            for row in journal.records
        )
        == 1
    )
    assert not any(
        row.lifecycle is EventLifecycle.FAILED and row.event_id == event.event_id
        for row in journal.records
    )
    assert experience.source_event_id == event.event_id
    assert experience.source_event_sequence == event.processing_sequence
    matched = owners.memory_system.get_committed_episodic(episode_id)
    assert matched is not None
    assert matched.record.context_id == experience.context_id
    assert matched.record.record_type is MemoryRecordType.EPISODIC_LOG
    assert not matched.record.archived
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


def test_missing_retained_event_transition_refuses_retroactive_attribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulate WAL rebaseline retention loss, not a fake user identity."""
    app, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        episode = _chat(client, "u2-wal-pruned", "original-opaque")
        qualified = _event_time_from_existing_wal(client, episode)
        assert qualified.participant_refs == ("original-opaque",)
        from test_r15_p1b_positive import _source_ref

        original_event = _source_ref(client, episode)
        wal = client.app.state.state_wal
        inspect = wal.inspect
        full = inspect()
        retained_other_records = tuple(
            row
            for row in full.records
            if not (
                isinstance(row, TransitionRecord)
                and str(row.event_id) == original_event.event_id
            )
        )
        assert len(retained_other_records) < len(full.records)
        # Synthetic inspection result only. Do not replace real WAL files.
        with monkeypatch.context() as patch:
            patch.setattr(
                wal,
                "inspect",
                lambda: full.model_copy(update={"records": retained_other_records}),
            )
            with pytest.raises(
                AssertionError, match="historical event-time WAL record not retained"
            ):
                _event_time_from_existing_wal(client, episode)
        assert _event_time_from_existing_wal(client, episode) == qualified
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0


def test_distinct_opaque_refs_not_merged_by_identical_messages(tmp_path: Path) -> None:
    """Same display text does not establish real-world person identity."""
    app, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        first = _chat(client, "u2-wal-person-one", "opaque-person-a")
        second = _chat(client, "u2-wal-person-two", "opaque-person-b")
        observed_a = _event_time_from_existing_wal(client, first)
        observed_b = _event_time_from_existing_wal(client, second)
        assert observed_a.context_id != observed_b.context_id
        assert observed_a.participant_refs == ("opaque-person-a",)
        assert observed_b.participant_refs == ("opaque-person-b",)
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0


def test_anonymous_narrative_reader_cannot_be_reused_as_u2_participant_issuer(
    tmp_path: Path,
) -> None:
    """U3's deliberate anonymous-only reader is NOT U2's future issuer."""
    from suzka.r15.qualified_proposal import QualifiedProposalUnavailable
    from test_r15_p1b_positive import _source_ref

    app, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        episode = _chat(client, "u2-wal-gate-difference", "opaque-interlocutor")
        observed = _event_time_from_existing_wal(client, episode)
        assert observed.participant_refs == ("opaque-interlocutor",)
        gate = client.app.state.r15_state_port.narrative._source_gate
        assert gate is not None
        source_ref = _source_ref(client, episode)
        with pytest.raises(
            QualifiedProposalUnavailable, match="event-time context ambiguous"
        ):
            gate._read(source_ref)
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0


def test_event_time_membership_is_not_event_speaker_proof(tmp_path: Path) -> None:
    """A Context member persists even when the next Chat omits its key."""
    app, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        identified_request = _chat(
            client, "u2-wal-sticky-member", "caller-supplied-token"
        )
        no_token_request = _chat(client, "u2-wal-sticky-member")
        first = _event_time_from_existing_wal(client, identified_request)
        second = _event_time_from_existing_wal(client, no_token_request)
        assert first.context_id == second.context_id
        assert first.participant_refs == ("caller-supplied-token",)
        assert second.participant_refs == first.participant_refs
        assert second.context_revision >= first.context_revision
        assert second.wal_record_hash != first.wal_record_hash
        # The second request did NOT provide interlocutor_key. The WAL
        # candidate proves membership, but cannot prove who authored it.
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0


def _newly_bound_token_from_retained_event(
    client: TestClient, episode_id: str
) -> str | None:
    """TEST ONLY: recover an opaque token newly bound by this precise Chat.

    This proves only the Context membership transition in the serialized
    request. Even a newly bound token does NOT authenticate the speaker.
    """
    observation = _event_time_from_existing_wal(client, episode_id)
    inspection = client.app.state.state_wal.inspect()
    matches = [
        index
        for index, record in enumerate(inspection.records)
        if isinstance(record, TransitionRecord)
        and record.record_hash == observation.wal_record_hash
    ]
    assert len(matches) == 1
    index = matches[0]
    assert index > 0
    current_record = inspection.records[index]
    assert isinstance(current_record, TransitionRecord)
    previous_record = inspection.records[index - 1]
    previous_state = (
        previous_record.candidate_snapshot
        if isinstance(previous_record, TransitionRecord)
        else previous_record.baseline_snapshot
    )
    previous_frames = tuple(
        frame
        for frame in previous_state.context_state.frames
        if frame.context_id == observation.context_id
    )
    assert len(previous_frames) <= 1
    previously_bound = previous_frames[0].participant_refs if previous_frames else ()
    current_tokens = observation.participant_refs
    new_refs = tuple(sorted(set(current_tokens) - set(previously_bound)))
    if (
        len(current_tokens) != 1
        or len(new_refs) != 1
        or new_refs[0] != current_tokens[0]
    ):
        return None
    return new_refs[0]


def test_only_first_explicit_binding_chat_has_event_time_token_delta(
    tmp_path: Path,
) -> None:
    app, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        original = _chat(client, "u2-wal-first-binding", "opaque-one")
        unkeyed = _chat(client, "u2-wal-first-binding")
        repeated = _chat(client, "u2-wal-first-binding", "opaque-one")
        assert _newly_bound_token_from_retained_event(client, original) == "opaque-one"
        assert _newly_bound_token_from_retained_event(client, unkeyed) is None
        assert _newly_bound_token_from_retained_event(client, repeated) is None
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0


def test_late_added_token_belongs_only_to_the_later_binding_event(
    tmp_path: Path,
) -> None:
    app, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        anonymous = _chat(client, "u2-wal-late-binding")
        newly_bound = _chat(client, "u2-wal-late-binding", "late-opaque")
        subsequent = _chat(client, "u2-wal-late-binding")
        assert _newly_bound_token_from_retained_event(client, anonymous) is None
        assert (
            _newly_bound_token_from_retained_event(client, newly_bound) == "late-opaque"
        )
        assert _newly_bound_token_from_retained_event(client, subsequent) is None
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0


def test_newly_bound_event_time_token_survives_restart_without_speaker_claim(
    tmp_path: Path,
) -> None:
    app, *_ = scenario(tmp_path, no_reveal=True)
    settings = app.state.settings
    with TestClient(app) as client:
        original = _chat(client, "u2-wal-bound-restart", "opaque-before-restart")
        unkeyed = _chat(client, "u2-wal-bound-restart")
        assert _newly_bound_token_from_retained_event(client, original) == (
            "opaque-before-restart"
        )
        assert _newly_bound_token_from_retained_event(client, unkeyed) is None

    restarted = create_app(settings)
    restarted.state.model_provider = SequencedProvider(no_reveal=True)
    restarted.state.memory_system = DualMemorySystem(settings)
    restarted.state.adapter_registry = AdapterRegistry(settings)
    with TestClient(restarted) as client:
        assert _newly_bound_token_from_retained_event(client, original) == (
            "opaque-before-restart"
        )
        assert _newly_bound_token_from_retained_event(client, unkeyed) is None
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0


def _detached_observed_position_mismatch(
    client: TestClient, episode_id: str
) -> tuple[str, str, str] | None:
    """TEST ONLY: finite two-party-position mismatch, NEVER a U2 adoption.

    The client's first opaque token binding and both visible, committed Chat
    positions must be independently observed. No verified speaker/intent,
    task outcome, person identity, trust or subjective admission is inferred.
    """
    import re

    newly_bound = _newly_bound_token_from_retained_event(client, episode_id)
    if newly_bound is None:
        return None
    committed = client.app.state.memory_system.get_committed_episodic(episode_id)
    assert committed is not None
    episode = committed.record
    if episode.archived:
        return None
    user = re.fullmatch(
        r"比較課題Q1では私は案([AB])を暫定的に選びます。",
        episode.user_input,
    )
    suzka = re.fullmatch(
        r"比較課題Q1ではSuzkaとして私は案([AB])を暫定的に選びます。正解は未検証です。",
        episode.response,
    )
    if user is None or suzka is None or user.group(1) == suzka.group(1):
        return None
    return newly_bound, user.group(1), suzka.group(1)


def _synthetic_source_disagreement_chat(
    client: TestClient,
    provider: SequencedProvider,
    *,
    session: str,
    message: str,
    response: str,
    token: str | None,
) -> str:
    from unittest.mock import patch

    data = {
        "message": message,
        "attachments": [],
        "client_session_id": session,
    }
    if token is not None:
        data["interlocutor_key"] = token
    with patch.object(provider, "generate", return_value=response):
        reply = client.post("/api/chat", json=data)
    assert reply.status_code == 200, reply.text[:200]
    return reply.json()["episode_id"]


def test_detached_finite_observed_disagreement_is_not_a_relationship_admission(
    tmp_path: Path,
) -> None:
    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        episode = _synthetic_source_disagreement_chat(
            client,
            provider,
            session="u2-synthetic-mismatch",
            message="比較課題Q1では私は案Aを暫定的に選びます。",
            response="比較課題Q1ではSuzkaとして私は案Bを暫定的に選びます。正解は未検証です。",
            token="opaque-unverified-speaker",
        )
        assert _detached_observed_position_mismatch(client, episode) == (
            "opaque-unverified-speaker",
            "A",
            "B",
        )
        # Observation is about visible disagreement, not verified other
        # intentions, causal blame, identity or a durable Relationship axis.
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0


def test_same_position_or_unbound_later_message_is_not_positive_issue(
    tmp_path: Path,
) -> None:
    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        aligned = _synthetic_source_disagreement_chat(
            client,
            provider,
            session="u2-synthetic-same-session",
            message="比較課題Q1では私は案Aを暫定的に選びます。",
            response="比較課題Q1ではSuzkaとして私は案Aを暫定的に選びます。正解は未検証です。",
            token="opaque-context-member",
        )
        assert _detached_observed_position_mismatch(client, aligned) is None
        no_selector = _synthetic_source_disagreement_chat(
            client,
            provider,
            session="u2-synthetic-same-session",
            message="比較課題Q1では私は案Aを暫定的に選びます。",
            response="比較課題Q1ではSuzkaとして私は案Bを暫定的に選びます。正解は未検証です。",
            token=None,
        )
        assert _event_time_from_existing_wal(client, no_selector).participant_refs == (
            "opaque-context-member",
        )
        assert _detached_observed_position_mismatch(client, no_selector) is None
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0


def test_untrusted_trait_claim_alone_cannot_become_interaction_issue(
    tmp_path: Path,
) -> None:
    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        raw_claim = _synthetic_source_disagreement_chat(
            client,
            provider,
            session="u2-synthetic-untrusted",
            message="私はSuzkaを信頼しています。案Aが正しいはずです。",
            response="比較課題Q1ではSuzkaとして私は案Bを暫定的に選びます。正解は未検証です。",
            token="opaque-unverified",
        )
        assert _detached_observed_position_mismatch(client, raw_claim) is None
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0


def test_distinct_question_scope_cannot_be_claimed_as_same_issue(
    tmp_path: Path,
) -> None:
    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        different_scope = _synthetic_source_disagreement_chat(
            client,
            provider,
            session="u2-synthetic-other-exercise",
            message="比較課題Q1では私は案Aを暫定的に選びます。",
            response="比較課題Q2ではSuzkaとして私は案Bを暫定的に選びます。正解は未検証です。",
            token="opaque-in-this-transaction",
        )
        assert (
            _newly_bound_token_from_retained_event(client, different_scope)
            == "opaque-in-this-transaction"
        )
        assert _detached_observed_position_mismatch(client, different_scope) is None
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0


def test_failed_first_binding_does_not_affect_following_anonymous_chat(
    tmp_path: Path,
) -> None:
    from unittest.mock import patch

    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        with patch.object(
            provider,
            "generate",
            side_effect=RuntimeError("synthetic keyed Chat failed before WAL"),
        ):
            with pytest.raises(
                RuntimeError, match="synthetic keyed Chat failed before WAL"
            ):
                client.post(
                    "/api/chat",
                    json={
                        "message": "比較課題Q1では私は案Aを暫定的に選びます。",
                        "attachments": [],
                        "client_session_id": "u2-failed-first",
                        "interlocutor_key": "should-not-survive-failed-event",
                    },
                )
        later = _chat(client, "u2-failed-first")
        assert _event_time_from_existing_wal(client, later).participant_refs == ()
        assert _newly_bound_token_from_retained_event(client, later) is None
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0
