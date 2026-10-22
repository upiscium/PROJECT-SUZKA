"""R15 U2 MVI revised acceptance — strictly TEST-ONLY real owner feasibility.

This is a detached postcommit technical projection, NOT an authorized Actor
InteractionStance producer, current inner feeling, verified person, identity
issuer, or persistent Relationship writer. A production actor-bound *current*
event source and prompt/use gating require separate Human code approval.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from suzka.experience import ExperienceLifecycle
from suzka.memory.dual_memory_system import (
    EpisodicMemoryFormatError,
    EpisodicMemoryReadError,
)
from suzka.memory.experience_store import ExperienceStoreError
from suzka.memory.experience_participant import experience_id_for_event
from suzka.memory.memory_schema import MemoryRecordType
from suzka.r15.common import (
    ClaimedOrigin,
    Concept,
    EventRef,
    InterpretationStatus,
    SourceDisposition,
    SourceKind,
    SourceLifecycle,
    SourceWitness,
    source_disposition,
)
from suzka.r15.stance import (
    ActorCausation,
    FeltAssociation,
    InteractionStance,
    context_projection_checksum,
    emotion_projection_checksum,
)
from suzka.runtime.agent_runtime import AgentEventSource, AgentEventType
from suzka.runtime.event_journal import (
    EventJournalError,
    EventLifecycle,
    ParticipantOutcome,
)
from suzka.runtime.state_wal import BaselineRecord, StateWALError, TransitionRecord

from test_r15_p1_owner_bridge import scenario


@dataclass(frozen=True, slots=True)
class _ScopedUnknownObservation:
    """Context-qualified unverified association. Never a Relationship root."""

    event: EventRef
    context_id: str
    opaque_token: str | None
    wal_record_hash: str
    relationship_axes: tuple[tuple[str, InterpretationStatus], ...]


def _send(client: TestClient, session: str, token: str | None = None) -> str:
    request = {
        "message": "Context単位の匿名・一時的な関係認識の検証。",
        "attachments": [],
        "client_session_id": session,
    }
    if token is not None:
        request["interlocutor_key"] = token
    response = client.post("/api/chat", json=request)
    assert response.status_code == 200, response.text[:220]
    return response.json()["episode_id"]


def _latest_committed_technical_evidence(
    client: TestClient, episode_id: str
) -> tuple[_ScopedUnknownObservation, InteractionStance] | None:
    """Fail closed on retired/corrupt/unavailable technical Source.

    The output is a detached TEST-ONLY postcommit feasibility observation,
    NOT an internal Actor-time current-cognition or real speaker issuer.
    """
    try:
        return _scoped_evidence_from_real_committed_owners(client, episode_id)
    except (
        ExperienceStoreError,
        EventJournalError,
        StateWALError,
        EpisodicMemoryReadError,
        EpisodicMemoryFormatError,
        OSError,
    ):
        return None


def _scoped_evidence_from_real_committed_owners(
    client: TestClient, episode_id: str
) -> tuple[_ScopedUnknownObservation, InteractionStance] | None:
    """Join real committed R07/R12/DB1/R09/R10/WAL, technical only.

    The test constructs Witness values AFTER the Chat. Those hashes
    corroborate a technical snapshot and are NOT trusted root authentication.
    Production would require a current serialized Actor-time producer.
    """
    from suzka.r15.contracts import RelationshipAxisKind

    owner = client.app.state
    journal = owner.event_journal.inspect()
    if (
        journal.external_reconciliation_required
        or journal.open_transactions
        or journal.reconciliation_required_transactions
    ):
        return None
    candidates = []
    for row in journal.records:
        if (
            row.lifecycle is not EventLifecycle.COMPLETED
            or row.event_type is not AgentEventType.CHAT
            or row.source is not AgentEventSource.API_CHAT
            or row.event_id is None
            or row.processing_sequence is None
        ):
            continue
        experience = owner.experience_store.load_committed_current(
            experience_id_for_event(row.event_id, row.processing_sequence)
        ).record
        if experience.source_episode_id == episode_id:
            candidates.append((row, experience))
    if len(candidates) != 1:
        return None
    done, experience = candidates[0]
    if experience.lifecycle is not ExperienceLifecycle.ACTIVE:
        return None
    if (
        experience.source_event_id != done.event_id
        or experience.source_event_sequence != done.processing_sequence
    ):
        return None
    admitted = [
        row
        for row in journal.records
        if row.event_id == done.event_id and row.lifecycle is EventLifecycle.ACCEPTED
    ]
    terminal = [
        row
        for row in journal.records
        if row.event_id == done.event_id
        and row.processing_sequence == done.processing_sequence
        and row.lifecycle is EventLifecycle.TRANSACTION_COMPLETED
    ]
    prepared = [
        row
        for row in journal.records
        if row.event_id == done.event_id
        and row.processing_sequence == done.processing_sequence
        and row.lifecycle is EventLifecycle.TRANSACTION_PREPARED
    ]
    if len(admitted) != 1 or len(terminal) != 1 or len(prepared) != 1:
        return None
    tx_id = prepared[0].transaction_id
    if (
        tx_id is None
        or terminal[0].transaction_id != tx_id
        or done.transaction_id is not None  # COMPLETED is WAL-linked instead
        or not (admitted[0].timestamp <= terminal[0].timestamp <= done.timestamp)
        or experience.created_at > terminal[0].timestamp
    ):
        return None
    for required in ("memory.episodic", "memory.experience", "session.turn"):
        qualified = [
            row
            for row in journal.records
            if row.event_id == done.event_id
            and row.processing_sequence == done.processing_sequence
            and row.lifecycle is EventLifecycle.PARTICIPANT_FINALIZED
            and row.transaction_id == tx_id
            and row.participant_id == required
            and row.participant_outcome is ParticipantOutcome.FINALIZED
        ]
        if len(qualified) != 1:
            return None
    # EventRef.occurred_at here denotes the journal ACCEPTED timestamp,
    # NOT exact model-generation or authenticated subjective event time.
    event = EventRef(done.event_id, done.processing_sequence, admitted[0].timestamp)
    db1 = owner.memory_system.get_committed_episodic(episode_id)
    if (
        db1 is None
        or db1.record.archived
        or db1.record.record_type is not MemoryRecordType.EPISODIC_LOG
        or db1.record.context_id != experience.context_id
        or db1.record.source_channel != "chat"
    ):
        return None

    wal = owner.state_wal.inspect()
    if (
        not wal.exists
        or wal.active_manifest is None
        or wal.active_manifest.external_reconciliation_required
    ):
        return None
    matches = [
        (index, row)
        for index, row in enumerate(wal.records)
        if isinstance(row, TransitionRecord)
        and str(row.event_id) == done.event_id
        and row.processing_sequence == done.processing_sequence
        and row.event_type == "chat"
        and row.event_source == "api.chat"
        and row.generation_id == wal.active_manifest.active_generation_id
    ]
    if len(matches) != 1:
        return None
    index, transition = matches[0]
    if (
        done.wal_generation_id != str(transition.generation_id)
        or done.wal_record_id != str(transition.record_id)
        or done.wal_record_hash != transition.record_hash
        or done.snapshot_hash != transition.candidate_snapshot_hash
        or not (admitted[0].timestamp <= transition.created_at <= terminal[0].timestamp)
    ):
        return None
    snapshot = transition.candidate_snapshot
    if (
        snapshot.schema_version != 10
        or snapshot.last_processed_event_sequence != done.processing_sequence
        or index < 1
    ):
        return None
    context = snapshot.context_state
    frames = tuple(
        frame for frame in context.frames if frame.context_id == experience.context_id
    )
    if len(frames) != 1:
        return None
    frame = frames[0]
    if (
        frame.status != "active"
        or frame.source_session_id != db1.record.source_session_id
        or frame.source_channel != db1.record.source_channel
    ):
        return None

    # A later event or an unrelated selected Context means this old projection
    # is no longer the current serialized event. No historical reattribution.
    actual_state = owner.agent_state_store.load()
    live = owner.main_loop.context_registry.state
    if (
        actual_state.last_processed_event_sequence != done.processing_sequence
        or actual_state.context_state != context
        or actual_state.emotion_state != snapshot.emotion_state
        or live.current_context_id != frame.context_id
        or owner.main_loop.emotion_engine.state.valence
        != snapshot.emotion_state.valence
        or owner.main_loop.emotion_engine.state.arousal
        != snapshot.emotion_state.arousal
        or owner.main_loop.emotion_engine.state.optimal_loss
        != snapshot.emotion_state.optimal_loss
    ):
        return None
    live_frames = tuple(f for f in live.frames if f.context_id == frame.context_id)
    if len(live_frames) != 1:
        return None
    live_frame = live_frames[0]
    if (
        live_frame.participant_refs != frame.participant_refs
        or live_frame.source_session_id != frame.source_session_id
        or live_frame.status.value != frame.status
    ):
        return None

    # Comparison with the immediately preceding retained snapshot proves a
    # new *Context association*, not an authenticated author/speaker.
    prior = wal.records[index - 1]
    if not isinstance(prior, (BaselineRecord, TransitionRecord)):
        return None
    old = (
        prior.candidate_snapshot
        if isinstance(prior, TransitionRecord)
        else prior.baseline_snapshot
    )
    previous = tuple(
        f for f in old.context_state.frames if f.context_id == frame.context_id
    )
    if len(previous) > 1:
        return None
    previous_tokens = previous[0].participant_refs if previous else ()
    newly_bound = tuple(sorted(set(frame.participant_refs) - set(previous_tokens)))
    token = (
        newly_bound[0] if len(frame.participant_refs) == len(newly_bound) == 1 else None
    )

    # Every INHERITED as well as newly bound member can collide across
    # Contexts, even if this event has no speaker selector. Refuse the entire
    # tagged Context projection rather than accidentally carrying a pooled
    # raw token. A negative-only local registry guard is not person proof.
    for member in frame.participant_refs:
        appearances = tuple(f for f in live.frames if member in f.participant_refs)
        if len(appearances) != 1 or appearances[0].context_id != frame.context_id:
            return None

    emotion = snapshot.emotion_state
    valence = emotion.valence.hex()
    arousal = emotion.arousal.hex()
    optimal = emotion.optimal_loss.hex()
    context_witness = SourceWitness(
        SourceKind.CONTEXT,
        frame.context_id,
        context.revision,
        context_projection_checksum(
            event, frame.context_id, context.revision, frame.participant_refs
        ),
        transition.record_hash,
        SourceLifecycle.ACTIVE,
        ClaimedOrigin.DECLARATIVE,
        event,
    )
    emotion_witness = SourceWitness(
        SourceKind.EMOTION,
        "emotion.global",
        done.processing_sequence,
        emotion_projection_checksum(event, valence, arousal, optimal),
        transition.candidate_snapshot_hash,
        SourceLifecycle.ACTIVE,
        ClaimedOrigin.DECLARATIVE,
        event,
    )
    # The technical witness origins/record hashes are test assertions about
    # a WAL snapshot, never an authenticated R09/R10 Subject issuer.
    assert (
        source_disposition(Concept.INTERACTION_ASSOCIATION, context_witness)
        is SourceDisposition.REQUIRES_TRUSTED_ROOT
    )
    assert (
        source_disposition(Concept.INTERACTION_ASSOCIATION, emotion_witness)
        is SourceDisposition.REQUIRES_TRUSTED_ROOT
    )

    stance = InteractionStance(
        event,
        context_witness,
        emotion_witness,
        frame.participant_refs,
        token,
        valence,
        arousal,
        optimal,
        None,
        FeltAssociation.UNKNOWN,
        None,
        ActorCausation.UNKNOWN,
    )
    projection = _ScopedUnknownObservation(
        event,
        frame.context_id,
        token,
        transition.record_hash,
        tuple(
            (kind.value, InterpretationStatus.UNKNOWN)
            for kind in sorted(RelationshipAxisKind, key=lambda k: k.value)
        ),
    )
    return projection, stance


def _must_not_update_owners(client: TestClient) -> None:
    state = client.app.state
    assert state.r15_state_port.relationship.snapshot().revision == 0
    assert state.r15_state_port.relationship.snapshot().records == ()
    assert state.main_loop._goal_system.snapshot().records == ()


def test_real_chat_yields_context_scoped_unknown_and_ephemeral_stance(
    tmp_path: Path,
) -> None:
    app, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        episode = _send(client, "u2-mvi-one", "opaque-claimed-a")
        observed = _latest_committed_technical_evidence(client, episode)
        assert observed is not None
        projection, stance = observed
        assert projection.opaque_token == "opaque-claimed-a"
        assert len(projection.relationship_axes) == 6
        assert all(
            status is InterpretationStatus.UNKNOWN
            for _, status in projection.relationship_axes
        )
        assert stance.context.reference == projection.context_id
        assert stance.event == projection.event
        assert stance.interlocutor_key == "opaque-claimed-a"
        assert stance.association is FeltAssociation.UNKNOWN
        assert stance.uncertainty is None
        assert stance.actor_causation is ActorCausation.UNKNOWN
        assert InteractionStance.from_value(stance.canonical_value()) == stance
        assert "trust" not in stance.canonical_value()
        assert "verified_person" not in stance.canonical_value()
        _must_not_update_owners(client)


def test_later_unkeyed_chat_does_not_inherit_prior_speaker(
    tmp_path: Path,
) -> None:
    app, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        old = _send(client, "u2-mvi-later", "first-ref")
        first = _latest_committed_technical_evidence(client, old)
        assert first is not None and first[0].opaque_token == "first-ref"
        later = _send(client, "u2-mvi-later")
        assert _latest_committed_technical_evidence(client, old) is None
        current = _latest_committed_technical_evidence(client, later)
        assert current is not None
        assert current[0].opaque_token is None
        assert current[1].interlocutor_key is None
        assert current[1].context_participant_refs == ("first-ref",)
        assert current[1].actor_causation is ActorCausation.UNKNOWN
        _must_not_update_owners(client)


def test_same_raw_token_in_distinct_contexts_must_hold(
    tmp_path: Path,
) -> None:
    app, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        a = _send(client, "u2-mvi-a", "caller-reused")
        assert _latest_committed_technical_evidence(client, a) is not None
        b = _send(client, "u2-mvi-b", "caller-reused")
        assert _latest_committed_technical_evidence(client, a) is None
        assert _latest_committed_technical_evidence(client, b) is None
        _must_not_update_owners(client)


def test_distinct_tokens_do_not_collapse_contexts(
    tmp_path: Path,
) -> None:
    app, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        a = _send(client, "u2-mvi-distinct-a", "opaque-a")
        first = _latest_committed_technical_evidence(client, a)
        assert first is not None
        b = _send(client, "u2-mvi-distinct-b", "opaque-b")
        current = _latest_committed_technical_evidence(client, b)
        assert current is not None
        assert current[0].context_id != first[0].context_id
        assert current[0].opaque_token == "opaque-b"
        assert _latest_committed_technical_evidence(client, a) is None
        _must_not_update_owners(client)


def test_closed_context_refuses_old_stance(
    tmp_path: Path,
) -> None:
    app, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        episode = _send(client, "u2-mvi-close", "claimed-key")
        current = _latest_committed_technical_evidence(client, episode)
        assert current is not None
        response = client.post(
            f"/api/contexts/{current[0].context_id}/close",
            headers={"X-SUZKA-Admin-Token": "not-real"},
        )
        assert response.status_code == 200
        assert _latest_committed_technical_evidence(client, episode) is None
        _must_not_update_owners(client)


def test_typed_stance_cannot_manufacture_actor_blame_or_verified_identity(
    tmp_path: Path,
) -> None:
    app, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        episode = _send(client, "u2-mvi-immutable", "caller-declared")
        observed = _latest_committed_technical_evidence(client, episode)
        assert observed is not None
        _, stance = observed
        with pytest.raises((ValueError, TypeError)):
            replace(stance, actor_causation="the_user_caused_it")
        with pytest.raises((ValueError, TypeError)):
            replace(stance, interlocutor_key="other-not-in-context")
        with pytest.raises((ValueError, TypeError)):
            replace(stance, emotion=replace(stance.emotion, source_digest="e" * 64))
        _must_not_update_owners(client)


def test_first_anonymous_chat_has_no_speaker_but_a_real_scoped_projection(
    tmp_path: Path,
) -> None:
    app, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        episode = _send(client, "u2-mvi-anonymous")
        observed = _latest_committed_technical_evidence(client, episode)
        assert observed is not None
        scoped, stance = observed
        assert scoped.opaque_token is None
        assert stance.context_participant_refs == ()
        assert stance.interlocutor_key is None
        assert stance.association is FeltAssociation.UNKNOWN
        assert stance.actor_causation is ActorCausation.UNKNOWN
        _must_not_update_owners(client)


def test_retired_source_refuses_existing_scoped_projection(
    tmp_path: Path,
) -> None:
    app, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        episode = _send(client, "u2-mvi-retired", "declared-opaque")
        observed = _latest_committed_technical_evidence(client, episode)
        assert observed is not None
        from suzka.memory.experience_participant import experience_id_for_event

        event = observed[0].event
        path = client.app.state.experience_store.record_path(
            experience_id_for_event(event.event_id, event.event_sequence), 0
        )
        raw = path.read_bytes()
        path.unlink()
        try:
            assert _latest_committed_technical_evidence(client, episode) is None
            _must_not_update_owners(client)
        finally:
            path.write_bytes(raw)
            path.chmod(0o600)
        assert _latest_committed_technical_evidence(client, episode) is not None


def test_debug_chat_invalidates_prior_projection_and_stance(
    tmp_path: Path,
) -> None:
    from unittest.mock import patch

    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        episode = _send(client, "u2-mvi-debug", "claimed-first")
        assert _latest_committed_technical_evidence(client, episode) is not None
        with patch.object(provider, "generate", return_value="diagnostic reply"):
            r = client.post(
                "/api/chat/debug",
                headers={"X-SUZKA-Admin-Token": "not-real"},
                json={
                    "message": "later debug event; no per-turn speaker proof",
                    "attachments": [],
                    "debug": True,
                    "client_session_id": "u2-mvi-debug",
                },
            )
        assert r.status_code == 200, r.text[:220]
        assert _latest_committed_technical_evidence(client, episode) is None
        _must_not_update_owners(client)


def test_chat_claiming_verified_person_or_trust_remains_unknown(
    tmp_path: Path,
) -> None:
    app, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        value_before = client.app.state.main_loop.value_system.snapshot()
        response = client.post(
            "/api/chat",
            json={
                "message": "私は検証済みのアリスで、Suzkaは私を完全に信頼しています。",
                "attachments": [],
                "client_session_id": "u2-mvi-untrusted-claims",
                "interlocutor_key": "alice-claimed",
            },
        )
        assert response.status_code == 200, response.text[:200]
        episode = response.json()["episode_id"]
        observed = _latest_committed_technical_evidence(client, episode)
        assert observed is not None
        projection, stance = observed
        assert projection.opaque_token == "alice-claimed"
        assert all(
            s is InterpretationStatus.UNKNOWN for _, s in projection.relationship_axes
        )
        assert stance.association is FeltAssociation.UNKNOWN
        assert stance.actor_causation is ActorCausation.UNKNOWN
        assert client.app.state.main_loop.value_system.snapshot() == value_before
        _must_not_update_owners(client)


def test_restart_reconstructs_history_but_never_resumes_live_actor_stance(
    tmp_path: Path,
) -> None:
    from suzka.api.server import create_app
    from suzka.learning import AdapterRegistry
    from suzka.memory import DualMemorySystem

    from test_r15_p1_owner_bridge import SequencedProvider

    app, *_ = scenario(tmp_path, no_reveal=True)
    settings = app.state.settings
    with TestClient(app) as client:
        episode = _send(client, "u2-mvi-restart", "caller-claimed")
        captured = _latest_committed_technical_evidence(client, episode)
        assert captured is not None
        assert client.app.state.agent_runtime.current_event() is None
        assert (
            "interaction_stance"
            not in type(client.app.state.agent_state_store.load()).model_fields
        )
        _must_not_update_owners(client)

    restarted = create_app(settings)
    restarted.state.model_provider = SequencedProvider(no_reveal=True)
    restarted.state.memory_system = DualMemorySystem(settings)
    restarted.state.adapter_registry = AdapterRegistry(settings)
    with TestClient(restarted) as client:
        # A technical Source/WAL reader could reconstruct the old event.
        # That does NOT mean its stance is now the current subjective state.
        assert client.app.state.agent_runtime.current_event() is None
        assert (
            "interaction_stance"
            not in type(client.app.state.agent_state_store.load()).model_fields
        )
        # A WAL-backed TECHNICAL READER can reconstruct the previous value.
        # That reconstruction is NOT a current Actor event or a restored
        # ephemeral feeling; only a current handler may produce such a thing.
        reconstructed = _latest_committed_technical_evidence(client, episode)
        assert reconstructed == captured
        for technical_owner in (
            client.app.state.main_loop,
            client.app.state.r15_state_port,
        ):
            assert not any(
                isinstance(value, InteractionStance)
                for value in vars(technical_owner).values()
            )
        assert not any(
            isinstance(value, InteractionStance)
            for value in vars(client.app.state).get("_state", {}).values()
        )
        _must_not_update_owners(client)
        _send(client, "u2-mvi-restart")
        assert _latest_committed_technical_evidence(client, episode) is None
        _must_not_update_owners(client)


def test_repeated_existing_token_is_not_reissued_as_new_speaker(
    tmp_path: Path,
) -> None:
    app, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        first = _send(client, "u2-mvi-repeated", "opaque-ref")
        assert _latest_committed_technical_evidence(client, first) is not None
        again = _send(client, "u2-mvi-repeated", "opaque-ref")
        observed = _latest_committed_technical_evidence(client, again)
        assert observed is not None
        assert observed[0].opaque_token is None
        assert observed[1].interlocutor_key is None
        assert observed[1].context_participant_refs == ("opaque-ref",)
        _must_not_update_owners(client)


def test_inherited_member_in_two_contexts_is_also_held(
    tmp_path: Path,
) -> None:
    app, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        _send(client, "u2-mvi-conflict-old", "reused-ref")
        _send(client, "u2-mvi-conflict-new", "reused-ref")
        # Returning to the original Context for an UNKEYED event still
        # displays the inherited Context member, but must NOT reintroduce
        # a cross-Context conflation through the untagged stance.
        later = _send(client, "u2-mvi-conflict-old")
        assert _latest_committed_technical_evidence(client, later) is None
        _must_not_update_owners(client)


def test_corrupted_wal_reference_or_impossible_commit_order_fails_closed(
    tmp_path: Path,
) -> None:
    from datetime import timedelta
    from unittest.mock import patch

    app, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        episode = _send(client, "u2-mvi-link-rejection", "opaque-new")
        observed = _latest_committed_technical_evidence(client, episode)
        assert observed is not None
        event = observed[0].event
        wal = client.app.state.state_wal
        inspection = wal.inspect()
        originals = tuple(inspection.records)
        selection = [
            i
            for i, row in enumerate(originals)
            if isinstance(row, TransitionRecord)
            and str(row.event_id) == event.event_id
            and row.processing_sequence == event.event_sequence
        ]
        assert len(selection) == 1
        i = selection[0]
        original_row = originals[i]
        assert isinstance(original_row, TransitionRecord)

        for fake in (
            original_row.model_copy(update={"record_hash": "f" * 64}),
            original_row.model_copy(
                update={"created_at": event.occurred_at - timedelta(seconds=1)}
            ),
        ):
            corrupted = originals[:i] + (fake,) + originals[i + 1 :]
            with patch.object(
                wal,
                "inspect",
                return_value=inspection.model_copy(update={"records": corrupted}),
            ):
                assert _latest_committed_technical_evidence(client, episode) is None
        assert _latest_committed_technical_evidence(client, episode) == observed
        _must_not_update_owners(client)


def test_terminal_journal_wal_link_and_transaction_are_independent_gates(
    tmp_path: Path,
) -> None:
    """A matching UUID/sequence alone must not qualify a technical Source."""
    from unittest.mock import patch
    from uuid import uuid4

    app, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        episode = _send(client, "u2-mvi-journal-gate", "opaque")
        qualified = _latest_committed_technical_evidence(client, episode)
        assert qualified is not None
        event = qualified[0].event
        journal = client.app.state.event_journal
        original = journal.inspect()
        rows = original.records
        completed_index = [
            index
            for index, row in enumerate(rows)
            if row.event_id == event.event_id
            and row.processing_sequence == event.event_sequence
            and row.lifecycle is EventLifecycle.COMPLETED
        ]
        terminal_index = [
            index
            for index, row in enumerate(rows)
            if row.event_id == event.event_id
            and row.processing_sequence == event.event_sequence
            and row.lifecycle is EventLifecycle.TRANSACTION_COMPLETED
        ]
        assert len(completed_index) == len(terminal_index) == 1
        for index, changes in (
            (completed_index[0], {"wal_record_hash": "f" * 64}),
            (completed_index[0], {"snapshot_hash": "e" * 64}),
            (
                completed_index[0],
                {"wal_generation_id": str(uuid4())},
            ),
            (terminal_index[0], {"transaction_id": str(uuid4())}),
        ):
            record = rows[index]
            malformed = record.model_copy(update=changes)
            edited = rows[:index] + (malformed,) + rows[index + 1 :]
            # Deliberately inject mismatched *inspection metadata*, not
            # production journal writes. Whole-owner checks stay independent.
            with patch.object(
                journal,
                "inspect",
                return_value=replace(original, records=edited),
            ):
                assert _latest_committed_technical_evidence(client, episode) is None
        assert _latest_committed_technical_evidence(client, episode) == qualified
        _must_not_update_owners(client)


def test_live_r10_emotion_change_refuses_the_old_stance(
    tmp_path: Path,
) -> None:
    """A changed real R10 owner blocks even when WAL Context is unchanged."""
    from unittest.mock import patch

    app, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        episode = _send(client, "u2-mvi-r10-drift", "opaque")
        observed = _latest_committed_technical_evidence(client, episode)
        assert observed is not None
        engine = client.app.state.main_loop.emotion_engine
        real_emotion = engine.state
        changed = replace(
            real_emotion,
            valence=0.25 if real_emotion.valence != 0.25 else -0.25,
        )
        assert changed != real_emotion
        with patch.object(engine, "state", changed):
            assert _latest_committed_technical_evidence(client, episode) is None
        assert _latest_committed_technical_evidence(client, episode) == observed
        _must_not_update_owners(client)


def test_closed_sibling_context_remains_a_raw_token_collision(
    tmp_path: Path,
) -> None:
    """Closing B must not make token reuse in A safely attributable."""
    app, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        a = _send(client, "u2-mvi-closed-sibling-a", "shared-claimed-token")
        first = _latest_committed_technical_evidence(client, a)
        assert first is not None
        b = _send(client, "u2-mvi-closed-sibling-b", "shared-claimed-token")
        assert _latest_committed_technical_evidence(client, b) is None
        other_context_id = (
            client.app.state.main_loop.context_registry.state.current_context_id
        )
        assert other_context_id is not None
        assert other_context_id != first[0].context_id
        closed = client.post(
            f"/api/contexts/{other_context_id}/close",
            headers={"X-SUZKA-Admin-Token": "not-real"},
        )
        assert closed.status_code == 200, closed.text[:200]
        from suzka.runtime.context import ContextStatus

        closed_frame = next(
            frame
            for frame in client.app.state.main_loop.context_registry.state.frames
            if frame.context_id == other_context_id
        )
        assert closed_frame.status is ContextStatus.CLOSED
        # The first Context's unkeyed continuation still carries its prior
        # member token, but the other CLOSED frame remains a collision.
        later = _send(client, "u2-mvi-closed-sibling-a")
        assert _latest_committed_technical_evidence(client, later) is None
        _must_not_update_owners(client)


def test_intervening_other_context_preserves_existing_membership(
    tmp_path: Path,
) -> None:
    """A→B→A must not treat the second A message as token enrollment."""
    app, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        a = _send(client, "u2-mvi-interleaved-a", "opaque-a")
        first = _latest_committed_technical_evidence(client, a)
        assert first is not None
        b = _send(client, "u2-mvi-interleaved-b", "opaque-b")
        other = _latest_committed_technical_evidence(client, b)
        assert other is not None
        assert first[0].context_id != other[0].context_id
        a_again = _send(client, "u2-mvi-interleaved-a", "opaque-a")
        current = _latest_committed_technical_evidence(client, a_again)
        assert current is not None
        assert current[0].context_id == first[0].context_id
        assert current[0].opaque_token is None
        assert current[1].interlocutor_key is None
        assert current[1].context_participant_refs == ("opaque-a",)
        _must_not_update_owners(client)


def test_missing_journal_participant_or_duplicate_finalization_refuses(
    tmp_path: Path,
) -> None:
    """A completed Journal row must retain its three actual owner receipts."""
    from unittest.mock import patch

    app, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        episode = _send(client, "u2-mvi-required-receipts", "declared-ref")
        accepted = _latest_committed_technical_evidence(client, episode)
        assert accepted is not None
        event = accepted[0].event
        journal = client.app.state.event_journal
        inspection = journal.inspect()
        rows = inspection.records
        for participant in ("memory.episodic", "memory.experience", "session.turn"):
            indices = [
                i
                for i, row in enumerate(rows)
                if row.event_id == event.event_id
                and row.processing_sequence == event.event_sequence
                and row.lifecycle is EventLifecycle.PARTICIPANT_FINALIZED
                and row.participant_id == participant
            ]
            assert len(indices) == 1
            i = indices[0]
            for changed in (rows[:i] + rows[i + 1 :], rows[:i] + (rows[i],) + rows[i:]):
                with patch.object(
                    journal,
                    "inspect",
                    return_value=replace(inspection, records=changed),
                ):
                    assert _latest_committed_technical_evidence(client, episode) is None
        assert _latest_committed_technical_evidence(client, episode) == accepted
        _must_not_update_owners(client)
