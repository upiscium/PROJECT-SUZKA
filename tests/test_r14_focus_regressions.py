"""R14 U6 focus regressions across real R08 and R13 source authorities."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from suzka.attention.adapters import (
    project_attention_candidates,
    project_global_emotion,
)
from suzka.attention.common import (
    ATTENTION_HIGH_AROUSAL_MAX_FOCUS,
    ATTENTION_MAX_FOCUS,
    AttentionTargetKind,
    CandidateAvailability,
)
from suzka.attention.contracts import AttentionEvent
from suzka.attention.policy import (
    ATTENTION_POLICY_SWITCH_COST_UNITS,
    AttentionCandidateReason,
    AttentionContinuityBaseline,
)
from suzka.attention.system import AttentionSystem
from suzka.emotion_contracts import EmotionState
from suzka.motivation.commitment import CommitmentLifecycle
from suzka.motivation.goal import GoalLifecycle
from suzka.motivation.motivation import MotivationLifecycle
from suzka.runtime.working_memory import WorkingMemory
from suzka.working_memory_contracts import (
    MAX_PROJECTION_BYTES,
    WorkingMemorySourceKind,
)
from r14_attention_fixtures import coherent_r13_systems


_NOW = datetime(2026, 2, 1, tzinfo=UTC)


def _event(step: int) -> AttentionEvent:
    return AttentionEvent(
        f"event:r14:u6:focus:{step:04d}",
        100 + step,
        _NOW + timedelta(seconds=step),
    )


def _source_signature(snapshot: object) -> tuple[object, ...]:
    return (
        snapshot.authority_digest,
        snapshot.serialized_bytes,
        tuple(record.record_digest for record in snapshot.records),
    )


def _source_projections(
    working_memory: WorkingMemory,
    motivation: object,
    goals: object,
    commitments: object,
    event: AttentionEvent,
) -> tuple[object, ...]:
    """Capture one complete R08 membership/view and the exact R13 roots."""

    items = working_memory.items
    revision = working_memory.revision
    view = working_memory.select(lambda item: f"temporary source {item.source_id}")
    assert view.revision == revision
    assert working_memory.items == items
    assert working_memory.revision == revision
    projections = project_attention_candidates(
        working_memory_items=items,
        working_memory_revision=revision,
        working_memory_view=view,
        motivation_snapshot=motivation,
        goal_snapshot=goals,
        commitment_snapshot=commitments,
        event=event,
    )
    assert type(projections) is tuple
    assert all(projection.event == event for projection in projections)
    assert all(
        projection.source.source_event_sequence is None
        or projection.source.source_event_sequence < event.event_sequence
        for projection in projections
    )
    return projections


def _decision_by_id(result: object, candidate_id: str) -> object:
    competition = result.view.competition
    assert competition is not None
    return next(
        decision
        for decision in competition.decisions
        if decision.candidate_id == candidate_id
    )


@pytest.mark.parametrize(
    ("arousal", "focus_capacity"),
    [
        pytest.param(0.0, ATTENTION_MAX_FOCUS, id="normal-arousal-16-focus"),
        pytest.param(
            0.75,
            ATTENTION_HIGH_AROUSAL_MAX_FOCUS,
            id="high-arousal-8-focus",
        ),
    ],
)
def test_transient_r08_challengers_displace_then_restore_real_r13_focus(
    arousal: float,
    focus_capacity: int,
) -> None:
    motivation_system, goal_system, commitment_system = coherent_r13_systems(1)
    motivation_snapshot = motivation_system.snapshot()
    goal_snapshot = goal_system.snapshot()
    commitment_snapshot = commitment_system.snapshot()
    source_snapshots = (
        motivation_snapshot,
        goal_snapshot,
        commitment_snapshot,
    )
    source_signatures = tuple(_source_signature(item) for item in source_snapshots)

    assert len(motivation_snapshot.records) == 1
    assert motivation_snapshot.records[0].lifecycle is MotivationLifecycle.ACTIVE
    assert len(goal_snapshot.records) == 1
    assert goal_snapshot.records[0].lifecycle is GoalLifecycle.ADOPTED
    assert len(commitment_snapshot.records) == 1
    assert commitment_snapshot.records[0].lifecycle is CommitmentLifecycle.ACTIVE

    working_memory = WorkingMemory(
        item_capacity=ATTENTION_MAX_FOCUS,
        projection_max_bytes=MAX_PROJECTION_BYTES,
    )
    attention = AttentionSystem()

    def emotion_for(event: AttentionEvent) -> object:
        return project_global_emotion(EmotionState(arousal=arousal), event=event)

    first_event = _event(1)
    first_emotion = emotion_for(first_event)
    first_projections = _source_projections(
        working_memory,
        motivation_snapshot,
        goal_snapshot,
        commitment_snapshot,
        first_event,
    )
    r13_projections = {
        projection.target.kind: projection
        for projection in first_projections
        if projection.target.kind is not AttentionTargetKind.WORKING_MEMORY
    }
    goal_id = r13_projections[AttentionTargetKind.GOAL].candidate_id
    commitment_id = r13_projections[AttentionTargetKind.COMMITMENT].candidate_id
    initial_r13_ids = {
        projection.candidate_id for projection in r13_projections.values()
    }

    # The fixture's actual latest revision events are preserved in the typed
    # upstream witnesses, and every caller Attention event is later than them.
    for projection, record_snapshot in (
        (r13_projections[AttentionTargetKind.MOTIVATION], motivation_snapshot),
        (r13_projections[AttentionTargetKind.GOAL], goal_snapshot),
        (r13_projections[AttentionTargetKind.COMMITMENT], commitment_snapshot),
    ):
        latest = record_snapshot.records[0].revision_history[-1]
        assert projection.source.source_event_id == latest.event_id
        assert projection.source.source_event_sequence == latest.event_sequence
        assert projection.source.source_occurred_at == latest.created_at
        assert latest.event_sequence < first_event.event_sequence
    assert r13_projections[AttentionTargetKind.GOAL].signals.urgency == 0.0
    assert r13_projections[AttentionTargetKind.COMMITMENT].signals.urgency == 0.0

    initial = attention.refresh(
        first_projections,
        first_event,
        global_emotion=first_emotion,
    )
    assert initial.view.competition is not None
    assert initial.view.competition.focus_capacity == focus_capacity
    assert set(initial.snapshot.focused_ids) == initial_r13_ids
    assert initial.snapshot.unfinished_ids == ()

    transient_ids: list[str] = []
    for index in range(focus_capacity):
        admission = working_memory.admit(
            WorkingMemorySourceKind.EPISODIC,
            f"u6-transient-source-{index:02d}",
            activation=1.0,
            salience=1.0,
        )
        assert admission.retained is True
        transient_ids.append(admission.item.item_id)
    assert len(working_memory.items) == focus_capacity

    challenge_event = _event(2)
    challenge_emotion = emotion_for(challenge_event)
    challenge_projections = _source_projections(
        working_memory,
        motivation_snapshot,
        goal_snapshot,
        commitment_snapshot,
        challenge_event,
    )
    assert len(challenge_projections) == focus_capacity + len(initial_r13_ids)
    transient_candidate_ids = {
        projection.candidate_id
        for projection in challenge_projections
        if projection.target.kind is AttentionTargetKind.WORKING_MEMORY
    }
    assert len(transient_candidate_ids) == focus_capacity
    assert all(
        projection.event == challenge_event for projection in challenge_projections
    )
    challenge = attention.refresh(
        challenge_projections,
        challenge_event,
        global_emotion=challenge_emotion,
    )

    competition = challenge.view.competition
    assert competition is not None
    assert competition.focus_capacity == focus_capacity
    assert len(challenge.snapshot.focused_ids) == focus_capacity
    assert len(challenge.snapshot.candidates) == len(challenge_projections)
    assert {candidate.candidate_id for candidate in challenge.snapshot.candidates} == {
        projection.candidate_id for projection in challenge_projections
    }
    assert goal_id not in challenge.snapshot.focused_ids
    assert commitment_id not in challenge.snapshot.focused_ids
    assert {goal_id, commitment_id} <= set(challenge.snapshot.unfinished_ids)
    assert len(challenge.snapshot.unfinished_ids) <= ATTENTION_MAX_FOCUS
    assert competition.unfinished_ids == challenge.snapshot.unfinished_ids
    assert transient_candidate_ids.intersection(challenge.snapshot.focused_ids)

    goal_decision = _decision_by_id(challenge, goal_id)
    commitment_decision = _decision_by_id(challenge, commitment_id)
    assert goal_decision.availability is CandidateAvailability.ELIGIBLE
    assert commitment_decision.availability is CandidateAvailability.ELIGIBLE
    assert goal_decision.reason is AttentionCandidateReason.FOCUS_CAPACITY
    assert commitment_decision.reason is AttentionCandidateReason.FOCUS_CAPACITY
    for candidate_id in transient_candidate_ids:
        decision = _decision_by_id(challenge, candidate_id)
        assert decision.continuity_baseline is AttentionContinuityBaseline.NEW
        assert decision.prior_candidate_present is False
        assert decision.switch_cost_units == ATTENTION_POLICY_SWITCH_COST_UNITS
        assert decision.score_units > goal_decision.score_units
        assert decision.score_units > commitment_decision.score_units

    retained_by_id = {
        candidate.candidate_id: candidate for candidate in challenge.snapshot.candidates
    }
    for projection in first_projections:
        if projection.target.kind is not AttentionTargetKind.WORKING_MEMORY:
            retained = retained_by_id[projection.candidate_id]
            assert retained.source == projection.source
            assert retained.unattended_event_count == 1
            assert retained.focused_event_count == 0
    assert tuple(_source_signature(item) for item in source_snapshots) == (
        source_signatures
    )
    assert tuple(
        _source_signature(item)
        for item in (
            motivation_system.snapshot(),
            goal_system.snapshot(),
            commitment_system.snapshot(),
        )
    ) == source_signatures

    # An exact event retry reuses its receipt and continuity root; it cannot
    # spend another habituation or streak step.
    challenge_bytes = challenge.snapshot.canonical_bytes()
    challenge_counters = tuple(
        (
            candidate.candidate_id,
            candidate.habituation,
            candidate.inhibition,
            candidate.focused_event_count,
            candidate.unattended_event_count,
        )
        for candidate in challenge.snapshot.candidates
    )
    replay = attention.refresh(
        challenge_projections,
        challenge_event,
        global_emotion=challenge_emotion,
    )
    assert replay.replayed is True
    assert replay.receipt == challenge.receipt
    assert replay.snapshot.canonical_bytes() == challenge_bytes
    assert replay.snapshot.revision == challenge.snapshot.revision
    assert tuple(
        (
            candidate.candidate_id,
            candidate.habituation,
            candidate.inhibition,
            candidate.focused_event_count,
            candidate.unattended_event_count,
        )
        for candidate in replay.snapshot.candidates
    ) == challenge_counters

    # R08 legitimately removes the temporary current membership. The next
    # complete capture contains the unchanged R13 authorities without rebuilding
    # or replaying their source records.
    for item_id in transient_ids:
        assert working_memory.forget(item_id) is True
    assert working_memory.items == ()
    restore_event = _event(3)
    restore_projections = _source_projections(
        working_memory,
        motivation_snapshot,
        goal_snapshot,
        commitment_snapshot,
        restore_event,
    )
    assert len(restore_projections) == len(initial_r13_ids) == 3
    assert all(
        projection.target.kind is not AttentionTargetKind.WORKING_MEMORY
        for projection in restore_projections
    )
    assert all(
        projection.source.source_event_sequence < restore_event.event_sequence
        for projection in restore_projections
    )

    restored = attention.refresh(
        restore_projections,
        restore_event,
        global_emotion=emotion_for(restore_event),
    )
    assert restored.view.competition is not None
    assert restored.view.competition.focus_capacity == focus_capacity
    assert restored.snapshot.revision == challenge.snapshot.revision + 1
    assert set(restored.snapshot.focused_ids) == initial_r13_ids
    assert goal_id in restored.snapshot.focused_ids
    assert commitment_id in restored.snapshot.focused_ids
    assert transient_candidate_ids.isdisjoint(
        candidate.candidate_id for candidate in restored.snapshot.candidates
    )
    for projection in restore_projections:
        retained = next(
            candidate
            for candidate in restored.snapshot.candidates
            if candidate.candidate_id == projection.candidate_id
        )
        assert retained.source == projection.source
    assert tuple(_source_signature(item) for item in source_snapshots) == (
        source_signatures
    )
    assert tuple(
        _source_signature(item)
        for item in (
            motivation_system.snapshot(),
            goal_system.snapshot(),
            commitment_system.snapshot(),
        )
    ) == source_signatures
