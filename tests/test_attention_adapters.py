"""R14 U1 source-adapter provenance and boundedness tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
import os
from pathlib import Path
import subprocess
import sys

import pytest

from suzka.attention.adapters import (
    GlobalEmotionProjection,
    project_attention_candidates,
    project_commitment,
    project_global_emotion,
    project_goal,
    project_motivation,
    project_working_memory,
)
from suzka.attention.common import (
    ATTENTION_MAX_RENDERED_ITEM_BYTES,
    ATTENTION_PROMPT_BUDGET_BYTES,
    AttentionSourceKind,
    AttentionTargetKind,
    CandidateAvailability,
    SourceDigestKind,
    canonical_json,
    digest_payload,
)
from suzka.attention.contracts import (
    AttentionCandidateProjection,
    AttentionContinuity,
    AttentionEvent,
    AttentionSignalVector,
    AttentionTarget,
)
from suzka.attention.policy import AttentionSignalDimension, compete_attention
from suzka.context_contracts import ContextRelation
from suzka.emotion_contracts import EmotionState
from suzka.motivation.common import Deadline
from suzka.motivation.commitment import CommitmentLifecycle
from suzka.motivation.commitment_system import CommitmentSystemSnapshot
from suzka.motivation.goal import GoalLifecycle
from suzka.motivation.goal_system import GoalSystemSnapshot
from suzka.motivation.motivation import MotivationLifecycle
from suzka.motivation.projection import (
    CommitmentPromptEntry,
    GoalPromptEntry,
    MotivationPromptEntry,
    _commitment_value,
    _goal_value,
    _motivation_value,
)
from suzka.motivation.system import MotivationSystemSnapshot
from suzka.working_memory_contracts import (
    MAX_ITEM_CAPACITY,
    MAX_PROJECTION_BYTES,
    WorkingMemoryDecision,
    WorkingMemoryDecisionReason,
    WorkingMemoryItem,
    WorkingMemoryRetentionReason,
    WorkingMemorySelection,
    WorkingMemorySourceKind,
    WorkingMemoryContextProjection,
    WorkingMemoryView,
    working_memory_item_id,
)


NOW = datetime(2026, 2, 1, tzinfo=UTC)
EVENT = AttentionEvent("attention:event:current", 1_000, NOW)


def _item(
    source_id: str = "episode-source-one",
    *,
    source_kind: WorkingMemorySourceKind = WorkingMemorySourceKind.EPISODIC,
    activation: float = 0.8,
    salience: float = 0.6,
    retention_reason: WorkingMemoryRetentionReason = WorkingMemoryRetentionReason.RECENT,
) -> WorkingMemoryItem:
    return WorkingMemoryItem(
        item_id=working_memory_item_id(source_kind, source_id),
        source_kind=source_kind,
        source_id=source_id,
        activation=activation,
        salience=salience,
        retention_reason=retention_reason,
        created_revision=1,
        last_activated_revision=1,
    )


def _score(item: WorkingMemoryItem) -> float:
    bonus = (
        0.1
        if item.retention_reason is WorkingMemoryRetentionReason.REACTIVATED
        else 0.0
    )
    return 0.6 * item.activation + 0.4 * item.salience + bonus


def _view(
    item: WorkingMemoryItem,
    *,
    content: str | None = None,
    reason: WorkingMemoryDecisionReason = WorkingMemoryDecisionReason.UNRESOLVED_REFERENCE,
    relation: ContextRelation | None = None,
    compatibility: float | None = None,
    effective_score: float | None = None,
    context_projection: WorkingMemoryContextProjection | None = None,
) -> WorkingMemoryView:
    selected = reason is WorkingMemoryDecisionReason.SELECTED
    score = _score(item)
    decision = WorkingMemoryDecision(
        item_id=item.item_id,
        source_kind=item.source_kind,
        source_id=item.source_id,
        selected=selected,
        score=score,
        reason=reason,
        context_relation=relation,
        context_compatibility=compatibility,
        effective_score=effective_score,
        context_projection=context_projection,
    )
    selections: tuple[WorkingMemorySelection, ...] = ()
    projected_bytes = 0
    if selected:
        assert content is not None
        selections = (
            WorkingMemorySelection(
                item_id=item.item_id,
                source_kind=item.source_kind,
                source_id=item.source_id,
                rendered_content=content,
                score=score,
                reason=reason,
                source_context_id="context:source" if relation is not None else None,
                context_relation=relation,
                context_compatibility=compatibility,
                effective_score=effective_score,
                context_projection=context_projection,
            ),
        )
        projected_bytes = len(content.encode("utf-8"))
    return WorkingMemoryView(
        selected=selections,
        decisions=(decision,),
        projected_bytes=projected_bytes,
        item_capacity=1,
        projection_max_bytes=MAX_PROJECTION_BYTES,
        revision=1,
    )


def _source_systems(
    motivation_lifecycles: tuple[MotivationLifecycle, ...] = (
        MotivationLifecycle.ACTIVE,
    ),
    goal_lifecycles: tuple[GoalLifecycle, ...] = (GoalLifecycle.ADOPTED,),
    commitment_lifecycles: tuple[CommitmentLifecycle, ...] = (
        CommitmentLifecycle.ACTIVE,
    ),
    *,
    goal_deadline: Deadline | None = None,
    commitment_deadline: Deadline | None = None,
) -> tuple[
    MotivationSystemSnapshot,
    GoalSystemSnapshot,
    CommitmentSystemSnapshot,
]:
    from test_r13_projection import (
        _make_commitment_system,
        _make_goal_system,
        _make_motivation_system,
    )

    motivation = _make_motivation_system(motivation_lifecycles).snapshot()
    goal = _make_goal_system(goal_lifecycles, deadline=goal_deadline).snapshot()
    commitment = _make_commitment_system(
        commitment_lifecycles, deadline=commitment_deadline
    ).snapshot()
    return motivation, goal, commitment


def _empty_source_snapshots() -> tuple[
    MotivationSystemSnapshot,
    GoalSystemSnapshot,
    CommitmentSystemSnapshot,
]:
    return _source_systems((), (), ())


def test_adapter_import_is_dependency_neutral_in_a_fresh_process() -> None:
    script = """
import sys
from datetime import UTC, datetime
from suzka.attention.adapters import (
    GlobalEmotionProjection,
    project_attention_candidates,
    project_global_emotion,
)
from suzka.attention.contracts import AttentionEvent
from suzka.emotion_contracts import EmotionState

assert GlobalEmotionProjection is not None
assert project_attention_candidates is not None
assert project_global_emotion(
    EmotionState(arousal=0.5),
    event=AttentionEvent('attention:clean', 1, datetime(2026, 1, 1, tzinfo=UTC)),
)
for prefix in (
    'suzka.runtime', 'suzka.models', 'suzka.body', 'suzka.cognition',
    'suzka.scheduler', 'torch', 'transformers', 'peft',
):
    assert not any(
        name == prefix or name.startswith(prefix + '.') for name in sys.modules
    ), prefix
"""
    environment = os.environ.copy()
    environment.pop("SUZKA_CONFIG_PATH", None)
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env=environment,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0, result.stderr


def test_working_memory_without_view_retains_measured_member_signals() -> None:
    item = _item(activation=0.75, salience=0.25)

    projection = project_working_memory(item, revision=1, event=EVENT)

    assert projection.target == AttentionTarget(
        AttentionTargetKind.WORKING_MEMORY, item.item_id
    )
    assert projection.source.kind is AttentionSourceKind.WORKING_MEMORY
    assert projection.source.digest_kind is SourceDigestKind.ATTENTION_PROJECTION
    assert projection.source.reference == item.item_id
    assert projection.source.target_reference == item.item_id
    assert projection.source.revision == 1
    assert projection.source.source_event_id is None
    assert projection.availability is CandidateAvailability.UNAVAILABLE
    assert projection.signals == AttentionSignalVector(
        activation=0.75, salience=0.25
    )
    assert projection.rendered_bytes is None
    assert projection.rendered_digest is None
    assert "rendered_content" not in projection.__dataclass_fields__
    assert projection.source.digest == project_working_memory(
        item, revision=1, event=EVENT
    ).source.digest


def test_selected_working_memory_row_is_exact_and_may_exceed_prompt_budget() -> None:
    item = _item()
    content = "x" * (ATTENTION_PROMPT_BUDGET_BYTES + 1)
    view = _view(
        item,
        content=content,
        reason=WorkingMemoryDecisionReason.SELECTED,
    )

    projection = project_working_memory(item, revision=1, event=EVENT, view=view)

    row = {
        "item_id": item.item_id,
        "source_kind": item.source_kind.value,
        "source_id": item.source_id,
        "text": content,
    }
    assert projection.availability is CandidateAvailability.ELIGIBLE
    assert projection.rendered_bytes == len(canonical_json(row))
    assert projection.rendered_bytes > ATTENTION_PROMPT_BUDGET_BYTES
    assert projection.rendered_bytes <= ATTENTION_MAX_RENDERED_ITEM_BYTES
    assert projection.rendered_digest == digest_payload(
        b"PROJECT-SUZKA:R14:ATTENTION-WM-ROW:V1\0", row
    )
    assert content not in repr(projection)


def test_unavailable_working_memory_decision_preserves_exact_flat_context_only() -> None:
    item = _item()
    cases = (
        (ContextRelation.SAME_CONTEXT, 1.0, 1.0),
        (ContextRelation.PARENT_CHILD, 0.8, 0.8),
        (ContextRelation.RELATED, 0.75, 0.75),
        (ContextRelation.SHARED_INTERLOCUTOR, 0.65, 0.65),
        (ContextRelation.UNRELATED, 0.2, 0.2),
        (ContextRelation.LEGACY_UNKNOWN, 0.45, None),
        (ContextRelation.UNKNOWN_CONTEXT, 0.35, None),
    )
    for relation, compatibility, expected_signal in cases:
        view = _view(
            item,
            reason=WorkingMemoryDecisionReason.PROJECTION_BUDGET,
            relation=relation,
            compatibility=compatibility,
            effective_score=_score(item) * compatibility,
        )

        projection = project_working_memory(item, revision=1, event=EVENT, view=view)

        assert projection.availability is CandidateAvailability.UNAVAILABLE
        assert projection.signals.activation == item.activation
        assert projection.signals.salience == item.salience
        assert projection.signals.context_compatibility == expected_signal
        assert projection.rendered_bytes is None


def test_unsupported_context_projection_is_opaque_and_never_invented() -> None:
    class UntrustedContextProjection:
        @property
        def aggregate_compatibility(self) -> float:
            raise AssertionError("adapter inspected unsupported projection property")

    item = _item()
    context_evidence = UntrustedContextProjection()
    view = _view(
        item,
        reason=WorkingMemoryDecisionReason.PROJECTION_BUDGET,
        compatibility=0.8,
        effective_score=_score(item) * 0.8,
        context_projection=context_evidence,
    )

    projection = project_working_memory(item, revision=1, event=EVENT, view=view)

    assert projection.availability is CandidateAvailability.UNAVAILABLE
    assert projection.signals.activation == item.activation
    assert projection.signals.salience == item.salience
    assert projection.signals.context_compatibility is None


def test_composite_context_relation_metadata_never_becomes_a_flat_signal() -> None:
    class OpaqueContextProjection:
        @property
        def aggregate_compatibility(self) -> float:
            raise AssertionError("adapter inspected composite compatibility")

    item = _item()
    view = _view(
        item,
        reason=WorkingMemoryDecisionReason.PROJECTION_BUDGET,
        relation=ContextRelation.SAME_CONTEXT,
        compatibility=0.6,
        effective_score=_score(item) * 0.6,
        context_projection=OpaqueContextProjection(),
    )

    projection = project_working_memory(item, revision=1, event=EVENT, view=view)

    assert projection.signals.context_compatibility is None


def test_selected_view_requires_the_exact_same_composite_projection_object() -> None:
    class OpaqueContextProjection:
        @property
        def aggregate_compatibility(self) -> float:
            raise AssertionError("adapter inspected composite compatibility")

    item = _item()
    view = _view(
        item,
        content="bounded row",
        reason=WorkingMemoryDecisionReason.SELECTED,
        context_projection=OpaqueContextProjection(),
    )
    mismatched_selection = replace(
        view.selected[0], context_projection=OpaqueContextProjection()
    )
    mismatched_view = replace(view, selected=(mismatched_selection,))

    with pytest.raises(ValueError, match="does not match its decision"):
        project_working_memory(
            item, revision=1, event=EVENT, view=mismatched_view
        )


@pytest.mark.parametrize(
    "bad_view",
    [
        object(),
        replace(_view(_item()), revision=2),
        replace(_view(_item()), decisions=()),
        replace(
            _view(_item()),
            decisions=(
                replace(_view(_item()).decisions[0], score=0.123),
            ),
        ),
    ],
)
def test_working_memory_rejects_wrong_or_incoherent_views(bad_view: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        project_working_memory(_item(), revision=1, event=EVENT, view=bad_view)  # type: ignore[arg-type]


def test_working_memory_rejects_bad_items_and_future_member_revisions() -> None:
    item = _item()
    malformed_id = replace(item, item_id="wm-" + "0" * 64)
    future_item = replace(item, last_activated_revision=2)
    bad_score_view = replace(
        _view(item),
        decisions=(replace(_view(item).decisions[0], score=0.1),),
    )

    for malformed, revision, view in (
        (malformed_id, 1, None),
        (future_item, 1, None),
        (item, True, None),
        (item, 1, bad_score_view),
    ):
        with pytest.raises((TypeError, ValueError)):
            project_working_memory(
                malformed, revision=revision, event=EVENT, view=view  # type: ignore[arg-type]
            )


def test_motivation_projects_only_current_record_facts_and_exact_source_event() -> None:
    motivation_snapshot, _, _ = _source_systems()
    record = motivation_snapshot.records[0]
    original_digest = record.record_digest
    latest = record.revision_history[-1]

    projection = project_motivation(record, event=EVENT)

    assert projection.target.kind is AttentionTargetKind.MOTIVATION
    assert projection.target.reference == record.motivation_id
    assert projection.source.kind is AttentionSourceKind.MOTIVATION
    assert projection.source.digest_kind is SourceDigestKind.UPSTREAM_AUTHORITY
    assert projection.source.digest == original_digest
    assert projection.source.revision == record.revision
    assert projection.source.source_event_id == latest.event_id
    assert projection.source.source_event_sequence == latest.event_sequence
    assert projection.source.source_occurred_at == latest.created_at
    assert projection.availability is CandidateAvailability.ELIGIBLE
    assert projection.signals == AttentionSignalVector(
        strength=record.strength,
        persistence=record.persistence,
        satiation=record.satiation,
        uncertainty=record.uncertainty,
    )
    assert projection.rendered_bytes == len(
        canonical_json(_motivation_value(MotivationPromptEntry._from_record(record)))
    )
    assert record.record_digest == original_digest


def test_goal_and_commitment_rows_use_only_the_existing_minimal_serializers() -> None:
    _, goal_snapshot, commitment_snapshot = _source_systems()
    goal = goal_snapshot.records[0]
    commitment = commitment_snapshot.records[0]

    goal_projection = project_goal(goal, event=EVENT)
    commitment_projection = project_commitment(commitment, event=EVENT)

    assert goal_projection.target.kind is AttentionTargetKind.GOAL
    assert goal_projection.target.reference == goal.goal_id
    assert goal_projection.source.digest == goal.record_digest
    assert goal_projection.rendered_bytes == len(
        canonical_json(_goal_value(GoalPromptEntry._from_record(goal)))
    )
    assert commitment_projection.target.kind is AttentionTargetKind.COMMITMENT
    assert commitment_projection.target.reference == commitment.commitment_id
    assert commitment_projection.source.digest == commitment.record_digest
    assert commitment_projection.rendered_bytes == len(
        canonical_json(_commitment_value(CommitmentPromptEntry._from_record(commitment)))
    )
    assert commitment.subject not in repr(commitment_projection)
    assert goal_projection.signals.novelty is None
    assert commitment_projection.signals.novelty is None


def test_r13_clone_rejects_record_digest_drift_without_rewriting_source() -> None:
    motivation_snapshot, _, _ = _source_systems()
    record = motivation_snapshot.records[0]
    forged_digest = "0" * 64
    object.__setattr__(record, "record_digest", forged_digest)

    with pytest.raises(ValueError, match="digest"):
        project_motivation(record, event=EVENT)

    assert record.record_digest == forged_digest


def test_goal_and_commitment_deadline_urgency_uses_event_time_fixed_point() -> None:
    def goal_projection(
        deadline: Deadline | None,
    ) -> AttentionCandidateProjection:
        _, goal_snapshot, _ = _source_systems(goal_deadline=deadline)
        return project_goal(goal_snapshot.records[0], event=EVENT)

    def commitment_projection(
        deadline: Deadline | None,
    ) -> AttentionCandidateProjection:
        _, _, commitment_snapshot = _source_systems(commitment_deadline=deadline)
        return project_commitment(commitment_snapshot.records[0], event=EVENT)

    for project in (goal_projection, commitment_projection):
        assert project(Deadline.without_deadline()).signals.urgency == 0.0
        assert project(Deadline(NOW)).signals.urgency == 1.0
        assert project(Deadline(NOW - timedelta(microseconds=1))).signals.urgency == 1.0
        assert project(Deadline(NOW + timedelta(hours=24))).signals.urgency == 0.0
        assert project(Deadline(NOW + timedelta(hours=48))).signals.urgency == 0.0
    assert goal_projection(Deadline(NOW + timedelta(hours=16))).signals.urgency == 0.333333
    assert commitment_projection(Deadline(NOW + timedelta(hours=12))).signals.urgency == 0.5


@pytest.mark.parametrize("kind", (AttentionTargetKind.GOAL, AttentionTargetKind.COMMITMENT))
def test_no_deadline_and_distant_deadline_have_equal_measured_urgency_contribution(
    kind: AttentionTargetKind,
) -> None:
    def projection(deadline: Deadline) -> AttentionCandidateProjection:
        _, goals, commitments = _source_systems(
            goal_deadline=deadline, commitment_deadline=deadline
        )
        if kind is AttentionTargetKind.GOAL:
            return project_goal(goals.records[0], event=EVENT)
        return project_commitment(commitments.records[0], event=EVENT)

    no_deadline = projection(Deadline.without_deadline())
    distant = projection(Deadline(NOW + timedelta(hours=48)))
    assert no_deadline.signals.urgency == distant.signals.urgency == 0.0

    prior = AttentionContinuity.bootstrap()
    no_deadline_decision = compete_attention((no_deadline,), prior, EVENT).decisions[0]
    distant_decision = compete_attention((distant,), prior, EVENT).decisions[0]
    assert no_deadline_decision.base_score_units == distant_decision.base_score_units == 400_000
    assert no_deadline_decision.score_units == distant_decision.score_units
    assert AttentionSignalDimension.URGENCY not in no_deadline_decision.missing_dimensions
    assert no_deadline_decision.missing_dimensions == distant_decision.missing_dimensions


def test_r13_adapters_reject_future_events_and_ineligible_lifecycles() -> None:
    _, adopted_goals, active_commitments = _source_systems()
    goal = adopted_goals.records[0]
    latest_goal_event = goal.revision_history[-1]
    assert latest_goal_event.event_sequence is not None
    earlier = AttentionEvent(
        "attention:event:earlier",
        latest_goal_event.event_sequence - 1,
        NOW - timedelta(days=1),
    )
    with pytest.raises(ValueError):
        project_goal(goal, event=earlier)

    proposed_motivation, proposed_goal, proposed_commitment = _source_systems(
        (MotivationLifecycle.DORMANT,),
        (GoalLifecycle.PROPOSED,),
        (CommitmentLifecycle.PROPOSED,),
    )
    with pytest.raises(ValueError):
        project_motivation(proposed_motivation.records[0], event=EVENT)
    with pytest.raises(ValueError):
        project_goal(proposed_goal.records[0], event=EVENT)
    with pytest.raises(ValueError):
        project_commitment(proposed_commitment.records[0], event=EVENT)
    assert active_commitments.records[0].lifecycle is CommitmentLifecycle.ACTIVE


def test_snapshot_batch_requires_complete_working_memory_membership_coverage() -> None:
    item = _item()
    motivation, goals, commitments = _empty_source_snapshots()
    partial_view = _view(item)
    with pytest.raises(ValueError, match="cover the complete"):
        project_attention_candidates(
            working_memory_items=(item, _item("episode-source-two")),
            working_memory_revision=1,
            working_memory_view=partial_view,
            motivation_snapshot=motivation,
            goal_snapshot=goals,
            commitment_snapshot=commitments,
            event=EVENT,
        )


def test_snapshot_batch_accepts_empty_sources_and_keeps_unavailable_members() -> None:
    item = _item("semantic-member", source_kind=WorkingMemorySourceKind.SEMANTIC)
    motivation, goals, commitments = _empty_source_snapshots()

    result = project_attention_candidates(
        working_memory_items=(item,),
        working_memory_revision=1,
        working_memory_view=None,
        motivation_snapshot=motivation,
        goal_snapshot=goals,
        commitment_snapshot=commitments,
        event=EVENT,
    )

    assert len(result) == 1
    assert result[0].target.kind is AttentionTargetKind.WORKING_MEMORY
    assert result[0].source.reference == item.item_id
    assert result[0].availability is CandidateAvailability.UNAVAILABLE
    assert result[0].signals.activation == item.activation
    assert result[0].signals.salience == item.salience
    assert result[0].signals.context_compatibility is None


def test_snapshot_batch_rejects_duplicate_working_memory_membership() -> None:
    item = _item()
    motivation, goals, commitments = _empty_source_snapshots()
    with pytest.raises(ValueError, match="duplicate"):
        project_attention_candidates(
            working_memory_items=(item, item),
            working_memory_revision=1,
            working_memory_view=None,
            motivation_snapshot=motivation,
            goal_snapshot=goals,
            commitment_snapshot=commitments,
            event=EVENT,
        )


def test_snapshot_batch_covers_full_4192_source_universe_canonically() -> None:
    from test_r13_projection import (
        _make_commitment_system,
        _make_goal_system,
        _make_motivation_system,
    )

    items = tuple(
        _item(
            f"episode-member-{index:04d}",
            activation=0.5,
            salience=0.25,
        )
        for index in range(MAX_ITEM_CAPACITY)
    )
    decisions = tuple(
        WorkingMemoryDecision(
            item_id=item.item_id,
            source_kind=item.source_kind,
            source_id=item.source_id,
            selected=False,
            score=_score(item),
            reason=WorkingMemoryDecisionReason.UNRESOLVED_REFERENCE,
        )
        for item in reversed(items)
    )
    view = WorkingMemoryView(
        selected=(),
        decisions=decisions,
        projected_bytes=0,
        item_capacity=MAX_ITEM_CAPACITY,
        projection_max_bytes=MAX_PROJECTION_BYTES,
        revision=1,
    )
    count = 32
    motivation = _make_motivation_system((MotivationLifecycle.ACTIVE,) * count).snapshot()
    goals = _make_goal_system((GoalLifecycle.ADOPTED,) * count).snapshot()
    commitments = _make_commitment_system(
        (CommitmentLifecycle.ACTIVE,) * count
    ).snapshot()
    def project(
        items_to_project: tuple[WorkingMemoryItem, ...],
    ) -> tuple[AttentionCandidateProjection, ...]:
        return project_attention_candidates(
            working_memory_items=items_to_project,
            working_memory_revision=1,
            working_memory_view=view,
            motivation_snapshot=motivation,
            goal_snapshot=goals,
            commitment_snapshot=commitments,
            event=EVENT,
        )

    reverse_result = project(tuple(reversed(items)))
    forward_result = project(items)

    assert len(reverse_result) == MAX_ITEM_CAPACITY + 3 * count == 4_192
    assert reverse_result == forward_result
    assert tuple(item.candidate_id for item in reverse_result) == tuple(
        sorted(item.candidate_id for item in reverse_result)
    )
    assert sum(
        item.target.kind is AttentionTargetKind.WORKING_MEMORY
        and item.availability is CandidateAvailability.UNAVAILABLE
        for item in reverse_result
    ) == MAX_ITEM_CAPACITY
    assert sum(
        item.target.kind is not AttentionTargetKind.WORKING_MEMORY
        and item.availability is CandidateAvailability.ELIGIBLE
        for item in reverse_result
    ) == 3 * count


def test_global_emotion_is_sealed_event_scoped_resource_evidence_only() -> None:
    state = EmotionState(valence=-0.4, arousal=0.8, optimal_loss=-3.0)
    before = (state.valence, state.arousal, state.optimal_loss)

    projection = project_global_emotion(state, event=EVENT)
    original_fields = (
        projection.event,
        projection.event.event_id,
        projection.event.event_sequence,
        projection.event.occurred_at,
        projection.arousal,
        projection.projection_digest,
    )

    assert type(projection) is GlobalEmotionProjection
    assert projection.event == EVENT
    assert projection.arousal == 0.8
    assert len(projection.projection_digest) == 64
    assert (state.valence, state.arousal, state.optimal_loss) == before
    checked_copy = projection.validated_copy()
    assert checked_copy is not projection
    assert checked_copy.event is not projection.event
    assert checked_copy == projection
    assert (
        projection.event,
        projection.event.event_id,
        projection.event.event_sequence,
        projection.event.occurred_at,
        projection.arousal,
        projection.projection_digest,
    ) == original_fields
    with pytest.raises(TypeError):
        GlobalEmotionProjection(EVENT, 0.8)
    with pytest.raises(TypeError):
        project_global_emotion(0.8, event=EVENT)  # type: ignore[arg-type]
    for field_name, value in (
        ("valence", float("nan")),
        ("arousal", 1.1),
        ("optimal_loss", float("inf")),
    ):
        malformed = EmotionState()
        object.__setattr__(malformed, field_name, value)
        with pytest.raises((TypeError, ValueError)):
            project_global_emotion(malformed, event=EVENT)
    assert project_global_emotion(
        EmotionState(valence=0.9, arousal=0.8, optimal_loss=99.0), event=EVENT
    ).projection_digest == projection.projection_digest


@pytest.mark.parametrize(
    "tamper",
    ["digest", "arousal", "event_id", "event_sequence"],
)
def test_global_emotion_validated_copy_rejects_tampering_without_repair(
    tamper: str,
) -> None:
    invalid = project_global_emotion(EmotionState(arousal=0.4), event=EVENT)
    if tamper == "digest":
        object.__setattr__(invalid, "projection_digest", "0" * 64)
    elif tamper == "arousal":
        object.__setattr__(invalid, "arousal", 0.7)
    elif tamper == "event_id":
        object.__setattr__(invalid.event, "event_id", "attention:event:tampered")
    else:
        object.__setattr__(invalid.event, "event_sequence", True)
    original = (
        invalid.event,
        invalid.event.event_id,
        invalid.event.event_sequence,
        invalid.event.occurred_at,
        invalid.arousal,
        invalid.projection_digest,
    )

    with pytest.raises((TypeError, ValueError)):
        invalid.validated_copy()
    with pytest.raises((TypeError, ValueError)):
        invalid.__post_init__()

    assert (
        invalid.event,
        invalid.event.event_id,
        invalid.event.event_sequence,
        invalid.event.occurred_at,
        invalid.arousal,
        invalid.projection_digest,
    ) == original
