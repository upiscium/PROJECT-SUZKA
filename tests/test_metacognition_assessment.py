"""Deterministic U3 assessment math over real sealed source observations."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from suzka.attention.adapters import (
    project_global_emotion,
    project_goal,
    project_working_memory,
)
from suzka.attention.common import ATTENTION_FIXED_POINT_SCALE
from suzka.attention.contracts import AttentionEvent
from suzka.attention.system import AttentionSystem
from suzka.belief.records import (
    BeliefEpistemicStatus,
    BeliefEvidence,
    BeliefEvidenceType,
    BeliefLifecycle,
    BeliefProposition,
    BeliefRecord,
    BeliefSubjectAdmission,
    BeliefSubjectAdmissionReason,
    belief_record_digest,
)
from suzka.emotion_contracts import EmotionState
from suzka.metacognition.assessment import (
    METACOGNITION_QUALITY_WEIGHTS,
    assess_metacognition,
)
from suzka.metacognition.contracts import (
    EpistemicBoundary,
    MetacognitiveReasonCode,
)
from suzka.metacognition.evidence import observe_metacognition
from suzka.motivation.common import R13Reference, R13ReferenceKind, Deadline
from suzka.motivation.goal import (
    GoalAdmissionReason,
    GoalLifecycle,
    GoalRecord,
    GoalRevisionOperation,
    GoalRevisionReason,
    GoalRevisionRecord,
    GoalSubjectAdmission,
    goal_id_for_target,
    goal_proposal_digest,
)
from suzka.working_memory_contracts import (
    WorkingMemoryDecision,
    WorkingMemoryDecisionReason,
    WorkingMemoryItem,
    WorkingMemoryRetentionReason,
    WorkingMemorySelection,
    WorkingMemorySourceKind,
    WorkingMemoryView,
    working_memory_item_id,
)


NOW = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)
_SCALE = ATTENTION_FIXED_POINT_SCALE


def _event(sequence: int, *, label: str, at: datetime | None = None) -> AttentionEvent:
    return AttentionEvent(
        event_id=f"event:{label}:{sequence}",
        event_sequence=sequence,
        occurred_at=at or NOW + timedelta(seconds=sequence),
    )


def _belief(
    label: str,
    proposition: BeliefProposition,
    confidence: float,
    *,
    context: str = "context:assessment",
    event_sequence: int,
) -> BeliefRecord:
    evidence_ref = f"claim:{label}"
    evidence = (BeliefEvidence(evidence_ref, BeliefEvidenceType.EXTERNAL_CLAIM),)
    admission = BeliefSubjectAdmission(
        proposition.proposition_digest,
        (evidence_ref,),
        f"event:belief-admit:{label}",
        event_sequence,
        BeliefSubjectAdmissionReason.SUBJECT_ENDORSEMENT,
    )
    return BeliefRecord(
        belief_id=f"belief:{label}",
        proposition=proposition,
        lifecycle=BeliefLifecycle.ADOPTED,
        epistemic_status=BeliefEpistemicStatus.ESTABLISHED,
        confidence=confidence,
        context_scope=(context,),
        evidence=evidence,
        subject_admission=admission,
    )


def _goal_for_beliefs(
    beliefs: tuple[BeliefRecord, ...], *, root_event: AttentionEvent
) -> GoalRecord:
    evidence_refs = tuple(
        sorted(
            [
                R13Reference(R13ReferenceKind.BELIEF, record.belief_id)
                for record in beliefs
            ],
            key=lambda item: (item.reference, item.kind.value),
        )
    )
    origins = (R13Reference(R13ReferenceKind.USER_REQUEST, "request:assessment"),)
    target = R13Reference(R13ReferenceKind.STATE, "state:assessment-focus")
    deadline = Deadline()
    description = "Review the current evidence"
    proposal_digest = goal_proposal_digest(
        target,
        description,
        deadline=deadline,
        origin_refs=origins,
        evidence_refs=evidence_refs,
    )
    goal_id = goal_id_for_target(target)
    created_at = root_event.occurred_at - timedelta(seconds=7)
    adopted_at = root_event.occurred_at - timedelta(seconds=5)
    genesis = GoalRevisionRecord(
        goal_id=goal_id,
        revision=0,
        operation=GoalRevisionOperation.CREATE,
        reason=GoalRevisionReason.CREATION,
        created_at=created_at,
        previous_lifecycle_state=None,
        proposal_digest=proposal_digest,
        event_id="event:goal:create:assessment",
        event_sequence=root_event.event_sequence - 2,
        evidence_refs=tuple(sorted(item.reference for item in evidence_refs)),
    )
    admission = GoalSubjectAdmission(
        goal_id=goal_id,
        proposal_digest=proposal_digest,
        evidence_refs=tuple(sorted(item.reference for item in evidence_refs)),
        event_id="event:goal:admit:assessment",
        event_sequence=root_event.event_sequence - 1,
        reason=GoalAdmissionReason.SUBJECT_ENDORSEMENT,
    )
    adopted_evidence = tuple(
        sorted((*admission.evidence_refs, admission.admission_digest))
    )
    adopted = GoalRevisionRecord(
        goal_id=goal_id,
        revision=1,
        operation=GoalRevisionOperation.ADOPT,
        reason=GoalRevisionReason.SUBJECT_ADMISSION,
        created_at=adopted_at,
        previous_lifecycle_state=GoalLifecycle.PROPOSED,
        proposal_digest=proposal_digest,
        event_id=admission.event_id,
        event_sequence=admission.event_sequence,
        evidence_refs=adopted_evidence,
        previous_revision_digest=genesis.record_digest,
    )
    return GoalRecord(
        goal_id=goal_id,
        target=target,
        description=description,
        lifecycle=GoalLifecycle.ADOPTED,
        deadline=deadline,
        origin_refs=origins,
        evidence_refs=evidence_refs,
        subject_admission=admission,
        revision=1,
        revision_history=(genesis, adopted),
    )


def _working_memory(
    count: int,
    *,
    item_capacity: int | None = None,
    projection_budget: int = 512,
) -> tuple[tuple[WorkingMemoryItem, ...], WorkingMemoryView, int]:
    capacity = item_capacity if item_capacity is not None else max(count, 1)
    revision = 1 if count else 0
    items: list[WorkingMemoryItem] = []
    selections: list[WorkingMemorySelection] = []
    decisions: list[WorkingMemoryDecision] = []
    projected_bytes = 0
    for index in range(count):
        source_id = f"assessment-item-{index:02}"
        item = WorkingMemoryItem(
            item_id=working_memory_item_id(WorkingMemorySourceKind.EPISODIC, source_id),
            source_kind=WorkingMemorySourceKind.EPISODIC,
            source_id=source_id,
            activation=0.8,
            salience=0.6,
            retention_reason=WorkingMemoryRetentionReason.RECENT,
            created_revision=revision,
            last_activated_revision=revision,
        )
        score = 0.6 * item.activation + 0.4 * item.salience
        rendered = f"bounded source row {index:02}"
        selected_bytes = len(rendered.encode("utf-8"))
        projected_bytes += selected_bytes
        items.append(item)
        selections.append(
            WorkingMemorySelection(
                item_id=item.item_id,
                source_kind=item.source_kind,
                source_id=item.source_id,
                rendered_content=rendered,
                score=score,
                reason=WorkingMemoryDecisionReason.SELECTED,
            )
        )
        decisions.append(
            WorkingMemoryDecision(
                item_id=item.item_id,
                source_kind=item.source_kind,
                source_id=item.source_id,
                selected=True,
                score=score,
                reason=WorkingMemoryDecisionReason.SELECTED,
            )
        )
    view = WorkingMemoryView(
        selected=tuple(selections),
        decisions=tuple(decisions),
        projected_bytes=projected_bytes,
        item_capacity=capacity,
        projection_max_bytes=projection_budget,
        revision=revision,
    )
    return tuple(items), view, revision


def _root_attention(
    *,
    event: AttentionEvent,
    items: tuple[WorkingMemoryItem, ...],
    view: WorkingMemoryView,
    working_memory_revision: int,
    goal: GoalRecord | None = None,
    emotion: EmotionState | None = None,
):
    projections = tuple(
        project_working_memory(
            item,
            revision=working_memory_revision,
            event=event,
            view=view,
        )
        for item in items
    )
    if goal is not None:
        projections = (*projections, project_goal(goal, event=event))
    global_emotion = (
        None if emotion is None else project_global_emotion(emotion, event=event)
    )
    return AttentionSystem().refresh(
        projections,
        event,
        global_emotion=global_emotion,
    ).snapshot


def _observe(
    *,
    event: AttentionEvent,
    attention,
    items: tuple[WorkingMemoryItem, ...] | None,
    view: WorkingMemoryView | None,
    revision: int | None,
    emotion: EmotionState | None,
    goal: GoalRecord | None = None,
    beliefs: tuple[BeliefRecord, ...] | None = None,
    context: str | None = "context:assessment",
):
    return observe_metacognition(
        event,
        attention,
        working_memory_items=items,
        working_memory_revision=revision,
        working_memory_view=view,
        emotion_state=emotion,
        focus_records=() if goal is None else (goal,),
        belief_records=beliefs,
        current_context_id=context,
    )


def test_fixed_integer_quality_and_ceiling_coverage_confidence_are_order_invariant() -> None:
    root_event = _event(20, label="root")
    beliefs = (
        _belief(
            "first",
            BeliefProposition("Measured fact one", "subject", "has", "one"),
            0.8,
            event_sequence=2,
        ),
        _belief(
            "second",
            BeliefProposition("Measured fact two", "another-subject", "has", "two"),
            0.6,
            event_sequence=3,
        ),
    )
    goal = _goal_for_beliefs(beliefs, root_event=root_event)
    items, view, revision = _working_memory(7, item_capacity=7)
    emotion = EmotionState(valence=-0.9, arousal=0.8)
    attention = _root_attention(
        event=root_event,
        items=items,
        view=view,
        working_memory_revision=revision,
        goal=goal,
        emotion=emotion,
    )
    before_attention = attention.canonical_bytes()
    before_source_digests = tuple(belief_record_digest(record) for record in beliefs)
    before_goal_digest = goal.record_digest

    first_observation = _observe(
        event=root_event,
        attention=attention,
        items=items,
        view=view,
        revision=revision,
        emotion=emotion,
        goal=goal,
        beliefs=beliefs,
    )
    reordered_observation = _observe(
        event=root_event,
        attention=attention,
        items=items,
        view=view,
        revision=revision,
        emotion=emotion,
        goal=goal,
        beliefs=tuple(reversed(beliefs)),
    )
    first = assess_metacognition(first_observation)
    repeat = assess_metacognition(first_observation)
    reordered = assess_metacognition(reordered_observation)

    load_units = first_observation.cognitive_load_units
    saturation_units = first_observation.attention_saturation_units
    emotion_units = first_observation.emotion_influence_units
    assert load_units == _SCALE
    assert saturation_units == _SCALE
    assert emotion_units == 900_000
    assert load_units is not None
    assert saturation_units is not None
    assert emotion_units is not None
    assert first_observation.focus_count == 8

    load_weight, saturation_weight, emotion_weight = METACOGNITION_QUALITY_WEIGHTS
    expected_quality_units = _SCALE - (
        (
            load_weight * load_units
            + saturation_weight * saturation_units
            + emotion_weight * emotion_units
        )
        // sum(METACOGNITION_QUALITY_WEIGHTS)
    )
    assert first.cognitive_quality == expected_quality_units / _SCALE
    assert first.epistemic_boundary is EpistemicBoundary.UNCERTAIN
    assert first.epistemic_boundary is not EpistemicBoundary.SUFFICIENT
    assert first.focus_witness.event == root_event
    assert first.focus_witness.attention_revision == attention.revision
    assert first.focus_witness.attention_state_digest == attention.state_digest
    assert first.focus_witness.focused_ids == attention.focused_ids
    assert goal.revision_history[-1].event_sequence < root_event.event_sequence
    coverage_units = first_observation.belief_coverage_units
    assert coverage_units is not None
    assert first.evidence_sufficiency == coverage_units / _SCALE

    ceiling_units = first_observation.belief_confidence_ceiling_units
    assert ceiling_units is not None
    numerator, denominator = min(record.confidence for record in beliefs).as_integer_ratio()
    assert ceiling_units == numerator * _SCALE // denominator
    expected_confidence_units = (
        ceiling_units * coverage_units * expected_quality_units
    ) // (_SCALE**2)
    assert first.confidence == expected_confidence_units / _SCALE
    assert first.confidence is not None and first.confidence <= min(
        record.confidence for record in beliefs
    )
    assert MetacognitiveReasonCode.LIMITED_PROVENANCE in first.reason_codes
    assert MetacognitiveReasonCode.EVIDENCE_BOUNDARY in first.reason_codes
    assert first.reason_codes == tuple(
        sorted(set(first.reason_codes), key=lambda code: code.value)
    )
    assert len(first.reason_codes) <= 16
    assert first.canonical_bytes() == repeat.canonical_bytes()
    assert first.canonical_bytes() == reordered.canonical_bytes()
    assert first_observation.observation_digest == reordered_observation.observation_digest
    with pytest.raises(TypeError):
        assess_metacognition(first_observation, model_confidence=0.99)  # type: ignore[call-arg]

    assert attention.canonical_bytes() == before_attention
    assert tuple(belief_record_digest(record) for record in beliefs) == before_source_digests
    assert goal.record_digest == before_goal_digest


def test_complete_empty_working_memory_is_measured_zero_but_missing_view_is_unknown() -> None:
    root_event = _event(12, label="empty-wm-root")
    belief = _belief(
        "empty-view",
        BeliefProposition("The source is recorded", "subject", "has", "source"),
        0.7,
        event_sequence=2,
    )
    beliefs = (belief,)
    goal = _goal_for_beliefs(beliefs, root_event=root_event)
    items, complete_view, revision = _working_memory(0, item_capacity=8)
    emotion = EmotionState(valence=0.0, arousal=0.1)
    attention = _root_attention(
        event=root_event,
        items=items,
        view=complete_view,
        working_memory_revision=revision,
        goal=goal,
        emotion=emotion,
    )

    complete = _observe(
        event=root_event,
        attention=attention,
        items=items,
        view=complete_view,
        revision=revision,
        emotion=emotion,
        goal=goal,
        beliefs=beliefs,
    )
    missing = _observe(
        event=root_event,
        attention=attention,
        items=items,
        view=None,
        revision=revision,
        emotion=emotion,
        goal=goal,
        beliefs=beliefs,
    )
    complete_assessment = assess_metacognition(complete)
    missing_assessment = assess_metacognition(missing)

    assert complete.cognitive_load_units == 0
    assert missing.cognitive_load_units is None
    assert complete_assessment.cognitive_load == 0.0
    assert missing_assessment.cognitive_load is None
    assert complete_assessment.cognitive_quality is not None
    assert missing_assessment.cognitive_quality is None
    assert complete_assessment.epistemic_boundary is EpistemicBoundary.UNCERTAIN
    assert missing_assessment.epistemic_boundary is EpistemicBoundary.UNCERTAIN
    assert complete_assessment.canonical_bytes() != missing_assessment.canonical_bytes()
    assert complete_assessment.assessment_digest != missing_assessment.assessment_digest


def test_no_evidence_is_unknown_with_no_focus_or_numeric_values() -> None:
    event = _event(1, label="no-focus")
    items, view, revision = _working_memory(0, item_capacity=4)
    attention = _root_attention(
        event=event,
        items=items,
        view=view,
        working_memory_revision=revision,
    )
    observation = _observe(
        event=event,
        attention=attention,
        items=None,
        view=None,
        revision=None,
        emotion=None,
        beliefs=None,
        context=None,
    )
    result = assess_metacognition(observation)

    assert observation.focus_count == 0
    assert observation.belief_coverage_units is None
    assert observation.belief_confidence_ceiling_units is None
    assert result.epistemic_boundary is EpistemicBoundary.UNKNOWN
    assert result.evidence_sufficiency is None
    assert result.confidence is None
    assert result.cognitive_load is None
    assert result.attention_saturation is None
    assert result.emotion_influence is None
    assert result.cognitive_quality is None
    assert result.reason_codes.count(MetacognitiveReasonCode.MISSING_EVIDENCE) == 1


def test_no_focus_keeps_coverage_and_confidence_undefined_with_observed_resources() -> None:
    event = _event(3, label="no-focus-resources")
    items, view, revision = _working_memory(0, item_capacity=4)
    emotion = EmotionState(valence=0.0, arousal=0.1)
    attention = _root_attention(
        event=event,
        items=items,
        view=view,
        working_memory_revision=revision,
        emotion=emotion,
    )
    observation = _observe(
        event=event,
        attention=attention,
        items=items,
        view=view,
        revision=revision,
        emotion=emotion,
        beliefs=None,
    )
    result = assess_metacognition(observation)

    assert observation.focus_count == 0
    assert observation.belief_coverage_units is None
    assert result.epistemic_boundary is EpistemicBoundary.UNCERTAIN
    assert result.evidence_sufficiency is None
    assert result.confidence is None
    assert result.cognitive_load == 0.0
    assert result.attention_saturation == 0.0
    assert result.emotion_influence == 0.1
    assert result.cognitive_quality is not None


def test_resource_only_assessment_never_claims_factual_confidence() -> None:
    root_event = _event(8, label="resource-only-root")
    belief = _belief(
        "not-supplied",
        BeliefProposition("A recorded source exists", "subject", "has", "source"),
        0.9,
        event_sequence=2,
    )
    goal = _goal_for_beliefs((belief,), root_event=root_event)
    items, view, revision = _working_memory(0, item_capacity=4)
    emotion = EmotionState(valence=0.0, arousal=0.1)
    attention = _root_attention(
        event=root_event,
        items=items,
        view=view,
        working_memory_revision=revision,
        goal=goal,
        emotion=emotion,
    )
    observation = _observe(
        event=root_event,
        attention=attention,
        items=items,
        view=view,
        revision=revision,
        emotion=emotion,
        goal=goal,
        beliefs=None,
    )
    result = assess_metacognition(observation)

    assert result.epistemic_boundary is EpistemicBoundary.UNCERTAIN
    assert result.cognitive_quality is not None
    assert result.confidence is None
    assert MetacognitiveReasonCode.MISSING_EVIDENCE in result.reason_codes


def test_only_typed_current_structured_belief_conflicts_yield_contradiction() -> None:
    root_event = _event(14, label="conflict-root")
    context = "context:assessment"
    left = _belief(
        "conflict-left",
        BeliefProposition("The surface is warm", "surface", "temperature", "warm"),
        0.8,
        context=context,
        event_sequence=2,
    )
    right = _belief(
        "conflict-right",
        BeliefProposition("The surface is cool", "surface", "temperature", "cool"),
        0.7,
        context=context,
        event_sequence=3,
    )
    support = _belief(
        "conflict-support",
        BeliefProposition(
            "The thermometer is in the room",
            "thermometer",
            "located",
            "room",
        ),
        0.9,
        context=context,
        event_sequence=4,
    )
    beliefs = (left, right, support)
    goal = _goal_for_beliefs(beliefs, root_event=root_event)
    items, view, revision = _working_memory(0, item_capacity=4)
    emotion = EmotionState(valence=0.0, arousal=0.1)
    attention = _root_attention(
        event=root_event,
        items=items,
        view=view,
        working_memory_revision=revision,
        goal=goal,
        emotion=emotion,
    )
    observation = _observe(
        event=root_event,
        attention=attention,
        items=items,
        view=view,
        revision=revision,
        emotion=emotion,
        goal=goal,
        beliefs=beliefs,
        context=context,
    )
    result = assess_metacognition(observation)

    assert observation.contradictory is True
    assert observation.focus_count == 1
    assert observation.belief_coverage_units is not None
    assert observation.belief_confidence_ceiling_units is not None
    assert observation.cognitive_load_units is not None
    assert observation.attention_saturation_units is not None
    assert observation.emotion_influence_units is not None
    assert any(
        witness.confidence_ceiling is not None
        for witness in observation.evidence
    )
    assert result.epistemic_boundary is EpistemicBoundary.CONTRADICTORY
    assert result.evidence_sufficiency == 0.0
    assert result.confidence == 0.0
    assert MetacognitiveReasonCode.CONTRADICTORY_EVIDENCE in result.reason_codes
    assert MetacognitiveReasonCode.CONFLICTING_SOURCES in result.reason_codes

    no_conflict_observation = _observe(
        event=root_event,
        attention=attention,
        items=items,
        view=view,
        revision=revision,
        emotion=emotion,
        goal=goal,
        beliefs=(left,),
        context=context,
    )
    no_conflict = assess_metacognition(no_conflict_observation)
    assert no_conflict_observation.contradictory is False
    assert no_conflict.epistemic_boundary is EpistemicBoundary.UNCERTAIN
    # The other declared links were not observed. Do not treat their absence
    # as measured zero coverage or manufacture a numeric confidence.
    assert no_conflict_observation.belief_coverage_units is None
    assert no_conflict.confidence is None
    assert MetacognitiveReasonCode.UNOBSERVED_SOURCE in no_conflict.reason_codes
    assert MetacognitiveReasonCode.CONTRADICTORY_EVIDENCE not in no_conflict.reason_codes
