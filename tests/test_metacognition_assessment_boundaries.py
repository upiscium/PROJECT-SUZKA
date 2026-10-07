"""U3 metacognition source, privacy, freshness, and non-authority boundaries."""

from __future__ import annotations

from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta
import inspect
from pathlib import Path
import subprocess
import sys

import pytest

from suzka.attention.adapters import project_goal, project_working_memory
from suzka.attention.bounds import maximum_attention_continuity_fixture
from suzka.attention.common import (
    ATTENTION_HIGH_AROUSAL_THRESHOLD,
    ATTENTION_MAX_FOCUS,
    AttentionSourceKind,
    SourceDigestKind,
)
from suzka.attention.contracts import AttentionContinuity, AttentionEvent
from suzka.attention.system import AttentionSystem
from suzka.belief.records import (
    BELIEF_MAX_EVIDENCE,
    BeliefEpistemicStatus,
    BeliefEvidence,
    BeliefEvidenceType,
    BeliefLifecycle,
    BeliefProposition,
    BeliefRecord,
    BeliefRevisionOperation,
    BeliefRevisionReason,
    BeliefRevisionRecord,
    BeliefSubjectAdmission,
    BeliefSubjectAdmissionReason,
    belief_record_digest,
)
from suzka.belief.system import BeliefSystem, belief_id_for_proposition
from suzka.emotion_contracts import EmotionState
from suzka.metacognition.assessment import assess_metacognition
from suzka.metacognition.contracts import (
    EpistemicBoundary,
    EvidenceCondition,
    METACOGNITION_MAX_ASSESSMENT_BYTES,
    METACOGNITION_MAX_REASON_CODES,
    MetacognitiveEvidenceWitness,
    MetacognitiveReasonCode,
    SourceEventOrigin,
)
from suzka.metacognition.evidence import (
    METACOGNITION_UNITS_SCALE,
    observe_metacognition,
)
from suzka.motivation.common import R13Reference, R13ReferenceKind
from suzka.motivation.goal import (
    GoalAdmissionReason,
    GoalRecord,
    GoalRevisionOperation,
    GoalRevisionReason,
    GoalRevisionRecord,
    GoalSubjectAdmission,
    goal_id_for_target,
    goal_proposal_digest,
)
from suzka.motivation.goal_system import GoalMutationEvidence, GoalSystem
from suzka.working_memory_contracts import (
    MAX_PROJECTION_BYTES,
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


def _event(
    sequence: int,
    *,
    event_id: str | None = None,
    seconds: int | None = None,
) -> AttentionEvent:
    offset = sequence if seconds is None else seconds
    return AttentionEvent(
        event_id or f"metacognition:event:{sequence}",
        sequence,
        NOW + timedelta(seconds=offset),
    )


def _working_memory_case(
    event: AttentionEvent,
    *,
    rendered_content: str,
    focus_records: tuple[GoalRecord, ...] = (),
) -> tuple[WorkingMemoryItem, WorkingMemoryView, AttentionSystem]:
    item = WorkingMemoryItem(
        item_id=working_memory_item_id(
            WorkingMemorySourceKind.EPISODIC, "metacognition-source-item"
        ),
        source_kind=WorkingMemorySourceKind.EPISODIC,
        source_id="metacognition-source-item",
        activation=0.8,
        salience=0.6,
        retention_reason=WorkingMemoryRetentionReason.RECENT,
        created_revision=1,
        last_activated_revision=1,
    )
    score = 0.6 * item.activation + 0.4 * item.salience
    selection = WorkingMemorySelection(
        item_id=item.item_id,
        source_kind=item.source_kind,
        source_id=item.source_id,
        rendered_content=rendered_content,
        score=score,
        reason=WorkingMemoryDecisionReason.SELECTED,
    )
    decision = WorkingMemoryDecision(
        item_id=item.item_id,
        source_kind=item.source_kind,
        source_id=item.source_id,
        selected=True,
        score=score,
        reason=WorkingMemoryDecisionReason.SELECTED,
    )
    view = WorkingMemoryView(
        selected=(selection,),
        decisions=(decision,),
        projected_bytes=len(rendered_content.encode("utf-8")),
        item_capacity=1,
        projection_max_bytes=MAX_PROJECTION_BYTES,
        revision=1,
    )
    projection = project_working_memory(
        item,
        revision=1,
        event=event,
        view=view,
    )
    goal_projections = tuple(
        project_goal(record, event=event) for record in focus_records
    )
    attention = AttentionSystem()
    result = attention.refresh((projection, *goal_projections), event)
    expected_focus = {projection.candidate_id} | {
        item.candidate_id for item in goal_projections
    }
    assert expected_focus.issubset(result.snapshot.focused_ids)
    return item, view, attention


def _active_goal(
    index: int,
    *,
    belief_ids: tuple[str, ...] = (),
    target_belief_id: str | None = None,
) -> GoalRecord:
    target = (
        R13Reference(R13ReferenceKind.BELIEF, target_belief_id)
        if target_belief_id is not None
        else R13Reference(R13ReferenceKind.VALUE, f"value:metacognition-goal:{index}")
    )
    origins = (R13Reference(R13ReferenceKind.USER_REQUEST, f"request:goal:{index}"),)
    evidence_refs = (
        tuple(
            sorted(
                (R13Reference(R13ReferenceKind.BELIEF, item) for item in belief_ids),
                key=lambda item: (item.reference, item.kind.value),
            )
        )
        if belief_ids
        else (R13Reference(R13ReferenceKind.EVENT, f"event:goal-evidence:{index}"),)
    )
    description = f"Adopted metacognition test goal {index}"
    proposal_digest = goal_proposal_digest(
        target,
        description,
        origin_refs=origins,
        evidence_refs=evidence_refs,
    )
    goal_id = goal_id_for_target(target)
    genesis_event = _event(1, event_id="goal:genesis:shared", seconds=0)
    genesis = GoalRevisionRecord(
        goal_id=goal_id,
        revision=0,
        operation=GoalRevisionOperation.CREATE,
        reason=GoalRevisionReason.CREATION,
        created_at=genesis_event.occurred_at,
        previous_lifecycle_state=None,
        proposal_digest=proposal_digest,
        event_id=genesis_event.event_id,
        event_sequence=genesis_event.event_sequence,
        evidence_refs=tuple(sorted(item.reference for item in evidence_refs)),
    )
    proposal = GoalRecord(
        goal_id=goal_id,
        target=target,
        description=description,
        origin_refs=origins,
        evidence_refs=evidence_refs,
        revision_history=(genesis,),
    )
    system = GoalSystem()
    system.ingest_proposal(
        proposal,
        GoalMutationEvidence(
            event_id=genesis_event.event_id,
            event_sequence=genesis_event.event_sequence,
            recorded_at=genesis_event.occurred_at,
            evidence_refs=evidence_refs,
        ),
    )
    admission_event = _event(2, event_id="goal:admission:shared", seconds=1)
    admission = GoalSubjectAdmission(
        goal_id=goal_id,
        proposal_digest=proposal.proposal_digest,
        evidence_refs=tuple(sorted(item.reference for item in evidence_refs)),
        event_id=admission_event.event_id,
        event_sequence=admission_event.event_sequence,
        reason=GoalAdmissionReason.SUBJECT_ENDORSEMENT,
    )
    return system.adopt(
        goal_id,
        admission,
        GoalMutationEvidence(
            event_id=admission_event.event_id,
            event_sequence=admission_event.event_sequence,
            recorded_at=admission_event.occurred_at,
            evidence_refs=evidence_refs,
        ),
    )


def _focused_goals(
    event: AttentionEvent,
    goals: tuple[GoalRecord, ...],
) -> tuple[AttentionSystem, AttentionContinuity]:
    projections = tuple(project_goal(record, event=event) for record in goals)
    attention = AttentionSystem()
    result = attention.refresh(projections, event)
    assert {item.candidate_id for item in projections}.issubset(
        result.snapshot.focused_ids
    )
    return attention, result.snapshot


def _active_belief(
    event: AttentionEvent,
    index: int,
    *,
    context_id: str = "context:assessment",
    subject: str | None = None,
    predicate: str = "predicate:state",
    object_value: str | None = None,
) -> BeliefRecord:
    evidence_ref = f"event:belief-evidence:{index}"
    proposition = BeliefProposition(
        f"structured belief proposition {index}",
        subject=subject or f"subject:{index}",
        predicate=predicate,
        object=object_value or f"object:{index}",
    )
    admission = BeliefSubjectAdmission(
        proposition_digest=proposition.proposition_digest,
        evidence_refs=(evidence_ref,),
        event_id=event.event_id,
        event_sequence=event.event_sequence,
        reason=BeliefSubjectAdmissionReason.SUBJECT_ENDORSEMENT,
    )
    creation_revision = BeliefRevisionRecord(
        belief_id=belief_id_for_proposition(proposition.proposition_digest),
        revision=0,
        operation=BeliefRevisionOperation.CREATE,
        reason=BeliefRevisionReason.CREATION,
        created_at=NOW,
        # Goal and Belief rows were created by this one shared global event.
        event_id="goal:genesis:shared",
        event_sequence=1,
        evidence_refs=(evidence_ref,),
    )
    belief_id = creation_revision.belief_id
    adoption_revision = BeliefRevisionRecord(
        belief_id=belief_id,
        revision=1,
        operation=BeliefRevisionOperation.ADOPT,
        reason=BeliefRevisionReason.SUBJECT_ADMISSION,
        created_at=event.occurred_at,
        event_id=event.event_id,
        event_sequence=event.event_sequence,
        evidence_refs=tuple(
            sorted((*admission.evidence_refs, admission.admission_digest))
        ),
        previous_revision_digest=creation_revision.record_digest,
    )
    record = BeliefRecord(
        belief_id=belief_id,
        proposition=proposition,
        lifecycle=BeliefLifecycle.ADOPTED,
        epistemic_status=BeliefEpistemicStatus.PROBABLE,
        confidence=0.7,
        context_scope=(context_id,),
        evidence=(BeliefEvidence(evidence_ref, BeliefEvidenceType.EXPERIENCE),),
        subject_admission=admission,
        revision=1,
        revision_history=(creation_revision, adoption_revision),
    )
    # Keep fixtures aligned with actual R12 identity, history, and lifecycle
    # checks without invoking its runtime-event mutation boundary.
    BeliefSystem((record,)).validate()
    return record


def _proposed_private_belief(text: str) -> BeliefRecord:
    return BeliefRecord(
        belief_id="belief:private-proposal",
        proposition=BeliefProposition(
            text,
            subject="private:subject",
            predicate="private:predicate",
            object="private:object",
        ),
        lifecycle=BeliefLifecycle.PROPOSED,
        epistemic_status=BeliefEpistemicStatus.UNKNOWN,
        confidence=0.5,
    )


def _assert_assessment_numeric_fields_none(assessment: object) -> None:
    for name in (
        "evidence_sufficiency",
        "confidence",
        "cognitive_load",
        "attention_saturation",
        "emotion_influence",
        "cognitive_quality",
    ):
        assert getattr(assessment, name) is None


def test_observation_is_sealed_event_bound_and_detects_checksum_tampering() -> None:
    event = _event(1)
    attention = AttentionContinuity.bootstrap()
    observed = observe_metacognition(event, attention)

    assert observed.event == event
    assert observed.focus_witness.event == event
    assert observed.focus_witness.attention_revision == attention.revision == 0
    assert observed.focus_witness.attention_state_digest == attention.state_digest
    assert observed.focus_witness.focused_ids == attention.focused_ids == ()
    assert observed.focus_count == 0
    assert observed.validated_copy() == observed
    assert observed.validated_copy() is not observed
    assert not hasattr(observed, "__dict__")
    assert {item.name for item in fields(observed)} == {
        "event",
        "focus_witness",
        "evidence",
        "focus_count",
        "cognitive_load_units",
        "attention_saturation_units",
        "emotion_influence_units",
        "belief_coverage_units",
        "belief_confidence_ceiling_units",
        "contradictory",
        "reason_codes",
        "observation_digest",
    }
    with pytest.raises((TypeError, ValueError)):
        type(observed)()

    original_digest = observed.observation_digest
    object.__setattr__(observed, "observation_digest", "0" * 64)
    with pytest.raises((TypeError, ValueError), match="digest|checksum|observation"):
        observed.validated_copy()
    assert observed.observation_digest == "0" * 64
    assert original_digest != observed.observation_digest


def test_observer_rejects_untyped_inputs_and_untrusted_overrides() -> None:
    event = _event(1)
    attention = AttentionContinuity.bootstrap()

    expected_parameters = (
        "event",
        "attention",
        "working_memory_items",
        "working_memory_revision",
        "working_memory_view",
        "emotion_state",
        "focus_records",
        "belief_records",
        "current_context_id",
    )
    signature = inspect.signature(observe_metacognition)
    assert tuple(signature.parameters) == expected_parameters
    assert all(
        parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for parameter in signature.parameters.values()
    )

    for bad_inputs in (
        {"event": {"event_id": event.event_id}},
        {"attention": {"revision": 0}},
        {"working_memory_items": []},
        {"working_memory_revision": True},
        {"working_memory_view": {}},
        {"emotion_state": {"arousal": 0.5}},
        {"focus_records": []},
        {"belief_records": []},
    ):
        with pytest.raises((TypeError, ValueError)):
            observe_metacognition(event, attention, **bad_inputs)

    forged_event = _event(1)
    object.__setattr__(forged_event, "event_sequence", True)
    with pytest.raises((TypeError, ValueError)):
        observe_metacognition(forged_event, attention)

    forged_emotion = EmotionState(arousal=0.5)
    object.__setattr__(forged_emotion, "arousal", float("nan"))
    with pytest.raises((TypeError, ValueError)):
        observe_metacognition(event, attention, emotion_state=forged_emotion)

    item, view, wm_attention = _working_memory_case(
        event,
        rendered_content="temporary source content",
    )
    malformed_item = replace(item, activation=True)
    with pytest.raises((TypeError, ValueError)):
        observe_metacognition(
            event,
            wm_attention.snapshot(),
            working_memory_items=(malformed_item,),
            working_memory_revision=view.revision,
            working_memory_view=view,
        )

    for unsupported in (
        "confidence",
        "confidence_ceiling",
        "cognitive_load",
        "evidence",
        "source_witness",
        "model_self_report",
        "value_records",
        "experience_records",
    ):
        with pytest.raises(TypeError):
            observe_metacognition(
                event,
                attention,
                **{unsupported: 0.9},
            )

    fabricated_witness = MetacognitiveEvidenceWitness(
        source_kind=AttentionSourceKind.EMOTION,
        reference="fabricated-source",
        source_revision=None,
        digest="a" * 64,
        digest_kind=SourceDigestKind.UPSTREAM_AUTHORITY,
        condition=EvidenceCondition.SUPPORTING,
        confidence_ceiling=1.0,
    )
    with pytest.raises((TypeError, ValueError)):
        observe_metacognition(
            event,
            attention,
            belief_records=(fabricated_witness,),  # type: ignore[arg-type]
        )


def test_attention_event_freshness_and_sequence_identity_are_enforced() -> None:
    first = _event(1)
    second = _event(2)
    attention = AttentionSystem()
    attention.refresh((), first)
    latest = attention.refresh((), second).snapshot
    original = latest.canonical_bytes()

    later_observation = observe_metacognition(_event(3), latest)
    assert later_observation.focus_witness.attention_revision == latest.revision
    assert later_observation.focus_witness.attention_state_digest == latest.state_digest
    assert later_observation.focus_witness.focused_ids == latest.focused_ids

    mismatched_same_sequence = _event(2, event_id="metacognition:different-id")
    same_id_different_time = AttentionEvent(
        second.event_id,
        second.event_sequence,
        second.occurred_at + timedelta(microseconds=1),
    )
    regressing_time = _event(3, seconds=1)
    for invalid_event in (
        first,
        mismatched_same_sequence,
        same_id_different_time,
        regressing_time,
    ):
        with pytest.raises((TypeError, ValueError)):
            observe_metacognition(invalid_event, latest)

    assert latest.canonical_bytes() == original


def test_source_confidence_uses_only_current_active_belief_records() -> None:
    event = _event(10)
    active = _active_belief(event, 1)
    goal = _active_goal(
        1,
        belief_ids=(active.belief_id,),
        target_belief_id=active.belief_id,
    )
    _, attention = _focused_goals(event, (goal,))
    observation = observe_metacognition(
        event,
        attention,
        focus_records=(goal,),
        belief_records=(active,),
        current_context_id="context:assessment",
    )
    belief_witnesses = tuple(
        item
        for item in observation.evidence
        if item.source_kind is AttentionSourceKind.BELIEF
    )
    assert belief_witnesses
    assert all(item.digest == belief_record_digest(active) for item in belief_witnesses)
    assert all(item.source_revision == active.revision for item in belief_witnesses)
    assert all(item.confidence_ceiling is not None for item in belief_witnesses)
    assert all(
        item.confidence_ceiling <= active.confidence for item in belief_witnesses
    )
    assert all(
        item.reference != active.proposition.canonical_text
        for item in belief_witnesses
    )
    assert all(item.source_event == event for item in belief_witnesses)
    assert all(
        item.source_event_origin is SourceEventOrigin.UPSTREAM_EVENT
        for item in belief_witnesses
    )
    assessment = assess_metacognition(observation)
    assert assessment.epistemic_boundary is EpistemicBoundary.UNCERTAIN
    assert assessment.epistemic_boundary is not EpistemicBoundary.SUFFICIENT
    assert assessment.evidence_sufficiency == 1.0
    assert assessment.confidence is None

    future_event = _event(11)
    future_record = _active_belief(future_event, 2)
    future_goal = _active_goal(
        2,
        belief_ids=(future_record.belief_id,),
        target_belief_id=future_record.belief_id,
    )
    _, future_attention = _focused_goals(event, (future_goal,))
    with pytest.raises((TypeError, ValueError), match="future|event|source"):
        observe_metacognition(
            event,
            future_attention,
            focus_records=(future_goal,),
            belief_records=(future_record,),
            current_context_id="context:assessment",
        )

    proposed = _proposed_private_belief("unadopted belief has no confidence authority")
    proposed_goal = _active_goal(
        3,
        belief_ids=(proposed.belief_id,),
        target_belief_id=proposed.belief_id,
    )
    _, proposed_attention = _focused_goals(event, (proposed_goal,))
    unsupported = observe_metacognition(
        event,
        proposed_attention,
        focus_records=(proposed_goal,),
        belief_records=(proposed,),
        current_context_id="context:assessment",
    )
    assert len(unsupported.evidence) == 1
    assert unsupported.evidence[0].condition is EvidenceCondition.UNKNOWN
    assert all(
        item.source_kind is not AttentionSourceKind.BELIEF
        or item.confidence_ceiling is None
        for item in unsupported.evidence
    )
    assert unsupported.belief_coverage_units is None
    unknown_assessment = assess_metacognition(unsupported)
    assert unknown_assessment.epistemic_boundary is EpistemicBoundary.UNKNOWN
    _assert_assessment_numeric_fields_none(unknown_assessment)

    outside_scope = observe_metacognition(
        event,
        attention,
        focus_records=(goal,),
        belief_records=(active,),
        current_context_id="context:unrelated",
    )
    scoped_witness = next(
        item
        for item in outside_scope.evidence
        if item.source_kind is AttentionSourceKind.BELIEF
    )
    assert scoped_witness.condition is EvidenceCondition.UNKNOWN
    assert scoped_witness.confidence_ceiling is None
    assert outside_scope.belief_confidence_ceiling_units is None
    assert outside_scope.belief_coverage_units is None
    outside_scope_assessment = assess_metacognition(outside_scope)
    assert outside_scope_assessment.epistemic_boundary is EpistemicBoundary.UNKNOWN
    _assert_assessment_numeric_fields_none(outside_scope_assessment)

    missing = observe_metacognition(
        event,
        proposed_attention,
        focus_records=(proposed_goal,),
        belief_records=(),
        current_context_id="context:assessment",
    )
    assert missing.belief_coverage_units is None
    assert MetacognitiveReasonCode.UNOBSERVED_SOURCE in missing.reason_codes
    missing_assessment = assess_metacognition(missing)
    assert missing_assessment.epistemic_boundary is EpistemicBoundary.UNKNOWN
    _assert_assessment_numeric_fields_none(missing_assessment)


def test_observation_copy_rejects_tampered_source_witness_without_repair() -> None:
    event = _event(1)
    observed = observe_metacognition(
        event,
        AttentionContinuity.bootstrap(),
        emotion_state=EmotionState(valence=0.2, arousal=0.3, optimal_loss=0.8),
    )
    emotion_witnesses = tuple(
        item
        for item in observed.evidence
        if item.source_kind is AttentionSourceKind.EMOTION
    )
    assert emotion_witnesses
    assert all(item.confidence_ceiling is None for item in emotion_witnesses)

    forged = emotion_witnesses[0]
    object.__setattr__(forged, "confidence_ceiling", 0.25)
    with pytest.raises(
        (TypeError, ValueError),
        match="confidence|source|evidence|digest",
    ):
        observed.validated_copy()
    assert forged.confidence_ceiling == 0.25


def test_source_text_is_not_retained_or_scored_and_assessment_is_pure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = _event(10)
    rendered_secret = "PRIVATE-WM-RENDERED-SENTINEL-9f3a"
    proposition_secret = "PRIVATE-BELIEF-PROMPT-SENTINEL-4c2d"
    proposed = _proposed_private_belief(proposition_secret)
    goal = _active_goal(
        30,
        belief_ids=(proposed.belief_id,),
        target_belief_id=proposed.belief_id,
    )
    item, view, attention = _working_memory_case(
        event,
        rendered_content=rendered_secret,
        focus_records=(goal,),
    )
    emotion = EmotionState(valence=0.1, arousal=0.9, optimal_loss=0.4)
    before_snapshot = attention.snapshot()
    before_view = attention.selected_view()
    before_attention_bytes = before_snapshot.canonical_bytes()
    before_belief_digest = belief_record_digest(proposed)

    def forbid_attention_policy(*args: object, **kwargs: object) -> object:
        pytest.fail("metacognition attempted to rerank or refresh Attention")

    import suzka.attention.system as attention_system_module

    monkeypatch.setattr(
        attention_system_module,
        "compete_attention",
        forbid_attention_policy,
    )
    monkeypatch.setattr(
        attention_system_module,
        "select_attention_prompt",
        forbid_attention_policy,
    )
    monkeypatch.setattr(AttentionSystem, "refresh", forbid_attention_policy)

    observation = observe_metacognition(
        event,
        before_snapshot,
        working_memory_items=(item,),
        working_memory_revision=view.revision,
        working_memory_view=view,
        emotion_state=emotion,
        focus_records=(goal,),
        belief_records=(proposed,),
        current_context_id="context:private-scope",
    )
    assessment = assess_metacognition(observation)
    public_output = repr(observation) + assessment.canonical_bytes().decode("ascii")

    assert rendered_secret not in public_output
    assert proposition_secret not in public_output
    assert all(
        rendered_secret not in witness.reference
        and proposition_secret not in witness.reference
        for witness in observation.evidence
    )
    assert observation.contradictory is False
    assert assessment.epistemic_boundary is EpistemicBoundary.UNCERTAIN
    assert attention.snapshot() == before_snapshot
    assert attention.selected_view() == before_view
    assert attention.snapshot().canonical_bytes() == before_attention_bytes
    assert view.selected[0].rendered_content == rendered_secret
    assert belief_record_digest(proposed) == before_belief_digest
    assert emotion == EmotionState(valence=0.1, arousal=0.9, optimal_loss=0.4)


def test_combined_source_evidence_overflow_is_explicit_not_truncated() -> None:
    event = _event(10)
    records = tuple(
        _active_belief(event, index)
        for index in range(BELIEF_MAX_EVIDENCE)
    )
    assert len(records) == BELIEF_MAX_EVIDENCE
    first_goal = _active_goal(
        40,
        belief_ids=tuple(item.belief_id for item in records[:16]),
    )
    second_goal = _active_goal(
        41,
        belief_ids=tuple(item.belief_id for item in records[16:]),
    )
    goals = (first_goal, second_goal)
    _, attention = _focused_goals(event, goals)

    complete_belief_batch = observe_metacognition(
        event,
        attention,
        focus_records=goals,
        belief_records=records,
        current_context_id="context:assessment",
    )
    assert len(complete_belief_batch.evidence) == BELIEF_MAX_EVIDENCE
    assert complete_belief_batch.belief_coverage_units == METACOGNITION_UNITS_SCALE
    assert (
        assess_metacognition(complete_belief_batch).epistemic_boundary
        is EpistemicBoundary.UNCERTAIN
    )

    with pytest.raises(
        (OverflowError, ValueError),
        match="capacity|bound|evidence|witness",
    ):
        observe_metacognition(
            event,
            attention,
            emotion_state=EmotionState(arousal=0.5),
            focus_records=goals,
            belief_records=records,
            current_context_id="context:assessment",
        )


def test_structured_conflict_without_support_has_no_confidence_ceiling() -> None:
    event = _event(10)
    left = _active_belief(
        event,
        1,
        subject="subject:shared",
        predicate="predicate:shared",
        object_value="object:left",
    )
    right = _active_belief(
        event,
        2,
        subject="subject:shared",
        predicate="predicate:shared",
        object_value="object:right",
    )
    goal = _active_goal(
        50,
        belief_ids=(left.belief_id, right.belief_id),
        target_belief_id=left.belief_id,
    )
    _, attention = _focused_goals(event, (goal,))

    observation = observe_metacognition(
        event,
        attention,
        focus_records=(goal,),
        belief_records=(left, right),
        current_context_id="context:assessment",
    )
    belief_witnesses = tuple(
        item
        for item in observation.evidence
        if item.source_kind is AttentionSourceKind.BELIEF
    )
    assert observation.contradictory is True
    assert len(belief_witnesses) == 2
    assert all(
        item.condition is EvidenceCondition.CONTRADICTORY
        for item in belief_witnesses
    )
    assert all(item.confidence_ceiling is None for item in belief_witnesses)
    assert observation.belief_confidence_ceiling_units is None

    assessment = assess_metacognition(observation)
    assert assessment.epistemic_boundary is EpistemicBoundary.CONTRADICTORY
    assert assessment.confidence is None


def test_partial_coverage_counts_only_resolved_linked_focal_targets() -> None:
    event = _event(10)
    active = _active_belief(event, 3)
    linked_goal = _active_goal(
        60,
        belief_ids=(active.belief_id,),
        target_belief_id=active.belief_id,
    )
    unlinked_goal = _active_goal(61)
    goals = (linked_goal, unlinked_goal)
    item, view, attention = _working_memory_case(
        event,
        rendered_content="temporary bounded working-memory row",
        focus_records=goals,
    )

    observation = observe_metacognition(
        event,
        attention.snapshot(),
        working_memory_items=(item,),
        working_memory_revision=view.revision,
        working_memory_view=view,
        focus_records=goals,
        belief_records=(active,),
        emotion_state=EmotionState(valence=0.1, arousal=0.2),
        current_context_id="context:assessment",
    )
    assert observation.focus_count == 3
    assert observation.belief_coverage_units == METACOGNITION_UNITS_SCALE // 3

    assessment = assess_metacognition(observation)
    assert assessment.epistemic_boundary is EpistemicBoundary.UNCERTAIN
    assert assessment.epistemic_boundary is not EpistemicBoundary.SUFFICIENT
    assert assessment.evidence_sufficiency == (
        METACOGNITION_UNITS_SCALE // 3
    ) / METACOGNITION_UNITS_SCALE
    assert MetacognitiveReasonCode.PARTIAL_COVERAGE in assessment.reason_codes
    load = observation.cognitive_load_units
    saturation = observation.attention_saturation_units
    emotion = observation.emotion_influence_units
    ceiling = observation.belief_confidence_ceiling_units
    coverage = observation.belief_coverage_units
    assert load is not None and saturation is not None and emotion is not None
    assert ceiling is not None and coverage is not None
    quality = METACOGNITION_UNITS_SCALE - (
        400 * load + 350 * saturation + 250 * emotion
    ) // 1000
    expected_units = ceiling * coverage * quality // METACOGNITION_UNITS_SCALE**2
    assert assessment.cognitive_quality == quality / METACOGNITION_UNITS_SCALE
    assert assessment.confidence == expected_units / METACOGNITION_UNITS_SCALE
    assert (
        assessment.confidence is not None
        and assessment.confidence <= active.confidence
    )


def test_arousal_threshold_changes_only_observed_focus_capacity() -> None:
    event = _event(10)
    goals = tuple(_active_goal(70 + index) for index in range(8))
    _, attention = _focused_goals(event, goals)
    assert len(attention.focused_ids) == 8

    absent_emotion = observe_metacognition(
        event,
        attention,
        working_memory_items=(),
        working_memory_revision=0,
        focus_records=goals,
    )
    assert absent_emotion.attention_saturation_units == METACOGNITION_UNITS_SCALE // 2
    assert absent_emotion.emotion_influence_units is None
    assert all(
        item.source_kind is not AttentionSourceKind.EMOTION
        for item in absent_emotion.evidence
    )

    just_below_threshold = observe_metacognition(
        event,
        attention,
        emotion_state=EmotionState(arousal=ATTENTION_HIGH_AROUSAL_THRESHOLD - 0.000001),
        focus_records=goals,
    )
    at_threshold = observe_metacognition(
        event,
        attention,
        emotion_state=EmotionState(arousal=ATTENTION_HIGH_AROUSAL_THRESHOLD),
        focus_records=goals,
    )
    assert just_below_threshold.focus_count == at_threshold.focus_count == 8
    assert (
        just_below_threshold.attention_saturation_units
        == METACOGNITION_UNITS_SCALE // 2
    )
    assert at_threshold.attention_saturation_units == METACOGNITION_UNITS_SCALE
    assert just_below_threshold.emotion_influence_units is not None
    assert at_threshold.emotion_influence_units == 750_000
    assert (
        just_below_threshold.emotion_influence_units
        < at_threshold.emotion_influence_units
    )


def test_all_unknown_nonempty_source_witnesses_assess_to_all_unknown_metrics() -> None:
    event = _event(10)
    proposed = _proposed_private_belief("proposed but not ordinarily active")
    goal = _active_goal(
        90,
        belief_ids=(proposed.belief_id,),
        target_belief_id=proposed.belief_id,
    )
    _, attention = _focused_goals(event, (goal,))
    observation = observe_metacognition(
        event,
        attention,
        focus_records=(goal,),
        belief_records=(proposed,),
        current_context_id="context:assessment",
    )
    assert observation.evidence
    assert all(
        item.condition is EvidenceCondition.UNKNOWN for item in observation.evidence
    )

    assessment = assess_metacognition(observation)
    assert assessment.epistemic_boundary is EpistemicBoundary.UNKNOWN
    _assert_assessment_numeric_fields_none(assessment)


def test_maximum_attention_root_and_restore_remain_unchanged_by_explicit_assessment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    maximum = maximum_attention_continuity_fixture()
    assert maximum.last_event is not None
    assert len(maximum.candidates) == 4_192
    assert len(maximum.focused_ids) == ATTENTION_MAX_FOCUS
    expected_bytes = maximum.canonical_bytes()
    restored = AttentionSystem(maximum)
    restored_snapshot = restored.snapshot()
    restored_view = restored.selected_view()
    assert restored_view.competition is None
    assert restored_view.prompt is None

    def forbid_attention_policy(*args: object, **kwargs: object) -> object:
        pytest.fail("restoration or assessment invoked Attention policy")

    import suzka.attention.system as attention_system_module

    monkeypatch.setattr(
        attention_system_module,
        "compete_attention",
        forbid_attention_policy,
    )
    monkeypatch.setattr(
        attention_system_module,
        "select_attention_prompt",
        forbid_attention_policy,
    )
    monkeypatch.setattr(AttentionSystem, "refresh", forbid_attention_policy)

    observation = observe_metacognition(maximum.last_event, restored_snapshot)
    assessment = assess_metacognition(observation)
    assert assessment.event == maximum.last_event
    assert assessment.epistemic_boundary is EpistemicBoundary.UNKNOWN
    assert assessment.confidence is None
    assert assessment.evidence_sufficiency is None
    assert assessment.cognitive_load is None
    assert assessment.attention_saturation is None
    assert assessment.emotion_influence is None
    assert assessment.cognitive_quality is None
    assert MetacognitiveReasonCode.MISSING_EVIDENCE in assessment.reason_codes
    assert len(assessment.reason_codes) <= METACOGNITION_MAX_REASON_CODES
    assert observation.focus_witness.attention_revision == maximum.revision
    assert observation.focus_witness.attention_state_digest == maximum.state_digest
    assert observation.focus_witness.focused_ids == maximum.focused_ids
    assert observation.focus_count == ATTENTION_MAX_FOCUS

    encoded = assessment.canonical_bytes()
    assert len(encoded) <= METACOGNITION_MAX_ASSESSMENT_BYTES
    assert type(assessment).from_json(encoded).canonical_bytes() == encoded
    assert maximum.canonical_bytes() == expected_bytes
    assert restored.snapshot() == restored_snapshot
    assert restored.snapshot().canonical_bytes() == expected_bytes
    assert restored.selected_view() == restored_view
    assert restored.selected_view().competition is None
    assert restored.selected_view().prompt is None
    assert {item.name for item in fields(restored_snapshot)} == {
        "schema_version",
        "policy_version",
        "revision",
        "last_event",
        "candidates",
        "focused_ids",
        "unfinished_ids",
        "revision_history",
        "receipts",
        "revision_anchor",
        "receipt_anchor",
    }

    # Fresh evidence is observed and assessed only by this explicit U3 call;
    # it does not populate a durable assessment view on the restored owner.
    explicit = observe_metacognition(
        maximum.last_event,
        restored_snapshot,
        emotion_state=EmotionState(arousal=0.2),
    )
    assess_metacognition(explicit)
    assert restored.snapshot() == restored_snapshot
    assert restored.selected_view() == restored_view
    assert restored.selected_view().competition is None
    assert restored.selected_view().prompt is None


def test_fresh_import_of_metacognition_modules_is_dependency_neutral() -> None:
    code = """
import importlib
import sys

for module in (
    'suzka.metacognition',
    'suzka.metacognition.evidence',
    'suzka.metacognition.assessment',
):
    importlib.import_module(module)

for prefix in (
    'suzka.runtime', 'suzka.memory', 'suzka.cognition', 'suzka.models',
    'suzka.provider', 'suzka.providers', 'suzka.scheduler', 'suzka.body',
    'torch', 'transformers',
):
    assert not any(
        name == prefix or name.startswith(prefix + '.')
        for name in sys.modules
    ), prefix
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
