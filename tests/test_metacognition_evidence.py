"""U3 event-scoped typed metacognition observation tests."""

from __future__ import annotations

from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
import subprocess
import sys

import pytest

from suzka.attention.adapters import (
    project_commitment,
    project_goal,
    project_motivation,
    project_working_memory,
)
from suzka.attention.common import AttentionSourceKind, SourceDigestKind
from suzka.attention.contracts import AttentionContinuity, AttentionEvent
from suzka.attention.system import AttentionSystem
from suzka.belief.records import (
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
)
from suzka.emotion_contracts import EmotionState
from suzka.metacognition.contracts import (
    EvidenceCondition,
    MetacognitiveReasonCode,
)
from suzka.metacognition.evidence import (
    METACOGNITION_MAX_BELIEF_RECORDS,
    METACOGNITION_MAX_OBSERVATION_BYTES,
    MetacognitionObservation,
    derive_metacognition_observation_max_bytes,
    observe_metacognition,
)
from suzka.motivation.commitment import (
    CommitmentAdmissionReason,
    CommitmentLifecycle,
    CommitmentRecord,
    CommitmentRevisionOperation,
    CommitmentRevisionReason,
    CommitmentRevisionRecord,
    CommitmentSubjectAdmission,
    commitment_id_for_proposal,
    commitment_proposal_digest,
)
from suzka.motivation.common import Deadline, R13Reference, R13ReferenceKind
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
from suzka.motivation.motivation import (
    MotivationKind,
    MotivationLifecycle,
    MotivationRecord,
    MotivationRevisionOperation,
    MotivationRevisionReason,
    MotivationRevisionRecord,
    motivation_id_for_target,
    motivation_state_digest,
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


def _event(
    event_id: str = "observation-event",
    sequence: int = 1,
    occurred_at: datetime = NOW,
) -> AttentionEvent:
    return AttentionEvent(event_id, sequence, occurred_at)


def _working_memory_item(
    *,
    revision: int = 1,
    activation: float = 0.5,
    source_id: str = "episode-one",
) -> WorkingMemoryItem:
    source_kind = WorkingMemorySourceKind.EPISODIC
    return WorkingMemoryItem(
        item_id=working_memory_item_id(source_kind, source_id),
        source_kind=source_kind,
        source_id=source_id,
        activation=activation,
        salience=0.5,
        retention_reason=WorkingMemoryRetentionReason.RECENT,
        created_revision=revision,
        last_activated_revision=revision,
    )


def _working_memory_view(
    item: WorkingMemoryItem,
    *,
    revision: int = 1,
    content: str = "rendered-private-source",
    item_capacity: int = 2,
    projection_max_bytes: int = 100,
) -> WorkingMemoryView:
    score = 0.6 * item.activation + 0.4 * item.salience
    decision = WorkingMemoryDecision(
        item_id=item.item_id,
        source_kind=item.source_kind,
        source_id=item.source_id,
        selected=True,
        score=score,
        reason=WorkingMemoryDecisionReason.SELECTED,
    )
    selection = WorkingMemorySelection(
        item_id=item.item_id,
        source_kind=item.source_kind,
        source_id=item.source_id,
        rendered_content=content,
        score=score,
        reason=WorkingMemoryDecisionReason.SELECTED,
    )
    return WorkingMemoryView(
        selected=(selection,),
        decisions=(decision,),
        projected_bytes=len(content.encode("utf-8")),
        item_capacity=item_capacity,
        projection_max_bytes=projection_max_bytes,
        revision=revision,
    )


def _focused_working_memory(
    event: AttentionEvent,
    *,
    item: WorkingMemoryItem | None = None,
    revision: int = 1,
    view: WorkingMemoryView | None = None,
) -> tuple[WorkingMemoryItem, AttentionContinuity]:
    current = _working_memory_item(revision=revision) if item is None else item
    projection = project_working_memory(
        current,
        revision=revision,
        event=event,
        view=view,
    )
    attention = AttentionSystem().refresh((projection,), event).snapshot
    return current, attention


def _belief(
    belief_id: str,
    *,
    confidence: float = 0.75,
    lifecycle: BeliefLifecycle = BeliefLifecycle.ADOPTED,
    epistemic_status: BeliefEpistemicStatus = BeliefEpistemicStatus.PROBABLE,
    context_scope: tuple[str, ...] = (),
    valid_from: datetime | None = None,
    structured: tuple[str, str, str] | None = None,
    text: str | None = None,
    revision: BeliefRevisionRecord | None = None,
) -> BeliefRecord:
    if structured is None:
        proposition = BeliefProposition(
            text or f"proposition for {belief_id}"
        )
    else:
        subject, predicate, object_value = structured
        proposition = BeliefProposition(
            text or f"{subject} {predicate} {object_value}",
            subject=subject,
            predicate=predicate,
            object=object_value,
        )
    evidence_ref = f"experience-{belief_id}"
    evidence = (BeliefEvidence(evidence_ref, BeliefEvidenceType.EXPERIENCE),)
    admission = None
    if lifecycle is BeliefLifecycle.ADOPTED:
        admission = BeliefSubjectAdmission(
            proposition_digest=proposition.proposition_digest,
            evidence_refs=(evidence_ref,),
            event_id="belief-admission-shared",
            event_sequence=1,
            reason=BeliefSubjectAdmissionReason.SUBJECT_ENDORSEMENT,
        )
    history = () if revision is None else (revision,)
    return BeliefRecord(
        belief_id=belief_id,
        proposition=proposition,
        lifecycle=lifecycle,
        epistemic_status=epistemic_status,
        confidence=confidence,
        context_scope=context_scope,
        valid_from=valid_from,
        evidence=evidence,
        subject_admission=admission,
        revision=0,
        revision_history=history,
    )


def _proposed_belief_with_source_event(
    belief_id: str,
    *,
    event_id: str,
    event_sequence: int,
    created_at: datetime,
) -> BeliefRecord:
    proposition = BeliefProposition(f"proposed proposition for {belief_id}")
    evidence_ref = f"experience-{belief_id}"
    revision = BeliefRevisionRecord(
        belief_id=belief_id,
        revision=0,
        operation=BeliefRevisionOperation.CREATE,
        reason=BeliefRevisionReason.CREATION,
        created_at=created_at,
        event_id=event_id,
        event_sequence=event_sequence,
        evidence_refs=(evidence_ref,),
    )
    return BeliefRecord(
        belief_id=belief_id,
        proposition=proposition,
        lifecycle=BeliefLifecycle.PROPOSED,
        epistemic_status=BeliefEpistemicStatus.UNKNOWN,
        confidence=0.5,
        evidence=(BeliefEvidence(evidence_ref, BeliefEvidenceType.EXPERIENCE),),
        revision_history=(revision,),
    )


def _adopted_belief_with_event_regression(
    belief_id: str,
    *,
    create_event_sequence: int,
    create_event_time: datetime,
    create_event_id: str = "belief-create-event",
    adoption_event_id: str,
    adoption_event_sequence: int,
    adoption_event_time: datetime,
    admission_event_id: str | None = None,
    admission_event_sequence: int | None = None,
) -> BeliefRecord:
    proposition = BeliefProposition(f"adopted proposition for {belief_id}")
    evidence_ref = f"experience-{belief_id}"
    evidence = (BeliefEvidence(evidence_ref, BeliefEvidenceType.EXPERIENCE),)
    current_admission_event_id = (
        adoption_event_id if admission_event_id is None else admission_event_id
    )
    current_admission_event_sequence = (
        adoption_event_sequence
        if admission_event_sequence is None
        else admission_event_sequence
    )
    admission = BeliefSubjectAdmission(
        proposition_digest=proposition.proposition_digest,
        evidence_refs=(evidence_ref,),
        event_id=current_admission_event_id,
        event_sequence=current_admission_event_sequence,
        reason=BeliefSubjectAdmissionReason.SUBJECT_ENDORSEMENT,
    )
    creation = BeliefRevisionRecord(
        belief_id=belief_id,
        revision=0,
        operation=BeliefRevisionOperation.CREATE,
        reason=BeliefRevisionReason.CREATION,
        created_at=create_event_time,
        event_id=create_event_id,
        event_sequence=create_event_sequence,
        evidence_refs=(evidence_ref,),
    )
    adoption = BeliefRevisionRecord(
        belief_id=belief_id,
        revision=1,
        operation=BeliefRevisionOperation.ADOPT,
        reason=BeliefRevisionReason.SUBJECT_ADMISSION,
        created_at=adoption_event_time,
        previous_revision_digest=creation.record_digest,
        event_id=adoption_event_id,
        event_sequence=adoption_event_sequence,
        evidence_refs=tuple(sorted((evidence_ref, admission.admission_digest))),
    )
    return BeliefRecord(
        belief_id=belief_id,
        proposition=proposition,
        lifecycle=BeliefLifecycle.ADOPTED,
        epistemic_status=BeliefEpistemicStatus.PROBABLE,
        confidence=0.75,
        evidence=evidence,
        subject_admission=admission,
        revision=1,
        revision_history=(creation, adoption),
    )


def _focused_motivation(
    event: AttentionEvent,
    belief_ids: tuple[str, ...],
    *,
    use_belief_as_source: bool = False,
    related_belief_ids: tuple[str, ...] = (),
) -> tuple[MotivationRecord, AttentionContinuity]:
    belief_refs = tuple(
        R13Reference(R13ReferenceKind.BELIEF, belief_id)
        for belief_id in belief_ids
    )
    source = (
        belief_refs[0]
        if use_belief_as_source and belief_refs
        else R13Reference(R13ReferenceKind.EXPERIENCE, "experience-motivation")
    )
    evidence_refs = tuple(
        sorted(
            set((source, *belief_refs)),
            key=lambda item: (item.reference, item.kind.value),
        )
    )
    related_refs = tuple(
        R13Reference(R13ReferenceKind.BELIEF, belief_id)
        for belief_id in related_belief_ids
    )
    target = belief_refs[0] if belief_refs else R13Reference(
        R13ReferenceKind.VALUE, "value-one"
    )
    motivation_id = motivation_id_for_target(MotivationKind.INTEREST, target)
    lifecycle = MotivationLifecycle.ACTIVE
    state_digest = motivation_state_digest(
        motivation_id=motivation_id,
        kind=MotivationKind.INTEREST,
        target=target,
        lifecycle=lifecycle,
        source_evidence=source,
        evidence_refs=evidence_refs,
        related_refs=related_refs,
    )
    revision = MotivationRevisionRecord(
        motivation_id=motivation_id,
        revision=0,
        operation=MotivationRevisionOperation.CREATE,
        reason=MotivationRevisionReason.CREATION,
        created_at=event.occurred_at,
        previous_lifecycle_state=None,
        state_digest=state_digest,
        event_id=event.event_id,
        event_sequence=event.event_sequence,
        evidence_refs=tuple(sorted(item.reference for item in evidence_refs)),
    )
    record = MotivationRecord(
        motivation_id=motivation_id,
        kind=MotivationKind.INTEREST,
        target=target,
        lifecycle=lifecycle,
        source_evidence=source,
        evidence_refs=evidence_refs,
        related_refs=related_refs,
        revision=0,
        revision_history=(revision,),
    )
    projected = project_motivation(record, event=event)
    attention = AttentionSystem().refresh((projected,), event).snapshot
    return record, attention


def _focused_goal(
    event: AttentionEvent,
    belief_id: str,
) -> tuple[GoalRecord, AttentionContinuity]:
    target = R13Reference(R13ReferenceKind.BELIEF, belief_id)
    origin = R13Reference(R13ReferenceKind.USER_REQUEST, f"request-{belief_id}")
    evidence_refs = (target,)
    deadline = Deadline()
    description = f"review linked belief {belief_id}"
    goal_id = goal_id_for_target(target)
    proposal_digest = goal_proposal_digest(
        target,
        description,
        deadline=deadline,
        origin_refs=(origin,),
        evidence_refs=evidence_refs,
    )
    genesis = GoalRevisionRecord(
        goal_id=goal_id,
        revision=0,
        operation=GoalRevisionOperation.CREATE,
        reason=GoalRevisionReason.CREATION,
        created_at=event.occurred_at - timedelta(seconds=1),
        previous_lifecycle_state=None,
        proposal_digest=proposal_digest,
        event_id=f"create-{belief_id}",
        event_sequence=event.event_sequence - 1,
        evidence_refs=("goal-genesis",),
    )
    admission = GoalSubjectAdmission(
        goal_id=goal_id,
        proposal_digest=proposal_digest,
        evidence_refs=(belief_id,),
        event_id=event.event_id,
        event_sequence=event.event_sequence,
        reason=GoalAdmissionReason.SUBJECT_ENDORSEMENT,
    )
    adopted = GoalRevisionRecord(
        goal_id=goal_id,
        revision=1,
        operation=GoalRevisionOperation.ADOPT,
        reason=GoalRevisionReason.SUBJECT_ADMISSION,
        created_at=event.occurred_at,
        previous_lifecycle_state=GoalLifecycle.PROPOSED,
        proposal_digest=proposal_digest,
        event_id=event.event_id,
        event_sequence=event.event_sequence,
        evidence_refs=tuple(sorted((*admission.evidence_refs, admission.admission_digest))),
        previous_revision_digest=genesis.record_digest,
    )
    record = GoalRecord(
        goal_id=goal_id,
        target=target,
        description=description,
        lifecycle=GoalLifecycle.ADOPTED,
        deadline=deadline,
        origin_refs=(origin,),
        evidence_refs=evidence_refs,
        subject_admission=admission,
        revision=1,
        revision_history=(genesis, adopted),
    )
    attention = AttentionSystem().refresh((project_goal(record, event=event),), event).snapshot
    return record, attention


def _focused_commitment(
    event: AttentionEvent,
    belief_id: str,
) -> tuple[CommitmentRecord, AttentionContinuity]:
    beneficiary = R13Reference(R13ReferenceKind.EXTERNAL_PARTY, "party-beneficiary")
    scope = ("scope-review-belief",)
    deadline = Deadline()
    evidence_refs = (R13Reference(R13ReferenceKind.BELIEF, belief_id),)
    origin_refs = (R13Reference(R13ReferenceKind.USER_REQUEST, f"request-{belief_id}"),)
    subject = f"review linked belief {belief_id}"
    proposal_digest = commitment_proposal_digest(
        subject=subject,
        beneficiary=beneficiary,
        scope=scope,
        deadline=deadline,
        evidence_refs=evidence_refs,
        origin_refs=origin_refs,
    )
    commitment_id = commitment_id_for_proposal(proposal_digest)
    genesis = CommitmentRevisionRecord(
        commitment_id=commitment_id,
        revision=0,
        operation=CommitmentRevisionOperation.CREATE,
        reason=CommitmentRevisionReason.CREATION,
        created_at=event.occurred_at - timedelta(seconds=1),
        previous_lifecycle_state=None,
        event_id=f"create-{belief_id}",
        event_sequence=event.event_sequence - 1,
        evidence_refs=(belief_id,),
    )
    admission = CommitmentSubjectAdmission(
        commitment_id=commitment_id,
        proposal_digest=proposal_digest,
        beneficiary=beneficiary,
        scope=scope,
        deadline=deadline,
        evidence_refs=(belief_id,),
        event_id=event.event_id,
        event_sequence=event.event_sequence,
        reason=CommitmentAdmissionReason.SUBJECT_ENDORSEMENT,
    )
    active = CommitmentRevisionRecord(
        commitment_id=commitment_id,
        revision=1,
        operation=CommitmentRevisionOperation.ADMIT,
        reason=CommitmentRevisionReason.SUBJECT_ADMISSION,
        created_at=event.occurred_at,
        previous_lifecycle_state=CommitmentLifecycle.PROPOSED,
        event_id=event.event_id,
        event_sequence=event.event_sequence,
        evidence_refs=tuple(sorted((*admission.evidence_refs, admission.admission_digest))),
        previous_revision_digest=genesis.record_digest,
    )
    record = CommitmentRecord(
        subject=subject,
        beneficiary=beneficiary,
        scope=scope,
        deadline=deadline,
        evidence_refs=evidence_refs,
        origin_refs=origin_refs,
        lifecycle=CommitmentLifecycle.ACTIVE,
        subject_admission=admission,
        revision=1,
        revision_history=(genesis, active),
    )
    attention = AttentionSystem().refresh((project_commitment(record, event=event),), event).snapshot
    return record, attention


def _believed_witnesses(observation: MetacognitionObservation):
    return tuple(
        item
        for item in observation.evidence
        if item.source_kind is AttentionSourceKind.BELIEF
    )


def test_empty_observation_is_sealed_unknown_and_event_bound() -> None:
    event = _event()
    observation = observe_metacognition(event, AttentionContinuity.bootstrap())

    assert observation.event == event
    assert observation.focus_witness.event == event
    assert observation.focus_count == 0
    assert observation.evidence == ()
    assert observation.cognitive_load_units is None
    assert observation.attention_saturation_units is None
    assert observation.emotion_influence_units is None
    assert observation.belief_coverage_units is None
    assert observation.belief_confidence_ceiling_units is None
    assert MetacognitionObservation.validated_copy(observation) == observation
    assert not hasattr(observation, "__dict__")
    assert {item.name for item in fields(observation)} == {
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
    assert b"rendered-private-source" not in observation.canonical_bytes()
    with pytest.raises(TypeError, match="created by observe_metacognition"):
        MetacognitionObservation(event=event)  # type: ignore[call-arg]


def test_missing_working_memory_view_and_complete_empty_view_are_distinct() -> None:
    event = _event()
    attention = AttentionContinuity.bootstrap()
    missing = observe_metacognition(
        event,
        attention,
        working_memory_items=(),
        working_memory_revision=0,
    )
    empty_view = WorkingMemoryView(
        selected=(),
        decisions=(),
        projected_bytes=0,
        item_capacity=4,
        projection_max_bytes=128,
        revision=0,
    )
    measured = observe_metacognition(
        event,
        attention,
        working_memory_items=(),
        working_memory_revision=0,
        working_memory_view=empty_view,
    )

    assert missing.cognitive_load_units is None
    assert measured.cognitive_load_units == 0
    assert missing.attention_saturation_units == measured.attention_saturation_units == 0
    assert any(
        code is MetacognitiveReasonCode.LOAD_UNOBSERVED
        for code in missing.reason_codes
    )
    assert all(
        item.source_kind is AttentionSourceKind.WORKING_MEMORY
        and item.condition is EvidenceCondition.SUPPORTING
        and item.confidence_ceiling is None
        for item in measured.evidence
    )


def test_emotion_zero_is_observed_while_absence_is_unknown() -> None:
    event = _event()
    attention = AttentionContinuity.bootstrap()
    missing = observe_metacognition(event, attention)
    zero = observe_metacognition(
        event,
        attention,
        emotion_state=EmotionState(valence=0.0, arousal=0.0, optimal_loss=-7.5),
    )

    assert missing.emotion_influence_units is None
    assert zero.emotion_influence_units == 0
    assert zero.attention_saturation_units == 0
    emotion = next(item for item in zero.evidence if item.source_kind is AttentionSourceKind.EMOTION)
    assert emotion.reference == "current-emotion"
    assert emotion.source_revision is None
    assert emotion.digest_kind is SourceDigestKind.ATTENTION_PROJECTION
    assert emotion.condition is EvidenceCondition.SUPPORTING
    assert emotion.confidence_ceiling is None


def test_current_working_memory_join_derives_load_without_retaining_rendered_content() -> None:
    event = _event()
    item = _working_memory_item()
    view = _working_memory_view(item, content="private rendered payload")
    item, attention = _focused_working_memory(event, item=item, view=view)

    unavailable = observe_metacognition(
        event,
        attention,
        working_memory_items=(item,),
        working_memory_revision=1,
    )
    measured = observe_metacognition(
        event,
        attention,
        working_memory_items=(item,),
        working_memory_revision=1,
        working_memory_view=view,
    )

    assert unavailable.focus_count == 1
    assert unavailable.cognitive_load_units is None
    assert measured.cognitive_load_units == 500_000
    assert measured.attention_saturation_units == 62_500
    assert b"private rendered payload" not in measured.canonical_bytes()
    assert "private rendered payload" not in repr(measured)

    changed_item = replace(item, activation=0.9)
    with pytest.raises(ValueError, match="stale relative to current input"):
        observe_metacognition(
            event,
            attention,
            working_memory_items=(changed_item,),
            working_memory_revision=1,
        )

    newer_item = item
    newer_view = _working_memory_view(newer_item, revision=2)
    _, newer_attention = _focused_working_memory(
        event,
        item=newer_item,
        revision=2,
        view=newer_view,
    )
    with pytest.raises(ValueError, match="stale relative to current input"):
        observe_metacognition(
            event,
            newer_attention,
            working_memory_items=(newer_item,),
            working_memory_revision=1,
        )


def test_working_memory_view_order_is_canonical_and_rendered_bytes_are_preflighted() -> None:
    event = _event("working-memory-view-order")
    items = (
        _working_memory_item(source_id="episode-alpha"),
        _working_memory_item(source_id="episode-beta"),
    )
    content_by_id = {
        items[0].item_id: "alpha",
        items[1].item_id: "beta",
    }

    def view_for(ordered_items: tuple[WorkingMemoryItem, ...]) -> WorkingMemoryView:
        decisions = tuple(
            WorkingMemoryDecision(
                item_id=item.item_id,
                source_kind=item.source_kind,
                source_id=item.source_id,
                selected=True,
                score=0.5,
                reason=WorkingMemoryDecisionReason.SELECTED,
            )
            for item in ordered_items
        )
        selections = tuple(
            WorkingMemorySelection(
                item_id=item.item_id,
                source_kind=item.source_kind,
                source_id=item.source_id,
                rendered_content=content_by_id[item.item_id],
                score=0.5,
                reason=WorkingMemoryDecisionReason.SELECTED,
            )
            for item in ordered_items
        )
        return WorkingMemoryView(
            selected=selections,
            decisions=decisions,
            projected_bytes=sum(len(item.rendered_content.encode("utf-8")) for item in selections),
            item_capacity=4,
            projection_max_bytes=100,
            revision=1,
        )

    first = observe_metacognition(
        event,
        AttentionContinuity.bootstrap(),
        working_memory_items=items,
        working_memory_revision=1,
        working_memory_view=view_for(items),
    )
    reversed_items = tuple(reversed(items))
    reversed_view = view_for(reversed_items)
    second = observe_metacognition(
        event,
        AttentionContinuity.bootstrap(),
        working_memory_items=reversed_items,
        working_memory_revision=1,
        working_memory_view=reversed_view,
    )
    assert first.evidence[0].witness_digest == second.evidence[0].witness_digest
    assert first.observation_digest == second.observation_digest

    oversized_content = "x" * 100_000
    item = items[0]
    oversized_view = WorkingMemoryView(
        selected=(
            WorkingMemorySelection(
                item_id=item.item_id,
                source_kind=item.source_kind,
                source_id=item.source_id,
                rendered_content=oversized_content,
                score=0.5,
                reason=WorkingMemoryDecisionReason.SELECTED,
            ),
        ),
        decisions=(
            WorkingMemoryDecision(
                item_id=item.item_id,
                source_kind=item.source_kind,
                source_id=item.source_id,
                selected=True,
                score=0.5,
                reason=WorkingMemoryDecisionReason.SELECTED,
            ),
        ),
        projected_bytes=0,
        item_capacity=1,
        projection_max_bytes=100,
        revision=1,
    )
    with pytest.raises(ValueError, match="exceeds projection_max_bytes"):
        observe_metacognition(
            event,
            AttentionContinuity.bootstrap(),
            working_memory_items=(item,),
            working_memory_revision=1,
            working_memory_view=oversized_view,
        )


def test_attention_event_fences_reject_future_and_reused_sequence_identity() -> None:
    root_event = _event("attention-current", 3, NOW + timedelta(seconds=3))
    item = _working_memory_item()
    _, attention = _focused_working_memory(root_event, item=item)

    with pytest.raises(ValueError, match="last_event is in the future"):
        observe_metacognition(_event("older", 2, NOW + timedelta(seconds=2)), attention)
    with pytest.raises(ValueError, match="exact same event"):
        observe_metacognition(
            _event("different-at-same-sequence", 3, root_event.occurred_at),
            attention,
        )
    with pytest.raises(ValueError, match="last_event time is in the future"):
        observe_metacognition(
            _event("new-sequence-regressed-time", 4, NOW + timedelta(seconds=2)),
            attention,
        )
    newer = _event("newer-observation", 4, NOW + timedelta(seconds=4))
    accepted = observe_metacognition(newer, attention)
    assert accepted.event == newer
    assert accepted.focus_witness.event == newer


def test_reused_event_ids_are_fenced_against_attention_belief_and_admission_sources() -> None:
    old_source_event = _event("shared-r13-source-event", 2, NOW + timedelta(seconds=2))
    focus_record, _ = _focused_motivation(old_source_event, ())
    later_attention_event = _event("attention-later", 4, NOW + timedelta(seconds=4))
    attention = AttentionSystem().refresh(
        (project_motivation(focus_record, event=later_attention_event),),
        later_attention_event,
    ).snapshot
    with pytest.raises(ValueError, match="conflicts with retained Attention event identity"):
        observe_metacognition(
            _event(old_source_event.event_id, 5, NOW + timedelta(seconds=5)),
            attention,
        )

    current = _event("belief-current", 2, NOW + timedelta(seconds=2))
    record, attention = _focused_motivation(current, ("belief-source-id-reuse",))
    old_belief_source = _proposed_belief_with_source_event(
        "belief-source-id-reuse",
        event_id="shared-belief-source-event",
        event_sequence=1,
        created_at=NOW + timedelta(seconds=1),
    )
    with pytest.raises(ValueError, match="reused event IDs"):
        observe_metacognition(
            _event("shared-belief-source-event", 3, NOW + timedelta(seconds=3)),
            attention,
            focus_records=(record,),
            belief_records=(old_belief_source,),
        )

    admission_record, admission_attention = _focused_motivation(
        current,
        ("belief-admission-id-reuse",),
    )
    with pytest.raises(ValueError, match="admission event ID was reused"):
        observe_metacognition(
            _event("belief-admission-shared", 3, NOW + timedelta(seconds=3)),
            admission_attention,
            focus_records=(admission_record,),
            belief_records=(_belief("belief-admission-id-reuse"),),
        )


def test_belief_source_collisions_reject_conflicting_triples_but_allow_exact_shared_events() -> None:
    event = _event("belief-collision-current", 2, NOW + timedelta(seconds=2))
    focus_record, attention = _focused_motivation(
        event,
        ("belief-collision-left", "belief-collision-right"),
    )
    left = _proposed_belief_with_source_event(
        "belief-collision-left",
        event_id="shared-belief-create",
        event_sequence=1,
        created_at=NOW + timedelta(seconds=1),
    )
    conflicting_right = _proposed_belief_with_source_event(
        "belief-collision-right",
        event_id="shared-belief-create",
        event_sequence=1,
        created_at=NOW + timedelta(seconds=1, microseconds=1),
    )
    with pytest.raises(ValueError, match="one event ID identifies conflicting source events"):
        observe_metacognition(
            event,
            attention,
            focus_records=(focus_record,),
            belief_records=(left, conflicting_right),
        )

    exact_shared_right = _proposed_belief_with_source_event(
        "belief-collision-right",
        event_id="shared-belief-create",
        event_sequence=1,
        created_at=NOW + timedelta(seconds=1),
    )
    accepted = observe_metacognition(
        event,
        attention,
        focus_records=(focus_record,),
        belief_records=(left, exact_shared_right),
    )
    assert len(_believed_witnesses(accepted)) == 2
    assert all(item.source_event == _believed_witnesses(accepted)[0].source_event for item in _believed_witnesses(accepted))

    sequence_event = _event("belief-sequence-collision-current", 4, NOW + timedelta(seconds=4))
    sequence_focus, sequence_attention = _focused_motivation(
        sequence_event,
        ("belief-sequence-left", "belief-sequence-right"),
    )
    sequence_left = _proposed_belief_with_source_event(
        "belief-sequence-left",
        event_id="belief-left-source-event",
        event_sequence=3,
        created_at=NOW + timedelta(seconds=3),
    )
    sequence_right = _proposed_belief_with_source_event(
        "belief-sequence-right",
        event_id="belief-right-source-event",
        event_sequence=3,
        created_at=NOW + timedelta(seconds=3),
    )
    input_before = (sequence_left.revision_history, sequence_right.revision_history)
    with pytest.raises(ValueError, match="one event sequence identifies conflicting source events"):
        observe_metacognition(
            sequence_event,
            sequence_attention,
            focus_records=(sequence_focus,),
            belief_records=(sequence_left, sequence_right),
        )
    assert input_before == (sequence_left.revision_history, sequence_right.revision_history)


def test_exact_shared_belief_admission_identity_is_allowed_but_sequence_collision_is_not() -> None:
    event = _event("admission-identity-current", 2, NOW + timedelta(seconds=2))
    focus_record, attention = _focused_motivation(
        event,
        ("belief-admission-left", "belief-admission-right"),
    )
    left = _belief("belief-admission-left")
    right = _belief("belief-admission-right")

    accepted = observe_metacognition(
        event,
        attention,
        focus_records=(focus_record,),
        belief_records=(left, right),
    )
    assert len(_believed_witnesses(accepted)) == 2

    assert right.subject_admission is not None
    conflicting_admission = replace(
        right.subject_admission,
        event_id="different-admission-at-sequence-one",
    )
    conflicting_right = replace(right, subject_admission=conflicting_admission)
    with pytest.raises(ValueError, match="Belief admission sequence identifies different event IDs"):
        observe_metacognition(
            event,
            attention,
            focus_records=(focus_record,),
            belief_records=(left, conflicting_right),
        )


def test_linked_belief_uses_only_current_typed_r13_references_and_scope_gate() -> None:
    event = _event("belief-observation", 2, NOW + timedelta(seconds=2))
    record, attention = _focused_motivation(event, ("belief-one",))
    belief = _belief("belief-one", confidence=0.73)

    active = observe_metacognition(
        event,
        attention,
        focus_records=(record,),
        belief_records=(belief,),
    )
    witness = _believed_witnesses(active)[0]
    assert witness.condition is EvidenceCondition.SUPPORTING
    assert witness.confidence_ceiling == 0.73
    assert active.belief_confidence_ceiling_units == 729_999
    assert active.belief_coverage_units == 1_000_000
    assert witness.source_event is None

    partially_supplied_record, partial_attention = _focused_motivation(
        event,
        ("belief-one", "belief-not-supplied"),
    )
    missing_link = observe_metacognition(
        event,
        partial_attention,
        focus_records=(partially_supplied_record,),
        belief_records=(belief,),
    )
    assert missing_link.belief_coverage_units is None
    assert missing_link.belief_confidence_ceiling_units == 729_999
    assert MetacognitiveReasonCode.UNOBSERVED_SOURCE in missing_link.reason_codes

    scoped = _belief(
        "belief-one",
        confidence=0.73,
        context_scope=("context-home",),
    )
    out_of_scope = observe_metacognition(
        event,
        attention,
        focus_records=(record,),
        belief_records=(scoped,),
    )
    assert _believed_witnesses(out_of_scope)[0].condition is EvidenceCondition.UNKNOWN
    assert out_of_scope.belief_confidence_ceiling_units is None

    in_scope = observe_metacognition(
        event,
        attention,
        focus_records=(record,),
        belief_records=(scoped,),
        current_context_id="context-home",
    )
    assert _believed_witnesses(in_scope)[0].condition is EvidenceCondition.SUPPORTING

    proposed = _belief(
        "belief-one",
        lifecycle=BeliefLifecycle.PROPOSED,
        epistemic_status=BeliefEpistemicStatus.UNKNOWN,
    )
    unknown = observe_metacognition(
        event,
        attention,
        focus_records=(record,),
        belief_records=(proposed,),
    )
    assert _believed_witnesses(unknown)[0].condition is EvidenceCondition.UNKNOWN
    assert _believed_witnesses(unknown)[0].confidence_ceiling is None
    assert unknown.attention_saturation_units is None
    assert unknown.belief_coverage_units is None
    assert unknown.cognitive_load_units is None
    assert unknown.emotion_influence_units is None
    assert unknown.belief_confidence_ceiling_units is None

    missing = observe_metacognition(
        event,
        attention,
        focus_records=(record,),
        belief_records=(),
    )
    assert missing.evidence == ()
    assert missing.belief_coverage_units is None
    assert missing.belief_confidence_ceiling_units is None
    assert MetacognitiveReasonCode.UNOBSERVED_SOURCE in missing.reason_codes

    with pytest.raises(ValueError, match="unrelated to focused R13 links"):
        observe_metacognition(
            event,
            attention,
            focus_records=(record,),
            belief_records=(_belief("not-linked"),),
        )
    with pytest.raises(TypeError, match="exact Motivation/Goal/Commitment"):
        observe_metacognition(event, attention, focus_records=(object(),))  # type: ignore[arg-type]


def test_known_partial_belief_coverage_uses_the_entire_current_focus_count() -> None:
    event = _event("partial-coverage", 2, NOW + timedelta(seconds=2))
    linked_record, _ = _focused_motivation(event, ("belief-partial",))
    unlinked_record, _ = _focused_motivation(event, ())
    attention = AttentionSystem().refresh(
        (
            project_motivation(linked_record, event=event),
            project_motivation(unlinked_record, event=event),
        ),
        event,
    ).snapshot
    observation = observe_metacognition(
        event,
        attention,
        focus_records=(linked_record, unlinked_record),
        belief_records=(_belief("belief-partial"),),
    )

    assert observation.focus_count == 2
    assert observation.belief_coverage_units == 500_000
    assert MetacognitiveReasonCode.PARTIAL_COVERAGE in observation.reason_codes


def test_belief_links_are_taken_from_motivation_related_goal_and_commitment_refs() -> None:
    event = _event("typed-link-observation", 2, NOW + timedelta(seconds=2))
    belief = _belief("belief-linked")
    sources = (
        _focused_motivation(event, (), related_belief_ids=("belief-linked",)),
        _focused_goal(event, "belief-linked"),
        _focused_commitment(event, "belief-linked"),
    )
    for record, attention in sources:
        observation = observe_metacognition(
            event,
            attention,
            focus_records=(record,),
            belief_records=(belief,),
        )
        witness = _believed_witnesses(observation)[0]
        assert witness.reference == "belief-linked"
        assert witness.condition is EvidenceCondition.SUPPORTING
        assert observation.belief_coverage_units == 1_000_000

    related_record, related_attention = sources[0]
    object.__setattr__(related_record, "related_refs", list(related_record.related_refs))
    with pytest.raises(TypeError, match="exact tuple"):
        observe_metacognition(
            event,
            related_attention,
            focus_records=(related_record,),
            belief_records=(belief,),
        )


def test_future_belief_revision_is_rejected_before_observation() -> None:
    event = _event("current", 2, NOW + timedelta(seconds=2))
    focus_record, attention = _focused_motivation(event, ("belief-future",))
    proposition = BeliefProposition("future revision proof")
    evidence_ref = "experience-belief-future"
    creation = BeliefRevisionRecord(
        belief_id="belief-future",
        revision=0,
        operation=BeliefRevisionOperation.CREATE,
        reason=BeliefRevisionReason.CREATION,
        created_at=NOW,
        event_id="belief-future-create",
        event_sequence=1,
        evidence_refs=(evidence_ref,),
    )
    admission = BeliefSubjectAdmission(
        proposition_digest=proposition.proposition_digest,
        evidence_refs=(evidence_ref,),
        event_id="future-belief-event",
        event_sequence=10,
        reason=BeliefSubjectAdmissionReason.SUBJECT_ENDORSEMENT,
    )
    adoption = BeliefRevisionRecord(
        belief_id="belief-future",
        revision=1,
        operation=BeliefRevisionOperation.ADOPT,
        reason=BeliefRevisionReason.SUBJECT_ADMISSION,
        created_at=NOW + timedelta(seconds=10),
        previous_revision_digest=creation.record_digest,
        event_id="future-belief-event",
        event_sequence=10,
        evidence_refs=tuple(sorted((evidence_ref, admission.admission_digest))),
    )
    belief = BeliefRecord(
        belief_id="belief-future",
        proposition=proposition,
        lifecycle=BeliefLifecycle.ADOPTED,
        epistemic_status=BeliefEpistemicStatus.PROBABLE,
        confidence=0.5,
        evidence=(BeliefEvidence(evidence_ref, BeliefEvidenceType.EXPERIENCE),),
        subject_admission=admission,
        revision=1,
        revision_history=(creation, adoption),
    )

    with pytest.raises(ValueError, match="future"):
        observe_metacognition(
            event,
            attention,
            focus_records=(focus_record,),
            belief_records=(belief,),
        )


def test_belief_retained_history_rejects_regressing_event_sequences_before_support() -> None:
    event = _event("belief-history-current", 6, NOW + timedelta(seconds=6))
    focus_record, attention = _focused_motivation(event, ("belief-history-order",))
    belief = _adopted_belief_with_event_regression(
        "belief-history-order",
        create_event_sequence=4,
        create_event_time=NOW + timedelta(seconds=4),
        adoption_event_id="belief-adoption-lower-sequence",
        adoption_event_sequence=3,
        adoption_event_time=NOW + timedelta(seconds=5),
    )

    assert belief.is_ordinary_active(at=event.occurred_at)
    with pytest.raises(ValueError, match="event sequences must be nondecreasing"):
        observe_metacognition(
            event,
            attention,
            focus_records=(focus_record,),
            belief_records=(belief,),
        )


def test_belief_create_and_adoption_may_share_only_the_exact_event() -> None:
    event = _event("belief-shared-event-current", 2, NOW + timedelta(seconds=2))
    focus_record, attention = _focused_motivation(event, ("belief-shared-event",))
    belief = _adopted_belief_with_event_regression(
        "belief-shared-event",
        create_event_sequence=1,
        create_event_time=NOW + timedelta(seconds=1),
        create_event_id="belief-shared-create-adopt",
        adoption_event_id="belief-shared-create-adopt",
        adoption_event_sequence=1,
        adoption_event_time=NOW + timedelta(seconds=1),
    )

    observation = observe_metacognition(
        event,
        attention,
        focus_records=(focus_record,),
        belief_records=(belief,),
    )
    witness = _believed_witnesses(observation)[0]
    assert witness.condition is EvidenceCondition.SUPPORTING
    assert witness.source_event == AttentionEvent(
        "belief-shared-create-adopt",
        1,
        NOW + timedelta(seconds=1),
    )


def test_retained_belief_history_must_bind_its_current_admission() -> None:
    event = _event("admission-binding-current", 5, NOW + timedelta(seconds=5))
    focus_record, attention = _focused_motivation(event, ("belief-admission-binding",))

    wrong_event_id = _adopted_belief_with_event_regression(
        "belief-admission-binding",
        create_event_sequence=1,
        create_event_time=NOW + timedelta(seconds=1),
        adoption_event_id="belief-adopted-event",
        adoption_event_sequence=2,
        adoption_event_time=NOW + timedelta(seconds=2),
        admission_event_id="other-admission-event",
        admission_event_sequence=2,
    )
    assert wrong_event_id.is_ordinary_active(at=event.occurred_at)
    with pytest.raises(ValueError, match="does not match its subject admission event"):
        observe_metacognition(
            event,
            attention,
            focus_records=(focus_record,),
            belief_records=(wrong_event_id,),
        )

    wrong_sequence = _adopted_belief_with_event_regression(
        "belief-admission-binding",
        create_event_sequence=1,
        create_event_time=NOW + timedelta(seconds=1),
        adoption_event_id="belief-adopted-event",
        adoption_event_sequence=2,
        adoption_event_time=NOW + timedelta(seconds=2),
        admission_event_id="belief-adopted-event",
        admission_event_sequence=3,
    )
    assert wrong_sequence.is_ordinary_active(at=event.occurred_at)
    with pytest.raises(ValueError, match="does not match its subject admission event"):
        observe_metacognition(
            event,
            attention,
            focus_records=(focus_record,),
            belief_records=(wrong_sequence,),
        )

    valid = _adopted_belief_with_event_regression(
        "belief-admission-binding",
        create_event_sequence=1,
        create_event_time=NOW + timedelta(seconds=1),
        adoption_event_id="belief-adopted-event",
        adoption_event_sequence=2,
        adoption_event_time=NOW + timedelta(seconds=2),
    )
    creation, adoption = valid.revision_history
    missing_admission_digest = replace(adoption, evidence_refs=("experience-belief-admission-binding",))
    unbound = replace(
        valid,
        revision_history=(creation, missing_admission_digest),
    )
    assert unbound.is_ordinary_active(at=event.occurred_at)
    with pytest.raises(ValueError, match="does not bind its subject admission"):
        observe_metacognition(
            event,
            attention,
            focus_records=(focus_record,),
            belief_records=(unbound,),
        )


def test_structured_conflict_is_distinct_from_text_difference() -> None:
    event = _event("conflict-observation", 2, NOW + timedelta(seconds=2))
    left = _belief(
        "belief-left",
        structured=("water", "boils_at", "100C"),
    )
    right = _belief(
        "belief-right",
        structured=("water", "boils_at", "90C"),
    )
    focus_record, attention = _focused_motivation(event, ("belief-left", "belief-right"))
    conflict = observe_metacognition(
        event,
        attention,
        focus_records=(focus_record,),
        belief_records=(left, right),
    )

    assert conflict.contradictory is True
    assert conflict.belief_confidence_ceiling_units is None
    assert conflict.belief_coverage_units == 1_000_000
    assert all(item.condition is EvidenceCondition.CONTRADICTORY for item in _believed_witnesses(conflict))
    assert MetacognitiveReasonCode.CONTRADICTORY_EVIDENCE in conflict.reason_codes
    assert MetacognitiveReasonCode.CONFLICTING_SOURCES in conflict.reason_codes

    unstructured_left = _belief("belief-left", text="water boils at 100C")
    unstructured_right = _belief("belief-right", text="water boils at 90C")
    unstructured = observe_metacognition(
        event,
        attention,
        focus_records=(focus_record,),
        belief_records=(unstructured_left, unstructured_right),
    )
    assert unstructured.contradictory is False
    assert all(item.condition is EvidenceCondition.SUPPORTING for item in _believed_witnesses(unstructured))


def test_witness_count_uses_the_full_bound_and_fails_one_over_without_truncation() -> None:
    event = _event("full-belief-bound", 2, NOW + timedelta(seconds=2))
    belief_ids = tuple(f"belief-{index:02d}" for index in range(METACOGNITION_MAX_BELIEF_RECORDS))
    focus_record, attention = _focused_motivation(
        event,
        belief_ids,
        use_belief_as_source=True,
    )
    beliefs = tuple(_belief(belief_id) for belief_id in belief_ids)

    full = observe_metacognition(
        event,
        attention,
        focus_records=(focus_record,),
        belief_records=beliefs,
    )
    assert len(full.evidence) == METACOGNITION_MAX_BELIEF_RECORDS
    assert len(full.canonical_bytes()) <= METACOGNITION_MAX_OBSERVATION_BYTES
    assert derive_metacognition_observation_max_bytes() == METACOGNITION_MAX_OBSERVATION_BYTES

    with pytest.raises(ValueError, match="witness bound"):
        observe_metacognition(
            event,
            attention,
            emotion_state=EmotionState(),
            focus_records=(focus_record,),
            belief_records=beliefs,
        )


def test_observation_and_source_digests_reject_post_publication_tampering() -> None:
    event = _event()
    observation = observe_metacognition(
        event,
        AttentionContinuity.bootstrap(),
        emotion_state=EmotionState(),
    )
    object.__setattr__(observation.focus_witness, "attention_state_digest", "0" * 64)
    with pytest.raises(ValueError, match="observation digest"):
        observation.validated_copy()

    witness_tamper = observe_metacognition(
        _event("witness-tamper"),
        AttentionContinuity.bootstrap(),
        emotion_state=EmotionState(),
    )
    object.__setattr__(witness_tamper.evidence[0], "witness_digest", "0" * 64)
    with pytest.raises(ValueError, match="published fields"):
        witness_tamper.validated_copy()

    event = _event("belief-tamper", 2, NOW + timedelta(seconds=2))
    focus_record, attention = _focused_motivation(event, ("belief-tamper",))
    belief = _belief("belief-tamper")
    object.__setattr__(belief.proposition, "proposition_digest", "0" * 64)
    with pytest.raises(ValueError, match="Belief source record is invalid"):
        observe_metacognition(
            event,
            attention,
            focus_records=(focus_record,),
            belief_records=(belief,),
        )

    malformed_record = _belief("belief-list-tamper")
    list_event = _event("belief-list-tamper", 2, NOW + timedelta(seconds=2))
    focus_record, attention = _focused_motivation(list_event, ("belief-list-tamper",))
    object.__setattr__(malformed_record, "evidence", list(malformed_record.evidence))
    with pytest.raises(ValueError, match="Belief source record is invalid"):
        observe_metacognition(
            list_event,
            attention,
            focus_records=(focus_record,),
            belief_records=(malformed_record,),
        )


def test_emotion_and_working_memory_topology_reject_partial_or_invalid_inputs() -> None:
    event = _event()
    attention = AttentionContinuity.bootstrap()
    with pytest.raises(ValueError, match="supplied together"):
        observe_metacognition(
            event,
            attention,
            working_memory_items=(),
        )
    with pytest.raises(ValueError, match="requires its complete membership"):
        observe_metacognition(
            event,
            attention,
            working_memory_view=WorkingMemoryView((), (), 0, 1, 1, 0),
        )
    with pytest.raises(ValueError, match="exact floats"):
        malformed = EmotionState()
        object.__setattr__(malformed, "optimal_loss", True)
        observe_metacognition(event, attention, emotion_state=malformed)

    oversized_source = WorkingMemoryItem(
        item_id="wm-" + "0" * 64,
        source_kind=WorkingMemorySourceKind.EPISODIC,
        source_id="x" * 1_000_000,
        activation=0.5,
        salience=0.5,
        retention_reason=WorkingMemoryRetentionReason.RECENT,
        created_revision=1,
        last_activated_revision=1,
    )
    with pytest.raises(ValueError, match="source_id exceeds its source bound"):
        observe_metacognition(
            event,
            attention,
            working_memory_items=(oversized_source,),
            working_memory_revision=1,
        )


def test_fresh_observation_import_does_not_load_runtime_memory_or_model_authorities() -> None:
    code = """
import importlib
import sys
importlib.import_module('suzka.metacognition.evidence')
for prefix in (
    'suzka.runtime', 'suzka.memory', 'suzka.cognition', 'suzka.models',
    'suzka.provider', 'suzka.providers', 'suzka.scheduler',
    'torch', 'transformers',
):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules), prefix
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
