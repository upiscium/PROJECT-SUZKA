"""Pure source adapters for bounded R14 Attention candidate projections.

Adapters copy exact current source facts into sealed Attention values. They do
not select, admit, resolve, persist, schedule, or mutate source authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
import math
from typing import cast

from suzka.attention.common import (
    ATTENTION_MAX_RENDERED_ITEM_BYTES,
    ATTENTION_MAX_R13_RECORDS_PER_DOMAIN,
    AttentionSourceKind,
    AttentionTargetKind,
    CandidateAvailability,
    SourceDigestKind,
    canonical_json,
    digest_payload,
    validate_digest,
)
from suzka.attention.contracts import (
    AttentionCandidateProjection,
    AttentionEvent,
    AttentionSignalVector,
    AttentionSourceWitness,
    AttentionTarget,
)
from suzka.context_contracts import ContextRelation
from suzka.emotion_contracts import EmotionState
from suzka.identifiers import validate_identifier
from suzka.limits import MAX_PERSISTED_REVISION
from suzka.motivation.commitment import (
    CommitmentLifecycle,
    CommitmentRecord,
    CommitmentRevisionRecord,
    commitment_record_digest,
)
from suzka.motivation.commitment_system import (
    COMMITMENT_SYSTEM_MAX_EVENT_RECEIPTS,
    CommitmentSystemEventReceipt,
    CommitmentSystemSnapshot,
)
from suzka.motivation.goal import (
    GoalLifecycle,
    GoalRecord,
    GoalRevisionRecord,
    goal_record_digest,
)
from suzka.motivation.goal_system import (
    GOAL_SYSTEM_MAX_EVENT_RECEIPTS,
    GoalSystemEventReceipt,
    GoalSystemSnapshot,
)
from suzka.motivation.motivation import (
    MotivationLifecycle,
    MotivationInterpretationCandidate,
    MotivationRecord,
    MotivationRevisionRecord,
    motivation_record_digest,
)
from suzka.motivation.projection import (
    CommitmentPromptEntry,
    GoalPromptEntry,
    MotivationPromptEntry,
    _commitment_value,
    _goal_value,
    _motivation_value,
)
from suzka.motivation.system import (
    MOTIVATION_MAX_CANDIDATES,
    MOTIVATION_MAX_EVENT_RECEIPTS,
    MOTIVATION_MAX_EVIDENCE_LEDGER,
    MOTIVATION_MAX_GOAL_PROPOSAL_WITNESSES,
    MotivationEvidenceLedgerEntry,
    MotivationEventReceipt,
    MotivationGoalProposalWitness,
    MotivationSystemSnapshot,
)
from suzka.working_memory_contracts import (
    MAX_ITEM_CAPACITY,
    MAX_PROJECTION_BYTES,
    WorkingMemoryDecision,
    WorkingMemoryDecisionReason,
    WorkingMemoryItem,
    WorkingMemoryRetentionReason,
    WorkingMemorySelection,
    WorkingMemorySourceKind,
    WorkingMemoryView,
    _validate_source,
    working_memory_item_id,
)


_WM_STATE_PROJECTION_DOMAIN = b"PROJECT-SUZKA:R14:ATTENTION-WM-STATE:V1\0"
_WM_RENDERED_ROW_DOMAIN = b"PROJECT-SUZKA:R14:ATTENTION-WM-ROW:V1\0"
_R13_RENDERED_ROW_DOMAIN = b"PROJECT-SUZKA:R14:ATTENTION-R13-ROW:V1\0"
_GLOBAL_EMOTION_PROJECTION_DOMAIN = (
    b"PROJECT-SUZKA:R14:ATTENTION-GLOBAL-EMOTION:V1\0"
)
_CONTEXT_RELATION_SCORES: dict[ContextRelation, float] = {
    ContextRelation.SAME_CONTEXT: 1.0,
    ContextRelation.PARENT_CHILD: 0.8,
    ContextRelation.RELATED: 0.75,
    ContextRelation.SHARED_INTERLOCUTOR: 0.65,
    ContextRelation.LEGACY_UNKNOWN: 0.45,
    ContextRelation.UNKNOWN_CONTEXT: 0.35,
    ContextRelation.UNRELATED: 0.2,
}
_UNKNOWN_CONTEXT_RELATIONS = frozenset(
    {ContextRelation.LEGACY_UNKNOWN, ContextRelation.UNKNOWN_CONTEXT}
)
_URGENCY_HORIZON_MICROSECONDS = 86_400 * 1_000_000
_URGENCY_FIXED_POINT_SCALE = 1_000_000


def _copy_event(event: object) -> AttentionEvent:
    if type(event) is not AttentionEvent:
        raise TypeError("event must be an exact AttentionEvent")
    try:
        return replace(event)
    except Exception:
        raise ValueError("Attention event is invalid") from None


def _bounded_revision(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value <= MAX_PERSISTED_REVISION:
        raise ValueError(f"{name} must be a bounded non-negative exact integer")
    return value


def _strict_unit_float(value: object, name: str) -> float:
    if (
        type(value) is not float
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} must be a finite exact float in [0, 1]")
    return value


def _validate_wm_identity(
    item_id: object,
    source_kind: object,
    source_id: object,
) -> tuple[str, WorkingMemorySourceKind, str]:
    if type(source_kind) is not WorkingMemorySourceKind:
        raise TypeError("source_kind must be an exact WorkingMemorySourceKind")
    if type(source_id) is not str:
        raise TypeError("source_id must be an exact string")
    try:
        _validate_source(source_kind, source_id)
    except Exception:
        raise ValueError("Working Memory source identity is invalid") from None
    expected_item_id = working_memory_item_id(source_kind, source_id)
    if type(item_id) is not str or item_id != expected_item_id:
        raise ValueError("Working Memory item_id does not match its source identity")
    return item_id, source_kind, source_id


def _validate_working_memory_item(
    item: object, revision: int
) -> WorkingMemoryItem:
    if type(item) is not WorkingMemoryItem:
        raise TypeError("item must be an exact WorkingMemoryItem")
    _validate_wm_identity(item.item_id, item.source_kind, item.source_id)
    _strict_unit_float(item.activation, "activation")
    _strict_unit_float(item.salience, "salience")
    if type(item.retention_reason) is not WorkingMemoryRetentionReason:
        raise ValueError("Working Memory retention_reason is invalid")
    created_revision = _bounded_revision(item.created_revision, "created_revision")
    last_activated_revision = _bounded_revision(
        item.last_activated_revision, "last_activated_revision"
    )
    if created_revision > revision or last_activated_revision > revision:
        raise ValueError("Working Memory item revision exceeds the supplied revision")
    return item


def _r08_score(item: WorkingMemoryItem) -> float:
    retention_bonus = (
        0.1
        if item.retention_reason is WorkingMemoryRetentionReason.REACTIVATED
        else 0.0
    )
    return 0.6 * item.activation + 0.4 * item.salience + retention_bonus


def _validate_score(value: object, item: WorkingMemoryItem, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite exact float")
    if value != _r08_score(item):
        raise ValueError(f"{name} does not match the Working Memory membership facts")
    return value


def _validate_flat_context(
    *,
    relation: object,
    compatibility: object,
    effective_score: object,
    base_score: float,
    context_projection_present: bool,
) -> float | None:
    checked_relation: ContextRelation | None = None
    if relation is not None:
        if type(relation) is not ContextRelation:
            raise TypeError("context_relation must be an exact ContextRelation or None")
        checked_relation = cast(ContextRelation, relation)
    checked_compatibility: float | None = None
    if compatibility is not None:
        checked_compatibility = _strict_unit_float(
            compatibility, "context_compatibility"
        )
    if effective_score is not None:
        if type(effective_score) is not float or not math.isfinite(effective_score):
            raise ValueError("effective_score must be a finite exact float")

    if context_projection_present:
        if checked_relation is not None and checked_compatibility is None:
            raise ValueError("a composite Context projection requires its compatibility score")
        if checked_compatibility is None:
            if effective_score is not None:
                raise ValueError("effective_score requires its compatibility score")
        elif effective_score != base_score * checked_compatibility:
            raise ValueError("effective_score does not match the bounded R08 score")
        return None

    if checked_relation is not None:
        if checked_compatibility is None:
            raise ValueError("a flat ContextRelation requires its compatibility score")
        expected = _CONTEXT_RELATION_SCORES[checked_relation]
        if checked_compatibility != expected:
            raise ValueError("ContextRelation compatibility does not match R09")
    elif checked_compatibility is not None:
        raise ValueError("compatibility without a flat relation requires a Context projection")

    if checked_compatibility is None:
        if effective_score is not None:
            raise ValueError("effective_score requires its compatibility score")
    elif effective_score != base_score * checked_compatibility:
        raise ValueError("effective_score does not match the bounded R08 score")

    if checked_relation is None or checked_relation in _UNKNOWN_CONTEXT_RELATIONS:
        return None
    return checked_compatibility


def _validate_view(
    view: object,
    revision: int,
) -> tuple[dict[str, WorkingMemoryDecision], dict[str, WorkingMemorySelection]]:
    if type(view) is not WorkingMemoryView:
        raise TypeError("view must be an exact WorkingMemoryView or None")
    if type(view.selected) is not tuple or type(view.decisions) is not tuple:
        raise TypeError("Working Memory view rows must be exact tuples")
    if _bounded_revision(view.revision, "view.revision") != revision:
        raise ValueError("Working Memory view revision does not match the supplied revision")
    if type(view.item_capacity) is not int or not 1 <= view.item_capacity <= MAX_ITEM_CAPACITY:
        raise ValueError("Working Memory view item_capacity is invalid")
    if (
        type(view.projection_max_bytes) is not int
        or not 1 <= view.projection_max_bytes <= MAX_PROJECTION_BYTES
    ):
        raise ValueError("Working Memory view projection_max_bytes is invalid")
    if (
        type(view.projected_bytes) is not int
        or not 0 <= view.projected_bytes <= view.projection_max_bytes
    ):
        raise ValueError("Working Memory view projected_bytes is invalid")
    if len(view.decisions) > view.item_capacity or len(view.selected) > view.item_capacity:
        raise ValueError("Working Memory view exceeds its configured item capacity")

    decisions: dict[str, WorkingMemoryDecision] = {}
    decision_sources: set[tuple[WorkingMemorySourceKind, str]] = set()
    for decision in view.decisions:
        if type(decision) is not WorkingMemoryDecision:
            raise TypeError("Working Memory decisions must have the exact source type")
        item_id, source_kind, source_id = _validate_wm_identity(
            decision.item_id, decision.source_kind, decision.source_id
        )
        if item_id in decisions or (source_kind, source_id) in decision_sources:
            raise ValueError("Working Memory decisions contain duplicate membership")
        if type(decision.selected) is not bool:
            raise TypeError("Working Memory decision selected must be an exact bool")
        if type(decision.reason) is not WorkingMemoryDecisionReason:
            raise TypeError("Working Memory decision reason is invalid")
        if decision.selected is not (
            decision.reason is WorkingMemoryDecisionReason.SELECTED
        ):
            raise ValueError("Working Memory decision selection and reason disagree")
        if type(decision.score) is not float or not math.isfinite(decision.score):
            raise ValueError("Working Memory decision score must be a finite exact float")
        _validate_flat_context(
            relation=decision.context_relation,
            compatibility=decision.context_compatibility,
            effective_score=decision.effective_score,
            base_score=decision.score,
            context_projection_present=decision.context_projection is not None,
        )
        decisions[item_id] = decision
        decision_sources.add((source_kind, source_id))

    selections: dict[str, WorkingMemorySelection] = {}
    selected_bytes = 0
    for selection in view.selected:
        if type(selection) is not WorkingMemorySelection:
            raise TypeError("Working Memory selections must have the exact source type")
        item_id, source_kind, source_id = _validate_wm_identity(
            selection.item_id, selection.source_kind, selection.source_id
        )
        if item_id in selections:
            raise ValueError("Working Memory selections contain duplicate identities")
        if type(selection.rendered_content) is not str:
            raise TypeError("Working Memory rendered content must be an exact string")
        try:
            selected_bytes += len(selection.rendered_content.encode("utf-8"))
        except UnicodeEncodeError:
            raise ValueError("Working Memory rendered content is not valid UTF-8") from None
        if type(selection.reason) is not WorkingMemoryDecisionReason or (
            selection.reason is not WorkingMemoryDecisionReason.SELECTED
        ):
            raise ValueError("Working Memory selections must carry the selected reason")
        if type(selection.score) is not float or not math.isfinite(selection.score):
            raise ValueError("Working Memory selection score must be a finite exact float")
        selection_decision = decisions.get(item_id)
        if (
            selection_decision is None
            or not selection_decision.selected
            or selection_decision.source_kind is not source_kind
            or selection_decision.source_id != source_id
            or selection_decision.score != selection.score
            or selection_decision.reason is not selection.reason
            or selection_decision.context_relation is not selection.context_relation
            or selection_decision.context_compatibility != selection.context_compatibility
            or selection_decision.effective_score != selection.effective_score
            or selection_decision.context_projection is not selection.context_projection
        ):
            raise ValueError("Working Memory selection does not match its decision row")
        if selection.source_context_id is not None:
            validate_identifier(selection.source_context_id)
        _validate_flat_context(
            relation=selection.context_relation,
            compatibility=selection.context_compatibility,
            effective_score=selection.effective_score,
            base_score=selection.score,
            context_projection_present=selection.context_projection is not None,
        )
        selections[item_id] = selection

    if set(selections) != {
        item_id for item_id, decision in decisions.items() if decision.selected
    }:
        raise ValueError("Working Memory selected rows do not match selected decisions")
    decision_selection_order = tuple(
        item_id for item_id, decision in decisions.items() if decision.selected
    )
    if tuple(selections) != decision_selection_order:
        raise ValueError("Working Memory selections do not preserve decision order")
    if selected_bytes != view.projected_bytes:
        raise ValueError("Working Memory projected_bytes does not match its selected content")
    return decisions, selections


def _validate_decision_for_item(
    item: WorkingMemoryItem,
    decision: WorkingMemoryDecision,
    selection: WorkingMemorySelection | None,
) -> None:
    if (
        decision.item_id != item.item_id
        or decision.source_kind is not item.source_kind
        or decision.source_id != item.source_id
    ):
        raise ValueError("Working Memory decision source identity does not match the item")
    _validate_score(decision.score, item, "decision score")
    if decision.selected:
        if selection is None:
            raise ValueError("selected Working Memory decision lacks its rendered selection")
        _validate_score(selection.score, item, "selection score")
    elif selection is not None:
        raise ValueError("unselected Working Memory decision has a rendered selection")


def _rendered_evidence(
    row: dict[str, object], domain: bytes
) -> tuple[int, str]:
    encoded = canonical_json(row)
    rendered_bytes = len(encoded)
    if not 1 <= rendered_bytes <= ATTENTION_MAX_RENDERED_ITEM_BYTES:
        raise ValueError("Attention rendered source row exceeds its derived byte bound")
    return rendered_bytes, digest_payload(domain, row)


def _wm_state_digest(item: WorkingMemoryItem, revision: int) -> str:
    return digest_payload(
        _WM_STATE_PROJECTION_DOMAIN,
        {
            "activation": item.activation.hex(),
            "created_revision": item.created_revision,
            "item_id": item.item_id,
            "last_activated_revision": item.last_activated_revision,
            "retention_reason": item.retention_reason.value,
            "salience": item.salience.hex(),
            "source_id": item.source_id,
            "source_kind": item.source_kind.value,
            "working_memory_revision": revision,
        },
    )


def _wm_projection(
    item: WorkingMemoryItem,
    revision: int,
    event: AttentionEvent,
    decision: WorkingMemoryDecision | None,
    selection: WorkingMemorySelection | None,
) -> AttentionCandidateProjection:
    target = AttentionTarget(AttentionTargetKind.WORKING_MEMORY, item.item_id)
    source = AttentionSourceWitness(
        kind=AttentionSourceKind.WORKING_MEMORY,
        reference=item.item_id,
        revision=revision,
        digest=_wm_state_digest(item, revision),
        digest_kind=SourceDigestKind.ATTENTION_PROJECTION,
        target_kind=AttentionTargetKind.WORKING_MEMORY,
        target_reference=item.item_id,
    )
    compatibility: float | None = None
    if decision is not None:
        compatibility = _validate_flat_context(
            relation=decision.context_relation,
            compatibility=decision.context_compatibility,
            effective_score=decision.effective_score,
            base_score=decision.score,
            context_projection_present=decision.context_projection is not None,
        )
    signals = AttentionSignalVector(
        activation=item.activation,
        salience=item.salience,
        context_compatibility=compatibility,
    )
    if selection is None:
        return AttentionCandidateProjection._create(
            target=target,
            source=source,
            signals=signals,
            event=event,
            availability=CandidateAvailability.UNAVAILABLE,
            rendered_bytes=None,
            rendered_digest=None,
        )

    row: dict[str, object] = {
        "item_id": item.item_id,
        "source_kind": item.source_kind.value,
        "source_id": item.source_id,
        "text": selection.rendered_content,
    }
    rendered_bytes, rendered_digest = _rendered_evidence(
        row, _WM_RENDERED_ROW_DOMAIN
    )
    return AttentionCandidateProjection._create(
        target=target,
        source=source,
        signals=signals,
        event=event,
        availability=CandidateAvailability.ELIGIBLE,
        rendered_bytes=rendered_bytes,
        rendered_digest=rendered_digest,
    )


def project_working_memory(
    item: WorkingMemoryItem,
    *,
    revision: int,
    event: AttentionEvent,
    view: WorkingMemoryView | None = None,
) -> AttentionCandidateProjection:
    """Project one exact current WM member, retaining facts if content is absent."""

    revision = _bounded_revision(revision, "working_memory revision")
    current_item = _validate_working_memory_item(item, revision)
    current_event = _copy_event(event)
    decision: WorkingMemoryDecision | None = None
    selection: WorkingMemorySelection | None = None
    if view is not None:
        decisions, selections = _validate_view(view, revision)
        decision = decisions.get(current_item.item_id)
        if decision is None:
            raise ValueError("Working Memory view lacks the current item's decision row")
        selection = selections.get(current_item.item_id)
        _validate_decision_for_item(current_item, decision, selection)
    return _wm_projection(current_item, revision, current_event, decision, selection)


def _copy_r13_event_fields(
    *, event_id: object, event_sequence: object, occurred_at: object, name: str
) -> tuple[str | None, int | None, datetime | None]:
    if event_id is None and event_sequence is None:
        return None, None, None
    if type(event_id) is not str or type(event_sequence) is not int:
        raise ValueError(f"{name} source event identity is incomplete")
    if type(occurred_at) is not datetime:
        raise ValueError(f"{name} source event time is invalid")
    return event_id, event_sequence, occurred_at


def _latest_event_fields(
    history: object,
    revision_record_type: type[object],
    name: str,
    current_event: AttentionEvent | None = None,
) -> tuple[str | None, int | None, datetime | None]:
    if type(history) is not tuple or not history:
        raise ValueError(f"{name} revision history is empty or malformed")
    if any(type(item) is not revision_record_type for item in history):
        raise TypeError(f"{name} revision history has a non-exact record type")
    latest = history[-1]
    latest_time = getattr(latest, "created_at", None)
    if type(latest_time) is not datetime:
        raise ValueError(f"{name} latest revision time is invalid")
    if current_event is not None and latest_time > current_event.occurred_at:
        raise ValueError(f"{name} latest revision is in the future of the Attention event")
    return _copy_r13_event_fields(
        event_id=getattr(latest, "event_id", None),
        event_sequence=getattr(latest, "event_sequence", None),
        occurred_at=latest_time,
        name=name,
    )


def _r13_witness(
    *,
    source_kind: AttentionSourceKind,
    target_kind: AttentionTargetKind,
    reference: str,
    revision: int,
    record_digest: str,
    event_fields: tuple[str | None, int | None, datetime | None],
) -> AttentionSourceWitness:
    source_event_id, source_event_sequence, source_occurred_at = event_fields
    return AttentionSourceWitness(
        kind=source_kind,
        reference=reference,
        revision=revision,
        digest=record_digest,
        digest_kind=SourceDigestKind.UPSTREAM_AUTHORITY,
        target_kind=target_kind,
        target_reference=reference,
        source_event_id=source_event_id,
        source_event_sequence=source_event_sequence,
        source_occurred_at=source_occurred_at,
    )


def _r13_projection(
    *,
    target_kind: AttentionTargetKind,
    source_kind: AttentionSourceKind,
    reference: str,
    revision: int,
    record_digest: str,
    event_fields: tuple[str | None, int | None, datetime | None],
    signals: AttentionSignalVector,
    event: AttentionEvent,
    row: dict[str, object],
) -> AttentionCandidateProjection:
    target = AttentionTarget(target_kind, reference)
    source = _r13_witness(
        source_kind=source_kind,
        target_kind=target_kind,
        reference=reference,
        revision=revision,
        record_digest=record_digest,
        event_fields=event_fields,
    )
    rendered_bytes, rendered_digest = _rendered_evidence(
        row, _R13_RENDERED_ROW_DOMAIN
    )
    return AttentionCandidateProjection._create(
        target=target,
        source=source,
        signals=signals,
        event=event,
        availability=CandidateAvailability.ELIGIBLE,
        rendered_bytes=rendered_bytes,
        rendered_digest=rendered_digest,
    )


def _clone_motivation(record: MotivationRecord) -> MotivationRecord:
    if type(record) is not MotivationRecord:
        raise TypeError("record must be an exact MotivationRecord")
    source_digest = record.record_digest
    try:
        current = replace(record)
        computed_digest = motivation_record_digest(current)
    except Exception:
        raise ValueError("Motivation record is invalid") from None
    if current.record_digest != source_digest or computed_digest != source_digest:
        raise ValueError("Motivation record digest does not match its current state")
    _latest_event_fields(
        current.revision_history, MotivationRevisionRecord, "Motivation"
    )
    return current


def _project_motivation_current(
    record: MotivationRecord, event: AttentionEvent
) -> AttentionCandidateProjection:
    if record.lifecycle is not MotivationLifecycle.ACTIVE:
        raise ValueError("only active Motivations may be projected")
    entry = MotivationPromptEntry._from_record(record)
    event_fields = _latest_event_fields(
        record.revision_history,
        MotivationRevisionRecord,
        "Motivation",
        current_event=event,
    )
    signals = AttentionSignalVector(
        strength=record.strength,
        persistence=record.persistence,
        satiation=record.satiation,
        uncertainty=record.uncertainty,
    )
    return _r13_projection(
        target_kind=AttentionTargetKind.MOTIVATION,
        source_kind=AttentionSourceKind.MOTIVATION,
        reference=record.motivation_id,
        revision=record.revision,
        record_digest=record.record_digest,
        event_fields=event_fields,
        signals=signals,
        event=event,
        row=_motivation_value(entry),
    )


def project_motivation(
    record: MotivationRecord,
    *,
    event: AttentionEvent,
) -> AttentionCandidateProjection:
    """Project one exact ACTIVE Motivation without running R13 policy."""

    current_event = _copy_event(event)
    current_record = _clone_motivation(record)
    return _project_motivation_current(current_record, current_event)


def _clone_goal(record: GoalRecord) -> GoalRecord:
    if type(record) is not GoalRecord:
        raise TypeError("record must be an exact GoalRecord")
    source_digest = record.record_digest
    try:
        current = replace(record)
        computed_digest = goal_record_digest(current)
    except Exception:
        raise ValueError("Goal record is invalid") from None
    if current.record_digest != source_digest or computed_digest != source_digest:
        raise ValueError("Goal record digest does not match its current state")
    _latest_event_fields(current.revision_history, GoalRevisionRecord, "Goal")
    return current


def _deadline_urgency(deadline: datetime | None, event: AttentionEvent) -> float | None:
    if deadline is None:
        return None
    if type(deadline) is not datetime:
        raise TypeError("deadline must be an exact datetime or None")
    remaining = deadline - event.occurred_at
    remaining_microseconds = (
        remaining.days * 86_400 * 1_000_000
        + remaining.seconds * 1_000_000
        + remaining.microseconds
    )
    if remaining_microseconds <= 0:
        return 1.0
    if remaining_microseconds >= _URGENCY_HORIZON_MICROSECONDS:
        return 0.0
    units = (
        (_URGENCY_HORIZON_MICROSECONDS - remaining_microseconds)
        * _URGENCY_FIXED_POINT_SCALE
        // _URGENCY_HORIZON_MICROSECONDS
    )
    return units / _URGENCY_FIXED_POINT_SCALE


def _project_goal_current(
    record: GoalRecord, event: AttentionEvent
) -> AttentionCandidateProjection:
    if record.lifecycle is not GoalLifecycle.ADOPTED:
        raise ValueError("only adopted Goals may be projected")
    entry = GoalPromptEntry._from_record(record)
    event_fields = _latest_event_fields(
        record.revision_history, GoalRevisionRecord, "Goal", current_event=event
    )
    signals = AttentionSignalVector(
        urgency=_deadline_urgency(entry.deadline.at, event)
    )
    return _r13_projection(
        target_kind=AttentionTargetKind.GOAL,
        source_kind=AttentionSourceKind.GOAL,
        reference=record.goal_id,
        revision=record.revision,
        record_digest=record.record_digest,
        event_fields=event_fields,
        signals=signals,
        event=event,
        row=_goal_value(entry),
    )


def project_goal(
    record: GoalRecord,
    *,
    event: AttentionEvent,
) -> AttentionCandidateProjection:
    """Project one exact ADOPTED Goal and its explicit deadline facts."""

    current_event = _copy_event(event)
    current_record = _clone_goal(record)
    return _project_goal_current(current_record, current_event)


def _clone_commitment(record: CommitmentRecord) -> CommitmentRecord:
    if type(record) is not CommitmentRecord:
        raise TypeError("record must be an exact CommitmentRecord")
    source_digest = record.record_digest
    try:
        current = replace(record)
        computed_digest = commitment_record_digest(current)
    except Exception:
        raise ValueError("Commitment record is invalid") from None
    if current.record_digest != source_digest or computed_digest != source_digest:
        raise ValueError("Commitment record digest does not match its current state")
    _latest_event_fields(
        current.revision_history, CommitmentRevisionRecord, "Commitment"
    )
    return current


def _project_commitment_current(
    record: CommitmentRecord, event: AttentionEvent
) -> AttentionCandidateProjection:
    if record.lifecycle is not CommitmentLifecycle.ACTIVE:
        raise ValueError("only active Commitments may be projected")
    entry = CommitmentPromptEntry._from_record(record)
    event_fields = _latest_event_fields(
        record.revision_history,
        CommitmentRevisionRecord,
        "Commitment",
        current_event=event,
    )
    signals = AttentionSignalVector(
        urgency=_deadline_urgency(entry.deadline.at, event)
    )
    return _r13_projection(
        target_kind=AttentionTargetKind.COMMITMENT,
        source_kind=AttentionSourceKind.COMMITMENT,
        reference=record.commitment_id,
        revision=record.revision,
        record_digest=record.record_digest,
        event_fields=event_fields,
        signals=signals,
        event=event,
        row=_commitment_value(entry),
    )


def project_commitment(
    record: CommitmentRecord,
    *,
    event: AttentionEvent,
) -> AttentionCandidateProjection:
    """Project one exact ACTIVE Commitment without subject-admission authority."""

    current_event = _copy_event(event)
    current_record = _clone_commitment(record)
    return _project_commitment_current(current_record, current_event)


def _clone_motivation_snapshot(
    snapshot: MotivationSystemSnapshot,
) -> tuple[MotivationRecord, ...]:
    if type(snapshot) is not MotivationSystemSnapshot:
        raise TypeError("motivation_snapshot must be an exact MotivationSystemSnapshot")
    if type(snapshot.records) is not tuple or len(snapshot.records) > ATTENTION_MAX_R13_RECORDS_PER_DOMAIN:
        raise ValueError("Motivation snapshot records exceed their exact source bound")
    for name, values, maximum, row_type in (
        (
            "evidence_ledger",
            snapshot.evidence_ledger,
            MOTIVATION_MAX_EVIDENCE_LEDGER,
            MotivationEvidenceLedgerEntry,
        ),
        ("candidates", snapshot.candidates, MOTIVATION_MAX_CANDIDATES, MotivationInterpretationCandidate),
        ("event_receipts", snapshot.event_receipts, MOTIVATION_MAX_EVENT_RECEIPTS, MotivationEventReceipt),
        (
            "goal_proposal_witnesses",
            snapshot.goal_proposal_witnesses,
            MOTIVATION_MAX_GOAL_PROPOSAL_WITNESSES,
            MotivationGoalProposalWitness,
        ),
    ):
        _validate_snapshot_rows(values, name, maximum, row_type)
    records = tuple(_clone_motivation(record) for record in snapshot.records)
    try:
        current = replace(snapshot, records=records)
    except Exception:
        raise ValueError("Motivation system snapshot is invalid") from None
    if (
        current.authority_digest != snapshot.authority_digest
        or current.serialized_bytes != snapshot.serialized_bytes
    ):
        raise ValueError("Motivation system snapshot digest does not match its contents")
    return current.records


def _clone_goal_snapshot(snapshot: GoalSystemSnapshot) -> tuple[GoalRecord, ...]:
    if type(snapshot) is not GoalSystemSnapshot:
        raise TypeError("goal_snapshot must be an exact GoalSystemSnapshot")
    if type(snapshot.records) is not tuple or len(snapshot.records) > ATTENTION_MAX_R13_RECORDS_PER_DOMAIN:
        raise ValueError("Goal snapshot records exceed their exact source bound")
    _validate_snapshot_rows(
        snapshot.event_receipts,
        "Goal event_receipts",
        GOAL_SYSTEM_MAX_EVENT_RECEIPTS,
        GoalSystemEventReceipt,
    )
    records = tuple(_clone_goal(record) for record in snapshot.records)
    try:
        current = replace(snapshot, records=records)
    except Exception:
        raise ValueError("Goal system snapshot is invalid") from None
    if (
        current.authority_digest != snapshot.authority_digest
        or current.serialized_bytes != snapshot.serialized_bytes
    ):
        raise ValueError("Goal system snapshot digest does not match its contents")
    return current.records


def _clone_commitment_snapshot(
    snapshot: CommitmentSystemSnapshot,
) -> tuple[CommitmentRecord, ...]:
    if type(snapshot) is not CommitmentSystemSnapshot:
        raise TypeError("commitment_snapshot must be an exact CommitmentSystemSnapshot")
    if type(snapshot.records) is not tuple or len(snapshot.records) > ATTENTION_MAX_R13_RECORDS_PER_DOMAIN:
        raise ValueError("Commitment snapshot records exceed their exact source bound")
    _validate_snapshot_rows(
        snapshot.event_receipts,
        "Commitment event_receipts",
        COMMITMENT_SYSTEM_MAX_EVENT_RECEIPTS,
        CommitmentSystemEventReceipt,
    )
    records = tuple(_clone_commitment(record) for record in snapshot.records)
    try:
        current = replace(snapshot, records=records)
    except Exception:
        raise ValueError("Commitment system snapshot is invalid") from None
    if (
        current.authority_digest != snapshot.authority_digest
        or current.serialized_bytes != snapshot.serialized_bytes
    ):
        raise ValueError("Commitment system snapshot digest does not match its contents")
    return current.records


def _validate_snapshot_rows(
    values: object,
    name: str,
    maximum: int,
    row_type: type[object],
) -> None:
    if type(values) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")
    if len(values) > maximum:
        raise ValueError(f"{name} exceeds its source bound")
    if any(type(item) is not row_type for item in values):
        raise TypeError(f"{name} contains a non-exact source row type")


def _validate_wm_universe(
    items: object, revision: object
) -> tuple[tuple[WorkingMemoryItem, ...], int, dict[str, WorkingMemoryItem]]:
    checked_revision = _bounded_revision(revision, "working_memory_revision")
    if type(items) is not tuple:
        raise TypeError("working_memory_items must be an exact tuple")
    if len(items) > MAX_ITEM_CAPACITY:
        raise ValueError("Working Memory membership exceeds its source capacity")
    current_items: list[WorkingMemoryItem] = []
    by_id: dict[str, WorkingMemoryItem] = {}
    by_source: set[tuple[WorkingMemorySourceKind, str]] = set()
    for item in items:
        current_item = _validate_working_memory_item(item, checked_revision)
        source_reference = (current_item.source_kind, current_item.source_id)
        if current_item.item_id in by_id or source_reference in by_source:
            raise ValueError("Working Memory membership contains duplicate identities")
        by_id[current_item.item_id] = current_item
        by_source.add(source_reference)
        current_items.append(current_item)
    return tuple(current_items), checked_revision, by_id


def project_attention_candidates(
    *,
    working_memory_items: tuple[WorkingMemoryItem, ...],
    working_memory_revision: int,
    working_memory_view: WorkingMemoryView | None,
    motivation_snapshot: MotivationSystemSnapshot,
    goal_snapshot: GoalSystemSnapshot,
    commitment_snapshot: CommitmentSystemSnapshot,
    event: AttentionEvent,
) -> tuple[AttentionCandidateProjection, ...]:
    """Project the complete current WM/R13 candidate universe in stable order."""

    current_event = _copy_event(event)
    items, revision, items_by_id = _validate_wm_universe(
        working_memory_items, working_memory_revision
    )
    decisions: dict[str, WorkingMemoryDecision] = {}
    selections: dict[str, WorkingMemorySelection] = {}
    if working_memory_view is not None:
        decisions, selections = _validate_view(working_memory_view, revision)
        if set(decisions) != set(items_by_id):
            raise ValueError(
                "Working Memory view decisions do not cover the complete current membership"
            )
        for item_id, item in items_by_id.items():
            _validate_decision_for_item(
                item, decisions[item_id], selections.get(item_id)
            )

    motivations = _clone_motivation_snapshot(motivation_snapshot)
    goals = _clone_goal_snapshot(goal_snapshot)
    commitments = _clone_commitment_snapshot(commitment_snapshot)

    projections = [
        _wm_projection(
            item,
            revision,
            current_event,
            decisions.get(item.item_id),
            selections.get(item.item_id),
        )
        for item in items
    ]
    projections.extend(
        _project_motivation_current(record, current_event)
        for record in motivations
        if record.lifecycle is MotivationLifecycle.ACTIVE
    )
    projections.extend(
        _project_goal_current(record, current_event)
        for record in goals
        if record.lifecycle is GoalLifecycle.ADOPTED
    )
    projections.extend(
        _project_commitment_current(record, current_event)
        for record in commitments
        if record.lifecycle is CommitmentLifecycle.ACTIVE
    )
    projections.sort(key=lambda projection: projection.candidate_id)
    candidate_ids = tuple(projection.candidate_id for projection in projections)
    if candidate_ids != tuple(sorted(set(candidate_ids))):
        raise ValueError("Attention source projections contain duplicate candidate identities")
    return tuple(projections)


@dataclass(frozen=True, slots=True, init=False)
class GlobalEmotionProjection:
    """Sealed event-scoped arousal evidence for the Attention resource ceiling."""

    event: AttentionEvent
    arousal: float
    projection_digest: str = field(init=False)

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("GlobalEmotionProjection values are created by source adapters")

    @classmethod
    def _create(
        cls, *, event: AttentionEvent, arousal: float
    ) -> GlobalEmotionProjection:
        if cls is not GlobalEmotionProjection:
            raise TypeError("GlobalEmotionProjection cannot be subclassed")
        current_event = _copy_event(event)
        current_arousal = _strict_unit_float(arousal, "global Emotion arousal")
        result = object.__new__(cls)
        object.__setattr__(result, "event", current_event)
        object.__setattr__(result, "arousal", current_arousal)
        object.__setattr__(
            result,
            "projection_digest",
            _global_emotion_projection_digest(current_event, current_arousal),
        )
        result.__post_init__()
        return result

    def __post_init__(self) -> None:
        if type(self.event) is not AttentionEvent:
            raise TypeError("event must be an exact AttentionEvent")
        try:
            current_event = replace(self.event)
        except Exception:
            raise ValueError("global Emotion event is invalid") from None
        arousal = _strict_unit_float(self.arousal, "global Emotion arousal")
        declared_digest = validate_digest(
            self.projection_digest, "global Emotion projection_digest"
        )
        expected_digest = _global_emotion_projection_digest(current_event, arousal)
        if declared_digest != expected_digest:
            raise ValueError("Global Emotion projection digest does not match its contents")

    def validated_copy(self) -> GlobalEmotionProjection:
        """Return a checked independent copy, rejecting a stale supplied digest."""

        if type(self) is not GlobalEmotionProjection:
            raise TypeError("value must be an exact GlobalEmotionProjection")
        source_event = self.event
        source_arousal = self.arousal
        source_state_digest = self.projection_digest
        current_event = _copy_event(source_event)
        current_arousal = _strict_unit_float(
            source_arousal, "global Emotion arousal"
        )
        declared_digest = validate_digest(
            source_state_digest, "global Emotion projection_digest"
        )
        copied = GlobalEmotionProjection._create(
            event=current_event,
            arousal=current_arousal,
        )
        if copied.projection_digest != declared_digest:
            raise ValueError("Global Emotion projection digest does not match its contents")
        return copied


def _global_emotion_projection_digest(
    event: AttentionEvent, arousal: float
) -> str:
    return digest_payload(
        _GLOBAL_EMOTION_PROJECTION_DOMAIN,
        {"arousal": arousal.hex(), "event": event.canonical_value()},
    )


def project_global_emotion(
    state: EmotionState,
    *,
    event: AttentionEvent,
) -> GlobalEmotionProjection:
    """Project only exact current arousal for the resource ceiling, never ranking."""

    if type(state) is not EmotionState:
        raise TypeError("state must be an exact EmotionState")
    current_event = _copy_event(event)
    try:
        current_state = replace(state)
    except Exception:
        raise ValueError("Emotion state is invalid") from None
    # R10 intentionally permits finite negative optimal_loss on its raw update
    # path; Attention validates but never changes that existing semantics.
    if type(current_state.optimal_loss) is not float or not math.isfinite(
        current_state.optimal_loss
    ):
        raise ValueError("Emotion optimal_loss must remain a finite exact float")
    _strict_unit_float(current_state.arousal, "global Emotion arousal")
    return GlobalEmotionProjection._create(
        event=current_event,
        arousal=current_state.arousal,
    )


__all__ = [
    "GlobalEmotionProjection",
    "project_attention_candidates",
    "project_commitment",
    "project_global_emotion",
    "project_goal",
    "project_motivation",
    "project_working_memory",
]
