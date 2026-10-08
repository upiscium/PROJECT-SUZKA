"""Pure event-scoped metacognition observations derived from typed inputs.

An observation is a detached, bounded measurement packet.  It is not an
assessment, source authenticator, memory of prior observations, or producer of
world-truth claims.  Checksums bind the value's declared fields; they do not
authenticate who supplied the inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
import hashlib
import math
from typing import Final, TypeAlias, TypeVar, cast

from suzka.attention.adapters import (
    _validate_decision_for_item,
    _validate_view,
    _validate_working_memory_item,
    _wm_state_digest,
    project_commitment,
    project_goal,
    project_motivation,
    project_working_memory,
)
from suzka.context_contracts import ContextRelation
from suzka.attention.common import (
    ATTENTION_MAX_EVENT_RECEIPTS,
    ATTENTION_FIXED_POINT_SCALE,
    ATTENTION_HIGH_AROUSAL_MAX_FOCUS,
    ATTENTION_HIGH_AROUSAL_THRESHOLD,
    ATTENTION_MAX_FOCUS,
    ATTENTION_MAX_R13_RECORDS_PER_DOMAIN,
    ATTENTION_MAX_REVISION_HISTORY,
    AttentionSourceKind,
    AttentionTargetKind,
    SourceDigestKind,
    canonical_datetime,
    canonical_json,
    digest_payload,
    exact_enum,
    validate_digest,
)
from suzka.attention.contracts import (
    AttentionCandidateContinuity,
    AttentionContinuity,
    AttentionEventReceipt,
    AttentionEvent,
    AttentionReceiptAnchor,
    AttentionRevisionAnchor,
    AttentionRevisionEvidence,
    AttentionSourceWitness,
    AttentionTarget,
)
from suzka.belief.records import (
    BELIEF_MAX_COMPONENT_CODEPOINTS,
    BELIEF_MAX_CONTEXTS,
    BELIEF_MAX_EVIDENCE,
    BELIEF_MAX_PROPOSITION_CODEPOINTS,
    BELIEF_MAX_REVISIONS,
    BELIEF_MAX_REVISION_WITNESSES,
    BeliefEvidence,
    BeliefEvidenceType,
    BeliefLifecycle,
    BeliefProposition,
    BeliefRecord,
    BeliefRevisionOperation,
    BeliefRevisionReason,
    BeliefRevisionRecord,
    BeliefSubjectAdmission,
    belief_record_digest,
    build_conflict_candidate,
    validate_proposition_digest,
    validate_revision_digest,
)
from suzka.emotion_contracts import EmotionState
from suzka.identifiers import MAX_IDENTIFIER_CODEPOINTS, validate_identifier
from suzka.limits import MAX_PERSISTED_REVISION
from suzka.metacognition.contracts import (
    METACOGNITION_MAX_EVIDENCE_WITNESSES,
    METACOGNITION_MAX_REASON_CODES,
    EvidenceCondition,
    FocusAssessmentWitness,
    MetacognitiveEvidenceWitness,
    MetacognitiveReasonCode,
    SourceEventOrigin,
)
from suzka.motivation.commitment import (
    CommitmentLifecycle,
    CommitmentRecord,
    CommitmentRevisionRecord,
    CommitmentSubjectAdmission,
    CommitmentSubjectTransitionProof,
    commitment_record_digest,
)
from suzka.motivation.common import (
    R13_MAX_CONFLICT_REFS,
    R13_MAX_DEPENDENCY_REFS,
    R13_MAX_EVIDENCE_REFS,
    R13_MAX_RELATED_REFS,
    R13_MAX_REVISION_HISTORY,
    R13_MAX_REVISION_WITNESSES,
    R13_MAX_SCOPE_CODEPOINTS,
    R13_MAX_SCOPE_ITEMS,
    R13Reference,
    R13ReferenceKind,
    RevisionCompactionAnchor,
)
from suzka.motivation.goal import (
    GoalLifecycle,
    GoalRecord,
    GoalRevisionRecord,
    GoalSubjectAdmission,
    GoalSubjectTransitionProof,
    goal_record_digest,
)
from suzka.motivation.motivation import (
    MotivationLifecycle,
    MotivationRecord,
    MotivationRevisionRecord,
    motivation_record_digest,
)
from suzka.working_memory_contracts import (
    MAX_ITEM_CAPACITY,
    MAX_PROJECTION_BYTES,
    MAX_SOURCE_ID_BYTES,
    WorkingMemoryDecision,
    WorkingMemoryItem,
    WorkingMemorySelection,
    WorkingMemorySourceKind,
    WorkingMemoryView,
)


METACOGNITION_UNITS_SCALE: Final[int] = ATTENTION_FIXED_POINT_SCALE
METACOGNITION_MAX_FOCUS_RECORDS: Final[int] = ATTENTION_MAX_FOCUS
METACOGNITION_MAX_BELIEF_RECORDS: Final[int] = METACOGNITION_MAX_EVIDENCE_WITNESSES
_UTF8_PREFLIGHT_CHUNK_CODEPOINTS: Final[int] = 4_096

_OBSERVATION_DOMAIN: Final[bytes] = (
    b"PROJECT-SUZKA:R14:METACOGNITION-OBSERVATION:V1\0"
)
_WORKING_MEMORY_OBSERVATION_DOMAIN: Final[bytes] = (
    b"PROJECT-SUZKA:R14:METACOGNITION-WORKING-MEMORY:V1\0"
)
_EMOTION_OBSERVATION_DOMAIN: Final[bytes] = (
    b"PROJECT-SUZKA:R14:METACOGNITION-EMOTION:V1\0"
)

_R13FocusRecord: TypeAlias = MotivationRecord | GoalRecord | CommitmentRecord
_EnumT = TypeVar("_EnumT", bound=Enum)


def _copy_event(value: object, name: str = "event") -> AttentionEvent:
    if type(value) is not AttentionEvent:
        raise TypeError(f"{name} must be an exact AttentionEvent")
    _checked_identifier(value.event_id, f"{name}.event_id")
    return AttentionEvent(value.event_id, value.event_sequence, value.occurred_at)


def _checked_identifier(value: object, name: str) -> str:
    if type(value) is not str or len(value) > MAX_IDENTIFIER_CODEPOINTS:
        raise ValueError(f"{name} exceeds its bounded identifier size")
    return validate_identifier(value)


def _checked_digest(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise ValueError(f"{name} must be a bounded SHA-256 digest")
    return validate_digest(value, name)


def _event_not_future(source: AttentionEvent, current: AttentionEvent) -> None:
    if source.event_sequence > current.event_sequence:
        raise ValueError("source event is in the future of the observation event")
    if source.occurred_at > current.occurred_at:
        raise ValueError("source event time is in the future of the observation event")
    if source.event_sequence == current.event_sequence and source != current:
        raise ValueError("equal event sequences must identify the exact same event")
    if source.event_id == current.event_id and source != current:
        raise ValueError("reused event IDs must identify the exact same event")


def _preflight_attention(value: AttentionContinuity) -> None:
    maximum_candidates = MAX_ITEM_CAPACITY + 3 * ATTENTION_MAX_R13_RECORDS_PER_DOMAIN
    candidates = _preflight_exact_tuple(value.candidates, "Attention candidates", maximum_candidates)
    focused = _preflight_exact_tuple(value.focused_ids, "Attention focused_ids", ATTENTION_MAX_FOCUS)
    unfinished = _preflight_exact_tuple(value.unfinished_ids, "Attention unfinished_ids", ATTENTION_MAX_FOCUS)
    revisions = _preflight_exact_tuple(
        value.revision_history,
        "Attention revision_history",
        ATTENTION_MAX_REVISION_HISTORY,
    )
    receipts = _preflight_exact_tuple(
        value.receipts,
        "Attention receipts",
        ATTENTION_MAX_EVENT_RECEIPTS,
    )
    if type(value.revision) is not int or not 0 <= value.revision <= MAX_PERSISTED_REVISION:
        raise ValueError("Attention revision must be a bounded exact integer")
    for name, identities in (("focused_ids", focused), ("unfinished_ids", unfinished)):
        for candidate_id in identities:
            _checked_digest(candidate_id, f"Attention {name} item")
    if value.last_event is not None:
        _copy_event(value.last_event, "Attention last_event")
    candidate_counts = {kind: 0 for kind in AttentionTargetKind}
    for item in candidates:
        if type(item) is not AttentionCandidateContinuity:
            raise TypeError("Attention candidates must be exact continuity values")
        candidate = cast(AttentionCandidateContinuity, item)
        if type(candidate.target) is not AttentionTarget:
            raise TypeError("Attention candidate target must be exact")
        if type(candidate.target.kind) is not AttentionTargetKind:
            raise TypeError("Attention candidate target kind must be exact")
        candidate_counts[candidate.target.kind] += 1
        _checked_identifier(candidate.target.reference, "Attention candidate target reference")
        if type(candidate.source) is not AttentionSourceWitness:
            raise TypeError("Attention candidate source must be exact")
        _checked_identifier(candidate.source.reference, "Attention source reference")
        _checked_identifier(candidate.source.target_reference, "Attention source target_reference")
        _checked_digest(candidate.source.digest, "Attention source digest")
        _checked_digest(candidate.record_digest, "Attention candidate record_digest")
        source_event_id = candidate.source.source_event_id
        source_event_sequence = candidate.source.source_event_sequence
        source_event_time = candidate.source.source_occurred_at
        if source_event_id is None:
            if source_event_sequence is not None or source_event_time is not None:
                raise ValueError("Attention source event identity is incomplete")
        else:
            if type(source_event_sequence) is not int or type(source_event_time) is not datetime:
                raise ValueError("Attention source event identity is incomplete")
            source_event = AttentionEvent(
                _checked_identifier(source_event_id, "Attention source event_id"),
                cast(int, source_event_sequence),
                cast(datetime, source_event_time),
            )
            _copy_event(source_event, "Attention source event")
    for kind, count in candidate_counts.items():
        maximum = (
            MAX_ITEM_CAPACITY
            if kind is AttentionTargetKind.WORKING_MEMORY
            else ATTENTION_MAX_R13_RECORDS_PER_DOMAIN
        )
        if count > maximum:
            raise ValueError("Attention candidate count exceeds its source bound")
    if any(type(item) is not AttentionRevisionEvidence for item in revisions):
        raise TypeError("Attention revision_history must contain exact evidence")
    checked_revisions = cast(tuple[AttentionRevisionEvidence, ...], revisions)
    for revision in checked_revisions:
        _checked_digest(revision.previous_state_digest, "Attention revision previous_state_digest")
        _checked_digest(revision.state_digest, "Attention revision state_digest")
        if revision.previous_revision_digest is not None:
            _checked_digest(revision.previous_revision_digest, "Attention previous revision digest")
        _checked_digest(revision.record_digest, "Attention revision record_digest")
        _copy_event(revision.event, "Attention revision event")
    if any(type(item) is not AttentionEventReceipt for item in receipts):
        raise TypeError("Attention receipts must contain exact receipt values")
    checked_receipts = cast(tuple[AttentionEventReceipt, ...], receipts)
    for receipt in checked_receipts:
        _checked_digest(receipt.input_digest, "Attention receipt input_digest")
        _checked_digest(receipt.result_state_digest, "Attention receipt result_state_digest")
        if receipt.previous_receipt_digest is not None:
            _checked_digest(receipt.previous_receipt_digest, "Attention previous receipt digest")
        _checked_digest(receipt.receipt_digest, "Attention receipt digest")
        _copy_event(receipt.event, "Attention receipt event")
    if value.revision_anchor is not None:
        if type(value.revision_anchor) is not AttentionRevisionAnchor:
            raise TypeError("Attention revision_anchor must be exact")
        _checked_digest(value.revision_anchor.through_state_digest, "Attention revision anchor state_digest")
        _checked_digest(value.revision_anchor.through_revision_digest, "Attention revision anchor revision_digest")
        _copy_event(value.revision_anchor.through_event, "Attention revision anchor event")
    if value.receipt_anchor is not None:
        if type(value.receipt_anchor) is not AttentionReceiptAnchor:
            raise TypeError("Attention receipt_anchor must be exact")
        _checked_digest(value.receipt_anchor.through_result_state_digest, "Attention receipt anchor state_digest")
        _checked_digest(value.receipt_anchor.through_receipt_digest, "Attention receipt anchor receipt_digest")
        _copy_event(value.receipt_anchor.through_event, "Attention receipt anchor event")


def _copy_attention(value: object) -> AttentionContinuity:
    if type(value) is not AttentionContinuity:
        raise TypeError("attention must be an exact AttentionContinuity")
    try:
        _preflight_attention(value)
        return AttentionContinuity.from_canonical_value(value.canonical_value())
    except Exception as error:
        raise ValueError("Attention continuity is invalid or changed after publication") from error


def _attention_events(snapshot: AttentionContinuity) -> tuple[AttentionEvent, ...]:
    events: list[AttentionEvent] = []
    if snapshot.last_event is not None:
        events.append(snapshot.last_event)
    events.extend(item.event for item in snapshot.revision_history)
    events.extend(item.event for item in snapshot.receipts)
    for candidate in snapshot.candidates:
        source_event = candidate.source.event()
        if source_event is not None:
            events.append(source_event)
    if snapshot.revision_anchor is not None:
        events.append(snapshot.revision_anchor.through_event)
    if snapshot.receipt_anchor is not None:
        events.append(snapshot.receipt_anchor.through_event)
    return tuple(events)


def _validate_attention_event(
    snapshot: AttentionContinuity,
    event: AttentionEvent,
) -> None:
    last_event = snapshot.last_event
    if last_event is not None:
        if last_event.event_sequence > event.event_sequence:
            raise ValueError("Attention last_event is in the future of the observation")
        if last_event.occurred_at > event.occurred_at:
            raise ValueError("Attention last_event time is in the future of the observation")
        if last_event.event_sequence == event.event_sequence and last_event != event:
            raise ValueError("equal Attention event sequences must identify the exact same event")

    for retained in _attention_events(snapshot):
        if retained.event_id == event.event_id or retained.event_sequence == event.event_sequence:
            if retained != event:
                raise ValueError(
                    "observation event conflicts with retained Attention event identity"
                )


def _validate_source_event_identity_consistency(
    snapshot: AttentionContinuity,
    event: AttentionEvent,
    beliefs: tuple[BeliefRecord, ...],
) -> None:
    by_id: dict[str, AttentionEvent] = {}
    by_sequence: dict[int, AttentionEvent] = {}

    def add_full_event(source: AttentionEvent) -> None:
        previous_by_id = by_id.get(source.event_id)
        if previous_by_id is not None and previous_by_id != source:
            raise ValueError("one event ID identifies conflicting source events")
        previous_by_sequence = by_sequence.get(source.event_sequence)
        if previous_by_sequence is not None and previous_by_sequence != source:
            raise ValueError("one event sequence identifies conflicting source events")
        by_id[source.event_id] = source
        by_sequence[source.event_sequence] = source

    for source in (*_attention_events(snapshot), event):
        add_full_event(source)

    admissions = tuple(
        cast(BeliefSubjectAdmission, belief.subject_admission)
        for belief in beliefs
        if belief.subject_admission is not None
    )
    for belief in beliefs:
        for revision in belief.revision_history:
            if revision.event_sequence is None:
                if revision.event_id is not None:
                    raise ValueError("Belief revision event identity is incomplete")
                continue
            if type(revision.event_id) is not str:
                raise ValueError("Belief revision event identity is incomplete")
            add_full_event(
                AttentionEvent(
                    _checked_identifier(revision.event_id, "Belief revision event_id"),
                    revision.event_sequence,
                    revision.created_at,
                )
            )

    admissions_by_id: dict[str, tuple[str, int]] = {}
    admissions_by_sequence: dict[int, tuple[str, int]] = {}
    for admission in admissions:
        identity = (admission.event_id, admission.event_sequence)
        prior_id = admissions_by_id.get(admission.event_id)
        if prior_id is not None and prior_id != identity:
            raise ValueError("one Belief admission event ID identifies different sequences")
        prior_sequence = admissions_by_sequence.get(admission.event_sequence)
        if prior_sequence is not None and prior_sequence != identity:
            raise ValueError("one Belief admission sequence identifies different event IDs")
        full_by_id = by_id.get(admission.event_id)
        if full_by_id is not None and (
            full_by_id.event_id,
            full_by_id.event_sequence,
        ) != identity:
            raise ValueError("Belief admission conflicts with a known full source event")
        full_by_sequence = by_sequence.get(admission.event_sequence)
        if full_by_sequence is not None and (
            full_by_sequence.event_id,
            full_by_sequence.event_sequence,
        ) != identity:
            raise ValueError("Belief admission conflicts with a known event sequence")
        admissions_by_id[admission.event_id] = identity
        admissions_by_sequence[admission.event_sequence] = identity


def _copy_working_memory_item(
    item: object,
    revision: int,
) -> WorkingMemoryItem:
    if type(item) is not WorkingMemoryItem:
        raise TypeError("item must be an exact WorkingMemoryItem")
    if type(item.item_id) is not str or len(item.item_id) > 67:
        raise ValueError("Working Memory item_id exceeds its identity bound")
    if type(item.source_kind) is not WorkingMemorySourceKind:
        raise TypeError("Working Memory source_kind must be exact")
    if type(item.source_id) is not str or len(item.source_id) > MAX_SOURCE_ID_BYTES:
        raise ValueError("Working Memory source_id exceeds its source bound")
    _validate_working_memory_item(item, revision)
    checked = cast(WorkingMemoryItem, item)
    return WorkingMemoryItem(
        item_id=checked.item_id,
        source_kind=checked.source_kind,
        source_id=checked.source_id,
        activation=checked.activation,
        salience=checked.salience,
        retention_reason=checked.retention_reason,
        created_revision=checked.created_revision,
        last_activated_revision=checked.last_activated_revision,
    )


def _copy_working_memory(
    items: object,
    revision: object,
) -> tuple[tuple[WorkingMemoryItem, ...], int, dict[str, WorkingMemoryItem]]:
    if type(revision) is not int or not 0 <= revision <= MAX_PERSISTED_REVISION:
        raise ValueError("working_memory_revision must be a bounded exact integer")
    checked_revision = cast(int, revision)
    if type(items) is not tuple:
        raise TypeError("working_memory_items must be an exact tuple")
    if len(items) > MAX_ITEM_CAPACITY:
        raise ValueError("Working Memory membership exceeds its source capacity")
    copied: list[WorkingMemoryItem] = []
    by_id: dict[str, WorkingMemoryItem] = {}
    source_ids: set[tuple[object, str]] = set()
    for item in items:
        current = _copy_working_memory_item(item, checked_revision)
        source_identity = (current.source_kind, current.source_id)
        if current.item_id in by_id or source_identity in source_ids:
            raise ValueError("Working Memory membership contains duplicate identities")
        copied.append(current)
        by_id[current.item_id] = current
        source_ids.add(source_identity)
    ordered = tuple(sorted(copied, key=lambda item: item.item_id))
    return ordered, checked_revision, by_id


def _relation_value(value: ContextRelation | None) -> str | None:
    if value is None:
        return None
    return value.value


def _bounded_utf8_size(value: object, byte_limit: int) -> int:
    if type(value) is not str:
        raise TypeError("Working Memory rendered content must be an exact string")
    if len(value) > byte_limit:
        raise ValueError("Working Memory rendered content exceeds projection_max_bytes")
    total = 0
    for offset in range(0, len(value), _UTF8_PREFLIGHT_CHUNK_CODEPOINTS):
        chunk = value[offset : offset + _UTF8_PREFLIGHT_CHUNK_CODEPOINTS]
        try:
            total += len(chunk.encode("utf-8"))
        except UnicodeEncodeError:
            raise ValueError("Working Memory rendered content is not valid UTF-8") from None
        if total > byte_limit:
            raise ValueError("Working Memory rendered content exceeds projection_max_bytes")
    return total


def _preflight_working_memory_view(
    value: object,
    revision: int,
) -> WorkingMemoryView:
    if type(value) is not WorkingMemoryView:
        raise TypeError("working_memory_view must be an exact WorkingMemoryView or None")
    checked_view = cast(WorkingMemoryView, value)
    if type(checked_view.selected) is not tuple or type(checked_view.decisions) is not tuple:
        raise TypeError("Working Memory view rows must be exact tuples")
    if (
        type(checked_view.item_capacity) is not int
        or not 1 <= checked_view.item_capacity <= MAX_ITEM_CAPACITY
    ):
        raise ValueError("Working Memory view item_capacity is invalid")
    if (
        type(checked_view.projection_max_bytes) is not int
        or not 1 <= checked_view.projection_max_bytes <= MAX_PROJECTION_BYTES
    ):
        raise ValueError("Working Memory view projection_max_bytes is invalid")
    if type(checked_view.projected_bytes) is not int or not 0 <= checked_view.projected_bytes <= checked_view.projection_max_bytes:
        raise ValueError("Working Memory view projected_bytes is invalid")
    if type(checked_view.revision) is not int or checked_view.revision != revision:
        raise ValueError("Working Memory view revision does not match the supplied revision")
    if len(checked_view.decisions) > checked_view.item_capacity or len(checked_view.selected) > checked_view.item_capacity:
        raise ValueError("Working Memory view exceeds its configured item capacity")
    if any(type(item) is not WorkingMemoryDecision for item in checked_view.decisions):
        raise TypeError("Working Memory decisions must have the exact source type")
    if any(type(item) is not WorkingMemorySelection for item in checked_view.selected):
        raise TypeError("Working Memory selections must have the exact source type")
    decisions = cast(tuple[WorkingMemoryDecision, ...], checked_view.decisions)
    selections = cast(tuple[WorkingMemorySelection, ...], checked_view.selected)

    view_rows: tuple[WorkingMemoryDecision | WorkingMemorySelection, ...] = (
        *decisions,
        *selections,
    )
    for row in view_rows:
        if (
            type(row.item_id) is not str
            or len(row.item_id) > 67
            or type(row.source_id) is not str
            or len(row.source_id) > MAX_SOURCE_ID_BYTES
            or type(row.source_kind) is not WorkingMemorySourceKind
        ):
            raise ValueError("Working Memory view row identity exceeds its source bound")
    for selection in selections:
        if selection.source_context_id is not None:
            _checked_identifier(selection.source_context_id, "Working Memory source_context_id")

    for selection in selections:
        if type(selection.rendered_content) is not str:
            raise TypeError("Working Memory rendered content must be an exact string")
        if len(selection.rendered_content) > checked_view.projection_max_bytes:
            raise ValueError("Working Memory rendered content exceeds projection_max_bytes")

    selected_bytes = 0
    for selection in selections:
        selected_bytes += _bounded_utf8_size(
            selection.rendered_content,
            checked_view.projection_max_bytes - selected_bytes,
        )
    if selected_bytes != checked_view.projected_bytes:
        raise ValueError("Working Memory projected_bytes does not match its selected content")
    return checked_view


def _working_memory_digest(
    items: tuple[WorkingMemoryItem, ...],
    revision: int,
    view: WorkingMemoryView | None,
    decisions: dict[str, WorkingMemoryDecision] | None,
    selections: dict[str, WorkingMemorySelection] | None,
) -> str:
    view_value: dict[str, object] | None = None
    if view is not None:
        assert decisions is not None and selections is not None
        view_value = {
            "decisions": [
                {
                    "context_compatibility": None
                    if decision.context_compatibility is None
                    else decision.context_compatibility.hex(),
                    "context_relation": _relation_value(decision.context_relation),
                    "effective_score": None
                    if decision.effective_score is None
                    else decision.effective_score.hex(),
                    "item_id": decision.item_id,
                    "reason": decision.reason.value,
                    "score": decision.score.hex(),
                    "selected": decision.selected,
                }
                for decision in sorted(decisions.values(), key=lambda item: item.item_id)
            ],
            "item_capacity": view.item_capacity,
            "projected_bytes": view.projected_bytes,
            "projection_max_bytes": view.projection_max_bytes,
            "revision": view.revision,
            "selected": [
                {
                    "context_compatibility": None
                    if selection.context_compatibility is None
                    else selection.context_compatibility.hex(),
                    "context_relation": _relation_value(selection.context_relation),
                    "effective_score": None
                    if selection.effective_score is None
                    else selection.effective_score.hex(),
                    "item_id": selection.item_id,
                    "reason": selection.reason.value,
                    "rendered_digest": hashlib.sha256(
                        selection.rendered_content.encode("utf-8")
                    ).hexdigest(),
                    "score": selection.score.hex(),
                }
                for selection in sorted(selections.values(), key=lambda item: item.item_id)
            ],
        }
    return digest_payload(
        _WORKING_MEMORY_OBSERVATION_DOMAIN,
        {
            "items": [
                {
                    "item_id": item.item_id,
                    "source_digest": _wm_state_digest(item, revision),
                }
                for item in items
            ],
            "revision": revision,
            "view": view_value,
        },
    )


def _copy_emotion(value: object) -> EmotionState:
    if type(value) is not EmotionState:
        raise TypeError("emotion_state must be an exact EmotionState or None")
    if any(
        type(getattr(value, name)) is not float
        for name in ("valence", "arousal", "optimal_loss")
    ):
        raise ValueError("Emotion state fields must remain exact floats")
    try:
        checked = replace(value)
    except Exception as error:
        raise ValueError("Emotion state is invalid") from error
    if (
        type(checked.valence) is not float
        or type(checked.arousal) is not float
        or type(checked.optimal_loss) is not float
        or not math.isfinite(checked.optimal_loss)
        or checked.valence != value.valence
        or checked.arousal != value.arousal
        or checked.optimal_loss != value.optimal_loss
    ):
        raise ValueError("Emotion state changed or failed strict revalidation")
    return checked


def _emotion_digest(state: EmotionState, event: AttentionEvent) -> str:
    return digest_payload(
        _EMOTION_OBSERVATION_DOMAIN,
        {
            "arousal": state.arousal.hex(),
            "event": event.canonical_value(),
            "valence": state.valence.hex(),
        },
    )


def _preflight_r13_reference(value: object, name: str) -> R13Reference:
    if type(value) is not R13Reference:
        raise TypeError(f"{name} must be an exact R13Reference")
    if type(value.kind) is not R13ReferenceKind:
        raise TypeError(f"{name}.kind must be exact")
    copied = R13Reference(value.kind, _checked_identifier(value.reference, f"{name}.reference"))
    if copied != value:
        raise ValueError(f"{name} is not already canonical")
    return copied


def _preflight_r13_references(
    value: object,
    name: str,
    maximum: int,
    *,
    allow_empty: bool = True,
) -> None:
    references = _preflight_exact_tuple(value, name, maximum)
    if not allow_empty and not references:
        raise ValueError(f"{name} must be non-empty")
    checked = tuple(
        _preflight_r13_reference(item, f"{name} item") for item in references
    )
    if checked != tuple(sorted(checked, key=lambda item: (item.reference, item.kind.value))):
        raise ValueError(f"{name} must already be canonically ordered")


def _preflight_identifier_tuple(value: object, name: str, maximum: int) -> None:
    identifiers = _preflight_exact_tuple(value, name, maximum)
    checked = tuple(_checked_identifier(item, name) for item in identifiers)
    if checked != tuple(sorted(set(checked))):
        raise ValueError(f"{name} must already be sorted and unique")


def _preflight_r13_revisions(
    value: object,
    name: str,
    revision_type: type[object],
) -> None:
    revisions = _preflight_exact_tuple(value, name, R13_MAX_REVISION_HISTORY)
    if any(type(item) is not revision_type for item in revisions):
        raise TypeError(f"{name} must contain exact source revision records")
    for item in revisions:
        _checked_digest(getattr(item, "record_digest"), f"{name} record_digest")
        references = _preflight_exact_tuple(
            getattr(item, "evidence_refs"),
            f"{name} evidence_refs",
            R13_MAX_REVISION_WITNESSES,
        )
        for reference in references:
            _checked_identifier(reference, f"{name} evidence_ref")
        event_id = getattr(item, "event_id")
        if event_id is not None:
            _checked_identifier(event_id, f"{name} event_id")
        previous = getattr(item, "previous_revision_digest")
        if previous is not None:
            _checked_digest(previous, f"{name} previous_revision_digest")


def _preflight_revision_anchor(value: object, name: str) -> None:
    if type(value) is not RevisionCompactionAnchor:
        raise TypeError(f"{name} must be exact")
    anchor = cast(RevisionCompactionAnchor, value)
    for field_name in (
        "through_state",
        "through_previous_state",
        "through_operation",
        "through_reason",
    ):
        field_value = getattr(anchor, field_name)
        if field_value is not None and (
            type(field_value) is not str or len(field_value) > 256
        ):
            raise ValueError(f"{name} {field_name} exceeds its source bound")
    references = _preflight_exact_tuple(
        anchor.through_evidence_refs,
        f"{name} through_evidence_refs",
        R13_MAX_REVISION_WITNESSES,
    )
    for reference in references:
        _checked_identifier(reference, f"{name} evidence_ref")
    _checked_identifier(anchor.authority_id, f"{name} authority_id")
    _checked_identifier(anchor.through_event_id, f"{name} through_event_id")
    for field_name in (
        "through_digest",
        "through_previous_revision_digest",
        "through_proposal_digest",
        "through_state_digest",
    ):
        field_value = getattr(anchor, field_name)
        if field_value is not None:
            _checked_digest(field_value, f"{name} {field_name}")
    try:
        copied = replace(anchor)
    except Exception as error:
        raise ValueError(f"{name} failed immutable source revalidation") from error
    if copied != anchor:
        raise ValueError(f"{name} changed or was normalized after publication")


def _preflight_motivation(record: MotivationRecord) -> None:
    _checked_identifier(record.motivation_id, "Motivation motivation_id")
    _checked_digest(record.record_digest, "Motivation record_digest")
    _preflight_r13_reference(record.target, "Motivation target")
    _preflight_r13_reference(record.source_evidence, "Motivation source_evidence")
    _preflight_r13_references(
        record.evidence_refs,
        "Motivation evidence_refs",
        R13_MAX_EVIDENCE_REFS,
        allow_empty=False,
    )
    _preflight_r13_references(
        record.conflict_refs,
        "Motivation conflict_refs",
        R13_MAX_CONFLICT_REFS,
    )
    _preflight_r13_references(
        record.related_refs,
        "Motivation related_refs",
        R13_MAX_RELATED_REFS,
    )
    _preflight_r13_revisions(
        record.revision_history,
        "Motivation revision_history",
        MotivationRevisionRecord,
    )
    if record.history_anchor is not None:
        _preflight_revision_anchor(record.history_anchor, "Motivation history_anchor")


def _preflight_goal(record: GoalRecord) -> None:
    _checked_identifier(record.goal_id, "Goal goal_id")
    _checked_digest(record.record_digest, "Goal record_digest")
    _checked_digest(record.proposal_digest, "Goal proposal_digest")
    _preflight_r13_reference(record.target, "Goal target")
    for name in ("origin_refs", "evidence_refs", "outcome_evidence_refs"):
        _preflight_r13_references(
            getattr(record, name),
            f"Goal {name}",
            R13_MAX_EVIDENCE_REFS,
            allow_empty=name != "origin_refs" and name != "evidence_refs",
        )
    _preflight_identifier_tuple(record.dependencies, "Goal dependencies", R13_MAX_DEPENDENCY_REFS)
    _preflight_identifier_tuple(record.conflicts, "Goal conflicts", R13_MAX_CONFLICT_REFS)
    if type(record.description) is not str or len(record.description) > 1_024:
        raise ValueError("Goal description exceeds its source bound")
    if record.subject_admission is not None and type(record.subject_admission) is not GoalSubjectAdmission:
        raise TypeError("Goal subject_admission must be exact")
    if record.subject_admission is not None:
        _checked_identifier(record.subject_admission.goal_id, "Goal admission goal_id")
        _checked_digest(record.subject_admission.proposal_digest, "Goal admission proposal_digest")
        _checked_digest(record.subject_admission.admission_digest, "Goal admission digest")
        admission_refs = _preflight_exact_tuple(
            record.subject_admission.evidence_refs,
            "Goal admission evidence_refs",
            R13_MAX_EVIDENCE_REFS,
        )
        for reference in admission_refs:
            _checked_identifier(reference, "Goal admission evidence_ref")
        _checked_identifier(record.subject_admission.event_id, "Goal admission event_id")
        admission = record.subject_admission
        declared_admission_digest = admission.admission_digest
        try:
            checked_admission = replace(admission)
        except Exception as error:
            raise ValueError("Goal subject admission is invalid") from error
        if (
            checked_admission.admission_digest != declared_admission_digest
            or checked_admission != admission
        ):
            raise ValueError("Goal subject admission changed after publication")
    proofs = _preflight_exact_tuple(
        record.subject_transition_proofs,
        "Goal subject_transition_proofs",
        R13_MAX_REVISION_HISTORY + 1,
    )
    if any(type(proof) is not GoalSubjectTransitionProof for proof in proofs):
        raise TypeError("Goal transition proofs must be exact")
    for proof in proofs:
        proof_value = cast(GoalSubjectTransitionProof, proof)
        _checked_digest(proof_value.proposal_digest, "Goal transition proposal_digest")
        _checked_digest(proof_value.transition_digest, "Goal transition digest")
        proof_refs = _preflight_exact_tuple(
            proof_value.evidence_refs,
            "Goal transition evidence_refs",
            R13_MAX_EVIDENCE_REFS,
        )
        for reference in proof_refs:
            _checked_identifier(reference, "Goal transition evidence_ref")
        _checked_identifier(proof_value.event_id, "Goal transition event_id")
        declared_transition_digest = proof_value.transition_digest
        try:
            checked_proof = replace(proof_value)
        except Exception as error:
            raise ValueError("Goal transition proof is invalid") from error
        if (
            checked_proof.transition_digest != declared_transition_digest
            or checked_proof != proof_value
        ):
            raise ValueError("Goal transition proof changed after publication")
    _preflight_r13_revisions(
        record.revision_history,
        "Goal revision_history",
        GoalRevisionRecord,
    )
    if record.history_anchor is not None:
        _preflight_revision_anchor(record.history_anchor, "Goal history_anchor")


def _preflight_commitment(record: CommitmentRecord) -> None:
    _checked_identifier(record.commitment_id, "Commitment commitment_id")
    _checked_digest(record.record_digest, "Commitment record_digest")
    _checked_digest(record.proposal_digest, "Commitment proposal_digest")
    _preflight_r13_reference(record.beneficiary, "Commitment beneficiary")
    for name in ("evidence_refs", "origin_refs", "outcome_evidence_refs"):
        _preflight_r13_references(
            getattr(record, name),
            f"Commitment {name}",
            R13_MAX_EVIDENCE_REFS,
            allow_empty=name == "outcome_evidence_refs",
        )
    _preflight_r13_references(
        record.related_goal_refs,
        "Commitment related_goal_refs",
        R13_MAX_RELATED_REFS,
    )
    _preflight_r13_references(
        record.desire_refs,
        "Commitment desire_refs",
        R13_MAX_RELATED_REFS,
    )
    scope = _preflight_exact_tuple(record.scope, "Commitment scope", R13_MAX_SCOPE_ITEMS)
    if not scope or any(type(item) is not str or len(item) > R13_MAX_SCOPE_CODEPOINTS for item in scope):
        raise ValueError("Commitment scope exceeds its source bound")
    if type(record.subject) is not str or len(record.subject) > 1_024:
        raise ValueError("Commitment subject exceeds its source bound")
    if record.subject_admission is not None and type(record.subject_admission) is not CommitmentSubjectAdmission:
        raise TypeError("Commitment subject_admission must be exact")
    if record.subject_admission is not None:
        _checked_identifier(record.subject_admission.commitment_id, "Commitment admission id")
        _checked_digest(record.subject_admission.proposal_digest, "Commitment admission proposal_digest")
        _checked_digest(record.subject_admission.admission_digest, "Commitment admission digest")
        admission_refs = _preflight_exact_tuple(
            record.subject_admission.evidence_refs,
            "Commitment admission evidence_refs",
            R13_MAX_EVIDENCE_REFS,
        )
        for reference in admission_refs:
            _checked_identifier(reference, "Commitment admission evidence_ref")
        _checked_identifier(record.subject_admission.event_id, "Commitment admission event_id")
        admission = record.subject_admission
        declared_admission_digest = admission.admission_digest
        try:
            checked_admission = replace(admission)
        except Exception as error:
            raise ValueError("Commitment subject admission is invalid") from error
        if (
            checked_admission.admission_digest != declared_admission_digest
            or checked_admission != admission
        ):
            raise ValueError("Commitment subject admission changed after publication")
    proofs = _preflight_exact_tuple(
        record.subject_transition_proofs,
        "Commitment subject_transition_proofs",
        R13_MAX_REVISION_HISTORY + 1,
    )
    if any(type(proof) is not CommitmentSubjectTransitionProof for proof in proofs):
        raise TypeError("Commitment transition proofs must be exact")
    for proof in proofs:
        proof_value = cast(CommitmentSubjectTransitionProof, proof)
        _checked_identifier(proof_value.commitment_id, "Commitment transition id")
        _checked_digest(proof_value.proposal_digest, "Commitment transition proposal_digest")
        _checked_digest(proof_value.transition_digest, "Commitment transition digest")
        proof_refs = _preflight_exact_tuple(
            proof_value.evidence_refs,
            "Commitment transition evidence_refs",
            R13_MAX_EVIDENCE_REFS,
        )
        for reference in proof_refs:
            _checked_identifier(reference, "Commitment transition evidence_ref")
        _checked_identifier(proof_value.event_id, "Commitment transition event_id")
        declared_transition_digest = proof_value.transition_digest
        try:
            checked_proof = replace(proof_value)
        except Exception as error:
            raise ValueError("Commitment transition proof is invalid") from error
        if (
            checked_proof.transition_digest != declared_transition_digest
            or checked_proof != proof_value
        ):
            raise ValueError("Commitment transition proof changed after publication")
    _preflight_r13_revisions(
        record.revision_history,
        "Commitment revision_history",
        CommitmentRevisionRecord,
    )
    if record.history_anchor is not None:
        _preflight_revision_anchor(record.history_anchor, "Commitment history_anchor")


def _copy_motivation(record: object) -> MotivationRecord:
    if type(record) is not MotivationRecord:
        raise TypeError("focus_records must contain exact R13 record types")
    _preflight_motivation(record)
    declared_digest = record.record_digest
    try:
        checked = replace(record)
        computed_digest = motivation_record_digest(checked)
    except Exception as error:
        raise ValueError("Motivation source record is invalid") from error
    if checked.record_digest != declared_digest or computed_digest != declared_digest:
        raise ValueError("Motivation source record digest does not match its current state")
    if checked.lifecycle is not MotivationLifecycle.ACTIVE:
        raise ValueError("focus_records may contain only current ACTIVE Motivations")
    return checked


def _copy_goal(record: object) -> GoalRecord:
    if type(record) is not GoalRecord:
        raise TypeError("focus_records must contain exact R13 record types")
    _preflight_goal(record)
    declared_digest = record.record_digest
    try:
        checked = replace(record)
        computed_digest = goal_record_digest(checked)
    except Exception as error:
        raise ValueError("Goal source record is invalid") from error
    if checked.record_digest != declared_digest or computed_digest != declared_digest:
        raise ValueError("Goal source record digest does not match its current state")
    if checked.lifecycle is not GoalLifecycle.ADOPTED:
        raise ValueError("focus_records may contain only current ADOPTED Goals")
    return checked


def _copy_commitment(record: object) -> CommitmentRecord:
    if type(record) is not CommitmentRecord:
        raise TypeError("focus_records must contain exact R13 record types")
    _preflight_commitment(record)
    declared_digest = record.record_digest
    try:
        checked = replace(record)
        computed_digest = commitment_record_digest(checked)
    except Exception as error:
        raise ValueError("Commitment source record is invalid") from error
    if checked.record_digest != declared_digest or computed_digest != declared_digest:
        raise ValueError("Commitment source record digest does not match its current state")
    if checked.lifecycle is not CommitmentLifecycle.ACTIVE:
        raise ValueError("focus_records may contain only current ACTIVE Commitments")
    return checked


def _copy_belief_evidence(item: object) -> BeliefEvidence:
    if type(item) is not BeliefEvidence:
        raise TypeError("Belief evidence must contain exact BeliefEvidence rows")
    return BeliefEvidence(item.evidence_ref, item.evidence_type)


def _preflight_exact_tuple(value: object, name: str, maximum: int) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")
    if len(value) > maximum:
        raise ValueError(f"{name} exceeds its source bound")
    return cast(tuple[object, ...], value)


def _preflight_belief(record: BeliefRecord) -> None:
    _checked_identifier(record.belief_id, "Belief belief_id")
    evidence = _preflight_exact_tuple(record.evidence, "Belief evidence", BELIEF_MAX_EVIDENCE)
    if any(type(item) is not BeliefEvidence for item in evidence):
        raise TypeError("Belief evidence must contain exact BeliefEvidence rows")
    for item in evidence:
        evidence_item = cast(BeliefEvidence, item)
        if type(evidence_item.evidence_type) is not BeliefEvidenceType:
            raise TypeError("Belief evidence_type must be exact")
        _checked_identifier(evidence_item.evidence_ref, "Belief evidence_ref")
    evidence_refs = tuple(cast(BeliefEvidence, item).evidence_ref for item in evidence)
    if len(set(evidence_refs)) != len(evidence_refs):
        raise ValueError("Belief evidence references must be unique")
    canonical_evidence = tuple(
        sorted(
            (cast(BeliefEvidence, item) for item in evidence),
            key=lambda item: (item.evidence_ref, item.evidence_type.value),
        )
    )
    if tuple(evidence) != canonical_evidence:
        raise ValueError("Belief evidence must already be canonically ordered")

    context_scope = _preflight_exact_tuple(
        record.context_scope,
        "Belief context_scope",
        BELIEF_MAX_CONTEXTS,
    )
    checked_contexts = tuple(
        _checked_identifier(item, "Belief context_scope item") for item in context_scope
    )
    if checked_contexts != tuple(sorted(set(checked_contexts))):
        raise ValueError("Belief context_scope must already be sorted and unique")

    revisions = _preflight_exact_tuple(
        record.revision_history,
        "Belief revision_history",
        BELIEF_MAX_REVISIONS,
    )
    if any(type(item) is not BeliefRevisionRecord for item in revisions):
        raise TypeError("Belief revision_history must contain exact revision records")
    for item in revisions:
        revision = cast(BeliefRevisionRecord, item)
        _checked_digest(revision.record_digest, "Belief revision record_digest")
        _preflight_exact_tuple(
            revision.evidence_refs,
            "Belief revision evidence_refs",
            BELIEF_MAX_REVISION_WITNESSES,
        )
        for reference in revision.evidence_refs:
            _checked_identifier(reference, "Belief revision evidence_ref")
        if revision.event_id is not None:
            _checked_identifier(revision.event_id, "Belief revision event_id")
        if revision.previous_revision_digest is not None:
            _checked_digest(
                revision.previous_revision_digest,
                "Belief revision previous_revision_digest",
            )

    admission = record.subject_admission
    if admission is not None:
        if type(admission) is not BeliefSubjectAdmission:
            raise TypeError("Belief subject_admission must be exact")
        _checked_digest(admission.proposition_digest, "Belief admission proposition_digest")
        _checked_digest(admission.admission_digest, "Belief admission digest")
        admission_refs = _preflight_exact_tuple(
            admission.evidence_refs,
            "Belief admission evidence_refs",
            BELIEF_MAX_EVIDENCE,
        )
        for admission_reference in admission_refs:
            _checked_identifier(admission_reference, "Belief admission evidence_ref")
        _checked_identifier(admission.event_id, "Belief admission event_id")

    if record.supersedes_id is not None:
        _checked_identifier(record.supersedes_id, "Belief supersedes_id")
    if record.superseded_by_id is not None:
        _checked_identifier(record.superseded_by_id, "Belief superseded_by_id")
    if record.history_anchor_digest is not None:
        _checked_digest(record.history_anchor_digest, "Belief history_anchor_digest")

    proposition = record.proposition
    if type(proposition) is not BeliefProposition:
        raise TypeError("Belief proposition must be an exact BeliefProposition")
    if type(proposition.canonical_text) is not str or len(
        proposition.canonical_text
    ) > BELIEF_MAX_PROPOSITION_CODEPOINTS:
        raise ValueError("Belief proposition text exceeds its source bound")
    _checked_digest(proposition.proposition_digest, "Belief proposition_digest")
    for name in ("subject", "predicate", "object"):
        value = getattr(proposition, name)
        if value is not None and (
            type(value) is not str or len(value) > BELIEF_MAX_COMPONENT_CODEPOINTS
        ):
            raise ValueError("Belief structured component exceeds its source bound")


def _copy_belief_revision(item: object) -> BeliefRevisionRecord:
    if type(item) is not BeliefRevisionRecord:
        raise TypeError("Belief revision history must contain exact revision records")
    declared_digest = item.record_digest
    try:
        checked = replace(item)
    except Exception as error:
        raise ValueError("Belief revision record is invalid") from error
    if checked.record_digest != declared_digest:
        raise ValueError("Belief revision digest does not match its current state")
    validate_revision_digest(checked)
    return checked


def _copy_belief(record: object) -> BeliefRecord:
    if type(record) is not BeliefRecord:
        raise TypeError("belief_records must contain exact BeliefRecord values")
    try:
        _preflight_belief(record)
        validate_proposition_digest(record.proposition)
        proposition = replace(record.proposition)
        evidence = tuple(_copy_belief_evidence(item) for item in record.evidence)
        revisions = tuple(_copy_belief_revision(item) for item in record.revision_history)
        admission = record.subject_admission
        if admission is not None:
            declared_admission_digest = admission.admission_digest
            admission = replace(admission)
            if admission.admission_digest != declared_admission_digest:
                raise ValueError("Belief admission digest does not match its current state")
        checked = replace(
            record,
            proposition=proposition,
            evidence=evidence,
            revision_history=revisions,
            subject_admission=admission,
        )
        checked_digest = belief_record_digest(checked)
        source_digest = belief_record_digest(record)
    except Exception as error:
        raise ValueError("Belief source record is invalid or changed after publication") from error
    if checked_digest != source_digest:
        raise ValueError("Belief source record changed while making its detached copy")
    if type(checked.evidence) is not tuple or len(checked.evidence) > BELIEF_MAX_EVIDENCE:
        raise ValueError("Belief evidence exceeds its source bound")
    if len(checked.context_scope) > BELIEF_MAX_CONTEXTS:
        raise ValueError("Belief context scope exceeds its source bound")
    return checked


def _project_current_r13(
    record: _R13FocusRecord,
    event: AttentionEvent,
) -> tuple[str, AttentionSourceWitness]:
    if type(record) is MotivationRecord:
        projection = project_motivation(record, event=event)
        return projection.candidate_id, projection.source
    if type(record) is GoalRecord:
        projection = project_goal(record, event=event)
        return projection.candidate_id, projection.source
    if type(record) is CommitmentRecord:
        projection = project_commitment(record, event=event)
        return projection.candidate_id, projection.source
    raise TypeError("focus_records must contain exact R13 record types")


def _belief_links(record: _R13FocusRecord) -> tuple[str, ...]:
    references: list[str] = []
    target = getattr(record, "target", None)
    if target is not None:
        if type(target) is not R13Reference:
            raise TypeError("R13 target must be an exact R13Reference")
        checked_target = cast(R13Reference, target)
        if type(checked_target.kind) is not R13ReferenceKind:
            raise TypeError("R13 target kind must be exact")
        if checked_target.kind is R13ReferenceKind.BELIEF:
            references.append(_checked_identifier(checked_target.reference, "R13 Belief target"))
    for supplied_reference in record.evidence_refs:
        if type(supplied_reference) is not R13Reference:
            raise TypeError("R13 evidence must contain exact R13Reference values")
        reference = cast(R13Reference, supplied_reference)
        if type(reference.kind) is not R13ReferenceKind:
            raise TypeError("R13 evidence kind must be exact")
        if reference.kind is R13ReferenceKind.BELIEF:
            references.append(_checked_identifier(reference.reference, "R13 Belief evidence"))
    if type(record) is MotivationRecord:
        for supplied_reference in record.related_refs:
            if type(supplied_reference) is not R13Reference:
                raise TypeError("Motivation related_refs must contain exact R13Reference values")
            reference = cast(R13Reference, supplied_reference)
            if type(reference.kind) is not R13ReferenceKind:
                raise TypeError("Motivation related reference kind must be exact")
            if reference.kind is R13ReferenceKind.BELIEF:
                references.append(_checked_identifier(reference.reference, "Motivation related Belief ref"))
    return tuple(sorted(set(references)))


def _copy_focus_records(
    values: object,
    snapshot: AttentionContinuity,
    event: AttentionEvent,
) -> tuple[tuple[_R13FocusRecord, ...], dict[str, set[str]]]:
    if type(values) is not tuple:
        raise TypeError("focus_records must be an exact tuple")
    if len(values) > METACOGNITION_MAX_FOCUS_RECORDS:
        raise ValueError("focus_records exceeds the current-focus bound")
    focused = set(snapshot.focused_ids)
    candidate_by_id = {item.candidate_id: item for item in snapshot.candidates}
    seen_candidates: set[str] = set()
    copied: list[_R13FocusRecord] = []
    links: dict[str, set[str]] = {}
    for value in values:
        if type(value) is MotivationRecord:
            record: _R13FocusRecord = _copy_motivation(value)
        elif type(value) is GoalRecord:
            record = _copy_goal(value)
        elif type(value) is CommitmentRecord:
            record = _copy_commitment(value)
        else:
            raise TypeError("focus_records must contain exact Motivation/Goal/Commitment records")
        candidate_id, projected_source = _project_current_r13(record, event)
        if candidate_id not in focused:
            raise ValueError("focus_records contains a non-focused R13 source record")
        if candidate_id in seen_candidates:
            raise ValueError("focus_records contains a duplicate focused R13 record")
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None or candidate.source.canonical_value() != projected_source.canonical_value():
            raise ValueError("R13 source record is stale relative to current Attention focus")
        seen_candidates.add(candidate_id)
        copied.append(record)
        for belief_id in _belief_links(record):
            links.setdefault(belief_id, set()).add(candidate_id)
    return tuple(copied), links


def _latest_belief_source_event(
    record: BeliefRecord,
    event: AttentionEvent,
) -> AttentionEvent | None:
    if not record.revision_history:
        return None
    latest = record.revision_history[-1]
    event_sequence = latest.event_sequence
    if latest.created_at > event.occurred_at:
        raise ValueError("Belief latest revision is in the future of the observation event")
    if event_sequence is None:
        if latest.event_id is not None:
            raise ValueError("Belief latest revision event identity is incomplete")
        return None
    if type(latest.event_id) is not str:
        raise ValueError("Belief latest revision event identity is incomplete")
    source_event = AttentionEvent(
        _checked_identifier(latest.event_id, "Belief latest revision event_id"),
        event_sequence,
        latest.created_at,
    )
    _event_not_future(source_event, event)
    return source_event


def _validate_belief_revision_history(record: BeliefRecord) -> None:
    """Recheck provider-free R12 retained chronology and lifecycle transitions."""

    history = record.revision_history
    if not history:
        return
    reasons = {
        BeliefRevisionOperation.CREATE: BeliefRevisionReason.CREATION,
        BeliefRevisionOperation.ADOPT: BeliefRevisionReason.SUBJECT_ADMISSION,
        BeliefRevisionOperation.CORRECT: BeliefRevisionReason.CORRECTION,
        BeliefRevisionOperation.SUPERSEDE: BeliefRevisionReason.SUPERSESSION,
        BeliefRevisionOperation.RETRACT: BeliefRevisionReason.RETRACTION,
        BeliefRevisionOperation.EXPIRE: BeliefRevisionReason.EXPIRATION,
    }
    transitions = {
        BeliefLifecycle.PROPOSED: {
            BeliefRevisionOperation.ADOPT: BeliefLifecycle.ADOPTED,
            BeliefRevisionOperation.SUPERSEDE: BeliefLifecycle.SUPERSEDED,
            BeliefRevisionOperation.RETRACT: BeliefLifecycle.RETRACTED,
            BeliefRevisionOperation.EXPIRE: BeliefLifecycle.EXPIRED,
        },
        BeliefLifecycle.ADOPTED: {
            BeliefRevisionOperation.CORRECT: BeliefLifecycle.ADOPTED,
            BeliefRevisionOperation.SUPERSEDE: BeliefLifecycle.SUPERSEDED,
            BeliefRevisionOperation.RETRACT: BeliefLifecycle.RETRACTED,
            BeliefRevisionOperation.EXPIRE: BeliefLifecycle.EXPIRED,
        },
        BeliefLifecycle.SUPERSEDED: {},
        BeliefLifecycle.RETRACTED: {},
        BeliefLifecycle.EXPIRED: {},
    }
    initial_state = {
        BeliefRevisionOperation.CREATE: BeliefLifecycle.PROPOSED,
        BeliefRevisionOperation.ADOPT: BeliefLifecycle.ADOPTED,
        BeliefRevisionOperation.CORRECT: BeliefLifecycle.ADOPTED,
        BeliefRevisionOperation.SUPERSEDE: BeliefLifecycle.SUPERSEDED,
        BeliefRevisionOperation.RETRACT: BeliefLifecycle.RETRACTED,
        BeliefRevisionOperation.EXPIRE: BeliefLifecycle.EXPIRED,
    }
    state: BeliefLifecycle | None = None
    previous: BeliefRevisionRecord | None = None
    for revision in history:
        if revision.reason is not reasons[revision.operation]:
            raise ValueError("Belief revision operation and reason disagree")
        if previous is None:
            if revision.revision == 0 and revision.operation is not BeliefRevisionOperation.CREATE:
                raise ValueError("Belief history must begin with creation")
            if revision.revision > 0 and revision.operation is BeliefRevisionOperation.CREATE:
                raise ValueError("compacted Belief history cannot recreate a record")
            state = initial_state[revision.operation]
        else:
            if revision.created_at < previous.created_at:
                raise ValueError("Belief revision timestamps must be nondecreasing")
            prior_sequence = previous.event_sequence
            current_sequence = revision.event_sequence
            if prior_sequence is not None and current_sequence is not None:
                if current_sequence < prior_sequence:
                    raise ValueError("Belief revision event sequences must be nondecreasing")
                if current_sequence == prior_sequence:
                    shared_create_adoption = (
                        previous.operation is BeliefRevisionOperation.CREATE
                        and revision.operation is BeliefRevisionOperation.ADOPT
                        and previous.event_id == revision.event_id
                        and previous.created_at == revision.created_at
                    )
                    if not shared_create_adoption:
                        raise ValueError("only exact creation/adoption event reuse is permitted")
                elif previous.event_id is not None and revision.event_id == previous.event_id:
                    raise ValueError("Belief revision event IDs cannot be reused")
            if state is None:
                raise AssertionError("Belief retained lifecycle state was lost")
            next_state = transitions[state].get(revision.operation)
            if next_state is None:
                raise ValueError("Belief revision history contains an invalid lifecycle transition")
            state = next_state
        previous = revision
    if state is not record.lifecycle:
        raise ValueError("Belief revision history does not reach the current lifecycle")
    if record.lifecycle in {
        BeliefLifecycle.ADOPTED,
        BeliefLifecycle.SUPERSEDED,
        BeliefLifecycle.RETRACTED,
        BeliefLifecycle.EXPIRED,
    }:
        admission = record.subject_admission
        if admission is None:
            raise ValueError("current Belief history requires its subject admission")
        latest = history[-1]
        if admission.admission_digest not in latest.evidence_refs:
            raise ValueError("latest Belief revision does not bind its subject admission")
        if latest.event_id is not None or latest.event_sequence is not None:
            if (
                latest.event_id is None
                or latest.event_sequence is None
                or (admission.event_id, admission.event_sequence)
                != (latest.event_id, latest.event_sequence)
            ):
                raise ValueError("latest Belief revision does not match its subject admission event")


def _validate_belief_not_future(
    record: BeliefRecord,
    event: AttentionEvent,
) -> AttentionEvent | None:
    admission = record.subject_admission
    if admission is not None:
        if admission.event_sequence > event.event_sequence:
            raise ValueError("Belief admission is in the future of the observation event")
        if (
            admission.event_sequence == event.event_sequence
            and admission.event_id != event.event_id
        ):
            raise ValueError("equal Belief admission sequences must identify the same event")
        if (
            admission.event_id == event.event_id
            and admission.event_sequence != event.event_sequence
        ):
            raise ValueError("Belief admission event ID was reused at a different sequence")
    _validate_belief_revision_history(record)
    for revision in record.revision_history:
        if revision.created_at > event.occurred_at:
            raise ValueError("Belief revision is in the future of the observation event")
        if revision.event_sequence is not None:
            if type(revision.event_id) is not str:
                raise ValueError("Belief revision event identity is incomplete")
            revision_event = AttentionEvent(
                _checked_identifier(revision.event_id, "Belief revision event_id"),
                revision.event_sequence,
                revision.created_at,
            )
            _event_not_future(revision_event, event)
    return _latest_belief_source_event(record, event)


def _belief_witness(
    record: BeliefRecord,
    *,
    condition: EvidenceCondition,
    confidence_ceiling: float | None,
    source_event: AttentionEvent | None,
) -> MetacognitiveEvidenceWitness:
    return MetacognitiveEvidenceWitness(
        source_kind=AttentionSourceKind.BELIEF,
        reference=record.belief_id,
        source_revision=record.revision,
        digest=belief_record_digest(record),
        digest_kind=SourceDigestKind.UPSTREAM_AUTHORITY,
        condition=condition,
        confidence_ceiling=confidence_ceiling,
        source_event=source_event,
        source_event_origin=(
            None if source_event is None else SourceEventOrigin.UPSTREAM_EVENT
        ),
    )


def _copy_belief_records(
    values: object,
    event: AttentionEvent,
    *,
    provided_links: dict[str, set[str]],
) -> tuple[tuple[BeliefRecord, ...], dict[str, AttentionEvent | None]]:
    if type(values) is not tuple:
        raise TypeError("belief_records must be an exact tuple or None")
    if len(values) > METACOGNITION_MAX_BELIEF_RECORDS:
        raise ValueError("belief_records exceeds its bounded source count")
    copied: list[BeliefRecord] = []
    source_events: dict[str, AttentionEvent | None] = {}
    seen_ids: set[str] = set()
    for value in values:
        record = _copy_belief(value)
        if record.belief_id in seen_ids:
            raise ValueError("belief_records contains duplicate Belief identities")
        if record.belief_id not in provided_links:
            raise ValueError("belief_records contains a record unrelated to focused R13 links")
        seen_ids.add(record.belief_id)
        source_events[record.belief_id] = _validate_belief_not_future(record, event)
        copied.append(record)
    return tuple(copied), source_events


def _is_ordinary_active(
    record: BeliefRecord,
    event: AttentionEvent,
    current_context_id: str | None,
) -> bool:
    return record.is_ordinary_active(
        at=event.occurred_at,
        context_id=current_context_id,
    )


def _contradictory_belief_ids(
    records: tuple[BeliefRecord, ...],
    links: dict[str, set[str]],
    event: AttentionEvent,
    current_context_id: str | None,
) -> set[str]:
    contradictory: set[str] = set()
    for index, left in enumerate(records):
        if not _is_ordinary_active(left, event, current_context_id):
            continue
        for right in records[index + 1 :]:
            if not _is_ordinary_active(right, event, current_context_id):
                continue
            if not (links[left.belief_id] & links[right.belief_id]):
                continue
            candidate = build_conflict_candidate(
                left.proposition,
                right.proposition,
                left.context_scope,
                right.context_scope,
            )
            if candidate is not None:
                contradictory.update((left.belief_id, right.belief_id))
    return contradictory


def _fixed_units(value: float) -> int:
    numerator, denominator = value.as_integer_ratio()
    return numerator * METACOGNITION_UNITS_SCALE // denominator


def _make_reason_codes(values: set[MetacognitiveReasonCode]) -> tuple[MetacognitiveReasonCode, ...]:
    result = tuple(sorted(values, key=lambda code: code.value))
    if len(result) > METACOGNITION_MAX_REASON_CODES:
        raise ValueError("observation reason codes exceed the metacognition bound")
    return result


def _copy_focus_witness(value: FocusAssessmentWitness) -> FocusAssessmentWitness:
    if type(value) is not FocusAssessmentWitness:
        raise TypeError("focus_witness must be an exact FocusAssessmentWitness")
    return FocusAssessmentWitness(
        attention_revision=value.attention_revision,
        attention_state_digest=value.attention_state_digest,
        focused_ids=tuple(value.focused_ids),
        event=_copy_event(value.event, "focus_witness.event"),
    )


def _copy_evidence_witness(value: object) -> MetacognitiveEvidenceWitness:
    if type(value) is not MetacognitiveEvidenceWitness:
        raise TypeError("evidence must contain exact MetacognitiveEvidenceWitness values")
    _checked_identifier(value.reference, "evidence reference")
    declared_digest = value.witness_digest
    checked = MetacognitiveEvidenceWitness(
        source_kind=value.source_kind,
        reference=value.reference,
        source_revision=value.source_revision,
        digest=value.digest,
        digest_kind=value.digest_kind,
        condition=value.condition,
        confidence_ceiling=value.confidence_ceiling,
        source_event=None
        if value.source_event is None
        else _copy_event(value.source_event, "evidence.source_event"),
        source_event_origin=value.source_event_origin,
    )
    if checked.witness_digest != declared_digest:
        raise ValueError("evidence witness digest does not match its published fields")
    return checked


def _units(value: object, name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or not 0 <= value <= METACOGNITION_UNITS_SCALE:
        raise ValueError(f"{name} must be an exact fixed-point integer or None")
    return value


def _observation_payload(observation: MetacognitionObservation) -> dict[str, object]:
    return {
        "attention_saturation_units": observation.attention_saturation_units,
        "belief_confidence_ceiling_units": observation.belief_confidence_ceiling_units,
        "belief_coverage_units": observation.belief_coverage_units,
        "cognitive_load_units": observation.cognitive_load_units,
        "contradictory": observation.contradictory,
        "emotion_influence_units": observation.emotion_influence_units,
        "event": observation.event.canonical_value(),
        "evidence": [item.canonical_value() for item in observation.evidence],
        "focus_count": observation.focus_count,
        "focus_witness": observation.focus_witness.canonical_value(),
        "reason_codes": [code.value for code in observation.reason_codes],
    }


@dataclass(frozen=True, slots=True, init=False)
class MetacognitionObservation:
    """Sealed numeric and typed-provenance facts observed for one event."""

    event: AttentionEvent
    focus_witness: FocusAssessmentWitness
    evidence: tuple[MetacognitiveEvidenceWitness, ...]
    focus_count: int
    cognitive_load_units: int | None
    attention_saturation_units: int | None
    emotion_influence_units: int | None
    belief_coverage_units: int | None
    belief_confidence_ceiling_units: int | None
    contradictory: bool
    reason_codes: tuple[MetacognitiveReasonCode, ...]
    observation_digest: str = field(init=False)

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("MetacognitionObservation values are created by observe_metacognition")

    @classmethod
    def _create(
        cls,
        *,
        event: AttentionEvent,
        focus_witness: FocusAssessmentWitness,
        evidence: tuple[MetacognitiveEvidenceWitness, ...],
        focus_count: int,
        cognitive_load_units: int | None,
        attention_saturation_units: int | None,
        emotion_influence_units: int | None,
        belief_coverage_units: int | None,
        belief_confidence_ceiling_units: int | None,
        contradictory: bool,
        reason_codes: tuple[MetacognitiveReasonCode, ...],
    ) -> MetacognitionObservation:
        if cls is not MetacognitionObservation:
            raise TypeError("MetacognitionObservation cannot be subclassed")
        if type(event) is not AttentionEvent:
            raise TypeError("event must be an exact AttentionEvent")
        if type(focus_witness) is not FocusAssessmentWitness:
            raise TypeError("focus_witness must be an exact FocusAssessmentWitness")
        if type(focus_witness.focused_ids) is not tuple:
            raise TypeError("focus_witness.focused_ids must be an exact tuple")
        if len(focus_witness.focused_ids) > ATTENTION_MAX_FOCUS:
            raise ValueError("focus_witness exceeds the Attention focus bound")
        if type(evidence) is not tuple:
            raise TypeError("evidence must be an exact tuple")
        if len(evidence) > METACOGNITION_MAX_EVIDENCE_WITNESSES:
            raise ValueError("evidence exceeds its witness bound")
        if any(type(item) is not MetacognitiveEvidenceWitness for item in evidence):
            raise TypeError("evidence must contain exact MetacognitiveEvidenceWitness values")
        if type(reason_codes) is not tuple:
            raise TypeError("reason_codes must be an exact tuple")
        if len(reason_codes) > METACOGNITION_MAX_REASON_CODES:
            raise ValueError("reason_codes exceeds its bound")
        current_event = _copy_event(event)
        current_focus = _copy_focus_witness(focus_witness)
        copied_evidence = tuple(_copy_evidence_witness(item) for item in evidence)
        result = object.__new__(cls)
        for name, value in (
            ("event", current_event),
            ("focus_witness", current_focus),
            ("evidence", copied_evidence),
            ("focus_count", focus_count),
            ("cognitive_load_units", cognitive_load_units),
            ("attention_saturation_units", attention_saturation_units),
            ("emotion_influence_units", emotion_influence_units),
            ("belief_coverage_units", belief_coverage_units),
            ("belief_confidence_ceiling_units", belief_confidence_ceiling_units),
            ("contradictory", contradictory),
            ("reason_codes", tuple(reason_codes)),
        ):
            object.__setattr__(result, name, value)
        result.__post_init__()
        object.__setattr__(
            result,
            "observation_digest",
            digest_payload(_OBSERVATION_DOMAIN, _observation_payload(result)),
        )
        return result

    def __post_init__(self) -> None:
        if type(self.event) is not AttentionEvent:
            raise TypeError("event must be an exact AttentionEvent")
        self.event.__post_init__()
        if type(self.focus_witness) is not FocusAssessmentWitness:
            raise TypeError("focus_witness must be an exact FocusAssessmentWitness")
        self.focus_witness.__post_init__()
        if self.focus_witness.event != self.event:
            raise ValueError("focus witness must bind the exact observation event")
        if type(self.focus_count) is not int or self.focus_count != len(
            self.focus_witness.focused_ids
        ):
            raise ValueError("focus_count must match the bound Attention focus")
        if type(self.evidence) is not tuple:
            raise TypeError("evidence must be an exact tuple")
        if len(self.evidence) > METACOGNITION_MAX_EVIDENCE_WITNESSES:
            raise ValueError("evidence exceeds its witness bound")
        if any(type(item) is not MetacognitiveEvidenceWitness for item in self.evidence):
            raise TypeError("evidence must contain exact MetacognitiveEvidenceWitness values")
        for item in self.evidence:
            item.validate_for(self.event)
            if item.source_kind not in {
                AttentionSourceKind.WORKING_MEMORY,
                AttentionSourceKind.EMOTION,
                AttentionSourceKind.BELIEF,
            }:
                raise ValueError("observation evidence source kind is outside U3 scope")
            if item.source_kind in {
                AttentionSourceKind.WORKING_MEMORY,
                AttentionSourceKind.EMOTION,
            } and (
                item.condition is not EvidenceCondition.SUPPORTING
                or item.confidence_ceiling is not None
            ):
                raise ValueError("resource evidence is supporting only and has no confidence ceiling")
            if item.source_kind is AttentionSourceKind.BELIEF:
                if item.condition is EvidenceCondition.SUPPORTING:
                    if item.confidence_ceiling is None:
                        raise ValueError("supporting Belief observation requires recorded confidence")
                elif item.confidence_ceiling is not None:
                    raise ValueError("non-supporting Belief evidence has no confidence ceiling")
        evidence_digests = tuple(item.witness_digest for item in self.evidence)
        if evidence_digests != tuple(sorted(set(evidence_digests))):
            raise ValueError("evidence witnesses must be uniquely sorted by witness digest")
        if not 0 <= self.focus_count <= ATTENTION_MAX_FOCUS:
            raise ValueError("focus_count must be an exact integer within the Attention bound")
        for name in (
            "cognitive_load_units",
            "attention_saturation_units",
            "emotion_influence_units",
            "belief_coverage_units",
            "belief_confidence_ceiling_units",
        ):
            _units(getattr(self, name), name)
        if type(self.contradictory) is not bool:
            raise TypeError("contradictory must be an exact bool")
        if type(self.reason_codes) is not tuple:
            raise TypeError("reason_codes must be an exact tuple")
        if len(self.reason_codes) > METACOGNITION_MAX_REASON_CODES:
            raise ValueError("reason_codes exceeds its bound")
        codes = tuple(
            exact_enum(code, MetacognitiveReasonCode, "reason code")
            for code in self.reason_codes
        )
        if codes != tuple(sorted(set(codes), key=lambda code: code.value)):
            raise ValueError("reason_codes must be uniquely sorted")
        required_reasons = {
            MetacognitiveReasonCode.EVIDENCE_BOUNDARY,
            MetacognitiveReasonCode.LIMITED_PROVENANCE,
            MetacognitiveReasonCode.QUALITY_UNOBSERVED,
        }
        if not required_reasons <= set(codes):
            raise ValueError("observation must preserve its bounded evidence boundary")
        has_contradiction = any(
            item.source_kind is AttentionSourceKind.BELIEF
            and item.condition is EvidenceCondition.CONTRADICTORY
            for item in self.evidence
        )
        if self.contradictory is not has_contradiction:
            raise ValueError("contradictory flag must match typed Belief conflict evidence")
        if has_contradiction and not {
            MetacognitiveReasonCode.CONFLICTING_SOURCES,
            MetacognitiveReasonCode.CONTRADICTORY_EVIDENCE,
        } <= set(codes):
            raise ValueError("observed conflict requires its bounded reason codes")
        if not has_contradiction and MetacognitiveReasonCode.CONTRADICTORY_EVIDENCE in codes:
            raise ValueError("contradictory reason requires contradictory evidence")
        if not has_contradiction and MetacognitiveReasonCode.CONFLICTING_SOURCES in codes:
            raise ValueError("conflicting-sources reason requires observed structured conflict")
        confidence_ceilings = tuple(
            item.confidence_ceiling
            for item in self.evidence
            if item.source_kind is AttentionSourceKind.BELIEF
            and item.condition is EvidenceCondition.SUPPORTING
            and item.confidence_ceiling is not None
        )
        expected_confidence_units = (
            None if not confidence_ceilings else _fixed_units(min(confidence_ceilings))
        )
        if self.belief_confidence_ceiling_units != expected_confidence_units:
            raise ValueError("Belief confidence units must match supporting recorded confidence")
        if self.cognitive_load_units is None and MetacognitiveReasonCode.LOAD_UNOBSERVED not in codes:
            raise ValueError("unobserved cognitive load requires its bounded reason")
        if (
            self.emotion_influence_units is None
            and MetacognitiveReasonCode.EMOTION_INFLUENCE_UNOBSERVED not in codes
        ):
            raise ValueError("unobserved Emotion influence requires its bounded reason")
        if (
            self.attention_saturation_units is None
            and MetacognitiveReasonCode.SATURATION_UNOBSERVED not in codes
        ):
            raise ValueError("unobserved attention saturation requires its bounded reason")
        if expected_confidence_units is None and MetacognitiveReasonCode.MISSING_EVIDENCE not in codes:
            raise ValueError("unobserved Belief confidence requires the missing-evidence reason")
        if (
            self.belief_coverage_units is not None
            and self.belief_coverage_units < METACOGNITION_UNITS_SCALE
            and MetacognitiveReasonCode.PARTIAL_COVERAGE not in codes
        ):
            raise ValueError("partial Belief coverage requires its bounded reason")
        if (
            self.belief_coverage_units is None
            and MetacognitiveReasonCode.PARTIAL_COVERAGE in codes
        ):
            raise ValueError("unknown Belief coverage cannot be labeled partial")
        if any(item.condition is EvidenceCondition.UNKNOWN for item in self.evidence):
            if MetacognitiveReasonCode.UNOBSERVED_SOURCE not in codes:
                raise ValueError("unknown source evidence requires its bounded reason")
        if not self.evidence or all(
            item.condition is EvidenceCondition.UNKNOWN for item in self.evidence
        ):
            if any(
                getattr(self, name) is not None
                for name in (
                    "cognitive_load_units",
                    "attention_saturation_units",
                    "emotion_influence_units",
                    "belief_coverage_units",
                    "belief_confidence_ceiling_units",
                )
            ):
                raise ValueError("empty or all-unknown evidence cannot carry numeric units")

    def validated_copy(self) -> MetacognitionObservation:
        """Return a detached clone and reject, rather than repair, tampering."""

        if type(self) is not MetacognitionObservation:
            raise TypeError("observation must be an exact MetacognitionObservation")
        declared_digest = validate_digest(self.observation_digest, "observation_digest")
        checked = MetacognitionObservation._create(
            event=self.event,
            focus_witness=self.focus_witness,
            evidence=self.evidence,
            focus_count=self.focus_count,
            cognitive_load_units=self.cognitive_load_units,
            attention_saturation_units=self.attention_saturation_units,
            emotion_influence_units=self.emotion_influence_units,
            belief_coverage_units=self.belief_coverage_units,
            belief_confidence_ceiling_units=self.belief_confidence_ceiling_units,
            contradictory=self.contradictory,
            reason_codes=self.reason_codes,
        )
        if checked.observation_digest != declared_digest:
            raise ValueError("observation digest does not match its published fields")
        return checked

    def _payload(self) -> dict[str, object]:
        self.__post_init__()
        return _observation_payload(self)

    def canonical_value(self) -> dict[str, object]:
        checked = self.validated_copy()
        return {
            **_observation_payload(checked),
            "observation_digest": checked.observation_digest,
        }

    def canonical_bytes(self) -> bytes:
        encoded = canonical_json(self.canonical_value())
        if len(encoded) > METACOGNITION_MAX_OBSERVATION_BYTES:
            raise ValueError("metacognition observation exceeds its derived byte envelope")
        return encoded


def observe_metacognition(
    event: AttentionEvent,
    attention: AttentionContinuity,
    *,
    working_memory_items: tuple[WorkingMemoryItem, ...] | None = None,
    working_memory_revision: int | None = None,
    working_memory_view: WorkingMemoryView | None = None,
    emotion_state: EmotionState | None = None,
    focus_records: tuple[MotivationRecord | GoalRecord | CommitmentRecord, ...] = (),
    belief_records: tuple[BeliefRecord, ...] | None = None,
    current_context_id: str | None = None,
) -> MetacognitionObservation:
    """Derive a strictly event-scoped observation from current typed values.

    Supplied records are validated as complete caller-provided source snapshots
    within their declared bounds.  This function performs no source lookup,
    policy replay, mutation, persistence, or history retention.
    """

    current_event = _copy_event(event)
    current_attention = _copy_attention(attention)
    _validate_attention_event(current_attention, current_event)

    if working_memory_view is not None and working_memory_items is None:
        raise ValueError("Working Memory view requires its complete membership snapshot")
    if (working_memory_items is None) != (working_memory_revision is None):
        raise ValueError("Working Memory membership and revision must be supplied together")

    items: tuple[WorkingMemoryItem, ...] | None = None
    item_by_id: dict[str, WorkingMemoryItem] = {}
    memory_revision: int | None = None
    checked_view: WorkingMemoryView | None = None
    decisions: dict[str, WorkingMemoryDecision] | None = None
    selections: dict[str, WorkingMemorySelection] | None = None
    if working_memory_items is not None:
        items, memory_revision, item_by_id = _copy_working_memory(
            working_memory_items,
            working_memory_revision,
        )
        assert memory_revision is not None
        if working_memory_view is not None:
            checked_view = _preflight_working_memory_view(
                working_memory_view,
                memory_revision,
            )
            decisions, selections = _validate_view(checked_view, memory_revision)
            if set(decisions) != set(item_by_id):
                raise ValueError(
                    "Working Memory view decisions do not cover complete current membership"
                )
            for item_id, item in item_by_id.items():
                _validate_decision_for_item(item, decisions[item_id], selections.get(item_id))

        focused_working_memory = tuple(
            item
            for item in current_attention.candidates
            if item.candidate_id in current_attention.focused_ids
            and item.target.kind.value == "working_memory"
        )
        for candidate in focused_working_memory:
            source_item = item_by_id.get(candidate.target.reference)
            if source_item is None:
                raise ValueError("focused Working Memory source is missing from current membership")
            current_projection = project_working_memory(
                source_item,
                revision=memory_revision,
                event=current_event,
                view=checked_view,
            )
            if current_projection.source.canonical_value() != candidate.source.canonical_value():
                raise ValueError("focused Working Memory source is stale relative to current input")

    checked_emotion = None if emotion_state is None else _copy_emotion(emotion_state)
    current_context = (
        None
        if current_context_id is None
        else _checked_identifier(current_context_id, "current_context_id")
    )
    _, belief_links = _copy_focus_records(
        focus_records,
        current_attention,
        current_event,
    )

    if belief_records is None:
        checked_beliefs: tuple[BeliefRecord, ...] = ()
        belief_source_events: dict[str, AttentionEvent | None] = {}
    else:
        checked_beliefs, belief_source_events = _copy_belief_records(
            belief_records,
            current_event,
            provided_links=belief_links,
        )

    _validate_source_event_identity_consistency(
        current_attention,
        current_event,
        checked_beliefs,
    )
    contradictory_ids = _contradictory_belief_ids(
        checked_beliefs,
        belief_links,
        current_event,
        current_context,
    )
    belief_by_id = {record.belief_id: record for record in checked_beliefs}
    evidence: list[MetacognitiveEvidenceWitness] = []
    reasons: set[MetacognitiveReasonCode] = {
        MetacognitiveReasonCode.EVIDENCE_BOUNDARY,
        MetacognitiveReasonCode.LIMITED_PROVENANCE,
        MetacognitiveReasonCode.QUALITY_UNOBSERVED,
    }
    if set(belief_links).difference(belief_by_id):
        reasons.add(MetacognitiveReasonCode.UNOBSERVED_SOURCE)

    if items is not None and memory_revision is not None:
        evidence.append(
            MetacognitiveEvidenceWitness(
                source_kind=AttentionSourceKind.WORKING_MEMORY,
                reference="working-memory",
                source_revision=memory_revision,
                digest=_working_memory_digest(
                    items,
                    memory_revision,
                    checked_view,
                    decisions,
                    selections,
                ),
                digest_kind=SourceDigestKind.ATTENTION_PROJECTION,
                condition=EvidenceCondition.SUPPORTING,
                confidence_ceiling=None,
                source_event=current_event,
                source_event_origin=SourceEventOrigin.ATTENTION_EVENT,
            )
        )

    if checked_emotion is not None:
        evidence.append(
            MetacognitiveEvidenceWitness(
                source_kind=AttentionSourceKind.EMOTION,
                reference="current-emotion",
                source_revision=None,
                digest=_emotion_digest(checked_emotion, current_event),
                digest_kind=SourceDigestKind.ATTENTION_PROJECTION,
                condition=EvidenceCondition.SUPPORTING,
                confidence_ceiling=None,
                source_event=current_event,
                source_event_origin=SourceEventOrigin.ATTENTION_EVENT,
            )
        )

    observed_belief_confidences: list[float] = []
    covered_targets: set[str] = set()
    ordinary_active_by_id: dict[str, bool] = {}
    focus_count = len(current_attention.focused_ids)
    for belief_id in sorted(belief_by_id):
        record = belief_by_id[belief_id]
        linked_targets = belief_links[belief_id]
        ordinary_active = _is_ordinary_active(record, current_event, current_context)
        ordinary_active_by_id[belief_id] = ordinary_active
        if ordinary_active:
            covered_targets.update(linked_targets)
        if ordinary_active and belief_id in contradictory_ids:
            condition = EvidenceCondition.CONTRADICTORY
            ceiling = None
        elif ordinary_active:
            condition = EvidenceCondition.SUPPORTING
            ceiling = record.confidence
            observed_belief_confidences.append(record.confidence)
        else:
            condition = EvidenceCondition.UNKNOWN
            ceiling = None
            reasons.add(MetacognitiveReasonCode.UNOBSERVED_SOURCE)
        evidence.append(
            _belief_witness(
                record,
                condition=condition,
                confidence_ceiling=ceiling,
                source_event=belief_source_events[belief_id],
            )
        )

    if contradictory_ids:
        reasons.update(
            {
                MetacognitiveReasonCode.CONFLICTING_SOURCES,
                MetacognitiveReasonCode.CONTRADICTORY_EVIDENCE,
            }
        )

    if not observed_belief_confidences:
        reasons.add(MetacognitiveReasonCode.MISSING_EVIDENCE)
    if checked_view is None:
        reasons.add(MetacognitiveReasonCode.LOAD_UNOBSERVED)
    if checked_emotion is None:
        reasons.add(MetacognitiveReasonCode.EMOTION_INFLUENCE_UNOBSERVED)

    cognitive_load_units: int | None = None
    if checked_view is not None and items is not None:
        item_load = len(items) * METACOGNITION_UNITS_SCALE // checked_view.item_capacity
        byte_load = (
            checked_view.projected_bytes
            * METACOGNITION_UNITS_SCALE
            // checked_view.projection_max_bytes
        )
        cognitive_load_units = max(item_load, byte_load)

    emotion_influence_units: int | None = None
    if checked_emotion is not None:
        emotion_influence_units = _fixed_units(
            max(checked_emotion.arousal, abs(checked_emotion.valence))
        )

    has_typed_observation = items is not None or checked_emotion is not None or bool(checked_beliefs)
    attention_saturation_units: int | None = None
    if has_typed_observation:
        focus_capacity = (
            ATTENTION_HIGH_AROUSAL_MAX_FOCUS
            if checked_emotion is not None
            and checked_emotion.arousal >= ATTENTION_HIGH_AROUSAL_THRESHOLD
            else ATTENTION_MAX_FOCUS
        )
        attention_saturation_units = min(
            focus_count * METACOGNITION_UNITS_SCALE // focus_capacity,
            METACOGNITION_UNITS_SCALE,
        )
    else:
        reasons.add(MetacognitiveReasonCode.SATURATION_UNOBSERVED)

    all_links_resolved = not set(belief_links).difference(belief_by_id)
    coverage_is_observed = (
        belief_records is not None
        and focus_count > 0
        and bool(checked_beliefs)
        and all_links_resolved
        and all(ordinary_active_by_id.values())
    )
    belief_coverage_units: int | None = None
    if coverage_is_observed:
        belief_coverage_units = (
            len(covered_targets) * METACOGNITION_UNITS_SCALE // focus_count
        )
        if belief_coverage_units < METACOGNITION_UNITS_SCALE:
            reasons.add(MetacognitiveReasonCode.PARTIAL_COVERAGE)

    belief_confidence_ceiling_units = (
        None
        if not observed_belief_confidences
        else _fixed_units(min(observed_belief_confidences))
    )

    if not evidence or all(
        item.condition is EvidenceCondition.UNKNOWN for item in evidence
    ):
        cognitive_load_units = None
        attention_saturation_units = None
        emotion_influence_units = None
        belief_coverage_units = None
        belief_confidence_ceiling_units = None

    if attention_saturation_units is None:
        reasons.add(MetacognitiveReasonCode.SATURATION_UNOBSERVED)

    if len(evidence) > METACOGNITION_MAX_EVIDENCE_WITNESSES:
        raise ValueError("observation evidence exceeds its witness bound")
    ordered_evidence = tuple(
        sorted(evidence, key=lambda item: item.witness_digest)
    )
    evidence_digests = tuple(item.witness_digest for item in ordered_evidence)
    if evidence_digests != tuple(sorted(set(evidence_digests))):
        raise ValueError("observation evidence witnesses are not unique")

    focus_witness = FocusAssessmentWitness(
        attention_revision=current_attention.revision,
        attention_state_digest=current_attention.state_digest,
        focused_ids=current_attention.focused_ids,
        event=current_event,
    )
    observation = MetacognitionObservation._create(
        event=current_event,
        focus_witness=focus_witness,
        evidence=ordered_evidence,
        focus_count=focus_count,
        cognitive_load_units=cognitive_load_units,
        attention_saturation_units=attention_saturation_units,
        emotion_influence_units=emotion_influence_units,
        belief_coverage_units=belief_coverage_units,
        belief_confidence_ceiling_units=belief_confidence_ceiling_units,
        contradictory=bool(contradictory_ids),
        reason_codes=_make_reason_codes(reasons),
    )
    observation.canonical_bytes()
    return observation


def _longest_enum_value(enum_type: type[_EnumT]) -> str:
    return max((cast(str, item.value) for item in enum_type), key=len)


def _maximum_event_value() -> dict[str, object]:
    return {
        "event_id": "e" * MAX_IDENTIFIER_CODEPOINTS,
        "event_sequence": 2**63 - 1,
        "occurred_at": canonical_datetime(datetime.max.replace(tzinfo=UTC)),
    }


def _maximum_fraction_hex() -> str:
    return float.fromhex("0x0.0000000000001p-1022").hex()


def derive_metacognition_observation_max_bytes() -> int:
    """Derive a finite canonical envelope from observation's own field bounds."""

    maximum_event = _maximum_event_value()
    maximum_digest = "f" * 64
    focus_value = {
        "attention_revision": MAX_PERSISTED_REVISION,
        "attention_state_digest": maximum_digest,
        "event": maximum_event,
        "focused_ids": [maximum_digest] * ATTENTION_MAX_FOCUS,
    }
    maximum_evidence = {
        "condition": _longest_enum_value(EvidenceCondition),
        "confidence_ceiling": _maximum_fraction_hex(),
        "digest": maximum_digest,
        "digest_kind": _longest_enum_value(SourceDigestKind),
        "reference": "a" * MAX_IDENTIFIER_CODEPOINTS,
        "source_event": maximum_event,
        "source_event_origin": _longest_enum_value(SourceEventOrigin),
        "source_kind": _longest_enum_value(AttentionSourceKind),
        "source_revision": MAX_PERSISTED_REVISION,
        "witness_digest": maximum_digest,
    }
    maximum_payload = {
        "attention_saturation_units": METACOGNITION_UNITS_SCALE,
        "belief_confidence_ceiling_units": METACOGNITION_UNITS_SCALE,
        "belief_coverage_units": METACOGNITION_UNITS_SCALE,
        "cognitive_load_units": METACOGNITION_UNITS_SCALE,
        "contradictory": False,
        "emotion_influence_units": METACOGNITION_UNITS_SCALE,
        "event": maximum_event,
        "evidence": [maximum_evidence] * METACOGNITION_MAX_EVIDENCE_WITNESSES,
        "focus_count": ATTENTION_MAX_FOCUS,
        "focus_witness": focus_value,
        "reason_codes": [
            _longest_enum_value(MetacognitiveReasonCode)
        ]
        * METACOGNITION_MAX_REASON_CODES,
        "observation_digest": maximum_digest,
    }
    return len(canonical_json(maximum_payload))


METACOGNITION_MAX_OBSERVATION_BYTES: Final[int] = (
    derive_metacognition_observation_max_bytes()
)


__all__ = [
    "METACOGNITION_MAX_BELIEF_RECORDS",
    "METACOGNITION_MAX_FOCUS_RECORDS",
    "METACOGNITION_MAX_OBSERVATION_BYTES",
    "METACOGNITION_UNITS_SCALE",
    "MetacognitionObservation",
    "derive_metacognition_observation_max_bytes",
    "observe_metacognition",
]
