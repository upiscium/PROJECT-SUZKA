"""Schema-derived R14 U1 persistence and resource bounds.

The capacity fixture is assembled through the exact immutable contracts and
their canonical serializers.  Production imports never reach ``suzka.runtime``;
the neutral Working Memory capacity is read lazily from its owner contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from typing import Final

from suzka.attention.common import (
    ATTENTION_FIXED_POINT_SCALE,
    ATTENTION_MAX_COUNTER,
    ATTENTION_MAX_EVENT_RECEIPTS,
    ATTENTION_MAX_EVENT_SEQUENCE,
    ATTENTION_MAX_FOCUS,
    ATTENTION_MAX_R13_RECORDS_PER_DOMAIN,
    ATTENTION_MAX_REVISION,
    ATTENTION_MAX_REVISION_HISTORY,
    ATTENTION_POLICY_VERSION,
    ATTENTION_SCHEMA_VERSION,
    AttentionRevisionReason,
    AttentionSourceKind,
    AttentionTargetKind,
    CandidateAvailability,
    SourceDigestKind,
    canonical_json,
)
from suzka.motivation.common import R13_AGENT_STATE_CAP_BYTES
from suzka.attention.contracts import (
    AttentionCandidateContinuity,
    AttentionContinuity,
    AttentionEvent,
    AttentionEventReceipt,
    AttentionReceiptAnchor,
    AttentionRevisionAnchor,
    AttentionRevisionEvidence,
    AttentionSourceWitness,
    AttentionTarget,
    attention_state_digest,
)


# Frozen baseline capacity evidence supplied by the integrated AgentState v8
# schema.  The value bound below derives the Attention field's remaining room
# from this base, the common hard cap, the future reserve, and exact JSON root
# key/comma overhead.  The v8 schema projector is imported only from tests.
ATTENTION_AGENT_STATE_V8_BASE_MAX_BYTES: Final[int] = 109_573_688
ATTENTION_AGENT_STATE_FUTURE_RESERVE_BYTES: Final[int] = 16 * 1024 * 1024
ATTENTION_AGENT_STATE_MAX_BYTES: Final[int] = R13_AGENT_STATE_CAP_BYTES
ATTENTION_FIELD_NAME: Final[str] = "attention_state"
ATTENTION_ROOT_FIELD_OVERHEAD_BYTES: Final[int] = (
    len(canonical_json(ATTENTION_FIELD_NAME)) + 2
)
ATTENTION_STATE_MAX_VALUE_BYTES: Final[int] = (
    ATTENTION_AGENT_STATE_MAX_BYTES
    - ATTENTION_AGENT_STATE_V8_BASE_MAX_BYTES
    - ATTENTION_AGENT_STATE_FUTURE_RESERVE_BYTES
    - ATTENTION_ROOT_FIELD_OVERHEAD_BYTES
)

_MAX_TIMESTAMP: Final[datetime] = datetime.max.replace(tzinfo=UTC)
_MAX_DIGEST: Final[str] = "f" * 64
_MAX_EVENT_ID_PREFIX: Final[str] = "e" * 124
_SOURCE_KIND_BY_TARGET: Final[dict[AttentionTargetKind, AttentionSourceKind]] = {
    AttentionTargetKind.WORKING_MEMORY: AttentionSourceKind.WORKING_MEMORY,
    AttentionTargetKind.MOTIVATION: AttentionSourceKind.MOTIVATION,
    AttentionTargetKind.GOAL: AttentionSourceKind.GOAL,
    AttentionTargetKind.COMMITMENT: AttentionSourceKind.COMMITMENT,
}


def attention_candidate_capacity() -> int:
    """Return the full supported current identity universe from its owners."""

    from suzka.working_memory_contracts import MAX_ITEM_CAPACITY

    if type(MAX_ITEM_CAPACITY) is not int or MAX_ITEM_CAPACITY <= 0:
        raise RuntimeError("neutral Working Memory capacity is not a positive integer")
    return MAX_ITEM_CAPACITY + 3 * ATTENTION_MAX_R13_RECORDS_PER_DOMAIN


def attention_candidate_capacity_by_kind() -> tuple[tuple[str, int], ...]:
    """Expose the source-owned per-kind split used by validation and sizing."""

    from suzka.working_memory_contracts import MAX_ITEM_CAPACITY

    if type(MAX_ITEM_CAPACITY) is not int or MAX_ITEM_CAPACITY <= 0:
        raise RuntimeError("neutral Working Memory capacity is not a positive integer")
    return (
        (AttentionTargetKind.COMMITMENT.value, ATTENTION_MAX_R13_RECORDS_PER_DOMAIN),
        (AttentionTargetKind.GOAL.value, ATTENTION_MAX_R13_RECORDS_PER_DOMAIN),
        (AttentionTargetKind.MOTIVATION.value, ATTENTION_MAX_R13_RECORDS_PER_DOMAIN),
        (AttentionTargetKind.WORKING_MEMORY.value, MAX_ITEM_CAPACITY),
    )


def _max_event_id(index: int) -> str:
    return f"{_MAX_EVENT_ID_PREFIX}{index:04x}"


def _maximum_current_event() -> AttentionEvent:
    return AttentionEvent(
        event_id=_max_event_id(ATTENTION_MAX_REVISION_HISTORY - 1),
        event_sequence=ATTENTION_MAX_EVENT_SEQUENCE,
        occurred_at=_MAX_TIMESTAMP,
    )


def _maximum_targets() -> tuple[AttentionTarget, ...]:
    from suzka.working_memory_contracts import MAX_ITEM_CAPACITY

    targets = [
        AttentionTarget(
            AttentionTargetKind.WORKING_MEMORY,
            f"wm-{index:064x}",
        )
        for index in range(MAX_ITEM_CAPACITY)
    ]
    for kind in (
        AttentionTargetKind.MOTIVATION,
        AttentionTargetKind.GOAL,
        AttentionTargetKind.COMMITMENT,
    ):
        targets.extend(
            AttentionTarget(kind, f"{index:064x}")
            for index in range(ATTENTION_MAX_R13_RECORDS_PER_DOMAIN)
        )
    return tuple(sorted(targets, key=lambda target: target.candidate_id))


def _maximum_primary_source(
    target: AttentionTarget,
    current_event: AttentionEvent,
) -> AttentionSourceWitness:
    wm_target = target.kind is AttentionTargetKind.WORKING_MEMORY
    return AttentionSourceWitness(
        kind=_SOURCE_KIND_BY_TARGET[target.kind],
        reference=target.reference,
        revision=ATTENTION_MAX_REVISION,
        digest=_MAX_DIGEST,
        digest_kind=(
            SourceDigestKind.ATTENTION_PROJECTION
            if wm_target
            else SourceDigestKind.UPSTREAM_AUTHORITY
        ),
        target_kind=target.kind,
        target_reference=target.reference,
        source_event_id=None if wm_target else current_event.event_id,
        source_event_sequence=None
        if wm_target
        else current_event.event_sequence,
        source_occurred_at=None if wm_target else current_event.occurred_at,
    )


def _maximum_candidate_row(
    target: AttentionTarget,
    current_event: AttentionEvent,
    availability: CandidateAvailability,
    *,
    focused: bool,
) -> AttentionCandidateContinuity:
    return AttentionCandidateContinuity(
        target=target,
        source=_maximum_primary_source(target, current_event),
        availability=availability,
        habituation=ATTENTION_FIXED_POINT_SCALE,
        inhibition=ATTENTION_FIXED_POINT_SCALE,
        focused_event_count=ATTENTION_MAX_COUNTER if focused else 0,
        unattended_event_count=0 if focused else ATTENTION_MAX_COUNTER,
    )


@lru_cache(maxsize=1)
def maximum_attention_continuity_fixture() -> AttentionContinuity:
    """Build the full legal shape: every supported candidate and proof slot."""

    current_event = _maximum_current_event()
    targets = _maximum_targets()
    candidates: list[AttentionCandidateContinuity] = []
    for index, target in enumerate(targets):
        candidates.append(
            _maximum_candidate_row(
                target,
                current_event,
                CandidateAvailability.ELIGIBLE
                if index < ATTENTION_MAX_FOCUS
                else CandidateAvailability.UNAVAILABLE,
                focused=index < ATTENTION_MAX_FOCUS,
            )
        )
    current_candidates = tuple(candidates)
    focused_ids = tuple(
        sorted(
            item.candidate_id
            for item in current_candidates
            if item.focused_event_count > 0
        )
    )
    unfinished_ids = focused_ids
    current_revision = ATTENTION_MAX_REVISION
    state_digest = attention_state_digest(
        schema_version=ATTENTION_SCHEMA_VERSION,
        policy_version=ATTENTION_POLICY_VERSION,
        revision=current_revision,
        last_event=current_event,
        candidates=current_candidates,
        focused_ids=focused_ids,
        unfinished_ids=unfinished_ids,
    )

    revision_anchor_event = AttentionEvent(
        event_id="a" * 128,
        event_sequence=ATTENTION_MAX_EVENT_SEQUENCE - ATTENTION_MAX_REVISION_HISTORY,
        occurred_at=_MAX_TIMESTAMP,
    )
    revision_anchor = AttentionRevisionAnchor(
        through_revision=current_revision - ATTENTION_MAX_REVISION_HISTORY,
        through_event=revision_anchor_event,
        through_state_digest="a" * 64,
        through_revision_digest="b" * 64,
    )
    revision_evidence: list[AttentionRevisionEvidence] = []
    previous_revision_digest = revision_anchor.through_revision_digest
    previous_state_digest = revision_anchor.through_state_digest
    for index in range(ATTENTION_MAX_REVISION_HISTORY):
        revision = current_revision - ATTENTION_MAX_REVISION_HISTORY + 1 + index
        event = AttentionEvent(
            event_id=_max_event_id(index),
            event_sequence=ATTENTION_MAX_EVENT_SEQUENCE
            - ATTENTION_MAX_REVISION_HISTORY
            + 1
            + index,
            occurred_at=_MAX_TIMESTAMP,
        )
        item_state_digest = (
            state_digest if index == ATTENTION_MAX_REVISION_HISTORY - 1 else f"{index:064x}"
        )
        item = AttentionRevisionEvidence(
            revision=revision,
            event=event,
            previous_state_digest=previous_state_digest,
            state_digest=item_state_digest,
            previous_revision_digest=previous_revision_digest,
            focused_ids=focused_ids,
            unfinished_ids=unfinished_ids,
            reason=AttentionRevisionReason.STATE_UPDATE,
        )
        revision_evidence.append(item)
        previous_revision_digest = item.record_digest
        previous_state_digest = item.state_digest

    receipt_anchor_event = AttentionEvent(
        event_id="r" * 128,
        event_sequence=ATTENTION_MAX_EVENT_SEQUENCE - ATTENTION_MAX_EVENT_RECEIPTS,
        occurred_at=_MAX_TIMESTAMP,
    )
    receipt_anchor = AttentionReceiptAnchor(
        through_event=receipt_anchor_event,
        through_result_state_digest="c" * 64,
        through_receipt_digest="d" * 64,
    )
    receipts: list[AttentionEventReceipt] = []
    previous_receipt_digest: str | None = receipt_anchor.through_receipt_digest
    for index in range(ATTENTION_MAX_EVENT_RECEIPTS):
        sequence = (
            ATTENTION_MAX_EVENT_SEQUENCE
            - ATTENTION_MAX_EVENT_RECEIPTS
            + 1
            + index
        )
        if sequence == revision_anchor_event.event_sequence:
            event = revision_anchor_event
        elif sequence == current_event.event_sequence:
            event = current_event
        elif sequence >= ATTENTION_MAX_EVENT_SEQUENCE - ATTENTION_MAX_REVISION_HISTORY + 1:
            revision_index = sequence - (
                ATTENTION_MAX_EVENT_SEQUENCE
                - ATTENTION_MAX_REVISION_HISTORY
                + 1
            )
            event = revision_evidence[revision_index].event
        else:
            event = AttentionEvent(
                event_id=f"r{index:03d}" + "q" * 124,
                event_sequence=sequence,
                occurred_at=_MAX_TIMESTAMP,
            )
        result_state_digest = next(
            (
                evidence.state_digest
                for evidence in revision_evidence
                if evidence.event == event
            ),
            revision_anchor.through_state_digest
            if event == revision_anchor_event
            else "f" * 64,
        )
        receipt = AttentionEventReceipt(
            event=event,
            input_digest="e" * 64,
            result_state_digest=result_state_digest,
            previous_receipt_digest=previous_receipt_digest,
        )
        receipts.append(receipt)
        previous_receipt_digest = receipt.receipt_digest

    return AttentionContinuity(
        schema_version=ATTENTION_SCHEMA_VERSION,
        policy_version=ATTENTION_POLICY_VERSION,
        revision=current_revision,
        last_event=current_event,
        candidates=current_candidates,
        focused_ids=focused_ids,
        unfinished_ids=unfinished_ids,
        revision_history=tuple(revision_evidence),
        receipts=tuple(receipts),
        revision_anchor=revision_anchor,
        receipt_anchor=receipt_anchor,
    )


@dataclass(frozen=True, slots=True)
class AttentionCandidateRowEnvelope:
    """Serializer-derived legal row maxima for one primary target domain."""

    target_kind: str
    eligible_focused_row_bytes: int
    unavailable_nonfocused_row_bytes: int
    maximum_nonfocused_row_bytes: int
    field_maxima: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if type(self.target_kind) is not str:
            raise TypeError("target_kind must be an exact string")
        AttentionTargetKind(self.target_kind)
        for name in (
            "eligible_focused_row_bytes",
            "unavailable_nonfocused_row_bytes",
            "maximum_nonfocused_row_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact byte count")
        if self.maximum_nonfocused_row_bytes != self.unavailable_nonfocused_row_bytes:
            raise ValueError("unavailable must be the longest legal nonfocused status")
        if self.field_maxima != tuple(sorted(self.field_maxima)):
            raise ValueError("candidate field maxima must use canonical key order")

    @property
    def field_maxima_by_name(self) -> dict[str, int]:
        return dict(self.field_maxima)


@lru_cache(maxsize=1)
def derive_attention_candidate_row_envelopes() -> tuple[AttentionCandidateRowEnvelope, ...]:
    """Derive exact canonical row/field lengths for every legal primary source."""

    event = _maximum_current_event()
    envelopes: list[AttentionCandidateRowEnvelope] = []
    for kind in AttentionTargetKind:
        reference = (
            "wm-" + "f" * 64
            if kind is AttentionTargetKind.WORKING_MEMORY
            else "f" * 64
        )
        target = AttentionTarget(kind, reference)
        rows = (
            _maximum_candidate_row(
                target,
                event,
                CandidateAvailability.ELIGIBLE,
                focused=True,
            ),
            _maximum_candidate_row(
                target,
                event,
                CandidateAvailability.ELIGIBLE,
                focused=False,
            ),
            _maximum_candidate_row(
                target,
                event,
                CandidateAvailability.UNAVAILABLE,
                focused=False,
            ),
            _maximum_candidate_row(
                target,
                event,
                CandidateAvailability.INACTIVE,
                focused=False,
            ),
        )
        values = tuple(row.canonical_value() for row in rows)
        lengths = tuple(len(canonical_json(value)) for value in values)
        eligible_focused_bytes = lengths[0]
        nonfocused_max = max(lengths[1:])
        if nonfocused_max != lengths[2]:
            raise AssertionError("unavailable is not the longest legal nonfocused row")
        field_names = set(values[0])
        if any(set(value) != field_names for value in values[1:]):
            raise AssertionError("candidate row serializer field set is not closed")
        field_maxima = tuple(
            sorted(
                (
                    name,
                    max(len(canonical_json(value[name])) for value in values),
                )
                for name in field_names
            )
        )
        envelopes.append(
            AttentionCandidateRowEnvelope(
                target_kind=kind.value,
                eligible_focused_row_bytes=eligible_focused_bytes,
                unavailable_nonfocused_row_bytes=lengths[2],
                maximum_nonfocused_row_bytes=nonfocused_max,
                field_maxima=field_maxima,
            )
        )
    return tuple(envelopes)


def _candidate_array_envelope_bytes(
    fixture: AttentionContinuity,
    envelopes: tuple[AttentionCandidateRowEnvelope, ...],
) -> int:
    envelope_by_kind = {item.target_kind: item for item in envelopes}
    candidates_by_kind = {
        kind: tuple(
            candidate
            for candidate in fixture.candidates
            if candidate.target.kind.value == kind
        )
        for kind in envelope_by_kind
    }
    total_row_bytes = 0
    row_count = 0
    for kind, candidates in candidates_by_kind.items():
        envelope = envelope_by_kind[kind]
        focused_count = sum(
            candidate.availability is CandidateAvailability.ELIGIBLE
            and candidate.focused_event_count > 0
            for candidate in candidates
        )
        unavailable_count = sum(
            candidate.availability is CandidateAvailability.UNAVAILABLE
            for candidate in candidates
        )
        if focused_count + unavailable_count != len(candidates):
            raise ValueError("maximum fixture contains an unmodeled candidate row status")
        if any(
            candidate.availability is CandidateAvailability.ELIGIBLE
            and candidate.focused_event_count == 0
            for candidate in candidates
        ):
            raise ValueError("maximum fixture must assign focus to every eligible row")
        total_row_bytes += (
            focused_count * envelope.eligible_focused_row_bytes
            + unavailable_count * envelope.unavailable_nonfocused_row_bytes
        )
        row_count += len(candidates)
    return 2 + total_row_bytes + max(0, row_count - 1)


def _array_row_envelope_bytes(values: Sequence[object]) -> int:
    row_sizes = tuple(len(canonical_json(value)) for value in values)
    if row_sizes and len(set(row_sizes)) != 1:
        raise AssertionError("maximum retained proof rows do not share their schema envelope")
    return 2 + sum(row_sizes) + max(0, len(row_sizes) - 1)


def _canonical_object_envelope_bytes(field_maxima: tuple[tuple[str, int], ...]) -> int:
    fields = tuple(sorted(field_maxima))
    return 2 + sum(
        len(canonical_json(name)) + 1 + value_bytes
        for name, value_bytes in fields
    ) + max(0, len(fields) - 1)


@dataclass(frozen=True, slots=True)
class AttentionSchemaSizeBudget:
    """Exact canonical JSON maximum for the declared Attention continuity schema."""

    candidate_count: int
    candidate_counts: tuple[tuple[str, int], ...]
    candidate_row_envelopes: tuple[AttentionCandidateRowEnvelope, ...]
    candidate_array_bytes: int
    candidate_array_envelope_bytes: int
    revision_history_array_bytes: int
    revision_history_array_envelope_bytes: int
    receipts_array_bytes: int
    receipts_array_envelope_bytes: int
    field_maxima: tuple[tuple[str, int], ...]
    attention_state_max_bytes: int
    attention_state_envelope_bytes: int
    maximum_value_bytes: int = ATTENTION_STATE_MAX_VALUE_BYTES

    def __post_init__(self) -> None:
        if type(self.candidate_count) is not int or self.candidate_count <= 0:
            raise ValueError("candidate_count must be a positive exact integer")
        if type(self.attention_state_max_bytes) is not int or self.attention_state_max_bytes <= 0:
            raise ValueError("attention_state_max_bytes must be a positive exact integer")
        if self.attention_state_max_bytes > self.maximum_value_bytes:
            raise ValueError("Attention continuity exceeds its reserved AgentState value bound")
        if self.candidate_array_bytes != self.candidate_array_envelope_bytes:
            raise ValueError("candidate-array row envelope does not equal the legal fixture")
        if self.revision_history_array_bytes != self.revision_history_array_envelope_bytes:
            raise ValueError("revision-history row envelope does not equal the full fixture")
        if self.receipts_array_bytes != self.receipts_array_envelope_bytes:
            raise ValueError("receipt row envelope does not equal the full fixture")
        if self.attention_state_max_bytes != self.attention_state_envelope_bytes:
            raise ValueError("Attention root field envelope does not equal its canonical JSON")
        if self.field_maxima != tuple(sorted(self.field_maxima)):
            raise ValueError("field maxima must be in canonical key order")

    @property
    def margin_bytes(self) -> int:
        return self.maximum_value_bytes - self.attention_state_max_bytes

    @property
    def field_maxima_by_name(self) -> dict[str, int]:
        return dict(self.field_maxima)

    @property
    def candidate_counts_by_kind(self) -> dict[str, int]:
        return dict(self.candidate_counts)


@lru_cache(maxsize=1)
def derive_attention_schema_size_budget() -> AttentionSchemaSizeBudget:
    """Derive bytes and prove the full candidate array against legal row maxima.

    Every primary row has a fixed-width typed target reference and matching
    source reference. Source kinds, digest kinds, optional event witnesses,
    counters, and statuses are enumerated from the actual closed contracts.
    The full fixture has 16 eligible focused rows and the remaining rows use
    the longest legal nonfocused status (unavailable). Stable IDs and hash
    outputs are fixed-width; changing their values cannot increase row bytes.
    """

    fixture = maximum_attention_continuity_fixture()
    value = fixture.canonical_value()
    encoded = canonical_json(value)
    row_envelopes = derive_attention_candidate_row_envelopes()
    candidate_array = canonical_json(
        [candidate.canonical_value() for candidate in fixture.candidates]
    )
    candidate_array_envelope = _candidate_array_envelope_bytes(
        fixture,
        row_envelopes,
    )
    if len(candidate_array) != candidate_array_envelope:
        raise AssertionError(
            "serializer-derived candidate row envelope does not equal the full fixture"
        )
    revision_history_values = [
        item.canonical_value() for item in fixture.revision_history
    ]
    revision_history_array = canonical_json(revision_history_values)
    revision_history_envelope = _array_row_envelope_bytes(revision_history_values)
    receipts_values = [item.canonical_value() for item in fixture.receipts]
    receipts_array = canonical_json(receipts_values)
    receipts_envelope = _array_row_envelope_bytes(receipts_values)
    field_maxima = tuple(
        sorted((name, len(canonical_json(field_value))) for name, field_value in value.items())
    )
    state_envelope = _canonical_object_envelope_bytes(field_maxima)
    if len(encoded) != state_envelope:
        raise AssertionError("Attention root field envelope does not equal the full fixture")
    counts = tuple(
        sorted(
            (
                kind.value,
                sum(candidate.target.kind is kind for candidate in fixture.candidates),
            )
            for kind in AttentionTargetKind
        )
    )
    if len(fixture.candidates) != attention_candidate_capacity():
        raise AssertionError("maximum Attention fixture omits supported current identities")
    if set(dict(field_maxima)) != set(value):
        raise AssertionError("Attention size field map does not match the serializer")
    return AttentionSchemaSizeBudget(
        candidate_count=len(fixture.candidates),
        candidate_counts=counts,
        candidate_row_envelopes=row_envelopes,
        candidate_array_bytes=len(candidate_array),
        candidate_array_envelope_bytes=candidate_array_envelope,
        revision_history_array_bytes=len(revision_history_array),
        revision_history_array_envelope_bytes=revision_history_envelope,
        receipts_array_bytes=len(receipts_array),
        receipts_array_envelope_bytes=receipts_envelope,
        field_maxima=field_maxima,
        attention_state_max_bytes=len(encoded),
        attention_state_envelope_bytes=state_envelope,
    )


def validate_attention_schema_size_budget(
    budget: AttentionSchemaSizeBudget | None = None,
) -> AttentionSchemaSizeBudget:
    expected = derive_attention_schema_size_budget()
    actual = expected if budget is None else budget
    if type(actual) is not AttentionSchemaSizeBudget or actual != expected:
        raise ValueError("Attention schema size budget does not match its declared bounds")
    return actual


def canonical_attention_schema_size_budget(
    budget: AttentionSchemaSizeBudget | None = None,
) -> bytes:
    actual = validate_attention_schema_size_budget(budget)
    return canonical_json(
        {
            "attention_state_max_bytes": actual.attention_state_max_bytes,
            "attention_state_envelope_bytes": actual.attention_state_envelope_bytes,
            "candidate_array_bytes": actual.candidate_array_bytes,
            "candidate_array_envelope_bytes": actual.candidate_array_envelope_bytes,
            "candidate_count": actual.candidate_count,
            "candidate_counts": dict(actual.candidate_counts),
            "candidate_row_envelopes": [
                {
                    "eligible_focused_row_bytes": item.eligible_focused_row_bytes,
                    "field_maxima": dict(item.field_maxima),
                    "maximum_nonfocused_row_bytes": item.maximum_nonfocused_row_bytes,
                    "target_kind": item.target_kind,
                    "unavailable_nonfocused_row_bytes": item.unavailable_nonfocused_row_bytes,
                }
                for item in actual.candidate_row_envelopes
            ],
            "field_maxima": dict(actual.field_maxima),
            "maximum_value_bytes": actual.maximum_value_bytes,
            "margin_bytes": actual.margin_bytes,
            "receipts_array_bytes": actual.receipts_array_bytes,
            "receipts_array_envelope_bytes": actual.receipts_array_envelope_bytes,
            "revision_history_array_bytes": actual.revision_history_array_bytes,
            "revision_history_array_envelope_bytes": actual.revision_history_array_envelope_bytes,
        }
    )


__all__ = [
    "ATTENTION_AGENT_STATE_FUTURE_RESERVE_BYTES",
    "ATTENTION_AGENT_STATE_MAX_BYTES",
    "ATTENTION_AGENT_STATE_V8_BASE_MAX_BYTES",
    "ATTENTION_FIELD_NAME",
    "ATTENTION_ROOT_FIELD_OVERHEAD_BYTES",
    "ATTENTION_STATE_MAX_VALUE_BYTES",
    "AttentionCandidateRowEnvelope",
    "AttentionSchemaSizeBudget",
    "attention_candidate_capacity",
    "attention_candidate_capacity_by_kind",
    "canonical_attention_schema_size_budget",
    "derive_attention_schema_size_budget",
    "derive_attention_candidate_row_envelopes",
    "maximum_attention_continuity_fixture",
    "validate_attention_schema_size_budget",
]
