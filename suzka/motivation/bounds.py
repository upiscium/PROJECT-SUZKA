"""Schema-derived R13 U1 size budgets.

The bounds here are contract inputs, not runtime capacity authority.  They
construct worst-case canonical JSON shapes from every declared field bound and
derive exact byte maxima.  R13's future AgentState v8 projection is checked
against the integrated v7 schema, its exact root envelope, and the reserved
R14+ capacity under the existing 128 MiB whole-state cap.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import math
from typing import Final

from suzka.motivation.common import (
    R12_BELIEF_SCHEMA_MAX_BYTES,
    R13_AGENT_STATE_CAP_BYTES,
    R13_MAX_CONFLICT_REFS,
    R13_MAX_CANDIDATES_PER_DOMAIN,
    R13_MAX_DEPENDENCY_REFS,
    R13_MAX_EVIDENCE_REFS,
    R13_MAX_EVENT_SEQUENCE,
    R13_MAX_IDENTIFIER_CODEPOINTS,
    R13_MAX_RECORDS_PER_DOMAIN,
    R13_MAX_RELATED_REFS,
    R13_MAX_REVISION,
    R13_MAX_REVISION_HISTORY,
    R13_MAX_REVISION_WITNESSES,
    R13_MAX_SHORT_TEXT_CODEPOINTS,
    R13_MAX_SCOPE_CODEPOINTS,
    R13_MAX_SCOPE_ITEMS,
    R13_MAX_TEXT_CODEPOINTS,
    Deadline,
    R13Reference,
    R13ReferenceKind,
    R13_SCHEMA_VERSION,
    canonical_json,
)
from suzka.motivation.commitment import (
    COMMITMENT_BENEFICIARY_KINDS,
    COMMITMENT_EVIDENCE_KINDS,
    COMMITMENT_ORIGIN_KINDS,
    COMMITMENT_OUTCOME_KINDS,
    CommitmentLifecycle,
    CommitmentRevisionOperation,
    CommitmentRevisionReason,
    CommitmentRevisionRecord,
    CommitmentSubjectTransitionProof,
    commitment_id_for_proposal,
)
from suzka.motivation.goal import (
    GOAL_EVIDENCE_KINDS,
    GOAL_ORIGIN_KINDS,
    GOAL_OUTCOME_KINDS,
    GOAL_TARGET_KINDS,
    GoalLifecycle,
    GoalRevisionOperation,
    GoalRevisionReason,
    GoalRevisionRecord,
    GoalSubjectTransitionProof,
)
from suzka.motivation.motivation import (
    MOTIVATION_RELATED_KINDS,
    MOTIVATION_SOURCE_KINDS,
    MOTIVATION_TARGET_KINDS,
    MotivationCandidateClassification,
    MotivationKind,
    MotivationLifecycle,
    MotivationRevisionOperation,
    MotivationRevisionReason,
    MotivationRevisionRecord,
)

# Frozen capacity inputs copied from the integrated #277 AgentState v7 public
# budget.  Tests cross-check these values and the exact outer-envelope formula
# against AgentState's schema projector; the pure R13 package must not import
# ``suzka.runtime`` just to read its persisted-state size constants.
R13_AGENT_STATE_V7_BASE_MAX_BYTES: Final[int] = 85_826_260
R13_AGENT_STATE_FUTURE_RESERVE_BYTES: Final[int] = 16 * 1024 * 1024


def _project_agent_state_v8_without_reserve(r13_aggregate_bytes: int) -> int:
    """Project R13's top-level field over the complete integrated v7 base."""

    if type(r13_aggregate_bytes) is not int or r13_aggregate_bytes <= 0:
        raise ValueError("R13 aggregate maximum must be a positive byte count")
    schema_version_delta = len(canonical_json(8)) - len(canonical_json(7))
    added_root_field_overhead = len(canonical_json("motivation_state")) + 2
    return (
        R13_AGENT_STATE_V7_BASE_MAX_BYTES
        + schema_version_delta
        + added_root_field_overhead
        + r13_aggregate_bytes
    )


R13_MAX_HISTORY = R13_MAX_REVISION_HISTORY
R13_MAX_TEXT = R13_MAX_TEXT_CODEPOINTS
R13_MAX_SCOPE_TEXT = R13_MAX_SCOPE_CODEPOINTS
# References and event identifiers use the shared identifier bound.  Digest
# identities are narrower because validate_digest requires exactly 64 hex
# characters.
R13_MAX_REF = R13_MAX_IDENTIFIER_CODEPOINTS
R13_MAX_ID = 64
R13_MAX_DIGEST = 64
R13_MAX_DATETIME = "9999-12-31T23:59:59.999999Z"
_R13_MAX_DATETIME_VALUE = datetime(
    9999,
    12,
    31,
    23,
    59,
    59,
    999999,
    tzinfo=timezone.utc,
)
# ``bounded_fraction`` accepts every finite float in [0, 1].  The smallest
# positive subnormal has the widest ``float.hex`` representation in that
# domain (the minimum normal has the same exponent width), so derive the
# payload-width witness from the platform's actual IEEE-754 boundary rather
# than from an arbitrary representative value.
R13_MAX_FLOAT_HEX = math.nextafter(0.0, 1.0).hex()


def _max_text(codepoints: int) -> str:
    """Return a valid UTF-8 scalar sample with the largest JSON expansion."""

    return "\U00010000" * codepoints


def _max_ref(kind: str) -> dict[str, str]:
    return {"kind": kind, "reference": "r" * (R13_MAX_REF - 2) + "00"}


def _longest_string(values: Iterable[str]) -> str:
    return max(values, key=lambda value: (len(value), value))


def _longest_enum_value(enum_type: type[Enum]) -> str:
    return _longest_string(str(member.value) for member in enum_type)


def _longest_enum_member(enum_type: type[Enum]) -> Enum:
    return max(
        enum_type,
        key=lambda member: (len(str(member.value)), str(member.value)),
    )


def _longest_reference_kind(
    allowed_kinds: frozenset[R13ReferenceKind],
) -> str:
    return _longest_string(kind.value for kind in allowed_kinds)


def _max_typed_refs(count: int, kind: str) -> list[dict[str, str]]:
    return [
        {
            "kind": kind,
            "reference": "r" * (R13_MAX_REF - 2) + f"{index:02d}",
        }
        for index in range(count)
    ]


def _max_ids(count: int) -> list[str]:
    return [
        "i" * (R13_MAX_REF - 2) + f"{index:02d}"
        for index in range(count)
    ]


def _max_digest() -> str:
    return "d" * R13_MAX_DIGEST


def _max_digest_at(index: int) -> str:
    return "d" * (R13_MAX_DIGEST - 2) + f"{index:02x}"


def _revision_shape(record: object, authority_key: str) -> dict[str, object]:
    operation = getattr(record, "operation")
    reason = getattr(record, "reason")
    previous_state = getattr(record, "previous_lifecycle_state")
    shape = {
        "created_at": R13_MAX_DATETIME,
        "event_id": getattr(record, "event_id"),
        "event_sequence": getattr(record, "event_sequence"),
        "evidence_refs": list(getattr(record, "evidence_refs")),
        authority_key: getattr(record, authority_key),
        "operation": operation.value,
        "previous_revision_digest": getattr(record, "previous_revision_digest"),
        "previous_lifecycle_state": None
        if previous_state is None
        else getattr(previous_state, "value"),
        "record_digest": _max_digest(),
        "reason": reason.value,
        "revision": getattr(record, "revision"),
    }
    if hasattr(record, "state_digest"):
        shape["state_digest"] = _max_digest()
    if hasattr(record, "proposal_digest"):
        shape["proposal_digest"] = _max_digest()
    return shape


def _max_revision(
    authority_key: str,
    revision_record: Callable[..., object],
    operation_enum: type[Enum],
    reason_enum: type[Enum],
    lifecycle_enum: type[Enum],
) -> dict[str, object]:
    """Return the largest valid non-genesis revision representation.

    Revision operation/reason compatibility is enforced by each domain's
    revision record.  Enumerating those actual enum values avoids using a
    longer value from another domain (or an invalid operation/reason pair) as
    a sizing witness.
    """

    candidates: list[dict[str, object]] = []
    for operation in operation_enum:
        for reason in reason_enum:
            try:
                record = revision_record(
                    **{
                        authority_key: _max_digest(),
                        "revision": R13_MAX_REVISION,
                        "operation": operation,
                        "reason": reason,
                        "created_at": _R13_MAX_DATETIME_VALUE,
                        "event_id": "e" * R13_MAX_REF,
                        "event_sequence": R13_MAX_EVENT_SEQUENCE,
                        "evidence_refs": tuple(
                            _max_ids(R13_MAX_REVISION_WITNESSES)
                        ),
                        "previous_lifecycle_state": _longest_enum_member(
                            lifecycle_enum
                        ),
                        "previous_revision_digest": _max_digest(),
                        **(
                            {"state_digest": _max_digest()}
                            if hasattr(revision_record, "__dataclass_fields__")
                            and "state_digest"
                            in revision_record.__dataclass_fields__
                            else {}
                        ),
                        **(
                            {"proposal_digest": _max_digest()}
                            if hasattr(revision_record, "__dataclass_fields__")
                            and "proposal_digest"
                            in revision_record.__dataclass_fields__
                            else {}
                        ),
                    }
                )
            except ValueError:
                continue
            candidates.append(_revision_shape(record, authority_key))

    if not candidates:
        raise AssertionError("R13 revision vocabulary has no valid bounded pair")
    return max(
        candidates,
        key=lambda shape: (
            len(canonical_json(shape)),
            str(shape["operation"]),
            str(shape["reason"]),
        ),
    )


def _max_anchor(
    transitions: tuple[tuple[str, str, str, str], ...],
    *,
    includes_proposal_digest: bool = False,
    includes_state_digest: bool = False,
) -> dict[str, object]:
    candidates = []
    for state, previous_state, operation, reason in transitions:
        candidates.append(
            {
                "authority_id": "a" * R13_MAX_ID,
                "through_digest": _max_digest(),
                "through_created_at": R13_MAX_DATETIME,
                "through_event_id": "e" * R13_MAX_REF,
                "through_event_sequence": (
                    R13_MAX_EVENT_SEQUENCE - R13_MAX_REVISION_HISTORY
                ),
                "through_evidence_refs": _max_ids(R13_MAX_REVISION_WITNESSES),
                "through_operation": operation,
                "through_previous_revision_digest": _max_digest(),
                "through_previous_state": previous_state,
                "through_proposal_digest": (
                    _max_digest() if includes_proposal_digest else None
                ),
                "through_reason": reason,
                "through_revision": (
                    R13_MAX_REVISION - R13_MAX_REVISION_HISTORY
                ),
                "through_state": state,
                "through_state_digest": (
                    _max_digest() if includes_state_digest else None
                ),
            }
        )
    if not candidates:
        raise AssertionError("R13 compaction anchor has no transition witnesses")
    return max(candidates, key=lambda candidate: len(canonical_json(candidate)))


def _max_scope_items() -> list[str]:
    return [
        "\U00010000" * (R13_MAX_SCOPE_TEXT - 2) + f"{index:02d}"
        for index in range(R13_MAX_SCOPE_ITEMS)
    ]


def _max_goal_subject_transition_proof_shape() -> dict[str, object]:
    candidates = []
    for operation, reason, previous_state in (
        ("adopt", "subject_admission", "deferred"),
        ("defer", "subject_deferred", "adopted"),
        ("defer", "subject_deferred", "deferred"),
        ("abandon", "subject_abandoned", "adopted"),
        ("abandon", "subject_abandoned", "deferred"),
    ):
        proof = GoalSubjectTransitionProof(
            goal_id=_max_digest(),
            proposal_digest=_max_digest(),
            operation=GoalRevisionOperation(operation),
            reason=GoalRevisionReason(reason),
            previous_lifecycle_state=GoalLifecycle(previous_state),
            evidence_refs=tuple(_max_ids(R13_MAX_EVIDENCE_REFS)),
            event_id="e" * R13_MAX_REF,
            event_sequence=R13_MAX_EVENT_SEQUENCE,
        )
        candidates.append(proof.canonical_value())
    return max(candidates, key=lambda value: len(canonical_json(value)))


def _max_commitment_subject_transition_proof_shape() -> dict[str, object]:
    beneficiary = R13Reference(
        R13ReferenceKind.EXTERNAL_PARTY,
        "r" * (R13_MAX_REF - 2) + "00",
    )
    candidates = []
    proposal_digest = _max_digest()
    for operation, reason in (
        ("release", "subject_release"),
        ("renegotiate", "subject_renegotiation"),
    ):
        proof = CommitmentSubjectTransitionProof(
            commitment_id=commitment_id_for_proposal(proposal_digest),
            proposal_digest=proposal_digest,
            beneficiary=beneficiary,
            scope=tuple(_max_scope_items()),
            deadline=Deadline(_R13_MAX_DATETIME_VALUE),
            operation=CommitmentRevisionOperation(operation),
            reason=CommitmentRevisionReason(reason),
            previous_lifecycle_state=CommitmentLifecycle.ACTIVE,
            evidence_refs=tuple(_max_ids(R13_MAX_EVIDENCE_REFS)),
            event_id="e" * R13_MAX_REF,
            event_sequence=R13_MAX_EVENT_SEQUENCE,
        )
        candidates.append(proof.canonical_value())
    return max(candidates, key=lambda value: len(canonical_json(value)))


def max_motivation_record_shape() -> dict[str, object]:
    source_kind = _longest_reference_kind(MOTIVATION_SOURCE_KINDS)
    target_kind = _longest_reference_kind(MOTIVATION_TARGET_KINDS)
    conflict_kind = _longest_reference_kind(
        MOTIVATION_SOURCE_KINDS | {R13ReferenceKind.STATE}
    )
    related_kind = _longest_reference_kind(MOTIVATION_RELATED_KINDS)
    revision = _max_revision(
        "motivation_id",
        MotivationRevisionRecord,
        MotivationRevisionOperation,
        MotivationRevisionReason,
        MotivationLifecycle,
    )
    return {
        "conflict_refs": _max_typed_refs(R13_MAX_CONFLICT_REFS, conflict_kind),
        "evidence_refs": _max_typed_refs(R13_MAX_EVIDENCE_REFS, source_kind),
        "history_anchor": _max_anchor(
            (
                ("active", "satiated", "update", "evidence_update"),
                ("dormant", "satiated", "update", "decay"),
                ("satiated", "satiated", "update", "satiation"),
                ("retired", "satiated", "retire", "retirement"),
            ),
            includes_state_digest=True,
        ),
        "kind": _longest_enum_value(MotivationKind),
        "lifecycle": _longest_enum_value(MotivationLifecycle),
        "motivation_id": _max_digest(),
        "persistence": R13_MAX_FLOAT_HEX,
        "related_refs": _max_typed_refs(R13_MAX_RELATED_REFS, related_kind),
        "revision": R13_MAX_REVISION,
        "revision_history": [dict(revision) for _ in range(R13_MAX_HISTORY)],
        "satiation": R13_MAX_FLOAT_HEX,
        "schema_version": R13_SCHEMA_VERSION,
        "source_evidence": _max_ref(source_kind),
        "strength": R13_MAX_FLOAT_HEX,
        "target": _max_ref(target_kind),
        "uncertainty": R13_MAX_FLOAT_HEX,
    }


def max_goal_record_shape() -> dict[str, object]:
    target_kind = _longest_reference_kind(GOAL_TARGET_KINDS)
    origin_kind = _longest_reference_kind(GOAL_ORIGIN_KINDS)
    evidence_kind = _longest_reference_kind(GOAL_EVIDENCE_KINDS)
    outcome_kind = _longest_reference_kind(GOAL_OUTCOME_KINDS)
    revision = _max_revision(
        "goal_id",
        GoalRevisionRecord,
        GoalRevisionOperation,
        GoalRevisionReason,
        GoalLifecycle,
    )
    return {
        "conflicts": _max_ids(R13_MAX_CONFLICT_REFS),
        "deadline": R13_MAX_DATETIME,
        "dependencies": _max_ids(R13_MAX_DEPENDENCY_REFS),
        "description": _max_text(R13_MAX_TEXT),
        "evidence_refs": _max_typed_refs(R13_MAX_EVIDENCE_REFS, evidence_kind),
        "goal_id": _max_digest(),
        "history_anchor": _max_anchor(
            (
                ("adopted", "proposed", "adopt", "subject_admission"),
                ("deferred", "deferred", "defer", "subject_deferred"),
                ("abandoned", "deferred", "abandon", "subject_abandoned"),
                ("completed", "deferred", "complete", "verified_outcome"),
                ("failed", "deferred", "fail", "verified_outcome"),
            ),
            includes_proposal_digest=True,
        ),
        "lifecycle": _longest_enum_value(GoalLifecycle),
        "origin_refs": _max_typed_refs(R13_MAX_EVIDENCE_REFS, origin_kind),
        "outcome_evidence_refs": _max_typed_refs(
            R13_MAX_EVIDENCE_REFS, outcome_kind
        ),
        "proposal_digest": _max_digest(),
        "revision": R13_MAX_REVISION,
        "revision_history": [dict(revision) for _ in range(R13_MAX_HISTORY)],
        "schema_version": R13_SCHEMA_VERSION,
        "subject_admission": _max_digest(),
        "subject_transition_proofs": [
            _max_goal_subject_transition_proof_shape()
            for _ in range(R13_MAX_REVISION_HISTORY + 1)
        ],
        "target": _max_ref(target_kind),
    }


def max_commitment_record_shape() -> dict[str, object]:
    beneficiary_kind = _longest_reference_kind(COMMITMENT_BENEFICIARY_KINDS)
    evidence_kind = _longest_reference_kind(COMMITMENT_EVIDENCE_KINDS)
    origin_kind = _longest_reference_kind(COMMITMENT_ORIGIN_KINDS)
    outcome_kind = _longest_reference_kind(COMMITMENT_OUTCOME_KINDS)
    revision = _max_revision(
        "commitment_id",
        CommitmentRevisionRecord,
        CommitmentRevisionOperation,
        CommitmentRevisionReason,
        CommitmentLifecycle,
    )
    return {
        "admission": _max_digest(),
        "beneficiary": _max_ref(beneficiary_kind),
        "commitment_id": _max_digest(),
        "deadline": R13_MAX_DATETIME,
        "desire_refs": _max_typed_refs(
            R13_MAX_RELATED_REFS, R13ReferenceKind.MOTIVATION.value
        ),
        "evidence_refs": _max_typed_refs(R13_MAX_EVIDENCE_REFS, evidence_kind),
        "history_anchor": _max_anchor(
            (
                ("active", "proposed", "admit", "subject_admission"),
                ("released", "active", "release", "subject_release"),
                ("renegotiated", "active", "renegotiate", "subject_renegotiation"),
                ("fulfilled", "active", "fulfill", "verified_outcome"),
                ("breached", "active", "breach", "verified_outcome"),
            ),
        ),
        "lifecycle": _longest_enum_value(CommitmentLifecycle),
        "origin_refs": _max_typed_refs(R13_MAX_EVIDENCE_REFS, origin_kind),
        "outcome_evidence_refs": _max_typed_refs(
            R13_MAX_EVIDENCE_REFS, outcome_kind
        ),
        "proposal_digest": _max_digest(),
        "related_goal_refs": _max_typed_refs(
            R13_MAX_RELATED_REFS, R13ReferenceKind.GOAL.value
        ),
        "revision": R13_MAX_REVISION,
        "revision_history": [dict(revision) for _ in range(R13_MAX_HISTORY)],
        "schema_version": R13_SCHEMA_VERSION,
        "scope": _max_scope_items(),
        "subject": _max_text(R13_MAX_TEXT),
        "subject_transition_proofs": [
            _max_commitment_subject_transition_proof_shape()
        ],
    }


def max_motivation_candidate_shape() -> dict[str, object]:
    source_kind = _longest_reference_kind(MOTIVATION_SOURCE_KINDS)
    target_kind = _longest_reference_kind(MOTIVATION_TARGET_KINDS)
    return {
        "candidate_digest": _max_digest(),
        "classification": _longest_enum_value(MotivationCandidateClassification),
        "event_id": "e" * R13_MAX_REF,
        "event_sequence": R13_MAX_EVENT_SEQUENCE,
        "model_identity": {
            "model_id": _max_text(R13_MAX_SHORT_TEXT_CODEPOINTS),
            "model_key": "model." + "a" * R13_MAX_DIGEST,
            "provider_name": _max_text(R13_MAX_SHORT_TEXT_CODEPOINTS),
        },
        "source_evidence_refs": _max_typed_refs(
            R13_MAX_EVIDENCE_REFS, source_kind
        ),
        "suggested_kind": _longest_enum_value(MotivationKind),
        "suggested_persistence": R13_MAX_FLOAT_HEX,
        "suggested_satiation": R13_MAX_FLOAT_HEX,
        "suggested_strength": R13_MAX_FLOAT_HEX,
        "suggested_target": _max_ref(target_kind),
        "suggested_uncertainty": R13_MAX_FLOAT_HEX,
    }


def _max_section(
    record_shape: dict[str, object],
    *,
    count: int = R13_MAX_RECORDS_PER_DOMAIN,
) -> dict[str, object]:
    identity_key = next(
        (
            key
            for key in ("motivation_id", "goal_id", "commitment_id", "candidate_digest")
            if key in record_shape
        ),
        None,
    )
    records = []
    for index in range(count):
        record = dict(record_shape)
        if identity_key is not None:
            record[identity_key] = _max_digest_at(index)
        records.append(record)
    return {
        "authority_digest": _max_digest(),
        "records": records,
        "schema_version": R13_SCHEMA_VERSION,
    }


def _max_revision_payload_bytes(revision: dict[str, object]) -> int:
    return len(canonical_json(revision))


def _first_revision_shape(record: dict[str, object]) -> dict[str, object]:
    history = record["revision_history"]
    if (
        not isinstance(history, list)
        or not history
        or not isinstance(history[0], dict)
    ):
        raise AssertionError("R13 record shape has no revision witness")
    return history[0]


@dataclass(frozen=True, slots=True)
class R13SchemaSizeBudget:
    """Exact maxima derived from the declared U1 schema bounds."""

    motivation_revision_bytes: int
    goal_revision_bytes: int
    commitment_revision_bytes: int
    motivation_record_bytes: int
    goal_record_bytes: int
    commitment_record_bytes: int
    motivation_candidate_bytes: int
    motivation_section_bytes: int
    goal_section_bytes: int
    commitment_section_bytes: int
    motivation_candidate_section_bytes: int
    aggregate_r13_bytes: int
    agent_state_cap_bytes: int = R13_AGENT_STATE_CAP_BYTES
    agent_state_v7_base_bytes: int = R13_AGENT_STATE_V7_BASE_MAX_BYTES
    agent_state_future_state_reserve_bytes: int = R13_AGENT_STATE_FUTURE_RESERVE_BYTES
    projected_agent_state_v8_bytes: int = 0
    projected_agent_state_v8_with_reserve_bytes: int = 0
    r12_belief_schema_bytes: int = R12_BELIEF_SCHEMA_MAX_BYTES

    @property
    def remaining_after_r13_bytes(self) -> int:
        """Bytes below the hard cap before reserving R14+ capacity."""

        return self.agent_state_cap_bytes - self.projected_agent_state_v8_bytes

    @property
    def remaining_after_reserve_bytes(self) -> int:
        """Additional headroom after the requested future-state reserve."""

        return (
            self.agent_state_cap_bytes
            - self.projected_agent_state_v8_with_reserve_bytes
        )

    @property
    def has_provisional_capacity_headroom(self) -> bool:
        return (
            self.projected_agent_state_v8_with_reserve_bytes
            <= self.agent_state_cap_bytes
            and self.remaining_after_r13_bytes
            >= self.agent_state_future_state_reserve_bytes
        )

    def canonical_value(self) -> dict[str, int]:
        return {
            "aggregate_r13_bytes": self.aggregate_r13_bytes,
            "agent_state_cap_bytes": self.agent_state_cap_bytes,
            "agent_state_future_state_reserve_bytes": (
                self.agent_state_future_state_reserve_bytes
            ),
            "agent_state_v7_base_bytes": self.agent_state_v7_base_bytes,
            "commitment_record_bytes": self.commitment_record_bytes,
            "commitment_revision_bytes": self.commitment_revision_bytes,
            "commitment_section_bytes": self.commitment_section_bytes,
            "goal_record_bytes": self.goal_record_bytes,
            "goal_revision_bytes": self.goal_revision_bytes,
            "goal_section_bytes": self.goal_section_bytes,
            "motivation_candidate_bytes": self.motivation_candidate_bytes,
            "motivation_candidate_section_bytes": self.motivation_candidate_section_bytes,
            "motivation_record_bytes": self.motivation_record_bytes,
            "motivation_revision_bytes": self.motivation_revision_bytes,
            "motivation_section_bytes": self.motivation_section_bytes,
            "projected_agent_state_v8_bytes": self.projected_agent_state_v8_bytes,
            "projected_agent_state_v8_with_reserve_bytes": (
                self.projected_agent_state_v8_with_reserve_bytes
            ),
            "r12_belief_schema_bytes": self.r12_belief_schema_bytes,
            "remaining_after_reserve_bytes": self.remaining_after_reserve_bytes,
            "remaining_after_r13_bytes": self.remaining_after_r13_bytes,
        }


def derive_r13_schema_size_budget() -> R13SchemaSizeBudget:
    motivation = max_motivation_record_shape()
    goal = max_goal_record_shape()
    commitment = max_commitment_record_shape()
    candidate = max_motivation_candidate_shape()
    motivation_section = _max_section(motivation)
    goal_section = _max_section(goal)
    commitment_section = _max_section(commitment)
    candidate_section = _max_section(candidate, count=R13_MAX_CANDIDATES_PER_DOMAIN)
    aggregate = canonical_json(
        {
            "commitment": commitment_section,
            "goal": goal_section,
            "motivation": motivation_section,
            "motivation_candidates": candidate_section,
            "schema_version": R13_SCHEMA_VERSION,
        }
    )
    aggregate_bytes = len(aggregate)
    projected_v8 = _project_agent_state_v8_without_reserve(aggregate_bytes)
    projected_v8_with_reserve = (
        projected_v8 + R13_AGENT_STATE_FUTURE_RESERVE_BYTES
    )
    return R13SchemaSizeBudget(
        motivation_revision_bytes=_max_revision_payload_bytes(
            _first_revision_shape(motivation)
        ),
        goal_revision_bytes=_max_revision_payload_bytes(_first_revision_shape(goal)),
        commitment_revision_bytes=_max_revision_payload_bytes(
            _first_revision_shape(commitment)
        ),
        motivation_record_bytes=len(canonical_json(motivation)),
        goal_record_bytes=len(canonical_json(goal)),
        commitment_record_bytes=len(canonical_json(commitment)),
        motivation_candidate_bytes=len(canonical_json(candidate)),
        motivation_section_bytes=len(canonical_json(motivation_section)),
        goal_section_bytes=len(canonical_json(goal_section)),
        commitment_section_bytes=len(canonical_json(commitment_section)),
        motivation_candidate_section_bytes=len(canonical_json(candidate_section)),
        aggregate_r13_bytes=aggregate_bytes,
        projected_agent_state_v8_bytes=projected_v8,
        projected_agent_state_v8_with_reserve_bytes=projected_v8_with_reserve,
    )


R13_SCHEMA_SIZE_BUDGET: Final = derive_r13_schema_size_budget()
R13_SCHEMA_MAX_MOTIVATION_RECORD_BYTES: Final = (
    R13_SCHEMA_SIZE_BUDGET.motivation_record_bytes
)
R13_SCHEMA_MAX_GOAL_RECORD_BYTES: Final = R13_SCHEMA_SIZE_BUDGET.goal_record_bytes
R13_SCHEMA_MAX_COMMITMENT_RECORD_BYTES: Final = (
    R13_SCHEMA_SIZE_BUDGET.commitment_record_bytes
)
R13_SCHEMA_MAX_MOTIVATION_SECTION_BYTES: Final = (
    R13_SCHEMA_SIZE_BUDGET.motivation_section_bytes
)
R13_SCHEMA_MAX_GOAL_SECTION_BYTES: Final = R13_SCHEMA_SIZE_BUDGET.goal_section_bytes
R13_SCHEMA_MAX_COMMITMENT_SECTION_BYTES: Final = (
    R13_SCHEMA_SIZE_BUDGET.commitment_section_bytes
)
R13_SCHEMA_MAX_MOTIVATION_CANDIDATE_BYTES: Final = (
    R13_SCHEMA_SIZE_BUDGET.motivation_candidate_bytes
)
R13_SCHEMA_MAX_MOTIVATION_CANDIDATE_SECTION_BYTES: Final = (
    R13_SCHEMA_SIZE_BUDGET.motivation_candidate_section_bytes
)
R13_SCHEMA_MAX_AGGREGATE_BYTES: Final = R13_SCHEMA_SIZE_BUDGET.aggregate_r13_bytes


def validate_r13_schema_size_budget(
    budget: R13SchemaSizeBudget | None = None,
    ) -> R13SchemaSizeBudget:
    """Recompute and fail closed if the declared U1 budget is not feasible."""

    expected = derive_r13_schema_size_budget()
    actual = expected if budget is None else budget
    if actual != expected:
        raise ValueError("R13 schema budget does not match its declared bounds")
    if not actual.has_provisional_capacity_headroom:
        raise ValueError(
            "R13 AgentState v8 projection does not preserve the R14+ reserve under the hard cap"
        )
    return actual


def canonical_r13_schema_size_budget(
    budget: R13SchemaSizeBudget | None = None,
) -> bytes:
    return canonical_json(validate_r13_schema_size_budget(budget).canonical_value())


__all__ = [
    "R13_MAX_HISTORY",
    "R13_MAX_ID",
    "R13_MAX_REF",
    "R13_MAX_SCOPE_TEXT",
    "R13_MAX_TEXT",
    "R13_SCHEMA_MAX_AGGREGATE_BYTES",
    "R13_SCHEMA_MAX_COMMITMENT_RECORD_BYTES",
    "R13_SCHEMA_MAX_COMMITMENT_SECTION_BYTES",
    "R13_SCHEMA_MAX_GOAL_RECORD_BYTES",
    "R13_SCHEMA_MAX_GOAL_SECTION_BYTES",
    "R13_SCHEMA_MAX_MOTIVATION_CANDIDATE_BYTES",
    "R13_SCHEMA_MAX_MOTIVATION_CANDIDATE_SECTION_BYTES",
    "R13_SCHEMA_MAX_MOTIVATION_RECORD_BYTES",
    "R13_SCHEMA_MAX_MOTIVATION_SECTION_BYTES",
    "max_motivation_candidate_shape",
    "R13_SCHEMA_SIZE_BUDGET",
    "R13SchemaSizeBudget",
    "canonical_r13_schema_size_budget",
    "derive_r13_schema_size_budget",
    "max_commitment_record_shape",
    "max_goal_record_shape",
    "max_motivation_record_shape",
    "validate_r13_schema_size_budget",
]
