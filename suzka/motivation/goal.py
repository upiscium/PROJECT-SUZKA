"""Pure, bounded Goal contracts for R13 U1.

These values represent proposals, lifecycle facts, and admission witnesses.
They do not select winners, execute plans, schedule work, or admit state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Final, Iterable

from suzka.identifiers import validate_identifier
from suzka.motivation.common import (
    R13_MAX_CONFLICT_REFS,
    R13_MAX_DEPENDENCY_REFS,
    R13_MAX_EVIDENCE_REFS,
    R13_MAX_REVISION,
    R13_MAX_REVISION_HISTORY,
    R13_SCHEMA_VERSION,
    Deadline,
    R13Reference,
    R13ReferenceKind,
    RevisionCompactionAnchor,
    bounded_nonnegative_int,
    canonical_event,
    canonical_datetime,
    canonical_identifier_refs,
    canonical_json,
    canonical_references,
    canonical_revision_witnesses,
    coerce_deadline,
    digest_payload,
    normalize_text,
    utc_datetime,
    validate_digest,
    validate_revision_suffix,
)


GOAL_DOMAIN: Final = b"PROJECT-SUZKA:R13:GOAL:V1\0"
GOAL_ID_DOMAIN: Final = b"PROJECT-SUZKA:R13:GOAL-ID:V1\0"
GOAL_PROPOSAL_DOMAIN: Final = b"PROJECT-SUZKA:R13:GOAL-PROPOSAL:V1\0"
GOAL_ADMISSION_DOMAIN: Final = b"PROJECT-SUZKA:R13:GOAL-ADMISSION:V1\0"
GOAL_SUBJECT_TRANSITION_DOMAIN: Final = (
    b"PROJECT-SUZKA:R13:GOAL-SUBJECT-TRANSITION:V1\0"
)
GOAL_REVISION_DOMAIN: Final = b"PROJECT-SUZKA:R13:GOAL-REVISION:V1\0"

GOAL_TARGET_KINDS: Final = frozenset(
    {
        R13ReferenceKind.STATE,
        R13ReferenceKind.EXPERIENCE,
        R13ReferenceKind.VALUE,
        R13ReferenceKind.BELIEF,
        R13ReferenceKind.MOTIVATION,
        R13ReferenceKind.GOAL,
        R13ReferenceKind.COMMITMENT,
    }
)
GOAL_ORIGIN_KINDS: Final = frozenset(
    {
        R13ReferenceKind.MOTIVATION,
        R13ReferenceKind.EXPERIENCE,
        R13ReferenceKind.VALUE,
        R13ReferenceKind.BELIEF,
        R13ReferenceKind.USER_REQUEST,
        R13ReferenceKind.OPERATOR_PROPOSAL,
        R13ReferenceKind.EXTERNAL_REQUEST,
        R13ReferenceKind.SYSTEM,
    }
)
GOAL_EVIDENCE_KINDS: Final = GOAL_ORIGIN_KINDS | {
    R13ReferenceKind.EMOTION,
    R13ReferenceKind.EVENT,
    R13ReferenceKind.GOAL,
    R13ReferenceKind.COMMITMENT,
}
GOAL_OUTCOME_KINDS: Final = frozenset(
    {
        R13ReferenceKind.EVENT,
        R13ReferenceKind.EXPERIENCE,
        R13ReferenceKind.STATE,
    }
)


class _ClosedStrEnum(str, Enum):
    pass


class GoalLifecycle(_ClosedStrEnum):
    PROPOSED = "proposed"
    ADOPTED = "adopted"
    DEFERRED = "deferred"
    ABANDONED = "abandoned"
    COMPLETED = "completed"
    FAILED = "failed"


class GoalAdmissionReason(_ClosedStrEnum):
    SUBJECT_ENDORSEMENT = "subject_endorsement"
    SUBJECT_REVIEW = "subject_review"
    SUBJECT_CORRECTION = "subject_correction"


class GoalRevisionOperation(_ClosedStrEnum):
    CREATE = "create"
    ADOPT = "adopt"
    DEFER = "defer"
    ABANDON = "abandon"
    COMPLETE = "complete"
    FAIL = "fail"


class GoalRevisionReason(_ClosedStrEnum):
    CREATION = "creation"
    SUBJECT_ADMISSION = "subject_admission"
    SUBJECT_DEFERRED = "subject_deferred"
    SUBJECT_ABANDONED = "subject_abandoned"
    VERIFIED_OUTCOME = "verified_outcome"


def _enum(value: object, enum_type: type[Enum], name: str) -> None:
    if type(value) is not enum_type:
        raise TypeError(f"{name} must be a {enum_type.__name__}")


def _refs(
    value: object,
    name: str,
    maximum: int,
    allowed_kinds: frozenset[R13ReferenceKind],
    *,
    allow_empty: bool = True,
) -> tuple[R13Reference, ...]:
    return canonical_references(
        value,
        name,
        maximum=maximum,
        allow_empty=allow_empty,
        allowed_kinds=allowed_kinds,
    )


def goal_id_for_target(target: R13Reference) -> str:
    """Return stable Goal identity derived from the immutable target."""

    if not isinstance(target, R13Reference) or target.kind not in GOAL_TARGET_KINDS:
        raise ValueError("target is not an authorized Goal target reference")
    return digest_payload(GOAL_ID_DOMAIN, {"target": target.canonical_value()})


def goal_id_for_proposal(
    target: R13Reference,
    description: str,
    *,
    origin_refs: tuple[R13Reference, ...] = (),
) -> str:
    """Return deterministic identity; mutable lifecycle/proof fields are absent."""

    normalize_text(description, "description", 1_024)
    _refs(origin_refs, "origin_refs", R13_MAX_EVIDENCE_REFS, GOAL_ORIGIN_KINDS)
    return goal_id_for_target(target)


def _proposal_fields(
    *,
    target: R13Reference,
    description: str,
    deadline: Deadline,
    origin_refs: tuple[R13Reference, ...],
    evidence_refs: tuple[R13Reference, ...],
    dependencies: tuple[str, ...],
    conflicts: tuple[str, ...],
) -> dict[str, object]:
    return {
        "deadline": deadline.canonical_value(),
        "dependencies": list(dependencies),
        "description": description,
        "evidence_refs": [item.canonical_value() for item in evidence_refs],
        "conflicts": list(conflicts),
        "origin_refs": [item.canonical_value() for item in origin_refs],
        "target": target.canonical_value(),
    }


def goal_proposal_digest(
    target: R13Reference,
    description: str,
    *,
    deadline: Deadline | datetime | None = None,
    origin_refs: tuple[R13Reference, ...],
    evidence_refs: tuple[R13Reference, ...],
    dependencies: tuple[str, ...] = (),
    conflicts: tuple[str, ...] = (),
) -> str:
    target = target if isinstance(target, R13Reference) else target
    if not isinstance(target, R13Reference) or target.kind not in GOAL_TARGET_KINDS:
        raise ValueError("target is not an authorized Goal target reference")
    text = normalize_text(description, "description", 1_024)
    deadline_value = coerce_deadline(deadline)
    origins = _refs(
        origin_refs,
        "origin_refs",
        R13_MAX_EVIDENCE_REFS,
        GOAL_ORIGIN_KINDS,
        allow_empty=False,
    )
    evidence = _refs(
        evidence_refs,
        "evidence_refs",
        R13_MAX_EVIDENCE_REFS,
        GOAL_EVIDENCE_KINDS,
        allow_empty=False,
    )
    dependency_refs = canonical_identifier_refs(
        dependencies,
        "dependencies",
        maximum=R13_MAX_DEPENDENCY_REFS,
    )
    conflict_refs = canonical_identifier_refs(
        conflicts,
        "conflicts",
        maximum=R13_MAX_CONFLICT_REFS,
    )
    goal_id = goal_id_for_target(target)
    if any(
        item.kind is R13ReferenceKind.GOAL and item.reference == goal_id
        for item in evidence
    ):
        raise ValueError("a Goal cannot use itself as evidence")
    if goal_id in dependency_refs or goal_id in conflict_refs:
        raise ValueError("a Goal cannot depend on or conflict with itself")
    return digest_payload(
        GOAL_PROPOSAL_DOMAIN,
        _proposal_fields(
            target=target,
            description=text,
            deadline=deadline_value,
            origin_refs=origins,
            evidence_refs=evidence,
            dependencies=dependency_refs,
            conflicts=conflict_refs,
        ),
    )


@dataclass(frozen=True, slots=True)
class GoalSubjectAdmission:
    """Proof-shaped value; R16 is the first planned producer."""

    goal_id: str
    proposal_digest: str
    evidence_refs: tuple[str, ...]
    event_id: str
    event_sequence: int
    reason: GoalAdmissionReason
    admission_digest: str = field(init=False)

    def __post_init__(self) -> None:
        goal_id = validate_identifier(self.goal_id)
        proposal_digest = validate_digest(self.proposal_digest, "proposal_digest")
        evidence_refs = canonical_identifier_refs(
            self.evidence_refs,
            "evidence_refs",
            maximum=R13_MAX_EVIDENCE_REFS,
            allow_empty=False,
        )
        event_id, event_sequence = canonical_event(
            self.event_id,
            self.event_sequence,
        )
        if event_id is None or event_sequence is None:
            raise AssertionError("required admission event was lost")
        _enum(self.reason, GoalAdmissionReason, "reason")
        object.__setattr__(self, "goal_id", goal_id)
        object.__setattr__(self, "proposal_digest", proposal_digest)
        object.__setattr__(self, "evidence_refs", evidence_refs)
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "event_sequence", event_sequence)
        object.__setattr__(
            self,
            "admission_digest",
            digest_payload(
                GOAL_ADMISSION_DOMAIN,
                {
                    "event_id": event_id,
                    "event_sequence": event_sequence,
                    "evidence_refs": list(evidence_refs),
                    "goal_id": goal_id,
                    "proposal_digest": proposal_digest,
                    "reason": self.reason.value,
                },
            ),
        )

    @property
    def evidence_ref_ids(self) -> tuple[str, ...]:
        return self.evidence_refs

    @property
    def digest(self) -> str:
        return self.admission_digest


@dataclass(frozen=True, slots=True)
class GoalSubjectTransitionProof:
    """Fresh event-bound proof shape for a post-admission subject decision."""

    goal_id: str
    proposal_digest: str
    operation: GoalRevisionOperation
    reason: GoalRevisionReason
    previous_lifecycle_state: GoalLifecycle
    evidence_refs: tuple[str, ...]
    event_id: str
    event_sequence: int
    transition_digest: str = field(init=False)

    def __post_init__(self) -> None:
        goal_id = validate_identifier(self.goal_id)
        proposal_digest = validate_digest(self.proposal_digest, "proposal_digest")
        _enum(self.operation, GoalRevisionOperation, "operation")
        _enum(self.reason, GoalRevisionReason, "reason")
        _enum(
            self.previous_lifecycle_state,
            GoalLifecycle,
            "previous_lifecycle_state",
        )
        allowed = {
            (
                GoalRevisionOperation.ADOPT,
                GoalRevisionReason.SUBJECT_ADMISSION,
                GoalLifecycle.DEFERRED,
            ),
            (
                GoalRevisionOperation.DEFER,
                GoalRevisionReason.SUBJECT_DEFERRED,
                GoalLifecycle.ADOPTED,
            ),
            (
                GoalRevisionOperation.DEFER,
                GoalRevisionReason.SUBJECT_DEFERRED,
                GoalLifecycle.DEFERRED,
            ),
            (
                GoalRevisionOperation.ABANDON,
                GoalRevisionReason.SUBJECT_ABANDONED,
                GoalLifecycle.ADOPTED,
            ),
            (
                GoalRevisionOperation.ABANDON,
                GoalRevisionReason.SUBJECT_ABANDONED,
                GoalLifecycle.DEFERRED,
            ),
        }
        if (
            self.operation,
            self.reason,
            self.previous_lifecycle_state,
        ) not in allowed:
            raise ValueError("Goal subject transition proof has an invalid transition")
        evidence_refs = canonical_identifier_refs(
            self.evidence_refs,
            "evidence_refs",
            maximum=R13_MAX_EVIDENCE_REFS,
            allow_empty=False,
        )
        event_id, event_sequence = canonical_event(
            self.event_id,
            self.event_sequence,
        )
        if event_id is None or event_sequence is None:
            raise AssertionError("required Goal transition event was lost")
        object.__setattr__(self, "goal_id", goal_id)
        object.__setattr__(self, "proposal_digest", proposal_digest)
        object.__setattr__(self, "evidence_refs", evidence_refs)
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "event_sequence", event_sequence)
        object.__setattr__(
            self,
            "transition_digest",
            digest_payload(
                GOAL_SUBJECT_TRANSITION_DOMAIN,
                {
                    "event_id": event_id,
                    "event_sequence": event_sequence,
                    "evidence_refs": list(evidence_refs),
                    "goal_id": goal_id,
                    "operation": self.operation.value,
                    "previous_lifecycle_state": self.previous_lifecycle_state.value,
                    "proposal_digest": proposal_digest,
                    "reason": self.reason.value,
                },
            ),
        )

    @property
    def digest(self) -> str:
        return self.transition_digest

    def canonical_value(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "event_sequence": self.event_sequence,
            "evidence_refs": list(self.evidence_refs),
            "goal_id": self.goal_id,
            "operation": self.operation.value,
            "previous_lifecycle_state": self.previous_lifecycle_state.value,
            "proposal_digest": self.proposal_digest,
            "reason": self.reason.value,
            "transition_digest": self.transition_digest,
        }


def _revision_fields(record: GoalRevisionRecord) -> dict[str, object]:
    return {
        "created_at": record.created_at.isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        ),
        "event_id": record.event_id,
        "event_sequence": record.event_sequence,
        "evidence_refs": list(record.evidence_refs),
        "goal_id": record.goal_id,
        "operation": record.operation.value,
        "previous_revision_digest": record.previous_revision_digest,
        "previous_lifecycle_state": None
        if record.previous_lifecycle_state is None
        else record.previous_lifecycle_state.value,
        "proposal_digest": record.proposal_digest,
        "reason": record.reason.value,
        "revision": record.revision,
    }


def canonical_goal_revision_payload(record: GoalRevisionRecord) -> bytes:
    if not isinstance(record, GoalRevisionRecord):
        raise TypeError("record must be GoalRevisionRecord")
    return GOAL_REVISION_DOMAIN + canonical_json(_revision_fields(record))


def goal_revision_digest(record: GoalRevisionRecord) -> str:
    return digest_payload(GOAL_REVISION_DOMAIN, _revision_fields(record))


@dataclass(frozen=True, slots=True)
class GoalRevisionRecord:
    goal_id: str
    revision: int
    operation: GoalRevisionOperation
    reason: GoalRevisionReason
    created_at: datetime
    previous_lifecycle_state: GoalLifecycle | None
    proposal_digest: str
    event_id: str | None = None
    event_sequence: int | None = None
    evidence_refs: tuple[str, ...] = ()
    previous_revision_digest: str | None = None
    record_digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_id", validate_identifier(self.goal_id))
        object.__setattr__(
            self,
            "revision",
            bounded_nonnegative_int(
                self.revision,
                "revision",
                maximum=R13_MAX_REVISION,
            ),
        )
        _enum(self.operation, GoalRevisionOperation, "operation")
        _enum(self.reason, GoalRevisionReason, "reason")
        compatibility = {
            (GoalRevisionOperation.CREATE, GoalRevisionReason.CREATION),
            (GoalRevisionOperation.ADOPT, GoalRevisionReason.SUBJECT_ADMISSION),
            (GoalRevisionOperation.DEFER, GoalRevisionReason.SUBJECT_DEFERRED),
            (GoalRevisionOperation.ABANDON, GoalRevisionReason.SUBJECT_ABANDONED),
            (GoalRevisionOperation.COMPLETE, GoalRevisionReason.VERIFIED_OUTCOME),
            (GoalRevisionOperation.FAIL, GoalRevisionReason.VERIFIED_OUTCOME),
        }
        if (self.operation, self.reason) not in compatibility:
            raise ValueError("Goal revision operation and reason are incompatible")
        object.__setattr__(self, "created_at", utc_datetime(self.created_at, "created_at"))
        if self.previous_lifecycle_state is not None:
            _enum(
                self.previous_lifecycle_state,
                GoalLifecycle,
                "previous_lifecycle_state",
            )
        if self.revision == 0 and self.previous_lifecycle_state is not None:
            raise ValueError("Goal genesis cannot have a previous lifecycle state")
        if self.revision > 0 and self.previous_lifecycle_state is None:
            raise ValueError("non-genesis Goal revision requires its prior state")
        object.__setattr__(
            self,
            "proposal_digest",
            validate_digest(self.proposal_digest, "proposal_digest"),
        )
        event_id, event_sequence = canonical_event(
            self.event_id,
            self.event_sequence,
        )
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "event_sequence", event_sequence)
        object.__setattr__(
            self,
            "evidence_refs",
            canonical_revision_witnesses(self.evidence_refs),
        )
        if self.revision == 0:
            if self.operation is not GoalRevisionOperation.CREATE:
                raise ValueError("revision zero must create the Goal")
            if self.previous_revision_digest is not None:
                raise ValueError("genesis revision cannot have a previous digest")
        else:
            if self.operation is GoalRevisionOperation.CREATE:
                raise ValueError("non-genesis revisions cannot create the Goal")
            if self.previous_revision_digest is None:
                raise ValueError("non-genesis revision requires a previous digest")
        if self.previous_revision_digest is not None:
            validate_digest(self.previous_revision_digest, "previous_revision_digest")
        object.__setattr__(self, "record_digest", goal_revision_digest(self))

    @property
    def authority_id(self) -> str:
        return self.goal_id


def _require_goal_transition(
    *,
    goal_id: str,
    lifecycle: GoalLifecycle,
    revision_history: tuple[GoalRevisionRecord, ...],
    history_anchor: RevisionCompactionAnchor | None,
    admission: GoalSubjectAdmission | None,
    subject_transition_proofs: tuple[GoalSubjectTransitionProof, ...],
    outcome_evidence_refs: tuple[R13Reference, ...],
    proposal_digest: str,
) -> None:
    """Derive lifecycle through the complete retained transition suffix."""

    transitions = {
        GoalLifecycle.PROPOSED: {
            GoalRevisionOperation.ADOPT: GoalLifecycle.ADOPTED,
            GoalRevisionOperation.DEFER: GoalLifecycle.DEFERRED,
            GoalRevisionOperation.ABANDON: GoalLifecycle.ABANDONED,
        },
        GoalLifecycle.ADOPTED: {
            GoalRevisionOperation.DEFER: GoalLifecycle.DEFERRED,
            GoalRevisionOperation.ABANDON: GoalLifecycle.ABANDONED,
            GoalRevisionOperation.COMPLETE: GoalLifecycle.COMPLETED,
            GoalRevisionOperation.FAIL: GoalLifecycle.FAILED,
        },
        GoalLifecycle.DEFERRED: {
            GoalRevisionOperation.ADOPT: GoalLifecycle.ADOPTED,
            GoalRevisionOperation.DEFER: GoalLifecycle.DEFERRED,
            GoalRevisionOperation.ABANDON: GoalLifecycle.ABANDONED,
            GoalRevisionOperation.COMPLETE: GoalLifecycle.COMPLETED,
            GoalRevisionOperation.FAIL: GoalLifecycle.FAILED,
        },
        GoalLifecycle.ABANDONED: {},
        GoalLifecycle.COMPLETED: {},
        GoalLifecycle.FAILED: {},
    }
    subject_operations = {
        GoalRevisionOperation.ADOPT,
        GoalRevisionOperation.DEFER,
        GoalRevisionOperation.ABANDON,
    }
    used_proofs: set[str] = set()

    def validate_subject_transition(
        *,
        previous_state: GoalLifecycle,
        operation: GoalRevisionOperation,
        reason: GoalRevisionReason,
        event_id: str | None,
        event_sequence: int | None,
        evidence_refs: tuple[str, ...],
        initial: bool,
    ) -> None:
        if operation not in subject_operations:
            return
        if initial:
            if admission is None:
                raise ValueError("initial Goal decision lacks subject admission")
            expected_evidence = tuple(
                sorted((*admission.evidence_refs, admission.admission_digest))
            )
            if evidence_refs != expected_evidence:
                raise ValueError(
                    "initial Goal decision must bind its exact subject admission"
                )
            if event_id != admission.event_id or event_sequence != admission.event_sequence:
                raise ValueError(
                    "initial Goal decision is not bound to its admission event"
                )
            return

        matches = [
            proof
            for proof in subject_transition_proofs
            if proof.transition_digest in evidence_refs
        ]
        if len(matches) != 1:
            raise ValueError("Goal transition requires one fresh subject proof")
        proof = matches[0]
        expected_evidence = tuple(
            sorted((*proof.evidence_refs, proof.transition_digest))
        )
        if (
            proof.goal_id != goal_id
            or proof.proposal_digest != proposal_digest
            or proof.operation is not operation
            or proof.reason is not reason
            or proof.previous_lifecycle_state is not previous_state
            or proof.event_id != event_id
            or proof.event_sequence != event_sequence
            or evidence_refs != expected_evidence
        ):
            raise ValueError("Goal subject proof does not match its transition")
        if proof.transition_digest in used_proofs:
            raise ValueError("Goal subject proof cannot authorize multiple transitions")
        used_proofs.add(proof.transition_digest)

    if history_anchor is None:
        state: GoalLifecycle | None = None
    else:
        try:
            state = GoalLifecycle(history_anchor.through_state)
            anchor_operation = GoalRevisionOperation(history_anchor.through_operation)
            anchor_reason = GoalRevisionReason(history_anchor.through_reason)
            anchor_previous_state = (
                None
                if history_anchor.through_previous_state is None
                else GoalLifecycle(history_anchor.through_previous_state)
            )
        except ValueError as error:
            raise ValueError(
                "history anchor contains invalid Goal transition data"
            ) from error
        if (
            history_anchor.through_proposal_digest is None
            or history_anchor.through_state_digest is not None
            or history_anchor.through_proposal_digest != proposal_digest
        ):
            raise ValueError("Goal history anchor has invalid proposal evidence")
        anchored_revision = GoalRevisionRecord(
            goal_id=history_anchor.authority_id,
            revision=history_anchor.through_revision,
            operation=anchor_operation,
            reason=anchor_reason,
            created_at=history_anchor.through_created_at,
            previous_lifecycle_state=anchor_previous_state,
            proposal_digest=history_anchor.through_proposal_digest,
            event_id=history_anchor.through_event_id,
            event_sequence=history_anchor.through_event_sequence,
            evidence_refs=history_anchor.through_evidence_refs,
            previous_revision_digest=history_anchor.through_previous_revision_digest,
        )
        if anchored_revision.record_digest != history_anchor.through_digest:
            raise ValueError("Goal history anchor does not match its revision digest")
        if anchor_previous_state is None:
            if (
                history_anchor.through_revision != 0
                or anchor_operation is not GoalRevisionOperation.CREATE
                or anchor_reason is not GoalRevisionReason.CREATION
                or state is not GoalLifecycle.PROPOSED
            ):
                raise ValueError("Goal anchor has an invalid genesis transition")
        else:
            if transitions[anchor_previous_state].get(anchor_operation) is not state:
                raise ValueError("Goal anchor hides an invalid lifecycle transition")
            validate_subject_transition(
                previous_state=anchor_previous_state,
                operation=anchor_operation,
                reason=anchor_reason,
                event_id=history_anchor.through_event_id,
                event_sequence=history_anchor.through_event_sequence,
                evidence_refs=history_anchor.through_evidence_refs,
                initial=anchor_previous_state is GoalLifecycle.PROPOSED,
            )

    for item in revision_history:
        if state is None:
            if (
                item.revision != 0
                or item.previous_lifecycle_state is not None
                or item.operation is not GoalRevisionOperation.CREATE
                or item.reason is not GoalRevisionReason.CREATION
            ):
                raise ValueError("Goal history must begin with creation")
            state = GoalLifecycle.PROPOSED
            continue

        next_state = transitions[state].get(item.operation)
        if next_state is None:
            raise ValueError("Goal history contains an invalid lifecycle transition")
        if item.previous_lifecycle_state is not state:
            raise ValueError("Goal revision prior lifecycle state is inconsistent")
        validate_subject_transition(
            previous_state=state,
            operation=item.operation,
            reason=item.reason,
            event_id=item.event_id,
            event_sequence=item.event_sequence,
            evidence_refs=item.evidence_refs,
            initial=state is GoalLifecycle.PROPOSED,
        )
        if item.operation in {
            GoalRevisionOperation.COMPLETE,
            GoalRevisionOperation.FAIL,
        }:
            required_outcomes = {item.reference for item in outcome_evidence_refs}
            if not required_outcomes.issubset(item.evidence_refs):
                raise ValueError(
                    "Goal lifecycle transition lacks its verified outcome evidence"
                )
        state = next_state

    if state is not lifecycle:
        raise ValueError("Goal history does not reach its lifecycle")
    proof_digests = {proof.transition_digest for proof in subject_transition_proofs}
    if len(proof_digests) != len(subject_transition_proofs):
        raise ValueError("Goal subject transition proofs must be unique")
    if used_proofs != proof_digests:
        raise ValueError("Goal subject transition proofs must be referenced exactly once")


def _record_fields(record: GoalRecord) -> dict[str, object]:
    return {
        "conflicts": list(record.conflicts),
        "deadline": record.deadline.canonical_value(),
        "dependencies": list(record.dependencies),
        "description": record.description,
        "evidence_refs": [item.canonical_value() for item in record.evidence_refs],
        "goal_id": record.goal_id,
        "history_anchor": None
        if record.history_anchor is None
        else {
            "authority_id": record.history_anchor.authority_id,
            "through_digest": record.history_anchor.through_digest,
            "through_created_at": canonical_datetime(
                record.history_anchor.through_created_at
            ),
            "through_event_id": record.history_anchor.through_event_id,
            "through_event_sequence": record.history_anchor.through_event_sequence,
            "through_evidence_refs": list(
                record.history_anchor.through_evidence_refs
            ),
            "through_operation": record.history_anchor.through_operation,
            "through_previous_revision_digest": (
                record.history_anchor.through_previous_revision_digest
            ),
            "through_previous_state": record.history_anchor.through_previous_state,
            "through_proposal_digest": (
                record.history_anchor.through_proposal_digest
            ),
            "through_reason": record.history_anchor.through_reason,
            "through_revision": record.history_anchor.through_revision,
            "through_state": record.history_anchor.through_state,
            "through_state_digest": record.history_anchor.through_state_digest,
        },
        "lifecycle": record.lifecycle.value,
        "origin_refs": [item.canonical_value() for item in record.origin_refs],
        "outcome_evidence_refs": [
            item.canonical_value() for item in record.outcome_evidence_refs
        ],
        "proposal_digest": record.proposal_digest,
        "revision": record.revision,
        "revision_history": [
            {**_revision_fields(item), "record_digest": item.record_digest}
            for item in record.revision_history
        ],
        "schema_version": record.schema_version,
        "subject_transition_proofs": [
            proof.canonical_value() for proof in record.subject_transition_proofs
        ],
        "subject_admission": None
        if record.subject_admission is None
        else record.subject_admission.admission_digest,
        "target": record.target.canonical_value(),
    }


def canonical_goal_payload(record: GoalRecord) -> bytes:
    if not isinstance(record, GoalRecord):
        raise TypeError("record must be GoalRecord")
    return GOAL_DOMAIN + canonical_json(_record_fields(record))


def goal_record_digest(record: GoalRecord) -> str:
    return digest_payload(GOAL_DOMAIN, _record_fields(record))


@dataclass(frozen=True, slots=True)
class GoalRecord:
    goal_id: str
    target: R13Reference
    description: str
    lifecycle: GoalLifecycle = GoalLifecycle.PROPOSED
    deadline: Deadline = field(default_factory=Deadline)
    origin_refs: tuple[R13Reference, ...] = ()
    evidence_refs: tuple[R13Reference, ...] = ()
    dependencies: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    outcome_evidence_refs: tuple[R13Reference, ...] = ()
    subject_admission: GoalSubjectAdmission | None = None
    subject_transition_proofs: tuple[GoalSubjectTransitionProof, ...] = ()
    revision: int = 0
    revision_history: tuple[GoalRevisionRecord, ...] = ()
    history_anchor: RevisionCompactionAnchor | None = None
    schema_version: int = R13_SCHEMA_VERSION
    proposal_digest: str = field(init=False)
    record_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.target, R13Reference) or self.target.kind not in GOAL_TARGET_KINDS:
            raise ValueError("target is not an authorized Goal target reference")
        goal_id = validate_identifier(self.goal_id)
        expected_id = goal_id_for_target(self.target)
        if goal_id != expected_id:
            raise ValueError("goal_id does not match the immutable target")
        description = normalize_text(self.description, "description", 1_024)
        _enum(self.lifecycle, GoalLifecycle, "lifecycle")
        deadline = coerce_deadline(self.deadline)
        origins = _refs(
            self.origin_refs,
            "origin_refs",
            R13_MAX_EVIDENCE_REFS,
            GOAL_ORIGIN_KINDS,
            allow_empty=False,
        )
        evidence = _refs(
            self.evidence_refs,
            "evidence_refs",
            R13_MAX_EVIDENCE_REFS,
            GOAL_EVIDENCE_KINDS,
            allow_empty=False,
        )
        dependencies = canonical_identifier_refs(
            self.dependencies,
            "dependencies",
            maximum=R13_MAX_DEPENDENCY_REFS,
        )
        conflicts = canonical_identifier_refs(
            self.conflicts,
            "conflicts",
            maximum=R13_MAX_CONFLICT_REFS,
        )
        if goal_id in dependencies or goal_id in conflicts:
            raise ValueError("a Goal cannot depend on or conflict with itself")
        outcomes = _refs(
            self.outcome_evidence_refs,
            "outcome_evidence_refs",
            R13_MAX_EVIDENCE_REFS,
            GOAL_OUTCOME_KINDS,
        )
        if self.subject_admission is not None and not isinstance(
            self.subject_admission, GoalSubjectAdmission
        ):
            raise TypeError("subject_admission must be GoalSubjectAdmission")
        if type(self.subject_transition_proofs) is not tuple or any(
            not isinstance(proof, GoalSubjectTransitionProof)
            for proof in self.subject_transition_proofs
        ):
            raise TypeError(
                "subject_transition_proofs must contain GoalSubjectTransitionProof values"
            )
        if len(self.subject_transition_proofs) > R13_MAX_REVISION_HISTORY + 1:
            raise ValueError("subject_transition_proofs exceeds its bound")
        proof_digests = tuple(
            proof.transition_digest for proof in self.subject_transition_proofs
        )
        if proof_digests != tuple(sorted(set(proof_digests))):
            raise ValueError(
                "subject_transition_proofs must be sorted and unique"
            )
        if self.lifecycle is GoalLifecycle.PROPOSED:
            if self.subject_admission is not None:
                raise ValueError("proposed Goals cannot carry subject admission")
            if outcomes:
                raise ValueError("proposed Goals cannot carry outcome evidence")
        else:
            if self.subject_admission is None:
                raise ValueError("non-proposed Goals require subject admission")
            if self.subject_admission.goal_id != goal_id:
                raise ValueError("Goal admission identity mismatch")
            if self.subject_admission.evidence_refs != tuple(
                item.reference for item in evidence
            ):
                raise ValueError("Goal admission evidence must exactly match Goal evidence")
        if self.lifecycle in (GoalLifecycle.COMPLETED, GoalLifecycle.FAILED):
            if not outcomes:
                raise ValueError("verified terminal outcomes require outcome evidence")
        elif outcomes:
            raise ValueError("outcome evidence is terminal-only")
        if type(self.schema_version) is not int or self.schema_version != R13_SCHEMA_VERSION:
            raise ValueError("unsupported Goal schema version")
        if type(self.revision_history) is not tuple or any(
            not isinstance(item, GoalRevisionRecord) for item in self.revision_history
        ):
            raise TypeError("revision_history must contain GoalRevisionRecord values")
        revision = bounded_nonnegative_int(
            self.revision,
            "revision",
            maximum=R13_MAX_REVISION,
        )
        proposal = goal_proposal_digest(
            self.target,
            description,
            deadline=deadline,
            origin_refs=origins,
            evidence_refs=evidence,
            dependencies=dependencies,
            conflicts=conflicts,
        )
        if any(item.proposal_digest != proposal for item in self.revision_history):
            raise ValueError("Goal revision history does not bind its proposal")
        if self.subject_admission is not None:
            if self.subject_admission.proposal_digest != proposal:
                raise ValueError("Goal admission proposal digest mismatch")
        object.__setattr__(self, "goal_id", goal_id)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "deadline", deadline)
        object.__setattr__(self, "origin_refs", origins)
        object.__setattr__(self, "evidence_refs", evidence)
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "conflicts", conflicts)
        object.__setattr__(self, "outcome_evidence_refs", outcomes)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "proposal_digest", proposal)
        validate_revision_suffix(
            authority_id=goal_id,
            current_revision=revision,
            revision_history=self.revision_history,
            history_anchor=self.history_anchor,
            revision_digest=goal_revision_digest,
        )
        _require_goal_transition(
            goal_id=goal_id,
            lifecycle=self.lifecycle,
            revision_history=self.revision_history,
            history_anchor=self.history_anchor,
            admission=self.subject_admission,
            subject_transition_proofs=self.subject_transition_proofs,
            outcome_evidence_refs=outcomes,
            proposal_digest=proposal,
        )
        object.__setattr__(self, "record_digest", goal_record_digest(self))

    @property
    def admission(self) -> GoalSubjectAdmission | None:
        return self.subject_admission

    @property
    def history(self) -> tuple[GoalRevisionRecord, ...]:
        return self.revision_history

    @property
    def history_anchor_digest(self) -> str | None:
        return None if self.history_anchor is None else self.history_anchor.through_digest


def validate_goal_reference_graph(goals: Iterable[GoalRecord]) -> None:
    """Validate references/cycles without ranking or suspending any Goal."""

    records = tuple(goals)
    if any(not isinstance(goal, GoalRecord) for goal in records):
        raise TypeError("goal graph must contain GoalRecord values")
    by_id = {goal.goal_id: goal for goal in records}
    if len(by_id) != len(records):
        raise ValueError("goal graph contains duplicate identities")
    for goal in records:
        for reference in (*goal.dependencies, *goal.conflicts):
            if reference not in by_id:
                raise ValueError("goal graph contains a missing reference")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(goal_id: str) -> None:
        if goal_id in visiting:
            raise ValueError("goal dependency graph contains a cycle")
        if goal_id in visited:
            return
        visiting.add(goal_id)
        for dependency in by_id[goal_id].dependencies:
            visit(dependency)
        visiting.remove(goal_id)
        visited.add(goal_id)

    for goal_id in by_id:
        visit(goal_id)


GoalTargetReference = R13Reference
GoalAdmissionProof = GoalSubjectAdmission


__all__ = [
    "GOAL_ADMISSION_DOMAIN",
    "GOAL_DOMAIN",
    "GOAL_EVIDENCE_KINDS",
    "GOAL_ID_DOMAIN",
    "GOAL_ORIGIN_KINDS",
    "GOAL_PROPOSAL_DOMAIN",
    "GOAL_REVISION_DOMAIN",
    "GOAL_SUBJECT_TRANSITION_DOMAIN",
    "GOAL_TARGET_KINDS",
    "GoalAdmissionProof",
    "GoalAdmissionReason",
    "GoalLifecycle",
    "GoalRecord",
    "GoalRevisionOperation",
    "GoalRevisionReason",
    "GoalRevisionRecord",
    "GoalSubjectAdmission",
    "GoalSubjectTransitionProof",
    "GoalTargetReference",
    "canonical_goal_payload",
    "canonical_goal_revision_payload",
    "goal_id_for_proposal",
    "goal_id_for_target",
    "goal_proposal_digest",
    "goal_record_digest",
    "goal_revision_digest",
    "validate_goal_reference_graph",
]
