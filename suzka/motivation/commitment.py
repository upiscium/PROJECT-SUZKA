"""Pure, immutable Commitment contracts for R13 U1.

An external or operator proposal is not active responsibility.  This module
defines the proof-shaped admission value and lifecycle facts, but it does not
provide an admission producer, outcome producer, runtime, or persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Final

from suzka.identifiers import validate_identifier
from suzka.motivation.common import (
    R13_MAX_EVIDENCE_REFS,
    R13_MAX_RELATED_REFS,
    R13_MAX_REVISION,
    R13_MAX_SCOPE_CODEPOINTS,
    R13_MAX_SCOPE_ITEMS,
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


COMMITMENT_DOMAIN: Final = b"PROJECT-SUZKA:R13:COMMITMENT:V1\0"
COMMITMENT_ID_DOMAIN: Final = b"PROJECT-SUZKA:R13:COMMITMENT-ID:V1\0"
COMMITMENT_PROPOSAL_DOMAIN: Final = (
    b"PROJECT-SUZKA:R13:COMMITMENT-PROPOSAL:V1\0"
)
COMMITMENT_ADMISSION_DOMAIN: Final = (
    b"PROJECT-SUZKA:R13:COMMITMENT-ADMISSION:V1\0"
)
COMMITMENT_REVISION_DOMAIN: Final = (
    b"PROJECT-SUZKA:R13:COMMITMENT-REVISION:V1\0"
)

COMMITMENT_BENEFICIARY_KINDS: Final = frozenset(
    {R13ReferenceKind.SUBJECT, R13ReferenceKind.EXTERNAL_PARTY}
)
COMMITMENT_ORIGIN_KINDS: Final = frozenset(
    {
        R13ReferenceKind.GOAL,
        R13ReferenceKind.MOTIVATION,
        R13ReferenceKind.USER_REQUEST,
        R13ReferenceKind.OPERATOR_PROPOSAL,
        R13ReferenceKind.EXTERNAL_REQUEST,
        R13ReferenceKind.SYSTEM,
    }
)
COMMITMENT_EVIDENCE_KINDS: Final = COMMITMENT_ORIGIN_KINDS | {
    R13ReferenceKind.EXPERIENCE,
    R13ReferenceKind.EMOTION,
    R13ReferenceKind.VALUE,
    R13ReferenceKind.BELIEF,
    R13ReferenceKind.EVENT,
    R13ReferenceKind.GOAL,
    R13ReferenceKind.COMMITMENT,
}
COMMITMENT_OUTCOME_KINDS: Final = frozenset(
    {
        R13ReferenceKind.EVENT,
        R13ReferenceKind.EXPERIENCE,
        R13ReferenceKind.STATE,
    }
)


class _ClosedStrEnum(str, Enum):
    pass


class CommitmentLifecycle(_ClosedStrEnum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    RELEASED = "released"
    RENEGOTIATED = "renegotiated"
    FULFILLED = "fulfilled"
    BREACHED = "breached"


class CommitmentAdmissionReason(_ClosedStrEnum):
    SUBJECT_ENDORSEMENT = "subject_endorsement"
    SUBJECT_REVIEW = "subject_review"
    SUBJECT_RESPONSIBILITY = "subject_responsibility"


class CommitmentRevisionOperation(_ClosedStrEnum):
    CREATE = "create"
    ADMIT = "admit"
    RELEASE = "release"
    RENEGOTIATE = "renegotiate"
    FULFILL = "fulfill"
    BREACH = "breach"


class CommitmentRevisionReason(_ClosedStrEnum):
    CREATION = "creation"
    SUBJECT_ADMISSION = "subject_admission"
    SUBJECT_RELEASE = "subject_release"
    SUBJECT_RENEGOTIATION = "subject_renegotiation"
    VERIFIED_OUTCOME = "verified_outcome"


def _enum(value: object, enum_type: type[Enum], name: str) -> None:
    if type(value) is not enum_type:
        raise TypeError(f"{name} must be a {enum_type.__name__}")


def _references(
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


def _scope(value: object) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError("scope must be a tuple")
    if not value:
        raise ValueError("scope must be non-empty")
    if len(value) > R13_MAX_SCOPE_ITEMS:
        raise ValueError("scope exceeds its item bound")
    result = tuple(
        normalize_text(item, "scope item", R13_MAX_SCOPE_CODEPOINTS)
        for item in value
    )
    if result != tuple(sorted(set(result))):
        raise ValueError("scope must be canonically ordered and unique")
    return result


def _proposal_fields(
    *,
    subject: str,
    beneficiary: R13Reference,
    scope: tuple[str, ...],
    deadline: Deadline,
    evidence_refs: tuple[R13Reference, ...],
    origin_refs: tuple[R13Reference, ...],
    related_goal_refs: tuple[R13Reference, ...],
    desire_refs: tuple[R13Reference, ...],
) -> dict[str, object]:
    return {
        "beneficiary": beneficiary.canonical_value(),
        "deadline": deadline.canonical_value(),
        "desire_refs": [item.canonical_value() for item in desire_refs],
        "evidence_refs": [item.canonical_value() for item in evidence_refs],
        "origin_refs": [item.canonical_value() for item in origin_refs],
        "related_goal_refs": [item.canonical_value() for item in related_goal_refs],
        "scope": list(scope),
        "subject": subject,
    }


def commitment_proposal_digest(
    *,
    subject: str,
    beneficiary: R13Reference,
    scope: tuple[str, ...],
    deadline: Deadline | datetime | None = None,
    evidence_refs: tuple[R13Reference, ...],
    origin_refs: tuple[R13Reference, ...],
    related_goal_refs: tuple[R13Reference, ...] = (),
    desire_refs: tuple[R13Reference, ...] = (),
) -> str:
    subject = normalize_text(subject, "subject", 1_024)
    if not isinstance(beneficiary, R13Reference) or beneficiary.kind not in COMMITMENT_BENEFICIARY_KINDS:
        raise ValueError("beneficiary has an unauthorized reference kind")
    scope = _scope(scope)
    deadline = coerce_deadline(deadline)
    evidence = _references(
        evidence_refs,
        "evidence_refs",
        R13_MAX_EVIDENCE_REFS,
        COMMITMENT_EVIDENCE_KINDS,
        allow_empty=False,
    )
    origins = _references(
        origin_refs,
        "origin_refs",
        R13_MAX_EVIDENCE_REFS,
        COMMITMENT_ORIGIN_KINDS,
        allow_empty=False,
    )
    goals = _references(
        related_goal_refs,
        "related_goal_refs",
        R13_MAX_RELATED_REFS,
        frozenset({R13ReferenceKind.GOAL}),
    )
    desires = _references(
        desire_refs,
        "desire_refs",
        R13_MAX_RELATED_REFS,
        frozenset({R13ReferenceKind.MOTIVATION}),
    )
    return digest_payload(
        COMMITMENT_PROPOSAL_DOMAIN,
        _proposal_fields(
            subject=subject,
            beneficiary=beneficiary,
            scope=scope,
            deadline=deadline,
            evidence_refs=evidence,
            origin_refs=origins,
            related_goal_refs=goals,
            desire_refs=desires,
        ),
    )


def commitment_id_for_proposal(proposal_digest: str) -> str:
    return digest_payload(
        COMMITMENT_ID_DOMAIN,
        {"proposal_digest": validate_digest(proposal_digest, "proposal_digest")},
    )


def commitment_id_for_fields(
    *,
    subject: str,
    beneficiary: R13Reference,
    scope: tuple[str, ...],
    deadline: Deadline | datetime | None = None,
    evidence_refs: tuple[R13Reference, ...],
    origin_refs: tuple[R13Reference, ...],
    related_goal_refs: tuple[R13Reference, ...] = (),
    desire_refs: tuple[R13Reference, ...] = (),
) -> str:
    return commitment_id_for_proposal(
        commitment_proposal_digest(
            subject=subject,
            beneficiary=beneficiary,
            scope=scope,
            deadline=deadline,
            evidence_refs=evidence_refs,
            origin_refs=origin_refs,
            related_goal_refs=related_goal_refs,
            desire_refs=desire_refs,
        )
    )


@dataclass(frozen=True, slots=True)
class CommitmentSubjectAdmission:
    """Exact responsibility proof shape; it does not authenticate its producer."""

    commitment_id: str
    proposal_digest: str
    beneficiary: R13Reference
    scope: tuple[str, ...]
    deadline: Deadline
    evidence_refs: tuple[str, ...]
    event_id: str
    event_sequence: int
    reason: CommitmentAdmissionReason
    admission_digest: str = field(init=False)

    def __post_init__(self) -> None:
        commitment_id = validate_identifier(self.commitment_id)
        proposal_digest = validate_digest(self.proposal_digest, "proposal_digest")
        if commitment_id != commitment_id_for_proposal(proposal_digest):
            raise ValueError("commitment_id is not derived from proposal_digest")
        if not isinstance(self.beneficiary, R13Reference) or self.beneficiary.kind not in COMMITMENT_BENEFICIARY_KINDS:
            raise ValueError("beneficiary has an unauthorized reference kind")
        scope = _scope(self.scope)
        deadline = coerce_deadline(self.deadline)
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
        _enum(self.reason, CommitmentAdmissionReason, "reason")
        object.__setattr__(self, "commitment_id", commitment_id)
        object.__setattr__(self, "proposal_digest", proposal_digest)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "deadline", deadline)
        object.__setattr__(self, "evidence_refs", evidence_refs)
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "event_sequence", event_sequence)
        object.__setattr__(
            self,
            "admission_digest",
            digest_payload(
                COMMITMENT_ADMISSION_DOMAIN,
                {
                    "beneficiary": self.beneficiary.canonical_value(),
                    "commitment_id": commitment_id,
                    "deadline": deadline.canonical_value(),
                    "event_id": event_id,
                    "event_sequence": event_sequence,
                    "evidence_refs": list(evidence_refs),
                    "proposal_digest": proposal_digest,
                    "reason": self.reason.value,
                    "scope": list(scope),
                },
            ),
        )

    @property
    def digest(self) -> str:
        return self.admission_digest


def _revision_fields(record: CommitmentRevisionRecord) -> dict[str, object]:
    return {
        "commitment_id": record.commitment_id,
        "created_at": record.created_at.isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        ),
        "event_id": record.event_id,
        "event_sequence": record.event_sequence,
        "evidence_refs": list(record.evidence_refs),
        "operation": record.operation.value,
        "previous_revision_digest": record.previous_revision_digest,
        "previous_lifecycle_state": None
        if record.previous_lifecycle_state is None
        else record.previous_lifecycle_state.value,
        "reason": record.reason.value,
        "revision": record.revision,
    }


def canonical_commitment_revision_payload(
    record: CommitmentRevisionRecord,
) -> bytes:
    if not isinstance(record, CommitmentRevisionRecord):
        raise TypeError("record must be CommitmentRevisionRecord")
    return COMMITMENT_REVISION_DOMAIN + canonical_json(_revision_fields(record))


def commitment_revision_digest(record: CommitmentRevisionRecord) -> str:
    return digest_payload(COMMITMENT_REVISION_DOMAIN, _revision_fields(record))


@dataclass(frozen=True, slots=True)
class CommitmentRevisionRecord:
    commitment_id: str
    revision: int
    operation: CommitmentRevisionOperation
    reason: CommitmentRevisionReason
    created_at: datetime
    previous_lifecycle_state: CommitmentLifecycle | None
    event_id: str | None = None
    event_sequence: int | None = None
    evidence_refs: tuple[str, ...] = ()
    previous_revision_digest: str | None = None
    record_digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "commitment_id", validate_identifier(self.commitment_id))
        object.__setattr__(
            self,
            "revision",
            bounded_nonnegative_int(
                self.revision,
                "revision",
                maximum=R13_MAX_REVISION,
            ),
        )
        _enum(self.operation, CommitmentRevisionOperation, "operation")
        _enum(self.reason, CommitmentRevisionReason, "reason")
        compatibility = {
            (CommitmentRevisionOperation.CREATE, CommitmentRevisionReason.CREATION),
            (CommitmentRevisionOperation.ADMIT, CommitmentRevisionReason.SUBJECT_ADMISSION),
            (CommitmentRevisionOperation.RELEASE, CommitmentRevisionReason.SUBJECT_RELEASE),
            (
                CommitmentRevisionOperation.RENEGOTIATE,
                CommitmentRevisionReason.SUBJECT_RENEGOTIATION,
            ),
            (CommitmentRevisionOperation.FULFILL, CommitmentRevisionReason.VERIFIED_OUTCOME),
            (CommitmentRevisionOperation.BREACH, CommitmentRevisionReason.VERIFIED_OUTCOME),
        }
        if (self.operation, self.reason) not in compatibility:
            raise ValueError("Commitment revision operation and reason are incompatible")
        object.__setattr__(self, "created_at", utc_datetime(self.created_at, "created_at"))
        if self.previous_lifecycle_state is not None:
            _enum(
                self.previous_lifecycle_state,
                CommitmentLifecycle,
                "previous_lifecycle_state",
            )
        if self.revision == 0 and self.previous_lifecycle_state is not None:
            raise ValueError("Commitment genesis cannot have a previous lifecycle state")
        if self.revision > 0 and self.previous_lifecycle_state is None:
            raise ValueError("non-genesis Commitment revision requires its prior state")
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
            if self.operation is not CommitmentRevisionOperation.CREATE:
                raise ValueError("revision zero must create the Commitment")
            if self.previous_revision_digest is not None:
                raise ValueError("genesis revision cannot have a previous digest")
        else:
            if self.operation is CommitmentRevisionOperation.CREATE:
                raise ValueError("non-genesis revisions cannot create the Commitment")
            if self.previous_revision_digest is None:
                raise ValueError("non-genesis revision requires a previous digest")
        if self.previous_revision_digest is not None:
            validate_digest(self.previous_revision_digest, "previous_revision_digest")
        object.__setattr__(self, "record_digest", commitment_revision_digest(self))

    @property
    def authority_id(self) -> str:
        return self.commitment_id


def _require_commitment_transition(
    *,
    lifecycle: CommitmentLifecycle,
    revision_history: tuple[CommitmentRevisionRecord, ...],
    history_anchor: RevisionCompactionAnchor | None,
    admission: CommitmentSubjectAdmission | None,
    outcome_evidence_refs: tuple[R13Reference, ...],
) -> None:
    """Derive responsibility lifecycle through the retained transition suffix."""

    transitions = {
        CommitmentLifecycle.PROPOSED: {
            CommitmentRevisionOperation.ADMIT: CommitmentLifecycle.ACTIVE,
        },
        CommitmentLifecycle.ACTIVE: {
            CommitmentRevisionOperation.RELEASE: CommitmentLifecycle.RELEASED,
            CommitmentRevisionOperation.RENEGOTIATE: CommitmentLifecycle.RENEGOTIATED,
            CommitmentRevisionOperation.FULFILL: CommitmentLifecycle.FULFILLED,
            CommitmentRevisionOperation.BREACH: CommitmentLifecycle.BREACHED,
        },
        CommitmentLifecycle.RELEASED: {},
        CommitmentLifecycle.RENEGOTIATED: {},
        CommitmentLifecycle.FULFILLED: {},
        CommitmentLifecycle.BREACHED: {},
    }

    if history_anchor is None:
        state: CommitmentLifecycle | None = None
    else:
        try:
            state = CommitmentLifecycle(history_anchor.through_state)
            anchor_operation = CommitmentRevisionOperation(
                history_anchor.through_operation
            )
            anchor_reason = CommitmentRevisionReason(history_anchor.through_reason)
            anchor_previous_state = (
                None
                if history_anchor.through_previous_state is None
                else CommitmentLifecycle(history_anchor.through_previous_state)
            )
        except ValueError as error:
            raise ValueError(
                "history anchor contains invalid Commitment transition data"
            ) from error
        if (
            history_anchor.through_proposal_digest is not None
            or history_anchor.through_state_digest is not None
        ):
            raise ValueError("Commitment history anchor contains foreign digest fields")
        anchored_revision = CommitmentRevisionRecord(
            commitment_id=history_anchor.authority_id,
            revision=history_anchor.through_revision,
            operation=anchor_operation,
            reason=anchor_reason,
            created_at=history_anchor.through_created_at,
            previous_lifecycle_state=anchor_previous_state,
            event_id=history_anchor.through_event_id,
            event_sequence=history_anchor.through_event_sequence,
            evidence_refs=history_anchor.through_evidence_refs,
            previous_revision_digest=history_anchor.through_previous_revision_digest,
        )
        if anchored_revision.record_digest != history_anchor.through_digest:
            raise ValueError(
                "Commitment history anchor does not match its revision digest"
            )
        if anchor_previous_state is None:
            if (
                history_anchor.through_revision != 0
                or anchor_operation is not CommitmentRevisionOperation.CREATE
                or anchor_reason is not CommitmentRevisionReason.CREATION
                or state is not CommitmentLifecycle.PROPOSED
            ):
                raise ValueError(
                    "Commitment anchor has an invalid genesis transition"
                )
        else:
            if transitions[anchor_previous_state].get(anchor_operation) is not state:
                raise ValueError(
                    "Commitment anchor hides an invalid lifecycle transition"
                )
            if admission is None or admission.admission_digest not in (
                history_anchor.through_evidence_refs
            ):
                raise ValueError(
                    "Commitment anchor lacks its subject admission witness"
                )
            if anchor_previous_state is CommitmentLifecycle.PROPOSED and (
                history_anchor.through_event_id != admission.event_id
                or history_anchor.through_event_sequence != admission.event_sequence
            ):
                raise ValueError(
                    "Commitment anchor is not bound to its admission event"
                )

    for item in revision_history:
        if state is None:
            if (
                item.revision != 0
                or item.previous_lifecycle_state is not None
                or item.operation is not CommitmentRevisionOperation.CREATE
                or item.reason is not CommitmentRevisionReason.CREATION
            ):
                raise ValueError("Commitment history must begin with creation")
            state = CommitmentLifecycle.PROPOSED
            continue

        next_state = transitions[state].get(item.operation)
        if next_state is None:
            raise ValueError(
                "Commitment history contains an invalid lifecycle transition"
            )
        if item.previous_lifecycle_state is not state:
            raise ValueError(
                "Commitment revision prior lifecycle state is inconsistent"
            )
        if admission is None or admission.admission_digest not in item.evidence_refs:
            raise ValueError(
                "Commitment transition lacks its subject admission witness"
            )
        if state is CommitmentLifecycle.PROPOSED and (
            item.event_id != admission.event_id
            or item.event_sequence != admission.event_sequence
        ):
            raise ValueError(
                "Commitment activation is not bound to its admission event"
            )
        if item.operation in {
            CommitmentRevisionOperation.FULFILL,
            CommitmentRevisionOperation.BREACH,
        }:
            required_outcomes = {item.reference for item in outcome_evidence_refs}
            if not required_outcomes.issubset(item.evidence_refs):
                raise ValueError(
                    "Commitment transition lacks its verified outcome evidence"
                )
        state = next_state

    if state is not lifecycle:
        raise ValueError("Commitment history does not reach its lifecycle")


def _record_fields(record: CommitmentRecord) -> dict[str, object]:
    return {
        "admission": None
        if record.subject_admission is None
        else record.subject_admission.admission_digest,
        "beneficiary": record.beneficiary.canonical_value(),
        "commitment_id": record.commitment_id,
        "deadline": record.deadline.canonical_value(),
        "desire_refs": [item.canonical_value() for item in record.desire_refs],
        "evidence_refs": [item.canonical_value() for item in record.evidence_refs],
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
        "related_goal_refs": [
            item.canonical_value() for item in record.related_goal_refs
        ],
        "revision": record.revision,
        "revision_history": [
            {**_revision_fields(item), "record_digest": item.record_digest}
            for item in record.revision_history
        ],
        "schema_version": record.schema_version,
        "scope": list(record.scope),
        "subject": record.subject,
    }


def canonical_commitment_payload(record: CommitmentRecord) -> bytes:
    if not isinstance(record, CommitmentRecord):
        raise TypeError("record must be CommitmentRecord")
    return COMMITMENT_DOMAIN + canonical_json(_record_fields(record))


def commitment_record_digest(record: CommitmentRecord) -> str:
    if not isinstance(record, CommitmentRecord):
        raise TypeError("record must be CommitmentRecord")
    return digest_payload(COMMITMENT_DOMAIN, _record_fields(record))


@dataclass(frozen=True, slots=True)
class CommitmentRecord:
    subject: str
    beneficiary: R13Reference
    scope: tuple[str, ...]
    deadline: Deadline = field(default_factory=Deadline)
    evidence_refs: tuple[R13Reference, ...] = ()
    origin_refs: tuple[R13Reference, ...] = ()
    related_goal_refs: tuple[R13Reference, ...] = ()
    desire_refs: tuple[R13Reference, ...] = ()
    lifecycle: CommitmentLifecycle = CommitmentLifecycle.PROPOSED
    subject_admission: CommitmentSubjectAdmission | None = None
    outcome_evidence_refs: tuple[R13Reference, ...] = ()
    revision: int = 0
    revision_history: tuple[CommitmentRevisionRecord, ...] = ()
    history_anchor: RevisionCompactionAnchor | None = None
    schema_version: int = R13_SCHEMA_VERSION
    commitment_id: str = field(init=False)
    proposal_digest: str = field(init=False)
    record_digest: str = field(init=False)

    def __post_init__(self) -> None:
        subject = normalize_text(self.subject, "subject", 1_024)
        if not isinstance(self.beneficiary, R13Reference) or self.beneficiary.kind not in COMMITMENT_BENEFICIARY_KINDS:
            raise ValueError("beneficiary has an unauthorized reference kind")
        scope = _scope(self.scope)
        deadline = coerce_deadline(self.deadline)
        evidence = _references(
            self.evidence_refs,
            "evidence_refs",
            R13_MAX_EVIDENCE_REFS,
            COMMITMENT_EVIDENCE_KINDS,
            allow_empty=False,
        )
        origins = _references(
            self.origin_refs,
            "origin_refs",
            R13_MAX_EVIDENCE_REFS,
            COMMITMENT_ORIGIN_KINDS,
            allow_empty=False,
        )
        related_goals = _references(
            self.related_goal_refs,
            "related_goal_refs",
            R13_MAX_RELATED_REFS,
            frozenset({R13ReferenceKind.GOAL}),
        )
        desires = _references(
            self.desire_refs,
            "desire_refs",
            R13_MAX_RELATED_REFS,
            frozenset({R13ReferenceKind.MOTIVATION}),
        )
        outcomes = _references(
            self.outcome_evidence_refs,
            "outcome_evidence_refs",
            R13_MAX_EVIDENCE_REFS,
            COMMITMENT_OUTCOME_KINDS,
        )
        if self.subject_admission is not None and not isinstance(
            self.subject_admission, CommitmentSubjectAdmission
        ):
            raise TypeError("subject_admission must be CommitmentSubjectAdmission")
        _enum(self.lifecycle, CommitmentLifecycle, "lifecycle")
        if self.lifecycle is CommitmentLifecycle.PROPOSED:
            if self.subject_admission is not None:
                raise ValueError("proposed Commitments cannot carry subject admission")
            if outcomes:
                raise ValueError("proposed Commitments cannot carry outcome evidence")
        else:
            if self.subject_admission is None:
                raise ValueError("non-proposed Commitments require subject admission")
            if outcomes and self.lifecycle not in {
                CommitmentLifecycle.FULFILLED,
                CommitmentLifecycle.BREACHED,
            }:
                raise ValueError("outcome evidence is terminal-only")
        if self.lifecycle in {
            CommitmentLifecycle.FULFILLED,
            CommitmentLifecycle.BREACHED,
        } and not outcomes:
            raise ValueError("verified terminal outcomes require outcome evidence")
        proposal = commitment_proposal_digest(
            subject=subject,
            beneficiary=self.beneficiary,
            scope=scope,
            deadline=deadline,
            evidence_refs=evidence,
            origin_refs=origins,
            related_goal_refs=related_goals,
            desire_refs=desires,
        )
        commitment_id = commitment_id_for_proposal(proposal)
        if self.subject_admission is not None:
            admission = self.subject_admission
            if admission.commitment_id != commitment_id:
                raise ValueError("Commitment admission identity mismatch")
            if admission.proposal_digest != proposal:
                raise ValueError("Commitment admission proposal digest mismatch")
            if admission.beneficiary != self.beneficiary:
                raise ValueError("Commitment admission beneficiary mismatch")
            if admission.scope != scope or admission.deadline != deadline:
                raise ValueError("Commitment admission scope/deadline mismatch")
            if admission.evidence_refs != tuple(item.reference for item in evidence):
                raise ValueError("Commitment admission evidence must exactly match Commitment")
        revision = bounded_nonnegative_int(
            self.revision,
            "revision",
            maximum=R13_MAX_REVISION,
        )
        if type(self.schema_version) is not int or self.schema_version != R13_SCHEMA_VERSION:
            raise ValueError("unsupported Commitment schema version")
        if type(self.revision_history) is not tuple or any(
            not isinstance(item, CommitmentRevisionRecord)
            for item in self.revision_history
        ):
            raise TypeError("revision_history must contain CommitmentRevisionRecord values")
        object.__setattr__(self, "subject", subject)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "deadline", deadline)
        object.__setattr__(self, "evidence_refs", evidence)
        object.__setattr__(self, "origin_refs", origins)
        object.__setattr__(self, "related_goal_refs", related_goals)
        object.__setattr__(self, "desire_refs", desires)
        object.__setattr__(self, "outcome_evidence_refs", outcomes)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "proposal_digest", proposal)
        object.__setattr__(self, "commitment_id", commitment_id)
        validate_revision_suffix(
            authority_id=commitment_id,
            current_revision=revision,
            revision_history=self.revision_history,
            history_anchor=self.history_anchor,
            revision_digest=commitment_revision_digest,
        )
        _require_commitment_transition(
            lifecycle=self.lifecycle,
            revision_history=self.revision_history,
            history_anchor=self.history_anchor,
            admission=self.subject_admission,
            outcome_evidence_refs=outcomes,
        )
        object.__setattr__(self, "record_digest", commitment_record_digest(self))

    @property
    def admission(self) -> CommitmentSubjectAdmission | None:
        return self.subject_admission

    @property
    def history(self) -> tuple[CommitmentRevisionRecord, ...]:
        return self.revision_history

    @property
    def history_anchor_digest(self) -> str | None:
        return None if self.history_anchor is None else self.history_anchor.through_digest


CommitmentAdmissionProof = CommitmentSubjectAdmission
CommitmentResponsibilityAdmission = CommitmentSubjectAdmission


__all__ = [
    "COMMITMENT_ADMISSION_DOMAIN",
    "COMMITMENT_BENEFICIARY_KINDS",
    "COMMITMENT_DOMAIN",
    "COMMITMENT_EVIDENCE_KINDS",
    "COMMITMENT_ID_DOMAIN",
    "COMMITMENT_ORIGIN_KINDS",
    "COMMITMENT_PROPOSAL_DOMAIN",
    "COMMITMENT_REVISION_DOMAIN",
    "CommitmentAdmissionProof",
    "CommitmentAdmissionReason",
    "CommitmentLifecycle",
    "CommitmentRecord",
    "CommitmentResponsibilityAdmission",
    "CommitmentRevisionOperation",
    "CommitmentRevisionReason",
    "CommitmentRevisionRecord",
    "CommitmentSubjectAdmission",
    "canonical_commitment_payload",
    "canonical_commitment_revision_payload",
    "commitment_id_for_fields",
    "commitment_id_for_proposal",
    "commitment_proposal_digest",
    "commitment_record_digest",
    "commitment_revision_digest",
]
