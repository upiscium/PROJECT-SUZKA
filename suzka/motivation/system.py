"""Bounded, process-local Motivation mutation authority for R13 U2.

The public U1 values remain the state and revision contracts.  This module
accepts only closed typed evidence, applies one deterministic policy, and
retains exact evidence/provenance witnesses for idempotency.  It deliberately
does not call models, integrate with AgentRuntime, persist state, or own Goal
admission.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
import json
import math
from typing import TYPE_CHECKING, Final

from suzka.identity.origin import OriginActor, OriginInputKind
from suzka.identifiers import validate_identifier
from suzka.motivation.common import (
    R13_MAX_CANDIDATES_PER_DOMAIN,
    R13_MAX_EVIDENCE_REFS,
    R13_MAX_EVENT_SEQUENCE,
    R13_MAX_RECORDS_PER_DOMAIN,
    R13_MAX_REVISION,
    R13_MAX_REVISION_HISTORY,
    R13Reference,
    R13ReferenceKind,
    RevisionCompactionAnchor,
    bounded_fraction,
    canonical_datetime,
    canonical_event,
    canonical_json,
    canonical_references,
    digest_payload,
    utc_datetime,
    validate_digest,
)
from suzka.motivation.commitment import CommitmentLifecycle, CommitmentRecord
from suzka.motivation.goal import (
    GOAL_EVIDENCE_KINDS,
    GoalLifecycle,
    GoalRecord,
    GoalRevisionOperation,
    GoalRevisionReason,
    GoalRevisionRecord,
    goal_id_for_target,
    goal_proposal_digest,
)
from suzka.motivation.motivation import (
    MOTIVATION_SOURCE_KINDS,
    MOTIVATION_TARGET_KINDS,
    MotivationInterpretationCandidate,
    MotivationKind,
    MotivationLifecycle,
    MotivationRecord,
    MotivationRevisionOperation,
    MotivationRevisionReason,
    MotivationRevisionRecord,
    motivation_id_for_target,
    motivation_record_digest,
    motivation_state_digest,
)

if TYPE_CHECKING:
    from suzka.belief.records import BeliefRecord
    from suzka.identity.value_system import ValueState


MOTIVATION_SYSTEM_SCHEMA_VERSION: Final = 1
MOTIVATION_SYSTEM_DOMAIN: Final = b"PROJECT-SUZKA:R13:MOTIVATION-SYSTEM:V1\0"
MOTIVATION_EVIDENCE_DOMAIN: Final = b"PROJECT-SUZKA:R13:MOTIVATION-EVIDENCE:V1\0"
MOTIVATION_EVENT_DOMAIN: Final = b"PROJECT-SUZKA:R13:MOTIVATION-EVENT:V1\0"
MOTIVATION_MAX_RECORDS: Final = R13_MAX_RECORDS_PER_DOMAIN
MOTIVATION_MAX_EVIDENCE_PER_RECORD: Final = R13_MAX_EVIDENCE_REFS
MOTIVATION_MAX_EVIDENCE_LEDGER: Final = (
    MOTIVATION_MAX_RECORDS * MOTIVATION_MAX_EVIDENCE_PER_RECORD
)
MOTIVATION_MAX_ORIGIN_REFS: Final = R13_MAX_EVIDENCE_REFS
MOTIVATION_MAX_CANDIDATES: Final = R13_MAX_CANDIDATES_PER_DOMAIN
MOTIVATION_MAX_EVENT_RECEIPTS: Final = 1_024
MOTIVATION_MAX_REVISION: Final = R13_MAX_REVISION
MOTIVATION_MAX_EVIDENCE_PER_EVENT: Final = 1
MOTIVATION_SYSTEM_MAX_SERIALIZED_BYTES: Final = 10 * 1024 * 1024
MOTIVATION_MAX_ELAPSED_SECONDS: Final = 10 * 365 * 24 * 60 * 60
MOTIVATION_DECAY_SCALE_SECONDS: Final = 30 * 24 * 60 * 60
MOTIVATION_PERSISTENCE_DECAY_SECONDS: Final = 365 * 24 * 60 * 60
MOTIVATION_SATIATION_STEP: Final = 0.25
MOTIVATION_REVIEW_STEP: Final = 0.25
MOTIVATION_GOAL_PROPOSAL_MIN_STRENGTH: Final = 0.2
MOTIVATION_MAX_GOALS_PER_EVENT: Final = 1
MOTIVATION_MAX_GOALS_PER_MOTIVATION: Final = 1
MOTIVATION_MAX_GOAL_PROPOSAL_WITNESSES: Final = (
    MOTIVATION_MAX_RECORDS * MOTIVATION_MAX_GOALS_PER_MOTIVATION
)

_SOURCE_WEIGHTS: Final[dict[R13ReferenceKind, float]] = {
    R13ReferenceKind.EXPERIENCE: 0.30,
    R13ReferenceKind.EMOTION: 0.25,
    R13ReferenceKind.VALUE: 0.30,
    R13ReferenceKind.BELIEF: 0.20,
    R13ReferenceKind.MOTIVATION: 0.15,
    R13ReferenceKind.GOAL: 0.15,
    R13ReferenceKind.COMMITMENT: 0.20,
}
_ORIGIN_KINDS: Final = GOAL_EVIDENCE_KINDS | {R13ReferenceKind.EVENT}


class MotivationDomainError(ValueError):
    """Base class for fail-closed Motivation authority errors."""


class MotivationCapacityExceeded(MotivationDomainError):
    """A bounded Motivation record, ledger, history, or snapshot is full."""


class MotivationEvidenceConflict(MotivationDomainError):
    """An evidence identity was reused with conflicting content/provenance."""


class MotivationEventOperation(str, Enum):
    EVIDENCE = "evidence"
    CANDIDATE = "candidate"
    DECAY = "decay"
    SATIATION = "satiation"
    REVIEW = "review"
    RETIRE = "retire"


@dataclass(frozen=True, slots=True)
class MotivationExperienceEvidence:
    """Minimal typed R12 projection consumed without importing R10/model stacks."""

    experience_ref: R13Reference
    source_event_id: str
    source_event_sequence: int
    created_at: datetime
    active: bool
    novelty: float | None
    novelty_valid: bool
    goal_progress: float | None
    threat: float | None
    emotion_valence: float
    subjective_salience: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.experience_ref, R13Reference)
            or self.experience_ref.kind is not R13ReferenceKind.EXPERIENCE
        ):
            raise ValueError("experience_ref must be an Experience reference")
        object.__setattr__(self, "source_event_id", validate_identifier(self.source_event_id))
        if (
            type(self.source_event_sequence) is not int
            or not 1 <= self.source_event_sequence <= R13_MAX_EVENT_SEQUENCE
        ):
            raise ValueError("source_event_sequence must be a positive bounded integer")
        object.__setattr__(self, "created_at", utc_datetime(self.created_at, "created_at"))
        if type(self.active) is not bool or type(self.novelty_valid) is not bool:
            raise TypeError("active and novelty_valid must be exact booleans")
        if (self.novelty is None) == self.novelty_valid:
            raise ValueError("novelty and novelty_valid disagree")
        if self.novelty is not None:
            object.__setattr__(self, "novelty", bounded_fraction(self.novelty, "novelty"))
        for name in ("threat", "subjective_salience"):
            value = getattr(self, name)
            object.__setattr__(self, name, None if value is None else bounded_fraction(value, name))
        if self.goal_progress is not None:
            value = self.goal_progress
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("goal_progress must be a finite number")
            numeric = float(value)
            if not math.isfinite(numeric) or not -1.0 <= numeric <= 1.0:
                raise ValueError("goal_progress must be finite and in [-1, 1]")
            object.__setattr__(self, "goal_progress", numeric)
        if self.threat is not None:
            object.__setattr__(self, "threat", bounded_fraction(self.threat, "threat"))
        if isinstance(self.emotion_valence, bool) or not isinstance(self.emotion_valence, (int, float)):
            raise TypeError("emotion_valence must be a finite number")
        valence = float(self.emotion_valence)
        if not math.isfinite(valence) or not -1.0 <= valence <= 1.0:
            raise ValueError("emotion_valence must be finite and in [-1, 1]")
        object.__setattr__(self, "emotion_valence", valence)


def _exact_enum(value: object, enum_type: type[Enum], name: str) -> None:
    if type(value) is not enum_type:
        raise TypeError(f"{name} must be a {enum_type.__name__}")


def _goal_description(motivation_id: str, kind: MotivationKind) -> str:
    return f"consider {kind.value} motivation {motivation_id}"


def _source_reference(value: object) -> R13Reference:
    if not isinstance(value, R13Reference):
        raise TypeError("source_ref must be an R13Reference")
    if value.kind not in MOTIVATION_SOURCE_KINDS:
        raise ValueError("source_ref is outside the closed R13 evidence set")
    return value


def _source_time(
    event_id: str | None,
    event_sequence: int | None,
    observed_at: datetime | None,
) -> tuple[str | None, int | None, datetime | None]:
    checked_id, checked_sequence = canonical_event(
        event_id,
        event_sequence,
        require=False,
    )
    if observed_at is not None:
        observed_at = utc_datetime(observed_at, "observed_at")
    return checked_id, checked_sequence, observed_at


def _canonical_identity_refs(
    refs: object,
    name: str,
    *,
    maximum: int,
    allowed_kinds: frozenset[R13ReferenceKind],
) -> tuple[R13Reference, ...]:
    return canonical_references(
        refs,
        name,
        maximum=maximum,
        allow_empty=False,
        allowed_kinds=allowed_kinds,
    )


@dataclass(frozen=True, slots=True)
class MotivationEvidence:
    """Reference-first authoritative evidence projection for one Motive.

    Source adapters should prefer the ``from_*`` factories: they validate
    active/adopted upstream records and copy only exact references and bounded
    numeric measurements.  No source text, transcript, rationale, or model
    output is part of this value.
    """

    source_ref: R13Reference
    target: R13Reference
    kind: MotivationKind
    salience: float
    confidence: float
    persistence: float
    uncertainty: float
    origin_refs: tuple[R13Reference, ...]
    source_event_id: str | None = None
    source_event_sequence: int | None = None
    observed_at: datetime | None = None
    authority_witness_digest: str | None = None

    def __post_init__(self) -> None:
        source = _source_reference(self.source_ref)
        if not isinstance(self.target, R13Reference):
            raise TypeError("target must be an R13Reference")
        if self.target.kind not in MOTIVATION_TARGET_KINDS:
            raise ValueError("target is outside the U1 Motivation target set")
        _exact_enum(self.kind, MotivationKind, "kind")
        if not isinstance(self.origin_refs, tuple):
            raise TypeError("origin_refs must be a tuple")
        origins = _canonical_identity_refs(
            self.origin_refs,
            "origin_refs",
            maximum=MOTIVATION_MAX_ORIGIN_REFS,
            allowed_kinds=_ORIGIN_KINDS,
        )
        if source in origins:
            raise ValueError("source_ref cannot be repeated as an origin reference")
        event_id, event_sequence, observed_at = _source_time(
            self.source_event_id,
            self.source_event_sequence,
            self.observed_at,
        )
        authority_witness = self.authority_witness_digest
        if authority_witness is not None:
            authority_witness = validate_digest(
                authority_witness,
                "authority_witness_digest",
            )
        for name in ("salience", "confidence", "persistence", "uncertainty"):
            object.__setattr__(self, name, bounded_fraction(getattr(self, name), name))
        object.__setattr__(self, "source_ref", source)
        object.__setattr__(self, "origin_refs", origins)
        object.__setattr__(self, "source_event_id", event_id)
        object.__setattr__(self, "source_event_sequence", event_sequence)
        object.__setattr__(self, "observed_at", observed_at)
        object.__setattr__(self, "authority_witness_digest", authority_witness)

    @property
    def motivation_id(self) -> str:
        return motivation_id_for_target(self.kind, self.target)

    @property
    def evidence_digest(self) -> str:
        return digest_payload(MOTIVATION_EVIDENCE_DOMAIN, self.canonical_value())

    def canonical_value(self) -> dict[str, object]:
        return {
            "confidence": self.confidence.hex(),
            "authority_witness_digest": self.authority_witness_digest,
            "kind": self.kind.value,
            "observed_at": canonical_datetime(self.observed_at),
            "origin_refs": [item.canonical_value() for item in self.origin_refs],
            "persistence": self.persistence.hex(),
            "salience": self.salience.hex(),
            "source_event_id": self.source_event_id,
            "source_event_sequence": self.source_event_sequence,
            "source_ref": self.source_ref.canonical_value(),
            "target": self.target.canonical_value(),
            "uncertainty": self.uncertainty.hex(),
        }

    @classmethod
    def from_value(cls, value: ValueState) -> MotivationEvidence:
        """Project an active Value without changing its recorded origin."""

        from suzka.identity.value_system import ValueState

        if not isinstance(value, ValueState):
            raise TypeError("value must be ValueState")
        if not value.is_active():
            raise MotivationDomainError("only active Values may support Motivation")
        origin = value.origin
        actor_reference_kind = {
            OriginActor.SELF: R13ReferenceKind.VALUE,
            OriginActor.USER: R13ReferenceKind.USER_REQUEST,
            OriginActor.OPERATOR: R13ReferenceKind.OPERATOR_PROPOSAL,
            OriginActor.EXTERNAL_SOURCE: R13ReferenceKind.EXTERNAL_REQUEST,
            OriginActor.SYSTEM: R13ReferenceKind.SYSTEM,
        }.get(origin.actor)
        if actor_reference_kind is None:
            raise MotivationDomainError("Value origin is not authoritative for Motivation")
        origins: list[R13Reference] = [
            R13Reference(actor_reference_kind, origin.origin_id)
        ]
        # A self-endorsed Value may still name an originating request.  Keep
        # that reference instead of relabelling repeated requests as intrinsic.
        if (
            origin.actor is OriginActor.SELF
            and origin.input_kind is OriginInputKind.REQUEST
            and origin.source_ref is not None
        ):
            origins.append(
                R13Reference(R13ReferenceKind.EXTERNAL_REQUEST, origin.source_ref)
            )
        event_id, event_sequence = origin.event_id, origin.event_sequence
        if (event_id is None) != (event_sequence is None):
            raise MotivationDomainError("Value origin has an incomplete event witness")
        return cls(
            source_ref=R13Reference(R13ReferenceKind.VALUE, value.value_id),
            target=R13Reference(R13ReferenceKind.VALUE, value.value_id),
            kind=MotivationKind.DESIRE
            if value.polarity > 0
            else MotivationKind.AVERSION,
            salience=value.strength,
            confidence=value.confidence,
            persistence=value.stability,
            uncertainty=1.0 - value.confidence,
            origin_refs=tuple(
                sorted(origins, key=lambda item: (item.reference, item.kind.value))
            ),
            source_event_id=event_id,
            source_event_sequence=event_sequence,
        )

    @classmethod
    def from_experience(
        cls, experience: MotivationExperienceEvidence
    ) -> MotivationEvidence | None:
        """Project an active R12 appraisal without importing cognition/models."""

        if not isinstance(experience, MotivationExperienceEvidence):
            raise TypeError("experience must be MotivationExperienceEvidence")
        if not experience.active:
            raise MotivationDomainError("only active Experiences may support Motivation")
        progress = experience.goal_progress or 0.0
        threat = experience.threat or 0.0
        if threat > 0.0 and threat >= max(0.0, progress):
            kind = MotivationKind.AVERSION
        elif progress > 0.0:
            kind = MotivationKind.DESIRE
        elif experience.novelty_valid and (experience.novelty or 0.0) > 0.0:
            kind = MotivationKind.INTEREST
        elif experience.emotion_valence < 0.0:
            kind = MotivationKind.AVERSION
        else:
            return None
        event_ref = R13Reference(R13ReferenceKind.EVENT, experience.source_event_id)
        return cls(
            source_ref=experience.experience_ref,
            target=experience.experience_ref,
            kind=kind,
            salience=experience.subjective_salience,
            confidence=1.0 if experience.novelty_valid else 0.5,
            persistence=0.5,
            uncertainty=0.0 if experience.novelty_valid else 0.5,
            origin_refs=(event_ref,),
            source_event_id=experience.source_event_id,
            source_event_sequence=experience.source_event_sequence,
            observed_at=experience.created_at,
        )

    @classmethod
    def from_belief(cls, belief: BeliefRecord) -> MotivationEvidence:
        """Project only an explicitly adopted R12 Belief."""

        from suzka.belief.records import (
            BeliefEvidenceType,
            BeliefLifecycle,
            BeliefRecord,
        )

        if not isinstance(belief, BeliefRecord):
            raise TypeError("belief must be BeliefRecord")
        if (
            belief.lifecycle is not BeliefLifecycle.ADOPTED
            or belief.subject_admission is None
        ):
            raise MotivationDomainError("only subject-adopted Beliefs may support Motivation")
        if belief.revision_history:
            latest = belief.revision_history[-1]
            source_event_id = latest.event_id
            source_event_sequence = latest.event_sequence
            observed_at = latest.created_at
        else:
            # An immutable U1 adopted record may carry its explicit admission
            # before an authority has appended its first local revision.
            source_event_id = belief.subject_admission.event_id
            source_event_sequence = belief.subject_admission.event_sequence
            observed_at = None
        if source_event_id is None or source_event_sequence is None:
            raise MotivationDomainError("adopted Belief lacks its latest event witness")
        evidence_origin_kind = {
            BeliefEvidenceType.EXPERIENCE: R13ReferenceKind.EXPERIENCE,
            BeliefEvidenceType.SEMANTIC_MEMORY: R13ReferenceKind.BELIEF,
            BeliefEvidenceType.EPISODIC_MEMORY: R13ReferenceKind.EXPERIENCE,
            BeliefEvidenceType.EXTERNAL_CLAIM: R13ReferenceKind.EXTERNAL_REQUEST,
            BeliefEvidenceType.MODEL_INFERENCE: R13ReferenceKind.BELIEF,
            BeliefEvidenceType.OPERATOR_CORRECTION: R13ReferenceKind.OPERATOR_PROPOSAL,
        }
        origins = [
            R13Reference(evidence_origin_kind[item.evidence_type], item.evidence_ref)
            for item in belief.evidence
        ]
        origins = sorted(origins, key=lambda item: (item.reference, item.kind.value))
        return cls(
            source_ref=R13Reference(R13ReferenceKind.BELIEF, belief.belief_id),
            target=R13Reference(R13ReferenceKind.BELIEF, belief.belief_id),
            kind=MotivationKind.INTEREST,
            salience=belief.confidence,
            confidence=belief.confidence,
            persistence=belief.confidence,
            uncertainty=1.0 - belief.confidence,
            origin_refs=tuple(origins),
            source_event_id=source_event_id,
            source_event_sequence=source_event_sequence,
            observed_at=observed_at,
            authority_witness_digest=belief.subject_admission.admission_digest,
        )

    @classmethod
    def from_emotion(
        cls,
        emotion_ref: R13Reference,
        *,
        valence: float,
        arousal: float,
        origin_refs: tuple[R13Reference, ...],
        source_event_id: str,
        source_event_sequence: int,
        observed_at: datetime | None = None,
    ) -> MotivationEvidence:
        """Project the exact current R10 Emotion state under its event ref."""

        if not isinstance(emotion_ref, R13Reference) or emotion_ref.kind is not R13ReferenceKind.EMOTION:
            raise ValueError("emotion_ref must be an Emotion reference")
        if isinstance(valence, bool) or not isinstance(valence, (int, float)):
            raise TypeError("valence must be a finite number")
        if isinstance(arousal, bool) or not isinstance(arousal, (int, float)):
            raise TypeError("arousal must be a finite number")
        checked_valence = float(valence)
        checked_arousal = bounded_fraction(arousal, "arousal")
        if not math.isfinite(checked_valence) or not -1.0 <= checked_valence <= 1.0:
            raise ValueError("valence must be finite and in [-1, 1]")
        if checked_valence < 0.0:
            kind = MotivationKind.AVERSION
            target = "state:emotion:negative"
        elif checked_valence > 0.0:
            kind = MotivationKind.DESIRE
            target = "state:emotion:positive"
        elif checked_arousal >= 0.5:
            kind = MotivationKind.DRIVE
            target = "state:emotion:activated"
        else:
            kind = MotivationKind.INTEREST
            target = "state:emotion:resting"
        return cls(
            source_ref=emotion_ref,
            target=R13Reference(R13ReferenceKind.STATE, target),
            kind=kind,
            salience=max(abs(checked_valence), checked_arousal),
            confidence=1.0,
            persistence=0.25 + 0.5 * checked_arousal,
            uncertainty=0.0,
            origin_refs=origin_refs,
            source_event_id=source_event_id,
            source_event_sequence=source_event_sequence,
            observed_at=observed_at,
        )

    @classmethod
    def from_goal(cls, goal: GoalRecord) -> MotivationEvidence:
        """Project an adopted Goal without acquiring Goal lifecycle authority."""

        if not isinstance(goal, GoalRecord):
            raise TypeError("goal must be GoalRecord")
        if goal.lifecycle is not GoalLifecycle.ADOPTED:
            raise MotivationDomainError("only adopted Goals may support Motivation")
        if not goal.revision_history:
            raise MotivationDomainError("adopted Goal lacks its latest event witness")
        latest = goal.revision_history[-1]
        if latest.event_id is None or latest.event_sequence is None:
            raise MotivationDomainError("adopted Goal lacks its latest event witness")
        return cls(
            source_ref=R13Reference(R13ReferenceKind.GOAL, goal.goal_id),
            target=R13Reference(R13ReferenceKind.GOAL, goal.goal_id),
            kind=MotivationKind.DRIVE,
            salience=0.5,
            confidence=1.0,
            persistence=0.75,
            uncertainty=0.0,
            origin_refs=goal.origin_refs,
            source_event_id=latest.event_id,
            source_event_sequence=latest.event_sequence,
            observed_at=latest.created_at,
            authority_witness_digest=(
                None
                if goal.subject_admission is None
                else goal.subject_admission.admission_digest
            ),
        )

    @classmethod
    def from_commitment(cls, commitment: CommitmentRecord) -> MotivationEvidence:
        """Project an active Commitment without changing its responsibility."""

        if not isinstance(commitment, CommitmentRecord):
            raise TypeError("commitment must be CommitmentRecord")
        if commitment.lifecycle is not CommitmentLifecycle.ACTIVE:
            raise MotivationDomainError("only active Commitments may support Motivation")
        if not commitment.revision_history:
            raise MotivationDomainError("active Commitment lacks its latest event witness")
        latest = commitment.revision_history[-1]
        if latest.event_id is None or latest.event_sequence is None:
            raise MotivationDomainError("active Commitment lacks its latest event witness")
        return cls(
            source_ref=R13Reference(
                R13ReferenceKind.COMMITMENT,
                commitment.commitment_id,
            ),
            target=R13Reference(
                R13ReferenceKind.COMMITMENT,
                commitment.commitment_id,
            ),
            kind=MotivationKind.DRIVE,
            salience=0.5,
            confidence=1.0,
            persistence=0.75,
            uncertainty=0.0,
            origin_refs=commitment.origin_refs,
            source_event_id=latest.event_id,
            source_event_sequence=latest.event_sequence,
            observed_at=latest.created_at,
            authority_witness_digest=(
                None
                if commitment.subject_admission is None
                else commitment.subject_admission.admission_digest
            ),
        )


@dataclass(frozen=True, slots=True)
class MotivationMutationEvidence:
    """Caller-supplied canonical event contract; never reads a system clock."""

    event_id: str
    event_sequence: int
    recorded_at: datetime
    evidence_refs: tuple[R13Reference, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", validate_identifier(self.event_id))
        if (
            type(self.event_sequence) is not int
            or not 1 <= self.event_sequence <= R13_MAX_EVENT_SEQUENCE
        ):
            raise ValueError("event_sequence must be a positive bounded integer")
        object.__setattr__(
            self,
            "recorded_at",
            utc_datetime(self.recorded_at, "recorded_at"),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _canonical_identity_refs(
                self.evidence_refs,
                "evidence_refs",
                maximum=R13_MAX_EVIDENCE_REFS,
                allowed_kinds=MOTIVATION_SOURCE_KINDS,
            ),
        )

    def canonical_value(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "event_sequence": self.event_sequence,
            "evidence_refs": [item.canonical_value() for item in self.evidence_refs],
            "recorded_at": canonical_datetime(self.recorded_at),
        }


@dataclass(frozen=True, slots=True)
class MotivationEvidenceLedgerEntry:
    """Exact accepted evidence identity and its first mutation event."""

    evidence: MotivationEvidence
    motivation_id: str
    event_id: str
    event_sequence: int

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, MotivationEvidence):
            raise TypeError("evidence must be MotivationEvidence")
        if validate_identifier(self.motivation_id) != self.evidence.motivation_id:
            raise ValueError("evidence ledger Motivation identity mismatch")
        object.__setattr__(self, "event_id", validate_identifier(self.event_id))
        if type(self.event_sequence) is not int or not 1 <= self.event_sequence <= R13_MAX_EVENT_SEQUENCE:
            raise ValueError("event_sequence must be a positive bounded integer")
        if (
            self.evidence.source_event_sequence is not None
            and self.evidence.source_event_sequence > self.event_sequence
        ):
            raise ValueError("evidence source event is in the future")

    def canonical_value(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "event_sequence": self.event_sequence,
            "evidence": self.evidence.canonical_value(),
            "motivation_id": self.motivation_id,
        }


@dataclass(frozen=True, slots=True)
class MotivationEventReceipt:
    """Bounded exact replay witness for one committed Motivation mutation."""

    event_id: str
    event_sequence: int
    recorded_at: datetime
    operation: MotivationEventOperation
    input_digest: str
    motivation_ids: tuple[str, ...]
    evidence_refs: tuple[R13Reference, ...]
    evidence_digest: str | None = None
    candidate_digest: str | None = None
    elapsed_seconds: float | None = None
    goal_proposal_digest: str | None = None
    receipt_digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", validate_identifier(self.event_id))
        if type(self.event_sequence) is not int or not 1 <= self.event_sequence <= R13_MAX_EVENT_SEQUENCE:
            raise ValueError("event_sequence must be a positive bounded integer")
        object.__setattr__(self, "recorded_at", utc_datetime(self.recorded_at, "recorded_at"))
        _exact_enum(self.operation, MotivationEventOperation, "operation")
        if type(self.input_digest) is not str or len(self.input_digest) != 64 or any(
            character not in "0123456789abcdef" for character in self.input_digest
        ):
            raise ValueError("input_digest must be a lowercase SHA-256 digest")
        if type(self.motivation_ids) is not tuple:
            raise TypeError("motivation_ids must be a tuple")
        ids = tuple(validate_identifier(item) for item in self.motivation_ids)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("motivation_ids must be sorted and unique")
        refs = canonical_references(
            self.evidence_refs,
            "evidence_refs",
            maximum=R13_MAX_EVIDENCE_REFS,
            allow_empty=False,
            allowed_kinds=MOTIVATION_SOURCE_KINDS,
        )
        for name in ("evidence_digest", "candidate_digest"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, validate_digest(value, name))
        if self.goal_proposal_digest is not None:
            object.__setattr__(
                self,
                "goal_proposal_digest",
                validate_digest(self.goal_proposal_digest, "goal_proposal_digest"),
            )
        if self.operation is MotivationEventOperation.EVIDENCE:
            if self.evidence_digest is None or self.elapsed_seconds is not None:
                raise ValueError("evidence receipt requires its exact evidence digest")
        elif self.operation is MotivationEventOperation.CANDIDATE:
            if (
                self.evidence_digest is not None
                or self.candidate_digest is None
                or self.elapsed_seconds is not None
                or self.goal_proposal_digest is not None
            ):
                raise ValueError("candidate receipt requires its exact candidate digest")
        elif self.operation is MotivationEventOperation.DECAY:
            if (
                self.evidence_digest is not None
                or self.candidate_digest is not None
                or self.elapsed_seconds is None
                or self.goal_proposal_digest is not None
            ):
                raise ValueError("decay receipt requires its exact elapsed-time input")
            object.__setattr__(self, "elapsed_seconds", _bounded_elapsed(self.elapsed_seconds))
        elif (
            self.evidence_digest is not None
            or self.candidate_digest is not None
            or self.elapsed_seconds is not None
            or self.goal_proposal_digest is not None
        ):
            raise ValueError("unexpected operation-specific event receipt input")
        object.__setattr__(self, "motivation_ids", ids)
        object.__setattr__(self, "evidence_refs", refs)
        object.__setattr__(self, "receipt_digest", digest_payload(MOTIVATION_EVENT_DOMAIN, self.canonical_value()))

    def canonical_value(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "event_sequence": self.event_sequence,
            "evidence_digest": self.evidence_digest,
            "evidence_refs": [item.canonical_value() for item in self.evidence_refs],
            "elapsed_seconds": (
                None
                if self.elapsed_seconds is None
                else self.elapsed_seconds.hex()
            ),
            "input_digest": self.input_digest,
            "candidate_digest": self.candidate_digest,
            "goal_proposal_digest": self.goal_proposal_digest,
            "motivation_ids": list(self.motivation_ids),
            "operation": self.operation.value,
            "recorded_at": canonical_datetime(self.recorded_at),
        }


@dataclass(frozen=True, slots=True)
class MotivationGoalProposalWitness:
    """First exact event at which one Motive became Goal-proposal eligible."""

    motivation_id: str
    kind: MotivationKind
    event_id: str
    event_sequence: int
    recorded_at: datetime
    evidence_ref: R13Reference
    eligible_strength: float
    motivation_revision: int
    motivation_revision_digest: str
    motivation_state_digest: str
    goal_id: str = field(init=False)
    proposal_digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "motivation_id", validate_identifier(self.motivation_id))
        _exact_enum(self.kind, MotivationKind, "kind")
        object.__setattr__(self, "event_id", validate_identifier(self.event_id))
        if type(self.event_sequence) is not int or not 1 <= self.event_sequence <= R13_MAX_EVENT_SEQUENCE:
            raise ValueError("event_sequence must be a positive bounded integer")
        object.__setattr__(self, "recorded_at", utc_datetime(self.recorded_at, "recorded_at"))
        evidence_ref = _source_reference(self.evidence_ref)
        strength = bounded_fraction(self.eligible_strength, "eligible_strength")
        if strength < MOTIVATION_GOAL_PROPOSAL_MIN_STRENGTH:
            raise ValueError("eligible_strength does not meet the Goal proposal threshold")
        if (
            type(self.motivation_revision) is not int
            or not 0 <= self.motivation_revision <= R13_MAX_REVISION
        ):
            raise ValueError("motivation_revision must be a bounded exact integer")
        revision_digest = validate_digest(
            self.motivation_revision_digest,
            "motivation_revision_digest",
        )
        state_digest = validate_digest(
            self.motivation_state_digest,
            "motivation_state_digest",
        )
        object.__setattr__(self, "evidence_ref", evidence_ref)
        object.__setattr__(self, "eligible_strength", strength)
        object.__setattr__(self, "motivation_revision_digest", revision_digest)
        object.__setattr__(self, "motivation_state_digest", state_digest)
        motivation_ref = R13Reference(R13ReferenceKind.MOTIVATION, self.motivation_id)
        goal_id = goal_id_for_target(motivation_ref)
        proposal_digest = goal_proposal_digest(
            motivation_ref,
            _goal_description(self.motivation_id, self.kind),
            origin_refs=(motivation_ref,),
            evidence_refs=tuple(
                sorted(
                    (motivation_ref, evidence_ref),
                    key=lambda item: (item.reference, item.kind.value),
                )
            ),
        )
        object.__setattr__(self, "goal_id", goal_id)
        object.__setattr__(self, "proposal_digest", proposal_digest)

    def canonical_value(self) -> dict[str, object]:
        return {
            "eligible_strength": self.eligible_strength.hex(),
            "event_id": self.event_id,
            "event_sequence": self.event_sequence,
            "evidence_ref": self.evidence_ref.canonical_value(),
            "goal_id": self.goal_id,
            "kind": self.kind.value,
            "motivation_id": self.motivation_id,
            "motivation_revision": self.motivation_revision,
            "motivation_revision_digest": self.motivation_revision_digest,
            "motivation_state_digest": self.motivation_state_digest,
            "proposal_digest": self.proposal_digest,
            "recorded_at": canonical_datetime(self.recorded_at),
        }

    def to_goal_record(self) -> GoalRecord:
        motivation_ref = R13Reference(R13ReferenceKind.MOTIVATION, self.motivation_id)
        evidence_refs = tuple(
            sorted(
                (motivation_ref, self.evidence_ref),
                key=lambda item: (item.reference, item.kind.value),
            )
        )
        revision = GoalRevisionRecord(
            goal_id=self.goal_id,
            revision=0,
            operation=GoalRevisionOperation.CREATE,
            reason=GoalRevisionReason.CREATION,
            created_at=self.recorded_at,
            previous_lifecycle_state=None,
            proposal_digest=self.proposal_digest,
            event_id=self.event_id,
            event_sequence=self.event_sequence,
            evidence_refs=tuple(sorted(item.reference for item in evidence_refs)),
        )
        return GoalRecord(
            goal_id=self.goal_id,
            target=motivation_ref,
            description=_goal_description(self.motivation_id, self.kind),
            lifecycle=GoalLifecycle.PROPOSED,
            origin_refs=(motivation_ref,),
            evidence_refs=evidence_refs,
            revision_history=(revision,),
        )


def _record_value(record: MotivationRecord) -> dict[str, object]:
    encoded = motivation_record_digest(record)
    if encoded != record.record_digest:
        raise MotivationDomainError("Motivation record digest is inconsistent")
    from suzka.motivation.motivation import MOTIVATION_DOMAIN, canonical_motivation_payload

    payload = canonical_motivation_payload(record)
    return json.loads(payload[len(MOTIVATION_DOMAIN) :])


def _candidate_value(candidate: MotivationInterpretationCandidate) -> dict[str, object]:
    from suzka.motivation.motivation import (
        MOTIVATION_CANDIDATE_DOMAIN,
        canonical_motivation_candidate_payload,
    )

    payload = canonical_motivation_candidate_payload(candidate)
    return json.loads(payload[len(MOTIVATION_CANDIDATE_DOMAIN) :])


@dataclass(frozen=True, slots=True)
class MotivationSystemSnapshot:
    """Exact immutable export projection; it is not a persistence format."""

    records: tuple[MotivationRecord, ...]
    evidence_ledger: tuple[MotivationEvidenceLedgerEntry, ...]
    candidates: tuple[MotivationInterpretationCandidate, ...]
    event_receipts: tuple[MotivationEventReceipt, ...]
    goal_proposal_witnesses: tuple[MotivationGoalProposalWitness, ...] = ()
    schema_version: int = MOTIVATION_SYSTEM_SCHEMA_VERSION
    authority_digest: str = field(init=False)
    serialized_bytes: int = field(init=False)

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != MOTIVATION_SYSTEM_SCHEMA_VERSION:
            raise ValueError("unsupported Motivation system snapshot version")
        for name, values, maximum, item_type in (
            ("records", self.records, MOTIVATION_MAX_RECORDS, MotivationRecord),
            ("evidence_ledger", self.evidence_ledger, MOTIVATION_MAX_EVIDENCE_LEDGER, MotivationEvidenceLedgerEntry),
            ("candidates", self.candidates, MOTIVATION_MAX_CANDIDATES, MotivationInterpretationCandidate),
            ("event_receipts", self.event_receipts, MOTIVATION_MAX_EVENT_RECEIPTS, MotivationEventReceipt),
            (
                "goal_proposal_witnesses",
                self.goal_proposal_witnesses,
                MOTIVATION_MAX_GOAL_PROPOSAL_WITNESSES,
                MotivationGoalProposalWitness,
            ),
        ):
            if type(values) is not tuple:
                raise TypeError(f"{name} must be a tuple")
            if len(values) > maximum:
                raise MotivationCapacityExceeded(f"{name} exceeds its bound")
            if any(not isinstance(item, item_type) for item in values):
                raise TypeError(f"{name} contains an invalid value")
        record_ids = tuple(item.motivation_id for item in self.records)
        if record_ids != tuple(sorted(set(record_ids))):
            raise ValueError("Motivation records must be sorted and unique")
        evidence_ids = tuple(
            (item.evidence.source_ref.kind.value, item.evidence.source_ref.reference)
            for item in self.evidence_ledger
        )
        if evidence_ids != tuple(sorted(set(evidence_ids))):
            raise ValueError("Motivation evidence ledger must be sorted and unique")
        candidate_ids = tuple(item.candidate_digest for item in self.candidates)
        if candidate_ids != tuple(sorted(set(candidate_ids))):
            raise ValueError("Motivation candidates must be sorted and unique")
        proposal_ids = tuple(item.motivation_id for item in self.goal_proposal_witnesses)
        if proposal_ids != tuple(sorted(set(proposal_ids))):
            raise ValueError("Goal proposal witnesses must be sorted and unique")
        if len({item.event_id for item in self.goal_proposal_witnesses}) != len(
            self.goal_proposal_witnesses
        ):
            raise ValueError("one event cannot exceed the Goal proposal budget")
        event_sequences = tuple(item.event_sequence for item in self.event_receipts)
        if event_sequences != tuple(sorted(set(event_sequences))):
            raise ValueError("Motivation event receipts must be sequence ordered")
        if len({item.event_id for item in self.event_receipts}) != len(self.event_receipts):
            raise ValueError("Motivation event IDs cannot be reused")
        if any(
            later.recorded_at < earlier.recorded_at
            for earlier, later in zip(self.event_receipts, self.event_receipts[1:])
        ):
            raise ValueError("Motivation event timestamps must not regress")
        record_map = {item.motivation_id: item for item in self.records}
        receipt_map = {item.event_id: item for item in self.event_receipts}
        for receipt in self.event_receipts:
            if any(item not in record_map for item in receipt.motivation_ids):
                raise ValueError("Motivation event receipt references a missing record")
            expected_motivation_refs = tuple(
                R13Reference(R13ReferenceKind.MOTIVATION, item)
                for item in receipt.motivation_ids
            )
            if receipt.operation is MotivationEventOperation.EVIDENCE:
                if len(receipt.motivation_ids) != 1 or len(receipt.evidence_refs) != 1:
                    raise ValueError("evidence receipt must bind one Motive and one source")
            elif receipt.operation is MotivationEventOperation.CANDIDATE:
                if receipt.motivation_ids or receipt.candidate_digest is None:
                    raise ValueError("candidate receipt cannot carry Motivation authority")
            elif receipt.operation is MotivationEventOperation.DECAY:
                if receipt.evidence_refs != expected_motivation_refs:
                    raise ValueError("decay receipt must bind its exact Motive set")
            elif len(receipt.motivation_ids) != 1 or receipt.evidence_refs != expected_motivation_refs:
                raise ValueError("review/retire receipt must bind one exact Motive")
            expected_digest: str
            if receipt.operation is MotivationEventOperation.EVIDENCE:
                continue
            if receipt.operation is MotivationEventOperation.CANDIDATE:
                expected_digest = _event_input_digest(
                    receipt.operation,
                    {"candidate_digest": receipt.candidate_digest},
                )
                if receipt.input_digest != expected_digest:
                    raise ValueError("candidate event receipt input digest is inconsistent")
                continue
            if receipt.operation is MotivationEventOperation.DECAY:
                if receipt.elapsed_seconds is None:
                    raise ValueError("decay receipt lacks elapsed-time input")
                expected_digest = _event_input_digest(
                    receipt.operation,
                    {
                        "elapsed_seconds": receipt.elapsed_seconds.hex(),
                        "motivation_ids": list(receipt.motivation_ids),
                    },
                )
            elif receipt.operation is MotivationEventOperation.SATIATION:
                expected_digest = _event_input_digest(
                    receipt.operation,
                    {
                        "motivation_id": receipt.motivation_ids[0],
                        "satiation_step": MOTIVATION_SATIATION_STEP.hex(),
                    },
                )
            elif receipt.operation is MotivationEventOperation.REVIEW:
                expected_digest = _event_input_digest(
                    receipt.operation,
                    {
                        "motivation_id": receipt.motivation_ids[0],
                        "satiation_step": (-MOTIVATION_REVIEW_STEP).hex(),
                    },
                )
            else:
                expected_digest = _event_input_digest(
                    receipt.operation,
                    {"motivation_id": receipt.motivation_ids[0]},
                )
            if receipt.input_digest != expected_digest:
                raise ValueError("Motivation event receipt input digest is inconsistent")
        evidence_by_motivation: dict[str, list[R13Reference]] = {}
        for entry in self.evidence_ledger:
            record = record_map.get(entry.motivation_id)
            if record is None:
                raise ValueError("Motivation evidence ledger references a missing record")
            if entry.evidence.source_ref not in record.evidence_refs:
                raise ValueError("Motivation evidence ledger is not represented by its record")
            entry_receipt = receipt_map.get(entry.event_id)
            if (
                entry_receipt is None
                or entry_receipt.event_sequence != entry.event_sequence
                or entry_receipt.operation is not MotivationEventOperation.EVIDENCE
                or entry_receipt.motivation_ids != (entry.motivation_id,)
                or entry_receipt.evidence_refs != (entry.evidence.source_ref,)
            ):
                raise ValueError("Motivation evidence ledger lacks its exact event receipt")
            if entry_receipt.evidence_digest != entry.evidence.evidence_digest:
                raise ValueError("Motivation receipt does not bind exact evidence provenance")
            expected_input_digest = _event_input_digest(
                MotivationEventOperation.EVIDENCE,
                {
                    "candidate_digest": entry_receipt.candidate_digest,
                    "evidence_digest": entry.evidence.evidence_digest,
                },
            )
            if entry_receipt.input_digest != expected_input_digest:
                raise ValueError("Motivation evidence receipt digest is inconsistent")
            if entry_receipt.candidate_digest is not None and not any(
                item.candidate_digest == entry_receipt.candidate_digest
                and item.event_id == entry.event_id
                and item.event_sequence == entry.event_sequence
                and item.source_evidence_refs == (entry.evidence.source_ref,)
                for item in self.candidates
            ):
                raise ValueError("Motivation evidence receipt lacks its exact candidate")
            evidence_by_motivation.setdefault(entry.motivation_id, []).append(
                entry.evidence.source_ref
            )
        for candidate in self.candidates:
            candidate_receipt = receipt_map.get(candidate.event_id)
            if (
                candidate_receipt is None
                or candidate_receipt.event_sequence != candidate.event_sequence
                or candidate_receipt.candidate_digest != candidate.candidate_digest
                or candidate_receipt.evidence_refs != candidate.source_evidence_refs
                or candidate_receipt.operation
                not in {
                    MotivationEventOperation.CANDIDATE,
                    MotivationEventOperation.EVIDENCE,
                }
            ):
                raise ValueError("Motivation candidate lacks its exact event receipt")
        for receipt in self.event_receipts:
            if receipt.operation is MotivationEventOperation.CANDIDATE and not any(
                item.candidate_digest == receipt.candidate_digest
                and item.event_id == receipt.event_id
                and item.event_sequence == receipt.event_sequence
                and item.source_evidence_refs == receipt.evidence_refs
                for item in self.candidates
            ):
                raise ValueError("candidate receipt lacks its exact candidate value")
        proposal_by_id = {
            item.motivation_id: item for item in self.goal_proposal_witnesses
        }
        for witness in self.goal_proposal_witnesses:
            record = record_map.get(witness.motivation_id)
            witness_receipt = receipt_map.get(witness.event_id)
            if record is None or record.kind is not witness.kind:
                raise ValueError("Goal proposal witness references a missing Motive")
            if witness.evidence_ref not in record.evidence_refs:
                raise ValueError("Goal proposal witness source is absent from its Motive")
            if (
                witness_receipt is None
                or witness_receipt.operation is not MotivationEventOperation.EVIDENCE
                or witness_receipt.event_sequence != witness.event_sequence
                or witness_receipt.recorded_at != witness.recorded_at
                or witness_receipt.motivation_ids != (witness.motivation_id,)
                or witness_receipt.evidence_refs != (witness.evidence_ref,)
                or witness_receipt.goal_proposal_digest != witness.proposal_digest
            ):
                raise ValueError("Goal proposal witness lacks its exact eligibility event")
            if not any(
                item.motivation_id == witness.motivation_id
                and item.event_id == witness.event_id
                and item.event_sequence == witness.event_sequence
                and item.evidence.source_ref == witness.evidence_ref
                for item in self.evidence_ledger
            ):
                raise ValueError("Goal proposal witness lacks its accepted source evidence")
            retained_revision = next(
                (
                    item
                    for item in record.revision_history
                    if item.event_id == witness.event_id
                    and item.event_sequence == witness.event_sequence
                ),
                None,
            )
            if retained_revision is not None:
                if (
                    retained_revision.revision != witness.motivation_revision
                    or retained_revision.record_digest != witness.motivation_revision_digest
                    or retained_revision.state_digest != witness.motivation_state_digest
                ):
                    raise ValueError("Goal proposal witness differs from its Motivation revision")
            elif record.history_anchor is not None:
                anchor = record.history_anchor
                if (
                    anchor.through_event_id == witness.event_id
                    and anchor.through_event_sequence == witness.event_sequence
                ):
                    if (
                        anchor.through_revision != witness.motivation_revision
                        or anchor.through_digest != witness.motivation_revision_digest
                        or anchor.through_state_digest != witness.motivation_state_digest
                    ):
                        raise ValueError("Goal proposal witness differs from its compaction anchor")
                elif (
                    witness.motivation_revision >= anchor.retained_from_revision
                    or witness.event_sequence > anchor.through_event_sequence
                ):
                    raise ValueError("Goal proposal witness revision is absent from its Motive history")
            else:
                raise ValueError("Goal proposal witness revision is absent from its Motive history")
        for record in self.records:
            if (
                record.lifecycle is MotivationLifecycle.ACTIVE
                and record.strength >= MOTIVATION_GOAL_PROPOSAL_MIN_STRENGTH
                and record.motivation_id not in proposal_by_id
            ):
                raise ValueError("eligible Motivation is missing its first Goal proposal witness")
        for receipt in self.event_receipts:
            if receipt.goal_proposal_digest is not None and not any(
                item.event_id == receipt.event_id
                and item.proposal_digest == receipt.goal_proposal_digest
                for item in self.goal_proposal_witnesses
            ):
                raise ValueError("event receipt references a missing Goal proposal witness")
        for receipt in self.event_receipts:
            if receipt.operation is MotivationEventOperation.EVIDENCE:
                matches = tuple(
                    item
                    for item in self.evidence_ledger
                    if item.event_id == receipt.event_id
                    and item.event_sequence == receipt.event_sequence
                    and item.motivation_id in receipt.motivation_ids
                    and item.evidence.source_ref in receipt.evidence_refs
                )
                if len(matches) != 1:
                    raise ValueError("evidence receipt must have one exact accepted input")
        for record in self.records:
            record_entries = tuple(
                sorted(
                    (
                        item
                        for item in self.evidence_ledger
                        if item.motivation_id == record.motivation_id
                    ),
                    key=lambda item: item.event_sequence,
                )
            )
            if not record_entries or record.source_evidence != record_entries[0].evidence.source_ref:
                raise ValueError("Motivation source evidence is not its first accepted input")
            refs = tuple(
                sorted(
                    evidence_by_motivation.get(record.motivation_id, []),
                    key=lambda item: (item.reference, item.kind.value),
                )
            )
            if refs != record.evidence_refs:
                raise ValueError("Motivation record evidence differs from its exact ledger")
            if record.revision_history[-1].evidence_refs != tuple(
                sorted(item.reference for item in record.evidence_refs)
            ):
                raise ValueError("latest Motivation revision does not bind current evidence")

            witnesses: list[
                tuple[str, int, tuple[str, ...], str, str, datetime]
            ] = []
            if record.history_anchor is not None:
                anchor = record.history_anchor
                witnesses.append(
                    (
                        anchor.through_event_id,
                        anchor.through_event_sequence,
                        anchor.through_evidence_refs,
                        anchor.through_operation,
                        anchor.through_reason,
                        anchor.through_created_at,
                    )
                )
            witnesses.extend(
                (
                    item.event_id or "",
                    item.event_sequence or 0,
                    item.evidence_refs,
                    item.operation.value,
                    item.reason.value,
                    item.created_at,
                )
                for item in record.revision_history
            )
            for (
                event_id,
                event_sequence,
                event_refs,
                operation,
                reason,
                created_at,
            ) in witnesses:
                revision_receipt = receipt_map.get(event_id)
                if (
                    revision_receipt is None
                    or revision_receipt.event_sequence != event_sequence
                    or revision_receipt.recorded_at != created_at
                    or record.motivation_id not in revision_receipt.motivation_ids
                ):
                    raise ValueError("Motivation revision lacks its exact event receipt")
                revision_source_refs = tuple(
                    sorted(
                        item.evidence.source_ref.reference
                        for item in record_entries
                        if item.event_sequence <= event_sequence
                    )
                )
                if event_refs != revision_source_refs:
                    raise ValueError("Motivation revision evidence differs from its ledger")
                if operation == MotivationRevisionOperation.CREATE.value:
                    allowed_receipts = {MotivationEventOperation.EVIDENCE}
                elif operation == MotivationRevisionOperation.RETIRE.value:
                    allowed_receipts = {MotivationEventOperation.RETIRE}
                elif reason == MotivationRevisionReason.DECAY.value:
                    allowed_receipts = {MotivationEventOperation.DECAY}
                elif reason == MotivationRevisionReason.SATIATION.value:
                    allowed_receipts = {MotivationEventOperation.SATIATION}
                else:
                    allowed_receipts = {
                        MotivationEventOperation.EVIDENCE,
                        MotivationEventOperation.REVIEW,
                    }
                if revision_receipt.operation not in allowed_receipts:
                    raise ValueError("Motivation revision operation disagrees with its receipt")
                if revision_receipt.operation is MotivationEventOperation.EVIDENCE:
                    if not any(
                        item.event_id == event_id
                        and item.event_sequence == event_sequence
                        and item.motivation_id == record.motivation_id
                        for item in self.evidence_ledger
                    ):
                        raise ValueError("evidence revision lacks its exact accepted input")
        payload = self.canonical_value()
        encoded = canonical_json(payload)
        if len(encoded) > MOTIVATION_SYSTEM_MAX_SERIALIZED_BYTES:
            raise MotivationCapacityExceeded("Motivation snapshot exceeds its byte bound")
        object.__setattr__(self, "serialized_bytes", len(encoded))
        object.__setattr__(self, "authority_digest", digest_payload(MOTIVATION_SYSTEM_DOMAIN, payload))

    def canonical_value(self) -> dict[str, object]:
        return {
            "candidates": [_candidate_value(item) for item in self.candidates],
            "event_receipts": [item.canonical_value() for item in self.event_receipts],
            "evidence_ledger": [item.canonical_value() for item in self.evidence_ledger],
            "goal_proposal_witnesses": [
                item.canonical_value() for item in self.goal_proposal_witnesses
            ],
            "records": [_record_value(item) for item in self.records],
            "schema_version": self.schema_version,
        }


def _event_input_digest(
    operation: MotivationEventOperation,
    payload: object,
) -> str:
    return digest_payload(
        MOTIVATION_EVENT_DOMAIN,
        {"operation": operation.value, "payload": payload},
    )


def _lifecycle_after_revision(
    revision: MotivationRevisionRecord,
) -> MotivationLifecycle:
    if revision.operation is MotivationRevisionOperation.CREATE:
        return MotivationLifecycle.ACTIVE
    if revision.operation is MotivationRevisionOperation.RETIRE:
        return MotivationLifecycle.RETIRED
    return {
        MotivationRevisionReason.EVIDENCE_UPDATE: MotivationLifecycle.ACTIVE,
        MotivationRevisionReason.DECAY: MotivationLifecycle.DORMANT,
        MotivationRevisionReason.SATIATION: MotivationLifecycle.SATIATED,
    }[revision.reason]


def _append_revision(
    current: MotivationRecord,
    revision: MotivationRevisionRecord,
) -> tuple[tuple[MotivationRevisionRecord, ...], RevisionCompactionAnchor | None]:
    history = current.revision_history + (revision,)
    anchor = current.history_anchor
    if len(history) > R13_MAX_REVISION_HISTORY:
        dropped = history[0]
        if dropped.event_id is None or dropped.event_sequence is None:
            raise MotivationDomainError("cannot compact a revision without its event")
        anchor = RevisionCompactionAnchor(
            authority_id=dropped.motivation_id,
            through_revision=dropped.revision,
            through_digest=dropped.record_digest,
            through_created_at=dropped.created_at,
            through_evidence_refs=dropped.evidence_refs,
            through_previous_revision_digest=dropped.previous_revision_digest,
            through_state=_lifecycle_after_revision(dropped).value,
            through_previous_state=(
                None
                if dropped.previous_lifecycle_state is None
                else dropped.previous_lifecycle_state.value
            ),
            through_operation=dropped.operation.value,
            through_reason=dropped.reason.value,
            through_event_id=dropped.event_id,
            through_event_sequence=dropped.event_sequence,
            through_state_digest=dropped.state_digest,
        )
        history = history[1:]
    return history, anchor


def _state_digest(
    record: MotivationRecord,
    *,
    lifecycle: MotivationLifecycle,
    strength: float,
    persistence: float,
    satiation: float,
    uncertainty: float,
    evidence_refs: tuple[R13Reference, ...],
) -> str:
    return motivation_state_digest(
        motivation_id=record.motivation_id,
        kind=record.kind,
        target=record.target,
        lifecycle=lifecycle,
        source_evidence=record.source_evidence,
        evidence_refs=evidence_refs,
        strength=strength,
        persistence=persistence,
        satiation=satiation,
        uncertainty=uncertainty,
        conflict_refs=record.conflict_refs,
        related_refs=record.related_refs,
        schema_version=record.schema_version,
    )


class MotivationSystem:
    """Serialized-by-caller process-local Motivation state authority.

    The caller owns event ordering.  Every mutation requires an explicit
    ``MotivationMutationEvidence``; this class has no clock, timer, runtime,
    model, persistence, or external-store dependency.
    """

    MAX_RECORDS: Final = MOTIVATION_MAX_RECORDS

    def __init__(self) -> None:
        self._records: tuple[MotivationRecord, ...] = ()
        self._evidence_ledger: tuple[MotivationEvidenceLedgerEntry, ...] = ()
        self._candidates: tuple[MotivationInterpretationCandidate, ...] = ()
        self._event_receipts: tuple[MotivationEventReceipt, ...] = ()
        self._goal_proposal_witnesses: tuple[MotivationGoalProposalWitness, ...] = ()

    @property
    def records(self) -> tuple[MotivationRecord, ...]:
        return self._records

    def get(self, motivation_id: str) -> MotivationRecord | None:
        identifier = validate_identifier(motivation_id)
        return next(
            (item for item in self._records if item.motivation_id == identifier),
            None,
        )

    def snapshot(self) -> MotivationSystemSnapshot:
        return MotivationSystemSnapshot(
            records=self._records,
            evidence_ledger=self._evidence_ledger,
            candidates=self._candidates,
            event_receipts=self._event_receipts,
            goal_proposal_witnesses=self._goal_proposal_witnesses,
        )

    export = snapshot

    def validate(self) -> None:
        self.snapshot()

    def ingest_candidate(
        self,
        candidate: MotivationInterpretationCandidate,
        event: MotivationMutationEvidence,
    ) -> MotivationInterpretationCandidate:
        """Retain model-inference evidence without granting it authority."""

        if not isinstance(candidate, MotivationInterpretationCandidate):
            raise TypeError("candidate must be MotivationInterpretationCandidate")
        if not isinstance(event, MotivationMutationEvidence):
            raise TypeError("event must be MotivationMutationEvidence")
        if (
            candidate.event_id != event.event_id
            or candidate.event_sequence != event.event_sequence
            or candidate.source_evidence_refs != event.evidence_refs
        ):
            raise MotivationDomainError("candidate is not bound to the exact event evidence")
        existing = next(
            (item for item in self._candidates if item.candidate_digest == candidate.candidate_digest),
            None,
        )
        input_digest = _event_input_digest(
            MotivationEventOperation.CANDIDATE,
            {"candidate_digest": candidate.candidate_digest},
        )
        existing_receipt = next(
            (item for item in self._event_receipts if item.event_id == event.event_id),
            None,
        )
        if existing_receipt is not None:
            if (
                existing_receipt.event_sequence == event.event_sequence
                and existing_receipt.recorded_at == event.recorded_at
                and existing_receipt.operation is MotivationEventOperation.EVIDENCE
                and existing_receipt.candidate_digest == candidate.candidate_digest
                and existing_receipt.evidence_refs == event.evidence_refs
            ):
                if existing is None:
                    raise MotivationDomainError("evidence receipt lost its accepted candidate")
                return existing
            prior = self._matching_event(
                event,
                MotivationEventOperation.CANDIDATE,
                input_digest,
            )
            if prior is not None:
                if existing is None:
                    raise MotivationDomainError("candidate receipt lost its candidate value")
                return existing
        if existing is not None:
            raise MotivationDomainError("candidate ledger value lacks its event receipt")
        if len(self._candidates) >= MOTIVATION_MAX_CANDIDATES:
            raise MotivationCapacityExceeded("Motivation candidate ledger is full")
        self._check_new_event_capacity(event)
        candidates = tuple(
            sorted((*self._candidates, candidate), key=lambda item: item.candidate_digest)
        )
        receipt = self._make_receipt(
            event,
            MotivationEventOperation.CANDIDATE,
            input_digest,
            (),
            candidate_digest=candidate.candidate_digest,
        )
        self._replace_state(
            candidates=candidates,
            event_receipts=(*self._event_receipts, receipt),
        )
        return candidate

    def apply_evidence(
        self,
        evidence: MotivationEvidence,
        event: MotivationMutationEvidence,
        *,
        candidate: MotivationInterpretationCandidate | None = None,
    ) -> MotivationRecord:
        """Create or reinforce exactly one Motivation from typed evidence."""

        if not isinstance(evidence, MotivationEvidence):
            raise TypeError("evidence must be MotivationEvidence")
        if not isinstance(event, MotivationMutationEvidence):
            raise TypeError("event must be MotivationMutationEvidence")
        if event.evidence_refs != (evidence.source_ref,):
            raise MotivationDomainError("event evidence must exactly match the typed source")
        if len(event.evidence_refs) > MOTIVATION_MAX_EVIDENCE_PER_EVENT:
            raise MotivationCapacityExceeded("Motivation event evidence budget is exhausted")
        if candidate is not None:
            if not isinstance(candidate, MotivationInterpretationCandidate):
                raise TypeError("candidate must be MotivationInterpretationCandidate")
            if (
                candidate.event_id != event.event_id
                or candidate.event_sequence != event.event_sequence
                or candidate.source_evidence_refs != event.evidence_refs
            ):
                raise MotivationDomainError("candidate does not match authoritative event evidence")

        input_digest = _event_input_digest(
            MotivationEventOperation.EVIDENCE,
            {
                "candidate_digest": None if candidate is None else candidate.candidate_digest,
                "evidence_digest": evidence.evidence_digest,
            },
        )
        prior = self._matching_event(event, MotivationEventOperation.EVIDENCE, input_digest)
        if prior is not None:
            result = self.get(evidence.motivation_id)
            if result is None:
                raise MotivationDomainError("replayed event lost its Motivation")
            return result
        self._check_event_order(event)

        evidence_entry = next(
            (
                item
                for item in self._evidence_ledger
                if item.evidence.source_ref == evidence.source_ref
            ),
            None,
        )
        if evidence_entry is not None:
            if evidence_entry.evidence != evidence:
                raise MotivationEvidenceConflict(
                    "evidence identity was reused with different source content or origin"
                )
            if candidate is not None:
                self.ingest_candidate(candidate, event)
            current = self.get(evidence_entry.motivation_id)
            if current is None:
                raise MotivationDomainError("accepted evidence lost its Motivation record")
            return current

        self._validate_source_time(evidence, event)
        motivation_id = evidence.motivation_id
        current = self.get(motivation_id)
        if current is not None and current.lifecycle is MotivationLifecycle.RETIRED:
            raise MotivationDomainError("retired Motivations cannot be reactivated")
        if len(self._evidence_ledger) >= MOTIVATION_MAX_EVIDENCE_LEDGER:
            raise MotivationCapacityExceeded("Motivation evidence identity ledger is full")
        if current is None and len(self._records) >= MOTIVATION_MAX_RECORDS:
            raise MotivationCapacityExceeded("Motivation record capacity is exhausted")
        if current is not None and len(current.evidence_refs) >= MOTIVATION_MAX_EVIDENCE_PER_RECORD:
            raise MotivationCapacityExceeded("Motivation evidence reference bound is exhausted")
        if current is not None and current.revision >= R13_MAX_REVISION:
            raise MotivationCapacityExceeded("Motivation revision bound is exhausted")

        self._check_new_event_capacity(event)

        gain = _SOURCE_WEIGHTS[evidence.source_ref.kind] * evidence.salience * evidence.confidence
        gain = min(0.35, max(0.0, gain))
        if current is None:
            if (
                evidence.source_ref.kind is R13ReferenceKind.MOTIVATION
                and evidence.source_ref.reference == motivation_id
            ):
                raise MotivationDomainError("a Motivation cannot use itself as source evidence")
            lifecycle = MotivationLifecycle.ACTIVE
            strength = gain
            persistence = min(1.0, evidence.persistence)
            satiation = 0.0
            uncertainty = evidence.uncertainty
            evidence_refs: tuple[R13Reference, ...] = (evidence.source_ref,)
            source_evidence = evidence.source_ref
            state_digest = motivation_state_digest(
                motivation_id=motivation_id,
                kind=evidence.kind,
                target=evidence.target,
                lifecycle=lifecycle,
                source_evidence=source_evidence,
                evidence_refs=evidence_refs,
                strength=strength,
                persistence=persistence,
                satiation=satiation,
                uncertainty=uncertainty,
            )
            revision = MotivationRevisionRecord(
                motivation_id=motivation_id,
                revision=0,
                operation=MotivationRevisionOperation.CREATE,
                reason=MotivationRevisionReason.CREATION,
                created_at=event.recorded_at,
                previous_lifecycle_state=None,
                state_digest=state_digest,
                event_id=event.event_id,
                event_sequence=event.event_sequence,
                evidence_refs=(evidence.source_ref.reference,),
            )
            updated = MotivationRecord(
                motivation_id=motivation_id,
                kind=evidence.kind,
                target=evidence.target,
                lifecycle=lifecycle,
                source_evidence=source_evidence,
                evidence_refs=evidence_refs,
                strength=strength,
                persistence=persistence,
                satiation=satiation,
                uncertainty=uncertainty,
                revision=0,
                revision_history=(revision,),
            )
        else:
            evidence_refs = tuple(
                sorted(
                    (*current.evidence_refs, evidence.source_ref),
                    key=lambda item: (item.reference, item.kind.value),
                )
            )
            lifecycle = MotivationLifecycle.ACTIVE
            strength = current.strength + (1.0 - current.strength) * gain
            persistence_gain = min(0.2, gain * evidence.persistence)
            persistence = current.persistence + (1.0 - current.persistence) * persistence_gain
            satiation = max(0.0, current.satiation - gain * 0.25)
            uncertainty = current.uncertainty * (1.0 - gain) + evidence.uncertainty * gain
            updated = self._append_state_revision(
                current,
                event,
                operation=MotivationRevisionOperation.UPDATE,
                reason=MotivationRevisionReason.EVIDENCE_UPDATE,
                lifecycle=lifecycle,
                strength=strength,
                persistence=persistence,
                satiation=satiation,
                uncertainty=uncertainty,
                evidence_refs=evidence_refs,
            )

        proposal_witnesses = self._goal_proposal_witnesses
        proposal_count_for_motivation = sum(
            item.motivation_id == motivation_id for item in proposal_witnesses
        )
        has_proposal = proposal_count_for_motivation > 0
        was_eligible = (
            current is not None
            and current.lifecycle is MotivationLifecycle.ACTIVE
            and current.strength >= MOTIVATION_GOAL_PROPOSAL_MIN_STRENGTH
        )
        if was_eligible and not has_proposal:
            raise MotivationDomainError(
                "eligible Motivation is missing its first Goal proposal witness"
            )
        creates_proposal = (
            not has_proposal
            and not was_eligible
            and updated.lifecycle is MotivationLifecycle.ACTIVE
            and updated.strength >= MOTIVATION_GOAL_PROPOSAL_MIN_STRENGTH
        )
        proposal_witness: MotivationGoalProposalWitness | None = None
        if creates_proposal:
            if proposal_count_for_motivation >= MOTIVATION_MAX_GOALS_PER_MOTIVATION:
                raise MotivationCapacityExceeded("Motivation Goal proposal budget is exhausted")
            if len(proposal_witnesses) >= MOTIVATION_MAX_GOAL_PROPOSAL_WITNESSES:
                raise MotivationCapacityExceeded("Motivation Goal proposal ledger is full")
            if sum(item.event_id == event.event_id for item in proposal_witnesses) >= MOTIVATION_MAX_GOALS_PER_EVENT:
                raise MotivationCapacityExceeded("Motivation event Goal proposal budget is exhausted")
            proposal_witness = MotivationGoalProposalWitness(
                motivation_id=motivation_id,
                kind=evidence.kind,
                event_id=event.event_id,
                event_sequence=event.event_sequence,
                recorded_at=event.recorded_at,
                evidence_ref=evidence.source_ref,
                eligible_strength=updated.strength,
                motivation_revision=updated.revision_history[-1].revision,
                motivation_revision_digest=updated.revision_history[-1].record_digest,
                motivation_state_digest=updated.revision_history[-1].state_digest,
            )
            proposal_witnesses = tuple(
                sorted(
                    (*proposal_witnesses, proposal_witness),
                    key=lambda item: item.motivation_id,
                )
            )

        ledger_entry = MotivationEvidenceLedgerEntry(
            evidence=evidence,
            motivation_id=motivation_id,
            event_id=event.event_id,
            event_sequence=event.event_sequence,
        )
        evidence_ledger = tuple(
            sorted(
                (*self._evidence_ledger, ledger_entry),
                key=lambda item: (
                    item.evidence.source_ref.kind.value,
                    item.evidence.source_ref.reference,
                ),
            )
        )
        candidates = self._candidates
        if candidate is not None and not any(
            item.candidate_digest == candidate.candidate_digest for item in candidates
        ):
            if len(candidates) >= MOTIVATION_MAX_CANDIDATES:
                raise MotivationCapacityExceeded("Motivation candidate ledger is full")
            candidates = tuple(
                sorted((*candidates, candidate), key=lambda item: item.candidate_digest)
            )
        receipt = self._make_receipt(
            event,
            MotivationEventOperation.EVIDENCE,
            input_digest,
            (motivation_id,),
            evidence_digest=evidence.evidence_digest,
            candidate_digest=None if candidate is None else candidate.candidate_digest,
            goal_proposal_digest=(
                None if proposal_witness is None else proposal_witness.proposal_digest
            ),
        )
        records = tuple(
            sorted(
                (
                    updated if item.motivation_id == motivation_id else item
                    for item in self._records
                ),
                key=lambda item: item.motivation_id,
            )
        )
        if current is None:
            records = tuple(sorted((*records, updated), key=lambda item: item.motivation_id))
        self._replace_state(
            records=records,
            evidence_ledger=evidence_ledger,
            candidates=candidates,
            event_receipts=(*self._event_receipts, receipt),
            goal_proposal_witnesses=proposal_witnesses,
        )
        return updated

    def decay(
        self,
        event: MotivationMutationEvidence,
        *,
        elapsed_seconds: float,
    ) -> tuple[MotivationRecord, ...]:
        """Apply explicit elapsed-time decay to an exact bounded Motive set."""

        if not isinstance(event, MotivationMutationEvidence):
            raise TypeError("event must be MotivationMutationEvidence")
        elapsed = _bounded_elapsed(elapsed_seconds)
        motivation_ids = self._motivation_ids_from_event(event)
        current_records = tuple(self._require_record(item) for item in motivation_ids)
        input_digest = _event_input_digest(
            MotivationEventOperation.DECAY,
            {"elapsed_seconds": elapsed.hex(), "motivation_ids": list(motivation_ids)},
        )
        prior = self._matching_event(event, MotivationEventOperation.DECAY, input_digest)
        if prior is not None:
            return tuple(self._require_record(item) for item in prior.motivation_ids)
        self._check_new_event_capacity(event)
        if elapsed == 0.0:
            return current_records

        updated_records: list[MotivationRecord] = []
        changed = False
        for current in current_records:
            if current.lifecycle is MotivationLifecycle.RETIRED:
                raise MotivationDomainError("retired Motivations cannot decay")
            scale = MOTIVATION_DECAY_SCALE_SECONDS * (1.0 + 9.0 * current.persistence)
            strength_factor = max(0.0, 1.0 - elapsed / scale)
            persistence_factor = max(
                0.0,
                1.0 - elapsed / MOTIVATION_PERSISTENCE_DECAY_SECONDS,
            )
            uncertainty_factor = min(1.0, elapsed / MOTIVATION_PERSISTENCE_DECAY_SECONDS)
            strength = current.strength * strength_factor
            persistence = current.persistence * persistence_factor
            uncertainty = current.uncertainty + (1.0 - current.uncertainty) * uncertainty_factor
            satiation = max(0.0, current.satiation - 0.05 * min(1.0, elapsed / scale))
            lifecycle = MotivationLifecycle.DORMANT
            if (strength, persistence, satiation, uncertainty, lifecycle) == (
                current.strength,
                current.persistence,
                current.satiation,
                current.uncertainty,
                current.lifecycle,
            ):
                updated_records.append(current)
                continue
            if current.revision >= R13_MAX_REVISION:
                raise MotivationCapacityExceeded("Motivation revision bound is exhausted")
            updated_records.append(
                self._append_state_revision(
                    current,
                    event,
                    operation=MotivationRevisionOperation.UPDATE,
                    reason=MotivationRevisionReason.DECAY,
                    lifecycle=lifecycle,
                    strength=strength,
                    persistence=persistence,
                    satiation=satiation,
                    uncertainty=uncertainty,
                    evidence_refs=current.evidence_refs,
                )
            )
            changed = True
        if not changed:
            return current_records
        receipt = self._make_receipt(
            event,
            MotivationEventOperation.DECAY,
            input_digest,
            motivation_ids,
            elapsed_seconds=elapsed,
        )
        update_map = {item.motivation_id: item for item in updated_records}
        records = tuple(
            update_map.get(item.motivation_id, item) for item in self._records
        )
        self._replace_state(records=records, event_receipts=(*self._event_receipts, receipt))
        return tuple(updated_records)

    def satiate(
        self,
        motivation_id: str,
        event: MotivationMutationEvidence,
    ) -> MotivationRecord:
        """Apply one fixed, bounded satiation step; no caller score is accepted."""

        return self._review_transition(
            motivation_id,
            event,
            operation=MotivationEventOperation.SATIATION,
            reason=MotivationRevisionReason.SATIATION,
            satiation_step=MOTIVATION_SATIATION_STEP,
        )

    def review(
        self,
        motivation_id: str,
        event: MotivationMutationEvidence,
    ) -> MotivationRecord:
        """Revisit one Motive with a fixed satiation relief step."""

        return self._review_transition(
            motivation_id,
            event,
            operation=MotivationEventOperation.REVIEW,
            reason=MotivationRevisionReason.EVIDENCE_UPDATE,
            satiation_step=-MOTIVATION_REVIEW_STEP,
        )

    def retire(
        self,
        motivation_id: str,
        event: MotivationMutationEvidence,
    ) -> MotivationRecord:
        """Explicitly retire a Motive; retirement is terminal under U1."""

        current = self._require_record(motivation_id)
        self._require_single_motivation_event(event, current.motivation_id)
        input_digest = _event_input_digest(
            MotivationEventOperation.RETIRE,
            {"motivation_id": current.motivation_id},
        )
        prior = self._matching_event(event, MotivationEventOperation.RETIRE, input_digest)
        if prior is not None:
            return self._require_record(current.motivation_id)
        self._check_event_order(event)
        if current.lifecycle is MotivationLifecycle.RETIRED:
            return current
        if current.revision >= R13_MAX_REVISION:
            raise MotivationCapacityExceeded("Motivation revision bound is exhausted")
        self._check_new_event_capacity(event)
        updated = self._append_state_revision(
            current,
            event,
            operation=MotivationRevisionOperation.RETIRE,
            reason=MotivationRevisionReason.RETIREMENT,
            lifecycle=MotivationLifecycle.RETIRED,
            strength=current.strength,
            persistence=current.persistence,
            satiation=current.satiation,
            uncertainty=current.uncertainty,
            evidence_refs=current.evidence_refs,
        )
        receipt = self._make_receipt(
            event,
            MotivationEventOperation.RETIRE,
            input_digest,
            (current.motivation_id,),
        )
        self._replace_one(updated, receipt)
        return updated

    def propose_goal(self, motivation_id: str) -> GoalRecord | None:
        """Return the exact first-eligibility proposal, if one was produced.

        The eligibility witness is created atomically with the event that first
        raises an active Motivation to the fixed threshold. Later updates do
        not rewrite its identity, evidence, or CREATE event.
        """
        current = self._require_record(motivation_id)
        witness = next(
            (
                item
                for item in self._goal_proposal_witnesses
                if item.motivation_id == current.motivation_id
            ),
            None,
        )
        return None if witness is None else witness.to_goal_record()

    def _review_transition(
        self,
        motivation_id: str,
        event: MotivationMutationEvidence,
        *,
        operation: MotivationEventOperation,
        reason: MotivationRevisionReason,
        satiation_step: float,
    ) -> MotivationRecord:
        current = self._require_record(motivation_id)
        self._require_single_motivation_event(event, current.motivation_id)
        input_digest = _event_input_digest(
            operation,
            {"motivation_id": current.motivation_id, "satiation_step": satiation_step.hex()},
        )
        prior = self._matching_event(event, operation, input_digest)
        if prior is not None:
            return current
        self._check_event_order(event)
        if current.lifecycle is MotivationLifecycle.RETIRED:
            raise MotivationDomainError("retired Motivations cannot be reviewed")
        new_satiation = min(1.0, max(0.0, current.satiation + satiation_step))
        lifecycle = (
            MotivationLifecycle.SATIATED
            if reason is MotivationRevisionReason.SATIATION
            else MotivationLifecycle.ACTIVE
        )
        if (new_satiation, lifecycle) == (current.satiation, current.lifecycle):
            return current
        if current.revision >= R13_MAX_REVISION:
            raise MotivationCapacityExceeded("Motivation revision bound is exhausted")
        self._check_new_event_capacity(event)
        updated = self._append_state_revision(
            current,
            event,
            operation=MotivationRevisionOperation.UPDATE,
            reason=reason,
            lifecycle=lifecycle,
            strength=current.strength,
            persistence=current.persistence,
            satiation=new_satiation,
            uncertainty=current.uncertainty,
            evidence_refs=current.evidence_refs,
        )
        receipt = self._make_receipt(event, operation, input_digest, (current.motivation_id,))
        self._replace_one(updated, receipt)
        return updated

    def _append_state_revision(
        self,
        current: MotivationRecord,
        event: MotivationMutationEvidence,
        *,
        operation: MotivationRevisionOperation,
        reason: MotivationRevisionReason,
        lifecycle: MotivationLifecycle,
        strength: float,
        persistence: float,
        satiation: float,
        uncertainty: float,
        evidence_refs: tuple[R13Reference, ...],
    ) -> MotivationRecord:
        if current.revision >= R13_MAX_REVISION:
            raise MotivationCapacityExceeded("Motivation revision bound is exhausted")
        next_revision = current.revision + 1
        state_digest = _state_digest(
            current,
            lifecycle=lifecycle,
            strength=strength,
            persistence=persistence,
            satiation=satiation,
            uncertainty=uncertainty,
            evidence_refs=evidence_refs,
        )
        revision = MotivationRevisionRecord(
            motivation_id=current.motivation_id,
            revision=next_revision,
            operation=operation,
            reason=reason,
            created_at=event.recorded_at,
            previous_lifecycle_state=current.lifecycle,
            state_digest=state_digest,
            previous_revision_digest=current.revision_history[-1].record_digest,
            event_id=event.event_id,
            event_sequence=event.event_sequence,
            evidence_refs=tuple(sorted(item.reference for item in evidence_refs)),
        )
        history, anchor = _append_revision(current, revision)
        return replace(
            current,
            lifecycle=lifecycle,
            evidence_refs=evidence_refs,
            strength=strength,
            persistence=persistence,
            satiation=satiation,
            uncertainty=uncertainty,
            revision=next_revision,
            revision_history=history,
            history_anchor=anchor,
        )

    @staticmethod
    def _validate_source_time(
        evidence: MotivationEvidence,
        event: MotivationMutationEvidence,
    ) -> None:
        if evidence.observed_at is not None and evidence.observed_at > event.recorded_at:
            raise MotivationDomainError("Motivation evidence timestamp is in the future")
        if evidence.source_event_sequence is not None:
            if evidence.source_event_sequence > event.event_sequence:
                raise MotivationDomainError("Motivation source event is in the future")
            if evidence.source_event_sequence == event.event_sequence and evidence.source_event_id != event.event_id:
                raise MotivationDomainError("same-sequence evidence must name the exact event")

    @staticmethod
    def _motivation_ids_from_event(
        event: MotivationMutationEvidence,
    ) -> tuple[str, ...]:
        if not event.evidence_refs or any(
            item.kind is not R13ReferenceKind.MOTIVATION
            for item in event.evidence_refs
        ):
            raise MotivationDomainError("time/review events must name exact Motivation refs")
        return tuple(item.reference for item in event.evidence_refs)

    @staticmethod
    def _require_single_motivation_event(
        event: MotivationMutationEvidence,
        motivation_id: str,
    ) -> None:
        if not isinstance(event, MotivationMutationEvidence):
            raise TypeError("event must be MotivationMutationEvidence")
        expected = (R13Reference(R13ReferenceKind.MOTIVATION, motivation_id),)
        if event.evidence_refs != expected:
            raise MotivationDomainError("event must bind the exact Motivation identity")

    def _require_record(self, motivation_id: str) -> MotivationRecord:
        record = self.get(motivation_id)
        if record is None:
            raise MotivationDomainError("Motivation is not present")
        return record

    def _matching_event(
        self,
        event: MotivationMutationEvidence,
        operation: MotivationEventOperation,
        input_digest: str,
    ) -> MotivationEventReceipt | None:
        existing = next(
            (item for item in self._event_receipts if item.event_id == event.event_id),
            None,
        )
        if existing is None:
            return None
        if (
            existing.event_sequence == event.event_sequence
            and existing.recorded_at == event.recorded_at
            and existing.operation is operation
            and existing.input_digest == input_digest
            and existing.evidence_refs == event.evidence_refs
        ):
            return existing
        raise MotivationEvidenceConflict("Motivation event identity was reused with conflicting input")

    def _check_event_order(self, event: MotivationMutationEvidence) -> None:
        if any(item.event_sequence == event.event_sequence for item in self._event_receipts):
            raise MotivationEvidenceConflict("Motivation event sequence was already used")
        if self._event_receipts:
            latest = self._event_receipts[-1]
            if event.event_sequence <= latest.event_sequence:
                raise MotivationDomainError("Motivation event sequence regressed")
            if event.recorded_at < latest.recorded_at:
                raise MotivationDomainError("Motivation event time regressed")

    def _check_new_event_capacity(self, event: MotivationMutationEvidence) -> None:
        self._check_event_order(event)
        if len(self._event_receipts) >= MOTIVATION_MAX_EVENT_RECEIPTS:
            raise MotivationCapacityExceeded("Motivation event receipt ledger is full")

    @staticmethod
    def _make_receipt(
        event: MotivationMutationEvidence,
        operation: MotivationEventOperation,
        input_digest: str,
        motivation_ids: tuple[str, ...],
        *,
        evidence_digest: str | None = None,
        candidate_digest: str | None = None,
        elapsed_seconds: float | None = None,
        goal_proposal_digest: str | None = None,
    ) -> MotivationEventReceipt:
        return MotivationEventReceipt(
            event_id=event.event_id,
            event_sequence=event.event_sequence,
            recorded_at=event.recorded_at,
            operation=operation,
            input_digest=input_digest,
            motivation_ids=tuple(sorted(set(motivation_ids))),
            evidence_refs=event.evidence_refs,
            evidence_digest=evidence_digest,
            candidate_digest=candidate_digest,
            elapsed_seconds=elapsed_seconds,
            goal_proposal_digest=goal_proposal_digest,
        )

    def _replace_one(
        self,
        record: MotivationRecord,
        receipt: MotivationEventReceipt,
    ) -> None:
        records = tuple(
            sorted(
                (record if item.motivation_id == record.motivation_id else item for item in self._records),
                key=lambda item: item.motivation_id,
            )
        )
        if not any(item.motivation_id == record.motivation_id for item in self._records):
            records = tuple(sorted((*records, record), key=lambda item: item.motivation_id))
        self._replace_state(records=records, event_receipts=(*self._event_receipts, receipt))

    def _replace_state(
        self,
        *,
        records: tuple[MotivationRecord, ...] | None = None,
        evidence_ledger: tuple[MotivationEvidenceLedgerEntry, ...] | None = None,
        candidates: tuple[MotivationInterpretationCandidate, ...] | None = None,
        event_receipts: tuple[MotivationEventReceipt, ...] | None = None,
        goal_proposal_witnesses: tuple[MotivationGoalProposalWitness, ...] | None = None,
    ) -> None:
        proposed = MotivationSystemSnapshot(
            records=self._records if records is None else records,
            evidence_ledger=(
                self._evidence_ledger if evidence_ledger is None else evidence_ledger
            ),
            candidates=self._candidates if candidates is None else candidates,
            event_receipts=(
                self._event_receipts if event_receipts is None else event_receipts
            ),
            goal_proposal_witnesses=(
                self._goal_proposal_witnesses
                if goal_proposal_witnesses is None
                else goal_proposal_witnesses
            ),
        )
        self._records = proposed.records
        self._evidence_ledger = proposed.evidence_ledger
        self._candidates = proposed.candidates
        self._event_receipts = proposed.event_receipts
        self._goal_proposal_witnesses = proposed.goal_proposal_witnesses


def _bounded_elapsed(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("elapsed_seconds must be a finite non-negative number")
    elapsed = float(value)
    if not math.isfinite(elapsed) or not 0.0 <= elapsed <= MOTIVATION_MAX_ELAPSED_SECONDS:
        raise ValueError("elapsed_seconds is outside the closed bounded interval")
    return elapsed


__all__ = [
    "MOTIVATION_DECAY_SCALE_SECONDS",
    "MOTIVATION_GOAL_PROPOSAL_MIN_STRENGTH",
    "MOTIVATION_MAX_CANDIDATES",
    "MOTIVATION_MAX_ELAPSED_SECONDS",
    "MOTIVATION_MAX_EVIDENCE_LEDGER",
    "MOTIVATION_MAX_EVIDENCE_PER_EVENT",
    "MOTIVATION_MAX_EVIDENCE_PER_RECORD",
    "MOTIVATION_MAX_EVENT_RECEIPTS",
    "MOTIVATION_MAX_GOAL_PROPOSAL_WITNESSES",
    "MOTIVATION_MAX_GOALS_PER_EVENT",
    "MOTIVATION_MAX_GOALS_PER_MOTIVATION",
    "MOTIVATION_MAX_ORIGIN_REFS",
    "MOTIVATION_MAX_RECORDS",
    "MOTIVATION_MAX_REVISION",
    "MOTIVATION_PERSISTENCE_DECAY_SECONDS",
    "MOTIVATION_REVIEW_STEP",
    "MOTIVATION_SATIATION_STEP",
    "MOTIVATION_SYSTEM_DOMAIN",
    "MOTIVATION_SYSTEM_MAX_SERIALIZED_BYTES",
    "MOTIVATION_SYSTEM_SCHEMA_VERSION",
    "MotivationCapacityExceeded",
    "MotivationDomainError",
    "MotivationEventOperation",
    "MotivationEventReceipt",
    "MotivationEvidence",
    "MotivationEvidenceConflict",
    "MotivationEvidenceLedgerEntry",
    "MotivationExperienceEvidence",
    "MotivationGoalProposalWitness",
    "MotivationMutationEvidence",
    "MotivationSystem",
    "MotivationSystemSnapshot",
]
