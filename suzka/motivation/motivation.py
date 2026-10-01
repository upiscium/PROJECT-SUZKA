"""Pure, immutable Motivation contracts for R13 U1.

An interpretation candidate is typed model evidence, not an authority.  This
module contains no producer, mutable system, runtime, persistence, or model
invocation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Final

from suzka.identifiers import validate_identifier
from suzka.motivation.common import (
    R13_MAX_CONFLICT_REFS,
    R13_MAX_EVIDENCE_REFS,
    R13_MAX_RELATED_REFS,
    R13_MAX_REVISION,
    R13_SCHEMA_VERSION,
    MotivationModelIdentity,
    R13Reference,
    R13ReferenceKind,
    RevisionCompactionAnchor,
    bounded_fraction,
    bounded_nonnegative_int,
    canonical_datetime,
    canonical_event,
    canonical_json,
    canonical_revision_witnesses,
    digest_payload,
    utc_datetime,
    validate_digest,
    validate_revision_suffix,
)


MOTIVATION_DOMAIN: Final = b"PROJECT-SUZKA:R13:MOTIVATION:V1\0"
MOTIVATION_STATE_DOMAIN: Final = b"PROJECT-SUZKA:R13:MOTIVATION-STATE:V1\0"
MOTIVATION_REVISION_DOMAIN: Final = b"PROJECT-SUZKA:R13:MOTIVATION-REVISION:V1\0"
MOTIVATION_CANDIDATE_DOMAIN: Final = (
    b"PROJECT-SUZKA:R13:MOTIVATION-INTERPRETATION-CANDIDATE:V1\0"
)

MOTIVATION_SOURCE_KINDS: Final = frozenset(
    {
        R13ReferenceKind.EXPERIENCE,
        R13ReferenceKind.EMOTION,
        R13ReferenceKind.VALUE,
        R13ReferenceKind.BELIEF,
        R13ReferenceKind.MOTIVATION,
        R13ReferenceKind.GOAL,
        R13ReferenceKind.COMMITMENT,
    }
)
MOTIVATION_TARGET_KINDS: Final = frozenset(
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
MOTIVATION_RELATED_KINDS: Final = frozenset(
    {
        R13ReferenceKind.EXPERIENCE,
        R13ReferenceKind.VALUE,
        R13ReferenceKind.BELIEF,
        R13ReferenceKind.GOAL,
        R13ReferenceKind.COMMITMENT,
    }
)


class MotivationKind(str, Enum):
    DRIVE = "drive"
    INTEREST = "interest"
    DESIRE = "desire"
    AVERSION = "aversion"


class MotivationSourceKind(str, Enum):
    EXPERIENCE = "experience"
    EMOTION = "emotion"
    VALUE = "value"
    BELIEF = "belief"
    MOTIVATION = "motivation"
    GOAL = "goal"
    COMMITMENT = "commitment"


class MotivationLifecycle(str, Enum):
    ACTIVE = "active"
    DORMANT = "dormant"
    SATIATED = "satiated"
    RETIRED = "retired"


class MotivationRevisionOperation(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    RETIRE = "retire"


class MotivationRevisionReason(str, Enum):
    CREATION = "creation"
    EVIDENCE_UPDATE = "evidence_update"
    DECAY = "decay"
    SATIATION = "satiation"
    RETIREMENT = "retirement"


class MotivationCandidateClassification(str, Enum):
    MODEL_INFERENCE = "model_inference"


def _enum(value: object, enum_type: type[Enum], name: str) -> None:
    if type(value) is not enum_type:
        raise TypeError(f"{name} must be a {enum_type.__name__}")


def _typed_reference(
    value: object,
    name: str,
    allowed_kinds: frozenset[R13ReferenceKind],
) -> R13Reference:
    if not isinstance(value, R13Reference):
        raise TypeError(f"{name} must be an R13Reference")
    if value.kind not in allowed_kinds:
        raise ValueError(f"{name} contains an unauthorized reference kind")
    return value


def _source(value: object, name: str = "source_evidence") -> R13Reference:
    return _typed_reference(value, name, MOTIVATION_SOURCE_KINDS)


def _target(value: object, name: str = "target") -> R13Reference:
    return _typed_reference(value, name, MOTIVATION_TARGET_KINDS)


def _typed_refs(
    value: object,
    name: str,
    maximum: int,
    allowed_kinds: frozenset[R13ReferenceKind],
    *,
    allow_empty: bool = True,
) -> tuple[R13Reference, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    if not allow_empty and not value:
        raise ValueError(f"{name} must be non-empty")
    if len(value) > maximum:
        raise ValueError(f"{name} exceeds its bound")
    references: list[R13Reference] = []
    for item in value:
        if not isinstance(item, R13Reference):
            raise TypeError(f"{name} must contain R13Reference values")
        if item.kind not in allowed_kinds:
            raise ValueError(f"{name} contains an unauthorized reference kind")
        references.append(item)
    typed_identities = tuple((item.kind.value, item.reference) for item in references)
    if len(set(typed_identities)) != len(typed_identities):
        raise ValueError(f"{name} typed references must be unique")
    ordered = tuple(
        sorted(references, key=lambda item: (item.reference, item.kind.value))
    )
    if tuple(references) != ordered:
        raise ValueError(f"{name} must be canonically ordered")
    return ordered


def motivation_id_for_target(kind: MotivationKind, target: R13Reference) -> str:
    """Return stable identity derived only from Motivation kind and target."""

    _enum(kind, MotivationKind, "kind")
    target = _target(target)
    return digest_payload(
        MOTIVATION_DOMAIN,
        {"kind": kind.value, "target": target.canonical_value()},
    )


def motivation_state_digest(
    *,
    motivation_id: str,
    kind: MotivationKind,
    target: R13Reference,
    lifecycle: MotivationLifecycle,
    source_evidence: R13Reference,
    evidence_refs: tuple[R13Reference, ...],
    strength: float = 0.0,
    persistence: float = 0.0,
    satiation: float = 0.0,
    uncertainty: float = 0.0,
    conflict_refs: tuple[R13Reference, ...] = (),
    related_refs: tuple[R13Reference, ...] = (),
    schema_version: int = R13_SCHEMA_VERSION,
) -> str:
    """Digest all current Motivation state that a revision can change."""

    _enum(kind, MotivationKind, "kind")
    target = _target(target)
    motivation_id = validate_identifier(motivation_id)
    if motivation_id != motivation_id_for_target(kind, target):
        raise ValueError("motivation_id does not match immutable kind and target")
    _enum(lifecycle, MotivationLifecycle, "lifecycle")
    source = _source(source_evidence)
    evidence = _typed_refs(
        evidence_refs,
        "evidence_refs",
        R13_MAX_EVIDENCE_REFS,
        MOTIVATION_SOURCE_KINDS,
        allow_empty=False,
    )
    if source not in evidence:
        raise ValueError("source_evidence must be present in evidence_refs")
    conflicts = _typed_refs(
        conflict_refs,
        "conflict_refs",
        R13_MAX_CONFLICT_REFS,
        MOTIVATION_SOURCE_KINDS | {R13ReferenceKind.STATE},
    )
    related = _typed_refs(
        related_refs,
        "related_refs",
        R13_MAX_RELATED_REFS,
        MOTIVATION_RELATED_KINDS,
    )
    if any(
        item.kind is R13ReferenceKind.MOTIVATION
        and item.reference == motivation_id
        for item in (*evidence, *conflicts, *related)
    ):
        raise ValueError("a Motivation cannot reference itself")
    fractions = {
        name: bounded_fraction(value, name)
        for name, value in (
            ("strength", strength),
            ("persistence", persistence),
            ("satiation", satiation),
            ("uncertainty", uncertainty),
        )
    }
    if type(schema_version) is not int or schema_version != R13_SCHEMA_VERSION:
        raise ValueError("unsupported Motivation schema version")
    return digest_payload(
        MOTIVATION_STATE_DOMAIN,
        {
            "conflict_refs": [item.canonical_value() for item in conflicts],
            "evidence_refs": [item.canonical_value() for item in evidence],
            "kind": kind.value,
            "lifecycle": lifecycle.value,
            "motivation_id": motivation_id,
            "persistence": fractions["persistence"].hex(),
            "related_refs": [item.canonical_value() for item in related],
            "satiation": fractions["satiation"].hex(),
            "schema_version": schema_version,
            "source_evidence": source.canonical_value(),
            "strength": fractions["strength"].hex(),
            "target": target.canonical_value(),
            "uncertainty": fractions["uncertainty"].hex(),
        },
    )


def _revision_fields(record: MotivationRevisionRecord) -> dict[str, object]:
    return {
        "created_at": record.created_at.isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        ),
        "event_id": record.event_id,
        "event_sequence": record.event_sequence,
        "evidence_refs": list(record.evidence_refs),
        "motivation_id": record.motivation_id,
        "operation": record.operation.value,
        "previous_revision_digest": record.previous_revision_digest,
        "previous_lifecycle_state": None
        if record.previous_lifecycle_state is None
        else record.previous_lifecycle_state.value,
        "reason": record.reason.value,
        "revision": record.revision,
        "state_digest": record.state_digest,
    }


def canonical_motivation_revision_payload(
    record: MotivationRevisionRecord,
) -> bytes:
    if not isinstance(record, MotivationRevisionRecord):
        raise TypeError("record must be MotivationRevisionRecord")
    return MOTIVATION_REVISION_DOMAIN + canonical_json(_revision_fields(record))


def motivation_revision_digest(record: MotivationRevisionRecord) -> str:
    return digest_payload(MOTIVATION_REVISION_DOMAIN, _revision_fields(record))


@dataclass(frozen=True, slots=True)
class MotivationRevisionRecord:
    motivation_id: str
    revision: int
    operation: MotivationRevisionOperation
    reason: MotivationRevisionReason
    created_at: datetime
    previous_lifecycle_state: MotivationLifecycle | None
    state_digest: str
    event_id: str | None = None
    event_sequence: int | None = None
    evidence_refs: tuple[str, ...] = ()
    previous_revision_digest: str | None = None
    record_digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "motivation_id", validate_identifier(self.motivation_id))
        object.__setattr__(
            self,
            "revision",
            bounded_nonnegative_int(
                self.revision,
                "revision",
                maximum=R13_MAX_REVISION,
            ),
        )
        _enum(self.operation, MotivationRevisionOperation, "operation")
        _enum(self.reason, MotivationRevisionReason, "reason")
        compatibility = {
            (MotivationRevisionOperation.CREATE, MotivationRevisionReason.CREATION),
            (MotivationRevisionOperation.UPDATE, MotivationRevisionReason.EVIDENCE_UPDATE),
            (MotivationRevisionOperation.UPDATE, MotivationRevisionReason.DECAY),
            (MotivationRevisionOperation.UPDATE, MotivationRevisionReason.SATIATION),
            (MotivationRevisionOperation.RETIRE, MotivationRevisionReason.RETIREMENT),
        }
        if (self.operation, self.reason) not in compatibility:
            raise ValueError("Motivation revision operation and reason are incompatible")
        object.__setattr__(self, "created_at", utc_datetime(self.created_at, "created_at"))
        if self.previous_lifecycle_state is not None:
            _enum(
                self.previous_lifecycle_state,
                MotivationLifecycle,
                "previous_lifecycle_state",
            )
        if self.revision == 0 and self.previous_lifecycle_state is not None:
            raise ValueError("Motivation genesis cannot have a previous lifecycle state")
        if self.revision > 0 and self.previous_lifecycle_state is None:
            raise ValueError("non-genesis Motivation revision requires its prior state")
        object.__setattr__(
            self,
            "state_digest",
            validate_digest(self.state_digest, "state_digest"),
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
            if self.operation is not MotivationRevisionOperation.CREATE:
                raise ValueError("revision zero must create the Motivation")
            if self.previous_revision_digest is not None:
                raise ValueError("genesis revision cannot have a previous digest")
        else:
            if self.operation is MotivationRevisionOperation.CREATE:
                raise ValueError("non-genesis revisions cannot create the Motivation")
            if self.previous_revision_digest is None:
                raise ValueError("non-genesis revision requires a previous digest")
        if self.previous_revision_digest is not None:
            validate_digest(self.previous_revision_digest, "previous_revision_digest")
        object.__setattr__(self, "record_digest", motivation_revision_digest(self))

    @property
    def authority_id(self) -> str:
        return self.motivation_id


def _record_fields(record: MotivationRecord) -> dict[str, object]:
    return {
        "conflict_refs": [item.canonical_value() for item in record.conflict_refs],
        "evidence_refs": [item.canonical_value() for item in record.evidence_refs],
        "history_anchor": None
        if record.history_anchor is None
        else {
            "authority_id": record.history_anchor.authority_id,
            "through_created_at": canonical_datetime(
                record.history_anchor.through_created_at
            ),
            "through_event_id": record.history_anchor.through_event_id,
            "through_event_sequence": record.history_anchor.through_event_sequence,
            "through_evidence_refs": list(
                record.history_anchor.through_evidence_refs
            ),
            "through_digest": record.history_anchor.through_digest,
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
        "kind": record.kind.value,
        "lifecycle": record.lifecycle.value,
        "motivation_id": record.motivation_id,
        "persistence": record.persistence.hex(),
        "related_refs": [item.canonical_value() for item in record.related_refs],
        "revision": record.revision,
        "revision_history": [
            {**_revision_fields(item), "record_digest": item.record_digest}
            for item in record.revision_history
        ],
        "satiation": record.satiation.hex(),
        "schema_version": record.schema_version,
        "source_evidence": record.source_evidence.canonical_value(),
        "strength": record.strength.hex(),
        "target": record.target.canonical_value(),
        "uncertainty": record.uncertainty.hex(),
    }


def canonical_motivation_payload(record: MotivationRecord) -> bytes:
    if not isinstance(record, MotivationRecord):
        raise TypeError("record must be MotivationRecord")
    return MOTIVATION_DOMAIN + canonical_json(_record_fields(record))


def motivation_record_digest(record: MotivationRecord) -> str:
    return digest_payload(MOTIVATION_DOMAIN, _record_fields(record))


def _require_motivation_transition(
    *,
    lifecycle: MotivationLifecycle,
    revision_history: tuple[MotivationRevisionRecord, ...],
    history_anchor: RevisionCompactionAnchor | None,
) -> None:
    """Derive lifecycle through the complete retained transition suffix."""

    transitions = {
        MotivationLifecycle.ACTIVE: {
            (MotivationRevisionOperation.UPDATE, MotivationRevisionReason.EVIDENCE_UPDATE): MotivationLifecycle.ACTIVE,
            (MotivationRevisionOperation.UPDATE, MotivationRevisionReason.DECAY): MotivationLifecycle.DORMANT,
            (MotivationRevisionOperation.UPDATE, MotivationRevisionReason.SATIATION): MotivationLifecycle.SATIATED,
            (MotivationRevisionOperation.RETIRE, MotivationRevisionReason.RETIREMENT): MotivationLifecycle.RETIRED,
        },
        MotivationLifecycle.DORMANT: {
            (MotivationRevisionOperation.UPDATE, MotivationRevisionReason.EVIDENCE_UPDATE): MotivationLifecycle.ACTIVE,
            (MotivationRevisionOperation.UPDATE, MotivationRevisionReason.DECAY): MotivationLifecycle.DORMANT,
            (MotivationRevisionOperation.UPDATE, MotivationRevisionReason.SATIATION): MotivationLifecycle.SATIATED,
            (MotivationRevisionOperation.RETIRE, MotivationRevisionReason.RETIREMENT): MotivationLifecycle.RETIRED,
        },
        MotivationLifecycle.SATIATED: {
            (MotivationRevisionOperation.UPDATE, MotivationRevisionReason.EVIDENCE_UPDATE): MotivationLifecycle.ACTIVE,
            (MotivationRevisionOperation.UPDATE, MotivationRevisionReason.DECAY): MotivationLifecycle.DORMANT,
            (MotivationRevisionOperation.UPDATE, MotivationRevisionReason.SATIATION): MotivationLifecycle.SATIATED,
            (MotivationRevisionOperation.RETIRE, MotivationRevisionReason.RETIREMENT): MotivationLifecycle.RETIRED,
        },
        MotivationLifecycle.RETIRED: {},
    }

    if history_anchor is None:
        state: MotivationLifecycle | None = None
    else:
        try:
            state = MotivationLifecycle(history_anchor.through_state)
            anchor_operation = MotivationRevisionOperation(
                history_anchor.through_operation
            )
            anchor_reason = MotivationRevisionReason(history_anchor.through_reason)
            anchor_previous_state = (
                None
                if history_anchor.through_previous_state is None
                else MotivationLifecycle(history_anchor.through_previous_state)
            )
        except ValueError as error:
            raise ValueError(
                "history anchor contains invalid Motivation transition data"
            ) from error
        if (
            history_anchor.through_state_digest is None
            or history_anchor.through_proposal_digest is not None
        ):
            raise ValueError("Motivation history anchor has invalid digest fields")
        anchored_revision = MotivationRevisionRecord(
            motivation_id=history_anchor.authority_id,
            revision=history_anchor.through_revision,
            operation=anchor_operation,
            reason=anchor_reason,
            created_at=history_anchor.through_created_at,
            previous_lifecycle_state=anchor_previous_state,
            state_digest=history_anchor.through_state_digest,
            event_id=history_anchor.through_event_id,
            event_sequence=history_anchor.through_event_sequence,
            evidence_refs=history_anchor.through_evidence_refs,
            previous_revision_digest=history_anchor.through_previous_revision_digest,
        )
        if anchored_revision.record_digest != history_anchor.through_digest:
            raise ValueError("Motivation history anchor does not match its revision digest")

        operation_reason = (anchor_operation, anchor_reason)
        if anchor_previous_state is None:
            if (
                history_anchor.through_revision != 0
                or operation_reason
                != (
                    MotivationRevisionOperation.CREATE,
                    MotivationRevisionReason.CREATION,
                )
                or state is not MotivationLifecycle.ACTIVE
            ):
                raise ValueError("Motivation anchor has an invalid genesis transition")
        elif transitions[anchor_previous_state].get(operation_reason) is not state:
            raise ValueError("Motivation anchor hides an invalid lifecycle transition")
    for item in revision_history:
        operation_reason = (item.operation, item.reason)
        if state is None:
            if (
                item.revision != 0
                or item.previous_lifecycle_state is not None
                or operation_reason
                != (
                    MotivationRevisionOperation.CREATE,
                    MotivationRevisionReason.CREATION,
                )
            ):
                raise ValueError("Motivation history must begin with creation")
            state = MotivationLifecycle.ACTIVE
            continue
        if item.previous_lifecycle_state is not state:
            raise ValueError("Motivation revision prior lifecycle state is inconsistent")
        next_state = transitions[state].get(operation_reason)
        if next_state is None:
            raise ValueError(
                "Motivation history contains an invalid lifecycle transition"
            )
        state = next_state
    if state is not lifecycle:
        raise ValueError("Motivation history does not reach its lifecycle")


@dataclass(frozen=True, slots=True)
class MotivationRecord:
    """Bounded current Motivation state; it does not mutate authority."""

    motivation_id: str
    kind: MotivationKind
    target: R13Reference
    lifecycle: MotivationLifecycle
    source_evidence: R13Reference
    evidence_refs: tuple[R13Reference, ...]
    strength: float = 0.0
    persistence: float = 0.0
    satiation: float = 0.0
    uncertainty: float = 0.0
    conflict_refs: tuple[R13Reference, ...] = ()
    related_refs: tuple[R13Reference, ...] = ()
    revision: int = 0
    revision_history: tuple[MotivationRevisionRecord, ...] = ()
    history_anchor: RevisionCompactionAnchor | None = None
    schema_version: int = R13_SCHEMA_VERSION
    record_digest: str = field(init=False)

    def __post_init__(self) -> None:
        _enum(self.kind, MotivationKind, "kind")
        target = _target(self.target)
        expected_id = motivation_id_for_target(self.kind, target)
        if validate_identifier(self.motivation_id) != expected_id:
            raise ValueError("motivation_id does not match immutable kind and target")
        _enum(self.lifecycle, MotivationLifecycle, "lifecycle")
        source = _source(self.source_evidence)
        evidence = _typed_refs(
            self.evidence_refs,
            "evidence_refs",
            R13_MAX_EVIDENCE_REFS,
            MOTIVATION_SOURCE_KINDS,
            allow_empty=False,
        )
        if source not in evidence:
            raise ValueError("source_evidence must be present in evidence_refs")
        conflict_refs = _typed_refs(
            self.conflict_refs,
            "conflict_refs",
            R13_MAX_CONFLICT_REFS,
            MOTIVATION_SOURCE_KINDS | {R13ReferenceKind.STATE},
        )
        related_refs = _typed_refs(
            self.related_refs,
            "related_refs",
            R13_MAX_RELATED_REFS,
            MOTIVATION_RELATED_KINDS,
        )
        if any(
            item.kind is R13ReferenceKind.MOTIVATION
            and item.reference == self.motivation_id
            for item in (*evidence, *conflict_refs, *related_refs)
        ):
            raise ValueError("a Motivation cannot reference itself")
        for name in ("strength", "persistence", "satiation", "uncertainty"):
            object.__setattr__(self, name, bounded_fraction(getattr(self, name), name))
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "source_evidence", source)
        object.__setattr__(self, "evidence_refs", evidence)
        object.__setattr__(self, "conflict_refs", conflict_refs)
        object.__setattr__(self, "related_refs", related_refs)
        object.__setattr__(
            self,
            "revision",
            bounded_nonnegative_int(
                self.revision,
                "revision",
                maximum=R13_MAX_REVISION,
            ),
        )
        if type(self.schema_version) is not int or self.schema_version != R13_SCHEMA_VERSION:
            raise ValueError("unsupported Motivation schema version")
        if type(self.revision_history) is not tuple or any(
            not isinstance(item, MotivationRevisionRecord)
            for item in self.revision_history
        ):
            raise TypeError("revision_history must contain MotivationRevisionRecord values")
        validate_revision_suffix(
            authority_id=self.motivation_id,
            current_revision=self.revision,
            revision_history=self.revision_history,
            history_anchor=self.history_anchor,
            revision_digest=motivation_revision_digest,
        )
        state_digest = motivation_state_digest(
            motivation_id=self.motivation_id,
            kind=self.kind,
            target=target,
            lifecycle=self.lifecycle,
            source_evidence=source,
            evidence_refs=evidence,
            strength=self.strength,
            persistence=self.persistence,
            satiation=self.satiation,
            uncertainty=self.uncertainty,
            conflict_refs=conflict_refs,
            related_refs=related_refs,
            schema_version=self.schema_version,
        )
        if self.revision_history[-1].state_digest != state_digest:
            raise ValueError("Motivation revision does not bind its current state")
        _require_motivation_transition(
            lifecycle=self.lifecycle,
            revision_history=self.revision_history,
            history_anchor=self.history_anchor,
        )
        object.__setattr__(self, "record_digest", motivation_record_digest(self))

    @property
    def source_evidence_refs(self) -> tuple[R13Reference, ...]:
        """All typed evidence refs, retained as a compatibility spelling."""

        return self.evidence_refs

    @property
    def origin(self) -> R13Reference:
        return self.source_evidence

    @property
    def history(self) -> tuple[MotivationRevisionRecord, ...]:
        return self.revision_history

    @property
    def history_anchor_digest(self) -> str | None:
        return None if self.history_anchor is None else self.history_anchor.through_digest

    @property
    def history_anchor_revision(self) -> int | None:
        return None if self.history_anchor is None else self.history_anchor.through_revision


@dataclass(frozen=True, slots=True)
class MotivationInterpretationCandidate:
    """Bounded model inference evidence, never authoritative Motivation state."""

    source_evidence_refs: tuple[R13Reference, ...]
    event_id: str
    event_sequence: int
    model_identity: MotivationModelIdentity
    suggested_kind: MotivationKind
    suggested_target: R13Reference
    suggested_strength: float = 0.0
    suggested_persistence: float = 0.0
    suggested_satiation: float = 0.0
    suggested_uncertainty: float = 0.0
    classification: MotivationCandidateClassification = (
        MotivationCandidateClassification.MODEL_INFERENCE
    )
    candidate_digest: str = field(init=False)

    def __post_init__(self) -> None:
        evidence = _typed_refs(
            self.source_evidence_refs,
            "source_evidence_refs",
            R13_MAX_EVIDENCE_REFS,
            MOTIVATION_SOURCE_KINDS,
            allow_empty=False,
        )
        event_id, event_sequence = canonical_event(
            self.event_id,
            self.event_sequence,
        )
        if event_id is None or event_sequence is None:
            raise AssertionError("required candidate event was lost")
        _enum(self.suggested_kind, MotivationKind, "suggested_kind")
        target = _target(self.suggested_target, "suggested_target")
        if not isinstance(self.model_identity, MotivationModelIdentity):
            raise TypeError("model_identity must be MotivationModelIdentity")
        _enum(self.classification, MotivationCandidateClassification, "classification")
        values: dict[str, object] = {
            "suggested_strength": self.suggested_strength,
            "suggested_persistence": self.suggested_persistence,
            "suggested_satiation": self.suggested_satiation,
            "suggested_uncertainty": self.suggested_uncertainty,
        }
        for name, value in values.items():
            object.__setattr__(self, name, bounded_fraction(value, name))
        object.__setattr__(self, "source_evidence_refs", evidence)
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "event_sequence", event_sequence)
        object.__setattr__(self, "model_identity", self.model_identity)
        object.__setattr__(self, "suggested_target", target)
        payload = {
            "classification": self.classification.value,
            "event_id": event_id,
            "event_sequence": event_sequence,
            "model_identity": {
                "model_key": self.model_identity.model_key,
                "model_id": self.model_identity.model_id,
                "provider_name": self.model_identity.provider_name,
            },
            "source_evidence_refs": [item.canonical_value() for item in evidence],
            "suggested_kind": self.suggested_kind.value,
            "suggested_persistence": self.suggested_persistence.hex(),
            "suggested_satiation": self.suggested_satiation.hex(),
            "suggested_strength": self.suggested_strength.hex(),
            "suggested_target": target.canonical_value(),
            "suggested_uncertainty": self.suggested_uncertainty.hex(),
        }
        object.__setattr__(
            self,
            "candidate_digest",
            digest_payload(MOTIVATION_CANDIDATE_DOMAIN, payload),
        )

    @property
    def motivation_id(self) -> str:
        """The proposed target identity, without making it authoritative."""

        return motivation_id_for_target(self.suggested_kind, self.suggested_target)

    @property
    def model_key(self) -> str:
        return self.model_identity.model_key

    @property
    def source_evidence(self) -> R13Reference:
        return self.source_evidence_refs[0]

    @property
    def origin(self) -> R13Reference:
        return self.source_evidence


def _candidate_fields(
    candidate: MotivationInterpretationCandidate,
) -> dict[str, object]:
    return {
        "classification": candidate.classification.value,
        "event_id": candidate.event_id,
        "event_sequence": candidate.event_sequence,
        "model_identity": {
            "model_key": candidate.model_identity.model_key,
            "model_id": candidate.model_identity.model_id,
            "provider_name": candidate.model_identity.provider_name,
        },
        "source_evidence_refs": [
            item.canonical_value() for item in candidate.source_evidence_refs
        ],
        "suggested_kind": candidate.suggested_kind.value,
        "suggested_persistence": candidate.suggested_persistence.hex(),
        "suggested_satiation": candidate.suggested_satiation.hex(),
        "suggested_strength": candidate.suggested_strength.hex(),
        "suggested_target": candidate.suggested_target.canonical_value(),
        "suggested_uncertainty": candidate.suggested_uncertainty.hex(),
    }


def canonical_motivation_candidate_payload(
    candidate: MotivationInterpretationCandidate,
) -> bytes:
    if not isinstance(candidate, MotivationInterpretationCandidate):
        raise TypeError("candidate must be MotivationInterpretationCandidate")
    return MOTIVATION_CANDIDATE_DOMAIN + canonical_json(_candidate_fields(candidate))


def motivation_candidate_digest(candidate: MotivationInterpretationCandidate) -> str:
    if not isinstance(candidate, MotivationInterpretationCandidate):
        raise TypeError("candidate must be MotivationInterpretationCandidate")
    return digest_payload(MOTIVATION_CANDIDATE_DOMAIN, _candidate_fields(candidate))


# Descriptive aliases for downstream contract imports.
MotivationSourceReference = R13Reference
MotivationCandidate = MotivationInterpretationCandidate
canonical_motivation_revision_payload = canonical_motivation_revision_payload
motivation_digest = motivation_record_digest


__all__ = [
    "MOTIVATION_CANDIDATE_DOMAIN",
    "MOTIVATION_DOMAIN",
    "MOTIVATION_RELATED_KINDS",
    "MOTIVATION_REVISION_DOMAIN",
    "MOTIVATION_SOURCE_KINDS",
    "MOTIVATION_TARGET_KINDS",
    "MotivationCandidate",
    "MotivationCandidateClassification",
    "MotivationInterpretationCandidate",
    "MotivationKind",
    "MotivationLifecycle",
    "MotivationRecord",
    "MotivationRevisionOperation",
    "MotivationRevisionReason",
    "MotivationRevisionRecord",
    "MotivationSourceKind",
    "MotivationSourceReference",
    "canonical_motivation_payload",
    "canonical_motivation_candidate_payload",
    "canonical_motivation_revision_payload",
    "motivation_candidate_digest",
    "motivation_digest",
    "motivation_id_for_target",
    "motivation_record_digest",
    "motivation_revision_digest",
]
