"""Intrinsic, bounded Belief authority for R12 U5.

The immutable values in :mod:`suzka.belief.records` remain useful as pure
contracts, but they do not decide which values become authoritative.  This
module owns that decision and keeps every mutation event-bound.  It does not
listen to Memory, invoke a model, or expose an HTTP/API producer.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from functools import wraps
import hashlib
import json
from threading import RLock
from typing import Callable, Concatenate, Final, Iterable, ParamSpec, TypeVar

from suzka.belief.records import (
    BELIEF_MAX_EVENT_SEQUENCE,
    BELIEF_MAX_REVISIONS,
    BELIEF_MAX_REVISION,
    BELIEF_MAX_REVISION_WITNESSES,
    BeliefEpistemicStatus,
    BeliefEvidence,
    BeliefLifecycle,
    BeliefProposition,
    BeliefRecord,
    BeliefRevisionOperation,
    BeliefRevisionReason,
    BeliefRevisionRecord,
    BeliefSubjectAdmission,
    belief_record_digest,
)
from suzka.identifiers import validate_identifier


BELIEF_MAX_RECORDS: Final = 256
BELIEF_MAX_SERIALIZED_BYTES: Final = 64 * 1024 * 1024
BELIEF_SYSTEM_SCHEMA_VERSION: Final = 1
BELIEF_ID_DOMAIN: Final = b"PROJECT-SUZKA:R12:BELIEF-ID:V1\0"
BELIEF_SYSTEM_DOMAIN: Final = b"PROJECT-SUZKA:R12:BELIEF-SYSTEM:V1\0"

_P = ParamSpec("_P")
_T = TypeVar("_T")


class BeliefDomainError(ValueError):
    """Base class for bounded Belief authority failures."""


class BeliefCapacityExceeded(BeliefDomainError):
    """The intrinsic Belief authority bound would be exceeded."""


class BeliefConflict(BeliefDomainError):
    """A cross-record Belief relationship is inconsistent."""


def _locked_mutation(
    method: Callable[Concatenate[BeliefSystem, _P], _T],
) -> Callable[Concatenate[BeliefSystem, _P], _T]:
    """Serialize each read-modify-write Belief transition."""

    @wraps(method)
    def wrapped(self: BeliefSystem, *args: _P.args, **kwargs: _P.kwargs) -> _T:
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapped


def belief_id_for_proposition(proposition_digest: str) -> str:
    """Return the deterministic authority identity for one proposition."""

    if type(proposition_digest) is not str or len(proposition_digest) != 64:
        raise ValueError("proposition_digest must be a SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in proposition_digest):
        raise ValueError("proposition_digest must be a lowercase SHA-256 digest")
    return "belief-" + hashlib.sha256(
        BELIEF_ID_DOMAIN + proposition_digest.encode("ascii")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class BeliefMutationEvidence:
    """The minimal event witness required by one intrinsic mutation."""

    event_id: str
    event_sequence: int
    recorded_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", validate_identifier(self.event_id))
        if (
            type(self.event_sequence) is not int
            or not 1 <= self.event_sequence <= BELIEF_MAX_EVENT_SEQUENCE
        ):
            raise ValueError("event_sequence must be a positive bounded integer")
        if not isinstance(self.recorded_at, datetime):
            raise TypeError("recorded_at must be a datetime")
        if self.recorded_at.tzinfo is None or self.recorded_at.utcoffset() is None:
            raise ValueError("recorded_at must be timezone-aware")
        object.__setattr__(
            self, "recorded_at", self.recorded_at.astimezone(timezone.utc)
        )

    @classmethod
    def from_event(cls, event: object) -> BeliefMutationEvidence:
        """Build evidence from an active AgentEvent without importing runtime code."""

        event_id = getattr(event, "event_id", None)
        event_sequence = getattr(event, "processing_sequence", None)
        recorded_at = getattr(event, "requested_at", None)
        if (
            type(event_id) is not str
            or type(event_sequence) is not int
            or not isinstance(recorded_at, datetime)
        ):
            raise BeliefDomainError("Belief mutation requires an active event")
        return cls(event_id, event_sequence, recorded_at)


def _canonical_system_digest(records: tuple[BeliefRecord, ...]) -> str:
    payload = {
        "records": [belief_record_digest(record) for record in records],
        "schema_version": BELIEF_SYSTEM_SCHEMA_VERSION,
    }
    return hashlib.sha256(
        BELIEF_SYSTEM_DOMAIN
        + json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()


def _canonical_record_payload(record: BeliefRecord) -> dict[str, object]:
    admission = record.subject_admission
    return {
        "belief_id": record.belief_id,
        "confidence": record.confidence.hex(),
        "context_scope": list(record.context_scope),
        "epistemic_status": record.epistemic_status.value,
        "evidence": [
            {
                "evidence_ref": item.evidence_ref,
                "evidence_type": item.evidence_type.value,
            }
            for item in record.evidence
        ],
        "history_anchor_digest": record.history_anchor_digest,
        "lifecycle": record.lifecycle.value,
        "proposition": {
            "canonical_text": record.proposition.canonical_text,
            "object": record.proposition.object,
            "predicate": record.proposition.predicate,
            "proposition_digest": record.proposition.proposition_digest,
            "subject": record.proposition.subject,
        },
        "revision": record.revision,
        "revision_history": [
            {
                "belief_id": revision.belief_id,
                "created_at": revision.created_at.isoformat(),
                "event_id": revision.event_id,
                "event_sequence": revision.event_sequence,
                "evidence_refs": list(revision.evidence_refs),
                "operation": revision.operation.value,
                "previous_revision_digest": revision.previous_revision_digest,
                "reason": revision.reason.value,
                "record_digest": revision.record_digest,
                "revision": revision.revision,
            }
            for revision in record.revision_history
        ],
        "schema_version": record.schema_version,
        "subject_admission": (
            None
            if admission is None
            else {
                "admission_digest": admission.admission_digest,
                "event_id": admission.event_id,
                "event_sequence": admission.event_sequence,
                "evidence_refs": list(admission.evidence_refs),
                "proposition_digest": admission.proposition_digest,
                "reason": admission.reason.value,
            }
        ),
        "superseded_by_id": record.superseded_by_id,
        "supersedes_id": record.supersedes_id,
        "valid_from": None if record.valid_from is None else record.valid_from.isoformat(),
        "valid_until": None if record.valid_until is None else record.valid_until.isoformat(),
    }


def canonical_belief_system_bytes(records: tuple[BeliefRecord, ...]) -> bytes:
    """Return the deterministic representation used for size admission."""

    return json.dumps(
        {
            "records": [_canonical_record_payload(record) for record in records],
            "schema_version": BELIEF_SYSTEM_SCHEMA_VERSION,
        },
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


@dataclass(frozen=True, slots=True)
class BeliefSystemSnapshot:
    """Canonical immutable projection of the complete Belief authority."""

    records: tuple[BeliefRecord, ...] = ()
    schema_version: int = BELIEF_SYSTEM_SCHEMA_VERSION
    authority_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != BELIEF_SYSTEM_SCHEMA_VERSION:
            raise ValueError("unsupported Belief system schema version")
        if type(self.records) is not tuple:
            raise TypeError("records must be a tuple")
        if len(self.records) > BELIEF_MAX_RECORDS:
            raise BeliefCapacityExceeded("Belief authority exceeds its record bound")
        if any(not isinstance(record, BeliefRecord) for record in self.records):
            raise TypeError("records must contain BeliefRecord values")
        ids = tuple(record.belief_id for record in self.records)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("Belief records must be sorted and unique")
        if len(canonical_belief_system_bytes(self.records)) > BELIEF_MAX_SERIALIZED_BYTES:
            raise BeliefCapacityExceeded("Belief snapshot exceeds its serialized byte bound")
        object.__setattr__(self, "authority_digest", _canonical_system_digest(self.records))


def _revision_evidence_refs(
    record: BeliefRecord,
    admission: BeliefSubjectAdmission | None = None,
    additional_admissions: tuple[BeliefSubjectAdmission, ...] = (),
) -> tuple[str, ...]:
    refs = {evidence.evidence_ref for evidence in record.evidence}
    admissions = (() if admission is None else (admission,)) + additional_admissions
    refs.update(item.admission_digest for item in admissions)
    if len(refs) > BELIEF_MAX_REVISION_WITNESSES:
        raise BeliefCapacityExceeded("Belief revision witnesses exceed their bound")
    return tuple(sorted(refs))


class BeliefSystem:
    """Serialized intrinsic Belief authority with no automatic producer."""

    MAX_RECORDS: Final = BELIEF_MAX_RECORDS

    def __init__(
        self,
        records: Iterable[BeliefRecord] = (),
        *,
        event_provider: Callable[[], object | None] | None = None,
    ) -> None:
        self._lock = RLock()
        if event_provider is not None and not callable(event_provider):
            raise TypeError("event_provider must be callable")
        self._event_provider = event_provider
        canonical = tuple(sorted(tuple(records), key=lambda record: record.belief_id))
        self._replace_records(canonical)

    @classmethod
    def restore_snapshot(
        cls,
        snapshot: BeliefSystemSnapshot,
        *,
        event_provider: Callable[[], object | None] | None = None,
    ) -> BeliefSystem:
        if not isinstance(snapshot, BeliefSystemSnapshot):
            raise TypeError("snapshot must be BeliefSystemSnapshot")
        system = cls(snapshot.records, event_provider=event_provider)
        if system.snapshot().authority_digest != snapshot.authority_digest:
            raise BeliefDomainError("Belief authority digest does not match records")
        return system

    @property
    def records(self) -> tuple[BeliefRecord, ...]:
        with self._lock:
            return tuple(self._records)

    def get(self, belief_id: str) -> BeliefRecord | None:
        identifier = validate_identifier(belief_id)
        with self._lock:
            return self._record_map.get(identifier)

    def snapshot(self) -> BeliefSystemSnapshot:
        with self._lock:
            return BeliefSystemSnapshot(tuple(self._records))

    def validate(self) -> None:
        with self._lock:
            self._validate_records(self._records)

    def ordinary_active(
        self,
        *,
        at: datetime | None = None,
        context_id: str | None = None,
    ) -> tuple[BeliefRecord, ...]:
        with self._lock:
            return tuple(
                record
                for record in self._records
                if record.is_ordinary_active(at=at, context_id=context_id)
            )

    def create_proposal(
        self,
        proposition: BeliefProposition,
        event: BeliefMutationEvidence | None = None,
        *,
        epistemic_status: BeliefEpistemicStatus = BeliefEpistemicStatus.UNKNOWN,
        confidence: float = 0.0,
        context_scope: tuple[str, ...] = (),
        valid_from: datetime | None = None,
        valid_until: datetime | None = None,
        evidence: tuple[BeliefEvidence, ...] = (),
    ) -> BeliefRecord:
        if not isinstance(proposition, BeliefProposition):
            raise TypeError("proposition must be BeliefProposition")
        return self.propose(
            BeliefRecord(
                belief_id=belief_id_for_proposition(proposition.proposition_digest),
                proposition=proposition,
                lifecycle=BeliefLifecycle.PROPOSED,
                epistemic_status=epistemic_status,
                confidence=confidence,
                context_scope=context_scope,
                valid_from=valid_from,
                valid_until=valid_until,
                evidence=evidence,
            ),
            event,
        )

    def propose(
        self, record: BeliefRecord, event: BeliefMutationEvidence | None = None
    ) -> BeliefRecord:
        evidence = self._require_event(event)
        if not isinstance(record, BeliefRecord):
            raise TypeError("record must be BeliefRecord")
        with self._lock:
            self._require_new_identity(record)
            if record.lifecycle is not BeliefLifecycle.PROPOSED:
                raise BeliefDomainError("only proposed Beliefs may be created")
            if record.subject_admission is not None:
                raise BeliefDomainError("proposals cannot carry subject admission")
            if record.revision != 0:
                raise BeliefDomainError("proposals must start at revision zero")
            revision = BeliefRevisionRecord(
                belief_id=record.belief_id,
                revision=0,
                operation=BeliefRevisionOperation.CREATE,
                reason=BeliefRevisionReason.CREATION,
                created_at=evidence.recorded_at,
                event_id=evidence.event_id,
                event_sequence=evidence.event_sequence,
                evidence_refs=_revision_evidence_refs(record),
            )
            created = replace(record, revision_history=(revision,))
            self._replace_records((*self._records, created))
            return created

    @_locked_mutation
    def adopt(
        self,
        belief_id: str,
        admission: BeliefSubjectAdmission,
        event: BeliefMutationEvidence | None = None,
        *,
        epistemic_status: BeliefEpistemicStatus | None = None,
        confidence: float | None = None,
    ) -> BeliefRecord:
        evidence = self._require_event(event)
        if not isinstance(admission, BeliefSubjectAdmission):
            raise TypeError("admission must be BeliefSubjectAdmission")
        current = self._require_record(belief_id)
        if current.lifecycle is not BeliefLifecycle.PROPOSED:
            raise BeliefDomainError("only proposed Beliefs may be adopted")
        admission = self._require_subject_admission(admission, current, evidence)
        revision = self._next_revision(
            current,
            BeliefRevisionOperation.ADOPT,
            BeliefRevisionReason.SUBJECT_ADMISSION,
            evidence,
            admission,
        )
        revision_history, history_anchor_digest = self._append_revision(current, revision)
        adopted = replace(
            current,
            lifecycle=BeliefLifecycle.ADOPTED,
            epistemic_status=(
                current.epistemic_status
                if epistemic_status is None
                else epistemic_status
            ),
            confidence=current.confidence if confidence is None else confidence,
            subject_admission=admission,
            revision=current.revision + 1,
            revision_history=revision_history,
            history_anchor_digest=history_anchor_digest,
        )
        self._replace_one(adopted)
        return adopted

    @_locked_mutation
    def correct(
        self,
        belief_id: str,
        corrected: BeliefRecord,
        event: BeliefMutationEvidence | None = None,
        *,
        admission: BeliefSubjectAdmission | None = None,
    ) -> BeliefRecord:
        evidence = self._require_event(event)
        current = self._require_record(belief_id)
        if not isinstance(corrected, BeliefRecord):
            raise TypeError("corrected must be BeliefRecord")
        if current.lifecycle is not BeliefLifecycle.ADOPTED:
            raise BeliefDomainError("only adopted Beliefs may be corrected")
        if corrected.belief_id != current.belief_id:
            raise BeliefDomainError("correction cannot change Belief identity")
        if corrected.proposition.proposition_digest != current.proposition.proposition_digest:
            raise BeliefDomainError("correction cannot change proposition identity")
        if corrected.lifecycle is not current.lifecycle:
            raise BeliefDomainError("correction cannot change Belief lifecycle")
        if corrected.supersedes_id != current.supersedes_id or corrected.superseded_by_id != current.superseded_by_id:
            raise BeliefDomainError("correction cannot change supersession links")
        subject_admission = self._require_subject_admission(
            admission, corrected, evidence
        )
        revision = self._next_revision(
            current,
            BeliefRevisionOperation.CORRECT,
            BeliefRevisionReason.CORRECTION,
            evidence,
            subject_admission,
        )
        revision_history, history_anchor_digest = self._append_revision(current, revision)
        updated = replace(
            corrected,
            subject_admission=subject_admission,
            revision=current.revision + 1,
            revision_history=revision_history,
            history_anchor_digest=history_anchor_digest,
        )
        self._replace_one(updated)
        return updated

    @_locked_mutation
    def supersede(
        self,
        belief_id: str,
        successor: BeliefRecord,
        event: BeliefMutationEvidence | None = None,
        *,
        predecessor_admission: BeliefSubjectAdmission | None = None,
        successor_admission: BeliefSubjectAdmission | None = None,
    ) -> tuple[BeliefRecord, BeliefRecord]:
        evidence = self._require_event(event)
        current = self._require_record(belief_id)
        if not isinstance(successor, BeliefRecord):
            raise TypeError("successor must be BeliefRecord")
        self._require_new_identity(successor)
        if successor.supersedes_id != current.belief_id:
            raise BeliefConflict("successor must name the superseded Belief")
        if current.lifecycle in {
            BeliefLifecycle.SUPERSEDED,
            BeliefLifecycle.RETRACTED,
            BeliefLifecycle.EXPIRED,
        }:
            raise BeliefConflict("terminal Beliefs cannot be superseded")
        if (
            successor.lifecycle is not BeliefLifecycle.PROPOSED
            or successor.subject_admission is not None
            or successor.revision != 0
            or successor.revision_history
        ):
            raise BeliefDomainError("a successor must begin at revision zero")
        predecessor_admission = self._require_subject_admission(
            predecessor_admission, current, evidence
        )
        successor_admission = self._require_subject_admission(
            successor_admission, successor, evidence
        )
        old_revision = self._next_revision(
            current,
            BeliefRevisionOperation.SUPERSEDE,
            BeliefRevisionReason.SUPERSESSION,
            evidence,
            predecessor_admission,
            (successor_admission,),
        )
        old_history, old_history_anchor_digest = self._append_revision(
            current, old_revision
        )
        successor_revision = BeliefRevisionRecord(
            belief_id=successor.belief_id,
            revision=0,
            operation=BeliefRevisionOperation.CREATE,
            reason=BeliefRevisionReason.CREATION,
            created_at=evidence.recorded_at,
            event_id=evidence.event_id,
            event_sequence=evidence.event_sequence,
            evidence_refs=_revision_evidence_refs(successor),
        )
        successor_adoption_revision = BeliefRevisionRecord(
            belief_id=successor.belief_id,
            revision=1,
            operation=BeliefRevisionOperation.ADOPT,
            reason=BeliefRevisionReason.SUBJECT_ADMISSION,
            created_at=evidence.recorded_at,
            previous_revision_digest=successor_revision.record_digest,
            event_id=evidence.event_id,
            event_sequence=evidence.event_sequence,
            evidence_refs=_revision_evidence_refs(successor, successor_admission),
        )
        superseded = replace(
            current,
            lifecycle=BeliefLifecycle.SUPERSEDED,
            subject_admission=predecessor_admission,
            superseded_by_id=successor.belief_id,
            revision=current.revision + 1,
            revision_history=old_history,
            history_anchor_digest=old_history_anchor_digest,
        )
        created_successor = replace(
            successor,
            lifecycle=BeliefLifecycle.ADOPTED,
            subject_admission=successor_admission,
            revision=1,
            revision_history=(successor_revision, successor_adoption_revision),
        )
        self._replace_records(
            tuple(
                superseded if item.belief_id == superseded.belief_id else item
                for item in self._records
            )
            + (created_successor,)
        )
        return superseded, created_successor

    @_locked_mutation
    def retract(
        self,
        belief_id: str,
        event: BeliefMutationEvidence | None = None,
        *,
        admission: BeliefSubjectAdmission | None = None,
    ) -> BeliefRecord:
        return self._terminalize(
            belief_id,
            BeliefLifecycle.RETRACTED,
            BeliefRevisionOperation.RETRACT,
            BeliefRevisionReason.RETRACTION,
            event,
            admission,
        )

    @_locked_mutation
    def expire(
        self,
        belief_id: str,
        event: BeliefMutationEvidence | None = None,
        *,
        admission: BeliefSubjectAdmission | None = None,
    ) -> BeliefRecord:
        return self._terminalize(
            belief_id,
            BeliefLifecycle.EXPIRED,
            BeliefRevisionOperation.EXPIRE,
            BeliefRevisionReason.EXPIRATION,
            event,
            admission,
        )

    def _terminalize(
        self,
        belief_id: str,
        lifecycle: BeliefLifecycle,
        operation: BeliefRevisionOperation,
        reason: BeliefRevisionReason,
        event: BeliefMutationEvidence | None,
        admission: BeliefSubjectAdmission | None,
    ) -> BeliefRecord:
        evidence = self._require_event(event)
        current = self._require_record(belief_id)
        if current.lifecycle in {
            BeliefLifecycle.SUPERSEDED,
            BeliefLifecycle.RETRACTED,
            BeliefLifecycle.EXPIRED,
        }:
            raise BeliefDomainError("Belief is already terminal")
        subject_admission = self._require_subject_admission(
            admission, current, evidence
        )
        revision = self._next_revision(
            current, operation, reason, evidence, subject_admission
        )
        revision_history, history_anchor_digest = self._append_revision(current, revision)
        updated = replace(
            current,
            lifecycle=lifecycle,
            subject_admission=subject_admission,
            revision=current.revision + 1,
            revision_history=revision_history,
            history_anchor_digest=history_anchor_digest,
        )
        self._replace_one(updated)
        return updated

    def _next_revision(
        self,
        current: BeliefRecord,
        operation: BeliefRevisionOperation,
        reason: BeliefRevisionReason,
        event: BeliefMutationEvidence,
        admission: BeliefSubjectAdmission | None = None,
        additional_admissions: tuple[BeliefSubjectAdmission, ...] = (),
    ) -> BeliefRevisionRecord:
        if not current.revision_history or current.revision_history[-1].revision != current.revision:
            raise BeliefDomainError("Belief history does not include current authority")
        if current.revision >= BELIEF_MAX_REVISION:
            raise BeliefCapacityExceeded("Belief revision bound is exhausted")
        return BeliefRevisionRecord(
            belief_id=current.belief_id,
            revision=current.revision + 1,
            operation=operation,
            reason=reason,
            created_at=event.recorded_at,
            previous_revision_digest=current.revision_history[-1].record_digest,
            event_id=event.event_id,
            event_sequence=event.event_sequence,
            evidence_refs=_revision_evidence_refs(
                current, admission, additional_admissions
            ),
        )

    @staticmethod
    def _append_revision(
        current: BeliefRecord, revision: BeliefRevisionRecord
    ) -> tuple[tuple[BeliefRevisionRecord, ...], str | None]:
        """Append one revision while retaining a cryptographic history anchor."""

        history = current.revision_history + (revision,)
        anchor = current.history_anchor_digest
        if len(history) > BELIEF_MAX_REVISIONS:
            anchor = history[0].record_digest
            history = history[1:]
        return history, anchor

    def _require_event(
        self, supplied: BeliefMutationEvidence | None
    ) -> BeliefMutationEvidence:
        if self._event_provider is None:
            raise BeliefDomainError("Belief mutation requires an active runtime event")
        try:
            active_event = self._event_provider()
        except Exception:
            raise BeliefDomainError("Belief mutation event is unavailable") from None
        if active_event is None:
            raise BeliefDomainError("Belief mutation requires an active runtime event")
        try:
            from suzka.runtime.agent_runtime import AgentEvent

            if not isinstance(active_event, AgentEvent):
                raise BeliefDomainError("Belief event context is not an AgentRuntime event")
            if isinstance(active_event, BeliefMutationEvidence):
                raise BeliefDomainError("Belief event context is not an AgentRuntime event")
            if not hasattr(active_event, "event_type") or not hasattr(active_event, "source"):
                raise BeliefDomainError("Belief event context is not an AgentRuntime event")
            evidence = BeliefMutationEvidence.from_event(active_event)
        except (TypeError, ValueError, BeliefDomainError):
            raise BeliefDomainError("Belief mutation event is invalid") from None
        if supplied is not None:
            if not isinstance(supplied, BeliefMutationEvidence):
                raise TypeError("event must be BeliefMutationEvidence")
            if supplied != evidence:
                raise BeliefDomainError(
                    "caller-supplied Belief event does not match the active runtime event"
                )
        return evidence

    @staticmethod
    def _require_subject_admission(
        admission: BeliefSubjectAdmission | None,
        target: BeliefRecord,
        event: BeliefMutationEvidence,
    ) -> BeliefSubjectAdmission:
        if not isinstance(admission, BeliefSubjectAdmission):
            raise BeliefDomainError("Belief mutation requires subject admission")
        if (
            admission.event_id != event.event_id
            or admission.event_sequence != event.event_sequence
        ):
            raise BeliefDomainError("subject admission is not bound to the active event")
        if admission.proposition_digest != target.proposition.proposition_digest:
            raise BeliefDomainError("subject admission proposition mismatch")
        if tuple(item.evidence_ref for item in target.evidence) != admission.evidence_refs:
            raise BeliefDomainError("subject admission evidence does not match")
        return admission

    def _require_record(self, belief_id: str) -> BeliefRecord:
        record = self.get(belief_id)
        if record is None:
            raise BeliefDomainError("Belief is not present")
        return record

    @staticmethod
    def _require_new_identity(record: BeliefRecord) -> None:
        expected = belief_id_for_proposition(record.proposition.proposition_digest)
        if record.belief_id != expected:
            raise BeliefDomainError("Belief identity is not proposition-derived")

    def _replace_one(self, record: BeliefRecord) -> None:
        self._replace_records(
            tuple(record if item.belief_id == record.belief_id else item for item in self._records)
        )

    def _replace_records(self, records: tuple[BeliefRecord, ...]) -> None:
        canonical = tuple(sorted(records, key=lambda record: record.belief_id))
        self._validate_records(canonical)
        if len(canonical_belief_system_bytes(canonical)) > BELIEF_MAX_SERIALIZED_BYTES:
            raise BeliefCapacityExceeded("Belief snapshot exceeds its serialized byte bound")
        self._records = canonical
        self._record_map = {record.belief_id: record for record in canonical}

    @classmethod
    def _validate_records(cls, records: tuple[BeliefRecord, ...]) -> None:
        if len(records) > BELIEF_MAX_RECORDS:
            raise BeliefCapacityExceeded("Belief authority exceeds its record bound")
        ids = tuple(record.belief_id for record in records)
        if ids != tuple(sorted(set(ids))):
            raise BeliefDomainError("Belief records must be sorted and unique")
        record_map = {record.belief_id: record for record in records}
        for record in records:
            cls._require_record_shape(record, record_map)
        for record in records:
            if record.supersedes_id is not None:
                target = record_map.get(record.supersedes_id)
                if target is None or target.superseded_by_id != record.belief_id:
                    raise BeliefConflict("supersession links are not reciprocal")
            if record.superseded_by_id is not None:
                successor = record_map.get(record.superseded_by_id)
                if successor is None or successor.supersedes_id != record.belief_id:
                    raise BeliefConflict("supersession links are not reciprocal")
                successor_admission = successor.subject_admission
                if (
                    successor_admission is None
                    or successor_admission.admission_digest
                    not in record.revision_history[-1].evidence_refs
                ):
                    raise BeliefConflict(
                        "supersession revision is missing successor admission proof"
                    )
                latest = record.revision_history[-1]
                if (
                    successor_admission.event_id != latest.event_id
                    or successor_admission.event_sequence != latest.event_sequence
                ):
                    raise BeliefConflict(
                        "supersession admissions must share the SUPERSEDE event"
                    )
        cls._validate_supersession_graph(records, record_map)

    @staticmethod
    def _validate_supersession_graph(
        records: tuple[BeliefRecord, ...], record_map: dict[str, BeliefRecord]
    ) -> None:
        """Reject reciprocal supersession cycles during every authority load."""

        state: dict[str, int] = {record.belief_id: 0 for record in records}
        for root in sorted(state):
            if state[root] == 2:
                continue
            current: str | None = root
            path: list[str] = []
            while current is not None:
                color = state[current]
                if color == 1:
                    raise BeliefConflict("supersession graph contains a cycle")
                if color == 2:
                    break
                state[current] = 1
                path.append(current)
                successor_id = record_map[current].superseded_by_id
                if successor_id is not None and successor_id not in record_map:
                    raise BeliefConflict("supersession successor is missing")
                current = successor_id
            for belief_id in path:
                state[belief_id] = 2

    @classmethod
    def _require_record_shape(
        cls, record: BeliefRecord, record_map: dict[str, BeliefRecord]
    ) -> None:
        if not isinstance(record, BeliefRecord):
            raise TypeError("records must contain BeliefRecord values")
        expected = belief_id_for_proposition(record.proposition.proposition_digest)
        if record.belief_id != expected:
            raise BeliefDomainError("Belief identity is not proposition-derived")
        if not record.revision_history or record.revision_history[-1].revision != record.revision:
            raise BeliefDomainError("Belief history must end at the current revision")
        if len(record.revision_history) > BELIEF_MAX_REVISIONS:
            raise BeliefCapacityExceeded("Belief revision history exceeds its bound")
        for revision in record.revision_history:
            if revision.event_id is None or revision.event_sequence is None:
                raise BeliefDomainError("authoritative revisions require event evidence")
        cls._validate_revision_history(record)
        latest_operation = record.revision_history[-1].operation
        expected_operation = {
            BeliefLifecycle.PROPOSED: BeliefRevisionOperation.CREATE,
            BeliefLifecycle.ADOPTED: {
                BeliefRevisionOperation.ADOPT,
                BeliefRevisionOperation.CORRECT,
            },
            BeliefLifecycle.SUPERSEDED: BeliefRevisionOperation.SUPERSEDE,
            BeliefLifecycle.RETRACTED: BeliefRevisionOperation.RETRACT,
            BeliefLifecycle.EXPIRED: BeliefRevisionOperation.EXPIRE,
        }[record.lifecycle]
        if isinstance(expected_operation, set):
            valid_operation = latest_operation in expected_operation
        else:
            valid_operation = latest_operation is expected_operation
        if not valid_operation:
            raise BeliefDomainError("Belief lifecycle does not match its latest revision")
        if record.lifecycle is BeliefLifecycle.PROPOSED and record.subject_admission is not None:
            raise BeliefDomainError("proposals cannot carry subject admission")
        if record.lifecycle in {
            BeliefLifecycle.ADOPTED,
            BeliefLifecycle.RETRACTED,
            BeliefLifecycle.EXPIRED,
        }:
            admission = record.subject_admission
            latest = record.revision_history[-1]
            if admission is None:
                raise BeliefDomainError("terminal or adopted Beliefs require subject admission")
            if (
                admission.event_id != latest.event_id
                or admission.event_sequence != latest.event_sequence
                or admission.admission_digest not in latest.evidence_refs
            ):
                raise BeliefDomainError(
                    "subject admission does not match the latest Belief revision"
                )
        if record.lifecycle is BeliefLifecycle.SUPERSEDED:
            admission = record.subject_admission
            latest = record.revision_history[-1]
            if admission is None:
                raise BeliefConflict(
                    "superseded Beliefs require predecessor admission proof"
                )
            if (
                admission.event_id != latest.event_id
                or admission.event_sequence != latest.event_sequence
                or admission.admission_digest not in latest.evidence_refs
            ):
                raise BeliefConflict(
                    "supersession revision is missing predecessor admission proof"
                )
        if record.supersedes_id == record.belief_id or record.superseded_by_id == record.belief_id:
            raise BeliefConflict("a Belief cannot supersede itself")
        if record.lifecycle is BeliefLifecycle.SUPERSEDED and record.superseded_by_id is None:
            raise BeliefConflict("superseded Beliefs require a successor")
        del record_map

    @staticmethod
    def _validate_revision_history(record: BeliefRecord) -> None:
        """Validate reachable lifecycle transitions in the retained history."""

        reasons = {
            BeliefRevisionOperation.CREATE: BeliefRevisionReason.CREATION,
            BeliefRevisionOperation.ADOPT: BeliefRevisionReason.SUBJECT_ADMISSION,
            BeliefRevisionOperation.CORRECT: BeliefRevisionReason.CORRECTION,
            BeliefRevisionOperation.SUPERSEDE: BeliefRevisionReason.SUPERSESSION,
            BeliefRevisionOperation.RETRACT: BeliefRevisionReason.RETRACTION,
            BeliefRevisionOperation.EXPIRE: BeliefRevisionReason.EXPIRATION,
        }
        history = record.revision_history
        first = history[0]
        if first.revision == 0 and first.operation is not BeliefRevisionOperation.CREATE:
            raise BeliefDomainError("Belief history must begin with creation")
        if first.revision > 0 and first.operation is BeliefRevisionOperation.CREATE:
            raise BeliefDomainError("compacted Belief history cannot recreate a record")

        state: BeliefLifecycle | None = None
        previous: BeliefRevisionRecord | None = None
        for revision in history:
            expected_reason = reasons[revision.operation]
            if revision.reason is not expected_reason:
                raise BeliefDomainError("Belief revision operation and reason disagree")
            if previous is not None:
                if (
                    revision.event_sequence is None
                    or previous.event_sequence is None
                    or revision.event_sequence < previous.event_sequence
                ):
                    raise BeliefDomainError(
                        "Belief revision events must increase monotonically"
                    )
                if revision.event_sequence == previous.event_sequence and not (
                    previous.operation is BeliefRevisionOperation.CREATE
                    and revision.operation is BeliefRevisionOperation.ADOPT
                    and revision.event_id == previous.event_id
                ):
                    raise BeliefDomainError(
                        "only successor creation and adoption may share an event"
                    )
            if state is None:
                if revision.operation is BeliefRevisionOperation.CREATE:
                    state = BeliefLifecycle.PROPOSED
                elif revision.operation in {
                    BeliefRevisionOperation.ADOPT,
                    BeliefRevisionOperation.CORRECT,
                }:
                    state = BeliefLifecycle.ADOPTED
                elif revision.operation is BeliefRevisionOperation.SUPERSEDE:
                    state = BeliefLifecycle.SUPERSEDED
                elif revision.operation is BeliefRevisionOperation.RETRACT:
                    state = BeliefLifecycle.RETRACTED
                else:
                    state = BeliefLifecycle.EXPIRED
            else:
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
                next_state = transitions[state].get(revision.operation)
                if next_state is None:
                    raise BeliefDomainError(
                        "Belief revision history contains an invalid lifecycle transition"
                    )
                state = next_state
            previous = revision

        if state is not record.lifecycle:
            raise BeliefDomainError("Belief history does not reach its lifecycle")


__all__ = [
    "BELIEF_ID_DOMAIN",
    "BELIEF_MAX_RECORDS",
    "BELIEF_SYSTEM_DOMAIN",
    "BELIEF_SYSTEM_SCHEMA_VERSION",
    "BeliefCapacityExceeded",
    "BeliefConflict",
    "BeliefDomainError",
    "BeliefMutationEvidence",
    "BeliefSystem",
    "BeliefSystemSnapshot",
    "belief_id_for_proposition",
]
