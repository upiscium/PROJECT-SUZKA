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
    BELIEF_MAX_EVIDENCE,
    BELIEF_MAX_REVISIONS,
    BELIEF_MAX_REVISION,
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
        object.__setattr__(self, "authority_digest", _canonical_system_digest(self.records))


def _revision_evidence_refs(
    record: BeliefRecord,
    admission: BeliefSubjectAdmission | None = None,
) -> tuple[str, ...]:
    refs = {evidence.evidence_ref for evidence in record.evidence}
    if admission is not None:
        refs.add(admission.admission_digest)
    if len(refs) > BELIEF_MAX_EVIDENCE:
        raise BeliefCapacityExceeded("Belief revision evidence exceeds its bound")
    return tuple(sorted(refs))


class BeliefSystem:
    """Serialized intrinsic Belief authority with no automatic producer."""

    MAX_RECORDS: Final = BELIEF_MAX_RECORDS

    def __init__(self, records: Iterable[BeliefRecord] = ()) -> None:
        self._lock = RLock()
        canonical = tuple(sorted(tuple(records), key=lambda record: record.belief_id))
        self._replace_records(canonical)

    @classmethod
    def restore_snapshot(cls, snapshot: BeliefSystemSnapshot) -> BeliefSystem:
        if not isinstance(snapshot, BeliefSystemSnapshot):
            raise TypeError("snapshot must be BeliefSystemSnapshot")
        system = cls(snapshot.records)
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
        event: BeliefMutationEvidence,
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
        self, record: BeliefRecord, event: BeliefMutationEvidence
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
        event: BeliefMutationEvidence,
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
        if admission.proposition_digest != current.proposition.proposition_digest:
            raise BeliefDomainError("subject admission proposition mismatch")
        if tuple(item.evidence_ref for item in current.evidence) != admission.evidence_refs:
            raise BeliefDomainError("subject admission evidence does not match")
        if (
            admission.event_id != evidence.event_id
            or admission.event_sequence != evidence.event_sequence
        ):
            raise BeliefDomainError("subject admission is not bound to the active event")
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
        event: BeliefMutationEvidence,
    ) -> BeliefRecord:
        evidence = self._require_event(event)
        current = self._require_record(belief_id)
        if not isinstance(corrected, BeliefRecord):
            raise TypeError("corrected must be BeliefRecord")
        if corrected.belief_id != current.belief_id:
            raise BeliefDomainError("correction cannot change Belief identity")
        if corrected.proposition.proposition_digest != current.proposition.proposition_digest:
            raise BeliefDomainError("correction cannot change proposition identity")
        if corrected.lifecycle is not current.lifecycle:
            raise BeliefDomainError("correction cannot change Belief lifecycle")
        if corrected.subject_admission != current.subject_admission:
            raise BeliefDomainError("correction cannot change subject admission")
        if corrected.supersedes_id != current.supersedes_id or corrected.superseded_by_id != current.superseded_by_id:
            raise BeliefDomainError("correction cannot change supersession links")
        revision = self._next_revision(
            current,
            BeliefRevisionOperation.CORRECT,
            BeliefRevisionReason.CORRECTION,
            evidence,
        )
        revision_history, history_anchor_digest = self._append_revision(current, revision)
        updated = replace(
            corrected,
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
        event: BeliefMutationEvidence,
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
        if successor.revision != 0 or successor.revision_history:
            raise BeliefDomainError("a successor must begin at revision zero")
        if successor.lifecycle is BeliefLifecycle.ADOPTED:
            admission = successor.subject_admission
            if admission is None or (
                admission.event_id != evidence.event_id
                or admission.event_sequence != evidence.event_sequence
            ):
                raise BeliefDomainError(
                    "an adopted successor must bind admission to the active event"
                )
        old_revision = self._next_revision(
            current,
            BeliefRevisionOperation.SUPERSEDE,
            BeliefRevisionReason.SUPERSESSION,
            evidence,
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
            evidence_refs=_revision_evidence_refs(successor, successor.subject_admission),
        )
        superseded = replace(
            current,
            lifecycle=BeliefLifecycle.SUPERSEDED,
            superseded_by_id=successor.belief_id,
            revision=current.revision + 1,
            revision_history=old_history,
            history_anchor_digest=old_history_anchor_digest,
        )
        created_successor = replace(successor, revision_history=(successor_revision,))
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
        self, belief_id: str, event: BeliefMutationEvidence
    ) -> BeliefRecord:
        return self._terminalize(
            belief_id,
            BeliefLifecycle.RETRACTED,
            BeliefRevisionOperation.RETRACT,
            BeliefRevisionReason.RETRACTION,
            event,
        )

    @_locked_mutation
    def expire(self, belief_id: str, event: BeliefMutationEvidence) -> BeliefRecord:
        return self._terminalize(
            belief_id,
            BeliefLifecycle.EXPIRED,
            BeliefRevisionOperation.EXPIRE,
            BeliefRevisionReason.EXPIRATION,
            event,
        )

    def _terminalize(
        self,
        belief_id: str,
        lifecycle: BeliefLifecycle,
        operation: BeliefRevisionOperation,
        reason: BeliefRevisionReason,
        event: BeliefMutationEvidence,
    ) -> BeliefRecord:
        evidence = self._require_event(event)
        current = self._require_record(belief_id)
        if current.lifecycle in {
            BeliefLifecycle.SUPERSEDED,
            BeliefLifecycle.RETRACTED,
            BeliefLifecycle.EXPIRED,
        }:
            raise BeliefDomainError("Belief is already terminal")
        revision = self._next_revision(current, operation, reason, evidence)
        revision_history, history_anchor_digest = self._append_revision(current, revision)
        updated = replace(
            current,
            lifecycle=lifecycle,
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
            evidence_refs=_revision_evidence_refs(current, admission),
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

    @staticmethod
    def _require_event(event: BeliefMutationEvidence) -> BeliefMutationEvidence:
        if not isinstance(event, BeliefMutationEvidence):
            raise TypeError("event must be BeliefMutationEvidence")
        return event

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

    @staticmethod
    def _require_record_shape(
        record: BeliefRecord, record_map: dict[str, BeliefRecord]
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
        if record.supersedes_id == record.belief_id or record.superseded_by_id == record.belief_id:
            raise BeliefConflict("a Belief cannot supersede itself")
        if record.lifecycle is BeliefLifecycle.SUPERSEDED and record.superseded_by_id is None:
            raise BeliefConflict("superseded Beliefs require a successor")
        del record_map


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
