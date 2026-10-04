"""Bounded process-local Goal lifecycle authority for R13 U3.

The immutable U1 Goal values remain the schema and proof contract.  This
system accepts exact proposals and caller-supplied subject proofs, validates a
complete replacement snapshot, and then publishes one bounded state change.
It has no runtime, model, persistence, scheduler, outcome-verification, or
Commitment dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
import json
from threading import RLock
from typing import Final

from suzka.identifiers import validate_identifier
from suzka.motivation.common import (
    R13_MAX_EVENT_SEQUENCE,
    R13_MAX_EVIDENCE_REFS,
    R13_MAX_RECORDS_PER_DOMAIN,
    R13_MAX_REVISION,
    R13_MAX_REVISION_HISTORY,
    R13Reference,
    RevisionCompactionAnchor,
    canonical_datetime,
    canonical_json,
    canonical_references,
    digest_payload,
    utc_datetime,
    validate_digest,
)
from suzka.motivation.goal import (
    GOAL_EVIDENCE_KINDS,
    GOAL_DOMAIN,
    GoalLifecycle,
    GoalRecord,
    GoalRevisionOperation,
    GoalRevisionReason,
    GoalRevisionRecord,
    GoalSubjectAdmission,
    GoalSubjectTransitionProof,
    canonical_goal_payload,
    goal_record_digest,
    validate_goal_reference_graph,
)


GOAL_SYSTEM_SCHEMA_VERSION: Final = 1
GOAL_SYSTEM_DOMAIN: Final = b"PROJECT-SUZKA:R13:GOAL-SYSTEM:V1\0"
GOAL_SYSTEM_EVENT_DOMAIN: Final = b"PROJECT-SUZKA:R13:GOAL-SYSTEM-EVENT:V1\0"
GOAL_SYSTEM_MAX_RECORDS: Final = R13_MAX_RECORDS_PER_DOMAIN
GOAL_SYSTEM_MAX_EVENT_RECEIPTS: Final = 1_024
GOAL_SYSTEM_MAX_SERIALIZED_BYTES: Final = 32 * 1024 * 1024
GOAL_SYSTEM_MAX_EVIDENCE_REFS_PER_EVENT: Final = R13_MAX_EVIDENCE_REFS

class GoalSystemError(ValueError):
    """Base class for bounded Goal authority failures."""


class GoalSystemCapacityExceeded(GoalSystemError):
    """A Goal record, receipt, revision, or snapshot bound would be exceeded."""


class GoalSystemConflict(GoalSystemError):
    """A Goal identity or event was reused with conflicting contract data."""


class GoalSystemEventOperation(str, Enum):
    INGEST_PROPOSAL = "ingest_proposal"
    ADOPT = "adopt"
    DEFER = "defer"
    RESUME = "resume"
    ABANDON = "abandon"


def _exact_enum(value: object, enum_type: type[Enum], name: str) -> None:
    if type(value) is not enum_type:
        raise TypeError(f"{name} must be a {enum_type.__name__}")


def _goal_revision_value(revision: GoalRevisionRecord) -> dict[str, object]:
    return {
        "created_at": canonical_datetime(revision.created_at),
        "event_id": revision.event_id,
        "event_sequence": revision.event_sequence,
        "evidence_refs": list(revision.evidence_refs),
        "goal_id": revision.goal_id,
        "operation": revision.operation.value,
        "previous_lifecycle_state": (
            None
            if revision.previous_lifecycle_state is None
            else revision.previous_lifecycle_state.value
        ),
        "previous_revision_digest": revision.previous_revision_digest,
        "proposal_digest": revision.proposal_digest,
        "reason": revision.reason.value,
        "record_digest": revision.record_digest,
        "revision": revision.revision,
    }


def _goal_event_input_digest(
    *,
    operation: GoalSystemEventOperation,
    goal_id: str,
    proposal_digest: str,
    evidence_refs: tuple[R13Reference, ...],
    proposal_record_digest: str | None = None,
    admission_digest: str | None = None,
    transition_digest: str | None = None,
) -> str:
    return digest_payload(
        GOAL_SYSTEM_EVENT_DOMAIN,
        {
            "admission_digest": admission_digest,
            "evidence_refs": [item.canonical_value() for item in evidence_refs],
            "goal_id": goal_id,
            "operation": operation.value,
            "proposal_digest": proposal_digest,
            "proposal_record_digest": proposal_record_digest,
            "transition_digest": transition_digest,
        },
    )


@dataclass(frozen=True, slots=True)
class GoalMutationEvidence:
    """Explicit canonical event and typed evidence contract for Goal mutation."""

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
            canonical_references(
                self.evidence_refs,
                "evidence_refs",
                maximum=GOAL_SYSTEM_MAX_EVIDENCE_REFS_PER_EVENT,
                allow_empty=False,
                allowed_kinds=GOAL_EVIDENCE_KINDS,
            ),
        )

    def canonical_value(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "event_sequence": self.event_sequence,
            "recorded_at": canonical_datetime(self.recorded_at),
            "evidence_refs": [item.canonical_value() for item in self.evidence_refs],
        }


@dataclass(frozen=True, slots=True)
class GoalSystemEventReceipt:
    """Exact bounded witness for one GoalSystem mutation or proposal ingestion."""

    event_id: str
    event_sequence: int
    recorded_at: datetime
    operation: GoalSystemEventOperation
    goal_id: str
    proposal_digest: str
    input_digest: str
    evidence_refs: tuple[R13Reference, ...]
    proposal_record_digest: str | None = None
    proposal_genesis: GoalRevisionRecord | None = None
    revision_witness: GoalRevisionRecord | None = None
    admission_digest: str | None = None
    transition_digest: str | None = None
    receipt_digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", validate_identifier(self.event_id))
        if (
            type(self.event_sequence) is not int
            or not 1 <= self.event_sequence <= R13_MAX_EVENT_SEQUENCE
        ):
            raise ValueError("event_sequence must be a positive bounded integer")
        object.__setattr__(self, "recorded_at", utc_datetime(self.recorded_at, "recorded_at"))
        _exact_enum(self.operation, GoalSystemEventOperation, "operation")
        goal_id = validate_identifier(self.goal_id)
        proposal_digest = validate_digest(self.proposal_digest, "proposal_digest")
        input_digest = validate_digest(self.input_digest, "input_digest")
        evidence_refs = canonical_references(
            self.evidence_refs,
            "evidence_refs",
            maximum=GOAL_SYSTEM_MAX_EVIDENCE_REFS_PER_EVENT,
            allow_empty=False,
            allowed_kinds=GOAL_EVIDENCE_KINDS,
        )
        for name in (
            "proposal_record_digest",
            "admission_digest",
            "transition_digest",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, validate_digest(value, name))
        if self.operation is GoalSystemEventOperation.INGEST_PROPOSAL:
            genesis = (
                self.proposal_genesis
                if isinstance(self.proposal_genesis, GoalRevisionRecord)
                else None
            )
            genesis_sequence = None if genesis is None else genesis.event_sequence
            genesis_event_id = None if genesis is None else genesis.event_id
            if (
                self.proposal_record_digest is None
                or genesis is None
                or self.revision_witness is not None
                or self.admission_digest is not None
                or self.transition_digest is not None
            ):
                raise ValueError("proposal receipt requires its exact genesis witness")
            if (
                genesis.goal_id != goal_id
                or genesis.proposal_digest != proposal_digest
                or genesis.revision != 0
                or genesis.operation is not GoalRevisionOperation.CREATE
                or genesis.reason is not GoalRevisionReason.CREATION
                or genesis_sequence is None
                or genesis_event_id is None
                or genesis_sequence > self.event_sequence
                or genesis.created_at > self.recorded_at
                or (
                    (
                        genesis_event_id == self.event_id
                        or genesis_sequence == self.event_sequence
                    )
                    and (
                        genesis_event_id != self.event_id
                        or genesis_sequence != self.event_sequence
                        or genesis.created_at != self.recorded_at
                    )
                )
            ):
                raise ValueError("proposal receipt genesis witness is inconsistent")
        elif self.operation is GoalSystemEventOperation.ADOPT:
            revision = (
                self.revision_witness
                if isinstance(self.revision_witness, GoalRevisionRecord)
                else None
            )
            if (
                self.proposal_record_digest is not None
                or self.proposal_genesis is not None
                or revision is None
                or self.admission_digest is None
                or self.transition_digest is not None
            ):
                raise ValueError("admission receipt requires its exact revision and proof")
            expected_witnesses = tuple(
                sorted(
                    (*[item.reference for item in evidence_refs], self.admission_digest)
                )
            )
            if (
                revision.goal_id != goal_id
                or revision.proposal_digest != proposal_digest
                or revision.operation is not GoalRevisionOperation.ADOPT
                or revision.reason is not GoalRevisionReason.SUBJECT_ADMISSION
                or revision.previous_lifecycle_state is not GoalLifecycle.PROPOSED
                or revision.event_id != self.event_id
                or revision.event_sequence != self.event_sequence
                or revision.created_at != self.recorded_at
                or revision.evidence_refs != expected_witnesses
            ):
                raise ValueError("admission receipt revision does not match its exact proof")
        else:
            revision = (
                self.revision_witness
                if isinstance(self.revision_witness, GoalRevisionRecord)
                else None
            )
            if (
                self.proposal_record_digest is not None
                or self.proposal_genesis is not None
                or revision is None
                or self.admission_digest is not None
                or self.transition_digest is None
            ):
                raise ValueError("subject transition receipt requires its exact revision/proof")
            operation_and_reason = {
                GoalSystemEventOperation.DEFER: (
                    GoalRevisionOperation.DEFER,
                    GoalRevisionReason.SUBJECT_DEFERRED,
                ),
                GoalSystemEventOperation.RESUME: (
                    GoalRevisionOperation.ADOPT,
                    GoalRevisionReason.SUBJECT_ADMISSION,
                ),
                GoalSystemEventOperation.ABANDON: (
                    GoalRevisionOperation.ABANDON,
                    GoalRevisionReason.SUBJECT_ABANDONED,
                ),
            }[self.operation]
            expected_witnesses = tuple(
                sorted(
                    (*[item.reference for item in evidence_refs], self.transition_digest)
                )
            )
            if (
                revision.goal_id != goal_id
                or revision.proposal_digest != proposal_digest
                or (revision.operation, revision.reason) != operation_and_reason
                or revision.event_id != self.event_id
                or revision.event_sequence != self.event_sequence
                or revision.created_at != self.recorded_at
                or revision.evidence_refs != expected_witnesses
            ):
                raise ValueError("subject transition receipt revision is inconsistent")
            previous_lifecycle = revision.previous_lifecycle_state
            if previous_lifecycle is None:
                raise ValueError("subject transition receipt lacks its prior lifecycle")
            try:
                proof = GoalSubjectTransitionProof(
                    goal_id=revision.goal_id,
                    proposal_digest=revision.proposal_digest,
                    operation=revision.operation,
                    reason=revision.reason,
                    previous_lifecycle_state=previous_lifecycle,
                    evidence_refs=tuple(item.reference for item in evidence_refs),
                    event_id=self.event_id,
                    event_sequence=self.event_sequence,
                )
            except (TypeError, ValueError) as error:
                raise ValueError("subject transition receipt contains an invalid proof") from error
            if proof.transition_digest != self.transition_digest:
                raise ValueError("subject transition receipt digest does not bind its proof")
        object.__setattr__(self, "goal_id", goal_id)
        object.__setattr__(self, "proposal_digest", proposal_digest)
        object.__setattr__(self, "input_digest", input_digest)
        object.__setattr__(self, "evidence_refs", evidence_refs)
        expected = _goal_event_input_digest(
            operation=self.operation,
            goal_id=goal_id,
            proposal_digest=proposal_digest,
            evidence_refs=evidence_refs,
            proposal_record_digest=self.proposal_record_digest,
            admission_digest=self.admission_digest,
            transition_digest=self.transition_digest,
        )
        if input_digest != expected:
            raise ValueError("GoalSystem event input digest is inconsistent")
        object.__setattr__(
            self,
            "receipt_digest",
            digest_payload(GOAL_SYSTEM_EVENT_DOMAIN, self.canonical_value()),
        )

    def canonical_value(self) -> dict[str, object]:
        return {
            "admission_digest": self.admission_digest,
            "event_id": self.event_id,
            "event_sequence": self.event_sequence,
            "evidence_refs": [item.canonical_value() for item in self.evidence_refs],
            "goal_id": self.goal_id,
            "input_digest": self.input_digest,
            "operation": self.operation.value,
            "proposal_genesis": (
                None
                if self.proposal_genesis is None
                else _goal_revision_value(self.proposal_genesis)
            ),
            "proposal_digest": self.proposal_digest,
            "proposal_record_digest": self.proposal_record_digest,
            "recorded_at": canonical_datetime(self.recorded_at),
            "revision_witness": (
                None
                if self.revision_witness is None
                else _goal_revision_value(self.revision_witness)
            ),
            "transition_digest": self.transition_digest,
        }


def _goal_revision_after(
    revision: GoalRevisionRecord,
) -> GoalLifecycle:
    if revision.operation is GoalRevisionOperation.CREATE:
        return GoalLifecycle.PROPOSED
    if revision.operation is GoalRevisionOperation.ADOPT:
        return GoalLifecycle.ADOPTED
    if revision.operation is GoalRevisionOperation.DEFER:
        return GoalLifecycle.DEFERRED
    if revision.operation is GoalRevisionOperation.ABANDON:
        return GoalLifecycle.ABANDONED
    if revision.operation is GoalRevisionOperation.COMPLETE:
        return GoalLifecycle.COMPLETED
    return GoalLifecycle.FAILED


def _append_goal_revision(
    current: GoalRecord,
    revision: GoalRevisionRecord,
) -> tuple[tuple[GoalRevisionRecord, ...], RevisionCompactionAnchor | None]:
    history = current.revision_history + (revision,)
    anchor = current.history_anchor
    if len(history) > R13_MAX_REVISION_HISTORY:
        dropped = history[0]
        if dropped.event_id is None or dropped.event_sequence is None:
            raise GoalSystemError("cannot compact a Goal revision without its event")
        anchor = RevisionCompactionAnchor(
            authority_id=dropped.goal_id,
            through_revision=dropped.revision,
            through_digest=dropped.record_digest,
            through_created_at=dropped.created_at,
            through_evidence_refs=dropped.evidence_refs,
            through_previous_revision_digest=dropped.previous_revision_digest,
            through_state=_goal_revision_after(dropped).value,
            through_previous_state=(
                None
                if dropped.previous_lifecycle_state is None
                else dropped.previous_lifecycle_state.value
            ),
            through_operation=dropped.operation.value,
            through_reason=dropped.reason.value,
            through_event_id=dropped.event_id,
            through_event_sequence=dropped.event_sequence,
            through_proposal_digest=dropped.proposal_digest,
        )
        history = history[1:]
    return history, anchor


def _goal_system_record_value(record: GoalRecord) -> dict[str, object]:
    if goal_record_digest(record) != record.record_digest:
        raise GoalSystemError("Goal record digest is inconsistent")
    payload = canonical_goal_payload(record)
    return json.loads(payload[len(GOAL_DOMAIN) :])


def _system_operation_for_revision(
    revision: GoalRevisionRecord,
) -> GoalSystemEventOperation:
    if revision.operation is GoalRevisionOperation.ADOPT:
        if revision.previous_lifecycle_state is GoalLifecycle.PROPOSED:
            return GoalSystemEventOperation.ADOPT
        if revision.previous_lifecycle_state is GoalLifecycle.DEFERRED:
            return GoalSystemEventOperation.RESUME
        raise ValueError("Goal adoption revision has an invalid prior lifecycle")
    if revision.operation is GoalRevisionOperation.DEFER:
        return GoalSystemEventOperation.DEFER
    if revision.operation is GoalRevisionOperation.ABANDON:
        return GoalSystemEventOperation.ABANDON
    raise ValueError("GoalSystem cannot represent a terminal outcome revision")


def _validate_revision_receipt(
    revision: GoalRevisionRecord,
    receipts: tuple[GoalSystemEventReceipt, ...],
) -> None:
    operation = _system_operation_for_revision(revision)
    matches = tuple(
        receipt
        for receipt in receipts
        if receipt.event_id == revision.event_id
        and receipt.event_sequence == revision.event_sequence
    )
    if len(matches) != 1:
        raise ValueError("Goal revision must have exactly one matching system receipt")
    receipt = matches[0]
    proof_digest = (
        receipt.admission_digest
        if operation is GoalSystemEventOperation.ADOPT
        else receipt.transition_digest
    )
    if proof_digest is None:
        raise ValueError("Goal revision receipt lacks its exact subject proof")
    expected_revision_evidence = tuple(
        sorted((*[item.reference for item in receipt.evidence_refs], proof_digest))
    )
    if (
        receipt.operation is not operation
        or receipt.goal_id != revision.goal_id
        or receipt.proposal_digest != revision.proposal_digest
        or receipt.recorded_at != revision.created_at
        or receipt.revision_witness != revision
        or revision.evidence_refs != expected_revision_evidence
    ):
        raise ValueError("Goal revision does not match its exact system receipt")


@dataclass(frozen=True, slots=True)
class GoalSystemSnapshot:
    """Bounded exact Goal authority projection, not a persistence artifact."""

    records: tuple[GoalRecord, ...] = ()
    event_receipts: tuple[GoalSystemEventReceipt, ...] = ()
    schema_version: int = GOAL_SYSTEM_SCHEMA_VERSION
    authority_digest: str = field(init=False)
    serialized_bytes: int = field(init=False)

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != GOAL_SYSTEM_SCHEMA_VERSION:
            raise ValueError("unsupported GoalSystem snapshot version")
        if type(self.records) is not tuple or any(
            not isinstance(item, GoalRecord) for item in self.records
        ):
            raise TypeError("records must contain GoalRecord values")
        if len(self.records) > GOAL_SYSTEM_MAX_RECORDS:
            raise GoalSystemCapacityExceeded("Goal record capacity is exhausted")
        if type(self.event_receipts) is not tuple or any(
            not isinstance(item, GoalSystemEventReceipt) for item in self.event_receipts
        ):
            raise TypeError("event_receipts must contain GoalSystemEventReceipt values")
        if len(self.event_receipts) > GOAL_SYSTEM_MAX_EVENT_RECEIPTS:
            raise GoalSystemCapacityExceeded("Goal event receipt capacity is exhausted")
        goal_ids = tuple(item.goal_id for item in self.records)
        if goal_ids != tuple(sorted(set(goal_ids))):
            raise ValueError("Goal records must be sorted and unique")
        if any(
            item.lifecycle in {GoalLifecycle.COMPLETED, GoalLifecycle.FAILED}
            for item in self.records
        ):
            raise GoalSystemError("GoalSystem cannot produce terminal outcomes")
        try:
            validate_goal_reference_graph(self.records)
        except (TypeError, ValueError) as error:
            raise GoalSystemError("Goal graph is invalid") from error
        event_sequences = tuple(item.event_sequence for item in self.event_receipts)
        if event_sequences != tuple(sorted(set(event_sequences))):
            raise ValueError("GoalSystem event receipts must be sequence ordered")
        event_ids = tuple(item.event_id for item in self.event_receipts)
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("GoalSystem event IDs cannot be reused")
        if any(
            later.recorded_at < earlier.recorded_at
            for earlier, later in zip(self.event_receipts, self.event_receipts[1:])
        ):
            raise ValueError("GoalSystem event timestamps must not regress")
        record_map = {item.goal_id: item for item in self.records}
        receipts_by_goal: dict[str, list[GoalSystemEventReceipt]] = {}
        for receipt in self.event_receipts:
            record = record_map.get(receipt.goal_id)
            if record is None:
                raise ValueError("Goal event receipt references a missing Goal")
            receipts_by_goal.setdefault(receipt.goal_id, []).append(receipt)
            if receipt.proposal_digest != record.proposal_digest:
                raise ValueError("Goal event receipt proposal digest mismatch")
            if receipt.operation in {
                GoalSystemEventOperation.INGEST_PROPOSAL,
                GoalSystemEventOperation.ADOPT,
            } and receipt.evidence_refs != record.evidence_refs:
                raise ValueError("Goal proposal/admission event must bind exact Goal evidence")
            if receipt.operation is GoalSystemEventOperation.ADOPT:
                admission = record.subject_admission
                if (
                    admission is None
                    or receipt.admission_digest != admission.admission_digest
                    or receipt.event_id != admission.event_id
                    or receipt.event_sequence != admission.event_sequence
                ):
                    raise ValueError("Goal admission receipt does not match current Goal authority")
        for record in self.records:
            goal_receipts = tuple(receipts_by_goal.get(record.goal_id, []))
            ingestions = tuple(
                item
                for item in goal_receipts
                if item.operation is GoalSystemEventOperation.INGEST_PROPOSAL
            )
            if not ingestions:
                raise ValueError("Goal authority lacks its exact proposal ingestion receipt")
            proposal_record_digests = {
                item.proposal_record_digest for item in ingestions
            }
            if len(proposal_record_digests) != 1:
                raise ValueError("Goal proposal ingestion receipts disagree on genesis contract")
            genesis_values = {item.proposal_genesis for item in ingestions}
            if len(genesis_values) != 1:
                raise ValueError("Goal proposal ingestion receipts disagree on genesis evidence")
            genesis = next(iter(genesis_values))
            if not isinstance(genesis, GoalRevisionRecord):
                raise ValueError("Goal proposal ingestion lacks its genesis revision")
            if genesis.evidence_refs != tuple(
                sorted(item.reference for item in record.evidence_refs)
            ):
                raise ValueError("Goal genesis revision does not bind exact proposal evidence")
            initial_record = replace(
                record,
                lifecycle=GoalLifecycle.PROPOSED,
                outcome_evidence_refs=(),
                subject_admission=None,
                subject_transition_proofs=(),
                revision=0,
                revision_history=(genesis,),
                history_anchor=None,
            )
            if goal_record_digest(initial_record) != next(iter(proposal_record_digests)):
                raise ValueError("proposal receipt does not bind the exact Goal genesis record")

            mutation_receipts = tuple(
                item
                for item in goal_receipts
                if item.operation is not GoalSystemEventOperation.INGEST_PROPOSAL
            )
            # Genesis may precede ingestion, but subject mutation cannot precede
            # the first GoalSystem ingestion, even after revision compaction.
            first_ingestion = ingestions[0]
            if goal_receipts[0].operation is not GoalSystemEventOperation.INGEST_PROPOSAL or any(
                item.event_sequence <= first_ingestion.event_sequence
                or item.recorded_at < first_ingestion.recorded_at
                for item in mutation_receipts
            ):
                raise ValueError("Goal subject mutation precedes first proposal ingestion")
            if len(mutation_receipts) != record.revision:
                raise ValueError("Goal revision count differs from its retained mutation receipts")
            if any(item.revision_witness is None for item in mutation_receipts):
                raise ValueError("Goal mutation receipt lacks its exact revision witness")
            mutation_revisions = tuple(
                item.revision_witness for item in mutation_receipts
            )
            if tuple(item.revision for item in mutation_revisions if item is not None) != tuple(
                range(1, record.revision + 1)
            ):
                raise ValueError("Goal mutation receipts do not cover the exact revision prefix")
            admission_receipts = tuple(
                item
                for item in mutation_receipts
                if item.operation is GoalSystemEventOperation.ADOPT
            )
            if (record.lifecycle is GoalLifecycle.PROPOSED and admission_receipts) or (
                record.lifecycle is not GoalLifecycle.PROPOSED
                and len(admission_receipts) != 1
            ):
                raise ValueError("Goal admission receipt count differs from its lifecycle")

            previous_revision = genesis
            previous_state = GoalLifecycle.PROPOSED
            revision_digests = {genesis.revision: genesis.record_digest}
            for receipt, revision in zip(
                mutation_receipts,
                mutation_revisions,
                strict=True,
            ):
                if revision is None:
                    raise ValueError("Goal mutation receipt lost its revision witness")
                if (
                    revision.goal_id != record.goal_id
                    or revision.proposal_digest != record.proposal_digest
                    or revision.previous_revision_digest != previous_revision.record_digest
                    or revision.previous_lifecycle_state is not previous_state
                ):
                    raise ValueError("Goal mutation receipt breaks the exact revision chain")
                operation = _system_operation_for_revision(revision)
                if receipt.operation is not operation:
                    raise ValueError("Goal revision prefix contains an inconsistent operation")
                allowed_next = {
                    GoalLifecycle.PROPOSED: {
                        GoalSystemEventOperation.ADOPT,
                    },
                    GoalLifecycle.ADOPTED: {
                        GoalSystemEventOperation.DEFER,
                        GoalSystemEventOperation.ABANDON,
                    },
                    GoalLifecycle.DEFERRED: {
                        GoalSystemEventOperation.DEFER,
                        GoalSystemEventOperation.RESUME,
                        GoalSystemEventOperation.ABANDON,
                    },
                }.get(previous_state, set())
                if operation not in allowed_next:
                    raise ValueError("Goal mutation receipt has an invalid lifecycle transition")
                previous_state = _goal_revision_after(revision)
                previous_revision = revision
                revision_digests[revision.revision] = revision.record_digest

            if (
                previous_revision.revision != record.revision
                or previous_revision.record_digest != record.revision_history[-1].record_digest
                or previous_state is not record.lifecycle
            ):
                raise ValueError("Goal record differs from its exact mutation revision prefix")
            if any(
                revision_digests.get(item.revision) != item.record_digest
                for item in record.revision_history
            ):
                raise ValueError("Goal retained revision differs from its exact event witness")
            anchor = record.history_anchor
            if (
                anchor is not None
                and revision_digests.get(anchor.through_revision) != anchor.through_digest
            ):
                raise ValueError("Goal compaction anchor differs from its exact event witness")

            for revision in record.revision_history:
                if revision.operation is not GoalRevisionOperation.CREATE:
                    _validate_revision_receipt(revision, goal_receipts)

        payload = self.canonical_value()
        encoded = canonical_json(payload)
        if len(encoded) > GOAL_SYSTEM_MAX_SERIALIZED_BYTES:
            raise GoalSystemCapacityExceeded("GoalSystem snapshot exceeds its byte bound")
        object.__setattr__(self, "serialized_bytes", len(encoded))
        object.__setattr__(self, "authority_digest", digest_payload(GOAL_SYSTEM_DOMAIN, payload))

    def canonical_value(self) -> dict[str, object]:
        return {
            "event_receipts": [item.canonical_value() for item in self.event_receipts],
            "records": [_goal_system_record_value(item) for item in self.records],
            "schema_version": self.schema_version,
        }


def _goal_input_digest(
    operation: GoalSystemEventOperation,
    *,
    goal_id: str,
    proposal_digest: str,
    evidence_refs: tuple[R13Reference, ...],
    proposal_record_digest: str | None = None,
    admission_digest: str | None = None,
    transition_digest: str | None = None,
) -> str:
    return _goal_event_input_digest(
        operation=operation,
        goal_id=goal_id,
        proposal_digest=proposal_digest,
        evidence_refs=evidence_refs,
        proposal_record_digest=proposal_record_digest,
        admission_digest=admission_digest,
        transition_digest=transition_digest,
    )


class GoalSystem:
    """Bounded process-local proposal/admission/lifecycle authority for Goals."""

    MAX_RECORDS: Final = GOAL_SYSTEM_MAX_RECORDS

    def __init__(self) -> None:
        self._lock = RLock()
        self._records: tuple[GoalRecord, ...] = ()
        self._event_receipts: tuple[GoalSystemEventReceipt, ...] = ()

    @property
    def records(self) -> tuple[GoalRecord, ...]:
        with self._lock:
            return self._records

    def get(self, goal_id: str) -> GoalRecord | None:
        identifier = validate_identifier(goal_id)
        with self._lock:
            return next((item for item in self._records if item.goal_id == identifier), None)

    def snapshot(self) -> GoalSystemSnapshot:
        with self._lock:
            return GoalSystemSnapshot(self._records, self._event_receipts)

    export = snapshot

    def validate(self) -> None:
        self.snapshot()

    def ingest_proposal(
        self,
        proposal: GoalRecord,
        event: GoalMutationEvidence,
    ) -> GoalRecord:
        """Admit a U1 Goal record into process-local proposal state only."""

        with self._lock:
            if not isinstance(proposal, GoalRecord):
                raise TypeError("proposal must be GoalRecord")
            if not isinstance(event, GoalMutationEvidence):
                raise TypeError("event must be GoalMutationEvidence")
            proposal_record_digest = goal_record_digest(proposal)
            input_digest = _goal_input_digest(
                GoalSystemEventOperation.INGEST_PROPOSAL,
                goal_id=proposal.goal_id,
                proposal_digest=proposal.proposal_digest,
                evidence_refs=event.evidence_refs,
                proposal_record_digest=proposal_record_digest,
            )
            prior = self._matching_event(
                event,
                GoalSystemEventOperation.INGEST_PROPOSAL,
                proposal.goal_id,
                proposal.proposal_digest,
                input_digest,
            )
            if prior is not None:
                current = self.get(proposal.goal_id)
                if current is None:
                    raise GoalSystemError("replayed proposal event lost its Goal")
                return current
            self._validate_proposal_input(proposal, event)
            current = self.get(proposal.goal_id)
            self._check_new_event_capacity(event)
            if current is not None:
                prior_ingestion = next(
                    (
                        item
                        for item in self._event_receipts
                        if item.goal_id == proposal.goal_id
                        and item.operation is GoalSystemEventOperation.INGEST_PROPOSAL
                    ),
                    None,
                )
                if (
                    current.proposal_digest != proposal.proposal_digest
                    or prior_ingestion is None
                    or prior_ingestion.proposal_record_digest != proposal_record_digest
                ):
                    raise GoalSystemConflict(
                        "same Goal identity was reused with a different proposal contract"
                    )
                updated_records = self._records
            else:
                if len(self._records) >= GOAL_SYSTEM_MAX_RECORDS:
                    raise GoalSystemCapacityExceeded("Goal record capacity is exhausted")
                updated_records = tuple(
                    sorted((*self._records, proposal), key=lambda item: item.goal_id)
                )
                self._validate_graph(updated_records)
            receipt = self._make_receipt(
                event,
                GoalSystemEventOperation.INGEST_PROPOSAL,
                proposal.goal_id,
                proposal.proposal_digest,
                input_digest,
                proposal_record_digest=proposal_record_digest,
                proposal_genesis=proposal.revision_history[0],
            )
            self._replace_state(
                records=updated_records,
                event_receipts=(*self._event_receipts, receipt),
            )
            return self._require_record(proposal.goal_id)

    def adopt(
        self,
        goal_id: str,
        admission: GoalSubjectAdmission,
        event: GoalMutationEvidence,
    ) -> GoalRecord:
        """Consume the exact caller-supplied initial subject admission."""

        with self._lock:
            current = self._require_record(goal_id)
            if not isinstance(admission, GoalSubjectAdmission):
                raise TypeError("admission must be GoalSubjectAdmission")
            if not isinstance(event, GoalMutationEvidence):
                raise TypeError("event must be GoalMutationEvidence")
            input_digest = _goal_input_digest(
                GoalSystemEventOperation.ADOPT,
                goal_id=current.goal_id,
                proposal_digest=current.proposal_digest,
                evidence_refs=event.evidence_refs,
                admission_digest=admission.admission_digest,
            )
            prior = self._matching_event(
                event,
                GoalSystemEventOperation.ADOPT,
                current.goal_id,
                current.proposal_digest,
                input_digest,
            )
            if prior is not None:
                return current
            self._validate_admission_input(current, admission, event)
            self._check_new_event_capacity(event)
            if current.lifecycle is not GoalLifecycle.PROPOSED:
                raise GoalSystemError("only proposed Goals may be adopted")
            if current.revision >= R13_MAX_REVISION:
                raise GoalSystemCapacityExceeded("Goal revision bound is exhausted")
            revision = GoalRevisionRecord(
                goal_id=current.goal_id,
                revision=current.revision + 1,
                operation=GoalRevisionOperation.ADOPT,
                reason=GoalRevisionReason.SUBJECT_ADMISSION,
                created_at=event.recorded_at,
                previous_lifecycle_state=GoalLifecycle.PROPOSED,
                proposal_digest=current.proposal_digest,
                previous_revision_digest=current.revision_history[-1].record_digest,
                event_id=event.event_id,
                event_sequence=event.event_sequence,
                evidence_refs=tuple(
                    sorted((*admission.evidence_refs, admission.admission_digest))
                ),
            )
            history, anchor = _append_goal_revision(current, revision)
            updated = replace(
                current,
                lifecycle=GoalLifecycle.ADOPTED,
                subject_admission=admission,
                revision=current.revision + 1,
                revision_history=history,
                history_anchor=anchor,
            )
            receipt = self._make_receipt(
                event,
                GoalSystemEventOperation.ADOPT,
                current.goal_id,
                current.proposal_digest,
                input_digest,
                admission_digest=admission.admission_digest,
                revision_witness=revision,
            )
            self._replace_one(updated, receipt)
            return updated

    def defer(
        self,
        goal_id: str,
        proof: GoalSubjectTransitionProof,
        event: GoalMutationEvidence,
    ) -> GoalRecord:
        return self._subject_transition(
            goal_id,
            proof,
            event,
            system_operation=GoalSystemEventOperation.DEFER,
            revision_operation=GoalRevisionOperation.DEFER,
            reason=GoalRevisionReason.SUBJECT_DEFERRED,
            allowed_previous=frozenset({GoalLifecycle.ADOPTED, GoalLifecycle.DEFERRED}),
            next_lifecycle=GoalLifecycle.DEFERRED,
        )

    def resume(
        self,
        goal_id: str,
        proof: GoalSubjectTransitionProof,
        event: GoalMutationEvidence,
    ) -> GoalRecord:
        return self._subject_transition(
            goal_id,
            proof,
            event,
            system_operation=GoalSystemEventOperation.RESUME,
            revision_operation=GoalRevisionOperation.ADOPT,
            reason=GoalRevisionReason.SUBJECT_ADMISSION,
            allowed_previous=frozenset({GoalLifecycle.DEFERRED}),
            next_lifecycle=GoalLifecycle.ADOPTED,
        )

    def abandon(
        self,
        goal_id: str,
        proof: GoalSubjectTransitionProof,
        event: GoalMutationEvidence,
    ) -> GoalRecord:
        return self._subject_transition(
            goal_id,
            proof,
            event,
            system_operation=GoalSystemEventOperation.ABANDON,
            revision_operation=GoalRevisionOperation.ABANDON,
            reason=GoalRevisionReason.SUBJECT_ABANDONED,
            allowed_previous=frozenset({GoalLifecycle.ADOPTED, GoalLifecycle.DEFERRED}),
            next_lifecycle=GoalLifecycle.ABANDONED,
        )

    def _subject_transition(
        self,
        goal_id: str,
        proof: GoalSubjectTransitionProof,
        event: GoalMutationEvidence,
        *,
        system_operation: GoalSystemEventOperation,
        revision_operation: GoalRevisionOperation,
        reason: GoalRevisionReason,
        allowed_previous: frozenset[GoalLifecycle],
        next_lifecycle: GoalLifecycle,
    ) -> GoalRecord:
        with self._lock:
            current = self._require_record(goal_id)
            if not isinstance(proof, GoalSubjectTransitionProof):
                raise TypeError("proof must be GoalSubjectTransitionProof")
            if not isinstance(event, GoalMutationEvidence):
                raise TypeError("event must be GoalMutationEvidence")
            input_digest = _goal_input_digest(
                system_operation,
                goal_id=current.goal_id,
                proposal_digest=current.proposal_digest,
                evidence_refs=event.evidence_refs,
                transition_digest=proof.transition_digest,
            )
            prior = self._matching_event(
                event,
                system_operation,
                current.goal_id,
                current.proposal_digest,
                input_digest,
            )
            if prior is not None:
                return current
            self._validate_transition_input(
                current,
                proof,
                event,
                system_operation=system_operation,
                revision_operation=revision_operation,
                reason=reason,
            )
            self._check_new_event_capacity(event)
            if current.lifecycle not in allowed_previous:
                raise GoalSystemError("Goal lifecycle does not allow this subject transition")
            if proof.previous_lifecycle_state is not current.lifecycle:
                raise GoalSystemError("Goal transition proof has a stale prior lifecycle")
            if proof.transition_digest in {
                item.transition_digest for item in current.subject_transition_proofs
            } or any(
                receipt.transition_digest == proof.transition_digest
                for receipt in self._event_receipts
            ):
                raise GoalSystemConflict("Goal transition proof was already consumed")
            if current.revision >= R13_MAX_REVISION:
                raise GoalSystemCapacityExceeded("Goal revision bound is exhausted")
            revision = GoalRevisionRecord(
                goal_id=current.goal_id,
                revision=current.revision + 1,
                operation=revision_operation,
                reason=reason,
                created_at=event.recorded_at,
                previous_lifecycle_state=current.lifecycle,
                proposal_digest=current.proposal_digest,
                previous_revision_digest=current.revision_history[-1].record_digest,
                event_id=event.event_id,
                event_sequence=event.event_sequence,
                evidence_refs=tuple(
                    sorted((*proof.evidence_refs, proof.transition_digest))
                ),
            )
            history, anchor = _append_goal_revision(current, revision)
            proof_set = (*current.subject_transition_proofs, proof)
            required_witnesses = {
                item
                for revision_item in history
                for item in revision_item.evidence_refs
            }
            if anchor is not None:
                required_witnesses.update(anchor.through_evidence_refs)
            retained_proofs = tuple(
                sorted(
                    (
                        item
                        for item in proof_set
                        if item.transition_digest in required_witnesses
                    ),
                    key=lambda item: item.transition_digest,
                )
            )
            updated = replace(
                current,
                lifecycle=next_lifecycle,
                subject_transition_proofs=retained_proofs,
                revision=current.revision + 1,
                revision_history=history,
                history_anchor=anchor,
            )
            receipt = self._make_receipt(
                event,
                system_operation,
                current.goal_id,
                current.proposal_digest,
                input_digest,
                transition_digest=proof.transition_digest,
                revision_witness=revision,
            )
            self._replace_one(updated, receipt)
            return updated

    @staticmethod
    def _validate_proposal_input(
        proposal: GoalRecord,
        event: GoalMutationEvidence,
    ) -> None:
        if not isinstance(proposal, GoalRecord):
            raise TypeError("proposal must be GoalRecord")
        if not isinstance(event, GoalMutationEvidence):
            raise TypeError("event must be GoalMutationEvidence")
        if proposal.lifecycle is not GoalLifecycle.PROPOSED:
            raise GoalSystemError("GoalSystem ingests proposal state only")
        if (
            proposal.subject_admission is not None
            or proposal.subject_transition_proofs
            or proposal.outcome_evidence_refs
            or proposal.revision != 0
            or proposal.history_anchor is not None
            or len(proposal.revision_history) != 1
        ):
            raise GoalSystemError("proposal input contains pre-admitted or advanced Goal state")
        genesis = proposal.revision_history[0]
        if (
            genesis.operation is not GoalRevisionOperation.CREATE
            or genesis.reason is not GoalRevisionReason.CREATION
            or genesis.event_id is None
            or genesis.event_sequence is None
        ):
            raise GoalSystemError("proposal input lacks exact genesis evidence")
        if event.evidence_refs != proposal.evidence_refs:
            raise GoalSystemError("proposal event evidence must exactly match Goal evidence")
        if genesis.evidence_refs != tuple(
            sorted(item.reference for item in proposal.evidence_refs)
        ):
            raise GoalSystemError("proposal genesis must bind exact Goal evidence")
        if genesis.event_sequence > event.event_sequence or genesis.created_at > event.recorded_at:
            raise GoalSystemError("proposal genesis is in the future of its ingestion event")
        if (
            genesis.event_id == event.event_id
            or genesis.event_sequence == event.event_sequence
        ) and (
            genesis.event_id != event.event_id
            or genesis.event_sequence != event.event_sequence
            or genesis.created_at != event.recorded_at
        ):
            raise GoalSystemError("proposal genesis event identity must match exactly")

    @staticmethod
    def _validate_admission_input(
        current: GoalRecord,
        admission: GoalSubjectAdmission,
        event: GoalMutationEvidence,
    ) -> None:
        if not isinstance(admission, GoalSubjectAdmission):
            raise TypeError("admission must be GoalSubjectAdmission")
        if not isinstance(event, GoalMutationEvidence):
            raise TypeError("event must be GoalMutationEvidence")
        expected_refs = tuple(sorted(item.reference for item in current.evidence_refs))
        if (
            admission.goal_id != current.goal_id
            or admission.proposal_digest != current.proposal_digest
            or admission.evidence_refs != expected_refs
            or admission.event_id != event.event_id
            or admission.event_sequence != event.event_sequence
            or event.evidence_refs != current.evidence_refs
        ):
            raise GoalSystemError("Goal subject admission does not match exact proposal/event evidence")

    @staticmethod
    def _validate_transition_input(
        current: GoalRecord,
        proof: GoalSubjectTransitionProof,
        event: GoalMutationEvidence,
        *,
        system_operation: GoalSystemEventOperation,
        revision_operation: GoalRevisionOperation,
        reason: GoalRevisionReason,
    ) -> None:
        if not isinstance(proof, GoalSubjectTransitionProof):
            raise TypeError("proof must be GoalSubjectTransitionProof")
        if not isinstance(event, GoalMutationEvidence):
            raise TypeError("event must be GoalMutationEvidence")
        expected_event_evidence = tuple(sorted(item.reference for item in event.evidence_refs))
        if (
            proof.goal_id != current.goal_id
            or proof.proposal_digest != current.proposal_digest
            or proof.operation is not revision_operation
            or proof.reason is not reason
            or proof.event_id != event.event_id
            or proof.event_sequence != event.event_sequence
            or proof.evidence_refs != expected_event_evidence
        ):
            raise GoalSystemError("Goal transition proof does not match exact Goal/event evidence")
        del system_operation

    def _validate_graph(self, records: tuple[GoalRecord, ...]) -> None:
        try:
            validate_goal_reference_graph(records)
        except (TypeError, ValueError) as error:
            raise GoalSystemConflict("Goal dependency/conflict facts are invalid") from error

    def _require_record(self, goal_id: str) -> GoalRecord:
        current = self.get(goal_id)
        if current is None:
            raise GoalSystemError("Goal is not present")
        return current

    def _matching_event(
        self,
        event: GoalMutationEvidence,
        operation: GoalSystemEventOperation,
        goal_id: str,
        proposal_digest: str,
        input_digest: str,
    ) -> GoalSystemEventReceipt | None:
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
            and existing.goal_id == goal_id
            and existing.proposal_digest == proposal_digest
            and existing.input_digest == input_digest
            and existing.evidence_refs == event.evidence_refs
        ):
            return existing
        raise GoalSystemConflict("Goal event identity was reused with conflicting input")

    def _check_event_order(self, event: GoalMutationEvidence) -> None:
        if any(item.event_sequence == event.event_sequence for item in self._event_receipts):
            raise GoalSystemConflict("Goal event sequence was already used")
        if self._event_receipts:
            latest = self._event_receipts[-1]
            if event.event_sequence <= latest.event_sequence:
                raise GoalSystemError("Goal event sequence regressed")
            if event.recorded_at < latest.recorded_at:
                raise GoalSystemError("Goal event time regressed")

    def _check_new_event_capacity(self, event: GoalMutationEvidence) -> None:
        self._check_event_order(event)
        if len(self._event_receipts) >= GOAL_SYSTEM_MAX_EVENT_RECEIPTS:
            raise GoalSystemCapacityExceeded("Goal event receipt capacity is exhausted")

    @staticmethod
    def _make_receipt(
        event: GoalMutationEvidence,
        operation: GoalSystemEventOperation,
        goal_id: str,
        proposal_digest: str,
        input_digest: str,
        *,
        proposal_record_digest: str | None = None,
        proposal_genesis: GoalRevisionRecord | None = None,
        revision_witness: GoalRevisionRecord | None = None,
        admission_digest: str | None = None,
        transition_digest: str | None = None,
    ) -> GoalSystemEventReceipt:
        return GoalSystemEventReceipt(
            event_id=event.event_id,
            event_sequence=event.event_sequence,
            recorded_at=event.recorded_at,
            operation=operation,
            goal_id=goal_id,
            proposal_digest=proposal_digest,
            input_digest=input_digest,
            evidence_refs=event.evidence_refs,
            proposal_record_digest=proposal_record_digest,
            proposal_genesis=proposal_genesis,
            revision_witness=revision_witness,
            admission_digest=admission_digest,
            transition_digest=transition_digest,
        )

    def _replace_one(
        self,
        record: GoalRecord,
        receipt: GoalSystemEventReceipt,
    ) -> None:
        records = tuple(
            sorted(
                (record if item.goal_id == record.goal_id else item for item in self._records),
                key=lambda item: item.goal_id,
            )
        )
        self._replace_state(records=records, event_receipts=(*self._event_receipts, receipt))

    def _replace_state(
        self,
        *,
        records: tuple[GoalRecord, ...] | None = None,
        event_receipts: tuple[GoalSystemEventReceipt, ...] | None = None,
    ) -> None:
        proposed = GoalSystemSnapshot(
            records=self._records if records is None else records,
            event_receipts=(
                self._event_receipts if event_receipts is None else event_receipts
            ),
        )
        self._records = proposed.records
        self._event_receipts = proposed.event_receipts


__all__ = [
    "GOAL_SYSTEM_DOMAIN",
    "GOAL_SYSTEM_MAX_EVENT_RECEIPTS",
    "GOAL_SYSTEM_MAX_EVIDENCE_REFS_PER_EVENT",
    "GOAL_SYSTEM_MAX_RECORDS",
    "GOAL_SYSTEM_MAX_SERIALIZED_BYTES",
    "GOAL_SYSTEM_SCHEMA_VERSION",
    "GoalMutationEvidence",
    "GoalSystem",
    "GoalSystemCapacityExceeded",
    "GoalSystemConflict",
    "GoalSystemError",
    "GoalSystemEventOperation",
    "GoalSystemEventReceipt",
    "GoalSystemSnapshot",
]
