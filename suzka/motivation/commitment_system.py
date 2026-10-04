"""Bounded process-local Commitment lifecycle authority for R13 U4.

The immutable U1 Commitment values remain the schema and proof contract. This
system ingests exact proposals and consumes only caller-supplied subject
admission/transition proofs. The proof shape and digest establish internal
binding, not producer authentication: a trusted caller must supply them, and
R16 is the first planned production admission owner. It has no runtime,
inference, persistence, scheduler, outcome-verification, or Goal/Desire
transition behavior.
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
    R13_MAX_REVISION_WITNESSES,
    R13Reference,
    canonical_datetime,
    canonical_json,
    canonical_references,
    digest_payload,
    utc_datetime,
    validate_digest,
)
from suzka.motivation.commitment import (
    COMMITMENT_DOMAIN,
    COMMITMENT_EVIDENCE_KINDS,
    CommitmentLifecycle,
    CommitmentRecord,
    CommitmentRevisionOperation,
    CommitmentRevisionReason,
    CommitmentRevisionRecord,
    CommitmentSubjectAdmission,
    CommitmentSubjectTransitionProof,
    canonical_commitment_payload,
    commitment_record_digest,
)


COMMITMENT_SYSTEM_SCHEMA_VERSION: Final = 1
COMMITMENT_SYSTEM_DOMAIN: Final = b"PROJECT-SUZKA:R13:COMMITMENT-SYSTEM:V1\0"
COMMITMENT_SYSTEM_EVENT_DOMAIN: Final = (
    b"PROJECT-SUZKA:R13:COMMITMENT-SYSTEM-EVENT:V1\0"
)
COMMITMENT_SYSTEM_MAX_RECORDS: Final = R13_MAX_RECORDS_PER_DOMAIN
COMMITMENT_SYSTEM_MAX_EVENT_RECEIPTS: Final = 256
COMMITMENT_SYSTEM_MAX_EVIDENCE_REFS_PER_EVENT: Final = R13_MAX_EVIDENCE_REFS
COMMITMENT_SYSTEM_MAX_SERIALIZED_BYTES: Final = 16 * 1024 * 1024


class CommitmentSystemError(ValueError):
    """Base class for bounded Commitment authority failures."""


class CommitmentSystemConflict(CommitmentSystemError):
    """A Commitment identity, event, or proof was reused inconsistently."""


class CommitmentSystemCapacityExceeded(CommitmentSystemError):
    """A bounded Commitment record, receipt, revision, or snapshot is full."""


class CommitmentSystemEventOperation(str, Enum):
    INGEST_PROPOSAL = "ingest_proposal"
    ACCEPT = "accept"
    RELEASE = "release"
    RENEGOTIATE = "renegotiate"


def _exact_enum(value: object, enum_type: type[Enum], name: str) -> None:
    if type(value) is not enum_type:
        raise TypeError(f"{name} must be a {enum_type.__name__}")


def _admission_value(admission: CommitmentSubjectAdmission) -> dict[str, object]:
    """Encode the complete U1 admission (U1's record payload stores its digest)."""

    return {
        "admission_digest": admission.admission_digest,
        "beneficiary": admission.beneficiary.canonical_value(),
        "commitment_id": admission.commitment_id,
        "deadline": admission.deadline.canonical_value(),
        "event_id": admission.event_id,
        "event_sequence": admission.event_sequence,
        "evidence_refs": list(admission.evidence_refs),
        "proposal_digest": admission.proposal_digest,
        "reason": admission.reason.value,
        "scope": list(admission.scope),
    }


def _revision_value(revision: CommitmentRevisionRecord) -> dict[str, object]:
    return {
        "commitment_id": revision.commitment_id,
        "created_at": canonical_datetime(revision.created_at),
        "event_id": revision.event_id,
        "event_sequence": revision.event_sequence,
        "evidence_refs": list(revision.evidence_refs),
        "operation": revision.operation.value,
        "previous_lifecycle_state": (
            None
            if revision.previous_lifecycle_state is None
            else revision.previous_lifecycle_state.value
        ),
        "previous_revision_digest": revision.previous_revision_digest,
        "reason": revision.reason.value,
        "record_digest": revision.record_digest,
        "revision": revision.revision,
    }


def _event_input_digest(
    *,
    operation: CommitmentSystemEventOperation,
    commitment_id: str,
    proposal_digest: str,
    evidence_refs: tuple[R13Reference, ...],
    proposal_record_digest: str | None = None,
    admission_digest: str | None = None,
    transition_digest: str | None = None,
) -> str:
    return digest_payload(
        COMMITMENT_SYSTEM_EVENT_DOMAIN,
        {
            "admission_digest": admission_digest,
            "commitment_id": commitment_id,
            "evidence_refs": [item.canonical_value() for item in evidence_refs],
            "operation": operation.value,
            "proposal_digest": proposal_digest,
            "proposal_record_digest": proposal_record_digest,
            "transition_digest": transition_digest,
        },
    )


@dataclass(frozen=True, slots=True)
class CommitmentMutationEvidence:
    """Explicit UTC event identity and bounded typed witnesses for a mutation."""

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
                maximum=COMMITMENT_SYSTEM_MAX_EVIDENCE_REFS_PER_EVENT,
                allow_empty=False,
                allowed_kinds=COMMITMENT_EVIDENCE_KINDS,
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
class CommitmentSystemEventReceipt:
    """Exact bounded witness for one proposal ingestion or subject mutation."""

    event_id: str
    event_sequence: int
    recorded_at: datetime
    operation: CommitmentSystemEventOperation
    commitment_id: str
    proposal_digest: str
    input_digest: str
    evidence_refs: tuple[R13Reference, ...]
    proposal_record_digest: str | None = None
    proposal_genesis: CommitmentRevisionRecord | None = None
    revision_witness: CommitmentRevisionRecord | None = None
    admission_digest: str | None = None
    admission: CommitmentSubjectAdmission | None = None
    transition_digest: str | None = None
    transition_proof: CommitmentSubjectTransitionProof | None = None
    receipt_digest: str = field(init=False)

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
        _exact_enum(self.operation, CommitmentSystemEventOperation, "operation")
        commitment_id = validate_identifier(self.commitment_id)
        proposal_digest = validate_digest(self.proposal_digest, "proposal_digest")
        input_digest = validate_digest(self.input_digest, "input_digest")
        evidence_refs = canonical_references(
            self.evidence_refs,
            "evidence_refs",
            maximum=COMMITMENT_SYSTEM_MAX_EVIDENCE_REFS_PER_EVENT,
            allow_empty=False,
            allowed_kinds=COMMITMENT_EVIDENCE_KINDS,
        )
        for name in (
            "proposal_record_digest",
            "admission_digest",
            "transition_digest",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, validate_digest(value, name))

        if self.operation is CommitmentSystemEventOperation.INGEST_PROPOSAL:
            genesis = self.proposal_genesis
            if (
                self.proposal_record_digest is None
                or not isinstance(genesis, CommitmentRevisionRecord)
                or self.revision_witness is not None
                or self.admission_digest is not None
                or self.admission is not None
                or self.transition_digest is not None
                or self.transition_proof is not None
            ):
                raise ValueError("proposal receipt requires its exact genesis witness")
            if (
                genesis.commitment_id != commitment_id
                or genesis.revision != 0
                or genesis.operation is not CommitmentRevisionOperation.CREATE
                or genesis.reason is not CommitmentRevisionReason.CREATION
                or genesis.previous_lifecycle_state is not None
                or genesis.event_id is None
                or genesis.event_sequence is None
                or genesis.event_sequence > self.event_sequence
                or genesis.created_at > self.recorded_at
                or (
                    genesis.event_id == self.event_id
                    or genesis.event_sequence == self.event_sequence
                )
                and (
                    genesis.event_id != self.event_id
                    or genesis.event_sequence != self.event_sequence
                    or genesis.created_at != self.recorded_at
                )
            ):
                raise ValueError("proposal receipt genesis witness is inconsistent")
        elif self.operation is CommitmentSystemEventOperation.ACCEPT:
            revision = self.revision_witness
            admission = self.admission
            if (
                self.proposal_record_digest is not None
                or self.proposal_genesis is not None
                or not isinstance(revision, CommitmentRevisionRecord)
                or not isinstance(admission, CommitmentSubjectAdmission)
                or self.admission_digest is None
                or self.transition_digest is not None
                or self.transition_proof is not None
            ):
                raise ValueError("accept receipt requires its exact revision and admission")
            expected_evidence = tuple(
                sorted((*[item.reference for item in evidence_refs], self.admission_digest))
            )
            if (
                admission.admission_digest != self.admission_digest
                or admission.commitment_id != commitment_id
                or admission.proposal_digest != proposal_digest
                or admission.event_id != self.event_id
                or admission.event_sequence != self.event_sequence
                or revision.commitment_id != commitment_id
                or revision.revision < 1
                or revision.operation is not CommitmentRevisionOperation.ADMIT
                or revision.reason is not CommitmentRevisionReason.SUBJECT_ADMISSION
                or revision.previous_lifecycle_state is not CommitmentLifecycle.PROPOSED
                or revision.event_id != self.event_id
                or revision.event_sequence != self.event_sequence
                or revision.created_at != self.recorded_at
                or revision.evidence_refs != expected_evidence
            ):
                raise ValueError("accept receipt does not bind its exact admission")
        else:
            revision = self.revision_witness
            proof = self.transition_proof
            transition = {
                CommitmentSystemEventOperation.RELEASE: (
                    CommitmentRevisionOperation.RELEASE,
                    CommitmentRevisionReason.SUBJECT_RELEASE,
                ),
                CommitmentSystemEventOperation.RENEGOTIATE: (
                    CommitmentRevisionOperation.RENEGOTIATE,
                    CommitmentRevisionReason.SUBJECT_RENEGOTIATION,
                ),
            }[self.operation]
            if (
                self.proposal_record_digest is not None
                or self.proposal_genesis is not None
                or not isinstance(revision, CommitmentRevisionRecord)
                or not isinstance(proof, CommitmentSubjectTransitionProof)
                or self.admission_digest is not None
                or self.admission is not None
                or self.transition_digest is None
            ):
                raise ValueError("transition receipt requires its exact revision and proof")
            expected_evidence = tuple(
                sorted((*[item.reference for item in evidence_refs], self.transition_digest))
            )
            if (
                proof.transition_digest != self.transition_digest
                or proof.commitment_id != commitment_id
                or proof.proposal_digest != proposal_digest
                or proof.event_id != self.event_id
                or proof.event_sequence != self.event_sequence
                or proof.operation is not transition[0]
                or proof.reason is not transition[1]
                or proof.previous_lifecycle_state is not CommitmentLifecycle.ACTIVE
                or revision.commitment_id != commitment_id
                or revision.revision < 1
                or revision.operation is not transition[0]
                or revision.reason is not transition[1]
                or revision.previous_lifecycle_state is not CommitmentLifecycle.ACTIVE
                or revision.event_id != self.event_id
                or revision.event_sequence != self.event_sequence
                or revision.created_at != self.recorded_at
                or revision.evidence_refs != expected_evidence
            ):
                raise ValueError("transition receipt does not bind its exact proof")

        object.__setattr__(self, "commitment_id", commitment_id)
        object.__setattr__(self, "proposal_digest", proposal_digest)
        object.__setattr__(self, "input_digest", input_digest)
        object.__setattr__(self, "evidence_refs", evidence_refs)
        expected_input = _event_input_digest(
            operation=self.operation,
            commitment_id=commitment_id,
            proposal_digest=proposal_digest,
            evidence_refs=evidence_refs,
            proposal_record_digest=self.proposal_record_digest,
            admission_digest=self.admission_digest,
            transition_digest=self.transition_digest,
        )
        if input_digest != expected_input:
            raise ValueError("CommitmentSystem event input digest is inconsistent")
        object.__setattr__(
            self,
            "receipt_digest",
            digest_payload(COMMITMENT_SYSTEM_EVENT_DOMAIN, self.canonical_value()),
        )

    def canonical_value(self) -> dict[str, object]:
        return {
            "admission": (
                None if self.admission is None else _admission_value(self.admission)
            ),
            "admission_digest": self.admission_digest,
            "commitment_id": self.commitment_id,
            "event_id": self.event_id,
            "event_sequence": self.event_sequence,
            "evidence_refs": [item.canonical_value() for item in self.evidence_refs],
            "input_digest": self.input_digest,
            "operation": self.operation.value,
            "proposal_digest": self.proposal_digest,
            "proposal_genesis": (
                None
                if self.proposal_genesis is None
                else _revision_value(self.proposal_genesis)
            ),
            "proposal_record_digest": self.proposal_record_digest,
            "recorded_at": canonical_datetime(self.recorded_at),
            "revision_witness": (
                None
                if self.revision_witness is None
                else _revision_value(self.revision_witness)
            ),
            "transition_digest": self.transition_digest,
            "transition_proof": (
                None
                if self.transition_proof is None
                else self.transition_proof.canonical_value()
            ),
        }


def _system_record_value(record: CommitmentRecord) -> dict[str, object]:
    if commitment_record_digest(record) != record.record_digest:
        raise CommitmentSystemError("Commitment record digest is inconsistent")
    payload = canonical_commitment_payload(record)
    value = json.loads(payload[len(COMMITMENT_DOMAIN) :])
    if not isinstance(value, dict):
        raise CommitmentSystemError("Commitment canonical payload is invalid")
    value["admission"] = (
        None
        if record.subject_admission is None
        else _admission_value(record.subject_admission)
    )
    value["record_digest"] = record.record_digest
    return value


def _lifecycle_after(
    revision: CommitmentRevisionRecord,
) -> CommitmentLifecycle:
    if revision.operation is CommitmentRevisionOperation.CREATE:
        return CommitmentLifecycle.PROPOSED
    if revision.operation is CommitmentRevisionOperation.ADMIT:
        return CommitmentLifecycle.ACTIVE
    if revision.operation is CommitmentRevisionOperation.RELEASE:
        return CommitmentLifecycle.RELEASED
    if revision.operation is CommitmentRevisionOperation.RENEGOTIATE:
        return CommitmentLifecycle.RENEGOTIATED
    raise CommitmentSystemError("CommitmentSystem cannot represent this outcome")


@dataclass(frozen=True, slots=True)
class CommitmentSystemSnapshot:
    """Bounded, exact Commitment authority projection; not a restore artifact."""

    records: tuple[CommitmentRecord, ...] = ()
    event_receipts: tuple[CommitmentSystemEventReceipt, ...] = ()
    schema_version: int = COMMITMENT_SYSTEM_SCHEMA_VERSION
    authority_digest: str = field(init=False)
    serialized_bytes: int = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != COMMITMENT_SYSTEM_SCHEMA_VERSION
        ):
            raise ValueError("unsupported CommitmentSystem snapshot version")
        if type(self.records) is not tuple or any(
            not isinstance(item, CommitmentRecord) for item in self.records
        ):
            raise TypeError("records must contain CommitmentRecord values")
        if len(self.records) > COMMITMENT_SYSTEM_MAX_RECORDS:
            raise CommitmentSystemCapacityExceeded(
                "Commitment record capacity is exhausted"
            )
        if type(self.event_receipts) is not tuple or any(
            not isinstance(item, CommitmentSystemEventReceipt)
            for item in self.event_receipts
        ):
            raise TypeError(
                "event_receipts must contain CommitmentSystemEventReceipt values"
            )
        if len(self.event_receipts) > COMMITMENT_SYSTEM_MAX_EVENT_RECEIPTS:
            raise CommitmentSystemCapacityExceeded(
                "Commitment event receipt capacity is exhausted"
            )

        commitment_ids = tuple(item.commitment_id for item in self.records)
        if commitment_ids != tuple(sorted(set(commitment_ids))):
            raise ValueError("Commitment records must be sorted and unique")
        if any(
            item.lifecycle
            not in {
                CommitmentLifecycle.PROPOSED,
                CommitmentLifecycle.ACTIVE,
                CommitmentLifecycle.RELEASED,
                CommitmentLifecycle.RENEGOTIATED,
            }
            for item in self.records
        ):
            raise CommitmentSystemError(
                "CommitmentSystem cannot contain outcome or unknown lifecycle state"
            )
        if any(
            item.history_anchor is not None or item.outcome_evidence_refs
            for item in self.records
        ):
            raise CommitmentSystemError(
                "CommitmentSystem cannot contain compacted history or outcome evidence"
            )

        sequences = tuple(item.event_sequence for item in self.event_receipts)
        if sequences != tuple(sorted(set(sequences))):
            raise ValueError("CommitmentSystem receipts must be sequence ordered")
        event_ids = tuple(item.event_id for item in self.event_receipts)
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("CommitmentSystem event IDs cannot be reused")
        if any(
            later.recorded_at < earlier.recorded_at
            for earlier, later in zip(self.event_receipts, self.event_receipts[1:])
        ):
            raise ValueError("CommitmentSystem event timestamps must not regress")

        record_map = {item.commitment_id: item for item in self.records}
        receipts_by_commitment: dict[str, list[CommitmentSystemEventReceipt]] = {}
        for receipt in self.event_receipts:
            record = record_map.get(receipt.commitment_id)
            if record is None:
                raise ValueError("Commitment event receipt references a missing record")
            if receipt.proposal_digest != record.proposal_digest:
                raise ValueError("Commitment event receipt proposal digest mismatch")
            receipts_by_commitment.setdefault(receipt.commitment_id, []).append(receipt)

        # One upstream event may propose several Commitments or coincide with
        # an ingestion. Its identity, sequence, and time must agree across all
        # retained genesis witnesses and system receipts.
        event_by_id = {
            item.event_id: (item.event_sequence, item.recorded_at)
            for item in self.event_receipts
        }
        event_by_sequence = {
            item.event_sequence: (item.event_id, item.recorded_at)
            for item in self.event_receipts
        }
        for record in self.records:
            commitment_receipts = tuple(
                receipts_by_commitment.get(record.commitment_id, ())
            )
            ingestions = tuple(
                item
                for item in commitment_receipts
                if item.operation is CommitmentSystemEventOperation.INGEST_PROPOSAL
            )
            if not ingestions:
                raise ValueError(
                    "Commitment authority lacks its exact proposal ingestion receipt"
                )
            if commitment_receipts[0] is not ingestions[0]:
                raise ValueError(
                    "Commitment subject mutation precedes proposal ingestion"
                )

            proposal_record_digests = {
                item.proposal_record_digest for item in ingestions
            }
            genesis_values = {item.proposal_genesis for item in ingestions}
            if len(proposal_record_digests) != 1 or len(genesis_values) != 1:
                raise ValueError(
                    "Commitment proposal ingestion receipts disagree on genesis"
                )
            genesis = next(iter(genesis_values))
            if not isinstance(genesis, CommitmentRevisionRecord):
                raise ValueError("Commitment proposal receipt lacks its genesis revision")
            genesis_id = genesis.event_id
            genesis_sequence = genesis.event_sequence
            if genesis_id is None or genesis_sequence is None:
                raise ValueError("Commitment genesis lacks its event identity")
            known_id = event_by_id.get(genesis_id)
            known_sequence = event_by_sequence.get(genesis_sequence)
            if (
                known_id is not None
                and known_id != (genesis_sequence, genesis.created_at)
            ) or (
                known_sequence is not None
                and known_sequence != (genesis_id, genesis.created_at)
            ):
                raise ValueError("Commitment genesis event identity conflicts across proposals")
            event_by_id[genesis_id] = (genesis_sequence, genesis.created_at)
            event_by_sequence[genesis_sequence] = (genesis_id, genesis.created_at)
            proposal_evidence = tuple(item.reference for item in record.evidence_refs)
            if genesis.evidence_refs != tuple(sorted(proposal_evidence)):
                raise ValueError(
                    "Commitment genesis does not bind exact proposal evidence"
                )
            if any(item.evidence_refs != record.evidence_refs for item in ingestions):
                raise ValueError(
                    "Commitment proposal ingestion must bind exact typed evidence"
                )
            initial_record = replace(
                record,
                lifecycle=CommitmentLifecycle.PROPOSED,
                subject_admission=None,
                subject_transition_proofs=(),
                outcome_evidence_refs=(),
                revision=0,
                revision_history=(genesis,),
                history_anchor=None,
            )
            expected_proposal_digest = next(iter(proposal_record_digests))
            if commitment_record_digest(initial_record) != expected_proposal_digest:
                raise ValueError(
                    "proposal receipt does not bind the exact Commitment genesis record"
                )

            first_ingestion = ingestions[0]
            for ingestion in ingestions:
                if (
                    genesis_sequence > ingestion.event_sequence
                    or genesis.created_at > ingestion.recorded_at
                    or (
                        genesis_id == ingestion.event_id
                        or genesis_sequence == ingestion.event_sequence
                    )
                    and (
                        genesis_id != ingestion.event_id
                        or genesis_sequence != ingestion.event_sequence
                        or genesis.created_at != ingestion.recorded_at
                    )
                ):
                    raise ValueError(
                        "Commitment genesis event does not precede its ingestion"
                    )

            mutations = tuple(
                item
                for item in commitment_receipts
                if item.operation is not CommitmentSystemEventOperation.INGEST_PROPOSAL
            )
            if len(mutations) != record.revision:
                raise ValueError(
                    "Commitment revision count differs from its mutation receipts"
                )
            if any(item.revision_witness is None for item in mutations):
                raise ValueError("Commitment mutation receipt lacks its revision witness")
            mutation_revisions = tuple(
                item.revision_witness for item in mutations
            )
            if tuple(
                item.revision for item in mutation_revisions if item is not None
            ) != tuple(range(1, record.revision + 1)):
                raise ValueError(
                    "Commitment mutation receipts do not cover the exact revision prefix"
                )
            if record.revision > R13_MAX_REVISION_HISTORY:
                raise CommitmentSystemCapacityExceeded(
                    "Commitment revision history exceeds its retained bound"
                )
            if len(record.revision_history) != record.revision + 1:
                raise ValueError(
                    "CommitmentSystem requires the complete un-compacted revision history"
                )
            if record.revision_history[0] != genesis:
                raise ValueError("Commitment retained history differs from exact genesis")
            if record.revision_history[1:] != mutation_revisions:
                raise ValueError(
                    "Commitment retained history differs from exact mutation receipts"
                )

            admissions = tuple(
                item
                for item in mutations
                if item.operation is CommitmentSystemEventOperation.ACCEPT
            )
            if (record.lifecycle is CommitmentLifecycle.PROPOSED and admissions) or (
                record.lifecycle is not CommitmentLifecycle.PROPOSED
                and len(admissions) != 1
            ):
                raise ValueError(
                    "Commitment admission receipt count differs from its lifecycle"
                )
            if record.lifecycle is CommitmentLifecycle.PROPOSED:
                if record.subject_admission is not None or record.subject_transition_proofs:
                    raise ValueError("proposed Commitment contains subject proof payloads")
            elif not admissions or admissions[0].admission != record.subject_admission:
                raise ValueError(
                    "Commitment record does not retain its exact admission payload"
                )

            proof_receipts = tuple(
                item
                for item in mutations
                if item.operation
                in {
                    CommitmentSystemEventOperation.RELEASE,
                    CommitmentSystemEventOperation.RENEGOTIATE,
                }
            )
            proof_values = tuple(
                item.transition_proof for item in proof_receipts
            )
            if any(item is None for item in proof_values):
                raise ValueError("Commitment transition receipt lacks its exact proof")
            expected_proofs = tuple(
                sorted(
                    (item for item in proof_values if item is not None),
                    key=lambda item: item.transition_digest,
                )
            )
            if record.subject_transition_proofs != expected_proofs:
                raise ValueError(
                    "Commitment record does not retain its exact transition proof payload"
                )

            previous_revision = genesis
            previous_state = CommitmentLifecycle.PROPOSED
            for receipt, revision in zip(
                mutations,
                mutation_revisions,
                strict=True,
            ):
                if revision is None:
                    raise ValueError("Commitment mutation receipt lost its revision")
                if (
                    revision.commitment_id != record.commitment_id
                    or revision.revision != previous_revision.revision + 1
                    or revision.previous_revision_digest
                    != previous_revision.record_digest
                    or revision.previous_lifecycle_state is not previous_state
                    or receipt.event_sequence <= first_ingestion.event_sequence
                    or receipt.recorded_at < first_ingestion.recorded_at
                ):
                    raise ValueError(
                        "Commitment mutation receipt breaks the exact revision chain"
                    )
                allowed_operation = {
                    CommitmentLifecycle.PROPOSED: {
                        CommitmentSystemEventOperation.ACCEPT
                    },
                    CommitmentLifecycle.ACTIVE: {
                        CommitmentSystemEventOperation.RELEASE,
                        CommitmentSystemEventOperation.RENEGOTIATE,
                    },
                }.get(previous_state, set())
                if receipt.operation not in allowed_operation:
                    raise ValueError(
                        "Commitment mutation receipt has an invalid lifecycle transition"
                    )
                if _lifecycle_after(revision) is CommitmentLifecycle.PROPOSED:
                    raise ValueError("Commitment mutation cannot reopen a lifecycle")
                previous_state = _lifecycle_after(revision)
                previous_revision = revision

            if (
                previous_revision != record.revision_history[-1]
                or previous_state is not record.lifecycle
            ):
                raise ValueError(
                    "Commitment record differs from its exact mutation revision prefix"
                )

        payload = self.canonical_value()
        encoded = canonical_json(payload)
        if len(encoded) > COMMITMENT_SYSTEM_MAX_SERIALIZED_BYTES:
            raise CommitmentSystemCapacityExceeded(
                "CommitmentSystem snapshot exceeds its byte bound"
            )
        object.__setattr__(self, "serialized_bytes", len(encoded))
        object.__setattr__(
            self,
            "authority_digest",
            digest_payload(COMMITMENT_SYSTEM_DOMAIN, payload),
        )

    def canonical_value(self) -> dict[str, object]:
        return {
            "event_receipts": [
                item.canonical_value() for item in self.event_receipts
            ],
            "records": [_system_record_value(item) for item in self.records],
            "schema_version": self.schema_version,
        }


def _revision_after(
    current: CommitmentRecord,
    *,
    event: CommitmentMutationEvidence,
    operation: CommitmentRevisionOperation,
    reason: CommitmentRevisionReason,
    previous_state: CommitmentLifecycle,
    evidence_refs: tuple[str, ...],
) -> CommitmentRevisionRecord:
    if current.revision >= R13_MAX_REVISION:
        raise CommitmentSystemCapacityExceeded(
            "Commitment revision bound is exhausted"
        )
    if current.revision_history and current.revision_history[-1].revision == current.revision:
        previous_digest = current.revision_history[-1].record_digest
    else:
        raise CommitmentSystemError("Commitment current revision witness is missing")
    if len(evidence_refs) > R13_MAX_REVISION_WITNESSES:
        raise CommitmentSystemCapacityExceeded(
            "Commitment revision evidence exceeds its witness bound"
        )
    return CommitmentRevisionRecord(
        commitment_id=current.commitment_id,
        revision=current.revision + 1,
        operation=operation,
        reason=reason,
        created_at=event.recorded_at,
        previous_lifecycle_state=previous_state,
        event_id=event.event_id,
        event_sequence=event.event_sequence,
        evidence_refs=evidence_refs,
        previous_revision_digest=previous_digest,
    )


def _append_revision(
    current: CommitmentRecord,
    revision: CommitmentRevisionRecord,
) -> tuple[CommitmentRevisionRecord, ...]:
    history = (*current.revision_history, revision)
    if len(history) > R13_MAX_REVISION_HISTORY:
        raise CommitmentSystemCapacityExceeded(
            "Commitment revision history bound is exhausted"
        )
    return history


class CommitmentSystem:
    """Bounded process-local proposal/admission/release authority."""

    MAX_RECORDS: Final = COMMITMENT_SYSTEM_MAX_RECORDS

    def __init__(self) -> None:
        self._lock = RLock()
        self._records: tuple[CommitmentRecord, ...] = ()
        self._event_receipts: tuple[CommitmentSystemEventReceipt, ...] = ()

    @property
    def records(self) -> tuple[CommitmentRecord, ...]:
        with self._lock:
            return self._records

    def get(self, commitment_id: str) -> CommitmentRecord | None:
        identifier = validate_identifier(commitment_id)
        with self._lock:
            return next(
                (item for item in self._records if item.commitment_id == identifier),
                None,
            )

    def snapshot(self) -> CommitmentSystemSnapshot:
        with self._lock:
            return CommitmentSystemSnapshot(self._records, self._event_receipts)

    export = snapshot

    def validate(self) -> None:
        self.snapshot()

    def ingest_proposal(
        self,
        proposal: CommitmentRecord,
        event: CommitmentMutationEvidence,
    ) -> CommitmentRecord:
        """Ingest a U1 proposal genesis without admitting responsibility."""

        with self._lock:
            self.snapshot()
            if not isinstance(proposal, CommitmentRecord):
                raise TypeError("proposal must be CommitmentRecord")
            if not isinstance(event, CommitmentMutationEvidence):
                raise TypeError("event must be CommitmentMutationEvidence")
            self._validate_proposal_input(proposal, event)
            proposal_digest = commitment_record_digest(proposal)
            input_digest = _event_input_digest(
                operation=CommitmentSystemEventOperation.INGEST_PROPOSAL,
                commitment_id=proposal.commitment_id,
                proposal_digest=proposal.proposal_digest,
                evidence_refs=event.evidence_refs,
                proposal_record_digest=proposal_digest,
            )
            prior = self._matching_event(
                event,
                CommitmentSystemEventOperation.INGEST_PROPOSAL,
                proposal.commitment_id,
                proposal.proposal_digest,
                input_digest,
            )
            current = self.get(proposal.commitment_id)
            if prior is not None:
                if current is None:
                    raise CommitmentSystemError("replayed proposal event lost its record")
                return current
            if current is not None:
                first_ingestion = next(
                    (
                        item
                        for item in self._event_receipts
                        if item.commitment_id == proposal.commitment_id
                        and item.operation
                        is CommitmentSystemEventOperation.INGEST_PROPOSAL
                    ),
                    None,
                )
                if (
                    current.proposal_digest != proposal.proposal_digest
                    or first_ingestion is None
                    or first_ingestion.proposal_record_digest != proposal_digest
                    or first_ingestion.proposal_genesis != proposal.revision_history[0]
                ):
                    raise CommitmentSystemConflict(
                        "same Commitment identity was reused with a different proposal"
                    )
            else:
                if len(self._records) >= COMMITMENT_SYSTEM_MAX_RECORDS:
                    raise CommitmentSystemCapacityExceeded(
                        "Commitment record capacity is exhausted"
                    )
            self._check_new_event_capacity(event)
            receipt = self._make_receipt(
                event,
                CommitmentSystemEventOperation.INGEST_PROPOSAL,
                proposal.commitment_id,
                proposal.proposal_digest,
                input_digest,
                proposal_record_digest=proposal_digest,
                proposal_genesis=proposal.revision_history[0],
            )
            records = self._records
            if current is None:
                records = tuple(
                    sorted((*records, proposal), key=lambda item: item.commitment_id)
                )
            self._replace_state(records=records, event_receipts=(*self._event_receipts, receipt))
            result = self.get(proposal.commitment_id)
            if result is None:
                raise CommitmentSystemError("ingested Commitment was not published")
            return result

    def accept(
        self,
        commitment_id: str,
        admission: CommitmentSubjectAdmission,
        event: CommitmentMutationEvidence,
    ) -> CommitmentRecord:
        """Consume the exact caller-supplied PROPOSED-to-ACTIVE admission."""

        with self._lock:
            self.snapshot()
            current = self._require_record(commitment_id)
            if not isinstance(admission, CommitmentSubjectAdmission):
                raise TypeError("admission must be CommitmentSubjectAdmission")
            if not isinstance(event, CommitmentMutationEvidence):
                raise TypeError("event must be CommitmentMutationEvidence")
            input_digest = _event_input_digest(
                operation=CommitmentSystemEventOperation.ACCEPT,
                commitment_id=current.commitment_id,
                proposal_digest=current.proposal_digest,
                evidence_refs=event.evidence_refs,
                admission_digest=admission.admission_digest,
            )
            prior = self._matching_event(
                event,
                CommitmentSystemEventOperation.ACCEPT,
                current.commitment_id,
                current.proposal_digest,
                input_digest,
            )
            if prior is not None:
                return current
            self._validate_admission_input(current, admission, event)
            self._check_new_event_capacity(event)
            if current.lifecycle is not CommitmentLifecycle.PROPOSED:
                raise CommitmentSystemError("only proposed Commitments may be accepted")
            if current.revision != 0 or current.revision_history[0].operation is not CommitmentRevisionOperation.CREATE:
                raise CommitmentSystemError("Commitment admission requires the exact genesis state")
            witnesses = tuple(
                sorted((*admission.evidence_refs, admission.admission_digest))
            )
            revision = _revision_after(
                current,
                event=event,
                operation=CommitmentRevisionOperation.ADMIT,
                reason=CommitmentRevisionReason.SUBJECT_ADMISSION,
                previous_state=CommitmentLifecycle.PROPOSED,
                evidence_refs=witnesses,
            )
            updated = replace(
                current,
                lifecycle=CommitmentLifecycle.ACTIVE,
                subject_admission=admission,
                revision=current.revision + 1,
                revision_history=_append_revision(current, revision),
            )
            receipt = self._make_receipt(
                event,
                CommitmentSystemEventOperation.ACCEPT,
                current.commitment_id,
                current.proposal_digest,
                input_digest,
                revision_witness=revision,
                admission_digest=admission.admission_digest,
                admission=admission,
            )
            self._replace_one(updated, receipt)
            return updated

    def release(
        self,
        commitment_id: str,
        proof: CommitmentSubjectTransitionProof,
        event: CommitmentMutationEvidence,
    ) -> CommitmentRecord:
        return self._subject_transition(
            commitment_id,
            proof,
            event,
            operation=CommitmentSystemEventOperation.RELEASE,
            revision_operation=CommitmentRevisionOperation.RELEASE,
            reason=CommitmentRevisionReason.SUBJECT_RELEASE,
            lifecycle=CommitmentLifecycle.RELEASED,
        )

    def renegotiate(
        self,
        commitment_id: str,
        proof: CommitmentSubjectTransitionProof,
        event: CommitmentMutationEvidence,
    ) -> CommitmentRecord:
        return self._subject_transition(
            commitment_id,
            proof,
            event,
            operation=CommitmentSystemEventOperation.RENEGOTIATE,
            revision_operation=CommitmentRevisionOperation.RENEGOTIATE,
            reason=CommitmentRevisionReason.SUBJECT_RENEGOTIATION,
            lifecycle=CommitmentLifecycle.RENEGOTIATED,
        )

    def _subject_transition(
        self,
        commitment_id: str,
        proof: CommitmentSubjectTransitionProof,
        event: CommitmentMutationEvidence,
        *,
        operation: CommitmentSystemEventOperation,
        revision_operation: CommitmentRevisionOperation,
        reason: CommitmentRevisionReason,
        lifecycle: CommitmentLifecycle,
    ) -> CommitmentRecord:
        with self._lock:
            self.snapshot()
            current = self._require_record(commitment_id)
            if not isinstance(proof, CommitmentSubjectTransitionProof):
                raise TypeError("proof must be CommitmentSubjectTransitionProof")
            if not isinstance(event, CommitmentMutationEvidence):
                raise TypeError("event must be CommitmentMutationEvidence")
            input_digest = _event_input_digest(
                operation=operation,
                commitment_id=current.commitment_id,
                proposal_digest=current.proposal_digest,
                evidence_refs=event.evidence_refs,
                transition_digest=proof.transition_digest,
            )
            prior = self._matching_event(
                event,
                operation,
                current.commitment_id,
                current.proposal_digest,
                input_digest,
            )
            if prior is not None:
                return current
            self._validate_transition_input(
                current,
                proof,
                event,
                operation=revision_operation,
                reason=reason,
            )
            self._check_new_event_capacity(event)
            if current.lifecycle is not CommitmentLifecycle.ACTIVE:
                raise CommitmentSystemError(
                    "only active Commitments may be released or renegotiated"
                )
            if current.subject_admission is None:
                raise CommitmentSystemError(
                    "active Commitment lacks its exact subject admission"
                )
            if proof.transition_digest in {
                item.transition_digest
                for item in current.subject_transition_proofs
            } or any(
                item.transition_digest == proof.transition_digest
                for item in self._event_receipts
            ):
                raise CommitmentSystemConflict(
                    "Commitment transition proof was already consumed"
                )
            revision = _revision_after(
                current,
                event=event,
                operation=revision_operation,
                reason=reason,
                previous_state=CommitmentLifecycle.ACTIVE,
                evidence_refs=tuple(
                    sorted((*proof.evidence_refs, proof.transition_digest))
                ),
            )
            proofs = tuple(
                sorted(
                    (*current.subject_transition_proofs, proof),
                    key=lambda item: item.transition_digest,
                )
            )
            updated = replace(
                current,
                lifecycle=lifecycle,
                subject_transition_proofs=proofs,
                revision=current.revision + 1,
                revision_history=_append_revision(current, revision),
            )
            receipt = self._make_receipt(
                event,
                operation,
                current.commitment_id,
                current.proposal_digest,
                input_digest,
                revision_witness=revision,
                transition_digest=proof.transition_digest,
                transition_proof=proof,
            )
            self._replace_one(updated, receipt)
            return updated

    @staticmethod
    def _validate_proposal_input(
        proposal: CommitmentRecord,
        event: CommitmentMutationEvidence,
    ) -> None:
        if not isinstance(proposal, CommitmentRecord):
            raise TypeError("proposal must be CommitmentRecord")
        if not isinstance(event, CommitmentMutationEvidence):
            raise TypeError("event must be CommitmentMutationEvidence")
        if proposal.lifecycle is not CommitmentLifecycle.PROPOSED:
            raise CommitmentSystemError("CommitmentSystem ingests proposal state only")
        if (
            proposal.subject_admission is not None
            or proposal.subject_transition_proofs
            or proposal.outcome_evidence_refs
            or proposal.revision != 0
            or proposal.history_anchor is not None
            or len(proposal.revision_history) != 1
        ):
            raise CommitmentSystemError(
                "proposal input contains pre-admitted or advanced Commitment state"
            )
        genesis = proposal.revision_history[0]
        if (
            genesis.operation is not CommitmentRevisionOperation.CREATE
            or genesis.reason is not CommitmentRevisionReason.CREATION
            or genesis.previous_lifecycle_state is not None
            or genesis.event_id is None
            or genesis.event_sequence is None
        ):
            raise CommitmentSystemError("proposal input lacks exact genesis evidence")
        if event.evidence_refs != proposal.evidence_refs:
            raise CommitmentSystemError(
                "proposal event evidence must exactly match Commitment evidence"
            )
        if genesis.evidence_refs != tuple(
            sorted(item.reference for item in proposal.evidence_refs)
        ):
            raise CommitmentSystemError(
                "proposal genesis must bind exact Commitment evidence"
            )
        if (
            genesis.event_sequence > event.event_sequence
            or genesis.created_at > event.recorded_at
        ):
            raise CommitmentSystemError(
                "proposal genesis is in the future of its ingestion event"
            )
        if (
            genesis.event_id == event.event_id
            or genesis.event_sequence == event.event_sequence
        ) and (
            genesis.event_id != event.event_id
            or genesis.event_sequence != event.event_sequence
            or genesis.created_at != event.recorded_at
        ):
            raise CommitmentSystemError(
                "proposal genesis event identity must match exactly"
            )

    @staticmethod
    def _validate_admission_input(
        current: CommitmentRecord,
        admission: CommitmentSubjectAdmission,
        event: CommitmentMutationEvidence,
    ) -> None:
        if not isinstance(admission, CommitmentSubjectAdmission):
            raise TypeError("admission must be CommitmentSubjectAdmission")
        if not isinstance(event, CommitmentMutationEvidence):
            raise TypeError("event must be CommitmentMutationEvidence")
        expected_evidence = tuple(sorted(item.reference for item in current.evidence_refs))
        if (
            admission.commitment_id != current.commitment_id
            or admission.proposal_digest != current.proposal_digest
            or admission.beneficiary != current.beneficiary
            or admission.scope != current.scope
            or admission.deadline != current.deadline
            or admission.evidence_refs != expected_evidence
            or admission.event_id != event.event_id
            or admission.event_sequence != event.event_sequence
            or event.evidence_refs != current.evidence_refs
        ):
            raise CommitmentSystemError(
                "Commitment subject admission does not match exact proposal/event evidence"
            )

    @staticmethod
    def _validate_transition_input(
        current: CommitmentRecord,
        proof: CommitmentSubjectTransitionProof,
        event: CommitmentMutationEvidence,
        *,
        operation: CommitmentRevisionOperation,
        reason: CommitmentRevisionReason,
    ) -> None:
        if not isinstance(proof, CommitmentSubjectTransitionProof):
            raise TypeError("proof must be CommitmentSubjectTransitionProof")
        if not isinstance(event, CommitmentMutationEvidence):
            raise TypeError("event must be CommitmentMutationEvidence")
        expected_evidence = tuple(sorted(item.reference for item in event.evidence_refs))
        if (
            proof.commitment_id != current.commitment_id
            or proof.proposal_digest != current.proposal_digest
            or proof.beneficiary != current.beneficiary
            or proof.scope != current.scope
            or proof.deadline != current.deadline
            or proof.operation is not operation
            or proof.reason is not reason
            or proof.previous_lifecycle_state is not CommitmentLifecycle.ACTIVE
            or proof.event_id != event.event_id
            or proof.event_sequence != event.event_sequence
            or proof.evidence_refs != expected_evidence
        ):
            raise CommitmentSystemError(
                "Commitment transition proof does not match exact authority/event evidence"
            )

    def _require_record(self, commitment_id: str) -> CommitmentRecord:
        current = self.get(commitment_id)
        if current is None:
            raise CommitmentSystemError("Commitment is not present")
        return current

    def _matching_event(
        self,
        event: CommitmentMutationEvidence,
        operation: CommitmentSystemEventOperation,
        commitment_id: str,
        proposal_digest: str,
        input_digest: str,
    ) -> CommitmentSystemEventReceipt | None:
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
            and existing.commitment_id == commitment_id
            and existing.proposal_digest == proposal_digest
            and existing.input_digest == input_digest
            and existing.evidence_refs == event.evidence_refs
        ):
            return existing
        raise CommitmentSystemConflict(
            "Commitment event identity was reused with conflicting input"
        )

    def _check_new_event_capacity(self, event: CommitmentMutationEvidence) -> None:
        if any(item.event_id == event.event_id for item in self._event_receipts):
            raise CommitmentSystemConflict("Commitment event ID was already used")
        if any(
            item.event_sequence == event.event_sequence
            for item in self._event_receipts
        ):
            raise CommitmentSystemConflict("Commitment event sequence was already used")
        if self._event_receipts:
            latest = self._event_receipts[-1]
            if event.event_sequence <= latest.event_sequence:
                raise CommitmentSystemError("Commitment event sequence regressed")
            if event.recorded_at < latest.recorded_at:
                raise CommitmentSystemError("Commitment event time regressed")
        if len(self._event_receipts) >= COMMITMENT_SYSTEM_MAX_EVENT_RECEIPTS:
            raise CommitmentSystemCapacityExceeded(
                "Commitment event receipt capacity is exhausted"
            )

    @staticmethod
    def _make_receipt(
        event: CommitmentMutationEvidence,
        operation: CommitmentSystemEventOperation,
        commitment_id: str,
        proposal_digest: str,
        input_digest: str,
        *,
        proposal_record_digest: str | None = None,
        proposal_genesis: CommitmentRevisionRecord | None = None,
        revision_witness: CommitmentRevisionRecord | None = None,
        admission_digest: str | None = None,
        admission: CommitmentSubjectAdmission | None = None,
        transition_digest: str | None = None,
        transition_proof: CommitmentSubjectTransitionProof | None = None,
    ) -> CommitmentSystemEventReceipt:
        return CommitmentSystemEventReceipt(
            event_id=event.event_id,
            event_sequence=event.event_sequence,
            recorded_at=event.recorded_at,
            operation=operation,
            commitment_id=commitment_id,
            proposal_digest=proposal_digest,
            input_digest=input_digest,
            evidence_refs=event.evidence_refs,
            proposal_record_digest=proposal_record_digest,
            proposal_genesis=proposal_genesis,
            revision_witness=revision_witness,
            admission_digest=admission_digest,
            admission=admission,
            transition_digest=transition_digest,
            transition_proof=transition_proof,
        )

    def _replace_one(
        self,
        record: CommitmentRecord,
        receipt: CommitmentSystemEventReceipt,
    ) -> None:
        records = tuple(
            sorted(
                (
                    record if item.commitment_id == record.commitment_id else item
                    for item in self._records
                ),
                key=lambda item: item.commitment_id,
            )
        )
        self._replace_state(
            records=records,
            event_receipts=(*self._event_receipts, receipt),
        )

    def _replace_state(
        self,
        *,
        records: tuple[CommitmentRecord, ...] | None = None,
        event_receipts: tuple[CommitmentSystemEventReceipt, ...] | None = None,
    ) -> None:
        proposed = CommitmentSystemSnapshot(
            records=self._records if records is None else records,
            event_receipts=(
                self._event_receipts if event_receipts is None else event_receipts
            ),
        )
        self._records = proposed.records
        self._event_receipts = proposed.event_receipts


__all__ = [
    "COMMITMENT_SYSTEM_DOMAIN",
    "COMMITMENT_SYSTEM_EVENT_DOMAIN",
    "COMMITMENT_SYSTEM_MAX_EVENT_RECEIPTS",
    "COMMITMENT_SYSTEM_MAX_EVIDENCE_REFS_PER_EVENT",
    "COMMITMENT_SYSTEM_MAX_RECORDS",
    "COMMITMENT_SYSTEM_MAX_SERIALIZED_BYTES",
    "COMMITMENT_SYSTEM_SCHEMA_VERSION",
    "CommitmentMutationEvidence",
    "CommitmentSystem",
    "CommitmentSystemCapacityExceeded",
    "CommitmentSystemConflict",
    "CommitmentSystemError",
    "CommitmentSystemEventOperation",
    "CommitmentSystemEventReceipt",
    "CommitmentSystemSnapshot",
]
