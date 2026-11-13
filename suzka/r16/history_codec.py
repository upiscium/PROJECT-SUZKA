"""R16 U4-1: canonical SubjectHistory values (#376 v1, IP-9).

Pure, bounded, immutable values for the one ``subject.history`` R07
participant: the outer ``CommittedDecisionV1`` issuer envelope around an
unchanged U1 ``DecisionRecord`` (#371 D-P1-7), its typed source references and
the post-internal-commit receipt (#376 D1-3).

These values own no authority on their own.  A checksum binds bytes, not an
issuer: a Decision is adopted only by a FINALIZED SubjectHistory row whose
R07 transaction and event are terminal in the retained Journal (#376 §C).
HOLD is not representable here; R16 HOLD changes no state (#354) and its
receipt belongs to the later R19 additive schema (#376 v2 §2).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Final, Self
from uuid import UUID

from suzka.limits import MAX_PERSISTED_EVENT_SEQUENCE, MAX_PERSISTED_REVISION
from suzka.r16.contracts import (
    MAX_JSON_BYTES,
    DecisionRecord,
    ReasonCode,
    canonical_json,
    checksum,
    closed,
    digest,
    identifier,
    integer,
    parse_enum,
    parse_json,
    sequence,
)


SUBJECT_HISTORY_PARTICIPANT_ID: Final = "subject.history"
HISTORY_VALUE_VERSION: Final = 1
GENESIS: Final = "GENESIS"
MAX_POLICY_INPUT_BYTES: Final = 8_192
MAX_PENDING_BYTES: Final = 131_072
MAX_SOURCE_REFS: Final = 32
DECISION_ID_PREFIX: Final = "decision-"

_DECISION_DOMAIN: Final = b"PROJECT-SUZKA:R16:COMMITTED-DECISION:V1\0"
_DECISION_ID_DOMAIN: Final = b"PROJECT-SUZKA:R16:COMMITTED-DECISION-ID:V1\0"
_OPERATION_DOMAIN: Final = b"PROJECT-SUZKA:R16:SUBJECT-HISTORY-OPERATION:V1\0"
_RECEIPT_DOMAIN: Final = b"PROJECT-SUZKA:R16:SUBJECT-HISTORY-POSTCOMMIT:V1\0"


class PolicyKind(str, Enum):
    SUCCESSOR_POLICY_V1 = "SUCCESSOR_POLICY_V1"
    SUCCESSOR_POLICY_V2 = "SUCCESSOR_POLICY_V2"


class PolicyRow(str, Enum):
    """Policy rows that issue a Decision; HOLD rows (0/1/2/2b/7) never do."""

    ROW_3 = "3"
    ROW_4 = "4"
    ROW_4B = "4b"
    ROW_5 = "5"
    ROW_5B = "5b"
    ROW_6 = "6"


# #374 F1 D3 §3-2: the "b" rows exist only in SuccessorPolicyV2, and row 5 is
# the only Decision row whose output is NO_OP.
_V2_ONLY_ROWS: Final = frozenset({PolicyRow.ROW_4B, PolicyRow.ROW_5B})


class SourceRole(str, Enum):
    GOAL = "GOAL"
    VALUE = "VALUE"
    CONTEXT = "CONTEXT"
    EXPERIENCE = "EXPERIENCE"
    COMMITMENT = "COMMITMENT"
    FEEDBACK = "FEEDBACK"
    OUTCOME = "OUTCOME"
    R15_NARRATIVE_REVISION = "R15_NARRATIVE_REVISION"
    MOTIVATION = "MOTIVATION"


def canonical_uuid(value: object, name: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be a canonical UUID string")
    try:
        parsed = UUID(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a canonical UUID string") from error
    if str(parsed) != value:
        raise ValueError(f"{name} must be a canonical UUID string")
    return value


def event_sequence(value: object, name: str) -> int:
    return integer(value, name, minimum=1, maximum=MAX_PERSISTED_EVENT_SEQUENCE)


def decision_id_for(
    incarnation_id: str,
    issuer_event_id: str,
    issuer_sequence: int,
    review_intent_key: str,
    policy_input_digest: str,
) -> str:
    """#376 D1-4: domain-separated, incarnation-bound Decision identity."""

    body = [
        digest(incarnation_id, "incarnation_id"),
        canonical_uuid(issuer_event_id, "issuer_event_id"),
        event_sequence(issuer_sequence, "issuer_sequence"),
        digest(review_intent_key, "review_intent_key"),
        digest(policy_input_digest, "policy_input_digest"),
    ]
    return DECISION_ID_PREFIX + checksum(_DECISION_ID_DOMAIN, body)


def decision_id(value: object, name: str) -> str:
    if (
        type(value) is not str
        or not value.startswith(DECISION_ID_PREFIX)
        or len(value) != len(DECISION_ID_PREFIX) + 64
    ):
        raise ValueError(f"{name} must be a SubjectHistory Decision identity")
    digest(value[len(DECISION_ID_PREFIX) :], name)
    return value


def policy_input_digest(payload: bytes) -> str:
    return sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class HistorySourceRef:
    """A digest-bound source reference; never a fetched or qualified root."""

    role: SourceRole
    kind: str
    ref_id: str
    revision: int
    digest: str

    def __post_init__(self) -> None:
        if type(self.role) is not SourceRole:
            raise TypeError("source role must be a SourceRole")
        identifier(self.kind, "source kind")
        identifier(self.ref_id, "source ref_id")
        integer(self.revision, "source revision", maximum=MAX_PERSISTED_REVISION)
        digest(self.digest, "source digest")

    def canonical_value(self) -> dict[str, object]:
        return {
            "digest": self.digest,
            "kind": self.kind,
            "ref_id": self.ref_id,
            "revision": self.revision,
            "role": self.role.value,
        }

    @property
    def sort_key(self) -> tuple[str, str]:
        return (self.role.value, self.ref_id)

    @classmethod
    def from_value(cls, value: object) -> Self:
        row = closed(value, ("digest", "kind", "ref_id", "revision", "role"), "source")
        return cls(
            parse_enum(row["role"], SourceRole, "source role"),
            identifier(row["kind"], "source kind"),
            identifier(row["ref_id"], "source ref_id"),
            integer(row["revision"], "source revision", maximum=MAX_PERSISTED_REVISION),
            digest(row["digest"], "source digest"),
        )


_DECISION_FIELDS: Final = (
    "decision",
    "decision_id",
    "incarnation_id",
    "issuer_event_id",
    "issuer_sequence",
    "lineage_scope",
    "policy_input",
    "policy_kind",
    "policy_row",
    "predecessor_digest",
    "predecessor_key",
    "review_intent_key",
    "schema_version",
    "source_refs",
)


@dataclass(frozen=True, slots=True)
class CommittedDecisionV1:
    """The immutable outer issuer envelope persisted by ``subject.history``.

    The envelope binds the unchanged U1 ``DecisionRecord`` to one accepted
    event, one incarnation, one lineage predecessor, one policy row and the
    exact sealed policy input.  Validation rejects internally inconsistent
    values only; it cannot tell whether the inputs were fresh or whether the
    caller was the trusted issuer (#376 §B).
    """

    incarnation_id: str
    decision_id: str
    issuer_event_id: str
    issuer_sequence: int
    lineage_scope: str
    review_intent_key: str
    predecessor_key: str
    predecessor_digest: str | None
    decision: DecisionRecord
    policy_kind: PolicyKind
    policy_row: PolicyRow
    policy_input_bytes: bytes
    source_refs: tuple[HistorySourceRef, ...]

    def __post_init__(self) -> None:
        digest(self.incarnation_id, "incarnation_id")
        decision_id(self.decision_id, "decision_id")
        canonical_uuid(self.issuer_event_id, "issuer_event_id")
        event_sequence(self.issuer_sequence, "issuer_sequence")
        identifier(self.lineage_scope, "lineage_scope")
        digest(self.review_intent_key, "review_intent_key")
        if self.predecessor_key == GENESIS:
            if self.predecessor_digest is not None:
                raise ValueError("a GENESIS Decision has no predecessor digest")
        else:
            decision_id(self.predecessor_key, "predecessor_key")
            digest(self.predecessor_digest, "predecessor_digest")
            if self.predecessor_key == self.decision_id:
                raise ValueError("a Decision cannot be its own predecessor")
        if type(self.decision) is not DecisionRecord:
            raise TypeError("decision must be an exact U1 DecisionRecord")
        self.decision.__post_init__()
        if self.decision.decision_id != self.decision_id:
            raise ValueError("U1 decision identity must equal the committed identity")
        if type(self.policy_kind) is not PolicyKind:
            raise TypeError("policy_kind must be a PolicyKind")
        if type(self.policy_row) is not PolicyRow:
            raise TypeError("policy_row must be a PolicyRow")
        if (
            self.policy_kind is PolicyKind.SUCCESSOR_POLICY_V1
            and self.policy_row in _V2_ONLY_ROWS
        ):
            raise ValueError("SuccessorPolicyV1 has no b rows")
        expected_reason = (
            ReasonCode.NO_OP_PREFERRED
            if self.policy_row is PolicyRow.ROW_5
            else ReasonCode.PLAN_PREFERRED
        )
        if self.decision.reason is not expected_reason:
            raise ValueError("U1 decision reason does not match its policy row")
        if type(self.policy_input_bytes) is not bytes:
            raise TypeError("policy_input_bytes must be bytes")
        policy_input = parse_json(self.policy_input_bytes, MAX_POLICY_INPUT_BYTES)
        if type(policy_input) is not dict:
            raise ValueError("policy input must be one canonical JSON object")
        sequence(self.source_refs, "source_refs", MAX_SOURCE_REFS, json_input=False)
        if not self.source_refs:
            raise ValueError("a committed Decision needs at least one source")
        for item in self.source_refs:
            if type(item) is not HistorySourceRef:
                raise TypeError("source_refs must be exact HistorySourceRef values")
            item.__post_init__()
        keys = tuple(item.sort_key for item in self.source_refs)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("source refs must be sorted and unique by role and ID")
        expected_id = decision_id_for(
            self.incarnation_id,
            self.issuer_event_id,
            self.issuer_sequence,
            self.review_intent_key,
            policy_input_digest(self.policy_input_bytes),
        )
        if self.decision_id != expected_id:
            raise ValueError("decision_id does not bind its event and policy input")
        if len(self.canonical_bytes()) > MAX_PENDING_BYTES:
            raise ValueError("committed Decision exceeds its pending byte bound")

    def _body(self) -> dict[str, object]:
        return {
            "decision": self.decision.canonical_value(),
            "decision_id": self.decision_id,
            "incarnation_id": self.incarnation_id,
            "issuer_event_id": self.issuer_event_id,
            "issuer_sequence": self.issuer_sequence,
            "lineage_scope": self.lineage_scope,
            "policy_input": parse_json(self.policy_input_bytes, MAX_POLICY_INPUT_BYTES),
            "policy_kind": self.policy_kind.value,
            "policy_row": self.policy_row.value,
            "predecessor_digest": self.predecessor_digest,
            "predecessor_key": self.predecessor_key,
            "review_intent_key": self.review_intent_key,
            "schema_version": HISTORY_VALUE_VERSION,
            "source_refs": [item.canonical_value() for item in self.source_refs],
        }

    def canonical_value(self) -> dict[str, object]:
        body = self._body()
        return {**body, "content_digest": checksum(_DECISION_DOMAIN, body)}

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.canonical_value())

    @property
    def content_digest(self) -> str:
        return checksum(_DECISION_DOMAIN, self._body())

    @property
    def decision_bytes(self) -> bytes:
        return self.decision.canonical_bytes()

    @property
    def policy_input_digest(self) -> str:
        return policy_input_digest(self.policy_input_bytes)

    @property
    def is_genesis(self) -> bool:
        return self.predecessor_key == GENESIS

    @classmethod
    def from_json(cls, payload: bytes) -> Self:
        row = closed(
            parse_json(payload, MAX_PENDING_BYTES),
            (*_DECISION_FIELDS, "content_digest"),
            "CommittedDecisionV1",
        )
        integer(
            row["schema_version"],
            "schema_version",
            minimum=HISTORY_VALUE_VERSION,
            maximum=HISTORY_VALUE_VERSION,
        )
        claimed = digest(row["content_digest"], "content_digest")
        decision = DecisionRecord.from_value(row["decision"])
        if len(decision.canonical_bytes()) > MAX_JSON_BYTES:
            raise ValueError("U1 decision exceeds its byte bound")
        predecessor = row["predecessor_digest"]
        result = cls(
            digest(row["incarnation_id"], "incarnation_id"),
            decision_id(row["decision_id"], "decision_id"),
            canonical_uuid(row["issuer_event_id"], "issuer_event_id"),
            event_sequence(row["issuer_sequence"], "issuer_sequence"),
            identifier(row["lineage_scope"], "lineage_scope"),
            digest(row["review_intent_key"], "review_intent_key"),
            _predecessor_key(row["predecessor_key"]),
            None if predecessor is None else digest(predecessor, "predecessor_digest"),
            decision,
            parse_enum(row["policy_kind"], PolicyKind, "policy_kind"),
            parse_enum(row["policy_row"], PolicyRow, "policy_row"),
            canonical_json(row["policy_input"]),
            tuple(
                HistorySourceRef.from_value(item)
                for item in sequence(
                    row["source_refs"], "source_refs", MAX_SOURCE_REFS, json_input=True
                )
            ),
        )
        if result.content_digest != claimed:
            raise ValueError("committed Decision digest does not bind its value")
        if result.canonical_bytes() != payload:
            raise ValueError("committed Decision is not canonical")
        return result


def _predecessor_key(value: object) -> str:
    if value == GENESIS:
        return GENESIS
    return decision_id(value, "predecessor_key")


def history_operation_digest(decision: CommittedDecisionV1) -> str:
    """The R07 operation digest of one ``R16_DECISION_ISSUE`` operation."""

    return sha256(_OPERATION_DOMAIN + decision.canonical_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class PostcommitReceiptV1:
    """Read-only R06 proof captured before the SQLite FINALIZE (#376 D1-3).

    It holds only evidence that existed before the FINALIZE: never the later
    PARTICIPANT_FINALIZED, TRANSACTION_COMPLETED/RECONCILED or event terminal.
    It is derivative of the live validator, not self-attesting authority.
    """

    transaction_id: str
    event_id: str
    processing_sequence: int
    prepared_state_hash_before: str
    prepared_state_hash_after: str
    snapshot_sequence: int
    snapshot_hash: str
    wal_generation_id: str
    wal_record_id: str
    wal_record_hash: str

    def __post_init__(self) -> None:
        canonical_uuid(self.transaction_id, "transaction_id")
        canonical_uuid(self.event_id, "event_id")
        event_sequence(self.processing_sequence, "processing_sequence")
        digest(self.prepared_state_hash_before, "prepared_state_hash_before")
        digest(self.prepared_state_hash_after, "prepared_state_hash_after")
        integer(
            self.snapshot_sequence,
            "snapshot_sequence",
            maximum=MAX_PERSISTED_EVENT_SEQUENCE,
        )
        digest(self.snapshot_hash, "snapshot_hash")
        canonical_uuid(self.wal_generation_id, "wal_generation_id")
        canonical_uuid(self.wal_record_id, "wal_record_id")
        digest(self.wal_record_hash, "wal_record_hash")
        if self.snapshot_hash != self.prepared_state_hash_after:
            raise ValueError("committed snapshot must equal the prepared after-state")

    def canonical_value(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "prepared_state_hash_after": self.prepared_state_hash_after,
            "prepared_state_hash_before": self.prepared_state_hash_before,
            "processing_sequence": self.processing_sequence,
            "snapshot_hash": self.snapshot_hash,
            "snapshot_sequence": self.snapshot_sequence,
            "transaction_id": self.transaction_id,
            "wal_generation_id": self.wal_generation_id,
            "wal_record_hash": self.wal_record_hash,
            "wal_record_id": self.wal_record_id,
        }

    @property
    def receipt_digest(self) -> str:
        return checksum(_RECEIPT_DOMAIN, self.canonical_value())


__all__ = [
    "DECISION_ID_PREFIX",
    "GENESIS",
    "HISTORY_VALUE_VERSION",
    "MAX_PENDING_BYTES",
    "MAX_POLICY_INPUT_BYTES",
    "MAX_SOURCE_REFS",
    "SUBJECT_HISTORY_PARTICIPANT_ID",
    "CommittedDecisionV1",
    "HistorySourceRef",
    "PolicyKind",
    "PolicyRow",
    "PostcommitReceiptV1",
    "SourceRole",
    "canonical_uuid",
    "decision_id_for",
    "history_operation_digest",
    "policy_input_digest",
]
