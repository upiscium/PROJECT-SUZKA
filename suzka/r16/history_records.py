"""IP-14a-1: SubjectHistory v2 record operations (#411 comment 6910958369 §2-§3).

One R17 / R18 subject.history operation carries exactly one typed record in
a ``HistoryRecordOperationV1`` envelope (its ``pending_bytes``). The
envelope binds the record to its R07 operation (incarnation, event,
sequence, transaction kind); every record that carries its own event fields
must name the same event. From the envelope this module derives, without
any store access:

- the table row (record id, bytes, digest and key columns) FINALIZE writes;
- the keys PREPARE reserves (§2a), so uniqueness is decided before R07
  commits, not at FINALIZE.

Pure: no store, Journal or clock. Validation rejects inconsistent values;
whether a writer was allowed to issue them is the writer's (IP-16..IP-19).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Final, Self

from suzka.r16.contracts import (
    canonical_json,
    checksum,
    closed,
    digest,
    identifier,
    integer,
    parse_enum,
    parse_json,
)
from suzka.r16.history_codec import MAX_PENDING_BYTES, canonical_uuid, event_sequence
from suzka.r17.contracts import DecisionFeedbackV1, ExpectedObservationV1, OutcomeAssessmentV1
from suzka.r18.contracts import (
    ActionIntentV1,
    ApprovalChallengeV1,
    Direction,
    ExecutionClaimV1,
    InverseIntentV1,
    PolicyVerdictV1,
    VerifiedOutcomeV1,
)

ENVELOPE_SCHEMA_VERSION: Final = 1
# Human decision §11-3 (2026-11-16): a new challenge may follow a recorded
# expiry, at most three per intent.
MAX_CHALLENGES_PER_INTENT: Final = 3
_OPERATION_DOMAIN_V2: Final = b"PROJECT-SUZKA:R16:SUBJECT-HISTORY-RECORD-OPERATION:V2\0"
_EXPIRATION_DOMAIN: Final = b"PROJECT-SUZKA:R18:APPROVAL-EXPIRATION:V1\0"


class HistoryOperationKind(str, Enum):
    R16_DECISION_ISSUE = "R16_DECISION_ISSUE"
    R07_BINDING_ONLY_ABORT = "R07_BINDING_ONLY_ABORT"
    R18_INTENT_ISSUE = "R18_INTENT_ISSUE"
    R17_EXPECTATION_COMMIT = "R17_EXPECTATION_COMMIT"
    R18_POLICY_COMMIT = "R18_POLICY_COMMIT"
    R18_CHALLENGE_ISSUE = "R18_CHALLENGE_ISSUE"
    R18_APPROVAL_EXPIRE = "R18_APPROVAL_EXPIRE"
    R18_EXECUTION_CLAIM = "R18_EXECUTION_CLAIM"
    R18_OUTCOME_COMMIT = "R18_OUTCOME_COMMIT"
    R17_ASSESSMENT_COMMIT = "R17_ASSESSMENT_COMMIT"
    R17_FEEDBACK_COMMIT = "R17_FEEDBACK_COMMIT"


R16_OPERATION_KINDS: Final = frozenset(
    {HistoryOperationKind.R16_DECISION_ISSUE, HistoryOperationKind.R07_BINDING_ONLY_ABORT}
)
RECORD_OPERATION_KINDS: Final = frozenset(set(HistoryOperationKind) - set(R16_OPERATION_KINDS))
# r18.enabled gates every R17 / R18 kind (R17 follows R18 Outcomes, §7).
R18_GATED_KINDS: Final = RECORD_OPERATION_KINDS


class RecordKind(str, Enum):
    ACTION_INTENT = "ACTION_INTENT"
    INVERSE_INTENT = "INVERSE_INTENT"
    ACTION_FORECAST = "ACTION_FORECAST"
    POLICY_VERDICT = "POLICY_VERDICT"
    APPROVAL_CHALLENGE = "APPROVAL_CHALLENGE"
    APPROVAL_EXPIRATION = "APPROVAL_EXPIRATION"
    EXECUTION_CLAIM = "EXECUTION_CLAIM"
    VERIFIED_OUTCOME = "VERIFIED_OUTCOME"
    OUTCOME_ASSESSMENT = "OUTCOME_ASSESSMENT"
    DECISION_FEEDBACK = "DECISION_FEEDBACK"


OPERATION_RECORD_KINDS: Final[Mapping[HistoryOperationKind, frozenset[RecordKind]]] = {
    HistoryOperationKind.R18_INTENT_ISSUE: frozenset({RecordKind.ACTION_INTENT, RecordKind.INVERSE_INTENT}),
    HistoryOperationKind.R17_EXPECTATION_COMMIT: frozenset({RecordKind.ACTION_FORECAST}),
    HistoryOperationKind.R18_POLICY_COMMIT: frozenset({RecordKind.POLICY_VERDICT}),
    HistoryOperationKind.R18_CHALLENGE_ISSUE: frozenset({RecordKind.APPROVAL_CHALLENGE}),
    HistoryOperationKind.R18_APPROVAL_EXPIRE: frozenset({RecordKind.APPROVAL_EXPIRATION}),
    HistoryOperationKind.R18_EXECUTION_CLAIM: frozenset({RecordKind.EXECUTION_CLAIM}),
    HistoryOperationKind.R18_OUTCOME_COMMIT: frozenset({RecordKind.VERIFIED_OUTCOME}),
    HistoryOperationKind.R17_ASSESSMENT_COMMIT: frozenset({RecordKind.OUTCOME_ASSESSMENT}),
    HistoryOperationKind.R17_FEEDBACK_COMMIT: frozenset({RecordKind.DECISION_FEEDBACK}),
}
# The tables each operation kind writes at FINALIZE (§3); used by the
# finalize_complete trigger and the startup check.
OPERATION_TABLES: Final[Mapping[HistoryOperationKind, tuple[str, ...]]] = {
    HistoryOperationKind.R18_INTENT_ISSUE: ("action_intents",),
    HistoryOperationKind.R17_EXPECTATION_COMMIT: ("action_forecasts",),
    HistoryOperationKind.R18_POLICY_COMMIT: ("policy_verdicts",),
    HistoryOperationKind.R18_CHALLENGE_ISSUE: ("approval_challenges",),
    HistoryOperationKind.R18_APPROVAL_EXPIRE: ("approval_expirations",),
    HistoryOperationKind.R18_EXECUTION_CLAIM: ("approval_nonce_consumptions", "execution_claims"),
    HistoryOperationKind.R18_OUTCOME_COMMIT: ("verified_outcomes",),
    HistoryOperationKind.R17_ASSESSMENT_COMMIT: ("outcome_assessments",),
    HistoryOperationKind.R17_FEEDBACK_COMMIT: ("decision_feedback",),
}
RECORD_TABLES: Final = tuple(
    dict.fromkeys(table for tables in OPERATION_TABLES.values() for table in tables)
)


class ExpirationReason(str, Enum):
    EXPIRED = "EXPIRED"
    # The challenge's boot ended (#380 P2-2): a restart voids it.
    BOOT_CHANGED = "BOOT_CHANGED"


@dataclass(frozen=True, slots=True)
class ApprovalExpirationV1:
    """A challenge that ended without a Claim, recorded once (§2 item 3).

    Monotonic by construction: once recorded, the Claim gate refuses the
    challenge even if the clock later moves back."""

    challenge_digest: str
    intent_id: str
    direction: Direction
    reason: ExpirationReason

    def __post_init__(self) -> None:
        digest(self.challenge_digest, "challenge_digest")
        identifier(self.intent_id, "intent_id")
        if type(self.direction) is not Direction:
            raise TypeError("direction must be a Direction")
        if type(self.reason) is not ExpirationReason:
            raise TypeError("reason must be an ExpirationReason")

    @classmethod
    def for_challenge(cls, challenge: ApprovalChallengeV1, reason: ExpirationReason) -> Self:
        if type(challenge) is not ApprovalChallengeV1:
            raise TypeError("challenge must be an ApprovalChallengeV1")
        return cls(challenge.challenge_digest, challenge.intent_id, challenge.direction, reason)

    def canonical_value(self) -> dict[str, object]:
        return {
            "challenge_digest": self.challenge_digest,
            "direction": self.direction.value,
            "intent_id": self.intent_id,
            "reason": self.reason.value,
            "schema_version": 1,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.canonical_value())

    @property
    def expiration_digest(self) -> str:
        return checksum(_EXPIRATION_DOMAIN, self.canonical_value())

    @classmethod
    def from_json(cls, payload: bytes) -> Self:
        row = closed(
            parse_json(payload, 4096),
            ("challenge_digest", "direction", "intent_id", "reason", "schema_version"),
            "approval expiration",
        )
        if row["schema_version"] != 1 or type(row["schema_version"]) is not int:
            raise ValueError("approval expiration schema_version is not V1")
        value = cls(
            _str(row["challenge_digest"]),
            _str(row["intent_id"]),
            parse_enum(row["direction"], Direction, "direction"),
            parse_enum(row["reason"], ExpirationReason, "reason"),
        )
        if value.canonical_bytes() != payload:
            raise ValueError("approval expiration bytes are not canonical")
        return value


def _str(value: object) -> str:
    if type(value) is not str:
        raise TypeError("expected a string")
    return value


def _int(value: object) -> int:
    if type(value) is not int:
        raise TypeError("expected an integer")
    return value


# ----------------------------------------------------------- derived rows


@dataclass(frozen=True, slots=True)
class RecordRow:
    """One row a FINALIZE writes: the table, its columns and their values."""

    table: str
    columns: tuple[tuple[str, object], ...]

    def value(self, name: str) -> object:
        return dict(self.columns)[name]


@dataclass(frozen=True, slots=True)
class DecodedRecord:
    record_kind: RecordKind
    record: object
    rows: tuple[RecordRow, ...]
    reserved_keys: tuple[str, ...]
    event: tuple[str, int] | None


def _common(record_id: str, record_bytes: bytes, record_digest: str) -> tuple[tuple[str, object], ...]:
    return (("record_id", record_id), ("record_bytes", record_bytes), ("record_digest", record_digest))


def _intent(value: ActionIntentV1) -> DecodedRecord:
    payload = value.canonical_bytes()
    row = RecordRow(
        "action_intents",
        (
            *_common(value.intent_id, payload, value.intent_digest),
            ("direction", Direction.FORWARD.value),
            ("obligation_group_key", value.obligation_group_key),
            ("operation_key", value.forward_key),
            ("forward_operation_key", value.forward_key),
            ("decision_id", value.decision_id),
        ),
    )
    return DecodedRecord(
        RecordKind.ACTION_INTENT,
        value,
        (row,),
        # IP-16 D1 §9 (#373 A1-5): Intents are unique per incarnation; only
        # Claims, nonces and Outcomes are unique across incarnations.
        (
            f"operation:{value.incarnation_id}:{value.forward_key}",
            f"group:{value.incarnation_id}:{value.obligation_group_key}:FORWARD",
        ),
        (value.admitting_event_id, value.admitting_sequence),
    )


def _inverse(value: InverseIntentV1) -> DecodedRecord:
    payload = value.canonical_bytes()
    row = RecordRow(
        "action_intents",
        (
            *_common(value.intent_id, payload, value.intent_digest),
            ("direction", Direction.INVERSE.value),
            ("obligation_group_key", value.obligation_group_key),
            ("operation_key", value.inverse_key),
            ("forward_operation_key", value.forward_key),
            ("decision_id", None),
        ),
    )
    return DecodedRecord(
        RecordKind.INVERSE_INTENT,
        value,
        (row,),
        (
            f"operation:{value.incarnation_id}:{value.inverse_key}",
            f"group:{value.incarnation_id}:{value.obligation_group_key}:INVERSE",
        ),
        (value.admitting_event_id, value.admitting_sequence),
    )


def _forecast(value: ExpectedObservationV1) -> DecodedRecord:
    row = RecordRow(
        "action_forecasts",
        (
            *_common(value.forecast_id, value.canonical_bytes(), value.forecast_digest),
            ("action_intent_id", value.action_intent_id),
            ("forward_key", value.forward_key),
        ),
    )
    return DecodedRecord(
        RecordKind.ACTION_FORECAST,
        value,
        (row,),
        (
            f"forecast-intent:{value.action_intent_id}",
            f"forecast-key:{value.incarnation_id}:{value.forward_key}",
        ),
        (value.admitting_event_id, value.admitting_sequence),
    )


def _verdict(value: PolicyVerdictV1) -> DecodedRecord:
    row = RecordRow(
        "policy_verdicts",
        (
            *_common(value.verdict_digest, value.canonical_bytes(), value.verdict_digest),
            ("intent_id", value.intent_id),
            ("direction", value.direction.value),
            ("verdict", value.verdict.value),
        ),
    )
    return DecodedRecord(
        RecordKind.POLICY_VERDICT,
        value,
        (row,),
        (f"verdict:{value.intent_id}",),
        (value.policy_event_id, value.policy_sequence),
    )


def _challenge(value: ApprovalChallengeV1, attempt: int) -> DecodedRecord:
    row = RecordRow(
        "approval_challenges",
        (
            *_common(value.challenge_digest, value.canonical_bytes(), value.challenge_digest),
            ("intent_id", value.intent_id),
            ("direction", value.direction.value),
            ("attempt", attempt),
            ("nonce", value.nonce),
            ("operation_key", value.operation_key),
            ("boot_id", value.boot_id),
            ("expires_utc", value.expires_utc),
        ),
    )
    return DecodedRecord(
        RecordKind.APPROVAL_CHALLENGE,
        value,
        (row,),
        (f"challenge:{value.intent_id}:{attempt}", f"nonce:{value.nonce}"),
        None,
    )


def _expiration(value: ApprovalExpirationV1) -> DecodedRecord:
    row = RecordRow(
        "approval_expirations",
        (
            *_common(value.challenge_digest, value.canonical_bytes(), value.expiration_digest),
            ("intent_id", value.intent_id),
            ("direction", value.direction.value),
            ("reason", value.reason.value),
        ),
    )
    return DecodedRecord(
        RecordKind.APPROVAL_EXPIRATION,
        value,
        (row,),
        (f"challenge-end:{value.challenge_digest}",),
        None,
    )


def _claim(value: ExecutionClaimV1) -> DecodedRecord:
    consumption = RecordRow(
        "approval_nonce_consumptions",
        (
            ("nonce", value.nonce),
            ("challenge_digest", value.challenge_digest),
            ("claim_id", value.claim_id),
        ),
    )
    claim = RecordRow(
        "execution_claims",
        (
            *_common(value.claim_id, value.canonical_bytes(), value.claim_digest),
            ("direction", value.direction.value),
            ("operation_key", value.operation_key),
            ("nonce", value.nonce),
            ("challenge_digest", value.challenge_digest),
        ),
    )
    return DecodedRecord(
        RecordKind.EXECUTION_CLAIM,
        value,
        (consumption, claim),
        (f"challenge-end:{value.challenge_digest}", f"claim:{value.operation_key}"),
        (value.claim_event_id, value.claim_sequence),
    )


def _outcome(value: VerifiedOutcomeV1) -> DecodedRecord:
    row = RecordRow(
        "verified_outcomes",
        (
            ("verified_outcome_id", value.verified_outcome_id),
            ("outcome_revision", value.outcome_revision),
            ("record_bytes", value.canonical_bytes()),
            ("record_digest", value.outcome_digest),
            ("direction", value.direction.value),
            ("operation_key", value.operation_key),
            ("outcome_state", value.outcome_state.value),
        ),
    )
    return DecodedRecord(
        RecordKind.VERIFIED_OUTCOME,
        value,
        (row,),
        (f"outcome:{value.operation_key}:{value.outcome_revision}",),
        (value.verifier_event_id, value.verifier_sequence),
    )


def _assessment(value: OutcomeAssessmentV1) -> DecodedRecord:
    row = RecordRow(
        "outcome_assessments",
        (
            *_common(value.assessment_id, value.canonical_bytes(), value.assessment_digest),
            ("verified_outcome_id", value.verified_outcome_id),
            ("outcome_revision", value.outcome_revision),
            ("forecast_id", value.forecast_id),
        ),
    )
    return DecodedRecord(
        RecordKind.OUTCOME_ASSESSMENT,
        value,
        (row,),
        (f"assessment:{value.verified_outcome_id}:{value.outcome_revision}",),
        (value.admitting_event_id, value.admitting_sequence),
    )


def _feedback(value: DecisionFeedbackV1) -> DecodedRecord:
    row = RecordRow(
        "decision_feedback",
        (
            *_common(value.feedback_id, value.canonical_bytes(), value.feedback_digest),
            ("assessment_id", value.assessment_id),
        ),
    )
    return DecodedRecord(
        RecordKind.DECISION_FEEDBACK,
        value,
        (row,),
        (f"feedback:{value.assessment_id}",),
        (value.admitting_event_id, value.admitting_sequence),
    )


_DECODERS: Final[Mapping[RecordKind, Callable[[bytes], object]]] = {
    RecordKind.ACTION_INTENT: ActionIntentV1.from_json,
    RecordKind.INVERSE_INTENT: InverseIntentV1.from_json,
    RecordKind.ACTION_FORECAST: ExpectedObservationV1.from_json,
    RecordKind.POLICY_VERDICT: PolicyVerdictV1.from_json,
    RecordKind.APPROVAL_CHALLENGE: ApprovalChallengeV1.from_json,
    RecordKind.APPROVAL_EXPIRATION: ApprovalExpirationV1.from_json,
    RecordKind.EXECUTION_CLAIM: ExecutionClaimV1.from_json,
    RecordKind.VERIFIED_OUTCOME: VerifiedOutcomeV1.from_json,
    RecordKind.OUTCOME_ASSESSMENT: OutcomeAssessmentV1.from_json,
    RecordKind.DECISION_FEEDBACK: DecisionFeedbackV1.from_json,
}
_TYPES: Final[Mapping[RecordKind, type]] = {
    RecordKind.ACTION_INTENT: ActionIntentV1,
    RecordKind.INVERSE_INTENT: InverseIntentV1,
    RecordKind.ACTION_FORECAST: ExpectedObservationV1,
    RecordKind.POLICY_VERDICT: PolicyVerdictV1,
    RecordKind.APPROVAL_CHALLENGE: ApprovalChallengeV1,
    RecordKind.APPROVAL_EXPIRATION: ApprovalExpirationV1,
    RecordKind.EXECUTION_CLAIM: ExecutionClaimV1,
    RecordKind.VERIFIED_OUTCOME: VerifiedOutcomeV1,
    RecordKind.OUTCOME_ASSESSMENT: OutcomeAssessmentV1,
    RecordKind.DECISION_FEEDBACK: DecisionFeedbackV1,
}


def _decode(kind: RecordKind, record: object, attempt: int | None) -> DecodedRecord:
    if kind is RecordKind.APPROVAL_CHALLENGE:
        if attempt is None:
            raise ValueError("a challenge record carries its attempt")
        integer(attempt, "attempt", minimum=1, maximum=MAX_CHALLENGES_PER_INTENT)
    elif attempt is not None:
        raise ValueError("only a challenge record carries an attempt")
    if isinstance(record, ActionIntentV1):
        return _intent(record)
    if isinstance(record, InverseIntentV1):
        return _inverse(record)
    if isinstance(record, ExpectedObservationV1):
        return _forecast(record)
    if isinstance(record, PolicyVerdictV1):
        return _verdict(record)
    if isinstance(record, ApprovalChallengeV1):
        assert attempt is not None
        return _challenge(record, attempt)
    if isinstance(record, ApprovalExpirationV1):
        return _expiration(record)
    if isinstance(record, ExecutionClaimV1):
        return _claim(record)
    if isinstance(record, VerifiedOutcomeV1):
        return _outcome(record)
    if isinstance(record, OutcomeAssessmentV1):
        return _assessment(record)
    if isinstance(record, DecisionFeedbackV1):
        return _feedback(record)
    raise TypeError("unknown record type")


# ----------------------------------------------------------------- envelope

_ENVELOPE_FIELDS: Final = (
    "attempt",
    "event_id",
    "incarnation_id",
    "operation_kind",
    "processing_sequence",
    "record",
    "record_kind",
    "schema_version",
    "transaction_kind",
)


@dataclass(frozen=True, slots=True)
class HistoryRecordOperationV1:
    """The ``pending_bytes`` of one R17 / R18 subject.history operation."""

    operation_kind: HistoryOperationKind
    incarnation_id: str
    event_id: str
    processing_sequence: int
    transaction_kind: str
    record_kind: RecordKind
    record: object
    attempt: int | None = None

    def __post_init__(self) -> None:
        if type(self.operation_kind) is not HistoryOperationKind:
            raise TypeError("operation_kind must be a HistoryOperationKind")
        if self.operation_kind not in RECORD_OPERATION_KINDS:
            raise ValueError("R16 operations are not record operations")
        digest(self.incarnation_id, "incarnation_id")
        canonical_uuid(self.event_id, "event_id")
        event_sequence(self.processing_sequence, "processing_sequence")
        if type(self.transaction_kind) is not str or not 1 <= len(self.transaction_kind) <= 64:
            raise ValueError("transaction_kind must be 1..64 characters")
        if type(self.record_kind) is not RecordKind:
            raise TypeError("record_kind must be a RecordKind")
        if self.record_kind not in OPERATION_RECORD_KINDS[self.operation_kind]:
            raise ValueError("the record kind does not belong to the operation kind")
        if type(self.record) is not _TYPES[self.record_kind]:
            raise TypeError("the record is not of its record kind")
        decoded = self.decoded()
        if decoded.event is not None and decoded.event != (self.event_id, self.processing_sequence):
            raise ValueError("the record names another event than its operation")
        if len(self.canonical_bytes()) > MAX_PENDING_BYTES:
            raise ValueError("the record operation exceeds the pending bound")

    def decoded(self) -> DecodedRecord:
        return _decode(self.record_kind, self.record, self.attempt)

    @property
    def reserved_keys(self) -> tuple[str, ...]:
        return self.decoded().reserved_keys

    @property
    def rows(self) -> tuple[RecordRow, ...]:
        return self.decoded().rows

    def canonical_value(self) -> dict[str, object]:
        return {
            "attempt": self.attempt,
            "event_id": self.event_id,
            "incarnation_id": self.incarnation_id,
            "operation_kind": self.operation_kind.value,
            "processing_sequence": self.processing_sequence,
            "record": parse_json(_record_bytes(self.record), MAX_PENDING_BYTES),
            "record_kind": self.record_kind.value,
            "schema_version": ENVELOPE_SCHEMA_VERSION,
            "transaction_kind": self.transaction_kind,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.canonical_value())

    @property
    def operation_digest(self) -> str:
        return record_operation_digest(self.canonical_bytes())

    @classmethod
    def from_bytes(cls, payload: bytes) -> Self:
        if type(payload) is not bytes or len(payload) > MAX_PENDING_BYTES:
            raise ValueError("record operation bytes are invalid")
        row = closed(parse_json(payload, MAX_PENDING_BYTES), _ENVELOPE_FIELDS, "record operation")
        if row["schema_version"] != ENVELOPE_SCHEMA_VERSION or type(row["schema_version"]) is not int:
            raise ValueError("record operation schema_version is not V1")
        kind = parse_enum(row["record_kind"], RecordKind, "record_kind")
        attempt = row["attempt"]
        if attempt is not None and type(attempt) is not int:
            raise TypeError("attempt must be an integer")
        record = _DECODERS[kind](canonical_json(row["record"]))
        value = cls(
            operation_kind=parse_enum(row["operation_kind"], HistoryOperationKind, "operation_kind"),
            incarnation_id=_str(row["incarnation_id"]),
            event_id=_str(row["event_id"]),
            processing_sequence=_int(row["processing_sequence"]),
            transaction_kind=_str(row["transaction_kind"]),
            record_kind=kind,
            record=record,
            attempt=attempt,
        )
        if value.canonical_bytes() != payload:
            raise ValueError("record operation bytes are not canonical")
        return value


def _record_bytes(record: object) -> bytes:
    payload = getattr(record, "canonical_bytes")()
    if type(payload) is not bytes:
        raise TypeError("a record encodes to bytes")
    return payload


def record_operation_digest(pending_bytes: bytes) -> str:
    """The R07 operation digest of one record operation (§3)."""
    return sha256(_OPERATION_DOMAIN_V2 + pending_bytes).hexdigest()


__all__ = [
    "ENVELOPE_SCHEMA_VERSION",
    "MAX_CHALLENGES_PER_INTENT",
    "OPERATION_RECORD_KINDS",
    "OPERATION_TABLES",
    "R16_OPERATION_KINDS",
    "R18_GATED_KINDS",
    "RECORD_OPERATION_KINDS",
    "RECORD_TABLES",
    "ApprovalExpirationV1",
    "DecodedRecord",
    "ExpirationReason",
    "HistoryOperationKind",
    "HistoryRecordOperationV1",
    "RecordKind",
    "RecordRow",
    "record_operation_digest",
]
