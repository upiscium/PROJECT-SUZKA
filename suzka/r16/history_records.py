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

IP-14b-1 (D1 r2 §4-2) adds the R19 / cycle record kinds of schema v4 (Wake,
claim, attempt outcome, cycle closure, terminal, Outbox message, Inbox ACK)
and the HD-3 (a) plan-hold receipt, each one record in the same envelope
(``ENVELOPE_SCHEMA_VERSION`` stays 1). ``R16_CYCLE_DECISION`` is the one
record kind that also writes R16 Decision rows; the store fills its lineage
columns from the embedded Decision.

Pure: no store, Journal or clock. Validation rejects inconsistent values;
whether a writer was allowed to issue them is the writer's (IP-16..IP-21).
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
from suzka.r16.cycle_contracts import CycleClosureV1, PlanHoldReceiptV1, ResultKind
from suzka.r16.history_codec import (
    MAX_PENDING_BYTES,
    CommittedDecisionV1,
    canonical_uuid,
    event_sequence,
)
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
from suzka.r19.contracts import (
    STORED_ATTEMPT_OUTCOMES,
    InboxAckV1,
    OutboxMessageV1,
    WakeAttemptOutcomeV1,
    WakeClaimV1,
    WakeTerminalV1,
    WakeV1,
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
    # IP-14b-1 (schema v4, D1 r2 §4-1).
    R19_WAKE_MATERIALIZE = "R19_WAKE_MATERIALIZE"
    R19_WAKE_CLAIM = "R19_WAKE_CLAIM"
    R19_WAKE_ATTEMPT_OUTCOME = "R19_WAKE_ATTEMPT_OUTCOME"
    R16_PLAN_HOLD = "R16_PLAN_HOLD"
    R16_CYCLE_DECISION = "R16_CYCLE_DECISION"
    R16_CYCLE_RECEIPT = "R16_CYCLE_RECEIPT"
    R19_WAKE_TERMINAL = "R19_WAKE_TERMINAL"
    R19_OUTBOX_EMIT = "R19_OUTBOX_EMIT"
    R19_INBOX_ACK = "R19_INBOX_ACK"


_K = HistoryOperationKind
# The two kinds that are not a record envelope.
R16_OPERATION_KINDS: Final = frozenset({_K.R16_DECISION_ISSUE, _K.R07_BINDING_ONLY_ABORT})
# Every kind whose pending bytes are a HistoryRecordOperationV1 envelope.
RECORD_OPERATION_KINDS: Final = frozenset(set(HistoryOperationKind) - set(R16_OPERATION_KINDS))
# r18.enabled gates every R17 / R18 kind (R17 follows R18 Outcomes, §7).
R18_GATED_KINDS: Final = frozenset(
    {
        _K.R18_INTENT_ISSUE,
        _K.R17_EXPECTATION_COMMIT,
        _K.R18_POLICY_COMMIT,
        _K.R18_CHALLENGE_ISSUE,
        _K.R18_APPROVAL_EXPIRE,
        _K.R18_EXECUTION_CLAIM,
        _K.R18_OUTCOME_COMMIT,
        _K.R17_ASSESSMENT_COMMIT,
        _K.R17_FEEDBACK_COMMIT,
    }
)
# IP-14b D1 r2 §6-1: r19.enabled gates the eight R19 / cycle kinds.
R19_GATED_KINDS: Final = frozenset(
    {
        _K.R19_WAKE_MATERIALIZE,
        _K.R19_WAKE_CLAIM,
        _K.R19_WAKE_ATTEMPT_OUTCOME,
        _K.R16_CYCLE_DECISION,
        _K.R16_CYCLE_RECEIPT,
        _K.R19_WAKE_TERMINAL,
        _K.R19_OUTBOX_EMIT,
        _K.R19_INBOX_ACK,
    }
)
# HD-3 (a): subject_history.plan_hold_receipts gates only this kind.
PLAN_HOLD_GATED_KINDS: Final = frozenset({_K.R16_PLAN_HOLD})
assert R18_GATED_KINDS | R19_GATED_KINDS | PLAN_HOLD_GATED_KINDS == RECORD_OPERATION_KINDS
# The record kinds that also write R16 lineage columns and Decision rows.
LINEAGE_RECORD_KINDS: Final = frozenset({_K.R16_CYCLE_DECISION})


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
    WAKE = "WAKE"
    WAKE_CLAIM = "WAKE_CLAIM"
    WAKE_ATTEMPT_OUTCOME = "WAKE_ATTEMPT_OUTCOME"
    PLAN_HOLD = "PLAN_HOLD"
    CYCLE_CLOSURE = "CYCLE_CLOSURE"
    WAKE_TERMINAL = "WAKE_TERMINAL"
    OUTBOX_MESSAGE = "OUTBOX_MESSAGE"
    INBOX_ACK = "INBOX_ACK"


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
    _K.R19_WAKE_MATERIALIZE: frozenset({RecordKind.WAKE}),
    _K.R19_WAKE_CLAIM: frozenset({RecordKind.WAKE_CLAIM}),
    _K.R19_WAKE_ATTEMPT_OUTCOME: frozenset({RecordKind.WAKE_ATTEMPT_OUTCOME}),
    _K.R16_PLAN_HOLD: frozenset({RecordKind.PLAN_HOLD}),
    _K.R16_CYCLE_DECISION: frozenset({RecordKind.CYCLE_CLOSURE}),
    _K.R16_CYCLE_RECEIPT: frozenset({RecordKind.CYCLE_CLOSURE}),
    _K.R19_WAKE_TERMINAL: frozenset({RecordKind.WAKE_TERMINAL}),
    _K.R19_OUTBOX_EMIT: frozenset({RecordKind.OUTBOX_MESSAGE}),
    _K.R19_INBOX_ACK: frozenset({RecordKind.INBOX_ACK}),
}
# The tables each operation kind writes at FINALIZE (§3); used by the
# finalize_complete trigger and the startup check. ``R16_CYCLE_DECISION``
# also names the v1 ``history_decisions`` (its source rows are optional in
# SQL, as for ``R16_DECISION_ISSUE``).
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
    # IP-14b-1 (v4), in the CREATE order of D1 r2 §3-1.
    _K.R19_WAKE_MATERIALIZE: ("wakeups",),
    _K.R19_WAKE_CLAIM: ("wake_claims",),
    _K.R19_WAKE_ATTEMPT_OUTCOME: ("wake_attempt_outcomes",),
    _K.R16_CYCLE_DECISION: ("history_decisions", "logical_cycle_resolutions"),
    _K.R16_CYCLE_RECEIPT: ("logical_cycle_resolutions",),
    _K.R19_WAKE_TERMINAL: ("wake_terminals",),
    _K.R19_OUTBOX_EMIT: ("outbox_messages",),
    _K.R19_INBOX_ACK: ("outbox_acks",),
    _K.R16_PLAN_HOLD: ("plan_hold_receipts",),
}
# The v1 tables have their own triggers and are never record tables.
_V1_TABLES: Final = frozenset({"history_decisions", "history_source_refs"})
RECORD_TABLES: Final = tuple(
    dict.fromkeys(
        table
        for tables in OPERATION_TABLES.values()
        for table in tables
        if table not in _V1_TABLES
    )
)
# The v2 / v3 record tables (R17 / R18) and the v4 ones (R19 / cycle / hold).
V2_RECORD_TABLES: Final = tuple(
    table
    for table in RECORD_TABLES
    if any(table in OPERATION_TABLES[kind] for kind in R18_GATED_KINDS)
)
V4_RECORD_TABLES: Final = tuple(table for table in RECORD_TABLES if table not in V2_RECORD_TABLES)


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


# ------------------------------------------------- v4 records (IP-14b-1)
# Keys (D1 r2 §2-4). Wakes, cycles, messages and ACKs are cognition records
# of one incarnation (IP-16 §9); their ids already hash the incarnation.


def _same_incarnation(value: str, incarnation_id: str) -> None:
    if value != incarnation_id:
        raise ValueError("the record belongs to another incarnation than its operation")


def _wake(value: WakeV1, incarnation_id: str) -> DecodedRecord:
    _same_incarnation(value.incarnation_id, incarnation_id)
    outcome = value.verified_outcome_key
    row = RecordRow(
        "wakeups",
        (
            *_common(value.wakeup_id, value.canonical_bytes(), value.wake_digest),
            ("logical_cycle_id", value.logical_cycle_id),
            ("cause_event_id", value.cause_event_id),
            ("verified_outcome_id", None if outcome is None else outcome[0]),
            ("outcome_revision", None if outcome is None else outcome[1]),
        ),
    )
    keys = [f"wake:{value.wakeup_id}", f"wake-cause:{incarnation_id}:{value.cause_event_id}"]
    if outcome is not None:
        keys.append(f"wake-outcome:{incarnation_id}:{outcome[0]}:{outcome[1]}")
    return DecodedRecord(RecordKind.WAKE, value, (row,), tuple(keys), None)


def _wake_claim(value: WakeClaimV1, incarnation_id: str) -> DecodedRecord:
    _same_incarnation(value.incarnation_id, incarnation_id)
    row = RecordRow(
        "wake_claims",
        (
            *_common(value.claim_id, value.canonical_bytes(), value.claim_digest),
            ("wakeup_id", value.wakeup_id),
            ("logical_cycle_id", value.logical_cycle_id),
            ("attempt_index", value.attempt_index),
        ),
    )
    return DecodedRecord(
        RecordKind.WAKE_CLAIM,
        value,
        (row,),
        (f"wake-attempt:{value.wakeup_id}:{value.attempt_index}",),
        (value.claim_event_id, value.claim_sequence),
    )


def _attempt_outcome(value: WakeAttemptOutcomeV1) -> DecodedRecord:
    if value.outcome not in STORED_ATTEMPT_OUTCOMES:
        # LEASE_EXPIRED is derived from the claim and the clock (D1 r2 §1-1).
        raise ValueError("only a REFUSED or DEFERRED_BUDGET attempt outcome is recorded")
    row = RecordRow(
        "wake_attempt_outcomes",
        _common(value.claim_id, value.canonical_bytes(), value.attempt_digest),
    )
    return DecodedRecord(
        RecordKind.WAKE_ATTEMPT_OUTCOME, value, (row,), (f"attempt-end:{value.claim_id}",), None
    )


def _plan_hold(value: PlanHoldReceiptV1, incarnation_id: str) -> DecodedRecord:
    _same_incarnation(value.incarnation_id, incarnation_id)
    row = RecordRow(
        "plan_hold_receipts",
        (
            *_common(value.hold_id, value.canonical_bytes(), value.hold_digest),
            ("lineage_scope", value.lineage_scope),
            ("review_intent_key", value.review_intent_key),
        ),
    )
    return DecodedRecord(
        RecordKind.PLAN_HOLD,
        value,
        (row,),
        (f"plan-hold:{incarnation_id}:{value.review_intent_key}",),
        (value.event_id, value.event_sequence),
    )


def _cycle_closure(value: CycleClosureV1, incarnation_id: str) -> DecodedRecord:
    resolution = value.resolution
    _same_incarnation(resolution.incarnation_id, incarnation_id)
    decision = value.decision
    row = RecordRow(
        "logical_cycle_resolutions",
        (
            *_common(resolution.resolution_id, resolution.canonical_bytes(), resolution.resolution_digest),
            ("logical_cycle_id", resolution.logical_cycle_id),
            ("review_intent_key", resolution.review_intent_key),
            ("claim_id", resolution.claim_id),
            ("result_kind", resolution.result_kind.value),
            ("result_id", resolution.result_id),
            ("result_digest", resolution.result_digest),
            ("decision_id", None if decision is None else decision.decision_id),
            ("receipt_bytes", value.receipt_bytes),
        ),
    )
    # The Decision and source rows of R16_CYCLE_DECISION are written by the
    # store's Decision insert (history_source_refs has no operation columns).
    return DecodedRecord(
        RecordKind.CYCLE_CLOSURE,
        value,
        (row,),
        (
            f"cycle:{incarnation_id}:{resolution.logical_cycle_id}",
            f"attempt-end:{resolution.claim_id}",
        ),
        value.event,
    )


def _wake_terminal(value: WakeTerminalV1, incarnation_id: str) -> DecodedRecord:
    row = RecordRow(
        "wake_terminals",
        (
            *_common(value.terminal_id, value.canonical_bytes(), value.terminal_digest),
            ("wakeup_id", value.wakeup_id),
            ("terminal", value.terminal.value),
            ("reason", value.reason.value),
            ("superseding_wakeup_id", value.superseding_wakeup_id),
        ),
    )
    return DecodedRecord(
        RecordKind.WAKE_TERMINAL,
        value,
        (row,),
        (f"cycle:{incarnation_id}:{value.logical_cycle_id}",),
        (value.event_id, value.event_sequence),
    )


def _outbox_message(value: OutboxMessageV1, incarnation_id: str) -> DecodedRecord:
    _same_incarnation(value.incarnation_id, incarnation_id)
    row = RecordRow(
        "outbox_messages",
        (
            *_common(value.message_id, value.canonical_bytes(), value.message_digest),
            ("source_id", value.source_id),
            ("result_digest", value.result_digest),
            ("template", value.template.value),
            ("context_id", value.context_id),
        ),
    )
    return DecodedRecord(RecordKind.OUTBOX_MESSAGE, value, (row,), (f"outbox:{value.message_id}",), None)


def _inbox_ack(value: InboxAckV1) -> DecodedRecord:
    row = RecordRow(
        "outbox_acks",
        (
            *_common(value.ack_id, value.canonical_bytes(), value.ack_digest),
            ("message_id", value.message_id),
            ("context_id", value.context_id),
        ),
    )
    return DecodedRecord(
        RecordKind.INBOX_ACK,
        value,
        (row,),
        (f"ack:{value.message_id}",),
        (value.event_id, value.event_sequence),
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
    RecordKind.WAKE: WakeV1.from_json,
    RecordKind.WAKE_CLAIM: WakeClaimV1.from_json,
    RecordKind.WAKE_ATTEMPT_OUTCOME: WakeAttemptOutcomeV1.from_json,
    RecordKind.PLAN_HOLD: PlanHoldReceiptV1.from_json,
    RecordKind.CYCLE_CLOSURE: CycleClosureV1.from_json,
    RecordKind.WAKE_TERMINAL: WakeTerminalV1.from_json,
    RecordKind.OUTBOX_MESSAGE: OutboxMessageV1.from_json,
    RecordKind.INBOX_ACK: InboxAckV1.from_json,
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
    RecordKind.WAKE: WakeV1,
    RecordKind.WAKE_CLAIM: WakeClaimV1,
    RecordKind.WAKE_ATTEMPT_OUTCOME: WakeAttemptOutcomeV1,
    RecordKind.PLAN_HOLD: PlanHoldReceiptV1,
    RecordKind.CYCLE_CLOSURE: CycleClosureV1,
    RecordKind.WAKE_TERMINAL: WakeTerminalV1,
    RecordKind.OUTBOX_MESSAGE: OutboxMessageV1,
    RecordKind.INBOX_ACK: InboxAckV1,
}


def _decode(kind: RecordKind, record: object, attempt: int | None, incarnation_id: str) -> DecodedRecord:
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
    if isinstance(record, WakeV1):
        return _wake(record, incarnation_id)
    if isinstance(record, WakeClaimV1):
        return _wake_claim(record, incarnation_id)
    if isinstance(record, WakeAttemptOutcomeV1):
        return _attempt_outcome(record)
    if isinstance(record, PlanHoldReceiptV1):
        return _plan_hold(record, incarnation_id)
    if isinstance(record, CycleClosureV1):
        return _cycle_closure(record, incarnation_id)
    if isinstance(record, WakeTerminalV1):
        return _wake_terminal(record, incarnation_id)
    if isinstance(record, OutboxMessageV1):
        return _outbox_message(record, incarnation_id)
    if isinstance(record, InboxAckV1):
        return _inbox_ack(record)
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
        if type(self.record) is CycleClosureV1:
            # The kind matches the result: a Decision <=> R16_CYCLE_DECISION.
            is_decision = self.record.resolution.result_kind is ResultKind.DECISION_ISSUED
            if is_decision != (self.operation_kind is HistoryOperationKind.R16_CYCLE_DECISION):
                raise ValueError("a cycle Decision is exactly an R16_CYCLE_DECISION operation")
        decoded = self.decoded()
        if decoded.event is not None and decoded.event != (self.event_id, self.processing_sequence):
            raise ValueError("the record names another event than its operation")
        if type(self.record) is WakeV1 and not self.record.horizon_sequence < self.processing_sequence:
            raise ValueError("a Wake is materialized after its creation horizon")
        if len(self.canonical_bytes()) > MAX_PENDING_BYTES:
            raise ValueError("the record operation exceeds the pending bound")

    def decoded(self) -> DecodedRecord:
        return _decode(self.record_kind, self.record, self.attempt, self.incarnation_id)

    @property
    def reserved_keys(self) -> tuple[str, ...]:
        return self.decoded().reserved_keys

    @property
    def rows(self) -> tuple[RecordRow, ...]:
        return self.decoded().rows

    @property
    def cycle_decision(self) -> CommittedDecisionV1 | None:
        """The embedded Decision of an ``R16_CYCLE_DECISION`` operation."""
        if type(self.record) is CycleClosureV1:
            return self.record.decision
        return None

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
    "LINEAGE_RECORD_KINDS",
    "MAX_CHALLENGES_PER_INTENT",
    "OPERATION_RECORD_KINDS",
    "OPERATION_TABLES",
    "PLAN_HOLD_GATED_KINDS",
    "R16_OPERATION_KINDS",
    "R18_GATED_KINDS",
    "R19_GATED_KINDS",
    "RECORD_OPERATION_KINDS",
    "RECORD_TABLES",
    "V2_RECORD_TABLES",
    "V4_RECORD_TABLES",
    "ApprovalExpirationV1",
    "DecodedRecord",
    "ExpirationReason",
    "HistoryOperationKind",
    "HistoryRecordOperationV1",
    "RecordKind",
    "RecordRow",
    "record_operation_digest",
]
