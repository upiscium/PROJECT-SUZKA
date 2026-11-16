"""IP-13b: R16-owned pure values that close a Wake-origin cycle (D-R19 §2/§4).

A Wake re-evaluation (``R16_WAKE_REVIEW``, IP-20) ends in exactly one result
per logical cycle (#369 A1-1 / A2-2, #373 A1-9 / A3-2): a new
CommittedDecision, an ExistingDecisionReceipt (same evidence, the head is
confirmed) or a HoldReceipt. ``LogicalCycleResolutionV1`` points at that one
result. Nothing here issues, persists or reads anything; the R16 issuer and
the SubjectHistory DDL (IP-14b) own that.

R16 must not import ``suzka.r19``: the R19 claim arrives as its id and digest.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Final, Self

from suzka.r16.contracts import (
    canonical_json,
    checksum,
    closed,
    digest,
    parse_enum,
    parse_json,
)
from suzka.r16.history_codec import (
    DECISION_ID_PREFIX,
    CommittedDecisionV1,
    PolicyKind,
    PolicyRow,
    canonical_uuid,
    decision_id,
    event_sequence,
)
from suzka.r16.successor_policy import SuccessorRow

SCHEMA_VERSION: Final = 1
MAX_RECORD_BYTES: Final = 4096
WAKE_CLAIM_ID_PREFIX: Final = "wake-claim-"
HOLD_RECEIPT_ID_PREFIX: Final = "cycle-hold-"
EXISTING_RECEIPT_ID_PREFIX: Final = "cycle-existing-"
RESOLUTION_ID_PREFIX: Final = "cycle-resolution-"
_UTC_FORMAT: Final = "%Y-%m-%dT%H:%M:%SZ"

_REVIEW_INTENT_DOMAIN: Final = b"PROJECT-SUZKA:R16:WAKE-REVIEW-INTENT:V1\0"
_HOLD_ID_DOMAIN: Final = b"PROJECT-SUZKA:R16:CYCLE-HOLD-ID:V1\0"
_HOLD_DOMAIN: Final = b"PROJECT-SUZKA:R16:CYCLE-HOLD:V1\0"
_EXISTING_ID_DOMAIN: Final = b"PROJECT-SUZKA:R16:CYCLE-EXISTING-ID:V1\0"
_EXISTING_DOMAIN: Final = b"PROJECT-SUZKA:R16:CYCLE-EXISTING:V1\0"
_RESOLUTION_ID_DOMAIN: Final = b"PROJECT-SUZKA:R16:CYCLE-RESOLUTION-ID:V1\0"
_RESOLUTION_DOMAIN: Final = b"PROJECT-SUZKA:R16:CYCLE-RESOLUTION:V1\0"

# D-R19 §4: policy rows that close a cycle with a terminal HOLD. Row 1 from
# an EXPIRED / INELIGIBLE Commitment is terminal; row 1 from an UNAVAILABLE
# source is a refuse (R19 WakeAttemptOutcome) and never reaches a receipt.
_HOLD_ROWS: Final = frozenset(
    {SuccessorRow.ROW_0, SuccessorRow.ROW_1, SuccessorRow.ROW_2, SuccessorRow.ROW_2B, SuccessorRow.ROW_7}
)
_POLICY_KINDS: Final = frozenset({PolicyKind.SUCCESSOR_POLICY_V2, PolicyKind.SUCCESSOR_POLICY_V3})


class HoldReason(str, Enum):
    """The terminal HOLD reasons of D-R19 §4 (closed)."""

    # A policy HOLD row; ``policy_row`` names it.
    POLICY_HOLD = "POLICY_HOLD"
    # The current fingerprint differs from the frozen one (steps 3 / 7), or
    # a FINALIZED row already holds the Decision slot.
    STALE_SOURCE = "STALE_SOURCE"
    # Same evidence as the head, but the replayed row differs (step 5).
    DECISION_SOURCE_DRIFT = "DECISION_SOURCE_DRIFT"
    # The Commitment deadline passed at the frozen time T (step 4).
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    PLAN_SOURCE_CONFLICT = "PLAN_SOURCE_CONFLICT"


class ResultKind(str, Enum):
    DECISION_ISSUED = "DECISION_ISSUED"
    EXISTING_DECISION = "EXISTING_DECISION"
    HOLD = "HOLD"


# ------------------------------------------------------------------ helpers


def _str(value: object) -> str:
    if type(value) is not str:
        raise TypeError("expected a string")
    return value


def _int(value: object) -> int:
    if type(value) is not int:
        raise TypeError("expected an integer")
    return value


def utc(value: object, name: str) -> datetime:
    """A canonical ``YYYY-MM-DDTHH:MM:SSZ`` timestamp."""
    if type(value) is not str:
        raise TypeError(f"{name} must be a UTC timestamp string")
    try:
        parsed = datetime.strptime(value, _UTC_FORMAT)
    except ValueError as error:
        raise ValueError(f"{name} must be a UTC timestamp") from error
    if parsed.strftime(_UTC_FORMAT) != value:
        raise ValueError(f"{name} must be a canonical UTC timestamp")
    return parsed.replace(tzinfo=timezone.utc)


def prefixed(value: object, prefix: str, name: str) -> str:
    """``<prefix><sha256 hex>``."""
    if type(value) is not str or not value.startswith(prefix) or len(value) != len(prefix) + 64:
        raise ValueError(f"{name} must be a {prefix}<sha256> identity")
    digest(value[len(prefix) :], name)
    return value


def _parse(payload: bytes, names: tuple[str, ...], name: str) -> dict[str, object]:
    if type(payload) is not bytes:
        raise TypeError(f"{name} payload must be bytes")
    if len(payload) > MAX_RECORD_BYTES:
        raise ValueError(f"{name} exceeds its {MAX_RECORD_BYTES}-byte bound")
    row = closed(parse_json(payload, MAX_RECORD_BYTES), names, name)
    if row["schema_version"] != SCHEMA_VERSION or type(row["schema_version"]) is not int:
        raise ValueError(f"{name} schema_version is not V1")
    return row


def _bounded(payload: bytes, name: str) -> bytes:
    if len(payload) > MAX_RECORD_BYTES:
        raise ValueError(f"{name} exceeds its {MAX_RECORD_BYTES}-byte bound")
    return payload


def _claim(claim_id: object, claim_digest: object) -> None:
    prefixed(claim_id, WAKE_CLAIM_ID_PREFIX, "claim_id")
    digest(claim_digest, "claim_digest")


# ------------------------------------------------------ frozen predecessor


@dataclass(frozen=True, slots=True)
class FrozenPredecessor:
    """The head frozen at Wake creation (#369 A2-1), never re-read per attempt.

    Either the committed predecessor Decision (id and committed digest) or a
    witnessed absence at the creation horizon (the event sequence of the
    snapshot that found no head).
    """

    decision_id: str | None
    decision_digest: str | None
    absence_horizon_sequence: int | None

    def __post_init__(self) -> None:
        present = self.decision_id is not None
        if present != (self.decision_digest is not None) or present == (
            self.absence_horizon_sequence is not None
        ):
            raise ValueError("a frozen predecessor is a Decision or a witnessed absence")
        if present:
            decision_id(self.decision_id, "predecessor decision_id")
            digest(self.decision_digest, "predecessor decision_digest")
        else:
            event_sequence(self.absence_horizon_sequence, "absence_horizon_sequence")

    @classmethod
    def decision(cls, decision_id: str, decision_digest: str) -> Self:
        return cls(decision_id, decision_digest, None)

    @classmethod
    def absence(cls, horizon_sequence: int) -> Self:
        return cls(None, None, horizon_sequence)

    def canonical_value(self) -> dict[str, object]:
        return {
            "absence_horizon_sequence": self.absence_horizon_sequence,
            "decision_digest": self.decision_digest,
            "decision_id": self.decision_id,
        }

    @classmethod
    def from_value(cls, value: object) -> Self:
        row = closed(
            value, ("absence_horizon_sequence", "decision_digest", "decision_id"), "frozen predecessor"
        )
        horizon, ident, value_digest = (
            row["absence_horizon_sequence"],
            row["decision_id"],
            row["decision_digest"],
        )
        return cls(
            None if ident is None else _str(ident),
            None if value_digest is None else _str(value_digest),
            None if horizon is None else _int(horizon),
        )


def wake_review_intent_key(
    logical_cycle_id: str,
    predecessor: FrozenPredecessor,
    policy_kind: PolicyKind,
    evidence_fingerprint: str,
) -> str:
    """#369 A2-1: the Wake-origin ``review_intent_key``.

    Built only from the frozen cycle, the frozen predecessor (or absence),
    the versioned policy identity and the frozen fingerprint. It excludes the
    claim / attempt, the admitting event, ``now_utc`` and future WAL proof, so
    every retry of one cycle has the same key.
    """

    digest(logical_cycle_id, "logical_cycle_id")
    if type(predecessor) is not FrozenPredecessor:
        raise TypeError("predecessor must be a FrozenPredecessor")
    if type(policy_kind) is not PolicyKind or policy_kind not in _POLICY_KINDS:
        raise ValueError("policy_kind must be SuccessorPolicy V2 or V3")
    digest(evidence_fingerprint, "evidence_fingerprint")
    return checksum(
        _REVIEW_INTENT_DOMAIN,
        {
            "evidence_fingerprint": evidence_fingerprint,
            "logical_cycle_id": logical_cycle_id,
            "policy_kind": policy_kind.value,
            "predecessor": predecessor.canonical_value(),
        },
    )


# ------------------------------------------------------------ hold receipt

_HOLD_BODY: Final = (
    "claim_digest",
    "claim_event_id",
    "claim_id",
    "current_fingerprint",
    "hold_event_id",
    "hold_sequence",
    "logical_cycle_id",
    "policy_row",
    "prepare_checked_at_utc",
    "reason",
    "review_intent_key",
    "schema_version",
    "sealed_input_digest",
)


@dataclass(frozen=True, slots=True)
class HoldReceiptV1:
    """A terminal HOLD that closes one cycle (D-R19 §2, #369 D2-3).

    ``policy_row`` is present exactly for a policy HOLD row. Refuses
    (UNAVAILABLE sources, capacity, a pending slot, the clock) never become a
    HoldReceipt; they are R19 WakeAttemptOutcomes and count as attempts.
    """

    logical_cycle_id: str
    review_intent_key: str
    claim_id: str
    claim_digest: str
    claim_event_id: str
    hold_event_id: str
    hold_sequence: int
    reason: HoldReason
    policy_row: SuccessorRow | None
    current_fingerprint: str
    sealed_input_digest: str
    prepare_checked_at_utc: str

    def __post_init__(self) -> None:
        digest(self.logical_cycle_id, "logical_cycle_id")
        digest(self.review_intent_key, "review_intent_key")
        _claim(self.claim_id, self.claim_digest)
        canonical_uuid(self.claim_event_id, "claim_event_id")
        canonical_uuid(self.hold_event_id, "hold_event_id")
        if self.hold_event_id == self.claim_event_id:
            raise ValueError("the HOLD is a later event than the claim")
        event_sequence(self.hold_sequence, "hold_sequence")
        if type(self.reason) is not HoldReason:
            raise TypeError("reason must be a HoldReason")
        if self.policy_row is not None and type(self.policy_row) is not SuccessorRow:
            raise TypeError("policy_row must be a SuccessorRow")
        if (self.reason is HoldReason.POLICY_HOLD) != (self.policy_row is not None):
            raise ValueError("a policy row is recorded exactly for a policy HOLD")
        if self.policy_row is not None and self.policy_row not in _HOLD_ROWS:
            raise ValueError("a policy HOLD names a HOLD row (0, 1, 2, 2b or 7)")
        digest(self.current_fingerprint, "current_fingerprint")
        digest(self.sealed_input_digest, "sealed_input_digest")
        utc(self.prepare_checked_at_utc, "prepare_checked_at_utc")
        _bounded(self.canonical_bytes(), "hold receipt")

    def _body(self) -> dict[str, object]:
        return {
            "claim_digest": self.claim_digest,
            "claim_event_id": self.claim_event_id,
            "claim_id": self.claim_id,
            "current_fingerprint": self.current_fingerprint,
            "hold_event_id": self.hold_event_id,
            "hold_sequence": self.hold_sequence,
            "logical_cycle_id": self.logical_cycle_id,
            "policy_row": None if self.policy_row is None else self.policy_row.value,
            "prepare_checked_at_utc": self.prepare_checked_at_utc,
            "reason": self.reason.value,
            "review_intent_key": self.review_intent_key,
            "schema_version": SCHEMA_VERSION,
            "sealed_input_digest": self.sealed_input_digest,
        }

    @property
    def hold_id(self) -> str:
        return HOLD_RECEIPT_ID_PREFIX + checksum(_HOLD_ID_DOMAIN, self._body())

    @property
    def hold_digest(self) -> str:
        return checksum(_HOLD_DOMAIN, self._body())

    def canonical_value(self) -> dict[str, object]:
        return {**self._body(), "hold_digest": self.hold_digest, "hold_id": self.hold_id}

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.canonical_value())

    @classmethod
    def from_json(cls, payload: bytes) -> Self:
        row = _parse(payload, (*_HOLD_BODY, "hold_digest", "hold_id"), "hold receipt")
        policy_row = row["policy_row"]
        value = cls(
            logical_cycle_id=_str(row["logical_cycle_id"]),
            review_intent_key=_str(row["review_intent_key"]),
            claim_id=_str(row["claim_id"]),
            claim_digest=_str(row["claim_digest"]),
            claim_event_id=_str(row["claim_event_id"]),
            hold_event_id=_str(row["hold_event_id"]),
            hold_sequence=_int(row["hold_sequence"]),
            reason=parse_enum(row["reason"], HoldReason, "reason"),
            policy_row=None
            if policy_row is None
            else parse_enum(policy_row, SuccessorRow, "policy_row"),
            current_fingerprint=_str(row["current_fingerprint"]),
            sealed_input_digest=_str(row["sealed_input_digest"]),
            prepare_checked_at_utc=_str(row["prepare_checked_at_utc"]),
        )
        if value.canonical_bytes() != payload:
            raise ValueError("hold receipt bytes are not canonical")
        return value


# ------------------------------------------------- existing-decision receipt

_EXISTING_BODY: Final = (
    "claim_digest",
    "claim_id",
    "current_fingerprint",
    "event_id",
    "event_sequence",
    "head_committed_digest",
    "head_decision_id",
    "logical_cycle_id",
    "prepare_checked_at_utc",
    "reproduced_policy_row",
    "review_intent_key",
    "schema_version",
)


@dataclass(frozen=True, slots=True)
class ExistingDecisionReceiptV1:
    """Same evidence as the head, and the head's row replays (D-R19 §4 step 5).

    It confirms the existing Decision without issuing a new one. Human
    decision §8-2 (2026-11-15): S1 counts it as "continue".
    """

    logical_cycle_id: str
    review_intent_key: str
    claim_id: str
    claim_digest: str
    event_id: str
    event_sequence: int
    prepare_checked_at_utc: str
    current_fingerprint: str
    head_decision_id: str
    head_committed_digest: str
    reproduced_policy_row: PolicyRow

    def __post_init__(self) -> None:
        digest(self.logical_cycle_id, "logical_cycle_id")
        digest(self.review_intent_key, "review_intent_key")
        _claim(self.claim_id, self.claim_digest)
        canonical_uuid(self.event_id, "event_id")
        event_sequence(self.event_sequence, "event_sequence")
        utc(self.prepare_checked_at_utc, "prepare_checked_at_utc")
        digest(self.current_fingerprint, "current_fingerprint")
        decision_id(self.head_decision_id, "head_decision_id")
        digest(self.head_committed_digest, "head_committed_digest")
        if type(self.reproduced_policy_row) is not PolicyRow:
            raise TypeError("reproduced_policy_row must be a Decision PolicyRow")
        _bounded(self.canonical_bytes(), "existing-decision receipt")

    def _body(self) -> dict[str, object]:
        return {
            "claim_digest": self.claim_digest,
            "claim_id": self.claim_id,
            "current_fingerprint": self.current_fingerprint,
            "event_id": self.event_id,
            "event_sequence": self.event_sequence,
            "head_committed_digest": self.head_committed_digest,
            "head_decision_id": self.head_decision_id,
            "logical_cycle_id": self.logical_cycle_id,
            "prepare_checked_at_utc": self.prepare_checked_at_utc,
            "reproduced_policy_row": self.reproduced_policy_row.value,
            "review_intent_key": self.review_intent_key,
            "schema_version": SCHEMA_VERSION,
        }

    @property
    def receipt_id(self) -> str:
        return EXISTING_RECEIPT_ID_PREFIX + checksum(_EXISTING_ID_DOMAIN, self._body())

    @property
    def receipt_digest(self) -> str:
        return checksum(_EXISTING_DOMAIN, self._body())

    def canonical_value(self) -> dict[str, object]:
        return {
            **self._body(),
            "receipt_digest": self.receipt_digest,
            "receipt_id": self.receipt_id,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.canonical_value())

    @classmethod
    def from_json(cls, payload: bytes) -> Self:
        row = _parse(
            payload, (*_EXISTING_BODY, "receipt_digest", "receipt_id"), "existing-decision receipt"
        )
        value = cls(
            logical_cycle_id=_str(row["logical_cycle_id"]),
            review_intent_key=_str(row["review_intent_key"]),
            claim_id=_str(row["claim_id"]),
            claim_digest=_str(row["claim_digest"]),
            event_id=_str(row["event_id"]),
            event_sequence=_int(row["event_sequence"]),
            prepare_checked_at_utc=_str(row["prepare_checked_at_utc"]),
            current_fingerprint=_str(row["current_fingerprint"]),
            head_decision_id=_str(row["head_decision_id"]),
            head_committed_digest=_str(row["head_committed_digest"]),
            reproduced_policy_row=parse_enum(
                row["reproduced_policy_row"], PolicyRow, "reproduced_policy_row"
            ),
        )
        if value.canonical_bytes() != payload:
            raise ValueError("existing-decision receipt bytes are not canonical")
        return value


# --------------------------------------------------------------- resolution

_RESULT_PREFIX: Final = {
    ResultKind.DECISION_ISSUED: DECISION_ID_PREFIX,
    ResultKind.EXISTING_DECISION: EXISTING_RECEIPT_ID_PREFIX,
    ResultKind.HOLD: HOLD_RECEIPT_ID_PREFIX,
}
_RESOLUTION_BODY: Final = (
    "claim_digest",
    "claim_id",
    "incarnation_id",
    "logical_cycle_id",
    "result_digest",
    "result_id",
    "result_kind",
    "review_intent_key",
    "schema_version",
)


@dataclass(frozen=True, slots=True)
class LogicalCycleResolutionV1:
    """The one result of a logical cycle (#369 A2-2); UNIQUE
    ``(incarnation_id, logical_cycle_id)`` in IP-14b.

    It points at exactly one CommittedDecision, ExistingDecisionReceipt or
    HoldReceipt, written in the same subject.history operation.
    """

    incarnation_id: str
    logical_cycle_id: str
    review_intent_key: str
    claim_id: str
    claim_digest: str
    result_kind: ResultKind
    result_id: str
    result_digest: str

    def __post_init__(self) -> None:
        digest(self.incarnation_id, "incarnation_id")
        digest(self.logical_cycle_id, "logical_cycle_id")
        digest(self.review_intent_key, "review_intent_key")
        _claim(self.claim_id, self.claim_digest)
        if type(self.result_kind) is not ResultKind:
            raise TypeError("result_kind must be a ResultKind")
        prefixed(self.result_id, _RESULT_PREFIX[self.result_kind], "result_id")
        digest(self.result_digest, "result_digest")

    @classmethod
    def of_decision(
        cls,
        logical_cycle_id: str,
        claim_id: str,
        claim_digest: str,
        decision: CommittedDecisionV1,
    ) -> Self:
        """A Wake-origin Decision carries the cycle's review_intent_key."""
        if type(decision) is not CommittedDecisionV1:
            raise TypeError("decision must be a CommittedDecisionV1")
        return cls(
            decision.incarnation_id,
            logical_cycle_id,
            decision.review_intent_key,
            claim_id,
            claim_digest,
            ResultKind.DECISION_ISSUED,
            decision.decision_id,
            decision.content_digest,
        )

    @classmethod
    def of_hold(cls, incarnation_id: str, receipt: HoldReceiptV1) -> Self:
        if type(receipt) is not HoldReceiptV1:
            raise TypeError("receipt must be a HoldReceiptV1")
        return cls(
            incarnation_id,
            receipt.logical_cycle_id,
            receipt.review_intent_key,
            receipt.claim_id,
            receipt.claim_digest,
            ResultKind.HOLD,
            receipt.hold_id,
            receipt.hold_digest,
        )

    @classmethod
    def of_existing(cls, incarnation_id: str, receipt: ExistingDecisionReceiptV1) -> Self:
        if type(receipt) is not ExistingDecisionReceiptV1:
            raise TypeError("receipt must be an ExistingDecisionReceiptV1")
        return cls(
            incarnation_id,
            receipt.logical_cycle_id,
            receipt.review_intent_key,
            receipt.claim_id,
            receipt.claim_digest,
            ResultKind.EXISTING_DECISION,
            receipt.receipt_id,
            receipt.receipt_digest,
        )

    def canonical_value(self) -> dict[str, object]:
        return {
            "claim_digest": self.claim_digest,
            "claim_id": self.claim_id,
            "incarnation_id": self.incarnation_id,
            "logical_cycle_id": self.logical_cycle_id,
            "result_digest": self.result_digest,
            "result_id": self.result_id,
            "result_kind": self.result_kind.value,
            "review_intent_key": self.review_intent_key,
            "schema_version": SCHEMA_VERSION,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.canonical_value())

    @property
    def resolution_id(self) -> str:
        """One per cycle: built from the cycle identity only."""
        return RESOLUTION_ID_PREFIX + checksum(
            _RESOLUTION_ID_DOMAIN,
            {"incarnation_id": self.incarnation_id, "logical_cycle_id": self.logical_cycle_id},
        )

    @property
    def resolution_digest(self) -> str:
        return checksum(_RESOLUTION_DOMAIN, self.canonical_value())

    @classmethod
    def from_json(cls, payload: bytes) -> Self:
        row = _parse(payload, _RESOLUTION_BODY, "cycle resolution")
        value = cls(
            incarnation_id=_str(row["incarnation_id"]),
            logical_cycle_id=_str(row["logical_cycle_id"]),
            review_intent_key=_str(row["review_intent_key"]),
            claim_id=_str(row["claim_id"]),
            claim_digest=_str(row["claim_digest"]),
            result_kind=parse_enum(row["result_kind"], ResultKind, "result_kind"),
            result_id=_str(row["result_id"]),
            result_digest=_str(row["result_digest"]),
        )
        if value.canonical_bytes() != payload:
            raise ValueError("cycle resolution bytes are not canonical")
        return value


__all__ = [
    "EXISTING_RECEIPT_ID_PREFIX",
    "HOLD_RECEIPT_ID_PREFIX",
    "RESOLUTION_ID_PREFIX",
    "WAKE_CLAIM_ID_PREFIX",
    "ExistingDecisionReceiptV1",
    "FrozenPredecessor",
    "HoldReason",
    "HoldReceiptV1",
    "LogicalCycleResolutionV1",
    "ResultKind",
    "wake_review_intent_key",
]
