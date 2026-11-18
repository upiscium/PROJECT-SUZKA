"""R19 IP-13b: pure Wake / claim / completion / Outbox / Inbox values (D-R19 D1).

Nothing here creates, claims, schedules, persists or delivers anything, and
nothing reads a store or the clock. IP-14b adds the tables, IP-20 the Wake
ledger and Scheduler, IP-21 the Outbox and Inbox. The R16-owned cycle
results (HoldReceipt, ExistingDecisionReceipt, LogicalCycleResolution) live
in ``suzka.r16.cycle_contracts``; R19 refers to them by id and digest.

Identities follow #369 A1-1 / A1-3 / A2-1: ``wakeup_id`` and
``logical_cycle_id`` are built from the incarnation, Goal, kind and frozen
evidence fingerprint only, never from an event, attempt, claim or clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Final, Self

from suzka.limits import MAX_PERSISTED_REVISION
from suzka.r16.contracts import (
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
from suzka.r16.cycle_contracts import (
    WAKE_CLAIM_ID_PREFIX,
    FrozenPredecessor,
    LogicalCycleResolutionV1,
    ResultKind,
    prefixed,
    utc,
)
from suzka.r16.history_codec import PolicyRow, canonical_uuid, event_sequence
from suzka.r16.successor_policy import (
    Interpretation,
    LineageState,
    PolicyRef,
    PredecessorState,
    RequiredSourceState,
    SuccessorPolicyInputV2,
)
from suzka.r17.contracts import FEEDBACK_REF_KIND
from suzka.r18.contracts import OUTCOME_REF_KIND, GoalOrigin

SCHEMA_VERSION: Final = 1
MAX_RECORD_BYTES: Final = 4096
MAX_ORIGIN_REFS: Final = 4
WAKEUP_ID_PREFIX: Final = "wakeup-"
TERMINAL_ID_PREFIX: Final = "wake-terminal-"
MESSAGE_ID_PREFIX: Final = "outbox-"
ACK_ID_PREFIX: Final = "inbox-ack-"
REPLY_ID_PREFIX: Final = "inbox-reply-"
GOAL_ADMISSION_REF_KIND: Final = "goal-admission"
COMMITMENT_ADMISSION_REF_KIND: Final = "commitment-admission"
ASSESSMENT_REF_KIND: Final = "r17-assessment"
COMMITMENT_REF_KIND: Final = "commitment"
OBSERVATION_REF_KIND: Final = "observation"
# IP-14b D1 r2 §4-2 (P2-2): the only Inbox trust claim the MVI makes. A
# client that the SUZKA HTTP API admits, through the existing chat / API route
# (direct loopback, or via Caddy from the networks the firewall already
# admits to :8080), acknowledged this message. It is neither proof that a
# person read it nor an approval. (#373 A1-3's SO_PEERCRED operator socket was
# dropped by the 2026-11-18 scope decision, so its value would be false.)
INBOX_ASSURANCE: Final = "SUZKA_API_CLIENT_V1"

# HP-5 (#411, 2026-11-15).
MAX_ATTEMPTS: Final = 3
MAX_WAKE_TTL: Final = timedelta(days=7)
MAX_MESSAGE_TTL: Final = timedelta(days=7)
ATTEMPT_MAX_EVENTS: Final = 4
ATTEMPT_MAX_MODEL_CALLS: Final = 2
ATTEMPT_MAX_SECONDS: Final = 10
CYCLE_MAX_EVENTS: Final = 12
_UTC_FORMAT: Final = "%Y-%m-%dT%H:%M:%SZ"

_WAKEUP_DOMAIN: Final = b"PROJECT-SUZKA:R19:WAKEUP-ID:V1\0"
_CYCLE_DOMAIN: Final = b"PROJECT-SUZKA:R19:LOGICAL-CYCLE-ID:V1\0"
_WAKE_DOMAIN: Final = b"PROJECT-SUZKA:R19:WAKE:V1\0"
_CLAIM_ID_DOMAIN: Final = b"PROJECT-SUZKA:R19:WAKE-CLAIM-ID:V1\0"
_CLAIM_DOMAIN: Final = b"PROJECT-SUZKA:R19:WAKE-CLAIM:V1\0"
_ATTEMPT_DOMAIN: Final = b"PROJECT-SUZKA:R19:WAKE-ATTEMPT:V1\0"
_TERMINAL_ID_DOMAIN: Final = b"PROJECT-SUZKA:R19:WAKE-TERMINAL-ID:V1\0"
_TERMINAL_DOMAIN: Final = b"PROJECT-SUZKA:R19:WAKE-TERMINAL:V1\0"
_DEDUPE_DOMAIN: Final = b"PROJECT-SUZKA:R19:OUTBOX-DEDUPE:V1\0"
_MESSAGE_ID_DOMAIN: Final = b"PROJECT-SUZKA:R19:OUTBOX-MESSAGE-ID:V1\0"
_MESSAGE_DOMAIN: Final = b"PROJECT-SUZKA:R19:OUTBOX-MESSAGE:V1\0"
_ACK_ID_DOMAIN: Final = b"PROJECT-SUZKA:R19:INBOX-ACK-ID:V1\0"
_ACK_DOMAIN: Final = b"PROJECT-SUZKA:R19:INBOX-ACK:V1\0"
_REPLY_ID_DOMAIN: Final = b"PROJECT-SUZKA:R19:INBOX-REPLY-ID:V1\0"
_REPLY_DOMAIN: Final = b"PROJECT-SUZKA:R19:INBOX-REPLY:V1\0"


class WakeKind(str, Enum):
    GOAL_REEVALUATION = "GOAL_REEVALUATION"


class WakeCause(str, Enum):
    """The closed cause enum (#369 D2-1)."""

    GOAL_ADMITTED = "GOAL_ADMITTED"
    COMMITMENT_ADMITTED = "COMMITMENT_ADMITTED"
    FEEDBACK_ADMITTED = "FEEDBACK_ADMITTED"
    # Never a feedback-positive R20 PASS (#369 A1-2).
    OUTCOME_VERIFIED_NO_FEEDBACK = "OUTCOME_VERIFIED_NO_FEEDBACK"


class AttemptOutcome(str, Enum):
    DEFERRED_BUDGET = "DEFERRED_BUDGET"
    # Derived only (IP-14b D1 r2 §1-1): "now >= the claim's lease" is a value
    # of IP-20's attempt view and is never stored.
    LEASE_EXPIRED = "LEASE_EXPIRED"
    REFUSED = "REFUSED"


# The outcomes SubjectHistory stores (``wake_attempt_outcomes``).
STORED_ATTEMPT_OUTCOMES: Final = frozenset({AttemptOutcome.DEFERRED_BUDGET, AttemptOutcome.REFUSED})


class RefuseReason(str, Enum):
    """D-R19 §4 refuses: no resolution is written and the attempt counts."""

    # (2'): a required source, predecessor, lineage or interpretation is
    # UNAVAILABLE (also policy row 1 from an UNAVAILABLE state).
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    WAKE_TERMINAL = "WAKE_TERMINAL"
    CLAIM_NOT_LATEST = "CLAIM_NOT_LATEST"
    OUTSIDE_WAKE_WINDOW = "OUTSIDE_WAKE_WINDOW"
    INCARNATION_MISMATCH = "INCARNATION_MISMATCH"
    LINEAGE_MISMATCH = "LINEAGE_MISMATCH"
    # Another PREPARED reservation of the same cycle exists.
    CYCLE_RESERVED = "CYCLE_RESERVED"
    SNAPSHOT_CHANGED = "SNAPSHOT_CHANGED"
    CLOCK_UNVERIFIABLE = "CLOCK_UNVERIFIABLE"
    HISTORY_CAPACITY = "HISTORY_CAPACITY"
    # DECISION_SLOT_OCCUPIED by an in-flight PREPARED row (a FINALIZED one is
    # a STALE_SOURCE terminal HOLD instead).
    DECISION_SLOT_PENDING = "DECISION_SLOT_PENDING"


class WakeTerminalKind(str, Enum):
    FAILED_TERMINAL = "FAILED_TERMINAL"
    CANCELLED = "CANCELLED"


class TerminalReason(str, Enum):
    EXPIRED_WAKE = "EXPIRED_WAKE"
    ATTEMPTS_EXHAUSTED = "ATTEMPTS_EXHAUSTED"
    SUPERSEDED = "SUPERSEDED"


class OutboxTemplate(str, Enum):
    DECISION_ISSUED = "DECISION_ISSUED"
    DECISION_CONFIRMED = "DECISION_CONFIRMED"
    DECISION_HELD = "DECISION_HELD"
    CYCLE_FAILED = "CYCLE_FAILED"


class OutboxReason(str, Enum):
    """Coarse codes only: no policy row and no HOLD reason (2b / 4b / 5b
    would reveal the R15 class, #374)."""

    PLAN_SELECTED = "PLAN_SELECTED"
    NO_ACTION = "NO_ACTION"
    HELD = "HELD"
    CONFIRMED = "CONFIRMED"
    FAILED = "FAILED"


class AckStatus(str, Enum):
    ACKNOWLEDGED = "ACKNOWLEDGED"
    REJECTED = "REJECTED"


_CAUSE_REF_KIND: Final = {
    WakeCause.GOAL_ADMITTED: GOAL_ADMISSION_REF_KIND,
    WakeCause.COMMITMENT_ADMITTED: COMMITMENT_ADMISSION_REF_KIND,
    WakeCause.FEEDBACK_ADMITTED: FEEDBACK_REF_KIND,
    WakeCause.OUTCOME_VERIFIED_NO_FEEDBACK: ASSESSMENT_REF_KIND,
}
_OUTCOME_CAUSES: Final = frozenset(
    {WakeCause.FEEDBACK_ADMITTED, WakeCause.OUTCOME_VERIFIED_NO_FEEDBACK}
)
_FAILED_REASONS: Final = frozenset({TerminalReason.EXPIRED_WAKE, TerminalReason.ATTEMPTS_EXHAUSTED})
_TEMPLATE_REASONS: Final = {
    OutboxTemplate.DECISION_ISSUED: frozenset({OutboxReason.PLAN_SELECTED, OutboxReason.NO_ACTION}),
    OutboxTemplate.DECISION_CONFIRMED: frozenset({OutboxReason.CONFIRMED}),
    OutboxTemplate.DECISION_HELD: frozenset({OutboxReason.HELD}),
    OutboxTemplate.CYCLE_FAILED: frozenset({OutboxReason.FAILED}),
}
_RESULT_TEMPLATE: Final = {
    ResultKind.DECISION_ISSUED: OutboxTemplate.DECISION_ISSUED,
    ResultKind.EXISTING_DECISION: OutboxTemplate.DECISION_CONFIRMED,
    ResultKind.HOLD: OutboxTemplate.DECISION_HELD,
}
_NO_ACTION_ROWS: Final = frozenset({PolicyRow.ROW_5, PolicyRow.ROW_5B})


# ------------------------------------------------------------------ helpers


def _str(value: object) -> str:
    if type(value) is not str:
        raise TypeError("expected a string")
    return value


def _int(value: object) -> int:
    if type(value) is not int:
        raise TypeError("expected an integer")
    return value


def _bool(value: object) -> bool:
    if type(value) is not bool:
        raise TypeError("expected a boolean")
    return value


def _opt_str(value: object) -> str | None:
    return None if value is None else _str(value)


def _stamp(value: datetime) -> str:
    return value.strftime(_UTC_FORMAT)


def _revision(value: object, name: str, minimum: int = 0) -> int:
    return integer(value, name, minimum=minimum, maximum=MAX_PERSISTED_REVISION)


def _event(event_id: object, sequence_value: object, name: str) -> None:
    canonical_uuid(event_id, f"{name}_event_id")
    event_sequence(sequence_value, f"{name}_sequence")


def _parse(payload: bytes, names: tuple[str, ...], name: str) -> dict[str, object]:
    if type(payload) is not bytes:
        raise TypeError(f"{name} payload must be bytes")
    if len(payload) > MAX_RECORD_BYTES:
        raise ValueError(f"{name} exceeds its {MAX_RECORD_BYTES}-byte bound")
    row = closed(parse_json(payload, MAX_RECORD_BYTES), names, name)
    if row["schema_version"] != SCHEMA_VERSION or type(row["schema_version"]) is not int:
        raise ValueError(f"{name} schema_version is not V1")
    return row


def _bounded(payload: bytes, name: str) -> None:
    if len(payload) > MAX_RECORD_BYTES:
        raise ValueError(f"{name} exceeds its {MAX_RECORD_BYTES}-byte bound")


def _ref(value: object, name: str) -> PolicyRef:
    if type(value) is not PolicyRef:
        raise TypeError(f"{name} must be a PolicyRef")
    value.__post_init__()
    return value


# --------------------------------------------------------------- identities


def _cycle_body(
    incarnation_id: str, goal_id: str, kind: WakeKind, evidence_fingerprint: str
) -> dict[str, object]:
    digest(incarnation_id, "incarnation_id")
    identifier(goal_id, "goal_id")
    if type(kind) is not WakeKind:
        raise TypeError("kind must be a WakeKind")
    digest(evidence_fingerprint, "evidence_fingerprint")
    return {
        "evidence_fingerprint": evidence_fingerprint,
        "goal_id": goal_id,
        "incarnation_id": incarnation_id,
        "kind": kind.value,
    }


def wakeup_id(incarnation_id: str, goal_id: str, kind: WakeKind, evidence_fingerprint: str) -> str:
    """#369 A1-3: the UNIQUE Wake key. The cause is provenance, not a key part."""
    return WAKEUP_ID_PREFIX + checksum(
        _WAKEUP_DOMAIN, _cycle_body(incarnation_id, goal_id, kind, evidence_fingerprint)
    )


def logical_cycle_id(
    incarnation_id: str, goal_id: str, kind: WakeKind, evidence_fingerprint: str
) -> str:
    """#369 A1-1 / A2-1: one result per logical cycle, across all attempts."""
    return checksum(_CYCLE_DOMAIN, _cycle_body(incarnation_id, goal_id, kind, evidence_fingerprint))


def wake_fingerprint(value: SuccessorPolicyInputV2) -> str | None:
    """The frozen evidence fingerprint, or None when a Wake may not be made.

    D-R19 §3: a Wake is made only from an input whose required source,
    predecessor, lineage and interpretation are all qualified; freezing an
    UNAVAILABLE input would make every re-evaluation refuse. The fingerprint
    is SuccessorPolicyInput's own and covers the R15 ``(stage,
    interpretation)`` pair.
    """

    if type(value) is not SuccessorPolicyInputV2:
        raise TypeError("value must be a SuccessorPolicyInputV2")
    if (
        value.required_source is not RequiredSourceState.QUALIFIED
        or value.predecessor not in (PredecessorState.QUALIFIED, PredecessorState.ABSENT_INITIAL)
        or value.lineage is not LineageState.CONSISTENT
        or value.interpretation is Interpretation.UNAVAILABLE
    ):
        return None
    return value.evidence_fingerprint()


# ----------------------------------------------------------------- schedule


@dataclass(frozen=True, slots=True)
class WakeTiming:
    """The D-R19 §5 numbers, in seconds (``r19.*`` settings in IP-20)."""

    goal_wake_delay_s: int = 3600
    commitment_wake_delay_s: int = 3600
    feedback_wake_delay_s: int = 60
    deadline_margin_s: int = 600
    claim_lease_s: int = 120
    tick_s: int = 30

    def __post_init__(self) -> None:
        for name in (
            "goal_wake_delay_s",
            "commitment_wake_delay_s",
            "feedback_wake_delay_s",
            "deadline_margin_s",
            "claim_lease_s",
            "tick_s",
        ):
            integer(getattr(self, name), name, minimum=0, maximum=int(MAX_WAKE_TTL.total_seconds()))
        if self.claim_lease_s == 0 or self.tick_s == 0:
            raise ValueError("claim lease and tick must be positive")


def wake_window(
    cause: WakeCause,
    *,
    cause_utc: str,
    created_utc: str,
    goal_deadline_utc: str | None,
    commitment_deadline_utc: str | None,
    timing: WakeTiming,
) -> tuple[str, str] | None:
    """``(not_before_utc, expires_at_utc)``, or None for ``wake_not_schedulable``.

    - GOAL: cause time (the admission's ``requested_at``) + goal delay.
    - COMMITMENT: cause time + commitment delay, but no later than the
      Commitment deadline − margin.
    - FEEDBACK / OUTCOME: creation time + feedback delay, but no later than
      expires − margin.
    - expires = min(created + 7 days, Goal deadline, Commitment deadline).

    No Wake is made when not_before is later than expires − margin: the
    deadline is not silently lost (#369 D2-A3); the caller records
    ``wake_not_schedulable``. For FEEDBACK / OUTCOME the cap makes that
    literal rule unreachable, so for those two causes no Wake is made either
    when expires − margin is already before the creation time. GOAL /
    COMMITMENT keep the literal rule (a late materialize may be due at once).
    """

    if type(cause) is not WakeCause:
        raise TypeError("cause must be a WakeCause")
    if type(timing) is not WakeTiming:
        raise TypeError("timing must be a WakeTiming")
    caused = utc(cause_utc, "cause_utc")
    created = utc(created_utc, "created_utc")
    if caused > created:
        raise ValueError("a Wake is created at or after its cause")
    deadlines = [
        utc(value, name)
        for value, name in (
            (goal_deadline_utc, "goal_deadline_utc"),
            (commitment_deadline_utc, "commitment_deadline_utc"),
        )
        if value is not None
    ]
    expires = min([created + MAX_WAKE_TTL, *deadlines])
    margin = timedelta(seconds=timing.deadline_margin_s)
    if cause is WakeCause.GOAL_ADMITTED:
        not_before = caused + timedelta(seconds=timing.goal_wake_delay_s)
    elif cause is WakeCause.COMMITMENT_ADMITTED:
        not_before = caused + timedelta(seconds=timing.commitment_wake_delay_s)
        if commitment_deadline_utc is not None:
            not_before = min(not_before, utc(commitment_deadline_utc, "commitment_deadline_utc") - margin)
    else:
        not_before = min(created + timedelta(seconds=timing.feedback_wake_delay_s), expires - margin)
    if not_before > expires - margin:
        return None
    if cause in _OUTCOME_CAUSES and expires - margin < created:
        return None
    return _stamp(not_before), _stamp(expires)


# --------------------------------------------------------------------- wake

_WAKE_BODY: Final = (
    "cause",
    "cause_event_id",
    "cause_ref",
    "cause_sequence",
    "commitment_deadline_utc",
    "context_id",
    "evidence_fingerprint",
    "experience_id",
    "expires_at_utc",
    "frozen_predecessor",
    "goal_digest",
    "goal_id",
    "goal_origin",
    "goal_revision",
    "horizon_sequence",
    "incarnation_id",
    "journal_lineage_id",
    "kind",
    "logical_cycle_id",
    "not_before_utc",
    "origin_refs",
    "schema_version",
)


@dataclass(frozen=True, slots=True)
class WakeV1:
    """One scheduled Goal re-evaluation (D-R19 §2, #369 D2-1 / A1-3 / A2-1).

    ``evidence_fingerprint`` and ``frozen_predecessor`` are frozen at the
    same creation horizon (``horizon_sequence``). ``origin_refs`` names the
    evidence behind the cause (at most 4, sorted, distinct): the Commitment
    for COMMITMENT_ADMITTED, and exactly one ``r18-outcome`` for the two
    Outcome causes (one Wake per verified Outcome revision, D-R19 §3).
    """

    kind: WakeKind
    cause: WakeCause
    cause_event_id: str
    cause_sequence: int
    cause_ref: PolicyRef
    goal_id: str
    goal_revision: int
    goal_digest: str
    experience_id: str
    goal_origin: GoalOrigin
    origin_refs: tuple[PolicyRef, ...]
    evidence_fingerprint: str
    frozen_predecessor: FrozenPredecessor
    horizon_sequence: int
    incarnation_id: str
    journal_lineage_id: str
    not_before_utc: str
    expires_at_utc: str
    commitment_deadline_utc: str | None
    context_id: str

    def __post_init__(self) -> None:
        if type(self.cause) is not WakeCause:
            raise TypeError("cause must be a WakeCause")
        _event(self.cause_event_id, self.cause_sequence, "cause")
        _ref(self.cause_ref, "cause_ref")
        if self.cause_ref.kind != _CAUSE_REF_KIND[self.cause]:
            raise ValueError(f"a {self.cause.value} Wake cites a {_CAUSE_REF_KIND[self.cause]} ref")
        identifier(self.goal_id, "goal_id")
        _revision(self.goal_revision, "goal_revision", minimum=1)
        digest(self.goal_digest, "goal_digest")
        identifier(self.experience_id, "experience_id")
        if type(self.goal_origin) is not GoalOrigin:
            raise TypeError("goal_origin must be a GoalOrigin")
        refs = sequence(self.origin_refs, "origin_refs", MAX_ORIGIN_REFS, json_input=False)
        for item in refs:
            _ref(item, "origin ref")
        keys = [item.sort_key for item in self.origin_refs]
        if keys != sorted(set(keys)):
            raise ValueError("origin refs are sorted and distinct")
        kinds = [item.kind for item in self.origin_refs]
        if (kinds.count(OUTCOME_REF_KIND) == 1) != (self.cause in _OUTCOME_CAUSES) or kinds.count(
            OUTCOME_REF_KIND
        ) > 1:
            raise ValueError("exactly the Outcome causes cite one verified Outcome")
        if kinds.count(COMMITMENT_REF_KIND) > 1:
            raise ValueError("a Wake cites at most one Commitment")
        if self.cause is WakeCause.COMMITMENT_ADMITTED and COMMITMENT_REF_KIND not in kinds:
            raise ValueError("a COMMITMENT_ADMITTED Wake cites its Commitment")
        if type(self.frozen_predecessor) is not FrozenPredecessor:
            raise TypeError("frozen_predecessor must be a FrozenPredecessor")
        self.frozen_predecessor.__post_init__()
        event_sequence(self.horizon_sequence, "horizon_sequence")
        absence = self.frozen_predecessor.absence_horizon_sequence
        if absence is not None and absence != self.horizon_sequence:
            raise ValueError("predecessor absence and fingerprint share one horizon")
        identifier(self.journal_lineage_id, "journal_lineage_id")
        identifier(self.context_id, "context_id")
        # Derives and checks the identity inputs.
        logical_cycle_id(self.incarnation_id, self.goal_id, self.kind, self.evidence_fingerprint)
        not_before = utc(self.not_before_utc, "not_before_utc")
        expires = utc(self.expires_at_utc, "expires_at_utc")
        if not not_before < expires:
            raise ValueError("a Wake becomes due before it expires")
        if self.commitment_deadline_utc is not None and expires > utc(
            self.commitment_deadline_utc, "commitment_deadline_utc"
        ):
            raise ValueError("a Wake never outlives its Commitment deadline")
        _bounded(self.canonical_bytes(), "wake")

    @property
    def wakeup_id(self) -> str:
        return wakeup_id(self.incarnation_id, self.goal_id, self.kind, self.evidence_fingerprint)

    @property
    def logical_cycle_id(self) -> str:
        return logical_cycle_id(self.incarnation_id, self.goal_id, self.kind, self.evidence_fingerprint)

    @property
    def verified_outcome_key(self) -> tuple[str, int] | None:
        """UNIQUE across both Outcome causes (D-R19 §3)."""
        for item in self.origin_refs:
            if item.kind == OUTCOME_REF_KIND:
                return item.ref_id, item.revision
        return None

    def commitment_ref(self) -> PolicyRef | None:
        return next((item for item in self.origin_refs if item.kind == COMMITMENT_REF_KIND), None)

    def _body(self) -> dict[str, object]:
        return {
            "cause": self.cause.value,
            "cause_event_id": self.cause_event_id,
            "cause_ref": self.cause_ref.canonical_value(),
            "cause_sequence": self.cause_sequence,
            "commitment_deadline_utc": self.commitment_deadline_utc,
            "context_id": self.context_id,
            "evidence_fingerprint": self.evidence_fingerprint,
            "experience_id": self.experience_id,
            "expires_at_utc": self.expires_at_utc,
            "frozen_predecessor": self.frozen_predecessor.canonical_value(),
            "goal_digest": self.goal_digest,
            "goal_id": self.goal_id,
            "goal_origin": self.goal_origin.value,
            "goal_revision": self.goal_revision,
            "horizon_sequence": self.horizon_sequence,
            "incarnation_id": self.incarnation_id,
            "journal_lineage_id": self.journal_lineage_id,
            "kind": self.kind.value,
            "logical_cycle_id": self.logical_cycle_id,
            "not_before_utc": self.not_before_utc,
            "origin_refs": [item.canonical_value() for item in self.origin_refs],
            "schema_version": SCHEMA_VERSION,
        }

    @property
    def wake_digest(self) -> str:
        return checksum(_WAKE_DOMAIN, self._body())

    def canonical_value(self) -> dict[str, object]:
        return {**self._body(), "wake_digest": self.wake_digest, "wakeup_id": self.wakeup_id}

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.canonical_value())

    @classmethod
    def from_json(cls, payload: bytes) -> Self:
        row = _parse(payload, (*_WAKE_BODY, "wake_digest", "wakeup_id"), "wake")
        value = cls(
            kind=parse_enum(row["kind"], WakeKind, "kind"),
            cause=parse_enum(row["cause"], WakeCause, "cause"),
            cause_event_id=_str(row["cause_event_id"]),
            cause_sequence=_int(row["cause_sequence"]),
            cause_ref=PolicyRef.from_value(row["cause_ref"]),
            goal_id=_str(row["goal_id"]),
            goal_revision=_int(row["goal_revision"]),
            goal_digest=_str(row["goal_digest"]),
            experience_id=_str(row["experience_id"]),
            goal_origin=parse_enum(row["goal_origin"], GoalOrigin, "goal_origin"),
            origin_refs=tuple(
                PolicyRef.from_value(item)
                for item in sequence(row["origin_refs"], "origin_refs", MAX_ORIGIN_REFS, json_input=True)
            ),
            evidence_fingerprint=_str(row["evidence_fingerprint"]),
            frozen_predecessor=FrozenPredecessor.from_value(row["frozen_predecessor"]),
            horizon_sequence=_int(row["horizon_sequence"]),
            incarnation_id=_str(row["incarnation_id"]),
            journal_lineage_id=_str(row["journal_lineage_id"]),
            not_before_utc=_str(row["not_before_utc"]),
            expires_at_utc=_str(row["expires_at_utc"]),
            commitment_deadline_utc=_opt_str(row["commitment_deadline_utc"]),
            context_id=_str(row["context_id"]),
        )
        if value.canonical_bytes() != payload:
            raise ValueError("wake bytes are not canonical")
        return value


def may_supersede(old: WakeV1, new: WakeV1) -> bool:
    """D-R19 §3 / #369 A2-3: may a new FEEDBACK_ADMITTED Wake cancel an older,
    unclaimed same-Goal Wake? Otherwise both stay. Whether ``old`` is still
    unclaimed is the caller's (store) fact."""

    if type(old) is not WakeV1 or type(new) is not WakeV1:
        raise TypeError("expected WakeV1 values")
    if new.cause is not WakeCause.FEEDBACK_ADMITTED or old.wakeup_id == new.wakeup_id:
        return False
    old_commitment, new_commitment = old.commitment_ref(), new.commitment_ref()
    same_commitment = old_commitment is None or (
        old_commitment == new_commitment and old.commitment_deadline_utc == new.commitment_deadline_utc
    )
    return (
        old.incarnation_id == new.incarnation_id
        and (old.goal_id, old.goal_revision) == (new.goal_id, new.goal_revision)
        and same_commitment
        and utc(new.not_before_utc, "not_before_utc") <= utc(old.expires_at_utc, "expires_at_utc")
        and old.context_id == new.context_id
    )


# -------------------------------------------------------------------- claim

_CLAIM_BODY: Final = (
    "attempt_index",
    "claim_event_id",
    "claim_lease_expires_utc",
    "claim_sequence",
    "evidence_fingerprint",
    "incarnation_id",
    "logical_cycle_id",
    "schema_version",
    "wakeup_id",
)


@dataclass(frozen=True, slots=True)
class WakeClaimV1:
    """A terminal R07 claim of one attempt, before the R16 event (#369 D2-3).

    Build new claims with ``for_wake``, which checks the window and the
    lease; the plain constructor is for decoding.
    """

    wakeup_id: str
    logical_cycle_id: str
    evidence_fingerprint: str
    incarnation_id: str
    attempt_index: int
    claim_event_id: str
    claim_sequence: int
    claim_lease_expires_utc: str

    def __post_init__(self) -> None:
        prefixed(self.wakeup_id, WAKEUP_ID_PREFIX, "wakeup_id")
        digest(self.logical_cycle_id, "logical_cycle_id")
        digest(self.evidence_fingerprint, "evidence_fingerprint")
        digest(self.incarnation_id, "incarnation_id")
        integer(self.attempt_index, "attempt_index", minimum=1, maximum=MAX_ATTEMPTS)
        _event(self.claim_event_id, self.claim_sequence, "claim")
        utc(self.claim_lease_expires_utc, "claim_lease_expires_utc")

    @classmethod
    def for_wake(
        cls,
        wake: WakeV1,
        attempt_index: int,
        *,
        claim_event_id: str,
        claim_sequence: int,
        claimed_utc: str,
        timing: WakeTiming,
    ) -> Self:
        """Refuses a Wake that is not due or has expired. The lease ends at
        min(claimed + lease, Wake expiry, Commitment deadline)."""
        if type(wake) is not WakeV1:
            raise TypeError("wake must be a WakeV1")
        if type(timing) is not WakeTiming:
            raise TypeError("timing must be a WakeTiming")
        now = utc(claimed_utc, "claimed_utc")
        limits = [utc(wake.expires_at_utc, "expires_at_utc")]
        if wake.commitment_deadline_utc is not None:
            limits.append(utc(wake.commitment_deadline_utc, "commitment_deadline_utc"))
        if not utc(wake.not_before_utc, "not_before_utc") <= now < min(limits):
            raise ValueError("the Wake is not due or has expired")
        lease = min([now + timedelta(seconds=timing.claim_lease_s), *limits])
        return cls(
            wake.wakeup_id,
            wake.logical_cycle_id,
            wake.evidence_fingerprint,
            wake.incarnation_id,
            attempt_index,
            claim_event_id,
            claim_sequence,
            _stamp(lease),
        )

    def _body(self) -> dict[str, object]:
        return {
            "attempt_index": self.attempt_index,
            "claim_event_id": self.claim_event_id,
            "claim_lease_expires_utc": self.claim_lease_expires_utc,
            "claim_sequence": self.claim_sequence,
            "evidence_fingerprint": self.evidence_fingerprint,
            "incarnation_id": self.incarnation_id,
            "logical_cycle_id": self.logical_cycle_id,
            "schema_version": SCHEMA_VERSION,
            "wakeup_id": self.wakeup_id,
        }

    @property
    def claim_id(self) -> str:
        return WAKE_CLAIM_ID_PREFIX + checksum(_CLAIM_ID_DOMAIN, self._body())

    @property
    def claim_digest(self) -> str:
        return checksum(_CLAIM_DOMAIN, self._body())

    def canonical_value(self) -> dict[str, object]:
        return {**self._body(), "claim_digest": self.claim_digest, "claim_id": self.claim_id}

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.canonical_value())

    def in_window(self, wake: WakeV1, now_utc: str) -> bool:
        """D-R19 §4 (2): ``not_before ≤ now < min(Wake expiry, lease)``, and
        before the Commitment deadline (the three deadlines of #373 A4-4)."""
        if type(wake) is not WakeV1:
            raise TypeError("wake must be a WakeV1")
        if wake.wakeup_id != self.wakeup_id:
            raise ValueError("the claim belongs to another Wake")
        now = utc(now_utc, "now_utc")
        limits = [
            utc(wake.expires_at_utc, "expires_at_utc"),
            utc(self.claim_lease_expires_utc, "claim_lease_expires_utc"),
        ]
        if wake.commitment_deadline_utc is not None:
            limits.append(utc(wake.commitment_deadline_utc, "commitment_deadline_utc"))
        return utc(wake.not_before_utc, "not_before_utc") <= now < min(limits)

    @classmethod
    def from_json(cls, payload: bytes) -> Self:
        row = _parse(payload, (*_CLAIM_BODY, "claim_digest", "claim_id"), "wake claim")
        value = cls(
            wakeup_id=_str(row["wakeup_id"]),
            logical_cycle_id=_str(row["logical_cycle_id"]),
            evidence_fingerprint=_str(row["evidence_fingerprint"]),
            incarnation_id=_str(row["incarnation_id"]),
            attempt_index=_int(row["attempt_index"]),
            claim_event_id=_str(row["claim_event_id"]),
            claim_sequence=_int(row["claim_sequence"]),
            claim_lease_expires_utc=_str(row["claim_lease_expires_utc"]),
        )
        if value.canonical_bytes() != payload:
            raise ValueError("wake claim bytes are not canonical")
        return value


def next_attempt_index(prior_attempts: int) -> int | None:
    """The next attempt of one cycle, or None when all 3 are used."""
    integer(prior_attempts, "prior_attempts", minimum=0, maximum=MAX_ATTEMPTS)
    return None if prior_attempts >= MAX_ATTEMPTS else prior_attempts + 1


# ------------------------------------------------------------------ attempt

_ATTEMPT_BODY: Final = ("claim_id", "outcome", "reason", "schema_version")


@dataclass(frozen=True, slots=True)
class WakeAttemptOutcomeV1:
    """An attempt that produced no result; it counts toward the 3 attempts.

    IP-14b D1 r2 (D-1): a REFUSED or DEFERRED_BUDGET outcome is decided by the
    handler before PREPARE and recorded by its own ``R19_WAKE_ATTEMPT_OUTCOME``
    operation; nothing is substituted inside ``prepare``. The row records the
    reason only: it never shortens the gap before the next claim, which is at
    least the previous claim's lease (P2-A). LEASE_EXPIRED is derived from the
    claim and the clock and is never stored."""

    claim_id: str
    outcome: AttemptOutcome
    reason: RefuseReason | None

    def __post_init__(self) -> None:
        prefixed(self.claim_id, WAKE_CLAIM_ID_PREFIX, "claim_id")
        if type(self.outcome) is not AttemptOutcome:
            raise TypeError("outcome must be an AttemptOutcome")
        if self.reason is not None and type(self.reason) is not RefuseReason:
            raise TypeError("reason must be a RefuseReason")
        if (self.outcome is AttemptOutcome.REFUSED) != (self.reason is not None):
            raise ValueError("a refuse reason exists exactly for REFUSED")

    def canonical_value(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "outcome": self.outcome.value,
            "reason": None if self.reason is None else self.reason.value,
            "schema_version": SCHEMA_VERSION,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.canonical_value())

    @property
    def attempt_digest(self) -> str:
        return checksum(_ATTEMPT_DOMAIN, self.canonical_value())

    @classmethod
    def from_json(cls, payload: bytes) -> Self:
        row = _parse(payload, _ATTEMPT_BODY, "wake attempt outcome")
        reason = row["reason"]
        value = cls(
            claim_id=_str(row["claim_id"]),
            outcome=parse_enum(row["outcome"], AttemptOutcome, "outcome"),
            reason=None if reason is None else parse_enum(reason, RefuseReason, "reason"),
        )
        if value.canonical_bytes() != payload:
            raise ValueError("wake attempt bytes are not canonical")
        return value


# ----------------------------------------------------------------- terminal

_TERMINAL_BODY: Final = (
    "event_id",
    "event_sequence",
    "logical_cycle_id",
    "reason",
    "schema_version",
    "superseding_wakeup_id",
    "terminal",
    "wakeup_id",
)


@dataclass(frozen=True, slots=True)
class WakeTerminalV1:
    """FAILED_TERMINAL (expired / 3 attempts) or CANCELLED (SUPERSEDED).

    Neither is retried, and an unchanged fingerprint is never rescheduled.
    ``logical_cycle_id`` is the Wake's (IP-14b-1): the terminal reserves the
    cycle's ``cycle:`` key, so a cycle closes once, by a result or a terminal.
    """

    wakeup_id: str
    logical_cycle_id: str
    terminal: WakeTerminalKind
    reason: TerminalReason
    superseding_wakeup_id: str | None
    event_id: str
    event_sequence: int

    def __post_init__(self) -> None:
        prefixed(self.wakeup_id, WAKEUP_ID_PREFIX, "wakeup_id")
        digest(self.logical_cycle_id, "logical_cycle_id")
        if type(self.terminal) is not WakeTerminalKind:
            raise TypeError("terminal must be a WakeTerminalKind")
        if type(self.reason) is not TerminalReason:
            raise TypeError("reason must be a TerminalReason")
        failed = self.terminal is WakeTerminalKind.FAILED_TERMINAL
        if failed != (self.reason in _FAILED_REASONS):
            raise ValueError("FAILED_TERMINAL is expiry or exhaustion; CANCELLED is supersession")
        if (self.reason is TerminalReason.SUPERSEDED) != (self.superseding_wakeup_id is not None):
            raise ValueError("a superseding Wake is named exactly for SUPERSEDED")
        if self.superseding_wakeup_id is not None:
            prefixed(self.superseding_wakeup_id, WAKEUP_ID_PREFIX, "superseding_wakeup_id")
            if self.superseding_wakeup_id == self.wakeup_id:
                raise ValueError("a Wake does not supersede itself")
        _event(self.event_id, self.event_sequence, "terminal")

    def _body(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "event_sequence": self.event_sequence,
            "logical_cycle_id": self.logical_cycle_id,
            "reason": self.reason.value,
            "schema_version": SCHEMA_VERSION,
            "superseding_wakeup_id": self.superseding_wakeup_id,
            "terminal": self.terminal.value,
            "wakeup_id": self.wakeup_id,
        }

    @classmethod
    def for_wake(
        cls,
        wake: WakeV1,
        terminal: WakeTerminalKind,
        reason: TerminalReason,
        *,
        event_id: str,
        event_sequence: int,
        superseding: WakeV1 | None = None,
    ) -> Self:
        """The terminal of ``wake``: its id and its own logical cycle (whose
        ``cycle:`` key the terminal reserves). A superseding Wake must be of
        the same incarnation."""
        if type(wake) is not WakeV1:
            raise TypeError("wake must be a WakeV1")
        if superseding is not None:
            if type(superseding) is not WakeV1:
                raise TypeError("superseding must be a WakeV1")
            if superseding.incarnation_id != wake.incarnation_id:
                raise ValueError("a Wake is superseded only within its incarnation")
        return cls(
            wake.wakeup_id,
            wake.logical_cycle_id,
            terminal,
            reason,
            None if superseding is None else superseding.wakeup_id,
            event_id,
            event_sequence,
        )

    @property
    def terminal_id(self) -> str:
        """One terminal per Wake: built from the Wake only."""
        return TERMINAL_ID_PREFIX + checksum(_TERMINAL_ID_DOMAIN, {"wakeup_id": self.wakeup_id})

    @property
    def terminal_digest(self) -> str:
        return checksum(_TERMINAL_DOMAIN, self._body())

    def canonical_value(self) -> dict[str, object]:
        return {**self._body(), "terminal_digest": self.terminal_digest, "terminal_id": self.terminal_id}

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.canonical_value())

    @classmethod
    def from_json(cls, payload: bytes) -> Self:
        row = _parse(payload, (*_TERMINAL_BODY, "terminal_digest", "terminal_id"), "wake terminal")
        value = cls(
            wakeup_id=_str(row["wakeup_id"]),
            logical_cycle_id=_str(row["logical_cycle_id"]),
            terminal=parse_enum(row["terminal"], WakeTerminalKind, "terminal"),
            reason=parse_enum(row["reason"], TerminalReason, "reason"),
            superseding_wakeup_id=_opt_str(row["superseding_wakeup_id"]),
            event_id=_str(row["event_id"]),
            event_sequence=_int(row["event_sequence"]),
        )
        if value.canonical_bytes() != payload:
            raise ValueError("wake terminal bytes are not canonical")
        return value


# --------------------------------------------------------------- completion


def notification_required(resolution: LogicalCycleResolutionV1) -> bool:
    """IP-14b D1 r2 D-2: whether a resolved cycle gets one Outbox message.

    A pure function of the resolution (it replaces the removed
    ``WakeCompletionV1``). Every result kind has its own template
    (DECISION_ISSUED / DECISION_CONFIRMED / DECISION_HELD), so every
    resolution is notified; ``R19_WAKE_COMPLETE`` writes only that message."""

    if type(resolution) is not LogicalCycleResolutionV1:
        raise TypeError("resolution must be a LogicalCycleResolutionV1")
    if resolution.result_kind not in _RESULT_TEMPLATE:
        # A closed set: a new result kind needs its own decision here.
        raise ValueError("the resolution's result kind has no notification rule")
    return True


# ------------------------------------------------------------------- outbox

_MESSAGE_BODY: Final = (
    "context_id",
    "created_utc",
    "dedupe_key",
    "expires_utc",
    "incarnation_id",
    "reason_code",
    "result_digest",
    "schema_version",
    "source_id",
    "template",
)


def outbox_reason_for(resolution_kind: ResultKind, decision_row: PolicyRow | None) -> OutboxReason:
    """The coarse code for a resolution; a Decision needs its row."""
    if type(resolution_kind) is not ResultKind:
        raise TypeError("resolution_kind must be a ResultKind")
    if (resolution_kind is ResultKind.DECISION_ISSUED) != (decision_row is not None):
        raise ValueError("a Decision row is given exactly for DECISION_ISSUED")
    if resolution_kind is ResultKind.EXISTING_DECISION:
        return OutboxReason.CONFIRMED
    if resolution_kind is ResultKind.HOLD:
        return OutboxReason.HELD
    if type(decision_row) is not PolicyRow:
        raise TypeError("decision_row must be a PolicyRow")
    return OutboxReason.NO_ACTION if decision_row in _NO_ACTION_ROWS else OutboxReason.PLAN_SELECTED


@dataclass(frozen=True, slots=True)
class OutboxMessageV1:
    """One logical private Inbox message (D-R19 §2, #369 D2-5).

    ``source_id`` is the logical cycle (for a resolution) or the WakeTerminal
    id (for CYCLE_FAILED). The body is generated by code from the template;
    no policy row, HOLD reason, raw cognition or Experience text is carried.
    """

    source_id: str
    result_digest: str
    template: OutboxTemplate
    reason_code: OutboxReason
    incarnation_id: str
    context_id: str
    created_utc: str
    expires_utc: str

    def __post_init__(self) -> None:
        if type(self.template) is not OutboxTemplate:
            raise TypeError("template must be an OutboxTemplate")
        if self.template is OutboxTemplate.CYCLE_FAILED:
            prefixed(self.source_id, TERMINAL_ID_PREFIX, "source_id")
        else:
            digest(self.source_id, "source_id")
        digest(self.result_digest, "result_digest")
        if type(self.reason_code) is not OutboxReason:
            raise TypeError("reason_code must be an OutboxReason")
        if self.reason_code not in _TEMPLATE_REASONS[self.template]:
            raise ValueError("the reason code does not belong to the template")
        digest(self.incarnation_id, "incarnation_id")
        identifier(self.context_id, "context_id")
        created = utc(self.created_utc, "created_utc")
        expires = utc(self.expires_utc, "expires_utc")
        if not created < expires <= created + MAX_MESSAGE_TTL:
            raise ValueError("a message expires after creation, within 7 days")

    @classmethod
    def for_completion(
        cls,
        wake: WakeV1,
        resolution: LogicalCycleResolutionV1,
        decision_row: PolicyRow | None,
        *,
        created_utc: str,
        expires_utc: str,
    ) -> Self:
        """Context and incarnation come from the Wake, which must own the
        cycle. ``decision_row`` is the issued Decision's ``policy_row``; the
        caller (IP-21) reads it from that Decision."""
        if type(wake) is not WakeV1:
            raise TypeError("wake must be a WakeV1")
        if type(resolution) is not LogicalCycleResolutionV1:
            raise TypeError("resolution must be a LogicalCycleResolutionV1")
        if (resolution.incarnation_id, resolution.logical_cycle_id) != (
            wake.incarnation_id,
            wake.logical_cycle_id,
        ):
            raise ValueError("the resolution is not this Wake's cycle")
        return cls(
            resolution.logical_cycle_id,
            resolution.result_digest,
            _RESULT_TEMPLATE[resolution.result_kind],
            outbox_reason_for(resolution.result_kind, decision_row),
            wake.incarnation_id,
            wake.context_id,
            created_utc,
            expires_utc,
        )

    @classmethod
    def for_failed_cycle(
        cls,
        wake: WakeV1,
        terminal: WakeTerminalV1,
        *,
        created_utc: str,
        expires_utc: str,
    ) -> Self:
        if type(wake) is not WakeV1:
            raise TypeError("wake must be a WakeV1")
        if type(terminal) is not WakeTerminalV1:
            raise TypeError("terminal must be a WakeTerminalV1")
        if terminal.wakeup_id != wake.wakeup_id:
            raise ValueError("the terminal is not this Wake's")
        if terminal.terminal is not WakeTerminalKind.FAILED_TERMINAL:
            raise ValueError("only a FAILED_TERMINAL cycle is reported as failed")
        if terminal.logical_cycle_id != wake.logical_cycle_id:
            raise ValueError("the terminal is not this Wake's cycle")
        return cls(
            terminal.terminal_id,
            terminal.terminal_digest,
            OutboxTemplate.CYCLE_FAILED,
            OutboxReason.FAILED,
            wake.incarnation_id,
            wake.context_id,
            created_utc,
            expires_utc,
        )

    @property
    def dedupe_key(self) -> str:
        """Stable across crash and rescan; an EXISTING confirmation has its
        own result digest, so it is never absorbed by the original
        DECISION_ISSUED message."""
        return checksum(
            _DEDUPE_DOMAIN,
            {
                "context_id": self.context_id,
                "incarnation_id": self.incarnation_id,
                "result_digest": self.result_digest,
                "source_id": self.source_id,
                "template": self.template.value,
            },
        )

    @property
    def message_id(self) -> str:
        return MESSAGE_ID_PREFIX + checksum(_MESSAGE_ID_DOMAIN, {"dedupe_key": self.dedupe_key})

    def _body(self) -> dict[str, object]:
        return {
            "context_id": self.context_id,
            "created_utc": self.created_utc,
            "dedupe_key": self.dedupe_key,
            "expires_utc": self.expires_utc,
            "incarnation_id": self.incarnation_id,
            "reason_code": self.reason_code.value,
            "result_digest": self.result_digest,
            "schema_version": SCHEMA_VERSION,
            "source_id": self.source_id,
            "template": self.template.value,
        }

    @property
    def message_digest(self) -> str:
        return checksum(_MESSAGE_DOMAIN, self._body())

    def canonical_value(self) -> dict[str, object]:
        return {**self._body(), "message_digest": self.message_digest, "message_id": self.message_id}

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.canonical_value())

    @classmethod
    def from_json(cls, payload: bytes) -> Self:
        row = _parse(payload, (*_MESSAGE_BODY, "message_digest", "message_id"), "outbox message")
        value = cls(
            source_id=_str(row["source_id"]),
            result_digest=_str(row["result_digest"]),
            template=parse_enum(row["template"], OutboxTemplate, "template"),
            reason_code=parse_enum(row["reason_code"], OutboxReason, "reason_code"),
            incarnation_id=_str(row["incarnation_id"]),
            context_id=_str(row["context_id"]),
            created_utc=_str(row["created_utc"]),
            expires_utc=_str(row["expires_utc"]),
        )
        if value.canonical_bytes() != payload:
            raise ValueError("outbox message bytes are not canonical")
        return value


# -------------------------------------------------------------------- inbox

_ACK_BODY: Final = (
    "context_id",
    "event_id",
    "event_sequence",
    "inbox_assurance",
    "message_id",
    "occurred_utc",
    "schema_version",
    "session_selector",
    "status",
)


def _inbox_common(message_id: object, context_id: object, session_selector: object) -> None:
    prefixed(message_id, MESSAGE_ID_PREFIX, "message_id")
    identifier(context_id, "context_id")
    identifier(session_selector, "session_selector")


@dataclass(frozen=True, slots=True)
class InboxAckV1:
    """ACK / REJECT of one message (#369 D1, #373 A1-3).

    ACK means a client the SUZKA API admits acknowledged (``INBOX_ASSURANCE``);
    it does not mean a person read it, and it approves nothing. One status per message: the
    same status again is idempotent, the other one is a conflict."""

    message_id: str
    context_id: str
    session_selector: str
    status: AckStatus
    event_id: str
    event_sequence: int
    occurred_utc: str

    def __post_init__(self) -> None:
        _inbox_common(self.message_id, self.context_id, self.session_selector)
        if type(self.status) is not AckStatus:
            raise TypeError("status must be an AckStatus")
        _event(self.event_id, self.event_sequence, "ack")
        utc(self.occurred_utc, "occurred_utc")

    def _body(self) -> dict[str, object]:
        return {
            "context_id": self.context_id,
            "event_id": self.event_id,
            "event_sequence": self.event_sequence,
            "inbox_assurance": INBOX_ASSURANCE,
            "message_id": self.message_id,
            "occurred_utc": self.occurred_utc,
            "schema_version": SCHEMA_VERSION,
            "session_selector": self.session_selector,
            "status": self.status.value,
        }

    @property
    def ack_id(self) -> str:
        """One ACK slot per message (at most one ACK state transition)."""
        return ACK_ID_PREFIX + checksum(_ACK_ID_DOMAIN, {"message_id": self.message_id})

    @property
    def ack_digest(self) -> str:
        return checksum(_ACK_DOMAIN, self._body())

    def conflicts_with(self, other: InboxAckV1) -> bool:
        """True for an ACK↔REJECT (or wrong-Context) answer to the same message."""
        if type(other) is not InboxAckV1:
            raise TypeError("other must be an InboxAckV1")
        if other.message_id != self.message_id:
            return False
        return (self.status, self.context_id, self.session_selector) != (
            other.status,
            other.context_id,
            other.session_selector,
        )

    def canonical_value(self) -> dict[str, object]:
        return {**self._body(), "ack_digest": self.ack_digest, "ack_id": self.ack_id}

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.canonical_value())

    @classmethod
    def from_json(cls, payload: bytes) -> Self:
        row = _parse(payload, (*_ACK_BODY, "ack_digest", "ack_id"), "inbox ack")
        if row["inbox_assurance"] != INBOX_ASSURANCE:
            raise ValueError("unknown inbox assurance")
        value = cls(
            message_id=_str(row["message_id"]),
            context_id=_str(row["context_id"]),
            session_selector=_str(row["session_selector"]),
            status=parse_enum(row["status"], AckStatus, "status"),
            event_id=_str(row["event_id"]),
            event_sequence=_int(row["event_sequence"]),
            occurred_utc=_str(row["occurred_utc"]),
        )
        if value.canonical_bytes() != payload:
            raise ValueError("inbox ack bytes are not canonical")
        return value


_REPLY_BODY: Final = (
    "context_id",
    "inbox_assurance",
    "message_id",
    "observation_ref",
    "received_event_id",
    "received_sequence",
    "schema_version",
    "session_selector",
)


@dataclass(frozen=True, slots=True)
class InboxReplyV1:
    """A reply to one message, held as an Observation reference (#369 D0.2).

    The text enters through the accepted Chat / R12 route; this record only
    binds the reply to the message. It is never a Goal, Value, approval or
    verified Outcome."""

    message_id: str
    context_id: str
    session_selector: str
    received_event_id: str
    received_sequence: int
    observation_ref: PolicyRef

    def __post_init__(self) -> None:
        _inbox_common(self.message_id, self.context_id, self.session_selector)
        _event(self.received_event_id, self.received_sequence, "received")
        _ref(self.observation_ref, "observation_ref")
        if self.observation_ref.kind != OBSERVATION_REF_KIND:
            raise ValueError("a reply cites an observation ref")

    def _body(self) -> dict[str, object]:
        return {
            "context_id": self.context_id,
            "inbox_assurance": INBOX_ASSURANCE,
            "message_id": self.message_id,
            "observation_ref": self.observation_ref.canonical_value(),
            "received_event_id": self.received_event_id,
            "received_sequence": self.received_sequence,
            "schema_version": SCHEMA_VERSION,
            "session_selector": self.session_selector,
        }

    @property
    def reply_id(self) -> str:
        """A replayed event never makes a second reply."""
        return REPLY_ID_PREFIX + checksum(
            _REPLY_ID_DOMAIN,
            {"message_id": self.message_id, "received_event_id": self.received_event_id},
        )

    @property
    def reply_digest(self) -> str:
        return checksum(_REPLY_DOMAIN, self._body())

    def canonical_value(self) -> dict[str, object]:
        return {**self._body(), "reply_digest": self.reply_digest, "reply_id": self.reply_id}

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.canonical_value())

    @classmethod
    def from_json(cls, payload: bytes) -> Self:
        row = _parse(payload, (*_REPLY_BODY, "reply_digest", "reply_id"), "inbox reply")
        if row["inbox_assurance"] != INBOX_ASSURANCE:
            raise ValueError("unknown inbox assurance")
        value = cls(
            message_id=_str(row["message_id"]),
            context_id=_str(row["context_id"]),
            session_selector=_str(row["session_selector"]),
            received_event_id=_str(row["received_event_id"]),
            received_sequence=_int(row["received_sequence"]),
            observation_ref=PolicyRef.from_value(row["observation_ref"]),
        )
        if value.canonical_bytes() != payload:
            raise ValueError("inbox reply bytes are not canonical")
        return value
