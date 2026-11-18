"""R18 IP-13: pure governed-Action values and lifecycles (#368, #380 under H3).

Nothing here issues, persists, approves or executes anything, reads a store or
touches the target. Validation rejects internally inconsistent values only;
whether a value came from the trusted issuer, a live operator or the real
verifier is the job of IP-16 / IP-17 / IP-18.

The values follow the #368 producer order: ActionIntent → (R17 expectation)
→ PolicyVerdict → ApprovalChallenge → ExecutionClaim → ExecutionReceipt →
VerifiedOutcome. A challenge can only be built from this intent's ALLOW, and
the receipt is the executor's report, never the verifier's basis.

Under Human decision H3 (#380) there is no signer, signature or owner epoch:
the approval is a single-use server challenge bound to exact digests, and a
code derived from it that the operator types back. The physical ReviewArtifact
layout optimisation (#371 D6-A9/A10) is Post-R26; V1 fixes the artifact to the
six source references, the closed code-owned gate results and their summary.
"""

from __future__ import annotations

from base64 import b32encode
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from hashlib import sha256
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
from suzka.r16.history_codec import (
    PolicyKind,
    PolicyRow,
    canonical_uuid,
    decision_id,
    event_sequence,
)
from suzka.r16.successor_policy import InverseAuthorization, LedgerState
from suzka.r17.contracts import ObservedState, OutcomeState, VerifiedOutcomeView

SCHEMA_VERSION: Final = 1
MAX_ARTIFACT_BYTES: Final = 8192
MAX_RECORD_BYTES: Final = 16_384
# #380: a challenge is short-lived; the claim must happen before it expires.
MAX_APPROVAL_TTL: Final = timedelta(minutes=15)
CONFIRMATION_CODE_LENGTH: Final = 8
INTENT_ID_PREFIX: Final = "intent-"
INVERSE_INTENT_ID_PREFIX: Final = "inverse-intent-"
CLAIM_ID_PREFIX: Final = "claim-"
OUTCOME_ID_PREFIX: Final = "outcome-"
OUTCOME_REF_KIND: Final = "r18-outcome"
_UTC_FORMAT: Final = "%Y-%m-%dT%H:%M:%SZ"

_GROUP_DOMAIN: Final = b"PROJECT-SUZKA:R18:OBLIGATION-GROUP:V1\0"
_OPERATION_DOMAIN: Final = b"PROJECT-SUZKA:R18:OPERATION-KEY:V1\0"
_ARTIFACT_DOMAIN: Final = b"PROJECT-SUZKA:R18:REVIEW-ARTIFACT:V1\0"
_INTENT_ID_DOMAIN: Final = b"PROJECT-SUZKA:R18:INTENT-ID:V1\0"
_INTENT_DOMAIN: Final = b"PROJECT-SUZKA:R18:INTENT:V1\0"
_INVERSE_ID_DOMAIN: Final = b"PROJECT-SUZKA:R18:INVERSE-INTENT-ID:V1\0"
_INVERSE_DOMAIN: Final = b"PROJECT-SUZKA:R18:INVERSE-INTENT:V1\0"
_POLICY_DOMAIN: Final = b"PROJECT-SUZKA:R18:POLICY-VERDICT:V1\0"
_CHALLENGE_DOMAIN: Final = b"PROJECT-SUZKA:R18:APPROVAL-CHALLENGE:V1\0"
_CODE_DOMAIN: Final = b"PROJECT-SUZKA:R18:CONFIRMATION-CODE:V1\0"
_CLAIM_ID_DOMAIN: Final = b"PROJECT-SUZKA:R18:CLAIM-ID:V1\0"
_CLAIM_DOMAIN: Final = b"PROJECT-SUZKA:R18:CLAIM:V1\0"
_RECEIPT_DOMAIN: Final = b"PROJECT-SUZKA:R18:EXECUTION-RECEIPT:V1\0"
_OUTCOME_ID_DOMAIN: Final = b"PROJECT-SUZKA:R18:OUTCOME-ID:V1\0"
_OUTCOME_DOMAIN: Final = b"PROJECT-SUZKA:R18:OUTCOME:V1\0"


class Direction(str, Enum):
    FORWARD = "FORWARD"
    INVERSE = "INVERSE"


class ApprovalChannel(str, Enum):
    """How the Human's approval reached the Claim (IP-17b, #411 D1 r7 §12-1).

    A closed enum that names what the approval proves (§9): V1 is a deliberate
    request accepted from localhost or a trusted network, after the
    code-generated preview, with the typed confirmation code.  It does not
    prove that the Human made it.  A stronger channel is a new member.
    """

    LOCAL_LOOPBACK_TYPED_CODE_V1 = "LOCAL_LOOPBACK_TYPED_CODE_V1"


class GoalOrigin(str, Enum):
    """The origin of the Goal behind a Decision (#379; mirrors GoalOrigin)."""

    USER_REQUEST = "USER_REQUEST"
    SUBJECT_MOTIVATION = "SUBJECT_MOTIVATION"


class SourceRole(str, Enum):
    """The six Forward source roles C (#371 D6-A6)."""

    COMMITMENT = "COMMITMENT"
    CONTEXT = "CONTEXT"
    DECISION = "DECISION"
    EXPERIENCE = "EXPERIENCE"
    GOAL = "GOAL"
    VALUE = "VALUE"


class ReviewGate(str, Enum):
    """The closed, code-owned V1 gate roster; every gate is required."""

    COMMITMENT_ACTIVE_SAME_X = "COMMITMENT_ACTIVE_SAME_X"
    CONTEXT_CURRENT = "CONTEXT_CURRENT"
    DECISION_ROW4_TERMINAL = "DECISION_ROW4_TERMINAL"
    EXPERIENCE_TERMINAL = "EXPERIENCE_TERMINAL"
    GOAL_ADOPTED = "GOAL_ADOPTED"
    VALUE_QUALIFIED = "VALUE_QUALIFIED"


class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


class VerifiedState(str, Enum):
    """Terminal verifier results: forward (R17's OutcomeState) and inverse."""

    ACTIVE_VERIFIED_AUTHORIZED = "ACTIVE_VERIFIED_AUTHORIZED"
    VERIFIED_ABSENT = "VERIFIED_ABSENT"
    UNAUTHORIZED_EFFECT = "UNAUTHORIZED_EFFECT"
    REVERTED = "REVERTED"
    INVERSE_VERIFIED_ABSENT = "INVERSE_VERIFIED_ABSENT"


# A V1 Decision predates the R18 ledger states (#371 V2).
_INTENT_POLICY_KINDS: Final = frozenset(
    {PolicyKind.SUCCESSOR_POLICY_V2, PolicyKind.SUCCESSOR_POLICY_V3}
)
_GATE_ROLE: Final = {
    ReviewGate.COMMITMENT_ACTIVE_SAME_X: SourceRole.COMMITMENT,
    ReviewGate.CONTEXT_CURRENT: SourceRole.CONTEXT,
    ReviewGate.DECISION_ROW4_TERMINAL: SourceRole.DECISION,
    ReviewGate.EXPERIENCE_TERMINAL: SourceRole.EXPERIENCE,
    ReviewGate.GOAL_ADOPTED: SourceRole.GOAL,
    ReviewGate.VALUE_QUALIFIED: SourceRole.VALUE,
}
_FORWARD_STATES: Final = frozenset(
    {
        VerifiedState.ACTIVE_VERIFIED_AUTHORIZED,
        VerifiedState.VERIFIED_ABSENT,
        VerifiedState.UNAUTHORIZED_EFFECT,
    }
)
_INVERTIBLE: Final = frozenset(
    {VerifiedState.ACTIVE_VERIFIED_AUTHORIZED, VerifiedState.UNAUTHORIZED_EFFECT}
)


# ------------------------------------------------------------------ helpers


def _str(value: object) -> str:
    if type(value) is not str:
        raise TypeError("expected a string")
    return value


def _int(value: object) -> int:
    if type(value) is not int:
        raise TypeError("expected an integer")
    return value


def _revision(value: object, name: str, minimum: int = 0) -> int:
    return integer(value, name, minimum=minimum, maximum=MAX_PERSISTED_REVISION)


def _event(event_id: object, sequence_value: object, name: str) -> None:
    canonical_uuid(event_id, f"{name}_event_id")
    event_sequence(sequence_value, f"{name}_sequence")


def _utc(value: object, name: str) -> datetime:
    if type(value) is not str:
        raise TypeError(f"{name} must be a UTC timestamp string")
    try:
        parsed = datetime.strptime(value, _UTC_FORMAT)
    except ValueError as error:
        raise ValueError(f"{name} must be a UTC timestamp") from error
    if parsed.strftime(_UTC_FORMAT) != value:
        raise ValueError(f"{name} must be a canonical UTC timestamp")
    return parsed.replace(tzinfo=timezone.utc)


def _prefixed(value: object, prefix: str, name: str) -> str:
    if (
        type(value) is not str
        or not value.startswith(prefix)
        or len(value) != len(prefix) + 64
    ):
        raise ValueError(f"{name} must be a {prefix}<sha256> identity")
    digest(value[len(prefix) :], name)
    return value


def _intent_prefix(direction: Direction) -> str:
    return INTENT_ID_PREFIX if direction is Direction.FORWARD else INVERSE_INTENT_ID_PREFIX


def _bounded(payload: bytes, maximum: int, name: str) -> bytes:
    if len(payload) > maximum:
        raise ValueError(f"{name} exceeds its {maximum}-byte bound")
    return payload


def _parse(payload: bytes, maximum: int, names: tuple[str, ...], name: str) -> dict[str, object]:
    if type(payload) is not bytes:
        raise TypeError(f"{name} payload must be bytes")
    row = closed(parse_json(_bounded(payload, maximum, name), maximum), names, name)
    if row["schema_version"] != SCHEMA_VERSION or type(row["schema_version"]) is not int:
        raise ValueError(f"{name} schema_version is not V1")
    return row


# --------------------------------------------------------------------- keys


def obligation_group_key(
    subject_id: str,
    experience_content_digest: str,
    obligation_scope: str,
    target_namespace: str,
) -> str:
    """#373 A1-5 / A2-4: the stable physical obligation identity.

    Only incarnation-independent source content goes in: the subject, the
    content identity of Experience X, the obligation scope and the target
    namespace. No incarnation, receipt, revision or event metadata, so the
    key survives a true rollback and the target UNIQUE keeps one effect.
    """

    identifier(subject_id, "subject_id")
    digest(experience_content_digest, "experience_content_digest")
    identifier(obligation_scope, "obligation_scope")
    identifier(target_namespace, "target_namespace")
    return checksum(
        _GROUP_DOMAIN,
        {
            "experience_content_digest": experience_content_digest,
            "obligation_scope": obligation_scope,
            "subject_id": subject_id,
            "target_namespace": target_namespace,
        },
    )


def operation_key(group_key: str, direction: Direction) -> str:
    """The stable forward / inverse operation key of one obligation group.

    #380: the target holds UNIQUE(operation_key) and
    UNIQUE(obligation_group_key, direction); both keys are stable across
    retries, Decisions and restarts, so a FENCED key is never reissued.
    """

    digest(group_key, "obligation_group_key")
    if type(direction) is not Direction:
        raise TypeError("direction must be a Direction")
    return checksum(
        _OPERATION_DOMAIN, {"direction": direction.value, "obligation_group_key": group_key}
    )


# ---------------------------------------------------------- review artifact


@dataclass(frozen=True, slots=True)
class ReviewSourceRef:
    """One of the six Forward source roles, by digest only."""

    role: SourceRole
    ref_id: str
    revision: int
    digest: str

    def __post_init__(self) -> None:
        if type(self.role) is not SourceRole:
            raise TypeError("role must be a SourceRole")
        identifier(self.ref_id, "source ref_id")
        _revision(self.revision, "source revision")
        digest(self.digest, "source digest")

    def canonical_value(self) -> dict[str, object]:
        return {
            "digest": self.digest,
            "ref_id": self.ref_id,
            "revision": self.revision,
            "role": self.role.value,
        }

    @classmethod
    def from_value(cls, value: object) -> Self:
        row = closed(value, ("digest", "ref_id", "revision", "role"), "review source")
        return cls(
            parse_enum(row["role"], SourceRole, "role"),
            _str(row["ref_id"]),
            _int(row["revision"]),
            _str(row["digest"]),
        )


@dataclass(frozen=True, slots=True)
class GateResult:
    """One code-owned gate result, evaluated before the artifact is built."""

    gate: ReviewGate
    status: GateStatus

    def __post_init__(self) -> None:
        if type(self.gate) is not ReviewGate:
            raise TypeError("gate must be a ReviewGate")
        if type(self.status) is not GateStatus:
            raise TypeError("status must be a GateStatus")

    def canonical_value(self) -> dict[str, object]:
        return {"gate": self.gate.value, "status": self.status.value}

    @classmethod
    def from_value(cls, value: object) -> Self:
        row = closed(value, ("gate", "status"), "gate result")
        return cls(
            parse_enum(row["gate"], ReviewGate, "gate"),
            parse_enum(row["status"], GateStatus, "status"),
        )


@dataclass(frozen=True, slots=True)
class ReviewArtifactV1:
    """The bytes the fixed forward effect writes (#368 D1, #380 §3).

    It records which sources were checked and how every required gate came
    out, with UNKNOWN kept as UNKNOWN. It is not a subjective review, a
    verified review completion or a fulfilled Commitment. No raw subject ID,
    Experience text or hidden thought is in it.
    """

    target_namespace: str
    obligation_group_key: str
    sources: tuple[ReviewSourceRef, ...]
    gates: tuple[GateResult, ...]

    def __post_init__(self) -> None:
        identifier(self.target_namespace, "target_namespace")
        digest(self.obligation_group_key, "obligation_group_key")
        sequence(self.sources, "sources", len(SourceRole), json_input=False)
        sequence(self.gates, "gates", len(ReviewGate), json_input=False)
        for item in self.sources:
            if type(item) is not ReviewSourceRef:
                raise TypeError("sources must be ReviewSourceRef values")
            item.__post_init__()
        for gate in self.gates:
            if type(gate) is not GateResult:
                raise TypeError("gates must be GateResult values")
            gate.__post_init__()
        if tuple(item.role for item in self.sources) != tuple(sorted(SourceRole)):
            raise ValueError("an artifact names each of the six source roles once, in order")
        if tuple(gate.gate for gate in self.gates) != tuple(sorted(ReviewGate)):
            raise ValueError("an artifact carries the whole gate roster once, in order")
        _bounded(self.payload_bytes(), MAX_ARTIFACT_BYTES, "review artifact")

    @property
    def actionable(self) -> bool:
        """#371 D6-A9-R1: any required gate not PASS means no Action."""
        return all(gate.status is GateStatus.PASS for gate in self.gates)

    def source(self, role: SourceRole) -> ReviewSourceRef:
        return next(item for item in self.sources if item.role is role)

    def canonical_value(self) -> dict[str, object]:
        return {
            "gates": [gate.canonical_value() for gate in self.gates],
            "obligation_group_key": self.obligation_group_key,
            "schema_version": SCHEMA_VERSION,
            "sources": [item.canonical_value() for item in self.sources],
            "summary": "PASS" if self.actionable else "BLOCKED",
            "target_namespace": self.target_namespace,
        }

    def payload_bytes(self) -> bytes:
        """The exact bytes P the target stores (M_target = M_I = S_P(P))."""
        return canonical_json(self.canonical_value())

    @property
    def payload_digest(self) -> str:
        """The one payload digest the challenge binds and the verifier
        recomputes from the stored bytes (#380 D2 addendum 4)."""
        return payload_digest(self.payload_bytes())

    @classmethod
    def from_value(cls, value: object) -> Self:
        row = closed(
            value,
            ("gates", "obligation_group_key", "schema_version", "sources", "summary", "target_namespace"),
            "review artifact",
        )
        if row["schema_version"] != SCHEMA_VERSION or type(row["schema_version"]) is not int:
            raise ValueError("review artifact schema_version is not V1")
        artifact = cls(
            _str(row["target_namespace"]),
            _str(row["obligation_group_key"]),
            tuple(
                ReviewSourceRef.from_value(item)
                for item in sequence(row["sources"], "sources", len(SourceRole), json_input=True)
            ),
            tuple(
                GateResult.from_value(item)
                for item in sequence(row["gates"], "gates", len(ReviewGate), json_input=True)
            ),
        )
        if row["summary"] != artifact.canonical_value()["summary"]:
            raise ValueError("artifact summary does not follow from its gates")
        return artifact

    @classmethod
    def from_payload(cls, payload: bytes) -> Self:
        if type(payload) is not bytes:
            raise TypeError("payload must be bytes")
        artifact = cls.from_value(parse_json(_bounded(payload, MAX_ARTIFACT_BYTES, "artifact"), MAX_ARTIFACT_BYTES))
        if artifact.payload_bytes() != payload:
            raise ValueError("artifact bytes are not canonical")
        return artifact


def payload_digest(payload: bytes) -> str:
    """The #380 challenge / verifier payload-digest domain over P bytes."""
    if type(payload) is not bytes:
        raise TypeError("payload must be bytes")
    return sha256(_ARTIFACT_DOMAIN + payload).hexdigest()


# ------------------------------------------------------------ action intent

_INTENT_BODY: Final = (
    "admitting_event_id",
    "admitting_sequence",
    "approval_window_end_utc",
    "artifact",
    "decision_committed_digest",
    "decision_id",
    "direction",
    "forward_key",
    "goal_origin",
    "incarnation_id",
    "lineage_scope",
    "obligation_group_key",
    "policy_kind",
    "policy_row",
    "prestate",
    "schema_version",
    "selected_candidate_digest",
    "selected_candidate_id",
    "target_expected_revision",
    "target_instance_id",
    "target_namespace",
)


@dataclass(frozen=True, slots=True)
class ActionIntentV1:
    """The terminal forward ActionIntent (#368 D1, #380 §2-1).

    Only from a terminal row-4 CommittedDecision of a user-request Goal: a
    self-origin (MOTIVATION) Goal is refused (#368 2026-11-14, defence in depth
    with R16's Plan B ban). The artifact P is carried by value (#371 D6-A7).

    IP-17a (#411 D1 r7 §3a / §12-1): ``approval_window_end_utc`` is the end of
    the approval window, fixed by the issuer and bound into ``intent_digest``.
    At or after it an unclaimed forward Intent is closed
    (APPROVAL_EXPIRED_UNCLAIMED, derived by the ledger reader with no event).
    """

    lineage_scope: str
    incarnation_id: str
    admitting_event_id: str
    admitting_sequence: int
    decision_id: str
    decision_committed_digest: str
    policy_kind: PolicyKind
    policy_row: PolicyRow
    selected_candidate_id: str
    selected_candidate_digest: str
    goal_origin: GoalOrigin
    target_instance_id: str
    target_expected_revision: int
    artifact: ReviewArtifactV1
    approval_window_end_utc: str

    def __post_init__(self) -> None:
        identifier(self.lineage_scope, "lineage_scope")
        digest(self.incarnation_id, "incarnation_id")
        _event(self.admitting_event_id, self.admitting_sequence, "admitting")
        decision_id(self.decision_id, "decision_id")
        digest(self.decision_committed_digest, "decision_committed_digest")
        if type(self.policy_kind) is not PolicyKind:
            raise TypeError("policy_kind must be a PolicyKind")
        if self.policy_kind not in _INTENT_POLICY_KINDS:
            raise ValueError("an ActionIntent follows a SuccessorPolicy V2 or V3 Decision")
        if type(self.policy_row) is not PolicyRow or self.policy_row is not PolicyRow.ROW_4:
            raise ValueError("only a row 4 (Plan B) Decision yields a forward ActionIntent")
        identifier(self.selected_candidate_id, "selected_candidate_id")
        digest(self.selected_candidate_digest, "selected_candidate_digest")
        if type(self.goal_origin) is not GoalOrigin:
            raise TypeError("goal_origin must be a GoalOrigin")
        if self.goal_origin is not GoalOrigin.USER_REQUEST:
            raise ValueError("a self-origin Goal never yields an ActionIntent")
        canonical_uuid(self.target_instance_id, "target_instance_id")
        _revision(self.target_expected_revision, "target_expected_revision")
        if type(self.artifact) is not ReviewArtifactV1:
            raise TypeError("artifact must be a ReviewArtifactV1")
        self.artifact.__post_init__()
        if not self.artifact.actionable:
            raise ValueError("a required gate is not PASS: no ActionIntent")
        decision_source = self.artifact.source(SourceRole.DECISION)
        if (decision_source.ref_id, decision_source.digest) != (
            self.decision_id,
            self.decision_committed_digest,
        ):
            raise ValueError("the artifact's DECISION source is not this Decision")
        _utc(self.approval_window_end_utc, "approval_window_end_utc")
        _bounded(self.canonical_bytes(), MAX_RECORD_BYTES, "action intent")

    @property
    def target_namespace(self) -> str:
        return self.artifact.target_namespace

    @property
    def obligation_group_key(self) -> str:
        return self.artifact.obligation_group_key

    @property
    def forward_key(self) -> str:
        return operation_key(self.obligation_group_key, Direction.FORWARD)

    @property
    def payload_digest(self) -> str:
        return self.artifact.payload_digest

    @property
    def approval_window_end(self) -> datetime:
        """The window end as an aware UTC moment."""
        return _utc(self.approval_window_end_utc, "approval_window_end_utc")

    def window_open(self, now: datetime) -> bool:
        """§3a: ``now < approval_window_end_utc`` (an aware clock read)."""
        if type(now) is not datetime or now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("the window clock must be timezone-aware")
        return now < self.approval_window_end

    def _body(self) -> dict[str, object]:
        return {
            "admitting_event_id": self.admitting_event_id,
            "admitting_sequence": self.admitting_sequence,
            "approval_window_end_utc": self.approval_window_end_utc,
            "artifact": self.artifact.canonical_value(),
            "decision_committed_digest": self.decision_committed_digest,
            "decision_id": self.decision_id,
            "direction": Direction.FORWARD.value,
            "forward_key": self.forward_key,
            "goal_origin": self.goal_origin.value,
            "incarnation_id": self.incarnation_id,
            "lineage_scope": self.lineage_scope,
            "obligation_group_key": self.obligation_group_key,
            "policy_kind": self.policy_kind.value,
            "policy_row": self.policy_row.value,
            "prestate": "ABSENT",
            "schema_version": SCHEMA_VERSION,
            "selected_candidate_digest": self.selected_candidate_digest,
            "selected_candidate_id": self.selected_candidate_id,
            "target_expected_revision": self.target_expected_revision,
            "target_instance_id": self.target_instance_id,
            "target_namespace": self.target_namespace,
        }

    @property
    def intent_id(self) -> str:
        return INTENT_ID_PREFIX + checksum(_INTENT_ID_DOMAIN, self._body())

    @property
    def intent_digest(self) -> str:
        return checksum(_INTENT_DOMAIN, self._body())

    def canonical_value(self) -> dict[str, object]:
        return {**self._body(), "intent_digest": self.intent_digest, "intent_id": self.intent_id}

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.canonical_value())

    @classmethod
    def from_json(cls, payload: bytes) -> Self:
        row = _parse(
            payload, MAX_RECORD_BYTES, (*_INTENT_BODY, "intent_digest", "intent_id"), "action intent"
        )
        if row["direction"] != Direction.FORWARD.value or row["prestate"] != "ABSENT":
            raise ValueError("a V1 ActionIntent is a FORWARD from ABSENT")
        value = cls(
            lineage_scope=_str(row["lineage_scope"]),
            incarnation_id=_str(row["incarnation_id"]),
            admitting_event_id=_str(row["admitting_event_id"]),
            admitting_sequence=_int(row["admitting_sequence"]),
            decision_id=_str(row["decision_id"]),
            decision_committed_digest=_str(row["decision_committed_digest"]),
            policy_kind=parse_enum(row["policy_kind"], PolicyKind, "policy_kind"),
            policy_row=parse_enum(row["policy_row"], PolicyRow, "policy_row"),
            selected_candidate_id=_str(row["selected_candidate_id"]),
            selected_candidate_digest=_str(row["selected_candidate_digest"]),
            goal_origin=parse_enum(row["goal_origin"], GoalOrigin, "goal_origin"),
            target_instance_id=_str(row["target_instance_id"]),
            target_expected_revision=_int(row["target_expected_revision"]),
            artifact=ReviewArtifactV1.from_value(row["artifact"]),
            approval_window_end_utc=_str(row["approval_window_end_utc"]),
        )
        if value.canonical_bytes() != payload:
            raise ValueError("action intent bytes are not canonical")
        return value


# ----------------------------------------------------------- inverse intent

_INVERSE_BODY: Final = (
    "admitting_event_id",
    "admitting_sequence",
    "direction",
    "forward_key",
    "forward_outcome_digest",
    "forward_outcome_id",
    "forward_outcome_revision",
    "forward_outcome_state",
    "forward_payload_digest",
    "incarnation_id",
    "inverse_key",
    "obligation_group_key",
    "schema_version",
    "target_expected_revision",
    "target_instance_id",
    "target_namespace",
)


@dataclass(frozen=True, slots=True)
class InverseIntentV1:
    """The operator-initiated inverse of one verified forward effect.

    #368 D2-1 / #380 §2-7: a separate authority, not a subject Plan B WRITE.
    It needs its own approval and verification but no R17 forecast. It may
    start only after ACTIVE_VERIFIED_AUTHORIZED or UNAUTHORIZED_EFFECT.
    """

    incarnation_id: str
    admitting_event_id: str
    admitting_sequence: int
    obligation_group_key: str
    target_instance_id: str
    target_namespace: str
    target_expected_revision: int
    forward_outcome_id: str
    forward_outcome_revision: int
    forward_outcome_digest: str
    forward_outcome_state: VerifiedState
    forward_payload_digest: str

    def __post_init__(self) -> None:
        digest(self.incarnation_id, "incarnation_id")
        _event(self.admitting_event_id, self.admitting_sequence, "admitting")
        digest(self.obligation_group_key, "obligation_group_key")
        canonical_uuid(self.target_instance_id, "target_instance_id")
        identifier(self.target_namespace, "target_namespace")
        _revision(self.target_expected_revision, "target_expected_revision")
        _prefixed(self.forward_outcome_id, OUTCOME_ID_PREFIX, "forward_outcome_id")
        _revision(self.forward_outcome_revision, "forward_outcome_revision", minimum=1)
        digest(self.forward_outcome_digest, "forward_outcome_digest")
        if type(self.forward_outcome_state) is not VerifiedState:
            raise TypeError("forward_outcome_state must be a VerifiedState")
        if self.forward_outcome_state not in _INVERTIBLE:
            raise ValueError("an inverse starts only after a verified forward effect")
        digest(self.forward_payload_digest, "forward_payload_digest")
        _bounded(self.canonical_bytes(), MAX_RECORD_BYTES, "inverse intent")

    @property
    def forward_key(self) -> str:
        return operation_key(self.obligation_group_key, Direction.FORWARD)

    @property
    def inverse_key(self) -> str:
        return operation_key(self.obligation_group_key, Direction.INVERSE)

    def _body(self) -> dict[str, object]:
        return {
            "admitting_event_id": self.admitting_event_id,
            "admitting_sequence": self.admitting_sequence,
            "direction": Direction.INVERSE.value,
            "forward_key": self.forward_key,
            "forward_outcome_digest": self.forward_outcome_digest,
            "forward_outcome_id": self.forward_outcome_id,
            "forward_outcome_revision": self.forward_outcome_revision,
            "forward_outcome_state": self.forward_outcome_state.value,
            "forward_payload_digest": self.forward_payload_digest,
            "incarnation_id": self.incarnation_id,
            "inverse_key": self.inverse_key,
            "obligation_group_key": self.obligation_group_key,
            "schema_version": SCHEMA_VERSION,
            "target_expected_revision": self.target_expected_revision,
            "target_instance_id": self.target_instance_id,
            "target_namespace": self.target_namespace,
        }

    @property
    def intent_id(self) -> str:
        return INVERSE_INTENT_ID_PREFIX + checksum(_INVERSE_ID_DOMAIN, self._body())

    @property
    def intent_digest(self) -> str:
        return checksum(_INVERSE_DOMAIN, self._body())

    def canonical_value(self) -> dict[str, object]:
        return {**self._body(), "intent_digest": self.intent_digest, "intent_id": self.intent_id}

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.canonical_value())

    @classmethod
    def from_json(cls, payload: bytes) -> Self:
        row = _parse(
            payload, MAX_RECORD_BYTES, (*_INVERSE_BODY, "intent_digest", "intent_id"), "inverse intent"
        )
        if row["direction"] != Direction.INVERSE.value:
            raise ValueError("an inverse intent is INVERSE")
        value = cls(
            incarnation_id=_str(row["incarnation_id"]),
            admitting_event_id=_str(row["admitting_event_id"]),
            admitting_sequence=_int(row["admitting_sequence"]),
            obligation_group_key=_str(row["obligation_group_key"]),
            target_instance_id=_str(row["target_instance_id"]),
            target_namespace=_str(row["target_namespace"]),
            target_expected_revision=_int(row["target_expected_revision"]),
            forward_outcome_id=_str(row["forward_outcome_id"]),
            forward_outcome_revision=_int(row["forward_outcome_revision"]),
            forward_outcome_digest=_str(row["forward_outcome_digest"]),
            forward_outcome_state=parse_enum(
                row["forward_outcome_state"], VerifiedState, "forward_outcome_state"
            ),
            forward_payload_digest=_str(row["forward_payload_digest"]),
        )
        if value.canonical_bytes() != payload:
            raise ValueError("inverse intent bytes are not canonical")
        return value


# ------------------------------------------------------------------ policy


class PolicyVerdict(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"


class PolicyDenyReason(str, Enum):
    """Closed deny reasons of the R18 Policy owner (deny by default, #368)."""

    # The startup reconciliation sweep has not finished (#380 §2-6).
    ADMISSION_BLOCKED = "ADMISSION_BLOCKED"
    # A source, Goal, Value, Context or Commitment no longer matches the intent.
    SOURCE_NOT_CURRENT = "SOURCE_NOT_CURRENT"
    # The target is not the fixed namespace / instance / expected revision.
    TARGET_MISMATCH = "TARGET_MISMATCH"


_POLICY_BODY: Final = (
    "deny_reason",
    "direction",
    "intent_digest",
    "intent_id",
    "operation_key",
    "policy_event_id",
    "policy_sequence",
    "schema_version",
    "verdict",
)


@dataclass(frozen=True, slots=True)
class PolicyVerdictV1:
    """The private R18 Policy result for one intent (#368, #380 §2-2).

    It is evaluated after the R17 expectation commit and before any
    challenge. ALLOW only permits asking the operator; it is not an approval.
    DENY is terminal (POLICY_DENIED / INVERSE_POLICY_DENIED, no reissue).
    """

    direction: Direction
    intent_id: str
    intent_digest: str
    operation_key: str
    verdict: PolicyVerdict
    deny_reason: PolicyDenyReason | None
    policy_event_id: str
    policy_sequence: int

    def __post_init__(self) -> None:
        if type(self.direction) is not Direction:
            raise TypeError("direction must be a Direction")
        _prefixed(self.intent_id, _intent_prefix(self.direction), "intent_id")
        digest(self.intent_digest, "intent_digest")
        digest(self.operation_key, "operation_key")
        if type(self.verdict) is not PolicyVerdict:
            raise TypeError("verdict must be a PolicyVerdict")
        if self.deny_reason is not None and type(self.deny_reason) is not PolicyDenyReason:
            raise TypeError("deny_reason must be a PolicyDenyReason")
        if (self.verdict is PolicyVerdict.DENY) != (self.deny_reason is not None):
            raise ValueError("a deny reason exists exactly for DENY")
        _event(self.policy_event_id, self.policy_sequence, "policy")

    @property
    def denied_state(self) -> LedgerState | None:
        """The terminal ledger state of a DENY; None for ALLOW."""
        if self.verdict is PolicyVerdict.ALLOW:
            return None
        if self.direction is Direction.FORWARD:
            return LedgerState.POLICY_DENIED
        return LedgerState.INVERSE_POLICY_DENIED

    def canonical_value(self) -> dict[str, object]:
        return {
            "deny_reason": None if self.deny_reason is None else self.deny_reason.value,
            "direction": self.direction.value,
            "intent_digest": self.intent_digest,
            "intent_id": self.intent_id,
            "operation_key": self.operation_key,
            "policy_event_id": self.policy_event_id,
            "policy_sequence": self.policy_sequence,
            "schema_version": SCHEMA_VERSION,
            "verdict": self.verdict.value,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.canonical_value())

    @property
    def verdict_digest(self) -> str:
        return checksum(_POLICY_DOMAIN, self.canonical_value())

    @classmethod
    def from_json(cls, payload: bytes) -> Self:
        row = _parse(payload, MAX_RECORD_BYTES, _POLICY_BODY, "policy verdict")
        reason = row["deny_reason"]
        value = cls(
            direction=parse_enum(row["direction"], Direction, "direction"),
            intent_id=_str(row["intent_id"]),
            intent_digest=_str(row["intent_digest"]),
            operation_key=_str(row["operation_key"]),
            verdict=parse_enum(row["verdict"], PolicyVerdict, "verdict"),
            deny_reason=None
            if reason is None
            else parse_enum(reason, PolicyDenyReason, "deny_reason"),
            policy_event_id=_str(row["policy_event_id"]),
            policy_sequence=_int(row["policy_sequence"]),
        )
        if value.canonical_bytes() != payload:
            raise ValueError("policy verdict bytes are not canonical")
        return value


def _require_allow(
    policy: PolicyVerdictV1,
    direction: Direction,
    intent_id: str,
    intent_digest: str,
    key: str,
) -> None:
    """#368 producer order: a challenge only follows this intent's ALLOW."""
    if type(policy) is not PolicyVerdictV1:
        raise TypeError("policy must be a PolicyVerdictV1")
    if policy.verdict is not PolicyVerdict.ALLOW:
        raise ValueError("a denied intent gets no challenge")
    if (policy.direction, policy.intent_id, policy.intent_digest, policy.operation_key) != (
        direction,
        intent_id,
        intent_digest,
        key,
    ):
        raise ValueError("the policy verdict is for another intent")


# --------------------------------------------------------------- challenge

_CHALLENGE_BODY: Final = (
    "boot_id",
    "commitment_deadline_utc",
    "direction",
    "expectation_digest",
    "expires_utc",
    "intent_digest",
    "intent_id",
    "issued_utc",
    "nonce",
    "operation_key",
    "payload_digest",
    "schema_version",
    "target_expected_revision",
    "target_instance_id",
)


@dataclass(frozen=True, slots=True)
class ApprovalChallengeV1:
    """A single-use server challenge (#380 §2-3, D1 P2-2, D2 addendum 4).

    It binds intent, R17 expectation (forward only), target revision,
    operation key, payload digest, expiry, the Commitment deadline and boot.
    Consuming the nonce is once-only in SubjectHistory and happens in the
    same event as the Claim (IP-17c).

    ``boot_id`` (IP-17b, Human decision HD-10 (b), #411 D1 r7 §3): a fresh
    UUID4 per API process, separate from ``runtime_instance_id``.  #380 D2
    P2-1 called it the kernel boot unit; it is the process, so *any* Runtime
    restart voids every open challenge (BOOT_CHANGED), whether or not the host
    rebooted.

    ``commitment_deadline_utc`` (IP-17b, §12-1) is the qualified Commitment's
    deadline (``None``: it has none; always ``None`` for an inverse).  It is
    part of ``challenge_digest`` and so of the confirmation code the operator
    types, and the Claim carries it (A1-4).  A challenge never outlives it.

    Build new challenges with ``for_forward`` / ``for_inverse``, which
    require this intent's ALLOW; the plain constructor is for decoding.
    """

    nonce: str
    direction: Direction
    intent_id: str
    intent_digest: str
    expectation_digest: str | None
    target_instance_id: str
    target_expected_revision: int
    operation_key: str
    payload_digest: str
    issued_utc: str
    expires_utc: str
    boot_id: str
    commitment_deadline_utc: str | None

    def __post_init__(self) -> None:
        digest(self.nonce, "nonce")
        if type(self.direction) is not Direction:
            raise TypeError("direction must be a Direction")
        _prefixed(self.intent_id, _intent_prefix(self.direction), "intent_id")
        digest(self.intent_digest, "intent_digest")
        if self.direction is Direction.FORWARD:
            digest(self.expectation_digest, "expectation_digest")
        elif self.expectation_digest is not None:
            raise ValueError("an inverse needs no R17 expectation (#367 D2-4)")
        canonical_uuid(self.target_instance_id, "target_instance_id")
        _revision(self.target_expected_revision, "target_expected_revision")
        digest(self.operation_key, "operation_key")
        digest(self.payload_digest, "payload_digest")
        issued = _utc(self.issued_utc, "issued_utc")
        expires = _utc(self.expires_utc, "expires_utc")
        if not issued < expires <= issued + MAX_APPROVAL_TTL:
            raise ValueError("a challenge expires after issue, within the maximum TTL")
        canonical_uuid(self.boot_id, "boot_id")
        if self.commitment_deadline_utc is not None:
            deadline = _utc(self.commitment_deadline_utc, "commitment_deadline_utc")
            if self.direction is Direction.INVERSE:
                raise ValueError("an inverse challenge carries no Commitment deadline")
            if expires > deadline:
                raise ValueError("a challenge never outlives the Commitment deadline")

    @classmethod
    def for_forward(
        cls,
        intent: ActionIntentV1,
        expectation_digest: str,
        policy: PolicyVerdictV1,
        *,
        nonce: str,
        issued_utc: str,
        expires_utc: str,
        boot_id: str,
        commitment_deadline_utc: str | None,
    ) -> Self:
        if type(intent) is not ActionIntentV1:
            raise TypeError("intent must be an ActionIntentV1")
        _require_allow(policy, Direction.FORWARD, intent.intent_id, intent.intent_digest, intent.forward_key)
        if _utc(expires_utc, "expires_utc") > intent.approval_window_end:
            raise ValueError("a challenge never outlives its Intent's approval window")
        return cls(
            nonce,
            Direction.FORWARD,
            intent.intent_id,
            intent.intent_digest,
            expectation_digest,
            intent.target_instance_id,
            intent.target_expected_revision,
            intent.forward_key,
            intent.payload_digest,
            issued_utc,
            expires_utc,
            boot_id,
            commitment_deadline_utc,
        )

    @classmethod
    def for_inverse(
        cls,
        intent: InverseIntentV1,
        policy: PolicyVerdictV1,
        *,
        nonce: str,
        issued_utc: str,
        expires_utc: str,
        boot_id: str,
    ) -> Self:
        if type(intent) is not InverseIntentV1:
            raise TypeError("intent must be an InverseIntentV1")
        _require_allow(policy, Direction.INVERSE, intent.intent_id, intent.intent_digest, intent.inverse_key)
        return cls(
            nonce,
            Direction.INVERSE,
            intent.intent_id,
            intent.intent_digest,
            None,
            intent.target_instance_id,
            intent.target_expected_revision,
            intent.inverse_key,
            intent.forward_payload_digest,
            issued_utc,
            expires_utc,
            boot_id,
            None,
        )

    def canonical_value(self) -> dict[str, object]:
        return {
            "boot_id": self.boot_id,
            "commitment_deadline_utc": self.commitment_deadline_utc,
            "direction": self.direction.value,
            "expectation_digest": self.expectation_digest,
            "expires_utc": self.expires_utc,
            "intent_digest": self.intent_digest,
            "intent_id": self.intent_id,
            "issued_utc": self.issued_utc,
            "nonce": self.nonce,
            "operation_key": self.operation_key,
            "payload_digest": self.payload_digest,
            "schema_version": SCHEMA_VERSION,
            "target_expected_revision": self.target_expected_revision,
            "target_instance_id": self.target_instance_id,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.canonical_value())

    @property
    def challenge_digest(self) -> str:
        return checksum(_CHALLENGE_DOMAIN, self.canonical_value())

    @property
    def confirmation_code(self) -> str:
        """The short code the CLI shows and the operator types back (#380 D1
        P2-1). Derived from the whole challenge, so it changes with any
        bound value; it is a typing check, not a secret."""
        raw = sha256(_CODE_DOMAIN + self.challenge_digest.encode("ascii")).digest()
        return b32encode(raw).decode("ascii")[:CONFIRMATION_CODE_LENGTH]

    def is_live(self, now_utc: str, boot_id: str) -> bool:
        return (
            canonical_uuid(boot_id, "boot_id") == self.boot_id
            and _utc(self.issued_utc, "issued_utc")
            <= _utc(now_utc, "now_utc")
            < _utc(self.expires_utc, "expires_utc")
        )

    @classmethod
    def from_json(cls, payload: bytes) -> Self:
        row = _parse(payload, MAX_RECORD_BYTES, _CHALLENGE_BODY, "approval challenge")
        expectation = row["expectation_digest"]
        deadline = row["commitment_deadline_utc"]
        value = cls(
            nonce=_str(row["nonce"]),
            direction=parse_enum(row["direction"], Direction, "direction"),
            intent_id=_str(row["intent_id"]),
            intent_digest=_str(row["intent_digest"]),
            expectation_digest=None if expectation is None else _str(expectation),
            target_instance_id=_str(row["target_instance_id"]),
            target_expected_revision=_int(row["target_expected_revision"]),
            operation_key=_str(row["operation_key"]),
            payload_digest=_str(row["payload_digest"]),
            issued_utc=_str(row["issued_utc"]),
            expires_utc=_str(row["expires_utc"]),
            boot_id=_str(row["boot_id"]),
            commitment_deadline_utc=None if deadline is None else _str(deadline),
        )
        if value.canonical_bytes() != payload:
            raise ValueError("challenge bytes are not canonical")
        return value


# -------------------------------------------------------------------- claim

_CLAIM_BODY: Final = (
    "approval_channel",
    "boot_id",
    "challenge_digest",
    "claim_event_id",
    "claim_sequence",
    "claimed_utc",
    "commitment_deadline_utc",
    "direction",
    "intent_digest",
    "intent_id",
    "nonce",
    "operation_key",
    "runtime_instance_id",
    "schema_version",
)


@dataclass(frozen=True, slots=True)
class ExecutionClaimV1:
    """EXECUTION_CLAIM, terminal before the effect (#380 §2-4, D1 P2-2).

    The executor checks the same ``runtime_instance_id`` in the target CAS
    transaction (#380 D2 P2-1 / addendum 1).

    IP-17b (#411 D1 r7 §12-1) amends the shape only (the CLAIM handler is
    IP-17c): ``commitment_deadline_utc`` is the challenge's Commitment
    deadline (copied, A1-4) and ``approval_channel`` names what the approval
    proves.

    Build new claims with ``claim``, which refuses a stale challenge; the
    plain constructor is for decoding.
    """

    direction: Direction
    intent_id: str
    intent_digest: str
    operation_key: str
    challenge_digest: str
    nonce: str
    runtime_instance_id: str
    boot_id: str
    claim_event_id: str
    claim_sequence: int
    claimed_utc: str
    commitment_deadline_utc: str | None
    approval_channel: ApprovalChannel

    def __post_init__(self) -> None:
        if type(self.direction) is not Direction:
            raise TypeError("direction must be a Direction")
        _prefixed(self.intent_id, _intent_prefix(self.direction), "intent_id")
        digest(self.intent_digest, "intent_digest")
        digest(self.operation_key, "operation_key")
        digest(self.challenge_digest, "challenge_digest")
        digest(self.nonce, "nonce")
        canonical_uuid(self.runtime_instance_id, "runtime_instance_id")
        canonical_uuid(self.boot_id, "boot_id")
        _event(self.claim_event_id, self.claim_sequence, "claim")
        claimed = _utc(self.claimed_utc, "claimed_utc")
        if type(self.approval_channel) is not ApprovalChannel:
            raise TypeError("approval_channel must be an ApprovalChannel")
        if self.commitment_deadline_utc is not None:
            if self.direction is Direction.INVERSE:
                raise ValueError("an inverse claim carries no Commitment deadline")
            if claimed >= _utc(self.commitment_deadline_utc, "commitment_deadline_utc"):
                raise ValueError("a claim is made before the Commitment deadline")

    @classmethod
    def claim(
        cls,
        challenge: ApprovalChallengeV1,
        *,
        runtime_instance_id: str,
        boot_id: str,
        claim_event_id: str,
        claim_sequence: int,
        claimed_utc: str,
        approval_channel: ApprovalChannel,
    ) -> Self:
        """Refuses an expired challenge or one from another boot, or a claim at
        or after the challenge's Commitment deadline."""
        if not challenge.is_live(claimed_utc, boot_id):
            raise ValueError("the challenge expired or belongs to an earlier boot")
        return cls(
            challenge.direction,
            challenge.intent_id,
            challenge.intent_digest,
            challenge.operation_key,
            challenge.challenge_digest,
            challenge.nonce,
            runtime_instance_id,
            boot_id,
            claim_event_id,
            claim_sequence,
            claimed_utc,
            challenge.commitment_deadline_utc,
            approval_channel,
        )

    def _body(self) -> dict[str, object]:
        return {
            "approval_channel": self.approval_channel.value,
            "boot_id": self.boot_id,
            "challenge_digest": self.challenge_digest,
            "claim_event_id": self.claim_event_id,
            "claim_sequence": self.claim_sequence,
            "claimed_utc": self.claimed_utc,
            "commitment_deadline_utc": self.commitment_deadline_utc,
            "direction": self.direction.value,
            "intent_digest": self.intent_digest,
            "intent_id": self.intent_id,
            "nonce": self.nonce,
            "operation_key": self.operation_key,
            "runtime_instance_id": self.runtime_instance_id,
            "schema_version": SCHEMA_VERSION,
        }

    @property
    def claim_id(self) -> str:
        return CLAIM_ID_PREFIX + checksum(_CLAIM_ID_DOMAIN, self._body())

    @property
    def claim_digest(self) -> str:
        return checksum(_CLAIM_DOMAIN, self._body())

    def canonical_value(self) -> dict[str, object]:
        return {**self._body(), "claim_digest": self.claim_digest, "claim_id": self.claim_id}

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.canonical_value())

    @classmethod
    def from_json(cls, payload: bytes) -> Self:
        row = _parse(
            payload, MAX_RECORD_BYTES, (*_CLAIM_BODY, "claim_digest", "claim_id"), "execution claim"
        )
        deadline = row["commitment_deadline_utc"]
        value = cls(
            direction=parse_enum(row["direction"], Direction, "direction"),
            intent_id=_str(row["intent_id"]),
            intent_digest=_str(row["intent_digest"]),
            operation_key=_str(row["operation_key"]),
            challenge_digest=_str(row["challenge_digest"]),
            nonce=_str(row["nonce"]),
            runtime_instance_id=_str(row["runtime_instance_id"]),
            boot_id=_str(row["boot_id"]),
            claim_event_id=_str(row["claim_event_id"]),
            claim_sequence=_int(row["claim_sequence"]),
            claimed_utc=_str(row["claimed_utc"]),
            commitment_deadline_utc=None if deadline is None else _str(deadline),
            approval_channel=parse_enum(row["approval_channel"], ApprovalChannel, "approval_channel"),
        )
        if value.canonical_bytes() != payload:
            raise ValueError("claim bytes are not canonical")
        return value


# ------------------------------------------------------------------ receipt


class ExecutorResult(str, Enum):
    """What the executor reports about its one CAS attempt."""

    APPLIED = "APPLIED"
    # The startup sweep fenced the operation key first: not executed.
    FENCED = "FENCED"
    # Expected revision or runtime_instance_id did not match: not executed
    # (#380 D2 addendum 2; a definite result, not IN_DOUBT).
    CAS_REJECTED = "CAS_REJECTED"
    # IP-18a (HP-7 (a)): the Commitment deadline passed before the target
    # transaction could commit: not executed, nothing written. Receipt only.
    DEADLINE_ELAPSED = "DEADLINE_ELAPSED"


_RECEIPT_BODY: Final = (
    "claim_digest",
    "claim_id",
    "direction",
    "executor_result",
    "operation_key",
    "reported_payload_digest",
    "reported_revision",
    "runtime_instance_id",
    "schema_version",
)


@dataclass(frozen=True, slots=True)
class ExecutionReceiptV1:
    """The executor's own report after a Claim (#368 ``effect_receipts``).

    It is an observation of the executor, never proof of an effect: the
    verifier reads the target itself and never uses this as its basis (#380
    §2-5). A missing receipt after a Claim is IN_DOUBT, not a failure.
    """

    claim_id: str
    claim_digest: str
    direction: Direction
    operation_key: str
    runtime_instance_id: str
    executor_result: ExecutorResult
    reported_revision: int | None
    reported_payload_digest: str | None

    def __post_init__(self) -> None:
        _prefixed(self.claim_id, CLAIM_ID_PREFIX, "claim_id")
        digest(self.claim_digest, "claim_digest")
        if type(self.direction) is not Direction:
            raise TypeError("direction must be a Direction")
        digest(self.operation_key, "operation_key")
        canonical_uuid(self.runtime_instance_id, "runtime_instance_id")
        if type(self.executor_result) is not ExecutorResult:
            raise TypeError("executor_result must be an ExecutorResult")
        applied = self.executor_result is ExecutorResult.APPLIED
        if applied != (self.reported_revision is not None):
            raise ValueError("a reported revision exists exactly for APPLIED")
        if self.reported_revision is not None:
            _revision(self.reported_revision, "reported_revision", minimum=1)
        forward_applied = applied and self.direction is Direction.FORWARD
        if forward_applied != (self.reported_payload_digest is not None):
            raise ValueError("a reported payload digest exists exactly for an applied forward")
        if self.reported_payload_digest is not None:
            digest(self.reported_payload_digest, "reported_payload_digest")

    @classmethod
    def for_claim(
        cls,
        claim: ExecutionClaimV1,
        executor_result: ExecutorResult,
        *,
        reported_revision: int | None,
        reported_payload_digest: str | None,
    ) -> Self:
        if type(claim) is not ExecutionClaimV1:
            raise TypeError("claim must be an ExecutionClaimV1")
        return cls(
            claim.claim_id,
            claim.claim_digest,
            claim.direction,
            claim.operation_key,
            claim.runtime_instance_id,
            executor_result,
            reported_revision,
            reported_payload_digest,
        )

    def canonical_value(self) -> dict[str, object]:
        return {
            "claim_digest": self.claim_digest,
            "claim_id": self.claim_id,
            "direction": self.direction.value,
            "executor_result": self.executor_result.value,
            "operation_key": self.operation_key,
            "reported_payload_digest": self.reported_payload_digest,
            "reported_revision": self.reported_revision,
            "runtime_instance_id": self.runtime_instance_id,
            "schema_version": SCHEMA_VERSION,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.canonical_value())

    @property
    def receipt_digest(self) -> str:
        return checksum(_RECEIPT_DOMAIN, self.canonical_value())

    @classmethod
    def from_json(cls, payload: bytes) -> Self:
        row = _parse(payload, MAX_RECORD_BYTES, _RECEIPT_BODY, "execution receipt")
        revision = row["reported_revision"]
        reported = row["reported_payload_digest"]
        value = cls(
            claim_id=_str(row["claim_id"]),
            claim_digest=_str(row["claim_digest"]),
            direction=parse_enum(row["direction"], Direction, "direction"),
            operation_key=_str(row["operation_key"]),
            runtime_instance_id=_str(row["runtime_instance_id"]),
            executor_result=parse_enum(row["executor_result"], ExecutorResult, "executor_result"),
            reported_revision=None if revision is None else _int(revision),
            reported_payload_digest=None if reported is None else _str(reported),
        )
        if value.canonical_bytes() != payload:
            raise ValueError("receipt bytes are not canonical")
        return value


# ------------------------------------------------------------------ outcome

_OUTCOME_BODY: Final = (
    "authorization",
    "direction",
    "effect_operation_key",
    "foreign_change",
    "observed_payload_digest",
    "observed_revision",
    "observed_state",
    "operation_key",
    "outcome_revision",
    "outcome_state",
    "schema_version",
    "target_instance_id",
    "verifier_event_id",
    "verifier_sequence",
)


@dataclass(frozen=True, slots=True)
class VerifiedOutcomeV1:
    """One terminal result of the independent read-only verifier (#380 D2 P2-2).

    The verifier recomputes the payload digest from the stored effect bytes
    itself; the executor's ``payload_digest`` column and receipt are never the
    basis. ``authorization`` is the verifier's comparison with the nonce
    consumption record: forward effects are AUTHORIZED or UNPROVEN
    (UNAUTHORIZED_EFFECT), and REVERTED carries AUTHORIZED / UNPROVEN as the
    policy's ``inverse_authorization``.
    """

    operation_key: str
    direction: Direction
    outcome_revision: int
    outcome_state: VerifiedState
    authorization: InverseAuthorization | None
    target_instance_id: str
    observed_state: ObservedState
    observed_payload_digest: str | None
    observed_revision: int
    effect_operation_key: str | None
    foreign_change: bool
    verifier_event_id: str
    verifier_sequence: int

    def __post_init__(self) -> None:
        digest(self.operation_key, "operation_key")
        if type(self.direction) is not Direction:
            raise TypeError("direction must be a Direction")
        _revision(self.outcome_revision, "outcome_revision", minimum=1)
        if type(self.outcome_state) is not VerifiedState:
            raise TypeError("outcome_state must be a VerifiedState")
        forward = self.direction is Direction.FORWARD
        if forward != (self.outcome_state in _FORWARD_STATES):
            raise ValueError("outcome state does not belong to the operation direction")
        if self.authorization is not None and type(self.authorization) is not InverseAuthorization:
            raise TypeError("authorization must be an InverseAuthorization")
        canonical_uuid(self.target_instance_id, "target_instance_id")
        if type(self.observed_state) is not ObservedState:
            raise TypeError("observed_state must be an ObservedState")
        if (self.observed_payload_digest is None) != (
            self.observed_state is not ObservedState.ACTIVE
        ):
            raise ValueError("an observed payload digest exists only for ACTIVE")
        if self.observed_payload_digest is not None:
            digest(self.observed_payload_digest, "observed_payload_digest")
        _revision(self.observed_revision, "observed_revision")
        if self.effect_operation_key is not None:
            digest(self.effect_operation_key, "effect_operation_key")
        if type(self.foreign_change) is not bool:
            raise TypeError("foreign_change must be a bool")
        _event(self.verifier_event_id, self.verifier_sequence, "verifier")
        expected_authorization = {
            VerifiedState.ACTIVE_VERIFIED_AUTHORIZED: InverseAuthorization.AUTHORIZED,
            VerifiedState.UNAUTHORIZED_EFFECT: InverseAuthorization.UNPROVEN,
        }.get(self.outcome_state)
        if self.outcome_state is VerifiedState.REVERTED:
            if self.authorization is None:
                raise ValueError("REVERTED carries its inverse authorization")
        elif self.authorization is not expected_authorization:
            raise ValueError("authorization does not match the outcome state")
        if self.outcome_state in _INVERTIBLE and (
            self.observed_state is not ObservedState.ACTIVE
            or self.effect_operation_key != self.operation_key
        ):
            raise ValueError("an observed effect is this operation's ACTIVE effect row")
        if (
            self.outcome_state in (VerifiedState.VERIFIED_ABSENT, VerifiedState.REVERTED)
            and self.observed_state is not ObservedState.ABSENT
        ):
            raise ValueError("VERIFIED_ABSENT and REVERTED observe ABSENT")
        if self.outcome_state is VerifiedState.INVERSE_VERIFIED_ABSENT and (
            self.observed_state is not ObservedState.ACTIVE
            or self.effect_operation_key is None
            or self.effect_operation_key == self.operation_key
        ):
            # The inverse did not apply, so the forward effect row remains.
            raise ValueError("INVERSE_VERIFIED_ABSENT observes the forward effect ACTIVE")
        if self.observed_state is ObservedState.UNKNOWN:
            # An unreadable target stays IN_DOUBT; it is never a terminal Outcome.
            raise ValueError("a terminal verified Outcome never observes UNKNOWN")

    def _body(self) -> dict[str, object]:
        return {
            "authorization": None if self.authorization is None else self.authorization.value,
            "direction": self.direction.value,
            "effect_operation_key": self.effect_operation_key,
            "foreign_change": self.foreign_change,
            "observed_payload_digest": self.observed_payload_digest,
            "observed_revision": self.observed_revision,
            "observed_state": self.observed_state.value,
            "operation_key": self.operation_key,
            "outcome_revision": self.outcome_revision,
            "outcome_state": self.outcome_state.value,
            "schema_version": SCHEMA_VERSION,
            "target_instance_id": self.target_instance_id,
            "verifier_event_id": self.verifier_event_id,
            "verifier_sequence": self.verifier_sequence,
        }

    @property
    def verified_outcome_id(self) -> str:
        """Stable across revisions of the same operation's Outcome."""
        return OUTCOME_ID_PREFIX + checksum(
            _OUTCOME_ID_DOMAIN,
            {"direction": self.direction.value, "operation_key": self.operation_key},
        )

    @property
    def outcome_digest(self) -> str:
        return checksum(_OUTCOME_DOMAIN, self._body())

    def to_r17_view(self) -> VerifiedOutcomeView:
        """The witness fields R17 reads (D-R17 §3); forward Outcomes only."""
        if self.direction is not Direction.FORWARD:
            raise ValueError("R17 assesses forward Outcomes only")
        return VerifiedOutcomeView(
            verified_outcome_id=self.verified_outcome_id,
            outcome_revision=self.outcome_revision,
            outcome_digest=self.outcome_digest,
            outcome_state=OutcomeState(self.outcome_state.value),
            target_instance_id=self.target_instance_id,
            observed_state=self.observed_state,
            observed_payload_digest=self.observed_payload_digest,
            observed_revision=self.observed_revision,
            effect_operation_key=self.effect_operation_key,
            foreign_change=self.foreign_change,
        )

    def canonical_value(self) -> dict[str, object]:
        return {
            **self._body(),
            "outcome_digest": self.outcome_digest,
            "verified_outcome_id": self.verified_outcome_id,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.canonical_value())

    @classmethod
    def from_json(cls, payload: bytes) -> Self:
        row = _parse(
            payload,
            MAX_RECORD_BYTES,
            (*_OUTCOME_BODY, "outcome_digest", "verified_outcome_id"),
            "verified outcome",
        )
        authorization = row["authorization"]
        observed_payload = row["observed_payload_digest"]
        effect_key = row["effect_operation_key"]
        foreign = row["foreign_change"]
        if type(foreign) is not bool:
            raise TypeError("foreign_change must be a bool")
        value = cls(
            operation_key=_str(row["operation_key"]),
            direction=parse_enum(row["direction"], Direction, "direction"),
            outcome_revision=_int(row["outcome_revision"]),
            outcome_state=parse_enum(row["outcome_state"], VerifiedState, "outcome_state"),
            authorization=None
            if authorization is None
            else parse_enum(authorization, InverseAuthorization, "authorization"),
            target_instance_id=_str(row["target_instance_id"]),
            observed_state=parse_enum(row["observed_state"], ObservedState, "observed_state"),
            observed_payload_digest=None if observed_payload is None else _str(observed_payload),
            observed_revision=_int(row["observed_revision"]),
            effect_operation_key=None if effect_key is None else _str(effect_key),
            foreign_change=foreign,
            verifier_event_id=_str(row["verifier_event_id"]),
            verifier_sequence=_int(row["verifier_sequence"]),
        )
        if value.canonical_bytes() != payload:
            raise ValueError("outcome bytes are not canonical")
        return value


# ---------------------------------------------------------------- lifecycle

# #368 D1 forward lifecycle and #380 D1 P1-5 / D2 P2-3 inverse lifecycle.
# The code keeps the existing seven inverse LedgerStates: #380's
# INVERSE_NOT_CLAIMED(reason) is INVERSE_POLICY_DENIED (POLICY_DENIED) or
# INVERSE_APPROVAL_EXPIRED_UNCLAIMED (APPROVAL_EXPIRED). CANCELLED after a
# claim does not exist. CORRUPT is reachable from every non-terminal and
# terminal state when the target ledger is lost or crossed (#368 D1).
_FORWARD_VERIFIED: Final = frozenset(
    {
        LedgerState.ACTIVE_VERIFIED_AUTHORIZED,
        LedgerState.VERIFIED_ABSENT,
        LedgerState.UNAUTHORIZED_EFFECT,
    }
)
_INVERSE_VERIFIED: Final = frozenset(
    {LedgerState.REVERTED, LedgerState.INVERSE_VERIFIED_ABSENT}
)
# The startup sweep finds a target effect with no source-side Claim (#380
# D1 P1-2) when the restored source state is unclaimed: NONE, an issued
# intent, or one that was denied or expired before any Claim.
_PRIOR: Final = frozenset({LedgerState.PRIOR_INCARNATION_EFFECT_PRESENT})
_TRANSITIONS: Final[dict[LedgerState, frozenset[LedgerState]]] = {
    LedgerState.NONE: frozenset({LedgerState.INTENT_ISSUED}) | _PRIOR,
    LedgerState.INTENT_ISSUED: frozenset(
        {
            LedgerState.POLICY_DENIED,
            LedgerState.APPROVAL_EXPIRED_UNCLAIMED,
            LedgerState.CLAIMED,
        }
    )
    | _PRIOR,
    LedgerState.CLAIMED: frozenset({LedgerState.IN_DOUBT} | _FORWARD_VERIFIED),
    LedgerState.IN_DOUBT: _FORWARD_VERIFIED,
    LedgerState.ACTIVE_VERIFIED_AUTHORIZED: frozenset({LedgerState.INVERSE_INTENT_ISSUED}),
    LedgerState.UNAUTHORIZED_EFFECT: frozenset({LedgerState.INVERSE_INTENT_ISSUED}),
    LedgerState.INVERSE_INTENT_ISSUED: frozenset(
        {
            LedgerState.INVERSE_POLICY_DENIED,
            LedgerState.INVERSE_APPROVAL_EXPIRED_UNCLAIMED,
            LedgerState.INVERSE_CLAIMED,
        }
    ),
    LedgerState.INVERSE_CLAIMED: frozenset({LedgerState.INVERSE_IN_DOUBT} | _INVERSE_VERIFIED),
    LedgerState.INVERSE_IN_DOUBT: _INVERSE_VERIFIED,
    # Unclaimed terminals: never reissued, but a prior-incarnation effect
    # of the same group can still be found by the sweep.
    LedgerState.POLICY_DENIED: _PRIOR,
    LedgerState.APPROVAL_EXPIRED_UNCLAIMED: _PRIOR,
    # Terminal: no re-write, no automatic reissue of a FENCED key (#380 D2 P1-A).
    LedgerState.VERIFIED_ABSENT: frozenset(),
    LedgerState.INVERSE_POLICY_DENIED: frozenset(),
    LedgerState.INVERSE_APPROVAL_EXPIRED_UNCLAIMED: frozenset(),
    LedgerState.INVERSE_VERIFIED_ABSENT: frozenset(),
    LedgerState.REVERTED: frozenset(),
    LedgerState.PRIOR_INCARNATION_EFFECT_PRESENT: frozenset(),
    LedgerState.CORRUPT: frozenset(),
}


def allowed_transitions(current: LedgerState) -> frozenset[LedgerState]:
    if type(current) is not LedgerState:
        raise TypeError("current must be a LedgerState")
    extra = frozenset() if current is LedgerState.CORRUPT else frozenset({LedgerState.CORRUPT})
    return _TRANSITIONS[current] | extra


def transition(current: LedgerState, new: LedgerState) -> LedgerState:
    """One ledger step; anything outside the closed table is refused."""
    if type(new) is not LedgerState:
        raise TypeError("new must be a LedgerState")
    if new not in allowed_transitions(current):
        raise ValueError(f"{current.value} cannot move to {new.value}")
    return new


def ledger_state_of(outcome: VerifiedOutcomeV1) -> LedgerState:
    """The ledger state a terminal verified Outcome puts the operation in."""
    if type(outcome) is not VerifiedOutcomeV1:
        raise TypeError("outcome must be a VerifiedOutcomeV1")
    return LedgerState(outcome.outcome_state.value)
