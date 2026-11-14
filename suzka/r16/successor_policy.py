"""R16 IP-8b: SuccessorPolicyV2 and its sealed input (#374 F1 D3 §3, #371 D5/D6).

Pure, total and deterministic.  ``evaluate`` reads only one exact
``SuccessorPolicyInputV2`` and returns the first matching row of the ordered
table (D3 §3-2):

====  ===============================================  ==========================
Row   Condition                                        Output
====  ===============================================  ==========================
0     contradiction (states and D6-2 cross-refs)       HOLD(CONFLICTING_EVIDENCE)
1     a required source/lineage/commitment or the      HOLD(INSUFFICIENT_EVIDENCE)
      R15 interpretation is unavailable
2     the operation ledger is pending or unsafe        HOLD(INSUFFICIENT_EVIDENCE)
2b    interpretation = CONTRADICTORY                   HOLD(CONFLICTING_EVIDENCE)
3     initial, no Commitment                           Plan A
4     initial, ACTIVE Commitment, SETTLED/NOT_REVIEWED Plan B
4b    row 4 with interpretation = UNCERTAIN            Plan A (no WRITE)
5     verified effect + MATCH_SELF, SETTLED/NOT_REV.   NO_OP
5b    row 5 with interpretation = UNCERTAIN            Plan A
6     verified effect + any other Feedback             Plan A
7     every remaining tuple                            HOLD(INSUFFICIENT_EVIDENCE)
====  ===============================================  ==========================

No row grants WRITE: the R18 ActionIntent issuer separately admits only row
4 Plan B under its own Policy (#371 D6-4).  The input carries only IDs,
revisions, digests and event links, never raw text.  The state enums must
come from their Domain Owner adapters; this module cannot tell (D6-1).

Commitment states other than ABSENT/CONFLICT require the Commitment ref and
deadline (a self-origin Goal's INELIGIBLE, #379 D2, is about an existing
Commitment).  ``inverse_authorization`` is required exactly for REVERTED.

Known limit: R17/R18 do not exist yet, so ledger/Feedback/Outcome references
are generic digest-bound refs and D6-2's mandatory-reference rules are
checked by presence only.  Kind-exact predicates belong to R17/R18.  In
particular an inverse started after UNAUTHORIZED_EFFECT (#371 A2-1) is not
distinguishable from one after an authorized forward effect here, so a
MATCH_SELF under INVERSE_*/REVERTED is row 2 (HOLD) rather than row 0; R18
must check that the original forward effect was authorized.

R16 must not import ``suzka.r15``: the R15 read arrives as the closed strings
of ``suzka.runtime.r15_interpretation_reader`` and is mapped here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Final, Self

from suzka.limits import MAX_PERSISTED_EVENT_SEQUENCE, MAX_PERSISTED_REVISION
from suzka.r16.contracts import (
    SCHEMA_VERSION,
    UNCERTAINTY_SCALE,
    HoldAssessment,
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
from suzka.r16.history_codec import MAX_POLICY_INPUT_BYTES, PolicyKind, PolicyRow

INPUT_SCHEMA: Final = "SUCCESSOR_POLICY_INPUT_V2"
POLICY_KIND: Final = PolicyKind.SUCCESSOR_POLICY_V2
MAX_LEDGER_REFS: Final = 4
_FINGERPRINT_DOMAIN: Final = b"PROJECT-SUZKA:R16:SUCCESSOR-EVIDENCE-FINGERPRINT:V2\0"
_UTC_FORMAT: Final = "%Y-%m-%dT%H:%M:%SZ"
_REQUIRED_KINDS: Final = ("context", "experience", "goal", "value")


# --------------------------------------------------------------------- enums


class RequiredSourceState(str, Enum):
    QUALIFIED = "QUALIFIED"
    UNAVAILABLE = "UNAVAILABLE"
    CONFLICT = "CONFLICT"


class PredecessorState(str, Enum):
    ABSENT_INITIAL = "ABSENT_INITIAL"
    QUALIFIED = "QUALIFIED"
    UNAVAILABLE = "UNAVAILABLE"
    CONFLICT = "CONFLICT"


class LineageState(str, Enum):
    CONSISTENT = "CONSISTENT"
    MISMATCH = "MISMATCH"
    UNAVAILABLE = "UNAVAILABLE"


class CommitmentState(str, Enum):
    ABSENT = "ABSENT"
    ACTIVE = "ACTIVE"
    EXPIRED = "EXPIRED"
    INELIGIBLE = "INELIGIBLE"
    CONFLICT = "CONFLICT"


class LedgerState(str, Enum):
    NONE = "NONE"
    INTENT_ISSUED = "INTENT_ISSUED"
    POLICY_DENIED = "POLICY_DENIED"
    APPROVAL_EXPIRED_UNCLAIMED = "APPROVAL_EXPIRED_UNCLAIMED"
    CLAIMED = "CLAIMED"
    IN_DOUBT = "IN_DOUBT"
    VERIFIED_ABSENT = "VERIFIED_ABSENT"
    ACTIVE_VERIFIED_AUTHORIZED = "ACTIVE_VERIFIED_AUTHORIZED"
    UNAUTHORIZED_EFFECT = "UNAUTHORIZED_EFFECT"
    INVERSE_INTENT_ISSUED = "INVERSE_INTENT_ISSUED"
    INVERSE_CLAIMED = "INVERSE_CLAIMED"
    INVERSE_IN_DOUBT = "INVERSE_IN_DOUBT"
    # #371 D6-A2 A2-2: terminal inverse refusal / absence.
    INVERSE_POLICY_DENIED = "INVERSE_POLICY_DENIED"
    INVERSE_APPROVAL_EXPIRED_UNCLAIMED = "INVERSE_APPROVAL_EXPIRED_UNCLAIMED"
    INVERSE_VERIFIED_ABSENT = "INVERSE_VERIFIED_ABSENT"
    REVERTED = "REVERTED"
    CORRUPT = "CORRUPT"


class InverseAuthorization(str, Enum):
    """A2-2: separate immutable authorization of a REVERTED inverse."""

    AUTHORIZED = "AUTHORIZED"
    UNPROVEN = "UNPROVEN"


class FeedbackState(str, Enum):
    ABSENT = "ABSENT"
    MATCH_SELF = "MATCH_SELF"
    MATCH_OTHER_OR_UNKNOWN = "MATCH_OTHER_OR_UNKNOWN"
    MISMATCH = "MISMATCH"
    UNKNOWN = "UNKNOWN"
    INVALID = "INVALID"


class Interpretation(str, Enum):
    """#374 F1 D3 §3-1: the R16-defined view of one R15 interpretation."""

    SETTLED = "SETTLED"
    UNCERTAIN = "UNCERTAIN"
    CONTRADICTORY = "CONTRADICTORY"
    NOT_REVIEWED = "NOT_REVIEWED"
    UNAVAILABLE = "UNAVAILABLE"


class InterpretationStage(str, Enum):
    NONE = "NONE"
    S1 = "S1"
    S2 = "S2"


class SuccessorRow(str, Enum):
    """Every row ID, HOLD rows included (sealed into a HoldReceipt, #354)."""

    ROW_0 = "0"
    ROW_1 = "1"
    ROW_2 = "2"
    ROW_2B = "2b"
    ROW_3 = "3"
    ROW_4 = "4"
    ROW_4B = "4b"
    ROW_5 = "5"
    ROW_5B = "5b"
    ROW_6 = "6"
    ROW_7 = "7"


class PolicyOutputKind(str, Enum):
    HOLD = "HOLD"
    PLAN_A = "PLAN_A"
    PLAN_B = "PLAN_B"
    NO_OP = "NO_OP"


_PENDING_LEDGER: Final = frozenset(
    {
        LedgerState.INTENT_ISSUED,
        LedgerState.POLICY_DENIED,
        LedgerState.APPROVAL_EXPIRED_UNCLAIMED,
        LedgerState.CLAIMED,
        LedgerState.IN_DOUBT,
        LedgerState.VERIFIED_ABSENT,
        LedgerState.UNAUTHORIZED_EFFECT,
        LedgerState.INVERSE_INTENT_ISSUED,
        LedgerState.INVERSE_CLAIMED,
        LedgerState.INVERSE_IN_DOUBT,
        LedgerState.INVERSE_POLICY_DENIED,
        LedgerState.INVERSE_APPROVAL_EXPIRED_UNCLAIMED,
        LedgerState.INVERSE_VERIFIED_ABSENT,
        LedgerState.REVERTED,
    }
)
# D6-2: no verified Outcome may accompany a ledger with no claimed effect.
_OUTCOME_FORBIDDEN: Final = frozenset(
    {
        LedgerState.NONE,
        LedgerState.INTENT_ISSUED,
        LedgerState.POLICY_DENIED,
        LedgerState.APPROVAL_EXPIRED_UNCLAIMED,
    }
)
# Forward effect proven, then an inverse attempted or applied (A2-2/A2-3).
_INVERSE_LEDGER: Final = frozenset(
    {
        LedgerState.INVERSE_INTENT_ISSUED,
        LedgerState.INVERSE_CLAIMED,
        LedgerState.INVERSE_IN_DOUBT,
        LedgerState.INVERSE_POLICY_DENIED,
        LedgerState.INVERSE_APPROVAL_EXPIRED_UNCLAIMED,
        LedgerState.INVERSE_VERIFIED_ABSENT,
        LedgerState.REVERTED,
    }
)
# D6-2: Feedback needs a verified (or proven, then inverted) forward effect.
_FEEDBACK_ALLOWED_LEDGER: Final = frozenset(
    {LedgerState.ACTIVE_VERIFIED_AUTHORIZED, LedgerState.UNAUTHORIZED_EFFECT}
    | _INVERSE_LEDGER
)
# A2-3: a MATCH_SELF admitted at T survives a later inverse/REVERTED (it is
# then held by row 2); it can never accompany an unauthorized effect.
_MATCH_SELF_LEDGER: Final = frozenset(
    {LedgerState.ACTIVE_VERIFIED_AUTHORIZED} | _INVERSE_LEDGER
)
PREDECESSOR_DECISION_KIND: Final = "decision"
# D6-1: ABSENT_INITIAL stores the verified no-predecessor query horizon.
NO_PREDECESSOR_HORIZON_KIND: Final = "history-horizon"
_SETTLED_OR_NOT_REVIEWED: Final = frozenset(
    {Interpretation.SETTLED, Interpretation.NOT_REVIEWED}
)
_STAGE_INTERPRETATIONS: Final[Mapping[InterpretationStage, frozenset[Interpretation]]] = {
    InterpretationStage.NONE: frozenset(
        {Interpretation.NOT_REVIEWED, Interpretation.UNAVAILABLE}
    ),
    # S1 only ever yields PRESENT/UNKNOWN (IP-8a R15InterpretationRead).
    InterpretationStage.S1: frozenset({Interpretation.SETTLED, Interpretation.UNCERTAIN}),
    InterpretationStage.S2: frozenset(
        {Interpretation.SETTLED, Interpretation.UNCERTAIN, Interpretation.CONTRADICTORY}
    ),
}


# ----------------------------------------------------------------- values


def _utc(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a UTC timestamp string")
    try:
        parsed = datetime.strptime(value, _UTC_FORMAT)
    except ValueError as error:
        raise ValueError(f"{name} must be YYYY-MM-DDTHH:MM:SSZ") from error
    if parsed.strftime(_UTC_FORMAT) != value:
        raise ValueError(f"{name} is not canonical")
    return value


def utc_timestamp(moment: datetime) -> str:
    """The canonical sealed form of one trusted clock read (D5-1)."""

    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("the policy clock must be timezone-aware")
    return moment.astimezone(timezone.utc).strftime(_UTC_FORMAT)


@dataclass(frozen=True, slots=True)
class PolicyRef:
    """One digest-bound evidence reference; never a fetched source."""

    kind: str
    ref_id: str
    revision: int
    digest: str

    def __post_init__(self) -> None:
        identifier(self.kind, "policy ref kind")
        identifier(self.ref_id, "policy ref id")
        integer(self.revision, "policy ref revision", maximum=MAX_PERSISTED_REVISION)
        digest(self.digest, "policy ref digest")

    def canonical_value(self) -> dict[str, object]:
        return {
            "digest": self.digest,
            "kind": self.kind,
            "ref_id": self.ref_id,
            "revision": self.revision,
        }

    @classmethod
    def from_value(cls, value: object) -> Self:
        row = closed(value, ("digest", "kind", "ref_id", "revision"), "policy ref")
        return cls(
            _str(row["kind"]), _str(row["ref_id"]), _int(row["revision"]), _str(row["digest"])
        )

    @property
    def sort_key(self) -> tuple[str, str]:
        return (self.kind, self.ref_id)


@dataclass(frozen=True, slots=True)
class LineageRef:
    """R07 lineage, adopted baseline, latest qualified CLEARED, incarnation."""

    journal_lineage_id: str
    baseline_id: str
    baseline_hash: str
    cleared_record_hash: str | None
    incarnation_id: str

    def __post_init__(self) -> None:
        identifier(self.journal_lineage_id, "journal lineage id")
        identifier(self.baseline_id, "baseline id")
        digest(self.baseline_hash, "baseline hash")
        if self.cleared_record_hash is not None:
            digest(self.cleared_record_hash, "cleared record hash")
        digest(self.incarnation_id, "incarnation id")

    def canonical_value(self) -> dict[str, object]:
        return {
            "baseline_hash": self.baseline_hash,
            "baseline_id": self.baseline_id,
            "cleared_record_hash": self.cleared_record_hash,
            "incarnation_id": self.incarnation_id,
            "journal_lineage_id": self.journal_lineage_id,
        }

    @classmethod
    def from_value(cls, value: object) -> Self:
        row = closed(
            value,
            (
                "baseline_hash",
                "baseline_id",
                "cleared_record_hash",
                "incarnation_id",
                "journal_lineage_id",
            ),
            "lineage ref",
        )
        cleared = row["cleared_record_hash"]
        return cls(
            _str(row["journal_lineage_id"]),
            _str(row["baseline_id"]),
            _str(row["baseline_hash"]),
            None if cleared is None else _str(cleared),
            _str(row["incarnation_id"]),
        )


def _str(value: object) -> str:
    if type(value) is not str:
        raise TypeError("expected a JSON string")
    return value


def _int(value: object) -> int:
    if type(value) is not int:
        raise TypeError("expected a JSON integer")
    return value


def _optional_ref(value: object) -> PolicyRef | None:
    return None if value is None else PolicyRef.from_value(value)


@dataclass(frozen=True, slots=True)
class SuccessorPolicyInputV2:
    """The closed sealed input of SuccessorPolicyV2 (D6-1 + D3 §3-1).

    Construction checks types, bounds, the (stage, interpretation) pairs and
    the byte bound.  Cross-reference consistency is *not* a construction
    error: it is evaluated as row 0 (D6-2).
    """

    required_source: RequiredSourceState
    source_refs: tuple[PolicyRef, ...]
    predecessor: PredecessorState
    predecessor_ref: PolicyRef | None
    lineage: LineageState
    lineage_ref: LineageRef | None
    commitment: CommitmentState
    commitment_ref: PolicyRef | None
    commitment_deadline_utc: str | None
    ledger: LedgerState
    ledger_refs: tuple[PolicyRef, ...]
    inverse_authorization: InverseAuthorization | None
    feedback: FeedbackState
    feedback_ref: PolicyRef | None
    verified_outcome_ref: PolicyRef | None
    interpretation: Interpretation
    interpretation_stage: InterpretationStage
    as_of_r07_sequence: int
    # IP-10 D2 P2-1 (D6-A1): hash of the committed R06 snapshot_before.
    as_of_snapshot_hash: str
    event_id: str
    event_sequence: int
    now_utc: str

    def __post_init__(self) -> None:
        for name, kind in (
            ("required_source", RequiredSourceState),
            ("predecessor", PredecessorState),
            ("lineage", LineageState),
            ("commitment", CommitmentState),
            ("ledger", LedgerState),
            ("feedback", FeedbackState),
            ("interpretation", Interpretation),
            ("interpretation_stage", InterpretationStage),
        ):
            if type(getattr(self, name)) is not kind:
                raise TypeError(f"{name} must be an exact {kind.__name__}")
        _refs(self.source_refs, "source_refs", len(_REQUIRED_KINDS))
        _refs(self.ledger_refs, "ledger_refs", MAX_LEDGER_REFS)
        for name in ("predecessor_ref", "commitment_ref", "feedback_ref", "verified_outcome_ref"):
            value = getattr(self, name)
            if value is not None and type(value) is not PolicyRef:
                raise TypeError(f"{name} must be an exact PolicyRef")
        if (
            self.inverse_authorization is not None
            and type(self.inverse_authorization) is not InverseAuthorization
        ):
            raise TypeError("inverse_authorization must be an exact InverseAuthorization")
        if self.lineage_ref is not None and type(self.lineage_ref) is not LineageRef:
            raise TypeError("lineage_ref must be an exact LineageRef")
        if self.commitment_deadline_utc is not None:
            _utc(self.commitment_deadline_utc, "commitment deadline")
        if self.interpretation not in _STAGE_INTERPRETATIONS[self.interpretation_stage]:
            raise ValueError("interpretation and stage are not a closed combination")
        integer(
            self.as_of_r07_sequence,
            "as_of_r07_sequence",
            maximum=MAX_PERSISTED_EVENT_SEQUENCE,
        )
        digest(self.as_of_snapshot_hash, "as_of snapshot hash")
        identifier(self.event_id, "event id")
        integer(
            self.event_sequence,
            "event sequence",
            minimum=1,
            maximum=MAX_PERSISTED_EVENT_SEQUENCE,
        )
        _utc(self.now_utc, "now_utc")
        if len(self.canonical_bytes()) > MAX_POLICY_INPUT_BYTES:
            raise ValueError("policy input exceeds its byte bound")

    def canonical_value(self) -> dict[str, object]:
        return {
            "as_of": {
                "event_id": self.event_id,
                "event_sequence": self.event_sequence,
                "now_utc": self.now_utc,
                "r07_sequence": self.as_of_r07_sequence,
                "snapshot_hash": self.as_of_snapshot_hash,
            },
            "commitment": {
                "deadline_utc": self.commitment_deadline_utc,
                "ref": _ref_value(self.commitment_ref),
                "state": self.commitment.value,
            },
            "feedback": {
                "ref": _ref_value(self.feedback_ref),
                "state": self.feedback.value,
            },
            "interpretation": {
                "stage": self.interpretation_stage.value,
                "state": self.interpretation.value,
            },
            "ledger": {
                "inverse_authorization": None
                if self.inverse_authorization is None
                else self.inverse_authorization.value,
                "refs": [item.canonical_value() for item in self.ledger_refs],
                "state": self.ledger.value,
            },
            "lineage": {
                "ref": None
                if self.lineage_ref is None
                else self.lineage_ref.canonical_value(),
                "state": self.lineage.value,
            },
            "predecessor": {
                "ref": _ref_value(self.predecessor_ref),
                "state": self.predecessor.value,
            },
            "required_source": {
                "refs": [item.canonical_value() for item in self.source_refs],
                "state": self.required_source.value,
            },
            "schema": INPUT_SCHEMA,
            "verified_outcome": _ref_value(self.verified_outcome_ref),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.canonical_value())

    @classmethod
    def from_bytes(cls, payload: bytes) -> Self:
        """Strict inverse of ``canonical_bytes`` (sealed replay, D6-3).

        Every failure is a ``ValueError`` so callers fail closed on one type.
        """

        try:
            return cls._from_bytes(payload)
        except ValueError:
            raise
        except (TypeError, UnicodeDecodeError, RecursionError) as error:
            raise ValueError("policy input bytes are invalid") from error

    @classmethod
    def _from_bytes(cls, payload: bytes) -> Self:
        row = closed(
            parse_json(payload, MAX_POLICY_INPUT_BYTES),
            (
                "as_of",
                "commitment",
                "feedback",
                "interpretation",
                "ledger",
                "lineage",
                "predecessor",
                "required_source",
                "schema",
                "verified_outcome",
            ),
            "policy input",
        )
        if row["schema"] != INPUT_SCHEMA:
            raise ValueError("policy input schema is not V2")
        as_of = closed(
            row["as_of"],
            ("event_id", "event_sequence", "now_utc", "r07_sequence", "snapshot_hash"),
            "as_of",
        )
        commitment = closed(row["commitment"], ("deadline_utc", "ref", "state"), "commitment")
        feedback = closed(row["feedback"], ("ref", "state"), "feedback")
        interpretation = closed(row["interpretation"], ("stage", "state"), "interpretation")
        ledger = closed(row["ledger"], ("inverse_authorization", "refs", "state"), "ledger")
        inverse = ledger["inverse_authorization"]
        lineage = closed(row["lineage"], ("ref", "state"), "lineage")
        predecessor = closed(row["predecessor"], ("ref", "state"), "predecessor")
        required = closed(row["required_source"], ("refs", "state"), "required_source")
        deadline = commitment["deadline_utc"]
        value = cls(
            required_source=parse_enum(required["state"], RequiredSourceState, "required_source"),
            source_refs=tuple(
                PolicyRef.from_value(item)
                for item in sequence(
                    required["refs"], "source_refs", len(_REQUIRED_KINDS), json_input=True
                )
            ),
            predecessor=parse_enum(predecessor["state"], PredecessorState, "predecessor"),
            predecessor_ref=_optional_ref(predecessor["ref"]),
            lineage=parse_enum(lineage["state"], LineageState, "lineage"),
            lineage_ref=None if lineage["ref"] is None else LineageRef.from_value(lineage["ref"]),
            commitment=parse_enum(commitment["state"], CommitmentState, "commitment"),
            commitment_ref=_optional_ref(commitment["ref"]),
            commitment_deadline_utc=None if deadline is None else _str(deadline),
            ledger=parse_enum(ledger["state"], LedgerState, "ledger"),
            ledger_refs=tuple(
                PolicyRef.from_value(item)
                for item in sequence(ledger["refs"], "ledger_refs", MAX_LEDGER_REFS, json_input=True)
            ),
            inverse_authorization=None
            if inverse is None
            else parse_enum(inverse, InverseAuthorization, "inverse_authorization"),
            feedback=parse_enum(feedback["state"], FeedbackState, "feedback"),
            feedback_ref=_optional_ref(feedback["ref"]),
            verified_outcome_ref=_optional_ref(row["verified_outcome"]),
            interpretation=parse_enum(interpretation["state"], Interpretation, "interpretation"),
            interpretation_stage=parse_enum(
                interpretation["stage"], InterpretationStage, "interpretation stage"
            ),
            as_of_r07_sequence=_int(as_of["r07_sequence"]),
            as_of_snapshot_hash=_str(as_of["snapshot_hash"]),
            event_id=_str(as_of["event_id"]),
            event_sequence=_int(as_of["event_sequence"]),
            now_utc=_str(as_of["now_utc"]),
        )
        if value.canonical_bytes() != payload:
            raise ValueError("policy input bytes are not canonical")
        return value

    def evidence_fingerprint(self) -> str:
        """#369 wake fingerprint inputs, with (stage, interpretation) (D3 §3-3).

        Excludes the as-of event and clock: an unchanged source set yields the
        same fingerprint across Wakes.  A commit in R15 is not a Wake cause;
        a changed class is read here at the next Wake only.
        """

        body = self.canonical_value()
        del body["as_of"]
        del body["predecessor"]
        return checksum(_FINGERPRINT_DOMAIN, body)


def _ref_value(value: PolicyRef | None) -> dict[str, object] | None:
    return None if value is None else value.canonical_value()


def _refs(values: object, name: str, maximum: int) -> None:
    items = sequence(values, name, maximum, json_input=False)
    for item in items:
        if type(item) is not PolicyRef:
            raise TypeError(f"{name} must hold exact PolicyRef values")
    keys = [item.sort_key for item in items]  # type: ignore[attr-defined]
    if keys != sorted(set(keys)):
        raise ValueError(f"{name} must be sorted and unique by kind and ID")


# ----------------------------------------------------------------- policy


@dataclass(frozen=True, slots=True)
class SuccessorPolicyResult:
    row: SuccessorRow
    output: PolicyOutputKind
    hold_reason: ReasonCode | None
    # No row grants WRITE (D5-2 last column, D6-4).
    write_eligible: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.row) is not SuccessorRow
            or type(self.output) is not PolicyOutputKind
            or (self.hold_reason is not None and type(self.hold_reason) is not ReasonCode)
            or type(self.write_eligible) is not bool
        ):
            raise TypeError("policy result fields must be exact enum values")
        if (self.output is PolicyOutputKind.HOLD) != (self.hold_reason is not None):
            raise ValueError("exactly a HOLD carries a reason")
        if self.write_eligible:
            raise ValueError("SuccessorPolicyV2 never grants WRITE")
        if _ROW_OUTPUTS[self.row] != (self.output, self.hold_reason):
            raise ValueError("row and output do not match the table")

    @property
    def is_hold(self) -> bool:
        return self.output is PolicyOutputKind.HOLD

    @property
    def history_row(self) -> PolicyRow:
        """The SubjectHistory row for a Decision row; HOLD rows have none."""

        if self.is_hold:
            raise ValueError("a HOLD row issues no committed Decision")
        return PolicyRow(self.row.value)

    def hold_assessment(self) -> HoldAssessment:
        if self.hold_reason is None:
            raise ValueError("only a HOLD row has a HoldAssessment")
        return HoldAssessment(SCHEMA_VERSION, self.hold_reason, UNCERTAINTY_SCALE)


_CONFLICT: Final = (PolicyOutputKind.HOLD, ReasonCode.CONFLICTING_EVIDENCE)
_INSUFFICIENT: Final = (PolicyOutputKind.HOLD, ReasonCode.INSUFFICIENT_EVIDENCE)
_ROW_OUTPUTS: Final[Mapping[SuccessorRow, tuple[PolicyOutputKind, ReasonCode | None]]] = {
    SuccessorRow.ROW_0: _CONFLICT,
    SuccessorRow.ROW_1: _INSUFFICIENT,
    SuccessorRow.ROW_2: _INSUFFICIENT,
    SuccessorRow.ROW_2B: _CONFLICT,
    SuccessorRow.ROW_3: (PolicyOutputKind.PLAN_A, None),
    SuccessorRow.ROW_4: (PolicyOutputKind.PLAN_B, None),
    SuccessorRow.ROW_4B: (PolicyOutputKind.PLAN_A, None),
    SuccessorRow.ROW_5: (PolicyOutputKind.NO_OP, None),
    SuccessorRow.ROW_5B: (PolicyOutputKind.PLAN_A, None),
    SuccessorRow.ROW_6: (PolicyOutputKind.PLAN_A, None),
    SuccessorRow.ROW_7: _INSUFFICIENT,
}


def _result(row: SuccessorRow) -> SuccessorPolicyResult:
    output, reason = _ROW_OUTPUTS[row]
    return SuccessorPolicyResult(row, output, reason)


def cross_reference_conflicts(value: SuccessorPolicyInputV2) -> tuple[str, ...]:
    """D6-2 closed predicate list; any hit is row 0 (CONFLICTING_EVIDENCE)."""

    hits: list[str] = []
    kinds = tuple(item.kind for item in value.source_refs)
    if value.required_source is RequiredSourceState.QUALIFIED and kinds != _REQUIRED_KINDS:
        hits.append("required_source_refs")
    predecessor_kind = None if value.predecessor_ref is None else value.predecessor_ref.kind
    if value.predecessor is PredecessorState.QUALIFIED:
        if predecessor_kind != PREDECESSOR_DECISION_KIND:
            hits.append("predecessor_ref")
    elif value.predecessor is PredecessorState.ABSENT_INITIAL:
        if predecessor_kind != NO_PREDECESSOR_HORIZON_KIND:
            hits.append("predecessor_ref")
    elif predecessor_kind not in (None, PREDECESSOR_DECISION_KIND, NO_PREDECESSOR_HORIZON_KIND):
        hits.append("predecessor_ref")
    if value.lineage is LineageState.CONSISTENT and value.lineage_ref is None:
        hits.append("lineage_ref")
    if value.commitment is CommitmentState.ABSENT:
        if value.commitment_ref is not None or value.commitment_deadline_utc is not None:
            hits.append("absent_commitment_has_refs")
    elif value.commitment is not CommitmentState.CONFLICT and (
        value.commitment_ref is None or value.commitment_deadline_utc is None
    ):
        hits.append("commitment_ref")
    if (
        value.commitment is CommitmentState.ACTIVE
        and value.commitment_deadline_utc is not None
        and value.commitment_deadline_utc <= value.now_utc
    ):
        # Expiry is the adapter's (D6-4), but the same sealed clock must agree.
        hits.append("active_commitment_past_deadline")
    if value.ledger is LedgerState.NONE:
        if value.ledger_refs:
            hits.append("ledger_none_has_refs")
    elif value.ledger is not LedgerState.CORRUPT and not value.ledger_refs:
        hits.append("ledger_refs")
    if (value.ledger is LedgerState.REVERTED) != (value.inverse_authorization is not None):
        hits.append("inverse_authorization")
    if value.verified_outcome_ref is not None and value.ledger in _OUTCOME_FORBIDDEN:
        hits.append("outcome_without_effect")
    if (
        value.ledger is LedgerState.ACTIVE_VERIFIED_AUTHORIZED
        and value.verified_outcome_ref is None
    ):
        hits.append("verified_effect_without_outcome")
    if value.feedback is FeedbackState.ABSENT:
        if value.feedback_ref is not None:
            hits.append("absent_feedback_has_ref")
    elif value.feedback is not FeedbackState.INVALID:
        if value.feedback_ref is None:
            hits.append("feedback_ref")
        if value.ledger not in _FEEDBACK_ALLOWED_LEDGER:
            hits.append("feedback_without_verified_effect")
        if (
            value.feedback is FeedbackState.MATCH_SELF
            and value.ledger not in _MATCH_SELF_LEDGER
        ):
            hits.append("positive_feedback_without_authorized_effect")
    # D6-A1: as_of is committed state before the newly assigned event.
    if value.as_of_r07_sequence >= value.event_sequence:
        hits.append("as_of_not_before_event")
    every = [
        *value.source_refs,
        *value.ledger_refs,
        *(
            item
            for item in (
                value.predecessor_ref,
                value.commitment_ref,
                value.feedback_ref,
                value.verified_outcome_ref,
            )
            if item is not None
        ),
    ]
    keys = [item.sort_key for item in every]
    if len(keys) != len(set(keys)):
        hits.append("reference_identity_reused")
    return tuple(hits)


def evaluate(value: SuccessorPolicyInputV2) -> SuccessorPolicyResult:
    """The ordered total table; the first matching row wins."""

    if type(value) is not SuccessorPolicyInputV2:
        raise TypeError("SuccessorPolicyV2 needs an exact SuccessorPolicyInputV2")
    # Row 0: contradiction.
    if (
        value.required_source is RequiredSourceState.CONFLICT
        or value.predecessor is PredecessorState.CONFLICT
        or value.lineage is LineageState.MISMATCH
        or value.commitment is CommitmentState.CONFLICT
        or value.ledger is LedgerState.CORRUPT
        or value.feedback is FeedbackState.INVALID
        or (value.feedback is not FeedbackState.ABSENT and value.ledger is LedgerState.NONE)
        or (
            value.predecessor is PredecessorState.ABSENT_INITIAL
            and value.ledger is not LedgerState.NONE
        )
        or cross_reference_conflicts(value)
    ):
        return _result(SuccessorRow.ROW_0)
    # Row 1: insufficient (self-origin Goals arrive as INELIGIBLE, #379 D2).
    if (
        value.required_source is RequiredSourceState.UNAVAILABLE
        or value.predecessor is PredecessorState.UNAVAILABLE
        or value.lineage is LineageState.UNAVAILABLE
        or value.commitment in (CommitmentState.EXPIRED, CommitmentState.INELIGIBLE)
        or value.interpretation is Interpretation.UNAVAILABLE
    ):
        return _result(SuccessorRow.ROW_1)
    # Row 2: pending or unsafe operation ledger.
    if value.ledger in _PENDING_LEDGER:
        return _result(SuccessorRow.ROW_2)
    # Row 2b: a contradictory interpretation.
    if value.interpretation is Interpretation.CONTRADICTORY:
        return _result(SuccessorRow.ROW_2B)
    qualified = (
        value.required_source is RequiredSourceState.QUALIFIED
        and value.lineage is LineageState.CONSISTENT
    )
    initial = (
        qualified
        and value.ledger is LedgerState.NONE
        and value.feedback is FeedbackState.ABSENT
        and value.predecessor in (PredecessorState.ABSENT_INITIAL, PredecessorState.QUALIFIED)
    )
    # Row 3: initial, no obligation (interpretation is not consulted).
    if initial and value.commitment is CommitmentState.ABSENT:
        return _result(SuccessorRow.ROW_3)
    # Rows 4 / 4b: initial, accepted obligation.
    if initial and value.commitment is CommitmentState.ACTIVE:
        if value.interpretation in _SETTLED_OR_NOT_REVIEWED:
            return _result(SuccessorRow.ROW_4)
        return _result(SuccessorRow.ROW_4B)
    verified = (
        qualified
        and value.ledger is LedgerState.ACTIVE_VERIFIED_AUTHORIZED
        and value.commitment is CommitmentState.ACTIVE
        and value.predecessor is PredecessorState.QUALIFIED
        and value.verified_outcome_ref is not None
    )
    # Rows 5 / 5b: verified authorized effect + qualifying Feedback.
    if verified and value.feedback is FeedbackState.MATCH_SELF:
        if value.interpretation in _SETTLED_OR_NOT_REVIEWED:
            return _result(SuccessorRow.ROW_5)
        return _result(SuccessorRow.ROW_5B)
    # Row 6: verified authorized effect + other Feedback.
    if verified and value.feedback in (
        FeedbackState.ABSENT,
        FeedbackState.MATCH_OTHER_OR_UNKNOWN,
        FeedbackState.MISMATCH,
        FeedbackState.UNKNOWN,
    ):
        return _result(SuccessorRow.ROW_6)
    return _result(SuccessorRow.ROW_7)


def evaluate_sealed(payload: bytes) -> SuccessorPolicyResult:
    """Re-evaluate from stored ``policy_input_bytes`` (D6-3 replay)."""

    return evaluate(SuccessorPolicyInputV2.from_bytes(payload))


# ------------------------------------------------------- R15 interpretation


_R15_STATUS_MAP: Final[Mapping[tuple[str, str], Interpretation]] = {
    ("S1", "PRESENT"): Interpretation.SETTLED,
    ("S1", "UNKNOWN"): Interpretation.UNCERTAIN,
    ("S2", "PRESENT"): Interpretation.SETTLED,
    ("S2", "CONTESTED"): Interpretation.UNCERTAIN,
    ("S2", "UNKNOWN"): Interpretation.UNCERTAIN,
    ("S2", "CONTRADICTORY"): Interpretation.CONTRADICTORY,
}
_UNAVAILABLE_READ: Final = (Interpretation.UNAVAILABLE, InterpretationStage.NONE)


def interpretation_from_r15(
    outcome: object, stage: object, status: object
) -> tuple[Interpretation, InterpretationStage]:
    """#374 D3 §3-1 mapping of the closed R15 read strings (fail-closed).

    EPISODE -> the status class at its stage; NOT_ELIGIBLE -> NOT_REVIEWED;
    UNAVAILABLE or any value outside the closed set -> UNAVAILABLE.
    """

    if type(outcome) is not str or type(stage) is not str:
        return _UNAVAILABLE_READ
    if outcome == "NOT_ELIGIBLE" and stage == "NONE" and status is None:
        return Interpretation.NOT_REVIEWED, InterpretationStage.NONE
    if outcome == "EPISODE" and type(status) is str:
        mapped = _R15_STATUS_MAP.get((stage, status))
        if mapped is not None:
            return mapped, InterpretationStage(stage)
    return _UNAVAILABLE_READ


# ------------------------------------------------------- byte-bound proof


def worst_case_input() -> SuccessorPolicyInputV2:
    """The largest encodable input: every bounded field at its maximum.

    Identifiers are ASCII (no JSON escaping), so maximal lengths and maximal
    integers give the maximal canonical length.  Every ref uses a maximal
    kind and ID; every enum takes its longest member, and the
    (stage, interpretation) pair is the longest closed pair.  The test
    asserts this fits ``MAX_POLICY_INPUT_BYTES`` and that no single-field
    change makes it longer (D6-3 / D3 P2-4 legal-bound proof).
    """

    from suzka.identifiers import MAX_IDENTIFIER_CODEPOINTS

    def ident(prefix: str) -> str:
        return (prefix + "-" + "x" * MAX_IDENTIFIER_CODEPOINTS)[:MAX_IDENTIFIER_CODEPOINTS]

    def ref(name: str) -> PolicyRef:
        return PolicyRef(ident(name + "k"), ident(name), MAX_PERSISTED_REVISION, "f" * 64)

    def longest(kind: type[Enum]) -> Enum:
        return max(kind, key=lambda member: len(str(member.value)))

    def refs(prefix: str, count: int) -> tuple[PolicyRef, ...]:
        return tuple(
            sorted((ref(f"{prefix}{index}") for index in range(count)), key=lambda r: r.sort_key)
        )

    stage, interpretation = max(
        (
            (stage, item)
            for stage, allowed in _STAGE_INTERPRETATIONS.items()
            for item in allowed
        ),
        key=lambda pair: len(pair[0].value) + len(pair[1].value),
    )
    deadline = "9999-12-31T23:59:59Z"
    return SuccessorPolicyInputV2(
        required_source=longest(RequiredSourceState),  # type: ignore[arg-type]
        source_refs=refs("source", len(_REQUIRED_KINDS)),
        predecessor=longest(PredecessorState),  # type: ignore[arg-type]
        predecessor_ref=ref("predecessor"),
        lineage=longest(LineageState),  # type: ignore[arg-type]
        lineage_ref=LineageRef(ident("lineage"), ident("baseline"), "f" * 64, "f" * 64, "f" * 64),
        commitment=longest(CommitmentState),  # type: ignore[arg-type]
        commitment_ref=ref("commitment"),
        commitment_deadline_utc=deadline,
        ledger=longest(LedgerState),  # type: ignore[arg-type]
        ledger_refs=refs("ledger", MAX_LEDGER_REFS),
        inverse_authorization=longest(InverseAuthorization),  # type: ignore[arg-type]
        feedback=longest(FeedbackState),  # type: ignore[arg-type]
        feedback_ref=ref("feedback"),
        verified_outcome_ref=ref("outcome"),
        interpretation=interpretation,
        interpretation_stage=stage,
        as_of_r07_sequence=MAX_PERSISTED_EVENT_SEQUENCE,
        as_of_snapshot_hash="f" * 64,
        event_id=ident("event"),
        event_sequence=MAX_PERSISTED_EVENT_SEQUENCE,
        now_utc=deadline,
    )


__all__ = [
    "INPUT_SCHEMA",
    "MAX_LEDGER_REFS",
    "POLICY_KIND",
    "CommitmentState",
    "FeedbackState",
    "Interpretation",
    "InterpretationStage",
    "InverseAuthorization",
    "NO_PREDECESSOR_HORIZON_KIND",
    "PREDECESSOR_DECISION_KIND",
    "LedgerState",
    "LineageRef",
    "LineageState",
    "PolicyOutputKind",
    "PolicyRef",
    "PredecessorState",
    "RequiredSourceState",
    "SuccessorPolicyInputV2",
    "SuccessorPolicyResult",
    "SuccessorRow",
    "cross_reference_conflicts",
    "evaluate",
    "evaluate_sealed",
    "interpretation_from_r15",
    "utc_timestamp",
    "worst_case_input",
]

