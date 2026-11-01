"""R16 U1: finite, immutable Plan/Decision/Explanation *values* only.

No value authenticates an issuer, qualifies a Source, admits a Goal/Commitment,
commits state, or authorizes execution. R13 Goal and Value references below are
declarations, not fetched roots or proof of subjective adoption. Freshness against
live upstream state requires a separately reviewed future owner (D01 is gated).

All capacity limits are local provisional U1 bounds, NOT selected Human D03
limits and NOT an AgentState budget proof. schema_version=1 is local to these
values; it introduces no AgentState version, persistence or runtime integration.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import ClassVar, Final, Self

from suzka.limits import MAX_PERSISTED_REVISION
from suzka.r15.common import (
    canonical_json,
    checksum,
    closed,
    digest,
    exact_enum,
    identifier,
    integer,
    parse_enum,
    parse_json,
    sequence,
)

SCHEMA_VERSION: Final = 1
MAX_STEPS: Final = 4
MAX_EDGES: Final = 6
MAX_CANDIDATES: Final = 3
MAX_JSON_BYTES: Final = 65_536
UNCERTAINTY_SCALE: Final = 1_000_000


class StepKind(str, Enum):
    """Pure reasoning labels; none denotes a command, tool or executable step."""

    ASSESS = "assess"
    CHECK = "check"
    COMPARE = "compare"


class CandidateKind(str, Enum):
    PLAN = "plan"
    NO_OP = "no_op"


class ReasonCode(str, Enum):
    PLAN_PREFERRED = "plan_preferred"
    NO_OP_PREFERRED = "no_op_preferred"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"


class ExplanationCategory(str, Enum):
    PLAN = "plan"
    NO_OP = "no_op"
    REFUSAL = "refusal"


class ExplanationStatus(str, Enum):
    SELECTED = "selected"
    HOLD = "hold"


def _version(value: object) -> int:
    return integer(
        value, "schema_version", minimum=SCHEMA_VERSION, maximum=SCHEMA_VERSION
    )


def _encode(value: object) -> object:
    if isinstance(value, _Contract):
        return value.canonical_value()
    if isinstance(value, Enum):
        return value.value
    if type(value) is tuple:
        return [_encode(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class _Contract:
    """Shared strict codec, with a distinct digest domain for every pure shape."""

    schema_version: int
    _domain: ClassVar[bytes]

    def __post_init__(self) -> None:
        _version(self.schema_version)

    def canonical_value(self) -> dict[str, object]:
        self.__post_init__()
        body = {
            field.name: _encode(getattr(self, field.name)) for field in fields(self)
        }
        return {**body, "content_digest": checksum(self._domain, body)}

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.canonical_value())

    @property
    def content_digest(self) -> str:
        return digest(self.canonical_value()["content_digest"], "content_digest")

    @classmethod
    def from_value(cls, value: object) -> Self:
        row = closed(
            value,
            (*(field.name for field in fields(cls)), "content_digest"),
            cls.__name__,
        )
        _version(row["schema_version"])
        claimed = digest(row["content_digest"], "content_digest")
        result = cls._from_fields(row)
        if result.content_digest != claimed:
            raise ValueError("R16 content digest does not bind the declared value")
        # Byte comparison also distinguishes bool/int/float, unlike dict equality.
        if result.canonical_bytes() != canonical_json(row):
            raise ValueError("R16 value is not canonical")
        return result

    @classmethod
    def from_json(cls, payload: bytes) -> Self:
        # The reused pure parser bounds bytes/nesting and rejects duplicate keys,
        # malformed UTF-8, nonfinite numbers and every noncanonical JSON encoding.
        return cls.from_value(parse_json(payload, MAX_JSON_BYTES))

    @classmethod
    def _from_fields(cls, row: dict[str, object]) -> Self:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class PlanStep(_Contract):
    step_id: str
    kind: StepKind
    depends_on: tuple[str, ...]
    _domain: ClassVar[bytes] = b"PROJECT-SUZKA:R16:PLAN-STEP:V1\0"

    def __post_init__(self) -> None:
        _Contract.__post_init__(self)
        identifier(self.step_id, "step_id")
        exact_enum(self.kind, StepKind, "step kind")
        deps = sequence(self.depends_on, "depends_on", MAX_STEPS - 1, json_input=False)
        for item in deps:
            identifier(item, "dependency")
        if self.depends_on != tuple(sorted(set(self.depends_on))):
            raise ValueError("dependencies must be sorted and unique")
        if self.step_id in self.depends_on:
            raise ValueError("a step cannot depend on itself")

    @classmethod
    def _from_fields(cls, row: dict[str, object]) -> Self:
        return cls(
            _version(row["schema_version"]),
            identifier(row["step_id"], "step_id"),
            parse_enum(row["kind"], StepKind, "step kind"),
            tuple(
                identifier(item, "dependency")
                for item in sequence(
                    row["depends_on"], "depends_on", MAX_STEPS - 1, json_input=True
                )
            ),
        )


def _topological_order(steps: tuple[PlanStep, ...]) -> tuple[str, ...]:
    remaining = {step.step_id: set(step.depends_on) for step in steps}
    order: list[str] = []
    while remaining:
        ready = sorted(key for key, deps in remaining.items() if not deps)
        if not ready:
            raise ValueError("Plan dependencies contain a cycle")
        # Reconsider ready nodes after each removal: lexicographically least
        # available ID wins, independent of insertion order or hash randomization.
        next_id = ready[0]
        order.append(next_id)
        remaining.pop(next_id)
        for deps in remaining.values():
            deps.discard(next_id)
    return tuple(order)


@dataclass(frozen=True, slots=True)
class PlanDAG(_Contract):
    """A complete, non-executable Plan, in canonical topological step order."""

    plan_id: str
    steps: tuple[PlanStep, ...]
    _domain: ClassVar[bytes] = b"PROJECT-SUZKA:R16:PLAN-DAG:V1\0"

    def __post_init__(self) -> None:
        _Contract.__post_init__(self)
        identifier(self.plan_id, "plan_id")
        sequence(self.steps, "steps", MAX_STEPS, json_input=False)
        if len(self.steps) < 2:
            raise ValueError("a complete Plan requires at least two steps")
        for step in self.steps:
            if type(step) is not PlanStep:
                raise TypeError("Plan steps must be exact PlanStep values")
            step.__post_init__()
        ids = tuple(step.step_id for step in self.steps)
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate Plan step ID")
        if sum(len(step.depends_on) for step in self.steps) > MAX_EDGES:
            raise ValueError("Plan edges exceed the local provisional U1 bound")
        if any(not set(step.depends_on).issubset(ids) for step in self.steps):
            raise ValueError("dangling Plan dependency")
        if ids != _topological_order(self.steps):
            raise ValueError("Plan steps must use canonical topological order")

    @property
    def topological_order(self) -> tuple[str, ...]:
        self.__post_init__()
        return _topological_order(self.steps)

    @classmethod
    def _from_fields(cls, row: dict[str, object]) -> Self:
        return cls(
            _version(row["schema_version"]),
            identifier(row["plan_id"], "plan_id"),
            tuple(
                PlanStep.from_value(item)
                for item in sequence(row["steps"], "steps", MAX_STEPS, json_input=True)
            ),
        )


@dataclass(frozen=True, slots=True)
class GoalEvidenceRef(_Contract):
    """Exact R13 Goal identity/revision/digest assertion, NOT Source admission."""

    goal_id: str
    revision: int
    goal_digest: str
    _domain: ClassVar[bytes] = b"PROJECT-SUZKA:R16:GOAL-EVIDENCE-REF:V1\0"

    def __post_init__(self) -> None:
        _Contract.__post_init__(self)
        # R13 Goal identities are SHA256 IDs, not descriptions or target text.
        digest(self.goal_id, "goal_id")
        integer(self.revision, "Goal revision", maximum=MAX_PERSISTED_REVISION)
        digest(self.goal_digest, "goal_digest")

    @classmethod
    def _from_fields(cls, row: dict[str, object]) -> Self:
        return cls(
            _version(row["schema_version"]),
            digest(row["goal_id"], "goal_id"),
            integer(row["revision"], "Goal revision", maximum=MAX_PERSISTED_REVISION),
            digest(row["goal_digest"], "goal_digest"),
        )


@dataclass(frozen=True, slots=True)
class ValueEvidenceRef(_Contract):
    """Exact Value identity/revision/digest assertion, NOT issuer qualification."""

    value_id: str
    revision: int
    value_digest: str
    _domain: ClassVar[bytes] = b"PROJECT-SUZKA:R16:VALUE-EVIDENCE-REF:V1\0"

    def __post_init__(self) -> None:
        _Contract.__post_init__(self)
        identifier(self.value_id, "value_id")
        integer(self.revision, "Value revision", maximum=MAX_PERSISTED_REVISION)
        digest(self.value_digest, "value_digest")

    @classmethod
    def _from_fields(cls, row: dict[str, object]) -> Self:
        return cls(
            _version(row["schema_version"]),
            identifier(row["value_id"], "value_id"),
            integer(row["revision"], "Value revision", maximum=MAX_PERSISTED_REVISION),
            digest(row["value_digest"], "value_digest"),
        )


@dataclass(frozen=True, slots=True)
class PlanCandidate(_Contract):
    """A Plan or explicit NO_OP, with private declarative evidence references."""

    candidate_id: str
    kind: CandidateKind
    goal: GoalEvidenceRef
    value: ValueEvidenceRef
    plan: PlanDAG | None
    _domain: ClassVar[bytes] = b"PROJECT-SUZKA:R16:PLAN-CANDIDATE:V1\0"

    def __post_init__(self) -> None:
        _Contract.__post_init__(self)
        identifier(self.candidate_id, "candidate_id")
        exact_enum(self.kind, CandidateKind, "candidate kind")
        if (
            type(self.goal) is not GoalEvidenceRef
            or type(self.value) is not ValueEvidenceRef
        ):
            raise TypeError("candidate requires exact Goal/Value evidence refs")
        self.goal.__post_init__()
        self.value.__post_init__()
        if self.kind is CandidateKind.NO_OP:
            if self.plan is not None:
                raise ValueError("NO_OP cannot contain a Plan")
        else:
            if type(self.plan) is not PlanDAG:
                raise TypeError("Plan candidate requires an exact complete PlanDAG")
            self.plan.__post_init__()

    @classmethod
    def _from_fields(cls, row: dict[str, object]) -> Self:
        return cls(
            _version(row["schema_version"]),
            identifier(row["candidate_id"], "candidate_id"),
            parse_enum(row["kind"], CandidateKind, "candidate kind"),
            GoalEvidenceRef.from_value(row["goal"]),
            ValueEvidenceRef.from_value(row["value"]),
            None if row["plan"] is None else PlanDAG.from_value(row["plan"]),
        )


@dataclass(frozen=True, slots=True)
class CandidateSelection(_Contract):
    """Bind a selection to exact candidate content, not just a reusable ID."""

    candidate_id: str
    candidate_digest: str
    _domain: ClassVar[bytes] = b"PROJECT-SUZKA:R16:CANDIDATE-SELECTION:V1\0"

    def __post_init__(self) -> None:
        _Contract.__post_init__(self)
        identifier(self.candidate_id, "candidate_id")
        digest(self.candidate_digest, "candidate_digest")

    @classmethod
    def _from_fields(cls, row: dict[str, object]) -> Self:
        return cls(
            _version(row["schema_version"]),
            identifier(row["candidate_id"], "candidate_id"),
            digest(row["candidate_digest"], "candidate_digest"),
        )


@dataclass(frozen=True, slots=True)
class DecisionRecord(_Contract):
    """A selection *value*, never a committed record or execution authorization.

    Exactly two distinct Plans and one explicit NO_OP compare the same declared
    Goal/Value snapshot. This rejects internal stale/conflicting references, not
    externally stale roots. The selected candidate's digest must match exactly.
    HOLD is intentionally absent from this schema and its candidate kind enum.
    """

    decision_id: str
    candidates: tuple[PlanCandidate, ...]
    selected: CandidateSelection
    reason: ReasonCode
    uncertainty: int
    _domain: ClassVar[bytes] = b"PROJECT-SUZKA:R16:DECISION-RECORD:V1\0"

    def __post_init__(self) -> None:
        _Contract.__post_init__(self)
        identifier(self.decision_id, "decision_id")
        sequence(self.candidates, "candidates", MAX_CANDIDATES, json_input=False)
        if len(self.candidates) != MAX_CANDIDATES:
            raise ValueError("U1 requires two Plan candidates and one explicit NO_OP")
        for candidate in self.candidates:
            if type(candidate) is not PlanCandidate:
                raise TypeError("candidates must be exact PlanCandidate values")
            candidate.__post_init__()
        ids = tuple(candidate.candidate_id for candidate in self.candidates)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("candidate IDs must be sorted and unique")
        plans = tuple(item.plan for item in self.candidates if item.plan is not None)
        if len(plans) != 2 or plans[0].plan_id == plans[1].plan_id:
            raise ValueError("two distinct Plan identities and one NO_OP are required")
        if plans[0].steps == plans[1].steps:
            raise ValueError("competing Plans cannot be identical steps under new IDs")
        first = self.candidates[0]
        if any(
            item.goal != first.goal or item.value != first.value
            for item in self.candidates
        ):
            raise ValueError("candidates must compare one exact evidence snapshot")
        if type(self.selected) is not CandidateSelection:
            raise TypeError("selection must be an exact CandidateSelection")
        self.selected.__post_init__()
        matching = tuple(
            item
            for item in self.candidates
            if item.candidate_id == self.selected.candidate_id
        )
        if not matching or matching[0].content_digest != self.selected.candidate_digest:
            raise ValueError("selection is absent or refers to stale candidate content")
        exact_enum(self.reason, ReasonCode, "decision reason")
        expected = (
            ReasonCode.PLAN_PREFERRED
            if matching[0].kind is CandidateKind.PLAN
            else ReasonCode.NO_OP_PREFERRED
        )
        if self.reason is not expected:
            raise ValueError("decision requires matching selection reason, not HOLD")
        integer(self.uncertainty, "uncertainty", maximum=UNCERTAINTY_SCALE)

    def explanation(self) -> ExplanationProjection:
        self.__post_init__()
        category = (
            ExplanationCategory.PLAN
            if self.reason is ReasonCode.PLAN_PREFERRED
            else ExplanationCategory.NO_OP
        )
        return ExplanationProjection(
            SCHEMA_VERSION,
            category,
            ExplanationStatus.SELECTED,
            self.reason,
            self.uncertainty,
        )

    @classmethod
    def _from_fields(cls, row: dict[str, object]) -> Self:
        return cls(
            _version(row["schema_version"]),
            identifier(row["decision_id"], "decision_id"),
            tuple(
                PlanCandidate.from_value(item)
                for item in sequence(
                    row["candidates"], "candidates", MAX_CANDIDATES, json_input=True
                )
            ),
            CandidateSelection.from_value(row["selected"]),
            parse_enum(row["reason"], ReasonCode, "decision reason"),
            integer(row["uncertainty"], "uncertainty", maximum=UNCERTAINTY_SCALE),
        )


@dataclass(frozen=True, slots=True)
class HoldAssessment(_Contract):
    """Non-authoritative refusal/assessment; no selection or DecisionRecord.

    Its own canonical assessment bytes are not a canonical committed decision.
    No conversion to DecisionRecord, admission, publication or commit API exists.
    """

    reason: ReasonCode
    uncertainty: int
    _domain: ClassVar[bytes] = b"PROJECT-SUZKA:R16:HOLD-ASSESSMENT:V1\0"

    def __post_init__(self) -> None:
        _Contract.__post_init__(self)
        exact_enum(self.reason, ReasonCode, "assessment reason")
        if self.reason not in (
            ReasonCode.INSUFFICIENT_EVIDENCE,
            ReasonCode.CONFLICTING_EVIDENCE,
        ):
            raise ValueError("HOLD requires a refusal reason, not a selected decision")
        integer(self.uncertainty, "uncertainty", maximum=UNCERTAINTY_SCALE)

    def explanation(self) -> ExplanationProjection:
        self.__post_init__()
        return ExplanationProjection(
            SCHEMA_VERSION,
            ExplanationCategory.REFUSAL,
            ExplanationStatus.HOLD,
            self.reason,
            self.uncertainty,
        )

    @classmethod
    def _from_fields(cls, row: dict[str, object]) -> Self:
        return cls(
            _version(row["schema_version"]),
            parse_enum(row["reason"], ReasonCode, "assessment reason"),
            integer(row["uncertainty"], "uncertainty", maximum=UNCERTAINTY_SCALE),
        )


@dataclass(frozen=True, slots=True)
class ExplanationProjection(_Contract):
    """Public-safe vocabulary only; neither identifiers nor free text fit here.

    This value does NOT validate a publisher's authority. Construction, parsing
    and checksum verification establish shape only, never publication permission.
    """

    category: ExplanationCategory
    status: ExplanationStatus
    reason: ReasonCode
    uncertainty: int
    _domain: ClassVar[bytes] = b"PROJECT-SUZKA:R16:EXPLANATION-PROJECTION:V1\0"

    def __post_init__(self) -> None:
        _Contract.__post_init__(self)
        exact_enum(self.category, ExplanationCategory, "explanation category")
        exact_enum(self.status, ExplanationStatus, "explanation status")
        exact_enum(self.reason, ReasonCode, "explanation reason")
        integer(self.uncertainty, "uncertainty", maximum=UNCERTAINTY_SCALE)
        if self.category is ExplanationCategory.REFUSAL:
            if self.status is not ExplanationStatus.HOLD or self.reason not in (
                ReasonCode.INSUFFICIENT_EVIDENCE,
                ReasonCode.CONFLICTING_EVIDENCE,
            ):
                raise ValueError("refusal projection must describe a HOLD assessment")
        else:
            expected = (
                ReasonCode.PLAN_PREFERRED
                if self.category is ExplanationCategory.PLAN
                else ReasonCode.NO_OP_PREFERRED
            )
            if (
                self.status is not ExplanationStatus.SELECTED
                or self.reason is not expected
            ):
                raise ValueError("selection projection category/status/reason disagree")

    @classmethod
    def _from_fields(cls, row: dict[str, object]) -> Self:
        return cls(
            _version(row["schema_version"]),
            parse_enum(row["category"], ExplanationCategory, "explanation category"),
            parse_enum(row["status"], ExplanationStatus, "explanation status"),
            parse_enum(row["reason"], ReasonCode, "explanation reason"),
            integer(row["uncertainty"], "uncertainty", maximum=UNCERTAINTY_SCALE),
        )
