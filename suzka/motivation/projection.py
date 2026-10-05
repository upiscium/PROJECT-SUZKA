"""Pure, immutable R13 state projection for prompt construction.

This module copies only current active Motivations, adopted Goals, and active
Commitments.  It deliberately has no admission, evidence, history, runtime,
model, or persistence authority.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from typing import Final

from suzka.identifiers import MAX_IDENTIFIER_CODEPOINTS
from suzka.motivation.commitment import (
    COMMITMENT_BENEFICIARY_KINDS,
    CommitmentLifecycle,
    CommitmentRecord,
    commitment_id_for_proposal,
)
from suzka.motivation.common import (
    R13_MAX_SCOPE_CODEPOINTS,
    R13_MAX_SCOPE_ITEMS,
    Deadline,
    R13Reference,
    R13ReferenceKind,
    bounded_fraction,
    canonical_json,
    normalize_text,
    validate_digest,
)
from suzka.motivation.goal import (
    GOAL_TARGET_KINDS,
    GoalLifecycle,
    GoalRecord,
    goal_id_for_target,
)
from suzka.motivation.commitment_system import CommitmentSystemSnapshot
from suzka.motivation.goal_system import GoalSystemSnapshot
from suzka.motivation.motivation import (
    MOTIVATION_TARGET_KINDS,
    MotivationKind,
    MotivationLifecycle,
    MotivationRecord,
    motivation_id_for_target,
)
from suzka.motivation.system import MotivationSystemSnapshot


R13_PROMPT_MAX_RECORDS_PER_DOMAIN: Final = 32
R13_PROMPT_SCHEMA_VERSION: Final = 1

_MOTIVATION_FLOAT_FIELDS: Final = (
    "strength",
    "persistence",
    "satiation",
    "uncertainty",
)
_PROMPT_INTRO: Final = (
    "A current Motivation is not a Goal; an adopted Goal is not a Commitment. "
    "An active Commitment is accepted responsibility."
)
_RENDER_SECTIONS: Final = (
    ("motivations", "Current Motivations:"),
    ("goals", "Adopted Goals:"),
    ("commitments", "Active Commitments:"),
)


def _checked_reference(
    value: object,
    name: str,
    allowed_kinds: frozenset[R13ReferenceKind],
) -> R13Reference:
    if type(value) is not R13Reference:
        raise TypeError(f"{name} must be an exact R13Reference")
    checked = R13Reference(value.kind, value.reference)
    if checked.kind not in allowed_kinds:
        raise ValueError(f"{name} contains an unauthorized reference kind")
    return checked


def _canonical_text(value: object, name: str, maximum: int) -> str:
    text = normalize_text(value, name, maximum)
    if text != value:
        raise ValueError(f"{name} must be canonical text")
    return text


def _checked_deadline(value: object) -> Deadline:
    if type(value) is not Deadline:
        raise TypeError("deadline must be an exact Deadline")
    return Deadline(value.at)


@dataclass(frozen=True, slots=True, init=False)
class MotivationPromptEntry:
    """Minimal bounded current Motivation data safe to render in a prompt."""

    motivation_id: str
    kind: MotivationKind
    target: R13Reference
    strength: float
    persistence: float
    satiation: float
    uncertainty: float

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError(
            "MotivationPromptEntry instances are created by R13PromptView"
        )

    @classmethod
    def _from_record(cls, record: MotivationRecord) -> MotivationPromptEntry:
        if type(record) is not MotivationRecord:
            raise TypeError("record must be an exact MotivationRecord")
        if record.lifecycle is not MotivationLifecycle.ACTIVE:
            raise ValueError("only active Motivations may enter a prompt view")
        current = replace(record)
        entry = object.__new__(cls)
        object.__setattr__(entry, "motivation_id", current.motivation_id)
        object.__setattr__(entry, "kind", current.kind)
        object.__setattr__(entry, "target", current.target)
        for name in _MOTIVATION_FLOAT_FIELDS:
            object.__setattr__(entry, name, getattr(current, name))
        entry.__post_init__()
        return entry

    def __post_init__(self) -> None:
        motivation_id = validate_digest(self.motivation_id, "motivation_id")
        if type(self.kind) is not MotivationKind:
            raise TypeError("kind must be a MotivationKind")
        target = _checked_reference(self.target, "target", MOTIVATION_TARGET_KINDS)
        if motivation_id_for_target(self.kind, target) != motivation_id:
            raise ValueError("motivation_id does not match kind and target")
        object.__setattr__(self, "motivation_id", motivation_id)
        object.__setattr__(self, "target", target)
        for name in _MOTIVATION_FLOAT_FIELDS:
            value = getattr(self, name)
            if type(value) is not float:
                raise TypeError(f"{name} must be an exact float")
            object.__setattr__(self, name, bounded_fraction(value, name))


@dataclass(frozen=True, slots=True, init=False)
class GoalPromptEntry:
    """Minimal bounded adopted Goal data safe to render in a prompt."""

    goal_id: str
    target: R13Reference
    description: str
    deadline: Deadline

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("GoalPromptEntry instances are created by R13PromptView")

    @classmethod
    def _from_record(cls, record: GoalRecord) -> GoalPromptEntry:
        if type(record) is not GoalRecord:
            raise TypeError("record must be an exact GoalRecord")
        if record.lifecycle is not GoalLifecycle.ADOPTED:
            raise ValueError("only adopted Goals may enter a prompt view")
        current = replace(record)
        entry = object.__new__(cls)
        object.__setattr__(entry, "goal_id", current.goal_id)
        object.__setattr__(entry, "target", current.target)
        object.__setattr__(entry, "description", current.description)
        object.__setattr__(entry, "deadline", current.deadline)
        entry.__post_init__()
        return entry

    def __post_init__(self) -> None:
        goal_id = validate_digest(self.goal_id, "goal_id")
        target = _checked_reference(self.target, "target", GOAL_TARGET_KINDS)
        if goal_id_for_target(target) != goal_id:
            raise ValueError("goal_id does not match target")
        description = _canonical_text(self.description, "description", 1_024)
        deadline = _checked_deadline(self.deadline)
        object.__setattr__(self, "goal_id", goal_id)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "deadline", deadline)


@dataclass(frozen=True, slots=True, init=False)
class CommitmentPromptEntry:
    """Minimal bounded current Commitment data; never includes its subject."""

    commitment_id: str
    beneficiary: R13Reference
    scope: tuple[str, ...]
    deadline: Deadline

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError(
            "CommitmentPromptEntry instances are created by R13PromptView"
        )

    @classmethod
    def _from_record(cls, record: CommitmentRecord) -> CommitmentPromptEntry:
        if type(record) is not CommitmentRecord:
            raise TypeError("record must be an exact CommitmentRecord")
        if record.lifecycle is not CommitmentLifecycle.ACTIVE:
            raise ValueError("only active Commitments may enter a prompt view")
        current = replace(record)
        if current.commitment_id != record.commitment_id or record.commitment_id != (
            commitment_id_for_proposal(record.proposal_digest)
        ):
            raise ValueError("commitment_id does not match its proposal digest")
        entry = object.__new__(cls)
        object.__setattr__(entry, "commitment_id", current.commitment_id)
        object.__setattr__(entry, "beneficiary", current.beneficiary)
        object.__setattr__(entry, "scope", current.scope)
        object.__setattr__(entry, "deadline", current.deadline)
        entry.__post_init__()
        return entry

    def __post_init__(self) -> None:
        commitment_id = validate_digest(self.commitment_id, "commitment_id")
        if type(self.beneficiary) is not R13Reference:
            raise TypeError("beneficiary must be an exact R13Reference")
        beneficiary = _checked_reference(
            self.beneficiary,
            "beneficiary",
            COMMITMENT_BENEFICIARY_KINDS,
        )
        if type(self.scope) is not tuple:
            raise TypeError("scope must be a tuple")
        if not self.scope or len(self.scope) > R13_MAX_SCOPE_ITEMS:
            raise ValueError("scope must be non-empty and within its item bound")
        scope = tuple(
            _canonical_text(item, "scope item", R13_MAX_SCOPE_CODEPOINTS)
            for item in self.scope
        )
        if scope != tuple(sorted(set(scope))):
            raise ValueError("scope must be canonically ordered and unique")
        deadline = _checked_deadline(self.deadline)
        object.__setattr__(self, "commitment_id", commitment_id)
        object.__setattr__(self, "beneficiary", beneficiary)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "deadline", deadline)


def _motivation_value(entry: MotivationPromptEntry) -> dict[str, object]:
    return {
        "kind": entry.kind.value,
        "motivation_id": entry.motivation_id,
        "persistence": entry.persistence.hex(),
        "satiation": entry.satiation.hex(),
        "strength": entry.strength.hex(),
        "target": entry.target.canonical_value(),
        "uncertainty": entry.uncertainty.hex(),
    }


def _goal_value(entry: GoalPromptEntry) -> dict[str, object]:
    return {
        "deadline": entry.deadline.canonical_value(),
        "description": entry.description,
        "goal_id": entry.goal_id,
        "target": entry.target.canonical_value(),
    }


def _commitment_value(entry: CommitmentPromptEntry) -> dict[str, object]:
    return {
        "beneficiary": entry.beneficiary.canonical_value(),
        "commitment_id": entry.commitment_id,
        "deadline": entry.deadline.canonical_value(),
        "scope": list(entry.scope),
    }


def _canonical_view_value(
    motivations: tuple[dict[str, object], ...],
    goals: tuple[dict[str, object], ...],
    commitments: tuple[dict[str, object], ...],
) -> dict[str, object]:
    return {
        "commitments": list(commitments),
        "goals": list(goals),
        "motivations": list(motivations),
        "schema_version": R13_PROMPT_SCHEMA_VERSION,
    }


def _canonical_ascii(value: object) -> str:
    return canonical_json(value).decode("ascii")


def _render_value(value: dict[str, object]) -> str:
    lines = [_PROMPT_INTRO]
    for section_name, heading in _RENDER_SECTIONS:
        entries = value[section_name]
        if type(entries) is not list:
            raise TypeError("canonical prompt sections must be lists")
        lines.append(heading)
        lines.extend(_canonical_ascii(entry) for entry in entries)
    return "\n".join(lines)


def _longest_enum_value(values: Iterable[Enum]) -> str:
    return max((item.value for item in values), key=len)


def _maximum_reference_value(
    kinds: frozenset[R13ReferenceKind],
) -> dict[str, str]:
    return {
        "kind": _longest_enum_value(kinds),
        "reference": "a" * MAX_IDENTIFIER_CODEPOINTS,
    }


def _maximum_scope_value() -> list[str]:
    # Distinct, canonically ordered valid scope values retain the full
    # twelve-ASCII-byte JSON escape budget for every supplementary codepoint.
    return [
        chr(0x10000 + index) + chr(0x10000) * (R13_MAX_SCOPE_CODEPOINTS - 1)
        for index in range(R13_MAX_SCOPE_ITEMS)
    ]


def _maximum_prompt_value() -> dict[str, object]:
    # Keep this shape in lockstep with the three canonical entry serializers
    # above; every text field uses its contract bound and every domain uses its
    # complete record budget. The 25-byte float token is a conservative
    # envelope for bounded float.hex() output.
    max_digest = "a" * 64
    max_float_hex = "x" * 25
    max_deadline = Deadline(datetime.max.replace(tzinfo=UTC)).canonical_value()
    non_bmp = "\U00010000"
    max_motivation = {
        "kind": _longest_enum_value(MotivationKind),
        "motivation_id": max_digest,
        "persistence": max_float_hex,
        "satiation": max_float_hex,
        "strength": max_float_hex,
        "target": _maximum_reference_value(MOTIVATION_TARGET_KINDS),
        "uncertainty": max_float_hex,
    }
    max_goal = {
        "deadline": max_deadline,
        "description": non_bmp * 1_024,
        "goal_id": max_digest,
        "target": _maximum_reference_value(GOAL_TARGET_KINDS),
    }
    max_commitment = {
        "beneficiary": _maximum_reference_value(COMMITMENT_BENEFICIARY_KINDS),
        "commitment_id": max_digest,
        "deadline": max_deadline,
        "scope": _maximum_scope_value(),
    }
    return _canonical_view_value(
        tuple(dict(max_motivation) for _ in range(R13_PROMPT_MAX_RECORDS_PER_DOMAIN)),
        tuple(dict(max_goal) for _ in range(R13_PROMPT_MAX_RECORDS_PER_DOMAIN)),
        tuple(
            dict(max_commitment)
            for _ in range(R13_PROMPT_MAX_RECORDS_PER_DOMAIN)
        ),
    )


_MAXIMUM_PROMPT_VALUE: Final = _maximum_prompt_value()
R13_PROMPT_MAX_SERIALIZED_BYTES: Final = len(
    canonical_json(_MAXIMUM_PROMPT_VALUE)
)
R13_PROMPT_MAX_RENDERED_BYTES: Final = len(
    _render_value(_MAXIMUM_PROMPT_VALUE).encode("utf-8")
)


@dataclass(frozen=True, slots=True)
class R13PromptView:
    """A fully bounded immutable projection of all active R13 records."""

    motivations: tuple[MotivationPromptEntry, ...] = ()
    goals: tuple[GoalPromptEntry, ...] = ()
    commitments: tuple[CommitmentPromptEntry, ...] = ()
    serialized_bytes: int = field(init=False)

    def __post_init__(self) -> None:
        for name, entries, entry_type, id_attribute in (
            ("motivations", self.motivations, MotivationPromptEntry, "motivation_id"),
            ("goals", self.goals, GoalPromptEntry, "goal_id"),
            ("commitments", self.commitments, CommitmentPromptEntry, "commitment_id"),
        ):
            if type(entries) is not tuple:
                raise TypeError(f"{name} must be an exact tuple")
            if len(entries) > R13_PROMPT_MAX_RECORDS_PER_DOMAIN:
                raise ValueError(f"{name} exceeds its per-domain prompt bound")
            if any(type(entry) is not entry_type for entry in entries):
                raise TypeError(f"{name} must contain exact {entry_type.__name__} values")
            for entry in entries:
                entry.__post_init__()
            ids = tuple(getattr(entry, id_attribute) for entry in entries)
            if ids != tuple(sorted(set(ids))):
                raise ValueError(f"{name} must be sorted by unique stable identity")

        encoded = canonical_json(self.canonical_value())
        if len(encoded) > R13_PROMPT_MAX_SERIALIZED_BYTES:
            raise ValueError("R13 prompt view exceeds its derived serialized byte bound")
        object.__setattr__(self, "serialized_bytes", len(encoded))

    @classmethod
    def from_snapshots(
        cls,
        motivation: MotivationSystemSnapshot,
        goal: GoalSystemSnapshot,
        commitment: CommitmentSystemSnapshot,
    ) -> R13PromptView:
        """Validate exact authority snapshots, then copy every active record."""

        for name, snapshot, snapshot_type in (
            ("motivation", motivation, MotivationSystemSnapshot),
            ("goal", goal, GoalSystemSnapshot),
            ("commitment", commitment, CommitmentSystemSnapshot),
        ):
            if type(snapshot) is not snapshot_type:
                raise TypeError(f"{name} must be an exact {snapshot_type.__name__}")

        validated_motivation = replace(motivation)
        validated_goal = replace(goal)
        validated_commitment = replace(commitment)

        if any(type(record) is not MotivationRecord for record in validated_motivation.records):
            raise TypeError("Motivation snapshot contains a non-canonical record type")
        if any(type(record) is not GoalRecord for record in validated_goal.records):
            raise TypeError("Goal snapshot contains a non-canonical record type")
        if any(
            type(record) is not CommitmentRecord
            for record in validated_commitment.records
        ):
            raise TypeError("Commitment snapshot contains a non-canonical record type")

        motivations = tuple(
            sorted(
                (
                    MotivationPromptEntry._from_record(record)
                    for record in validated_motivation.records
                    if record.lifecycle is MotivationLifecycle.ACTIVE
                ),
                key=lambda entry: entry.motivation_id,
            )
        )
        goals = tuple(
            sorted(
                (
                    GoalPromptEntry._from_record(record)
                    for record in validated_goal.records
                    if record.lifecycle is GoalLifecycle.ADOPTED
                ),
                key=lambda entry: entry.goal_id,
            )
        )
        commitments = tuple(
            sorted(
                (
                    CommitmentPromptEntry._from_record(record)
                    for record in validated_commitment.records
                    if record.lifecycle is CommitmentLifecycle.ACTIVE
                ),
                key=lambda entry: entry.commitment_id,
            )
        )
        return cls(motivations, goals, commitments)

    def canonical_value(self) -> dict[str, object]:
        """Return only the minimal current-state prompt schema."""

        return _canonical_view_value(
            tuple(_motivation_value(entry) for entry in self.motivations),
            tuple(_goal_value(entry) for entry in self.goals),
            tuple(_commitment_value(entry) for entry in self.commitments),
        )

    def render(self) -> str:
        """Render all projected state without ranking, truncation, or injection."""

        rendered = _render_value(self.canonical_value())
        if len(rendered.encode("utf-8")) > R13_PROMPT_MAX_RENDERED_BYTES:
            raise ValueError("R13 prompt view exceeds its derived rendered byte bound")
        return rendered


__all__ = [
    "CommitmentPromptEntry",
    "GoalPromptEntry",
    "MotivationPromptEntry",
    "R13_PROMPT_MAX_RECORDS_PER_DOMAIN",
    "R13_PROMPT_MAX_RENDERED_BYTES",
    "R13_PROMPT_MAX_SERIALIZED_BYTES",
    "R13_PROMPT_SCHEMA_VERSION",
    "R13PromptView",
]
