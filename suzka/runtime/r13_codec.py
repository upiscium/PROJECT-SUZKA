"""Lossless closed node-table codec for R13 authority, owned by persistence.

Only the explicitly registered immutable U1-U4 values can appear. Shared refs,
proofs, genesis and revision witnesses are interned in deterministic postorder;
indexes are backward-only and decoding invokes value validators, never mutation
policy. Computed digests are retained and checked. Exact re-encoding rejects
duplicate/unreachable/reordered nodes and noncanonical scalar spellings.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
import json
from types import UnionType
from typing import Annotated, Any, Callable, Final, Self, Union, get_args, get_origin, get_type_hints, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from pydantic.config import ExtraValues

from suzka.motivation.common import (
    Deadline, MotivationModelIdentity, R13Reference, RevisionCompactionAnchor,
    canonical_datetime,
)
from suzka.motivation.commitment import (
    CommitmentRecord, CommitmentRevisionRecord, CommitmentSubjectAdmission,
    CommitmentSubjectTransitionProof,
)
from suzka.motivation.commitment_system import (
    CommitmentSystemEventReceipt, CommitmentSystemSnapshot,
)
from suzka.motivation.goal import (
    GoalRecord, GoalRevisionRecord, GoalSubjectAdmission, GoalSubjectTransitionProof,
)
from suzka.motivation.goal_system import GoalSystemEventReceipt, GoalSystemSnapshot
from suzka.motivation.motivation import (
    MotivationInterpretationCandidate, MotivationRecord, MotivationRevisionRecord,
)
from suzka.motivation.system import (
    MotivationEvidence, MotivationEvidenceLedgerEntry, MotivationEventReceipt,
    MotivationGoalProposalWitness, MotivationSystemSnapshot,
)


R13_CODEC_SCHEMA_VERSION: Final = 1
_TYPES: Final = (
    Deadline, MotivationModelIdentity, R13Reference, RevisionCompactionAnchor,
    MotivationRecord, MotivationRevisionRecord, MotivationInterpretationCandidate,
    MotivationEvidence, MotivationEvidenceLedgerEntry, MotivationEventReceipt,
    MotivationGoalProposalWitness, MotivationSystemSnapshot,
    GoalRecord, GoalRevisionRecord, GoalSubjectAdmission, GoalSubjectTransitionProof,
    GoalSystemEventReceipt, GoalSystemSnapshot,
    CommitmentRecord, CommitmentRevisionRecord, CommitmentSubjectAdmission,
    CommitmentSubjectTransitionProof, CommitmentSystemEventReceipt, CommitmentSystemSnapshot,
)
_REGISTRY: Final = {item.__name__: item for item in _TYPES}
_ROOTS: Final = {
    "motivation": MotivationSystemSnapshot,
    "goal": GoalSystemSnapshot,
    "commitment": CommitmentSystemSnapshot,
}

# Counts follow actual snapshot cross-links, not the process-local byte guards.
# Revisions shared by records/receipts are interned; Commitment has only two
# subject transitions, and all retained scopes remain present in proof nodes.
_COUNTS: Final = {
    "motivation": {
        MotivationSystemSnapshot: 1, MotivationRecord: 32,
        MotivationRevisionRecord: 256, RevisionCompactionAnchor: 32,
        MotivationInterpretationCandidate: 32, MotivationModelIdentity: 32,
        MotivationEvidenceLedgerEntry: 1024, MotivationEvidence: 1024,
        MotivationEventReceipt: 1024, MotivationGoalProposalWitness: 32,
        # Record refs (81 each), ledger target+32 origins (source in records),
        # candidate refs (33 each), and 32 Motive-ID receipt refs.
        R13Reference: 32 * 81 + 1024 * 33 + 32 * 33 + 32,
    },
    "goal": {
        GoalSystemSnapshot: 1, GoalRecord: 32, GoalSystemEventReceipt: 1024,
        GoalRevisionRecord: 1056, GoalSubjectAdmission: 32,
        GoalSubjectTransitionProof: 32 * 9, RevisionCompactionAnchor: 32,
        Deadline: 32,
        # Upper bound without needing the maximum simultaneous transition count.
        R13Reference: 32 * 65 + 1024 * 32,
    },
    "commitment": {
        CommitmentSystemSnapshot: 1, CommitmentRecord: 32,
        CommitmentSystemEventReceipt: 256, CommitmentRevisionRecord: 96,
        CommitmentSubjectAdmission: 32, CommitmentSubjectTransitionProof: 32,
        Deadline: 32,
        R13Reference: 32 * 97 + 64 * 32,
    },
}
_MAX_NODES: Final = max(sum(value.values()) for value in _COUNTS.values())
_BoundedString = Annotated[str, Field(strict=True, max_length=1024)]


def _json_bytes(value: object) -> bytes:
    # Same UTF-8, compact, sorted-key encoding as AgentState's canonical root.
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":")).encode("utf-8")


def _encode_scalar(value: object, visit: Any, intern: Any) -> JsonValue:
    if type(value) in _TYPES:
        return cast(int, visit(value))
    if isinstance(value, Enum):
        return cast(str, value.value)
    if isinstance(value, datetime):
        return canonical_datetime(value)
    if type(value) is float:
        return value.hex()
    if type(value) is tuple:
        return [_encode_scalar(item, visit, intern) for item in value]
    if type(value) is str:
        return cast(int, intern(value))
    if value is None or type(value) in (int, bool):
        return cast(JsonValue, value)
    raise ValueError("unsupported R13 codec value")


def _encode_nodes(root: object) -> tuple[list[list[JsonValue]], list[str]]:
    nodes: list[list[JsonValue]] = []
    indexes: dict[bytes, int] = {}
    strings: list[str] = []
    string_indexes: dict[str, int] = {}

    def intern(value: str) -> int:
        if value not in string_indexes:
            string_indexes[value] = len(strings)
            strings.append(value)
        return string_indexes[value]

    def visit(value: object) -> int:
        if type(value) not in _TYPES:
            raise ValueError("unregistered R13 codec value")
        value_fields = fields(cast(Any, value))
        init = [_encode_scalar(getattr(value, item.name), visit, intern)
                for item in value_fields if item.init]
        computed = [_encode_scalar(getattr(value, item.name), visit, intern)
                    for item in value_fields if not item.init]
        row: list[JsonValue] = [type(value).__name__, init, computed]
        identity = _json_bytes(row)
        if identity not in indexes:
            indexes[identity] = len(nodes)
            nodes.append(row)
        return indexes[identity]

    if visit(root) != len(nodes) - 1:
        raise ValueError("R13 codec root must be last")
    return nodes, strings


def _decode_scalar(
    value: JsonValue, annotation: object, prior: list[object], strings: tuple[str, ...]
) -> object:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (Union, UnionType):
        for choice in args:
            try:
                return _decode_scalar(value, choice, prior, strings)
            except (TypeError, ValueError):
                continue
        raise ValueError("R13 codec union value is invalid")
    if annotation is type(None):
        if value is not None:
            raise ValueError("R13 codec null required")
        return None
    if annotation in _TYPES:
        if type(value) is not int or not 0 <= value < len(prior):
            raise ValueError("R13 codec index is not backward and bounded")
        result = prior[value]
        if type(result) is not annotation:
            raise ValueError("R13 codec index has wrong value type")
        return result
    if origin is tuple:
        if not isinstance(value, list) or len(args) != 2 or args[1] is not Ellipsis:
            raise ValueError("R13 codec tuple shape is invalid")
        return tuple(_decode_scalar(item, args[0], prior, strings) for item in value)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        if type(value) is not str:
            raise ValueError("R13 codec enum must be a string")
        return annotation(value)
    if annotation is datetime:
        if type(value) is not str:
            raise ValueError("R13 codec datetime must be canonical UTC")
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if canonical_datetime(result) != value or result.tzinfo is None:
            raise ValueError("R13 codec datetime is noncanonical")
        return result
    if annotation is float:
        if type(value) is not str or len(value) > 25:
            raise ValueError("R13 codec float must be canonical hexadecimal")
        result_float = float.fromhex(value)
        if result_float.hex() != value:
            raise ValueError("R13 codec float is noncanonical")
        return result_float
    if annotation is str:
        if type(value) is not int or not 0 <= value < len(strings):
            raise ValueError("R13 codec string index is invalid")
        return strings[value]
    if annotation in (int, bool) and type(value) is annotation:
        return value
    raise ValueError("R13 codec scalar has wrong type")


def _decode_nodes(domain: str, nodes: list[list[JsonValue]], strings: tuple[str, ...]) -> object:
    if domain not in _ROOTS or not nodes or len(nodes) > sum(_COUNTS[domain].values()):
        raise ValueError("R13 codec domain or node bound is invalid")
    prior: list[object] = []
    counts: Counter[type[object]] = Counter()
    for row in nodes:
        if not isinstance(row, list) or len(row) != 3 or type(row[0]) is not str:
            raise ValueError("R13 codec node shape is invalid")
        value_type = _REGISTRY.get(row[0])
        if value_type is None or value_type not in _COUNTS[domain]:
            raise ValueError("R13 codec contains a foreign/future node type")
        counts[value_type] += 1
        if counts[value_type] > _COUNTS[domain][value_type]:
            raise ValueError("R13 codec node-type count exceeds its bound")
        init_values, computed_values = row[1], row[2]
        if not isinstance(init_values, list) or not isinstance(computed_values, list):
            raise ValueError("R13 codec field lists are invalid")
        value_fields = fields(value_type)
        init_fields = tuple(item for item in value_fields if item.init)
        computed_fields = tuple(item for item in value_fields if not item.init)
        if len(init_fields) != len(init_values) or len(computed_fields) != len(computed_values):
            raise ValueError("R13 codec node field count is invalid")
        hints = get_type_hints(value_type)
        kwargs = {item.name: _decode_scalar(value, hints[item.name], prior, strings)
                  for item, value in zip(init_fields, init_values, strict=True)}
        reconstructed = cast(Callable[..., object], value_type)(**kwargs)
        expected = [_decode_scalar(value, hints[item.name], prior, strings)
                    for item, value in zip(computed_fields, computed_values, strict=True)]
        actual = [getattr(reconstructed, item.name) for item in computed_fields]
        if expected != actual:
            raise ValueError("R13 codec computed digest/size witness was tampered")
        prior.append(reconstructed)
    root = prior[-1]
    if type(root) is not _ROOTS[domain] or _encode_nodes(root) != (nodes, list(strings)):
        raise ValueError("R13 codec table is noncanonical or not exactly reconstructible")
    return root


class _BoundedCodecModel(BaseModel):
    """Bound raw codec JSON before Pydantic materializes its table structures."""

    @classmethod
    def model_validate_json(
        cls, json_data: str | bytes | bytearray, *, strict: bool | None = None,
        extra: ExtraValues | None = None, context: Any | None = None,
        by_alias: bool | None = None, by_name: bool | None = None,
    ) -> Self:
        maxima = r13_codec_schema_maxima()
        limit = (max(maxima[name] for name in _ROOTS)
                 if cls is R13GraphSnapshot else maxima["total"])
        if len(json_data) > limit or (
            isinstance(json_data, str) and len(json_data.encode("utf-8")) > limit
        ):
            raise ValueError("R13 serialized codec exceeds its schema byte bound")
        return super().model_validate_json(
            json_data, strict=strict, extra=extra, context=context,
            by_alias=by_alias, by_name=by_name,
        )


def _require_bounded_closed_rows(value: object) -> object:
    if not isinstance(value, dict):
        return value
    rows = value.get("nodes")
    if not isinstance(rows, (list, tuple)) or len(rows) > _MAX_NODES:
        raise ValueError("R13 codec node table exceeds its bound")
    for row in rows:
        if (not isinstance(row, list) or len(row) != 3
                or type(row[0]) is not str or len(row[0]) > 64):
            raise ValueError("R13 codec node shape is invalid")
        for values, maximum in ((row[1], 32), (row[2], 4)):
            if not isinstance(values, list) or len(values) > maximum:
                raise ValueError("R13 codec field list exceeds its bound")
            for item in values:
                items = item if isinstance(item, list) else [item]
                if len(items) > 1024 or any(
                    type(scalar) not in (str, int, bool, type(None))
                    or (type(scalar) is str and len(scalar) > 64)
                    for scalar in items
                ):
                    raise ValueError("R13 codec field shape exceeds its bound")
    return value


class R13GraphSnapshot(_BoundedCodecModel):
    """Closed validated normalization of one immutable domain snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: int = Field(strict=True)
    domain: str = Field(max_length=10, strict=True)
    nodes: tuple[JsonValue, ...] = Field(max_length=_MAX_NODES)
    strings: tuple[_BoundedString, ...] = Field(max_length=120_000)

    @model_validator(mode="before")
    @classmethod
    def require_bounded_closed_rows(cls, value: object) -> object:
        return _require_bounded_closed_rows(value)

    @model_validator(mode="after")
    def validate_table(self) -> R13GraphSnapshot:
        if self.schema_version != R13_CODEC_SCHEMA_VERSION:
            raise ValueError("unsupported R13 codec version")
        self.restore()
        return self

    def restore(self) -> object:
        try:
            if self.schema_version != R13_CODEC_SCHEMA_VERSION:
                raise ValueError("unsupported R13 codec version")
            _require_bounded_closed_rows(self.model_dump(mode="python"))
            if len(self.strings) > 120_000 or any(
                type(item) is not str or len(item) > 1024 for item in self.strings
            ):
                raise ValueError("R13 codec string table exceeds its bound")
            if len(self.strings) != len(set(self.strings)):
                raise ValueError("duplicate R13 codec strings")
            return _decode_nodes(self.domain, cast(list[list[JsonValue]], list(self.nodes)),
                                 self.strings)
        except (ValueError, TypeError, OverflowError, KeyError, AttributeError, IndexError) as error:
            raise ValueError("invalid R13 persisted authority table") from error

    @classmethod
    def capture(cls, snapshot: object) -> R13GraphSnapshot:
        domain = next((key for key, value in _ROOTS.items() if type(snapshot) is value), None)
        if domain is None or not is_dataclass(snapshot):
            raise ValueError("capture requires an exact R13 domain snapshot")
        nodes, strings = _encode_nodes(snapshot)
        return cls(schema_version=R13_CODEC_SCHEMA_VERSION, domain=domain,
                   nodes=cast(tuple[JsonValue, ...], tuple(nodes)),
                   strings=tuple(strings))


class R13StateSnapshot(_BoundedCodecModel):
    """Complete persisted R13 section, with no optional/default-empty domains."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: int = Field(strict=True)
    motivation: R13GraphSnapshot
    goal: R13GraphSnapshot
    commitment: R13GraphSnapshot

    @model_validator(mode="after")
    def validate_domains(self) -> R13StateSnapshot:
        if self.schema_version != R13_CODEC_SCHEMA_VERSION:
            raise ValueError("unsupported R13 persisted state version")
        if (self.motivation.domain, self.goal.domain, self.commitment.domain) != (
            "motivation", "goal", "commitment"
        ):
            raise ValueError("R13 persisted domains are misplaced")
        return self

    @classmethod
    def empty(cls) -> R13StateSnapshot:
        return cls(
            schema_version=R13_CODEC_SCHEMA_VERSION,
            motivation=R13GraphSnapshot.capture(MotivationSystemSnapshot((), (), (), ())),
            goal=R13GraphSnapshot.capture(GoalSystemSnapshot()),
            commitment=R13GraphSnapshot.capture(CommitmentSystemSnapshot()),
        )


# Field bounds used for Gate 0, shared with executable table-limit validation.
_ARRAY_COUNTS: Final = {
    "scope": 32, "evidence_refs": 64, "origin_refs": 32,
    "conflict_refs": 32, "related_refs": 16, "dependencies": 32,
    "conflicts": 32, "outcome_evidence_refs": 32, "subject_transition_proofs": 9,
    "revision_history": 8, "related_goal_refs": 16, "desire_refs": 16,
    "source_evidence_refs": 32, "motivation_ids": 32, "through_evidence_refs": 64,
    "records": 32, "candidates": 32, "goal_proposal_witnesses": 32,
    "event_receipts": 1024, "evidence_ledger": 1024,
}
_TEXT_CHARS: Final = {"description": 1024, "subject": 1024,
                     "provider_name": 256, "model_id": 256}


def _max_scalar(annotation: object, name: str, max_index: int) -> object:
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Union, UnionType):
        choices = [_max_scalar(item, name, max_index) for item in args]
        return max(choices, key=lambda item: len(_json_bytes(item)))
    if annotation is type(None):
        return None
    if annotation in _TYPES:
        return max_index
    if origin is tuple:
        count = _ARRAY_COUNTS[name]
        return [_max_scalar(args[0], name, max_index)] * count
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return max((item.value for item in annotation), key=lambda item: len(_json_bytes(item)))
    if annotation is datetime:
        return "9999-12-31T23:59:59.999999Z"
    if annotation is float:
        return "-0x0.0000000000001p-1022"
    if annotation is bool:
        return False
    if annotation is int:
        return 2**63 - 1
    if annotation is str:
        return 119_999  # bounded interned-string index (including computed digests)
    raise ValueError("R13 sizing encountered an unbounded field")


def r13_codec_schema_maxima() -> dict[str, int]:
    """Derive conservative value-byte maxima from the actual node row schema.

    This is an upper bound, not a claim all scalar extremes can coexist in a
    valid snapshot. It does not use the process-local byte guards as maxima.
    """

    result: dict[str, int] = {}
    for domain, counts in _COUNTS.items():
        max_index = sum(counts.values()) - 1
        rows_bytes = 0
        strings_bytes = 0
        strings_count = 0
        for value_type, count in counts.items():
            hints = get_type_hints(value_type)
            init: list[object] = []
            computed: list[object] = []
            for item in fields(value_type):
                value = _max_scalar(hints[item.name], item.name, max_index)
                (init if item.init else computed).append(value)
                annotation = hints[item.name]
                if annotation is str or str in get_args(annotation):
                    text_chars = _TEXT_CHARS.get(item.name)
                    width = 6 * text_chars + 2 if text_chars is not None else 130
                    strings_bytes += count * width
                    strings_count += count
            if value_type is CommitmentRecord:
                strings_bytes += count * 32 * (6 * 256 + 2)
                strings_count += count * 32
            rows_bytes += count * len(_json_bytes([value_type.__name__, init, computed]))
        # Goal/Commitment revision and proof evidence lists reuse the exact
        # strings of typed refs plus already retained proof digests. Motivation
        # revisions/anchors use at most 1024 typed-reference SHA-256 witnesses.
        if domain == "motivation":
            strings_bytes += 1024 * 66
            strings_count += 1024
        if strings_count > 120_000:
            raise ValueError("R13 sizing exceeds its string-table index bound")
        envelope = len(_json_bytes({"schema_version": 1, "domain": domain,
                                    "nodes": [], "strings": []}))
        result[domain] = (envelope + rows_bytes + max(0, sum(counts.values()) - 1)
                          + strings_bytes + max(0, strings_count - 1))
    result["total"] = len(_json_bytes({"schema_version": 1, "motivation": None,
                                      "goal": None, "commitment": None})) - 12 + sum(result.values())
    return result
