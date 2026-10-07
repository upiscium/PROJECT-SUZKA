"""Bounded codec helpers for the durable Attention continuity field.

This module deliberately depends only on the immutable Attention contracts. It
does not import AgentState, the WAL, or recovery code, so those persistence
owners can call the raw preflight before materializing their enclosing JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Final, cast

from suzka.attention.bounds import (
    attention_candidate_capacity_by_kind,
    derive_attention_schema_size_budget,
)
from suzka.attention.common import (
    ATTENTION_MAX_EVENT_RECEIPTS,
    ATTENTION_MAX_FOCUS,
    ATTENTION_MAX_REVISION_HISTORY,
    AttentionRevisionReason,
    AttentionSourceKind,
    AttentionTargetKind,
    CandidateAvailability,
    SourceDigestKind,
)
from suzka.attention.contracts import (
    AttentionCandidateContinuity,
    AttentionContinuity,
    AttentionEvent,
    AttentionEventReceipt,
    AttentionReceiptAnchor,
    AttentionRevisionAnchor,
    AttentionRevisionEvidence,
    AttentionSourceWitness,
    AttentionTarget,
    _ATTENTION_MAX_JSON_NESTING,
)


_ATTENTION_SIZE_BUDGET: Final = derive_attention_schema_size_budget()
AGENT_STATE_ATTENTION_MAX_SERIALIZED_BYTES: Final[int] = (
    _ATTENTION_SIZE_BUDGET.attention_state_max_bytes
)

_ROOT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "authority_digest",
        "candidates",
        "focused_ids",
        "last_event",
        "policy_version",
        "receipt_anchor",
        "receipts",
        "revision",
        "revision_anchor",
        "revision_history",
        "schema_version",
        "state_digest",
        "unfinished_ids",
    }
)
_ROOT_ARRAY_LIMITS: Final[dict[str, int]] = {
    "candidates": _ATTENTION_SIZE_BUDGET.candidate_count,
    "focused_ids": ATTENTION_MAX_FOCUS,
    "unfinished_ids": ATTENTION_MAX_FOCUS,
    "revision_history": ATTENTION_MAX_REVISION_HISTORY,
    "receipts": ATTENTION_MAX_EVENT_RECEIPTS,
}
_ROOT_VALUE_KINDS: Final[dict[str, frozenset[str]]] = {
    "authority_digest": frozenset({"string"}),
    "candidates": frozenset({"array"}),
    "focused_ids": frozenset({"array"}),
    "last_event": frozenset({"object", "null"}),
    "policy_version": frozenset({"integer"}),
    "receipt_anchor": frozenset({"object", "null"}),
    "receipts": frozenset({"array"}),
    "revision": frozenset({"integer"}),
    "revision_anchor": frozenset({"object", "null"}),
    "revision_history": frozenset({"array"}),
    "schema_version": frozenset({"integer"}),
    "state_digest": frozenset({"string"}),
    "unfinished_ids": frozenset({"array"}),
}
_SNAPSHOT_FIELDS: Final[frozenset[str]] = frozenset(
    {"baseline_snapshot", "candidate_snapshot"}
)
_MAX_RAW_JSON_NESTING: Final[int] = 4_096
_MAX_CANONICAL_TREE_NODES: Final[int] = 1_000_000
_MAX_CANONICAL_OBJECT_FIELDS: Final[int] = 16
_MAX_CANONICAL_ARRAY_ITEMS: Final[int] = 4_192
_DIGEST_LENGTH: Final[int] = 64
_IDENTIFIER_LENGTH: Final[int] = 128
_JSON_STRING_ESCAPES: Final[dict[int, str]] = {
    34: '"',
    47: "/",
    92: "\\",
    98: "\b",
    102: "\f",
    110: "\n",
    114: "\r",
    116: "\t",
}


def _bounded_ascii(value: object, *, maximum: int, field: str) -> None:
    if (
        type(value) is not str
        or len(value) > maximum
        or not value.isascii()
    ):
        raise ValueError(f"{field} is outside its Attention persistence bound")


def _bounded_exact_int(value: object, field: str) -> None:
    if type(value) is not int or value.bit_length() > 64:
        raise ValueError(f"{field} is outside its Attention persistence bound")


def _check_event(value: object) -> None:
    if type(value) is not AttentionEvent:
        raise TypeError("Attention events must be exact values")
    _bounded_ascii(value.event_id, maximum=_IDENTIFIER_LENGTH, field="event ID")
    _bounded_exact_int(value.event_sequence, "event sequence")
    if type(value.occurred_at) is not datetime:
        raise TypeError("Attention event timestamps must be exact datetimes")


def _check_source(value: object) -> None:
    if type(value) is not AttentionSourceWitness:
        raise TypeError("Attention sources must be exact values")
    if (
        type(value.kind) is not AttentionSourceKind
        or type(value.target_kind) is not AttentionTargetKind
        or type(value.digest_kind) is not SourceDigestKind
    ):
        raise TypeError("Attention source enums must be exact closed values")
    _bounded_ascii(value.reference, maximum=_IDENTIFIER_LENGTH, field="source reference")
    _bounded_ascii(
        value.target_reference,
        maximum=_IDENTIFIER_LENGTH,
        field="source target reference",
    )
    _bounded_ascii(value.digest, maximum=_DIGEST_LENGTH, field="source digest")
    _bounded_exact_int(value.revision, "source revision")
    if value.source_event_id is not None:
        _bounded_ascii(
            value.source_event_id,
            maximum=_IDENTIFIER_LENGTH,
            field="source event ID",
        )
    if value.source_event_sequence is not None:
        _bounded_exact_int(value.source_event_sequence, "source event sequence")
    if value.source_occurred_at is not None and type(value.source_occurred_at) is not datetime:
        raise TypeError("Attention source timestamps must be exact datetimes")


def _check_id_tuple(value: object, field: str) -> None:
    if type(value) is not tuple or len(value) > ATTENTION_MAX_FOCUS:
        raise ValueError(f"{field} exceeds its Attention persistence bound")
    for item in value:
        _bounded_ascii(item, maximum=_DIGEST_LENGTH, field=field)


def _preflight_typed(value: AttentionContinuity) -> None:
    if type(value) is not AttentionContinuity:
        raise TypeError("snapshot must be an exact AttentionContinuity")
    for name in ("revision", "schema_version", "policy_version"):
        _bounded_exact_int(getattr(value, name), name)
    collections = (
        (value.candidates, "candidates", 4_192),
        (value.focused_ids, "focused_ids", ATTENTION_MAX_FOCUS),
        (value.unfinished_ids, "unfinished_ids", ATTENTION_MAX_FOCUS),
        (value.revision_history, "revision_history", ATTENTION_MAX_REVISION_HISTORY),
        (value.receipts, "receipts", ATTENTION_MAX_EVENT_RECEIPTS),
    )
    for collection, name, maximum in collections:
        if type(collection) is not tuple or len(collection) > maximum:
            raise ValueError(f"{name} exceeds its Attention persistence bound")
    counts = {kind: 0 for kind in AttentionTargetKind}
    for candidate in value.candidates:
        if type(candidate) is not AttentionCandidateContinuity:
            raise TypeError("candidates must contain exact Attention continuity values")
        if type(candidate.target) is not AttentionTarget:
            raise TypeError("candidate targets must be exact AttentionTarget values")
        _bounded_ascii(
            candidate.target.reference,
            maximum=_IDENTIFIER_LENGTH,
            field="candidate reference",
        )
        if type(candidate.target.kind) is not AttentionTargetKind:
            raise TypeError("candidate kind must be an exact AttentionTargetKind")
        if type(candidate.availability) is not CandidateAvailability:
            raise TypeError("candidate availability must be an exact closed value")
        for name in (
            "habituation",
            "inhibition",
            "focused_event_count",
            "unattended_event_count",
        ):
            _bounded_exact_int(getattr(candidate, name), name)
        counts[candidate.target.kind] += 1
        _check_source(candidate.source)
        _bounded_exact_int(candidate.source.revision, "source revision")
        if candidate.source.source_event_sequence is not None:
            _bounded_exact_int(
                candidate.source.source_event_sequence,
                "source event sequence",
            )
        _bounded_ascii(candidate.record_digest, maximum=_DIGEST_LENGTH, field="candidate digest")
    per_kind = dict(attention_candidate_capacity_by_kind())
    if any(counts[kind] > per_kind[kind.value] for kind in counts):
        raise ValueError("candidate count exceeds its source-authority bound")
    _check_id_tuple(value.focused_ids, "focused_ids")
    _check_id_tuple(value.unfinished_ids, "unfinished_ids")
    if value.last_event is not None:
        _check_event(value.last_event)
        _bounded_exact_int(value.last_event.event_sequence, "event sequence")
    if value.revision_anchor is not None:
        if type(value.revision_anchor) is not AttentionRevisionAnchor:
            raise TypeError("revision_anchor must be an exact AttentionRevisionAnchor")
        _check_event(value.revision_anchor.through_event)
        _bounded_exact_int(value.revision_anchor.through_revision, "revision anchor")
        _bounded_ascii(
            value.revision_anchor.through_state_digest,
            maximum=_DIGEST_LENGTH,
            field="revision anchor digest",
        )
        _bounded_ascii(
            value.revision_anchor.through_revision_digest,
            maximum=_DIGEST_LENGTH,
            field="revision anchor witness",
        )
        _bounded_ascii(
            value.revision_anchor.anchor_digest,
            maximum=_DIGEST_LENGTH,
            field="revision anchor checksum",
        )
    if value.receipt_anchor is not None:
        if type(value.receipt_anchor) is not AttentionReceiptAnchor:
            raise TypeError("receipt_anchor must be an exact AttentionReceiptAnchor")
        _check_event(value.receipt_anchor.through_event)
        _bounded_ascii(
            value.receipt_anchor.through_result_state_digest,
            maximum=_DIGEST_LENGTH,
            field="receipt anchor state digest",
        )
        _bounded_ascii(
            value.receipt_anchor.through_receipt_digest,
            maximum=_DIGEST_LENGTH,
            field="receipt anchor witness",
        )
        _bounded_ascii(
            value.receipt_anchor.anchor_digest,
            maximum=_DIGEST_LENGTH,
            field="receipt anchor checksum",
        )
    for evidence in value.revision_history:
        if type(evidence) is not AttentionRevisionEvidence:
            raise TypeError("revision_history must contain exact evidence values")
        _check_event(evidence.event)
        _bounded_exact_int(evidence.revision, "history revision")
        _bounded_exact_int(evidence.event.event_sequence, "history event sequence")
        if type(evidence.reason) is not AttentionRevisionReason:
            raise TypeError("revision reason must be an exact closed value")
        _check_id_tuple(evidence.focused_ids, "revision focused_ids")
        _check_id_tuple(evidence.unfinished_ids, "revision unfinished_ids")
        for field_name in (
            "previous_state_digest",
            "state_digest",
            "record_digest",
        ):
            _bounded_ascii(
                getattr(evidence, field_name),
                maximum=_DIGEST_LENGTH,
                field=field_name,
            )
        if evidence.previous_revision_digest is not None:
            _bounded_ascii(
                evidence.previous_revision_digest,
                maximum=_DIGEST_LENGTH,
                field="previous revision digest",
            )
    for receipt in value.receipts:
        if type(receipt) is not AttentionEventReceipt:
            raise TypeError("receipts must contain exact AttentionEventReceipt values")
        _check_event(receipt.event)
        _bounded_exact_int(receipt.event.event_sequence, "receipt event sequence")
        for field_name in (
            "input_digest",
            "result_state_digest",
            "receipt_digest",
        ):
            _bounded_ascii(
                getattr(receipt, field_name),
                maximum=_DIGEST_LENGTH,
                field=field_name,
            )
        if receipt.previous_receipt_digest is not None:
            _bounded_ascii(
                receipt.previous_receipt_digest,
                maximum=_DIGEST_LENGTH,
                field="previous receipt digest",
            )


def _preflight_mapping(value: dict[str, object]) -> None:
    if type(value) is not dict:
        raise TypeError("Attention canonical value must be an exact dictionary")
    if len(value) != len(_ROOT_FIELDS) or any(
        type(key) is not str or len(key) > _IDENTIFIER_LENGTH or not key.isascii()
        for key in value
    ) or set(value) != _ROOT_FIELDS:
        raise ValueError("Attention root has missing or unknown fields")
    for name, kinds in _ROOT_VALUE_KINDS.items():
        item = value[name]
        kind = (
            "null"
            if item is None
            else "array"
            if type(item) is list
            else "object"
            if type(item) is dict
            else "string"
            if type(item) is str
            else "integer"
            if type(item) is int
            else "other"
        )
        if kind not in kinds:
            raise TypeError(f"Attention root field {name} has the wrong literal type")
    for name, maximum in _ROOT_ARRAY_LIMITS.items():
        items = value[name]
        if type(items) is not list or len(items) > maximum:
            raise ValueError(f"{name} exceeds its Attention persistence bound")

    candidates = cast(list[object], value["candidates"])
    counts = {name: 0 for name, _ in attention_candidate_capacity_by_kind()}
    for candidate in candidates:
        if type(candidate) is not dict:
            continue
        target = candidate.get("target")
        if type(target) is not dict:
            continue
        candidate_kind: object = target.get("kind")
        if type(candidate_kind) is str and candidate_kind in counts:
            counts[candidate_kind] += 1
    if any(counts[name] > limit for name, limit in attention_candidate_capacity_by_kind()):
        raise ValueError("candidate count exceeds its source-authority bound")

    pending: list[tuple[object, int, bool]] = [(value, 0, False)]
    sizes: dict[int, int] = {}
    seen_containers: set[int] = set()
    nodes = 0
    while pending:
        item, depth, expanded = pending.pop()
        if depth > _ATTENTION_MAX_JSON_NESTING:
            raise ValueError("Attention canonical value exceeds its structural bound")
        if type(item) is str:
            _bounded_ascii(item, maximum=_IDENTIFIER_LENGTH, field="Attention string")
            nodes += 1
            size = _canonical_ascii_string_size(item)
            if size > AGENT_STATE_ATTENTION_MAX_SERIALIZED_BYTES:
                raise ValueError("Attention canonical value exceeds its field byte bound")
            sizes[id(item)] = size
        elif type(item) is dict:
            if len(item) > _MAX_CANONICAL_OBJECT_FIELDS:
                raise ValueError("Attention object has too many fields")
            identity = id(item)
            if not expanded:
                if identity in seen_containers:
                    raise ValueError("Attention canonical value is not a JSON tree")
                seen_containers.add(identity)
                nodes += 1
                pending.append((item, depth, True))
                for key, child in item.items():
                    _bounded_ascii(key, maximum=_IDENTIFIER_LENGTH, field="Attention key")
                    pending.append((child, depth + 1, False))
                    pending.append((key, depth + 1, False))
            else:
                size = 2 + max(0, len(item) - 1)
                for key, child in item.items():
                    size += sizes[id(key)] + 1 + sizes[id(child)]
                sizes[identity] = size
        elif type(item) is list:
            if len(item) > _MAX_CANONICAL_ARRAY_ITEMS:
                raise ValueError("Attention array exceeds its structural bound")
            identity = id(item)
            if not expanded:
                if identity in seen_containers:
                    raise ValueError("Attention canonical value is not a JSON tree")
                seen_containers.add(identity)
                nodes += 1
                pending.append((item, depth, True))
                pending.extend((child, depth + 1, False) for child in item)
            else:
                size = 2 + max(0, len(item) - 1)
                size += sum(sizes[id(child)] for child in item)
                sizes[identity] = size
        elif type(item) is int:
            if item.bit_length() > 64:
                raise ValueError("Attention integer exceeds its structural bound")
            nodes += 1
            sizes[id(item)] = len(str(item))
        elif item is not None:
            raise TypeError("Attention canonical value contains an unsupported literal")
        else:
            nodes += 1
            sizes[id(item)] = 4
        if nodes > _MAX_CANONICAL_TREE_NODES:
            raise ValueError("Attention canonical value exceeds its structural bound")
        if expanded:
            size = sizes[id(item)]
            if size > AGENT_STATE_ATTENTION_MAX_SERIALIZED_BYTES:
                raise ValueError("Attention canonical value exceeds its field byte bound")
    if sizes[id(value)] > AGENT_STATE_ATTENTION_MAX_SERIALIZED_BYTES:
        raise ValueError("Attention canonical value exceeds its field byte bound")


def _canonical_ascii_string_size(value: str) -> int:
    size = len(value) + 2
    for character in value:
        codepoint = ord(character)
        if character in {'"', "\\"} or character in "\b\f\n\r\t":
            size += 1
        elif codepoint < 32:
            size += 5
    return size


def validated_attention_continuity(value: object) -> AttentionContinuity:
    """Return a detached U1 value after resource preflight and checksum checks."""

    if type(value) is AttentionContinuity:
        _preflight_typed(value)
        canonical = value.canonical_value()
    elif type(value) is dict:
        _preflight_mapping(value)
        canonical = value
    else:
        raise TypeError("Attention state must be an exact continuity value or canonical dict")
    result = AttentionContinuity.from_canonical_value(canonical)
    if len(result.canonical_bytes()) > AGENT_STATE_ATTENTION_MAX_SERIALIZED_BYTES:
        raise ValueError("Attention continuity exceeds its AgentState field byte bound")
    return result


def _char(data: bytes | str, index: int) -> int:
    if isinstance(data, bytes):
        return data[index]
    return ord(data[index])


def _is_digit(value: int) -> bool:
    return 48 <= value <= 57


@dataclass(slots=True)
class _JSONFrame:
    kind: str
    state: str
    depth: int
    start: int
    parent: _JSONFrame | None
    parent_key: str | None
    snapshot: bool = False
    attention_root: bool = False
    in_attention: bool = False
    attention_depth: int = 0
    current_key: str | None = None
    keys: set[str] | None = None
    version_count: int = 0
    version_nine: bool = False
    attention_count: int = 0
    attention_span: tuple[int, int] | None = None
    baseline_snapshot_count: int = 0
    candidate_snapshot_count: int = 0
    baseline_snapshot_has_v9: bool = False
    candidate_snapshot_has_v9: bool = False
    array_limit: int = _MAX_CANONICAL_ARRAY_ITEMS
    array_count: int = 0


class _RawJSONScanner:
    """Finite-stack JSON syntax scanner with optional Attention root checks."""

    def __init__(
        self,
        data: bytes | str,
        *,
        attention_document: bool = False,
    ) -> None:
        self.data = data
        self.length = len(data)
        self.attention_document = attention_document

    def scan(self) -> None:
        index = self._whitespace(0)
        stack: list[_JSONFrame] = []
        index, root_frame = self._value(index, None, None, stack)
        if root_frame is not None:
            stack.append(root_frame)
        while stack:
            frame = stack[-1]
            index = self._whitespace(index)
            char = _char(self.data, index) if index < self.length else -1
            if frame.kind == "object":
                if frame.state in {"key_or_end", "key_after_comma"}:
                    if char == 125 and frame.state == "key_or_end":
                        index = self._close_container(index, stack)
                        continue
                    if char != 34:
                        raise ValueError("JSON object key is malformed")
                    key, index = self._string(index, in_attention=frame.in_attention)
                    frame.current_key = key
                    if frame.in_attention:
                        if key is None:
                            raise ValueError("Attention JSON keys must be bounded ASCII")
                        assert frame.keys is not None
                        if key in frame.keys:
                            raise ValueError("Attention JSON contains a duplicate object key")
                        frame.keys.add(key)
                        if len(frame.keys) > _MAX_CANONICAL_OBJECT_FIELDS:
                            raise ValueError("Attention object has too many fields")
                    if frame.snapshot and key == "schema_version":
                        frame.version_count += 1
                    if frame.snapshot and key == "attention_state":
                        frame.attention_count += 1
                    if frame.depth == 1 and key == "baseline_snapshot":
                        frame.baseline_snapshot_count += 1
                    if frame.depth == 1 and key == "candidate_snapshot":
                        frame.candidate_snapshot_count += 1
                    if frame.attention_root:
                        if key is None or key not in _ROOT_FIELDS:
                            raise ValueError("Attention root has an unknown field")
                        assert frame.keys is not None
                    frame.state = "colon"
                    continue
                if frame.state == "colon":
                    if char != 58:
                        raise ValueError("JSON object is missing a colon")
                    frame.state = "value"
                    index += 1
                    continue
                if frame.state == "value":
                    key = frame.current_key
                    start = index
                    if frame.snapshot and key == "attention_state" and frame.attention_span is None:
                        frame.attention_span = (start, -1)
                    index, child = self._value(index, frame, key, stack)
                    frame.state = "comma_or_end"
                    if child is not None:
                        stack.append(child)
                    continue
                if frame.state == "comma_or_end":
                    if char == 44:
                        frame.state = "key_after_comma"
                        index += 1
                        continue
                    if char == 125:
                        index = self._close_container(index, stack)
                        continue
                    raise ValueError("JSON object separator is malformed")
                raise AssertionError("unknown object scanner state")

            if frame.state in {"value_or_end", "value_after_comma"}:
                if char == 93 and frame.state == "value_or_end":
                    index = self._close_container(index, stack)
                    continue
                frame.array_count += 1
                if frame.array_count > frame.array_limit:
                    raise ValueError("Attention JSON array exceeds its preflight bound")
                key = str(frame.array_count - 1)
                index, child = self._value(index, frame, key, stack)
                frame.state = "comma_or_end"
                if child is not None:
                    stack.append(child)
                continue
            if frame.state == "comma_or_end":
                if char == 44:
                    frame.state = "value_after_comma"
                    index += 1
                    continue
                if char == 93:
                    index = self._close_container(index, stack)
                    continue
                raise ValueError("JSON array separator is malformed")
            raise AssertionError("unknown array scanner state")
        index = self._whitespace(index)
        if index != self.length:
            raise ValueError("JSON contains trailing data")

    def _whitespace(self, index: int) -> int:
        while index < self.length and _char(self.data, index) in (9, 10, 13, 32):
            index += 1
        return index

    def _string(self, start: int, *, in_attention: bool) -> tuple[str | None, int]:
        index = start + 1
        chars: list[str] = []
        count = 0
        too_long = False
        non_ascii = False
        while index < self.length:
            char = _char(self.data, index)
            index += 1
            if char == 34:
                if in_attention and (too_long or non_ascii):
                    raise ValueError("Attention JSON strings must be bounded ASCII")
                return (None if too_long or non_ascii else "".join(chars), index)
            if char < 32:
                raise ValueError("JSON string contains a control character")
            if char == 92:
                if index >= self.length:
                    raise ValueError("JSON string has a truncated escape")
                escape = _char(self.data, index)
                index += 1
                if escape in _JSON_STRING_ESCAPES:
                    codepoint = ord(_JSON_STRING_ESCAPES[escape])
                elif escape == 117:
                    if index + 4 > self.length:
                        raise ValueError("JSON string has a truncated Unicode escape")
                    digits = []
                    for offset in range(index, index + 4):
                        digit = _char(self.data, offset)
                        if not (
                            48 <= digit <= 57
                            or 65 <= digit <= 70
                            or 97 <= digit <= 102
                        ):
                            raise ValueError("JSON string has an invalid Unicode escape")
                        digits.append(chr(digit))
                    codepoint = int("".join(digits), 16)
                    index += 4
                else:
                    raise ValueError("JSON string has an invalid escape")
            else:
                codepoint = char
                if char >= 128 and type(self.data) is bytes:
                    codepoint, index = self._utf8_codepoint(index - 1)
            count += 1
            if codepoint > 127:
                non_ascii = True
            if count <= _IDENTIFIER_LENGTH:
                chars.append(chr(codepoint) if codepoint <= 0xFFFF else "?")
            else:
                too_long = True

        raise ValueError("JSON string is unterminated")

    def _utf8_codepoint(self, start: int) -> tuple[int, int]:
        lead = _char(self.data, start)
        if 0xC2 <= lead <= 0xDF:
            width = 2
            codepoint = lead & 0x1F
            minimum = 0x80
        elif 0xE0 <= lead <= 0xEF:
            width = 3
            codepoint = lead & 0x0F
            minimum = 0x800
        elif 0xF0 <= lead <= 0xF4:
            width = 4
            codepoint = lead & 0x07
            minimum = 0x10000
        else:
            raise ValueError("JSON string contains invalid UTF-8")
        if start + width > self.length:
            raise ValueError("JSON string contains truncated UTF-8")
        for offset in range(1, width):
            continuation = _char(self.data, start + offset)
            if not 0x80 <= continuation <= 0xBF:
                raise ValueError("JSON string contains invalid UTF-8")
            codepoint = (codepoint << 6) | (continuation & 0x3F)
        if (
            codepoint < minimum
            or 0xD800 <= codepoint <= 0xDFFF
            or codepoint > 0x10FFFF
        ):
            raise ValueError("JSON string contains invalid UTF-8")
        return codepoint, start + width

    def _number(self, index: int) -> tuple[int, str]:
        if _char(self.data, index) == 45:
            index += 1
            if index >= self.length:
                raise ValueError("JSON number is malformed")
        char = _char(self.data, index)
        if char == 48:
            index += 1
            if index < self.length and _is_digit(_char(self.data, index)):
                raise ValueError("JSON number has a leading zero")
        elif 49 <= char <= 57:
            while index < self.length and _is_digit(_char(self.data, index)):
                index += 1
        else:
            raise ValueError("JSON number is malformed")
        integer = True
        if index < self.length and _char(self.data, index) == 46:
            integer = False
            index += 1
            if index >= self.length or not _is_digit(_char(self.data, index)):
                raise ValueError("JSON fraction is malformed")
            while index < self.length and _is_digit(_char(self.data, index)):
                index += 1
        if index < self.length and _char(self.data, index) in (69, 101):
            integer = False
            index += 1
            if index < self.length and _char(self.data, index) in (43, 45):
                index += 1
            if index >= self.length or not _is_digit(_char(self.data, index)):
                raise ValueError("JSON exponent is malformed")
            while index < self.length and _is_digit(_char(self.data, index)):
                index += 1
        return index, "integer" if integer else "number"

    def _value(
        self,
        index: int,
        parent: _JSONFrame | None,
        key: str | None,
        stack: list[_JSONFrame],
    ) -> tuple[int, _JSONFrame | None]:
        if index >= self.length:
            raise ValueError("JSON value is missing")
        char = _char(self.data, index)
        if char == 123 or char == 91:
            if len(stack) >= _MAX_RAW_JSON_NESTING:
                raise ValueError("JSON nesting exceeds its scanner bound")
            is_attention_root = self.attention_document and parent is None
            is_snapshot = parent is None or (
                parent.depth == 1 and key in _SNAPSHOT_FIELDS
            )
            in_attention = is_attention_root or (
                parent is not None and parent.in_attention
            )
            attention_depth = (
                (parent.attention_depth if parent is not None else 0) + 1
                if in_attention
                else 0
            )
            if in_attention and attention_depth > _ATTENTION_MAX_JSON_NESTING:
                raise ValueError("Attention JSON exceeds its nesting bound")
            is_object = char == 123
            frame = _JSONFrame(
                kind="object" if is_object else "array",
                state="key_or_end" if is_object else "value_or_end",
                depth=len(stack) + 1,
                start=index,
                parent=parent,
                parent_key=key,
                snapshot=is_snapshot,
                attention_root=is_attention_root,
                in_attention=in_attention,
                attention_depth=attention_depth,
                keys=set() if in_attention else None,
            )
            if not is_object:
                frame.array_limit = _array_limit(
                    parent,
                    key,
                    in_attention,
                    self.length,
                )
            if parent is not None and parent.attention_root and key is not None:
                allowed = _ROOT_VALUE_KINDS.get(key)
                if allowed is None:
                    raise ValueError("Attention root has an unknown field")
                if frame.kind not in allowed:
                    raise TypeError(f"Attention root field {key} has the wrong literal type")
            return index + 1, frame
        if char == 34:
            _, end = self._string(index, in_attention=parent is not None and parent.in_attention)
            kind = "string"
        elif char == 116:
            end = self._literal(index, b"true")
            kind = "boolean"
        elif char == 102:
            end = self._literal(index, b"false")
            kind = "boolean"
        elif char == 110:
            end = self._literal(index, b"null")
            kind = "null"
        elif char == 45 or _is_digit(char):
            end, kind = self._number(index)
        else:
            raise ValueError("JSON value is malformed")
        if parent is not None and parent.in_attention:
            if kind == "boolean":
                raise TypeError("Attention JSON does not contain boolean literals")
            if kind == "number":
                raise TypeError("Attention JSON integer fields cannot be floats")
            if kind == "integer" and (
                _char(self.data, index) == 45 or end - index > 19
            ):
                raise ValueError("Attention JSON integer exceeds its literal bound")
        self._finish_value(parent, key, index, end, kind)
        return end, None

    def _literal(self, index: int, literal: bytes) -> int:
        for offset, expected in enumerate(literal):
            if index + offset >= self.length or _char(self.data, index + offset) != expected:
                raise ValueError("JSON literal is malformed")
        return index + len(literal)

    def _close_container(self, index: int, stack: list[_JSONFrame]) -> int:
        frame = stack.pop()
        end = index + 1
        kind = frame.kind
        if frame.depth == 1 and (
            frame.baseline_snapshot_count > 1 and frame.baseline_snapshot_has_v9
        ):
            raise ValueError("WAL has duplicate v9 baseline_snapshot envelope keys")
        if frame.depth == 1 and (
            frame.candidate_snapshot_count > 1 and frame.candidate_snapshot_has_v9
        ):
            raise ValueError("WAL has duplicate v9 candidate_snapshot envelope keys")
        if frame.attention_root:
            assert frame.keys is not None
            if frame.keys != _ROOT_FIELDS:
                raise ValueError("Attention root has missing or unknown fields")
        if frame.snapshot and frame.version_nine:
            if frame.version_count != 1:
                raise ValueError("AgentState v9 has duplicate schema_version keys")
            if frame.attention_count != 1 or frame.attention_span is None:
                raise ValueError("AgentState v9 requires exactly one attention_state field")
            start, attention_end = frame.attention_span
            if attention_end < 0:
                raise ValueError("AgentState v9 attention_state value is incomplete")
            raw_size = _raw_size(self.data, start, attention_end)
            if raw_size > AGENT_STATE_ATTENTION_MAX_SERIALIZED_BYTES:
                raise ValueError("AgentState attention_state exceeds its field byte bound")
            _preflight_attention_value(self.data[start:attention_end])
            if frame.parent is not None and frame.parent.depth == 1:
                if frame.parent_key == "baseline_snapshot":
                    frame.parent.baseline_snapshot_has_v9 = True
                elif frame.parent_key == "candidate_snapshot":
                    frame.parent.candidate_snapshot_has_v9 = True
        self._finish_value(frame.parent, frame.parent_key, frame.start, end, kind)
        return end

    def _finish_value(
        self,
        parent: _JSONFrame | None,
        key: str | None,
        start: int,
        end: int,
        kind: str,
    ) -> None:
        if parent is None or key is None:
            return
        if parent.snapshot and key == "schema_version":
            if kind == "integer" and _number_is_nine(self.data, start, end):
                parent.version_nine = True
            elif kind == "number" and _number_decodes_to_nine(
                self.data, start, end
            ):
                raise ValueError(
                    "AgentState v9 schema_version must be the exact integer literal 9"
                )
        if parent.snapshot and key == "attention_state" and parent.attention_span is not None:
            if parent.attention_span[0] == start:
                parent.attention_span = (start, end)
        if parent.attention_root:
            allowed = _ROOT_VALUE_KINDS[key]
            if kind not in allowed:
                raise TypeError(f"Attention root field {key} has the wrong literal type")
            if key in {"schema_version", "policy_version"}:
                token = self.data[start:end]
                if token != (b"1" if type(token) is bytes else "1"):
                    raise ValueError(f"Attention {key} must use its exact integer literal")
            if key == "revision" and kind != "integer":
                raise TypeError("Attention revision must be an exact JSON integer")


def _array_limit(
    parent: _JSONFrame | None,
    key: str | None,
    in_attention: bool,
    source_length: int,
) -> int:
    if not in_attention:
        return source_length
    if parent is not None and parent.attention_root and key in _ROOT_ARRAY_LIMITS:
        return _ROOT_ARRAY_LIMITS[key]
    if key in {"focused_ids", "unfinished_ids"}:
        return ATTENTION_MAX_FOCUS
    if key == "revision_history":
        return ATTENTION_MAX_REVISION_HISTORY
    if key == "receipts":
        return ATTENTION_MAX_EVENT_RECEIPTS
    return _MAX_CANONICAL_ARRAY_ITEMS


def _number_is_nine(data: bytes | str, start: int, end: int) -> bool:
    index = start
    if _char(data, index) == 45:
        return False
    fraction_digits = 0
    exponent = 0
    exponent_negative = False
    in_fraction = False
    in_exponent = False
    nonzero_digit = 0
    trailing_zero_digits = 0
    limit = end - start + 1
    while index < end:
        char = _char(data, index)
        if char == 46:
            in_fraction = True
        elif char in (69, 101):
            in_exponent = True
            if index + 1 < end and _char(data, index + 1) in (43, 45):
                exponent_negative = _char(data, index + 1) == 45
                index += 1
        elif in_exponent:
            exponent = min(limit, exponent * 10 + char - 48)
        elif _is_digit(char):
            if in_fraction:
                fraction_digits += 1
            if char != 48:
                if nonzero_digit != 0 or char != 57:
                    return False
                nonzero_digit = 57
                trailing_zero_digits = 0
            elif nonzero_digit != 0:
                trailing_zero_digits += 1
        index += 1
    if nonzero_digit != 57:
        return False
    if exponent_negative:
        exponent = -exponent
    return exponent - fraction_digits + trailing_zero_digits == 0


def _number_decodes_to_nine(data: bytes | str, start: int, end: int) -> bool:
    """Mirror CPython's JSON float conversion for version discriminator tokens."""

    try:
        return float(data[start:end]) == 9.0
    except (OverflowError, ValueError):
        return False


def _raw_size(data: bytes | str, start: int, end: int) -> int:
    if isinstance(data, bytes):
        return end - start
    for index in range(start, end):
        if ord(data[index]) > 127:
            raise ValueError("Attention JSON must be ASCII-only")
    return end - start


def _preflight_attention_value(value: bytes | str) -> None:
    _RawJSONScanner(value, attention_document=True).scan()
    if type(value) is bytes:
        AttentionContinuity.from_json(value)
    elif value.isascii():
        AttentionContinuity.from_json(value)
    else:
        raise ValueError("Attention JSON must be ASCII-only")


def preflight_attention_json(payload: bytes | str) -> None:
    """Preflight v9 AgentState roots and snapshots embedded in WAL records.

    The scanner recognizes JSON object keys (including escaped spellings), not
    quoted substrings. Legacy schemas retain their existing decoder and
    duplicate-key behavior; exact Attention parsing is applied only to v9.
    """

    if type(payload) not in (bytes, str):
        raise TypeError("JSON preflight input must be exact bytes or str")
    _RawJSONScanner(payload).scan()


__all__ = [
    "AGENT_STATE_ATTENTION_MAX_SERIALIZED_BYTES",
    "preflight_attention_json",
    "validated_attention_continuity",
]
