"""Dependency-light immutable primitives for the R14 Attention boundary.

The Attention package describes references and evidence only.  It does not own
Working Memory, R13 authorities, a runtime loop, a model, or a clock.  In
particular, an ``AttentionEvent`` always receives its event identity and UTC
time from its caller.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
import hashlib
import json
import math
import re
from typing import Final, TypeVar, cast

from suzka.identifiers import validate_identifier
from suzka.limits import MAX_PERSISTED_EVENT_SEQUENCE, MAX_PERSISTED_REVISION
from suzka.motivation.common import R13_MAX_RECORDS_PER_DOMAIN
from suzka.working_memory_contracts import (
    MAX_PROJECTION_BYTES,
    MAX_SOURCE_ID_BYTES,
    WorkingMemorySourceKind,
)


class _ClosedStrEnum(str, Enum):
    """String enum base for closed, persisted vocabularies."""


class AttentionTargetKind(_ClosedStrEnum):
    WORKING_MEMORY = "working_memory"
    MOTIVATION = "motivation"
    GOAL = "goal"
    COMMITMENT = "commitment"


class AttentionSourceKind(_ClosedStrEnum):
    WORKING_MEMORY = "working_memory"
    MOTIVATION = "motivation"
    GOAL = "goal"
    COMMITMENT = "commitment"
    CONTEXT = "context"
    EMOTION = "emotion"
    APPRAISAL = "appraisal"
    VALUE = "value"
    BELIEF = "belief"
    EXPERIENCE = "experience"


class SourceDigestKind(_ClosedStrEnum):
    UPSTREAM_AUTHORITY = "upstream_authority"
    ATTENTION_PROJECTION = "attention_projection"


class CandidateAvailability(_ClosedStrEnum):
    ELIGIBLE = "eligible"
    UNAVAILABLE = "unavailable"
    INACTIVE = "inactive"


class AttentionRevisionReason(_ClosedStrEnum):
    STATE_UPDATE = "state_update"


ATTENTION_SCHEMA_VERSION: Final[int] = 1
ATTENTION_POLICY_VERSION: Final[int] = 1
ATTENTION_MAX_FOCUS: Final[int] = 16
ATTENTION_HIGH_AROUSAL_THRESHOLD: Final[float] = 0.75
ATTENTION_HIGH_AROUSAL_MAX_FOCUS: Final[int] = 8
ATTENTION_PROMPT_BUDGET_BYTES: Final[int] = 131_072
ATTENTION_MAX_REVISION_HISTORY: Final[int] = 16
ATTENTION_MAX_EVENT_RECEIPTS: Final[int] = 256
ATTENTION_MAX_REVISION: Final[int] = MAX_PERSISTED_REVISION
ATTENTION_MAX_EVENT_SEQUENCE: Final[int] = MAX_PERSISTED_EVENT_SEQUENCE
ATTENTION_MAX_R13_RECORDS_PER_DOMAIN: Final[int] = R13_MAX_RECORDS_PER_DOMAIN
ATTENTION_MAX_COUNTER: Final[int] = MAX_PERSISTED_REVISION
ATTENTION_FIXED_POINT_SCALE: Final[int] = 1_000_000

_DIGEST_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}\Z")
_WORKING_MEMORY_REFERENCE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"wm-[0-9a-f]{64}\Z"
)
_ATTENTION_TARGET_DOMAIN: Final[bytes] = b"PROJECT-SUZKA:R14:ATTENTION-TARGET:V1\0"
_EnumT = TypeVar("_EnumT", bound=Enum)


def canonical_json(value: object) -> bytes:
    """Encode contract values as deterministic, compact, ASCII JSON."""

    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


# A resolved R08 payload may exceed the final Attention prompt budget. Preserve
# its exact byte witness so policy can explicitly omit it, never truncate it.
# Six ASCII JSON escape bytes per original UTF-8 byte cover even control bytes;
# the remaining bytes come from the actual closed WM row metadata serializer.
ATTENTION_MAX_RENDERED_ITEM_BYTES: Final[int] = 6 * MAX_PROJECTION_BYTES + len(
    canonical_json(
        {
            "item_id": "wm-" + "f" * 64,
            "source_kind": max((kind.value for kind in WorkingMemorySourceKind), key=len),
            "source_id": "a" * MAX_SOURCE_ID_BYTES,
            "text": "",
        }
    )
)


def digest_payload(domain: bytes, value: object) -> str:
    """Return a domain-separated SHA-256 digest of a closed canonical value."""

    if type(domain) is not bytes or not domain:
        raise TypeError("digest domain must be non-empty bytes")
    return hashlib.sha256(domain + canonical_json(value)).hexdigest()


def validate_digest(value: object, name: str = "digest") -> str:
    if type(value) is not str or _DIGEST_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def validate_working_memory_reference(value: object) -> str:
    if type(value) is not str or _WORKING_MEMORY_REFERENCE_PATTERN.fullmatch(value) is None:
        raise ValueError("Working Memory reference must be wm- followed by 64 lowercase hex characters")
    return value


def validate_target_reference(kind: AttentionTargetKind, value: object) -> str:
    if type(kind) is not AttentionTargetKind:
        raise TypeError("kind must be an AttentionTargetKind")
    if kind is AttentionTargetKind.WORKING_MEMORY:
        return validate_working_memory_reference(value)
    return validate_digest(value, "R13 target reference")


def stable_target_id(kind: AttentionTargetKind, reference: str) -> str:
    """Derive identity from both the closed target kind and canonical reference."""

    checked_kind = _exact_enum(kind, AttentionTargetKind, "kind")
    checked_reference = validate_target_reference(checked_kind, reference)
    return digest_payload(
        _ATTENTION_TARGET_DOMAIN,
        {"kind": checked_kind.value, "reference": checked_reference},
    )


def exact_enum(value: object, enum_type: type[_EnumT], name: str) -> _EnumT:
    """Require the actual closed enum type (never an equivalent free-form str)."""

    return _exact_enum(value, enum_type, name)


def _exact_enum(value: object, enum_type: type[_EnumT], name: str) -> _EnumT:
    if type(value) is not enum_type:
        raise TypeError(f"{name} must be a {enum_type.__name__}")
    return cast(_EnumT, value)


def bounded_fraction(value: object, name: str) -> float:
    """Validate a finite [0, 1] signal, rejecting booleans and preserving zero."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return 0.0 if result == 0.0 else result


def bounded_counter(value: object, name: str, *, maximum: int = ATTENTION_MAX_COUNTER) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be a bounded non-negative exact integer")
    return value


def bounded_positive_int(
    value: object,
    name: str,
    *,
    maximum: int = ATTENTION_MAX_EVENT_SEQUENCE,
) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be a positive bounded exact integer")
    return value


def bounded_nonnegative_int(
    value: object,
    name: str,
    *,
    maximum: int = ATTENTION_MAX_REVISION,
) -> int:
    return bounded_counter(value, name, maximum=maximum)


def utc_datetime(value: object, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def canonical_datetime(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def parse_canonical_datetime(value: object, name: str) -> datetime:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact timestamp string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} must be a canonical UTC timestamp") from error
    checked = utc_datetime(parsed, name)
    if canonical_datetime(checked) != value:
        raise ValueError(f"{name} must use canonical UTC microsecond precision")
    return checked


def parse_hex_fraction(value: object, name: str) -> float | None:
    """Decode the canonical fixed-width float-hex scalar; null remains unknown."""

    if value is None:
        return None
    if type(value) is not str:
        raise TypeError(f"{name} must be a canonical float.hex() string or null")
    try:
        parsed = float.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{name} is not a canonical finite fraction") from error
    checked = bounded_fraction(parsed, name)
    if checked.hex() != value:
        raise ValueError(f"{name} is not canonically encoded")
    return checked


def parse_enum(value: object, enum_type: type[_EnumT], name: str) -> _EnumT:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    try:
        return enum_type(value)
    except ValueError as error:
        raise ValueError(f"{name} is outside its closed vocabulary") from error


def parse_identifier(value: object, name: str) -> str:
    try:
        return validate_identifier(value)
    except (TypeError, ValueError) as error:
        raise type(error)(f"{name} must be a canonical bounded identifier") from error


def attention_focus_capacity(global_emotion_arousal: object) -> int:
    """Return the focus resource ceiling without assigning arousal to candidates."""

    arousal = bounded_fraction(global_emotion_arousal, "global_emotion_arousal")
    if arousal >= ATTENTION_HIGH_AROUSAL_THRESHOLD:
        return ATTENTION_HIGH_AROUSAL_MAX_FOCUS
    return ATTENTION_MAX_FOCUS


def prompt_budget_accepts(rendered_bytes: object) -> bool:
    """Pure byte-budget predicate; row accounting remains a later policy concern."""

    if type(rendered_bytes) is not int or rendered_bytes < 0:
        raise ValueError("rendered_bytes must be a non-negative exact integer")
    return rendered_bytes <= ATTENTION_PROMPT_BUDGET_BYTES


__all__ = [
    "ATTENTION_HIGH_AROUSAL_MAX_FOCUS",
    "ATTENTION_HIGH_AROUSAL_THRESHOLD",
    "ATTENTION_MAX_COUNTER",
    "ATTENTION_MAX_EVENT_RECEIPTS",
    "ATTENTION_MAX_EVENT_SEQUENCE",
    "ATTENTION_MAX_FOCUS",
    "ATTENTION_MAX_REVISION",
    "ATTENTION_MAX_REVISION_HISTORY",
    "ATTENTION_MAX_R13_RECORDS_PER_DOMAIN",
    "ATTENTION_MAX_RENDERED_ITEM_BYTES",
    "ATTENTION_FIXED_POINT_SCALE",
    "ATTENTION_POLICY_VERSION",
    "ATTENTION_PROMPT_BUDGET_BYTES",
    "ATTENTION_SCHEMA_VERSION",
    "AttentionRevisionReason",
    "AttentionSourceKind",
    "AttentionTargetKind",
    "CandidateAvailability",
    "SourceDigestKind",
    "attention_focus_capacity",
    "bounded_counter",
    "bounded_fraction",
    "bounded_nonnegative_int",
    "bounded_positive_int",
    "canonical_datetime",
    "canonical_json",
    "digest_payload",
    "exact_enum",
    "parse_canonical_datetime",
    "parse_enum",
    "parse_hex_fraction",
    "parse_identifier",
    "prompt_budget_accepts",
    "stable_target_id",
    "utc_datetime",
    "validate_digest",
    "validate_target_reference",
    "validate_working_memory_reference",
]
