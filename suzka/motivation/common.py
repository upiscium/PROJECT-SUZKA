"""Shared immutable primitives for the R13 contract boundary.

This module contains no authority, persistence, runtime, model, or scheduler
behavior.  The domain modules use these values for opaque references,
deadlines, canonical ordering, and bounded revision witnesses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
import re
from typing import Callable, Final, TypeVar

from suzka.identifiers import validate_identifier


class _ClosedStrEnum(str, Enum):
    """String enum base which rejects values outside its closed vocabulary."""


class R13ReferenceKind(_ClosedStrEnum):
    """Reference kinds understood by R13.

    Each domain module applies a narrower allow-list.  There is deliberately
    no free-form source-kind string in any R13 record.
    """

    EXPERIENCE = "experience"
    EMOTION = "emotion"
    VALUE = "value"
    BELIEF = "belief"
    MOTIVATION = "motivation"
    GOAL = "goal"
    COMMITMENT = "commitment"
    STATE = "state"
    EVENT = "event"
    USER_REQUEST = "user_request"
    OPERATOR_PROPOSAL = "operator_proposal"
    EXTERNAL_REQUEST = "external_request"
    SUBJECT = "subject"
    EXTERNAL_PARTY = "external_party"
    SYSTEM = "system"


R13_SCHEMA_VERSION: Final = 1
R13_MAX_EVENT_SEQUENCE: Final = 2**63 - 1
R13_MAX_REVISION: Final = 2**31 - 1
R13_MAX_IDENTIFIER_CODEPOINTS: Final = 128
R13_MAX_TEXT_CODEPOINTS: Final = 1_024
R13_MAX_SHORT_TEXT_CODEPOINTS: Final = 256
R13_MAX_SCOPE_ITEMS: Final = 32
R13_MAX_SCOPE_CODEPOINTS: Final = 256
R13_MAX_EVIDENCE_REFS: Final = 32
R13_MAX_RELATED_REFS: Final = 16
R13_MAX_DEPENDENCY_REFS: Final = 32
R13_MAX_CONFLICT_REFS: Final = 32
R13_MAX_REVISION_HISTORY: Final = 8
R13_MAX_REVISION_WITNESSES: Final = 64
R13_MAX_RECORDS_PER_DOMAIN: Final = 32
R13_MAX_CANDIDATES_PER_DOMAIN: Final = 32

R13_AGENT_STATE_CAP_BYTES: Final = 128 * 1024 * 1024
# Frozen R12 sizing evidence used only by U1's feasibility calculation.  U5
# still owns the final complete-v7 proof and may derive this from its schema.
R12_BELIEF_SCHEMA_MAX_BYTES: Final = 47_541_622

R13_REFERENCE_DOMAIN: Final = b"PROJECT-SUZKA:R13:REFERENCE:V1\0"
R13_MODEL_KEY_DOMAIN: Final = b"PROJECT-SUZKA:R10:CALIBRATION-MODEL:V1\0"
_DIGEST_PATTERN: Final = re.compile(r"[0-9a-f]{64}\Z")
_MODEL_KEY_PATTERN: Final = re.compile(r"model\.[0-9a-f]{64}\Z")


def _digest(value: object, name: str) -> str:
    if type(value) is not str or _DIGEST_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def validate_digest(value: object, name: str = "digest") -> str:
    """Validate and return a lowercase SHA-256 digest."""

    return _digest(value, name)


def validate_model_key(value: object) -> str:
    """Validate the opaque canonical model identity used by R12 evidence."""

    if type(value) is not str or _MODEL_KEY_PATTERN.fullmatch(value) is None:
        raise ValueError("model_key must be model.<64 lowercase hex characters>")
    return value


def _model_identity_part(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    if not value or value != value.strip() or any(
        character.isspace() or ord(character) < 32 or ord(character) == 127
        for character in value
    ):
        raise ValueError(f"{name} must be a bounded identity token")
    if len(value) > R13_MAX_SHORT_TEXT_CODEPOINTS:
        raise ValueError(f"{name} exceeds its code-point bound")
    return value


def model_identity_key(provider_name: str, model_id: str) -> str:
    """Derive the same opaque model key used by the R10 evidence contract."""

    provider = _model_identity_part(provider_name, "provider_name")
    model = _model_identity_part(model_id, "model_id")
    identity = json.dumps(
        [provider, model],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "model." + hashlib.sha256(R13_MODEL_KEY_DOMAIN + identity).hexdigest()


@dataclass(frozen=True, slots=True)
class MotivationModelIdentity:
    """Bounded canonical model identity witness for a candidate."""

    provider_name: str
    model_id: str
    model_key: str = field(init=False)

    def __post_init__(self) -> None:
        provider = _model_identity_part(self.provider_name, "provider_name")
        model = _model_identity_part(self.model_id, "model_id")
        object.__setattr__(self, "provider_name", provider)
        object.__setattr__(self, "model_id", model)
        object.__setattr__(self, "model_key", model_identity_key(provider, model))


def normalize_text(value: object, name: str, maximum: int) -> str:
    """Normalize bounded text without truncating it."""

    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    for character in value:
        codepoint = ord(character)
        if 0xD800 <= codepoint <= 0xDFFF:
            raise ValueError(f"{name} contains a non-Unicode scalar value")
        if codepoint == 0 or codepoint == 127 or (
            codepoint < 32 and character not in "\t\n\r"
        ):
            raise ValueError(f"{name} contains a prohibited control character")
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    if len(normalized) > maximum:
        raise ValueError(f"{name} exceeds its code-point bound")
    return normalized


def bounded_fraction(value: object, name: str) -> float:
    """Return a finite number in the closed interval [0, 1]."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def bounded_nonnegative_int(value: object, name: str, *, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be a bounded non-negative exact integer")
    return value


def bounded_positive_int(value: object, name: str, *, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be a positive bounded exact integer")
    return value


def canonical_json(value: object) -> bytes:
    """Encode a closed payload with deterministic ASCII JSON."""

    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def utc_datetime(value: object, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def canonical_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


@dataclass(frozen=True, slots=True)
class Deadline:
    """Explicit deadline/no-deadline value used by Goals and Commitments."""

    at: datetime | None = None

    def __post_init__(self) -> None:
        if self.at is not None:
            object.__setattr__(self, "at", utc_datetime(self.at, "deadline"))

    @classmethod
    def without_deadline(cls) -> Deadline:
        return cls()

    @property
    def is_set(self) -> bool:
        return self.at is not None

    def canonical_value(self) -> str | None:
        return canonical_datetime(self.at)


def coerce_deadline(value: object, name: str = "deadline") -> Deadline:
    if isinstance(value, Deadline):
        return value
    if value is None:
        return Deadline()
    if isinstance(value, datetime):
        return Deadline(value)
    raise TypeError(f"{name} must be a Deadline, datetime, or None")


@dataclass(frozen=True, slots=True)
class R13Reference:
    """Opaque, typed reference used for evidence, targets, and origins."""

    kind: R13ReferenceKind
    reference: str

    def __post_init__(self) -> None:
        if type(self.kind) is not R13ReferenceKind:
            raise TypeError("kind must be a R13ReferenceKind")
        object.__setattr__(self, "reference", validate_identifier(self.reference))

    @property
    def ref(self) -> str:
        return self.reference

    def canonical_value(self) -> dict[str, str]:
        return {"kind": self.kind.value, "reference": self.reference}


# Role aliases document intent without widening the vocabulary.
R13EvidenceReference = R13Reference
EvidenceReference = R13Reference
TargetReference = R13Reference
OriginReference = R13Reference


def canonical_references(
    value: object,
    name: str,
    *,
    maximum: int,
    allow_empty: bool = True,
    allowed_kinds: frozenset[R13ReferenceKind] | None = None,
) -> tuple[R13Reference, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    if not allow_empty and not value:
        raise ValueError(f"{name} must be non-empty")
    if len(value) > maximum:
        raise ValueError(f"{name} exceeds its bound")
    references: list[R13Reference] = []
    for item in value:
        if not isinstance(item, R13Reference):
            raise TypeError(f"{name} must contain R13Reference values")
        if allowed_kinds is not None and item.kind not in allowed_kinds:
            raise ValueError(f"{name} contains an unauthorized reference kind")
        references.append(item)
    if len({item.reference for item in references}) != len(references):
        raise ValueError(f"{name} references must be unique")
    ordered = tuple(sorted(references, key=lambda item: (item.reference, item.kind.value)))
    if references != list(ordered):
        raise ValueError(f"{name} must be canonically ordered")
    return ordered


def canonical_identifier_refs(
    value: object,
    name: str,
    *,
    maximum: int,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    if not allow_empty and not value:
        raise ValueError(f"{name} must be non-empty")
    if len(value) > maximum:
        raise ValueError(f"{name} exceeds its bound")
    refs = tuple(validate_identifier(item) for item in value)
    if refs != tuple(sorted(set(refs))):
        raise ValueError(f"{name} must be sorted and unique")
    return refs


def canonical_event(
    event_id: object,
    event_sequence: object,
    *,
    require: bool = True,
) -> tuple[str | None, int | None]:
    if require and (event_id is None or event_sequence is None):
        raise ValueError("event_id and event_sequence are required together")
    if not require and ((event_id is None) != (event_sequence is None)):
        raise ValueError("event_id and event_sequence must be supplied together")
    if event_id is None:
        return None, None
    return (
        validate_identifier(event_id),
        bounded_positive_int(
            event_sequence,
            "event_sequence",
            maximum=R13_MAX_EVENT_SEQUENCE,
        ),
    )


def canonical_revision_witnesses(
    value: object,
    name: str = "evidence_refs",
) -> tuple[str, ...]:
    return canonical_identifier_refs(
        value,
        name,
        maximum=R13_MAX_REVISION_WITNESSES,
        allow_empty=False,
    )


@dataclass(frozen=True, slots=True)
class RevisionCompactionAnchor:
    """Immutable digest witness for revisions before a retained suffix."""

    authority_id: str
    through_revision: int
    through_digest: str
    through_created_at: datetime
    through_evidence_refs: tuple[str, ...]
    through_previous_revision_digest: str | None
    through_state: str
    through_previous_state: str | None
    through_operation: str
    through_reason: str
    through_event_id: str
    through_event_sequence: int
    through_proposal_digest: str | None = None
    through_state_digest: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "authority_id", validate_identifier(self.authority_id))
        object.__setattr__(
            self,
            "through_revision",
            bounded_nonnegative_int(
                self.through_revision,
                "through_revision",
                maximum=R13_MAX_REVISION - 1,
            ),
        )
        if (self.through_revision == 0) != (
            self.through_previous_revision_digest is None
        ):
            raise ValueError(
                "genesis anchors must omit their previous revision digest"
            )
        if (self.through_revision == 0) != (self.through_previous_state is None):
            raise ValueError("genesis anchors must omit their previous lifecycle state")
        if self.through_previous_state is not None:
            object.__setattr__(
                self,
                "through_previous_state",
                _model_identity_part(
                    self.through_previous_state,
                    "through_previous_state",
                ),
            )
        if self.through_previous_revision_digest is not None:
            object.__setattr__(
                self,
                "through_previous_revision_digest",
                validate_digest(
                    self.through_previous_revision_digest,
                    "through_previous_revision_digest",
                ),
            )
        object.__setattr__(
            self,
            "through_digest",
            validate_digest(self.through_digest, "through_digest"),
        )
        object.__setattr__(
            self,
            "through_created_at",
            utc_datetime(self.through_created_at, "through_created_at"),
        )
        object.__setattr__(
            self,
            "through_evidence_refs",
            canonical_revision_witnesses(
                self.through_evidence_refs,
                "through_evidence_refs",
            ),
        )
        object.__setattr__(
            self,
            "through_state",
            _model_identity_part(self.through_state, "through_state"),
        )
        object.__setattr__(
            self,
            "through_operation",
            _model_identity_part(self.through_operation, "through_operation"),
        )
        object.__setattr__(
            self,
            "through_reason",
            _model_identity_part(self.through_reason, "through_reason"),
        )
        object.__setattr__(
            self,
            "through_event_id",
            validate_identifier(self.through_event_id),
        )
        object.__setattr__(
            self,
            "through_event_sequence",
            bounded_positive_int(
                self.through_event_sequence,
                "through_event_sequence",
                maximum=R13_MAX_EVENT_SEQUENCE,
            ),
        )
        if self.through_proposal_digest is not None:
            object.__setattr__(
                self,
                "through_proposal_digest",
                validate_digest(self.through_proposal_digest, "through_proposal_digest"),
            )
        if self.through_state_digest is not None:
            object.__setattr__(
                self,
                "through_state_digest",
                validate_digest(self.through_state_digest, "through_state_digest"),
            )

    @property
    def retained_from_revision(self) -> int:
        return self.through_revision + 1

    @property
    def digest(self) -> str:
        return self.through_digest


_RevisionT = TypeVar("_RevisionT")


def validate_revision_suffix(
    *,
    authority_id: str,
    current_revision: object,
    revision_history: tuple[_RevisionT, ...],
    history_anchor: RevisionCompactionAnchor | None,
    revision_digest: Callable[[_RevisionT], str],
    maximum_history: int = R13_MAX_REVISION_HISTORY,
) -> None:
    """Validate contiguous revision evidence without mutation.

    Event ID uniqueness inside the retained suffix is checked here. The
    EventJournal owns global event-ID uniqueness, including the compacted
    prefix, and remains the upstream authority for that contract.
    """

    authority_id = validate_identifier(authority_id)
    revision = bounded_nonnegative_int(
        current_revision,
        "revision",
        maximum=R13_MAX_REVISION,
    )
    if type(revision_history) is not tuple:
        raise TypeError("revision_history must be a tuple")
    if len(revision_history) > maximum_history:
        raise ValueError("revision_history exceeds its bound")
    if revision == 0 and len(revision_history) != 1:
        raise ValueError("revision zero requires its genesis evidence")
    if revision == 0 and history_anchor is not None:
        raise ValueError("revision zero cannot have a compaction anchor")
    if revision > 0 and not revision_history:
        raise ValueError("nonzero revision requires retained revision evidence")

    if history_anchor is not None:
        if not isinstance(history_anchor, RevisionCompactionAnchor):
            raise TypeError("history_anchor must be RevisionCompactionAnchor")
        if history_anchor.authority_id != authority_id:
            raise ValueError("history_anchor authority does not match the record")
        if len(revision_history) != maximum_history:
            raise ValueError("a compacted history requires its full bounded suffix")
        expected_first = history_anchor.retained_from_revision
        if expected_first < 1 or expected_first + len(revision_history) - 1 != revision:
            raise ValueError("history_anchor does not describe the retained suffix")
    elif revision >= maximum_history:
        raise ValueError("revision history requires a compaction anchor")

    previous: object | None = None
    previous_event_id: str | None = (
        history_anchor.through_event_id if history_anchor is not None else None
    )
    previous_event_sequence: int | None = (
        history_anchor.through_event_sequence if history_anchor is not None else None
    )
    previous_operation: str | None = (
        history_anchor.through_operation if history_anchor is not None else None
    )
    seen_event_ids = (
        {history_anchor.through_event_id} if history_anchor is not None else set()
    )
    for item in revision_history:
        item_id = getattr(item, "authority_id", None)
        item_revision = getattr(item, "revision", None)
        item_digest = getattr(item, "record_digest", None)
        item_previous = getattr(item, "previous_revision_digest", None)
        item_event_id = getattr(item, "event_id", None)
        item_event_sequence = getattr(item, "event_sequence", None)
        if item_id != authority_id:
            raise ValueError("revision_history cannot mix authority IDs")
        if type(item_revision) is not int or not 0 <= item_revision <= revision:
            raise ValueError("revision_history contains an invalid revision")
        if previous is None:
            if history_anchor is not None:
                if item_revision != history_anchor.retained_from_revision:
                    raise ValueError("retained suffix starts at the wrong revision")
                if item_previous != history_anchor.through_digest:
                    raise ValueError("retained suffix does not link to its anchor")
            elif item_revision != 0:
                raise ValueError("unanchored history must retain its genesis revision")
            elif item_previous is not None:
                raise ValueError("genesis revision cannot link to a prior digest")
        else:
            previous_revision = getattr(previous, "revision")
            previous_digest = getattr(previous, "record_digest")
            if item_revision != previous_revision + 1:
                raise ValueError("revision_history is non-monotonic")
            if item_previous != previous_digest:
                raise ValueError("revision_history has a broken digest link")
        validate_digest(item_digest, "record_digest")
        if item_previous is not None:
            validate_digest(item_previous, "previous_revision_digest")
        if item_event_id is None or item_event_sequence is None:
            raise ValueError("revision_history requires event witnesses")
        item_event_id = validate_identifier(item_event_id)
        item_event_sequence = bounded_positive_int(
            item_event_sequence,
            "event_sequence",
            maximum=R13_MAX_EVENT_SEQUENCE,
        )
        item_operation = getattr(getattr(item, "operation", None), "value", None)
        if type(item_operation) is not str:
            raise ValueError("revision_history contains an invalid operation")
        same_creation_admission_event = (
            item_event_sequence == previous_event_sequence
            and previous_operation == "create"
            and item_operation in {"adopt", "admit"}
            and item_event_id == previous_event_id
        )
        if item_event_id in seen_event_ids and not same_creation_admission_event:
            raise ValueError("revision event IDs cannot be reused")
        if previous_event_sequence is not None:
            if item_event_sequence < previous_event_sequence:
                raise ValueError("revision events must increase monotonically")
            if item_event_sequence == previous_event_sequence:
                if not same_creation_admission_event:
                    raise ValueError("revision events cannot reuse an event sequence")
            elif item_event_id == previous_event_id:
                raise ValueError("revision event IDs cannot be reused at new sequences")
        previous_event_id = item_event_id
        previous_event_sequence = item_event_sequence
        previous_operation = item_operation
        seen_event_ids.add(item_event_id)
        if item_digest != revision_digest(item):
            raise ValueError("revision record digest does not match its contents")
        previous = item

    if previous is not None and getattr(previous, "revision") != revision:
        raise ValueError("revision_history does not reach the current revision")


def digest_payload(domain: bytes, payload: object) -> str:
    return hashlib.sha256(domain + canonical_json(payload)).hexdigest()


def domain_digest(domain: bytes, payload: object) -> str:
    """Public spelling for deterministic domain-separated SHA-256 digests."""

    return digest_payload(domain, payload)


__all__ = [
    "Deadline",
    "EvidenceReference",
    "OriginReference",
    "R12_BELIEF_SCHEMA_MAX_BYTES",
    "R13_AGENT_STATE_CAP_BYTES",
    "R13_MAX_CONFLICT_REFS",
    "R13_MAX_CANDIDATES_PER_DOMAIN",
    "R13_MAX_DEPENDENCY_REFS",
    "R13_MAX_EVIDENCE_REFS",
    "R13_MAX_EVENT_SEQUENCE",
    "R13_MAX_IDENTIFIER_CODEPOINTS",
    "R13_MAX_RECORDS_PER_DOMAIN",
    "R13_MAX_RELATED_REFS",
    "R13_MAX_REVISION",
    "R13_MAX_REVISION_HISTORY",
    "R13_MAX_REVISION_WITNESSES",
    "R13_MAX_SCOPE_CODEPOINTS",
    "R13_MAX_SCOPE_ITEMS",
    "R13_MAX_SHORT_TEXT_CODEPOINTS",
    "R13_MAX_TEXT_CODEPOINTS",
    "R13_REFERENCE_DOMAIN",
    "R13_MODEL_KEY_DOMAIN",
    "R13_SCHEMA_VERSION",
    "R13EvidenceReference",
    "R13Reference",
    "R13ReferenceKind",
    "MotivationModelIdentity",
    "RevisionCompactionAnchor",
    "TargetReference",
    "bounded_fraction",
    "bounded_nonnegative_int",
    "bounded_positive_int",
    "canonical_datetime",
    "canonical_event",
    "canonical_identifier_refs",
    "canonical_json",
    "canonical_references",
    "canonical_revision_witnesses",
    "coerce_deadline",
    "digest_payload",
    "domain_digest",
    "normalize_text",
    "utc_datetime",
    "validate_digest",
    "validate_model_key",
    "model_identity_key",
    "validate_revision_suffix",
]
