"""Pure, immutable R14 Attention projections and continuity evidence.

This module stores bounded identities, source witnesses, fixed-point continuity
counters, and checksummed event/history evidence.  It has no mutation authority,
policy replay, runtime integration, scheduling, model access, or clock access.
Projection hashes are deterministic checksums, not producer authentication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from typing import Final

from suzka.attention.common import (
    ATTENTION_MAX_EVENT_RECEIPTS,
    ATTENTION_MAX_EVENT_SEQUENCE,
    ATTENTION_MAX_FOCUS,
    ATTENTION_MAX_R13_RECORDS_PER_DOMAIN,
    ATTENTION_MAX_RENDERED_ITEM_BYTES,
    ATTENTION_MAX_REVISION,
    ATTENTION_MAX_REVISION_HISTORY,
    ATTENTION_FIXED_POINT_SCALE,
    ATTENTION_POLICY_VERSION,
    ATTENTION_SCHEMA_VERSION,
    AttentionRevisionReason,
    AttentionSourceKind,
    AttentionTargetKind,
    CandidateAvailability,
    SourceDigestKind,
    bounded_counter,
    bounded_fraction,
    bounded_nonnegative_int,
    bounded_positive_int,
    canonical_datetime,
    canonical_json,
    digest_payload,
    exact_enum,
    parse_canonical_datetime,
    parse_enum,
    parse_hex_fraction,
    parse_identifier,
    stable_target_id,
    utc_datetime,
    validate_digest,
    validate_target_reference,
)


_ATTENTION_STATE_DOMAIN: Final[bytes] = b"PROJECT-SUZKA:R14:ATTENTION-STATE:V1\0"
_ATTENTION_MAX_CANONICAL_VALUE_BYTES: Final[int] = 7_866_805
_ATTENTION_MAX_JSON_NESTING: Final[int] = 64


class _AttentionJSONInputError(ValueError):
    """Stable validation failure raised by the bounded JSON decoder hooks."""
_SOURCE_KIND_TO_TARGET: Final[dict[AttentionSourceKind, AttentionTargetKind]] = {
    AttentionSourceKind.WORKING_MEMORY: AttentionTargetKind.WORKING_MEMORY,
    AttentionSourceKind.MOTIVATION: AttentionTargetKind.MOTIVATION,
    AttentionSourceKind.GOAL: AttentionTargetKind.GOAL,
    AttentionSourceKind.COMMITMENT: AttentionTargetKind.COMMITMENT,
}
_TARGET_KIND_TO_SOURCE: Final[dict[AttentionTargetKind, AttentionSourceKind]] = {
    target_kind: source_kind
    for source_kind, target_kind in _SOURCE_KIND_TO_TARGET.items()
}
_TARGET_SIGNAL_FIELDS: Final = (
    (AttentionTargetKind.WORKING_MEMORY, frozenset({"activation", "salience", "context_compatibility"})),
    (AttentionTargetKind.MOTIVATION, frozenset({"strength", "persistence", "satiation", "uncertainty"})),
    (AttentionTargetKind.GOAL, frozenset({"urgency"})),
    (AttentionTargetKind.COMMITMENT, frozenset({"urgency"})),
)
_SIGNAL_FIELDS: Final[tuple[str, ...]] = (
    "activation",
    "salience",
    "strength",
    "persistence",
    "satiation",
    "uncertainty",
    "urgency",
    "context_compatibility",
    "novelty",
)


def _closed_object(value: object, keys: frozenset[str], name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be an exact JSON object")
    if set(value) != keys or any(type(key) is not str for key in value):
        raise ValueError(f"{name} has missing or unknown fields")
    return value


def _json_list(value: object, name: str) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{name} must be a JSON array")
    return value


def _bounded_json_list(value: object, name: str, maximum: int) -> list[object]:
    result = _json_list(value, name)
    if len(result) > maximum:
        raise ValueError(f"{name} exceeds its bound")
    return result


def _bounded_canonical_value_bytes(value: object) -> bytes:
    try:
        encoded = canonical_json(value)
    except (OverflowError, RecursionError, TypeError, ValueError) as error:
        raise ValueError("Attention canonical input is invalid or too deeply nested") from error
    if len(encoded) > _ATTENTION_MAX_CANONICAL_VALUE_BYTES:
        raise ValueError("Attention canonical input exceeds its persisted byte bound")
    return encoded


def _check_json_nesting(text: str) -> None:
    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > _ATTENTION_MAX_JSON_NESTING:
                raise ValueError("Attention JSON exceeds its nesting bound")
        elif character in "]}":
            depth -= 1


def _optional_digest(value: object, name: str) -> str | None:
    return None if value is None else validate_digest(value, name)


def _fixed_point_value(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value <= ATTENTION_FIXED_POINT_SCALE:
        raise ValueError(
            f"{name} must be an exact fixed-point integer in [0, {ATTENTION_FIXED_POINT_SCALE}]"
        )
    return value


def _canonical_id_tuple(
    value: object,
    name: str,
    *,
    maximum: int = ATTENTION_MAX_FOCUS,
) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")
    if len(value) > maximum:
        raise ValueError(f"{name} exceeds its bound")
    ids = tuple(validate_digest(item, f"{name} item") for item in value)
    if ids != tuple(sorted(set(ids))):
        raise ValueError(f"{name} must be sorted and unique")
    return ids


def _parse_id_list(value: object, name: str) -> tuple[str, ...]:
    values = _bounded_json_list(value, name, ATTENTION_MAX_FOCUS)
    return _canonical_id_tuple(tuple(values), name)


def _event_not_future(source: AttentionEvent, current: AttentionEvent) -> None:
    if source.event_sequence > current.event_sequence:
        raise ValueError("source event is in the future of the Attention event")
    if source.occurred_at > current.occurred_at:
        raise ValueError("source event time is in the future of the Attention event")
    if source.event_sequence == current.event_sequence and source != current:
        raise ValueError("equal event sequences must identify the exact same event")


@dataclass(frozen=True, slots=True)
class AttentionEvent:
    """Caller-supplied positive sequence, opaque ID, and explicit UTC time."""

    event_id: str
    event_sequence: int
    occurred_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", parse_identifier(self.event_id, "event_id"))
        object.__setattr__(
            self,
            "event_sequence",
            bounded_positive_int(
                self.event_sequence,
                "event_sequence",
                maximum=ATTENTION_MAX_EVENT_SEQUENCE,
            ),
        )
        object.__setattr__(self, "occurred_at", utc_datetime(self.occurred_at, "occurred_at"))

    def canonical_value(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "event_sequence": self.event_sequence,
            "occurred_at": canonical_datetime(self.occurred_at),
        }

    @classmethod
    def from_canonical_value(cls, value: object) -> AttentionEvent:
        fields = _closed_object(
            value,
            frozenset({"event_id", "event_sequence", "occurred_at"}),
            "AttentionEvent",
        )
        return cls(
            event_id=parse_identifier(fields["event_id"], "event_id"),
            event_sequence=bounded_positive_int(
                fields["event_sequence"],
                "event_sequence",
                maximum=ATTENTION_MAX_EVENT_SEQUENCE,
            ),
            occurred_at=parse_canonical_datetime(fields["occurred_at"], "occurred_at"),
        )


@dataclass(frozen=True, slots=True)
class AttentionTarget:
    """Stable typed candidate identity; source-specific semantics stay upstream."""

    kind: AttentionTargetKind
    reference: str

    def __post_init__(self) -> None:
        kind = exact_enum(self.kind, AttentionTargetKind, "kind")
        object.__setattr__(
            self,
            "reference",
            validate_target_reference(kind, self.reference),
        )

    @property
    def candidate_id(self) -> str:
        return stable_target_id(self.kind, self.reference)

    def canonical_value(self) -> dict[str, str]:
        return {"kind": self.kind.value, "reference": self.reference}

    @classmethod
    def from_canonical_value(cls, value: object) -> AttentionTarget:
        fields = _closed_object(value, frozenset({"kind", "reference"}), "AttentionTarget")
        kind = parse_enum(fields["kind"], AttentionTargetKind, "target kind")
        reference = validate_target_reference(kind, fields["reference"])
        return cls(kind, reference)


@dataclass(frozen=True, slots=True)
class AttentionSourceWitness:
    """Reference-first evidence tied to one typed Attention target.

    Secondary evidence kinds can witness a target without becoming target
    kinds.  The explicit target pair prevents a source checksum from being
    transplanted to another candidate.  A primary WM/R13 witness uses the
    matching source kind/reference and the prescribed WM-derived or R13
    upstream digest classification.
    """

    kind: AttentionSourceKind
    reference: str
    revision: int
    digest: str
    digest_kind: SourceDigestKind
    target_kind: AttentionTargetKind
    target_reference: str
    source_event_id: str | None = None
    source_event_sequence: int | None = None
    source_occurred_at: datetime | None = None

    def __post_init__(self) -> None:
        kind = exact_enum(self.kind, AttentionSourceKind, "kind")
        target_kind = exact_enum(self.target_kind, AttentionTargetKind, "target_kind")
        digest_kind = exact_enum(self.digest_kind, SourceDigestKind, "digest_kind")
        if kind in _SOURCE_KIND_TO_TARGET:
            expected_target = _SOURCE_KIND_TO_TARGET[kind]
            if target_kind is not expected_target:
                raise ValueError("primary source kind does not match its target kind")
            reference = validate_target_reference(expected_target, self.reference)
            target_reference = validate_target_reference(target_kind, self.target_reference)
            if reference != target_reference:
                raise ValueError("primary source reference does not match its target")
            expected_digest_kind = (
                SourceDigestKind.ATTENTION_PROJECTION
                if kind is AttentionSourceKind.WORKING_MEMORY
                else SourceDigestKind.UPSTREAM_AUTHORITY
            )
            if digest_kind is not expected_digest_kind:
                raise ValueError("primary source witness digest kind does not match its authority")
        else:
            reference = parse_identifier(self.reference, "source reference")
            target_reference = validate_target_reference(target_kind, self.target_reference)
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "target_reference", target_reference)
        object.__setattr__(
            self,
            "revision",
            bounded_nonnegative_int(self.revision, "source revision"),
        )
        object.__setattr__(self, "digest", validate_digest(self.digest, "source digest"))
        if (self.source_event_id is None) != (self.source_event_sequence is None) or (
            self.source_event_id is None
        ) != (self.source_occurred_at is None):
            raise ValueError("source event ID, sequence, and time must be supplied together")
        if self.source_event_id is not None:
            object.__setattr__(
                self,
                "source_event_id",
                parse_identifier(self.source_event_id, "source_event_id"),
            )
            object.__setattr__(
                self,
                "source_event_sequence",
                bounded_positive_int(
                    self.source_event_sequence,
                    "source_event_sequence",
                    maximum=ATTENTION_MAX_EVENT_SEQUENCE,
                ),
            )
            object.__setattr__(
                self,
                "source_occurred_at",
                utc_datetime(self.source_occurred_at, "source_occurred_at"),
            )

    def event(self) -> AttentionEvent | None:
        if self.source_event_id is None:
            return None
        assert self.source_event_sequence is not None
        assert self.source_occurred_at is not None
        return AttentionEvent(
            self.source_event_id,
            self.source_event_sequence,
            self.source_occurred_at,
        )

    @property
    def witness_digest(self) -> str:
        """Checksum all witness fields; this does not authenticate a producer."""

        return digest_payload(
            b"PROJECT-SUZKA:R14:ATTENTION-SOURCE-WITNESS:V1\0",
            self._payload(),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "digest": self.digest,
            "digest_kind": self.digest_kind.value,
            "kind": self.kind.value,
            "reference": self.reference,
            "revision": self.revision,
            "source_event_id": self.source_event_id,
            "source_event_sequence": self.source_event_sequence,
            "source_occurred_at": None
            if self.source_occurred_at is None
            else canonical_datetime(self.source_occurred_at),
            "target_kind": self.target_kind.value,
            "target_reference": self.target_reference,
        }

    def validate_for(self, target: AttentionTarget, event: AttentionEvent) -> None:
        if type(target) is not AttentionTarget or type(event) is not AttentionEvent:
            raise TypeError("source witness validation requires exact target and event values")
        target.__post_init__()
        event.__post_init__()
        self.__post_init__()
        if self.target_kind is not target.kind or self.target_reference != target.reference:
            raise ValueError("source witness is bound to a different Attention target")
        source_event = self.event()
        if source_event is not None:
            _event_not_future(source_event, event)

    def validate_primary_for_target(self, target: AttentionTarget) -> None:
        """Require the main candidate witness to come from its target authority."""

        if type(target) is not AttentionTarget:
            raise TypeError("target must be an exact AttentionTarget")
        target.__post_init__()
        self.__post_init__()
        if self.kind is not _TARGET_KIND_TO_SOURCE[target.kind]:
            raise ValueError("primary source kind must match the candidate target domain")
        if self.reference != target.reference:
            raise ValueError("primary source reference must match the candidate target")
        if self.target_kind is not target.kind or self.target_reference != target.reference:
            raise ValueError("primary source target binding does not match the candidate")
        if (
            target.kind is AttentionTargetKind.WORKING_MEMORY
            and self.source_event_id is not None
        ):
            raise ValueError("Working Memory primary witnesses have no upstream event identity")

    def canonical_value(self) -> dict[str, object]:
        return {**self._payload(), "witness_digest": self.witness_digest}

    @classmethod
    def from_canonical_value(cls, value: object) -> AttentionSourceWitness:
        fields = _closed_object(
            value,
            frozenset(
                {
                    "digest",
                    "digest_kind",
                    "kind",
                    "reference",
                    "revision",
                    "source_event_id",
                    "source_event_sequence",
                    "source_occurred_at",
                    "target_kind",
                    "target_reference",
                    "witness_digest",
                }
            ),
            "AttentionSourceWitness",
        )
        source_time = fields["source_occurred_at"]
        kind = parse_enum(fields["kind"], AttentionSourceKind, "source kind")
        target_kind = parse_enum(
            fields["target_kind"], AttentionTargetKind, "source target kind"
        )
        reference = (
            validate_target_reference(_SOURCE_KIND_TO_TARGET[kind], fields["reference"])
            if kind in _SOURCE_KIND_TO_TARGET
            else parse_identifier(fields["reference"], "source reference")
        )
        source_event_id = (
            None
            if fields["source_event_id"] is None
            else parse_identifier(fields["source_event_id"], "source_event_id")
        )
        source_event_sequence = (
            None
            if fields["source_event_sequence"] is None
            else bounded_positive_int(
                fields["source_event_sequence"],
                "source_event_sequence",
                maximum=ATTENTION_MAX_EVENT_SEQUENCE,
            )
        )
        result = cls(
            kind=kind,
            reference=reference,
            revision=bounded_nonnegative_int(fields["revision"], "source revision"),
            digest=validate_digest(fields["digest"], "source digest"),
            digest_kind=parse_enum(
                fields["digest_kind"], SourceDigestKind, "source digest kind"
            ),
            target_kind=target_kind,
            target_reference=validate_target_reference(
                target_kind, fields["target_reference"]
            ),
            source_event_id=source_event_id,
            source_event_sequence=source_event_sequence,
            source_occurred_at=None
            if source_time is None
            else parse_canonical_datetime(source_time, "source_occurred_at"),
        )
        if result.witness_digest != fields["witness_digest"]:
            raise ValueError("source witness digest does not match its contents")
        return result


@dataclass(frozen=True, slots=True)
class AttentionSignalVector:
    """Measured normalized signals; ``None`` is unknown and 0.0 is measured zero."""

    activation: float | None = None
    salience: float | None = None
    strength: float | None = None
    persistence: float | None = None
    satiation: float | None = None
    uncertainty: float | None = None
    urgency: float | None = None
    context_compatibility: float | None = None
    novelty: float | None = None

    def __post_init__(self) -> None:
        for name in _SIGNAL_FIELDS:
            value = getattr(self, name)
            object.__setattr__(
                self,
                name,
                None if value is None else bounded_fraction(value, name),
            )

    @property
    def is_unknown(self) -> bool:
        return all(getattr(self, name) is None for name in _SIGNAL_FIELDS)

    def canonical_value(self) -> dict[str, str | None]:
        return {
            name: None if getattr(self, name) is None else getattr(self, name).hex()
            for name in _SIGNAL_FIELDS
        }

    @classmethod
    def from_canonical_value(cls, value: object) -> AttentionSignalVector:
        fields = _closed_object(value, frozenset(_SIGNAL_FIELDS), "AttentionSignalVector")
        return cls(
            **{
                name: parse_hex_fraction(fields[name], name)
                for name in _SIGNAL_FIELDS
            }
        )


def _source_payload(
    target: AttentionTarget,
    source: AttentionSourceWitness,
    signals: AttentionSignalVector,
    event: AttentionEvent,
    availability: CandidateAvailability,
    rendered_bytes: int | None,
    rendered_digest: str | None,
) -> dict[str, object]:
    return {
        "availability": availability.value,
        "event": event.canonical_value(),
        "rendered_bytes": rendered_bytes,
        "rendered_digest": rendered_digest,
        "signals": signals.canonical_value(),
        "source": source.canonical_value(),
        "target": target.canonical_value(),
    }


@dataclass(frozen=True, slots=True, init=False)
class AttentionCandidateProjection:
    """Ephemeral source-adapter product; no raw rendered source text is retained."""

    target: AttentionTarget
    source: AttentionSourceWitness
    signals: AttentionSignalVector
    event: AttentionEvent
    availability: CandidateAvailability
    rendered_bytes: int | None
    rendered_digest: str | None
    projection_digest: str = field(init=False)

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("AttentionCandidateProjection values are created by source adapters")

    @classmethod
    def _create(
        cls,
        *,
        target: AttentionTarget,
        source: AttentionSourceWitness,
        signals: AttentionSignalVector,
        event: AttentionEvent,
        availability: CandidateAvailability,
        rendered_bytes: int | None,
        rendered_digest: str | None,
    ) -> AttentionCandidateProjection:
        (
            checked_target,
            checked_source,
            checked_signals,
            checked_event,
            checked_availability,
            checked_rendered_bytes,
            checked_rendered_digest,
            payload,
        ) = cls._validated_components(
            target=target,
            source=source,
            signals=signals,
            event=event,
            availability=availability,
            rendered_bytes=rendered_bytes,
            rendered_digest=rendered_digest,
        )
        result = object.__new__(cls)
        for name, value in (
            ("target", checked_target),
            ("source", checked_source),
            ("signals", checked_signals),
            ("event", checked_event),
            ("availability", checked_availability),
            ("rendered_bytes", checked_rendered_bytes),
            ("rendered_digest", checked_rendered_digest),
        ):
            object.__setattr__(result, name, value)
        object.__setattr__(
            result,
            "projection_digest",
            digest_payload(b"PROJECT-SUZKA:R14:ATTENTION-PROJECTION:V1\0", payload),
        )
        return result

    @classmethod
    def _validated_components(
        cls,
        *,
        target: AttentionTarget,
        source: AttentionSourceWitness,
        signals: AttentionSignalVector,
        event: AttentionEvent,
        availability: CandidateAvailability,
        rendered_bytes: int | None,
        rendered_digest: str | None,
    ) -> tuple[
        AttentionTarget,
        AttentionSourceWitness,
        AttentionSignalVector,
        AttentionEvent,
        CandidateAvailability,
        int | None,
        str | None,
        dict[str, object],
    ]:
        if type(target) is not AttentionTarget:
            raise TypeError("target must be an exact AttentionTarget")
        if type(source) is not AttentionSourceWitness:
            raise TypeError("source must be an exact AttentionSourceWitness")
        if type(signals) is not AttentionSignalVector:
            raise TypeError("signals must be an exact AttentionSignalVector")
        if type(event) is not AttentionEvent:
            raise TypeError("event must be an exact AttentionEvent")
        for name in _SIGNAL_FIELDS:
            signal = getattr(signals, name)
            if signal is not None and type(signal) is not float:
                raise TypeError(f"projection signal {name} must remain an exact float")
        target_copy = AttentionTarget(target.kind, target.reference)
        source_copy = AttentionSourceWitness(
            kind=source.kind,
            reference=source.reference,
            revision=source.revision,
            digest=source.digest,
            digest_kind=source.digest_kind,
            target_kind=source.target_kind,
            target_reference=source.target_reference,
            source_event_id=source.source_event_id,
            source_event_sequence=source.source_event_sequence,
            source_occurred_at=source.source_occurred_at,
        )
        signals_copy = AttentionSignalVector(
            **{name: getattr(signals, name) for name in _SIGNAL_FIELDS}
        )
        event_copy = AttentionEvent(
            event.event_id,
            event.event_sequence,
            event.occurred_at,
        )
        checked_availability = exact_enum(
            availability, CandidateAvailability, "availability"
        )
        source_copy.validate_primary_for_target(target_copy)
        source_copy.validate_for(target_copy, event_copy)
        allowed_signals = next(
            fields for kind, fields in _TARGET_SIGNAL_FIELDS if kind is target_copy.kind
        )
        if any(
            getattr(signals_copy, name) is not None
            for name in _SIGNAL_FIELDS
            if name not in allowed_signals
        ):
            raise ValueError("signal has no reviewed adapter for this target kind")
        checked_rendered_bytes = rendered_bytes
        checked_rendered_digest = rendered_digest
        if checked_availability is CandidateAvailability.ELIGIBLE:
            if (
                type(checked_rendered_bytes) is not int
                or not 1
                <= checked_rendered_bytes
                <= ATTENTION_MAX_RENDERED_ITEM_BYTES
            ):
                raise ValueError("eligible candidates require bounded rendered byte accounting")
            checked_rendered_digest = validate_digest(
                checked_rendered_digest,
                "rendered_digest",
            )
        else:
            if checked_rendered_bytes is not None or checked_rendered_digest is not None:
                raise ValueError("non-eligible candidates cannot carry a rendered row")
        payload = _source_payload(
            target_copy,
            source_copy,
            signals_copy,
            event_copy,
            checked_availability,
            checked_rendered_bytes,
            checked_rendered_digest,
        )
        return (
            target_copy,
            source_copy,
            signals_copy,
            event_copy,
            checked_availability,
            checked_rendered_bytes,
            checked_rendered_digest,
            payload,
        )

    def __post_init__(self) -> None:
        """Revalidate without repairing or mutating a published projection."""

        expected = self.validated_copy()
        if expected.projection_digest != validate_digest(
            self.projection_digest,
            "projection_digest",
        ):
            raise ValueError("Attention projection digest does not match its fields")

    def validated_copy(self) -> AttentionCandidateProjection:
        """Return a cloned, revalidated packet or reject any post-publication change."""

        if type(self) is not AttentionCandidateProjection:
            raise TypeError("projection must be an exact AttentionCandidateProjection")
        declared_digest = validate_digest(self.projection_digest, "projection_digest")
        checked = AttentionCandidateProjection._create(
            target=self.target,
            source=self.source,
            signals=self.signals,
            event=self.event,
            availability=self.availability,
            rendered_bytes=self.rendered_bytes,
            rendered_digest=self.rendered_digest,
        )
        if checked.projection_digest != declared_digest:
            raise ValueError("Attention projection digest does not match its fields")
        return checked

    @property
    def candidate_id(self) -> str:
        return self.target.candidate_id

    def canonical_value(self) -> dict[str, object]:
        return {
            **_source_payload(
                self.target,
                self.source,
                self.signals,
                self.event,
                self.availability,
                self.rendered_bytes,
                self.rendered_digest,
            ),
            "candidate_id": self.candidate_id,
            "projection_digest": self.projection_digest,
        }


@dataclass(frozen=True, slots=True)
class AttentionCandidateContinuity:
    """Persisted reference-only attention continuity for one current target."""

    target: AttentionTarget
    source: AttentionSourceWitness
    availability: CandidateAvailability
    habituation: int = 0
    inhibition: int = 0
    focused_event_count: int = 0
    unattended_event_count: int = 0
    record_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.target) is not AttentionTarget:
            raise TypeError("target must be an exact AttentionTarget")
        if type(self.source) is not AttentionSourceWitness:
            raise TypeError("source must be an exact AttentionSourceWitness")
        self.target.__post_init__()
        self.source.__post_init__()
        self.source.validate_primary_for_target(self.target)
        availability = exact_enum(self.availability, CandidateAvailability, "availability")
        for name in ("habituation", "inhibition"):
            object.__setattr__(self, name, _fixed_point_value(getattr(self, name), name))
        for name in ("focused_event_count", "unattended_event_count"):
            object.__setattr__(self, name, bounded_counter(getattr(self, name), name))
        if self.focused_event_count and self.unattended_event_count:
            raise ValueError("focused and unattended event streaks are mutually exclusive")
        if (
            self.focused_event_count
            and availability is not CandidateAvailability.ELIGIBLE
        ):
            raise ValueError("only eligible candidates can have a focused event streak")
        object.__setattr__(
            self,
            "record_digest",
            digest_payload(
                b"PROJECT-SUZKA:R14:ATTENTION-CANDIDATE:V1\0",
                self._payload(),
            ),
        )

    @property
    def candidate_id(self) -> str:
        return self.target.candidate_id

    def _payload(self) -> dict[str, object]:
        return {
            "availability": self.availability.value,
            "candidate_id": self.candidate_id,
            "focused_event_count": self.focused_event_count,
            "habituation": self.habituation,
            "inhibition": self.inhibition,
            "source": self.source.canonical_value(),
            "target": self.target.canonical_value(),
            "unattended_event_count": self.unattended_event_count,
        }

    def canonical_value(self) -> dict[str, object]:
        return {**self._payload(), "record_digest": self.record_digest}

    @classmethod
    def from_canonical_value(cls, value: object) -> AttentionCandidateContinuity:
        fields = _closed_object(
            value,
            frozenset(
                {
                    "availability",
                    "candidate_id",
                    "focused_event_count",
                    "habituation",
                    "inhibition",
                    "record_digest",
                    "source",
                    "target",
                    "unattended_event_count",
                }
            ),
            "AttentionCandidateContinuity",
        )
        result = cls(
            target=AttentionTarget.from_canonical_value(fields["target"]),
            source=AttentionSourceWitness.from_canonical_value(fields["source"]),
            availability=parse_enum(
                fields["availability"], CandidateAvailability, "availability"
            ),
            habituation=_fixed_point_value(fields["habituation"], "habituation"),
            inhibition=_fixed_point_value(fields["inhibition"], "inhibition"),
            focused_event_count=bounded_counter(
                fields["focused_event_count"], "focused_event_count"
            ),
            unattended_event_count=bounded_counter(
                fields["unattended_event_count"], "unattended_event_count"
            ),
        )
        if result.candidate_id != fields["candidate_id"]:
            raise ValueError("candidate_id does not match its typed target")
        if result.record_digest != fields["record_digest"]:
            raise ValueError("candidate record digest does not match its contents")
        return result


@dataclass(frozen=True, slots=True)
class AttentionRevisionAnchor:
    """Immutable fence immediately before a retained revision-history suffix."""

    through_revision: int
    through_event: AttentionEvent
    through_state_digest: str
    through_revision_digest: str
    anchor_digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "through_revision",
            bounded_nonnegative_int(self.through_revision, "through_revision"),
        )
        if self.through_revision == 0:
            raise ValueError("revision-history anchors must fence a positive revision")
        if type(self.through_event) is not AttentionEvent:
            raise TypeError("through_event must be an exact AttentionEvent")
        self.through_event.__post_init__()
        object.__setattr__(
            self,
            "through_state_digest",
            validate_digest(self.through_state_digest, "through_state_digest"),
        )
        object.__setattr__(
            self,
            "through_revision_digest",
            validate_digest(self.through_revision_digest, "through_revision_digest"),
        )
        object.__setattr__(
            self,
            "anchor_digest",
            digest_payload(
                b"PROJECT-SUZKA:R14:ATTENTION-REVISION-ANCHOR:V1\0",
                self._payload(),
            ),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "through_event": self.through_event.canonical_value(),
            "through_revision": self.through_revision,
            "through_revision_digest": self.through_revision_digest,
            "through_state_digest": self.through_state_digest,
        }

    def canonical_value(self) -> dict[str, object]:
        return {**self._payload(), "anchor_digest": self.anchor_digest}

    @classmethod
    def from_canonical_value(cls, value: object) -> AttentionRevisionAnchor:
        fields = _closed_object(
            value,
            frozenset(
                {
                    "anchor_digest",
                    "through_event",
                    "through_revision",
                    "through_revision_digest",
                    "through_state_digest",
                }
            ),
            "AttentionRevisionAnchor",
        )
        result = cls(
            through_revision=bounded_nonnegative_int(
                fields["through_revision"], "through_revision"
            ),
            through_event=AttentionEvent.from_canonical_value(fields["through_event"]),
            through_state_digest=validate_digest(
                fields["through_state_digest"], "through_state_digest"
            ),
            through_revision_digest=validate_digest(
                fields["through_revision_digest"], "through_revision_digest"
            ),
        )
        if result.anchor_digest != fields["anchor_digest"]:
            raise ValueError("revision anchor digest does not match its contents")
        return result


@dataclass(frozen=True, slots=True)
class AttentionReceiptAnchor:
    """Independent immutable fence for the bounded event-receipt suffix."""

    through_event: AttentionEvent
    through_result_state_digest: str
    through_receipt_digest: str
    anchor_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.through_event) is not AttentionEvent:
            raise TypeError("through_event must be an exact AttentionEvent")
        self.through_event.__post_init__()
        object.__setattr__(
            self,
            "through_result_state_digest",
            validate_digest(self.through_result_state_digest, "through_result_state_digest"),
        )
        object.__setattr__(
            self,
            "through_receipt_digest",
            validate_digest(self.through_receipt_digest, "through_receipt_digest"),
        )
        object.__setattr__(
            self,
            "anchor_digest",
            digest_payload(
                b"PROJECT-SUZKA:R14:ATTENTION-RECEIPT-ANCHOR:V1\0",
                self._payload(),
            ),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "through_event": self.through_event.canonical_value(),
            "through_receipt_digest": self.through_receipt_digest,
            "through_result_state_digest": self.through_result_state_digest,
        }

    def canonical_value(self) -> dict[str, object]:
        return {**self._payload(), "anchor_digest": self.anchor_digest}

    @classmethod
    def from_canonical_value(cls, value: object) -> AttentionReceiptAnchor:
        fields = _closed_object(
            value,
            frozenset(
                {
                    "anchor_digest",
                    "through_event",
                    "through_receipt_digest",
                    "through_result_state_digest",
                }
            ),
            "AttentionReceiptAnchor",
        )
        result = cls(
            through_event=AttentionEvent.from_canonical_value(fields["through_event"]),
            through_result_state_digest=validate_digest(
                fields["through_result_state_digest"], "through_result_state_digest"
            ),
            through_receipt_digest=validate_digest(
                fields["through_receipt_digest"], "through_receipt_digest"
            ),
        )
        if result.anchor_digest != fields["anchor_digest"]:
            raise ValueError("receipt anchor digest does not match its contents")
        return result


@dataclass(frozen=True, slots=True)
class AttentionRevisionEvidence:
    """One immutable state-revision witness; it is validated, never replayed."""

    revision: int
    event: AttentionEvent
    previous_state_digest: str
    state_digest: str
    previous_revision_digest: str | None
    focused_ids: tuple[str, ...]
    unfinished_ids: tuple[str, ...]
    reason: AttentionRevisionReason
    record_digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "revision",
            bounded_positive_int(
                self.revision,
                "revision",
                maximum=ATTENTION_MAX_REVISION,
            ),
        )
        if type(self.event) is not AttentionEvent:
            raise TypeError("event must be an exact AttentionEvent")
        self.event.__post_init__()
        object.__setattr__(
            self,
            "previous_state_digest",
            validate_digest(self.previous_state_digest, "previous_state_digest"),
        )
        object.__setattr__(self, "state_digest", validate_digest(self.state_digest, "state_digest"))
        object.__setattr__(
            self,
            "previous_revision_digest",
            _optional_digest(self.previous_revision_digest, "previous_revision_digest"),
        )
        if (self.revision == 1) != (self.previous_revision_digest is None):
            raise ValueError("the first Attention revision alone omits its prior revision digest")
        object.__setattr__(self, "focused_ids", _canonical_id_tuple(self.focused_ids, "focused_ids"))
        object.__setattr__(
            self,
            "unfinished_ids",
            _canonical_id_tuple(self.unfinished_ids, "unfinished_ids"),
        )
        exact_enum(self.reason, AttentionRevisionReason, "reason")
        object.__setattr__(
            self,
            "record_digest",
            digest_payload(
                b"PROJECT-SUZKA:R14:ATTENTION-REVISION:V1\0",
                self._payload(),
            ),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "event": self.event.canonical_value(),
            "focused_ids": list(self.focused_ids),
            "previous_revision_digest": self.previous_revision_digest,
            "previous_state_digest": self.previous_state_digest,
            "reason": self.reason.value,
            "revision": self.revision,
            "state_digest": self.state_digest,
            "unfinished_ids": list(self.unfinished_ids),
        }

    def canonical_value(self) -> dict[str, object]:
        return {**self._payload(), "record_digest": self.record_digest}

    @classmethod
    def from_canonical_value(cls, value: object) -> AttentionRevisionEvidence:
        fields = _closed_object(
            value,
            frozenset(
                {
                    "event",
                    "focused_ids",
                    "previous_revision_digest",
                    "previous_state_digest",
                    "reason",
                    "record_digest",
                    "revision",
                    "state_digest",
                    "unfinished_ids",
                }
            ),
            "AttentionRevisionEvidence",
        )
        result = cls(
            revision=bounded_positive_int(
                fields["revision"], "revision", maximum=ATTENTION_MAX_REVISION
            ),
            event=AttentionEvent.from_canonical_value(fields["event"]),
            previous_state_digest=validate_digest(
                fields["previous_state_digest"], "previous_state_digest"
            ),
            state_digest=validate_digest(fields["state_digest"], "state_digest"),
            previous_revision_digest=_optional_digest(
                fields["previous_revision_digest"], "previous_revision_digest"
            ),
            focused_ids=_parse_id_list(fields["focused_ids"], "focused_ids"),
            unfinished_ids=_parse_id_list(fields["unfinished_ids"], "unfinished_ids"),
            reason=parse_enum(fields["reason"], AttentionRevisionReason, "reason"),
        )
        if result.record_digest != fields["record_digest"]:
            raise ValueError("revision evidence digest does not match its contents")
        return result


@dataclass(frozen=True, slots=True)
class AttentionEventReceipt:
    """Exact idempotency witness for one Attention event input and result."""

    event: AttentionEvent
    input_digest: str
    result_state_digest: str
    previous_receipt_digest: str | None = None
    receipt_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.event) is not AttentionEvent:
            raise TypeError("event must be an exact AttentionEvent")
        self.event.__post_init__()
        object.__setattr__(self, "input_digest", validate_digest(self.input_digest, "input_digest"))
        object.__setattr__(
            self,
            "result_state_digest",
            validate_digest(self.result_state_digest, "result_state_digest"),
        )
        object.__setattr__(
            self,
            "previous_receipt_digest",
            _optional_digest(self.previous_receipt_digest, "previous_receipt_digest"),
        )
        object.__setattr__(
            self,
            "receipt_digest",
            digest_payload(
                b"PROJECT-SUZKA:R14:ATTENTION-RECEIPT:V1\0",
                self._payload(),
            ),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "event": self.event.canonical_value(),
            "input_digest": self.input_digest,
            "previous_receipt_digest": self.previous_receipt_digest,
            "result_state_digest": self.result_state_digest,
        }

    def canonical_value(self) -> dict[str, object]:
        return {**self._payload(), "receipt_digest": self.receipt_digest}

    @classmethod
    def from_canonical_value(cls, value: object) -> AttentionEventReceipt:
        fields = _closed_object(
            value,
            frozenset(
                {
                    "event",
                    "input_digest",
                    "previous_receipt_digest",
                    "receipt_digest",
                    "result_state_digest",
                }
            ),
            "AttentionEventReceipt",
        )
        result = cls(
            event=AttentionEvent.from_canonical_value(fields["event"]),
            input_digest=validate_digest(fields["input_digest"], "input_digest"),
            result_state_digest=validate_digest(
                fields["result_state_digest"], "result_state_digest"
            ),
            previous_receipt_digest=_optional_digest(
                fields["previous_receipt_digest"], "previous_receipt_digest"
            ),
        )
        if result.receipt_digest != fields["receipt_digest"]:
            raise ValueError("event receipt digest does not match its contents")
        return result


def _candidate_values(
    candidates: tuple[AttentionCandidateContinuity, ...],
) -> list[dict[str, object]]:
    return [candidate.canonical_value() for candidate in candidates]


def attention_state_digest(
    *,
    schema_version: int,
    policy_version: int,
    revision: int,
    last_event: AttentionEvent | None,
    candidates: tuple[AttentionCandidateContinuity, ...],
    focused_ids: tuple[str, ...],
    unfinished_ids: tuple[str, ...],
) -> str:
    """Compute state identity without self-referential history or receipts."""

    payload = {
        "candidates": _candidate_values(candidates),
        "focused_ids": list(focused_ids),
        "last_event": None if last_event is None else last_event.canonical_value(),
        "policy_version": policy_version,
        "revision": revision,
        "schema_version": schema_version,
        "unfinished_ids": list(unfinished_ids),
    }
    return digest_payload(_ATTENTION_STATE_DOMAIN, payload)


@dataclass(frozen=True, slots=True)
class AttentionContinuity:
    """Closed canonical state value ready for a later AgentState owner."""

    schema_version: int = ATTENTION_SCHEMA_VERSION
    policy_version: int = ATTENTION_POLICY_VERSION
    revision: int = 0
    last_event: AttentionEvent | None = None
    candidates: tuple[AttentionCandidateContinuity, ...] = ()
    focused_ids: tuple[str, ...] = ()
    unfinished_ids: tuple[str, ...] = ()
    revision_history: tuple[AttentionRevisionEvidence, ...] = ()
    receipts: tuple[AttentionEventReceipt, ...] = ()
    revision_anchor: AttentionRevisionAnchor | None = None
    receipt_anchor: AttentionReceiptAnchor | None = None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != ATTENTION_SCHEMA_VERSION:
            raise ValueError("unsupported Attention schema version")
        if type(self.policy_version) is not int or self.policy_version != ATTENTION_POLICY_VERSION:
            raise ValueError("unsupported Attention policy version")
        object.__setattr__(
            self,
            "revision",
            bounded_nonnegative_int(self.revision, "revision"),
        )
        if self.last_event is not None and type(self.last_event) is not AttentionEvent:
            raise TypeError("last_event must be an exact AttentionEvent or None")
        if self.last_event is not None:
            self.last_event.__post_init__()
        if type(self.candidates) is not tuple:
            raise TypeError("candidates must be an exact tuple")
        if type(self.focused_ids) is not tuple or type(self.unfinished_ids) is not tuple:
            raise TypeError("focus and unfinished identities must be exact tuples")
        if type(self.revision_history) is not tuple or type(self.receipts) is not tuple:
            raise TypeError("history and receipts must be exact tuples")
        if len(self.revision_history) > ATTENTION_MAX_REVISION_HISTORY:
            raise ValueError("Attention revision history exceeds its retained window")
        if len(self.receipts) > ATTENTION_MAX_EVENT_RECEIPTS:
            raise ValueError("Attention event receipts exceed their retained window")
        if self.revision_anchor is not None and type(self.revision_anchor) is not AttentionRevisionAnchor:
            raise TypeError("revision_anchor must be an exact AttentionRevisionAnchor")
        if self.receipt_anchor is not None and type(self.receipt_anchor) is not AttentionReceiptAnchor:
            raise TypeError("receipt_anchor must be an exact AttentionReceiptAnchor")
        if self.revision_anchor is not None:
            self.revision_anchor.__post_init__()
        if self.receipt_anchor is not None:
            self.receipt_anchor.__post_init__()

        if self.revision == 0:
            if (
                self.last_event is not None
                or self.candidates
                or self.focused_ids
                or self.unfinished_ids
                or self.revision_history
                or self.receipts
                or self.revision_anchor is not None
                or self.receipt_anchor is not None
            ):
                raise ValueError("revision-zero Attention continuity must be the empty bootstrap")
            return
        if self.last_event is None:
            raise ValueError("nonempty Attention continuity requires its last event")
        current_event = self.last_event

        from suzka.working_memory_contracts import MAX_ITEM_CAPACITY

        if type(MAX_ITEM_CAPACITY) is not int or MAX_ITEM_CAPACITY <= 0:
            raise RuntimeError("neutral Working Memory capacity is not a positive integer")
        per_kind_limits = {
            AttentionTargetKind.WORKING_MEMORY: MAX_ITEM_CAPACITY,
            AttentionTargetKind.MOTIVATION: ATTENTION_MAX_R13_RECORDS_PER_DOMAIN,
            AttentionTargetKind.GOAL: ATTENTION_MAX_R13_RECORDS_PER_DOMAIN,
            AttentionTargetKind.COMMITMENT: ATTENTION_MAX_R13_RECORDS_PER_DOMAIN,
        }
        counts = {kind: 0 for kind in AttentionTargetKind}
        candidate_ids: list[str] = []
        focused_streak_ids: set[str] = set()
        for candidate in self.candidates:
            if type(candidate) is not AttentionCandidateContinuity:
                raise TypeError("candidates must contain exact AttentionCandidateContinuity values")
            candidate.__post_init__()
            counts[candidate.target.kind] += 1
            candidate_ids.append(candidate.candidate_id)
            if candidate.focused_event_count:
                focused_streak_ids.add(candidate.candidate_id)
            if max(
                candidate.focused_event_count,
                candidate.unattended_event_count,
            ) > current_event.event_sequence:
                raise ValueError("candidate event streak exceeds the current event sequence")
            candidate.source.validate_for(candidate.target, current_event)
        if candidate_ids != sorted(set(candidate_ids)):
            raise ValueError("current Attention candidates must be uniquely sorted by candidate ID")
        if any(counts[kind] > limit for kind, limit in per_kind_limits.items()):
            raise ValueError("current Attention candidate count exceeds a source authority bound")

        focused = _canonical_id_tuple(self.focused_ids, "focused_ids")
        unfinished = _canonical_id_tuple(self.unfinished_ids, "unfinished_ids")
        object.__setattr__(self, "focused_ids", focused)
        object.__setattr__(self, "unfinished_ids", unfinished)
        by_id = {candidate.candidate_id: candidate for candidate in self.candidates}
        for name, refs in (("focused_ids", focused), ("unfinished_ids", unfinished)):
            for candidate_id in refs:
                matched_candidate = by_id.get(candidate_id)
                if matched_candidate is None:
                    raise ValueError(f"{name} contains a dangling candidate identity")
                if matched_candidate.availability is not CandidateAvailability.ELIGIBLE:
                    raise ValueError(f"{name} contains a candidate that is not eligible")
        if set(focused) != focused_streak_ids:
            raise ValueError("focused IDs must exactly match eligible focused streaks")

        if any(type(item) is not AttentionRevisionEvidence for item in self.revision_history):
            raise TypeError("revision_history contains a non-canonical evidence value")
        if any(type(item) is not AttentionEventReceipt for item in self.receipts):
            raise TypeError("receipts contains a non-canonical receipt value")
        for revision_item in self.revision_history:
            revision_item.__post_init__()
        for receipt_item in self.receipts:
            receipt_item.__post_init__()

        self._validate_revision_history()
        self._validate_receipts()
        self._validate_retained_event_consistency()

    @classmethod
    def bootstrap(cls) -> AttentionContinuity:
        """Return the only valid empty revision-zero root."""

        return cls()

    @property
    def state_digest(self) -> str:
        return attention_state_digest(
            schema_version=self.schema_version,
            policy_version=self.policy_version,
            revision=self.revision,
            last_event=self.last_event,
            candidates=self.candidates,
            focused_ids=self.focused_ids,
            unfinished_ids=self.unfinished_ids,
        )

    @property
    def authority_digest(self) -> str:
        return digest_payload(
            b"PROJECT-SUZKA:R14:ATTENTION-AUTHORITY:V1\0",
            self._authority_payload(),
        )

    def _state_payload(self) -> dict[str, object]:
        return {
            "candidates": _candidate_values(self.candidates),
            "focused_ids": list(self.focused_ids),
            "last_event": None
            if self.last_event is None
            else self.last_event.canonical_value(),
            "policy_version": self.policy_version,
            "revision": self.revision,
            "schema_version": self.schema_version,
            "unfinished_ids": list(self.unfinished_ids),
        }

    def _authority_payload(self) -> dict[str, object]:
        return {
            "candidates": _candidate_values(self.candidates),
            "focused_ids": list(self.focused_ids),
            "last_event": None
            if self.last_event is None
            else self.last_event.canonical_value(),
            "policy_version": self.policy_version,
            "receipt_anchor": None
            if self.receipt_anchor is None
            else self.receipt_anchor.canonical_value(),
            "receipts": [item.canonical_value() for item in self.receipts],
            "revision": self.revision,
            "revision_anchor": None
            if self.revision_anchor is None
            else self.revision_anchor.canonical_value(),
            "revision_history": [item.canonical_value() for item in self.revision_history],
            "schema_version": self.schema_version,
            "state_digest": self.state_digest,
            "unfinished_ids": list(self.unfinished_ids),
        }

    def canonical_value(self) -> dict[str, object]:
        return {**self._authority_payload(), "authority_digest": self.authority_digest}

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.canonical_value())

    def _validate_revision_history(self) -> None:
        current_event = self.last_event
        if current_event is None:
            raise ValueError("revision history requires a current event")
        history = self.revision_history
        if self.revision <= ATTENTION_MAX_REVISION_HISTORY:
            if self.revision_anchor is not None:
                raise ValueError("uncompacted revision history cannot carry an anchor")
            if len(history) != self.revision:
                raise ValueError("uncompacted history must retain every non-genesis revision")
            previous_revision_digest: str | None = None
            previous_state_digest = AttentionContinuity.bootstrap().state_digest
            previous_event: AttentionEvent | None = None
        else:
            anchor = self.revision_anchor
            if type(anchor) is not AttentionRevisionAnchor:
                raise ValueError("compacted revision history requires its independent anchor")
            if len(history) != ATTENTION_MAX_REVISION_HISTORY:
                raise ValueError("compacted revision history requires its full retained suffix")
            if anchor.through_revision != self.revision - len(history):
                raise ValueError("revision-history anchor fence does not match the retained suffix")
            previous_revision_digest = anchor.through_revision_digest
            previous_state_digest = anchor.through_state_digest
            previous_event = anchor.through_event

        previous_revision = 0 if previous_event is None else self.revision - len(history)
        seen_ids = set() if previous_event is None else {previous_event.event_id}
        for item in history:
            if item.revision != previous_revision + 1:
                raise ValueError("revision history is not contiguous")
            if item.previous_revision_digest != previous_revision_digest:
                raise ValueError("revision history has a broken revision-digest link")
            if item.previous_state_digest != previous_state_digest:
                raise ValueError("revision history has a broken state-digest link")
            if previous_event is not None:
                if item.event.event_sequence <= previous_event.event_sequence:
                    raise ValueError("revision events must increase strictly")
                if item.event.occurred_at < previous_event.occurred_at:
                    raise ValueError("revision event times cannot regress")
            if item.event.event_id in seen_ids:
                raise ValueError("revision event IDs cannot be reused")
            seen_ids.add(item.event.event_id)
            if item.event.event_sequence > current_event.event_sequence:
                raise ValueError("revision evidence contains a future event")
            if item.event.occurred_at > current_event.occurred_at:
                raise ValueError("revision evidence contains a future event time")
            previous_revision = item.revision
            previous_revision_digest = item.record_digest
            previous_state_digest = item.state_digest
            previous_event = item.event
        last = history[-1]
        if last.revision != self.revision or last.state_digest != self.state_digest:
            raise ValueError("retained revision history does not bind the current state")
        if last.event != current_event:
            raise ValueError("retained revision history does not bind the current event")
        if last.focused_ids != self.focused_ids or last.unfinished_ids != self.unfinished_ids:
            raise ValueError("retained revision focus evidence does not bind current selections")
        if self.revision_anchor is not None:
            if self.revision_anchor.through_event.event_sequence >= history[0].event.event_sequence:
                raise ValueError("revision anchor event fence overlaps retained history")
            if self.revision_anchor.through_event.occurred_at > history[0].event.occurred_at:
                raise ValueError("revision anchor time fence overlaps retained history")

    def _validate_receipts(self) -> None:
        current_event = self.last_event
        if current_event is None:
            raise ValueError("event receipts require a current event")
        receipts = self.receipts
        if not receipts:
            raise ValueError("nonempty Attention continuity requires its last event receipt")
        anchor = self.receipt_anchor
        if anchor is None:
            previous_digest: str | None = None
            previous_event: AttentionEvent | None = None
        else:
            if len(receipts) != ATTENTION_MAX_EVENT_RECEIPTS:
                raise ValueError("compacted receipts require their full independent retained suffix")
            if anchor.through_event.event_sequence >= receipts[0].event.event_sequence:
                raise ValueError("receipt anchor event fence overlaps retained receipts")
            if anchor.through_event.occurred_at > receipts[0].event.occurred_at:
                raise ValueError("receipt anchor time fence overlaps retained receipts")
            previous_digest = anchor.through_receipt_digest
            previous_event = anchor.through_event

        seen_ids = set() if previous_event is None else {previous_event.event_id}
        for receipt in receipts:
            if receipt.previous_receipt_digest != previous_digest:
                raise ValueError("receipt chain has a broken digest link")
            if previous_event is not None:
                if receipt.event.event_sequence <= previous_event.event_sequence:
                    raise ValueError("receipt event sequences must increase strictly")
                if receipt.event.occurred_at < previous_event.occurred_at:
                    raise ValueError("receipt event times cannot regress")
            if receipt.event.event_id in seen_ids:
                raise ValueError("receipt event IDs cannot be reused")
            seen_ids.add(receipt.event.event_id)
            if receipt.event.event_sequence > current_event.event_sequence:
                raise ValueError("receipt contains a future event")
            if receipt.event.occurred_at > current_event.occurred_at:
                raise ValueError("receipt contains a future event time")
            previous_digest = receipt.receipt_digest
            previous_event = receipt.event
        last = receipts[-1]
        if last.event != current_event or last.result_state_digest != self.state_digest:
            raise ValueError("last event receipt does not bind current Attention state")
        receipt_events = {item.event.event_sequence: item.event for item in receipts}
        receipt_ids = {item.event.event_id: item.event for item in receipts}
        for evidence in self.revision_history:
            matching_receipt = receipt_events.get(evidence.event.event_sequence)
            if matching_receipt is not None and matching_receipt != evidence.event:
                raise ValueError("revision and receipt evidence disagree on a shared event")
            matching_id = receipt_ids.get(evidence.event.event_id)
            if matching_id is not None and matching_id != evidence.event:
                raise ValueError("revision and receipt evidence reuse an event ID")

    def _validate_retained_event_consistency(self) -> None:
        current_event = self.last_event
        if current_event is None:
            raise ValueError("retained event consistency requires a current event")
        retained_events = [current_event]
        retained_events.extend(item.event for item in self.revision_history)
        retained_events.extend(item.event for item in self.receipts)
        if self.revision_anchor is not None:
            retained_events.append(self.revision_anchor.through_event)
        if self.receipt_anchor is not None:
            retained_events.append(self.receipt_anchor.through_event)
        for candidate in self.candidates:
            source_event = candidate.source.event()
            if source_event is not None:
                retained_events.append(source_event)

        for index, event in enumerate(retained_events):
            for previous in retained_events[:index]:
                if (
                    event.event_id == previous.event_id
                    or event.event_sequence == previous.event_sequence
                ) and event != previous:
                    raise ValueError(
                        "retained event identities disagree on ID, sequence, or UTC time"
                    )

        state_witnesses: list[tuple[AttentionEvent, str]] = [
            (current_event, self.state_digest)
        ]
        state_witnesses.extend(
            (item.event, item.state_digest) for item in self.revision_history
        )
        state_witnesses.extend(
            (item.event, item.result_state_digest) for item in self.receipts
        )
        if self.revision_anchor is not None:
            state_witnesses.append(
                (
                    self.revision_anchor.through_event,
                    self.revision_anchor.through_state_digest,
                )
            )
        if self.receipt_anchor is not None:
            state_witnesses.append(
                (
                    self.receipt_anchor.through_event,
                    self.receipt_anchor.through_result_state_digest,
                )
            )
        for index, (event, state_digest) in enumerate(state_witnesses):
            for previous_event, previous_state_digest in state_witnesses[:index]:
                if (
                    event.event_id == previous_event.event_id
                    or event.event_sequence == previous_event.event_sequence
                ) and (
                    event != previous_event or state_digest != previous_state_digest
                ):
                    raise ValueError(
                        "retained event state witnesses disagree on event or state digest"
                    )

    @classmethod
    def from_canonical_value(cls, value: object) -> AttentionContinuity:
        _bounded_canonical_value_bytes(value)
        fields = _closed_object(
            value,
            frozenset(
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
            ),
            "AttentionContinuity",
        )
        candidate_values = _json_list(fields["candidates"], "candidates")
        revision_values = _json_list(fields["revision_history"], "revision_history")
        receipt_values = _json_list(fields["receipts"], "receipts")
        focused_values = _json_list(fields["focused_ids"], "focused_ids")
        unfinished_values = _json_list(fields["unfinished_ids"], "unfinished_ids")
        from suzka.attention.bounds import attention_candidate_capacity

        if len(candidate_values) > attention_candidate_capacity():
            raise ValueError("candidates exceeds its source-authority bound")
        if len(revision_values) > ATTENTION_MAX_REVISION_HISTORY:
            raise ValueError("revision_history exceeds its retained window")
        if len(receipt_values) > ATTENTION_MAX_EVENT_RECEIPTS:
            raise ValueError("receipts exceeds its retained window")
        if len(focused_values) > ATTENTION_MAX_FOCUS:
            raise ValueError("focused_ids exceeds its bound")
        if len(unfinished_values) > ATTENTION_MAX_FOCUS:
            raise ValueError("unfinished_ids exceeds its bound")

        last_event_value = fields["last_event"]
        revision_anchor_value = fields["revision_anchor"]
        receipt_anchor_value = fields["receipt_anchor"]
        schema_version = fields["schema_version"]
        policy_version = fields["policy_version"]
        if type(schema_version) is not int:
            raise TypeError("schema_version must be an exact integer")
        if type(policy_version) is not int:
            raise TypeError("policy_version must be an exact integer")
        result = cls(
            schema_version=schema_version,
            policy_version=policy_version,
            revision=bounded_nonnegative_int(fields["revision"], "revision"),
            last_event=None
            if last_event_value is None
            else AttentionEvent.from_canonical_value(last_event_value),
            candidates=tuple(
                AttentionCandidateContinuity.from_canonical_value(item)
                for item in candidate_values
            ),
            focused_ids=_canonical_id_tuple(tuple(focused_values), "focused_ids"),
            unfinished_ids=_canonical_id_tuple(tuple(unfinished_values), "unfinished_ids"),
            revision_history=tuple(
                AttentionRevisionEvidence.from_canonical_value(item)
                for item in revision_values
            ),
            receipts=tuple(
                AttentionEventReceipt.from_canonical_value(item)
                for item in receipt_values
            ),
            revision_anchor=None
            if revision_anchor_value is None
            else AttentionRevisionAnchor.from_canonical_value(revision_anchor_value),
            receipt_anchor=None
            if receipt_anchor_value is None
            else AttentionReceiptAnchor.from_canonical_value(receipt_anchor_value),
        )
        if result.state_digest != fields["state_digest"]:
            raise ValueError("Attention state digest does not match current state")
        if result.authority_digest != fields["authority_digest"]:
            raise ValueError("Attention authority digest does not match retained evidence")
        return result

    @classmethod
    def from_json(cls, value: bytes | str) -> AttentionContinuity:
        """Parse canonical JSON, rejecting duplicate keys and noncanonical bytes."""

        if type(value) is bytes:
            if len(value) > _ATTENTION_MAX_CANONICAL_VALUE_BYTES:
                raise ValueError("Attention JSON exceeds its persisted byte bound")
            try:
                text = value.decode("ascii")
            except UnicodeDecodeError as error:
                raise ValueError("Attention JSON must be ASCII-only") from error
        elif type(value) is str:
            if len(value) > _ATTENTION_MAX_CANONICAL_VALUE_BYTES:
                raise ValueError("Attention JSON exceeds its persisted byte bound")
            try:
                encoded = value.encode("ascii")
            except UnicodeEncodeError as error:
                raise ValueError("Attention JSON must be ASCII-only") from error
            if len(encoded) > _ATTENTION_MAX_CANONICAL_VALUE_BYTES:
                raise ValueError("Attention JSON exceeds its persisted byte bound")
            text = value
        else:
            raise TypeError("Attention JSON must be bytes or an exact string")
        _check_json_nesting(text)

        def reject_non_finite(value: str) -> object:
            raise _AttentionJSONInputError(
                f"Attention JSON contains a non-finite number: {value}"
            )

        def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, item in pairs:
                if key in result:
                    raise _AttentionJSONInputError(
                        "Attention JSON contains a duplicate object key"
                    )
                result[key] = item
            return result

        try:
            parsed = json.loads(
                text,
                object_pairs_hook=reject_duplicates,
                parse_constant=reject_non_finite,
            )
        except _AttentionJSONInputError:
            raise
        except (RecursionError, ValueError) as error:
            raise ValueError("Attention JSON is malformed or too deeply nested") from error
        result = cls.from_canonical_value(parsed)
        if result.canonical_bytes().decode("ascii") != text:
            raise ValueError("Attention JSON is not in canonical form")
        return result


__all__ = [
    "AttentionCandidateContinuity",
    "AttentionCandidateProjection",
    "AttentionContinuity",
    "AttentionEvent",
    "AttentionEventReceipt",
    "AttentionRevisionAnchor",
    "AttentionRevisionEvidence",
    "AttentionReceiptAnchor",
    "AttentionSignalVector",
    "AttentionSourceWitness",
    "AttentionTarget",
    "attention_state_digest",
]
