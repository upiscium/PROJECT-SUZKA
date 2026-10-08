"""Pure, immutable event-scoped metacognition contract shapes.

These values hold supplied measurements and typed provenance only.  They do
not derive, assess, authenticate, retain, or calibrate metacognitive state.
Digests are deterministic checksums, not producer authentication.  A trusted
U3 producer must own any future derivation and establish the authority of its
inputs; constructing one of these values alone establishes neither.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
import json
from typing import Final, TypeVar, cast

from suzka.attention.common import (
    ATTENTION_MAX_FOCUS,
    AttentionSourceKind,
    SourceDigestKind,
    bounded_fraction,
    bounded_nonnegative_int,
    canonical_datetime,
    canonical_json,
    digest_payload,
    exact_enum,
    parse_enum,
    parse_hex_fraction,
    parse_identifier,
    validate_digest,
)
from suzka.attention.contracts import AttentionEvent
from suzka.identifiers import MAX_IDENTIFIER_CODEPOINTS
from suzka.limits import MAX_PERSISTED_EVENT_SEQUENCE, MAX_PERSISTED_REVISION


METACOGNITION_MAX_EVIDENCE_WITNESSES: Final[int] = 32
METACOGNITION_MAX_REASON_CODES: Final[int] = 16

_ASSESSMENT_ID_DOMAIN: Final[bytes] = b"PROJECT-SUZKA:R14:METACOGNITION-ID:V1\0"
_ASSESSMENT_DIGEST_DOMAIN: Final[bytes] = (
    b"PROJECT-SUZKA:R14:METACOGNITION-ASSESSMENT:V1\0"
)
_EVIDENCE_WITNESS_DOMAIN: Final[bytes] = (
    b"PROJECT-SUZKA:R14:METACOGNITION-EVIDENCE-WITNESS:V1\0"
)
_EnumT = TypeVar("_EnumT", bound=Enum)
_OPTIONAL_REVISION_KINDS: Final[frozenset[AttentionSourceKind]] = frozenset(
    {AttentionSourceKind.EMOTION, AttentionSourceKind.APPRAISAL}
)


class _ClosedStrEnum(str, Enum):
    """Base for the small closed vocabularies in this contract."""


class EpistemicBoundary(_ClosedStrEnum):
    SUFFICIENT = "sufficient"
    UNCERTAIN = "uncertain"
    UNKNOWN = "unknown"
    CONTRADICTORY = "contradictory"


class EvidenceCondition(_ClosedStrEnum):
    SUPPORTING = "supporting"
    CONTRADICTORY = "contradictory"
    UNKNOWN = "unknown"


class SourceEventOrigin(_ClosedStrEnum):
    """Labels an optional source event as upstream or Attention-derived."""

    UPSTREAM_EVENT = "upstream_event"
    ATTENTION_EVENT = "attention_event"


class MetacognitiveReasonCode(_ClosedStrEnum):
    """Closed, non-freeform codes; these are not rationale or explanation text."""

    MISSING_EVIDENCE = "missing_evidence"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONTRADICTORY_EVIDENCE = "contradictory_evidence"
    UNOBSERVED_SOURCE = "unobserved_source"
    CONFLICTING_SOURCES = "conflicting_sources"
    LOW_CONFIDENCE = "low_confidence"
    STALE_SOURCE = "stale_source"
    FOCUS_UNCERTAIN = "focus_uncertain"
    LOAD_UNOBSERVED = "load_unobserved"
    SATURATION_UNOBSERVED = "saturation_unobserved"
    EMOTION_INFLUENCE_UNOBSERVED = "emotion_influence_unobserved"
    QUALITY_UNOBSERVED = "quality_unobserved"
    EVIDENCE_BOUNDARY = "evidence_boundary"
    PARTIAL_COVERAGE = "partial_coverage"
    LIMITED_PROVENANCE = "limited_provenance"
    PRODUCER_UNAVAILABLE = "producer_unavailable"


def _closed_object(value: object, keys: frozenset[str], name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be an exact JSON object")
    if set(value) != keys or any(type(key) is not str for key in value):
        raise ValueError(f"{name} has missing or unknown fields")
    return cast(dict[str, object], value)


def _json_list(value: object, name: str) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{name} must be a JSON array")
    return cast(list[object], value)


def _canonical_digest_tuple(
    value: object,
    name: str,
    *,
    maximum: int,
) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")
    if len(value) > maximum:
        raise ValueError(f"{name} exceeds its bound")
    checked = tuple(validate_digest(item, f"{name} item") for item in value)
    if checked != tuple(sorted(set(checked))):
        raise ValueError(f"{name} must be sorted and unique")
    return checked


def _parse_digest_list(value: object, name: str, *, maximum: int) -> tuple[str, ...]:
    items = _json_list(value, name)
    if len(items) > maximum:
        raise ValueError(f"{name} exceeds its bound")
    return _canonical_digest_tuple(
        tuple(items),
        name,
        maximum=maximum,
    )


def _event_not_future(source: AttentionEvent, current: AttentionEvent) -> None:
    if source.event_sequence > current.event_sequence:
        raise ValueError("source event is in the future of the assessment event")
    if source.occurred_at > current.occurred_at:
        raise ValueError("source event time is in the future of the assessment event")
    if source.event_sequence == current.event_sequence and source != current:
        raise ValueError("equal event sequences must identify the exact same event")


def _optional_fraction(value: object, name: str) -> float | None:
    return None if value is None else bounded_fraction(value, name)


@dataclass(frozen=True, slots=True)
class FocusAssessmentWitness:
    """Reference-only snapshot witness for the current Attention focus.

    The event may be the caller's current event before the corresponding
    Attention state has been persisted.  This shape intentionally does not
    demand an Attention receipt, history row, or canonical commit.
    """

    attention_revision: int
    attention_state_digest: str
    focused_ids: tuple[str, ...]
    event: AttentionEvent

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "attention_revision",
            bounded_nonnegative_int(
                self.attention_revision,
                "attention_revision",
                maximum=MAX_PERSISTED_REVISION,
            ),
        )
        object.__setattr__(
            self,
            "attention_state_digest",
            validate_digest(self.attention_state_digest, "attention_state_digest"),
        )
        object.__setattr__(
            self,
            "focused_ids",
            _canonical_digest_tuple(
                self.focused_ids,
                "focused_ids",
                maximum=ATTENTION_MAX_FOCUS,
            ),
        )
        if type(self.event) is not AttentionEvent:
            raise TypeError("event must be an exact AttentionEvent")
        self.event.__post_init__()
        if self.attention_revision == 0 and self.focused_ids:
            raise ValueError("revision-zero Attention focus must be empty")

    def canonical_value(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "attention_revision": self.attention_revision,
            "attention_state_digest": self.attention_state_digest,
            "event": self.event.canonical_value(),
            "focused_ids": list(self.focused_ids),
        }

    @classmethod
    def from_canonical_value(cls, value: object) -> FocusAssessmentWitness:
        fields = _closed_object(
            value,
            frozenset(
                {
                    "attention_revision",
                    "attention_state_digest",
                    "event",
                    "focused_ids",
                }
            ),
            "FocusAssessmentWitness",
        )
        return cls(
            attention_revision=bounded_nonnegative_int(
                fields["attention_revision"],
                "attention_revision",
                maximum=MAX_PERSISTED_REVISION,
            ),
            attention_state_digest=validate_digest(
                fields["attention_state_digest"], "attention_state_digest"
            ),
            focused_ids=_parse_digest_list(
                fields["focused_ids"], "focused_ids", maximum=ATTENTION_MAX_FOCUS
            ),
            event=AttentionEvent.from_canonical_value(fields["event"]),
        )


@dataclass(frozen=True, slots=True)
class MetacognitiveEvidenceWitness:
    """Opaque typed evidence reference, condition, provenance, and event binding.

    ``source_event`` is an exact shared AttentionEvent value, not a fabricated
    upstream event ID.  ``source_event_origin`` distinguishes an upstream
    event from an Attention-derived observation whenever one is available.
    A checksum and these caller-supplied labels do not authenticate a source.
    """

    source_kind: AttentionSourceKind
    reference: str
    source_revision: int | None
    digest: str
    digest_kind: SourceDigestKind
    condition: EvidenceCondition
    confidence_ceiling: float | None = None
    source_event: AttentionEvent | None = None
    source_event_origin: SourceEventOrigin | None = None
    witness_digest: str = field(init=False)

    def __post_init__(self) -> None:
        source_kind = exact_enum(self.source_kind, AttentionSourceKind, "source_kind")
        digest_kind = exact_enum(self.digest_kind, SourceDigestKind, "digest_kind")
        condition = exact_enum(self.condition, EvidenceCondition, "condition")
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "digest_kind", digest_kind)
        object.__setattr__(self, "condition", condition)
        object.__setattr__(self, "reference", parse_identifier(self.reference, "reference"))
        if self.source_revision is None:
            if source_kind not in _OPTIONAL_REVISION_KINDS:
                raise ValueError("source_revision is required for this source kind")
        else:
            object.__setattr__(
                self,
                "source_revision",
                bounded_nonnegative_int(
                    self.source_revision,
                    "source_revision",
                    maximum=MAX_PERSISTED_REVISION,
                ),
            )
        object.__setattr__(self, "digest", validate_digest(self.digest, "digest"))
        object.__setattr__(
            self,
            "confidence_ceiling",
            _optional_fraction(self.confidence_ceiling, "confidence_ceiling"),
        )
        if condition is not EvidenceCondition.SUPPORTING and self.confidence_ceiling is not None:
            raise ValueError("only supporting evidence may provide a confidence ceiling")
        if self.source_event is None:
            if self.source_event_origin is not None:
                raise ValueError("source event origin requires its exact source event")
        else:
            if type(self.source_event) is not AttentionEvent:
                raise TypeError("source_event must be an exact AttentionEvent or None")
            self.source_event.__post_init__()
            origin = exact_enum(
                self.source_event_origin, SourceEventOrigin, "source_event_origin"
            )
            expected_digest_kind = (
                SourceDigestKind.ATTENTION_PROJECTION
                if origin is SourceEventOrigin.ATTENTION_EVENT
                else SourceDigestKind.UPSTREAM_AUTHORITY
            )
            if digest_kind is not expected_digest_kind:
                raise ValueError("source event origin does not match its digest provenance")
        object.__setattr__(
            self,
            "witness_digest",
            digest_payload(_EVIDENCE_WITNESS_DOMAIN, self._payload()),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "condition": self.condition.value,
            "confidence_ceiling": None
            if self.confidence_ceiling is None
            else self.confidence_ceiling.hex(),
            "digest": self.digest,
            "digest_kind": self.digest_kind.value,
            "reference": self.reference,
            "source_event": None
            if self.source_event is None
            else self.source_event.canonical_value(),
            "source_event_origin": None
            if self.source_event_origin is None
            else self.source_event_origin.value,
            "source_kind": self.source_kind.value,
            "source_revision": self.source_revision,
        }

    def validate_for(self, event: AttentionEvent) -> None:
        if type(event) is not AttentionEvent:
            raise TypeError("event must be an exact AttentionEvent")
        event.__post_init__()
        self.__post_init__()
        if self.source_event is not None:
            _event_not_future(self.source_event, event)

    def canonical_value(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._payload(), "witness_digest": self.witness_digest}

    @classmethod
    def from_canonical_value(cls, value: object) -> MetacognitiveEvidenceWitness:
        fields = _closed_object(
            value,
            frozenset(
                {
                    "condition",
                    "confidence_ceiling",
                    "digest",
                    "digest_kind",
                    "reference",
                    "source_event",
                    "source_event_origin",
                    "source_kind",
                    "source_revision",
                    "witness_digest",
                }
            ),
            "MetacognitiveEvidenceWitness",
        )
        source_event_value = fields["source_event"]
        source_revision = fields["source_revision"]
        confidence_ceiling = fields["confidence_ceiling"]
        result = cls(
            source_kind=parse_enum(fields["source_kind"], AttentionSourceKind, "source_kind"),
            reference=parse_identifier(fields["reference"], "reference"),
            source_revision=None
            if source_revision is None
            else bounded_nonnegative_int(
                source_revision,
                "source_revision",
                maximum=MAX_PERSISTED_REVISION,
            ),
            digest=validate_digest(fields["digest"], "digest"),
            digest_kind=parse_enum(fields["digest_kind"], SourceDigestKind, "digest_kind"),
            condition=parse_enum(fields["condition"], EvidenceCondition, "condition"),
            confidence_ceiling=parse_hex_fraction(
                confidence_ceiling, "confidence_ceiling"
            ),
            source_event=None
            if source_event_value is None
            else AttentionEvent.from_canonical_value(source_event_value),
            source_event_origin=None
            if fields["source_event_origin"] is None
            else parse_enum(
                fields["source_event_origin"], SourceEventOrigin, "source_event_origin"
            ),
        )
        if result.witness_digest != fields["witness_digest"]:
            raise ValueError("evidence witness digest does not match its contents")
        return result


def _evidence_values(
    evidence: tuple[MetacognitiveEvidenceWitness, ...],
) -> list[dict[str, object]]:
    return [item.canonical_value() for item in evidence]


@dataclass(frozen=True, slots=True)
class MetacognitiveAssessment:
    """Caller-supplied event assessment shape; it is not an assessment producer."""

    event: AttentionEvent
    focus_witness: FocusAssessmentWitness
    evidence: tuple[MetacognitiveEvidenceWitness, ...] = ()
    evidence_sufficiency: float | None = None
    epistemic_boundary: EpistemicBoundary = EpistemicBoundary.UNKNOWN
    confidence: float | None = None
    cognitive_load: float | None = None
    attention_saturation: float | None = None
    emotion_influence: float | None = None
    cognitive_quality: float | None = None
    reason_codes: tuple[MetacognitiveReasonCode, ...] = (
        MetacognitiveReasonCode.MISSING_EVIDENCE,
    )
    assessment_id: str = field(init=False)
    assessment_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.event) is not AttentionEvent:
            raise TypeError("event must be an exact AttentionEvent")
        self.event.__post_init__()
        if type(self.focus_witness) is not FocusAssessmentWitness:
            raise TypeError("focus_witness must be an exact FocusAssessmentWitness")
        self.focus_witness.__post_init__()
        if self.focus_witness.event != self.event:
            raise ValueError("focus witness must bind the exact assessment event")
        if type(self.evidence) is not tuple:
            raise TypeError("evidence must be an exact tuple")
        if len(self.evidence) > METACOGNITION_MAX_EVIDENCE_WITNESSES:
            raise ValueError("evidence exceeds its witness bound")
        if any(type(item) is not MetacognitiveEvidenceWitness for item in self.evidence):
            raise TypeError("evidence must contain exact MetacognitiveEvidenceWitness values")
        for item in self.evidence:
            item.validate_for(self.event)
        witness_digests = tuple(item.witness_digest for item in self.evidence)
        if witness_digests != tuple(sorted(set(witness_digests))):
            raise ValueError("evidence witnesses must be uniquely sorted by witness digest")
        for name in (
            "evidence_sufficiency",
            "confidence",
            "cognitive_load",
            "attention_saturation",
            "emotion_influence",
            "cognitive_quality",
        ):
            object.__setattr__(
                self,
                name,
                _optional_fraction(getattr(self, name), name),
            )
        boundary = exact_enum(
            self.epistemic_boundary, EpistemicBoundary, "epistemic_boundary"
        )
        object.__setattr__(self, "epistemic_boundary", boundary)
        if type(self.reason_codes) is not tuple:
            raise TypeError("reason_codes must be an exact tuple")
        if len(self.reason_codes) > METACOGNITION_MAX_REASON_CODES:
            raise ValueError("reason_codes exceeds its bound")
        reasons = tuple(
            exact_enum(code, MetacognitiveReasonCode, "reason code")
            for code in self.reason_codes
        )
        if reasons != tuple(sorted(set(reasons), key=lambda code: code.value)):
            raise ValueError("reason_codes must be uniquely sorted")
        object.__setattr__(self, "reason_codes", reasons)

        observed = any(
            item.condition is not EvidenceCondition.UNKNOWN for item in self.evidence
        )
        supporting = tuple(
            item
            for item in self.evidence
            if item.condition is EvidenceCondition.SUPPORTING
        )
        contradictory = any(
            item.condition is EvidenceCondition.CONTRADICTORY for item in self.evidence
        )
        if boundary is EpistemicBoundary.SUFFICIENT and not supporting:
            raise ValueError("sufficient boundary requires supporting evidence")
        if boundary is EpistemicBoundary.UNCERTAIN and not observed:
            raise ValueError("uncertain boundary requires observed evidence")
        if contradictory and boundary is not EpistemicBoundary.CONTRADICTORY:
            raise ValueError("contradictory evidence requires its distinct boundary")
        if (
            MetacognitiveReasonCode.CONTRADICTORY_EVIDENCE in reasons
            and not contradictory
        ):
            raise ValueError("contradictory reason requires contradictory evidence")
        if boundary is EpistemicBoundary.CONTRADICTORY:
            if not contradictory:
                raise ValueError("contradictory boundary requires contradictory evidence")
            if MetacognitiveReasonCode.CONTRADICTORY_EVIDENCE not in reasons:
                raise ValueError("contradictory boundary requires its distinct reason code")
        if not observed:
            if boundary is not EpistemicBoundary.UNKNOWN:
                raise ValueError("unobserved evidence must remain epistemically unknown")
            if MetacognitiveReasonCode.MISSING_EVIDENCE not in reasons:
                raise ValueError("unobserved evidence requires the missing-evidence reason")
            if any(
                getattr(self, name) is not None
                for name in (
                    "evidence_sufficiency",
                    "confidence",
                    "cognitive_load",
                    "attention_saturation",
                    "emotion_influence",
                    "cognitive_quality",
                )
            ):
                raise ValueError("unobserved evidence cannot be encoded as a numeric value")
        if boundary is EpistemicBoundary.UNKNOWN:
            if self.confidence is not None:
                raise ValueError("unknown epistemic boundary must not carry confidence")
            if any(
                getattr(self, name) is not None
                for name in (
                    "evidence_sufficiency",
                    "cognitive_load",
                    "attention_saturation",
                    "emotion_influence",
                    "cognitive_quality",
                )
            ):
                raise ValueError("unknown epistemic boundary cannot carry numeric values")
        if self.confidence is not None:
            ceilings = tuple(
                item.confidence_ceiling
                for item in supporting
                if item.confidence_ceiling is not None
            )
            if not ceilings:
                raise ValueError("confidence requires a supporting typed confidence ceiling")
            if self.confidence > min(ceilings):
                raise ValueError("confidence exceeds its supporting evidence ceiling")

        object.__setattr__(self, "assessment_id", self._assessment_id())
        object.__setattr__(
            self,
            "assessment_digest",
            digest_payload(_ASSESSMENT_DIGEST_DOMAIN, self._payload()),
        )

    def _assessment_id(self) -> str:
        return digest_payload(_ASSESSMENT_ID_DOMAIN, self.event.canonical_value())

    def _payload(self) -> dict[str, object]:
        return {
            "attention_saturation": _fraction_value(self.attention_saturation),
            "cognitive_load": _fraction_value(self.cognitive_load),
            "cognitive_quality": _fraction_value(self.cognitive_quality),
            "confidence": _fraction_value(self.confidence),
            "emotion_influence": _fraction_value(self.emotion_influence),
            "epistemic_boundary": self.epistemic_boundary.value,
            "event": self.event.canonical_value(),
            "evidence": _evidence_values(self.evidence),
            "evidence_sufficiency": _fraction_value(self.evidence_sufficiency),
            "focus_witness": self.focus_witness.canonical_value(),
            "reason_codes": [code.value for code in self.reason_codes],
        }

    def canonical_value(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._payload(),
            "assessment_digest": self.assessment_digest,
            "assessment_id": self.assessment_id,
        }

    def canonical_bytes(self) -> bytes:
        value = self.canonical_value()
        encoded = canonical_json(value)
        if len(encoded) > METACOGNITION_MAX_ASSESSMENT_BYTES:
            raise ValueError("metacognitive assessment exceeds its schema envelope")
        return encoded

    @classmethod
    def from_canonical_value(cls, value: object) -> MetacognitiveAssessment:
        fields = _closed_object(
            value,
            frozenset(
                {
                    "assessment_digest",
                    "assessment_id",
                    "attention_saturation",
                    "cognitive_load",
                    "cognitive_quality",
                    "confidence",
                    "emotion_influence",
                    "epistemic_boundary",
                    "event",
                    "evidence",
                    "evidence_sufficiency",
                    "focus_witness",
                    "reason_codes",
                }
            ),
            "MetacognitiveAssessment",
        )
        evidence_values = _json_list(fields["evidence"], "evidence")
        if len(evidence_values) > METACOGNITION_MAX_EVIDENCE_WITNESSES:
            raise ValueError("evidence exceeds its witness bound")
        reason_values = _json_list(fields["reason_codes"], "reason_codes")
        if len(reason_values) > METACOGNITION_MAX_REASON_CODES:
            raise ValueError("reason_codes exceeds its bound")
        evidence = tuple(
            MetacognitiveEvidenceWitness.from_canonical_value(item)
            for item in evidence_values
        )
        result = cls(
            event=AttentionEvent.from_canonical_value(fields["event"]),
            focus_witness=FocusAssessmentWitness.from_canonical_value(
                fields["focus_witness"]
            ),
            evidence=evidence,
            evidence_sufficiency=parse_hex_fraction(
                fields["evidence_sufficiency"], "evidence_sufficiency"
            ),
            epistemic_boundary=parse_enum(
                fields["epistemic_boundary"], EpistemicBoundary, "epistemic_boundary"
            ),
            confidence=parse_hex_fraction(fields["confidence"], "confidence"),
            cognitive_load=parse_hex_fraction(fields["cognitive_load"], "cognitive_load"),
            attention_saturation=parse_hex_fraction(
                fields["attention_saturation"], "attention_saturation"
            ),
            emotion_influence=parse_hex_fraction(
                fields["emotion_influence"], "emotion_influence"
            ),
            cognitive_quality=parse_hex_fraction(
                fields["cognitive_quality"], "cognitive_quality"
            ),
            reason_codes=tuple(
                parse_enum(code, MetacognitiveReasonCode, "reason code")
                for code in reason_values
            ),
        )
        if result.assessment_id != fields["assessment_id"]:
            raise ValueError("assessment ID does not match its event")
        if result.assessment_digest != fields["assessment_digest"]:
            raise ValueError("assessment digest does not match its contents")
        return result

    @classmethod
    def from_json(cls, value: bytes | str) -> MetacognitiveAssessment:
        """Parse closed canonical JSON, rejecting duplicate keys and extra bytes."""

        if type(value) is bytes:
            try:
                text = value.decode("ascii")
            except UnicodeDecodeError as error:
                raise ValueError("metacognitive assessment JSON must be ASCII-only") from error
        elif type(value) is str:
            text = value
            try:
                text.encode("ascii")
            except UnicodeEncodeError as error:
                raise ValueError("metacognitive assessment JSON must be ASCII-only") from error
        else:
            raise TypeError("metacognitive assessment JSON must be bytes or an exact string")
        if len(text) > METACOGNITION_MAX_ASSESSMENT_BYTES:
            raise ValueError("metacognitive assessment JSON exceeds its schema envelope")

        def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError("metacognitive JSON contains a duplicate object key")
                result[key] = item
            return result

        def reject_constant(value: str) -> object:
            raise ValueError(f"metacognitive JSON contains invalid constant {value}")

        try:
            parsed = json.loads(
                text,
                object_pairs_hook=reject_duplicates,
                parse_constant=reject_constant,
            )
        except (json.JSONDecodeError, UnicodeEncodeError) as error:
            raise ValueError("metacognitive assessment JSON is malformed") from error
        result = cls.from_canonical_value(parsed)
        if result.canonical_bytes().decode("ascii") != text:
            raise ValueError("metacognitive assessment JSON is not in canonical form")
        return result


def _fraction_value(value: float | None) -> str | None:
    return None if value is None else value.hex()


def _longest_enum_value(enum_type: type[_EnumT]) -> str:
    return max((item.value for item in enum_type), key=lambda item: len(canonical_json(item)))


def _maximum_event_value() -> dict[str, object]:
    return {
        "event_id": "e" * MAX_IDENTIFIER_CODEPOINTS,
        "event_sequence": MAX_PERSISTED_EVENT_SEQUENCE,
        "occurred_at": canonical_datetime(datetime.max.replace(tzinfo=UTC)),
    }


def _maximum_fraction_hex() -> str:
    # For a finite binary64 fraction in [0, 1], a positive subnormal has the
    # longest canonical float.hex() form (fixed mantissa, four-digit exponent).
    smallest_positive = float.fromhex("0x0.0000000000001p-1022")
    return bounded_fraction(smallest_positive, "maximum fraction").hex()


def _maximum_evidence_row_bytes() -> int:
    event_value = _maximum_event_value()
    source_kind = _longest_enum_value(AttentionSourceKind)
    max_fraction = _maximum_fraction_hex()
    rows: list[dict[str, object]] = []
    for digest_kind, event_origin in (
        (SourceDigestKind.UPSTREAM_AUTHORITY, SourceEventOrigin.UPSTREAM_EVENT),
        (SourceDigestKind.ATTENTION_PROJECTION, SourceEventOrigin.ATTENTION_EVENT),
    ):
        for condition in EvidenceCondition:
            ceilings: tuple[str | None, ...] = (
                (None, max_fraction)
                if condition is EvidenceCondition.SUPPORTING
                else (None,)
            )
            for ceiling in ceilings:
                rows.append(
                    {
                        "condition": condition.value,
                        "confidence_ceiling": ceiling,
                        "digest": "f" * 64,
                        "digest_kind": digest_kind.value,
                        "reference": "a" * MAX_IDENTIFIER_CODEPOINTS,
                        "source_event": event_value,
                        "source_event_origin": event_origin.value,
                        "source_kind": source_kind,
                        "source_revision": MAX_PERSISTED_REVISION,
                        "witness_digest": "f" * 64,
                    }
                )
    return max(len(canonical_json(row)) for row in rows)


def derive_metacognition_assessment_max_bytes() -> int:
    """Derive a conservative canonical envelope from this schema's own bounds.

    This is an event-scoped serialization ceiling, not a persisted AgentState
    capacity.  Each closed field is sized using its longest legal identifier,
    enum, digest, timestamp, sequence/revision, fraction, or bounded array.
    Independent maxima are composed conservatively; the resulting envelope
    need not itself be a semantically valid assessment value.
    """

    event_value = _maximum_event_value()
    max_digest = "f" * 64
    focus_value = {
        "attention_revision": MAX_PERSISTED_REVISION,
        "attention_state_digest": max_digest,
        "event": event_value,
        "focused_ids": [f"{index:064x}" for index in range(ATTENTION_MAX_FOCUS)],
    }
    evidence_row = {
        "condition": _longest_enum_value(EvidenceCondition),
        "confidence_ceiling": _maximum_fraction_hex(),
        "digest": max_digest,
        "digest_kind": SourceDigestKind.ATTENTION_PROJECTION.value,
        "reference": "a" * MAX_IDENTIFIER_CODEPOINTS,
        "source_event": event_value,
        "source_event_origin": SourceEventOrigin.ATTENTION_EVENT.value,
        "source_kind": _longest_enum_value(AttentionSourceKind),
        "source_revision": MAX_PERSISTED_REVISION,
        "witness_digest": max_digest,
    }
    if len(canonical_json(evidence_row)) < _maximum_evidence_row_bytes():
        raise AssertionError("composed evidence row is below its legal schema maximum")
    fraction = _maximum_fraction_hex()
    assessment_value = {
        "assessment_digest": max_digest,
        "assessment_id": max_digest,
        "attention_saturation": fraction,
        "cognitive_load": fraction,
        "cognitive_quality": fraction,
        "confidence": fraction,
        "emotion_influence": fraction,
        "epistemic_boundary": _longest_enum_value(EpistemicBoundary),
        "event": event_value,
        "evidence": [evidence_row]
        * METACOGNITION_MAX_EVIDENCE_WITNESSES,
        "evidence_sufficiency": fraction,
        "focus_witness": focus_value,
        "reason_codes": [
            _longest_enum_value(MetacognitiveReasonCode)
        ]
        * METACOGNITION_MAX_REASON_CODES,
    }
    return len(canonical_json(assessment_value))


METACOGNITION_MAX_ASSESSMENT_BYTES: Final[int] = (
    derive_metacognition_assessment_max_bytes()
)


__all__ = [
    "EpistemicBoundary",
    "EvidenceCondition",
    "FocusAssessmentWitness",
    "METACOGNITION_MAX_ASSESSMENT_BYTES",
    "METACOGNITION_MAX_EVIDENCE_WITNESSES",
    "METACOGNITION_MAX_REASON_CODES",
    "MetacognitiveAssessment",
    "MetacognitiveEvidenceWitness",
    "MetacognitiveReasonCode",
    "SourceEventOrigin",
    "derive_metacognition_assessment_max_bytes",
]
