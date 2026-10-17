"""Closed, immutable R15 reference and interpretation vocabulary.

These checks validate SHAPE only. Neither a digest nor a constructed witness
authenticates a producer, commits an Experience, or adopts a subjective claim.
Only a future ordered runtime owner may compare a witness with a trusted root.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
import hashlib
import json
from typing import Final, TypeVar, cast

from suzka.identifiers import validate_identifier
from suzka.limits import MAX_PERSISTED_EVENT_SEQUENCE, MAX_PERSISTED_REVISION


SCHEMA_VERSION: Final = 1
MAX_RELATIONSHIPS: Final = 64
MAX_EPISODES: Final = 64
MAX_NARRATIVE_CLAIMS: Final = 32
MAX_SELF_CLAIMS: Final = 32
MAX_CAPABILITIES: Final = 16
MAX_LINKS: Final = 4
MAX_SUPPORT: Final = 4
MAX_COUNTER: Final = 4
MAX_REVISIONS: Final = 64
SCORE_SCALE: Final = 1_000_000
MAX_JSON_NESTING: Final = 32
_EnumT = TypeVar("_EnumT", bound=Enum)


class SourceKind(str, Enum):
    CONTEXT = "context"
    BINDING = "interlocutor_binding"
    EMOTION = "emotion"
    APPRAISAL = "appraisal"
    VALUE = "value"
    BELIEF = "belief"
    EXPERIENCE = "experience"
    MOTIVATION = "motivation"
    GOAL = "goal"
    COMMITMENT = "commitment"
    ATTENTION = "attention"
    METACOGNITION = "metacognition"
    OPERATOR_CLAIM = "operator_claim"
    MODEL_TEXT = "model_text"
    WEB_TEXT = "web_text"
    VERIFIED_OUTCOME = "verified_outcome"


class SourceLifecycle(str, Enum):
    ACTIVE = "active"
    ADOPTED = "adopted"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"
    UNAVAILABLE = "unavailable"
    DECLARATIVE = "declarative"


class ClaimedOrigin(str, Enum):
    SUBJECT_OBSERVATION = "subject_observation"
    DECLARATIVE = "declarative"
    OPERATOR_ASSERTION = "operator_assertion"
    MODEL_ASSERTION = "model_assertion"
    WEB_ASSERTION = "web_assertion"
    EXTERNAL_ASSERTION = "external_assertion"


class Concept(str, Enum):
    INTERLOCUTOR = "interlocutor"
    CLAIMED_IDENTITY = "claimed_identity"
    VERIFIED_PERSON = "verified_person"
    RELATIONSHIP = "relationship"
    NARRATIVE_EPISODE = "narrative_episode"
    NARRATIVE_CLAIM = "narrative_claim"
    SELF_HYPOTHESIS = "self_hypothesis"
    TASK_ATTEMPT = "task_attempt"
    VERIFIED_COMPETENCE = "verified_competence"
    INTERACTION_ASSOCIATION = "interaction_association"
    ACTOR_CAUSATION = "actor_causation"


class SourceDisposition(str, Enum):
    UNAVAILABLE = "unavailable"
    REQUIRES_TRUSTED_ROOT = "requires_trusted_root"


class UntrustedIngress(str, Enum):
    OPERATOR_API = "operator_api"
    PRIVILEGED_CORRECTION = "privileged_correction"
    INTERNAL_HANDLER = "internal_handler"
    SERIALIZED_STATE_PORT = "serialized_state_port"
    REPEATED_CLAIM = "repeated_claim"
    FORGED_SUBJECT_ADMISSION = "forged_subject_admission"
    MIGRATION = "migration"
    TECHNICAL_RESTORE = "technical_restore"
    MODEL_TEXT = "model_text"
    WEB_TEXT = "web_text"


def asserted_intrinsic_disposition(concept: Concept, ingress: UntrustedIngress) -> SourceDisposition:
    """A transport or self-endorsement *assertion* cannot adopt intrinsic truth.

    Technical restoration has its own exact-state compatibility gate in U5;
    it is never an assertion-based cognitive revision.
    """

    exact_enum(concept, Concept, "concept")
    exact_enum(ingress, UntrustedIngress, "ingress")
    return SourceDisposition.UNAVAILABLE


class InterpretationStatus(str, Enum):
    UNKNOWN = "unknown"
    PRESENT = "present"
    CONTESTED = "contested"
    CONTRADICTORY = "contradictory"


class Accessibility(str, Enum):
    CURRENT = "current"
    LATENT = "latent"
    REACTIVATABLE = "reactivatable"


class Inertia(str, Enum):
    TENTATIVE = "tentative"
    REVISABLE = "revisable"
    RESISTANT = "resistant"


class RevisionReason(str, Enum):
    INITIAL_INTERPRETATION = "initial_interpretation"
    QUALIFIED_REINTERPRETATION = "qualified_reinterpretation"
    CONTRARY_EVIDENCE = "contrary_evidence"
    ACCESSIBILITY_REVIEW = "accessibility_review"
    QUALIFIED_REACTIVATION = "qualified_reactivation"
    SOURCE_INELIGIBLE = "source_ineligible"


def exact_enum(value: object, kind: type[_EnumT], name: str) -> _EnumT:
    if type(value) is not kind:
        raise TypeError(f"{name} must be a {kind.__name__}")
    return cast(_EnumT, value)


def parse_enum(value: object, kind: type[_EnumT], name: str) -> _EnumT:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    try:
        return kind(value)
    except ValueError as error:
        raise ValueError(f"{name} is not a supported value") from error


def integer(value: object, name: str, *, maximum: int, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an exact bounded integer")
    return value


def identifier(value: object, name: str) -> str:
    try:
        return validate_identifier(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not a bounded opaque identifier") from error


def digest(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be 64 lowercase hex characters")
    return value


def utc(value: object, name: str) -> datetime:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a UTC datetime")
    offset = value.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise ValueError(f"{name} must be UTC, not another offset")
    return value


def utc_text(value: datetime) -> str:
    return utc(value, "time").astimezone(UTC).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def parse_utc(value: object, name: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise ValueError(f"{name} must be canonical UTC")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} is not a timestamp") from error
    if utc_text(result) != value:
        raise ValueError(f"{name} must have canonical microsecond precision")
    return result


def closed(value: object, names: tuple[str, ...], name: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != set(names) or any(
        type(key) is not str for key in value
    ):
        raise ValueError(f"{name} must have exactly its declared fields")
    return value


def sequence(value: object, name: str, maximum: int, *, json_input: bool) -> tuple[object, ...]:
    if json_input:
        if type(value) is not list:
            raise TypeError(f"{name} must be a JSON array")
        values = tuple(cast(list[object], value))
    else:
        if type(value) is not tuple:
            raise TypeError(f"{name} must be a tuple")
        values = cast(tuple[object, ...], value)
    if len(values) > maximum:
        raise ValueError(f"{name} exceeds its bound")
    return values


def encode(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if type(value) is datetime:
        return utc_text(value)
    if type(value) is tuple:
        return [encode(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: encode(getattr(value, item.name)) for item in fields(value)}
    return value


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def checksum(domain: bytes, value: object) -> str:
    # Checksums bind bytes, never a producer, root ownership or subject adoption.
    return hashlib.sha256(domain + canonical_json(value)).hexdigest()


def parse_json(payload: bytes, maximum: int) -> object:
    if type(payload) is not bytes or len(payload) > maximum:
        raise ValueError("R15 JSON exceeds its declared byte bound")
    text = payload.decode("utf-8")
    depth = 0
    quoted = False
    escaped = False
    for character in text:
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character in "[{":
            depth += 1
            if depth > MAX_JSON_NESTING:
                raise ValueError("R15 JSON nesting exceeds its bound")
        elif character in "]}":
            depth -= 1

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate R15 JSON key")
            result[key] = item
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"nonfinite R15 JSON constant: {value}")

    try:
        result = json.loads(text, object_pairs_hook=unique, parse_constant=reject_constant)
        if canonical_json(result) != payload:
            raise ValueError("R15 JSON must use its exact canonical encoding")
        return result
    except (RecursionError, OverflowError) as error:
        raise ValueError("invalid R15 JSON nesting") from error


@dataclass(frozen=True, slots=True)
class EventRef:
    event_id: str
    event_sequence: int
    occurred_at: datetime

    def __post_init__(self) -> None:
        identifier(self.event_id, "event_id")
        integer(self.event_sequence, "event_sequence", minimum=1, maximum=MAX_PERSISTED_EVENT_SEQUENCE)
        utc(self.occurred_at, "occurred_at")

    @classmethod
    def from_value(cls, value: object) -> EventRef:
        row = closed(value, ("event_id", "event_sequence", "occurred_at"), "EventRef")
        return cls(
            identifier(row["event_id"], "event_id"),
            integer(row["event_sequence"], "event_sequence", minimum=1, maximum=MAX_PERSISTED_EVENT_SEQUENCE),
            parse_utc(row["occurred_at"], "occurred_at"),
        )


@dataclass(frozen=True, slots=True)
class SourceWitness:
    """A declarative source tuple; its IDs, claimed origin and digests are NOT proof."""

    kind: SourceKind
    reference: str
    revision: int
    source_digest: str
    root_digest: str
    lifecycle: SourceLifecycle
    origin: ClaimedOrigin
    event: EventRef | None

    def __post_init__(self) -> None:
        exact_enum(self.kind, SourceKind, "kind")
        identifier(self.reference, "reference")
        integer(self.revision, "revision", maximum=MAX_PERSISTED_REVISION)
        digest(self.source_digest, "source_digest")
        digest(self.root_digest, "root_digest")
        exact_enum(self.lifecycle, SourceLifecycle, "lifecycle")
        exact_enum(self.origin, ClaimedOrigin, "origin")
        if self.event is not None and type(self.event) is not EventRef:
            raise TypeError("event must be an exact EventRef")
        if self.kind is SourceKind.EXPERIENCE and self.event is None:
            raise ValueError("Experience witness requires event and episode lineage")

    @classmethod
    def from_value(cls, value: object) -> SourceWitness:
        row = closed(value, (
            "kind", "reference", "revision", "source_digest", "root_digest", "lifecycle", "origin", "event"
        ), "SourceWitness")
        return cls(
            parse_enum(row["kind"], SourceKind, "kind"),
            identifier(row["reference"], "reference"),
            integer(row["revision"], "revision", maximum=MAX_PERSISTED_REVISION),
            digest(row["source_digest"], "source_digest"),
            digest(row["root_digest"], "root_digest"),
            parse_enum(row["lifecycle"], SourceLifecycle, "lifecycle"),
            parse_enum(row["origin"], ClaimedOrigin, "origin"),
            None if row["event"] is None else EventRef.from_value(row["event"]),
        )


_POTENTIAL_SOURCES: Final[dict[Concept, frozenset[SourceKind]]] = {
    Concept.INTERLOCUTOR: frozenset({SourceKind.CONTEXT}),
    Concept.CLAIMED_IDENTITY: frozenset({SourceKind.BINDING}),
    Concept.VERIFIED_PERSON: frozenset(),
    Concept.RELATIONSHIP: frozenset({SourceKind.EXPERIENCE}),
    Concept.NARRATIVE_EPISODE: frozenset({SourceKind.EXPERIENCE}),
    Concept.NARRATIVE_CLAIM: frozenset({SourceKind.EXPERIENCE, SourceKind.BELIEF}),
    Concept.SELF_HYPOTHESIS: frozenset({SourceKind.EXPERIENCE}),
    Concept.TASK_ATTEMPT: frozenset({SourceKind.EXPERIENCE}),
    Concept.VERIFIED_COMPETENCE: frozenset(),
    Concept.INTERACTION_ASSOCIATION: frozenset({SourceKind.CONTEXT, SourceKind.EMOTION}),
    Concept.ACTOR_CAUSATION: frozenset(),
}


def source_disposition(concept: Concept, witness: SourceWitness) -> SourceDisposition:
    """Return only an obligation, never an adoption or verified-source result."""

    exact_enum(concept, Concept, "concept")
    if type(witness) is not SourceWitness:
        raise TypeError("witness must be an exact SourceWitness")
    witness.__post_init__()
    if witness.kind not in _POTENTIAL_SOURCES[concept]:
        return SourceDisposition.UNAVAILABLE
    if witness.kind is SourceKind.EXPERIENCE and (
        witness.lifecycle is not SourceLifecycle.ACTIVE
        or witness.origin is not ClaimedOrigin.SUBJECT_OBSERVATION
    ):
        return SourceDisposition.UNAVAILABLE
    if witness.kind is SourceKind.BELIEF and (
        witness.lifecycle is not SourceLifecycle.ADOPTED
        or witness.origin is not ClaimedOrigin.SUBJECT_OBSERVATION
    ):
        return SourceDisposition.UNAVAILABLE
    if witness.kind is SourceKind.CONTEXT and witness.lifecycle is not SourceLifecycle.ACTIVE:
        return SourceDisposition.UNAVAILABLE
    if witness.kind is SourceKind.BINDING and witness.lifecycle is not SourceLifecycle.DECLARATIVE:
        return SourceDisposition.UNAVAILABLE
    if witness.kind is SourceKind.EMOTION and (
        witness.lifecycle is not SourceLifecycle.ACTIVE or witness.event is None
    ):
        return SourceDisposition.UNAVAILABLE
    if witness.origin in (
        ClaimedOrigin.OPERATOR_ASSERTION,
        ClaimedOrigin.MODEL_ASSERTION,
        ClaimedOrigin.WEB_ASSERTION,
        ClaimedOrigin.EXTERNAL_ASSERTION,
    ):
        return SourceDisposition.UNAVAILABLE
    # This return is NOT 'eligible'. Future U2-U4 must fetch the trusted
    # producer root and verify full lifecycle, finalized commit, event and
    # independent contrary evidence, origin and subject admission atomically.
    return SourceDisposition.REQUIRES_TRUSTED_ROOT


@dataclass(frozen=True, slots=True)
class Interpretation:
    statement_id: str
    status: InterpretationStatus
    estimate: int | None
    confidence: int | None
    support: tuple[SourceWitness, ...]
    counter: tuple[SourceWitness, ...]
    accessibility: Accessibility
    inertia: Inertia

    def __post_init__(self) -> None:
        digest(self.statement_id, "statement_id")
        exact_enum(self.status, InterpretationStatus, "status")
        exact_enum(self.accessibility, Accessibility, "accessibility")
        exact_enum(self.inertia, Inertia, "inertia")
        if self.estimate is not None:
            integer(self.estimate, "estimate", maximum=SCORE_SCALE)
        if self.confidence is not None:
            integer(self.confidence, "confidence", maximum=SCORE_SCALE)
        for name, values, limit in (("support", self.support, MAX_SUPPORT), ("counter", self.counter, MAX_COUNTER)):
            if type(values) is not tuple or len(values) > limit or any(type(item) is not SourceWitness for item in values):
                raise ValueError(f"{name} must be a bounded exact witness tuple")
            for item in values:
                item.__post_init__()
        identities = [(item.kind, item.reference, item.revision, item.root_digest) for item in self.support + self.counter]
        if len(set(identities)) != len(identities):
            raise ValueError("one source occurrence cannot corroborate or oppose itself")
        if self.status is InterpretationStatus.UNKNOWN:
            if self.estimate is not None or self.confidence is not None or self.support or self.counter:
                raise ValueError("unknown is missing evidence, not zero confidence")
        elif self.status is InterpretationStatus.PRESENT:
            if self.estimate is None or self.confidence is None or not self.support or self.counter:
                raise ValueError("present needs witnessed estimate and confidence")
        elif self.status is InterpretationStatus.CONTESTED:
            if self.estimate is None or self.confidence is None or not self.support or not self.counter:
                raise ValueError("contested needs retained support and contrary evidence")
        elif not self.support or not self.counter or self.estimate is not None or self.confidence is not None:
            raise ValueError("contradictory retains conflict without invented numeric certainty")

    def require_sources(self, concept: Concept) -> None:
        for item in self.support + self.counter:
            if source_disposition(concept, item) is SourceDisposition.UNAVAILABLE:
                raise ValueError("source shape cannot support this interpretation")

    @classmethod
    def from_value(cls, value: object) -> Interpretation:
        row = closed(value, (
            "statement_id", "status", "estimate", "confidence", "support", "counter", "accessibility", "inertia"
        ), "Interpretation")
        support = sequence(row["support"], "support", MAX_SUPPORT, json_input=True)
        counter = sequence(row["counter"], "counter", MAX_COUNTER, json_input=True)
        return cls(
            digest(row["statement_id"], "statement_id"),
            parse_enum(row["status"], InterpretationStatus, "status"),
            None if row["estimate"] is None else integer(row["estimate"], "estimate", maximum=SCORE_SCALE),
            None if row["confidence"] is None else integer(row["confidence"], "confidence", maximum=SCORE_SCALE),
            tuple(SourceWitness.from_value(item) for item in support),
            tuple(SourceWitness.from_value(item) for item in counter),
            parse_enum(row["accessibility"], Accessibility, "accessibility"),
            parse_enum(row["inertia"], Inertia, "inertia"),
        )


@dataclass(frozen=True, slots=True)
class RevisionProof:
    """Bounded interpretation revision checksum, not source authentication."""

    revision: int
    event: EventRef
    target_id: str
    before_digest: str
    after_digest: str
    support_digests: tuple[str, ...]
    counter_digests: tuple[str, ...]
    reason: RevisionReason
    previous_digest: str | None

    def __post_init__(self) -> None:
        integer(self.revision, "revision", minimum=1, maximum=MAX_PERSISTED_REVISION)
        if type(self.event) is not EventRef:
            raise TypeError("revision event must be exact")
        self.event.__post_init__()
        for name in ("target_id", "before_digest", "after_digest"):
            digest(getattr(self, name), name)
        if self.previous_digest is not None:
            digest(self.previous_digest, "previous_digest")
        exact_enum(self.reason, RevisionReason, "reason")
        for name, maximum in (("support_digests", MAX_SUPPORT), ("counter_digests", MAX_COUNTER)):
            items = getattr(self, name)
            if type(items) is not tuple or len(items) > maximum:
                raise ValueError(f"{name} exceeds its proof bound")
            for item in items:
                digest(item, name)
            if items != tuple(sorted(set(items))):
                raise ValueError(f"{name} must be ordered and unique")

    @property
    def record_digest(self) -> str:
        return checksum(b"PROJECT-SUZKA:R15:INTERPRETATION-REVISION:V1\0", encode(self))

    @classmethod
    def from_value(cls, value: object) -> RevisionProof:
        row = closed(value, (
            "revision", "event", "target_id", "before_digest", "after_digest", "support_digests", "counter_digests", "reason", "previous_digest"
        ), "RevisionProof")
        return cls(
            integer(row["revision"], "revision", minimum=1, maximum=MAX_PERSISTED_REVISION),
            EventRef.from_value(row["event"]),
            digest(row["target_id"], "target_id"),
            digest(row["before_digest"], "before_digest"),
            digest(row["after_digest"], "after_digest"),
            tuple(digest(item, "support_digest") for item in sequence(row["support_digests"], "support_digests", MAX_SUPPORT, json_input=True)),
            tuple(digest(item, "counter_digest") for item in sequence(row["counter_digests"], "counter_digests", MAX_COUNTER, json_input=True)),
            parse_enum(row["reason"], RevisionReason, "reason"),
            None if row["previous_digest"] is None else digest(row["previous_digest"], "previous_digest"),
        )


def validate_history(revision: int, anchor: str | None, history: tuple[RevisionProof, ...]) -> None:
    integer(revision, "root revision", maximum=MAX_PERSISTED_REVISION)
    if anchor is not None:
        digest(anchor, "history anchor")
    if type(history) is not tuple or len(history) > MAX_REVISIONS:
        raise ValueError("revision history exceeds its retained proof window")
    if not history:
        if revision != 0 or anchor is not None:
            raise ValueError("nonempty history is required after a revision")
        return
    if history[0].revision != revision - len(history) + 1 or (anchor is None) != (history[0].revision == 1):
        raise ValueError("revision suffix and anchor do not match")
    previous = anchor
    prior_event: EventRef | None = None
    for position, item in enumerate(history):
        if type(item) is not RevisionProof or item.revision != history[0].revision + position or item.previous_digest != previous:
            raise ValueError("revision proof chain is incomplete")
        if prior_event is not None and (item.event.event_sequence <= prior_event.event_sequence or item.event.occurred_at < prior_event.occurred_at):
            raise ValueError("revision event is out of order")
        previous = item.record_digest
        prior_event = item.event


def root_value(value: object, domain: bytes) -> dict[str, object]:
    payload = encode(value)
    if type(payload) is not dict:
        raise TypeError("root is not a closed contract")
    return {**payload, "state_digest": checksum(domain, payload)}


def check_root_digest(value: dict[str, object], domain: bytes) -> None:
    claimed = digest(value["state_digest"], "state_digest")
    if claimed != checksum(domain, {key: item for key, item in value.items() if key != "state_digest"}):
        raise ValueError("R15 root checksum does not bind its exact canonical fields")
