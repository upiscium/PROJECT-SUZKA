"""D6 pure same-event felt association, never a lasting or actor-causal claim.

U1 deliberately provides no builder, prompt/style consumer, timer or persistence
port. A later caller must fetch exact R09/R10 current serialized event snapshots
and (if supplied) a committed R15 prior projection before constructing this value.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from suzka.r15.common import (
    SCORE_SCALE,
    Concept,
    EventRef,
    SourceDisposition,
    SourceKind,
    SourceWitness,
    canonical_json,
    checksum,
    closed,
    digest,
    encode,
    exact_enum,
    identifier,
    integer,
    parse_enum,
    source_disposition,
)


class FeltAssociation(str, Enum):
    UNKNOWN = "unknown"
    TENTATIVE_INTERACTION = "tentative_interaction"


class ActorCausation(str, Enum):
    UNKNOWN = "unknown"


def _hex_float(value: object, name: str, *, minimum: float, maximum: float) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be canonical binary64 hex text")
    try:
        number = float.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{name} must be canonical binary64 hex text") from error
    if not math.isfinite(number) or not minimum <= number <= maximum or number.hex() != value:
        raise ValueError(f"{name} must be finite, canonical and within its range")
    return value


@dataclass(frozen=True, slots=True)
class InteractionStance:
    event: EventRef
    context: SourceWitness
    emotion: SourceWitness
    context_participant_refs: tuple[str, ...]
    interlocutor_key: str | None
    valence_hex: str
    arousal_hex: str
    optimal_loss_hex: str
    prior_committed_relationship_digest: str | None
    association: FeltAssociation
    uncertainty: int | None
    actor_causation: ActorCausation

    def __post_init__(self) -> None:
        if type(self.event) is not EventRef or type(self.context) is not SourceWitness or type(self.emotion) is not SourceWitness:
            raise TypeError("stance event and witnesses must be exact immutable values")
        self.event.__post_init__()
        self.context.__post_init__()
        self.emotion.__post_init__()
        if self.context.kind is not SourceKind.CONTEXT or self.emotion.kind is not SourceKind.EMOTION:
            raise ValueError("stance uses only current Context and global R10 Emotion")
        if self.emotion.event != self.event or self.context.event != self.event:
            raise ValueError("stance witnesses must belong to the exact current event")
        if any(
            source_disposition(Concept.INTERACTION_ASSOCIATION, witness)
            is SourceDisposition.UNAVAILABLE
            for witness in (self.context, self.emotion)
        ):
            raise ValueError("unavailable or asserted input cannot be a stance source")
        if type(self.context_participant_refs) is not tuple or len(self.context_participant_refs) > 32:
            raise ValueError("context participants exceed the R09 bound")
        for participant in self.context_participant_refs:
            identifier(participant, "participant ref")
        if self.context_participant_refs != tuple(sorted(set(self.context_participant_refs))):
            raise ValueError("context participant refs must be canonical and unique")
        if self.context.source_digest != context_projection_checksum(
            self.event, self.context.reference, self.context.revision, self.context_participant_refs
        ):
            raise ValueError("Context witness does not bind the event-scoped participant projection")
        if self.emotion.source_digest != emotion_projection_checksum(
            self.event, self.valence_hex, self.arousal_hex, self.optimal_loss_hex
        ):
            raise ValueError("global Emotion witness does not bind the exact stance scalars and event")
        if self.interlocutor_key is not None:
            identifier(self.interlocutor_key, "interlocutor_key")
            if self.interlocutor_key not in self.context_participant_refs:
                raise ValueError("tagged interlocutor must belong to the same selected Context")
        _hex_float(self.valence_hex, "valence", minimum=-1.0, maximum=1.0)
        _hex_float(self.arousal_hex, "arousal", minimum=0.0, maximum=1.0)
        _hex_float(self.optimal_loss_hex, "optimal_loss", minimum=0.0, maximum=float("inf"))
        if self.prior_committed_relationship_digest is not None:
            digest(self.prior_committed_relationship_digest, "prior relationship digest")
        exact_enum(self.association, FeltAssociation, "association")
        exact_enum(self.actor_causation, ActorCausation, "actor_causation")
        if self.association is FeltAssociation.UNKNOWN:
            if self.uncertainty is not None:
                raise ValueError("missing felt association has no numeric uncertainty")
        elif self.interlocutor_key is None or self.uncertainty is None:
            raise ValueError("tentative association needs a tagged interaction and uncertainty")
        if self.uncertainty is not None:
            integer(self.uncertainty, "uncertainty", minimum=1, maximum=SCORE_SCALE)

    def canonical_value(self) -> dict[str, object]:
        self.__post_init__()
        result = encode(self)
        assert type(result) is dict
        return result

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.canonical_value())

    @classmethod
    def from_value(cls, value: object) -> InteractionStance:
        row = closed(value, (
            "event", "context", "emotion", "context_participant_refs", "interlocutor_key", "valence_hex", "arousal_hex", "optimal_loss_hex",
            "prior_committed_relationship_digest", "association", "uncertainty", "actor_causation"
        ), "InteractionStance")
        result = cls(
            EventRef.from_value(row["event"]),
            SourceWitness.from_value(row["context"]),
            SourceWitness.from_value(row["emotion"]),
            tuple(identifier(item, "participant ref") for item in _participants(row["context_participant_refs"])),
            None if row["interlocutor_key"] is None else identifier(row["interlocutor_key"], "interlocutor_key"),
            _hex_float(row["valence_hex"], "valence", minimum=-1.0, maximum=1.0),
            _hex_float(row["arousal_hex"], "arousal", minimum=0.0, maximum=1.0),
            _hex_float(row["optimal_loss_hex"], "optimal_loss", minimum=0.0, maximum=float("inf")),
            None if row["prior_committed_relationship_digest"] is None else digest(row["prior_committed_relationship_digest"], "prior digest"),
            parse_enum(row["association"], FeltAssociation, "association"),
            None if row["uncertainty"] is None else integer(row["uncertainty"], "uncertainty", minimum=1, maximum=SCORE_SCALE),
            parse_enum(row["actor_causation"], ActorCausation, "actor_causation"),
        )
        if result.canonical_value() != row:
            raise ValueError("stance is noncanonical")
        return result


def _participants(value: object) -> list[object]:
    if type(value) is not list or len(value) > 32:
        raise ValueError("context participants must be a bounded JSON array")
    return value


def context_projection_checksum(event: EventRef, context_id: str, revision: int, participant_refs: tuple[str, ...]) -> str:
    """Event-bound R09 selection checksum; later owner still must compare trusted R09."""

    event.__post_init__()
    identifier(context_id, "context_id")
    integer(revision, "context revision", maximum=2**31 - 1)
    if type(participant_refs) is not tuple or len(participant_refs) > 32:
        raise ValueError("R09 participant references are bounded")
    for participant in participant_refs:
        identifier(participant, "participant ref")
    if participant_refs != tuple(sorted(set(participant_refs))):
        raise ValueError("R09 participant projection is not canonical")
    return checksum(
        b"PROJECT-SUZKA:R15:EVENT-CONTEXT-PROJECTION:V1\0",
        {"event": encode(event), "context_id": context_id, "revision": revision, "participant_refs": list(participant_refs)},
    )


def emotion_projection_checksum(event: EventRef, valence_hex: str, arousal_hex: str, optimal_loss_hex: str) -> str:
    """Pure optional checksum of explicit scalars; never R10 producer proof."""

    event.__post_init__()
    return checksum(
        b"PROJECT-SUZKA:R15:EVENT-GLOBAL-EMOTION-PROJECTION:V1\0",
        {
            "event": encode(event),
            "valence_hex": _hex_float(valence_hex, "valence", minimum=-1.0, maximum=1.0),
            "arousal_hex": _hex_float(arousal_hex, "arousal", minimum=0.0, maximum=1.0),
            "optimal_loss_hex": _hex_float(optimal_loss_hex, "optimal_loss", minimum=0.0, maximum=float("inf")),
        },
    )
