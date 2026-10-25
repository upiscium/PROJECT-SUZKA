"""Request-local U2 technical observation, never a Relationship or identity root."""

from __future__ import annotations

from dataclasses import dataclass

from suzka.r15.common import InterpretationStatus
from suzka.r15.contracts import RelationshipAxisKind
from suzka.r15.stance import ActorCausation, FeltAssociation, InteractionStance


@dataclass(frozen=True, slots=True)
class ScopedInteraction:
    """Opaque Context association and six explicitly unknown conceptual axes."""

    stance: InteractionStance
    axes: tuple[tuple[RelationshipAxisKind, InterpretationStatus], ...]

    def __post_init__(self) -> None:
        if (
            type(self.stance) is not InteractionStance
            or self.stance.association is not FeltAssociation.UNKNOWN
            or self.stance.uncertainty is not None
            or self.stance.actor_causation is not ActorCausation.UNKNOWN
            or self.axes
            != tuple((kind, InterpretationStatus.UNKNOWN) for kind in RelationshipAxisKind)
        ):
            raise ValueError("only a complete unknown interaction is supported")


def unknown_interaction(stance: InteractionStance) -> ScopedInteraction:
    return ScopedInteraction(
        stance,
        tuple((kind, InterpretationStatus.UNKNOWN) for kind in RelationshipAxisKind),
    )


def neutral_prompt_line(observation: ScopedInteraction) -> str:
    """The sole model-facing projection: no identifiers, evidence or scores."""
    observation.__post_init__()
    return (
        "Current interaction interpretation: UNKNOWN. A Context participant "
        "reference is not an authenticated speaker or person; do not infer "
        "trust, familiarity, reciprocity, caution, or interpersonal causation.\n\n"
    )
