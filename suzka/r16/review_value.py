"""The review-integrity Value predicate shared by U2b-S and U3-A (IP-11).

#379 D1 P2 / D5 P2-3: a self-origin Goal is proposed only where the same
Value that Plan review needs is qualified, so an adopted self Goal can never
occupy X while its Plan is permanently unavailable.  A neutral module keeps
``goal_owner`` and ``plan_preview`` free of an import cycle (D5 review P3-1).
Read-only; owns no state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from suzka.identity.origin import (
    OriginActor,
    OriginInputKind,
    ValueAdmissionStatus,
    validate_origin_id,
)
from suzka.identity.value_system import (
    recompute_seed_contract_digest,
    value_state_digest,
)

if TYPE_CHECKING:
    from suzka.runtime.main_loop import SuzkaMainLoop

REVIEW_VALUE_ID: Final = "honesty"


class ReviewValueUnavailable(ValueError):
    """The review Value is missing, inactive here, or of unqualified origin."""


def qualify_review_value(loop: SuzkaMainLoop, context_id: str) -> tuple[int, str, str]:
    """(revision, state digest, origin id) of the qualified review Value."""

    values = loop._value_system.snapshot()
    matches = tuple(
        value for value in values.values if value.value_id == REVIEW_VALUE_ID
    )
    if len(matches) != 1:
        raise ReviewValueUnavailable("review integrity Value missing")
    value = matches[0]
    if (
        not value.is_active()
        or not value.applies_to(context_id)
        or value.origin.actor is not OriginActor.SYSTEM
        or value.origin.input_kind is not OriginInputKind.CONFIG_SEED
        or value.origin.admission is not ValueAdmissionStatus.SYSTEM_AUTHORIZED
        or value.origin.source_ref != f"config-seed:{REVIEW_VALUE_ID}"
        or value.polarity != 1
        or value.seed_contract_digest is None
    ):
        raise ReviewValueUnavailable("review integrity Value not qualified")
    seed = tuple(
        item.to_declaration()
        for item in loop.settings.values.seeds
        if item.value_id == REVIEW_VALUE_ID
    )
    if (
        len(seed) != 1
        or recompute_seed_contract_digest(seed[0]) != value.seed_contract_digest
    ):
        raise ReviewValueUnavailable("review Value has unqualified seed origin")
    return value.revision, value_state_digest(value), validate_origin_id(value.origin)


__all__ = ["REVIEW_VALUE_ID", "ReviewValueUnavailable", "qualify_review_value"]
