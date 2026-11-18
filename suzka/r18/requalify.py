"""IP-17a: the forward Intent's source requalification (#411 D1 r7 §2-5).

Moved out of the IP-16a-2 Intent issuer so that the Intent issuer and the
R18 Policy owner (and later the approval owner and the CLAIM, IP-17b / 17c)
requalify a Decision's sources with one code path.  It reads only through
the R16 Plan preview owner and the Commitment system, inside the caller's own
serialized event; it writes nothing.

``requalify`` qualifies the Decision's Goal twice (equal, and the domain
snapshot unchanged) and checks, in order:

1. the Goal is a user request (``SELF_ORIGIN_GOAL``);
2. the qualified X keys the Decision's lineage and the qualified sources are
   the Decision's sealed ``source_refs`` (``SOURCE_DRIFT``);
3. the Commitment is ACTIVE at the sealed revision / digest
   (``COMMITMENT_NOT_ACTIVE``) and not within ``DEADLINE_MARGIN`` of its
   deadline (``DEADLINE_ELAPSED``).

Every failure is a ``RequalifyFailed`` with a closed ``RequalifyReason``;
each caller maps it to its own vocabulary (the Intent issuer to the
identically named HOLD reasons, the Policy owner to DENY or HOLD).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from suzka.motivation.commitment import CommitmentLifecycle
from suzka.r16.decision_issuer import (
    DEADLINE_MARGIN,
    _history_refs,
    lineage_scope_for,
)
from suzka.r16.goal_owner import GoalOrigin as R16GoalOrigin
from suzka.r16.history_codec import CommittedDecisionV1, SourceRole as HistorySourceRole
from suzka.runtime.agent_runtime import AgentEvent

if TYPE_CHECKING:
    from suzka.r16.plan_preview import _PlanPreviewOwner, _QualifiedPreview


class RequalifyReason(str, Enum):
    """Why a Decision's sources no longer qualify (closed, source-free)."""

    # The Decision does not name exactly one Goal.
    HEAD_CONFLICT = "head_conflict"
    SOURCE_UNAVAILABLE = "source_unavailable"
    SOURCE_CONFLICT = "source_conflict"
    SOURCE_DRIFT = "source_drift"
    SELF_ORIGIN_GOAL = "self_origin_goal"
    COMMITMENT_NOT_ACTIVE = "commitment_not_active"
    DEADLINE_ELAPSED = "deadline_elapsed"


class RequalifyFailed(Exception):
    def __init__(self, reason: RequalifyReason) -> None:
        super().__init__(reason.value)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class Requalified:
    """The requalified sources and the Commitment's obligation scope."""

    q: _QualifiedPreview
    initial: object
    obligation_scope: tuple[str, ...]


def qualify_twice(
    preview: _PlanPreviewOwner,
    decision: CommittedDecisionV1,
    event: AgentEvent,
    initial: object,
) -> _QualifiedPreview:
    """The Decision's Goal qualified twice, equal, with the snapshot unchanged."""

    from suzka.r16.plan_preview import _PlanSourceConflict, _PlanSourceUnavailable

    goals = [ref for ref in decision.source_refs if ref.role is HistorySourceRole.GOAL]
    if len(goals) != 1:
        raise RequalifyFailed(RequalifyReason.HEAD_CONFLICT)
    try:
        before = preview._qualify(goals[0].ref_id, event)
        after = preview._qualify(goals[0].ref_id, event)
    except _PlanSourceConflict:
        raise RequalifyFailed(RequalifyReason.SOURCE_CONFLICT) from None
    except _PlanSourceUnavailable:
        raise RequalifyFailed(RequalifyReason.SOURCE_UNAVAILABLE) from None
    if after != before or preview._state_snapshot() != initial:
        raise RequalifyFailed(RequalifyReason.SOURCE_UNAVAILABLE)
    return before


def check_sources(q: _QualifiedPreview, decision: CommittedDecisionV1) -> None:
    """Steps 1 and 2: a user-request Goal, and exactly the sealed sources."""

    if q.self_origin or q.goal_origin is not R16GoalOrigin.USER_REQUEST:
        raise RequalifyFailed(RequalifyReason.SELF_ORIGIN_GOAL)
    if (
        lineage_scope_for(q.experience_id) != decision.lineage_scope
        or _history_refs(q) != decision.source_refs
    ):
        raise RequalifyFailed(RequalifyReason.SOURCE_DRIFT)


def active_commitment(
    preview: _PlanPreviewOwner,
    q: _QualifiedPreview,
    clock: Callable[[], datetime],
) -> tuple[str, ...]:
    """Step 3: ACTIVE and before ``deadline - DEADLINE_MARGIN``; its scope."""

    if q.commitment_id is None:
        raise RequalifyFailed(RequalifyReason.COMMITMENT_NOT_ACTIVE)
    record = preview._loop._commitment_system.get(q.commitment_id)
    if (
        record is None
        or record.lifecycle is not CommitmentLifecycle.ACTIVE
        or record.revision != q.commitment_revision
        or record.record_digest != q.commitment_digest
    ):
        raise RequalifyFailed(RequalifyReason.COMMITMENT_NOT_ACTIVE)
    at = record.deadline.at
    if at is not None and at - clock() < DEADLINE_MARGIN:
        raise RequalifyFailed(RequalifyReason.DEADLINE_ELAPSED)
    return tuple(record.scope)


def commitment_deadline(
    preview: _PlanPreviewOwner, q: _QualifiedPreview
) -> datetime | None:
    """The qualified Commitment's deadline (None: no deadline)."""

    if q.commitment_id is None:
        raise RequalifyFailed(RequalifyReason.COMMITMENT_NOT_ACTIVE)
    record = preview._loop._commitment_system.get(q.commitment_id)
    if record is None:
        raise RequalifyFailed(RequalifyReason.COMMITMENT_NOT_ACTIVE)
    return record.deadline.at


def requalify(
    preview: _PlanPreviewOwner,
    decision: CommittedDecisionV1,
    event: AgentEvent,
    clock: Callable[[], datetime],
) -> Requalified:
    """§2-5: the whole requalification of one adopted Decision's sources."""

    initial = preview._state_snapshot()
    q = qualify_twice(preview, decision, event, initial)
    check_sources(q, decision)
    scope = active_commitment(preview, q, clock)
    return Requalified(q, initial, scope)


__all__ = [
    "RequalifyFailed",
    "RequalifyReason",
    "Requalified",
    "active_commitment",
    "check_sources",
    "commitment_deadline",
    "qualify_twice",
    "requalify",
]
