"""Closed owner outcomes and reason codes for R16 U2b / U2c / U3-A (IP-5).

#370 D2 P1-5 (#354 scope): an owner's result used to be a bare string
(``"hold"``) or an exception, with no reason. IP-6's driver reports
command results to the Chat screen through tickets, so each owner now
returns a value with a status and, for HOLD, one reason from a closed,
source-free vocabulary. Reasons never carry text, identifiers or private
R15 classes; an unlisted cause is ``SOURCE_UNAVAILABLE``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from suzka.identifiers import validate_identifier
from suzka.runtime.journal_source import JournalSourceReason, SourceUnavailable


class R16OwnerStatus(str, Enum):
    PROPOSED = "proposed"
    ADOPTED = "adopted"
    ACCEPTED = "accepted"
    HOLD = "hold"


class R16OwnerReason(str, Enum):
    """Why an owner held; a closed vocabulary (#370 D2 P1-5, #379 D3)."""

    # The target already has a Goal: this user's request, or a Goal of the
    # subject's own (IP-11 U2b-S) that occupies X.
    GOAL_ALREADY_EXISTS = "goal_already_exists"
    SELF_GOAL_OCCUPIES_TARGET = "self_goal_occupies_target"
    COMMITMENT_ALREADY_EXISTS = "commitment_already_exists"
    # The reviewed record is not (or no longer) waiting for this review.
    NOT_PENDING_REVIEW = "not_pending_review"
    GOAL_NOT_ADOPTED = "goal_not_adopted"
    # No independent, unchanged INTEREST from U2a for the target.
    MOTIVATION_UNAVAILABLE = "motivation_unavailable"
    COUNTER_MOTIVATION = "counter_motivation"
    CONFLICTING_GOAL = "conflicting_goal"
    PLAN_SOURCE_CONFLICT = "plan_source_conflict"
    DEADLINE_ELAPSED = "deadline_elapsed"
    # Source evidence (IP-4c): archived past the window, missing, or any
    # other unqualified or unavailable source, Context mismatch included
    # (#370 D2 P2-2: no cross-Context existence oracle).
    SOURCE_PAST_RETENTION = "source_past_retention"
    SOURCE_MISSING = "source_missing"
    SOURCE_UNAVAILABLE = "source_unavailable"
    # IP-10 (#371 D2): the CommittedDecision issuer's pre-policy refusals.
    # Sources moved with no new policy witness (#376 §C).
    DECISION_SOURCE_DRIFT = "decision_source_drift"
    # SubjectHistory cannot hold another Decision (checked before PREPARE).
    HISTORY_CAPACITY = "history_capacity"
    # The review intent or the lineage slot is already taken.
    DECISION_SLOT_OCCUPIED = "decision_slot_occupied"


_JOURNAL_REASONS = {
    JournalSourceReason.SOURCE_PAST_RETENTION: R16OwnerReason.SOURCE_PAST_RETENTION,
    JournalSourceReason.SOURCE_MISSING: R16OwnerReason.SOURCE_MISSING,
}


def owner_reason(error: BaseException) -> R16OwnerReason:
    """The closed reason an owner refusal carries; unlisted causes are generic."""

    reason = error.reason if isinstance(error, SourceUnavailable) else None
    if isinstance(reason, R16OwnerReason):
        return reason
    if isinstance(reason, JournalSourceReason):
        return _JOURNAL_REASONS.get(reason, R16OwnerReason.SOURCE_UNAVAILABLE)
    return R16OwnerReason.SOURCE_UNAVAILABLE


@dataclass(frozen=True, slots=True)
class R16OwnerOutcome:
    """One owner result: a status, the record it concerns, and a HOLD reason."""

    status: R16OwnerStatus
    subject_id: str | None = None
    reason: R16OwnerReason | None = None

    def __post_init__(self) -> None:
        if type(self.status) is not R16OwnerStatus:
            raise TypeError("owner status must be an exact R16OwnerStatus")
        if (self.status is R16OwnerStatus.HOLD) != (self.reason is not None):
            raise ValueError("exactly a HOLD carries a reason")
        if self.reason is not None and type(self.reason) is not R16OwnerReason:
            raise TypeError("owner reason must be an exact R16OwnerReason")
        if self.subject_id is not None:
            validate_identifier(self.subject_id)
        if self.status is not R16OwnerStatus.HOLD and self.subject_id is None:
            raise ValueError("a positive owner outcome names its record")

    @classmethod
    def hold(
        cls, reason: R16OwnerReason, subject_id: str | None = None
    ) -> R16OwnerOutcome:
        return cls(R16OwnerStatus.HOLD, subject_id, reason)
