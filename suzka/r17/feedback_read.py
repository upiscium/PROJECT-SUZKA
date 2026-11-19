"""R17 IP-19: the Decision issuer's Feedback reader (#367 D1 / D2-1, D-R17 §5).

Pure.  From the adopted R17 / R18 records of the current incarnation (the
same classification the lineage head came from), one forward operation key,
the ledger state the issuer derived for it and the successor Decision's
event sequence, return the ``FeedbackState`` and ``feedback_ref`` the issuer
seals.  No store, Journal or clock is read.

As-of (#367 D2-1): only records whose processing sequence is below
``before_sequence`` count.  A positive Feedback admitted at T therefore stays
the answer after a later inverse (the current inverse state is a separate
row-2 HOLD through the ledger).

Selection is D-R17 §5: the latest terminal forward Outcome revision first,
then the latest Feedback for exactly that revision; none for that revision is
ABSENT (an older revision's Feedback is never reused).  Only
``project_feedback_state`` emits INVALID, and a read that cannot be made
consistent is INVALID here (row 0), never a guess.

Records of *every* lineage count: the key is the obligation, not the lineage
(IP-16b).  An Assessment with no Feedback is ABSENT; a missing forecast of a
claimed forward operation is INVALID (#367 D2-2).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

from suzka.r16.history_codec import event_sequence
from suzka.r16.successor_policy import FeedbackState, LedgerState, PolicyRef
from suzka.r17.contracts import (
    DecisionFeedbackV1,
    ExpectedObservationV1,
    OutcomeAssessmentV1,
    TerminalOutcomeRevision,
    project_feedback_state,
    select_feedback,
)
from suzka.r18.contracts import (
    ActionIntentV1,
    Direction,
    ExecutionClaimV1,
    VerifiedOutcomeV1,
)
from suzka.r18.ledger import RecordAt


@dataclass(frozen=True, slots=True)
class FeedbackRead:
    """The Feedback witness the Decision issuer seals."""

    state: FeedbackState
    ref: PolicyRef | None


# The pre-IP-19 value, and the value whenever nothing about R17 is readable
# (no forward operation, a prior-incarnation claim, a facts-derived state).
ABSENT_FEEDBACK: Final = FeedbackRead(FeedbackState.ABSENT, None)
_INVALID: Final = FeedbackRead(FeedbackState.INVALID, None)


def read_feedback(
    records: Iterable[RecordAt],
    forward_key: str,
    ledger: LedgerState,
    before_sequence: int,
) -> FeedbackRead:
    """The §5 projection for one forward operation, as of ``before_sequence``."""

    event_sequence(before_sequence, "before_sequence")
    items = tuple(records)
    intents = [
        item.record
        for item in items
        if type(item.record) is ActionIntentV1 and item.record.forward_key == forward_key
    ]
    if not intents:
        return ABSENT_FEEDBACK
    if len(intents) != 1:
        return _INVALID
    intent = intents[0]
    assert type(intent) is ActionIntentV1
    forecasts = [
        item.record
        for item in items
        if type(item.record) is ExpectedObservationV1
        and item.record.forward_key == forward_key
        and item.sequence < before_sequence
    ]
    if len(forecasts) > 1:
        return _INVALID
    forecast = forecasts[0] if forecasts else None
    assert forecast is None or type(forecast) is ExpectedObservationV1
    claims = [
        item.record
        for item in items
        if type(item.record) is ExecutionClaimV1
        and item.record.direction is Direction.FORWARD
        and item.record.operation_key == forward_key
        and item.sequence < before_sequence
    ]
    if len(claims) > 1:
        return _INVALID
    claim = claims[0] if claims else None
    assert claim is None or type(claim) is ExecutionClaimV1
    outcomes = [
        (item.record, item.sequence)
        for item in items
        if type(item.record) is VerifiedOutcomeV1
        and item.record.direction is Direction.FORWARD
        and item.record.operation_key == forward_key
    ]
    outcome_ids = {record.verified_outcome_id for record, _ in outcomes}
    feedbacks = [
        item.record
        for item in items
        if type(item.record) is DecisionFeedbackV1
        and (
            item.record.verified_outcome_id in outcome_ids
            or (forecast is not None and item.record.forecast_id == forecast.forecast_id)
        )
    ]
    try:
        selected = select_feedback(
            (
                TerminalOutcomeRevision(
                    record.verified_outcome_id, record.outcome_revision, sequence
                )
                for record, sequence in outcomes
            ),
            feedbacks,
            before_sequence,
        )
    except (TypeError, ValueError):
        return _INVALID
    terminal = [record for record, sequence in outcomes if sequence < before_sequence]
    if not terminal and any(item.admitting_sequence < before_sequence for item in feedbacks):
        # A Feedback with no terminal Outcome to bind to (the store cannot
        # write one): never a positive state.
        return _INVALID
    latest = max(terminal, key=lambda record: record.outcome_revision) if terminal else None
    view = None if latest is None else latest.to_r17_view()
    assessment = None
    if selected is not None:
        bound = [
            item.record
            for item in items
            if type(item.record) is OutcomeAssessmentV1
            and item.record.assessment_id == selected.assessment_id
        ]
        assessment = bound[0] if len(bound) == 1 else None
    try:
        state = project_feedback_state(
            ledger,
            forecast,
            selected,
            assessment,
            view,
            action_intent_id=intent.intent_id,
            forward_key=forward_key,
            claim_sequence=None if claim is None else claim.claim_sequence,
        )
    except (TypeError, ValueError):
        return _INVALID
    ref = None if selected is None else selected.policy_ref()
    if state is FeedbackState.ABSENT:
        ref = None
    return FeedbackRead(state, ref)


__all__ = ["ABSENT_FEEDBACK", "FeedbackRead", "read_feedback"]
