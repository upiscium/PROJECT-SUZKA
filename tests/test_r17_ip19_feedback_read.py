"""IP-19: the Decision issuer's pure Feedback reader (#367 D2-1, D-R17 §5).

Every record here is a FIXTURE built from the IP-12 contracts (a nominal,
code-derived forecast); none of it is evidence of real-world learning
(D-R17 §6).  The pure function reads no store, Journal or clock.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from suzka.r16.history_codec import PolicyKind
from suzka.r16.successor_policy import (
    CommitmentState,
    FeedbackState,
    Interpretation,
    InterpretationStage,
    LedgerState,
    LineageRef,
    LineageState,
    PolicyRef,
    PredecessorState,
    RequiredSourceState,
    SuccessorPolicyInputV2,
    SuccessorRow,
    evaluate,
)
from suzka.r17.contracts import (
    Comparison,
    Contributor,
    DecisionFeedbackV1,
    OutcomeAssessmentV1,
)
from suzka.r17.feedback_read import ABSENT_FEEDBACK, FeedbackRead, read_feedback
from suzka.r18.contracts import (
    OUTCOME_REF_KIND,
    VerifiedOutcomeV1,
    VerifiedState,
)
from suzka.r18.ledger import RecordAt

import test_r16_ip16b_ledger_reader as ip16b
from test_r16_ip14a_history_v2 import Chain

H = "a" * 64
ACTIVE = LedgerState.ACTIVE_VERIFIED_AUTHORIZED


def _chain() -> Chain:
    return ip16b._fixture_chain()


def _at(chain: Chain, *records: object) -> list[RecordAt]:
    """Each record at its own processing sequence (the fixture chain's)."""

    steps = {
        id(chain.intent): 0,
        id(chain.forecast): 1,
        id(chain.verdict): 2,
        id(chain.claim): 4,
        id(chain.outcome): 5,
        id(chain.assessment): 6,
        id(chain.feedback): 7,
    }
    result = []
    for record in records:
        if id(record) in steps:
            sequence = chain.seq(steps[id(record)])
        elif record in chain.challenges:
            sequence = chain.seq(3)
        else:
            sequence = getattr(record, "admitting_sequence", None) or getattr(
                record, "verifier_sequence"
            )
        result.append(RecordAt(record, sequence))
    return result


def _until_claim(chain: Chain) -> list[object]:
    return [chain.intent, chain.forecast, chain.verdict, chain.challenges[0], chain.claim]


def _read(
    chain: Chain,
    records: list[object],
    ledger: LedgerState = ACTIVE,
    before: int | None = None,
) -> FeedbackRead:
    return read_feedback(
        _at(chain, *records),
        chain.intent.forward_key,
        ledger,
        chain.seq(20) if before is None else before,
    )


def _row(feedback: FeedbackRead, ledger: LedgerState = ACTIVE, version: PolicyKind = PolicyKind.SUCCESSOR_POLICY_V3) -> SuccessorRow:
    """The policy row of a qualified head with this ledger and Feedback."""

    ref = PolicyRef(OUTCOME_REF_KIND, "outcome-1", 1, H) if ledger is ACTIVE else None
    value = SuccessorPolicyInputV2(
        required_source=RequiredSourceState.QUALIFIED,
        source_refs=tuple(
            PolicyRef(kind, f"{kind}-1", 1, H) for kind in ("context", "experience", "goal", "value")
        ),
        predecessor=PredecessorState.QUALIFIED,
        predecessor_ref=PolicyRef("decision", "d-1", 2, H),
        lineage=LineageState.CONSISTENT,
        lineage_ref=LineageRef("lineage-1", "baseline-1", H, None, H),
        commitment=CommitmentState.ACTIVE,
        commitment_ref=PolicyRef("commitment", "c-1", 1, H),
        commitment_deadline_utc="9999-12-31T23:59:59Z",
        ledger=ledger,
        ledger_refs=(PolicyRef("r18-claim", "claim-1", 1, H),),
        inverse_authorization=None,
        feedback=feedback.state,
        feedback_ref=feedback.ref,
        verified_outcome_ref=ref,
        interpretation=Interpretation.SETTLED,
        interpretation_stage=InterpretationStage.S1,
        as_of_r07_sequence=1,
        as_of_snapshot_hash=H,
        event_id="event-1",
        event_sequence=2,
        now_utc="2026-11-17T00:00:00Z",
        version=version,
    )
    return evaluate(value).row


# ---------------------------------------------------------------- the table


def test_no_intent_for_the_key_is_absent_and_changes_nothing() -> None:
    chain = _chain()
    assert read_feedback([], chain.intent.forward_key, LedgerState.NONE, 5) == ABSENT_FEEDBACK
    # Another operation's records never leak into this key.
    assert (
        read_feedback(_at(chain, *_until_claim(chain)), "f" * 64, LedgerState.NONE, chain.seq(20))
        == ABSENT_FEEDBACK
    )


def test_a_verified_effect_with_no_assessment_or_feedback_is_absent_row_6() -> None:
    chain = _chain()
    read = _read(chain, [*_until_claim(chain), chain.outcome])
    assert read == ABSENT_FEEDBACK
    assert _row(read) is SuccessorRow.ROW_6
    # An Assessment alone changes nothing: it never reaches the policy.
    with_assessment = _read(chain, [*_until_claim(chain), chain.outcome, chain.assessment])
    assert with_assessment == ABSENT_FEEDBACK


@pytest.mark.parametrize("version", (PolicyKind.SUCCESSOR_POLICY_V2, PolicyKind.SUCCESSOR_POLICY_V3))
def test_match_self_feedback_is_row_5_no_op(version: PolicyKind) -> None:
    chain = _chain()
    records = [*_until_claim(chain), chain.outcome, chain.assessment, chain.feedback]
    read = _read(chain, records)
    assert read.state is FeedbackState.MATCH_SELF
    assert read.ref == chain.feedback.policy_ref()
    assert (read.ref.kind, read.ref.ref_id, read.ref.revision) == (
        "r17-feedback",
        chain.feedback.feedback_id,
        1,
    )
    assert _row(read, version=version) is SuccessorRow.ROW_5


def test_it_is_as_of_the_successor_event_not_the_latest() -> None:
    chain = _chain()
    records = [*_until_claim(chain), chain.outcome, chain.assessment, chain.feedback]
    # Up to and including the Feedback's own sequence it is not yet admitted.
    assert _read(chain, records, before=chain.feedback.admitting_sequence) == ABSENT_FEEDBACK
    assert _read(chain, records, before=chain.feedback.admitting_sequence + 1).state is (
        FeedbackState.MATCH_SELF
    )


def test_a_feedback_admitted_at_t_survives_a_later_inverse_and_the_ledger_is_row_2() -> None:
    chain = _chain()
    records = [*_until_claim(chain), chain.outcome, chain.assessment, chain.feedback]
    for ledger in (
        LedgerState.INVERSE_INTENT_ISSUED,
        LedgerState.INVERSE_CLAIMED,
        LedgerState.REVERTED,
    ):
        read = _read(chain, records, ledger)
        # T's Feedback is returned as is (#367 D2-1) ...
        assert read.state is FeedbackState.MATCH_SELF, ledger
    # ... and the current inverse state is a separate row-2 HOLD (V3).
    for ledger in (LedgerState.INVERSE_INTENT_ISSUED, LedgerState.INVERSE_CLAIMED):
        assert _row(_read(chain, records, ledger), ledger) is SuccessorRow.ROW_2


def test_a_claimed_operation_with_no_forecast_is_invalid_row_0() -> None:
    chain = _chain()
    records = [chain.intent, chain.verdict, chain.challenges[0], chain.claim, chain.outcome]
    read = _read(chain, records)
    assert read.state is FeedbackState.INVALID
    assert _row(read) is SuccessorRow.ROW_0
    # Before any claim a missing forecast is not yet a defect.
    assert _read(chain, [chain.intent], LedgerState.INTENT_ISSUED) == ABSENT_FEEDBACK


def test_a_forecast_after_the_claim_is_invalid() -> None:
    chain = _chain()
    late = replace(chain.forecast, admitting_sequence=chain.claim.claim_sequence + 1)
    records = [chain.intent, late, chain.verdict, chain.challenges[0], chain.claim, chain.outcome]
    at = [
        RecordAt(record, getattr(record, "admitting_sequence", chain.seq(4)))
        for record in records
    ]
    read = read_feedback(at, chain.intent.forward_key, ACTIVE, chain.seq(20))
    assert read.state is FeedbackState.INVALID


def test_a_stored_state_that_does_not_follow_is_invalid() -> None:
    chain = _chain()
    forged = replace(chain.feedback, feedback_state=FeedbackState.MISMATCH)
    records = [*_until_claim(chain), chain.outcome, chain.assessment, forged]
    assert _read(chain, records).state is FeedbackState.INVALID


def test_a_feedback_with_no_matching_assessment_or_outcome_is_invalid() -> None:
    chain = _chain()
    base = [*_until_claim(chain), chain.outcome]
    assert _read(chain, [*base, chain.feedback]).state is FeedbackState.INVALID  # no Assessment
    other = replace(chain.feedback, assessment_digest="b" * 64)
    assert _read(chain, [*base, chain.assessment, other]).state is FeedbackState.INVALID
    # A Feedback and no Outcome at all: never a positive state.
    assert _read(chain, [*_until_claim(chain), chain.assessment, chain.feedback]).state is (
        FeedbackState.INVALID
    )


def test_two_feedbacks_for_one_revision_are_invalid() -> None:
    chain = _chain()
    second = replace(chain.feedback, admitting_sequence=chain.seq(8), admitting_event_id=chain.event("fb2"))
    records = [*_until_claim(chain), chain.outcome, chain.assessment, chain.feedback, second]
    assert _read(chain, records).state is FeedbackState.INVALID


def test_a_newer_outcome_revision_never_reuses_the_older_feedback() -> None:
    chain = _chain()
    newer = replace(chain.outcome, outcome_revision=2, verifier_sequence=chain.seq(8),
                    verifier_event_id=chain.event("outcome2"))
    records = [*_until_claim(chain), chain.outcome, chain.assessment, chain.feedback, newer]
    read = read_feedback(
        [*_at(chain, *records[:-1]), RecordAt(newer, newer.verifier_sequence)],
        chain.intent.forward_key,
        ACTIVE,
        chain.seq(20),
    )
    assert read == ABSENT_FEEDBACK  # selector: latest terminal revision first
    # Before the newer revision was terminal, the older Feedback is the answer.
    older = read_feedback(
        [*_at(chain, *records[:-1]), RecordAt(newer, newer.verifier_sequence)],
        chain.intent.forward_key,
        ACTIVE,
        chain.seq(8),
    )
    assert older.state is FeedbackState.MATCH_SELF


# ------------------------------------------- F09 and unauthorized effects


def _foreign(chain: Chain) -> tuple[VerifiedOutcomeV1, OutcomeAssessmentV1, DecisionFeedbackV1]:
    outcome = replace(chain.outcome, foreign_change=True)
    assessment = OutcomeAssessmentV1.issue(
        chain.forecast, outcome.to_r17_view(), chain.event("assessment"), chain.seq(6)
    )
    feedback = DecisionFeedbackV1.issue(assessment, chain.event("feedback"), chain.seq(7))
    return outcome, assessment, feedback


def test_f09_a_foreign_change_is_environment_never_self_and_is_row_6() -> None:
    chain = _chain()
    outcome, assessment, feedback = _foreign(chain)
    assert assessment.contributors == (Contributor.ENVIRONMENT,)
    assert assessment.comparison is Comparison.MATCH
    at = [
        *_at(chain, *_until_claim(chain)),
        RecordAt(outcome, chain.seq(5)),
        RecordAt(assessment, chain.seq(6)),
        RecordAt(feedback, chain.seq(7)),
    ]
    read = read_feedback(at, chain.intent.forward_key, ACTIVE, chain.seq(20))
    assert read.state is FeedbackState.MATCH_OTHER_OR_UNKNOWN
    assert _row(read) is SuccessorRow.ROW_6  # never row 5 (no self credit)


def test_f09_a_stored_self_attribution_over_a_foreign_change_is_invalid() -> None:
    chain = _chain()
    outcome = replace(chain.outcome, foreign_change=True)
    # The Assessment / Feedback of the unchanged Outcome, replayed against an
    # Outcome that did see a foreign change: the digests no longer bind.
    at = [
        *_at(chain, *_until_claim(chain)),
        RecordAt(outcome, chain.seq(5)),
        RecordAt(chain.assessment, chain.seq(6)),
        RecordAt(chain.feedback, chain.seq(7)),
    ]
    read = read_feedback(at, chain.intent.forward_key, ACTIVE, chain.seq(20))
    assert read.state is FeedbackState.INVALID


@pytest.mark.parametrize(
    "state", (VerifiedState.UNAUTHORIZED_EFFECT, VerifiedState.VERIFIED_ABSENT)
)
def test_an_unauthorized_or_absent_outcome_never_yields_positive_feedback(state: VerifiedState) -> None:
    from test_r18_ip13_contracts import _outcome

    chain = _chain()
    outcome = _outcome(
        state,
        operation_key=chain.intent.forward_key,
        effect_operation_key=chain.intent.forward_key
        if state is VerifiedState.UNAUTHORIZED_EFFECT
        else None,
        target_instance_id=chain.intent.target_instance_id,
        verifier_event_id=chain.event("outcome"),
        verifier_sequence=chain.seq(5),
    )
    assessment = OutcomeAssessmentV1.issue(
        chain.forecast, outcome.to_r17_view(), chain.event("assessment"), chain.seq(6)
    )
    assert not assessment.positive_feedback_eligible
    assert Contributor.SELF not in assessment.contributors
    with pytest.raises(ValueError):
        DecisionFeedbackV1.issue(assessment, chain.event("feedback"), chain.seq(7))
    ledger = LedgerState(state.value)
    base = [
        *_at(chain, *_until_claim(chain)),
        RecordAt(outcome, chain.seq(5)),
        RecordAt(assessment, chain.seq(6)),
    ]
    # With no Feedback it is ABSENT (the verdict is the Assessment itself) ...
    assert read_feedback(base, chain.intent.forward_key, ledger, chain.seq(20)) == ABSENT_FEEDBACK
    # ... and a Feedback forged over it (fixture only) is INVALID, never positive.
    forged: Any = DecisionFeedbackV1(
        assessment.assessment_id,
        assessment.assessment_digest,
        assessment.forecast_id,
        assessment.forecast_digest,
        assessment.verified_outcome_id,
        assessment.outcome_revision,
        assessment.outcome_digest,
        FeedbackState.MATCH_SELF,
        chain.event("forged"),
        chain.seq(7),
    )
    read = read_feedback(
        [*base, RecordAt(forged, chain.seq(7))], chain.intent.forward_key, ledger, chain.seq(20)
    )
    assert read.state is FeedbackState.INVALID
    assert _row(read, ledger) in (SuccessorRow.ROW_0,)


def test_in_doubt_claimed_with_no_outcome_has_nothing_to_project() -> None:
    chain = _chain()
    read = _read(chain, _until_claim(chain), LedgerState.CLAIMED)
    assert read == ABSENT_FEEDBACK and _row(read, LedgerState.CLAIMED) is SuccessorRow.ROW_2

