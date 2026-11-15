"""IP-12: R17 pure forecast / assessment / feedback values (D-R17 D1, #367).

Every value built here is a FIXTURE ONLY. None of it is evidence that SUZKA
learned anything from a real environment (#367 D0 acceptance matrix).
"""

from __future__ import annotations

import ast
from dataclasses import replace
import inspect
import json
from typing import Any

import pytest

import suzka.r17 as r17
from suzka.r16.history_codec import PolicyKind
from suzka.r16.successor_policy import FeedbackState, LedgerState
from suzka.r17 import contracts
from suzka.r17.contracts import (
    MAX_RECORD_BYTES,
    Comparison,
    Contributor,
    DecisionFeedbackV1,
    ExpectedObservationV1,
    ObservedState,
    OutcomeAssessmentV1,
    OutcomeState,
    TerminalOutcomeRevision,
    VerifiedOutcomeView,
    assess,
    feedback_state_of,
    project_feedback_state,
    select_feedback,
)

H = "a" * 64
PAYLOAD = "b" * 64
FORWARD = "c" * 64
TARGET = "00000000-0000-4000-8000-0000000000aa"
EVENT_F = "00000000-0000-4000-8000-000000000001"
EVENT_A = "00000000-0000-4000-8000-000000000002"
EVENT_B = "00000000-0000-4000-8000-000000000003"


def _forecast(**changes: Any) -> ExpectedObservationV1:
    base = ExpectedObservationV1(
        lineage_scope="x-" + "d" * 64,
        incarnation_id=H,
        admitting_event_id=EVENT_F,
        admitting_sequence=20,
        as_of_r07_sequence=19,
        decision_id="decision-" + "e" * 64,
        decision_committed_digest=H,
        policy_kind=PolicyKind.SUCCESSOR_POLICY_V2,
        selected_candidate_id="candidate-b",
        selected_candidate_digest=H,
        action_intent_id="intent-1",
        action_intent_digest=H,
        obligation_group_key="f" * 64,
        forward_key=FORWARD,
        target_instance_id=TARGET,
        target_namespace="review-artifact",
        target_expected_revision=0,
        predicted_payload_digest=PAYLOAD,
    )
    return replace(base, **changes)


def _outcome(state: OutcomeState = OutcomeState.ACTIVE_VERIFIED_AUTHORIZED, **changes: Any) -> VerifiedOutcomeView:
    defaults: dict[str, Any] = {
        OutcomeState.ACTIVE_VERIFIED_AUTHORIZED: {
            "observed_state": ObservedState.ACTIVE,
            "observed_payload_digest": PAYLOAD,
            "effect_operation_key": FORWARD,
        },
        OutcomeState.VERIFIED_ABSENT: {
            "observed_state": ObservedState.ABSENT,
            "observed_payload_digest": None,
            "effect_operation_key": None,
        },
        OutcomeState.UNAUTHORIZED_EFFECT: {
            "observed_state": ObservedState.ACTIVE,
            "observed_payload_digest": PAYLOAD,
            "effect_operation_key": FORWARD,
        },
    }[state]
    fields = {
        "verified_outcome_id": "outcome-1",
        "outcome_revision": 1,
        "outcome_digest": H,
        "outcome_state": state,
        "target_instance_id": TARGET,
        "observed_revision": 1,
        "foreign_change": False,
        **defaults,
        **changes,
    }
    return VerifiedOutcomeView(**fields)


def _assessment(outcome: VerifiedOutcomeView | None = None) -> OutcomeAssessmentV1:
    return OutcomeAssessmentV1.issue(_forecast(), outcome or _outcome(), EVENT_A, 30)


# ------------------------------------------------------------------ codecs


def test_each_record_round_trips_through_its_canonical_bytes() -> None:
    forecast = _forecast()
    assessment = _assessment()
    feedback = DecisionFeedbackV1.issue(assessment, EVENT_B, 31)
    for value in (forecast, assessment, feedback):
        payload = value.canonical_bytes()
        assert len(payload) <= MAX_RECORD_BYTES
        assert type(value).from_json(payload) == value
        loaded = json.loads(payload)
        assert loaded == json.loads(json.dumps(loaded, sort_keys=True))


def test_identities_are_domain_separated_and_fit_policy_ref_ids() -> None:
    forecast = _forecast()
    assessment = _assessment()
    feedback = DecisionFeedbackV1.issue(assessment, EVENT_B, 31)
    assert forecast.forecast_id.startswith("forecast-")
    assert assessment.assessment_id.startswith("assessment-")
    assert feedback.feedback_id.startswith("feedback-")
    digests = {
        forecast.forecast_id[len("forecast-") :],
        forecast.forecast_digest,
        assessment.assessment_id[len("assessment-") :],
        assessment.assessment_digest,
        feedback.feedback_id[len("feedback-") :],
        feedback.feedback_digest,
    }
    assert len(digests) == 6
    ref = feedback.policy_ref()
    assert (ref.kind, ref.ref_id, ref.revision, ref.digest) == (
        "r17-feedback",
        feedback.feedback_id,
        1,
        feedback.feedback_digest,
    )


def test_v1_fixed_fields_are_written_and_enforced_on_parse() -> None:
    row = json.loads(_forecast().canonical_bytes())
    assert (row["prediction_kind"], row["predicted_state"], row["prestate"]) == (
        "EXPECT_ACTIVE_WITH_PAYLOAD",
        "ACTIVE",
        "ABSENT",
    )
    assert (row["uncertainty"], row["policy_row"]) == ("NOT_ASSESSED", "4")
    for key, value in (
        ("prediction_kind", "EXPECT_FAILURE"),
        ("uncertainty", "LOW"),
        ("policy_row", "5"),
        ("prestate", "ACTIVE"),
    ):
        bad = dict(row, **{key: value})
        with pytest.raises(ValueError):
            ExpectedObservationV1.from_json(json.dumps(bad, sort_keys=True, separators=(",", ":")).encode())


def test_parse_rejects_extra_missing_and_noncanonical_input() -> None:
    payload = _forecast().canonical_bytes()
    row = json.loads(payload)
    with pytest.raises(ValueError):
        ExpectedObservationV1.from_json(json.dumps({**row, "extra": 1}).encode())
    del row["forecast_id"]
    with pytest.raises(ValueError):
        ExpectedObservationV1.from_json(json.dumps(row).encode())
    with pytest.raises(ValueError):
        ExpectedObservationV1.from_json(json.dumps(json.loads(payload), indent=1).encode())
    tampered = dict(json.loads(payload), forecast_digest="0" * 64)
    with pytest.raises(ValueError):
        ExpectedObservationV1.from_json(
            json.dumps(tampered, sort_keys=True, separators=(",", ":")).encode()
        )


def test_records_are_bounded() -> None:
    with pytest.raises(ValueError):
        ExpectedObservationV1.from_json(b"{" + b" " * MAX_RECORD_BYTES + b"}")


# ---------------------------------------------------------------- forecast


@pytest.mark.parametrize(
    "changes",
    [
        {"as_of_r07_sequence": 20},
        {"decision_id": "not-a-decision"},
        {"target_instance_id": "not-a-uuid"},
        {"predicted_payload_digest": "B" * 64},
        {"policy_kind": "SUCCESSOR_POLICY_V2"},
        {"target_expected_revision": -1},
    ],
)
def test_forecast_rejects_inconsistent_fields(changes: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        _forecast(**changes)


def test_forecast_binds_its_policy_kind() -> None:
    v1 = _forecast(policy_kind=PolicyKind.SUCCESSOR_POLICY_V1)
    assert v1.forecast_digest != _forecast().forecast_digest


# -------------------------------------------------------------- assessment


def test_authorized_effect_matches_and_is_self_without_foreign_change() -> None:
    assert assess(_forecast(), _outcome()) == (Comparison.MATCH, (Contributor.SELF,))


def test_authorized_effect_with_foreign_change_is_environment() -> None:
    outcome = _outcome(foreign_change=True)
    assert assess(_forecast(), outcome) == (Comparison.MATCH, (Contributor.ENVIRONMENT,))


def test_authorized_effect_of_another_operation_is_not_self() -> None:
    outcome = _outcome(effect_operation_key="9" * 64)
    assert assess(_forecast(), outcome) == (Comparison.MATCH, (Contributor.UNKNOWN,))


def test_authorized_effect_with_a_different_payload_is_inconsistent() -> None:
    with pytest.raises(ValueError):
        assess(_forecast(), _outcome(observed_payload_digest="9" * 64))


def test_verified_absent_is_a_mismatch_never_self() -> None:
    assert assess(_forecast(), _outcome(OutcomeState.VERIFIED_ABSENT)) == (
        Comparison.MISMATCH,
        (Contributor.UNKNOWN,),
    )
    assert assess(
        _forecast(), _outcome(OutcomeState.VERIFIED_ABSENT, foreign_change=True)
    ) == (Comparison.MISMATCH, (Contributor.ENVIRONMENT,))


@pytest.mark.parametrize(
    ("changes", "comparison"),
    [
        ({}, Comparison.MATCH),
        ({"observed_payload_digest": "9" * 64}, Comparison.MISMATCH),
        (
            {
                "observed_state": ObservedState.UNKNOWN,
                "observed_payload_digest": None,
                "effect_operation_key": None,
            },
            Comparison.UNKNOWN,
        ),
    ],
)
def test_unauthorized_effect_compares_but_never_credits_self(
    changes: dict[str, Any], comparison: Comparison
) -> None:
    outcome = _outcome(OutcomeState.UNAUTHORIZED_EFFECT, **changes)
    assert assess(_forecast(), outcome) == (comparison, (Contributor.UNKNOWN,))


def test_receipt_alone_cannot_make_a_match() -> None:
    # An executor receipt is not a verifier observation; without an observed
    # ACTIVE effect there is no authorized outcome to build.
    with pytest.raises(ValueError):
        _outcome(observed_state=ObservedState.UNKNOWN, observed_payload_digest=None)


def test_assessment_of_another_target_is_refused() -> None:
    with pytest.raises(ValueError):
        assess(_forecast(), _outcome(target_instance_id="00000000-0000-4000-8000-0000000000bb"))


def test_assessment_record_invariants() -> None:
    assessment = _assessment()
    for changes in (
        {"contributors": (Contributor.OTHER,)},
        {"contributors": (Contributor.SELF, Contributor.UNKNOWN)},
        {"contributors": ()},
        {"comparison": Comparison.MISMATCH},
        {"outcome_state": OutcomeState.UNAUTHORIZED_EFFECT},  # with SELF
    ):
        with pytest.raises((TypeError, ValueError)):
            replace(assessment, **changes)
    with pytest.raises(ValueError):
        OutcomeAssessmentV1.issue(_forecast(), _outcome(), EVENT_A, 20)


def test_non_authorized_assessment_is_the_positive_ineligibility_verdict() -> None:
    for state in (OutcomeState.VERIFIED_ABSENT, OutcomeState.UNAUTHORIZED_EFFECT):
        assessment = _assessment(_outcome(state))
        assert not assessment.positive_feedback_eligible
        with pytest.raises(ValueError):
            DecisionFeedbackV1.issue(assessment, EVENT_B, 31)


# ---------------------------------------------------------------- feedback


def test_feedback_only_from_authorized_effect_and_never_absent_or_invalid() -> None:
    feedback = DecisionFeedbackV1.issue(_assessment(), EVENT_B, 31)
    assert feedback.feedback_state is FeedbackState.MATCH_SELF
    for state in (FeedbackState.ABSENT, FeedbackState.INVALID):
        with pytest.raises(ValueError):
            replace(feedback, feedback_state=state)
    with pytest.raises(ValueError):
        DecisionFeedbackV1.issue(_assessment(), EVENT_B, 29)


def test_feedback_state_of_each_comparison() -> None:
    # MISMATCH / UNKNOWN rows are defensive: an authorized effect always
    # matches in V1, so these are fixture-only paths.
    authorized = _assessment()
    assert feedback_state_of(authorized) is FeedbackState.MATCH_SELF
    environment = _assessment(_outcome(foreign_change=True))
    assert feedback_state_of(environment) is FeedbackState.MATCH_OTHER_OR_UNKNOWN
    mismatch = _assessment(_outcome(OutcomeState.VERIFIED_ABSENT))
    assert feedback_state_of(mismatch) is FeedbackState.MISMATCH
    unknown = _assessment(
        _outcome(
            OutcomeState.UNAUTHORIZED_EFFECT,
            observed_state=ObservedState.UNKNOWN,
            observed_payload_digest=None,
            effect_operation_key=None,
        )
    )
    assert feedback_state_of(unknown) is FeedbackState.UNKNOWN


# -------------------------------------------------------------- projection


def _chain() -> tuple[ExpectedObservationV1, OutcomeAssessmentV1, DecisionFeedbackV1]:
    forecast = _forecast()
    assessment = OutcomeAssessmentV1.issue(forecast, _outcome(), EVENT_A, 30)
    return forecast, assessment, DecisionFeedbackV1.issue(assessment, EVENT_B, 31)


def test_projection_of_a_valid_chain_is_match_self() -> None:
    forecast, assessment, feedback = _chain()
    state = project_feedback_state(
        LedgerState.ACTIVE_VERIFIED_AUTHORIZED, forecast, feedback, assessment
    )
    assert state is FeedbackState.MATCH_SELF


def test_projection_keeps_historical_match_self_after_an_inverse() -> None:
    forecast, assessment, feedback = _chain()
    for ledger in (LedgerState.INVERSE_IN_DOUBT, LedgerState.REVERTED):
        assert (
            project_feedback_state(ledger, forecast, feedback, assessment)
            is FeedbackState.MATCH_SELF
        )


def test_claimed_ledger_without_a_forecast_is_invalid() -> None:
    for ledger in (LedgerState.CLAIMED, LedgerState.IN_DOUBT, LedgerState.CORRUPT):
        assert project_feedback_state(ledger, None, None, None) is FeedbackState.INVALID
    for ledger in (LedgerState.NONE, LedgerState.INTENT_ISSUED):
        assert project_feedback_state(ledger, None, None, None) is FeedbackState.ABSENT


def test_no_feedback_is_absent_even_with_an_assessment() -> None:
    forecast = _forecast()
    absent = OutcomeAssessmentV1.issue(forecast, _outcome(OutcomeState.VERIFIED_ABSENT), EVENT_A, 30)
    assert (
        project_feedback_state(LedgerState.VERIFIED_ABSENT, forecast, None, absent)
        is FeedbackState.ABSENT
    )


def test_feedback_without_its_assessment_or_with_wrong_bindings_is_invalid() -> None:
    forecast, assessment, feedback = _chain()
    ledger = LedgerState.ACTIVE_VERIFIED_AUTHORIZED
    assert project_feedback_state(ledger, forecast, feedback, None) is FeedbackState.INVALID
    other_forecast = _forecast(action_intent_id="intent-2")
    assert (
        project_feedback_state(ledger, other_forecast, feedback, assessment)
        is FeedbackState.INVALID
    )
    other = OutcomeAssessmentV1.issue(forecast, _outcome(outcome_digest="9" * 64), EVENT_A, 30)
    assert project_feedback_state(ledger, forecast, feedback, other) is FeedbackState.INVALID


def test_feedback_from_a_non_authorized_outcome_is_invalid() -> None:
    # A forged record: Feedback bound to an UNAUTHORIZED_EFFECT assessment
    # (e.g. an inverse after an unauthorized effect).
    forecast = _forecast()
    bad = OutcomeAssessmentV1.issue(
        forecast, _outcome(OutcomeState.UNAUTHORIZED_EFFECT), EVENT_A, 30
    )
    forged = DecisionFeedbackV1(
        bad.assessment_id,
        bad.assessment_digest,
        bad.forecast_id,
        bad.forecast_digest,
        bad.verified_outcome_id,
        bad.outcome_revision,
        bad.outcome_digest,
        FeedbackState.MATCH_OTHER_OR_UNKNOWN,
        EVENT_B,
        31,
    )
    assert (
        project_feedback_state(LedgerState.REVERTED, forecast, forged, bad)
        is FeedbackState.INVALID
    )


def test_stored_state_that_differs_from_the_recomputed_one_is_invalid() -> None:
    forecast, assessment, feedback = _chain()
    tampered = replace(feedback, feedback_state=FeedbackState.MISMATCH)
    assert (
        project_feedback_state(
            LedgerState.ACTIVE_VERIFIED_AUTHORIZED, forecast, tampered, assessment
        )
        is FeedbackState.INVALID
    )


# ---------------------------------------------------------------- selector


def test_selector_takes_the_latest_terminal_outcome_revision_first() -> None:
    _, _, feedback = _chain()
    revisions = [
        TerminalOutcomeRevision("outcome-1", 1, 30),
        TerminalOutcomeRevision("outcome-1", 2, 40),
    ]
    # Revision 2 has no Feedback: the revision-1 Feedback is not used.
    assert select_feedback(revisions, [feedback], 50) is None
    # Before revision 2 was terminal, revision 1's Feedback is selected.
    assert select_feedback(revisions, [feedback], 35) == feedback


def test_selector_ignores_records_at_or_after_the_horizon() -> None:
    _, _, feedback = _chain()
    revisions = [TerminalOutcomeRevision("outcome-1", 1, 30)]
    assert select_feedback(revisions, [feedback], 31) is None
    assert select_feedback(revisions, [feedback], 32) == feedback
    assert select_feedback([], [feedback], 32) is None


def test_selector_refuses_duplicate_v1_records() -> None:
    _, _, feedback = _chain()
    revisions = [TerminalOutcomeRevision("outcome-1", 1, 30)]
    with pytest.raises(ValueError):
        select_feedback(revisions, [feedback, replace(feedback, admitting_sequence=33)], 40)


# ------------------------------------------------------------------ purity


def test_r17_imports_stay_pure() -> None:
    tree = ast.parse(inspect.getsource(contracts))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert imported == {
        "__future__",
        "collections.abc",
        "dataclasses",
        "enum",
        "typing",
        "suzka.limits",
        "suzka.r16.contracts",
        "suzka.r16.history_codec",
        "suzka.r16.successor_policy",
    }
    assert not any(isinstance(node, ast.Import) for node in ast.walk(tree))
    assert set(r17.__all__) == {
        name for name in dir(r17) if not name.startswith("_") and name != "contracts"
    }
