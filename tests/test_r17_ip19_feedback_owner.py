"""IP-19: the Assessment / Feedback owner and the issuer's Feedback witness
(#367 D1 / D2, D-R17 §3 / §4 / §5; #411 IP-19).

The chain up to the Outcome is built the way the IP-16b ledger tests build it:
a real row-4 Decision, Intent and forecast, then FIXTURE verdict, challenge,
claim and Outcome records committed through the real R07 runtime and the
subject.history participant (the real executor is IP-18's and has its own
flow tests).  Every Assessment / Feedback below is written by the real
``FeedbackOwnerV1`` events, and read back by the real Decision issuer.

All of it is a fixture of a code-derived nominal forecast: none of it is
evidence that the subject learned anything (D-R17 §6).
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient
import pytest

from suzka.r16.decision_issuer import DecisionIssueStatus
from suzka.r16.history_reader import SubjectHistoryReader
from suzka.r16.history_records import (
    HistoryOperationKind as K,
    RecordKind as R,
)
from suzka.r16.history_store import HistoryPhase
from suzka.r16.successor_policy import FeedbackState, InverseAuthorization, SuccessorRow
from suzka.r17.contracts import (
    Comparison,
    Contributor,
    DecisionFeedbackV1,
    ObservedState,
    OutcomeAssessmentV1,
    OutcomeState,
)
from suzka.r17.feedback_owner import (
    AssessmentCommitStatus,
    FeedbackCommitStatus,
    FeedbackOwnerV1,
    R17FeedbackReason,
    _Hold,
)
from suzka.r18.contracts import (
    Direction,
    ExecutionClaimV1,
    VerifiedOutcomeV1,
    VerifiedState,
)
from suzka.r18.ledger import RecordAt
from suzka.runtime.agent_runtime import AgentEventType, AgentRuntimeDurabilityError
from suzka.runtime.event_journal import EventLifecycle

import test_r16_ip10_decision_issuer as ip10
import test_r16_ip16b_ledger_reader as ip16b
import test_r18_ip16a2_intent_issuer as ip16a2
from test_r16_u2a_source_owner import _app

_settings = ip16a2._settings
_no_failed_events = ip16a2._no_failed_events
_Seen = ip16b._Seen

ASSESS = AgentEventType.R17_ASSESSMENT_COMMIT
FEEDBACK = AgentEventType.R17_FEEDBACK_COMMIT


@pytest.fixture
def app(tmp_path):  # type: ignore[no-untyped-def]
    return _app(_settings(tmp_path / "ip19"))


# ------------------------------------------------------------------ helpers


def _chain_to_claim(state):  # type: ignore[no-untyped-def]
    """Real Decision / Intent / forecast; fixture verdict, challenge, claim."""

    gid, _decision, intent, forecast = ip16b._forward(state)
    verdict = ip16b._verdict(state, intent)
    challenge = ip16b._challenge(state, intent, verdict, 1, forecast)
    claim = ip16b._claim(state, challenge)
    return gid, intent, forecast, claim


def _commit_outcome(  # type: ignore[no-untyped-def]
    state,
    intent,
    outcome_state: VerifiedState = VerifiedState.ACTIVE_VERIFIED_AUTHORIZED,
    *,
    foreign_change: bool = False,
    revision: int = 1,
):
    """A forward Outcome fixture, as the independent verifier would write it."""

    active = outcome_state is not VerifiedState.VERIFIED_ABSENT
    authorization = {
        VerifiedState.ACTIVE_VERIFIED_AUTHORIZED: InverseAuthorization.AUTHORIZED,
        VerifiedState.UNAUTHORIZED_EFFECT: InverseAuthorization.UNPROVEN,
    }.get(outcome_state)
    payload = intent.payload_digest
    if outcome_state is VerifiedState.UNAUTHORIZED_EFFECT:
        payload = "9" * 64  # the effect differs from what was approved

    def build(event):  # type: ignore[no-untyped-def]
        record = VerifiedOutcomeV1(
            operation_key=intent.forward_key,
            direction=Direction.FORWARD,
            outcome_revision=revision,
            outcome_state=outcome_state,
            authorization=authorization,
            target_instance_id=intent.target_instance_id,
            observed_state=ObservedState.ACTIVE if active else ObservedState.ABSENT,
            observed_payload_digest=payload if active else None,
            observed_revision=1,
            effect_operation_key=intent.forward_key if active else None,
            foreign_change=foreign_change,
            verifier_event_id=event.event_id,
            verifier_sequence=event.processing_sequence,
        )
        return ip16b._operation(state, event, K.R18_OUTCOME_COMMIT, R.VERIFIED_OUTCOME, record)

    return ip16b._commit_record(state, build).record


def _assess(state, outcome_id: str | None = None):  # type: ignore[no-untyped-def]
    return state.feedback_owner._submit_assessment(outcome_id).result(timeout=30).value


def _feedback(state, assessment_id: str | None = None):  # type: ignore[no-untyped-def]
    return state.feedback_owner._submit_feedback(assessment_id).result(timeout=30).value


def _rows(state, *kinds):  # type: ignore[no-untyped-def]
    return [
        row
        for row in state.subject_history_store.operations(ip10._incarnation(state))
        if row.operation_kind in kinds
    ]


def _adopted(state, record_type):  # type: ignore[no-untyped-def]
    found = SubjectHistoryReader(state.event_journal, state.subject_history_store).adopted_records()
    assert found.blocked is None
    return [item.record for item in found.records if type(item.record) is record_type]


def _events(state, event_type) -> int:  # type: ignore[no-untyped-def]
    return sum(
        1
        for row in state.event_journal.records
        if row.event_type is event_type and row.lifecycle is EventLifecycle.ACCEPTED
    )


# ------------------------------------------------- the positive path, row 5


def test_an_authorized_outcome_yields_match_self_feedback_and_row_5(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        seen = _Seen(monkeypatch)
        gid, intent, forecast, _claim = _chain_to_claim(state)
        outcome = _commit_outcome(state, intent)
        before = ip16b._preview(state, gid, seen)
        assert before.row is SuccessorRow.ROW_6
        assert seen.last.feedback is FeedbackState.ABSENT and seen.last.feedback_ref is None

        assessed = _assess(state)
        assert assessed.status is AssessmentCommitStatus.COMMITTED
        assessment = assessed.assessment
        assert assessment.verified_outcome_id == outcome.verified_outcome_id
        assert assessment.outcome_state is OutcomeState.ACTIVE_VERIFIED_AUTHORIZED
        assert assessment.comparison is Comparison.MATCH
        assert assessment.contributors == (Contributor.SELF,)
        assert assessment.forecast_id == forecast.forecast_id
        # An Assessment alone never reaches the policy: the witness is unchanged.
        again = ip16b._preview(state, gid, seen)
        assert (again.status, seen.last.feedback) == (DecisionIssueStatus.EXISTING, FeedbackState.ABSENT)
        assert _assess(state).status is AssessmentCommitStatus.NONE_LEFT

        fed = _feedback(state)
        assert fed.status is FeedbackCommitStatus.COMMITTED
        feedback = fed.feedback
        assert feedback.feedback_state is FeedbackState.MATCH_SELF
        assert feedback.assessment_id == assessment.assessment_id

        after = ip16b._preview(state, gid, seen)
        # MATCH_SELF with a verified authorized effect: row 5 (NO_OP), a
        # successor Decision because the Feedback witness changed.
        assert (after.status, after.row) == (DecisionIssueStatus.ISSUED, SuccessorRow.ROW_5)
        assert seen.last.feedback is FeedbackState.MATCH_SELF
        assert seen.last.feedback_ref == feedback.policy_ref()
        assert ip16b._preview(state, gid, seen).status is DecisionIssueStatus.EXISTING
        assert _feedback(state).status is FeedbackCommitStatus.NONE_LEFT

        # Idempotent by id: nothing more is written.
        assert _assess(state, outcome.verified_outcome_id).status is AssessmentCommitStatus.EXISTING
        assert _feedback(state, assessment.assessment_id).status is FeedbackCommitStatus.EXISTING
        for kind in (K.R17_ASSESSMENT_COMMIT, K.R17_FEEDBACK_COMMIT):
            assert [row.phase for row in _rows(state, kind)] == [HistoryPhase.FINALIZED]
        assert len(_adopted(state, OutcomeAssessmentV1)) == 1
        assert len(_adopted(state, DecisionFeedbackV1)) == 1
        _no_failed_events(state)


def test_a_feedback_admitted_at_t_survives_a_later_inverse_intent(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """#367 D2-1: T's Feedback is returned as is; the inverse is row 2 on its own."""

    from suzka.r16.history_records import HistoryOperationKind
    from suzka.r18.contracts import InverseIntentV1

    with TestClient(app) as client:
        state = client.app.state
        seen = _Seen(monkeypatch)
        gid, intent, _forecast, _claim = _chain_to_claim(state)
        outcome = _commit_outcome(state, intent)
        assert _assess(state).status is AssessmentCommitStatus.COMMITTED
        feedback = _feedback(state).feedback
        inverse = InverseIntentV1(
            incarnation_id=ip10._incarnation(state),
            admitting_event_id="00000000-0000-4000-8000-000000000000",
            admitting_sequence=1,
            obligation_group_key=intent.obligation_group_key,
            target_instance_id=intent.target_instance_id,
            target_namespace=intent.target_namespace,
            target_expected_revision=1,
            forward_outcome_id=outcome.verified_outcome_id,
            forward_outcome_revision=1,
            forward_outcome_digest=outcome.outcome_digest,
            forward_outcome_state=VerifiedState.ACTIVE_VERIFIED_AUTHORIZED,
            forward_payload_digest=intent.payload_digest,
        )

        def build(event):  # type: ignore[no-untyped-def]
            record = replace(
                inverse, admitting_event_id=event.event_id, admitting_sequence=event.processing_sequence
            )
            return ip16b._operation(
                state, event, HistoryOperationKind.R18_INTENT_ISSUE, R.INVERSE_INTENT, record
            )

        ip16b._commit_record(state, build)
        result = ip16b._preview(state, gid, seen)
        assert seen.last.feedback is FeedbackState.MATCH_SELF
        assert seen.last.feedback_ref == feedback.policy_ref()
        assert result.row is SuccessorRow.ROW_2  # the current inverse state
        # No new Feedback is written once the operation is no longer ACTIVE_VERIFIED_AUTHORIZED.
        assert _feedback(state).status is FeedbackCommitStatus.NONE_LEFT
        held = _feedback(state, feedback.assessment_id)
        assert held.status is FeedbackCommitStatus.EXISTING  # the one admitted at T
        _no_failed_events(state)


# ------------------------------------------- what can never be positive


@pytest.mark.parametrize(
    "outcome_state", (VerifiedState.UNAUTHORIZED_EFFECT, VerifiedState.VERIFIED_ABSENT)
)
def test_an_unauthorized_or_absent_outcome_gets_an_assessment_and_never_feedback(  # type: ignore[no-untyped-def]
    app, monkeypatch, outcome_state
) -> None:
    with TestClient(app) as client:
        state = client.app.state
        seen = _Seen(monkeypatch)
        gid, intent, _forecast, _claim = _chain_to_claim(state)
        _commit_outcome(state, intent, outcome_state)
        assessed = _assess(state)
        assert assessed.status is AssessmentCommitStatus.COMMITTED
        assessment = assessed.assessment
        # R17's terminal verdict: not eligible for positive Feedback.
        assert not assessment.positive_feedback_eligible
        assert assessment.outcome_state is OutcomeState(outcome_state.value)
        assert Contributor.SELF not in assessment.contributors
        assert assessment.comparison is Comparison.MISMATCH
        # No Feedback, by scan or by id.
        assert _feedback(state).status is FeedbackCommitStatus.NONE_LEFT
        held = _feedback(state, assessment.assessment_id)
        assert (held.status, held.reason) == (FeedbackCommitStatus.HOLD, R17FeedbackReason.NOT_ELIGIBLE)
        assert _rows(state, K.R17_FEEDBACK_COMMIT) == []
        # The subject is not held: the closed forward obligation is row 2c.
        result = ip16b._preview(state, gid, seen)
        assert seen.last.feedback is FeedbackState.ABSENT and seen.last.feedback_ref is None
        assert result.row is SuccessorRow.ROW_2C
        _no_failed_events(state)


def test_f09_a_foreign_change_is_environment_and_never_row_5(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """R20-F09: an external change gives the subject neither credit nor blame."""

    with TestClient(app) as client:
        state = client.app.state
        seen = _Seen(monkeypatch)
        gid, intent, _forecast, _claim = _chain_to_claim(state)
        _commit_outcome(state, intent, foreign_change=True)
        assessment = _assess(state).assessment
        assert assessment.contributors == (Contributor.ENVIRONMENT,)
        feedback = _feedback(state).feedback
        assert feedback.feedback_state is FeedbackState.MATCH_OTHER_OR_UNKNOWN
        result = ip16b._preview(state, gid, seen)
        assert seen.last.feedback is FeedbackState.MATCH_OTHER_OR_UNKNOWN
        assert result.row is SuccessorRow.ROW_6
        _no_failed_events(state)


def test_in_doubt_has_no_assessment_and_no_feedback(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A claim with no verified Outcome (IN_DOUBT) cannot yield any R17 record."""

    with TestClient(app) as client:
        state = client.app.state
        seen = _Seen(monkeypatch)
        gid, _intent, _forecast, _claim = _chain_to_claim(state)
        assert _assess(state).status is AssessmentCommitStatus.NONE_LEFT
        assert _feedback(state).status is FeedbackCommitStatus.NONE_LEFT
        held = _assess(state, "outcome-" + "0" * 64)
        assert (held.status, held.reason) == (AssessmentCommitStatus.HOLD, R17FeedbackReason.OUTCOME_UNKNOWN)
        held_fb = _feedback(state, "assessment-" + "0" * 64)
        assert held_fb.reason is R17FeedbackReason.ASSESSMENT_UNKNOWN
        assert _rows(state, K.R17_ASSESSMENT_COMMIT, K.R17_FEEDBACK_COMMIT) == []
        result = ip16b._preview(state, gid, seen)
        assert (result.row, seen.last.feedback) == (SuccessorRow.ROW_2, FeedbackState.ABSENT)
        _no_failed_events(state)


def test_a_stale_revision_is_never_given_feedback(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        _gid, intent, _forecast, _claim = _chain_to_claim(state)
        _commit_outcome(state, intent)
        assessment = _assess(state).assessment
        _commit_outcome(state, intent, revision=2)
        held = _feedback(state, assessment.assessment_id)
        assert (held.status, held.reason) == (FeedbackCommitStatus.HOLD, R17FeedbackReason.NOT_ELIGIBLE)
        # The new revision gets its own Assessment, then its own Feedback.
        second = _assess(state)
        assert second.status is AssessmentCommitStatus.COMMITTED
        assert second.assessment.outcome_revision == 2
        assert _feedback(state).assessment_id == second.assessment.assessment_id
        _no_failed_events(state)


def test_a_missing_forecast_writes_no_assessment_or_feedback(app) -> None:  # type: ignore[no-untyped-def]
    """No forecast for the forward key: the owner refuses, the projection is
    INVALID (row 0) and the cycle ends on a HOLD receipt (D-R17 §3)."""

    with TestClient(app) as client:
        state = client.app.state
        _gid, intent, _forecast, _claim = _chain_to_claim(state)
        outcome = _commit_outcome(state, intent)
        owner: FeedbackOwnerV1 = state.feedback_owner
        reader = SubjectHistoryReader(state.event_journal, state.subject_history_store)
        classified = reader.classify()
        records = tuple(
            RecordAt(item.record, item.processing_sequence)
            for item in classified.records
            if item.record_kind is not R.ACTION_FORECAST
        )
        with pytest.raises(_Hold) as raised:
            owner._context(classified, records, outcome)
        assert raised.value.reason is R17FeedbackReason.FORECAST_MISSING


def test_a_forecast_after_the_claim_is_not_assessed(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        _gid, intent, forecast, claim = _chain_to_claim(state)
        outcome = _commit_outcome(state, intent)
        owner: FeedbackOwnerV1 = state.feedback_owner
        classified = SubjectHistoryReader(state.event_journal, state.subject_history_store).classify()
        records = tuple(
            RecordAt(
                replace(item.record, admitting_sequence=claim.claim_sequence + 1)
                if item.record_kind is R.ACTION_FORECAST
                else item.record,
                item.processing_sequence,
            )
            for item in classified.records
        )
        assert isinstance(claim, ExecutionClaimV1) and forecast.admitting_sequence < claim.claim_sequence
        with pytest.raises(_Hold) as raised:
            owner._context(classified, records, outcome)
        assert raised.value.reason is R17FeedbackReason.BINDING_MISMATCH


def test_a_scan_skips_a_candidate_that_does_not_qualify(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """One broken group never blocks the others: the scan moves on."""

    with TestClient(app) as client:
        state = client.app.state
        _gid, intent, _forecast, _claim = _chain_to_claim(state)
        _commit_outcome(state, intent)
        owner: FeedbackOwnerV1 = state.feedback_owner
        real = owner._context

        def broken(classified, records, outcome):  # type: ignore[no-untyped-def]
            raise _Hold(R17FeedbackReason.BINDING_MISMATCH, outcome.verified_outcome_id)

        monkeypatch.setattr(owner, "_context", broken)
        assert _assess(state).status is AssessmentCommitStatus.NONE_LEFT
        held = _assess(state, _adopted(state, VerifiedOutcomeV1)[0].verified_outcome_id)
        assert held.reason is R17FeedbackReason.BINDING_MISMATCH
        monkeypatch.setattr(owner, "_context", real)
        assert _assess(state).status is AssessmentCommitStatus.COMMITTED


# --------------------------------------------------------- gating, commit


def test_disabled_r18_holds_before_any_read_and_the_driver_gets_no_owner(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(_app(_settings(tmp_path / "off", r18=False))) as client:
        state = client.app.state
        owner = state.feedback_owner
        assert type(owner) is FeedbackOwnerV1 and not owner.enabled
        assert state.r16_driver._feedback is None

        def no_read(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("read while r18 is disabled")

        with monkeypatch.context() as patch:
            patch.setattr(SubjectHistoryReader, "classify", no_read)
            results = [
                _assess(state),
                _assess(state, "outcome-" + "0" * 64),
                _feedback(state),
                _feedback(state, "assessment-" + "0" * 64),
            ]
        for held in results:
            assert held.reason is R17FeedbackReason.R18_DISABLED
        assert _rows(state, K.R17_ASSESSMENT_COMMIT, K.R17_FEEDBACK_COMMIT) == []
        assert state.subject_history_store.has_record_operations() is False
        _no_failed_events(state)


def test_production_with_r18_off_posts_no_feedback_event(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with TestClient(_app(_settings(tmp_path / "prod", r18=False))) as client:
        state = client.app.state
        assert _events(state, ASSESS) == 0 and _events(state, FEEDBACK) == 0
        state.r16_driver.notify_outcome_committed()  # a no-op without an owner
        assert state.r16_driver._queue.empty()
    # R18 on but an empty store and an inactive driver: still no event.
    with TestClient(_app(_settings(tmp_path / "empty"))) as client:
        state = client.app.state
        assert _events(state, ASSESS) == 0 and _events(state, FEEDBACK) == 0
        state.r16_driver.notify_outcome_committed()  # inactive: the activation scan covers it
        assert state.r16_driver._queue.empty()


def test_validate_commit_sees_exactly_its_own_prepared_row(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        owner = state.feedback_owner
        _gid, intent, _forecast, _claim = _chain_to_claim(state)
        _commit_outcome(state, intent)
        seen: list[str] = []
        real = owner._validate_commit

        def spy(event):  # type: ignore[no-untyped-def]
            if owner._pending is not None:
                seen.extend(row.phase.value for row in _rows(state, K.R17_ASSESSMENT_COMMIT))
            return real(event)

        owner._validate_commit = spy  # type: ignore[method-assign]
        try:
            assert _assess(state).status is AssessmentCommitStatus.COMMITTED
        finally:
            owner._validate_commit = real  # type: ignore[method-assign]
        assert seen == [HistoryPhase.PREPARED.value]
        assert owner._pending is None


@pytest.mark.parametrize("which", ("assessment", "feedback"))
def test_crash_after_prepare_is_tombstoned_then_recommitted(tmp_path, which) -> None:  # type: ignore[no-untyped-def]
    settings = _settings(tmp_path / f"crash-{which}")
    target = ASSESS if which == "assessment" else FEEDBACK
    kind = K.R17_ASSESSMENT_COMMIT if which == "assessment" else K.R17_FEEDBACK_COMMIT
    with TestClient(_app(settings)) as client:
        state = client.app.state
        _gid, intent, _forecast, _claim = _chain_to_claim(state)
        _commit_outcome(state, intent)
        if which == "feedback":
            assert _assess(state).status is AssessmentCommitStatus.COMMITTED

        def crash(event):  # type: ignore[no-untyped-def]
            if event.event_type is target:
                raise RuntimeError("simulated pre-WAL failure")

        state.feedback_owner._validate_commit = crash  # type: ignore[method-assign]
        with pytest.raises(AgentRuntimeDurabilityError):
            (_assess if which == "assessment" else _feedback)(state)
    with TestClient(_app(settings)) as client:
        state = client.app.state
        assert state.event_journal.inspect().open_transactions == ()
        assert [row.phase for row in _rows(state, kind)][-1] is HistoryPhase.ABORTED
        if which == "assessment":
            assert _assess(state).status is AssessmentCommitStatus.COMMITTED
        else:
            assert _feedback(state).status is FeedbackCommitStatus.COMMITTED


def test_crash_after_internal_commit_rolls_forward_then_existing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from suzka.r16.history_participant import SubjectHistoryRecordParticipant

    settings = _settings(tmp_path / "crash-finalize")
    with TestClient(_app(settings)) as client:
        state = client.app.state
        _gid, intent, _forecast, _claim = _chain_to_claim(state)
        outcome = _commit_outcome(state, intent)
        real = SubjectHistoryRecordParticipant.finalize

        def crash(self, binding):  # type: ignore[no-untyped-def]
            raise RuntimeError("simulated post-R06 failure")

        SubjectHistoryRecordParticipant.finalize = crash  # type: ignore[method-assign]
        try:
            with pytest.raises(AgentRuntimeDurabilityError):
                _assess(state)
        finally:
            SubjectHistoryRecordParticipant.finalize = real  # type: ignore[method-assign]
    with TestClient(_app(settings)) as client:
        state = client.app.state
        assert state.event_journal.inspect().open_transactions == ()
        assert [row.phase for row in _rows(state, K.R17_ASSESSMENT_COMMIT)] == [HistoryPhase.FINALIZED]
        assert _assess(state, outcome.verified_outcome_id).status is AssessmentCommitStatus.EXISTING
        assert _assess(state).status is AssessmentCommitStatus.NONE_LEFT


def test_an_unadopted_prepared_assessment_is_in_flight_then_not_yet_terminal(app) -> None:  # type: ignore[no-untyped-def]
    """A PREPARED / FINALIZED-but-not-adopted Assessment holds its key
    (transient), and a Feedback request for it is NOT_YET_TERMINAL."""

    with TestClient(app) as client:
        state = client.app.state
        owner = state.feedback_owner
        _gid, intent, _forecast, _claim = _chain_to_claim(state)
        _commit_outcome(state, intent)
        seen: list[Any] = []
        real = owner._validate_commit

        def spy(event):  # type: ignore[no-untyped-def]
            if owner._pending is not None:
                classified = SimpleNamespace(
                    rows=state.subject_history_store.operations(ip10._incarnation(state))
                )
                record = owner._pending.operation.record
                try:
                    owner._assessment_missing(classified, record.assessment_id)  # type: ignore[arg-type]
                except _Hold as hold:
                    seen.append(hold.reason)
                seen.append(
                    bool(state.subject_history_store.live_keys(owner._pending.operation.reserved_keys))
                )
            return real(event)

        owner._validate_commit = spy  # type: ignore[method-assign]
        try:
            assert _assess(state).status is AssessmentCommitStatus.COMMITTED
        finally:
            owner._validate_commit = real  # type: ignore[method-assign]
        assert seen == [R17FeedbackReason.NOT_YET_TERMINAL, True]
