"""IP-19 end to end: the real app with the real executor, Outcome owner and
Feedback owner, driven by the R16 driver (S0 steps 4 / 5 -> R17).

Every fixture is synthetic; nothing here touches a production target.  The
Feedback is a MATCH_SELF of a nominal, code-derived forecast: it shows the
pipeline runs, not that the subject learned (D-R17 §6).
"""

from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from suzka.r16.decision_issuer import DecisionIssueStatus
from suzka.r16.successor_policy import FeedbackState, SuccessorRow
from suzka.r17.contracts import DecisionFeedbackV1, OutcomeAssessmentV1
from suzka.r18.contracts import VerifiedOutcomeV1, VerifiedState

import test_r16_ip16b_ledger_reader as ip16b
import test_r18_ip17c_claim as ip17c
import test_r18_ip18a_flow as flow

_Clock = flow._Clock
_adopted = flow._adopted
_allowed = flow._allowed
_fake_timers = flow._fake_timers
_live = flow._live
_open = flow._open
_until = flow._until
_no_failed_events = flow._no_failed_events


@pytest.fixture
def settings(tmp_path):  # type: ignore[no-untyped-def]
    return flow._settings(tmp_path / "ip19flow")


def test_an_active_driver_turns_an_executed_outcome_into_feedback_and_row_5(settings) -> None:  # type: ignore[no-untyped-def]
    with TestClient(_live(settings)) as client:
        state = client.app.state
        _open(state)
        _fake_timers(state)
        _Clock(state)
        gid, intent, _forecast = _allowed(state)
        state.r16_driver.activate()
        _challenge, done = ip17c._confirm(state, intent.intent_id)
        assert done.claim is not None
        outcome = _until(lambda: (_adopted(state, VerifiedOutcomeV1) or [None])[0])
        assert outcome.outcome_state is VerifiedState.ACTIVE_VERIFIED_AUTHORIZED
        # The executor's notification starts the driver's chain: Assessment,
        # then the one Feedback, with no further call from the test.
        feedback = _until(lambda: (_adopted(state, DecisionFeedbackV1) or [None])[0])
        assessment = _adopted(state, OutcomeAssessmentV1)[0]
        assert feedback.assessment_id == assessment.assessment_id
        assert feedback.feedback_state is FeedbackState.MATCH_SELF
        result = ip16b._preview(state, gid)
        assert (result.status, result.row) == (DecisionIssueStatus.ISSUED, SuccessorRow.ROW_5)
        assert ip16b._preview(state, gid).status is DecisionIssueStatus.EXISTING
        # Exactly one of each (a repeat is EXISTING / NONE_LEFT, never a second row).
        assert len(_adopted(state, OutcomeAssessmentV1)) == 1
        assert len(_adopted(state, DecisionFeedbackV1)) == 1
        _no_failed_events(state)


def test_an_inactive_driver_writes_nothing_and_the_activation_scan_catches_up(settings) -> None:  # type: ignore[no-untyped-def]
    with TestClient(_live(settings)) as client:
        state = client.app.state
        _open(state)
        _fake_timers(state)
        _Clock(state)
        gid, intent, _forecast = _allowed(state)
        assert not state.r16_driver.active
        ip17c._confirm(state, intent.intent_id)
        _until(lambda: _adopted(state, VerifiedOutcomeV1))
        # The executor's notification is a no-op while the driver is inactive:
        # the Outcome stands alone (row 6, Feedback ABSENT).
        assert _adopted(state, OutcomeAssessmentV1) == []
        assert ip16b._preview(state, gid).row is SuccessorRow.ROW_6
        # Activating the driver (or a restart) runs the scan chain once.
        state.r16_driver.activate()
        _until(lambda: _adopted(state, DecisionFeedbackV1))
        assert ip16b._preview(state, gid).row is SuccessorRow.ROW_5
        _no_failed_events(state)
