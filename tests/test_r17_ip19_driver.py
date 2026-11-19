"""IP-19: the R16 driver's Assessment -> Feedback chain, the executor's
``on_outcome`` hook and the structural pins of the new owner.

The driver's thread is not started: steps run synchronously with fake owners,
as in the IP-16a / IP-17a driver tests.
"""

from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any

from suzka.r16.driver import R16Driver, _FeedbackScan
from suzka.r17.feedback_owner import (
    MAX_FEEDBACK_SCANS_PER_STARTUP,
    AssessmentCommitStatus,
    FeedbackCommitStatus,
    R17FeedbackReason,
    _AssessmentResult,
    _FeedbackResult,
)
from suzka.r18.executor import R18Executor
from suzka.r18.outcome_owner import OutcomeStatus, _OutcomeResult
from suzka.r18.sweep import SweepGate
from suzka.runtime.agent_runtime import AgentEventOutcome

import test_r18_ip16a2_intent_issuer as ip16a2
from test_r16_ip14a_history_v2 import Chain
import test_r16_ip16b_ledger_reader as ip16b

ROOT = Path(__file__).resolve().parent.parent


class _FakeFeedback(ip16a2._FakeOwner):
    def _submit_assessment(self, outcome_id: str | None) -> Future[Any]:
        return self._record("assessment", outcome_id)

    def _submit_feedback(self, assessment_id: str | None) -> Future[Any]:
        return self._record("feedback", assessment_id)


def _driver(*, active: bool = True, feedback: _FakeFeedback | None = None):  # type: ignore[no-untyped-def]
    owner = _FakeFeedback() if feedback is None else feedback
    clock = ip16a2._Clock()
    driver = R16Driver(
        ip16a2._FakeRuntime(),  # type: ignore[arg-type]
        *[ip16a2._FakeOwner() for _ in range(3)],  # type: ignore[arg-type]
        ip16a2._FakeOwner(),  # type: ignore[arg-type]
        active=active,
        clock=clock,
        feedback_owner=owner,  # type: ignore[arg-type]
    )
    return driver, owner, clock


def _resolve(future: Future[Any], value: object) -> None:
    future.set_result(AgentEventOutcome(ip16a2._event(), value))


def _assessed(assessment_id: str, *, eligible: bool = True) -> _AssessmentResult:
    fake = SimpleNamespace(
        assessment_id=assessment_id,
        verified_outcome_id="outcome-" + assessment_id,
        positive_feedback_eligible=eligible,
    )
    return _AssessmentResult(
        AssessmentCommitStatus.COMMITTED, "outcome-" + assessment_id, assessment_id, None, fake  # type: ignore[arg-type]
    )


def _fed(assessment_id: str) -> _FeedbackResult:
    fake = SimpleNamespace(feedback_id="feedback-" + assessment_id, assessment_id=assessment_id)
    return _FeedbackResult(
        FeedbackCommitStatus.COMMITTED, assessment_id, "feedback-" + assessment_id, None, fake  # type: ignore[arg-type]
    )


def _none_left_assessment() -> _AssessmentResult:
    return _AssessmentResult(AssessmentCommitStatus.NONE_LEFT, None, None, None)


def _none_left_feedback() -> _FeedbackResult:
    return _FeedbackResult(FeedbackCommitStatus.NONE_LEFT, None, None, None)


def _pump(driver: R16Driver) -> int:
    seen = 0
    while not driver._queue.empty():
        item = driver._queue.get_nowait()
        if isinstance(item, _FeedbackScan):
            driver._on_feedback_scan()
            seen += 1
    return seen


# ------------------------------------------------------------- the chain


def test_a_scan_chain_follows_each_authorized_assessment_with_its_feedback() -> None:
    driver, owner, _clock = _driver()
    driver._on_feedback_scan()
    assert owner.calls == [("assessment", None)]
    _resolve(owner.futures[-1], _assessed("a1"))
    ip16a2._drain(driver)
    # One Feedback for that Assessment, then the next Assessment scan.
    assert owner.calls[1:] == [("feedback", "a1"), ("assessment", None)]
    _resolve(owner.futures[1], _fed("a1"))
    _resolve(owner.futures[2], _assessed("a2", eligible=False))
    ip16a2._drain(driver)
    # An Assessment with no positive Feedback (VERIFIED_ABSENT /
    # UNAUTHORIZED_EFFECT) is followed by no Feedback request, only the scan.
    assert owner.calls[3:] == [("assessment", None)]
    _resolve(owner.futures[3], _none_left_assessment())
    ip16a2._drain(driver)
    # NONE_LEFT ends the Assessment links; the Feedback scan catches a Feedback
    # a restart lost, and continues only while it COMMITs.
    assert owner.calls[4:] == [("feedback", None)]
    _resolve(owner.futures[4], _fed("a3"))
    ip16a2._drain(driver)
    assert owner.calls[5:] == [("feedback", None)]
    _resolve(owner.futures[5], _none_left_feedback())
    ip16a2._drain(driver)
    assert len(owner.calls) == 6


def test_a_hold_on_the_assessment_scan_still_runs_the_feedback_scan() -> None:
    driver, owner, _clock = _driver()
    driver._on_feedback_scan()
    _resolve(
        owner.futures[-1],
        _AssessmentResult(
            AssessmentCommitStatus.HOLD, None, None, R17FeedbackReason.HISTORY_CONFLICT
        ),
    )
    ip16a2._drain(driver)
    assert owner.calls == [("assessment", None), ("feedback", None)]
    _resolve(
        owner.futures[-1],
        _FeedbackResult(FeedbackCommitStatus.HOLD, None, None, R17FeedbackReason.IN_FLIGHT),
    )
    ip16a2._drain(driver)
    assert len(owner.calls) == 2  # a scan HOLD ends the chain; no retry loop


def test_a_transient_feedback_hold_is_retried_with_the_bounded_delays() -> None:
    driver, owner, clock = _driver()
    driver._maybe_feedback("a1")
    for attempt in range(4):
        _resolve(
            owner.futures[-1],
            _FeedbackResult(
                FeedbackCommitStatus.HOLD,
                "a1",
                None,
                R17FeedbackReason.NOT_YET_TERMINAL,
            ),
        )
        ip16a2._drain(driver)
        clock.now += 10
        ip16a2._drain(driver)
        assert len(owner.calls) == min(attempt + 2, 4)  # one submit + at most 3 retries
    assert all(call == ("feedback", "a1") for call in owner.calls)
    # A non-transient HOLD is not retried; and one Feedback per Assessment.
    other, owner2, clock2 = _driver()
    other._maybe_feedback("a2")
    _resolve(
        owner2.futures[-1],
        _FeedbackResult(FeedbackCommitStatus.HOLD, "a2", None, R17FeedbackReason.NOT_ELIGIBLE),
    )
    ip16a2._drain(other)
    clock2.now += 10
    ip16a2._drain(other)
    other._maybe_feedback("a2")
    assert len(owner2.calls) == 1


def test_the_chain_is_bounded_per_scan() -> None:
    driver, owner, _clock = _driver()
    driver._on_feedback_scan()
    for index in range(MAX_FEEDBACK_SCANS_PER_STARTUP + 4):
        if index >= len(owner.futures):
            break
        future = owner.futures[index]
        name = owner.calls[index][0]
        if name == "assessment":
            _resolve(future, _assessed(f"a{index}", eligible=False))
        else:
            _resolve(future, _fed(f"a{index}"))
        ip16a2._drain(driver)
    assessments = [call for call in owner.calls if call[0] == "assessment"]
    assert len(assessments) == MAX_FEEDBACK_SCANS_PER_STARTUP


def test_an_outcome_notification_restarts_a_chain_with_fresh_links() -> None:
    driver, owner, _clock = _driver()
    driver._on_feedback_scan()
    for index in range(MAX_FEEDBACK_SCANS_PER_STARTUP):
        _resolve(owner.futures[index], _assessed(f"a{index}", eligible=False))
        ip16a2._drain(driver)
    used = len(owner.calls)
    assert driver._assess_scans == MAX_FEEDBACK_SCANS_PER_STARTUP
    driver.notify_outcome_committed()
    assert _pump(driver) == 1
    assert driver._assess_scans == 1 and len(owner.calls) == used + 1


def test_notify_is_a_no_op_while_inactive_stopped_or_without_an_owner() -> None:
    inactive, owner, _ = _driver(active=False)
    inactive.notify_outcome_committed()
    assert inactive._queue.empty()
    stopped, _owner, _ = _driver()
    stopped._stop.set()
    stopped.notify_outcome_committed()
    assert stopped._queue.empty()
    off, _owner2, _ = _driver(feedback=_FakeFeedback(enabled=False))
    off.notify_outcome_committed()
    off._post_feedback_scan()
    assert off._queue.empty() and not owner.calls
    none = R16Driver(
        ip16a2._FakeRuntime(),  # type: ignore[arg-type]
        *[ip16a2._FakeOwner() for _ in range(4)],  # type: ignore[arg-type]
        active=True,
    )
    none.notify_outcome_committed()
    assert none._queue.empty()


def test_the_activation_scan_is_posted_once_per_incarnation() -> None:
    driver, _owner, _clock = _driver(active=False)
    driver.start()
    try:
        driver.activate()
        driver.activate()
        assert driver._feedback_scan_posted
    finally:
        driver.stop()


# ----------------------------------------------------- the executor's hook


def test_an_outcome_result_calls_the_hook_once_and_a_failing_hook_changes_nothing(tmp_path: Path) -> None:
    from test_r18_ip18a_executor import _FakeOutcome

    chain = ip16b._fixture_chain()
    assert isinstance(chain, Chain)
    calls: list[int] = []

    def hook() -> None:
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("hook failure")

    executor = R18Executor(
        target_path=tmp_path / "target.sqlite3",
        runtime_instance_id="00000000-0000-4000-8000-0000000000ee",
        gate=SweepGate(enabled=True),
        outcome_owner=_FakeOutcome(tmp_path / "target.sqlite3", (None, None)),
        read_records=lambda: None,  # type: ignore[arg-type,return-value]
        has_exclusive_authority=lambda: True,
        on_outcome=hook,
        threaded=False,
    )
    result = _OutcomeResult(OutcomeStatus.COMMITTED, chain.claim.claim_id, None, chain.outcome)
    executor._outcome_value(chain.claim, result)
    executor._outcome_value(chain.claim, result)  # the hook raises: swallowed
    existing = _OutcomeResult(OutcomeStatus.EXISTING, chain.claim.claim_id, None, chain.outcome)
    executor._outcome_value(chain.claim, existing)
    assert len(calls) == 3
    # A HOLD is not an Outcome: the hook is not called.
    from suzka.r18.outcome_owner import R18OutcomeReason

    executor._outcome_value(
        chain.claim, _OutcomeResult(OutcomeStatus.HOLD, chain.claim.claim_id, R18OutcomeReason.UNDECIDABLE)
    )
    assert len(calls) == 3


# ---------------------------------------------------------- structural pins


def _text(path: str) -> str:
    return (ROOT / path).read_text()


def test_only_the_feedback_owner_submits_the_r17_assessment_and_feedback_events() -> None:
    users = sorted(
        str(path.relative_to(ROOT))
        for path in (ROOT / "suzka").rglob("*.py")
        if re.search(r"AgentEventType\.R17_(ASSESSMENT|FEEDBACK)_COMMIT", path.read_text())
    )
    # The server only checks the owner's availability for the event.
    assert users == [
        "suzka/api/server.py",
        "suzka/r17/feedback_owner.py",
        "suzka/runtime/event_journal.py",
    ]
    assert "AgentEventType.R17" not in _text("suzka/r16/driver.py")


def test_the_driver_decides_nothing_it_only_submits() -> None:
    text = _text("suzka/r16/driver.py")
    start = text.index("# -- IP-19: Assessment -> Feedback")
    end = text.index("def _on_rescan")
    chain = text[start:end]
    assert not re.search(r"classify|SubjectHistoryReader|_store\b|derive_ledger|read_feedback", chain)


def test_the_policy_allow_lists_that_pin_the_replay_digests_are_untouched() -> None:
    """IP-19 reads Feedback; it does not change which Feedback the policy
    admits (the adapter, not a narrower allow-list, is the defence), so the
    pinned V2 / V3 fingerprints cannot move."""

    from suzka.r16.successor_policy import (
        _FEEDBACK_ALLOWED_LEDGER,
        _MATCH_SELF_LEDGER,
        LedgerState,
    )

    assert LedgerState.UNAUTHORIZED_EFFECT in _FEEDBACK_ALLOWED_LEDGER
    assert LedgerState.UNAUTHORIZED_EFFECT not in _MATCH_SELF_LEDGER
    assert LedgerState.ACTIVE_VERIFIED_AUTHORIZED in _MATCH_SELF_LEDGER


def test_the_owner_never_opens_the_target_or_the_nonce_records() -> None:
    text = _text("suzka/r17/feedback_owner.py")
    for forbidden in ("TargetReader", "TargetStore", "verify_forward", "approval_nonce", "nonce_consum"):
        assert forbidden not in text, forbidden
    # R17 reads the Outcome through its own view, never R18's nonce records.
    assert "to_r17_view" in text


def test_the_new_events_are_admitted_in_the_soft_band_by_name() -> None:
    text = _text("suzka/runtime/event_journal.py")
    block = text[text.index("SOFT_BAND_ADMITTED_EVENT_TYPES: frozenset") :]
    assert "AgentEventType.R17_ASSESSMENT_COMMIT" in block
    assert "AgentEventType.R17_FEEDBACK_COMMIT" in block
