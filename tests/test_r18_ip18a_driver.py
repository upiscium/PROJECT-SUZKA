"""IP-18a (#411 IP-18 D1 r4 §3-6): the R16 driver re-submits an Intent / Policy
stage that met a closed AdmissionGate when the gate opens.

The driver's thread is not started: steps run synchronously, as in the IP-16a /
IP-17a driver tests (fake owners, a real ``SweepGate``).
"""

from __future__ import annotations

from concurrent.futures import Future
from typing import Any

from suzka.r16.driver import R16Driver, _GateOpen
from suzka.r16.successor_policy import SuccessorRow
from suzka.r18.intent_issuer import IntentIssueStatus, R18IntentReason, _IntentIssueResult
from suzka.r18.policy_owner import (
    MAX_POLICY_SCANS_PER_STARTUP,
    PolicyCommitStatus,
    R18PolicyReason,
    _PolicyResult,
)
from suzka.r18.sweep import SweepGate
from suzka.runtime.agent_runtime import AgentEventOutcome

import test_r18_ip16a2_intent_issuer as ip16a2
import test_r18_ip17a_policy as ip17a
import test_r17_ip16a3_expectation_issuer as ip16a3


def _gated_driver(gate: SweepGate | None, *, policy: bool = False):  # type: ignore[no-untyped-def]
    intent = ip16a2._FakeOwner()
    preview = ip16a2._FakeOwner()
    policy_owner = ip17a._FakePolicy() if policy else None
    clock = ip16a2._Clock()
    driver = R16Driver(
        ip16a2._FakeRuntime(),  # type: ignore[arg-type]
        *[ip16a2._FakeOwner() for _ in range(3)],  # type: ignore[arg-type]
        preview,  # type: ignore[arg-type]
        active=True,
        clock=clock,
        intent_issuer=intent,  # type: ignore[arg-type]
        expectation_issuer=ip16a3._FakeExpectation() if policy else None,  # type: ignore[arg-type]
        policy_owner=policy_owner,  # type: ignore[arg-type]
        gate=gate,
    )
    return driver, intent, preview, policy_owner, clock


def _pump(driver: R16Driver) -> int:
    """What the driver thread does with its own queue (gate transitions only)."""

    seen = 0
    while not driver._queue.empty():
        item = driver._queue.get_nowait()
        if isinstance(item, _GateOpen):
            driver._on_gate_open()
            seen += 1
    return seen


def _resolve(future: Future[Any], value: object) -> None:
    future.set_result(AgentEventOutcome(ip16a2._event(), value))


def _gate_closed() -> _IntentIssueResult:
    return _IntentIssueResult(IntentIssueStatus.HOLD, None, R18IntentReason.ADMISSION_GATE_CLOSED)


def _row4(driver: R16Driver, preview: ip16a2._FakeOwner, decision_id: str = "d1") -> None:
    driver._maybe_preview(f"goal-{len(preview.calls)}", "commitment_accepted")
    _resolve(preview.futures[-1], ip16a2._decision_result(SuccessorRow.ROW_4, decision_id=decision_id))
    ip16a2._drain(driver)


def test_a_row_4_head_met_while_the_gate_is_closed_is_queued_and_submitted_once_on_open() -> None:
    gate = SweepGate(enabled=True)
    driver, intent, preview, _policy, clock = _gated_driver(gate)
    driver._subscribe_gate()
    assert _pump(driver) == 0  # closed: no callback yet
    _row4(driver, preview)
    assert len(intent.calls) == 1
    _resolve(intent.futures[-1], _gate_closed())
    ip16a2._drain(driver)
    # Dropped for the moment: no retry loop, however long we wait ...
    for _ in range(5):
        clock.now += 60
        ip16a2._drain(driver)
    assert len(intent.calls) == 1
    assert "d1" in driver._gate_waiting and "d1" not in driver._intended
    # ... and re-submitted exactly once when the gate opens.
    gate.open()
    assert _pump(driver) == 1
    assert len(intent.calls) == 2 and not driver._gate_waiting
    _resolve(
        intent.futures[-1],
        _IntentIssueResult(IntentIssueStatus.HOLD, None, R18IntentReason.TARGET_UNAVAILABLE),
    )
    ip16a2._drain(driver)
    # The same Decision again (EXISTING) is not resubmitted; neither is an
    # unrelated second OPEN transition.
    _row4(driver, preview)
    gate.sweeping()
    gate.open()
    _pump(driver)
    assert len(intent.calls) == 2


def test_a_blocked_gate_gives_no_intent_and_no_retry_loop() -> None:
    gate = SweepGate(enabled=True)
    driver, intent, preview, _policy, clock = _gated_driver(gate)
    driver._subscribe_gate()
    _row4(driver, preview)
    _resolve(intent.futures[-1], _gate_closed())
    ip16a2._drain(driver)
    gate.blocked()
    for _ in range(10):
        clock.now += 120
        ip16a2._drain(driver)
        _pump(driver)
    assert len(intent.calls) == 1


def test_subscribing_to_an_already_open_gate_fires_at_once() -> None:
    gate = SweepGate(enabled=True)
    gate.open()
    driver, _intent, _preview, _policy, _clock = _gated_driver(gate)
    driver._subscribe_gate()
    assert _pump(driver) == 1
    driver._subscribe_gate()  # idempotent
    assert _pump(driver) == 0


def test_the_gate_callback_only_posts_and_is_a_no_op_after_stop() -> None:
    gate = SweepGate(enabled=True)
    driver, intent, preview, _policy, _clock = _gated_driver(gate)
    driver._subscribe_gate()
    _row4(driver, preview)
    _resolve(intent.futures[-1], _gate_closed())
    ip16a2._drain(driver)
    driver._stop.set()
    gate.open()
    assert not any(isinstance(i, _GateOpen) for i in list(driver._queue.queue))  # nothing posted
    assert "d1" in driver._gate_waiting  # the driver's dicts are not touched by the gate thread
    assert len(intent.calls) == 1


def test_a_policy_stage_that_met_a_closed_gate_is_rescanned_on_open_with_the_counter_reset() -> None:
    gate = SweepGate(enabled=True)
    driver, _intent, _preview, policy, _clock = _gated_driver(gate, policy=True)
    assert policy is not None
    driver._subscribe_gate()
    driver._maybe_policy("intent-a")
    _resolve(
        policy.futures[-1],
        _PolicyResult(PolicyCommitStatus.HOLD, "intent-a", R18PolicyReason.ADMISSION_GATE_CLOSED),
    )
    ip16a2._drain(driver)
    assert policy.calls == [("policy", "intent-a")] and "intent-a" not in driver._policed
    # A flapping target (OPEN, closed, OPEN ... 20 times) never exhausts the
    # per-incarnation Policy scan counter: one scan per OPEN transition.
    for round_number in range(20):
        driver._policy_scans = MAX_POLICY_SCANS_PER_STARTUP - 1  # nearly spent
        gate.open()
        _pump(driver)
        assert policy.calls[-1] == ("scan", None), round_number
        assert driver._policy_scans == 1
        _resolve(policy.futures[-1], ip17a._result(PolicyCommitStatus.NONE_LEFT, None))
        ip16a2._drain(driver)
        gate.sweeping()
    scans = [call for call in policy.calls if call[0] == "scan"]
    assert len(scans) == 20


def test_each_scan_chain_is_still_capped_at_the_per_startup_limit() -> None:
    gate = SweepGate(enabled=True)
    driver, _intent, _preview, policy, _clock = _gated_driver(gate, policy=True)
    assert policy is not None
    driver._subscribe_gate()
    gate.open()
    _pump(driver)
    n = 0
    while not policy.futures[-1].done():
        n += 1
        _resolve(policy.futures[-1], _PolicyResult(PolicyCommitStatus.ALLOWED, f"i-{n}", None, ip17a._verdict(f"i-{n}")))  # type: ignore[arg-type]
        ip16a2._drain(driver)
    assert len(policy.calls) == MAX_POLICY_SCANS_PER_STARTUP


def test_without_a_gate_the_driver_is_unchanged() -> None:
    driver, intent, preview, _policy, _clock = _gated_driver(None)
    driver._subscribe_gate()
    assert driver._queue.empty()
    _row4(driver, preview)
    assert len(intent.calls) == 1


def test_a_gate_closed_result_processed_after_the_gate_opened_is_resubmitted_at_once() -> None:
    """Review P2-1: the driver learns of the result on its next loop; the OPEN
    message may be handled first. The target must not wait for a later OPEN."""

    gate = SweepGate(enabled=True)
    driver, intent, preview, _policy, _clock = _gated_driver(gate)
    driver._subscribe_gate()
    _row4(driver, preview)
    # The result arrives (future resolved) but the loop has not drained it ...
    _resolve(intent.futures[-1], _gate_closed())
    # ... the gate opens and its message is handled first (nothing waiting yet).
    gate.open()
    assert _pump(driver) == 1 and len(intent.calls) == 1
    ip16a2._drain(driver)  # ... and only now the HOLD is processed
    assert len(intent.calls) == 2 and not driver._gate_waiting


def test_a_policy_gate_closed_result_processed_after_the_gate_opened_is_rescanned_at_once() -> None:
    gate = SweepGate(enabled=True)
    driver, _intent, _preview, policy, _clock = _gated_driver(gate, policy=True)
    assert policy is not None
    driver._subscribe_gate()
    driver._maybe_policy("intent-a")
    _resolve(
        policy.futures[-1],
        _PolicyResult(PolicyCommitStatus.HOLD, "intent-a", R18PolicyReason.ADMISSION_GATE_CLOSED),
    )
    gate.open()
    _pump(driver)
    n = len(policy.calls)
    ip16a2._drain(driver)
    assert policy.calls[n:] == [("scan", None)]


def test_a_completed_submission_wakes_the_driver_loop() -> None:
    from suzka.r16.driver import _Wake

    gate = SweepGate(enabled=True)
    driver, intent, preview, _policy, _clock = _gated_driver(gate)
    _row4(driver, preview)
    driver._queue.queue.clear()
    _resolve(intent.futures[-1], _gate_closed())
    assert any(isinstance(i, _Wake) for i in list(driver._queue.queue))
