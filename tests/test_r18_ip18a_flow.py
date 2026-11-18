"""IP-18a end to end: the real app with the real executor, gate, Outcome owner
and approval owner (S0 steps 4 / 5 / 9, F05..F08, the claim feeders).

Every fixture is synthetic; nothing here touches a production target.
"""

from __future__ import annotations

from hashlib import sha256
import logging
from pathlib import Path
import re
import sqlite3
import threading
import time

from fastapi.testclient import TestClient
import pytest

from suzka.r16.history_records import HistoryOperationKind
from suzka.r16.history_store import HistoryPhase
from suzka.r16.successor_policy import LedgerState, SuccessorRow
from suzka.r18.approval_owner import ApprovalChallengeOwnerV1, ApprovalStatus, R18ApprovalReason
from suzka.r18.contracts import (
    ExecutionClaimV1,
    VerifiedOutcomeV1,
    VerifiedState,
)
from suzka.r18.outcome_owner import OutcomeStatus, R18OutcomeReason
from suzka.r18.sweep import GateState
from suzka.r18.target import TargetStore
from suzka.runtime.agent_runtime import AgentEventSource, AgentEventType
from suzka.runtime.event_journal import EventLifecycle

import test_r16_ip16b_ledger_reader as ip16b
import test_r18_ip16a2_intent_issuer as ip16a2
import test_r18_ip17b_approval as ip17b
import test_r18_ip17c_claim as ip17c
from test_r16_u2a_source_owner import _app

K = HistoryOperationKind
_settings = ip16a2._settings
_allowed = ip17b._allowed
_Clock = ip17b._Clock
_adopted = ip17b._adopted
_fake_timers = ip17b._fake_timers
_no_failed_events = ip16a2._no_failed_events


def _until(predicate, timeout: float = 40.0):  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("condition not reached")


def _live(settings):  # type: ignore[no-untyped-def]
    return _app(settings, executor=True)


def _open(state) -> None:  # type: ignore[no-untyped-def]
    _until(lambda: state.r18_gate.is_open())


def _outcomes(state) -> list[VerifiedOutcomeV1]:  # type: ignore[no-untyped-def]
    return _adopted(state, VerifiedOutcomeV1)


def _outcome_events(state) -> int:  # type: ignore[no-untyped-def]
    return sum(
        1
        for row in state.event_journal.records
        if row.event_type is AgentEventType.R18_OUTCOME_COMMIT
        and row.lifecycle is EventLifecycle.ACCEPTED
    )


def _target(state) -> Path:  # type: ignore[no-untyped-def]
    return Path(state.settings.r18.target_path)


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _status(state, intent):  # type: ignore[no-untyped-def]
    store = TargetStore.open(_target(state))
    try:
        return store.status(intent.obligation_group_key)
    finally:
        store.close()


def _ledger(state, intent) -> LedgerState:  # type: ignore[no-untyped-def]
    return ip17c._ledger(state, intent)


class _Spy:
    """Counts ``TargetStore.forward_write`` calls."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls = 0
        real = TargetStore.forward_write
        spy = self

        def counting(store, *args, **kwargs):  # type: ignore[no-untyped-def]
            spy.calls += 1
            return real(store, *args, **kwargs)

        monkeypatch.setattr(TargetStore, "forward_write", counting)


@pytest.fixture
def settings(tmp_path):  # type: ignore[no-untyped-def]
    return _settings(tmp_path / "ip18a")


# ============================================================ S0 steps 4 / 5


def test_s0_confirm_claim_one_effect_and_an_independent_outcome(settings, monkeypatch, caplog) -> None:  # type: ignore[no-untyped-def]
    caplog.set_level(logging.INFO, logger="suzka.r18")
    spy = _Spy(monkeypatch)
    with TestClient(_live(settings)) as client:
        state = client.app.state
        owner = state.approval_owner
        _open(state)
        _fake_timers(state)
        _Clock(state)
        gid, intent, _forecast = _allowed(state)
        challenge, done = ip17c._confirm(state, intent.intent_id)
        assert done.status is ApprovalStatus.CLAIMED and done.claim is not None
        outcome = _until(lambda: (_outcomes(state) or [None])[0])
        assert outcome.outcome_state is VerifiedState.ACTIVE_VERIFIED_AUTHORIZED
        assert outcome.operation_key == done.claim.operation_key and not outcome.foreign_change
        assert spy.calls == 1  # one write, once
        status = _status(state, intent)
        assert status.artifact is not None and status.forward is not None
        assert status.forward.approval_ref == challenge.nonce
        assert status.forward.runtime_instance_id == owner.runtime_instance_id
        # Exactly one Outcome event and one row; the ledger now has a verified effect.
        assert _outcome_events(state) == 1 and len(_outcomes(state)) == 1
        assert _ledger(state, intent) is LedgerState.ACTIVE_VERIFIED_AUTHORIZED
        # An Outcome with a ref is not row 0 (IP-19's input exists).
        assert ip16b._preview(state, gid).row is not SuccessorRow.ROW_0
        # No second effect from a second confirm.
        again = owner.claim(intent.intent_id, challenge.nonce, challenge.confirmation_code)
        assert again.status is ApprovalStatus.HOLD
        assert spy.calls == 1
        # A repeated Outcome request for the same claim writes nothing.
        result = state.outcome_owner._submit_outcome(done.claim.claim_id).result(timeout=30).value
        assert result.status is OutcomeStatus.EXISTING and len(_outcomes(state)) == 1
        assert state.r18_executor.unsettled_ids() == frozenset()
        _no_failed_events(state)
    text = caplog.text
    for stage in ("enqueued", "write", "settle", "outcome_submit", "outcome_commit"):
        assert re.search(rf"R18_EXEC .*stage={stage} ", text), stage
    assert "R18_SWEEP step=S5" in text


def test_f05_nothing_is_enqueued_for_a_held_claim(settings) -> None:  # type: ignore[no-untyped-def]
    with TestClient(_live(settings)) as client:
        state = client.app.state
        owner = state.approval_owner
        _open(state)
        _fake_timers(state)
        _Clock(state)
        fed: list[ExecutionClaimV1] = []
        owner._on_claimed = fed.append  # type: ignore[assignment]
        _gid, intent, _forecast = _allowed(state)
        challenge = ip17c._issued(state, intent.intent_id)
        # wrong code, unknown nonce: HOLD, nothing consumed, nothing fed.
        held = owner.claim(intent.intent_id, challenge.nonce, "000000")
        assert held.status is ApprovalStatus.HOLD and held.reason is R18ApprovalReason.CODE_MISMATCH
        unknown = owner.claim(intent.intent_id, "f" * 64, challenge.confirmation_code)
        assert unknown.status is ApprovalStatus.HOLD
        assert fed == []
        done = owner.claim(intent.intent_id, challenge.nonce, challenge.confirmation_code)
        assert done.status is ApprovalStatus.CLAIMED
        _until(lambda: fed)
        # The reused nonce: HOLD NONCE_USED, nothing more is fed.
        reused = owner.claim(intent.intent_id, challenge.nonce, challenge.confirmation_code)
        assert reused.reason is R18ApprovalReason.NONCE_USED
        assert len({c.claim_id for c in fed}) == 1


def test_a_callback_failure_never_changes_the_api_result(settings) -> None:  # type: ignore[no-untyped-def]
    with TestClient(_live(settings)) as client:
        state = client.app.state
        owner = state.approval_owner
        _open(state)
        _fake_timers(state)
        _Clock(state)

        def boom(claim: ExecutionClaimV1) -> None:
            raise RuntimeError("feeder failure")

        owner._on_claimed = boom  # type: ignore[assignment]
        _gid, intent, _forecast = _allowed(state)
        _challenge, done = ip17c._confirm(state, intent.intent_id)
        assert done.status is ApprovalStatus.CLAIMED and done.claim is not None


def test_a_claim_that_commits_after_claim_gave_up_is_settled_in_the_same_process(  # type: ignore[no-untyped-def]
    settings, monkeypatch
) -> None:
    """The Runtime is one FIFO thread: a claim queued behind a long event can
    commit long after ``result(timeout=60)`` gave up. The done-callback (attached
    BEFORE the wait) feeds it; no restart, no scan."""

    with TestClient(_live(settings)) as client:
        state = client.app.state
        owner = state.approval_owner
        _open(state)
        _fake_timers(state)
        _Clock(state)
        _gid, intent, _forecast = _allowed(state)
        challenge = ip17c._issued(state, intent.intent_id)
        monkeypatch.setattr(
            ApprovalChallengeOwnerV1,
            "_run",
            staticmethod(lambda future: future.result(timeout=0.4).value),
        )
        release = threading.Event()
        entered = threading.Event()

        def long_event() -> None:
            entered.set()
            release.wait(30)

        blocker = state.agent_runtime.submit(
            AgentEventType.CHAT, AgentEventSource.API_CHAT, long_event
        )
        assert entered.wait(10)
        gave_up = owner.claim(intent.intent_id, challenge.nonce, challenge.confirmation_code)
        assert (gave_up.status, gave_up.reason) == (
            ApprovalStatus.HOLD,
            R18ApprovalReason.JOURNAL_UNAVAILABLE,
        )
        assert state.r18_executor.handed_ids() == frozenset()
        release.set()
        try:
            blocker.result(timeout=30)
        except Exception:  # the dummy CHAT handler may fail; irrelevant here
            pass
        # The claim committed after claim() returned HOLD; it is fed and settled.
        outcome = _until(lambda: (_outcomes(state) or [None])[0])
        assert outcome.outcome_state is VerifiedState.ACTIVE_VERIFIED_AUTHORIZED
        assert len(_adopted(state, ExecutionClaimV1)) == 1


def test_a_lost_callback_is_found_by_the_slow_backstop_scan(settings) -> None:  # type: ignore[no-untyped-def]
    with TestClient(_live(settings)) as client:
        state = client.app.state
        owner = state.approval_owner
        executor = state.r18_executor
        _open(state)
        _fake_timers(state)
        _Clock(state)
        owner._on_claimed = None
        _gid, intent, _forecast = _allowed(state)
        _challenge, done = ip17c._confirm(state, intent.intent_id)
        assert done.status is ApprovalStatus.CLAIMED
        assert executor.handed_ids() == frozenset() and _outcomes(state) == []
        base = time.monotonic()
        executor._monotonic = lambda: base + 10_000.0  # the ten minutes pass
        executor._next_backstop = 0.0
        executor.tick()
        outcome = _until(lambda: (_outcomes(state) or [None])[0])
        assert outcome.outcome_state is VerifiedState.ACTIVE_VERIFIED_AUTHORIZED


def test_a_flapping_gate_holds_the_claim_consumes_nothing_and_writes_once(settings, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    spy = _Spy(monkeypatch)
    with TestClient(_live(settings)) as client:
        state = client.app.state
        owner = state.approval_owner
        _open(state)
        _fake_timers(state)
        _Clock(state)
        _gid, intent, _forecast = _allowed(state)
        challenge = ip17c._issued(state, intent.intent_id)
        gate = state.r18_gate
        for _ in range(5):
            gate.sweeping()
            held = owner.confirm(intent.intent_id, challenge.nonce, challenge.confirmation_code)
            assert (held.status, held.reason) == (ApprovalStatus.HOLD, R18ApprovalReason.ADMISSION_GATE_CLOSED)
            gate.open()
        assert ip17c._claim_rows(state) == [] and ip17c._consumptions(state) == []
        done = owner.confirm(intent.intent_id, challenge.nonce, challenge.confirmation_code)
        assert done.status is ApprovalStatus.CLAIMED
        _until(lambda: _outcomes(state))
        assert spy.calls == 1 and len(_outcomes(state)) == 1


# ============================================================ S0 step 9 (restart)


def _claim_without_executor(settings):  # type: ignore[no-untyped-def]
    """Process 1: a claim is adopted and the executor never runs (P1)."""

    with TestClient(_app(settings)) as client:
        state = client.app.state
        _fake_timers(state)
        gid, intent, _forecast = _allowed(state)
        challenge, done = ip17c._confirm(state, intent.intent_id)
        assert done.status is ApprovalStatus.CLAIMED
        return gid, intent, challenge, done.claim


def test_p1_claim_adopted_executor_never_ran_the_sweep_settles_absent_no_write(settings, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    gid, intent, _challenge, claim = _claim_without_executor(settings)
    spy = _Spy(monkeypatch)
    with TestClient(_live(settings)) as client:
        state = client.app.state
        _open(state)
        outcome = _until(lambda: (_outcomes(state) or [None])[0])
        assert outcome.outcome_state is VerifiedState.VERIFIED_ABSENT
        assert spy.calls == 0  # forward_write 0 times for a claim of an earlier process
        status = _status(state, intent)
        assert status.artifact is None and status.forward is not None
        assert status.forward.kind.value == "FENCED"
        assert _ledger(state, intent) is LedgerState.VERIFIED_ABSENT
        assert len(_outcomes(state)) == 1
        assert state.r18_executor.unsettled_ids() == frozenset()
        assert claim.runtime_instance_id != state.approval_owner.runtime_instance_id
        _no_failed_events(state)
    # A third process: nothing left to do, no second Outcome.
    with TestClient(_live(settings)) as client:
        state = client.app.state
        _open(state)
        time.sleep(0.3)
        assert len(_outcomes(state)) == 1 and spy.calls == 0


def test_p3_p4_effect_committed_but_the_outcome_never_ran_the_sweep_reads_it_back(settings, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(_live(settings)) as client:
        state = client.app.state
        _open(state)
        _fake_timers(state)
        _Clock(state)
        # The Outcome event is never admitted in this process (P4).
        def refuse(claim_id: str):  # type: ignore[no-untyped-def]
            from suzka.runtime.agent_runtime import AgentRuntimeAdmissionBlocked

            raise AgentRuntimeAdmissionBlocked(None)  # type: ignore[arg-type]

        state.outcome_owner._submit_outcome = refuse  # type: ignore[method-assign]
        _gid, intent, _forecast = _allowed(state)
        _challenge, done = ip17c._confirm(state, intent.intent_id)
        assert done.status is ApprovalStatus.CLAIMED
        _until(lambda: _status(state, intent).artifact is not None)
        time.sleep(0.2)
        assert _outcomes(state) == []
    spy = _Spy(monkeypatch)
    with TestClient(_live(settings)) as client:
        state = client.app.state
        _open(state)
        outcome = _until(lambda: (_outcomes(state) or [None])[0])
        assert outcome.outcome_state is VerifiedState.ACTIVE_VERIFIED_AUTHORIZED
        assert spy.calls == 0  # nothing re-sent after the restart
        assert _status(state, intent).forward.revision_after == 1  # type: ignore[union-attr]
        assert len(_outcomes(state)) == 1


def test_p5_a_crash_inside_the_outcome_op_tombstones_then_exactly_one_outcome_exists(settings) -> None:  # type: ignore[no-untyped-def]
    """F06: the Outcome op rides the existing record-operation machinery."""

    from suzka.runtime.agent_runtime import AgentRuntimeStopped

    with TestClient(_live(settings)) as client:
        state = client.app.state
        _open(state)
        _fake_timers(state)
        _Clock(state)

        def crash(event):  # type: ignore[no-untyped-def]
            if event.event_type is AgentEventType.R18_OUTCOME_COMMIT:
                raise RuntimeError("simulated pre-WAL failure")

        state.outcome_owner._validate_commit = crash  # type: ignore[method-assign]
        _gid, intent, _forecast = _allowed(state)
        _challenge, done = ip17c._confirm(state, intent.intent_id)
        assert done.status is ApprovalStatus.CLAIMED
        _until(lambda: _status(state, intent).artifact is not None)

        def stopped() -> bool:
            try:
                state.agent_runtime.submit(AgentEventType.SLEEP, AgentEventSource.API_SLEEP_RUN, lambda: None)
            except AgentRuntimeStopped:
                return True
            except Exception:
                return False
            return False

        _until(stopped)  # the pre-WAL failure was a runtime fail-stop
    with TestClient(_live(settings)) as client:
        state = client.app.state
        _open(state)
        outcome = _until(lambda: (_outcomes(state) or [None])[0])
        assert outcome.outcome_state is VerifiedState.ACTIVE_VERIFIED_AUTHORIZED
        rows = [
            row
            for row in state.subject_history_store.operations(ip16a2.ip10._incarnation(state))
            if row.operation_kind is K.R18_OUTCOME_COMMIT
        ]
        assert [row.phase for row in rows].count(HistoryPhase.FINALIZED) == 1
        assert all(row.phase in (HistoryPhase.ABORTED, HistoryPhase.FINALIZED) for row in rows)
        assert len(_outcomes(state)) == 1
        assert state.event_journal.inspect().open_transactions == ()


# ============================================================ F09 support


def test_a_target_changed_before_the_verification_is_an_unauthorized_effect(settings) -> None:  # type: ignore[no-untyped-def]
    with TestClient(_live(settings)) as client:
        state = client.app.state
        _open(state)
        _fake_timers(state)
        _Clock(state)

        def refuse(claim_id: str):  # type: ignore[no-untyped-def]
            from suzka.runtime.agent_runtime import AgentRuntimeAdmissionBlocked

            raise AgentRuntimeAdmissionBlocked(None)  # type: ignore[arg-type]

        real = state.outcome_owner._submit_outcome
        state.outcome_owner._submit_outcome = refuse  # type: ignore[method-assign]
        _gid, intent, _forecast = _allowed(state)
        _challenge, done = ip17c._confirm(state, intent.intent_id)
        _until(lambda: _status(state, intent).artifact is not None)
        # An outside change, before the first verification.
        connection = sqlite3.connect(_target(state), isolation_level=None)
        try:
            triggers = connection.execute(
                "SELECT name, sql FROM sqlite_schema WHERE type = 'trigger' AND name LIKE 'target_artifacts_%'"
            ).fetchall()
            for name, _sql in triggers:
                connection.execute(f"DROP TRIGGER {name}")
            connection.execute("UPDATE target_artifacts SET payload = ?", (b"{}",))
            for _name, sql in triggers:
                connection.execute(sql)
        finally:
            connection.close()
        state.outcome_owner._submit_outcome = real  # type: ignore[method-assign]
        result = real(done.claim.claim_id).result(timeout=30).value
        assert result.status is OutcomeStatus.COMMITTED
        outcome = result.outcome
        assert outcome.outcome_state is VerifiedState.UNAUTHORIZED_EFFECT and outcome.foreign_change
        view = outcome.to_r17_view()
        assert view.foreign_change is True


# ============================================================ F08 / holds


def test_the_outcome_holds_for_an_unknown_claim_and_writes_nothing(settings) -> None:  # type: ignore[no-untyped-def]
    with TestClient(_live(settings)) as client:
        state = client.app.state
        _open(state)
        _fake_timers(state)
        _Clock(state)
        unknown = state.outcome_owner._submit_outcome("claim-" + "0" * 64).result(timeout=30).value
        assert (unknown.status, unknown.reason) == (OutcomeStatus.HOLD, R18OutcomeReason.CLAIM_UNKNOWN)
        assert _outcomes(state) == []


# ============================================================ soft band


def test_the_soft_band_admits_the_outcome_and_still_refuses_new_forward_work() -> None:
    from suzka.runtime.event_journal import (
        SOFT_BAND_ADMITTED_EVENT_TYPES,
        EventJournalAdmissionBand,
        EventJournalAdmissionStatus,
    )

    assert AgentEventType.R18_OUTCOME_COMMIT in SOFT_BAND_ADMITTED_EVENT_TYPES
    soft = EventJournalAdmissionStatus(
        True, "journal_soft_limit", 0, 1, 0, 1, True, False, False,
        band=EventJournalAdmissionBand.SOFT,
    )
    assert soft.admits(AgentEventType.R18_OUTCOME_COMMIT)
    assert soft.admits(AgentEventType.R18_APPROVAL_EXPIRE)
    for refused in (
        AgentEventType.R18_EXECUTION_CLAIM,
        AgentEventType.R18_ACTION_INTENT_ISSUE,
        AgentEventType.R18_POLICY_COMMIT,
        AgentEventType.R18_CHALLENGE_ISSUE,
        AgentEventType.R17_EXPECTATION_COMMIT,
    ):
        assert not soft.admits(refused)


# ============================================================ r18.enabled=false


def test_r18_off_has_no_executor_no_thread_no_event_and_a_disabled_gate(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _settings(tmp_path / "off", r18=False)
    target = Path(settings.r18.target_path)
    before = (_hash(target), target.stat().st_mtime_ns)
    with TestClient(_app(settings, executor=True)) as client:  # the seam does not matter when off
        state = client.app.state
        assert state.r18_executor is None
        assert state.r18_gate.state is GateState.DISABLED and not state.r18_gate.is_open()
        assert not [t for t in threading.enumerate() if t.name.startswith("suzka-r18")]
        for owner in (state.intent_issuer, state.policy_owner, state.approval_owner):
            assert owner._gate is state.r18_gate
        held = state.outcome_owner._submit_outcome("claim-" + "0" * 64).result(timeout=30).value
        assert (held.status, held.reason) == (OutcomeStatus.HOLD, R18OutcomeReason.R18_DISABLED)
        r18_events = [
            row for row in state.event_journal.records if row.event_type is not None and row.event_type.value.startswith(("r18_", "r17_"))
        ]
        # only the disabled HOLD of the line above: it writes nothing.
        assert all(row.event_type is AgentEventType.R18_OUTCOME_COMMIT for row in r18_events)
        assert state.r18_gate.blocked_target() is False
        assert state.decision_issuer._blocked_target() is False
    assert (_hash(target), target.stat().st_mtime_ns) == before  # never opened for writing


def test_r18_on_rotates_the_target_to_this_process_before_the_gate_opens(settings) -> None:  # type: ignore[no-untyped-def]
    with TestClient(_live(settings)) as client:
        state = client.app.state
        _open(state)
        store = TargetStore.open(_target(state))
        try:
            assert store.meta().runtime_instance_id == state.approval_owner.runtime_instance_id
        finally:
            store.close()
        assert state.r18_executor.gate is state.r18_gate
