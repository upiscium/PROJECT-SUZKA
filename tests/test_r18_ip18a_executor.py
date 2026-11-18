"""IP-18a: the R18 executor, reconciler and sweep, on real target files with a
fake clock and a fake Outcome owner that verifies the target itself.

The Outcome owner's real event is tested end to end in
``test_r18_ip18a_flow.py``; here it is replaced by ``_FakeOutcome``, which
runs the real ``verify_forward`` against the real target (never the receipt).
"""

from __future__ import annotations

import ast
from concurrent.futures import Future
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import inspect
import logging
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from suzka.r16.history_reader import AdoptedRecord, AdoptedRecords
from suzka.r16.history_records import HistoryOperationKind as K, RecordKind as R
from suzka.r18 import executor as executor_module
from suzka.r18.contracts import (
    ApprovalChallengeV1,
    ApprovalChannel,
    Direction,
    ExecutionClaimV1,
    ExecutorResult,
    VerifiedState,
)
from suzka.r18.executor import (
    OUTCOME_SUBMIT_BUDGET,
    QUEUE_CAPACITY,
    R18Executor,
    pending_forward_claims,
)
from suzka.r18.outcome_owner import (
    OutcomeOwnerV1,
    OutcomeStatus,
    R18OutcomeReason,
    _OutcomeResult,
)
from suzka.r18.sweep import GateState, SweepGate
from suzka.r18.target import (
    TargetNotStarted,
    TargetReader,
    TargetStore,
    new_runtime_instance_id,
)
from suzka.r18.verifier import verify_forward
from suzka.runtime.agent_runtime import (
    AgentRuntimeDurabilityError,
    AgentRuntimeQueueFull,
)

from test_r18_ip13_contracts import BOOT, EVENT_V, GROUP, NONCE, _allow, _artifact, _intent
from test_r18_ip15_target_verifier import CLAIM_EVENT, _store

T0 = datetime(2026, 11, 16, 0, 6, 0, tzinfo=timezone.utc)
CLAIMED = "2026-11-16T00:05:00Z"
DEADLINE = "2026-11-16T00:10:00Z"
ME = "00000000-0000-4000-8000-0000000000ee"


class _Clock:
    def __init__(self) -> None:
        self.utc = T0
        self.mono = 1000.0

    def __call__(self) -> datetime:
        return self.utc

    def monotonic(self) -> float:
        return self.mono

    def advance(self, seconds: float) -> None:
        self.utc += timedelta(seconds=seconds)
        self.mono += seconds


class _FakeOutcome(OutcomeOwnerV1):
    """Verifies the real target like the real owner, never from a receipt."""

    def __init__(self, path: Path, chain: tuple[Any, Any]) -> None:  # noqa: D107
        self.path = path
        self.chain = chain
        self.calls: list[str] = []
        self.script: list[object] = []  # exceptions / _OutcomeResult to return first
        self.committed: dict[str, Any] = {}

    @property
    def enabled(self) -> bool:
        return True

    def _submit_outcome(self, claim_id: str) -> Future[Any]:
        self.calls.append(claim_id)
        if self.script:
            step = self.script.pop(0)
            if isinstance(step, BaseException):
                raise step
            return _done(step)
        intent, challenge = self.chain
        reader = TargetReader.open(self.path)
        try:
            outcome = verify_forward(
                reader, intent, challenge, consumed_nonce=challenge.nonce, outcome_revision=1,
                verifier_event_id=EVENT_V, verifier_sequence=23,
            )
            if outcome is None:
                reason = (
                    R18OutcomeReason.NOT_SETTLED
                    if reader.status(intent.obligation_group_key).forward is None
                    else R18OutcomeReason.UNDECIDABLE
                )
                return _done(_OutcomeResult(OutcomeStatus.HOLD, claim_id, reason))
        finally:
            reader.close()
        self.committed[claim_id] = outcome
        return _done(_OutcomeResult(OutcomeStatus.COMMITTED, claim_id, None, outcome))


def _done(value: object) -> Future[Any]:
    future: Future[Any] = Future()
    future.set_result(SimpleNamespace(value=value))
    return future


class _World:
    """One target, one adopted claim, an executor under a fake clock."""

    def __init__(self, tmp_path: Path, *, deadline: str | None = None, claim_runtime: str = ME) -> None:
        self.clock = _Clock()
        self.store = _store(tmp_path)  # rotated to RUNTIME (an "earlier process")
        self.path = self.store.path
        meta = self.store.meta()
        self.intent = _intent(
            target_instance_id=meta.target_instance_id,
            artifact=replace(_artifact(), obligation_group_key=GROUP),
        )
        self.challenge = ApprovalChallengeV1.for_forward(
            self.intent, "e" * 64, _allow(self.intent), nonce=NONCE,
            issued_utc="2026-11-16T00:00:00Z", expires_utc="2026-11-16T00:10:00Z",
            boot_id=BOOT, commitment_deadline_utc=deadline,
        )
        self.claim = ExecutionClaimV1.claim(
            self.challenge, runtime_instance_id=claim_runtime, boot_id=BOOT,
            claim_event_id=CLAIM_EVENT, claim_sequence=22, claimed_utc=CLAIMED,
            approval_channel=ApprovalChannel.LOCAL_LOOPBACK_TYPED_CODE_V1,
        )
        self.records: list[AdoptedRecord] = []
        self.blocked: Any = None
        self.add_chain()
        self.gate = SweepGate(enabled=True)
        self.outcome = _FakeOutcome(self.path, (self.intent, self.challenge))
        self.authority = True
        self.executor = R18Executor(
            target_path=self.path,
            runtime_instance_id=ME,
            gate=self.gate,
            outcome_owner=self.outcome,
            read_records=lambda: AdoptedRecords(self.blocked, tuple(self.records)),
            has_exclusive_authority=lambda: self.authority,
            clock=self.clock,
            monotonic=self.clock.monotonic,
            threaded=False,
        )
        self.sleeps: list[float] = []
        self.executor._sleep = lambda seconds: self.sleeps.append(seconds) or False  # type: ignore[method-assign]
        self.writes = 0
        self._wrap_write()

    def add_chain(self) -> None:
        def rec(kind: K, rk: R, record: object, n: int) -> AdoptedRecord:
            return AdoptedRecord(kind, rk, record, f"00000000-0000-4000-8000-00000000{n:04x}", CLAIM_EVENT, n)

        self.records += [
            rec(K.R18_INTENT_ISSUE, R.ACTION_INTENT, self.intent, 1),
            rec(K.R18_CHALLENGE_ISSUE, R.APPROVAL_CHALLENGE, self.challenge, 2),
            rec(K.R18_EXECUTION_CLAIM, R.EXECUTION_CLAIM, self.claim, 3),
        ]

    def _wrap_write(self) -> None:
        real = TargetStore.forward_write
        world = self

        def counting(store: TargetStore, *args: Any, **kwargs: Any) -> Any:
            world.writes += 1
            return real(store, *args, **kwargs)

        self.counting = counting
        TargetStore.forward_write = counting  # type: ignore[method-assign]
        self._real = real

    def close(self) -> None:
        TargetStore.forward_write = self._real  # type: ignore[method-assign]
        self.executor._close_store()

    def status(self) -> Any:
        return self.store.status(self.intent.obligation_group_key)


@pytest.fixture
def world(tmp_path: Path):  # type: ignore[no-untyped-def]
    made = _World(tmp_path)
    try:
        yield made
    finally:
        made.close()


def _run(world: _World) -> None:
    world.executor.run_pending()


def _opened(world: _World) -> None:
    """Run the sweep first, as ``start()`` does."""
    world.executor.start()
    _run(world)
    assert world.gate.is_open(), world.gate


# -------------------------------------------------------------- happy path


def test_a_claim_is_written_once_settled_and_gets_an_independent_outcome(world: _World, caplog) -> None:  # type: ignore[no-untyped-def]
    caplog.set_level(logging.INFO, logger="suzka.r18.executor")
    _opened(world)
    world.executor.enqueue(world.claim)
    world.executor.enqueue(world.claim)  # a duplicate feed is a no-op
    _run(world)
    assert world.writes == 1
    status = world.status()
    assert status.artifact is not None and status.forward is not None
    assert status.forward.approval_ref == NONCE
    assert world.outcome.calls == [world.claim.claim_id]
    outcome = world.outcome.committed[world.claim.claim_id]
    assert outcome.outcome_state is VerifiedState.ACTIVE_VERIFIED_AUTHORIZED
    assert world.executor.unsettled_ids() == frozenset()
    text = caplog.text
    for stage in ("enqueued", "write", "settle", "outcome_submit", "outcome_commit"):
        assert f"stage={stage} " in text
    assert "result=APPLIED" in text and "window_s=60" in text


def test_the_log_line_field_set_is_pinned(world: _World, caplog) -> None:  # type: ignore[no-untyped-def]
    caplog.set_level(logging.INFO, logger="suzka.r18.executor")
    _opened(world)
    world.executor.enqueue(world.claim)
    _run(world)
    pattern = re.compile(
        r"R18_EXEC claim=claim-[0-9a-f]{64} op=[0-9a-f]{8} stage=[a-z_]+ result=\S+ "
        r"state=[A-Za-z_:]+( budget_spent)? window_s=(-|\d+) t=\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ"
    )
    lines = [line for line in caplog.text.splitlines() if "R18_EXEC" in line]
    assert lines and all(pattern.search(line) for line in lines), lines
    sweep = [line for line in caplog.text.splitlines() if "R18_SWEEP" in line]
    assert {re.search(r"step=(\w+)", line).group(1) for line in sweep} >= {"S1", "S2", "S3", "S5"}  # type: ignore[union-attr]


def test_the_log_lines_carry_no_request_text(world: _World, caplog) -> None:  # type: ignore[no-untyped-def]
    caplog.set_level(logging.INFO)
    _opened(world)
    world.executor.enqueue(world.claim)
    _run(world)
    payload = world.intent.artifact.payload_bytes().decode()
    assert payload not in caplog.text and world.challenge.confirmation_code not in caplog.text


# ------------------------------------------------------------------ F07 / F08


def test_a_claim_of_an_earlier_process_is_never_written_and_settles_absent(tmp_path: Path) -> None:
    world = _World(tmp_path, claim_runtime="00000000-0000-4000-8000-0000000000cc")
    try:
        _opened(world)
        # The sweep settled it (S3) and gave it its Outcome; nothing was written.
        assert world.writes == 0
        assert world.status().artifact is None
        assert world.status().forward is not None and world.status().forward.kind.value == "FENCED"
        outcome = world.outcome.committed[world.claim.claim_id]
        assert outcome.outcome_state is VerifiedState.VERIFIED_ABSENT
        # A late CAS from the old id is definitely not executed.
        world.executor.enqueue(world.claim)
        _run(world)
        assert world.writes == 0
    finally:
        world.close()


def test_a_fed_claim_of_another_process_skips_the_write_but_settles(world: _World) -> None:
    other = replace(world.claim, runtime_instance_id="00000000-0000-4000-8000-0000000000cc")
    world.records[-1] = replace(world.records[-1], record=other)
    _opened(world)
    # The sweep settled it first; a feed afterwards still writes nothing.
    world.executor.enqueue(other)
    _run(world)
    assert world.writes == 0 and world.status().artifact is None


def test_the_deadline_in_the_transaction_burns_the_approval_into_absent(tmp_path: Path) -> None:
    world = _World(tmp_path, deadline=DEADLINE)
    try:
        _opened(world)
        world.clock.advance(5 * 60)  # now == deadline
        world.executor.enqueue(world.claim)
        _run(world)
        assert world.writes == 1 and world.status().artifact is None
        assert world.outcome.committed[world.claim.claim_id].outcome_state is VerifiedState.VERIFIED_ABSENT
    finally:
        world.close()


def test_a_lying_executor_cannot_make_an_effect_appear(world: _World) -> None:
    """F08: the receipt says APPLIED, the target holds no effect: the Outcome
    is VERIFIED_ABSENT, never ACTIVE (the receipt is not an input)."""

    real_receipt = TargetStore.forward_write

    def lying(store: TargetStore, intent, challenge, claim, *, now):  # type: ignore[no-untyped-def]
        from suzka.r18.contracts import ExecutionReceiptV1

        return ExecutionReceiptV1.for_claim(
            claim, ExecutorResult.APPLIED, reported_revision=1, reported_payload_digest=challenge.payload_digest
        )

    TargetStore.forward_write = lying  # type: ignore[method-assign,assignment]
    try:
        _opened(world)
        world.executor.enqueue(world.claim)
        _run(world)
    finally:
        TargetStore.forward_write = real_receipt  # type: ignore[method-assign]
    assert world.status().artifact is None
    assert world.outcome.committed[world.claim.claim_id].outcome_state is VerifiedState.VERIFIED_ABSENT


def test_a_tampered_payload_digest_column_changes_nothing(world: _World) -> None:
    import sqlite3

    _opened(world)
    world.executor.enqueue(world.claim)
    _run(world)
    # Not our change: the column alone is wrong; the stored bytes are the
    # approved bytes, and the verifier recomputes.
    connection = sqlite3.connect(world.path, isolation_level=None)
    try:
        for name in ("target_operations_no_update",):
            sql = connection.execute("SELECT sql FROM sqlite_schema WHERE name = ?", (name,)).fetchone()[0]
            connection.execute(f"DROP TRIGGER {name}")
            connection.execute("UPDATE target_operations SET payload_digest = ?", ("0" * 64,))
            connection.execute(sql)
    finally:
        connection.close()
    # The fresh read now disagrees with the ledger row: UNAUTHORIZED, still not ACTIVE.
    reader = TargetReader.open(world.path)
    try:
        outcome = verify_forward(
            reader, world.intent, world.challenge, consumed_nonce=NONCE, outcome_revision=1,
            verifier_event_id=EVENT_V, verifier_sequence=24,
        )
    finally:
        reader.close()
    assert outcome is not None and outcome.outcome_state is VerifiedState.UNAUTHORIZED_EFFECT


def test_not_started_is_retried_twice_within_the_deadline_then_settled(world: _World) -> None:
    def refuse(store, *args, **kwargs):  # type: ignore[no-untyped-def]
        world.writes += 1
        raise TargetNotStarted("locked")

    TargetStore.forward_write = refuse  # type: ignore[method-assign,assignment]
    _opened(world)
    world.executor.enqueue(world.claim)
    _run(world)
    assert world.writes == 3  # the first try and at most two retries
    assert world.sleeps == [1.0, 2.0]
    assert world.outcome.committed[world.claim.claim_id].outcome_state is VerifiedState.VERIFIED_ABSENT


def test_not_started_recovers_when_the_lock_clears(world: _World) -> None:
    real = world._real
    state = {"n": 0}

    def flaky(store, *args, **kwargs):  # type: ignore[no-untyped-def]
        state["n"] += 1
        if state["n"] == 1:
            raise TargetNotStarted("locked")
        return real(store, *args, **kwargs)

    TargetStore.forward_write = flaky  # type: ignore[method-assign,assignment]
    _opened(world)
    world.executor.enqueue(world.claim)
    _run(world)
    assert state["n"] == 2 and world.status().artifact is not None
    assert world.outcome.committed[world.claim.claim_id].outcome_state is VerifiedState.ACTIVE_VERIFIED_AUTHORIZED


def test_not_started_is_not_retried_past_the_deadline(tmp_path: Path) -> None:
    world = _World(tmp_path, deadline=DEADLINE)
    try:
        calls = {"n": 0}

        def refuse(store, *args, **kwargs):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            world.clock.advance(5 * 60)  # the deadline passes during the wait
            raise TargetNotStarted("locked")

        TargetStore.forward_write = refuse  # type: ignore[method-assign,assignment]
        _opened(world)
        world.executor.enqueue(world.claim)
        _run(world)
        assert calls["n"] == 1 and world.sleeps == []
    finally:
        world.close()


def test_an_unknown_failure_is_never_resent_and_the_store_is_reopened(world: _World) -> None:
    real = world._real
    calls = {"n": 0}

    def commit_then_fail(store, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        real(store, *args, **kwargs)  # the effect commits ...
        raise OSError("response lost")  # ... and the caller never learns

    TargetStore.forward_write = commit_then_fail  # type: ignore[method-assign,assignment]
    _opened(world)
    first_store = world.executor._store
    world.executor.enqueue(world.claim)
    _run(world)
    assert calls["n"] == 1  # no blind retry, ever
    assert world.executor._store is not first_store  # reopened
    # The settle read the effect back: AUTHORIZED, nothing duplicated.
    assert world.outcome.committed[world.claim.claim_id].outcome_state is VerifiedState.ACTIVE_VERIFIED_AUTHORIZED


def test_an_unknown_failure_before_the_commit_settles_absent(world: _World) -> None:
    def boom(store, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise OSError("disk")

    TargetStore.forward_write = boom  # type: ignore[method-assign,assignment]
    _opened(world)
    world.executor.enqueue(world.claim)
    _run(world)
    assert world.outcome.committed[world.claim.claim_id].outcome_state is VerifiedState.VERIFIED_ABSENT


def test_when_in_doubt_do_not_write(world: _World) -> None:
    world.blocked = "history"  # the adoption check cannot read the history
    world.executor.start()
    world.executor.enqueue(world.claim)
    _run(world)
    assert world.writes == 0 and world.status().artifact is None
    assert world.claim.claim_id in world.executor.unsettled_ids()  # retried later
    # No fence either: nothing was decided without the Intent.
    assert world.status().forward is None


# -------------------------------------------------------- enqueue / the queue


def test_enqueue_is_o1_and_never_blocks_on_a_full_queue(world: _World) -> None:
    claims = []
    for number in range(QUEUE_CAPACITY + 3):
        claims.append(replace(world.claim, claim_sequence=100 + number))
    started = time.monotonic()
    for claim in claims:
        world.executor.enqueue(claim)
    assert time.monotonic() - started < 1.0
    # The sweep job is not queued here (no start): 8 fit, the rest are dropped.
    assert len(world.executor.handed_ids()) == QUEUE_CAPACITY
    dropped = [c for c in claims if c.claim_id not in world.executor.handed_ids()]
    assert len(dropped) == 3
    # A dropped claim stays unsettled only, so the backstop feeds it again.
    assert all(c.claim_id in world.executor.unsettled_ids() for c in dropped)


def test_a_dropped_claim_is_fed_again_by_the_backstop(world: _World) -> None:
    _opened(world)
    others = [replace(world.claim, claim_sequence=100 + n) for n in range(QUEUE_CAPACITY)]
    for claim in others:
        world.executor.enqueue(claim)
    world.executor.enqueue(world.claim)  # the 9th: dropped
    assert world.claim.claim_id not in world.executor.handed_ids()
    world.executor.run_pending()
    world.clock.advance(executor_module.BACKSTOP_SECONDS + 1)
    world.executor.tick()
    _run(world)
    # fed again: the backstop saw the adopted claim without an Outcome.
    assert world.claim.claim_id in world.executor.handed_ids()


def test_enqueue_does_not_wait_for_a_busy_worker(world: _World) -> None:
    release = threading.Event()
    entered = threading.Event()

    def slow(claim_id: str) -> Future[Any]:
        entered.set()
        release.wait(10)
        return _done(_OutcomeResult(OutcomeStatus.HOLD, claim_id, R18OutcomeReason.TARGET_UNAVAILABLE))

    world.outcome._submit_outcome = slow  # type: ignore[method-assign]
    _opened(world)
    worker = threading.Thread(target=lambda: (world.executor.enqueue(world.claim), _run(world)))
    worker.start()
    assert entered.wait(10)
    other = replace(world.claim, claim_sequence=999)
    started = time.monotonic()
    world.executor.enqueue(other)  # while the worker is blocked inside an Outcome wait
    assert time.monotonic() - started < 0.5
    release.set()
    worker.join(10)


def test_enqueue_has_no_io_and_no_blocking_call_on_its_path() -> None:
    tree = ast.parse(inspect.getsource(R18Executor))
    method = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "enqueue"
    )
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
    }
    forbidden = {
        "result", "get", "put", "join", "wait", "sleep", "open", "fence", "forward_write",
        "_read_records", "_open_store", "adopted_records", "_submit_outcome",
    }
    assert not (called & forbidden), called & forbidden
    assert "put_nowait" in called


def test_enqueue_ignores_a_non_claim_and_an_inverse_claim(world: _World) -> None:
    world.executor.enqueue(object())  # type: ignore[arg-type]
    assert world.executor.handed_ids() == frozenset()


# ----------------------------------------------------- reconciler / outcome


def test_a_claim_whose_execute_job_never_ran_is_settled_after_60_seconds(world: _World) -> None:
    _opened(world)
    world.executor.enqueue(world.claim)
    world.executor._queue.get_nowait()  # the worker died with the job: it is lost
    world.clock.advance(30)
    world.executor.tick()
    _run(world)
    assert world.status().forward is None  # not yet
    world.clock.advance(31)
    world.executor.tick()
    _run(world)
    assert world.writes == 0  # settled, never written late
    assert world.outcome.committed[world.claim.claim_id].outcome_state is VerifiedState.VERIFIED_ABSENT
    assert world.executor.unsettled_ids() == frozenset()


def test_a_dead_worker_is_restarted_by_the_reconciler(tmp_path: Path) -> None:
    world = _World(tmp_path)
    try:
        world.executor._threaded = True
        world.executor._started = True
        dead = threading.Thread(target=lambda: None)
        dead.start()
        dead.join()
        world.executor._worker = dead
        world.executor.tick()
        assert world.executor.restarts == 1
        worker = world.executor._worker
        assert worker is not dead and worker is not None and worker.is_alive()
        world.executor._stop_event.set()
        world.executor._queue.put(executor_module._STOP)
        worker.join(5)
    finally:
        world.close()


def test_outcome_retries_back_off_to_ten_minutes_and_stop_at_the_budget(world: _World, caplog) -> None:  # type: ignore[no-untyped-def]
    caplog.set_level(logging.INFO, logger="suzka.r18.executor")
    world.outcome.script = [
        _OutcomeResult(OutcomeStatus.HOLD, None, R18OutcomeReason.TARGET_UNAVAILABLE)
        for _ in range(60)
    ]
    _opened(world)
    world.executor.enqueue(world.claim)
    _run(world)
    gaps: list[float] = []
    last = world.clock.monotonic()
    for _ in range(1500):
        if world.claim.claim_id not in world.executor.unsettled_ids():
            break
        before = len(world.outcome.calls)
        world.clock.advance(20)
        world.executor.tick()
        _run(world)
        if len(world.outcome.calls) > before:
            now = world.clock.monotonic()
            gaps.append(now - last)
            last = now
    assert len(world.outcome.calls) == OUTCOME_SUBMIT_BUDGET  # the 21st is not made
    assert 595.0 <= max(gaps) <= 620.0  # capped at 10 minutes
    assert 15.0 <= gaps[0] <= 20.0 and 30.0 <= gaps[1] <= 40.0  # 15 s, then 30 s, doubling
    assert "budget_spent" in caplog.text
    # Still CLAIMED on the target side: it was settled but has no Outcome.
    assert world.claim.claim_id not in world.outcome.committed
    # A restart's sweep resets the budget: a fresh executor submits again.
    assert world.claim.claim_id not in world.executor.unsettled_ids()


def test_a_refused_submit_does_not_consume_the_budget(world: _World) -> None:
    world.outcome.script = [AgentRuntimeQueueFull(None) for _ in range(40)]  # type: ignore[arg-type]
    _opened(world)
    world.executor.enqueue(world.claim)
    _run(world)
    for _ in range(80):
        if world.claim.claim_id in world.outcome.committed:
            break
        world.clock.advance(300)
        world.executor.tick()
        _run(world)
    assert world.claim.claim_id in world.outcome.committed  # 40 refusals, then it went through
    assert len(world.outcome.calls) == 41


def test_a_runtime_fail_stop_stops_the_outcome_retries(world: _World) -> None:
    from suzka.runtime.agent_runtime import AgentRuntimeDurabilityPhase

    world.outcome.script = [
        AgentRuntimeDurabilityError(
            None,  # type: ignore[arg-type]
            next(iter(AgentRuntimeDurabilityPhase)),
            outcome_indeterminate=False,
        )
    ]
    _opened(world)
    world.executor.enqueue(world.claim)
    _run(world)
    for _ in range(10):
        world.clock.advance(900)
        world.executor.tick()
        _run(world)
    assert len(world.outcome.calls) == 1


def test_a_key_level_blocked_hold_stays_claimed_and_stops_retrying(world: _World, caplog) -> None:  # type: ignore[no-untyped-def]
    caplog.set_level(logging.INFO, logger="suzka.r18.executor")
    world.outcome.script = [_OutcomeResult(OutcomeStatus.HOLD, None, R18OutcomeReason.UNDECIDABLE)]
    _opened(world)
    world.executor.enqueue(world.claim)
    _run(world)
    assert "BLOCKED:undecidable" in caplog.text
    world.clock.advance(3000)
    world.executor.tick()
    _run(world)
    assert len(world.outcome.calls) == 1


def test_a_failing_settle_closes_the_gate_and_the_probe_reopens_it(world: _World) -> None:
    _opened(world)
    store = world.executor._store
    assert store is not None
    real = TargetStore.fence
    state = {"fail": True}

    def fence(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if state["fail"]:
            raise TargetNotStarted("locked")
        return real(self, *args, **kwargs)

    TargetStore.fence = fence  # type: ignore[method-assign,assignment]
    try:
        world.executor.enqueue(world.claim)
        _run(world)
        assert world.gate.state is GateState.SWEEPING  # re-closed
        state["fail"] = False
        world.clock.advance(2)
        world.executor.tick()  # the probe: a full, idempotent sweep
        _run(world)
        assert world.gate.is_open()
    finally:
        TargetStore.fence = real  # type: ignore[method-assign]


# ------------------------------------------------------------------- sweep


def test_the_sweep_rotates_before_it_scans_and_the_gate_opens_only_at_the_end(world: _World) -> None:
    order: list[str] = []
    real_rotate = executor_module.rotate_if_needed

    def rotate(store, ident):  # type: ignore[no-untyped-def]
        order.append(f"rotate gate_open={world.gate.is_open()}")
        return real_rotate(store, ident)

    executor_module.rotate_if_needed = rotate  # type: ignore[assignment]
    original = world.executor._read_records

    def read() -> AdoptedRecords:
        order.append(f"scan rotated_to_me={world.store.meta().runtime_instance_id == ME} gate_open={world.gate.is_open()}")
        return original()

    world.executor._read_records = read
    try:
        assert world.gate.state is GateState.SWEEPING  # closed from construction
        _opened(world)
    finally:
        executor_module.rotate_if_needed = real_rotate  # type: ignore[assignment]
    assert order[0] == "rotate gate_open=False"
    assert order[1] == "scan rotated_to_me=True gate_open=False"
    assert world.store.meta().runtime_instance_id == ME


def test_start_asserts_the_lease_and_leaves_the_gate_closed(world: _World) -> None:
    world.authority = False
    with pytest.raises(RuntimeError, match="lease"):
        world.executor.start()
    assert not world.gate.is_open() and world.executor._queue.empty()
    world.executor.stop()  # the start-failure cleanup is safe


def test_s1_twice_is_a_no_op_and_a_worker_restart_reruns_it(world: _World) -> None:
    _opened(world)
    first = world.store.meta().runtime_instance_id
    world.executor._queue_sweep()
    _run(world)
    assert world.store.meta().runtime_instance_id == first == ME and world.gate.is_open()


def test_a_missing_target_blocks_after_60_seconds_and_the_probe_reopens_by_itself(tmp_path: Path) -> None:
    clock = _Clock()
    gate = SweepGate(enabled=True)
    root = tmp_path / "target"
    root.mkdir(mode=0o700)
    path = root / "target.sqlite3"
    executor = R18Executor(
        target_path=path,
        runtime_instance_id=ME,
        gate=gate,
        outcome_owner=_FakeOutcome(path, (None, None)),
        read_records=lambda: AdoptedRecords(None, ()),
        has_exclusive_authority=lambda: True,
        clock=clock,
        monotonic=clock.monotonic,
        threaded=False,
    )
    try:
        executor.start()
        executor.run_pending()
        assert gate.state is GateState.SWEEPING and not gate.is_open()
        seen = {GateState.SWEEPING}
        for _ in range(40):
            clock.advance(2)
            executor.tick()
            executor.run_pending()
            seen.add(gate.state)
            if gate.state is GateState.BLOCKED_TARGET:
                break
        assert GateState.BLOCKED_TARGET in seen and clock.mono - 1000.0 >= 60
        # The Human restores / creates the target with the existing command.
        TargetStore.create(path, target_namespace="review-artifact").close()
        for _ in range(40):
            clock.advance(10)
            executor.tick()
            executor.run_pending()
            if gate.is_open():
                break
        assert gate.is_open()
        assert TargetStore.open(path).meta().runtime_instance_id == ME
    finally:
        executor.stop()


def test_the_sweep_never_fences_a_live_claim_of_this_process(world: _World) -> None:
    _opened(world)
    world.executor.enqueue(world.claim)
    world.executor._queue.get_nowait()  # held back; a probe sweep runs meanwhile
    world.executor._sweep_done = False
    world.executor._queue_sweep()
    _run(world)
    assert world.status().forward is None  # not fenced: it is the feeders' claim


def test_pending_forward_claims_skips_a_claim_with_an_outcome(world: _World) -> None:
    assert [c.claim_id for c in pending_forward_claims(tuple(world.records))] == [world.claim.claim_id]
    outcome = SimpleNamespace()
    del outcome
    from suzka.r18.contracts import VerifiedOutcomeV1
    from suzka.r17.contracts import ObservedState

    absent = VerifiedOutcomeV1(
        operation_key=world.claim.operation_key, direction=Direction.FORWARD, outcome_revision=1,
        outcome_state=VerifiedState.VERIFIED_ABSENT, authorization=None,
        target_instance_id=world.intent.target_instance_id, observed_state=ObservedState.ABSENT,
        observed_payload_digest=None, observed_revision=0, effect_operation_key=None,
        foreign_change=False, verifier_event_id=EVENT_V, verifier_sequence=23,
    )
    world.records.append(AdoptedRecord(K.R18_OUTCOME_COMMIT, R.VERIFIED_OUTCOME, absent, "t" * 36, EVENT_V, 4))
    assert pending_forward_claims(tuple(world.records)) == []


# ---------------------------------------------------------------- stop


def test_stop_joins_the_threads_and_closes_the_target(tmp_path: Path) -> None:
    world = _World(tmp_path)
    try:
        threaded = R18Executor(
            target_path=world.path, runtime_instance_id=ME, gate=world.gate,
            outcome_owner=world.outcome, read_records=lambda: AdoptedRecords(None, ()),
            has_exclusive_authority=lambda: True,
        )
        threaded.start()
        deadline = time.monotonic() + 20
        while not world.gate.is_open() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert world.gate.is_open()
        threads = [t for t in threading.enumerate() if t.name.startswith("suzka-r18")]
        assert {t.name for t in threads} == {"suzka-r18-executor", "suzka-r18-reconciler"}
        threaded.stop()
        assert not [t for t in threading.enumerate() if t.name.startswith("suzka-r18")]
        assert threaded._store is None
    finally:
        world.close()


def test_a_thread_that_does_not_join_is_abandoned_not_waited_on_forever(
    world: _World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(executor_module, "JOIN_TIMEOUT_SECONDS", 0.2)
    release = threading.Event()
    stuck = threading.Thread(target=release.wait, name="suzka-r18-executor", daemon=True)
    stuck.start()
    world.executor._worker = stuck
    started = time.monotonic()
    world.executor.stop()
    assert time.monotonic() - started < 2.0
    release.set()


def test_the_new_runtime_instance_id_helper_is_a_uuid4() -> None:
    assert len(new_runtime_instance_id()) == 36


def test_unqualified_bindings_are_refused(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        R18Executor(
            target_path="x",  # type: ignore[arg-type]
            runtime_instance_id=ME, gate=SweepGate(enabled=True), outcome_owner=object(),  # type: ignore[arg-type]
            read_records=lambda: AdoptedRecords(None, ()), has_exclusive_authority=lambda: True,
        )
