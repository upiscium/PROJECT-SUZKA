"""IP-18a: the forward-write deadline check (HP-7 (a)), ``TargetNotStarted``,
and the real ``SweepGate`` / S1 rotation, on real target files.

Every intent, challenge and claim here is a FIXTURE.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import sqlite3

import pytest

from suzka.r18 import target as target_module
from suzka.r18.admission import AdmissionGate, gate_open
from suzka.r18.contracts import (
    ApprovalChallengeV1,
    ApprovalChannel,
    Direction,
    ExecutionClaimV1,
    ExecutorResult,
    VerifiedState,
)
from suzka.r18.sweep import GateState, SweepGate, rotate_if_needed
from suzka.r18.target import (
    TargetBindingError,
    TargetNotStarted,
    TargetReader,
    TargetStore,
    TargetUnavailable,
)
from suzka.r18.verifier import verify_forward

from test_r18_ip13_contracts import BOOT, EVENT_V, GROUP, NONCE, _allow, _artifact, _intent
from test_r18_ip15_target_verifier import CLAIM_EVENT, RUNTIME, _store

DEADLINE = "2026-11-16T00:10:00Z"
CLAIMED = "2026-11-16T00:05:00Z"


def _at(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _forward(store: TargetStore, deadline: str | None):  # type: ignore[no-untyped-def]
    meta = store.meta()
    intent = _intent(
        target_instance_id=meta.target_instance_id,
        artifact=replace(_artifact(), obligation_group_key=GROUP),
    )
    challenge = ApprovalChallengeV1.for_forward(
        intent,
        "e" * 64,
        _allow(intent),
        nonce=NONCE,
        issued_utc="2026-11-16T00:00:00Z",
        expires_utc="2026-11-16T00:10:00Z",
        boot_id=BOOT,
        commitment_deadline_utc=deadline,
    )
    claim = ExecutionClaimV1.claim(
        challenge,
        runtime_instance_id=RUNTIME,
        boot_id=BOOT,
        claim_event_id=CLAIM_EVENT,
        claim_sequence=22,
        claimed_utc=CLAIMED,
        approval_channel=ApprovalChannel.LOCAL_LOOPBACK_TYPED_CODE_V1,
    )
    return intent, challenge, claim


def _hash(store: TargetStore) -> str:
    return sha256(Path(store.path).read_bytes()).hexdigest()


# ------------------------------------------------------- HP-7 (a) deadline


def test_a_write_before_the_deadline_is_applied(tmp_path: Path) -> None:
    store = _store(tmp_path)
    intent, challenge, claim = _forward(store, DEADLINE)
    receipt = store.forward_write(intent, challenge, claim, now=lambda: _at("2026-11-16T00:09:59Z"))
    assert receipt.executor_result is ExecutorResult.APPLIED


@pytest.mark.parametrize("moment", ["2026-11-16T00:10:00Z", "2026-11-16T00:10:01Z"])
def test_a_write_at_or_after_the_deadline_is_deadline_elapsed_and_writes_nothing(
    tmp_path: Path, moment: str
) -> None:
    store = _store(tmp_path)
    intent, challenge, claim = _forward(store, DEADLINE)
    before = _hash(store)
    receipt = store.forward_write(intent, challenge, claim, now=lambda: _at(moment))
    assert receipt.executor_result is ExecutorResult.DEADLINE_ELAPSED
    assert receipt.reported_revision is None and receipt.reported_payload_digest is None
    assert _hash(store) == before  # byte-identical: no row, no artifact
    status = store.status(intent.obligation_group_key)
    assert status.artifact is None and status.forward is None
    # The settle that follows verifies the key ABSENT, and a late CAS stays out.
    store.fence(intent.obligation_group_key, Direction.FORWARD)
    reader = TargetReader.open(store.path)
    try:
        outcome = verify_forward(
            reader, intent, challenge, consumed_nonce=NONCE, outcome_revision=1,
            verifier_event_id=EVENT_V, verifier_sequence=23,
        )
    finally:
        reader.close()
    assert outcome is not None and outcome.outcome_state is VerifiedState.VERIFIED_ABSENT


def test_the_deadline_is_read_after_begin_immediate_returned(tmp_path: Path) -> None:
    """The clock is read inside the transaction (lock held), not before it."""

    store = _store(tmp_path)
    intent, challenge, claim = _forward(store, DEADLINE)
    seen: list[bool] = []

    def now() -> datetime:
        seen.append(store._connection.in_transaction)
        return _at("2026-11-16T00:06:00Z")

    store.forward_write(intent, challenge, claim, now=now)
    assert seen == [True]


def test_no_deadline_means_no_check(tmp_path: Path) -> None:
    store = _store(tmp_path)
    intent, challenge, claim = _forward(store, None)
    calls: list[int] = []

    def now() -> datetime:
        calls.append(1)
        return _at("2030-01-01T00:00:00Z")

    receipt = store.forward_write(intent, challenge, claim, now=now)
    assert receipt.executor_result is ExecutorResult.APPLIED and calls == []


def test_a_naive_target_clock_is_refused_and_writes_nothing(tmp_path: Path) -> None:
    store = _store(tmp_path)
    intent, challenge, claim = _forward(store, DEADLINE)
    before = _hash(store)
    with pytest.raises(TargetBindingError):
        store.forward_write(intent, challenge, claim, now=lambda: datetime(2026, 11, 16, 0, 6))
    assert _hash(store) == before


def test_the_other_cas_results_come_before_the_deadline_check(tmp_path: Path) -> None:
    """A fenced key stays FENCED, never DEADLINE_ELAPSED (the sweep's fence wins)."""

    store = _store(tmp_path)
    intent, challenge, claim = _forward(store, DEADLINE)
    store.fence(intent.obligation_group_key, Direction.FORWARD)
    late = store.forward_write(intent, challenge, claim, now=lambda: _at("2026-11-16T00:11:00Z"))
    assert late.executor_result is ExecutorResult.FENCED


# -------------------------------------------------------- TargetNotStarted


def test_a_failed_begin_immediate_is_target_not_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(target_module, "_SQLITE_BUSY_TIMEOUT_SECONDS", 0.05)
    store = _store(tmp_path)
    intent, challenge, claim = _forward(store, None)
    holder = sqlite3.connect(store.path, isolation_level=None)
    try:
        holder.execute("BEGIN IMMEDIATE")
        with pytest.raises(TargetNotStarted) as raised:
            store.forward_write(intent, challenge, claim, now=lambda: _at("2026-11-16T00:06:00Z"))
        assert isinstance(raised.value, TargetUnavailable)
        holder.execute("ROLLBACK")
    finally:
        holder.close()
    # Nothing ran: the same write now succeeds, once.
    assert (
        store.forward_write(intent, challenge, claim, now=lambda: _at("2026-11-16T00:06:00Z")).executor_result
        is ExecutorResult.APPLIED
    )


def test_only_the_begin_branch_raises_target_not_started() -> None:
    import inspect

    source = inspect.getsource(TargetStore._write)
    assert source.count("TargetNotStarted") == 1
    assert source.index("TargetNotStarted") < source.index("yield connection")


# ------------------------------------------------------------- the SweepGate


def test_the_gate_is_closed_from_construction_and_opens_only_when_told() -> None:
    gate = SweepGate(enabled=True)
    assert isinstance(gate, AdmissionGate)
    assert gate.state is GateState.SWEEPING and not gate.is_open() and not gate_open(gate)
    gate.open()
    assert gate.is_open() and gate_open(gate)
    gate.sweeping()
    assert not gate.is_open()
    gate.blocked()
    assert gate.state is GateState.BLOCKED_TARGET and gate.blocked_target() and not gate.is_open()


def test_a_disabled_gate_is_closed_for_good_and_never_blocked_target() -> None:
    gate = SweepGate(enabled=False)
    calls: list[int] = []
    gate.subscribe(lambda: calls.append(1))
    for move in (gate.open, gate.sweeping, gate.blocked, gate.open):
        move()
    assert gate.state is GateState.DISABLED
    assert not gate.is_open() and not gate.blocked_target() and calls == []


def test_subscribe_fires_on_every_transition_into_open_and_at_once_when_open() -> None:
    gate = SweepGate(enabled=True)
    calls: list[str] = []
    gate.subscribe(lambda: calls.append("a"))
    assert calls == []  # closed: nothing yet
    gate.open()
    gate.open()  # no transition, no call
    assert calls == ["a"]
    gate.sweeping()
    gate.open()
    assert calls == ["a", "a"]
    late: list[str] = []
    gate.subscribe(lambda: late.append("late"))  # already OPEN: fires now
    assert late == ["late"]
    gate.blocked()
    gate.open()
    assert calls == ["a", "a", "a"] and late == ["late", "late"]


def test_a_failing_subscriber_does_not_stop_the_others_or_the_gate() -> None:
    gate = SweepGate(enabled=True)
    seen: list[int] = []

    def boom() -> None:
        raise RuntimeError("subscriber")

    gate.subscribe(boom)
    gate.subscribe(lambda: seen.append(1))
    gate.open()
    assert gate.is_open() and seen == [1]


def test_an_enabled_flag_must_be_a_bool() -> None:
    with pytest.raises(TypeError):
        SweepGate(enabled=1)  # type: ignore[arg-type]


# ------------------------------------------------------------------ S1


def test_s1_rotation_is_idempotent(tmp_path: Path) -> None:
    store = _store(tmp_path)  # rotated to RUNTIME already
    new = "00000000-0000-4000-8000-0000000000dd"
    assert rotate_if_needed(store, RUNTIME) is False
    assert rotate_if_needed(store, new) is True
    assert store.meta().runtime_instance_id == new
    before = _hash(store)
    assert rotate_if_needed(store, new) is False  # a retry after a lost response
    assert _hash(store) == before
    # the raw call still refuses an unchanged id: S1 must read meta first.
    with pytest.raises(TargetBindingError):
        store.rotate_runtime_instance(new)


def test_a_late_write_from_the_old_id_after_rotation_is_cas_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    intent, challenge, claim = _forward(store, DEADLINE)
    rotate_if_needed(store, "00000000-0000-4000-8000-0000000000dd")
    receipt = store.forward_write(intent, challenge, claim, now=lambda: _at("2026-11-16T00:06:00Z"))
    assert receipt.executor_result is ExecutorResult.CAS_REJECTED
    assert store.status(intent.obligation_group_key).artifact is None
