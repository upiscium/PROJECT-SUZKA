"""IP-18b (#411 IP-18 D1 r4 §3-1 S2-B / S4, §3-6 item 5, §9-2, §9-4, §9-5): the
sweep's target -> source direction, ``prior_claims()``, the published
``TargetFacts``, the Decision issuer's use of them, the BLOCKED_TARGET witness
with its 5-minute hysteresis, the preview re-post, and F12.

Every fixture is synthetic; nothing here touches a production target.
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any

from fastapi.testclient import TestClient
import pytest

from suzka.r16.decision_issuer import DecisionIssueStatus
from suzka.r16.driver import R16Driver, _BlockedClear
from suzka.r16.successor_policy import LedgerState, SuccessorRow
from suzka.r18.contracts import ApprovalChannel, Direction, ExecutionClaimV1, ExecutorResult, operation_key
from suzka.r18.executor import ClaimKeys
from suzka.r18.facts import (
    TARGET_BLOCKED_REF_KIND,
    TARGET_EFFECT_REF_KIND,
    TARGET_FENCE_REF_KIND,
    FactsHolder,
    TargetFact,
    TargetFacts,
)
from suzka.r18.ledger import PRIOR_CLAIM_REF_KIND
from suzka.r18.sweep import BLOCKED_MIN_DWELL_SECONDS, GateState, SweepGate
from suzka.r18.target import TargetCapacityError, TargetStore

import test_r16_ip16b_ledger_reader as ip16b
import test_r18_ip16a2_intent_issuer as ip16a2
import test_r18_ip18a_driver as ip18a_driver
import test_r18_ip18a_executor as ip18a
from test_r16_u2a_source_owner import _app
from test_r18_ip13_contracts import BOOT, GROUP
from test_r18_ip15_target_verifier import CLAIM_EVENT, RUNTIME
from test_r18_ip18a_executor import _World

FWD = operation_key(GROUP, Direction.FORWARD)


@pytest.fixture
def world(tmp_path: Path):  # type: ignore[no-untyped-def]
    made = _World(tmp_path)
    try:
        yield made
    finally:
        made.close()


# ====================================================================== gate


class _Mono:
    def __init__(self) -> None:
        self.now = 5000.0

    def __call__(self) -> float:
        return self.now


def _gate(mono: _Mono) -> SweepGate:
    return SweepGate(enabled=True, monotonic=mono)


def test_a_plain_sweep_never_latches_blocked_target() -> None:
    mono = _Mono()
    gate = _gate(mono)
    assert gate.blocked_target() is False  # SWEEPING from construction
    gate.open()
    assert gate.blocked_target() is False
    gate.sweeping()
    assert gate.blocked_target() is False


def test_blocked_target_clears_only_after_five_minutes_continuously_open() -> None:
    mono = _Mono()
    gate = _gate(mono)
    cleared: list[int] = []
    gate.subscribe_unblocked(lambda: cleared.append(1))
    gate.blocked()
    assert gate.blocked_target() is True
    gate.open()  # the target is back, but the witness holds for the dwell
    assert gate.is_open() and gate.blocked_target() is True
    mono.now += BLOCKED_MIN_DWELL_SECONDS - 1
    gate.poll()
    assert gate.blocked_target() is True and cleared == []
    mono.now += 1
    assert gate.blocked_target() is False
    assert cleared == [1]
    gate.poll()
    assert cleared == [1]  # once per clear


def test_a_flapping_target_stays_blocked_and_the_dwell_restarts() -> None:
    mono = _Mono()
    gate = _gate(mono)
    cleared: list[int] = []
    gate.subscribe_unblocked(lambda: cleared.append(1))
    gate.blocked()
    for _ in range(20):  # OPEN for 4 minutes, then the target fails again
        gate.open()
        mono.now += BLOCKED_MIN_DWELL_SECONDS - 60
        gate.poll()
        assert gate.blocked_target() is True
        gate.sweeping()
        mono.now += 5
        gate.blocked()
    assert cleared == []
    gate.open()
    mono.now += BLOCKED_MIN_DWELL_SECONDS
    gate.poll()
    assert gate.blocked_target() is False and cleared == [1]


def test_poll_alone_fires_the_unblocked_subscribers_and_a_failing_one_is_harmless() -> None:
    mono = _Mono()
    gate = _gate(mono)
    seen: list[int] = []

    def boom() -> None:
        raise RuntimeError("subscriber")

    gate.subscribe_unblocked(boom)
    gate.subscribe_unblocked(lambda: seen.append(1))
    gate.blocked()
    gate.open()
    mono.now += BLOCKED_MIN_DWELL_SECONDS
    gate.poll()
    assert seen == [1]


def test_a_disabled_gate_is_never_blocked() -> None:
    gate = SweepGate(enabled=False)
    gate.blocked()
    gate.open()
    assert gate.state is GateState.DISABLED and gate.blocked_target() is False


# ==================================================================== driver


def test_a_preview_run_while_blocked_is_repeated_once_when_the_block_clears() -> None:
    mono = _Mono()
    gate = _gate(mono)
    driver, _intent, preview, _policy, _clock = ip18a_driver._gated_driver(gate)
    gate.open()
    driver._maybe_preview("goal-calm", "commitment_accepted")  # not blocked: never repeated
    gate.blocked()
    driver._maybe_preview("goal-a", "commitment_accepted")
    driver._maybe_preview("goal-a", "goal_adopted")
    driver._maybe_preview("goal-b", "commitment_accepted")
    before = len(preview.calls)
    assert before == 4
    gate.subscribe_unblocked(lambda: None)
    driver._subscribe_gate()
    gate.open()
    assert gate.blocked_target() is True  # inside the dwell: nothing is posted
    assert not [i for i in _drain_queue(driver) if isinstance(i, _BlockedClear)]
    mono.now += BLOCKED_MIN_DWELL_SECONDS
    gate.poll()
    items = _drain_queue(driver)
    assert sum(isinstance(i, _BlockedClear) for i in items) == 1
    driver._on_blocked_clear()
    # ONE preview per goal that was previewed while blocked, none for goal-calm.
    assert [c[1] for c in preview.calls[before:]] == ["goal-a", "goal-b"]
    # A second clear with no new blocked preview posts nothing.
    driver._on_blocked_clear()
    assert len(preview.calls) == before + 2


def _drain_queue(driver: R16Driver) -> list[Any]:
    items = []
    while not driver._queue.empty():
        items.append(driver._queue.get_nowait())
    return items


def test_the_blocked_clear_callback_is_a_no_op_after_stop() -> None:
    mono = _Mono()
    gate = _gate(mono)
    driver, *_ = ip18a_driver._gated_driver(gate)
    driver._subscribe_gate()
    driver._stop.set()
    gate.blocked()
    gate.open()
    mono.now += BLOCKED_MIN_DWELL_SECONDS
    gate.poll()
    assert not [i for i in _drain_queue(driver) if isinstance(i, _BlockedClear)]


# ===================================================================== sweep


def _keys(world: _World, *, prior: tuple[tuple[str, str], ...] = (), current: frozenset[str] = frozenset()) -> None:
    world.facts = FactsHolder()  # type: ignore[attr-defined]
    world.calls = []  # type: ignore[attr-defined]

    def read() -> ClaimKeys:
        world.calls.append("keys")  # type: ignore[attr-defined]
        return ClaimKeys(prior, current)

    world.executor._read_claim_keys = read
    world.executor._facts = world.facts  # type: ignore[attr-defined]


def _prior_claim(world: _World) -> ExecutionClaimV1:
    """The claim of the earlier process (the target is rotated to RUNTIME)."""

    return ExecutionClaimV1.claim(
        world.challenge, runtime_instance_id=RUNTIME, boot_id=BOOT, claim_event_id=CLAIM_EVENT,
        claim_sequence=22, claimed_utc=ip18a.CLAIMED, approval_channel=ApprovalChannel.LOCAL_LOOPBACK_TYPED_CODE_V1,
    )


def _sweep(world: _World) -> None:
    world.executor.start()
    world.executor.run_pending()
    assert world.gate.is_open(), world.gate


def test_a_fenced_row_with_no_claim_anywhere_becomes_a_verified_absent_fact(world: _World) -> None:
    world.records.clear()
    world.store.fence(GROUP, Direction.FORWARD)
    _keys(world)
    _sweep(world)
    assert world.facts.get().get(FWD) is TargetFact.FENCED_NO_CLAIM
    assert world.outcome.calls == []  # no Outcome can exist for it
    assert world.writes == 0


def test_an_effect_row_with_no_claim_is_an_effect_fact_and_a_claimed_key_is_not(world: _World) -> None:
    world.records.clear()
    result = world.store.forward_write(world.intent, world.challenge, _prior_claim(world), now=world.clock())
    assert result.executor_result is ExecutorResult.APPLIED
    world.writes = 0
    _keys(world)
    _sweep(world)
    assert world.facts.get().get(FWD) is TargetFact.EFFECT_NO_CLAIM
    # The same row with a claim of this incarnation is the Outcome path's.
    _keys(world, current=frozenset({FWD}))
    world.executor._sweep_done = False
    world.executor._queue_sweep()
    world.executor.run_pending()
    assert world.facts.get().get(FWD) is None


def test_a_prior_incarnation_claim_is_fenced_gets_no_outcome_and_a_late_cas_is_rejected(world: _World) -> None:
    world.records.clear()
    _keys(world, prior=((FWD, GROUP),))
    _sweep(world)
    assert world.status().forward is not None and world.status().forward.kind.value == "FENCED"  # type: ignore[union-attr]
    assert world.facts.get().get(FWD) is TargetFact.FENCED_NO_CLAIM
    assert world.outcome.calls == []  # derive_ledger would report CORRUPT for one
    # The old process' late write: its id no longer matches the rotated target.
    late = world.store.forward_write(world.intent, world.challenge, _prior_claim(world), now=world.clock())
    assert late.executor_result is ExecutorResult.CAS_REJECTED
    assert world.status().artifact is None


def test_a_prior_incarnation_claim_whose_effect_is_present_is_an_effect_fact(world: _World) -> None:
    world.records.clear()
    world.store.forward_write(world.intent, world.challenge, _prior_claim(world), now=world.clock())
    _keys(world, prior=((FWD, GROUP),))
    _sweep(world)
    assert world.facts.get().get(FWD) is TargetFact.EFFECT_NO_CLAIM
    assert world.status().forward.kind.value == "EFFECT"  # type: ignore[union-attr]
    assert world.outcome.calls == []


def test_facts_are_identical_across_two_sweeps_of_the_same_files(world: _World) -> None:
    world.records.clear()
    _keys(world, prior=((FWD, GROUP),))
    _sweep(world)
    first = dict(world.facts.get().by_key)
    world.executor._sweep_done = False
    world.executor._queue_sweep()
    world.executor.run_pending()
    assert dict(world.facts.get().by_key) == first and first == {FWD: TargetFact.FENCED_NO_CLAIM}


def test_the_target_is_read_before_the_claim_keys_so_a_fresh_effect_is_never_misread(world: _World) -> None:
    world.records.clear()
    _keys(world)
    order: list[str] = []
    real = world.executor._read_claim_keys

    def keys() -> ClaimKeys:
        order.append("keys")
        return real()  # type: ignore[misc]

    snapshot = TargetStore.forward_snapshot

    def spy(store: TargetStore) -> Any:
        order.append("snapshot")
        return snapshot(store)

    world.executor._read_claim_keys = keys
    TargetStore.forward_snapshot = spy  # type: ignore[method-assign]
    try:
        _sweep(world)
    finally:
        TargetStore.forward_snapshot = snapshot  # type: ignore[method-assign]
    assert order == ["keys", "snapshot", "keys"]  # prior keys, snapshot, then the claim keys


def test_a_prior_key_past_the_fence_reserve_does_not_keep_the_gate_closed(world: _World, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    world.records.clear()
    _keys(world, prior=((FWD, GROUP),))

    def full(self: TargetStore, group: str, direction: Direction) -> Any:
        raise TargetCapacityError("no room")

    monkeypatch.setattr(TargetStore, "fence", full)
    _sweep(world)  # the gate opens; the key keeps CLAIMED(prior), row 2
    assert world.facts.get().get(FWD) is None


def test_without_claim_keys_the_sweep_is_ip18a_s(world: _World) -> None:
    world.records.clear()
    world.store.fence(GROUP, Direction.FORWARD)
    holder = FactsHolder()
    world.executor._facts = holder  # no read_claim_keys: S2-B / S4 do not run
    _sweep(world)
    assert not holder.get()


def test_the_sweep_logs_s4(world: _World, caplog) -> None:  # type: ignore[no-untyped-def]
    caplog.set_level(logging.INFO, logger="suzka.r18.executor")
    world.records.clear()
    _keys(world, prior=((FWD, GROUP),))
    _sweep(world)
    assert "R18_SWEEP step=S4 facts=1" in caplog.text and "prior=1" in caplog.text


# ================================================ the store's two new queries


def _until(predicate, timeout: float = 40.0):  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("condition not reached")


@pytest.fixture
def settings(tmp_path):  # type: ignore[no-untyped-def]
    return ip16a2._settings(tmp_path / "ip18b")


def _seen(monkeypatch):  # type: ignore[no-untyped-def]
    return ip16b._Seen(monkeypatch)


def test_prior_claims_joins_the_claim_to_its_intent_and_excludes_this_incarnation(settings) -> None:  # type: ignore[no-untyped-def]
    from test_r16_ip10_decision_issuer import _incarnation

    with TestClient(_app(settings)) as client:
        state = client.app.state
        _x, _gid, decision = ip16a2._row4(state)
        chain = ip16b._other_chain(state, decision.decision, upto=5)
        store = state.subject_history_store
        current = _incarnation(state)
        group = ip16b._expected_group(state)
        forward = operation_key(group, Direction.FORWARD)
        assert store.prior_claims(current) == ((forward, group),)
        assert store.prior_claims(ip16b.OTHER) == ()  # that incarnation IS the current one now
        assert store.current_claim_keys(current) == frozenset()
        assert store.current_claim_keys(ip16b.OTHER) == frozenset({chain.claim.operation_key})


# ============================================================ Decision issuer


def _row4_with_facts(state):  # type: ignore[no-untyped-def]
    _x, gid, decision = ip16a2._row4(state)
    forward = operation_key(ip16b._expected_group(state), Direction.FORWARD)
    return gid, decision, forward


@pytest.mark.parametrize(
    ("fact", "ledger", "kind"),
    (
        (TargetFact.FENCED_NO_CLAIM, LedgerState.VERIFIED_ABSENT, TARGET_FENCE_REF_KIND),
        (TargetFact.EFFECT_NO_CLAIM, LedgerState.PRIOR_INCARNATION_EFFECT_PRESENT, TARGET_EFFECT_REF_KIND),
    ),
)
def test_a_row_4_head_sealed_before_facts_gets_a_new_row_2c_decision_not_a_failure(  # type: ignore[no-untyped-def]
    settings, monkeypatch, fact, ledger, kind
) -> None:
    with TestClient(_app(settings)) as client:
        state = client.app.state
        seen = _seen(monkeypatch)
        gid, row4, forward = _row4_with_facts(state)
        state.r18_facts.publish(TargetFacts({forward: fact}))
        result = ip16b._preview(state, gid, seen)
        assert result.row is SuccessorRow.ROW_2C
        assert result.decision_id != row4.decision_id  # a NEW Decision, not the old head's replay
        assert seen.last.ledger is ledger
        assert [ref.kind for ref in seen.last.ledger_refs] == [kind]
        assert seen.last.verified_outcome_ref is None
        # The old head's Intent is refused: no second effect is possible.
        assert ip16a2._intent(state, row4.lineage_scope).reason is not None
        ip16a2._no_failed_events(state)


def test_facts_for_another_group_do_not_touch_this_obligation(settings, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(_app(settings)) as client:
        state = client.app.state
        seen = _seen(monkeypatch)
        gid, row4, _forward = _row4_with_facts(state)
        state.r18_facts.publish(TargetFacts({"f" * 64: TargetFact.FENCED_NO_CLAIM}))
        again = ip16b._preview(state, gid, seen)
        assert (again.status, again.decision_id) == (DecisionIssueStatus.EXISTING, row4.decision_id)
        assert seen.last.ledger is LedgerState.NONE


def test_facts_take_precedence_over_the_prior_claim_path(settings, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(_app(settings)) as client:
        state = client.app.state
        seen = _seen(monkeypatch)
        gid, row4, forward = _row4_with_facts(state)
        ip16b._other_chain(state, row4.decision, upto=5)
        held = ip16b._preview(state, gid, seen)  # facts not complete: CLAIMED(prior), row 2
        assert (held.row, seen.last.ledger) == (SuccessorRow.ROW_2, LedgerState.CLAIMED)
        assert [ref.kind for ref in seen.last.ledger_refs] == [PRIOR_CLAIM_REF_KIND]
        state.r18_facts.publish(TargetFacts({forward: TargetFact.FENCED_NO_CLAIM}))
        done = ip16b._preview(state, gid, seen)  # the sweep fenced it: facts win
        assert (done.row, seen.last.ledger) == (SuccessorRow.ROW_2C, LedgerState.VERIFIED_ABSENT)


def _fake_clock(state) -> _Mono:  # type: ignore[no-untyped-def]
    mono = _Mono()
    state.r18_gate._monotonic = mono
    return mono


def test_a_blocked_target_gives_plan_a_over_an_empty_ledger_and_clears_with_hysteresis(  # type: ignore[no-untyped-def]
    settings, monkeypatch
) -> None:
    with TestClient(_app(settings)) as client:
        state = client.app.state
        seen = _seen(monkeypatch)
        mono = _fake_clock(state)
        gid, row4, _forward = _row4_with_facts(state)
        state.r18_gate.blocked()
        plan_a = ip16b._preview(state, gid, seen)
        assert plan_a.row is SuccessorRow.ROW_2C and plan_a.decision_id != row4.decision_id
        assert seen.last.ledger is LedgerState.TARGET_UNAVAILABLE
        assert [ref.kind for ref in seen.last.ledger_refs] == [TARGET_BLOCKED_REF_KIND]
        # An identical request while it holds: the existing Decision, no churn.
        again = ip16b._preview(state, gid, seen)
        assert (again.status, again.decision_id) == (DecisionIssueStatus.EXISTING, plan_a.decision_id)
        # The target is back, but the witness holds for the dwell.
        state.r18_gate.open()
        assert ip16b._preview(state, gid, seen).decision_id == plan_a.decision_id
        mono.now += BLOCKED_MIN_DWELL_SECONDS
        back = ip16b._preview(state, gid, seen)
        assert back.row is SuccessorRow.ROW_4  # the subject returns to row 4
        assert seen.last.ledger is LedgerState.NONE
        ip16a2._no_failed_events(state)


def test_a_blocked_target_never_replaces_a_claimed_ledger(settings, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(_app(settings)) as client:
        state = client.app.state
        seen = _seen(monkeypatch)
        gid, row4, _forward = _row4_with_facts(state)
        ip16b._other_chain(state, row4.decision, upto=5)
        state.r18_gate.blocked()
        held = ip16b._preview(state, gid, seen)
        assert (held.row, seen.last.ledger) == (SuccessorRow.ROW_2, LedgerState.CLAIMED)


def test_a_blocked_target_with_no_r18_data_computes_no_group_key(settings, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(_app(settings)) as client:
        state = client.app.state
        seen = _seen(monkeypatch)
        x, gid = ip16b._qualified_goal(state)
        assert ip16b._issue(state, gid).row is SuccessorRow.ROW_3
        ip16b._qualified_commitment(state, x)
        # The obligation secret is lost with the target directory.
        monkeypatch.setattr(
            state.decision_issuer,
            "_group_key",
            lambda q: (_ for _ in ()).throw(AssertionError("no key is needed")),
        )
        state.r18_gate.blocked()
        result = ip16b._issue(state, gid)
        assert result.row is SuccessorRow.ROW_2C
        assert seen.last.ledger is LedgerState.TARGET_UNAVAILABLE


def test_a_lost_secret_does_not_hold_the_subject_while_the_target_is_blocked(settings, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from suzka.r16 import decision_issuer as module

    with TestClient(_app(settings)) as client:
        state = client.app.state
        seen = _seen(monkeypatch)
        gid, row4, _forward = _row4_with_facts(state)

        def lost(q: Any) -> str:
            raise module._LedgerSourceUnavailable

        state.r18_facts.publish(TargetFacts({"f" * 64: TargetFact.FENCED_NO_CLAIM}))  # data present
        monkeypatch.setattr(state.decision_issuer, "_group_key", lost)
        held = ip16b._preview(state, gid, seen)
        assert held.status is not DecisionIssueStatus.ISSUED or held.row is not SuccessorRow.ROW_2C
        state.r18_gate.blocked()
        free = ip16b._preview(state, gid, seen)
        assert free.row is SuccessorRow.ROW_2C and seen.last.ledger is LedgerState.TARGET_UNAVAILABLE


def test_with_r18_off_the_issuer_ignores_blocked_target_and_facts(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    settings = ip16a2._settings(tmp_path / "off", r18=False)
    with TestClient(_app(settings)) as client:
        state = client.app.state
        assert state.r18_gate.state is GateState.DISABLED
        state.r18_gate.blocked()
        assert state.decision_issuer._blocked_target() is False
        assert not state.r18_facts.get()


# ============================================================ F12, restarts


def test_f12_a_prior_incarnation_claim_is_fenced_by_the_restart_sweep_and_the_subject_re_decides_plan_a(  # type: ignore[no-untyped-def]
    settings, monkeypatch
) -> None:
    with TestClient(_app(settings)) as client:  # process 1: the claim exists, no executor
        state = client.app.state
        gid, row4, forward = _row4_with_facts(state)
        ip16b._other_chain(state, row4.decision, upto=5)
    calls = {"write": 0}
    real = TargetStore.forward_write

    def counting(store, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls["write"] += 1
        return real(store, *args, **kwargs)

    monkeypatch.setattr(TargetStore, "forward_write", counting)
    with TestClient(_app(settings, executor=True)) as client:  # process 2: the sweep
        state = client.app.state
        _until(lambda: state.r18_gate.is_open())
        assert state.r18_facts.get().get(forward) is TargetFact.FENCED_NO_CLAIM
        seen = _seen(monkeypatch)
        again = ip16b._preview(state, gid, seen)
        assert again.row is SuccessorRow.ROW_2C
        assert seen.last.ledger is LedgerState.VERIFIED_ABSENT
        assert [ref.kind for ref in seen.last.ledger_refs] == [TARGET_FENCE_REF_KIND]
        store = TargetStore.open(Path(state.settings.r18.target_path))
        try:
            row = store.status(ip16b._expected_group(state)).forward
        finally:
            store.close()
        assert row is not None and row.kind.value == "FENCED"
        assert state.subject_history_store._read("SELECT count(*) FROM verified_outcomes") == [(0,)]
        assert calls["write"] == 0
        ip16a2._no_failed_events(state)


def test_a_dropped_clear_message_is_still_handled_through_the_flag() -> None:
    mono = _Mono()
    gate = _gate(mono)
    driver, _intent, preview, _policy, _clock = ip18a_driver._gated_driver(gate)
    driver._subscribe_gate()
    gate.blocked()
    driver._maybe_preview("goal-a", "commitment_accepted")
    before = len(preview.calls)
    gate.open()
    mono.now += BLOCKED_MIN_DWELL_SECONDS
    gate.poll()
    _drain_queue(driver)  # the message was lost (queue full)
    assert driver._blocked_clear_flag.is_set()
    driver._on_blocked_clear()  # what the loop does when it sees the flag
    assert len(preview.calls) == before + 1 and not driver._blocked_clear_flag.is_set()
