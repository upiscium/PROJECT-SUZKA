"""IP-20c: the Wake claim / terminal owner and the Wake review owner on the real
app (IP-20 D1 r3 §4-4, §5, §6, §6-1..§6-4, §9-2).

The Wake is written by ``r19_wake_fixtures.materialize`` (the materializer's role:
IP-20b is not part of this unit); the claim, review and terminal events are the
real owners on the real R07 runtime.  Time is a forward-only fake: ``requested_at``
(T) and the owners' live clock (L) are settable, so leases, queue waits and
expiries are exercised without sleeping.
"""

from __future__ import annotations

import ast
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

import r19_wake_fixtures as fx
from suzka.r16 import decision_issuer as issuer_module
from suzka.r16.cycle_contracts import HoldReason, wake_review_intent_key
from suzka.r16.history_codec import GENESIS, PolicyKind
from suzka.r16.history_participant import SubjectHistoryRecordParticipant
from suzka.r16.history_reader import HeadStatus, SubjectHistoryReader
from suzka.r16.history_records import HistoryOperationKind, RecordKind
from suzka.r16.history_store import HistoryOperationRow, HistoryPhase, HistoryStoreCapacityError
from suzka.r16.successor_policy import SuccessorRow
from suzka.r19 import claim_owner, ledger, review_owner
from suzka.r19.claim_owner import (
    ClaimStatus,
    TerminalStatus,
    WakeClaimOwnerV1,
    WakeCommitRefused,
)
from suzka.r19.contracts import (
    AttemptOutcome,
    RefuseReason,
    TerminalReason,
    WakeAttemptOutcomeV1,
    WakeClaimV1,
    WakeTerminalKind,
    WakeTerminalV1,
)
from suzka.r19.ledger import WakeHold, WakeState, cycle_key
from suzka.r19.review_owner import ReviewStatus, WakeReviewOwnerV1
from suzka.runtime.agent_runtime import AgentRuntimeExecutionError
from test_r16_ip10_decision_issuer import _issue
from test_r16_u2a_source_owner import _app
from test_r16_u3a_plan_preview import _qualified_goal

K = HistoryOperationKind
WAKE_KINDS = frozenset(
    {
        K.R19_WAKE_MATERIALIZE,
        K.R19_WAKE_CLAIM,
        K.R19_WAKE_ATTEMPT_OUTCOME,
        K.R16_CYCLE_DECISION,
        K.R16_CYCLE_RECEIPT,
        K.R19_WAKE_TERMINAL,
    }
)
CONTRADICTORY = SimpleNamespace(outcome="EPISODE", stage="S2", status="CONTRADICTORY")


@pytest.fixture
def world(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    with TestClient(_app(fx.settings(tmp_path / "w"))) as client:
        state = client.app.state
        clock = fx.Time(state, monkeypatch)
        _x, gid = _qualified_goal(state)
        yield SimpleNamespace(state=state, time=clock, gid=gid, monkeypatch=monkeypatch)


def _count(state) -> int:  # type: ignore[no-untyped-def]
    return len(fx.operations(state))


def _wake_ops(state) -> list:  # type: ignore[no-untyped-def]
    return [row for row in fx.operations(state) if row.operation_kind in WAKE_KINDS]


def _entry(w, wake):  # type: ignore[no-untyped-def]
    return fx.entry(w.state, wake, w.time.now())


class _Prepares:
    """Counts ``participant.prepare`` (a prepare that raises fail-stops the runtime)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.count = 0
        real = SubjectHistoryRecordParticipant.prepare
        outer = self

        def spy(self_, binding):  # type: ignore[no-untyped-def]
            outer.count += 1
            return real(self_, binding)

        monkeypatch.setattr(SubjectHistoryRecordParticipant, "prepare", spy)


def _no_failed_events(state) -> None:  # type: ignore[no-untyped-def]
    from suzka.runtime.event_journal import EventLifecycle

    assert not any(
        record.lifecycle is EventLifecycle.FAILED for record in state.event_journal.inspect().records
    )


# --------------------------------------------------------------- R19 disabled


def test_disabled_owners_hold_before_reading_anything(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with TestClient(_app(fx.settings(tmp_path / "off", r19=False))) as client:
        state = client.app.state
        claim_owner, review_owner_ = state.wake_claim_owner, state.wake_review_owner
        assert claim_owner is not None and not claim_owner.enabled
        assert review_owner_ is not None and not review_owner_.enabled

        def unreadable(*args):  # type: ignore[no-untyped-def]
            raise AssertionError("a disabled owner read something")

        for owner in (claim_owner, review_owner_):
            owner._classify = unreadable  # type: ignore[method-assign]
            owner._clock = unreadable  # type: ignore[method-assign]
        wakeup = "wakeup-" + "0" * 64
        claimed = claim_owner._submit_wake_claim(wakeup).result(timeout=30).value
        assert (claimed.status, claimed.hold) == (ClaimStatus.HOLD, WakeHold.R19_DISABLED)
        ended = claim_owner._submit_terminal(wakeup, TerminalReason.EXPIRED_WAKE).result(timeout=30).value
        assert (ended.status, ended.hold) == (TerminalStatus.HOLD, WakeHold.R19_DISABLED)
        reviewed = review_owner_._submit_review("wake-claim-" + "0" * 64).result(timeout=30).value
        assert (reviewed.status, reviewed.hold) == (ReviewStatus.HOLD, WakeHold.R19_DISABLED)
        assert not state.subject_history_store.has_operations()
        _no_failed_events(state)


# --------------------------------------------------------------------- claim


def test_claims_are_ordered_gapped_and_bounded_at_three(world) -> None:  # type: ignore[no-untyped-def]
    state, clock = world.state, world.time
    wake = fx.materialize(state, world.gid, clock)
    before = _count(state)
    started = clock.now()
    first = fx.claim(state, wake)
    assert first.status is ClaimStatus.COMMITTED and first.claim.attempt_index == 1
    lease = datetime.strptime(first.claim.claim_lease_expires_utc, fx.STAMP).replace(tzinfo=timezone.utc)
    assert 119 <= (lease - started).total_seconds() <= 121  # T + claim_lease_s
    assert _entry(world, wake).state is WakeState.CLAIMED_LIVE
    after_first = _count(state)
    assert after_first == before + 1
    # The gap rule: nothing inside the lease, and no row for the hold.
    clock.advance(60)
    held = fx.claim(state, wake)
    assert (held.status, held.hold) == (ClaimStatus.HOLD, WakeHold.LEASE_LIVE)
    assert _count(state) == after_first
    # Attempts 2 and 3, each after the previous lease.
    clock.advance(61)  # 121 s after the first claim: its lease (120 s) is over
    for index in (2, 3):
        if index == 3:
            clock.advance(121)
        made = fx.claim(state, wake)
        assert made.status is ClaimStatus.COMMITTED and made.claim.attempt_index == index
    clock.advance(121)
    over = fx.claim(state, wake)
    assert (over.status, over.hold) == (ClaimStatus.HOLD, WakeHold.ATTEMPTS_USED)
    assert _entry(world, wake).state is WakeState.EXHAUSTED
    # The raw store has exactly the three claims and the attempt keys are live.
    keys = [f"wake-attempt:{wake.wakeup_id}:{n}" for n in (1, 2, 3)]
    assert state.subject_history_store.live_keys(tuple(keys)) == frozenset(keys)
    assert state.wake_claim_owner._pending is None
    _no_failed_events(state)


def test_a_claim_outside_its_window_or_for_an_unknown_wake_writes_nothing(world) -> None:  # type: ignore[no-untyped-def]
    state = world.state
    future = fx.materialize(state, world.gid, world.time, not_before=3600)
    before = _count(state)
    held = fx.claim(state, future)
    assert (held.status, held.hold) == (ClaimStatus.HOLD, WakeHold.OUTSIDE_WAKE_WINDOW)
    unknown = state.wake_claim_owner._submit_wake_claim("wakeup-" + "1" * 64).result(timeout=30).value
    assert (unknown.status, unknown.hold) == (ClaimStatus.HOLD, WakeHold.WAKE_UNKNOWN)
    assert _count(state) == before
    # Past the window it is the claim that is refused, not an attempt.
    world.time.advance(3601 + 7 * 86_400)
    expired = fx.claim(state, future)
    assert (expired.status, expired.hold) == (ClaimStatus.HOLD, WakeHold.OUTSIDE_WAKE_WINDOW)
    assert _count(state) == before


def test_a_claim_needs_a_verifiable_clock_and_a_short_queue_wait(world) -> None:  # type: ignore[no-untyped-def]
    state, clock = world.state, world.time
    wake = fx.materialize(state, world.gid, clock, not_before=-4 * 3600)
    before = _count(state)
    clock.skew = timedelta(seconds=11)
    late = fx.claim(state, wake)
    assert (late.status, late.hold) == (ClaimStatus.HOLD, WakeHold.QUEUE_DELAY)
    clock.skew = timedelta(seconds=-5)
    behind = fx.claim(state, wake)
    assert (behind.status, behind.hold) == (ClaimStatus.HOLD, WakeHold.CLOCK_UNVERIFIABLE)
    clock.skew = timedelta(0)
    assert _count(state) == before  # a claim that cannot be made is not an attempt
    # A rolled-back clock across a restart: T below the Journal high-water.
    state.agent_runtime._last_admitted_at = None
    clock.advance(-3 * 3600)
    rolled = fx.claim(state, wake)
    assert (rolled.status, rolled.hold) == (ClaimStatus.HOLD, WakeHold.CLOCK_UNVERIFIABLE)
    assert _count(state) == before
    clock.advance(3 * 3600)
    ok = fx.claim(state, wake)
    assert ok.status is ClaimStatus.COMMITTED
    _no_failed_events(state)


# -------------------------------------------------------------------- review


def test_a_review_issues_the_decision_and_closes_the_cycle_once(world) -> None:  # type: ignore[no-untyped-def]
    state, clock, monkeypatch = world.state, world.time, world.monkeypatch
    wake = fx.materialize(state, world.gid, clock)
    claimed = fx.claim(state, wake)
    calls: list[tuple[object, ...]] = []
    real = state.decision_issuer._evaluate

    def spy(event, goal_id, *, now, before_sequence):  # type: ignore[no-untyped-def]
        calls.append((event.requested_at, event.processing_sequence, now, before_sequence))
        return real(event, goal_id, now=now, before_sequence=before_sequence)

    monkeypatch.setattr(state.decision_issuer, "_evaluate", spy)
    result = fx.review(state, claimed.claim.claim_id)
    assert result.status is ReviewStatus.DECISION_ISSUED
    # A-14 / A-5: frozen T and the review event's own sequence.
    ((requested, sequence, now, before),) = calls
    assert (now, before) == (requested, sequence)
    entry = _entry(world, wake)
    assert entry.state is WakeState.CLOSED
    resolution = entry.resolution
    assert resolution is not None
    key = wake_review_intent_key(
        wake.logical_cycle_id, wake.frozen_predecessor, wake.policy_kind, wake.evidence_fingerprint
    )
    assert resolution.review_intent_key == key and resolution.result_id == result.decision_id
    head = SubjectHistoryReader(state.event_journal, state.subject_history_store).head(
        entry.closure.decision.lineage_scope
    )
    assert (head.status, head.depth) == (HeadStatus.ADOPTED, 1)
    assert head.decision.decision_id == result.decision_id
    assert head.decision.review_intent_key == key and head.decision.predecessor_key == GENESIS
    # One operation holds the Decision, its sources and the resolution; the
    # event terminal exists (the adopted view only counts it then).
    cycle_ops = [row for row in fx.operations(state) if row.operation_kind is K.R16_CYCLE_DECISION]
    assert len(cycle_ops) == 1 and cycle_ops[0].phase is HistoryPhase.FINALIZED
    assert cycle_ops[0].event_id == entry.closure.decision.issuer_event_id
    # The cycle closes once: no claim, review or terminal writes anything more.
    count = _count(state)
    again = fx.claim(state, wake)
    assert (again.status, again.resolution) == (ClaimStatus.EXISTING, resolution)
    repeat = fx.review(state, claimed.claim.claim_id)
    assert (repeat.status, repeat.resolution) == (ReviewStatus.EXISTING, resolution)
    done = fx.terminal(state, wake, TerminalReason.EXPIRED_WAKE)
    assert (done.status, done.hold) == (TerminalStatus.HOLD, WakeHold.WAKE_CLOSED)
    assert _count(state) == count
    assert fx.adopted(state, WakeTerminalV1) == []
    # The review never touches the issuer's own pending, and a PLAN_PREVIEW
    # after it sees the Wake's Decision as its head.
    assert state.decision_issuer._pending is None and state.wake_review_owner._pending is None
    preview = _issue(state, world.gid)
    assert preview.decision_id == result.decision_id
    _no_failed_events(state)


def test_the_same_evidence_closes_as_an_existing_receipt_never_failed(world) -> None:  # type: ignore[no-untyped-def]
    """S1: a Wake whose evidence a PLAN_PREVIEW Decision already holds."""

    state, clock = world.state, world.time
    wake = fx.materialize(state, world.gid, clock)
    issued = _issue(state, world.gid)
    claimed = fx.claim(state, wake)
    result = fx.review(state, claimed.claim.claim_id)
    assert result.status is ReviewStatus.EXISTING_DECISION
    entry = _entry(world, wake)
    receipt = entry.closure.result
    assert entry.state is WakeState.CLOSED and entry.terminal is None
    assert receipt.head_decision_id == issued.decision_id
    assert [row.operation_kind for row in _wake_ops(state)].count(K.R16_CYCLE_RECEIPT) == 1
    # No Decision was added and the cycle cannot be claimed or ended again.
    assert len([row for row in fx.operations(state) if row.operation_kind in (K.R16_DECISION_ISSUE, K.R16_CYCLE_DECISION)]) == 1
    count = _count(state)
    assert fx.claim(state, wake).status is ClaimStatus.EXISTING
    assert fx.terminal(state, wake, TerminalReason.EXPIRED_WAKE).hold is WakeHold.WAKE_CLOSED
    assert _count(state) == count


def test_source_drift_closes_as_a_hold_receipt_once(world) -> None:  # type: ignore[no-untyped-def]
    state, clock, monkeypatch = world.state, world.time, world.monkeypatch
    with monkeypatch.context() as patch:
        patch.setattr(issuer_module, "_context_digest", lambda q: "e" * 64)
        wake = fx.materialize(state, world.gid, clock)
    issued = _issue(state, world.gid)  # the real sources: a head with other source refs
    assert issued.decision is not None
    claimed = fx.claim(state, wake)
    monkeypatch.setattr(issuer_module, "_context_digest", lambda q: "e" * 64)
    result = fx.review(state, claimed.claim.claim_id)
    assert (result.status, result.hold_reason) == (
        ReviewStatus.HOLD_RECEIPT,
        HoldReason.DECISION_SOURCE_DRIFT,
    )
    entry = _entry(world, wake)
    assert entry.state is WakeState.CLOSED and entry.terminal is None
    assert fx.claim(state, wake).status is ClaimStatus.EXISTING


def test_a_changed_source_closes_as_stale_and_a_none_version_is_no_match(world) -> None:  # type: ignore[no-untyped-def]
    state, clock, monkeypatch = world.state, world.time, world.monkeypatch
    wake = fx.materialize(state, world.gid, clock)
    claimed = fx.claim(state, wake)
    monkeypatch.setattr(issuer_module, "_context_digest", lambda q: "c" * 64)
    result = fx.review(state, claimed.claim.claim_id)
    assert (result.status, result.hold_reason) == (ReviewStatus.HOLD_RECEIPT, HoldReason.STALE_SOURCE)
    entry = _entry(world, wake)
    assert entry.state is WakeState.CLOSED and entry.terminal is None
    receipt = entry.closure.result
    assert receipt.reason is HoldReason.STALE_SOURCE
    assert fx.claim(state, wake).status is ClaimStatus.EXISTING


def test_an_inexpressible_version_is_no_match_never_an_error(world) -> None:  # type: ignore[no-untyped-def]
    """Forward note 1: ``as_version`` returning None is "no match"."""

    state, clock, monkeypatch = world.state, world.time, world.monkeypatch
    wake = fx.materialize(state, world.gid, clock)
    claimed = fx.claim(state, wake)
    monkeypatch.setattr(review_owner, "as_version", lambda value, version: None)
    result = fx.review(state, claimed.claim.claim_id)
    assert (result.status, result.hold_reason) == (ReviewStatus.HOLD_RECEIPT, HoldReason.STALE_SOURCE)


def test_a_v2_wake_is_compared_in_its_own_version(world) -> None:  # type: ignore[no-untyped-def]
    state, clock = world.state, world.time
    wake = fx.materialize(state, world.gid, clock, policy_kind=PolicyKind.SUCCESSOR_POLICY_V2)
    assert wake.policy_kind is PolicyKind.SUCCESSOR_POLICY_V2
    claimed = fx.claim(state, wake)
    result = fx.review(state, claimed.claim.claim_id)
    assert result.status is ReviewStatus.DECISION_ISSUED
    decision = _entry(world, wake).closure.decision
    # The label follows the sealed bytes (current issuer), the key the Wake's version.
    assert decision.policy_kind is PolicyKind.SUCCESSOR_POLICY_V3
    assert decision.review_intent_key == wake_review_intent_key(
        wake.logical_cycle_id, wake.frozen_predecessor, PolicyKind.SUCCESSOR_POLICY_V2, wake.evidence_fingerprint
    )


def test_a_policy_hold_closes_with_a_hold_receipt_naming_its_row(world) -> None:  # type: ignore[no-untyped-def]
    state, clock, monkeypatch = world.state, world.time, world.monkeypatch
    monkeypatch.setattr(state.decision_issuer, "_interpretation", lambda *args: CONTRADICTORY)
    wake = fx.materialize(state, world.gid, clock)
    claimed = fx.claim(state, wake)
    result = fx.review(state, claimed.claim.claim_id)
    assert (result.status, result.hold_reason) == (ReviewStatus.HOLD_RECEIPT, HoldReason.POLICY_HOLD)
    entry = _entry(world, wake)
    assert entry.state is WakeState.CLOSED and entry.closure.result.policy_row is SuccessorRow.ROW_2B
    assert not any(row.operation_kind is K.R16_CYCLE_DECISION for row in fx.operations(state))


def test_an_insufficient_commitment_deadline_closes_as_a_hold_receipt(world) -> None:  # type: ignore[no-untyped-def]
    """§6 step (4): the deadline is checked against the LIVE clock with a 15 s
    margin, so the queue wait ``L - T`` is absorbed, not capped."""

    state, clock, monkeypatch = world.state, world.time, world.monkeypatch
    wake = fx.materialize(state, world.gid, clock)
    claimed = fx.claim(state, wake)
    seen: list[tuple[object, object]] = []

    def deadline(q, *, now, margin):  # type: ignore[no-untyped-def]
        seen.append((now(), margin))
        raise issuer_module._IssuerHold(issuer_module.R16OwnerReason.DEADLINE_ELAPSED)

    monkeypatch.setattr(state.decision_issuer, "_check_deadline", deadline)
    clock.skew = timedelta(seconds=40)
    result = fx.review(state, claimed.claim.claim_id)
    assert (result.status, result.hold_reason) == (
        ReviewStatus.HOLD_RECEIPT,
        HoldReason.INSUFFICIENT_EVIDENCE,
    )
    ((live, margin),) = seen
    assert margin.total_seconds() == 15
    assert (live - clock.now()).total_seconds() > 30  # L, not T


def test_a_review_waits_in_the_queue_up_to_its_lease(world) -> None:  # type: ignore[no-untyped-def]
    """§4-5: a 60 s queue wait inside the lease is not refused and not DEFERRED."""

    state, clock = world.state, world.time
    wake = fx.materialize(state, world.gid, clock)
    claimed = fx.claim(state, wake)
    clock.skew = timedelta(seconds=60)
    result = fx.review(state, claimed.claim.claim_id)
    assert result.status is ReviewStatus.DECISION_ISSUED


# ------------------------------------------------------- refusals and budget


def test_an_unavailable_source_is_a_recorded_refusal_and_three_exhaust_the_cycle(world) -> None:  # type: ignore[no-untyped-def]
    state, clock, monkeypatch = world.state, world.time, world.monkeypatch
    wake = fx.materialize(state, world.gid, clock)
    model_calls = state.model_provider.calls

    def unavailable(*args):  # type: ignore[no-untyped-def]
        raise RuntimeError("R15 is down")

    monkeypatch.setattr(state.decision_issuer, "_interpretation", unavailable)
    for attempt in (1, 2, 3):
        claimed = fx.claim(state, wake)
        assert claimed.status is ClaimStatus.COMMITTED and claimed.claim.attempt_index == attempt
        result = fx.review(state, claimed.claim.claim_id)
        assert result.status is ReviewStatus.ATTEMPT_RECORDED
        assert (result.outcome, result.refuse) == (AttemptOutcome.REFUSED, RefuseReason.SOURCE_UNAVAILABLE)
        entry = _entry(world, wake)
        assert entry.latest_outcome == WakeAttemptOutcomeV1(
            claimed.claim.claim_id, AttemptOutcome.REFUSED, RefuseReason.SOURCE_UNAVAILABLE
        )
        # The race of the two writers on ``attempt-end:``: a second review of
        # the ended attempt is a hold, never a fail-stop and never a row.
        count = _count(state)
        second = fx.review(state, claimed.claim.claim_id)
        assert (second.status, second.hold) == (ReviewStatus.HOLD, WakeHold.ATTEMPT_ENDED)
        assert _count(state) == count
        if attempt < 3:
            # The outcome never shortens the gap: the lease still bounds the next claim.
            inside = fx.claim(state, wake)
            assert (inside.status, inside.hold) == (ClaimStatus.HOLD, WakeHold.LEASE_LIVE)
            clock.advance(121)
    assert _entry(world, wake).state is WakeState.EXHAUSTED
    ended = fx.terminal(state, wake, TerminalReason.ATTEMPTS_EXHAUSTED)
    assert ended.status is TerminalStatus.COMMITTED
    assert ended.terminal.terminal is WakeTerminalKind.FAILED_TERMINAL
    assert _entry(world, wake).state is WakeState.FAILED
    # Row-writing events of the cycle: 1 materialize + 3 x (claim + review) + 1 terminal.
    assert len(_wake_ops(state)) == 1 + 3 * 2 + 1 <= 9
    assert state.model_provider.calls == model_calls  # no model call on the Wake path
    assert fx.claim(state, wake).hold is WakeHold.WAKE_TERMINAL
    _no_failed_events(state)


def test_the_other_recordable_refusals(world) -> None:  # type: ignore[no-untyped-def]
    state, clock, monkeypatch = world.state, world.time, world.monkeypatch
    prepares = _Prepares(monkeypatch)
    # LINEAGE_MISMATCH: the Wake's own Journal lineage differs from the current one.
    other = fx.materialize(state, world.gid, clock, lineage="another-lineage")
    claimed = fx.claim(state, other)
    result = fx.review(state, claimed.claim.claim_id)
    assert (result.status, result.refuse) == (ReviewStatus.ATTEMPT_RECORDED, RefuseReason.LINEAGE_MISMATCH)
    assert prepares.count == 3  # materialize, claim, outcome: each prepared cleanly
    # CLOCK_UNVERIFIABLE counts as an attempt (D-R19 §4 (2)): L < T.
    clock.advance(121)
    claimed = fx.claim(state, other)
    assert claimed.claim.attempt_index == 2
    clock.skew = timedelta(seconds=-5)
    result = fx.review(state, claimed.claim.claim_id)
    assert (result.status, result.refuse) == (ReviewStatus.ATTEMPT_RECORDED, RefuseReason.CLOCK_UNVERIFIABLE)
    clock.skew = timedelta(0)
    _no_failed_events(state)


def test_a_slow_review_records_deferred_budget(world) -> None:  # type: ignore[no-untyped-def]
    state, clock = world.state, world.time
    wake = fx.materialize(state, world.gid, clock)
    claimed = fx.claim(state, wake)
    ticks = iter([0.0, 100.0, 100.0, 100.0, 100.0])
    state.wake_review_owner._monotonic = lambda: next(ticks)
    result = fx.review(state, claimed.claim.claim_id)
    assert (result.status, result.outcome, result.refuse) == (
        ReviewStatus.ATTEMPT_RECORDED,
        AttemptOutcome.DEFERRED_BUDGET,
        None,
    )
    assert _entry(world, wake).latest_outcome.outcome is AttemptOutcome.DEFERRED_BUDGET
    # The gap still holds: DEFERRED_BUDGET writes a row, not an earlier claim.
    assert fx.claim(state, wake).hold is WakeHold.LEASE_LIVE


def test_decision_slots_and_moved_sources_are_pre_checked(world) -> None:  # type: ignore[no-untyped-def]
    state, clock, monkeypatch = world.state, world.time, world.monkeypatch
    prepares = _Prepares(monkeypatch)
    wake = fx.materialize(state, world.gid, clock)
    # Attempt 1: the sources moved while the Decision was built.
    claimed = fx.claim(state, wake)
    with monkeypatch.context() as patch:
        patch.setattr(WakeReviewOwnerV1, "_sources_moved", lambda self, event, evaluated: True)
        result = fx.review(state, claimed.claim.claim_id)
    assert (result.status, result.refuse) == (ReviewStatus.ATTEMPT_RECORDED, RefuseReason.SNAPSHOT_CHANGED)
    clock.advance(121)
    # Attempt 2: a PREPARED Decision holds the slot: recordable, not a fail-stop.
    claimed = fx.claim(state, wake)
    monkeypatch.setattr(review_owner, "_slot_holder", lambda rows, committed: HistoryPhase.PREPARED)
    result = fx.review(state, claimed.claim.claim_id)
    assert (result.status, result.refuse) == (
        ReviewStatus.ATTEMPT_RECORDED,
        RefuseReason.DECISION_SLOT_PENDING,
    )
    clock.advance(121)
    # Attempt 3: a FINALIZED holder closes the question (STALE_SOURCE).
    claimed = fx.claim(state, wake)
    monkeypatch.setattr(review_owner, "_slot_holder", lambda rows, committed: HistoryPhase.FINALIZED)
    result = fx.review(state, claimed.claim.claim_id)
    assert (result.status, result.hold_reason) == (ReviewStatus.HOLD_RECEIPT, HoldReason.STALE_SOURCE)
    assert prepares.count >= 6
    _no_failed_events(state)


def test_slot_holder_prefers_a_finalized_holder() -> None:
    def row(kind, phase, **keys):  # type: ignore[no-untyped-def]
        base = dict(
            incarnation_id="i", transaction_id="t" + str(phase), event_id="e", processing_sequence=1,
            transaction_kind="event_mutation", operation_kind=kind, operation_digest="d", phase=phase,
            pending_bytes=None, pending_digest=None, lineage_scope=None, review_intent_key=None,
            predecessor_key=None,
        )
        return HistoryOperationRow(**{**base, **keys})

    committed = SimpleNamespace(lineage_scope="s", review_intent_key="k", predecessor_key="p")
    held = dict(lineage_scope="s", review_intent_key="k", predecessor_key="p")
    assert review_owner._slot_holder([], committed) is None  # type: ignore[arg-type]
    prepared = row(K.R16_CYCLE_DECISION, HistoryPhase.PREPARED, **held)
    finalized = row(K.R16_DECISION_ISSUE, HistoryPhase.FINALIZED, **held)
    aborted = row(K.R16_CYCLE_DECISION, HistoryPhase.ABORTED, **held)
    assert review_owner._slot_holder([prepared], committed) is HistoryPhase.PREPARED  # type: ignore[arg-type]
    assert review_owner._slot_holder([prepared, finalized], committed) is HistoryPhase.FINALIZED  # type: ignore[arg-type]
    assert review_owner._slot_holder([aborted], committed) is None  # type: ignore[arg-type]
    unrelated = row(K.R19_WAKE_CLAIM, HistoryPhase.PREPARED, **held)
    assert review_owner._slot_holder([unrelated], committed) is None  # type: ignore[arg-type]


# --------------------------------------------- mirrored pre-checks (D1 §6-4)


def _claimed(world):  # type: ignore[no-untyped-def]
    wake = fx.materialize(world.state, world.gid, world.time)
    return wake, fx.claim(world.state, wake).claim


def _later_claim(world, wake, claim):  # type: ignore[no-untyped-def]
    """A FIXTURE claim 2 (adopted), so claim 1 is no longer the latest."""

    fx.write_record(
        world.state,
        K.R19_WAKE_CLAIM,
        RecordKind.WAKE_CLAIM,
        lambda event: WakeClaimV1(
            wake.wakeup_id, wake.logical_cycle_id, wake.evidence_fingerprint,
            wake.incarnation_id, 2, event.event_id, event.processing_sequence,
            claim.claim_lease_expires_utc,
        ),
    )


def _fixture_outcome(world, wake, claim):  # type: ignore[no-untyped-def]
    fx.write_record(
        world.state,
        K.R19_WAKE_ATTEMPT_OUTCOME,
        RecordKind.WAKE_ATTEMPT_OUTCOME,
        lambda event: WakeAttemptOutcomeV1(
            claim.claim_id, AttemptOutcome.REFUSED, RefuseReason.SOURCE_UNAVAILABLE
        ),
    )


def _fixture_terminal(world, wake, claim):  # type: ignore[no-untyped-def]
    fx.write_record(
        world.state,
        K.R19_WAKE_TERMINAL,
        RecordKind.WAKE_TERMINAL,
        lambda event: WakeTerminalV1.for_wake(
            wake, WakeTerminalKind.FAILED_TERMINAL, TerminalReason.EXPIRED_WAKE,
            event_id=event.event_id, event_sequence=event.processing_sequence,
        ),
    )


def _reserved_cycle(world, wake, claim):  # type: ignore[no-untyped-def]
    """A PREPARED closure of this cycle: its key is live but nothing is adopted."""

    store = world.state.subject_history_store
    real = store.live_keys
    world.monkeypatch.setattr(
        store, "live_keys",
        lambda keys: real(keys) | {key for key in keys if key.startswith("cycle:")},
    )


def _no_headroom(world, wake, claim):  # type: ignore[no-untyped-def]
    def full(length):  # type: ignore[no-untyped-def]
        raise HistoryStoreCapacityError("full")

    world.monkeypatch.setattr(world.state.subject_history_store, "check_headroom", full)


def _lease_over(world, wake, claim):  # type: ignore[no-untyped-def]
    world.time.skew = timedelta(seconds=125)


_MIRRORS = {
    "claim_not_latest": (_later_claim, WakeHold.CLAIM_NOT_LATEST),
    "attempt_end_live": (_fixture_outcome, WakeHold.ATTEMPT_ENDED),
    "wake_terminal": (_fixture_terminal, WakeHold.WAKE_TERMINAL),
    "cycle_key_live": (_reserved_cycle, WakeHold.CYCLE_RESERVED),
    "no_headroom": (_no_headroom, WakeHold.HISTORY_CAPACITY),
    "lease_over": (_lease_over, WakeHold.OUTSIDE_WAKE_WINDOW),
}


@pytest.mark.parametrize("case", sorted(_MIRRORS))
def test_a_violated_trigger_condition_is_a_hold_with_no_row_and_no_prepare(world, case) -> None:  # type: ignore[no-untyped-def]
    """D1 §6-4: with the condition violated the handler returns a HOLD,
    ``participant.prepare`` is never entered and nothing is written."""

    state = world.state
    wake, claim = _claimed(world)
    violate, expected = _MIRRORS[case]
    violate(world, wake, claim)
    prepares = _Prepares(world.monkeypatch)
    rows = _count(state)
    result = fx.review(state, claim.claim_id)
    assert (result.status, result.hold) == (ReviewStatus.HOLD, expected)
    assert prepares.count == 0
    assert _count(state) == rows
    _no_failed_events(state)


def test_an_unknown_claim_is_a_hold(world) -> None:  # type: ignore[no-untyped-def]
    state = world.state
    _claimed(world)
    prepares = _Prepares(world.monkeypatch)
    rows = _count(state)
    result = fx.review(state, "wake-claim-" + "9" * 64)
    assert (result.status, result.hold) == (ReviewStatus.HOLD, WakeHold.WAKE_UNKNOWN)
    assert prepares.count == 0 and _count(state) == rows


def test_one_review_per_claim_is_bounded_by_the_journal(world) -> None:  # type: ignore[no-untyped-def]
    """HD-20-2 (a): a review that STARTED (even one that wrote nothing) is the
    claim's one review; the bound is read from the Journal, so a restart or a
    crash loop adds none."""

    state, monkeypatch = world.state, world.monkeypatch
    wake, claim = _claimed(world)
    with monkeypatch.context() as patch:
        _no_headroom(SimpleNamespace(state=state, monkeypatch=patch), wake, claim)
        first = fx.review(state, claim.claim_id)
    assert (first.status, first.hold) == (ReviewStatus.HOLD, WakeHold.HISTORY_CAPACITY)
    prepares = _Prepares(monkeypatch)
    rows = _count(state)
    second = fx.review(state, claim.claim_id)
    assert (second.status, second.hold) == (ReviewStatus.HOLD, WakeHold.REVIEW_ALREADY_STARTED)
    assert prepares.count == 0 and _count(state) == rows
    # The attempt simply lapses: the next claim comes after the lease.
    world.time.advance(121)
    assert fx.claim(state, wake).claim.attempt_index == 2


# ------------------------------------------------------------------ terminals


def test_expired_wake_closes_a_wake_only_when_it_is_expired(world) -> None:  # type: ignore[no-untyped-def]
    state, clock = world.state, world.time
    wake = fx.materialize(state, world.gid, clock, not_before=-600, expires=300)
    rows = _count(state)
    early = fx.terminal(state, wake, TerminalReason.EXPIRED_WAKE)
    assert (early.status, early.hold) == (TerminalStatus.HOLD, WakeHold.NOT_DUE)
    assert _count(state) == rows
    clock.advance(301)
    done = fx.terminal(state, wake, TerminalReason.EXPIRED_WAKE)
    assert done.status is TerminalStatus.COMMITTED
    assert done.terminal.terminal is WakeTerminalKind.FAILED_TERMINAL
    assert done.terminal.logical_cycle_id == wake.logical_cycle_id
    assert _entry(world, wake).state is WakeState.FAILED
    # The terminal holds the cycle key: nothing else can close or claim this cycle.
    assert state.subject_history_store.live_keys((cycle_key(wake.incarnation_id, wake.logical_cycle_id),))
    assert fx.claim(state, wake).hold is WakeHold.WAKE_TERMINAL
    assert fx.terminal(state, wake, TerminalReason.EXPIRED_WAKE).hold is WakeHold.WAKE_TERMINAL
    _no_failed_events(state)


def test_a_terminal_with_a_foreign_cycle_id_is_refused_before_prepare(world) -> None:  # type: ignore[no-untyped-def]
    """A-4: ``wake_terminals_insert`` requires the Wake's own cycle key; a
    foreign id would ABORT in SQL (a runtime fail-stop), so the writer checks."""

    state, clock, monkeypatch = world.state, world.time, world.monkeypatch
    wake = fx.materialize(state, world.gid, clock, not_before=-600, expires=300)
    clock.advance(301)
    real = WakeClaimOwnerV1._build_terminal

    def foreign(entry, reason, newer, event, sequence):  # type: ignore[no-untyped-def]
        return replace(real(entry, reason, newer, event, sequence), logical_cycle_id="f" * 64)

    monkeypatch.setattr(WakeClaimOwnerV1, "_build_terminal", staticmethod(foreign))
    prepares = _Prepares(monkeypatch)
    rows = _count(state)
    held = fx.terminal(state, wake, TerminalReason.EXPIRED_WAKE)
    assert (held.status, held.hold) == (TerminalStatus.HOLD, WakeHold.COLLISION)
    assert prepares.count == 0 and _count(state) == rows


def _second_wake(world):  # type: ignore[no-untyped-def]
    """A newer Wake of the same Goal (another cause event and fingerprint).  The
    Outcome-cause Wake ``may_supersede`` wants needs a real verified Outcome row
    (a foreign key), which is IP-20b / IP-22 territory: the owner defers to
    ``may_supersede`` itself (the IP-13b contract), so the owner-level cases
    here fix its answer and the contract predicate keeps its own tests."""

    return fx.materialize(
        world.state, world.gid, world.time,
        cause_event_id=str(uuid4()), evidence_fingerprint="e" * 64,
    )


def test_superseded_needs_an_unclaimed_old_wake_and_may_supersede(world) -> None:  # type: ignore[no-untyped-def]
    state, monkeypatch = world.state, world.monkeypatch
    old = fx.materialize(state, world.gid, world.time)
    newer = _second_wake(world)
    assert newer.wakeup_id != old.wakeup_id
    prepares = _Prepares(monkeypatch)
    rows = _count(state)
    # The contract says no (a GOAL-cause Wake never supersedes): a hold, no row.
    refused = fx.terminal(state, old, TerminalReason.SUPERSEDED, newer.wakeup_id)
    assert (refused.status, refused.hold) == (TerminalStatus.HOLD, WakeHold.NOT_SUPERSEDABLE)
    unknown = fx.terminal(state, old, TerminalReason.SUPERSEDED, "wakeup-" + "7" * 64)
    assert unknown.hold is WakeHold.NOT_SUPERSEDABLE
    assert prepares.count == 0 and _count(state) == rows
    monkeypatch.setattr(ledger, "may_supersede", lambda old_wake, new_wake: True)
    done = fx.terminal(state, old, TerminalReason.SUPERSEDED, newer.wakeup_id)
    assert done.status is TerminalStatus.COMMITTED
    assert (done.terminal.terminal, done.terminal.superseding_wakeup_id) == (
        WakeTerminalKind.CANCELLED,
        newer.wakeup_id,
    )
    assert _entry(world, old).state is WakeState.CANCELLED
    assert _entry(world, newer).state is WakeState.DUE
    _no_failed_events(state)


def test_a_claimed_wake_is_not_superseded(world) -> None:  # type: ignore[no-untyped-def]
    state = world.state
    old = fx.materialize(state, world.gid, world.time)
    newer = _second_wake(world)
    fx.claim(state, old)
    world.monkeypatch.setattr(ledger, "may_supersede", lambda old_wake, new_wake: True)
    held = fx.terminal(state, old, TerminalReason.SUPERSEDED, newer.wakeup_id)
    assert (held.status, held.hold) == (TerminalStatus.HOLD, WakeHold.NOT_SUPERSEDABLE)


# ------------------------------------------------------------ owner plumbing


def test_a_failed_handler_leaves_no_pending_state_behind(world) -> None:  # type: ignore[no-untyped-def]
    """``_clear`` wiring: a failure mid-handler never makes the next event see
    "already pending" (the server calls ``_clear`` after every event)."""

    state, clock, monkeypatch = world.state, world.time, world.monkeypatch
    wake = fx.materialize(state, world.gid, clock)
    owner = state.wake_claim_owner
    real = WakeClaimOwnerV1._prepare

    def failing(self, event, incarnation, operation, result, **kwargs):  # type: ignore[no-untyped-def]
        real(self, event, incarnation, operation, result, **kwargs)  # sets _pending
        raise RuntimeError("handler failed after the pending state was set")

    with monkeypatch.context() as patch:
        patch.setattr(WakeClaimOwnerV1, "_prepare", failing)
        with pytest.raises(AgentRuntimeExecutionError):
            owner._submit_wake_claim(wake.wakeup_id).result(timeout=30)
    assert owner._pending is None and owner._inflight is None


def test_two_events_in_a_row_see_no_leftover_pending(world) -> None:  # type: ignore[no-untyped-def]
    state, clock = world.state, world.time
    wake = fx.materialize(state, world.gid, clock)
    first = fx.claim(state, wake)
    assert first.status is ClaimStatus.COMMITTED
    assert state.wake_claim_owner._pending is None
    held = fx.claim(state, wake)
    assert held.hold is WakeHold.LEASE_LIVE
    result = fx.review(state, first.claim.claim_id)
    assert result.status is ReviewStatus.DECISION_ISSUED
    assert state.wake_review_owner._pending is None
    assert fx.review(state, first.claim.claim_id).status is ReviewStatus.EXISTING


def test_validate_commit_rejects_an_unbound_or_stale_pending(world) -> None:  # type: ignore[no-untyped-def]
    state = world.state
    owner = state.wake_claim_owner
    other = SimpleNamespace(event_type=review_owner.AgentEventType.PLAN_PREVIEW_REVIEW)
    claim_event = SimpleNamespace(event_type=review_owner.AgentEventType.R19_WAKE_CLAIM)
    owner._validate_commit(other)  # nothing pending: fine
    owner._validate_commit(claim_event)
    owner._pending = claim_owner._Pending(  # type: ignore[assignment]
        SimpleNamespace(event_type=claim_event.event_type, mark=1), "i", "h", (0, ""), "t", None  # type: ignore[arg-type]
    )
    with pytest.raises(WakeCommitRefused, match="stale"):
        owner._validate_commit(other)
    with pytest.raises(WakeCommitRefused, match="unbound"):
        owner._validate_commit(claim_event)
    # Bound to its event but evaluated in another incarnation: a fail-stop.
    owner._pending = claim_owner._Pending(claim_event, "i", "h", (0, ""), "t", None)  # type: ignore[arg-type]
    with pytest.raises(WakeCommitRefused, match="incarnation"):
        owner._validate_commit(claim_event)  # type: ignore[arg-type]
    owner._pending = None


def test_the_wake_path_has_no_model_and_never_asks_the_runtime_if_it_is_idle() -> None:
    """HP-5's "2 model calls" is 0 on this path, and the claim -> review chain is
    not gated on ``AgentRuntime.is_idle()`` (only the Scheduler's claim is)."""

    root = Path(__file__).resolve().parents[1] / "suzka" / "r19"
    for name in ("claim_owner.py", "review_owner.py", "ledger.py"):
        source = (root / name).read_text()
        tree = ast.parse(source)
        imported = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        } | {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        assert not [module for module in imported if module.startswith("suzka.models")], name
        assert "is_idle" not in source and "pending_count" not in source, name


def test_a_hold_after_the_pending_state_was_set_leaves_none_behind(world) -> None:  # type: ignore[no-untyped-def]
    """A caught HOLD must never leave ``_pending`` set: ``_validate_commit`` would
    find no PREPARED row and fail-stop the runtime for what is a no-row HOLD."""

    state, clock, monkeypatch = world.state, world.time, world.monkeypatch
    wake = fx.materialize(state, world.gid, clock)
    from suzka.r16.history_store import HistoryStoreError
    from suzka.r19 import claim_owner as module

    def broken(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise HistoryStoreError("store gone")

    with monkeypatch.context() as patch:
        patch.setattr(module, "SubjectHistoryRecordParticipant", broken)
        held = fx.claim(state, wake)
    assert (held.status, held.hold) == (ClaimStatus.HOLD, WakeHold.HISTORY_UNAVAILABLE)
    assert state.wake_claim_owner._pending is None
    assert fx.claim(state, wake).status is ClaimStatus.COMMITTED
    _no_failed_events(state)
