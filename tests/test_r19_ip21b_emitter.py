"""IP-21b: the Outbox emitter (``R19_WAKE_COMPLETE``) and the Scheduler's
``_next_complete`` step (IP-21 D1 r3 §3, §4-4, §4-5, §8, §9, §10-1, §10-3).

Real app, real R07 runtime, real owners.  The Wake is written by
``r19_wake_fixtures.materialize``; claim, review, terminal and the new emission
are the real owners.  Time is a forward-only fake (``fx.Time``).  The 21a
carry-overs are here: a closure built from a REAL ``DECISION_ISSUED``
``CommittedDecisionV1`` (not a stubbed ``policy_row``) and the handler-side,
Journal-derived ``wake_clock_floor``.
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
from suzka.r16.cycle_contracts import HoldReason, ResultKind
from suzka.r16.history_codec import CommittedDecisionV1
from suzka.r16.history_participant import SubjectHistoryRecordParticipant
from suzka.r16.history_reader import HeadBlockReason, ReaderTransient, SubjectHistoryReader
from suzka.r16.history_records import HistoryOperationKind as K
from suzka.r16.history_store import HistoryPhase, HistoryStoreCapacityError, PrepareCollisions
from suzka.r19 import inbox, ledger, outbox_owner, scheduler as scheduler_module
from suzka.r19.claim_owner import WakeCommitRefused
from suzka.r19.contracts import (
    MAX_MESSAGE_TTL,
    TERMINAL_ID_PREFIX,
    OutboxMessageV1,
    OutboxReason,
    OutboxTemplate,
    TerminalReason,
    WakeTerminalV1,
    outbox_reason_for,
)
from suzka.r19.inbox import OutboxHold, emission_candidates, plan_emission
from suzka.r19.ledger import WakeState
from suzka.r19.outbox_owner import OutboxOwnerV1, OutboxStatus
from suzka.r19.scheduler import TickAction, WakeScheduler
from suzka.r19.raw_state import wake_clock_floor
from suzka.runtime.agent_runtime import AgentEventSource, AgentEventType, AgentRuntimeExecutionError
from suzka.runtime.event_journal import EventLifecycle
from test_r16_u2a_source_owner import _app
from test_r16_u3a_plan_preview import _qualified_goal

UTC = timezone.utc
STAMP = fx.STAMP


# ----------------------------------------------------------------- the world


@pytest.fixture
def w(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    app = _app(fx.settings(tmp_path / "w"))
    app.state.r19_scheduler_enabled = False  # the tests tick by hand
    with TestClient(app) as client:
        state = client.app.state
        clock = fx.Time(state, monkeypatch)
        state.wake_outbox_owner._clock = clock.live
        x, gid = _qualified_goal(state)
        yield SimpleNamespace(
            state=state,
            time=clock,
            gid=gid,
            x=x,
            monkeypatch=monkeypatch,
            mono=lambda: 1_000.0 + clock.offset.total_seconds(),
        )


class Prepares:
    """Counts ``participant.prepare`` (a prepare that raises fail-stops the runtime)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.count = 0
        real = SubjectHistoryRecordParticipant.prepare
        outer = self

        def spy(self_, binding):  # type: ignore[no-untyped-def]
            outer.count += 1
            return real(self_, binding)

        monkeypatch.setattr(SubjectHistoryRecordParticipant, "prepare", spy)


def no_failed_events(state) -> None:  # type: ignore[no-untyped-def]
    assert not [
        record
        for record in state.event_journal.inspect().records
        if record.lifecycle is EventLifecycle.FAILED
    ]


def outbox_rows(state):  # type: ignore[no-untyped-def]
    return fx.operations(state, K.R19_OUTBOX_EMIT)


def messages(state):  # type: ignore[no-untyped-def]
    return fx.adopted(state, OutboxMessageV1)


def adopted_records(state):  # type: ignore[no-untyped-def]
    found = SubjectHistoryReader(state.event_journal, state.subject_history_store).adopted_records()
    assert found.blocked is None
    return found.records


def count(state) -> int:  # type: ignore[no-untyped-def]
    return len(fx.operations(state))


def entry_of(w, wake):  # type: ignore[no-untyped-def]
    return fx.entry(w.state, wake, w.time.now())


def emit(w, wake_or_id):  # type: ignore[no-untyped-def]
    wakeup_id = wake_or_id if isinstance(wake_or_id, str) else wake_or_id.wakeup_id
    return w.state.wake_outbox_owner._submit_complete(wakeup_id).result(timeout=30).value


def closed(w, **changes):  # type: ignore[no-untyped-def]
    """A Wake closed by a real review (a real ``CommittedDecisionV1``)."""

    wake = fx.materialize(w.state, w.gid, w.time, **changes)
    claimed = fx.claim(w.state, wake)
    result = fx.review(w.state, claimed.claim.claim_id)
    return wake, claimed, result


def expired(w, **changes):  # type: ignore[no-untyped-def]
    """A FAILED Wake: an expired Wake closed by its EXPIRED_WAKE terminal."""

    wake = fx.materialize(w.state, w.gid, w.time, expires=-5, not_before=-4000, **changes)
    done = fx.terminal(w.state, wake, TerminalReason.EXPIRED_WAKE)
    assert done.terminal is not None
    return wake, done.terminal


def make(w, *, scanned: bool = True, **kwargs):  # type: ignore[no-untyped-def]
    state = w.state
    kwargs.setdefault("driver_active", lambda _inspection: True)
    kwargs.setdefault("outbox_owner", state.wake_outbox_owner)
    scheduler = WakeScheduler(
        state.agent_runtime,
        state.event_journal,
        state.wake_materializer,
        state.wake_claim_owner,
        state.wake_review_owner,
        timing=state.wake_claim_owner.timing,
        rescan_s=1800,
        clock=w.time.now,
        monotonic=w.mono,
        **kwargs,
    )
    if scanned:
        scheduler._scan_served = scheduler._scan_requested
    return scheduler


def settle(w, scheduler) -> None:  # type: ignore[no-untyped-def]
    import time as real_time

    assert scheduler.wait_idle(30)
    deadline = real_time.monotonic() + 10
    while not w.state.agent_runtime.is_idle():
        assert real_time.monotonic() < deadline
        real_time.sleep(0.005)


def step(w, scheduler):  # type: ignore[no-untyped-def]
    outcome = scheduler.tick()
    settle(w, scheduler)
    return outcome


def complete_events(state, lifecycle=EventLifecycle.STARTED):  # type: ignore[no-untyped-def]
    return [
        record
        for record in state.event_journal.inspect().records
        if record.event_type is AgentEventType.R19_WAKE_COMPLETE and record.lifecycle is lifecycle
    ]


# --------------------------------------------------------------- R19 disabled


def test_with_r19_off_no_owner_exists_and_nothing_submits_a_completion(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with TestClient(_app(fx.settings(tmp_path / "off", r19=False))) as client:
        state = client.app.state
        assert state.wake_outbox_owner is None
        assert state.wake_scheduler is None
        assert not state.subject_history_store.has_operations()
        assert not complete_events(state, EventLifecycle.ACCEPTED)


def test_with_r19_on_the_owner_is_wired_into_the_scheduler(tmp_path) -> None:  # type: ignore[no-untyped-def]
    app = _app(fx.settings(tmp_path / "on"))
    with TestClient(app) as client:
        state = client.app.state
        owner = state.wake_outbox_owner
        assert isinstance(owner, OutboxOwnerV1) and owner.enabled
        assert owner.outbox_ttl_s == app.state.settings.r19.outbox_ttl_s
        assert state.wake_scheduler._outbox is owner


def test_a_disabled_owner_holds_before_reading_anything(w) -> None:  # type: ignore[no-untyped-def]
    state = w.state
    owner = OutboxOwnerV1(
        state.decision_issuer,
        state.agent_runtime,
        enabled=False,
        timing=state.wake_claim_owner.timing,
        outbox_ttl_s=3600,
    )

    def unreadable(*args):  # type: ignore[no-untyped-def]
        raise AssertionError("a disabled owner read something")

    owner._classify = unreadable  # type: ignore[method-assign]
    owner._clock = unreadable  # type: ignore[method-assign]
    held = owner._submit_complete("wakeup-" + "0" * 64).result(timeout=30).value
    assert (held.status, held.hold) == (OutboxStatus.HOLD, OutboxHold.R19_DISABLED)
    assert not state.subject_history_store.has_operations()
    no_failed_events(state)


def test_the_store_switch_is_checked_even_if_the_owner_is_enabled(w) -> None:  # type: ignore[no-untyped-def]
    state, monkeypatch = w.state, w.monkeypatch
    wake, _claimed, _result = closed(w)
    prepares = Prepares(monkeypatch)
    monkeypatch.setattr(state.subject_history_store, "_r19_enabled", False)
    held = emit(w, wake)
    assert (held.status, held.hold) == (OutboxStatus.HOLD, OutboxHold.R19_DISABLED)
    assert prepares.count == 0 and outbox_rows(state) == []
    no_failed_events(state)


# ----------------------------------------- a real DECISION_ISSUED closure (21a carry-over)


def test_a_real_issued_decision_closure_gives_its_own_policy_row_code(w) -> None:  # type: ignore[no-untyped-def]
    """No stubbed ``policy_row``: the closure comes from a real review, the code
    from ``closure.result.policy_row`` of the issued ``CommittedDecisionV1``."""

    state = w.state
    wake, _claimed, result = closed(w)
    assert result.decision_id is not None
    entry = entry_of(w, wake)
    closure = entry.closure
    assert entry.state is WakeState.CLOSED and closure is not None
    assert isinstance(closure.result, CommittedDecisionV1)
    assert closure.resolution.result_kind is ResultKind.DECISION_ISSUED
    row = closure.result.policy_row
    assert inbox.decision_row_of(closure) is row
    expected = outbox_reason_for(ResultKind.DECISION_ISSUED, row)
    emitted = emit(w, wake)
    assert emitted.status is OutboxStatus.COMMITTED
    (message,) = messages(state)
    assert message.message_id == emitted.message_id
    assert message.template is OutboxTemplate.DECISION_ISSUED
    assert message.reason_code is expected
    assert expected in (OutboxReason.PLAN_SELECTED, OutboxReason.NO_ACTION)
    # The message reports this cycle and this result only.
    assert message.source_id == wake.logical_cycle_id
    assert message.result_digest == closure.resolution.result_digest
    assert (message.incarnation_id, message.context_id) == (wake.incarnation_id, wake.context_id)
    # The Scheduler's pre-filter plan (record bytes only) builds the same message
    # (its id and code) the handler wrote.
    plan = plan_emission(entry, w.time.stamp(), outbox_ttl_s=state.wake_outbox_owner.outbox_ttl_s)
    assert plan.hold is OutboxHold.NONE_LEFT or plan.message_id == message.message_id
    assert inbox.plan_emission(
        entry, w.time.stamp(), outbox_ttl_s=3600
    ).message_id == message.message_id
    no_failed_events(state)


def test_the_message_context_equals_the_wakes_context(w) -> None:  # type: ignore[no-untyped-def]
    wake, _terminal = expired(w)
    other, _c, _r = closed(w, cause_event_id=str(uuid4()), evidence_fingerprint="d" * 64)
    assert emit(w, wake).status is OutboxStatus.COMMITTED
    assert emit(w, other).status is OutboxStatus.COMMITTED
    by_source = {m.source_id: m for m in messages(w.state)}
    assert by_source[other.logical_cycle_id].context_id == other.context_id
    assert all(m.context_id == wake.context_id for m in by_source.values())


# ----------------------------------------------------- the clock (21a carry-over)


def test_the_handler_derives_the_journal_clock_floor_the_pre_filter_cannot(w) -> None:  # type: ignore[no-untyped-def]
    """The Scheduler's view has no floor (no Journal read); the handler reads the
    Journal high-water (the real ``ACCEPTED`` times, which the fake clock does not
    move).  A clock rolled back across a restart, to an instant past the claim
    bound but below a later Journal record, is a handler-side
    ``CLOCK_UNVERIFIABLE`` for a plan the pre-filter calls ready."""

    import time as real_time

    state, clock = w.state, w.time
    wake, _claimed, _result = closed(w)
    entry = entry_of(w, wake)
    real_time.sleep(4)
    # A later Journal record (any event): the high-water moves past the closure.
    state.agent_runtime.submit(
        AgentEventType.EMOTION_TICK, AgentEventSource.RUNTIME_EMOTION_TIMER, lambda: None
    ).result(timeout=30)
    state.agent_runtime._last_admitted_at = None  # a restart forgets the in-memory high-water
    clock.advance(-2.5)
    # Pre-filter: no floor, T is past L (the claim bound) -> ready.
    view = fx.view(state, clock.now())
    (plan,) = emission_candidates(view, inbox.adopted_message_ids(adopted_records(state)), outbox_ttl_s=3600)
    assert plan.ready
    prepares = Prepares(w.monkeypatch)
    rows = count(state)
    held = emit(w, wake)
    assert (held.status, held.hold) == (OutboxStatus.HOLD, OutboxHold.CLOCK_UNVERIFIABLE)
    assert prepares.count == 0 and count(state) == rows
    # The floor the handler used is the module's own (Journal high-water, then claim bound).
    floor = wake_clock_floor(
        state.event_journal.inspect().records,
        entry.latest_claim,
        claim_lease_s=state.wake_claim_owner.timing.claim_lease_s,
    )
    assert floor is not None and floor.replace(microsecond=0) > clock.now().replace(microsecond=0)
    clock.advance(2.5)
    assert emit(w, wake).status is OutboxStatus.COMMITTED
    no_failed_events(state)


def test_a_clock_below_the_claim_lower_bound_is_unverifiable_without_the_journal(w) -> None:  # type: ignore[no-untyped-def]
    state, clock = w.state, w.time
    wake, _claimed, _result = closed(w)
    state.agent_runtime._last_admitted_at = None
    clock.advance(-3 * 3600)
    held = emit(w, wake)
    assert (held.status, held.hold) == (OutboxStatus.HOLD, OutboxHold.CLOCK_UNVERIFIABLE)
    assert outbox_rows(state) == []


def test_ttl_is_the_earlier_of_emission_plus_ttl_and_the_result_bound(w) -> None:  # type: ignore[no-untyped-def]
    state, clock = w.state, w.time
    wake, _claimed, _result = closed(w)
    entry = entry_of(w, wake)
    lower = inbox.closing_lower_bound(entry)
    assert lower is not None
    owner = state.wake_outbox_owner
    owner._outbox_ttl_s = 3600
    emitted = emit(w, wake)
    assert emitted.status is OutboxStatus.COMMITTED
    (message,) = messages(state)
    created = datetime.strptime(message.created_utc, STAMP).replace(tzinfo=UTC)
    expires = datetime.strptime(message.expires_utc, STAMP).replace(tzinfo=UTC)
    assert expires == min(created + timedelta(seconds=3600), lower + MAX_MESSAGE_TTL)
    assert expires - created == timedelta(seconds=3600)
    # Default TTL: the bound from the result wins and never exceeds 7 days from it.
    other, _c, _r = closed(w, cause_event_id=str(uuid4()), evidence_fingerprint="c" * 64)
    owner._outbox_ttl_s = int(MAX_MESSAGE_TTL.total_seconds())
    clock.advance(600)
    assert emit(w, other).status is OutboxStatus.COMMITTED
    late = next(m for m in messages(state) if m.source_id == other.logical_cycle_id)
    other_lower = inbox.closing_lower_bound(entry_of(w, other))
    assert other_lower is not None
    assert datetime.strptime(late.expires_utc, STAMP).replace(tzinfo=UTC) == other_lower + MAX_MESSAGE_TTL
    created_late = datetime.strptime(late.created_utc, STAMP).replace(tzinfo=UTC)
    assert datetime.strptime(late.expires_utc, STAMP).replace(tzinfo=UTC) < created_late + MAX_MESSAGE_TTL


def test_a_result_older_than_seven_days_is_never_notified(w) -> None:  # type: ignore[no-untyped-def]
    state, clock = w.state, w.time
    wake, _claimed, _result = closed(w)
    clock.advance(7 * 86_400 + 600)
    prepares = Prepares(w.monkeypatch)
    rows = count(state)
    held = emit(w, wake)
    assert (held.status, held.hold) == (OutboxStatus.HOLD, OutboxHold.TOO_OLD)
    assert prepares.count == 0 and count(state) == rows
    # The pre-filter agrees (no Journal read): not ready, so it is never submitted.
    view = fx.view(state, clock.now())
    (plan,) = emission_candidates(view, frozenset(), outbox_ttl_s=3600)
    assert plan.hold is OutboxHold.TOO_OLD and not plan.ready
    no_failed_events(state)


def test_an_expired_wake_failure_is_anchored_at_the_wakes_expiry(w) -> None:  # type: ignore[no-untyped-def]
    state = w.state
    wake, terminal = expired(w)
    entry = entry_of(w, wake)
    assert entry.state is WakeState.FAILED
    assert inbox.closing_lower_bound(entry) == datetime.strptime(entry.expiry_utc, STAMP).replace(tzinfo=UTC)
    emitted = emit(w, wake)
    assert emitted.status is OutboxStatus.COMMITTED
    (message,) = messages(state)
    assert message.template is OutboxTemplate.CYCLE_FAILED
    assert message.reason_code is OutboxReason.FAILED
    assert message.source_id == terminal.terminal_id and message.result_digest == terminal.terminal_digest
    assert message.source_id.startswith(TERMINAL_ID_PREFIX)
    assert fx.adopted(state, WakeTerminalV1) == [terminal]


def test_an_exhausted_wake_failure_is_anchored_at_its_last_claim(w) -> None:  # type: ignore[no-untyped-def]
    state, clock, monkeypatch = w.state, w.time, w.monkeypatch

    def unavailable(*args):  # type: ignore[no-untyped-def]
        raise RuntimeError("R15 is down")

    monkeypatch.setattr(state.decision_issuer, "_interpretation", unavailable)
    wake = fx.materialize(state, w.gid, clock)
    for _ in range(3):
        claimed = fx.claim(state, wake)
        assert claimed.claim is not None
        fx.review(state, claimed.claim.claim_id)
        clock.advance(121)
    done = fx.terminal(state, wake, TerminalReason.ATTEMPTS_EXHAUSTED)
    assert done.terminal is not None
    entry = entry_of(w, wake)
    assert entry.state is WakeState.FAILED
    lease = datetime.strptime(entry.latest_claim.claim_lease_expires_utc, STAMP).replace(tzinfo=UTC)
    assert inbox.closing_lower_bound(entry) == lease - timedelta(seconds=120)
    assert emit(w, wake).status is OutboxStatus.COMMITTED
    (message,) = messages(state)
    assert message.template is OutboxTemplate.CYCLE_FAILED


# ----------------------------------------------------- the other result kinds


def test_hold_and_existing_results_have_their_own_templates(w) -> None:  # type: ignore[no-untyped-def]
    """DECISION_CONFIRMED (an EXISTING receipt) and DECISION_HELD (a HOLD receipt)
    each get their own template and code, whatever the HOLD reason.  (That an
    EXISTING message is never absorbed by an ISSUED one of the same Goal is the
    IP-13b dedupe-key test; a second real Wake of the Goal cannot be fixtured.)"""

    from suzka.r19.review_owner import ReviewStatus
    from test_r16_ip10_decision_issuer import _issue

    state = w.state
    confirmed_wake = fx.materialize(state, w.gid, w.time)
    held_wake = fx.materialize(
        state, w.gid, w.time, cause_event_id=str(uuid4()), evidence_fingerprint="9" * 64
    )
    _issue(state, w.gid)  # the head the first Wake's evidence already holds
    claimed = fx.claim(state, confirmed_wake)
    assert fx.review(state, claimed.claim.claim_id).status is ReviewStatus.EXISTING_DECISION
    claimed = fx.claim(state, held_wake)
    result = fx.review(state, claimed.claim.claim_id)
    assert (result.status, result.hold_reason) == (ReviewStatus.HOLD_RECEIPT, HoldReason.STALE_SOURCE)
    assert emit(w, confirmed_wake).status is OutboxStatus.COMMITTED
    assert emit(w, held_wake).status is OutboxStatus.COMMITTED
    by_source = {m.source_id: m for m in messages(state)}
    confirmed = by_source[confirmed_wake.logical_cycle_id]
    held = by_source[held_wake.logical_cycle_id]
    assert (confirmed.template, confirmed.reason_code) == (
        OutboxTemplate.DECISION_CONFIRMED,
        OutboxReason.CONFIRMED,
    )
    assert (held.template, held.reason_code) == (OutboxTemplate.DECISION_HELD, OutboxReason.HELD)
    assert confirmed.message_id != held.message_id
    assert len(outbox_rows(state)) == 2
    no_failed_events(state)


# ------------------------------------------------------- the handler order


def test_one_message_per_closed_cycle_a_second_emission_is_existing(w) -> None:  # type: ignore[no-untyped-def]
    state = w.state
    wake, _claimed, _result = closed(w)
    first = emit(w, wake)
    assert first.status is OutboxStatus.COMMITTED
    rows = count(state)
    prepares = Prepares(w.monkeypatch)
    second = emit(w, wake)
    assert (second.status, second.message_id) == (OutboxStatus.EXISTING, first.message_id)
    assert second.hold is None
    assert prepares.count == 0 and count(state) == rows
    assert len(outbox_rows(state)) == 1 and len(messages(state)) == 1
    assert state.wake_outbox_owner._pending is None
    no_failed_events(state)


def test_dedupe_comes_first_even_for_a_message_that_has_since_aged_out(w) -> None:  # type: ignore[no-untyped-def]
    wake, _claimed, _result = closed(w)
    first = emit(w, wake)
    assert first.status is OutboxStatus.COMMITTED
    w.time.advance(8 * 86_400)
    # TOO_OLD for a fresh emission, but this one exists: EXISTING, never re-emittable.
    again = emit(w, wake)
    assert (again.status, again.message_id) == (OutboxStatus.EXISTING, first.message_id)
    assert len(outbox_rows(w.state)) == 1


def test_unknown_unsettled_cancelled_and_corrupt_wakes_have_no_message(w) -> None:  # type: ignore[no-untyped-def]
    state, monkeypatch = w.state, w.monkeypatch
    unknown = emit(w, "wakeup-" + "7" * 64)
    assert (unknown.status, unknown.hold) == (OutboxStatus.HOLD, OutboxHold.WAKE_UNKNOWN)
    open_wake = fx.materialize(state, w.gid, w.time)
    assert emit(w, open_wake).hold is OutboxHold.NOT_SETTLED
    newer = fx.materialize(state, w.gid, w.time, cause_event_id=str(uuid4()), evidence_fingerprint="e" * 64)
    monkeypatch.setattr(ledger, "may_supersede", lambda old, new: True)
    done = fx.terminal(state, open_wake, TerminalReason.SUPERSEDED, newer.wakeup_id)
    assert done.terminal is not None
    prepares = Prepares(monkeypatch)
    rows = count(state)
    assert entry_of(w, open_wake).state is WakeState.CANCELLED
    assert emit(w, open_wake).hold is OutboxHold.NO_MESSAGE
    real = outbox_owner.derive_wake_view

    def corrupt(records, now, *, clock_floor):  # type: ignore[no-untyped-def]
        view = real(records, now, clock_floor=clock_floor)
        return replace(
            view,
            entries=tuple(replace(e, state=WakeState.CORRUPT, corrupt_reason="TEST") for e in view.entries),
        )

    with monkeypatch.context() as patch:
        patch.setattr(outbox_owner, "derive_wake_view", corrupt)
        assert emit(w, newer).hold is OutboxHold.WAKE_CORRUPT
    assert prepares.count == 0 and count(state) == rows and outbox_rows(state) == []
    no_failed_events(state)


def test_a_live_raw_key_is_in_flight_before_any_clock_check(w) -> None:  # type: ignore[no-untyped-def]
    state, monkeypatch = w.state, w.monkeypatch
    wake, _claimed, _result = closed(w)
    owner = state.wake_outbox_owner
    plan = plan_emission(entry_of(w, wake), w.time.stamp(), outbox_ttl_s=3600)
    assert plan.message_id is not None
    real = owner._live_keys
    monkeypatch.setattr(
        owner,
        "_live_keys",
        lambda *keys: frozenset(keys) if keys == (f"outbox:{plan.message_id}",) else real(*keys),
    )
    prepares = Prepares(monkeypatch)
    held = emit(w, wake)
    assert (held.status, held.hold) == (OutboxStatus.HOLD, OutboxHold.IN_FLIGHT)
    assert held.message_id == plan.message_id
    # Even a message that would be TOO_OLD is in flight first (D1 step 4 before 5).
    w.time.advance(8 * 86_400)
    assert emit(w, wake).hold is OutboxHold.IN_FLIGHT
    assert prepares.count == 0 and outbox_rows(state) == []


def test_every_history_failure_is_a_no_row_hold(w) -> None:  # type: ignore[no-untyped-def]
    state, monkeypatch = w.state, w.monkeypatch
    wake, _claimed, _result = closed(w)
    owner = state.wake_outbox_owner
    prepares = Prepares(monkeypatch)
    rows = count(state)
    with monkeypatch.context() as patch:
        def transient(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise ReaderTransient(HeadBlockReason.STORE_UNAVAILABLE)

        patch.setattr(SubjectHistoryReader, "classify", transient)
        assert emit(w, wake).hold is OutboxHold.HISTORY_UNAVAILABLE
    with monkeypatch.context() as patch:
        def full(length):  # type: ignore[no-untyped-def]
            raise HistoryStoreCapacityError("full")

        patch.setattr(state.subject_history_store, "check_headroom", full)
        assert emit(w, wake).hold is OutboxHold.HISTORY_CAPACITY
    with monkeypatch.context() as patch:
        patch.setattr(
            state.subject_history_store,
            "prepare_collisions",
            lambda operation: PrepareCollisions(frozenset(), True),
        )
        assert emit(w, wake).hold is OutboxHold.COLLISION
    assert owner._pending is None
    assert prepares.count == 0 and count(state) == rows and outbox_rows(state) == []
    assert emit(w, wake).status is OutboxStatus.COMMITTED
    no_failed_events(state)


def test_a_builder_failure_is_a_logged_hold_never_an_event_failure(w) -> None:  # type: ignore[no-untyped-def]
    state, monkeypatch = w.state, w.monkeypatch
    wake, _claimed, _result = closed(w)
    rows = count(state)

    def broken(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise ValueError("a builder refused")

    with monkeypatch.context() as patch:
        patch.setattr(outbox_owner, "HistoryRecordOperationV1", broken)
        held = emit(w, wake)
    assert (held.status, held.hold) == (OutboxStatus.HOLD, OutboxHold.UNREPRESENTABLE)
    with monkeypatch.context() as patch:
        patch.setattr(OutboxMessageV1, "for_completion", classmethod(broken))
        assert emit(w, wake).hold is OutboxHold.UNREPRESENTABLE
    assert count(state) == rows and outbox_rows(state) == []
    no_failed_events(state)


# ------------------------------------------------ mirrors of the SQL trigger


def _plan_with(monkeypatch, mutate):  # type: ignore[no-untyped-def]
    real = outbox_owner.plan_emission

    def planned(entry, now, **kwargs):  # type: ignore[no-untyped-def]
        plan = real(entry, now, **kwargs)
        if plan.message is None:
            return plan
        message = mutate(plan.message, entry)
        return inbox.EmissionPlan(
            plan.wakeup_id, plan.state, message.message_id, plan.lower_bound_utc, message, None
        )

    monkeypatch.setattr(outbox_owner, "plan_emission", planned)


_ISSUED_MIRRORS = {
    "digest": lambda m, e: replace(m, result_digest="a" * 64),
    "source": lambda m, e: replace(m, source_id="b" * 64),
    "template": lambda m, e: replace(m, template=OutboxTemplate.DECISION_HELD, reason_code=OutboxReason.HELD),
    "failed_without_terminal": lambda m, e: replace(
        m,
        template=OutboxTemplate.CYCLE_FAILED,
        reason_code=OutboxReason.FAILED,
        source_id=TERMINAL_ID_PREFIX + "c" * 64,
    ),
    "incarnation": lambda m, e: replace(m, incarnation_id="d" * 64),
    "context": lambda m, e: replace(m, context_id=m.context_id + "x"),
}


@pytest.mark.parametrize("case", sorted(_ISSUED_MIRRORS))
def test_a_violated_trigger_condition_is_a_hold_before_prepare(w, case) -> None:  # type: ignore[no-untyped-def]
    state, monkeypatch = w.state, w.monkeypatch
    wake, _claimed, _result = closed(w)
    prepares = Prepares(monkeypatch)
    rows = count(state)
    _plan_with(monkeypatch, _ISSUED_MIRRORS[case])
    held = emit(w, wake)
    assert (held.status, held.hold) == (OutboxStatus.HOLD, OutboxHold.UNREPRESENTABLE)
    assert prepares.count == 0 and count(state) == rows and outbox_rows(state) == []
    assert state.wake_outbox_owner._pending is None
    no_failed_events(state)


_FAILED_MIRRORS = {
    "digest": lambda m, e: replace(m, result_digest="a" * 64),
    "source": lambda m, e: replace(m, source_id=TERMINAL_ID_PREFIX + "e" * 64),
    "template": lambda m, e: replace(m, template=OutboxTemplate.DECISION_HELD, reason_code=OutboxReason.HELD,
                                     source_id="f" * 64),
}


@pytest.mark.parametrize("case", sorted(_FAILED_MIRRORS))
def test_a_failed_cycle_message_mirrors_the_terminal_condition(w, case) -> None:  # type: ignore[no-untyped-def]
    state, monkeypatch = w.state, w.monkeypatch
    wake, _terminal = expired(w)
    prepares = Prepares(monkeypatch)
    rows = count(state)
    _plan_with(monkeypatch, _FAILED_MIRRORS[case])
    held = emit(w, wake)
    assert (held.status, held.hold) == (OutboxStatus.HOLD, OutboxHold.UNREPRESENTABLE)
    assert prepares.count == 0 and count(state) == rows


def test_the_trigger_itself_refuses_an_unbound_message(w) -> None:  # type: ignore[no-untyped-def]
    """The mirrors above guard what the trigger enforces: a direct FINALIZE of a
    message whose digest names no adopted resolution ABORTs (store level)."""

    state = w.state
    wake, _claimed, _result = closed(w)
    entry = entry_of(w, wake)
    plan = plan_emission(entry, w.time.stamp(), outbox_ttl_s=3600)
    assert plan.message is not None
    bad = replace(plan.message, result_digest="a" * 64)
    with pytest.raises(Exception):
        fx.write_record(
            state,
            K.R19_OUTBOX_EMIT,
            fx.RecordKind.OUTBOX_MESSAGE,
            lambda event: bad,
            AgentEventType.R19_WAKE_COMPLETE,
            fx.AgentEventSource.RUNTIME_R19_WAKE_COMPLETE,
        )
    assert messages(state) == []


# ------------------------------------------------------------ owner plumbing


def test_two_emissions_in_a_row_see_no_leftover_pending(w) -> None:  # type: ignore[no-untyped-def]
    state = w.state
    first, _c, _r = closed(w)
    second, _c2, _r2 = closed(w, cause_event_id=str(uuid4()), evidence_fingerprint="b" * 64)
    owner = state.wake_outbox_owner
    assert emit(w, first).status is OutboxStatus.COMMITTED
    assert owner._pending is None and owner._inflight is None
    assert emit(w, second).status is OutboxStatus.COMMITTED
    assert owner._pending is None and owner._inflight is None
    assert len(outbox_rows(state)) == 2
    assert all(row.phase is HistoryPhase.FINALIZED for row in outbox_rows(state))
    no_failed_events(state)


def test_a_failed_handler_leaves_no_pending_state_behind(w) -> None:  # type: ignore[no-untyped-def]
    state, monkeypatch = w.state, w.monkeypatch
    wake, _claimed, _result = closed(w)
    owner = state.wake_outbox_owner
    real = OutboxOwnerV1._prepare

    def failing(self, event, incarnation, operation, result, **kwargs):  # type: ignore[no-untyped-def]
        real(self, event, incarnation, operation, result, **kwargs)  # sets _pending
        raise RuntimeError("handler failed after the pending state was set")

    with monkeypatch.context() as patch:
        patch.setattr(OutboxOwnerV1, "_prepare", failing)
        with pytest.raises(AgentRuntimeExecutionError):
            owner._submit_complete(wake.wakeup_id).result(timeout=30)
    assert owner._pending is None and owner._inflight is None


def test_a_hold_after_the_pending_state_was_set_leaves_none_behind(w) -> None:  # type: ignore[no-untyped-def]
    from suzka.r16.history_store import HistoryStoreError

    state, monkeypatch = w.state, w.monkeypatch
    wake, _claimed, _result = closed(w)
    from suzka.r19 import claim_owner as module

    def broken(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise HistoryStoreError("store gone")

    with monkeypatch.context() as patch:
        patch.setattr(module, "SubjectHistoryRecordParticipant", broken)
        held = emit(w, wake)
    assert (held.status, held.hold) == (OutboxStatus.HOLD, OutboxHold.HISTORY_UNAVAILABLE)
    assert state.wake_outbox_owner._pending is None
    assert emit(w, wake).status is OutboxStatus.COMMITTED


def test_validate_commit_rejects_an_unbound_or_stale_pending(w) -> None:  # type: ignore[no-untyped-def]
    from suzka.r19 import claim_owner

    owner = w.state.wake_outbox_owner
    other = SimpleNamespace(event_type=AgentEventType.PLAN_PREVIEW_REVIEW)
    mine = SimpleNamespace(event_type=AgentEventType.R19_WAKE_COMPLETE)
    owner._validate_commit(other)
    owner._validate_commit(mine)
    owner._pending = claim_owner._Pending(  # type: ignore[assignment]
        SimpleNamespace(event_type=mine.event_type, mark=1), "i", "h", (0, ""), "t", None  # type: ignore[arg-type]
    )
    with pytest.raises(WakeCommitRefused, match="stale"):
        owner._validate_commit(other)
    with pytest.raises(WakeCommitRefused, match="unbound"):
        owner._validate_commit(mine)
    owner._pending = claim_owner._Pending(mine, "i", "h", (0, ""), "t", None)  # type: ignore[arg-type]
    with pytest.raises(WakeCommitRefused, match="incarnation"):
        owner._validate_commit(mine)  # type: ignore[arg-type]
    owner._pending = None


def test_the_owner_has_no_model_and_the_scheduler_has_no_outbox_authority() -> None:
    root = Path(outbox_owner.__file__).resolve().parent
    for name, forbidden in (("outbox_owner.py", ("suzka.models",)), ("scheduler.py", ("suzka.models",))):
        tree = ast.parse((root / name).read_text())
        imported = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        } | {a.name for node in ast.walk(tree) if isinstance(node, ast.Import) for a in node.names}
        assert not [m for m in imported if m.startswith(forbidden)], name
    source = (root / "outbox_owner.py").read_text()
    assert "is_idle" not in source and "_pending = self._decision" not in source
    # Only the owner builds a record operation of the Outbox kind.
    assert "R19_OUTBOX_EMIT" not in (root / "scheduler.py").read_text()


# ----------------------------------------------------- the Scheduler step


def test_the_scheduler_completes_a_closed_cycle_once_and_idles_afterwards(w) -> None:  # type: ignore[no-untyped-def]
    state, clock = w.state, w.time
    wake, _claimed, _result = closed(w)
    scheduler = make(w)
    first = step(w, scheduler)
    assert (first.action, first.target) == (TickAction.COMPLETE, wake.wakeup_id)
    assert first.submitted
    assert len(messages(state)) == 1 and len(complete_events(state)) == 1
    # Idle: no periodic event, however many ticks.
    for _ in range(12):
        clock.advance(300)
        assert step(w, scheduler).action is TickAction.NOTHING_DUE
    assert len(complete_events(state, EventLifecycle.ACCEPTED)) == 1
    assert len(outbox_rows(state)) == 1
    # A fresh Scheduler (a restart) finds the message adopted and submits nothing.
    restarted = make(w)
    assert step(w, restarted).action is TickAction.NOTHING_DUE
    assert len(complete_events(state, EventLifecycle.ACCEPTED)) == 1
    no_failed_events(state)


def test_the_prefilter_checks_adoption_first_and_reads_no_journal(w) -> None:  # type: ignore[no-untyped-def]
    state, monkeypatch = w.state, w.monkeypatch
    closed(w)
    scheduler = make(w)
    assert step(w, scheduler).action is TickAction.COMPLETE
    seen: list[object] = []
    real = inbox.plan_emission

    def spy(entry, now, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(kwargs)
        return real(entry, now, **kwargs)

    monkeypatch.setattr(inbox, "plan_emission", spy)
    reads: list[object] = []
    real_floor = scheduler_module.wake_clock_floor
    monkeypatch.setattr(
        scheduler_module,
        "wake_clock_floor",
        lambda *a, **k: reads.append(1) or real_floor(*a, **k),
    )
    # Adopted already: the candidate is skipped (derived-only), nothing is read for it.
    assert step(w, scheduler).action is TickAction.NOTHING_DUE
    assert all(not kwargs.get("clock_floor") for kwargs in seen)
    assert reads == []
    assert emission_candidates(
        fx.view(state, w.time.now()),
        inbox.adopted_message_ids(adopted_records(state)),
        outbox_ttl_s=3600,
    ) == ()


def test_step_order_terminal_review_complete_scan_claim(w) -> None:  # type: ignore[no-untyped-def]
    state, clock = w.state, w.time
    scheduler = make(w, scanned=False)
    owed, _c, _r = closed(w)
    expiring = fx.materialize(
        state, w.gid, clock, expires=-5, not_before=-4000,
        cause_event_id=str(uuid4()), evidence_fingerprint="a" * 64,
    )
    live = fx.materialize(
        state, w.gid, clock, cause_event_id=str(uuid4()), evidence_fingerprint="b" * 64
    )
    fx.claim(state, live)
    due = fx.materialize(
        state, w.gid, clock, cause_event_id=str(uuid4()), evidence_fingerprint="c" * 64
    )
    order = [step(w, scheduler) for _ in range(5)]
    assert [o.action for o in order[:2]] == [TickAction.TERMINAL, TickAction.REVIEW]
    assert order[0].target == expiring.wakeup_id and order[1].target == live.wakeup_id
    # Then the completions (three closed cycles, entry order), before scan and claim.
    completes = [o for o in order[2:] if o.action is TickAction.COMPLETE]
    assert len(completes) == 3
    assert {o.target for o in completes} == {owed.wakeup_id, expiring.wakeup_id, live.wakeup_id}
    assert [o.action for o in order[2:]] == [TickAction.COMPLETE] * 3
    assert step(w, scheduler).action is TickAction.MATERIALIZE
    scheduler._scan_served = scheduler._scan_requested
    last = step(w, scheduler)
    assert (last.action, last.target) == (TickAction.CLAIM, due.wakeup_id)
    no_failed_events(state)


def test_a_completion_is_not_held_back_for_an_idle_runtime(w) -> None:  # type: ignore[no-untyped-def]
    state = w.state
    closed(w)
    scheduler = make(w)
    w.monkeypatch.setattr(state.agent_runtime, "is_idle", lambda: False)
    assert scheduler.tick().action is TickAction.COMPLETE
    assert scheduler.wait_idle(30)


def test_a_cancelled_wake_gets_no_candidate_and_a_failed_one_gets_cycle_failed(w) -> None:  # type: ignore[no-untyped-def]
    state, monkeypatch = w.state, w.monkeypatch
    old = fx.materialize(state, w.gid, w.time)
    newer = fx.materialize(state, w.gid, w.time, cause_event_id=str(uuid4()), evidence_fingerprint="e" * 64)
    monkeypatch.setattr(ledger, "may_supersede", lambda a, b: True)
    assert fx.terminal(state, old, TerminalReason.SUPERSEDED, newer.wakeup_id).terminal is not None
    failed, terminal = expired(w, cause_event_id=str(uuid4()), evidence_fingerprint="f" * 64)
    scheduler = make(w)
    done = step(w, scheduler)
    assert (done.action, done.target) == (TickAction.COMPLETE, failed.wakeup_id)
    (message,) = messages(state)
    assert message.template is OutboxTemplate.CYCLE_FAILED and message.source_id == terminal.terminal_id
    # The cancelled Wake never gets one.
    assert step(w, scheduler).action is not TickAction.COMPLETE
    assert len(messages(state)) == 1


def test_a_no_row_hold_is_capped_per_wake_and_a_permanent_one_sticks(w) -> None:  # type: ignore[no-untyped-def]
    state, clock = w.state, w.time
    wake, _claimed, _result = closed(w)
    owner = state.wake_outbox_owner
    calls: list[str] = []
    from concurrent.futures import Future

    def collision(wakeup_id):  # type: ignore[no-untyped-def]
        calls.append(wakeup_id)
        future: Future = Future()  # type: ignore[type-arg]
        future.set_result(
            SimpleNamespace(
                value=outbox_owner._OutboxResult(OutboxStatus.HOLD, wakeup_id, None, OutboxHold.COLLISION)
            )
        )
        return future

    owner._submit_complete = collision  # type: ignore[method-assign]
    scheduler = make(w)
    for _ in range(3 * 24 * 7 // 1):
        scheduler.tick()
        clock.advance(900)
    scheduler.wait_idle(30)
    # 3 per hour at most, over the 6.9 days the message stays writable: bounded
    # far below one per tick (~660) and never zero.
    assert 3 <= len(calls) <= 3 * 24 * 7 + 3, len(calls)
    assert len(calls) < 600

    # A permanent reason is stuck for the process lifetime: one event, then none.
    calls.clear()
    scheduler = make(w)

    def unrepresentable(wakeup_id):  # type: ignore[no-untyped-def]
        calls.append(wakeup_id)
        future: Future = Future()  # type: ignore[type-arg]
        future.set_result(
            SimpleNamespace(
                value=outbox_owner._OutboxResult(OutboxStatus.HOLD, wakeup_id, None, OutboxHold.UNREPRESENTABLE)
            )
        )
        return future

    owner._submit_complete = unrepresentable  # type: ignore[method-assign]
    clock.advance(-3 * 24 * 3600)  # still inside the writable window
    for _ in range(40):
        scheduler.tick()
        clock.advance(900)
    scheduler.wait_idle(30)
    assert calls == [wake.wakeup_id]
    # A restart (a fresh Scheduler) forgets it, and only it.
    again = make(w)
    again.tick()
    again.wait_idle(30)
    assert calls == [wake.wakeup_id] * 2


def test_the_queue_and_in_flight_holds_are_not_capped(w) -> None:  # type: ignore[no-untyped-def]
    state, clock = w.state, w.time
    closed(w)
    owner = state.wake_outbox_owner
    calls: list[str] = []
    from concurrent.futures import Future

    def in_flight(wakeup_id):  # type: ignore[no-untyped-def]
        calls.append(wakeup_id)
        future: Future = Future()  # type: ignore[type-arg]
        future.set_result(
            SimpleNamespace(
                value=outbox_owner._OutboxResult(OutboxStatus.HOLD, wakeup_id, None, OutboxHold.IN_FLIGHT)
            )
        )
        return future

    owner._submit_complete = in_flight  # type: ignore[method-assign]
    scheduler = make(w)
    for _ in range(8):
        assert scheduler.tick().action is TickAction.COMPLETE
        scheduler.wait_idle(30)
        clock.advance(301)
    assert len(calls) == 8


def test_the_scheduler_needs_an_enabled_outbox_owner_of_its_store(w) -> None:  # type: ignore[no-untyped-def]
    state = w.state
    state.wake_outbox_owner._enabled = False
    with pytest.raises(ValueError):
        make(w)
    state.wake_outbox_owner._enabled = True
    with pytest.raises(TypeError):
        make(w, outbox_owner=object())
    # Without an Outbox owner the step does nothing (the IP-20d behaviour).
    closed(w)
    plain = make(w, outbox_owner=None)
    assert plain.tick().action is TickAction.NOTHING_DUE


# ------------------------------------------------- end to end and recovery


def test_the_real_chain_from_a_goal_to_one_message_is_nine_events_or_fewer(w) -> None:  # type: ignore[no-untyped-def]
    state, clock = w.state, w.time
    scheduler = make(w, scanned=False)
    seen = []
    # The startup scan writes the Goal's Wake (and asks for another scan).
    for _ in range(3):
        seen.append(step(w, scheduler).action)
    clock.advance(3601)
    for _ in range(6):
        seen.append(step(w, scheduler).action)
        if len(messages(state)) == 1:
            break
    assert len(messages(state)) == 1, seen
    assert TickAction.CLAIM in seen and TickAction.COMPLETE in seen
    (message,) = messages(state)
    wake_rows = [
        row for row in fx.operations(state)
        if row.operation_kind in {
            K.R19_WAKE_MATERIALIZE, K.R19_WAKE_CLAIM, K.R19_WAKE_ATTEMPT_OUTCOME,
            K.R16_CYCLE_DECISION, K.R16_CYCLE_RECEIPT, K.R19_WAKE_TERMINAL, K.R19_OUTBOX_EMIT,
        }
    ]
    assert len(wake_rows) <= 9
    assert [row.operation_kind for row in wake_rows].count(K.R19_OUTBOX_EMIT) == 1
    assert message.context_id
    # No periodic events afterwards.
    before = len(complete_events(state, EventLifecycle.ACCEPTED))
    for _ in range(6):
        clock.advance(600)
        step(w, scheduler)
    assert len(complete_events(state, EventLifecycle.ACCEPTED)) == before == 1
    no_failed_events(state)


def test_a_cycle_without_its_message_is_notified_by_the_next_tick_after_a_restart(w) -> None:  # type: ignore[no-untyped-def]
    """§9-2: a crash between the resolution and its message needs no recovery
    code; a fresh Scheduler (and a fresh owner) derives and writes it."""

    state = w.state
    wake, _claimed, _result = closed(w)
    assert messages(state) == []
    fresh = OutboxOwnerV1(
        state.decision_issuer,
        state.agent_runtime,
        enabled=True,
        timing=state.wake_claim_owner.timing,
        outbox_ttl_s=state.wake_outbox_owner.outbox_ttl_s,
    )
    fresh._clock = w.time.live
    state.wake_outbox_owner = fresh
    scheduler = make(w, outbox_owner=fresh)
    done = step(w, scheduler)
    assert (done.action, done.target) == (TickAction.COMPLETE, wake.wakeup_id)
    assert len(messages(state)) == 1


def test_a_crash_before_the_wal_aborts_the_row_and_the_next_tick_writes_it_again(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """§9-1 / §9-3: an emission that fails before its event is durable fail-stops
    the runtime; the next boot aborts the PREPARED row (releasing ``outbox:``) and
    the Scheduler writes the same ``dedupe_key`` message."""

    import time as real_time

    settings = fx.settings(tmp_path / "crash")
    app = _app(settings)
    app.state.r19_scheduler_enabled = False
    with TestClient(app) as client:
        state = client.app.state
        _x, gid = _qualified_goal(state)
        wake = fx.materialize(state, gid, None)
        claimed = fx.claim(state, wake)
        assert fx.review(state, claimed.claim.claim_id).decision_id is not None
        owner = state.wake_outbox_owner

        def crash(event):  # type: ignore[no-untyped-def]
            if event.event_type is AgentEventType.R19_WAKE_COMPLETE:
                raise RuntimeError("simulated pre-WAL failure")

        owner._validate_commit = crash  # type: ignore[method-assign]
        world = SimpleNamespace(
            state=state,
            time=SimpleNamespace(now=lambda: datetime.now(UTC)),
            mono=real_time.monotonic,
        )
        scheduler = make(world)
        assert scheduler.tick().submitted
        assert scheduler.wait_idle(30)
        assert state.agent_runtime.status.value != "accepting"
    app = _app(settings)
    app.state.r19_scheduler_enabled = False
    with TestClient(app) as client:
        state = client.app.state
        clock = fx.Time(state, monkeypatch)
        state.wake_outbox_owner._clock = clock.live
        assert state.event_journal.inspect().open_transactions == ()
        rows = outbox_rows(state)
        assert len(rows) == 1 and rows[0].phase is HistoryPhase.ABORTED
        assert messages(state) == []
        world = SimpleNamespace(
            state=state, time=clock, mono=lambda: 1_000.0 + clock.offset.total_seconds()
        )
        restarted = make(world)
        assert step(world, restarted).action is TickAction.COMPLETE
        (message,) = messages(state)
        assert message.source_id == wake.logical_cycle_id
        assert state.event_journal.inspect().open_transactions == ()
        no_failed_events(state)
