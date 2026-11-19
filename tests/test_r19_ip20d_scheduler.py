"""IP-20d: the R19 Wake Scheduler and the F10 / R20-F10 acceptance suite
(IP-20 D1 r3 §4-1, §4-3, §7, §8, §9-3, §9-4).

Real app, real R07 runtime, real owners and (where the cause chain matters) the
real materializer.  The Scheduler is driven tick by tick from the test thread
(``r19_scheduler_enabled`` is the server's test seam), with a forward-only fake
wall clock and a fake monotonic clock that moves with it, so leases, backoffs
and 7-day expiries run without sleeping.  Where a handler result cannot be
produced naturally (an injected ``IN_FLIGHT``, a never-ending hold) the owner's
submit method is replaced by one that returns a completed future.
"""

from __future__ import annotations

import ast
from concurrent.futures import Future
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
import threading
import time as real_time

from fastapi.testclient import TestClient
import pytest

import r19_wake_fixtures as fx
from suzka.r16.history_reader import HeadBlockReason, ReaderTransient
from suzka.r16.history_records import HistoryOperationKind as K
from suzka.r16.history_store import HistoryPhase
from suzka.r19 import scheduler as scheduler_module
from suzka.r19.claim_owner import ClaimStatus
from suzka.r19.contracts import (
    AttemptOutcome,
    CYCLE_MAX_EVENTS,
    MAX_ATTEMPTS,
    RefuseReason,
    TerminalReason,
    WakeCause,
    WakeV1,
)
from suzka.r19.ledger import WakeHold, WakeState
from suzka.r19.materializer import MaterializeReason, MaterializeStatus
from suzka.r19.review_owner import ReviewStatus
from suzka.r19.scheduler import THREAD_NAME, TickAction, WakeScheduler
from suzka.runtime.agent_runtime import AgentEventSource, AgentEventType
from suzka.runtime.event_journal import EventLifecycle
from test_r16_ip10_decision_issuer import _issue
from test_r16_u2a_source_owner import _app
from test_r16_u3a_plan_preview import _qualified_commitment, _qualified_goal
import test_r18_ip16a2_intent_issuer as ip16a2

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
R19_EVENT_TYPES = frozenset(
    {
        AgentEventType.R19_WAKE_MATERIALIZE,
        AgentEventType.R19_WAKE_CLAIM,
        AgentEventType.R16_WAKE_REVIEW,
        AgentEventType.R19_WAKE_TERMINAL,
    }
)


# ----------------------------------------------------------------- the world


@pytest.fixture
def sw(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    app = _app(fx.settings(tmp_path / "w"))
    app.state.r19_scheduler_enabled = False  # the tests tick by hand
    with TestClient(app) as client:
        state = client.app.state
        clock = fx.Time(state, monkeypatch)
        x, gid = _qualified_goal(state)
        yield SimpleNamespace(
            state=state,
            time=clock,
            gid=gid,
            x=x,
            monkeypatch=monkeypatch,
            mono=lambda: 1_000.0 + clock.offset.total_seconds(),
        )


def make(w, *, scanned: bool = True, **kwargs):  # type: ignore[no-untyped-def]
    """A Scheduler over the world (a fresh one is a restart).  ``scanned`` marks
    the startup scan as served, so a test that wants a claim first gets one."""

    state = w.state
    kwargs.setdefault("driver_active", lambda _inspection: True)
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
    """The future is done and the runtime worker has left the event."""

    assert scheduler.wait_idle(30)
    deadline = real_time.monotonic() + 10
    while not w.state.agent_runtime.is_idle():
        assert real_time.monotonic() < deadline
        real_time.sleep(0.005)


def step(w, scheduler):  # type: ignore[no-untyped-def]
    outcome = scheduler.tick()
    settle(w, scheduler)
    return outcome


def until(w, scheduler, action, limit=4):  # type: ignore[no-untyped-def]
    """Ticks until ``action`` is chosen (a due scan may come first)."""

    seen = []
    for _ in range(limit):
        outcome = step(w, scheduler)
        seen.append(outcome.action)
        if outcome.action is action:
            return outcome
    raise AssertionError(f"{action} never chosen: {seen}")


def ops(state):  # type: ignore[no-untyped-def]
    return fx.operations(state)


def wake_rows(state):  # type: ignore[no-untyped-def]
    return [row for row in ops(state) if row.operation_kind in WAKE_KINDS]


def kinds(state):  # type: ignore[no-untyped-def]
    return [row.operation_kind for row in wake_rows(state)]


def entry_of(w, wake):  # type: ignore[no-untyped-def]
    return fx.entry(w.state, wake, w.time.now())


def journal_events(state, event_type, lifecycle=EventLifecycle.STARTED):  # type: ignore[no-untyped-def]
    return [
        record
        for record in state.event_journal.inspect().records
        if record.event_type is event_type and record.lifecycle is lifecycle
    ]


def no_failed_events(state) -> None:  # type: ignore[no-untyped-def]
    assert not [
        record
        for record in state.event_journal.inspect().records
        if record.lifecycle is EventLifecycle.FAILED
    ]


def done(value) -> Future:  # type: ignore[type-arg,no-untyped-def]
    future: Future = Future()  # type: ignore[type-arg]
    future.set_result(SimpleNamespace(value=value))
    return future


def refuse_interpretation(w) -> None:  # type: ignore[no-untyped-def]
    def unavailable(*args):  # type: ignore[no-untyped-def]
        raise RuntimeError("R15 is down")

    w.monkeypatch.setattr(w.state.decision_issuer, "_interpretation", unavailable)


# ------------------------------------------------------------ lifecycle (§8)


def test_disabled_has_no_scheduler_no_thread_and_submits_no_r19_event(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with TestClient(_app(fx.settings(tmp_path / "off", r19=False))) as client:
        state = client.app.state
        submitted: list[AgentEventType] = []
        real_submit = state.agent_runtime.submit

        def spy(event_type, source, handler):  # type: ignore[no-untyped-def]
            submitted.append(event_type)
            return real_submit(event_type, source, handler)

        state.agent_runtime.submit = spy  # type: ignore[method-assign]
        assert state.wake_scheduler is None
        assert state.r16_driver._on_cause_terminal is None
        assert not [t for t in threading.enumerate() if t.name.startswith("suzka-r19")]
        real_time.sleep(1.5)
        assert not set(submitted) & R19_EVENT_TYPES
        assert not state.subject_history_store.has_operations()


def test_enabled_scheduler_waits_for_the_driver_then_runs_the_startup_scan(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("SUZKA_P1B_TEST_TOKEN", ip16a2.TOKEN)
    base = ip16a2._settings(tmp_path / "on")
    settings = base.model_copy(
        update={"r19": base.r19.model_copy(update={"enabled": True, "tick_s": 1})}
    )
    app = _app(settings)
    with TestClient(app) as client:
        state = client.app.state
        scheduler = state.wake_scheduler
        assert isinstance(scheduler, WakeScheduler) and scheduler.running
        assert state.r16_driver._on_cause_terminal == scheduler.nudge
        threads = [t for t in threading.enumerate() if t.name == THREAD_NAME]
        assert len(threads) == 1
        # No activation event: the Scheduler exists and does nothing.
        real_time.sleep(2.5)
        assert not journal_events(state, AgentEventType.R19_WAKE_MATERIALIZE, EventLifecycle.ACCEPTED)
        assert client.post("/api/r16/driver/activate", headers=ip16a2.ADMIN).status_code == 200
        scheduler.nudge()
        deadline = real_time.monotonic() + 20
        while not journal_events(state, AgentEventType.R19_WAKE_MATERIALIZE, EventLifecycle.ACCEPTED):
            assert real_time.monotonic() < deadline, "the startup scan never ran"
            real_time.sleep(0.1)
    assert not [t for t in threading.enumerate() if t.name == THREAD_NAME]
    assert not scheduler.running


def test_the_scheduler_needs_enabled_owners_of_one_store(sw) -> None:  # type: ignore[no-untyped-def]
    state = sw.state
    state.wake_claim_owner._enabled = False
    with pytest.raises(ValueError):
        make(sw)
    state.wake_claim_owner._enabled = True
    with pytest.raises(TypeError):
        WakeScheduler(
            state.agent_runtime, state.event_journal, object(), state.wake_claim_owner,  # type: ignore[arg-type]
            state.wake_review_owner, timing=state.wake_claim_owner.timing, rescan_s=10,
        )


def test_the_scheduler_module_has_no_model_and_r16_never_imports_it() -> None:
    root = Path(scheduler_module.__file__).resolve().parents[1]
    imported: set[str] = set()
    tree = ast.parse(Path(scheduler_module.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert not [name for name in imported if "model" in name or "llm" in name or "provider" in name]
    # R19 imports R16, never the reverse: R16 may import only the R19 contracts
    # (the pure decoders the history store and records already use).
    offenders = []
    for path in (root / "r16").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = (
                [node.module] if isinstance(node, ast.ImportFrom) and node.module
                else [a.name for a in node.names] if isinstance(node, ast.Import) else []
            )
            if any(
                name.startswith("suzka.r19") and name != "suzka.r19.contracts" for name in names
            ):
                offenders.append((path.name, names))
    assert offenders == []


def test_the_cycle_event_budget_arithmetic_holds() -> None:
    # 1 materialize + 3 attempts x (claim + review) + 1 terminal + 1 complete (IP-21).
    assert 1 + MAX_ATTEMPTS * 2 + 1 + 1 == 9 <= CYCLE_MAX_EVENTS


# ----------------------------------------------------------- one item per tick


def test_repeated_ticks_make_one_claim_and_one_review_per_lease(sw) -> None:  # type: ignore[no-untyped-def]
    """F10: many ticks, exactly one claim per due Wake, no second review in the
    same lease (here the attempt is a recorded refusal, so the claim stays live)."""

    state, clock = sw.state, sw.time
    refuse_interpretation(sw)
    wake = fx.materialize(state, sw.gid, clock)
    scheduler = make(sw)
    first = step(sw, scheduler)
    assert first == scheduler_module.TickOutcome(TickAction.CLAIM, wake.wakeup_id)
    entry = entry_of(sw, wake)
    assert entry.attempts == 1 and entry.latest_outcome is not None
    assert entry.latest_outcome.reason is RefuseReason.SOURCE_UNAVAILABLE
    rows = len(ops(state))
    for _ in range(25):
        clock.advance(2)
        assert step(sw, scheduler).action is TickAction.NOTHING_DUE
    assert len(ops(state)) == rows
    assert len(journal_events(state, AgentEventType.R16_WAKE_REVIEW)) == 1
    assert len(journal_events(state, AgentEventType.R19_WAKE_CLAIM)) == 1
    no_failed_events(state)


def test_one_future_in_flight_and_the_claim_hands_its_slot_to_the_review(sw) -> None:  # type: ignore[no-untyped-def]
    state = sw.state
    wake = fx.materialize(state, sw.gid, sw.time)
    gate = Future()  # type: ignore[type-arg]
    state.wake_claim_owner._submit_wake_claim = lambda wakeup_id: gate  # type: ignore[method-assign]
    scheduler = make(sw)
    assert scheduler.tick().action is TickAction.CLAIM
    # Still in flight: nothing else is submitted, whatever is due.
    assert scheduler.tick().action is TickAction.BUSY
    gate.set_result(SimpleNamespace(value=SimpleNamespace(status=ClaimStatus.EXISTING)))
    assert scheduler.wait_idle(5)
    assert scheduler.tick().action is TickAction.CLAIM  # the slot is free again
    assert scheduler.submitted[TickAction.CLAIM] == 2
    assert entry_of(sw, wake).state is WakeState.DUE


def test_the_claim_to_review_chain_does_not_depend_on_is_idle(sw) -> None:  # type: ignore[no-untyped-def]
    """HD-20-6 (a): ``is_idle`` gates only the claim; the review is chained from
    the claim future's done-callback even on a runtime that is never idle."""

    state = sw.state
    wake = fx.materialize(state, sw.gid, sw.time)
    scheduler = make(sw)
    calls = []
    real_idle = state.agent_runtime.is_idle

    def counted() -> bool:
        calls.append(1)
        return real_idle()

    sw.monkeypatch.setattr(state.agent_runtime, "is_idle", counted)
    assert scheduler.tick().action is TickAction.CLAIM
    sw.monkeypatch.setattr(state.agent_runtime, "is_idle", lambda: calls.append(2) or False)
    assert scheduler.wait_idle(30)
    deadline = real_time.monotonic() + 10
    while state.agent_runtime._active is not None:
        assert real_time.monotonic() < deadline
        real_time.sleep(0.005)
    assert calls == [1]  # the one claim-time check; the chain never asked
    assert entry_of(sw, wake).state is WakeState.CLOSED
    assert [k for k in kinds(state)].count(K.R19_WAKE_CLAIM) == 1
    assert len(journal_events(state, AgentEventType.R16_WAKE_REVIEW)) == 1


def test_a_claim_waits_for_an_idle_runtime_and_a_review_does_not(sw) -> None:  # type: ignore[no-untyped-def]
    """HD-20-6 (a): a busy runtime (a running handler, nothing queued) delays the
    claim and burns no attempt; the Scheduler claims once it is idle."""

    state = sw.state
    wake = fx.materialize(state, sw.gid, sw.time)
    scheduler = make(sw)
    started, release = threading.Event(), threading.Event()

    def slow() -> None:
        started.set()
        release.wait(30)

    future = state.agent_runtime.submit(
        AgentEventType.EMOTION_TICK, AgentEventSource.RUNTIME_EMOTION_TIMER, slow
    )
    assert started.wait(10)
    assert state.agent_runtime.pending_count() == 0 and not state.agent_runtime.is_idle()
    for _ in range(3):
        assert scheduler.tick().action is TickAction.NOTHING_DUE
    assert entry_of(sw, wake).attempts == 0
    release.set()
    future.result(timeout=30)
    deadline = real_time.monotonic() + 10
    while not state.agent_runtime.is_idle():
        assert real_time.monotonic() < deadline
        real_time.sleep(0.005)
    assert step(sw, scheduler).action is TickAction.CLAIM
    assert entry_of(sw, wake).state is WakeState.CLOSED


def test_priority_terminal_then_supersede_then_review_then_scan_then_claim(sw) -> None:  # type: ignore[no-untyped-def]
    state, clock, monkeypatch = sw.state, sw.time, sw.monkeypatch
    scheduler = make(sw, scanned=False)
    # Claim candidate + due scan + a live claim awaiting its review + an
    # expired Wake: the terminal goes first, then the review, then the scan.
    expired = fx.materialize(state, sw.gid, clock, expires=-5, not_before=-4000)
    live = fx.materialize(
        state, sw.gid, clock, cause_event_id=str(uuid4()), evidence_fingerprint="a" * 64
    )
    fx.claim(state, live)
    due = fx.materialize(
        state, sw.gid, clock, cause_event_id=str(uuid4()), evidence_fingerprint="b" * 64
    )
    order = [step(sw, scheduler) for _ in range(3)]
    assert [item.action for item in order] == [
        TickAction.TERMINAL,
        TickAction.REVIEW,
        TickAction.MATERIALIZE,
    ]
    assert order[0].target == expired.wakeup_id and order[1].target == live.wakeup_id
    assert entry_of(sw, expired).state is WakeState.FAILED
    # After the scan (it found nothing new), the claim is last.
    scheduler._scan_served = scheduler._scan_requested
    last = step(sw, scheduler)
    assert (last.action, last.target) == (TickAction.CLAIM, due.wakeup_id)
    no_failed_events(state)
    del monkeypatch


def test_a_commitment_wake_supersedes_the_unclaimed_goal_wake_through_the_scheduler(sw) -> None:  # type: ignore[no-untyped-def]
    """Real chain: Goal then Commitment (HD-20-5 (b)).  The supersession is its
    own terminal event, before any claim; the Commitment Wake is then claimed."""

    state, clock = sw.state, sw.time
    scheduler = make(sw, scanned=False)
    first = step(sw, scheduler)
    assert first.action is TickAction.MATERIALIZE
    # COMMITTED asks for another scan (a scan writes at most one Wake).
    assert step(sw, scheduler).action is TickAction.MATERIALIZE
    assert step(sw, scheduler).action is TickAction.NOTHING_DUE
    goal_wake = fx.adopted(state, WakeV1)[0]
    assert goal_wake.cause is WakeCause.GOAL_ADMITTED
    _qualified_commitment(state, sw.x)
    scheduler.nudge()
    assert step(sw, scheduler).action is TickAction.MATERIALIZE  # the Commitment Wake
    # The supersession outranks the rescan that a COMMITTED scan asks for.
    again = step(sw, scheduler)
    assert (again.action, again.target) == (TickAction.SUPERSEDE, goal_wake.wakeup_id)
    assert entry_of(sw, goal_wake).state is WakeState.CANCELLED
    clock.advance(3601)
    claimed = until(sw, scheduler, TickAction.CLAIM)
    assert claimed.target != goal_wake.wakeup_id
    no_failed_events(state)


# ------------------------------------------------------------- the real cycle


def test_real_goal_cycle_end_to_end_without_a_user_message(sw) -> None:  # type: ignore[no-untyped-def]
    """Goal admitted -> materialize -> due -> claim -> review -> one Decision and
    its resolution, with no chat event after the Goal; the same evidence never
    makes a second Wake afterwards."""

    state, clock = sw.state, sw.time
    scheduler = make(sw, scanned=False)
    chat_before = len(journal_events(state, AgentEventType.CHAT, EventLifecycle.ACCEPTED))
    assert step(sw, scheduler).action is TickAction.MATERIALIZE
    assert step(sw, scheduler).action is TickAction.MATERIALIZE  # rescan: nothing left
    assert step(sw, scheduler).action is TickAction.NOTHING_DUE  # not due yet
    (wake,) = fx.adopted(state, WakeV1)
    clock.advance(3601)
    until(sw, scheduler, TickAction.CLAIM)
    entry = entry_of(sw, wake)
    assert entry.state is WakeState.CLOSED and entry.terminal is None
    assert entry.resolution is not None
    assert kinds(state).count(K.R19_WAKE_CLAIM) == 1
    assert len(wake_rows(state)) <= 9
    for _ in range(5):
        clock.advance(300)
        assert step(sw, scheduler).action in (TickAction.NOTHING_DUE, TickAction.MATERIALIZE)
    # No re-Wake for the same fingerprint, after the cycle closed.
    assert len(fx.adopted(state, WakeV1)) == 1
    assert len(journal_events(state, AgentEventType.CHAT, EventLifecycle.ACCEPTED)) == chat_before
    no_failed_events(state)


def test_a_duplicate_scan_and_a_second_wake_for_one_fingerprint_are_nothing(sw) -> None:  # type: ignore[no-untyped-def]
    state, clock = sw.state, sw.time
    scheduler = make(sw, scanned=False)
    step(sw, scheduler)
    step(sw, scheduler)
    (wake,) = fx.adopted(state, WakeV1)
    # The first cycle ends FAILED (expired); the evidence is the same, so a
    # rescan still makes no Wake (the ``wake:`` key is permanent).
    clock.advance(8 * 86_400)
    assert step(sw, scheduler).action is TickAction.TERMINAL
    assert entry_of(sw, wake).state is WakeState.FAILED
    scheduler.nudge()
    for _ in range(3):
        step(sw, scheduler)
        clock.advance(1900)
    assert len(fx.adopted(state, WakeV1)) == 1
    assert kinds(state).count(K.R19_WAKE_TERMINAL) == 1
    no_failed_events(state)


# --------------------------------------------------------------- the rescan


def test_startup_scan_backstop_and_nudge_are_bounded(sw) -> None:  # type: ignore[no-untyped-def]
    state, clock = sw.state, sw.time
    scheduler = make(sw, scanned=False)
    scans = lambda: len(journal_events(state, AgentEventType.R19_WAKE_MATERIALIZE, EventLifecycle.ACCEPTED))  # noqa: E731
    assert scans() == 0
    assert step(sw, scheduler).action is TickAction.MATERIALIZE  # startup (finds the Goal)
    assert step(sw, scheduler).action is TickAction.MATERIALIZE  # the rescan after COMMITTED
    assert scans() == 2
    for _ in range(10):  # nothing to do inside the backstop
        clock.advance(100)
        assert step(sw, scheduler).action is TickAction.NOTHING_DUE
    assert scans() == 2
    clock.advance(1000)  # past rescan_s (1800 s) since the last scan
    assert step(sw, scheduler).action is TickAction.MATERIALIZE
    assert scans() == 3
    scheduler.nudge()
    scheduler.nudge()  # coalesced
    assert step(sw, scheduler).action is TickAction.MATERIALIZE
    assert step(sw, scheduler).action is TickAction.NOTHING_DUE
    assert scans() == 4


def test_a_committed_scan_asks_for_another_until_none_left(sw) -> None:  # type: ignore[no-untyped-def]
    """One IN_FLIGHT candidate HOLDs the whole scan and a scan writes one Wake:
    the Scheduler rescans after COMMITTED (carry-over)."""

    state = sw.state
    results = iter(
        [
            SimpleNamespace(status=MaterializeStatus.COMMITTED, reason=None),
            SimpleNamespace(status=MaterializeStatus.COMMITTED, reason=None),
            SimpleNamespace(status=MaterializeStatus.NONE_LEFT, reason=None),
        ]
    )
    state.wake_materializer.submit_scan = lambda: done(next(results))  # type: ignore[method-assign]
    scheduler = make(sw, scanned=False)
    assert [step(sw, scheduler).action for _ in range(4)] == [
        TickAction.MATERIALIZE,
        TickAction.MATERIALIZE,
        TickAction.MATERIALIZE,
        TickAction.NOTHING_DUE,
    ]


def test_a_transient_scan_hold_backs_off_and_in_flight_is_never_capped(sw) -> None:  # type: ignore[no-untyped-def]
    state, clock = sw.state, sw.time
    calls = []

    def held():  # type: ignore[no-untyped-def]
        calls.append(1)
        return done(
            SimpleNamespace(status=MaterializeStatus.HOLD, reason=MaterializeReason.IN_FLIGHT)
        )

    state.wake_materializer.submit_scan = held  # type: ignore[method-assign]
    scheduler = make(sw, scanned=False)
    assert scheduler.tick().action is TickAction.MATERIALIZE
    assert scheduler.tick().action is TickAction.NOTHING_DUE  # backing off
    for _ in range(12):
        clock.advance(301)
        assert scheduler.tick().action is TickAction.MATERIALIZE
    assert len(calls) == 13  # uncapped: backoff only
    # A non-transient hold waits for the backstop instead.
    state.wake_materializer.submit_scan = lambda: done(  # type: ignore[method-assign]
        SimpleNamespace(status=MaterializeStatus.HOLD, reason=MaterializeReason.R19_DISABLED)
    )
    clock.advance(301)
    assert scheduler.tick().action is TickAction.MATERIALIZE
    clock.advance(301)
    assert scheduler.tick().action is TickAction.NOTHING_DUE


# ------------------------------------------------------- reads and fail paths


def test_a_derive_error_is_recorded_and_does_nothing_that_tick(sw) -> None:  # type: ignore[no-untyped-def]
    state = sw.state
    fx.materialize(state, sw.gid, sw.time)
    scheduler = make(sw)
    rows = len(ops(state))

    def broken(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise TypeError("an adopted record of this kind must be a WakeV1")

    with sw.monkeypatch.context() as patch:
        patch.setattr(scheduler_module, "derive_wake_view", broken)
        assert scheduler.tick().action is TickAction.DERIVE_ERROR
    assert scheduler.derive_errors == 1 and len(ops(state)) == rows
    assert step(sw, scheduler).action is TickAction.CLAIM  # the next tick is normal


def test_an_unreadable_history_or_inactive_driver_or_stopped_runtime_does_nothing(sw) -> None:  # type: ignore[no-untyped-def]
    state = sw.state
    fx.materialize(state, sw.gid, sw.time)
    rows = len(ops(state))
    inactive = make(sw, driver_active=lambda _inspection: False)
    assert inactive.tick().action is TickAction.DRIVER_INACTIVE
    scheduler = make(sw)
    from suzka.r16 import history_reader

    def transient(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise ReaderTransient(HeadBlockReason.STORE_UNAVAILABLE)

    with sw.monkeypatch.context() as patch:
        patch.setattr(history_reader.SubjectHistoryReader, "classify", transient)
        assert scheduler.tick().action is TickAction.HISTORY_UNAVAILABLE
    assert scheduler.read_errors == 1
    scheduler.stop()
    assert scheduler.tick().action is TickAction.STOPPED
    assert len(ops(state)) == rows


def test_a_runtime_that_stopped_is_never_submitted_to_and_the_thread_survives(sw) -> None:  # type: ignore[no-untyped-def]
    state = sw.state
    fx.materialize(state, sw.gid, sw.time)
    scheduler = make(sw)
    scheduler.start()
    try:
        assert scheduler.running
        scheduler.start()  # a second start creates no second thread
        assert len([t for t in threading.enumerate() if t.name == THREAD_NAME]) == 1
        scheduler._runtime = SimpleNamespace(  # type: ignore[assignment]
            status=scheduler_module.AgentRuntimeStatus.FAILED, is_idle=lambda: False
        )
        assert scheduler.tick().action is TickAction.NOT_ACCEPTING
        assert scheduler.running  # a failed runtime does not kill the thread
    finally:
        scheduler.stop()
    assert not scheduler.running


def test_an_event_that_fails_is_counted_and_backed_off(sw) -> None:  # type: ignore[no-untyped-def]
    state, clock = sw.state, sw.time
    fx.materialize(state, sw.gid, clock)
    failing = Future()  # type: ignore[type-arg]
    failing.set_exception(RuntimeError("handler failed"))
    state.wake_claim_owner._submit_wake_claim = lambda wakeup_id: failing  # type: ignore[method-assign]
    scheduler = make(sw)
    assert scheduler.tick().action is TickAction.CLAIM
    assert scheduler.tick().action is TickAction.NOTHING_DUE  # backing off
    clock.advance(31)
    assert scheduler.tick().action is TickAction.CLAIM


# -------------------------------------------------------- restart and recovery


def test_claim_after_restart_a_live_lease_without_a_started_review_gets_one(sw) -> None:  # type: ignore[no-untyped-def]
    state = sw.state
    wake = fx.materialize(state, sw.gid, sw.time)
    claimed = fx.claim(state, wake)  # the first Scheduler died right here
    assert claimed.status is ClaimStatus.COMMITTED
    assert not journal_events(state, AgentEventType.R16_WAKE_REVIEW)
    restarted = make(sw)
    outcome = step(sw, restarted)
    assert (outcome.action, outcome.target) == (TickAction.REVIEW, wake.wakeup_id)
    assert entry_of(sw, wake).state is WakeState.CLOSED
    assert kinds(state).count(K.R19_WAKE_CLAIM) == 1  # the same attempt, no new claim
    assert len(journal_events(state, AgentEventType.R16_WAKE_REVIEW)) == 1


def test_claim_after_restart_a_started_review_gets_none_and_the_attempt_lapses(sw) -> None:  # type: ignore[no-untyped-def]
    state, clock = sw.state, sw.time
    wake = fx.materialize(state, sw.gid, clock)
    claimed = fx.claim(state, wake)
    from suzka.r16.history_store import HistoryStoreCapacityError

    def full(length):  # type: ignore[no-untyped-def]
        raise HistoryStoreCapacityError("full")

    with sw.monkeypatch.context() as patch:
        patch.setattr(state.subject_history_store, "check_headroom", full)
        held = fx.review(state, claimed.claim.claim_id)
    assert (held.status, held.hold) == (ReviewStatus.HOLD, WakeHold.HISTORY_CAPACITY)
    restarted = make(sw)
    assert step(sw, restarted).action is TickAction.NOTHING_DUE  # no second review
    assert len(journal_events(state, AgentEventType.R16_WAKE_REVIEW)) == 1
    # A lapsed lease gets attempt 2 (and its chained review closes the cycle).
    clock.advance(121)
    assert step(sw, restarted).action is TickAction.CLAIM
    assert entry_of(sw, wake).attempts == 2
    assert len(journal_events(state, AgentEventType.R16_WAKE_REVIEW)) == 2


def test_claim_after_restart_a_lapsed_lease_without_any_review_gets_attempt_two(sw) -> None:  # type: ignore[no-untyped-def]
    state, clock = sw.state, sw.time
    wake = fx.materialize(state, sw.gid, clock)
    fx.claim(state, wake)
    clock.advance(121)
    restarted = make(sw)
    assert entry_of(sw, wake).state is WakeState.RETRYABLE
    assert step(sw, restarted).action is TickAction.CLAIM
    assert entry_of(sw, wake).attempts == 2
    assert entry_of(sw, wake).state is WakeState.CLOSED  # the chained review closed it


def test_crash_loop_inside_one_lease_adds_no_review_beyond_the_first(sw) -> None:  # type: ignore[no-untyped-def]
    """5 restarts between a claim and its lease end: at most one STARTED review
    for the claim, and the cycle's row-writing events stay <= 9."""

    state, clock = sw.state, sw.time
    wake = fx.materialize(state, sw.gid, clock)
    fx.claim(state, wake)
    from suzka.r16.history_store import HistoryStoreCapacityError

    def full(length):  # type: ignore[no-untyped-def]
        raise HistoryStoreCapacityError("full")

    outcomes = []
    for restart in range(5):
        scheduler = make(sw)
        clock.advance(2)
        if restart == 0:
            with sw.monkeypatch.context() as patch:
                patch.setattr(state.subject_history_store, "check_headroom", full)
                outcomes.append(step(sw, scheduler).action)
        else:
            outcomes.append(step(sw, scheduler).action)
    assert outcomes == [TickAction.REVIEW] + [TickAction.NOTHING_DUE] * 4
    assert len(journal_events(state, AgentEventType.R16_WAKE_REVIEW)) == 1
    assert len(wake_rows(state)) <= 9


# ----------------------------------------------------------------- the clock


def test_a_rolled_back_clock_waits_without_a_claim_or_a_terminal_or_a_double_write(sw) -> None:  # type: ignore[no-untyped-def]
    state, clock = sw.state, sw.time
    wake = fx.materialize(state, sw.gid, clock, not_before=-4 * 3600)
    scheduler = make(sw)
    rows = len(ops(state))
    clock.advance(-3 * 3600)
    for _ in range(5):
        assert scheduler.tick().action is TickAction.NOTHING_DUE
    assert len(ops(state)) == rows and entry_of(sw, wake).attempts == 0
    clock.advance(3 * 3600)
    assert step(sw, scheduler).action is TickAction.CLAIM  # the Wake closes when the clock is back
    assert entry_of(sw, wake).state is WakeState.CLOSED
    no_failed_events(state)


def test_a_review_below_its_claims_floor_is_a_recorded_refusal_that_counts(sw) -> None:  # type: ignore[no-untyped-def]
    state, clock = sw.state, sw.time
    wake = fx.materialize(state, sw.gid, clock, not_before=-4 * 3600)
    claimed = fx.claim(state, wake)
    clock.advance(-3 * 3600)
    state.agent_runtime._last_admitted_at = None  # the restart that follows a rollback
    restarted = make(sw)
    outcome = step(sw, restarted)
    assert (outcome.action, outcome.target) == (TickAction.REVIEW, wake.wakeup_id)
    entry = entry_of(sw, wake)
    assert entry.latest_outcome.claim_id == claimed.claim.claim_id
    assert entry.latest_outcome.reason is RefuseReason.CLOCK_UNVERIFIABLE
    assert entry.attempts == 1 and entry.state is not WakeState.FAILED
    no_failed_events(state)


def test_expired_pending_lease_is_derived_only_no_row(sw) -> None:  # type: ignore[no-untyped-def]
    state = sw.state
    wake = fx.materialize(state, sw.gid, sw.time)
    scheduler = make(sw)
    real = scheduler_module.derive_wake_view

    def pending(records, now, *, clock_floor):  # type: ignore[no-untyped-def]
        view = real(records, now, clock_floor=clock_floor)
        return replace(
            view,
            entries=tuple(replace(e, state=WakeState.EXPIRED_PENDING_LEASE) for e in view.entries),
        )

    rows = len(ops(state))
    with sw.monkeypatch.context() as patch:
        patch.setattr(scheduler_module, "derive_wake_view", pending)
        for _ in range(3):
            assert scheduler.tick().action is TickAction.NOTHING_DUE
    assert len(ops(state)) == rows
    del wake


def test_a_forward_clock_jump_retires_the_wake_once_and_never_re_wakes(sw) -> None:  # type: ignore[no-untyped-def]
    state, clock = sw.state, sw.time
    wake = fx.materialize(state, sw.gid, clock)
    scheduler = make(sw)
    clock.advance(9 * 86_400)
    first = step(sw, scheduler)
    assert (first.action, first.target) == (TickAction.TERMINAL, wake.wakeup_id)
    assert entry_of(sw, wake).state is WakeState.FAILED
    assert entry_of(sw, wake).terminal.reason is TerminalReason.EXPIRED_WAKE
    for _ in range(4):
        clock.advance(2000)
        assert step(sw, scheduler).action in (TickAction.NOTHING_DUE, TickAction.MATERIALIZE)
    assert kinds(state).count(K.R19_WAKE_TERMINAL) == 1
    assert len(fx.adopted(state, WakeV1)) == 1
    no_failed_events(state)


# ------------------------------------------------------------------- budgets


def test_budget_exhaustion_ends_after_three_leases_and_never_calls_a_model(sw) -> None:  # type: ignore[no-untyped-def]
    state, clock = sw.state, sw.time
    refuse_interpretation(sw)
    wake = fx.materialize(state, sw.gid, clock)
    scheduler = make(sw)
    model_calls = state.model_provider.calls
    started = clock.now()
    seen = []
    for _ in range(40):
        outcome = step(sw, scheduler)
        seen.append(outcome.action)
        if entry_of(sw, wake).state is WakeState.FAILED:
            break
        clock.advance(31)
    entry = entry_of(sw, wake)
    assert entry.state is WakeState.FAILED
    assert entry.terminal.reason is TerminalReason.ATTEMPTS_EXHAUSTED
    assert entry.attempts == 3
    # Three claims need two lease gaps (the EXHAUSTED state needs the third
    # attempt's outcome row, not its lease end, ledger §4-2).
    assert clock.now() - started >= timedelta(seconds=2 * 120 - 5)
    assert seen.count(TickAction.CLAIM) == 3 and seen.count(TickAction.TERMINAL) == 1
    assert len(wake_rows(state)) == 1 + 3 * 2 + 1 <= 9
    assert state.model_provider.calls == model_calls
    no_failed_events(state)


def test_a_deferred_budget_records_and_the_gap_still_waits_for_the_lease(sw) -> None:  # type: ignore[no-untyped-def]
    state, clock = sw.state, sw.time
    wake = fx.materialize(state, sw.gid, clock)
    ticks = iter([0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    state.wake_review_owner._monotonic = lambda: next(ticks)
    scheduler = make(sw)
    assert step(sw, scheduler).action is TickAction.CLAIM
    entry = entry_of(sw, wake)
    assert entry.latest_outcome.outcome is AttemptOutcome.DEFERRED_BUDGET
    clock.advance(30)
    assert step(sw, scheduler).action is TickAction.NOTHING_DUE  # the lease still holds
    clock.advance(91)
    assert step(sw, scheduler).action is TickAction.CLAIM
    assert entry_of(sw, wake).attempts == 2


def test_existing_and_drift_close_once_and_are_never_failed_or_claimed_again(sw) -> None:  # type: ignore[no-untyped-def]
    state, clock = sw.state, sw.time
    # EXISTING (S1): a PLAN_PREVIEW Decision already holds the evidence.
    wake = fx.materialize(state, sw.gid, clock)
    issued = _issue(state, sw.gid)
    assert issued.decision is not None
    scheduler = make(sw)
    assert step(sw, scheduler).action is TickAction.CLAIM
    entry = entry_of(sw, wake)
    assert entry.state is WakeState.CLOSED and entry.terminal is None
    assert kinds(state).count(K.R16_CYCLE_RECEIPT) == 1
    rows = len(ops(state))
    for _ in range(6):
        clock.advance(2000)
        assert step(sw, scheduler).action in (TickAction.NOTHING_DUE, TickAction.MATERIALIZE)
    assert entry_of(sw, wake).state is WakeState.CLOSED  # never EXPIRED, never FAILED
    assert kinds(state).count(K.R19_WAKE_CLAIM) == 1 and kinds(state).count(K.R19_WAKE_TERMINAL) == 0
    assert len(ops(state)) == rows  # no cycle rows, no scan rows
    no_failed_events(state)


def test_source_drift_closes_once_through_the_scheduler(sw) -> None:  # type: ignore[no-untyped-def]
    from suzka.r16 import decision_issuer as issuer_module
    from suzka.r16.cycle_contracts import HoldReason

    state, clock, monkeypatch = sw.state, sw.time, sw.monkeypatch
    with monkeypatch.context() as patch:
        patch.setattr(issuer_module, "_context_digest", lambda q: "e" * 64)
        wake = fx.materialize(state, sw.gid, clock)
    _issue(state, sw.gid)
    monkeypatch.setattr(issuer_module, "_context_digest", lambda q: "e" * 64)
    scheduler = make(sw)
    assert step(sw, scheduler).action is TickAction.CLAIM
    entry = entry_of(sw, wake)
    assert entry.state is WakeState.CLOSED and entry.terminal is None
    assert entry.closure.result.reason is HoldReason.DECISION_SOURCE_DRIFT
    for _ in range(4):
        clock.advance(2000)
        assert step(sw, scheduler).action in (TickAction.NOTHING_DUE, TickAction.MATERIALIZE)
    assert kinds(state).count(K.R19_WAKE_CLAIM) == 1 and kinds(state).count(K.R19_WAKE_TERMINAL) == 0


# ------------------------------------------------------- congestion and storms


def test_congestion_then_free_never_ends_the_wake(sw) -> None:  # type: ignore[no-untyped-def]
    """r3 P1-new: injected QUEUE_DELAY (a live clock 11 s ahead of T, for real)
    and IN_FLIGHT are not capped; when the queue is free the Wake is claimed and
    reviewed, and it never ends EXPIRED_WAKE because of the congestion."""

    state, clock = sw.state, sw.time
    wake = fx.materialize(state, sw.gid, clock)
    scheduler = make(sw)
    rows = len(ops(state))
    clock.skew = timedelta(seconds=11)
    for _ in range(8):  # more than the cap of 3 an hour
        assert scheduler.tick().action is TickAction.CLAIM
        settle(sw, scheduler)
        clock.advance(301)
    assert len(ops(state)) == rows  # a no-row HOLD: not an attempt
    assert entry_of(sw, wake).attempts == 0
    real_claim = state.wake_claim_owner._submit_wake_claim
    injected = []

    def in_flight(wakeup_id):  # type: ignore[no-untyped-def]
        injected.append(wakeup_id)
        return done(SimpleNamespace(status=ClaimStatus.HOLD, hold=WakeHold.IN_FLIGHT))

    state.wake_claim_owner._submit_wake_claim = in_flight  # type: ignore[method-assign]
    clock.skew = timedelta(0)
    for _ in range(8):
        assert scheduler.tick().action is TickAction.CLAIM
        clock.advance(301)
    assert len(injected) == 8
    state.wake_claim_owner._submit_wake_claim = real_claim  # type: ignore[method-assign]
    assert step(sw, scheduler).action is TickAction.CLAIM
    entry = entry_of(sw, wake)
    assert entry.state is WakeState.CLOSED and entry.attempts == 1
    assert kinds(state).count(K.R19_WAKE_TERMINAL) == 0
    no_failed_events(state)


def test_a_no_row_hold_storm_is_bounded_over_seven_days(sw) -> None:  # type: ignore[no-untyped-def]
    """A Wake whose claim a handler keeps refusing without a row: backoff and the
    hourly cap bound the events (far below the ~2000 an uncapped 300 s backoff
    would cost), and the Wake still ends by its own expiry."""

    state, clock = sw.state, sw.time
    wake = fx.materialize(state, sw.gid, clock)
    claims = []

    def collision(wakeup_id):  # type: ignore[no-untyped-def]
        claims.append(wakeup_id)
        return done(SimpleNamespace(status=ClaimStatus.HOLD, hold=WakeHold.COLLISION))

    state.wake_claim_owner._submit_wake_claim = collision  # type: ignore[method-assign]
    scheduler = make(sw)
    for _ in range(7 * 24 * 4 + 8):
        scheduler.tick()
        clock.advance(900)
    scheduler.wait_idle(30)
    assert 100 < len(claims) <= 3 * 24 * 7 + 3, len(claims)
    assert len(claims) < 2000
    entry = entry_of(sw, wake)
    assert entry.state is WakeState.FAILED
    assert entry.terminal.reason is TerminalReason.EXPIRED_WAKE
    no_failed_events(state)


def test_the_pre_filter_submits_nothing_a_handler_would_refuse(sw) -> None:  # type: ignore[no-untyped-def]
    state, clock = sw.state, sw.time
    submitted = []
    for name in ("_submit_wake_claim", "_submit_terminal"):
        real = getattr(state.wake_claim_owner, name)

        def counted(*args, _real=real, **kwargs):  # type: ignore[no-untyped-def]
            submitted.append(args)
            return _real(*args, **kwargs)

        setattr(state.wake_claim_owner, name, counted)
    scheduler = make(sw)
    # Not yet due, a Wake whose window cannot hold a claim, a rolled back clock.
    fx.materialize(state, sw.gid, clock, not_before=3600, expires=7 * 86_400 - 60)
    for _ in range(10):
        assert scheduler.tick().action is TickAction.NOTHING_DUE
        clock.advance(60)
    assert submitted == []


# -------------------------------------------------------------- thread reads


def test_the_scheduler_thread_reads_safely_beside_runtime_events_and_store_writes(sw) -> None:  # type: ignore[no-untyped-def]
    """§4-1 / P2-8: the tick (its own thread) runs against a stream of runtime
    events and a store write storm: no exception, and no submission that fails
    its own handler's checks."""

    state, clock = sw.state, sw.time
    scheduler = make(sw)
    outcomes: list[TickAction] = []
    errors: list[BaseException] = []
    stop = threading.Event()

    def ticker() -> None:
        try:
            while not stop.is_set():
                outcomes.append(scheduler.tick().action)
                scheduler.wait_idle(30)
        except BaseException as error:  # pragma: no cover - the assertion below
            errors.append(error)

    thread = threading.Thread(target=ticker, name="test-ticker")
    thread.start()
    try:
        for index in range(4):
            fx.materialize(
                state, sw.gid, clock,
                cause_event_id=str(uuid4()), evidence_fingerprint=f"{index:x}" * 64,
            )
            for _ in range(12):
                state.agent_runtime.submit(
                    AgentEventType.EMOTION_TICK, AgentEventSource.RUNTIME_EMOTION_TIMER, lambda: None
                ).result(timeout=30)
            real_time.sleep(0.05)
    finally:
        stop.set()
        thread.join(timeout=60)
    assert not thread.is_alive() and errors == []
    assert TickAction.ERROR not in outcomes and TickAction.DERIVE_ERROR not in outcomes
    assert TickAction.CLAIM in outcomes
    no_failed_events(state)
    # Every Wake the Scheduler touched is in a consistent state (nothing CORRUPT).
    view = fx.view(state, clock.now())
    assert not [e for e in view.entries if e.state is WakeState.CORRUPT]
    assert all(row.phase is not HistoryPhase.PREPARED for row in wake_rows(state))


# ---------------------------------------------------- Condition B: no Human


def test_every_wake_state_reaches_a_settled_state_with_the_scheduler_alone(sw) -> None:  # type: ignore[no-untyped-def]
    state, clock = sw.state, sw.time
    expired = fx.materialize(state, sw.gid, clock, expires=-5, not_before=-4000,
                             cause_event_id=str(uuid4()), evidence_fingerprint="1" * 64)
    fx.materialize(state, sw.gid, clock, cause_event_id=str(uuid4()), evidence_fingerprint="2" * 64)
    fx.materialize(state, sw.gid, clock, cause_event_id=str(uuid4()), evidence_fingerprint="3" * 64)
    scheduler = make(sw)
    refuse_interpretation(sw)
    settled = {WakeState.CLOSED, WakeState.FAILED, WakeState.CANCELLED}
    for _ in range(80):
        step(sw, scheduler)
        view = fx.view(state, clock.now())
        if all(e.state in settled for e in view.entries):
            break
        clock.advance(61)
    view = fx.view(state, clock.now())
    states = {e.wakeup_id: e.state for e in view.entries}
    assert all(item in settled for item in states.values()), states
    assert states[expired.wakeup_id] is WakeState.FAILED
    assert WakeState.CORRUPT not in states.values()
    no_failed_events(state)


# ------------------------------------------------------------- crash windows


@pytest.mark.parametrize("stage", ["claim", "review", "terminal"])
def test_a_crash_before_the_wal_aborts_and_the_next_tick_converges(tmp_path, monkeypatch, stage) -> None:  # type: ignore[no-untyped-def]
    """§7-1 / §9-3 crash windows: a claim, a review or a terminal that fails
    before its event is durable fail-stops the runtime; the next boot aborts the
    PREPARED row (releasing its keys) and the Scheduler converges: the claim and
    terminal are simply submitted again, a review lost after STARTED lapses with
    its lease and the next attempt closes the cycle.  (The roll-forward half of
    §7-1 belongs to R07 and is covered by the IP-9 crash matrix.)"""

    from datetime import datetime, timezone

    from suzka.runtime.agent_runtime import AgentEventType as T

    settings = fx.settings(tmp_path / "crash")
    crash_type = {
        "claim": T.R19_WAKE_CLAIM,
        "review": T.R16_WAKE_REVIEW,
        "terminal": T.R19_WAKE_TERMINAL,
    }[stage]
    owner_name = "wake_review_owner" if stage == "review" else "wake_claim_owner"

    def real_world(state):  # type: ignore[no-untyped-def]
        return SimpleNamespace(
            state=state,
            time=SimpleNamespace(now=lambda: datetime.now(timezone.utc)),
            mono=real_time.monotonic,
        )

    app = _app(settings)
    app.state.r19_scheduler_enabled = False
    with TestClient(app) as client:
        state = client.app.state
        _x, gid = _qualified_goal(state)
        if stage == "terminal":
            wake = fx.materialize(state, gid, None, expires=-5, not_before=-4000)
        else:
            wake = fx.materialize(state, gid, None)
        if stage == "review":
            assert fx.claim(state, wake).status is ClaimStatus.COMMITTED
        owner = getattr(state, owner_name)

        def crash(event):  # type: ignore[no-untyped-def]
            if event.event_type is crash_type:
                raise RuntimeError("simulated pre-WAL failure")

        owner._validate_commit = crash
        scheduler = make(real_world(state))
        assert scheduler.tick().submitted
        assert scheduler.wait_idle(30)
        assert state.agent_runtime.status.value != "accepting"
        # A stopped runtime is never submitted to again.
        assert scheduler.tick().action is TickAction.NOT_ACCEPTING
    app = _app(settings)
    app.state.r19_scheduler_enabled = False
    with TestClient(app) as client:
        state = client.app.state
        clock = fx.Time(state, monkeypatch)
        assert state.event_journal.inspect().open_transactions == ()
        aborted = [row for row in wake_rows(state) if row.phase is HistoryPhase.ABORTED]
        assert len(aborted) == 1
        world = SimpleNamespace(
            state=state, time=clock, mono=lambda: 1_000.0 + clock.offset.total_seconds()
        )
        entry = entry_of(world, wake)
        assert entry.terminal is None and entry.closure is None
        restarted = make(world)
        if stage == "review":
            # The review STARTED before the crash: none is repeated; it lapses.
            assert step(world, restarted).action is TickAction.NOTHING_DUE
            clock.advance(121)
        outcome = step(world, restarted)
        assert outcome.action in (TickAction.CLAIM, TickAction.TERMINAL)
        entry = fx.entry(state, wake, clock.now())
        if stage == "terminal":
            assert entry.state is WakeState.FAILED
            assert entry.terminal.reason is TerminalReason.EXPIRED_WAKE
        else:
            assert entry.state is WakeState.CLOSED
            assert entry.attempts == (2 if stage == "review" else 1)
        assert state.event_journal.inspect().open_transactions == ()


# ------------------------------------------------------- pacing (review P2)


def test_pacer_backoff_cap_hour_reset_and_state_reset() -> None:
    from suzka.r19.scheduler import BACKOFF_CAP_S, HOLD_CAP_PER_HOUR, WakePacer

    now = [0.0]
    pacer = WakePacer(lambda: now[0])
    # Backoff doubles 30, 60, ... and is capped at 300 s.
    delays = []
    for _ in range(6):
        pacer.record_hold("w", "COLLISION", "DUE", capped=False)
        assert pacer.blocked("w", "DUE")
        step_s = 0
        while pacer.blocked("w", "DUE"):
            now[0] += 1
            step_s += 1
        delays.append(step_s)
    assert delays == [30, 60, 120, 240, 300, 300] and max(delays) == BACKOFF_CAP_S
    # Uncapped holds never reach the hourly cap.
    pacer.clear("w")
    for _ in range(20):
        pacer.record_hold("w", "IN_FLIGHT", "DUE", capped=False)
        now[0] += 400
        assert not pacer.blocked("w", "DUE")
    # Capped holds: 3 per hour, then blocked past the backoff until the hour ends.
    pacer.clear("w")
    start = now[0]
    for _ in range(HOLD_CAP_PER_HOUR):
        pacer.record_hold("w", "COLLISION", "DUE", capped=True)
        now[0] += 301
    assert pacer.blocked("w", "DUE") and pacer.holds_of("w") == HOLD_CAP_PER_HOUR
    now[0] = start + 3601
    assert not pacer.blocked("w", "DUE")  # the hour reset the counter
    # A derived-state change also resets the cap.
    pacer.clear("w")
    for _ in range(HOLD_CAP_PER_HOUR):
        pacer.record_hold("w", "COLLISION", "DUE", capped=True)
        now[0] += 301
    assert pacer.blocked("w", "DUE") and not pacer.blocked("w", "EXPIRED")


def test_pacer_seven_day_bound_for_a_permanent_no_row_hold() -> None:
    from suzka.r19.scheduler import WakePacer

    now = [0.0]
    pacer = WakePacer(lambda: now[0])
    events = 0
    while now[0] < 7 * 86_400:
        if not pacer.blocked("w", "DUE"):
            events += 1
            pacer.record_hold("w", "COLLISION", "DUE", capped=True)
        now[0] += 30
    assert 0 < events <= 3 * 24 * 7 + 3 < 2000


def test_a_chain_that_cannot_be_submitted_is_resumed_by_the_next_tick(sw) -> None:  # type: ignore[no-untyped-def]
    from suzka.runtime.agent_runtime import AgentRuntimeQueueFull

    state = sw.state
    wake = fx.materialize(state, sw.gid, sw.time)
    scheduler = make(sw)
    real = state.wake_review_owner._submit_review

    def full(claim_id):  # type: ignore[no-untyped-def]
        raise AgentRuntimeQueueFull(None)  # type: ignore[arg-type]

    state.wake_review_owner._submit_review = full  # type: ignore[method-assign]
    assert step(sw, scheduler).action is TickAction.CLAIM
    assert entry_of(sw, wake).state is WakeState.CLAIMED_LIVE
    state.wake_review_owner._submit_review = real  # type: ignore[method-assign]
    assert step(sw, scheduler).action is TickAction.REVIEW
    assert entry_of(sw, wake).state is WakeState.CLOSED
    assert len(journal_events(state, AgentEventType.R16_WAKE_REVIEW)) == 1


def test_stop_does_not_chain_a_review_and_the_next_start_resumes_it(sw) -> None:  # type: ignore[no-untyped-def]
    state = sw.state
    wake = fx.materialize(state, sw.gid, sw.time)
    scheduler = make(sw)
    assert scheduler.tick().action is TickAction.CLAIM
    scheduler.stop()  # the claim runs; nothing is chained after stop
    assert scheduler.wait_idle(30)
    settle(sw, scheduler)
    chained = len(journal_events(state, AgentEventType.R16_WAKE_REVIEW))
    assert chained in (0, 1)
    restarted = make(sw)
    step(sw, restarted)
    # Exactly one review in all, whether or not the first one was chained.
    assert len(journal_events(state, AgentEventType.R16_WAKE_REVIEW)) == 1
    assert entry_of(sw, wake).state is WakeState.CLOSED


# ------------------------------------------------ review round 2 (feeder, server)


@pytest.mark.parametrize("which", ["durability", "stopped"])
def test_a_feeder_that_hits_a_dead_runtime_frees_the_slot_and_keeps_the_thread(sw, which) -> None:  # type: ignore[no-untyped-def]
    from suzka.runtime.agent_runtime import (
        AgentRuntimeDurabilityError,
        AgentRuntimeStatus,
        AgentRuntimeStopped,
    )

    state = sw.state
    fx.materialize(state, sw.gid, sw.time)
    scheduler = make(sw)
    scheduler.start()
    try:
        error_type = AgentRuntimeDurabilityError if which == "durability" else AgentRuntimeStopped

        def dead(claim_id):  # type: ignore[no-untyped-def]
            if which == "durability":
                raise AgentRuntimeDurabilityError(None, None, outcome_indeterminate=False)  # type: ignore[arg-type]
            raise error_type(None)  # type: ignore[arg-type]

        state.wake_review_owner._submit_review = dead  # type: ignore[method-assign]
        scheduler._driver_active = lambda _i: True
        scheduler.nudge()
        deadline = real_time.monotonic() + 30
        while not scheduler.submitted.get(TickAction.CLAIM):
            assert real_time.monotonic() < deadline
            real_time.sleep(0.05)
        assert scheduler.wait_idle(30)
        assert scheduler.running
        scheduler._runtime = SimpleNamespace(  # type: ignore[assignment]
            status=AgentRuntimeStatus.FAILED, is_idle=lambda: False
        )
        assert scheduler.tick().action is TickAction.NOT_ACCEPTING
        assert scheduler.running
    finally:
        scheduler.stop()


def test_a_settle_that_raises_still_releases_the_slot(sw) -> None:  # type: ignore[no-untyped-def]
    state = sw.state
    fx.materialize(state, sw.gid, sw.time)
    scheduler = make(sw)

    def broken(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("settle failed")

    scheduler._settle = broken  # type: ignore[method-assign]
    assert scheduler.tick().action is TickAction.CLAIM
    assert scheduler.wait_idle(30)  # _done's finally released the slot
    assert scheduler.tick().action is not TickAction.BUSY


def test_an_expired_newer_wake_does_not_supersede(sw) -> None:  # type: ignore[no-untyped-def]
    state, clock = sw.state, sw.time
    sw.monkeypatch.setattr(scheduler_module, "may_supersede", lambda old, new: True)
    old = fx.materialize(state, sw.gid, clock, not_before=-100)
    expired = fx.materialize(
        state, sw.gid, clock, cause_event_id=str(uuid4()), evidence_fingerprint="9" * 64,
        expires=-5, not_before=-4000,
    )
    scheduler = make(sw)

    def candidate(newer):  # type: ignore[no-untyped-def]
        view = fx.view(state, clock.now())
        read = SimpleNamespace(view=view)
        found = scheduler._superseder(read, view.get(old.wakeup_id))  # type: ignore[arg-type]
        return None if found is None else found.wakeup_id

    assert fx.view(state, clock.now()).get(expired.wakeup_id).state is WakeState.EXPIRED
    assert candidate(expired) is None
    live = fx.materialize(
        state, sw.gid, clock, cause_event_id=str(uuid4()), evidence_fingerprint="8" * 64
    )
    assert candidate(live) == live.wakeup_id


def test_a_failing_scheduler_construction_or_start_never_blocks_the_app(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from suzka.api import server as server_module

    class Broken:
        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("cannot build")

    monkeypatch.setattr(server_module, "WakeScheduler", Broken)
    with TestClient(_app(fx.settings(tmp_path / "bc"))) as client:
        state = client.app.state
        assert state.wake_scheduler is None
        assert state.r16_driver._on_cause_terminal is None
        assert client.get("/health").status_code == 200
    monkeypatch.undo()

    class NoStart(WakeScheduler):
        def start(self) -> None:
            raise RuntimeError("cannot start")

    monkeypatch.setattr(server_module, "WakeScheduler", NoStart)
    with TestClient(_app(fx.settings(tmp_path / "bs"))) as client:
        assert client.app.state.wake_scheduler is None
        assert client.get("/health").status_code == 200
