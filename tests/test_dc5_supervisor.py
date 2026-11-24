"""DC-5: the in-process subject supervisor (F15 slice D1 r3 sec. 4.1, 5.3, 10).

Two layers.  The scripted-cycle tests drive ``SubjectSupervisor`` with stub
cycles (a stub app, virtual time where a bound of minutes is the point) and
prove the state machine: restart only after the termination proof, the
terminal ``threads_not_stopped`` state, refusals that keep the process alive,
restart limits, SIGTERM in every phase.  The real-app tests run
``create_app(mode="continue")`` against a real Journal / Chroma / Runtime and
prove what the stubs cannot: nothing blocks the server loop, a failed start
leaks nothing, every cycle gets fresh identities, the legacy path is
untouched.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import gc
import logging
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import textwrap
import threading
import time
from types import SimpleNamespace
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from starlette.datastructures import State

import suzka.api.server as server
from dc_kit import FakeTimebase
from suzka.api.supervisor import (
    EXIT_CODE_THREADS_NOT_STOPPED,
    SubjectPhaseMiddleware,
    SubjectSupervisor,
    ReadyReport,
    EXIT_CODE_BOOT_TIMEOUT,
    failed_to_serving_bound_s,
    normal_failed_to_serving_s,
    phase_detail,
)
from suzka.config import Settings, load_settings
from suzka.config.schema import DegradedContinuationSettings
from suzka.models import DummyProvider
from suzka.r16.history_store import SubjectHistoryStore
from suzka.r18.sweep import GateState, SweepGate
from suzka.runtime import (
    AgentEventSource,
    AgentEventType,
    AgentRuntime,
    AgentRuntimeDurabilityPhase,
    AgentRuntimeStatus,
    EventJournalLease,
)
from suzka.runtime.degraded_continuation import (
    CycleRestart,
    DcState,
    DegradedContinuationStatus,
    IntegrityVerdict,
    classify_cycle_exception,
)
from suzka.runtime.event_journal import EventJournalLoadError
from suzka.runtime.startup_reconciliation import StartupReconciliationError
from suzka.runtime.thread_liveness import ThreadLedger

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"
SERVING = DcState.SERVING


def wait_until(
    predicate: Callable[[], bool], timeout: float = 10.0, what: str = "condition"
) -> None:
    end = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > end:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.002)


# ===================================================================== stubs


class FakeRuntime:
    """The part of ``AgentRuntime`` the supervisor reads."""

    def __init__(self) -> None:
        self.status = AgentRuntimeStatus.ACCEPTING
        self._callback: Callable[[Any], None] | None = None

    def set_on_failed(self, callback: Callable[[Any], None] | None) -> None:
        self._callback = callback
        if callback is not None and self.status is AgentRuntimeStatus.FAILED:
            callback(AgentRuntimeDurabilityPhase.TERMINAL_COMPLETION)

    def fail(self, *, notify: bool = True) -> None:
        self.status = AgentRuntimeStatus.FAILED
        if notify and self._callback is not None:
            self._callback(AgentRuntimeDurabilityPhase.TERMINAL_COMPLETION)

    def threads_alive(self) -> tuple[str, ...]:
        return ()


class ZombieOwner:
    """A thread owner whose thread ignores ``stop()`` (an R18 executor worker
    blocked in a job).  ``threads_alive`` is the real ``ThreadLedger``."""

    def __init__(self, name: str = "suzka-r18-executor") -> None:
        self.release = threading.Event()
        self.ledger = ThreadLedger()
        self.thread = threading.Thread(
            target=lambda: self.release.wait(60), name=name, daemon=True
        )
        self.thread.start()
        self.ledger.register(self.thread)

    def threads_alive(self) -> tuple[str, ...]:
        return self.ledger.alive()

    def stop(self) -> None:  # ignored on purpose
        pass

    def free(self) -> None:
        self.release.set()
        self.thread.join(5)


@dataclass
class Step:
    """One scripted cycle.  ``build`` runs on the cycle thread before the
    cycle is ready (raise to refuse); ``teardown`` runs on the same thread
    after the stop request."""

    build: Callable[[Any, "ScriptedCycles", int], None] | None = None
    teardown: Callable[[Any, "ScriptedCycles", int], None] | None = None
    serving: bool = True


def _install_runtime(app: Any, cycles: "ScriptedCycles", serving: bool) -> None:
    runtime = FakeRuntime()
    if not serving:
        runtime.status = AgentRuntimeStatus.CREATED
    app.state.agent_runtime = runtime
    cycles.runtimes.append(runtime)


@dataclass
class ScriptedCycles:
    script: list[Step]
    launches: int = 0
    active: int = 0
    max_active: int = 0
    runtimes: list[FakeRuntime] = field(default_factory=list)
    teardowns: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __call__(self, app: Any) -> Any:
        return self._cycle(app)

    @asynccontextmanager
    async def _cycle(self, app: Any) -> AsyncIterator[None]:
        with self._lock:
            self.launches += 1
            n = self.launches
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        step = self.script[min(n - 1, len(self.script) - 1)]
        try:
            if step.build is not None:
                step.build(app, self, n)
            _install_runtime(app, self, step.serving)
            try:
                yield
            finally:
                try:
                    if step.teardown is not None:
                        step.teardown(app, self, n)
                finally:
                    with self._lock:
                        self.teardowns += 1
        finally:
            with self._lock:
                self.active -= 1


def ok() -> Step:
    return Step()


def raises(factory: Callable[[], BaseException]) -> Step:
    def build(app: Any, cycles: ScriptedCycles, n: int) -> None:
        raise factory()

    return Step(build=build)


def stub_app() -> Any:
    app = SimpleNamespace(state=State())
    app.state.settings = object()  # a baseline key, kept across cycles
    return app


def ready_report(app: Any) -> ReadyReport:
    runtime = app.state.agent_runtime
    if runtime.status is AgentRuntimeStatus.ACCEPTING:
        return ReadyReport(serving=True)
    return ReadyReport(serving=False, reason="external_reconciliation_required")


def make(
    script: list[Step],
    *,
    tb: FakeTimebase | None = None,
    restart_delay: float = 0.01,
    reeval_delay: float = 0.01,
    exit_fn: Callable[[int], None] | None = None,
    app: Any = None,
    term_grace_s: float = 0.05,
    **settings: Any,
) -> tuple[SubjectSupervisor, ScriptedCycles, DegradedContinuationStatus, Any]:
    app = app or stub_app()
    cycles = ScriptedCycles(script)
    status = DegradedContinuationStatus()
    config = DegradedContinuationSettings(mode="continue", **settings)
    kwargs: dict[str, Any] = {}
    if tb is not None:
        kwargs["timebase"] = tb
    if exit_fn is not None:
        kwargs["exit_fn"] = exit_fn
    supervisor = SubjectSupervisor(
        app,
        config,
        status,
        cycle_factory=cycles,
        ready_report=ready_report,
        restart_delay_fn=lambda n: restart_delay,
        reeval_delay_fn=lambda n: reeval_delay,
        poll_s=0.01,
        term_grace_s=term_grace_s,
        **kwargs,
    )
    return supervisor, cycles, status, app


@pytest.fixture
def supervisors() -> Any:
    created: list[SubjectSupervisor] = []

    def register(supervisor: SubjectSupervisor) -> SubjectSupervisor:
        created.append(supervisor)
        return supervisor

    yield register
    for supervisor in created:
        supervisor.shutdown(timeout=2.0)


def phase_is(supervisor: SubjectSupervisor, *states: DcState) -> Callable[[], bool]:
    return lambda: supervisor.phase in states


# ============================================================ state machine


def test_first_cycle_serves_and_a_runtime_failure_restarts_with_a_new_instance(
    supervisors: Any,
) -> None:
    sup, cycles, status, app = make([ok(), ok()])
    supervisors(sup).start()
    wait_until(phase_is(sup, SERVING), what="first SERVING")
    first = app.state.agent_runtime
    first.fail()
    wait_until(lambda: cycles.launches == 2 and sup.phase is SERVING, what="restart")
    assert app.state.agent_runtime is not first
    assert first.status is AgentRuntimeStatus.FAILED  # the old one stays FAILED
    assert cycles.max_active == 1  # no second cycle while the first lived
    assert cycles.teardowns == 1
    assert status.snapshot()["dc_state"] == "SERVING"


def test_a_failure_the_poll_sees_without_a_callback_also_restarts(
    supervisors: Any,
) -> None:
    sup, cycles, _status, app = make([ok(), ok()])
    supervisors(sup).start()
    wait_until(phase_is(sup, SERVING))
    app.state.agent_runtime.fail(notify=False)
    wait_until(lambda: cycles.launches == 2 and sup.phase is SERVING)


@pytest.mark.parametrize(
    "factory, label, verdict",
    [
        (
            lambda: EventJournalLoadError("EventJournal exclusive authority is unavailable"),
            "event_journal_lease_busy",
            IntegrityVerdict.TRANSIENT,
        ),
        (lambda: OSError(5, "EIO"), "builtins.OSError", IntegrityVerdict.TRANSIENT),
        (
            lambda: StartupReconciliationError("ambiguous"),
            "suzka.runtime.startup_reconciliation.StartupReconciliationError",
            IntegrityVerdict.AMBIGUOUS,
        ),
        (lambda: RuntimeError("x"), "builtins.RuntimeError", IntegrityVerdict.PROTECTED_LOST),
        (lambda: ValueError("x"), "builtins.ValueError", IntegrityVerdict.PROTECTED_LOST),
        (lambda: AssertionError("x"), "builtins.AssertionError", IntegrityVerdict.PROTECTED_LOST),
        (lambda: MemoryError(), "builtins.MemoryError", IntegrityVerdict.PROTECTED_LOST),
        (
            lambda: EventJournalLoadError("corrupt tail"),
            "suzka.runtime.event_journal.EventJournalLoadError",
            IntegrityVerdict.PROTECTED_LOST,
        ),
    ],
)
def test_a_first_start_refusal_is_classified_and_the_process_stays_alive(
    supervisors: Any,
    factory: Callable[[], BaseException],
    label: str,
    verdict: IntegrityVerdict,
) -> None:
    assert classify_cycle_exception(factory()).verdict is verdict
    sup, cycles, status, _app = make([raises(factory), ok()], reeval_delay=0.2)
    supervisors(sup).start()
    wait_until(phase_is(sup, DcState.REFUSED_RUNNING), what="REFUSED_RUNNING")
    snapshot = status.snapshot()
    assert snapshot["dc_reason"] == label
    assert phase_detail(sup.phase) == "subject_refused"
    # alive and re-evaluating: the next attempt succeeds without anyone acting
    wait_until(phase_is(sup, SERVING), what="self-heal")
    assert cycles.launches == 2


def test_an_unknown_exception_class_is_named_never_undefined(
    supervisors: Any,
) -> None:
    class Strange(Exception):
        pass

    sup, _cycles, status, _app = make([raises(lambda: Strange("x"))], reeval_delay=5)
    supervisors(sup).start()
    wait_until(phase_is(sup, DcState.REFUSED_RUNNING))
    assert status.snapshot()["dc_reason"].endswith("Strange")


def test_a_refusal_loop_stays_alive_and_keeps_re_evaluating(supervisors: Any) -> None:
    sup, cycles, _status, _app = make([raises(lambda: RuntimeError("no"))])
    supervisors(sup).start()
    wait_until(lambda: cycles.launches >= 6, what="six re-evaluations")
    assert sup.phase in (DcState.REFUSED_RUNNING, DcState.RESTARTING)
    assert not sup.terminal
    assert cycles.max_active == 1


def test_a_refusal_at_restart_keeps_the_process_alive_and_re_evaluates(
    supervisors: Any,
) -> None:
    sup, cycles, status, _app = make(
        [ok(), raises(lambda: OSError(5, "EIO")), ok()], reeval_delay=0.2
    )
    exits: list[int] = []
    sup._exit_fn = exits.append
    supervisors(sup).start()
    wait_until(phase_is(sup, SERVING))
    sup._state.agent_runtime.fail()
    wait_until(phase_is(sup, DcState.REFUSED_RUNNING), what="restart refusal")
    assert status.snapshot()["dc_reason"] == "builtins.OSError"
    wait_until(lambda: cycles.launches == 3 and sup.phase is SERVING)
    assert exits == []  # refusals never exit the process


@pytest.mark.parametrize("signal_error", [KeyboardInterrupt, SystemExit])
def test_exit_signals_inside_a_cycle_are_a_visible_re_evaluated_refusal(
    supervisors: Any, signal_error: type[BaseException]
) -> None:
    sup, cycles, status, _app = make([raises(lambda: signal_error()), ok()], reeval_delay=0.2)
    supervisors(sup).start()
    wait_until(phase_is(sup, DcState.REFUSED_RUNNING))
    assert status.snapshot()["dc_reason"] == f"cycle_exit_requested:{signal_error.__name__}"
    wait_until(phase_is(sup, SERVING))  # not a dead supervisor
    assert cycles.launches == 2


def test_an_error_in_the_supervisor_itself_still_proves_the_stop_before_the_next_cycle() -> None:
    holder: dict[str, ZombieOwner] = {}
    calls = {"n": 0}

    def flaky_ready(app: Any) -> ReadyReport:
        calls["n"] += 1
        raise RuntimeError("ready report failed")

    sup, cycles, status, _app = make([_zombie_step(holder), ok()], t_teardown_max_s=0.2)
    sup._ready_report = flaky_ready
    sup.start()
    try:
        wait_until(lambda: sup.terminal, what="terminal: the zombie was not stopped")
        assert cycles.launches == 1  # no second cycle on top of a live thread
        assert status.snapshot()["dc_reason"].startswith("threads_not_stopped:")
    finally:
        holder["owner"].free()
        sup.shutdown(timeout=2.0)


def test_a_ready_cycle_without_a_started_runtime_is_reconciling_and_re_evaluated(
    supervisors: Any,
) -> None:
    sup, cycles, status, _app = make(
        [Step(serving=False), ok()], reeval_delay=0.15
    )
    supervisors(sup).start()
    wait_until(phase_is(sup, DcState.RECONCILING), what="RECONCILING")
    assert status.snapshot()["dc_reason"] == "external_reconciliation_required"
    assert phase_detail(sup.phase) == "subject_reconciling"
    wait_until(phase_is(sup, SERVING), what="re-evaluation reaches SERVING")
    assert cycles.launches == 2 and cycles.teardowns == 1


def test_restart_limit_then_one_attempt_per_re_evaluation(supervisors: Any) -> None:
    sup, cycles, status, app = make([ok()], reeval_delay=0.4)
    supervisors(sup).start()
    for expected in (1, 2):
        wait_until(lambda: cycles.launches == expected and sup.phase is SERVING)
        app.state.agent_runtime.fail()
    wait_until(lambda: cycles.launches == 3 and sup.phase is SERVING)
    app.state.agent_runtime.fail()  # the third failure inside the window
    wait_until(phase_is(sup, DcState.RESTART_LIMIT), what="RESTART_LIMIT")
    assert status.snapshot()["dc_reason"] == "restart_limit"
    launches = cycles.launches
    time.sleep(0.1)
    assert cycles.launches == launches  # no restart while the limit holds
    wait_until(lambda: cycles.launches == launches + 1 and sup.phase is SERVING)


def test_t_stable_serving_resets_the_restart_counter() -> None:
    tb = FakeTimebase()
    sup, cycles, _status, app = make([ok()], tb=tb, t_stable_s=600.0)
    sup.start()
    try:
        wait_until(phase_is(sup, SERVING))
        app.state.agent_runtime.fail()  # failure 1
        wait_until(lambda: cycles.launches == 2 and sup.phase is SERVING)
        app.state.agent_runtime.fail()  # failure 2
        wait_until(lambda: cycles.launches == 3 and sup.phase is SERVING)
        tb.sleep(700)  # stable for longer than T_stable
        app.state.agent_runtime.fail()  # would be failure 3, but the counter reset
        wait_until(lambda: cycles.launches == 4 and sup.phase is SERVING)
        assert sup.phase is SERVING
    finally:
        sup.shutdown(timeout=2.0)


def test_cycle_restart_is_not_a_failure_but_is_capped() -> None:
    sup, cycles, status, _app = make(
        [raises(lambda: CycleRestart("dc4_quarantine"))], restart_delay=0.01
    )
    sup.start()
    try:
        wait_until(phase_is(sup, DcState.RECONCILING), what="cap reached")
        assert status.snapshot()["dc_reason"] == "cycle_restart_loop"
        assert cycles.launches == 4  # first + cycle_restart_cap (3) restarts
        assert sup.phase is not DcState.RESTART_LIMIT  # never counted as failures
    finally:
        sup.shutdown(timeout=2.0)


def test_one_cycle_restart_then_a_normal_cycle_never_shows_a_refusal(
    supervisors: Any,
) -> None:
    seen: list[DcState] = []
    sup, cycles, _status, _app = make([raises(lambda: CycleRestart("r")), ok()])
    original = sup._publish

    def spy(state: DcState, reason: str | None = None) -> None:
        seen.append(state)
        original(state, reason)

    sup._publish = spy  # type: ignore[method-assign]
    supervisors(sup).start()
    wait_until(phase_is(sup, SERVING))
    assert DcState.REFUSED_RUNNING not in seen and cycles.launches == 2


def test_the_app_state_is_reset_between_cycles_and_the_baseline_is_kept(
    supervisors: Any,
) -> None:
    def build(app: Any, cycles: ScriptedCycles, n: int) -> None:
        app.state.event_journal = object()
        app.state.main_loop = object()

    sup, cycles, _status, app = make([Step(build=build), Step(build=build)])
    settings_before = app.state.settings
    supervisors(sup).start()
    wait_until(phase_is(sup, SERVING))
    first = (app.state.event_journal, app.state.main_loop)
    app.state.agent_runtime.fail()
    wait_until(lambda: cycles.launches == 2 and sup.phase is SERVING)
    assert app.state.event_journal is not first[0]
    assert app.state.main_loop is not first[1]
    assert app.state.settings is settings_before
    assert set(app.state._state) == {
        "settings",
        "event_journal",
        "main_loop",
        "agent_runtime",
    }


# ===================================================== the termination proof


def _zombie_step(holder: dict[str, ZombieOwner]) -> Step:
    def build(app: Any, cycles: ScriptedCycles, n: int) -> None:
        owner = ZombieOwner()
        holder["owner"] = owner
        app.state.r18_executor = owner
        gate = SweepGate(enabled=True)
        gate.open()
        app.state.r18_gate = gate

    return Step(build=build)


def test_zombie_executor_means_no_in_process_restart_and_a_terminal_refusal() -> None:
    holder: dict[str, ZombieOwner] = {}
    exits: list[int] = []
    sup, cycles, status, app = make(
        [_zombie_step(holder), ok()],
        t_teardown_max_s=0.2,
        exit_fn=exits.append,
    )
    sup.start()
    try:
        wait_until(phase_is(sup, SERVING))
        gate = app.state.r18_gate
        assert gate.state is GateState.OPEN
        app.state.agent_runtime.fail()
        wait_until(lambda: sup.terminal, what="the terminal state")
        assert sup.phase is DcState.REFUSED_RUNNING
        reason = status.snapshot()["dc_reason"]
        assert reason.startswith("threads_not_stopped:")
        assert "suzka-r18-executor" in reason
        assert gate.state is not GateState.OPEN  # R18 closed
        time.sleep(0.3)
        assert cycles.launches == 1  # no second cycle, ever, in this process
        assert cycles.max_active == 1
        assert exits == []  # exit_on_threads_not_stopped is off by default
        # freeing the thread later does not resurrect the supervisor
        holder["owner"].free()
        time.sleep(0.1)
        assert cycles.launches == 1 and sup.terminal
    finally:
        holder["owner"].free()
        sup.shutdown(timeout=2.0)


def test_the_exit_happens_only_in_the_terminal_state_and_only_when_enabled() -> None:
    holder: dict[str, ZombieOwner] = {}
    exits: list[int] = []
    sup, cycles, _status, app = make(
        [_zombie_step(holder)],
        t_teardown_max_s=0.2,
        exit_on_threads_not_stopped=True,
        exit_fn=exits.append,
    )
    sup.start()
    try:
        wait_until(phase_is(sup, SERVING))
        # an ordinary restart exits nothing
        app.state.agent_runtime.fail()
        wait_until(lambda: sup.terminal)
        assert exits == [EXIT_CODE_THREADS_NOT_STOPPED]
    finally:
        holder["owner"].free()
        sup.shutdown(timeout=2.0)


def test_refusals_and_restarts_never_exit_even_when_the_exit_is_enabled(
    supervisors: Any,
) -> None:
    exits: list[int] = []
    sup, cycles, _status, app = make(
        [raises(lambda: RuntimeError("x")), ok(), ok()],
        exit_on_threads_not_stopped=True,
        exit_fn=exits.append,
    )
    supervisors(sup).start()
    wait_until(phase_is(sup, SERVING))
    app.state.agent_runtime.fail()
    wait_until(lambda: cycles.launches == 3 and sup.phase is SERVING)
    assert exits == []


def test_hard_exit_ends_the_process_non_zero_from_a_non_main_thread() -> None:
    script = textwrap.dedent(
        """
        import threading, time
        import sys
        sys.path.insert(0, %r)
        from suzka.api.supervisor import hard_exit, EXIT_CODE_THREADS_NOT_STOPPED
        t = threading.Thread(target=lambda: hard_exit(EXIT_CODE_THREADS_NOT_STOPPED))
        t.start()
        time.sleep(30)  # the main thread would live on
        """
    ) % str(ROOT)
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=ROOT,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == EXIT_CODE_THREADS_NOT_STOPPED


def test_a_teardown_of_140_virtual_seconds_is_not_threads_not_stopped() -> None:
    tb = FakeTimebase()

    def slow_teardown(app: Any, cycles: ScriptedCycles, n: int) -> None:
        tb.sleep(140.0)  # the bounded joins add up to 140 s (slow but exiting)

    def slow_build(app: Any, cycles: ScriptedCycles, n: int) -> None:
        tb.sleep(90.0)  # the budgeted T_cycle_build

    sup, cycles, _status, app = make(
        [Step(teardown=slow_teardown), Step(build=slow_build)], tb=tb, restart_delay=1.0
    )
    sup.start()
    try:
        wait_until(phase_is(sup, SERVING), timeout=30)
        app.state.agent_runtime.fail()
        wait_until(lambda: cycles.launches == 2 and sup.phase is SERVING, timeout=30)
        assert not sup.terminal
        elapsed = sup.last_failed_to_serving_s
        assert elapsed is not None
        # FAILED observed -> teardown (140) + back-off (1) + build (90), all
        # inside T_failed_to_serving_max, measured from the FAILED observation.
        assert 140 + 1 + 90 <= elapsed <= 400  # a 140 s teardown is still "normal"
        assert elapsed <= failed_to_serving_bound_s(sup._settings) + 1.0
    finally:
        sup.shutdown(timeout=2.0)


def test_the_stated_bounds_add_up() -> None:
    settings = DegradedContinuationSettings()
    # D1's 400 s is the NORMAL case (the bounded joins, 150 s); a slow-but-exiting
    # teardown may use the whole proof window: 240 + 30 + 1 + 240.
    assert normal_failed_to_serving_s(settings) == 391.0 <= 400.0
    assert failed_to_serving_bound_s(settings) == 511.0
    assert settings.t_teardown_max_s >= 150 + 90  # joins, then the two unbounded
    assert settings.t_boot_max_s == 240


def test_a_teardown_that_never_returns_ends_terminal_within_the_bound() -> None:
    tb = FakeTimebase()
    forever = threading.Event()

    def stuck_teardown(app: Any, cycles: ScriptedCycles, n: int) -> None:
        forever.wait(60)  # an unbounded join that never comes back

    sup, cycles, status, app = make(
        [Step(teardown=stuck_teardown), ok()], tb=tb, t_teardown_max_s=240.0
    )
    sup.start()
    try:
        wait_until(phase_is(sup, SERVING), timeout=30)
        started = tb.now()
        app.state.agent_runtime.fail()
        wait_until(lambda: sup.terminal, timeout=30)
        assert tb.now() - started <= 240 + 30 + 5  # T_teardown_max + T_term_grace
        assert "suzka-dc-cycle" in status.snapshot()["dc_reason"]
        assert cycles.launches == 1
    finally:
        forever.set()
        sup.shutdown(timeout=2.0)


def test_the_gate_is_closed_before_the_teardown_runs_and_there_is_one_cycle_at_a_time(
    supervisors: Any,
) -> None:
    seen: list[GateState] = []

    def build(app: Any, cycles: ScriptedCycles, n: int) -> None:
        gate = SweepGate(enabled=True)
        gate.open()
        app.state.r18_gate = gate

    def teardown(app: Any, cycles: ScriptedCycles, n: int) -> None:
        seen.append(app.state.r18_gate.state)  # the executor stops after this point

    sup, cycles, _status, app = make(
        [Step(build=build, teardown=teardown), Step(build=build, teardown=teardown)]
    )
    supervisors(sup).start()
    wait_until(phase_is(sup, SERVING))
    app.state.agent_runtime.fail()
    wait_until(lambda: cycles.launches == 2 and sup.phase is SERVING)
    assert seen == [GateState.SWEEPING]
    assert cycles.max_active == 1


def test_a_failed_start_still_proves_its_threads_before_the_next_cycle() -> None:
    holder: dict[str, ZombieOwner] = {}

    def build(app: Any, cycles: ScriptedCycles, n: int) -> None:
        owner = ZombieOwner()
        holder["owner"] = owner
        app.state.wake_scheduler = owner
        raise RuntimeError("start failed after a thread was started")

    sup, cycles, status, _app = make([Step(build=build), ok()], t_teardown_max_s=0.2)
    sup.start()
    try:
        wait_until(lambda: sup.terminal, what="terminal after a failed start")
        assert "suzka-r18-executor" in status.snapshot()["dc_reason"]
        assert cycles.launches == 1
    finally:
        holder["owner"].free()
        sup.shutdown(timeout=2.0)


# ============================================================ boot bound


def test_a_build_past_t_boot_max_is_visible_and_a_late_ready_is_accepted() -> None:
    release = threading.Event()

    def slow(app: Any, cycles: ScriptedCycles, n: int) -> None:
        release.wait(30)

    sup, cycles, status, _app = make([Step(build=slow)], t_boot_max_s=0.2)
    sup.start()
    try:
        wait_until(phase_is(sup, DcState.REFUSED_RUNNING), what="boot_timeout")
        assert status.snapshot()["dc_reason"] == "boot_timeout"
        release.set()
        wait_until(phase_is(sup, SERVING), what="the late cycle is accepted")
        assert cycles.launches == 1
    finally:
        release.set()
        sup.shutdown(timeout=2.0)


# ============================================================== SIGTERM


def _shutdown_is_bounded(sup: SubjectSupervisor, bound: float) -> None:
    started = time.monotonic()
    assert sup.shutdown() is True
    assert time.monotonic() - started < bound


def test_sigterm_during_the_first_build() -> None:
    release = threading.Event()
    sup, cycles, _status, _app = make(
        [Step(build=lambda a, c, n: release.wait(30))], t_shutdown_join_s=0.3
    )
    sup.start()
    try:
        wait_until(lambda: cycles.launches == 1 and cycles.active == 1)
        _shutdown_is_bounded(sup, 1.5)  # the stuck build does not hold shutdown
    finally:
        release.set()
    wait_until(lambda: cycles.active == 0)  # the late build tears itself down
    assert cycles.teardowns == 1


def test_sigterm_while_serving_tears_the_cycle_down_first() -> None:
    sup, cycles, _status, _app = make([ok()])
    sup.start()
    wait_until(phase_is(sup, SERVING))
    _shutdown_is_bounded(sup, 2.0)
    assert cycles.teardowns == 1 and cycles.active == 0


def test_sigterm_during_restarting_waits_for_the_running_teardown_only() -> None:
    entered = threading.Event()
    release = threading.Event()

    def slow(app: Any, cycles: ScriptedCycles, n: int) -> None:
        entered.set()
        release.wait(30)

    sup, cycles, _status, app = make([Step(teardown=slow), ok()], t_shutdown_join_s=0.4)
    sup.start()
    try:
        wait_until(phase_is(sup, SERVING))
        app.state.agent_runtime.fail()
        assert entered.wait(5)
        assert sup.phase is DcState.RESTARTING
        _shutdown_is_bounded(sup, 1.5)
        assert cycles.launches == 1  # no new cycle after the stop request
    finally:
        release.set()


@pytest.mark.parametrize(
    "script, phase",
    [
        ([raises(lambda: RuntimeError("x"))], DcState.REFUSED_RUNNING),
        ([Step(serving=False)], DcState.RECONCILING),
    ],
    ids=("backoff-after-refusal", "reconciling-wait"),
)
def test_sigterm_cancels_a_long_back_off(script: list[Step], phase: DcState) -> None:
    sup, cycles, _status, _app = make(script, reeval_delay=3600.0)
    sup.start()
    wait_until(phase_is(sup, phase))
    _shutdown_is_bounded(sup, 1.5)
    assert cycles.active == 0


def test_sigterm_cancels_the_restart_back_off_and_the_restart_limit_wait() -> None:
    sup, cycles, _status, app = make([ok()], restart_delay=3600.0, reeval_delay=3600.0)
    sup.start()
    wait_until(phase_is(sup, SERVING))
    app.state.agent_runtime.fail()
    wait_until(phase_is(sup, DcState.RESTARTING))
    time.sleep(0.05)
    _shutdown_is_bounded(sup, 1.5)
    assert cycles.launches == 1


def test_sigterm_in_the_terminal_state() -> None:
    holder: dict[str, ZombieOwner] = {}
    sup, _cycles, _status, app = make([_zombie_step(holder)], t_teardown_max_s=0.1)
    sup.start()
    try:
        wait_until(phase_is(sup, SERVING))
        app.state.agent_runtime.fail()
        wait_until(lambda: sup.terminal)
        _shutdown_is_bounded(sup, 1.5)
    finally:
        holder["owner"].free()


def test_sigterm_in_the_restart_limit_state() -> None:
    sup, cycles, _status, app = make([ok()], reeval_delay=3600.0)
    sup.start()
    for expected in (1, 2, 3):
        wait_until(lambda: cycles.launches == expected and sup.phase is SERVING)
        app.state.agent_runtime.fail()
    wait_until(phase_is(sup, DcState.RESTART_LIMIT))
    _shutdown_is_bounded(sup, 1.5)


def test_the_supervisor_can_be_started_again_after_shutdown() -> None:
    sup, cycles, _status, _app = make([ok()])
    sup.start()
    wait_until(phase_is(sup, SERVING))
    assert sup.shutdown()
    sup.start()
    try:
        wait_until(lambda: cycles.launches == 2 and sup.phase is SERVING)
    finally:
        sup.shutdown(timeout=2.0)


# ===================================================== middleware / probes


class _PhaseHolder:
    def __init__(self, phase: DcState) -> None:
        self.phase = phase


@pytest.mark.parametrize("phase", list(DcState))
def test_phase_middleware_503s_everything_but_health_while_not_serving(
    phase: DcState,
) -> None:
    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"ok": "1"}

    @app.get("/health/dc")
    def health_dc() -> dict[str, str]:
        return {"dc": "1"}

    @app.get("/api/anything")
    def anything() -> dict[str, str]:
        return {"api": "1"}

    app.add_middleware(SubjectPhaseMiddleware, supervisor=_PhaseHolder(phase))
    client = TestClient(app)
    assert client.get("/health").status_code == 200  # in every phase
    assert client.get("/health/dc").status_code == 200
    response = client.get("/api/anything")
    if phase is SERVING:
        assert response.status_code == 200
    else:
        assert response.status_code == 503
        assert response.json() == {"detail": phase_detail(phase)}
        assert phase_detail(phase).startswith("subject_")


# ================================================= counts over many cycles


def test_threads_handlers_and_signals_are_constant_over_many_cycles() -> None:
    sup, cycles, _status, app = make(
        [ok()], restart_delay=0.0, reeval_delay=0.0, restart_limit=1000
    )
    root_handlers = len(logging.getLogger().handlers)
    sigterm = signal.getsignal(signal.SIGTERM)
    threads_before = {t.name for t in threading.enumerate()}
    sup.start()
    try:
        for expected in range(1, 7):
            wait_until(lambda: cycles.launches == expected and sup.phase is SERVING)
            if expected < 6:
                app.state.agent_runtime.fail()
    finally:
        sup.shutdown(timeout=3.0)
    assert len(logging.getLogger().handlers) == root_handlers
    assert signal.getsignal(signal.SIGTERM) == sigterm
    wait_until(
        lambda: {t.name for t in threading.enumerate()} <= threads_before,
        what="every supervisor / cycle thread gone",
    )


# ====================================================== architecture checks


def test_the_supervisor_module_never_opens_a_subject_history_store() -> None:
    tree = ast.parse((ROOT / "suzka" / "api" / "supervisor.py").read_text())
    names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    } | {
        node.attr if isinstance(node, ast.Attribute) else node.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.Name, ast.Attribute))
    }
    assert "open_or_create_store" not in names


def test_every_thread_creating_site_has_a_threads_alive_owner() -> None:
    sites: dict[str, int] = {}
    pattern = re.compile(
        r"(?<![A-Za-z_])(?:threading\.)?(Thread|Timer)\(|ThreadPoolExecutor|"
        r"to_thread\(|run_in_executor\(|multiprocessing|subprocess\."
    )
    for path in sorted((ROOT / "suzka").rglob("*.py")):
        count = 0
        for line in path.read_text().splitlines():
            code = line.split("#", 1)[0]
            if pattern.search(code) and not code.lstrip().startswith(("class ", "def ")):
                count += 1
        if count:
            sites[path.relative_to(ROOT).as_posix()] = count
    # The supervisor's own control / cycle threads are proven by the cycle-thread
    # check of the proof; server.py's single run_in_executor is the supervised
    # lifespan's shutdown call (it runs supervisor.shutdown, no owned thread).
    assert sites.pop("suzka/api/supervisor.py") == 2
    assert sites.pop("suzka/api/server.py") == 1
    # Pinned per file WITH counts: a new site (or a new kind of site) in an
    # owner fails here until it is registered in that owner's ThreadLedger.
    assert sites == {
        "suzka/runtime/agent_runtime.py": 1,
        "suzka/runtime/emotion_timer.py": 1,
        "suzka/r16/driver.py": 1,
        "suzka/r19/scheduler.py": 1,
        "suzka/r18/approval_owner.py": 2,
        "suzka/r18/executor.py": 2,
        # DC-6: the HARD-band watcher (found in app.state by threads_alive()).
        "suzka/runtime/hard_band_watcher.py": 1,
    }, (
        "a new thread site needs an owner with threads_alive() that lives in "
        "app.state (the termination proof finds owners by that method)"
    )


# --------------------------------------------- provider reuse / boot / launch


class _Provider:
    def __init__(self) -> None:
        self.released = False

    def release(self) -> None:
        self.released = True


def _provider_step(made: list[_Provider], *, serving: bool = True) -> Step:
    def build(app: Any, cycles: ScriptedCycles, n: int) -> None:
        provider = getattr(app.state, "model_provider", None)
        if provider is None:
            provider = _Provider()
            made.append(provider)
            app.state.model_provider = provider

    return Step(build=build, serving=serving)


def test_a_provider_of_a_cycle_that_never_served_is_reused_by_the_re_evaluation(
    supervisors: Any,
) -> None:
    made: list[_Provider] = []
    sup, cycles, _status, _app = make(
        [_provider_step(made, serving=False), _provider_step(made, serving=False), _provider_step(made)]
    )
    supervisors(sup).start()
    wait_until(phase_is(sup, SERVING))
    assert cycles.launches == 3
    assert len(made) == 1  # one model load for three cycles
    assert not made[0].released


def test_a_provider_that_served_is_released_and_rebuilt() -> None:
    made: list[_Provider] = []
    sup, cycles, _status, app = make([_provider_step(made)])
    sup.start()
    try:
        wait_until(phase_is(sup, SERVING))
        app.state.agent_runtime.fail()
        wait_until(lambda: cycles.launches == 2 and sup.phase is SERVING)
        assert len(made) == 2 and made[0].released and not made[1].released
        assert app.state.model_provider is made[1]
    finally:
        sup.shutdown(timeout=2.0)


def test_a_refused_start_keeps_the_provider_a_later_cycle_needs_not_rebuilt(
    supervisors: Any,
) -> None:
    made: list[_Provider] = []
    refuse_after = _provider_step(made)
    refusing = Step(
        build=lambda app, c, n: (
            refuse_after.build(app, c, n),  # type: ignore[misc]
            (_ for _ in ()).throw(RuntimeError("after the model load")),
        )
    )
    sup, cycles, _status, _app = make([refusing, refusing, _provider_step(made)])
    supervisors(sup).start()
    wait_until(phase_is(sup, SERVING))
    assert len(made) == 1  # not reloaded on each re-evaluation


def test_exit_on_boot_timeout_is_opt_in(supervisors: Any) -> None:
    for enabled in (False, True):
        release = threading.Event()
        exits: list[int] = []
        sup, _cycles, _status, _app = make(
            [Step(build=lambda a, c, n, r=release: r.wait(30))],
            t_boot_max_s=0.2,
            exit_on_boot_timeout=enabled,
            exit_fn=exits.append,
        )
        sup.start()
        try:
            wait_until(phase_is(sup, DcState.REFUSED_RUNNING))
            time.sleep(0.1)
            assert exits == ([EXIT_CODE_BOOT_TIMEOUT] if enabled else [])
        finally:
            release.set()
            sup.shutdown(timeout=2.0)


def test_thread_exhaustion_at_launch_is_a_refusal_not_a_long_wait(
    supervisors: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_start = threading.Thread.start
    state = {"failed": False}

    def start(self: threading.Thread) -> None:
        if self.name == "suzka-dc-cycle" and not state["failed"]:
            state["failed"] = True
            raise RuntimeError("can't start new thread")
        real_start(self)

    monkeypatch.setattr(threading.Thread, "start", start)
    sup, cycles, status, _app = make([ok()], reeval_delay=0.3, t_teardown_max_s=30.0)
    supervisors(sup).start()
    wait_until(phase_is(sup, DcState.REFUSED_RUNNING), timeout=3)
    assert status.snapshot()["dc_reason"] == "builtins.RuntimeError"
    assert not sup.terminal
    wait_until(phase_is(sup, SERVING), timeout=5)


# ------------------------------------------ real R18 executor: fencing, zombie


def _real_target(tmp_path: Path) -> Any:
    from test_r18_ip15_target_verifier import RUNTIME, _forward, _store

    store = _store(tmp_path)
    intent, challenge, claim = _forward(store)
    return store, intent, challenge, claim, RUNTIME


def test_dc5_zombie_executor_cannot_write_after_rotation(tmp_path: Path) -> None:
    from dataclasses import replace as _replace

    from suzka.r18.contracts import ExecutorResult
    from suzka.r18.target import TargetStore, new_runtime_instance_id

    store, intent, challenge, claim, _old = _real_target(tmp_path)
    successor = TargetStore.open(store.path)
    new_id = new_runtime_instance_id()
    successor.rotate_runtime_instance(new_id)  # the new cycle's S1, first
    now = lambda: __import__("datetime").datetime(2026, 11, 16, 0, 6, tzinfo=__import__("datetime").timezone.utc)  # noqa: E731
    # the old cycle's executor wakes up late with its in-flight claim
    late = store.forward_write(intent, challenge, claim, now=now)
    assert late.executor_result is ExecutorResult.CAS_REJECTED
    assert store.status(intent.obligation_group_key).artifact is None  # no effect
    # the new cycle writes it once under its own id
    fresh = _replace(claim, runtime_instance_id=new_id)
    assert successor.forward_write(intent, challenge, fresh, now=now).executor_result is (
        ExecutorResult.APPLIED
    )
    # and the zombie still cannot add a second one
    again = store.forward_write(intent, challenge, claim, now=now)
    assert again.executor_result is ExecutorResult.CAS_REJECTED
    active = successor.status(intent.obligation_group_key).artifact
    assert active is not None and active.revision == 1
    successor.close()
    store.close()


def test_dc5_stuck_thread_ends_refused_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A REAL threaded R18 executor whose worker is blocked inside a job: the
    legacy teardown abandons it after its join timeout, the thread proof sees
    it, the state is terminal and no second cycle (no second executor) starts."""

    from suzka.r18 import executor as executor_module
    from suzka.r18.executor import R18Executor
    from test_r18_ip18a_executor import ME, _World
    from suzka.r16.history_reader import AdoptedRecords

    monkeypatch.setattr(executor_module, "JOIN_TIMEOUT_SECONDS", 0.2)
    world = _World(tmp_path)
    entered = threading.Event()
    release = threading.Event()

    def blocked_submit(claim_id: str) -> Any:
        entered.set()
        release.wait(60)
        raise RuntimeError("released")

    world.outcome._submit_outcome = blocked_submit  # type: ignore[method-assign]
    executors: list[R18Executor] = []

    def build(app: Any, cycles: ScriptedCycles, n: int) -> None:
        executor = R18Executor(
            target_path=world.path,
            runtime_instance_id=ME,
            gate=world.gate,
            outcome_owner=world.outcome,
            read_records=lambda: AdoptedRecords(world.blocked, tuple(world.records)),
            has_exclusive_authority=lambda: True,
        )
        executors.append(executor)
        app.state.r18_executor = executor
        app.state.r18_gate = world.gate
        executor.start()
        executor.enqueue(world.claim)

    def teardown(app: Any, cycles: ScriptedCycles, n: int) -> None:
        app.state.r18_executor.stop()  # what the legacy teardown does

    sup, cycles, status, app = make(
        [Step(build=build, teardown=teardown), ok()], t_teardown_max_s=1.0
    )
    sup.start()
    try:
        wait_until(phase_is(sup, SERVING))
        assert entered.wait(20), "the worker never entered the blocked job"
        app.state.agent_runtime.fail()
        wait_until(lambda: sup.terminal, timeout=20, what="terminal")
        assert "suzka-r18-executor" in status.snapshot()["dc_reason"]
        assert world.gate.state is not GateState.OPEN  # R18 closed
        time.sleep(0.3)
        assert cycles.launches == 1 and len(executors) == 1  # no second executor
    finally:
        release.set()
        sup.shutdown(timeout=3.0)
        world.close()


# ====================================================== real app: helpers


def _settings(tmp_path: Path, **dc: Any) -> Settings:
    tmp_path.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp_path.chmod(0o700)
    settings = load_settings(CONFIG_PATH)
    return settings.model_copy(
        update={
            "memory": settings.memory.model_copy(
                update={
                    "persist_directory": tmp_path / "chroma",
                    "db1_collection": "hippocampus_dc5_test",
                    "db2_collection": "cortex_dc5_test",
                }
            ),
            "sleep": settings.sleep.model_copy(
                update={"dream_dataset_path": tmp_path / "dreams" / "dream_dataset.jsonl"}
            ),
            "qlora": settings.qlora.model_copy(
                update={"output_dir": tmp_path / "adapters", "dry_run": True}
            ),
            "adapter_registry": settings.adapter_registry.model_copy(
                update={
                    "path": tmp_path / "adapter_registry.json",
                    "eval_result_dir": tmp_path / "eval_results",
                    "eval_sets": [],
                }
            ),
            "agent_state": settings.agent_state.model_copy(
                update={"path": tmp_path / "agent_state.json"}
            ),
            "event_journal": settings.event_journal.model_copy(
                update={"path": tmp_path / "event_journal.jsonl"}
            ),
            "state_wal": settings.state_wal.model_copy(
                update={"directory": tmp_path / "private" / "state_wal"}
            ),
            "subject_history": settings.subject_history.model_copy(
                update={
                    "enabled": True,
                    "path": tmp_path / "history" / "subject_history.sqlite3",
                }
            ),
            "degraded_continuation": settings.degraded_continuation.model_copy(
                update=dc
            ),
        }
    )


def _fast(app: FastAPI, delay: float = 0.25) -> SubjectSupervisor:
    """Test seam: speed the back-offs up (the real ones are 1, 4, 16, 60, 300 s)."""

    supervisor: SubjectSupervisor = app.state.dc_supervisor
    supervisor._restart_delay = lambda n: 0.05
    supervisor._reeval_delay = lambda n: delay
    return supervisor


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    control = SimpleNamespace(
        calls=0,
        behaviour=lambda: DummyProvider(),
    )

    def fake_load(settings: Settings) -> Any:
        control.calls += 1
        return control.behaviour()

    monkeypatch.setattr(server, "load_model_provider", fake_load)
    return control


def _fd_count() -> int:
    return len(os.listdir("/proc/self/fd"))


# ===================================================== real app: the cycle


def test_real_app_serves_restarts_after_a_runtime_failure_with_fresh_identities(
    tmp_path: Path, provider: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids: list[str] = []
    real_new_id = server.new_runtime_instance_id

    def spy() -> str:
        value = real_new_id()
        ids.append(value)
        return value

    monkeypatch.setattr(server, "new_runtime_instance_id", spy)
    app = server.create_app(_settings(tmp_path, mode="continue"))
    supervisor = _fast(app)
    with TestClient(app) as client:
        assert supervisor.wait_serving(120)
        assert client.get("/health").json() == {
            "status": "ok",
            "project": app.state.settings.project.name,
        }
        first_runtime = app.state.agent_runtime
        first_journal = app.state.event_journal
        first_owner = app.state.approval_owner
        first_boot = first_owner.boot_id
        # an injected durability failure: the STARTED append of the next event
        monkeypatch.setattr(
            first_journal,
            "append_started",
            lambda *a, **k: (_ for _ in ()).throw(OSError(28, "ENOSPC")),
        )
        future = first_runtime.submit(
            AgentEventType.CHAT, AgentEventSource.API_CHAT, lambda: None
        )
        with pytest.raises(Exception):
            future.result(10)
        wait_until(
            lambda: supervisor.cycles_launched == 2 and supervisor.phase is SERVING,
            timeout=120,
            what="the supervised restart",
        )
        second_runtime = app.state.agent_runtime
        assert second_runtime is not first_runtime
        assert second_runtime.status is AgentRuntimeStatus.ACCEPTING
        assert first_runtime.status is AgentRuntimeStatus.FAILED
        with pytest.raises(RuntimeError, match="cannot be restarted"):
            first_runtime.start()  # the old instance still refuses
        assert app.state.event_journal is not first_journal
        assert app.state.approval_owner is not first_owner
        assert app.state.approval_owner.boot_id != first_boot  # fresh boot_id
        assert len(ids) == 2 and ids[0] != ids[1]  # fresh runtime_instance_id
        assert client.get("/health").json()["status"] == "ok"
        assert first_runtime.threads_alive() == ()
        assert supervisor.last_failed_to_serving_s is not None
        assert supervisor.last_failed_to_serving_s <= failed_to_serving_bound_s(
            supervisor._settings
        )
    assert supervisor._thread is not None and not supervisor._thread.is_alive()


def test_real_app_never_blocks_the_server_loop_during_a_slow_model_load(
    tmp_path: Path, provider: SimpleNamespace
) -> None:
    def slow() -> DummyProvider:
        time.sleep(5.0)
        return DummyProvider()

    provider.behaviour = slow
    app = server.create_app(_settings(tmp_path, mode="continue"))
    supervisor = app.state.dc_supervisor
    with TestClient(app) as client:
        wait_until(lambda: provider.calls == 1, timeout=60, what="the model load")
        for _ in range(5):
            started = time.monotonic()
            health = client.get("/health")
            dc = client.get("/health/dc")
            chat = client.post("/api/chat", json={"message": "hi", "attachments": []})
            elapsed = time.monotonic() - started
            assert elapsed < 1.0, "the server loop was blocked by the model load"
            assert health.status_code == 200
            assert health.json()["reason"] == "subject_starting"
            assert dc.json()["dc_state"] == "STARTING"
            assert chat.status_code == 503
            assert chat.json() == {"detail": "subject_starting"}
            time.sleep(0.2)
        assert supervisor.wait_serving(120)
        assert client.get("/health").json()["status"] == "ok"


def test_real_app_lease_busy_at_the_first_start_is_refused_running_then_heals(
    tmp_path: Path, provider: SimpleNamespace
) -> None:
    settings = _settings(tmp_path, mode="continue")
    holder = EventJournalLease(settings.event_journal.path)
    app = server.create_app(settings)
    supervisor = _fast(app)
    try:
        with TestClient(app) as client:
            wait_until(phase_is(supervisor, DcState.REFUSED_RUNNING), timeout=60)
            body = client.get("/health").json()
            assert body["status"] == "degraded"
            assert body["dc_state"] == "REFUSED_RUNNING"
            assert body["dc_reason"] == "event_journal_lease_busy"
            assert client.get("/health/dc").json()["dc_reason"] == "event_journal_lease_busy"
            assert client.post("/api/chat", json={"message": "x", "attachments": []}).status_code == 503
            holder.close()  # the cause clears: no human, no restart
            assert supervisor.wait_serving(120)
            assert client.get("/health").json()["status"] == "ok"
    finally:
        holder.close()


def test_real_app_corrupt_journal_refuses_and_keeps_answering(
    tmp_path: Path, provider: SimpleNamespace
) -> None:
    settings = _settings(tmp_path, mode="continue")
    settings.event_journal.path.write_bytes(b"\x00 this is not a journal\n")
    app = server.create_app(settings)
    supervisor = _fast(app)
    with TestClient(app) as client:
        wait_until(phase_is(supervisor, DcState.REFUSED_RUNNING), timeout=60)
        wait_until(lambda: supervisor.cycles_launched >= 3, timeout=60)
        # the Journal constructor failed: its lease is free between attempts
        wait_until(phase_is(supervisor, DcState.REFUSED_RUNNING), timeout=60)
        probe = EventJournalLease(settings.event_journal.path)
        probe.close()
        assert client.get("/health").status_code == 200
        assert client.get("/health/dc").json()["dc_state"] in (
            "REFUSED_RUNNING",
            "RESTARTING",
        )
        assert not supervisor.terminal


def test_real_app_failed_starts_leak_no_descriptor_thread_or_lease(
    tmp_path: Path, provider: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The model load fails AFTER the Journal, SubjectHistory and Chroma are
    open: the worst point for a leak.  Between attempts the Journal lease must
    be free, and the fd / thread counts must not grow with the attempts."""

    provider.behaviour = lambda: (_ for _ in ()).throw(RuntimeError("model load failed"))
    closed: list[str] = []
    real_close = SubjectHistoryStore.close

    def counting_close(self: SubjectHistoryStore) -> None:
        closed.append("history")
        real_close(self)

    monkeypatch.setattr(SubjectHistoryStore, "close", counting_close)
    settings = _settings(tmp_path, mode="continue")
    app = server.create_app(settings)
    supervisor = _fast(app, delay=0.4)
    samples: list[tuple[int, int]] = []
    with TestClient(app) as client:
        for attempt in range(1, 8):
            wait_until(lambda: provider.calls >= attempt, timeout=60)
            wait_until(phase_is(supervisor, DcState.REFUSED_RUNNING), timeout=60)
            time.sleep(0.05)
            # the lease is released promptly: another holder can take it now
            probe = EventJournalLease(settings.event_journal.path)
            probe.close()
            assert client.get("/health/dc").json()["dc_reason"] == "builtins.RuntimeError"
            # closed explicitly on the failure path, not left to the garbage
            # collector: one close per failed start that had opened the store
            assert len(closed) >= attempt
            gc.collect()
            samples.append((_fd_count(), threading.active_count()))
    warm = samples[2:]  # the first attempts warm lazy caches
    assert len({fds for fds, _ in warm}) == 1, samples
    assert len({count for _, count in warm}) == 1, samples


def test_the_baseline_includes_what_the_mounted_apis_set_at_create_app(
    tmp_path: Path, provider: SimpleNamespace
) -> None:
    app = server.create_app(
        _settings(tmp_path, mode="continue").model_copy(
            update={"r19": _settings(tmp_path).r19.model_copy(update={"enabled": True})}
        )
    )
    # mount_inbox_api sets app.state.inbox_limits at create_app time: not a
    # cycle key, so it must survive the reset between cycles.
    assert "inbox_limits" in app.state._state
    supervisor: SubjectSupervisor = app.state.dc_supervisor
    assert {"settings", "dc_status", "dc_supervisor", "inbox_limits"} <= (
        supervisor._baseline
    )


def test_the_baseline_includes_the_approval_api_and_guard_keys(
    tmp_path: Path, provider: SimpleNamespace
) -> None:
    base = _settings(tmp_path, mode="continue")
    app = server.create_app(
        base.model_copy(update={"r18": base.r18.model_copy(update={"enabled": True})})
    )
    supervisor: SubjectSupervisor = app.state.dc_supervisor
    set_by_mounts = {"approval_guard_config", "approval_limits"}
    assert set_by_mounts <= set(app.state._state)
    assert set_by_mounts <= supervisor._baseline  # captured LAST, after the mounts


def test_real_app_state_is_rebuilt_not_reused_across_a_restart(
    tmp_path: Path, provider: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = server.create_app(_settings(tmp_path, mode="continue"))
    supervisor = _fast(app)
    baseline = set(app.state._state)
    with TestClient(app):
        assert supervisor.wait_serving(120)
        added = set(app.state._state) - baseline
        assert {"agent_runtime", "event_journal", "main_loop", "memory_system"} <= added
        before = {key: getattr(app.state, key) for key in added}
        runtime = app.state.agent_runtime
        monkeypatch.setattr(
            app.state.event_journal,
            "append_started",
            lambda *a, **k: (_ for _ in ()).throw(OSError(5, "EIO")),
        )
        with pytest.raises(Exception):
            runtime.submit(
                AgentEventType.CHAT, AgentEventSource.API_CHAT, lambda: None
            ).result(10)
        wait_until(
            lambda: supervisor.cycles_launched == 2 and supervisor.phase is SERVING,
            timeout=120,
        )
        for key in ("agent_runtime", "event_journal", "main_loop", "memory_system",
                    "model_provider", "experience_store", "state_wal"):
            assert getattr(app.state, key) is not before[key], key
        assert set(app.state._state) - baseline == added  # nothing leftover, nothing missing


# ===================================================== the flag-off path


def test_the_default_mode_is_the_legacy_lifespan_exactly(
    tmp_path: Path, provider: SimpleNamespace
) -> None:
    settings = _settings(tmp_path)
    assert settings.degraded_continuation.mode == "refuse"
    app = server.create_app(settings)
    assert not hasattr(app.state, "dc_supervisor")
    assert not any(
        getattr(m, "cls", None) is SubjectPhaseMiddleware for m in app.user_middleware
    )
    with TestClient(app) as client:
        # ready as soon as the lifespan is entered, no phase, no dc fields
        assert client.get("/health").json() == {
            "status": "ok",
            "project": settings.project.name,
        }
        assert client.get("/health/dc").json() == {"dc_state": "INACTIVE"}
        assert app.state.agent_runtime.status is AgentRuntimeStatus.ACCEPTING


def test_the_default_mode_still_raises_at_a_refused_start(
    tmp_path: Path, provider: SimpleNamespace
) -> None:
    settings = _settings(tmp_path)
    settings.event_journal.path.write_bytes(b"\x00 not a journal\n")
    app = server.create_app(settings)
    with pytest.raises(EventJournalLoadError):
        with TestClient(app):
            pass


def test_the_default_mode_leaves_a_failed_runtime_failed(
    tmp_path: Path, provider: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = server.create_app(_settings(tmp_path))
    with TestClient(app) as client:
        runtime = app.state.agent_runtime
        monkeypatch.setattr(
            app.state.event_journal,
            "append_started",
            lambda *a, **k: (_ for _ in ()).throw(OSError(5, "EIO")),
        )
        with pytest.raises(Exception):
            runtime.submit(
                AgentEventType.CHAT, AgentEventSource.API_CHAT, lambda: None
            ).result(10)
        time.sleep(1.0)
        assert app.state.agent_runtime is runtime
        assert runtime.status is AgentRuntimeStatus.FAILED
        assert client.get("/health").json()["reason"] == "agent_runtime_unavailable"


def test_a_runtime_without_the_observer_behaves_as_before() -> None:
    runtime = AgentRuntime(2, allow_volatile=True)
    runtime.start()
    runtime.shutdown()  # no observer set: nothing to call, nothing changes
    seen: list[Any] = []
    failing = AgentRuntime(2, allow_volatile=True)
    failing.set_on_failed(seen.append)
    with failing._condition:
        failing._fail_stop_locked(phase=AgentRuntimeDurabilityPhase.ADMISSION)
    assert seen == [AgentRuntimeDurabilityPhase.ADMISSION]
    late: list[Any] = []
    failing.set_on_failed(late.append)  # already FAILED: told at once
    assert late == [AgentRuntimeDurabilityPhase.ADMISSION]
    failing.set_on_failed(lambda phase: 1 / 0)  # a raising observer is swallowed
    with failing._condition:
        failing._fail_stop_locked(phase=AgentRuntimeDurabilityPhase.ADMISSION)
    with pytest.raises(RuntimeError, match="cannot be restarted"):
        failing.start()


# ======================================== real routes on the SERVER loop (P2-1)


def _chat_and_reads(client: TestClient) -> None:
    chat = client.post("/api/chat", json={"message": "hello", "attachments": []})
    assert chat.status_code == 200, chat.text
    assert client.get("/api/memory/search", params={"query": "hello"}).status_code == 200  # memory_system
    assert client.get("/api/contexts").status_code == 200  # main_loop registry
    assert client.get("/health").json()["status"] == "ok"


def test_real_routes_work_on_the_server_loop_while_serving_and_after_a_restart(
    tmp_path: Path, provider: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = server.create_app(_settings(tmp_path, mode="continue"))
    supervisor = _fast(app)
    with TestClient(app) as client:
        assert supervisor.wait_serving(120)
        _chat_and_reads(client)  # objects built on the cycle thread's loop
        runtime = app.state.agent_runtime
        monkeypatch.setattr(
            app.state.event_journal,
            "append_started",
            lambda *a, **k: (_ for _ in ()).throw(OSError(5, "EIO")),
        )
        with pytest.raises(Exception):
            runtime.submit(
                AgentEventType.CHAT, AgentEventSource.API_CHAT, lambda: None
            ).result(10)
        wait_until(
            lambda: supervisor.cycles_launched == 2 and supervisor.phase is SERVING,
            timeout=120,
        )
        _chat_and_reads(client)  # and after the first loop was closed


def test_a_teardown_step_that_raises_does_not_skip_the_later_stops(
    tmp_path: Path, provider: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The legacy teardown is a plain sequence: the emotion timer's stop() is
    before ``agent_runtime.shutdown()``.  A raise there used to leave the
    Runtime worker alive and the supervisor in a permanent
    REFUSED_RUNNING(threads_not_stopped); now the later owner is still stopped
    and the next cycle starts."""

    app = server.create_app(_settings(tmp_path, mode="continue"))
    supervisor = _fast(app)
    with TestClient(app):
        assert supervisor.wait_serving(120)
        first_runtime = app.state.agent_runtime
        assert first_runtime.threads_alive() != ()

        def boom() -> None:
            raise RuntimeError("timer stop failed")

        monkeypatch.setattr(app.state.emotion_timer, "stop", boom)
        # a failure observation with the Runtime still ACCEPTING: the
        # teardown runs the legacy sequence, which raises midway
        with supervisor._cv:
            supervisor._failed = True
            supervisor._cv.notify_all()
        wait_until(
            lambda: supervisor.cycles_launched == 2 and supervisor.phase is SERVING,
            timeout=120,
            what="the next cycle after a raising teardown",
        )
        assert not supervisor.terminal
        assert first_runtime.threads_alive() == ()
        assert first_runtime.status is AgentRuntimeStatus.STOPPED


def test_transformers_provider_release_drops_model_and_processor() -> None:
    from suzka.models.transformers_provider import TransformersProvider

    provider = object.__new__(TransformersProvider)
    provider.model = object()  # type: ignore[assignment]
    provider.processor = object()  # type: ignore[assignment]
    provider.release()
    assert provider.model is None and provider.processor is None
