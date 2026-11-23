"""F15 DC-5: the in-process subject supervisor (#411, F15 slice D1 r3 sec. 4.1,
5.3).

One supervisor per process, only with ``degraded_continuation.mode ==
"continue"`` (the default is ``"refuse"``: no supervisor exists and the
legacy lifespan runs byte for byte).

Shape (all of it stays inside SUZKA, no host change):

* The server lifespan starts the supervisor and yields at once, so ``/health``
  and the 503 phase middleware answer from the first second.
* The supervisor runs on its own **control thread** (``suzka-dc-supervisor``).
  It runs each *cycle* (the unchanged startup / teardown of the subject) on a
  **dedicated cycle thread** (``suzka-dc-cycle``) with its **own event loop**:
  model load, Chroma open and reconciliation block only that thread, never the
  server loop.  The cycle thread enters the cycle context, announces ``ready``
  (or the classified exception), waits for a stop request and then leaves the
  context (teardown) on the same thread.
* After any ``AgentRuntime`` FAILED the supervisor tears the cycle down and
  starts a new one **only if every thread of the previous cycle is proven
  stopped**: the cycle thread has returned and ``threads_alive()`` of every
  thread owner is empty, within ``T_teardown_max`` + ``T_term_grace``.
  Otherwise the process ends in the TERMINAL state
  ``REFUSED_RUNNING(threads_not_stopped)``: no further cycle, R18 closed, and
  only then (HD-F15S-2 (e), opt-in) a non-zero process exit.
* A start-time refusal (the first start included, HD-new-1) is classified by
  exception class name (DC-1) and becomes ``REFUSED_RUNNING``: the process
  stays up, re-evaluating after ``T_reeval``.  Nothing is undefined: every
  ``Exception`` class has a label; only ``KeyboardInterrupt`` /
  ``SystemExit`` / cancellation (``BaseException``) are not failures.

Bounds and what is NOT bounded (stated, not hidden):

* ``T_boot_max`` is a *visibility* bound: a build that is still running at
  ``T_boot_max`` is published as ``REFUSED_RUNNING(boot_timeout)`` but is not
  killed (synchronous model load / Chroma open cannot be interrupted); when it
  finally completes the cycle is accepted and the state moves on.  A build that
  never returns leaves that visible state, and the thread is a daemon.
* The teardown proof waits ``T_teardown_max + T_term_grace`` from the stop
  request; one non-interruptible syscall inside the teardown is not bounded by
  this code.
* ``T_failed_to_serving_max`` (400 s) holds only without a stuck thread and
  with a build that stays inside ``T_boot_max``; with a stuck thread the state
  is terminal instead.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
import gc
import logging
import os
import threading
import time
from typing import Any, Protocol

from starlette.datastructures import State
from starlette.types import ASGIApp, Receive, Scope, Send

from suzka.config.schema import DegradedContinuationSettings
from suzka.runtime.agent_runtime import AgentRuntimeStatus
from suzka.runtime.degraded_continuation import (
    T_TERM_GRACE_SECONDS,
    CycleFailure,
    CycleRestart,
    DcState,
    DegradedContinuationStatus,
    IntegrityVerdict,
    classify_cycle_exception,
    must_propagate,
    reeval_delay,
    restart_delay,
)

__all__ = [
    "EXIT_CODE_THREADS_NOT_STOPPED",
    "ReadyReport",
    "SubjectPhaseMiddleware",
    "SubjectSupervisor",
    "Timebase",
    "THREAD_SOURCE_KEYS",
    "failed_to_serving_bound_s",
    "hard_exit",
    "phase_detail",
]

LOGGER = logging.getLogger(__name__)

#: ``EX_SOFTWARE``: the process ends because it cannot recover in-process.
EXIT_CODE_THREADS_NOT_STOPPED = 70

CYCLE_THREAD_NAME = "suzka-dc-cycle"
SUPERVISOR_THREAD_NAME = "suzka-dc-supervisor"

#: ``app.state`` keys whose objects own threads (D1 sec. 5.3 item 6 (b)); the
#: ``HardBandWatcher`` (DC-6) registers under the last key.
THREAD_SOURCE_KEYS: tuple[str, ...] = (
    "agent_runtime",
    "emotion_timer",
    "r16_driver",
    "wake_scheduler",
    "approval_owner",
    "r18_executor",
    "hard_band_watcher",
)

#: The bounded joins one after another (D1 sec. 3): Scheduler, approval owner,
#: executor worker, executor reconciler, R16 driver (30 s each).
_BOUNDED_JOINS_S = 150.0

#: Scheduling slack of ``shutdown()``'s join over ``T_shutdown_join``.
_JOIN_SLACK_S = 0.5


def failed_to_serving_bound_s(settings: DegradedContinuationSettings) -> float:
    """``T_failed_to_serving_max`` (D1 sec. 3): the worst normal teardown (the
    bounded joins, 150 s) + the first restart back-off (1 s) + ``T_boot_max``.
    With the defaults 391 s, inside the stated 400 s."""

    return _BOUNDED_JOINS_S + restart_delay(0) + float(settings.t_boot_max_s)


_PHASE_DETAIL: dict[DcState, str] = {
    DcState.INACTIVE: "subject_starting",
    DcState.STARTING: "subject_starting",
    DcState.RESTARTING: "subject_restarting",
    DcState.RECONCILING: "subject_reconciling",
    DcState.REFUSED_RUNNING: "subject_refused",
}


def phase_detail(state: DcState) -> str:
    """The 503 ``detail`` / ``/health`` reason of a non-serving phase."""

    return _PHASE_DETAIL.get(state, "subject_refused")


@dataclass(frozen=True)
class ReadyReport:
    """What a ready cycle is: ``serving`` (Runtime ACCEPTING) or not, and the
    legacy ``/health`` reason when it is not (today's DEGRADED-UP)."""

    serving: bool
    reason: str | None = None


class Timebase:
    """The supervisor's clock and its only blocking primitive.  Tests inject a
    fake that advances virtual time instead of sleeping."""

    def now(self) -> float:
        return time.monotonic()

    def wait_for(
        self,
        condition: threading.Condition,
        predicate: Callable[[], bool],
        timeout: float | None,
    ) -> bool:
        """Wait on ``condition`` (which must be held by the caller) until
        ``predicate()`` or ``timeout`` seconds; returns ``predicate()``."""

        return condition.wait_for(predicate, timeout)


def hard_exit(code: int) -> None:
    """End the process from a non-main thread.  ``sys.exit`` only ends the
    calling thread, and a signal to ourselves is turned by uvicorn into a
    graceful exit that systemd's ``Restart=on-failure`` treats as clean, so the
    one reliable way is ``os._exit``.  It is used only in the terminal
    ``threads_not_stopped`` state, where threads that will not stop make a
    graceful shutdown pointless; the Journal flock is released by process exit.
    """

    LOGGER.critical("exiting with status %d: threads of a previous cycle did not stop", code)
    for handler in list(logging.getLogger().handlers):
        try:
            handler.flush()
        except Exception:
            pass
    os._exit(code)


class _CycleFactory(Protocol):
    def __call__(self, app: Any) -> AbstractAsyncContextManager[None]: ...


class _Cycle:
    """One cycle's handle.  Flags are written under the supervisor's condition."""

    def __init__(self, index: int) -> None:
        self.index = index
        self.thread: threading.Thread | None = None
        self.stop = threading.Event()
        self.ready = False
        self.done = False
        self.failure: CycleFailure | None = None
        self.restart_reason: str | None = None
        self.shutdown_signal = False
        self.exit_label: str | None = None
        self.teardown_label: str | None = None


class _Proof:
    PROVEN = "proven"
    NOT_PROVEN = "not_proven"
    ABORTED = "aborted"


class SubjectSupervisor:
    """See the module docstring.  ``app`` is the FastAPI app (only its
    ``state`` is touched)."""

    def __init__(
        self,
        app: Any,
        settings: DegradedContinuationSettings,
        status: DegradedContinuationStatus,
        *,
        cycle_factory: _CycleFactory,
        ready_report: Callable[[Any], ReadyReport],
        timebase: Timebase | None = None,
        restart_delay_fn: Callable[[int], float] = restart_delay,
        reeval_delay_fn: Callable[[int], float] | None = None,
        exit_fn: Callable[[int], None] = hard_exit,
        poll_s: float = 0.25,
        term_grace_s: float = T_TERM_GRACE_SECONDS,
        extra_thread_sources: Callable[[Any], Iterable[tuple[str, object]]] | None = None,
    ) -> None:
        self._app = app
        self._state: State = app.state
        self._settings = settings
        self._status = status
        self._cycle_factory = cycle_factory
        self._ready_report = ready_report
        self._tb = timebase or Timebase()
        self._restart_delay = restart_delay_fn
        self._reeval_delay = reeval_delay_fn or (
            lambda attempt: reeval_delay(attempt, float(settings.reeval_max_backoff_s))
        )
        self._exit_fn = exit_fn
        self._poll_s = poll_s
        self._term_grace_s = term_grace_s
        self._extra_sources = extra_thread_sources
        # The keys of ``app.state`` at the end of ``create_app`` (see
        # ``capture_baseline``); everything a cycle adds beyond them is
        # removed before the next cycle.
        self._baseline: frozenset[str] = frozenset(self._state._state)

        self._cv = threading.Condition(threading.Lock())
        self._phase = DcState.INACTIVE
        self._stopping = False
        self._shutdown_deadline: float | None = None
        self._failed = False
        self._terminal = False
        self._handle: _Cycle | None = None
        self._cycles_launched = 0
        self._dirty = False
        self._thread: threading.Thread | None = None
        self._last_reason: str | None = None
        self._exit_code: int | None = None
        self._failed_at: float | None = None
        #: Seconds from the instant a Runtime FAILED was observed to the next
        #: SERVING, teardown included (the F15 bound ``T_failed_to_serving_max``).
        self.last_failed_to_serving_s: float | None = None

    # ------------------------------------------------------------- public

    @property
    def phase(self) -> DcState:
        return self._phase

    @property
    def terminal(self) -> bool:
        return self._terminal

    @property
    def cycles_launched(self) -> int:
        return self._cycles_launched

    @property
    def status(self) -> DegradedContinuationStatus:
        return self._status

    def capture_baseline(self) -> None:
        """Call once, at the very END of ``create_app`` (after every router
        and API mount has set its ``app.state`` keys): those keys, and the
        supervisor itself, are not the cycle's and must survive the reset."""

        self._baseline = frozenset(self._state._state)

    def start(self) -> None:
        with self._cv:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("SubjectSupervisor is already running")
            previous = self._handle
            if previous is not None and not previous.done:
                raise RuntimeError("the previous subject cycle is still running")
            self._stopping = False
            self._shutdown_deadline = None
            self._terminal = False
            self._failed = False
            thread = threading.Thread(
                target=self._run, name=SUPERVISOR_THREAD_NAME, daemon=True
            )
            self._thread = thread
        self._publish(DcState.STARTING)
        thread.start()

    def shutdown(self, timeout: float | None = None) -> bool:
        """SIGTERM path: set the stop flag, cancel any back-off, ask the live
        cycle to tear down, join for up to ``T_shutdown_join``.  Returns True
        when the control thread has ended.  A cycle still blocked in a model
        load is a daemon thread and dies with the process."""

        budget = float(
            self._settings.t_shutdown_join_s if timeout is None else timeout
        )
        with self._cv:
            if self._shutdown_deadline is None:
                self._shutdown_deadline = self._tb.now() + budget
            self._stopping = True
            handle = self._handle
            self._cv.notify_all()
        if handle is not None:
            handle.stop.set()
        thread = self._thread
        if thread is None:
            return True
        # The control thread's own wait ends at the same deadline; a little
        # slack so that it can return before this join gives up.
        thread.join(budget + _JOIN_SLACK_S)
        return not thread.is_alive()

    def wait_for_phase(
        self, states: Iterable[DcState], timeout: float
    ) -> bool:
        wanted = frozenset(states)
        with self._cv:
            return self._tb.wait_for(
                self._cv, lambda: self._phase in wanted, timeout
            )

    def wait_serving(self, timeout: float) -> bool:
        return self.wait_for_phase((DcState.SERVING,), timeout)

    # ----------------------------------------------------------- publishing

    def _publish(
        self, state: DcState, reason: str | None = None
    ) -> None:
        with self._cv:
            self._phase = state
            self._cv.notify_all()
        self._last_reason = reason
        self._status.publish(state, reason=reason)

    # ----------------------------------------------------------- the loop

    def _run(self) -> None:
        try:
            self._loop()
        except BaseException as error:  # the control thread must never die silently
            label = f"supervisor_internal_error:{type(error).__name__}"
            LOGGER.critical("subject supervisor ended unexpectedly: %s", label)
            self._publish(DcState.REFUSED_RUNNING, label)
            self._terminal = True
            self._abandon_cycle()  # a live cycle is told to tear down, not left

    def _loop(self) -> None:
        reeval_n = 0
        restart_run = 0
        failures: deque[float] = deque()
        delay = 0.0
        failed_at: float | None = None
        kind_first = True
        while True:
            if delay > 0 and self._sleep(delay):
                return
            if self._stop_requested():
                return
            try:
                outcome = self._one_cycle(
                    kind_first, reeval_n, restart_run, failures, failed_at
                )
            except Exception as error:  # a supervisor-logic error is a refusal too
                label = f"supervisor_internal_error:{type(error).__name__}"
                LOGGER.error("subject supervisor iteration failed: %s", label)
                self._publish(DcState.REFUSED_RUNNING, label)
                outcome = _Outcome(
                    delay=self._reeval_delay(reeval_n),
                    reeval_n=reeval_n + 1,
                    restart_run=restart_run,
                    failed_at=failed_at,
                )
                # Never launch the next cycle without the stop proof (P2-1).
                live = self._handle
                if live is not None:
                    proof, alive = self._conclude(live)
                    if proof == _Proof.ABORTED:
                        outcome = self._aborted(live)
                    elif proof == _Proof.NOT_PROVEN:
                        outcome = self._terminal_refusal(alive)
            kind_first = False
            if outcome.finished:
                return
            delay = outcome.delay
            reeval_n = outcome.reeval_n
            restart_run = outcome.restart_run
            failed_at = outcome.failed_at

    # One iteration: launch a cycle, follow it to its end, decide the next step.

    def _one_cycle(
        self,
        first: bool,
        reeval_n: int,
        restart_run: int,
        failures: deque[float],
        failed_at: float | None,
    ) -> "_Outcome":
        self._publish(
            DcState.STARTING if first else DcState.RESTARTING,
            None if first else self._last_reason,
        )
        handle = self._launch()
        build = self._await_build(handle)
        if build == "stop":
            self._end_for_shutdown(handle)
            return _Outcome(finished=True)
        if build == "done":
            return self._after_failed_build(handle, reeval_n, restart_run, failed_at)
        return self._after_ready(handle, reeval_n, restart_run, failures, failed_at)

    def _after_failed_build(
        self,
        handle: _Cycle,
        reeval_n: int,
        restart_run: int,
        failed_at: float | None,
    ) -> "_Outcome":
        proof, alive = self._conclude(handle)
        if proof == _Proof.ABORTED:
            return self._aborted(handle)
        if proof == _Proof.NOT_PROVEN:
            return self._terminal_refusal(alive)
        if handle.shutdown_signal:
            # KeyboardInterrupt / SystemExit / cancellation raised INSIDE a
            # cycle thread cannot end the process (it is not the main thread):
            # a visible, re-evaluated refusal, never a silent dead supervisor.
            label = "cycle_exit_requested:" + (handle.exit_label or "BaseException")
            self._publish(DcState.REFUSED_RUNNING, label)
            return _Outcome(
                delay=self._reeval_delay(reeval_n),
                reeval_n=reeval_n + 1,
                restart_run=restart_run,
                failed_at=failed_at,
            )
        if handle.restart_reason is not None:
            restart_run += 1
            if restart_run > int(self._settings.cycle_restart_cap):
                self._publish(DcState.RECONCILING, "cycle_restart_loop")
                return _Outcome(
                    delay=self._reeval_delay(reeval_n),
                    reeval_n=reeval_n + 1,
                    restart_run=0,
                    failed_at=failed_at,
                )
            return _Outcome(
                delay=0.0,
                reeval_n=reeval_n,
                restart_run=restart_run,
                failed_at=failed_at,
            )
        failure = handle.failure or CycleFailure(
            "unknown_cycle_failure", IntegrityVerdict.PROTECTED_LOST
        )
        self._publish(DcState.REFUSED_RUNNING, failure.label)
        LOGGER.warning(
            "subject start refused (%s, %s); re-evaluating",
            failure.label,
            failure.verdict.value,
        )
        return _Outcome(
            delay=self._reeval_delay(reeval_n),
            reeval_n=reeval_n + 1,
            restart_run=restart_run,
            failed_at=failed_at,
        )

    def _after_ready(
        self,
        handle: _Cycle,
        reeval_n: int,
        restart_run: int,
        failures: deque[float],
        failed_at: float | None,
    ) -> "_Outcome":
        report = self._ready_report(self._app)
        self._arm_failure_observer(handle)
        if not report.serving:
            self._publish(
                DcState.RECONCILING, report.reason or "external_reconciliation_required"
            )
            if self._sleep(self._reeval_delay(reeval_n)):
                self._end_for_shutdown(handle)
                return _Outcome(finished=True)
            return self._tear_down_then(
                handle,
                delay=0.0,
                reeval_n=reeval_n + 1,
                restart_run=restart_run,
                failed_at=failed_at,
            )
        self._publish(DcState.SERVING)
        serving_since = self._tb.now()
        reeval_n = 0
        if failed_at is not None:
            self.last_failed_to_serving_s = serving_since - failed_at
            failed_at = None
        if not self._await_failure_or_stop(handle):
            self._end_for_shutdown(handle)
            return _Outcome(finished=True)
        with self._cv:
            observed_at = self._failed_at
        failed_at = observed_at if observed_at is not None else self._tb.now()
        self._publish(DcState.RESTARTING, "runtime_failed")
        if failed_at - serving_since >= float(self._settings.t_stable_s):
            failures.clear()
            restart_run = 0
        proof, alive = self._conclude(handle)
        if proof == _Proof.ABORTED:
            return self._aborted(handle)
        if proof == _Proof.NOT_PROVEN:
            return self._terminal_refusal(alive)
        window = float(self._settings.restart_window_s)
        while failures and failed_at - failures[0] > window:
            failures.popleft()
        failures.append(failed_at)
        if len(failures) >= int(self._settings.restart_limit):
            self._publish(DcState.RESTART_LIMIT, "restart_limit")
            return _Outcome(
                delay=self._reeval_delay(reeval_n),
                reeval_n=reeval_n + 1,
                restart_run=restart_run,
                failed_at=failed_at,
            )
        return _Outcome(
            delay=self._restart_delay(len(failures) - 1),
            reeval_n=0,
            restart_run=restart_run,
            failed_at=failed_at,
        )

    def _tear_down_then(
        self,
        handle: _Cycle,
        *,
        delay: float,
        reeval_n: int,
        restart_run: int,
        failed_at: float | None,
    ) -> "_Outcome":
        proof, alive = self._conclude(handle)
        if proof == _Proof.ABORTED:
            return self._aborted(handle)
        if proof == _Proof.NOT_PROVEN:
            return self._terminal_refusal(alive)
        return _Outcome(
            delay=delay,
            reeval_n=reeval_n,
            restart_run=restart_run,
            failed_at=failed_at,
        )

    # --------------------------------------------------------- waiting helpers

    def _stop_requested(self) -> bool:
        with self._cv:
            return self._stopping

    def _sleep(self, seconds: float) -> bool:
        """Back-off.  Returns True when a stop was requested."""

        with self._cv:
            return self._tb.wait_for(self._cv, lambda: self._stopping, seconds)

    def _await_build(self, handle: _Cycle) -> str:
        def settled() -> bool:
            return self._stopping or handle.ready or handle.done

        with self._cv:
            met = self._tb.wait_for(
                self._cv, settled, float(self._settings.t_boot_max_s)
            )
        if not met:
            LOGGER.error(
                "subject cycle build exceeded T_boot_max (%ss); still waiting",
                self._settings.t_boot_max_s,
            )
            self._publish(DcState.REFUSED_RUNNING, "boot_timeout")
            with self._cv:
                self._tb.wait_for(self._cv, settled, None)
        with self._cv:
            if self._stopping:
                return "stop"
            return "ready" if handle.ready else "done"

    def _await_failure_or_stop(self, handle: _Cycle) -> bool:
        """Serving: wait for FAILED.  Returns False when a stop was requested."""

        while True:
            with self._cv:
                self._tb.wait_for(
                    self._cv,
                    lambda: self._stopping or self._failed or handle.done,
                    self._poll_s,
                )
                if self._stopping:
                    return False
                if self._failed or handle.done:
                    return True
            if self._runtime_status_failed():
                return True

    def _runtime_status_failed(self) -> bool:
        runtime = getattr(self._state, "agent_runtime", None)
        try:
            return getattr(runtime, "status", None) is AgentRuntimeStatus.FAILED
        except Exception:
            return False

    def _arm_failure_observer(self, handle: _Cycle) -> None:
        runtime = getattr(self._state, "agent_runtime", None)
        setter = getattr(runtime, "set_on_failed", None)
        if not callable(setter):
            return

        def on_failed(_phase: object) -> None:
            with self._cv:
                if self._handle is handle:
                    if not self._failed:
                        self._failed_at = self._tb.now()
                    self._failed = True
                    self._cv.notify_all()

        try:
            setter(on_failed)
        except Exception:
            LOGGER.warning("could not install the Runtime failure observer")

    # -------------------------------------------------------- launching a cycle

    def _launch(self) -> _Cycle:
        if self._dirty:
            self._reset_app_state()
        self._dirty = True
        self._cycles_launched += 1
        handle = _Cycle(self._cycles_launched)
        with self._cv:
            self._handle = handle
            self._failed = False
            self._failed_at = None
        thread = threading.Thread(
            target=self._cycle_thread, args=(handle,), name=CYCLE_THREAD_NAME, daemon=True
        )
        handle.thread = thread
        thread.start()
        return handle

    def _reset_app_state(self) -> None:
        """Remove every ``app.state`` key the previous cycle added, so the
        ``getattr(app.state, X, None) or <build>`` sites build fresh objects.
        Called only after the previous cycle is proven stopped."""

        values = self._state._state
        for key in [key for key in values if key not in self._baseline]:
            values.pop(key, None)
        gc.collect()

    def _abandon_cycle(self) -> None:
        handle = self._handle
        if handle is not None:
            handle.stop.set()

    def _cycle_thread(self, handle: _Cycle) -> None:
        try:
            asyncio.run(self._cycle_main(handle))
        except BaseException as error:  # the loop itself failed
            self._record(handle, error)
        finally:
            with self._cv:
                handle.done = True
                self._cv.notify_all()

    async def _cycle_main(self, handle: _Cycle) -> None:
        try:
            async with self._cycle_factory(self._app):
                with self._cv:
                    handle.ready = True
                    self._cv.notify_all()
                # Nothing else runs on this loop; blocking here is deliberate:
                # the teardown below must run on this same thread.
                handle.stop.wait()
        except BaseException as error:
            self._record(handle, error)

    def _record(self, handle: _Cycle, error: BaseException) -> None:
        """Classify and keep only the class name (never the exception or its
        traceback: a held traceback delays ``EventJournalLease.__del__``)."""

        after_ready = handle.ready
        restart = error.reason if isinstance(error, CycleRestart) else None
        propagate = must_propagate(error)
        failure = (
            None
            if restart is not None or propagate
            else classify_cycle_exception(error)
        )
        with self._cv:
            if after_ready:
                handle.teardown_label = (
                    failure.label if failure is not None else type(error).__name__
                )
            elif restart is not None:
                handle.restart_reason = restart
            elif propagate:
                handle.shutdown_signal = True
                handle.exit_label = type(error).__name__
            else:
                handle.failure = failure
        if after_ready:
            LOGGER.warning("subject cycle teardown raised %s", handle.teardown_label)

    # ----------------------------------------------- termination proof (sec. 5.3)

    def _collect_sources(self) -> list[tuple[str, object]]:
        found: list[tuple[str, object]] = []
        for key in THREAD_SOURCE_KEYS:
            owner = getattr(self._state, key, None)
            if owner is not None:
                found.append((key, owner))
        if self._extra_sources is not None:
            found.extend(self._extra_sources(self._app))
        return found

    @staticmethod
    def _alive_names(
        sources: Sequence[tuple[str, object]], handle: _Cycle
    ) -> tuple[str, ...]:
        alive: list[str] = []
        thread = handle.thread
        if thread is not None and thread.is_alive():
            alive.append(thread.name)
        for key, owner in sources:
            accessor = getattr(owner, "threads_alive", None)
            if not callable(accessor):
                continue  # an owner with no threads (test double)
            try:
                names = accessor()
            except Exception:
                # Not provable: count it as alive.
                alive.append(f"{key}:threads_alive_failed")
                continue
            alive.extend(str(name) for name in names)
        return tuple(alive)

    def _close_r18_gate(self) -> None:
        """Close the R18 gate before the producers stop.  ``SweepGate`` has no
        ``close``: ``sweeping()`` is its closed state.  No-op for a disabled
        gate, a missing one, or a test double."""

        gate = getattr(self._state, "r18_gate", None)
        closer = getattr(gate, "sweeping", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                LOGGER.warning("could not close the R18 gate")

    def _conclude(self, handle: _Cycle) -> tuple[str, tuple[str, ...]]:
        """Ask the cycle to tear down and prove every thread stopped, within
        ``T_teardown_max + T_term_grace`` of now.  The thread sources are read
        BEFORE ``app.state`` is reset (the reset happens only on the next
        launch, after a proven stop)."""

        handle.stop.set()
        self._close_r18_gate()
        sources = self._collect_sources()
        budget = float(self._settings.t_teardown_max_s) + self._term_grace_s
        deadline = self._tb.now() + budget
        while True:
            with self._cv:
                done = handle.done
                if self._stopping:
                    return _Proof.ABORTED, ()
            if done and not self._alive_names(sources, handle):
                return _Proof.PROVEN, ()
            remaining = deadline - self._tb.now()
            if remaining <= 0:
                break
            with self._cv:
                self._tb.wait_for(
                    self._cv, lambda: self._stopping, min(self._poll_s, remaining)
                )
        return _Proof.NOT_PROVEN, self._alive_names(sources, handle) or (
            "teardown_incomplete",
        )

    def _terminal_refusal(self, alive: tuple[str, ...]) -> "_Outcome":
        """No further cycle in this process (D1 sec. 5.3 item 6, F-14)."""

        reason = "threads_not_stopped:" + ",".join(alive)
        self._terminal = True
        self._close_r18_gate()
        LOGGER.critical("subject cycle threads did not stop: %s", ",".join(alive))
        self._publish(DcState.REFUSED_RUNNING, reason)
        with self._cv:
            stopping = self._stopping
        if self._settings.exit_on_threads_not_stopped and not stopping:
            self._exit_code = EXIT_CODE_THREADS_NOT_STOPPED
            self._exit_fn(EXIT_CODE_THREADS_NOT_STOPPED)
        # Terminal: idle until the process is told to stop.
        with self._cv:
            self._tb.wait_for(self._cv, lambda: self._stopping, None)
        return _Outcome(finished=True)

    def _aborted(self, handle: _Cycle) -> "_Outcome":
        """A stop request arrived during a teardown proof: let the teardown
        that is already running finish, within the shutdown deadline."""

        self._end_for_shutdown(handle)
        return _Outcome(finished=True)

    def _end_for_shutdown(self, handle: _Cycle) -> None:
        """Shutdown: tear down the live cycle, waiting at most until the
        shutdown deadline (``T_shutdown_join`` from the request)."""

        handle.stop.set()
        self._close_r18_gate()
        with self._cv:
            deadline = self._shutdown_deadline
        remaining = (
            float(self._settings.t_shutdown_join_s)
            if deadline is None
            else max(0.0, deadline - self._tb.now())
        )
        with self._cv:
            self._tb.wait_for(self._cv, lambda: handle.done, remaining)


@dataclass
class _Outcome:
    """What the loop does after one cycle."""

    finished: bool = False
    delay: float = 0.0
    reeval_n: int = 0
    restart_run: int = 0
    failed_at: float | None = None


class SubjectPhaseMiddleware:
    """Pure ASGI, outermost: while the supervisor is not ``SERVING`` every
    path except the health-class paths answers 503 ``{"detail": "<phase>"}``.
    The check reads one attribute; it never touches ``app.state``."""

    HEALTH_PATHS = frozenset({"/health", "/health/dc"})

    def __init__(self, app: ASGIApp, supervisor: SubjectSupervisor) -> None:
        self.app = app
        self._supervisor = supervisor

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        phase = self._supervisor.phase
        if phase is DcState.SERVING or scope.get("path") in self.HEALTH_PATHS:
            await self.app(scope, receive, send)
            return
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1013})
            return
        body = ('{"detail":"%s"}' % phase_detail(phase)).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 503,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"retry-after", b"5"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

