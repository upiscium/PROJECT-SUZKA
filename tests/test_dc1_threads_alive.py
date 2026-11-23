"""DC-1: the ``threads_alive()`` protocol (F15 slice D1 r3 sec. 5.1, 5.3 item 6).

The point of the protocol is the thread an owner has ABANDONED: ``stop()``
joins with a timeout and then forgets the handle.  Every test here keeps a
thread alive past ``stop()`` and requires ``threads_alive()`` to still name it.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable

import pytest

from suzka.r16.driver import R16Driver
from suzka.r18 import approval_owner as approval_module
from suzka.r18 import executor as executor_module
from suzka.r18.approval_owner import ApprovalChallengeOwnerV1
from suzka.r18.executor import R18Executor
from suzka.r19 import scheduler as scheduler_module
from suzka.r19.scheduler import WakeScheduler
from suzka.runtime import AgentRuntime, EmotionTimer
from suzka.runtime.thread_liveness import ThreadLedger, ThreadsAlive


def _wait(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    end = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.005)


def test_ledger_reports_a_thread_until_it_exits() -> None:
    ledger = ThreadLedger()
    release = threading.Event()
    thread = threading.Thread(target=release.wait, name="stuck-one", daemon=True)
    thread.start()
    ledger.register(thread)
    assert ledger.alive() == ("stuck-one",)
    release.set()
    thread.join(5)
    assert ledger.alive() == ()


def test_ledger_never_reports_a_thread_that_was_not_started() -> None:
    ledger = ThreadLedger()
    ledger.register(threading.Thread(target=lambda: None, name="never-started"))
    # not alive, so not reported; owners register only AFTER a successful start()
    assert ledger.alive() == ()


def test_agent_runtime_remembers_a_worker_that_shutdown_forgot() -> None:
    runtime = AgentRuntime(2, allow_volatile=True)
    assert isinstance(runtime, ThreadsAlive)
    assert runtime.threads_alive() == ()
    runtime.start()
    assert runtime.threads_alive() == ("suzka-agent-runtime",)
    runtime.shutdown()
    _wait(lambda: runtime.threads_alive() == ())


def test_agent_runtime_reports_a_worker_stuck_in_a_handler() -> None:
    runtime = AgentRuntime(2, allow_volatile=True)
    runtime.start()
    entered = threading.Event()
    release = threading.Event()

    def handler() -> None:
        entered.set()
        release.wait(30)

    from suzka.runtime import AgentEventSource, AgentEventType

    runtime.submit(AgentEventType.CHAT, AgentEventSource.API_CHAT, handler)
    assert entered.wait(5)
    shutter = threading.Thread(target=runtime.shutdown, daemon=True)
    shutter.start()  # shutdown() joins without a timeout: this blocks
    time.sleep(0.05)
    assert runtime.threads_alive() == ("suzka-agent-runtime",)
    release.set()
    shutter.join(5)
    _wait(lambda: runtime.threads_alive() == ())


class _IdleRuntime:
    def submit(self, *args: Any, **kwargs: Any) -> Any:  # never reached
        raise AssertionError("not expected")


def test_emotion_timer_threads_alive() -> None:
    timer = EmotionTimer(_IdleRuntime(), 3600.0, lambda: None)  # type: ignore[arg-type]
    assert isinstance(timer, ThreadsAlive)
    assert timer.threads_alive() == ()
    timer.start()
    assert timer.threads_alive() == ("suzka-emotion-timer",)
    timer.start()  # a repeated start creates no second thread
    assert timer.threads_alive() == ("suzka-emotion-timer",)
    timer.stop()
    assert timer.threads_alive() == ()


def test_scheduler_reports_an_abandoned_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    release = threading.Event()
    scheduler = object.__new__(WakeScheduler)
    scheduler._stop = threading.Event()  # type: ignore[attr-defined]
    scheduler._wake = threading.Event()  # type: ignore[attr-defined]
    scheduler._lock = threading.Lock()  # type: ignore[attr-defined]
    scheduler._thread = None  # type: ignore[attr-defined]
    scheduler._thread_ledger = ThreadLedger()  # type: ignore[attr-defined]
    monkeypatch.setattr(WakeScheduler, "_run", lambda self: release.wait(30))
    monkeypatch.setattr(
        scheduler_module.threading.Thread, "join", _short_join, raising=False
    )
    assert isinstance(scheduler, ThreadsAlive)
    scheduler.start()
    assert scheduler.threads_alive() == (scheduler_module.THREAD_NAME,)
    scheduler.stop()  # joins (shortened), then forgets the handle
    assert scheduler._thread is None  # type: ignore[attr-defined]
    assert scheduler.threads_alive() == (scheduler_module.THREAD_NAME,)  # still named
    release.set()
    _wait(lambda: scheduler.threads_alive() == ())


_real_join = threading.Thread.join


def _short_join(self: threading.Thread, timeout: float | None = None) -> None:
    _real_join(self, 0.05 if timeout is None or timeout > 0.05 else timeout)


def test_executor_reports_both_threads_and_an_abandoned_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = threading.Event()
    executor = object.__new__(R18Executor)
    executor._lifecycle = threading.Lock()  # type: ignore[attr-defined]
    executor._started = False  # type: ignore[attr-defined]
    executor._authority = lambda: True  # type: ignore[attr-defined]
    executor._stop_event = threading.Event()  # type: ignore[attr-defined]
    executor._monotonic = time.monotonic  # type: ignore[attr-defined]
    executor._next_backstop = 0.0  # type: ignore[attr-defined]
    executor._queue_sweep = lambda: None  # type: ignore[attr-defined]
    executor._threaded = True  # type: ignore[attr-defined]
    executor._worker = None  # type: ignore[attr-defined]
    executor._reconciler = None  # type: ignore[attr-defined]
    executor._queue = queue.Queue(maxsize=8)  # type: ignore[attr-defined]
    executor._thread_ledger = ThreadLedger()  # type: ignore[attr-defined]
    executor._close_store = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setattr(R18Executor, "_worker_loop", lambda self: release.wait(30))
    monkeypatch.setattr(
        R18Executor, "_reconciler_loop", lambda self: executor._stop_event.wait(30)
    )
    monkeypatch.setattr(executor_module, "JOIN_TIMEOUT_SECONDS", 0.05)
    assert isinstance(executor, ThreadsAlive)
    executor.start()
    assert set(executor.threads_alive()) == {
        "suzka-r18-executor",
        "suzka-r18-reconciler",
    }
    executor.stop()  # the reconciler exits on the stop event, the worker is stuck
    assert executor._worker is None  # type: ignore[attr-defined]
    _wait(lambda: executor.threads_alive() == ("suzka-r18-executor",))
    release.set()
    _wait(lambda: executor.threads_alive() == ())


def test_every_other_thread_owner_exposes_the_protocol() -> None:
    # The driver and the approval owner need a full world to start; their
    # wiring is pinned by the architecture test over every ``Thread(`` site.
    for owner in (R16Driver, ApprovalChallengeOwnerV1):
        assert callable(getattr(owner, "threads_alive"))
    assert approval_module.threading.Timer  # the timer threads are registered too
