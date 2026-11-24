"""DC-6: the HARD-band retention ladder (F15 slice D1 r3 sec. 4.4, 5.4, 10).

Layers:

* ``run_at_idle`` on a real ``AgentRuntime``.
* ``HardBandWatcher`` against fakes (a scripted Journal band, a scripted
  compaction, a manual clock): the ladder's order, filter, stop conditions,
  audit log, health overlay.
* The same watcher against a REAL ``EventJournal`` and compaction coordinator:
  what is archived, what the COMPACTION record says, that archives stay
  byte-identical and that an active-segment-only overflow stays HARD (F-5).
* The supervisor-owned origin of the HARD band across cycle rebuilds, on a
  virtual clock, and the termination proof that includes the watcher.
* The REAL application: HARD at start-up, HARD while serving, and the two
  flag-off equivalences (``mode = "refuse"``, ``hard_band_auto_shrink =
  false``).
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
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
from suzka.api.supervisor import ReadyReport, SubjectSupervisor
from suzka.config import Settings, load_settings
from suzka.config.schema import DegradedContinuationSettings
from suzka.models import DummyProvider
from suzka.runtime import (
    AgentEventSource,
    AgentEventType,
    AgentRuntime,
    AgentRuntimeStatus,
)
from suzka.runtime.degraded_continuation import (
    HARD_WINDOW_NOTE,
    DcState,
    DegradedContinuationStatus,
)
from suzka.runtime.event_journal import (
    EventJournal,
    EventJournalAdmissionBand,
    EventJournalCompactionOutcome,
    EventJournalLoadError,
)
from suzka.runtime.hard_band_watcher import (
    HARD_BAND_LOG_NAME,
    HARD_BAND_WATCHER_THREAD_NAME,
    HardBandWatcher,
)
from suzka.runtime.journal_compaction import (
    JournalCompactionCoordinator,
    JournalCompactionPolicy,
)
from tests.test_event_journal_compaction import _Clock, _fill, _journal, _segments

ROOT = Path(__file__).resolve().parents[1]
HARD = EventJournalAdmissionBand.HARD
NORMAL = EventJournalAdmissionBand.NORMAL
T_HARD = 600.0


def wait_until(
    predicate: Callable[[], bool], timeout: float = 10.0, what: str = "condition"
) -> None:
    end = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > end:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.005)


# =========================================================== run_at_idle


def _volatile_runtime() -> AgentRuntime:
    runtime = AgentRuntime(8, allow_volatile=True)
    runtime.start()
    return runtime


def test_run_at_idle_runs_when_idle_and_returns_true() -> None:
    runtime = _volatile_runtime()
    try:
        ran: list[int] = []
        assert runtime.run_at_idle(lambda: ran.append(1), 2.0) is True
        assert ran == [1]
    finally:
        runtime.shutdown()


def test_run_at_idle_times_out_without_running_while_an_event_is_active() -> None:
    runtime = _volatile_runtime()
    release = threading.Event()
    entered = threading.Event()

    def handler() -> None:
        entered.set()
        release.wait(10)

    try:
        future = runtime.submit(AgentEventType.CHAT, AgentEventSource.API_CHAT, handler)
        assert entered.wait(5)
        ran: list[int] = []
        started = time.monotonic()
        assert runtime.run_at_idle(lambda: ran.append(1), 0.2) is False
        assert 0.15 <= time.monotonic() - started < 3.0
        assert ran == []
        release.set()
        future.result(5)
        assert runtime.run_at_idle(lambda: ran.append(2), 2.0) is True
        assert ran == [2]
    finally:
        release.set()
        runtime.shutdown()


def test_run_at_idle_blocks_submit_for_the_duration_of_fn() -> None:
    runtime = _volatile_runtime()
    inside = threading.Event()
    leave = threading.Event()
    order: list[str] = []

    def work() -> None:
        order.append("fn-start")
        inside.set()
        leave.wait(10)
        order.append("fn-end")

    runner = threading.Thread(target=lambda: runtime.run_at_idle(work, 5.0))
    runner.start()
    try:
        assert inside.wait(5)
        submitted = threading.Event()

        def submit() -> None:
            runtime.submit(
                AgentEventType.CHAT,
                AgentEventSource.API_CHAT,
                lambda: order.append("event"),
            ).result(10)
            submitted.set()

        submitter = threading.Thread(target=submit)
        submitter.start()
        assert not submitted.wait(0.3), "a submit got in while fn was running"
        leave.set()
        runner.join(5)
        assert submitted.wait(5)
        submitter.join(5)
        assert order == ["fn-start", "fn-end", "event"]
    finally:
        leave.set()
        runtime.shutdown()


def test_run_at_idle_never_runs_on_a_runtime_that_is_not_accepting() -> None:
    created = AgentRuntime(8, allow_volatile=True)
    ran: list[int] = []
    assert created.run_at_idle(lambda: ran.append(1), 0.05) is False  # CREATED
    created.shutdown()
    assert created.run_at_idle(lambda: ran.append(1), 0.05) is False  # STOPPED
    assert ran == []


def test_run_at_idle_propagates_fn_errors_and_releases_the_lock() -> None:
    runtime = _volatile_runtime()
    try:
        with pytest.raises(ValueError):
            runtime.run_at_idle(lambda: (_ for _ in ()).throw(ValueError("x")), 1.0)
        assert runtime.run_at_idle(lambda: None, 1.0) is True
    finally:
        runtime.shutdown()


def test_run_at_idle_refuses_the_worker_thread() -> None:
    runtime = _volatile_runtime()
    try:
        future = runtime.submit(
            AgentEventType.CHAT,
            AgentEventSource.API_CHAT,
            lambda: runtime.run_at_idle(lambda: None, 0.05),
        )
        with pytest.raises(Exception) as caught:
            future.result(5)
        assert isinstance(caught.value.__cause__, RuntimeError)
    finally:
        runtime.shutdown()


# ============================================== the watcher against fakes


class FakeBandJournal:
    def __init__(self, band: EventJournalAdmissionBand = HARD) -> None:
        self.band = band
        self.live = 1000
        self.unreadable = False
        self.error: Exception | None = None
        self.status_calls = 0

    def admission_status(self) -> Any:
        self.status_calls += 1
        if self.error is not None:
            raise self.error
        if self.unreadable:
            raise EventJournalLoadError("size cannot be inspected")
        return SimpleNamespace(band=self.band, live_bytes=self.live)


class FakeCompaction:
    """``script[days]`` is a reason string (a non-run reason), ``"performed"``
    or ``"performed_clears"`` (performed and the band leaves HARD)."""

    def __init__(self, journal: FakeBandJournal, script: dict[int, str]) -> None:
        self.journal = journal
        self.script = script
        self.calls: list[int] = []

    def compact_with_window(self, retention_days: int) -> EventJournalCompactionOutcome:
        self.calls.append(retention_days)
        what = self.script.get(retention_days, "nothing_due")
        if what.startswith("performed"):
            self.journal.live -= 100
            if what == "performed_clears":
                self.journal.band = NORMAL
            return EventJournalCompactionOutcome(True, None, 1, 4, "a.jsonl.gz")
        return EventJournalCompactionOutcome(False, what)


class FakePort:
    def __init__(self) -> None:
        self.t = 5_000.0
        self.since: float | None = None
        self.restarts: list[str] = []
        self.restart_ok = True

    def now(self) -> float:
        return self.t

    def observe_hard_band(self, hard: bool) -> float | None:
        if not hard:
            self.since = None
        elif self.since is None:
            self.since = self.t
        return self.since

    def request_cycle_restart(self, reason: str) -> bool:
        self.restarts.append(reason)
        return self.restart_ok


class FakeRuntime:
    def __init__(self, status: AgentRuntimeStatus = AgentRuntimeStatus.ACCEPTING) -> None:
        self.status = status
        self.idle_calls = 0
        self.busy = False

    def run_at_idle(self, fn: Callable[[], object], timeout_s: float) -> bool:
        self.idle_calls += 1
        if self.busy:
            time.sleep(min(timeout_s, 0.01))
            return False
        fn()
        return True


@dataclass
class Rig:
    watcher: HardBandWatcher
    journal: FakeBandJournal
    compaction: FakeCompaction
    port: FakePort
    runtime: FakeRuntime
    status: DegradedContinuationStatus
    log: Path

    def lines(self) -> list[dict[str, Any]]:
        if not self.log.exists():
            return []
        return [json.loads(x) for x in self.log.read_text().splitlines()]

    def hard_state(self) -> str:
        return self.status.snapshot()["dc_state"]


def make_rig(
    tmp_path: Path,
    script: dict[int, str],
    *,
    ladder: tuple[int, ...] = (14, 7, 3, 1, 0),
    runtime_status: AgentRuntimeStatus = AgentRuntimeStatus.ACCEPTING,
    rung_timeout_s: float = 0.1,
    configured_days: int | None = None,
) -> Rig:
    journal = FakeBandJournal()
    compaction = FakeCompaction(journal, script)
    port = FakePort()
    runtime = FakeRuntime(runtime_status)
    status = DegradedContinuationStatus()
    status.publish(DcState.SERVING)
    log = tmp_path / "quarantine" / HARD_BAND_LOG_NAME
    watcher = HardBandWatcher(
        journal=journal,
        compaction=compaction,
        runtime=runtime,
        status=status,
        port=port,
        ladder=ladder,
        configured_days=configured_days,
        t_hard_s=T_HARD,
        log_path=log,
        rung_timeout_s=rung_timeout_s,
        wall_clock=lambda: datetime(2026, 11, 24, tzinfo=timezone.utc),
    )
    return Rig(watcher, journal, compaction, port, runtime, status, log)


def test_the_ladder_advances_only_after_t_hard_of_continuous_hard(
    tmp_path: Path,
) -> None:
    rig = make_rig(tmp_path, {14: "performed_clears"})
    rig.watcher.tick()
    assert rig.hard_state() == "HARD_WAIT"
    rig.port.t += T_HARD - 1
    rig.watcher.tick()
    assert rig.compaction.calls == [] and rig.hard_state() == "HARD_WAIT"
    rig.port.t += 1
    rig.watcher.tick()
    assert rig.compaction.calls == [14]


def test_a_non_hard_observation_resets_the_count(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, {14: "performed_clears"})
    rig.watcher.tick()
    rig.port.t += T_HARD - 1
    rig.journal.band = NORMAL
    rig.watcher.tick()  # cleared: the origin is gone
    assert rig.port.since is None and rig.hard_state() == "SERVING"
    rig.journal.band = HARD
    rig.port.t += 5
    rig.watcher.tick()  # a new origin
    rig.port.t += T_HARD - 1
    rig.watcher.tick()
    assert rig.compaction.calls == []
    rig.port.t += 1
    rig.watcher.tick()
    assert rig.compaction.calls == [14]


def _arm(rig: Rig) -> None:
    rig.watcher.tick()
    rig.port.t += T_HARD


def test_it_stops_at_the_first_sufficient_rung_and_records_each_step(
    tmp_path: Path,
) -> None:
    rig = make_rig(tmp_path, {14: "nothing_due", 7: "performed", 3: "performed_clears"})
    _arm(rig)
    rig.watcher.tick()
    assert rig.compaction.calls == [14, 7, 3]  # not 1, not 0
    assert [(x["rung_days"], x["outcome"]) for x in rig.lines()] == [
        (14, "nothing_due"),
        (7, "performed"),
        (3, "performed"),
    ]
    first = rig.lines()[1]
    assert first["performed"] is True
    assert first["live_bytes_before"] == 1000 and first["live_bytes_after"] == 900
    assert set(first) >= {"ts", "rung_days", "outcome", "live_bytes_before", "live_bytes_after"}
    snapshot = rig.status.snapshot()
    assert snapshot["dc_state"] == "SERVING"  # the band cleared
    assert snapshot["dc_hard_window_days"] == "3"
    assert snapshot["dc_hard_window_note"] == HARD_WINDOW_NOTE
    assert rig.status.health_fields()["dc_hard_window_days"] == "3"  # not quiet
    # a running Runtime needs no new cycle
    assert rig.port.restarts == [] and rig.runtime.idle_calls == 3


@pytest.mark.parametrize(
    "reason",
    [
        "previous_compaction_pending",
        "history_attestation_required",
        "history_attestation_refused",
        "deferred_history",
        "deferred_semantic",
        "boundary_not_clean",
        "clock_not_monotonic",
        "compaction_failed",
        "journal_schema_below_4",
        "some_future_reason",
    ],
)
def test_every_non_run_reason_stops_the_ladder_and_shows_the_reason(
    tmp_path: Path, reason: str
) -> None:
    rig = make_rig(tmp_path, {14: "nothing_due", 7: reason, 3: "performed_clears"})
    _arm(rig)
    rig.watcher.tick()
    assert rig.compaction.calls == [14, 7]  # never the next rung
    snapshot = rig.status.snapshot()
    assert snapshot["dc_state"] == "HARD_STUCK"
    assert snapshot["dc_hard_reason"] == reason
    assert snapshot["dc_reason"] == "journal_hard_limit"
    assert [x["outcome"] for x in rig.lines()] == ["nothing_due", reason]


def test_hard_stuck_retries_only_on_a_growing_re_evaluation_delay(
    tmp_path: Path,
) -> None:
    rig = make_rig(tmp_path, {14: "compaction_failed"})
    _arm(rig)
    rig.watcher.tick()
    assert rig.compaction.calls == [14]
    # polls inside the delay do nothing: never a tight loop
    for _ in range(50):
        rig.port.t += 0.01
        rig.watcher.tick()
    assert rig.compaction.calls == [14]
    delays: list[float] = []
    for _ in range(5):
        before = rig.port.t
        while len(rig.compaction.calls) == len(delays) + 1:
            rig.port.t += 0.5
            rig.watcher.tick()
        delays.append(rig.port.t - before)
    assert delays == sorted(delays)  # 1, 4, 16, 60, 300 s: growing
    assert delays[-1] > 100 > delays[0]


def test_stuck_then_the_cause_clears_and_the_ladder_continues(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, {14: "deferred_semantic"})
    _arm(rig)
    rig.watcher.tick()
    assert rig.hard_state() == "HARD_STUCK"
    rig.compaction.script = {14: "nothing_due", 7: "performed_clears"}
    rig.port.t += 5
    rig.watcher.tick()
    assert rig.compaction.calls[-2:] == [14, 7]
    assert rig.hard_state() == "SERVING"
    assert "dc_hard_reason" not in rig.status.snapshot()


def test_an_empty_ladder_is_hard_stuck_with_its_own_reason(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, {}, ladder=())
    _arm(rig)
    rig.watcher.tick()
    assert rig.compaction.calls == []
    assert rig.status.snapshot()["dc_hard_reason"] == "ladder_empty"
    assert rig.hard_state() == "HARD_STUCK"


def test_a_ladder_that_archives_but_stays_hard_ends_exhausted(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, {d: "performed" for d in (14, 7, 3, 1, 0)})
    _arm(rig)
    rig.watcher.tick()
    assert rig.compaction.calls == [14, 7, 3, 1, 0]
    snapshot = rig.status.snapshot()
    assert snapshot["dc_state"] == "HARD_STUCK"
    assert snapshot["dc_hard_reason"] == "ladder_exhausted"
    assert snapshot["dc_hard_window_days"] == "0"


def test_a_runtime_that_never_becomes_idle_stops_the_ladder(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, {14: "performed_clears"}, rung_timeout_s=0.1)
    rig.runtime.busy = True
    _arm(rig)
    rig.watcher.tick()
    assert rig.compaction.calls == []
    assert rig.status.snapshot()["dc_hard_reason"] == "run_at_idle_timeout"
    assert [x["outcome"] for x in rig.lines()] == ["run_at_idle_timeout"]


def test_a_unreadable_journal_neither_advances_nor_resets_the_origin(
    tmp_path: Path,
) -> None:
    rig = make_rig(tmp_path, {14: "performed_clears"})
    rig.watcher.tick()
    origin = rig.port.since
    rig.journal.unreadable = True
    rig.port.t += 2 * T_HARD
    rig.watcher.tick()
    assert rig.port.since == origin and rig.compaction.calls == []


def test_a_runtime_that_is_failing_is_left_to_the_supervisor(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, {14: "performed_clears"})
    rig.watcher.tick()
    rig.port.t += 2 * T_HARD
    rig.runtime.status = AgentRuntimeStatus.FAILED
    rig.watcher.tick()
    assert rig.compaction.calls == []


def test_at_start_up_the_ladder_compacts_directly_then_asks_for_one_new_cycle(
    tmp_path: Path,
) -> None:
    rig = make_rig(
        tmp_path,
        {14: "nothing_due", 7: "performed_clears"},
        runtime_status=AgentRuntimeStatus.CREATED,
    )
    _arm(rig)
    rig.watcher.tick()
    assert rig.compaction.calls == [14, 7]
    assert rig.runtime.idle_calls == 0  # no run_at_idle without a started Runtime
    assert rig.port.restarts == ["hard_band_cleared"]
    rig.watcher.tick()
    assert rig.port.restarts == ["hard_band_cleared"]  # asked once
    assert all(x["runtime_started"] is False for x in rig.lines())


def test_a_failed_log_write_never_stops_the_ladder(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, {14: "performed_clears"})
    (tmp_path / "quarantine").write_text("a file where the directory should be")
    _arm(rig)
    rig.watcher.tick()
    assert rig.compaction.calls == [14]
    assert rig.hard_state() == "SERVING"


def test_the_watcher_thread_polls_and_is_in_the_thread_ledger(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, {14: "performed_clears"})
    rig.watcher._poll_s = 0.01
    rig.watcher.start()
    try:
        assert rig.watcher.threads_alive() == (HARD_BAND_WATCHER_THREAD_NAME,)
        wait_until(lambda: rig.hard_state() == "HARD_WAIT", what="a poll")
        rig.port.t += T_HARD
        wait_until(lambda: rig.compaction.calls == [14], what="the ladder")
    finally:
        assert rig.watcher.stop() is True
    assert rig.watcher.threads_alive() == ()
    rig.watcher.stop()  # idempotent


def test_a_tick_that_raises_is_shown_and_the_thread_survives(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, {})
    rig.watcher._poll_s = 0.01
    calls = {"n": 0}
    real = rig.watcher.tick

    def flaky() -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        real()

    rig.watcher.tick = flaky  # type: ignore[method-assign]
    rig.watcher.start()
    try:
        wait_until(lambda: calls["n"] >= 1, what="the failing poll")
        rig.port.t += 1000.0  # past the back-off
        wait_until(lambda: calls["n"] >= 3, what="polls after the error")
        assert rig.watcher.threads_alive()
    finally:
        rig.watcher.stop()


# ====================================== real Journal, real coordinator


def _hard_journal(tmp_path: Path, days: int) -> tuple[Path, _Clock, EventJournal, int]:
    """A Journal with rotated segments that is HARD, ``days`` old."""

    path = tmp_path / "events.jsonl"
    clock = _Clock()
    value = _journal(path, clock)
    following = _fill(value, 1, 8)
    assert len(_segments(path)) >= 2
    value.close()
    clock.advance(days=days)
    live = path.stat().st_size + sum(item.stat().st_size for item in _segments(path))
    reopened = EventJournal(
        path, 3_000, 2, clock=clock, max_total_bytes=live - 1, soft_band_bytes=0
    )
    assert reopened.admission_status().band is HARD
    return path, clock, reopened, following


class _Spy:
    def __init__(self, inner: JournalCompactionCoordinator) -> None:
        self.inner = inner
        self.calls: list[int] = []

    def compact_with_window(self, retention_days: int) -> EventJournalCompactionOutcome:
        self.calls.append(retention_days)
        return self.inner.compact_with_window(retention_days)


def _real_rig(
    tmp_path: Path, journal: EventJournal, *, retention_days: int = 30, semantic: Any = None
) -> tuple[HardBandWatcher, _Spy, FakePort, DegradedContinuationStatus, JournalCompactionCoordinator]:
    coordinator = JournalCompactionCoordinator(
        journal, JournalCompactionPolicy(retention_days), semantic
    )
    spy = _Spy(coordinator)
    port = FakePort()
    status = DegradedContinuationStatus()
    status.publish(DcState.SERVING)
    settings = DegradedContinuationSettings(mode="continue")
    watcher = HardBandWatcher(
        journal=journal,
        compaction=spy,
        runtime=SimpleNamespace(status=AgentRuntimeStatus.CREATED),
        status=status,
        port=port,
        ladder=settings.effective_hard_ladder(retention_days),
        t_hard_s=T_HARD,
        log_path=tmp_path / "quarantine" / HARD_BAND_LOG_NAME,
    )
    return watcher, spy, port, status, coordinator


def test_the_ladder_frees_a_real_journal_at_the_first_sufficient_rung(
    tmp_path: Path,
) -> None:
    path, clock, journal, following = _hard_journal(tmp_path, days=10)
    watcher, spy, port, status, coordinator = _real_rig(tmp_path, journal)
    segments_before = _segments(path)
    watcher.tick()
    port.t += T_HARD
    watcher.tick()
    # 14 days: nothing is that old; 7 days: the segments go; 3, 1, 0 never run
    assert spy.calls == [14, 7]
    assert journal.admission_status().band is NORMAL
    assert len(_segments(path)) < len(segments_before)
    record = journal.inspect().latest_compaction
    assert record is not None and record.retention_days == 7  # the rung's window
    assert coordinator.policy.retention_days == 30  # the policy is untouched
    assert status.snapshot()["dc_hard_window_days"] == "7"
    lines = [
        json.loads(x)
        for x in (tmp_path / "quarantine" / HARD_BAND_LOG_NAME).read_text().splitlines()
    ]
    assert [(x["rung_days"], x["performed"]) for x in lines] == [(14, False), (7, True)]
    assert lines[1]["live_bytes_after"] < lines[1]["live_bytes_before"]
    assert port.restarts == ["hard_band_cleared"]  # the Runtime stub was never started
    journal.close()


def test_the_window_never_returns_and_archives_stay_byte_identical(
    tmp_path: Path,
) -> None:
    path, clock, journal, following = _hard_journal(tmp_path, days=10)
    watcher, spy, port, status, coordinator = _real_rig(tmp_path, journal)
    watcher.tick()
    port.t += T_HARD
    watcher.tick()
    archives = sorted(journal.archive_dir.iterdir())
    assert len(archives) == 1
    digest = hashlib.sha256(archives[0].read_bytes()).hexdigest()
    # the normal compaction (30 days) archives nothing that the ladder archived,
    # does not bring anything back, and a second ladder compaction leaves the
    # first archive's bytes alone
    _fill(journal, following, 6)
    clock.advance(days=8)
    assert coordinator.maybe_compact().reason == "nothing_due"
    assert coordinator.compact_with_window(0).performed
    first = [a for a in journal.archive_dir.iterdir() if a == archives[0]]
    assert first and hashlib.sha256(first[0].read_bytes()).hexdigest() == digest
    assert len(list(journal.archive_dir.iterdir())) == 2
    journal.close()


def test_an_overflow_inside_the_active_segment_stays_hard_f5(tmp_path: Path) -> None:
    # F-5 of the D1: archiving cannot free bytes that are not in rotated
    # segments.  The state stays HARD_STUCK and nothing is archived.
    path = tmp_path / "events.jsonl"
    clock = _Clock()
    value = _journal(path, clock, max_bytes=1_000_000)
    _fill(value, 1, 8)
    assert _segments(path) == []
    value.close()
    clock.advance(days=40)
    limited = EventJournal(
        path, 1_000_000, 2, clock=clock, max_total_bytes=path.stat().st_size - 1
    )
    assert limited.admission_status().band is HARD
    watcher, spy, port, status, _ = _real_rig(tmp_path, limited)
    watcher.tick()
    port.t += T_HARD
    watcher.tick()
    assert spy.calls == [14, 7, 3, 1, 0]
    snapshot = status.snapshot()
    assert snapshot["dc_state"] == "HARD_STUCK"
    assert snapshot["dc_hard_reason"] == "ladder_exhausted"
    assert not limited.archive_dir.exists() or not list(limited.archive_dir.iterdir())
    assert port.restarts == []
    limited.close()


def test_a_real_semantic_guard_stops_the_real_ladder(tmp_path: Path) -> None:
    path, clock, journal, _ = _hard_journal(tmp_path, days=10)
    semantic = SimpleNamespace(
        carried_tail=lambda: None, compaction_blocked=lambda high_water: True
    )
    watcher, spy, port, status, _ = _real_rig(tmp_path, journal, semantic=semantic)
    segments_before = _segments(path)
    watcher.tick()
    port.t += T_HARD
    watcher.tick()
    assert spy.calls == [14, 7]  # stopped at the first non-run reason
    assert status.snapshot()["dc_hard_reason"] == "deferred_semantic"
    assert _segments(path) == segments_before  # nothing archived
    journal.close()


def test_the_ladder_is_filtered_by_the_configured_window() -> None:
    settings = DegradedContinuationSettings(mode="continue")
    assert settings.effective_hard_ladder(30) == (14, 7, 3, 1, 0)
    assert settings.effective_hard_ladder(10) == (7, 3, 1, 0)
    assert settings.effective_hard_ladder(0) == ()
    assert DegradedContinuationSettings(
        hard_band_auto_shrink=False
    ).effective_hard_ladder(30) == ()


# ============================================ health overlay (status)


def test_the_overlay_replaces_serving_and_reconciling_and_nothing_else() -> None:
    status = DegradedContinuationStatus()
    status.publish(DcState.SERVING)
    assert status.health_fields() == {}
    status.note_hard_band(DcState.HARD_WAIT)
    fields = status.health_fields()
    assert fields["dc_state"] == "HARD_WAIT"
    assert fields["dc_reason"] == "journal_hard_limit"
    status.publish(DcState.RECONCILING, reason="journal_hard_limit")
    assert status.snapshot()["dc_state"] == "HARD_WAIT"
    assert status.snapshot()["dc_reason"] == "journal_hard_limit"
    status.publish(DcState.RESTARTING, reason="runtime_failed")
    assert status.snapshot()["dc_state"] == "RESTARTING"  # not hidden by it
    status.publish(DcState.SERVING)
    status.note_hard_band(None)
    assert status.health_fields() == {}
    with pytest.raises(ValueError):
        status.note_hard_band(DcState.SERVING)


def test_the_applied_window_is_kept_after_the_band_clears() -> None:
    status = DegradedContinuationStatus()
    status.publish(DcState.SERVING)
    status.note_hard_window(7)
    status.note_hard_window(14)  # never raises the shortest back up
    status.note_hard_window(3)
    status.note_hard_band(None)
    fields = status.health_fields()
    assert fields["dc_hard_window_days"] == "3"
    assert "does not return by itself" in fields["dc_hard_window_note"]
    assert fields["dc_state"] == "SERVING"


# ============================================== supervisor-owned origin


def _supervisor(
    cycles: Any, *, tb: FakeTimebase, reeval_delay: float = 60.0
) -> tuple[SubjectSupervisor, DegradedContinuationStatus, Any]:
    app = SimpleNamespace(state=State())
    app.state.settings = object()
    status = DegradedContinuationStatus()
    supervisor = SubjectSupervisor(
        app,
        DegradedContinuationSettings(mode="continue"),
        status,
        cycle_factory=cycles,
        ready_report=_ready_report,
        timebase=tb,
        restart_delay_fn=lambda n: 0.01,
        reeval_delay_fn=lambda n: reeval_delay,
        poll_s=0.01,
        term_grace_s=0.5,
    )
    return supervisor, status, app


def _ready_report(app: Any) -> ReadyReport:
    runtime = app.state.agent_runtime
    if runtime.status is AgentRuntimeStatus.ACCEPTING:
        return ReadyReport(serving=True)
    return ReadyReport(serving=False, reason="journal_hard_limit")


@dataclass
class HardWorld:
    """Scripted cycles that do what ``server._subject_cycle`` does for the
    watcher: build it before ``yield`` and stop it first in the teardown.  The
    Journal band is shared state (``journal``), like the files on disk."""

    supervisor_box: list[SubjectSupervisor]
    status: DegradedContinuationStatus
    tmp_path: Path
    script: dict[int, str]
    journal: FakeBandJournal = field(default_factory=FakeBandJournal)
    watchers: list[HardBandWatcher] = field(default_factory=list)
    compaction_times: list[float] = field(default_factory=list)
    leak_watcher: bool = False
    hard_clock: list[float] = field(default_factory=list)

    @asynccontextmanager
    async def __call__(self, app: Any) -> AsyncIterator[None]:
        supervisor = self.supervisor_box[0]
        journal = self.journal

        class Runtime:
            def __init__(self, status: AgentRuntimeStatus) -> None:
                self.status = status

            def threads_alive(self) -> tuple[str, ...]:
                return ()

            def run_at_idle(self, fn: Callable[[], object], timeout_s: float) -> bool:
                fn()
                return True

        # HARD at start-up: the Runtime is not started; otherwise it is.
        started = journal.band is not HARD
        runtime = Runtime(
            AgentRuntimeStatus.ACCEPTING if started else AgentRuntimeStatus.CREATED
        )
        app.state.agent_runtime = runtime
        world = self

        class Compaction(FakeCompaction):
            def compact_with_window(self, retention_days: int) -> Any:
                world.compaction_times.append(supervisor.now())
                return super().compact_with_window(retention_days)

        watcher = HardBandWatcher(
            journal=journal,
            compaction=Compaction(journal, self.script),
            runtime=runtime,
            status=self.status,
            port=supervisor,
            ladder=(14, 7, 3, 1, 0),
            t_hard_s=T_HARD,
            log_path=self.tmp_path / "quarantine" / HARD_BAND_LOG_NAME,
            poll_s=0.002,
            stop_join_s=5.0,
        )
        watcher.start()
        app.state.hard_band_watcher = watcher
        self.watchers.append(watcher)
        try:
            yield
        finally:
            if not self.leak_watcher:
                watcher.stop()


def test_hard_since_survives_cycle_rebuilds_and_the_ladder_runs_after_t_hard(
    tmp_path: Path,
) -> None:
    """HARD at start-up: every re-evaluation rebuilds the cycle (and its
    watcher); the origin is the supervisor's, so the ladder still runs after
    ``T_hard`` of CONTINUOUS HARD, and then the Runtime starts."""

    tb = FakeTimebase(step=0.5)
    status_box: list[SubjectSupervisor] = []
    world = HardWorld(status_box, DegradedContinuationStatus(), tmp_path, {7: "performed_clears"})
    supervisor, status, app = _supervisor(world, tb=tb, reeval_delay=60.0)
    world.status = status
    status.publish(DcState.STARTING)
    status_box.append(supervisor)
    supervisor.start()
    try:
        assert supervisor.wait_serving(5000)
        origin_unset = supervisor.hard_since
        assert origin_unset is None  # the band cleared: the origin is gone
        assert len(world.watchers) >= 11  # ~600 s of 60 s re-evaluations
        first_poll = world.compaction_times[0]
        # the first compaction came after T_hard of continuous HARD, measured
        # from the origin of the FIRST cycle (about t = 1000 on the fake clock)
        assert first_poll - 1000.0 >= T_HARD
        assert first_poll - 1000.0 < T_HARD + 120.0
        assert supervisor.cycles_launched == len(world.watchers)
        assert app.state.agent_runtime.status is AgentRuntimeStatus.ACCEPTING
        # every cycle's watcher was stopped by its cycle, all proven stopped
        wait_until(lambda: all(w.threads_alive() == () for w in world.watchers[:-1]))
        assert supervisor.phase is DcState.SERVING
    finally:
        supervisor.shutdown(timeout=5)
        for watcher in world.watchers:
            watcher.stop()


def test_a_restart_request_starts_the_next_cycle_without_waiting_the_re_evaluation(
    tmp_path: Path,
) -> None:
    """A real clock and a 300 s re-evaluation delay: only the watcher's request
    can make the second cycle start within the test."""

    @dataclass
    class Cycles:
        launches: int = 0

        @asynccontextmanager
        async def __call__(self, app: Any) -> AsyncIterator[None]:
            self.launches += 1
            runtime = SimpleNamespace(
                status=(
                    AgentRuntimeStatus.CREATED
                    if self.launches == 1
                    else AgentRuntimeStatus.ACCEPTING
                ),
                threads_alive=lambda: (),
            )
            app.state.agent_runtime = runtime
            yield

    cycles = Cycles()
    app = SimpleNamespace(state=State())
    app.state.settings = object()
    supervisor = SubjectSupervisor(
        app,
        DegradedContinuationSettings(mode="continue"),
        DegradedContinuationStatus(),
        cycle_factory=cycles,
        ready_report=_ready_report,
        reeval_delay_fn=lambda n: 300.0,
        poll_s=0.01,
    )
    supervisor.start()
    try:
        wait_until(lambda: supervisor.phase is DcState.RECONCILING, what="RECONCILING")
        assert supervisor.request_cycle_restart("hard_band_cleared") is True
        assert supervisor.wait_serving(10)
        assert cycles.launches == 2
    finally:
        supervisor.shutdown(timeout=5)
    assert supervisor.request_cycle_restart("late") is False  # nothing live


def test_the_supervisor_origin_is_set_once_and_cleared_only_by_a_non_hard_band() -> None:
    tb = FakeTimebase()
    app = SimpleNamespace(state=State())
    app.state.settings = object()
    supervisor = SubjectSupervisor(
        app,
        DegradedContinuationSettings(mode="continue"),
        DegradedContinuationStatus(),
        cycle_factory=lambda a: None,  # type: ignore[arg-type,return-value]
        ready_report=_ready_report,
        timebase=tb,
    )
    assert supervisor.hard_since is None
    first = supervisor.observe_hard_band(True)
    assert first == tb.now()
    tb._advance()
    assert supervisor.observe_hard_band(True) == first  # not restarted
    assert supervisor.hard_since == first
    assert supervisor.observe_hard_band(False) is None
    assert supervisor.hard_since is None
    tb._advance()
    assert supervisor.observe_hard_band(True) == tb.now()


def test_a_watcher_that_cannot_stop_is_the_terminal_threads_not_stopped_state(
    tmp_path: Path,
) -> None:
    """The DC-5 proof finds the watcher by ``threads_alive()``: a cycle whose
    teardown leaves it alive ends ``REFUSED_RUNNING(threads_not_stopped)``."""

    tb = FakeTimebase(step=0.5)
    box: list[SubjectSupervisor] = []
    world = HardWorld(box, DegradedContinuationStatus(), tmp_path, {})
    world.journal.band = NORMAL
    world.leak_watcher = True
    supervisor, status, app = _supervisor(world, tb=tb)
    world.status = status
    box.append(supervisor)
    supervisor.start()
    try:
        assert supervisor.wait_serving(30)
        names = supervisor._alive_names(
            supervisor._collect_sources(), supervisor._handle  # type: ignore[arg-type]
        )
        assert HARD_BAND_WATCHER_THREAD_NAME in names  # found by the method
        with supervisor._cv:
            supervisor._failed = True  # a Runtime failure: teardown + proof
            supervisor._cv.notify_all()
        wait_until(lambda: supervisor.terminal, timeout=30, what="the terminal state")
        assert supervisor.phase is DcState.REFUSED_RUNNING
        assert HARD_BAND_WATCHER_THREAD_NAME in (status.snapshot()["dc_reason"])
        assert len(world.watchers) == 1  # no second cycle started
    finally:
        for watcher in world.watchers:
            watcher.stop()
        supervisor.shutdown(timeout=5)


# ============================================================ real app


def _settings(
    tmp_path: Path,
    *,
    mode: str = "continue",
    max_bytes: int = 4096,
    max_total_bytes: int | None = None,
    retention_days: int = 30,
    **dc: Any,
) -> Settings:
    tmp_path.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp_path.chmod(0o700)
    settings = load_settings(ROOT / "config.yaml")
    return settings.model_copy(
        update={
            "memory": settings.memory.model_copy(
                update={
                    "persist_directory": tmp_path / "chroma",
                    "db1_collection": "hippocampus_dc6_test",
                    "db2_collection": "cortex_dc6_test",
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
                update={
                    "path": tmp_path / "event_journal.jsonl",
                    "max_bytes": max_bytes,
                    "max_total_bytes": max_total_bytes,
                    "soft_band_bytes": 0,
                    "retention_days": retention_days,
                }
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
            "emotion": settings.emotion.model_copy(update={"timer_enabled": False}),
            "degraded_continuation": settings.degraded_continuation.model_copy(
                update={"mode": mode, **dc}
            ),
        }
    )


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    control = SimpleNamespace(calls=0)

    def fake_load(settings: Settings) -> Any:
        control.calls += 1
        return DummyProvider()

    monkeypatch.setattr(server, "load_model_provider", fake_load)
    monkeypatch.setattr(server, "HARD_BAND_POLL_S", 0.05)
    return control


def _fast(app: FastAPI) -> SubjectSupervisor:
    supervisor: SubjectSupervisor = app.state.dc_supervisor
    supervisor._restart_delay = lambda n: 0.05
    supervisor._reeval_delay = lambda n: 0.3
    return supervisor


def _chat(client: TestClient, message: str) -> Any:
    return client.post("/api/chat", json={"message": message, "attachments": []})


def _segments_of(settings: Settings) -> list[Path]:
    return _segments(settings.event_journal.path)


def _live_bytes(settings: Settings) -> int:
    path = settings.event_journal.path
    return path.stat().st_size + sum(p.stat().st_size for p in _segments(path))


def _prepare_rotated_segments(tmp_path: Path) -> int:
    """Run 1 (an ordinary life, no capacity limit): chat until the Journal has
    rotated segments.  Returns the live bytes at the end."""

    settings = _settings(tmp_path, mode="refuse")
    with TestClient(server.create_app(settings)) as client:
        for number in range(40):
            assert _chat(client, f"hello {number}").status_code == 200
            if len(_segments_of(settings)) >= 2:
                break
    assert len(_segments_of(settings)) >= 2
    return _live_bytes(settings)


def _log_lines(settings: Settings) -> list[dict[str, Any]]:
    quarantine = settings.degraded_continuation.resolved_quarantine_dir(
        settings.event_journal.path
    )
    path = quarantine / HARD_BAND_LOG_NAME
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines()]


def _hard_threads() -> list[str]:
    return [
        t.name for t in threading.enumerate() if t.name == HARD_BAND_WATCHER_THREAD_NAME
    ]


def test_real_app_hard_at_startup_runs_the_ladder_then_starts_the_runtime(
    tmp_path: Path, provider: SimpleNamespace
) -> None:
    live = _prepare_rotated_segments(tmp_path)
    settings = _settings(
        tmp_path,
        max_total_bytes=live - 1,
        retention_days=10,
        t_hard_s=0.4,
    )
    app = server.create_app(settings)
    supervisor = _fast(app)
    with TestClient(app) as client:
        wait_until(
            lambda: supervisor.phase in (DcState.RECONCILING, DcState.SERVING),
            timeout=120,
            what="the first cycle",
        )
        # HARD at start-up: no admission, the Runtime is not started
        if supervisor.phase is DcState.RECONCILING:
            body = client.get("/health").json()
            assert body["reason"] == "journal_hard_limit"
            assert client.get("/health/dc").json()["dc_state"] in {
                "HARD_WAIT",
                "HARD_LADDER",
                "RECONCILING",
            }
        assert supervisor.wait_serving(120)
        # filtered ladder: retention_days = 10 runs only [7, 3, 1, 0]
        watcher = app.state.hard_band_watcher
        assert watcher._ladder == (7, 3, 1, 0)
        assert supervisor.cycles_launched >= 2  # a new cycle started the Runtime
        assert app.state.agent_runtime.status is AgentRuntimeStatus.ACCEPTING
        record = app.state.event_journal.inspect().latest_compaction
        assert record is not None and record.retention_days == 0  # the rung that freed
        assert app.state.event_journal.admission_status().band is NORMAL
        lines = _log_lines(settings)
        assert [x["rung_days"] for x in lines] == [10, 7, 3, 1, 0]  # 10 = rung 0
        assert [x["kind"] for x in lines][0] == "configured_window"
        assert [x["performed"] for x in lines] == [False, False, False, False, True]
        assert lines[-1]["live_bytes_after"] < lines[-1]["live_bytes_before"]
        health = client.get("/health").json()
        assert health["dc_hard_window_days"] == "0"
        assert "does not return by itself" in health["dc_hard_window_note"]
        assert health["status"] == "ok"
        # the subject answers again
        assert _chat(client, "after the ladder").status_code == 200
    assert _hard_threads() == []


def test_real_app_hard_while_serving_runs_the_ladder_at_idle_without_a_restart(
    tmp_path: Path, provider: SimpleNamespace
) -> None:
    _prepare_rotated_segments(tmp_path)
    settings = _settings(tmp_path, t_hard_s=0.4)
    app = server.create_app(settings)
    supervisor = _fast(app)
    with TestClient(app) as client:
        assert supervisor.wait_serving(120)
        assert _chat(client, "before").status_code == 200
        journal = app.state.event_journal
        runtime = app.state.agent_runtime
        launches = supervisor.cycles_launched
        # the band flips to HARD while the Runtime is running
        journal.max_total_bytes = _live_bytes(settings) - 1
        assert journal.admission_status().band is HARD
        assert _chat(client, "refused").status_code == 503
        wait_until(
            lambda: journal.admission_status().band is NORMAL,
            timeout=60,
            what="the ladder to free the Journal",
        )
        assert app.state.agent_runtime is runtime  # no restart
        assert supervisor.cycles_launched == launches
        wait_until(lambda: _chat(client, "after").status_code == 200, what="admission")
        assert any(x["performed"] for x in _log_lines(settings))
        assert client.get("/health/dc").json()["dc_hard_window_days"] == "0"
    assert _hard_threads() == []


def test_real_app_the_watcher_is_in_the_proof_and_the_cycle_stops_it(
    tmp_path: Path, provider: SimpleNamespace
) -> None:
    settings = _settings(tmp_path)
    app = server.create_app(settings)
    supervisor = _fast(app)
    with TestClient(app):
        assert supervisor.wait_serving(120)
        watcher = app.state.hard_band_watcher
        assert watcher.threads_alive() == (HARD_BAND_WATCHER_THREAD_NAME,)
        names = dict(supervisor._collect_sources())
        assert names["hard_band_watcher"] is watcher
        assert _hard_threads() == [HARD_BAND_WATCHER_THREAD_NAME]
        assert app.state.dc_status.health_fields() == {}  # NORMAL band: quiet
        # a Runtime failure: the cycle's teardown stops the watcher first
        with supervisor._cv:
            supervisor._failed = True
            supervisor._cv.notify_all()
        wait_until(
            lambda: supervisor.cycles_launched == 2 and supervisor.phase is DcState.SERVING,
            timeout=120,
            what="the supervised restart",
        )
        assert watcher.threads_alive() == ()
        assert app.state.hard_band_watcher is not watcher
        assert len(_hard_threads()) == 1
    assert _hard_threads() == []


def _hard_journal_settings(tmp_path: Path, **kwargs: Any) -> Settings:
    live = _prepare_rotated_segments(tmp_path)
    return _settings(tmp_path, max_total_bytes=live - 1, t_hard_s=0.2, **kwargs)


def test_flag_off_mode_refuse_the_hard_band_behaves_as_before(
    tmp_path: Path, provider: SimpleNamespace
) -> None:
    settings = _hard_journal_settings(tmp_path, mode="refuse")
    segments = [p.name for p in _segments_of(settings)]
    app = server.create_app(settings)
    with TestClient(app) as client:
        assert not hasattr(app.state, "dc_supervisor")
        assert not hasattr(app.state, "hard_band_watcher")
        assert _hard_threads() == []
        time.sleep(1.5)  # many T_hard
        body = client.get("/health").json()
        assert body["status"] == "degraded" and body["reason"] == "journal_hard_limit"
        assert not any(k.startswith("dc_") for k in body)
        assert _chat(client, "no").status_code == 503
        assert [p.name for p in _segments_of(settings)] == segments  # nothing archived
    assert _log_lines(settings) == []


def test_flag_off_auto_shrink_disabled_the_hard_band_behaves_as_before(
    tmp_path: Path, provider: SimpleNamespace
) -> None:
    settings = _hard_journal_settings(tmp_path, hard_band_auto_shrink=False)
    segments = [p.name for p in _segments_of(settings)]
    app = server.create_app(settings)
    supervisor = _fast(app)
    with TestClient(app) as client:
        wait_until(
            lambda: supervisor.phase is DcState.RECONCILING, timeout=120, what="RECONCILING"
        )
        launches = supervisor.cycles_launched
        time.sleep(1.5)  # many T_hard and many re-evaluations (each a new cycle)
        assert supervisor.cycles_launched > launches  # it did re-evaluate
        assert supervisor.phase is not DcState.SERVING
        assert not hasattr(app.state, "hard_band_watcher")
        assert _hard_threads() == []
        wait_until(
            lambda: client.get("/health").json().get("reason") == "journal_hard_limit",
            what="the legacy HARD reason",
        )
        wait_until(
            lambda: client.get("/health/dc").json()["dc_state"] == "RECONCILING",
            what="RECONCILING",
        )
        dc = client.get("/health/dc").json()
        assert "dc_hard_window_days" not in dc and "dc_hard_reason" not in dc
        assert [p.name for p in _segments_of(settings)] == segments
    assert _log_lines(settings) == []


# ======================================== rung 0: the configured window


def test_when_the_configured_window_suffices_nothing_is_shrunk(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, {30: "performed_clears"}, configured_days=30)
    _arm(rig)
    rig.watcher.tick()
    assert rig.compaction.calls == [30]  # no rung below the policy was tried
    snapshot = rig.status.snapshot()
    assert "dc_hard_window_days" not in snapshot and "dc_hard_window_note" not in snapshot
    assert rig.status.health_fields() == {}
    line = rig.lines()[0]
    assert line["kind"] == "configured_window" and line["performed"] is True


def test_the_ladder_starts_only_when_the_configured_window_is_not_enough(
    tmp_path: Path,
) -> None:
    rig = make_rig(
        tmp_path,
        {30: "performed", 14: "nothing_due", 7: "performed_clears"},
        configured_days=30,
    )
    _arm(rig)
    rig.watcher.tick()
    assert rig.compaction.calls == [30, 14, 7]
    # only the shrinking rung is the window the Human is told about
    assert rig.status.snapshot()["dc_hard_window_days"] == "7"


def test_a_configured_window_of_zero_still_gets_its_one_compaction(
    tmp_path: Path,
) -> None:
    rig = make_rig(tmp_path, {0: "performed_clears"}, ladder=(), configured_days=0)
    _arm(rig)
    rig.watcher.tick()
    assert rig.compaction.calls == [0]
    assert rig.hard_state() == "SERVING"
    assert "dc_hard_window_days" not in rig.status.snapshot()
    stuck = make_rig(tmp_path / "b", {}, ladder=(), configured_days=0)
    _arm(stuck)
    stuck.watcher.tick()
    assert stuck.compaction.calls == [0]
    assert stuck.status.snapshot()["dc_hard_reason"] == "ladder_empty"


# ===================================== audit log: changes only, capped


def test_a_permanently_stuck_ladder_does_not_grow_the_log(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, {}, configured_days=30)  # every rung: nothing_due
    _arm(rig)
    rig.watcher.tick()
    first = rig.lines()
    assert [x["rung_days"] for x in first] == [30, 14, 7, 3, 1, 0]
    for _ in range(40):
        rig.port.t += 400.0  # past every re-evaluation delay
        rig.watcher.tick()
    assert len(rig.compaction.calls) > 6 * 5  # it did retry
    assert rig.lines() == first  # and wrote nothing new


def test_a_change_of_outcome_and_every_performed_attempt_are_logged(
    tmp_path: Path,
) -> None:
    rig = make_rig(tmp_path, {14: "compaction_failed"})
    _arm(rig)
    rig.watcher.tick()
    rig.compaction.script = {14: "deferred_semantic"}
    rig.port.t += 400
    rig.watcher.tick()
    rig.compaction.script = {14: "performed", 7: "performed_clears"}
    rig.port.t += 400
    rig.watcher.tick()
    assert [(x["rung_days"], x["outcome"]) for x in rig.lines()] == [
        (14, "compaction_failed"),
        (14, "deferred_semantic"),
        (14, "performed"),
        (7, "performed"),
    ]


def test_the_log_is_capped_keeps_the_newest_lines_and_touches_nothing_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import suzka.runtime.hard_band_watcher as module

    monkeypatch.setattr(module, "_LOG_MAX_BYTES", 2_000)
    rig = make_rig(tmp_path, {})
    sibling = rig.log.parent / "manifest.json"
    sibling.parent.mkdir(parents=True)
    sibling.write_text("untouched")
    for number in range(60):
        rig.compaction.script = {}
        rig.watcher._last_outcome.clear()
        rig.watcher._log_attempt(14, "performed", True, number, True)
    size = rig.log.stat().st_size
    assert size < 2_000 + 400  # one line past the cap at most
    lines = rig.lines()
    assert lines and lines[-1]["live_bytes_before"] == 59  # the newest survives
    assert [x["live_bytes_before"] for x in lines] == sorted(
        x["live_bytes_before"] for x in lines
    )
    assert sorted(p.name for p in rig.log.parent.iterdir()) == [
        HARD_BAND_LOG_NAME,
        "manifest.json",
    ]
    assert sibling.read_text() == "untouched"


# ============================== a persistent unexpected error never spins


def test_a_persistent_unexpected_error_backs_off_and_is_not_shown_as_hard_stuck(
    tmp_path: Path,
) -> None:
    rig = make_rig(tmp_path, {})
    rig.journal.error = OSError(5, "EIO")  # not an EventJournalError
    rig.watcher._poll_s = 0.005
    rig.watcher.start()
    try:
        wait_until(lambda: rig.journal.status_calls >= 1, what="the first poll")
        time.sleep(0.3)  # ~60 polls; the back-off allows no more than a few ticks
        assert rig.journal.status_calls <= 3
        assert rig.watcher.threads_alive()
        assert rig.hard_state() == "SERVING"  # nothing claimed about HARD
        rig.port.t += 10_000.0  # past the back-off: it tries again
        wait_until(lambda: rig.journal.status_calls >= 2, what="a later retry")
    finally:
        rig.watcher.stop()


# ============================ a watcher that cannot start never drops the subject


def test_real_app_a_failed_watcher_start_is_not_fatal_and_leaves_no_stale_overlay(
    tmp_path: Path, provider: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(self: HardBandWatcher) -> None:
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(HardBandWatcher, "start", refuse)
    app = server.create_app(_settings(tmp_path))
    supervisor = _fast(app)
    # an overlay left by a previous cycle
    app.state.dc_status.note_hard_band(DcState.HARD_STUCK, "history_attestation_refused")
    with TestClient(app) as client:
        assert supervisor.wait_serving(120)
        assert not hasattr(app.state, "hard_band_watcher")
        assert app.state.dc_status.health_fields() == {}  # the stale overlay is gone
        assert client.get("/health").json()["status"] == "ok"
        assert _chat(client, "still here").status_code == 200
    assert _hard_threads() == []
