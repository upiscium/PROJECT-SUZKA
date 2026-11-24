"""F15 DC-6: the HARD-band retention ladder (#411, F15 slice D1 r3 sec. 4.4, 5.4).

The Journal's HARD band admits no new event.  Before DC-6 nothing could leave
it: compaction ran only at start-up and after a terminal completion, and a
HARD band produces no terminal completion.  With ``degraded_continuation.mode
= "continue"`` the supervisor now owns the instant HARD was first seen
(``hard_since``, kept across cycle rebuilds) and a ``HardBandWatcher`` thread
inside each cycle, after ``T_hard`` of continuous HARD, lowers the retention
window stepwise (``hard_ladder_days``, filtered to the steps below the
configured ``retention_days``) and tries one compaction per rung.

What this does and does not change (A-3, the Human's HD-F15-8 (a), which
overrides #381 D2 "the window is not shortened automatically"):

* Only what ``EventJournal.compact`` already proves safe is archived: whole
  rotated segments, never the active one, at a clean boundary, with the
  Semantic guard and the SubjectHistory attestation unchanged, byte-verified
  archives and a COMPACTION record that carries the rung's ``retention_days``.
  The ladder adds no way to remove anything.
* A shorter window lowers what the Journal retains for R15 narrative sources
  (R15 / R16 capability for events older than the effective window), makes a
  later true-rollback gate refusal (``participant_history_archived``) more
  likely, and turns older Decision proof into "derived evidence".  The window
  NEVER comes back by itself: the segments are archived.  ``/health`` and
  ``/health/dc`` say so (``dc_hard_window_days``, ``dc_hard_window_note``).
* Every rung attempt is one line in ``<quarantine_dir>/hard_band.jsonl``.
* The ladder stops at the first reason that is neither a compaction nor
  ``nothing_due`` (``previous_compaction_pending``, ``history_attestation_*``,
  ``deferred_*``, ``boundary_not_clean``, ``clock_not_monotonic``,
  ``compaction_failed``, ``journal_schema_below_4``, a runtime that never
  becomes idle): ``HARD_STUCK`` with that reason, a re-try after a growing
  ``T_reeval`` delay, never a tight loop.  HARD_STUCK is a known violation of
  "a safety stop must never stop the subject" when its cause is permanent
  (F-5), and it is shown, not hidden.
* Not here: epoch roll, ``max_total_bytes``, touching archives, a stored
  schema.  The watcher is created only with ``mode = "continue"`` AND
  ``hard_band_auto_shrink``; otherwise nothing of this module exists.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Protocol

from suzka.runtime.agent_runtime import AgentRuntimeStatus
from suzka.runtime.degraded_continuation import (
    T_LADDER_RUNG_SECONDS,
    DcState,
    DegradedContinuationStatus,
    reeval_delay,
)
from suzka.runtime.event_journal import (
    EventJournalAdmissionBand,
    EventJournalCompactionOutcome,
    EventJournalError,
)
from suzka.runtime.thread_liveness import ThreadLedger

__all__ = [
    "HARD_BAND_LOG_NAME",
    "HARD_BAND_WATCHER_THREAD_NAME",
    "HardBandPort",
    "HardBandWatcher",
]

LOGGER = logging.getLogger(__name__)

HARD_BAND_LOG_NAME = "hard_band.jsonl"
HARD_BAND_WATCHER_THREAD_NAME = "suzka-hard-band-watcher"
#: ``dc_hard_reason`` values the watcher itself produces.
REASON_LADDER_EMPTY = "ladder_empty"
REASON_LADDER_EXHAUSTED = "ladder_exhausted"
REASON_RUNTIME_NOT_IDLE = "run_at_idle_timeout"
REASON_RUNTIME_NOT_ACCEPTING = "runtime_not_accepting"
REASON_CLEARED_RESTART = "hard_band_cleared"
#: Slice of the run-at-idle wait, so a stop request is honoured within it.
_IDLE_SLICE_S = 1.0
_ENDING = frozenset(
    {
        AgentRuntimeStatus.FAILED,
        AgentRuntimeStatus.DRAINING,
        AgentRuntimeStatus.STOPPED,
    }
)


class HardBandPort(Protocol):
    """What the supervisor offers the watcher (the supervisor owns the origin
    of the HARD band, not the cycle)."""

    def now(self) -> float: ...

    def observe_hard_band(self, hard: bool) -> float | None:
        """Record one observation; returns the supervisor-held origin of the
        current continuous HARD (``None`` when the band is not HARD)."""

    def request_cycle_restart(self, reason: str) -> bool: ...


class _Compaction(Protocol):
    def compact_with_window(
        self, retention_days: int
    ) -> EventJournalCompactionOutcome: ...


class HardBandWatcher:
    """Polls the Journal band; after ``t_hard_s`` of continuous HARD (from the
    supervisor's origin) walks the ladder.  ``threads_alive()`` is part of the
    supervisor's cycle termination proof; ``stop()`` is called FIRST in a
    cycle's teardown (it uses the Runtime and the Journal)."""

    def __init__(
        self,
        *,
        journal: Any,
        compaction: _Compaction,
        runtime: Any,
        status: DegradedContinuationStatus,
        port: HardBandPort,
        ladder: tuple[int, ...],
        t_hard_s: float,
        log_path: Path | None,
        poll_s: float = 5.0,
        rung_timeout_s: float = T_LADDER_RUNG_SECONDS,
        reeval_max_backoff_s: float = 300.0,
        stop_join_s: float = 30.0,
        wall_clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._journal = journal
        self._compaction = compaction
        self._runtime = runtime
        self._status = status
        self._port = port
        self._ladder = tuple(ladder)
        self._t_hard_s = float(t_hard_s)
        self._log_path = log_path
        self._poll_s = float(poll_s)
        self._rung_timeout_s = float(rung_timeout_s)
        self._reeval_max_backoff_s = float(reeval_max_backoff_s)
        self._stop_join_s = float(stop_join_s)
        self._wall_clock = wall_clock or (lambda: datetime.now(timezone.utc))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._thread_ledger = ThreadLedger()
        # Per-cycle state only: the ORIGIN of the HARD band is the supervisor's.
        self._retry_at: float | None = None
        self._stuck_attempts = 0
        self._restart_requested = False

    # ----------------------------------------------------------- lifecycle

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("HardBandWatcher is already started")
        thread = threading.Thread(
            target=self._run, name=HARD_BAND_WATCHER_THREAD_NAME, daemon=True
        )
        self._thread = thread
        thread.start()
        self._thread_ledger.register(thread)

    def stop(self) -> bool:
        """Ask the thread to end and join it (bounded).  Idempotent.  Returns
        True when no watcher thread is alive afterwards."""

        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(self._stop_join_s)
        return not self.threads_alive()

    def threads_alive(self) -> tuple[str, ...]:
        return self._thread_ledger.alive()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as error:  # the watcher never dies silently
                LOGGER.warning("HARD-band watcher tick failed: %s", type(error).__name__)
                self._stuck(f"watcher_error:{type(error).__name__}")
            if self._stop.wait(self._poll_s):
                return

    # ---------------------------------------------------------------- tick

    def tick(self) -> None:
        """One poll.  Public so that a test can drive it with a fake clock."""

        if getattr(self._runtime, "status", None) in _ENDING:
            return  # the supervisor is ending this cycle
        try:
            band = self._journal.admission_status().band
        except EventJournalError:
            return  # not provable now: neither advance nor reset the origin
        hard = band is EventJournalAdmissionBand.HARD
        since = self._port.observe_hard_band(hard)
        if not hard or since is None:
            self._retry_at = None
            self._stuck_attempts = 0
            self._status.note_hard_band(None)
            return
        now = self._port.now()
        if now - since < self._t_hard_s:
            self._status.note_hard_band(DcState.HARD_WAIT)
            return
        if self._retry_at is not None and now < self._retry_at:
            return  # HARD_STUCK, waiting for its re-evaluation tick
        self._walk_ladder()

    # -------------------------------------------------------------- ladder

    def _runtime_started(self) -> bool:
        return getattr(self._runtime, "status", None) is AgentRuntimeStatus.ACCEPTING

    def _walk_ladder(self) -> None:
        if not self._ladder:
            self._stuck(REASON_LADDER_EMPTY)
            return
        self._status.note_hard_band(DcState.HARD_LADDER)
        for days in self._ladder:
            if self._stop.is_set():
                return
            started = self._runtime_started()
            before = self._live_bytes()
            reason, performed = self._attempt(days, started)
            self._log_attempt(days, reason, performed, before, started)
            if performed:
                self._status.note_hard_window(days)
                if not self._still_hard():
                    self._cleared(started)
                    return
                continue
            if reason == "nothing_due":
                continue
            self._stuck(reason)
            return
        if not self._still_hard():
            self._cleared(self._runtime_started())
            return
        self._stuck(REASON_LADDER_EXHAUSTED)

    def _attempt(self, days: int, started: bool) -> tuple[str, bool]:
        """One compaction attempt at ``days``.  Returns ``(reason, performed)``;
        ``reason`` is ``performed`` or the Journal's non-run reason, or one the
        watcher names itself."""

        box: list[EventJournalCompactionOutcome] = []

        def once() -> None:
            box.append(self._compaction.compact_with_window(days))

        if started:
            deadline = time.monotonic() + self._rung_timeout_s
            while True:
                if self._stop.is_set():
                    return "stopped", False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return REASON_RUNTIME_NOT_IDLE, False
                if self._runtime.run_at_idle(once, min(_IDLE_SLICE_S, remaining)):
                    break
                if not self._runtime_started():
                    return REASON_RUNTIME_NOT_ACCEPTING, False
        else:
            once()
        outcome = box[0]
        if outcome.performed:
            return "performed", True
        return outcome.reason or "compaction_failed", False

    def _still_hard(self) -> bool:
        try:
            band = self._journal.admission_status().band
        except EventJournalError:
            return True  # not provable: never claim the band cleared
        return band is EventJournalAdmissionBand.HARD

    def _cleared(self, started: bool) -> None:
        self._retry_at = None
        self._stuck_attempts = 0
        self._port.observe_hard_band(False)
        self._status.note_hard_band(None)
        if not started and not self._restart_requested:
            # The Runtime was never started in this cycle (HARD at start-up):
            # a new cycle starts it through the unchanged start block.
            self._restart_requested = self._port.request_cycle_restart(
                REASON_CLEARED_RESTART
            )

    def _stuck(self, reason: str) -> None:
        """Stop the ladder and show why; the next pass is a ``T_reeval`` tick
        away (1, 4, 16, 60, 300 s...), so this can never spin."""

        delay = reeval_delay(self._stuck_attempts, self._reeval_max_backoff_s)
        self._stuck_attempts += 1
        self._retry_at = self._port.now() + delay
        self._status.note_hard_band(DcState.HARD_STUCK, reason)
        LOGGER.warning("HARD-band ladder stopped: %s (next try in %.0f s)", reason, delay)

    # --------------------------------------------------------------- audit

    def _live_bytes(self) -> int | None:
        try:
            return int(self._journal.admission_status().live_bytes)
        except Exception:
            return None

    def _log_attempt(
        self,
        days: int,
        reason: str,
        performed: bool,
        before: int | None,
        started: bool,
    ) -> None:
        """One line per attempt; a failure to write is logged, never raised
        (the audit line must not stop the subject either)."""

        if self._log_path is None:
            return
        line = json.dumps(
            {
                "ts": self._wall_clock().isoformat(),
                "rung_days": days,
                "outcome": reason,
                "performed": performed,
                "live_bytes_before": before,
                "live_bytes_after": self._live_bytes(),
                "runtime_started": started,
            },
            sort_keys=True,
        )
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                self._log_path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                os.write(descriptor, (line + "\n").encode("utf-8"))
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            LOGGER.warning("could not append to the HARD-band log")
