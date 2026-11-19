"""R19 IP-20d: the Wake Scheduler (IP-20 D1 r3 §4-1, §4-3, §4-4, §7, §8).

``WakeScheduler`` is one daemon thread, ``suzka-r19-scheduler``, modelled on
``EmotionTimer``.  It owns **no authority**: each tick it reads the adopted
SubjectHistory and the retained Journal (read-only) and submits **at most one
item** through the owners' own submit methods; every handler re-checks everything
inside its serialized runtime event (D-R19 §5).

Per tick, by priority, the first eligible item of the first non-empty class:

1. a terminal: ``EXPIRED`` -> ``(FAILED_TERMINAL, EXPIRED_WAKE)``,
   ``EXHAUSTED`` -> ``(FAILED_TERMINAL, ATTEMPTS_EXHAUSTED)``;
2. a supersession: an unclaimed Wake that a newer Wake may supersede;
3. a review (first or resumed): a live claim with no attempt outcome and no
   ``STARTED`` review after it in the retained Journal (HD-20-2 (a));
4. the materialize scan: a nudge, the startup scan or the ``rescan_s`` backstop;
5. a claim: the oldest ``DUE`` / ``RETRYABLE`` Wake, only on an idle runtime
   (HD-20-6 (a), ``AgentRuntime.is_idle``).

A claim that commits is followed at once by ``R16_WAKE_REVIEW`` of that claim,
submitted from the claim future's done-callback (the next item never waits for
a tick, and the chain does not depend on ``is_idle``).  There is never more
than one future in flight: the claim future hands its slot to the review future.

Pre-filter.  Each submission is checked with the same pure functions the
handlers use (``claim_eligibility``, ``review_eligibility``,
``terminal_eligibility``), so the Scheduler submits only what a handler would
pass.  A handler that still ends in a no-row HOLD is paced by an in-memory
backoff (30 s, doubling, at most 300 s) and, except for ``QUEUE_DELAY`` and
``IN_FLIGHT`` (transient states of the runtime or the store, which must never
turn congestion into an ``EXPIRED_WAKE``), by a cap of 3 such events per Wake
per hour (D1 §4-3 (i)-(iii), HD-20-8).  Nothing in memory is needed for
correctness: attempts, leases, outcomes and results are rows, the one-review
bound is read from the Journal, so a restart resets only the pacing.

The Scheduler is **read-only on shared state**.  It touches no AgentState, no
Goal or Commitment; the SubjectHistory store serializes every call with its own
lock and ``EventJournal.inspect`` is already read from other threads (§4-1).
It is constructed (and its thread started) only when ``r19.enabled``, the
SubjectHistory is adopted and the issuer and owners are installed; a disabled
R19 has no thread and submits no event.  IP-21 adds ``R19_WAKE_COMPLETE``; no
Outbox or Inbox authority exists here.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import CancelledError, Future
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from functools import partial
import logging
import threading
import time
from typing import Final

from suzka.r16.activation import driver_activation
from suzka.r16.history_reader import ReaderTransient, SubjectHistoryReader
from suzka.r16.history_store import HistoryOperationRow, HistoryStoreError
from suzka.r16.successor_policy import utc_timestamp
from suzka.r19.claim_owner import ClaimStatus, TerminalStatus, WakeClaimOwnerV1
from suzka.r19.contracts import TerminalReason, WakeTiming, may_supersede
from suzka.r19.ledger import (
    SETTLED_STATES,
    UNCAPPED_HOLDS,
    WakeEntry,
    WakeHold,
    WakeState,
    WakeView,
    attempt_end_key,
    claim_eligibility,
    cycle_key,
    derive_wake_view,
    review_eligibility,
    terminal_eligibility,
    ReviewStep,
)
from suzka.r19.materializer import (
    MaterializeReason,
    MaterializeStatus,
    TRANSIENT_MATERIALIZE_REASONS,
    WakeMaterializerV1,
)
from suzka.r19.raw_state import raw_claims_of, review_started_after, wake_clock_floor
from suzka.r19.review_owner import ReviewStatus, WakeReviewOwnerV1
from suzka.runtime.agent_runtime import (
    AgentEventOutcome,
    AgentRuntime,
    AgentRuntimeAdmissionBlocked,
    AgentRuntimeDurabilityError,
    AgentRuntimeQueueFull,
    AgentRuntimeStatus,
    AgentRuntimeStopped,
)
from suzka.runtime.agent_state import AgentStateLoadError
from suzka.runtime.event_journal import EventJournal, EventJournalError, EventJournalInspection

LOGGER = logging.getLogger(__name__)

THREAD_NAME: Final = "suzka-r19-scheduler"
# D1 §4-3 backoff: 30 s, doubling, capped at 300 s (the IP-18 reconciler's pattern).
BACKOFF_FIRST_S: Final = 30.0
BACKOFF_CAP_S: Final = 300.0
# D1 §4-3 (ii): no-row HOLDs per Wake per hour, ``QUEUE_DELAY`` / ``IN_FLIGHT`` excluded.
HOLD_CAP_PER_HOUR: Final = 3
HOUR_S: Final = 3600.0
# A floor under the gap between two ticks, so a chain of completions cannot spin.
MIN_TICK_GAP_S: Final = 1.0

_MATERIALIZE_TARGET: Final = "materialize"
# A newer Wake in one of these states must not take an older one's follow-up.
_NO_SUPERSEDER: Final = frozenset(
    {
        WakeState.CORRUPT,
        WakeState.FAILED,
        WakeState.CANCELLED,
        WakeState.EXPIRED,
        WakeState.EXPIRED_PENDING_LEASE,
        WakeState.EXHAUSTED,
    }
)
_EVENT_FAILED: Final = "EVENT_FAILED"


class TickAction(str, Enum):
    """What one tick did (for logs and tests; never a state)."""

    STOPPED = "stopped"
    NOT_ACCEPTING = "not_accepting"
    BUSY = "busy"
    DRIVER_INACTIVE = "driver_inactive"
    HISTORY_UNAVAILABLE = "history_unavailable"
    # ``derive_wake_view`` raised ``TypeError``: recorded, nothing done this tick.
    DERIVE_ERROR = "derive_error"
    ERROR = "error"
    NOTHING_DUE = "nothing_due"
    # An item is eligible but cannot be submitted now (queue full, admission blocked).
    NOT_SUBMITTED = "not_submitted"
    TERMINAL = "terminal"
    SUPERSEDE = "supersede"
    REVIEW = "review"
    MATERIALIZE = "materialize"
    CLAIM = "claim"


@dataclass(frozen=True, slots=True)
class TickOutcome:
    action: TickAction
    target: str | None = None

    @property
    def submitted(self) -> bool:
        return self.action in _SUBMITTING


_SUBMITTING: Final = frozenset(
    {
        TickAction.TERMINAL,
        TickAction.SUPERSEDE,
        TickAction.REVIEW,
        TickAction.MATERIALIZE,
        TickAction.CLAIM,
    }
)


@dataclass(frozen=True, slots=True)
class _Ctx:
    """The one in-flight item."""

    action: TickAction
    target: str
    # The Wake's derived state when it was chosen (the cap resets on a change).
    state: str | None


@dataclass(slots=True)
class _Cap:
    hour_start: float
    count: int
    state: str | None


class WakePacer:
    """The in-memory pacing of D1 §4-3: a per-target backoff and an hourly cap on
    no-row HOLDs.  Pure bookkeeping over an injected monotonic clock; never read
    for correctness (a restart resets it, and only it)."""

    def __init__(self, monotonic: Callable[[], float]) -> None:
        self._monotonic = monotonic
        # (target, reason) -> (next_try, delay)
        self._backoff: dict[tuple[str, str], tuple[float, float]] = {}
        self._caps: dict[str, _Cap] = {}
        # Called from the Scheduler thread and from the runtime worker (callbacks).
        self._guard = threading.Lock()

    def blocked(self, target: str, state: str | None = None) -> bool:
        with self._guard:
            return self._blocked(target, state)

    def _blocked(self, target: str, state: str | None) -> bool:
        now = self._monotonic()
        if any(
            key[0] == target and next_try > now
            for key, (next_try, _delay) in self._backoff.items()
        ):
            return True
        cap = self._caps.get(target)
        if cap is None:
            return False
        if now - cap.hour_start >= HOUR_S or (state is not None and cap.state != state):
            # The hour is over, or the Wake's derived state changed: start afresh.
            del self._caps[target]
            return False
        return cap.count >= HOLD_CAP_PER_HOUR

    def record_hold(
        self, target: str, reason: str, state: str | None = None, *, capped: bool
    ) -> None:
        with self._guard:
            self._record_hold(target, reason, state, capped)

    def _record_hold(
        self, target: str, reason: str, state: str | None, capped: bool
    ) -> None:
        now = self._monotonic()
        key = (target, reason)
        previous = self._backoff.get(key)
        delay = (
            BACKOFF_FIRST_S
            # A hold long after the last one starts the backoff afresh.
            if previous is None or now - previous[0] > BACKOFF_CAP_S
            else min(previous[1] * 2, BACKOFF_CAP_S)
        )
        self._backoff[key] = (now + delay, delay)
        if not capped:
            return
        cap = self._caps.get(target)
        if cap is None or now - cap.hour_start >= HOUR_S:
            self._caps[target] = _Cap(now, 1, state)
        else:
            cap.count += 1
            cap.state = state

    def clear(self, target: str) -> None:
        with self._guard:
            for key in [key for key in self._backoff if key[0] == target]:
                del self._backoff[key]
            self._caps.pop(target, None)

    def prune(self, live_targets: frozenset[str]) -> None:
        """Forget targets that no longer name an open Wake (bounded memory)."""

        keep = live_targets | {_MATERIALIZE_TARGET}
        with self._guard:
            for key in [key for key in self._backoff if key[0] not in keep]:
                del self._backoff[key]
            for target in [target for target in self._caps if target not in keep]:
                del self._caps[target]

    def holds_of(self, target: str) -> int:
        with self._guard:
            cap = self._caps.get(target)
            return 0 if cap is None else cap.count


@dataclass(frozen=True, slots=True)
class _Read:
    """One tick's read-only inputs."""

    inspection: EventJournalInspection
    incarnation: str
    rows: tuple[HistoryOperationRow, ...]
    now: datetime
    view: WakeView


def _default_clock() -> datetime:
    return datetime.now(timezone.utc)


def _default_driver_active(inspection: EventJournalInspection) -> bool:
    return driver_activation(inspection) is not None


class WakeScheduler:
    """The R19 Scheduler (one thread, one item per tick, one future in flight)."""

    def __init__(
        self,
        runtime: AgentRuntime,
        journal: EventJournal,
        materializer: WakeMaterializerV1,
        claim_owner: WakeClaimOwnerV1,
        review_owner: WakeReviewOwnerV1,
        *,
        timing: WakeTiming,
        rescan_s: int,
        clock: Callable[[], datetime] = _default_clock,
        monotonic: Callable[[], float] = time.monotonic,
        driver_active: Callable[[EventJournalInspection], bool] = _default_driver_active,
        min_tick_gap_s: float = MIN_TICK_GAP_S,
    ) -> None:
        if (
            type(runtime) is not AgentRuntime
            or type(materializer) is not WakeMaterializerV1
            or type(claim_owner) is not WakeClaimOwnerV1
            or type(review_owner) is not WakeReviewOwnerV1
            or type(timing) is not WakeTiming
            or type(rescan_s) is not int
            or rescan_s < 1
        ):
            raise TypeError("unqualified Wake Scheduler binding")
        if not (
            materializer.enabled and claim_owner.enabled and review_owner.enabled
        ):
            # Constructed only for an enabled R19 (D1 §4-1, §8).
            raise ValueError("the Wake Scheduler needs enabled R19 owners")
        if claim_owner._store is not review_owner._store:
            raise ValueError("the Wake owners must share one SubjectHistory store")
        self._runtime = runtime
        self._journal = journal
        self._store = claim_owner._store
        self._materializer = materializer
        self._claims = claim_owner
        self._reviews = review_owner
        self._timing = timing
        self._rescan_s = rescan_s
        self._clock = clock
        self._monotonic = monotonic
        self._driver_active = driver_active
        self._min_tick_gap_s = float(min_tick_gap_s)
        self._pacer = WakePacer(monotonic)

        self._stop = threading.Event()
        self._wake = threading.Event()
        # One lock guards the thread handle, the in-flight slot and the nudge
        # counters; the condition lets ``wait_idle`` observe the slot.
        self._lock = threading.Lock()
        self._slot = threading.Condition(self._lock)
        self._thread: threading.Thread | None = None
        self._tick_lock = threading.Lock()
        self._future: Future[AgentEventOutcome[object]] | None = None
        self._ctx: _Ctx | None = None
        # The startup scan is request 1; a nudge is one more.  A scan that
        # finds nothing serves every request that was made before it was sent.
        self._scan_requested = 1
        self._scan_served = 0
        self._next_scan_at: float | None = None
        # Counters for tests and logs (never read for correctness).
        self.derive_errors = 0
        self.read_errors = 0
        self.submitted: dict[TickAction, int] = {}
        self._logged_orphans = False

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        """Start the thread once; a repeated start creates none."""

        with self._lock:
            if self._thread is not None or self._stop.is_set():
                return
            thread = threading.Thread(target=self._run, name=THREAD_NAME, daemon=True)
            self._thread = thread
        try:
            thread.start()
        except BaseException:
            with self._lock:
                self._thread = None
            raise

    def stop(self) -> None:
        """Stop and join before the runtime shuts down.  An accepted event is
        not cancelled (the runtime finishes it); no further one is chained."""

        self._stop.set()
        self._wake.set()
        with self._lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=30)
        with self._lock:
            if self._thread is thread:
                self._thread = None

    @property
    def running(self) -> bool:
        with self._lock:
            thread = self._thread
        return thread is not None and thread.is_alive()

    def nudge(self) -> None:
        """A cause of a Wake is terminal (the driver's ``on_cause_terminal``).
        Non-blocking by contract: it sets a counter and an event, no more."""

        if self._stop.is_set():
            return
        with self._lock:
            self._scan_requested += 1
        self._wake.set()

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """True when no future is in flight (a test and shutdown helper)."""

        with self._slot:
            return self._slot.wait_for(lambda: self._future is None, timeout)

    def _run(self) -> None:
        interval = float(self._timing.tick_s)
        while not self._stop.is_set():
            self.tick()
            if self._stop.is_set():
                return
            self._wake.wait(interval)
            self._wake.clear()
            # Completions and nudges wake the loop early; never faster than this.
            if self._stop.wait(self._min_tick_gap_s):
                return

    # ------------------------------------------------------------ the tick

    def tick(self) -> TickOutcome:
        """One read-only pass and at most one submission.  Never raises."""

        with self._tick_lock:
            try:
                return self._tick()
            except (ReaderTransient, HistoryStoreError, EventJournalError, AgentStateLoadError):
                self.read_errors += 1
                LOGGER.info("R19 scheduler: history unavailable this tick")
                return TickOutcome(TickAction.HISTORY_UNAVAILABLE)
            except Exception:
                LOGGER.warning("R19 scheduler: tick failed", exc_info=True)
                return TickOutcome(TickAction.ERROR)

    def _tick(self) -> TickOutcome:
        if self._stop.is_set():
            return TickOutcome(TickAction.STOPPED)
        if self._runtime.status is not AgentRuntimeStatus.ACCEPTING:
            return TickOutcome(TickAction.NOT_ACCEPTING)
        with self._lock:
            if self._future is not None:
                return TickOutcome(TickAction.BUSY)
        inspection = self._journal.inspect()
        # Re-read each tick: the activation can happen after the Scheduler starts.
        if not self._driver_active(inspection):
            return TickOutcome(TickAction.DRIVER_INACTIVE)
        classified = SubjectHistoryReader(self._journal, self._store).classify(
            inspection=inspection
        )
        if classified.global_block is not None:
            # D-R19 §5 (b): confirm via R07 that nothing is in doubt.
            return TickOutcome(TickAction.HISTORY_UNAVAILABLE)
        now = self._clock()
        try:
            view = derive_wake_view(classified.records, utc_timestamp(now), clock_floor=None)
        except TypeError:
            self.derive_errors += 1
            if self.derive_errors == 1 or self.derive_errors % 120 == 0:
                LOGGER.warning("R19 scheduler: the adopted Wake records could not be derived")
            return TickOutcome(TickAction.DERIVE_ERROR)
        if view.orphans and not self._logged_orphans:
            self._logged_orphans = True
            LOGGER.warning("R19 scheduler: %d adopted R19 record(s) name no Wake", len(view.orphans))
        read = _Read(inspection, classified.incarnation, classified.rows, now, view)
        self._pacer.prune(
            frozenset(e.wakeup_id for e in view.entries if e.state not in SETTLED_STATES)
        )
        for step in (
            self._next_terminal,
            self._next_supersede,
            self._next_review,
            self._next_scan,
            self._next_claim,
        ):
            chosen = step(read)
            if chosen is not None:
                return chosen
        return TickOutcome(TickAction.NOTHING_DUE)

    # ------------------------------------------------------------ pre-filters

    def _floor(self, read: _Read, entry: WakeEntry) -> datetime | None:
        return wake_clock_floor(
            read.inspection.records,
            entry.latest_claim,
            claim_lease_s=self._timing.claim_lease_s,
        )

    def _raw(self, read: _Read, entry: WakeEntry):  # type: ignore[no-untyped-def]
        return raw_claims_of(read.rows, entry.wakeup_id)

    def _blocked(self, entry: WakeEntry) -> bool:
        return self._pacer.blocked(entry.wakeup_id, entry.state.value)

    def _next_terminal(self, read: _Read) -> TickOutcome | None:
        wanted = {
            WakeState.EXPIRED: TerminalReason.EXPIRED_WAKE,
            WakeState.EXHAUSTED: TerminalReason.ATTEMPTS_EXHAUSTED,
        }
        for entry in read.view.entries:
            reason = wanted.get(entry.state)
            if reason is None or self._blocked(entry):
                continue
            if not self._terminal_passes(read, entry, reason, None):
                continue
            return self._submit(
                _Ctx(TickAction.TERMINAL, entry.wakeup_id, entry.state.value),
                partial(self._claims._submit_terminal, entry.wakeup_id, reason),
            )
        return None

    def _next_supersede(self, read: _Read) -> TickOutcome | None:
        for entry in read.view.entries:
            if entry.state not in (WakeState.SCHEDULED, WakeState.DUE) or entry.claims:
                continue
            if self._blocked(entry):
                continue
            newer = self._superseder(read, entry)
            if newer is None or not self._terminal_passes(
                read, entry, TerminalReason.SUPERSEDED, newer
            ):
                continue
            return self._submit(
                _Ctx(TickAction.SUPERSEDE, entry.wakeup_id, entry.state.value),
                partial(
                    self._claims._submit_terminal,
                    entry.wakeup_id,
                    TerminalReason.SUPERSEDED,
                    newer.wakeup_id,
                ),
            )
        return None

    @staticmethod
    def _superseder(read: _Read, entry: WakeEntry) -> WakeEntry | None:
        for other in read.view.entries:
            if (
                other.wakeup_id != entry.wakeup_id
                and other.wake.goal_id == entry.wake.goal_id
                # A newer Wake that already failed or was cancelled must not
                # take the older one's follow-up with it.
                and other.state
                not in _NO_SUPERSEDER
                and may_supersede(entry.wake, other.wake)
            ):
                return other
        return None

    def _terminal_passes(
        self,
        read: _Read,
        entry: WakeEntry,
        reason: TerminalReason,
        superseding: WakeEntry | None,
    ) -> bool:
        try:
            raw = self._raw(read, entry)
        except (TypeError, ValueError):
            LOGGER.warning("R19 scheduler: an undecodable pending claim envelope")
            return False
        check = terminal_eligibility(
            entry,
            reason,
            incarnation_id=read.incarnation,
            now=read.now,
            live=None,
            floor=self._floor(read, entry),
            raw_claims=raw,
            cycle_key_live=bool(
                self._store.live_keys((cycle_key(read.incarnation, entry.wake.logical_cycle_id),))
            ),
            superseding=superseding,
        )
        return check.proceed

    def _next_review(self, read: _Read) -> TickOutcome | None:
        for entry in read.view.entries:
            claim = entry.latest_claim
            if (
                entry.state is not WakeState.CLAIMED_LIVE
                or claim is None
                or entry.outcome_of(claim.claim_id) is not None
                or self._blocked(entry)
            ):
                continue
            try:
                raw = self._raw(read, entry)
            except (TypeError, ValueError):
                LOGGER.warning("R19 scheduler: an undecodable pending claim envelope")
                continue
            check = review_eligibility(
                entry,
                claim.claim_id,
                now=read.now,
                live=read.now,
                floor=self._floor(read, entry),
                journal_lineage_id=read.inspection.journal_lineage_id,
                incarnation_id=read.incarnation,
                raw_claims=raw,
                live_keys=self._store.live_keys(
                    (
                        cycle_key(read.incarnation, entry.wake.logical_cycle_id),
                        attempt_end_key(claim.claim_id),
                    )
                ),
                review_started=review_started_after(
                    read.inspection.records, claim.claim_sequence
                ),
            )
            # A recordable refusal also writes a row (and counts one attempt),
            # once: afterwards the claim has an outcome or a STARTED review.
            if check.step not in (ReviewStep.PROCEED, ReviewStep.REFUSE):
                continue
            return self._submit(
                _Ctx(TickAction.REVIEW, entry.wakeup_id, entry.state.value),
                partial(self._reviews._submit_review, claim.claim_id),
            )
        return None

    def _next_scan(self, read: _Read) -> TickOutcome | None:
        now = self._monotonic()
        with self._lock:
            due = self._scan_requested > self._scan_served or (
                self._next_scan_at is not None and now >= self._next_scan_at
            )
            requested = self._scan_requested
        if not due or self._pacer.blocked(_MATERIALIZE_TARGET):
            return None
        return self._submit(
            _Ctx(TickAction.MATERIALIZE, _MATERIALIZE_TARGET, str(requested)),
            self._materializer.submit_scan,
        )

    def _next_claim(self, read: _Read) -> TickOutcome | None:
        candidates = [
            entry
            for entry in read.view.entries
            if entry.state in (WakeState.DUE, WakeState.RETRYABLE)
            and not self._blocked(entry)
        ]
        if not candidates:
            return None
        # HD-20-6 (a): claim only when nothing waits and nothing runs.
        if not self._runtime.is_idle():
            return None
        for entry in candidates:
            try:
                raw = self._raw(read, entry)
            except (TypeError, ValueError):
                LOGGER.warning("R19 scheduler: an undecodable pending claim envelope")
                continue
            check = claim_eligibility(
                entry,
                incarnation_id=read.incarnation,
                now=read.now,
                live=None,
                floor=self._floor(read, entry),
                raw_claims=raw,
                cycle_key_live=bool(
                    self._store.live_keys(
                        (cycle_key(read.incarnation, entry.wake.logical_cycle_id),)
                    )
                ),
            )
            if not check.proceed:
                continue
            return self._submit(
                _Ctx(TickAction.CLAIM, entry.wakeup_id, entry.state.value),
                partial(self._claims._submit_wake_claim, entry.wakeup_id),
            )
        return None

    # ------------------------------------------------------------ submission

    def _submit(
        self,
        ctx: _Ctx,
        submit: Callable[[], Future[AgentEventOutcome[object]]],
    ) -> TickOutcome:
        # The slot is reserved under the lock, but the runtime is never called
        # while it is held: a fail-stop resolves futures (and runs their
        # callbacks, which take this lock) under the runtime's own lock.
        placeholder: Future[AgentEventOutcome[object]] = Future()
        with self._lock:
            if self._future is not None:
                return TickOutcome(TickAction.BUSY)
            self._future = placeholder
            self._ctx = ctx
        try:
            future = submit()
        except (AgentRuntimeQueueFull, AgentRuntimeAdmissionBlocked):
            # Transient: nothing was admitted; the next tick retries.
            self._release(placeholder)
            return TickOutcome(TickAction.NOT_SUBMITTED, ctx.target)
        except (AgentRuntimeStopped, AgentRuntimeDurabilityError):
            self._release(placeholder)
            return TickOutcome(TickAction.NOT_SUBMITTED, ctx.target)
        except (ValueError, TypeError):
            LOGGER.warning("R19 scheduler: a submission was refused", exc_info=True)
            self._pacer.record_hold(ctx.target, _EVENT_FAILED, ctx.state, capped=True)
            self._release(placeholder)
            return TickOutcome(TickAction.NOT_SUBMITTED, ctx.target)
        except BaseException:
            self._release(placeholder)
            raise
        with self._lock:
            self._future = future
            self.submitted[ctx.action] = self.submitted.get(ctx.action, 0) + 1
        future.add_done_callback(partial(self._done, ctx=ctx))
        return TickOutcome(ctx.action, ctx.target)

    def _release(self, future: Future[AgentEventOutcome[object]]) -> None:
        with self._slot:
            if self._future is future:
                self._future = None
                self._ctx = None
                self._slot.notify_all()

    def _done(self, future: Future[AgentEventOutcome[object]], ctx: _Ctx) -> None:
        """Runs on the runtime worker (or the submitter if already done): record
        the result, chain the review of a committed claim, free the slot.  It
        never raises and never blocks."""

        handed = False
        try:
            try:
                value = future.result().value
            except CancelledError:
                return
            except BaseException as error:
                self._failed(ctx, error)
                return
            handed = self._settle(ctx, value, future)
        except Exception:
            LOGGER.warning("R19 scheduler: a completion could not be recorded", exc_info=True)
        finally:
            if not handed:
                self._release(future)
            self._wake.set()

    def _failed(self, ctx: _Ctx, error: BaseException) -> None:
        if isinstance(error, (AgentRuntimeDurabilityError, AgentRuntimeStopped)):
            # The runtime fail-stopped or is draining: ``_tick`` sees the status
            # and submits nothing more.
            LOGGER.warning("R19 scheduler: the runtime stopped under an item")
            return
        LOGGER.warning("R19 scheduler: an item failed (%s)", type(error).__name__)
        self._pacer.record_hold(ctx.target, _EVENT_FAILED, ctx.state, capped=True)

    def _hold(self, ctx: _Ctx, reason: str, *, capped: bool) -> None:
        self._pacer.record_hold(ctx.target, reason, ctx.state, capped=capped)

    def _settle(
        self,
        ctx: _Ctx,
        value: object,
        future: Future[AgentEventOutcome[object]],
    ) -> bool:
        """Account for one result.  True when the slot was handed to a review."""

        status = getattr(value, "status", None)
        if ctx.action is TickAction.CLAIM:
            if status is ClaimStatus.COMMITTED:
                self._pacer.clear(ctx.target)
                claim = getattr(value, "claim", None)
                return claim is not None and self._chain_review(claim.claim_id, ctx, future)
            if status is ClaimStatus.HOLD:
                self._wake_hold(ctx, getattr(value, "hold", None))
            else:
                self._pacer.clear(ctx.target)
        elif ctx.action is TickAction.REVIEW:
            if status is ReviewStatus.HOLD:
                self._wake_hold(ctx, getattr(value, "hold", None))
            else:
                self._pacer.clear(ctx.target)
        elif ctx.action in (TickAction.TERMINAL, TickAction.SUPERSEDE):
            if status is TerminalStatus.HOLD:
                self._wake_hold(ctx, getattr(value, "hold", None))
            else:
                self._pacer.clear(ctx.target)
        elif ctx.action is TickAction.MATERIALIZE:
            self._settle_scan(ctx, value)
        return False

    def _wake_hold(self, ctx: _Ctx, hold: object) -> None:
        reason = hold.value if isinstance(hold, WakeHold) else _EVENT_FAILED
        self._hold(ctx, reason, capped=hold not in UNCAPPED_HOLDS)

    def _settle_scan(self, ctx: _Ctx, value: object) -> None:
        status = getattr(value, "status", None)
        requested = int(ctx.state or 0)
        now = self._monotonic()
        if status is MaterializeStatus.COMMITTED:
            # One IN_FLIGHT candidate HOLDs the whole scan and a scan writes at
            # most one Wake: rescan until nothing is left (carry-over, §3).
            self._pacer.clear(_MATERIALIZE_TARGET)
            with self._lock:
                self._scan_requested += 1
            return
        if status is MaterializeStatus.NONE_LEFT:
            self._pacer.clear(_MATERIALIZE_TARGET)
            with self._lock:
                self._scan_served = max(self._scan_served, requested)
                self._next_scan_at = now + self._rescan_s
            return
        reason = getattr(value, "reason", None)
        if reason in TRANSIENT_MATERIALIZE_REASONS:
            self._hold(
                ctx,
                reason.value,
                capped=reason is not MaterializeReason.IN_FLIGHT,
            )
            return
        # R19 disabled or a history conflict: not retried before the backstop.
        with self._lock:
            self._scan_served = max(self._scan_served, requested)
            self._next_scan_at = now + self._rescan_s

    def _chain_review(
        self,
        claim_id: str,
        ctx: _Ctx,
        claim_future: Future[AgentEventOutcome[object]],
    ) -> bool:
        """Submit the review of a committed claim from its done-callback and hand
        the single slot over.  False when it cannot be chained now: the next
        tick resumes it (step 3: a live claim with no ``STARTED`` review)."""

        if self._stop.is_set():
            return False
        review_ctx = _Ctx(TickAction.REVIEW, ctx.target, WakeState.CLAIMED_LIVE.value)
        with self._lock:
            if self._future is not claim_future:
                # Cannot happen (the slot is ours until released); never let a
                # second Future be in flight.
                LOGGER.warning("R19 scheduler: the slot is not the claim's")
                return False
        try:
            review = self._reviews._submit_review(claim_id)
        except (AgentRuntimeQueueFull, AgentRuntimeAdmissionBlocked):
            return False
        except (AgentRuntimeStopped, AgentRuntimeDurabilityError):
            return False
        except (ValueError, TypeError):
            LOGGER.warning("R19 scheduler: a review was refused", exc_info=True)
            return False
        with self._lock:
            self._future = review
            self._ctx = review_ctx
            self.submitted[TickAction.REVIEW] = self.submitted.get(TickAction.REVIEW, 0) + 1
        review.add_done_callback(partial(self._done, ctx=review_ctx))
        return True


__all__ = [
    "BACKOFF_CAP_S",
    "BACKOFF_FIRST_S",
    "HOLD_CAP_PER_HOUR",
    "HOUR_S",
    "THREAD_NAME",
    "TickAction",
    "TickOutcome",
    "WakePacer",
    "WakeScheduler",
]
