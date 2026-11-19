"""R18 IP-18a: the executor, the reconciler and the startup sweep
(#411 IP-18 D1 r4 §2-§4).

``R18Executor`` is a small set of threads inside the API process (so
``KillMode=control-group`` holds by construction):

* the **worker** owns the one ``TargetStore``. Rotation (S1), the forward
  write, the idempotent ``fence`` (settle) and the sweep all run here,
  serialized. Cognition never waits on it: the Runtime thread only runs the
  Outcome event (one read-only status read).
* the **reconciler** supervises the worker, settles claims that were not
  executed within 60 s, retries Outcome submits with back-off, runs the
  slow backstop scan and probes a failed target.

What drives it: an adopted ``ExecutionClaimV1``, fed by ``enqueue`` (the
approval owner's ``on_claimed`` hook, O(1) and non-blocking) and, as a
backstop, by the reconciler's scan of adopted claims without an Outcome.
There is no new "execute" R07 event.

Nothing here decides an outcome: the ``OutcomeOwnerV1`` event verifies the
target independently (the receipt is never an input, F08). "No blind
retry" (F07): only a ``TargetNotStarted`` (``BEGIN IMMEDIATE`` failed, so
nothing ran) is retried, at most twice within the Commitment deadline;
every other failure is "unknown" and is settled by a fence, never re-sent.

Known limits (documented, not defects): a settle that failed is retried by
the reconciler on the Outcome back-off, so it can wait up to 10 minutes after
the gate has re-opened. Gate OPEN transitions can flap when the target fails
at each settle (each OPEN posts one Policy scan and re-submits waiting
Intents); the probe back-off is therefore only reset by a successful settle.

Structured log lines (public-safe: ids and 8-hex digest prefixes only) are
the R20 transition trail: ``R18_EXEC ...`` and ``R18_SWEEP ...``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import logging
from pathlib import Path
import queue
import threading
import time
from typing import Final

from suzka.r16.history_reader import AdoptedRecord, AdoptedRecords
from suzka.r16.history_records import RecordKind
from suzka.r18.contracts import (
    ActionIntentV1,
    ApprovalChallengeV1,
    Direction,
    ExecutionClaimV1,
    ExecutorResult,
    VerifiedOutcomeV1,
)
from suzka.r18.facts import FactsHolder, compute_target_facts
from suzka.r18.outcome_owner import (
    TRANSIENT_OUTCOME_REASONS,
    OutcomeOwnerV1,
    OutcomeStatus,
    R18OutcomeReason,
    _OutcomeResult,
)
from suzka.r18.sweep import SweepGate, rotate_if_needed
from suzka.r18.target import (
    TargetBindingError,
    TargetCapacityError,
    TargetError,
    TargetNotStarted,
    TargetStore,
)
from suzka.runtime.agent_runtime import (
    AgentRuntimeAdmissionBlocked,
    AgentRuntimeDurabilityError,
    AgentRuntimeExecutionError,
    AgentRuntimeQueueFull,
    AgentRuntimeStopped,
)

LOGGER = logging.getLogger(__name__)

QUEUE_CAPACITY: Final = 8
JOIN_TIMEOUT_SECONDS: Final = 30.0
# A claim whose execute job has not completed this long after it was fed is
# settled by the reconciler (worker dead, queue full).
SETTLE_AFTER_SECONDS: Final = 60.0
RECONCILER_TICK_SECONDS: Final = 15.0
BACKSTOP_SECONDS: Final = 600.0
OUTCOME_BACKOFF_BASE_SECONDS: Final = 15.0
OUTCOME_BACKOFF_MAX_SECONDS: Final = 600.0
# Every submit appends a Journal event, even one that ends in a HOLD.
OUTCOME_SUBMIT_BUDGET: Final = 20
OUTCOME_WAIT_SECONDS: Final = 120.0
# TargetNotStarted: at most two more tries, after 1 s and 2 s.
NOT_STARTED_RETRY_DELAYS: Final = (1.0, 2.0)
# The adoption read before a write (review P3-c).
HISTORY_READ_RETRY_DELAYS: Final = (0.5, 1.0)
# BLOCKED_TARGET only after the target failed continuously for this long
# (one full back-off round); the probe then continues every 60 s.
BLOCKED_AFTER_SECONDS: Final = 60.0
PROBE_DELAYS: Final = (1.0, 2.0, 4.0, 8.0, 16.0, 30.0)
PROBE_STEADY_SECONDS: Final = 60.0
MAX_UNSETTLED: Final = 1024

_UTC_FORMAT: Final = "%Y-%m-%dT%H:%M:%SZ"


class _Stage(str, Enum):
    ENQUEUED = "enqueued"
    ADOPTION = "adoption"
    WRITE = "write"
    SETTLE = "settle"
    OUTCOME_SUBMIT = "outcome_submit"
    OUTCOME_COMMIT = "outcome_commit"


@dataclass(frozen=True, slots=True)
class ClaimKeys:
    """What the sweep needs from SubjectHistory beyond the adopted claims
    (IP-18b): the forward claims of earlier incarnations as
    ``(operation_key, obligation_group_key)``, and every forward key with a
    Claim in the current incarnation (adopted or not)."""

    prior: tuple[tuple[str, str], ...]
    current: frozenset[str]


@dataclass(slots=True)
class _Unsettled:
    """Everything the reconciler needs about one claim of this process."""

    claim: ExecutionClaimV1
    created: float
    # True until the target holds EFFECT or FENCED for the key.
    needs_settle: bool = True
    next_try: float = 0.0
    attempts: int = 0
    submits: int = 0
    settle_queued: bool = False


@dataclass(frozen=True, slots=True)
class _ExecuteJob:
    claim: ExecutionClaimV1


@dataclass(frozen=True, slots=True)
class _SettleJob:
    claim_id: str


@dataclass(frozen=True, slots=True)
class _OutcomeJob:
    claim_id: str


@dataclass(frozen=True, slots=True)
class _SweepJob:
    pass


@dataclass(frozen=True, slots=True)
class _BackstopJob:
    pass


_STOP: Final = object()


class _SweepRetry(Exception):
    """The sweep could not finish; the gate stays closed and it is retried."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime(_UTC_FORMAT)


# ------------------------------------------------------------ record helpers


def pending_forward_claims(records: tuple[AdoptedRecord, ...]) -> list[ExecutionClaimV1]:
    """Adopted forward claims of the current incarnation with no Outcome."""

    done = {
        item.record.operation_key
        for item in records
        if item.record_kind is RecordKind.VERIFIED_OUTCOME
        and type(item.record) is VerifiedOutcomeV1
        and item.record.direction is Direction.FORWARD
    }
    return [
        item.record
        for item in records
        if item.record_kind is RecordKind.EXECUTION_CLAIM
        and type(item.record) is ExecutionClaimV1
        and item.record.direction is Direction.FORWARD
        and item.record.operation_key not in done
    ]


def _chain_of(
    records: tuple[AdoptedRecord, ...], claim: ExecutionClaimV1
) -> tuple[ActionIntentV1, ApprovalChallengeV1] | None:
    """The claim's adopted Intent and challenge, or None (when in doubt, do
    not write)."""

    intents = [
        item.record
        for item in records
        if item.record_kind is RecordKind.ACTION_INTENT
        and type(item.record) is ActionIntentV1
        and item.record.intent_id == claim.intent_id
    ]
    challenges = [
        item.record
        for item in records
        if item.record_kind is RecordKind.APPROVAL_CHALLENGE
        and type(item.record) is ApprovalChallengeV1
        and item.record.challenge_digest == claim.challenge_digest
    ]
    if len(intents) != 1 or len(challenges) != 1:
        return None
    intent, challenge = intents[0], challenges[0]
    if (
        intent.intent_digest != claim.intent_digest
        or intent.forward_key != claim.operation_key
        or challenge.intent_id != intent.intent_id
        or challenge.nonce != claim.nonce
    ):
        return None
    return intent, challenge


def _adopted(records: tuple[AdoptedRecord, ...], claim: ExecutionClaimV1) -> bool:
    return any(
        item.record_kind is RecordKind.EXECUTION_CLAIM
        and type(item.record) is ExecutionClaimV1
        and item.record.claim_id == claim.claim_id
        and item.record.claim_digest == claim.claim_digest
        for item in records
    )


def _has_outcome(records: tuple[AdoptedRecord, ...], claim: ExecutionClaimV1) -> bool:
    return any(
        item.record_kind is RecordKind.VERIFIED_OUTCOME
        and type(item.record) is VerifiedOutcomeV1
        and item.record.direction is Direction.FORWARD
        and item.record.operation_key == claim.operation_key
        for item in records
    )


# ------------------------------------------------------------------ executor


class R18Executor:
    """The R18 forward executor (one per API process, ``r18.enabled`` only)."""

    def __init__(
        self,
        *,
        target_path: Path,
        runtime_instance_id: str,
        gate: SweepGate,
        outcome_owner: OutcomeOwnerV1,
        read_records: Callable[[], AdoptedRecords],
        has_exclusive_authority: Callable[[], bool],
        read_claim_keys: Callable[[], ClaimKeys] | None = None,
        facts: FactsHolder | None = None,
        clock: Callable[[], datetime] = _utc_now,
        monotonic: Callable[[], float] = time.monotonic,
        threaded: bool = True,
    ) -> None:
        if (
            not isinstance(target_path, Path)
            or type(gate) is not SweepGate
            or not isinstance(outcome_owner, OutcomeOwnerV1)
            or not outcome_owner.enabled
        ):
            raise TypeError("unqualified executor binding")
        self._path = target_path
        self._id = runtime_instance_id
        self._gate = gate
        self._outcome = outcome_owner
        self._read_records = read_records
        self._authority = has_exclusive_authority
        # IP-18b: S2-B / S4 run only when both are given (the server passes
        # them); without them the sweep is IP-18a's.
        self._read_claim_keys = read_claim_keys
        self._facts = facts
        self._clock = clock
        self._monotonic = monotonic
        self._threaded = threaded
        # The one short lock: never held across I/O or a blocking call.
        self._lock = threading.Lock()
        self._queue: queue.Queue[object] = queue.Queue(maxsize=QUEUE_CAPACITY)
        self._handed: set[str] = set()
        self._unsettled: dict[str, _Unsettled] = {}
        self._stop_event = threading.Event()
        self._started = False
        self._worker: threading.Thread | None = None
        self._reconciler: threading.Thread | None = None
        self._lifecycle = threading.Lock()
        # Worker-thread state.
        self._store: TargetStore | None = None
        self._sweep_done = False
        self._fail_stop = False
        # Probe schedule (guarded by ``_lock``).
        self._down_since: float | None = None
        self._probe_index = 0
        self._next_probe = 0.0
        self._sweep_queued = False
        self._next_backstop = 0.0
        self._sleep: Callable[[float], bool] = self._stop_event.wait
        self.restarts = 0

    # ------------------------------------------------------------- lifecycle

    @property
    def gate(self) -> SweepGate:
        return self._gate

    def start(self) -> None:
        """Assert the R07 lease, queue the sweep as the first job, start the
        worker and the reconciler. The gate stays closed until S5."""

        with self._lifecycle:
            if self._started:
                return
            if not self._authority():
                raise RuntimeError("the R18 executor needs the exclusive R07 lease")
            self._stop_event.clear()
            self._started = True
            self._next_backstop = self._monotonic() + BACKSTOP_SECONDS
            self._queue_sweep()
            if self._threaded:
                self._start_worker()
                reconciler = threading.Thread(
                    target=self._reconciler_loop, name="suzka-r18-reconciler", daemon=True
                )
                self._reconciler = reconciler
                reconciler.start()

    def _start_worker(self) -> None:
        worker = threading.Thread(target=self._worker_loop, name="suzka-r18-executor", daemon=True)
        self._worker = worker
        worker.start()

    def stop(self) -> None:
        """Set stop, join both threads (30 s each; a thread that does not
        join is logged and abandoned as a daemon), close the target."""

        with self._lifecycle:
            self._stop_event.set()
            try:
                self._queue.put_nowait(_STOP)
            except queue.Full:
                pass
            stuck = False
            for thread in (self._worker, self._reconciler):
                if thread is None or thread is threading.current_thread():
                    continue
                thread.join(timeout=JOIN_TIMEOUT_SECONDS)
                if thread.is_alive():
                    stuck = True
                    LOGGER.warning("R18 executor thread did not stop: %s", thread.name)
            self._worker = None
            self._reconciler = None
            self._started = False
            if not stuck:
                self._close_store()

    # --------------------------------------------------------------- enqueue

    def enqueue(self, claim: ExecutionClaimV1) -> None:
        """O(1), non-blocking (R3-1): runs on the Runtime thread inside the
        claim future's callback. One short lock, one ``put_nowait``; no
        history read, no target I/O, no ``result()``. A full queue leaves the
        claim in ``unsettled`` only, so the backstop feeds it again."""

        if type(claim) is not ExecutionClaimV1 or claim.direction is not Direction.FORWARD:
            return
        claim_id = claim.claim_id
        with self._lock:
            if claim_id in self._handed or self._stop_event.is_set():
                return
            self._handed.add(claim_id)
            if claim_id not in self._unsettled and len(self._unsettled) < MAX_UNSETTLED:
                self._unsettled[claim_id] = _Unsettled(
                    claim, self._monotonic(), next_try=self._monotonic() + SETTLE_AFTER_SECONDS
                )
            try:
                self._queue.put_nowait(_ExecuteJob(claim))
            except queue.Full:
                self._handed.discard(claim_id)
                dropped = True
            else:
                dropped = False
        self._log_exec(claim, _Stage.ENQUEUED, "DROPPED_QUEUE_FULL" if dropped else "QUEUED", "CLAIMED")

    # ----------------------------------------------------------- worker loop

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            item = self._queue.get()
            if item is _STOP:
                return
            self._run_job(item)

    def run_pending(self) -> None:
        """Run every queued job on the calling thread (tests and ``threaded=
        False``); the same code path as the worker."""

        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item is _STOP:
                return
            self._run_job(item)

    def _run_job(self, item: object) -> None:
        try:
            if isinstance(item, _SweepJob):
                self._sweep()
            elif isinstance(item, _ExecuteJob):
                self._execute(item.claim)
            elif isinstance(item, _SettleJob):
                self._settle_job(item.claim_id)
            elif isinstance(item, _OutcomeJob):
                self._outcome_job(item.claim_id)
            elif isinstance(item, _BackstopJob):
                self._backstop()
        except Exception as error:  # catch-all: log and continue
            LOGGER.warning("R18 executor job failed: %s", type(error).__name__)

    # ------------------------------------------------------------ the store

    def _open_store(self) -> TargetStore:
        if self._store is None:
            self._store = TargetStore.open(self._path)
        return self._store

    def _close_store(self) -> None:
        store, self._store = self._store, None
        if store is not None:
            try:
                store.close()
            except Exception:
                LOGGER.warning("R18 executor could not close the target")

    def _reopen(self) -> None:
        """After an unknown failure: a leaked open transaction would make every
        later BEGIN IMMEDIATE fail, and a reopen re-runs custody and schema."""

        self._close_store()

    # ------------------------------------------------------------- the sweep

    def _queue_sweep(self) -> None:
        with self._lock:
            if self._sweep_queued:
                return
            self._sweep_queued = True
        try:
            self._queue.put_nowait(_SweepJob())
        except queue.Full:
            with self._lock:
                self._sweep_queued = False

    def _sweep(self) -> None:
        """S1 rotate, S2 scan (A: source -> target, incl. prior-incarnation
        claims; B: target -> source), S3 settle, S4 publish ``TargetFacts``,
        S5 open."""

        with self._lock:
            self._sweep_queued = False
        try:
            store = self._open_store()
            rotated = rotate_if_needed(store, self._id)
            self._sweep_log("S1", f"rotated={'yes' if rotated else 'already'}")
            records = self._read_records()
            if records.blocked is not None:
                raise _SweepRetry("history")
            claims = [
                claim
                for claim in pending_forward_claims(records.records)
                # This process's claims belong to the feeders and the
                # reconciler: fencing one here would burn a live approval.
                if claim.runtime_instance_id != self._id
            ]
            self._sweep_log("S2", f"claims={len(claims)}")
            settled: list[ExecutionClaimV1] = []
            for claim in claims:
                chain = _chain_of(records.records, claim)
                if chain is None:
                    raise _SweepRetry("chain")
                row = store.fence(chain[0].obligation_group_key, Direction.FORWARD)
                self._log_exec(claim, _Stage.SETTLE, row.kind.value, "CLAIMED")
                settled.append(claim)
            claim_keys = None if self._read_claim_keys is None else self._read_claim_keys()
            fenced_prior = 0
            if claim_keys is not None:
                done = {claim.operation_key for claim in settled}
                for forward_key, group_key in claim_keys.prior:
                    if forward_key in done:
                        continue
                    try:
                        store.fence(group_key, Direction.FORWARD)
                    except TargetCapacityError:
                        # One key past the fence reserve is that key's problem
                        # (D1 §8): it keeps CLAIMED(prior), row 2, and must not
                        # keep the gate (and the subject's WRITE path) closed.
                        # Asymmetry on purpose: the S2-A fence of a CURRENT
                        # incarnation claim above still fails the whole sweep,
                        # because that claim needs its Outcome from this
                        # process; a prior claim never gets one.
                        LOGGER.warning(
                            "R18_SWEEP step=S3 prior_key=%s fence=capacity (left CLAIMED, row 2)",
                            forward_key[:8],
                        )
                        continue
                    fenced_prior += 1
            self._sweep_log("S3", f"settled={len(settled)} prior={fenced_prior}")
            if claim_keys is not None and self._facts is not None:
                # S2-B / S4. The target is read FIRST, the claim keys after:
                # a Claim precedes its effect, so every row in the snapshot
                # whose claim belongs to this incarnation is excluded.
                snapshot = store.forward_snapshot()
                current = self._read_claim_keys() if self._read_claim_keys else claim_keys
                facts = compute_target_facts(snapshot, current.current)
                self._facts.publish(facts)
                self._sweep_log("S4", f"facts={len(facts)}")
        except Exception as error:  # anything: the gate stays closed, with back-off
            self._reopen()
            self._target_failed(error)
            return
        self._sweep_done = True
        self._target_ok()
        self._sweep_log("S5", "gate=open")
        # The Outcome events come after the gate opens: they are not gated.
        for claim in settled:
            self._register_settled(claim)
            self._outcome_attempt(claim.claim_id)

    @staticmethod
    def _sweep_log(step: str, detail: str) -> None:
        LOGGER.info("R18_SWEEP step=%s %s t=%s", step, detail, _stamp(_utc_now()))

    # ------------------------------------------------------ target health

    def _target_failed(self, error: BaseException) -> None:
        """Re-close the gate, and BLOCKED_TARGET after 60 s of continuous
        failure; the probe (a full, idempotent sweep) runs on the back-off."""

        now = self._monotonic()
        with self._lock:
            if self._down_since is None:
                self._down_since = now
            elapsed = now - self._down_since
            if self._probe_index < len(PROBE_DELAYS) and elapsed < BLOCKED_AFTER_SECONDS:
                delay = PROBE_DELAYS[self._probe_index]
                self._probe_index += 1
            else:
                delay = PROBE_STEADY_SECONDS
            self._next_probe = now + delay
            blocked = elapsed >= BLOCKED_AFTER_SECONDS
        if blocked:
            self._gate.blocked()
        else:
            self._gate.sweeping()
        self._sweep_done = False
        self._sweep_log("FAIL", f"error={type(error).__name__} blocked={'yes' if blocked else 'no'}")

    def _target_ok(self) -> None:
        # Damping (review P3-d): the probe index is NOT reset here, only when a
        # settle succeeds (``_settle_ok``), so a target that fails at every
        # settle backs off further on each flap instead of re-opening the gate
        # every 1-2 s.
        with self._lock:
            self._down_since = None
        self._gate.open()

    def _settle_ok(self) -> None:
        with self._lock:
            self._probe_index = 0

    # ------------------------------------------------------- the execute job

    def _execute(self, claim: ExecutionClaimV1) -> None:
        """§2-3 in order: adoption check, process check, write, settle, hand
        to the Outcome event with the claim id only (F08)."""

        chain: tuple[ActionIntentV1, ApprovalChallengeV1] | None = None
        write = True
        # 1. Adoption check. When in doubt, do not write.
        records = self._read_with_retries(claim)
        if records is None or records.blocked is not None:
            write = False
            self._log_exec(claim, _Stage.ADOPTION, "ERROR(history)", "IN_DOUBT")
        else:
            chain = _chain_of(records.records, claim)
            if not _adopted(records.records, claim) or chain is None or _has_outcome(
                records.records, claim
            ):
                write = False
                self._log_exec(claim, _Stage.ADOPTION, "NOT_ADOPTED", "IN_DOUBT")
        # 2. Process check.
        if write and claim.runtime_instance_id != self._id:
            write = False
            self._log_exec(claim, _Stage.WRITE, "OTHER_PROCESS", "IN_DOUBT")
        # 3. Write.
        if write and chain is not None:
            self._write(claim, chain)
        # 4. Settle, always.
        if chain is None:
            self._settle_failed(claim.claim_id)
            return
        if not self._settle(claim, chain[0]):
            return
        # 5. Hand to the Outcome event.
        self._outcome_attempt(claim.claim_id)

    def _read_with_retries(self, claim: ExecutionClaimV1) -> AdoptedRecords | None:
        """The adoption read, with up to two short bounded retries (0.5 s, 1 s)
        within the deadline (review P3-c): a single transient read error must
        not burn the Human's approval into VERIFIED_ABSENT. After the retries
        the rule is unchanged: when in doubt, do not write."""

        records: AdoptedRecords | None = None
        for attempt in range(len(HISTORY_READ_RETRY_DELAYS) + 1):
            try:
                records = self._read_records()
            except Exception:
                records = None
            if records is not None and records.blocked is None:
                return records
            if attempt >= len(HISTORY_READ_RETRY_DELAYS) or not self._before_deadline(claim):
                break
            if self._sleep(HISTORY_READ_RETRY_DELAYS[attempt]):
                break
        return records

    def _write(self, claim: ExecutionClaimV1, chain: tuple[ActionIntentV1, ApprovalChallengeV1]) -> None:
        intent, challenge = chain
        attempts = 0
        while True:
            try:
                store = self._open_store()
                receipt = store.forward_write(intent, challenge, claim, now=self._clock)
            except TargetNotStarted as error:
                # Provably nothing ran: a bounded retry within the deadline.
                if attempts >= len(NOT_STARTED_RETRY_DELAYS) or not self._before_deadline(claim):
                    self._log_exec(claim, _Stage.WRITE, f"ERROR({type(error).__name__})", "IN_DOUBT")
                    return
                delay = NOT_STARTED_RETRY_DELAYS[attempts]
                attempts += 1
                if self._sleep(delay):
                    return
                continue
            except TargetBindingError as error:
                # Definite: nothing written (checked before the transaction).
                self._log_exec(claim, _Stage.WRITE, f"ERROR({type(error).__name__})", "CLAIMED")
                return
            except Exception as error:
                # Unknown: never a second write; reopen, then settle.
                self._reopen()
                self._log_exec(claim, _Stage.WRITE, f"UNKNOWN({type(error).__name__})", "IN_DOUBT")
                return
            window = "-"
            if receipt.executor_result is ExecutorResult.APPLIED:
                window = str(int(self._seconds_since(claim.claimed_utc)))
            self._log_exec(claim, _Stage.WRITE, receipt.executor_result.value, "CLAIMED", window)
            return

    def _before_deadline(self, claim: ExecutionClaimV1) -> bool:
        if claim.commitment_deadline_utc is None:
            return True
        deadline = datetime.strptime(claim.commitment_deadline_utc, _UTC_FORMAT).replace(
            tzinfo=timezone.utc
        )
        return self._clock() < deadline

    def _seconds_since(self, stamp: str) -> float:
        then = datetime.strptime(stamp, _UTC_FORMAT).replace(tzinfo=timezone.utc)
        return max(0.0, (self._clock() - then).total_seconds())

    def _settle(self, claim: ExecutionClaimV1, intent: ActionIntentV1) -> bool:
        """One idempotent fence: afterwards the target is decided for this key
        (EFFECT or FENCED). True when settled."""

        try:
            store = self._open_store()
            row = store.fence(intent.obligation_group_key, Direction.FORWARD)
        except (TargetError, OSError) as error:
            self._reopen()
            self._log_exec(claim, _Stage.SETTLE, f"ERROR({type(error).__name__})", "IN_DOUBT")
            self._settle_failed(claim.claim_id)
            self._target_failed(error)
            return False
        self._log_exec(claim, _Stage.SETTLE, row.kind.value, "IN_DOUBT")
        self._settle_ok()
        self._register_settled(claim)
        return True

    def _register_settled(self, claim: ExecutionClaimV1) -> None:
        with self._lock:
            entry = self._unsettled.get(claim.claim_id)
            if entry is None:
                if len(self._unsettled) >= MAX_UNSETTLED:
                    return
                entry = _Unsettled(claim, self._monotonic())
                self._unsettled[claim.claim_id] = entry
            entry.needs_settle = False
            entry.settle_queued = False
            entry.next_try = self._monotonic()

    def _settle_failed(self, claim_id: str, *, consumed: bool = False) -> None:
        now = self._monotonic()
        with self._lock:
            entry = self._unsettled.get(claim_id)
            if entry is None:
                return
            if consumed:
                entry.submits += 1
            entry.needs_settle = True
            entry.settle_queued = False
            entry.attempts += 1
            entry.next_try = now + self._backoff(entry.attempts)

    @staticmethod
    def _backoff(attempts: int) -> float:
        return min(OUTCOME_BACKOFF_BASE_SECONDS * (2 ** max(attempts - 1, 0)), OUTCOME_BACKOFF_MAX_SECONDS)

    def _settle_job(self, claim_id: str) -> None:
        """The reconciler's settle of a claim whose execute job never
        completed: no write, only the fence and the Outcome."""

        with self._lock:
            entry = self._unsettled.get(claim_id)
            claim = None if entry is None else entry.claim
        if claim is None:
            return
        try:
            records = self._read_records()
        except Exception:
            records = None
        chain = None
        if records is not None and records.blocked is None:
            if _has_outcome(records.records, claim):
                self._drop(claim_id)
                return
            chain = _chain_of(records.records, claim)
        if chain is None:
            self._settle_failed(claim_id)
            return
        if self._settle(claim, chain[0]):
            self._outcome_attempt(claim_id)

    # ------------------------------------------------------- the Outcome event

    def _outcome_job(self, claim_id: str) -> None:
        self._outcome_attempt(claim_id)

    def _outcome_attempt(self, claim_id: str) -> None:
        """One ``R18_OUTCOME_COMMIT`` submit, with the per-claim budget. A
        refused submit does not touch the Journal and does not consume it."""

        with self._lock:
            entry = self._unsettled.get(claim_id)
            if entry is None:
                return
            claim = entry.claim
            if self._fail_stop:
                return
            spent = entry.submits >= OUTCOME_SUBMIT_BUDGET
            # In flight: the reconciler must not queue a second submit.
            entry.next_try = self._monotonic() + OUTCOME_WAIT_SECONDS
        if spent:
            self._log_exec(claim, _Stage.OUTCOME_SUBMIT, "BUDGET_SPENT", "IN_DOUBT budget_spent")
            self._drop(claim_id)
            return
        try:
            future = self._outcome._submit_outcome(claim_id)
        except AgentRuntimeDurabilityError:
            self._fail_stop = True
            self._log_exec(claim, _Stage.OUTCOME_SUBMIT, "ERROR(AgentRuntimeDurabilityError)", "IN_DOUBT")
            return
        except (
            AgentRuntimeQueueFull,
            AgentRuntimeAdmissionBlocked,
            AgentRuntimeStopped,
            AgentRuntimeExecutionError,
        ) as error:
            self._outcome_retry(claim_id, consumed=False)
            self._log_exec(claim, _Stage.OUTCOME_SUBMIT, f"REFUSED({type(error).__name__})", "IN_DOUBT")
            return
        except Exception as error:
            self._outcome_retry(claim_id, consumed=False)
            self._log_exec(claim, _Stage.OUTCOME_SUBMIT, f"ERROR({type(error).__name__})", "IN_DOUBT")
            return
        self._log_exec(claim, _Stage.OUTCOME_SUBMIT, "SUBMITTED", "IN_DOUBT")
        try:
            value = future.result(timeout=OUTCOME_WAIT_SECONDS).value
        except AgentRuntimeDurabilityError:
            self._fail_stop = True
            return
        except Exception as error:
            self._outcome_retry(claim_id, consumed=True)
            self._log_exec(claim, _Stage.OUTCOME_COMMIT, f"ERROR({type(error).__name__})", "IN_DOUBT")
            return
        self._outcome_value(claim, value)

    def _outcome_value(self, claim: ExecutionClaimV1, value: object) -> None:
        claim_id = claim.claim_id
        if type(value) is not _OutcomeResult:
            self._outcome_retry(claim_id, consumed=True)
            return
        if value.status in (OutcomeStatus.COMMITTED, OutcomeStatus.EXISTING):
            assert value.outcome is not None
            self._log_exec(
                claim, _Stage.OUTCOME_COMMIT, value.status.value, value.outcome.outcome_state.value
            )
            self._drop(claim_id)
            return
        reason = value.reason
        assert reason is not None
        if reason is R18OutcomeReason.NOT_SETTLED:
            # Review P3-a: this submit counts, so a persistent NOT_SETTLED
            # reaches the per-claim budget like any other HOLD.
            self._settle_failed(claim_id, consumed=True)
        elif reason in TRANSIENT_OUTCOME_REASONS or reason in (
            R18OutcomeReason.TARGET_CORRUPT,
            R18OutcomeReason.SOURCE_UNAVAILABLE,
        ):
            self._outcome_retry(claim_id, consumed=True)
        else:
            # Key-level BLOCKED / conflict: stays CLAIMED (row 2), this
            # process stops spending events on it.
            self._drop(claim_id)
            self._log_exec(claim, _Stage.OUTCOME_COMMIT, "HOLD", f"BLOCKED:{reason.value}")
            return
        self._log_exec(claim, _Stage.OUTCOME_COMMIT, "HOLD", f"IN_DOUBT:{reason.value}")

    def _outcome_retry(self, claim_id: str, *, consumed: bool) -> None:
        now = self._monotonic()
        with self._lock:
            entry = self._unsettled.get(claim_id)
            if entry is None:
                return
            if consumed:
                entry.submits += 1
            entry.attempts += 1
            entry.next_try = now + self._backoff(entry.attempts)

    def _drop(self, claim_id: str) -> None:
        with self._lock:
            self._unsettled.pop(claim_id, None)

    # ---------------------------------------------------------------- backstop

    def _backstop(self) -> None:
        """The slow always-on scan: adopted claims of this incarnation without
        an Outcome that no feeder handed over (a lost callback, a restarted
        worker)."""

        try:
            records = self._read_records()
        except Exception:
            return
        if records.blocked is not None:
            return
        for claim in pending_forward_claims(records.records):
            self.enqueue(claim)

    # -------------------------------------------------------------- reconciler

    def _reconciler_loop(self) -> None:
        while not self._stop_event.wait(RECONCILER_TICK_SECONDS):
            try:
                self.tick()
            except Exception as error:  # supervised: never dies
                LOGGER.warning("R18 reconciler tick failed: %s", type(error).__name__)

    def tick(self) -> None:
        """One reconciler pass (public so tests drive it with a fake clock)."""

        now = self._monotonic()
        # Worker supervision: a BaseException that escaped.
        worker = self._worker
        if self._threaded and self._started and worker is not None and not worker.is_alive():
            if not self._stop_event.is_set():
                self.restarts += 1
                LOGGER.warning("R18 executor worker died; restarting")
                self._start_worker()
        # IP-18b: let BLOCKED_TARGET clear on time alone (hysteresis dwell).
        try:
            self._gate.poll()
        except Exception:  # pragma: no cover - defensive
            LOGGER.warning("R18 gate poll failed")
        # The probe of a failed target (a full, idempotent sweep).
        with self._lock:
            probe = not self._sweep_done and now >= self._next_probe and not self._sweep_queued
        if probe:
            self._queue_sweep()
        # The slow backstop.
        if now >= self._next_backstop:
            self._next_backstop = now + BACKSTOP_SECONDS
            self._put(_BackstopJob())
        # Unsettled claims: settle / Outcome retries.
        due: list[object] = []
        with self._lock:
            for claim_id, entry in self._unsettled.items():
                if now < entry.next_try:
                    continue
                if entry.needs_settle:
                    if not entry.settle_queued:
                        entry.settle_queued = True
                        due.append(_SettleJob(claim_id))
                else:
                    entry.next_try = now + SETTLE_AFTER_SECONDS
                    due.append(_OutcomeJob(claim_id))
        for job in due:
            if not self._put(job) and isinstance(job, _SettleJob):
                with self._lock:
                    waiting = self._unsettled.get(job.claim_id)
                    if waiting is not None:
                        waiting.settle_queued = False

    def _put(self, job: object) -> bool:
        try:
            self._queue.put_nowait(job)
            return True
        except queue.Full:
            return False

    # ------------------------------------------------------------- test hooks

    def unsettled_ids(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._unsettled)

    def handed_ids(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._handed)

    # ----------------------------------------------------------------- logging

    def _log_exec(
        self,
        claim: ExecutionClaimV1,
        stage: _Stage,
        result: str,
        state: str,
        window: str = "-",
    ) -> None:
        LOGGER.info(
            "R18_EXEC claim=%s op=%s stage=%s result=%s state=%s window_s=%s t=%s",
            claim.claim_id,
            claim.operation_key[:8],
            stage.value,
            result,
            state,
            window,
            _stamp(self._clock()),
        )


__all__ = [
    "BACKSTOP_SECONDS",
    "OUTCOME_SUBMIT_BUDGET",
    "QUEUE_CAPACITY",
    "R18Executor",
    "pending_forward_claims",
]
