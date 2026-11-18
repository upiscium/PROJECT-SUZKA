"""R18 IP-17b: the Approval owner (#411 D1 r7 §1, §3, §3b, §8a, §12-1).

``ApprovalChallengeOwnerV1`` owns three events and is the only writer of
``approval_challenges``, ``approval_expirations`` and (IP-17c) the nonce
consumption + ``execution_claims`` (through the subject.history record
participant):

* ``R18_CHALLENGE_ISSUE(prebuilt challenge, attempt, code)`` (source
  ``api.r18.approval``): written only at an operator confirm that carries the
  matching typed code.  Refused in the soft band (HP-8).
* ``R18_APPROVAL_EXPIRE(challenge | scan(after))`` (source
  ``runtime.r18.approval_expire``): written by a timer at ``expires + 1 s``,
  by the startup scan (BOOT_CHANGED) and by the lazy path.  ADMITTED in the
  soft band (HD-6 (a)): it only records that a challenge ended.

* ``R18_EXECUTION_CLAIM(nonce, code)`` (IP-17c, source ``api.r18.approval``,
  §6): nonce consumption and the claim in ONE R07 event.  Refused in the soft
  band (HP-8).  It re-checks everything §6-1 lists and writes the CLAIM only;
  it causes no effect (the executor is IP-18, §7).

All are gated by ``r18.enabled`` (HOLD ``R18_DISABLED`` before any read) and
are FORWARD only (an inverse is HOLD ``INVERSE_DEFERRED``).  A HOLD writes
nothing.  No handler waits for an operator.

The review is not an event (§3 / §4): ``review`` reads the ledger facts and
the target, builds the challenge IN MEMORY and writes nothing.  The pending
map holds at most one challenge per Intent; a live challenge (written, else
pending) is never replaced, so repeated reviews cannot keep the Human from
confirming.  The review does not requalify the Decision's sources: that
needs the live domain state of the Runtime thread (``Plan preview._qualify``
reads it against the running event), so it is done by ``R18_CHALLENGE_ISSUE``
itself, which re-runs the whole of §2 steps 2 and 4-7 at the confirm.

``boot_id`` (HD-10 (b)) is a fresh UUID4 per owner, i.e. per API process.

§3b: every ``approval_challenges_insert`` / ``approval_expirations_insert``
condition and UNIQUE is pre-checked from the same classification, so the
FINALIZE after the R07 commit cannot refuse what PREPARE admitted; an
operation row of the same Intent that is not adopted yet is IN_FLIGHT.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
import hmac
import logging
import secrets
import threading
import time
from typing import TYPE_CHECKING, Final, Protocol
from uuid import uuid4

from suzka.motivation.commitment import CommitmentLifecycle
from suzka.r16.decision_issuer import _sequence
from suzka.r16.history_attestation_digest import operations_digest
from suzka.r16.history_participant import (
    JournalCommitProofSource,
    SubjectHistoryRecordParticipant,
)
from suzka.r16.history_reader import ReaderClassification
from suzka.r16.history_records import (
    MAX_CHALLENGES_PER_INTENT,
    ApprovalExpirationV1,
    ExpirationReason,
    HistoryOperationKind,
    HistoryRecordOperationV1,
    RecordKind,
)
from suzka.r16.history_store import HistoryPhase, HistoryStoreError
from suzka.r18.admission import ALWAYS_OPEN_UNTIL_IP18, AdmissionGate, gate_open
from suzka.r18.contracts import (
    INVERSE_INTENT_ID_PREFIX,
    MAX_APPROVAL_TTL,
    ActionIntentV1,
    ApprovalChallengeV1,
    ApprovalChannel,
    Direction,
    ExecutionClaimV1,
    PolicyVerdict,
    SourceRole,
)
from suzka.r16.successor_policy import LedgerState
from suzka.r18.ledger import RecordAt, derive_ledger
from suzka.r18.requalify import RequalifyFailed
from suzka.r18.policy_owner import (
    ReviewArtifactPolicyV1,
    _adopted_intent,
    _bind,
    _Deny,
    _forecast_for,
    _PolicyHold,
    _verdict_for,
)
from suzka.runtime.agent_runtime import (
    AgentEvent,
    AgentEventOutcome,
    AgentEventSource,
    AgentEventType,
    AgentRuntime,
)
from suzka.runtime.agent_state import AgentStateLoadError
from suzka.runtime.event_journal import TransactionKind
from suzka.runtime.transaction_coordinator import (
    CoordinatedResult,
    TransactionCoordinator,
)

if TYPE_CHECKING:
    from suzka.r18.requalify import Requalified

LOGGER = logging.getLogger(__name__)

UTC_FORMAT: Final = "%Y-%m-%dT%H:%M:%SZ"
# The timer fires this long after ``expires_utc`` (D1 §3).
EXPIRE_DELAY: Final = timedelta(seconds=1)
# Bounds: pending challenges (one per Intent), chained startup scans, and the
# timer's retries of a transient HOLD (the lazy path / the window cover the rest).
MAX_PENDING_CHALLENGES: Final = 64
MAX_EXPIRE_SCANS_PER_STARTUP: Final = 64
MAX_TIMER_RETRIES: Final = 3
# Review coalescing (condition B): a review that has to read the ledger does so
# at most once per interval per Intent, and one read at a time.
MIN_REVIEW_INTERVAL: Final = 2.0
MAX_REVIEW_CACHE: Final = 256
# The list route shows at most this many open Intents (the newest).
MAX_LISTED_APPROVALS: Final = 16
# The startup scan retries a transient HOLD that names no challenge once.
STARTUP_SCAN_RETRY_DELAY: Final = 1.0
_MAX_ID: Final = 256
_EVENT_TYPES: Final = frozenset(
    {
        AgentEventType.R18_CHALLENGE_ISSUE,
        AgentEventType.R18_APPROVAL_EXPIRE,
        AgentEventType.R18_EXECUTION_CLAIM,
    }
)


def _fmt(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime(UTC_FORMAT)


def _parse(value: str) -> datetime:
    return datetime.strptime(value, UTC_FORMAT).replace(tzinfo=timezone.utc)


class ApprovalStatus(str, Enum):
    ISSUED = "challenge_issued"
    # The nonce is already written, live and unconsumed: go straight to the
    # CLAIM (IP-17c); nothing is written.
    EXISTING = "challenge_existing"
    EXPIRED = "approval_expired"
    EXPIRE_EXISTING = "approval_expire_existing"
    # IP-17c: the nonce is consumed and the claim written, in one event.
    CLAIMED = "execution_claimed"
    NONE_LEFT = "approval_none_left"
    HOLD = "hold"


class ReviewStatus(str, Enum):
    WRITTEN = "review_written_live"
    PENDING = "review_pending_live"
    BUILT = "review_built"
    HOLD = "hold"


class R18ApprovalReason(str, Enum):
    """Why the Approval owner held; a closed, source-free vocabulary."""

    R18_DISABLED = "r18_disabled"
    HISTORY_UNAVAILABLE = "history_unavailable"
    HISTORY_CONFLICT = "history_conflict"
    INTENT_UNKNOWN = "intent_unknown"
    NOT_YET_TERMINAL = "not_yet_terminal"
    INVERSE_DEFERRED = "inverse_deferred"
    FORECAST_MISSING = "forecast_missing"
    BINDING_MISMATCH = "binding_mismatch"
    HEAD_UNAVAILABLE = "head_unavailable"
    HEAD_CONFLICT = "head_conflict"
    LEDGER_UNSAFE = "ledger_unsafe"
    SOURCE_UNAVAILABLE = "source_unavailable"
    SOURCE_CONFLICT = "source_conflict"
    DEADLINE_ELAPSED = "deadline_elapsed"
    OBLIGATION_SECRET_UNAVAILABLE = "obligation_secret_unavailable"
    TARGET_UNAVAILABLE = "target_unavailable"
    ADMISSION_GATE_CLOSED = "admission_gate_closed"
    IN_FLIGHT = "in_flight"
    HISTORY_CAPACITY = "history_capacity"
    # Drift after ALLOW is a HOLD here, never a DENY (HD-9 (a), §6-3).
    SOURCE_NOT_CURRENT = "source_not_current"
    TARGET_MISMATCH = "target_mismatch"
    POLICY_PENDING = "policy_pending"
    POLICY_DENIED = "policy_denied"
    WINDOW_CLOSED = "window_closed"
    # The challenge is not (or no longer) live: expired or another boot.
    CHALLENGE_NOT_LIVE = "challenge_not_live"
    # A confirm names a nonce that is neither written nor pending.
    UNKNOWN_CHALLENGE = "unknown_challenge"
    CODE_MISMATCH = "code_mismatch"
    CHALLENGE_STALE = "challenge_stale"
    CHALLENGE_ALREADY_OPEN = "challenge_already_open"
    PREVIOUS_CHALLENGE_OPEN = "previous_challenge_open"
    NONCE_USED = "nonce_used"
    ATTEMPTS_EXHAUSTED = "attempts_exhausted"
    PENDING_FULL = "pending_full"
    # R18_APPROVAL_EXPIRE.
    CHALLENGE_UNKNOWN = "challenge_unknown"
    STILL_LIVE = "still_live"
    ALREADY_CONSUMED = "already_consumed"
    # A review is being computed for another caller (coalesced, not an error).
    REVIEW_BUSY = "review_busy"
    # The event could not be admitted or did not finish in time (soft or hard
    # journal band, a stopped Runtime, a timeout): nothing was written.
    JOURNAL_UNAVAILABLE = "journal_unavailable"


# Transient: the timer retries these (bounded); the rest wait for the lazy
# path, the window end or the next restart's scan.
TRANSIENT_APPROVAL_REASONS: Final = frozenset(
    {
        R18ApprovalReason.NOT_YET_TERMINAL,
        R18ApprovalReason.IN_FLIGHT,
        R18ApprovalReason.HISTORY_UNAVAILABLE,
        R18ApprovalReason.STILL_LIVE,
    }
)


@dataclass(frozen=True, slots=True)
class ApprovalResult:
    """The R18_CHALLENGE_ISSUE / R18_APPROVAL_EXPIRE value (public-safe: a
    status, ids and a closed reason; ``challenge`` is held for in-process use).
    """

    status: ApprovalStatus
    intent_id: str | None
    reason: R18ApprovalReason | None = None
    attempt: int | None = None
    challenge: ApprovalChallengeV1 | None = None
    expiration: ApprovalExpirationV1 | None = None
    # A HOLD PREVIOUS_CHALLENGE_OPEN names the dead challenge to expire first.
    challenge_digest: str | None = None
    claim: ExecutionClaimV1 | None = None

    def __post_init__(self) -> None:
        if type(self.status) is not ApprovalStatus:
            raise TypeError("approval status must be exact")
        if (self.status is ApprovalStatus.HOLD) != (self.reason is not None):
            raise ValueError("exactly a HOLD carries a reason")
        if self.reason is not None and type(self.reason) is not R18ApprovalReason:
            raise TypeError("approval reason must be an exact R18ApprovalReason")

    @property
    def is_authoritative(self) -> bool:
        return self.status in (
            ApprovalStatus.ISSUED,
            ApprovalStatus.EXISTING,
            ApprovalStatus.EXPIRED,
            ApprovalStatus.EXPIRE_EXISTING,
            ApprovalStatus.CLAIMED,
        )


@dataclass(frozen=True, slots=True)
class ApprovalFacts:
    """One Intent's approval facts, from one classification (IP-17c, §4:
    ledger facts only; no event, no challenge material, no code)."""

    intent_id: str
    ledger_state: str
    decision_id: str
    window_end_utc: str
    attempts_used: int
    live_challenge: bool
    challenge_expires_utc: str | None
    commitment_id: str
    commitment_deadline_utc: str | None
    target_instance_id: str
    operation_key: str


@dataclass(frozen=True, slots=True)
class FactsResult:
    """A facts read: the facts of one Intent (``status``), of every open
    Intent (``open_approvals``), or a closed HOLD reason."""

    reason: R18ApprovalReason | None = None
    items: tuple[ApprovalFacts, ...] = ()

    def __post_init__(self) -> None:
        if self.reason is not None and self.items:
            raise ValueError("a HOLD carries no facts")


@dataclass(frozen=True, slots=True)
class ReviewResult:
    """What a review shows: a challenge (never written by the review) or a HOLD."""

    status: ReviewStatus
    intent_id: str
    reason: R18ApprovalReason | None = None
    challenge: ApprovalChallengeV1 | None = None
    attempt: int | None = None
    window_end_utc: str | None = None
    # The Intent the challenge binds (the approval page recomputes its digest).
    intent: ActionIntentV1 | None = None

    def __post_init__(self) -> None:
        if (self.status is ReviewStatus.HOLD) != (self.reason is not None):
            raise ValueError("exactly a HOLD carries a reason")
        if (self.status is not ReviewStatus.HOLD) != (self.challenge is not None):
            raise ValueError("exactly a non-HOLD review carries a challenge")


class _Timer(Protocol):
    def start(self) -> None: ...

    def cancel(self) -> None: ...


TimerFactory = Callable[[float, Callable[[], None]], _Timer]


def _thread_timer(delay: float, callback: Callable[[], None]) -> _Timer:
    timer = threading.Timer(max(delay, 0.0), callback)
    timer.daemon = True
    return timer


class _ApprovalHold(Exception):
    def __init__(
        self,
        reason: R18ApprovalReason,
        intent_id: str | None = None,
        challenge_digest: str | None = None,
    ) -> None:
        super().__init__(reason.value)
        self.reason = reason
        self.intent_id = intent_id
        self.challenge_digest = challenge_digest


class _ApprovalCommitRefused(RuntimeError):
    """The pre-WAL check failed: a runtime fail-stop (no source text)."""


@dataclass(frozen=True, slots=True)
class _Target:
    """The handler's request."""

    kind: AgentEventType
    challenge: ApprovalChallengeV1 | None = None
    code: str | None = None
    challenge_digest: str | None = None
    after: str | None = None
    scan: bool = False
    # R18_EXECUTION_CLAIM: the confirmed nonce (written) and its Intent.
    nonce: str | None = None
    intent_id: str | None = None


@dataclass(frozen=True, slots=True)
class _PendingOperation:
    event: AgentEvent
    incarnation: str
    snapshot_hash: str
    rows_digest: tuple[int, str]
    transaction_id: str
    operation: HistoryRecordOperationV1
    initial_state: object
    kind: HistoryOperationKind


@dataclass(frozen=True, slots=True)
class _Facts:
    """One intent's approval facts from one classification."""

    intent: ActionIntentV1
    challenges: tuple[ApprovalChallengeV1, ...]  # processing order = attempts
    expired: frozenset[str]
    consumed: frozenset[str]
    expiries: tuple[ApprovalExpirationV1, ...]


def _valid_id(value: object) -> bool:
    return type(value) is str and 0 < len(value) <= _MAX_ID


class ApprovalChallengeOwnerV1:
    """The R18 Approval owner: challenge issue and expiry for forward Intents."""

    def __init__(
        self,
        policy_owner: ReviewArtifactPolicyV1,
        runtime: AgentRuntime,
        *,
        enabled: bool,
        ttl_seconds: int,
        admission_gate: AdmissionGate = ALWAYS_OPEN_UNTIL_IP18,
        boot_id: str | None = None,
        runtime_instance_id: str | None = None,
        timer_factory: TimerFactory = _thread_timer,
        on_claimed: Callable[[ExecutionClaimV1], None] | None = None,
    ) -> None:
        if (
            type(policy_owner) is not ReviewArtifactPolicyV1
            or type(runtime) is not AgentRuntime
            or policy_owner._runtime is not runtime
            or type(enabled) is not bool
            or type(ttl_seconds) is not int
            or not 1 <= ttl_seconds <= MAX_APPROVAL_TTL.total_seconds()
            or not isinstance(admission_gate, AdmissionGate)
        ):
            raise TypeError("unqualified Approval owner binding")
        self._policy = policy_owner
        self._decision = policy_owner._decision
        self._preview = policy_owner._preview
        self._runtime = runtime
        self._store = policy_owner._store
        self._journal = policy_owner._journal
        self._recovery = policy_owner._recovery
        self._enabled = enabled
        self._ttl = timedelta(seconds=ttl_seconds)
        self._clock: Callable[[], datetime] = policy_owner._clock
        self._gate = admission_gate
        # HD-10 (b): a fresh UUID per API process (this owner's lifetime).
        self._boot_id = boot_id if boot_id is not None else str(uuid4())
        # IP-17c (§6-1 step 7): the claim's ``runtime_instance_id`` is a
        # per-process UUID4 too (the executor, IP-18, checks the same value).
        self._runtime_instance_id = (
            runtime_instance_id if runtime_instance_id is not None else str(uuid4())
        )
        self._timer_factory = timer_factory
        # IP-18a (§2-2): the executor's feeder; never changes an API result.
        self._on_claimed = on_claimed
        self._inflight: tuple[object, _Target] | None = None
        self._pending: _PendingOperation | None = None
        self._lock = threading.Lock()
        # intent_id -> the one pending in-memory challenge (never written).
        self._pending_challenges: dict[str, ApprovalChallengeV1] = {}
        self._timers: dict[str, _Timer] = {}
        # Review coalescing: the live challenge shown per Intent, the last
        # computed result per Intent, and one ledger read at a time.
        self._shown: dict[str, ReviewResult] = {}
        self._reviewed: dict[str, tuple[float, ReviewResult]] = {}
        self._review_lock = threading.Lock()
        # Facts reads (list / status): one ledger read at a time, 2 s reuse.
        self._facts_lock = threading.Lock()
        self._facts_cache: dict[str, tuple[float, FactsResult]] = {}
        self._monotonic: Callable[[], float] = time.monotonic
        self._stop_event = threading.Event()
        self._stopped = False
        self._scan_thread: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def boot_id(self) -> str:
        return self._boot_id

    @property
    def runtime_instance_id(self) -> str:
        return self._runtime_instance_id

    # ----------------------------------------------------------- submission

    def _submit_issue(
        self, challenge: ApprovalChallengeV1, code: str
    ) -> Future[AgentEventOutcome[object]]:
        """``R18_CHALLENGE_ISSUE(prebuilt challenge, attempt, code)``; the
        handler computes the attempt itself."""

        if type(challenge) is not ApprovalChallengeV1 or type(code) is not str:
            raise ValueError("an issue names one challenge and its typed code")
        return self._submit(
            _Target(AgentEventType.R18_CHALLENGE_ISSUE, challenge=challenge, code=code),
            AgentEventSource.API_R18_APPROVAL,
        )

    def _submit_expire(self, challenge_digest: str) -> Future[AgentEventOutcome[object]]:
        """``R18_APPROVAL_EXPIRE(challenge)``."""

        if not _valid_id(challenge_digest):
            raise ValueError("an expiry names one challenge digest")
        return self._submit(
            _Target(AgentEventType.R18_APPROVAL_EXPIRE, challenge_digest=challenge_digest),
            AgentEventSource.RUNTIME_R18_APPROVAL_EXPIRE,
        )

    def _submit_expire_scan(self, after: str | None) -> Future[AgentEventOutcome[object]]:
        """``R18_APPROVAL_EXPIRE(scan(after))``: ``None`` scans from the start."""

        if after is not None and not _valid_id(after):
            raise ValueError("an expiry scan cursor is one challenge digest")
        return self._submit(
            _Target(AgentEventType.R18_APPROVAL_EXPIRE, after=after, scan=True),
            AgentEventSource.RUNTIME_R18_APPROVAL_EXPIRE,
        )

    def _submit_claim(
        self, intent_id: str, nonce: str, code: str
    ) -> Future[AgentEventOutcome[object]]:
        """``R18_EXECUTION_CLAIM(nonce, code)`` (IP-17c): nonce consumption and
        the claim in one R07 event.  Only ``approval_api`` reaches this, through
        ``confirm`` / ``claim`` (an architecture test pins it)."""

        if not (_valid_id(intent_id) and _valid_id(nonce) and type(code) is str):
            raise ValueError("a claim names one Intent, one nonce and its typed code")
        return self._submit(
            _Target(
                AgentEventType.R18_EXECUTION_CLAIM,
                code=code,
                nonce=nonce,
                intent_id=intent_id,
            ),
            AgentEventSource.API_R18_APPROVAL,
        )

    def _submit(
        self, target: _Target, source: AgentEventSource
    ) -> Future[AgentEventOutcome[object]]:
        nonce = object()

        def handle() -> object:
            if self._inflight is not None or self._pending is not None:
                raise _ApprovalCommitRefused("approval operation already pending")
            self._inflight = (nonce, target)
            try:
                return self._commit(nonce, source)
            except BaseException:
                self._pending = None
                raise
            finally:
                self._inflight = None

        return self._runtime.submit(target.kind, source, handle)

    def _current_event(self, nonce: object, source: AgentEventSource) -> AgentEvent:
        event = self._runtime.current_event()
        if (
            self._inflight is None
            or self._inflight[0] is not nonce
            or event is None
            or event.event_type is not self._inflight[1].kind
            or event.source is not source
            or type(event.processing_sequence) is not int
            or event.processing_sequence < 1
        ):
            raise _ApprovalCommitRefused("unbound approval event")
        return event

    # -------------------------------------------------------------- handler

    def _commit(self, nonce: object, source: AgentEventSource) -> object:
        self._pending = None
        event = self._current_event(nonce, source)
        assert self._inflight is not None
        target = self._inflight[1]
        intent_id = (
            target.intent_id
            if target.challenge is None
            else target.challenge.intent_id
        )
        try:
            if target.kind is AgentEventType.R18_CHALLENGE_ISSUE:
                return self._issue_checked(event, target)
            if target.kind is AgentEventType.R18_EXECUTION_CLAIM:
                return self._claim_checked(event, target)
            return self._expire_checked(event, target)
        except (HistoryStoreError, AgentStateLoadError):
            return self._hold(R18ApprovalReason.HISTORY_UNAVAILABLE, intent_id)
        except _ApprovalHold as hold:
            return self._hold(hold.reason, hold.intent_id or intent_id, hold.challenge_digest)
        except _PolicyHold as hold:
            return self._hold(_map(hold.reason.name), hold.intent_id or intent_id)
        except _Deny as denied:
            return self._hold(_map(denied.reason.name), intent_id)

    @staticmethod
    def _hold(
        reason: R18ApprovalReason,
        intent_id: str | None,
        challenge_digest: str | None = None,
    ) -> ApprovalResult:
        LOGGER.info("R18 approval owner result=hold reason=%s", reason.value)
        return ApprovalResult(
            ApprovalStatus.HOLD, intent_id, reason, challenge_digest=challenge_digest
        )

    # ------------------------------------------------------- shared facts

    def _facts(
        self, classified: ReaderClassification, intent_id: str
    ) -> _Facts:
        intent = self._policy._target_intent(classified, intent_id)
        ordered = sorted(
            classified.records, key=lambda item: (item.processing_sequence, item.transaction_id)
        )
        challenges: list[ApprovalChallengeV1] = []
        expiries: list[ApprovalExpirationV1] = []
        consumed: set[str] = set()
        for adopted in ordered:
            record = adopted.record
            if type(record) is ApprovalChallengeV1 and record.intent_id == intent_id:
                challenges.append(record)
            elif type(record) is ApprovalExpirationV1 and record.intent_id == intent_id:
                expiries.append(record)
            elif type(record) is ExecutionClaimV1 and record.intent_id == intent_id:
                consumed.add(record.challenge_digest)
        return _Facts(
            intent,
            tuple(challenges),
            frozenset(item.challenge_digest for item in expiries),
            frozenset(consumed),
            tuple(expiries),
        )

    def _forward_facts(
        self, classified: ReaderClassification, intent_id: str, now: datetime
    ) -> tuple[_Facts, str]:
        """§2 step 2-3 for an approval: the Intent, an ALLOW, the forecast
        (returns its digest), an open window and a pending ledger."""

        facts = self._facts(classified, intent_id)
        intent = facts.intent
        verdict = _verdict_for(classified, intent_id)
        if verdict is None:
            raise _ApprovalHold(R18ApprovalReason.POLICY_PENDING, intent_id)
        if verdict.verdict is not PolicyVerdict.ALLOW:
            raise _ApprovalHold(R18ApprovalReason.POLICY_DENIED, intent_id)
        if (verdict.direction, verdict.intent_digest, verdict.operation_key) != (
            Direction.FORWARD,
            intent.intent_digest,
            intent.forward_key,
        ):
            raise _ApprovalHold(R18ApprovalReason.HISTORY_CONFLICT, intent_id)
        forecast = _forecast_for(classified, intent_id)
        if forecast is None:
            raise _ApprovalHold(R18ApprovalReason.FORECAST_MISSING, intent_id)
        _bind(forecast, intent)
        if not intent.window_open(now):
            raise _ApprovalHold(R18ApprovalReason.WINDOW_CLOSED, intent_id)
        return facts, forecast.forecast_digest

    def _live_written(
        self, facts: _Facts, now: datetime
    ) -> list[tuple[int, ApprovalChallengeV1]]:
        """The written challenges that are live and unconsumed (with attempt)."""

        stamp = _fmt(now)
        return [
            (attempt, challenge)
            for attempt, challenge in enumerate(facts.challenges, start=1)
            if challenge.challenge_digest not in facts.expired
            and challenge.challenge_digest not in facts.consumed
            and challenge.is_live(stamp, self._boot_id)
        ]

    def _in_flight(
        self, classified: ReaderClassification, intent_id: str, kinds: frozenset[HistoryOperationKind]
    ) -> bool:
        """§3b: an operation row of this Intent (challenge / expiry / claim)
        that is not adopted yet holds its keys and its attempt count."""

        adopted = [item.record for item in classified.records]
        for row in classified.rows:
            if (
                row.operation_kind not in kinds
                or row.phase not in (HistoryPhase.PREPARED, HistoryPhase.FINALIZED)
                or row.pending_bytes is None
            ):
                continue
            try:
                operation = HistoryRecordOperationV1.from_bytes(row.pending_bytes)
            except (TypeError, ValueError):
                raise _ApprovalHold(R18ApprovalReason.HISTORY_CONFLICT, intent_id) from None
            record = operation.record
            if getattr(record, "intent_id", None) == intent_id and record not in adopted:
                return True
        return False

    def _commitment_deadline(self, intent: ActionIntentV1) -> datetime | None:
        """The Intent's Commitment deadline (read-only), whole seconds floored."""

        ref = intent.artifact.source(SourceRole.COMMITMENT)
        record = self._preview._loop._commitment_system.get(ref.ref_id)
        if (
            record is None
            or record.lifecycle is not CommitmentLifecycle.ACTIVE
            or record.record_digest != ref.digest
        ):
            raise _ApprovalHold(R18ApprovalReason.SOURCE_NOT_CURRENT, intent.intent_id)
        at = record.deadline.at
        return None if at is None else at.replace(microsecond=0)

    # --------------------------------------------------------------- review

    def review(self, intent_id: str) -> ReviewResult:
        """§3: the challenge a review shows.  Not an event; writes nothing.

        In order: a written live unconsumed challenge; else the pending
        in-memory challenge while it is live (never replaced); else a new one,
        built after §2 steps 2 and 4-7 minus the source requalification (see
        the module note) and held in the bounded map.

        Coalesced (condition B): reading the ledger holds the journal lock, so
        a caller looping on review must not stall the Runtime.  While a live
        challenge is shown for the Intent it is returned with no read at all;
        otherwise a computed result is reused for ``MIN_REVIEW_INTERVAL``, and
        only one read runs at a time (another caller gets the cached result,
        or HOLD ``REVIEW_BUSY``)."""

        if not self._enabled:
            return ReviewResult(ReviewStatus.HOLD, str(intent_id), R18ApprovalReason.R18_DISABLED)
        if not _valid_id(intent_id):
            return self._review_hold(intent_id, R18ApprovalReason.INTENT_UNKNOWN)
        shown = self._live_shown(intent_id)
        if shown is not None:
            return shown
        cached = self._cached_review(intent_id)
        if cached is not None:
            return cached
        if not self._review_lock.acquire(blocking=False):
            return self._cached_review(intent_id, ignore_age=True) or self._review_hold(
                intent_id, R18ApprovalReason.REVIEW_BUSY
            )
        try:
            # Another caller may have finished while this one waited.
            shown = self._live_shown(intent_id)
            if shown is not None:
                return shown
            cached = self._cached_review(intent_id)
            if cached is not None:
                return cached
            result = self._review_computed(intent_id)
        finally:
            self._review_lock.release()
        with self._lock:
            if result.status in (ReviewStatus.WRITTEN, ReviewStatus.PENDING, ReviewStatus.BUILT):
                self._shown[intent_id] = result
                self._reviewed.pop(intent_id, None)
            else:
                if len(self._reviewed) >= MAX_REVIEW_CACHE:
                    self._reviewed.clear()
                self._reviewed[intent_id] = (self._monotonic(), result)
        return result

    def _review_computed(self, intent_id: str) -> ReviewResult:
        try:
            return self._review(intent_id)
        except (HistoryStoreError, AgentStateLoadError):
            return self._review_hold(intent_id, R18ApprovalReason.HISTORY_UNAVAILABLE)
        except _ApprovalHold as hold:
            return self._review_hold(intent_id, hold.reason)
        except _PolicyHold as hold:
            return self._review_hold(intent_id, _map(hold.reason.name))
        except _Deny as denied:
            return self._review_hold(intent_id, _map(denied.reason.name))

    def _live_shown(self, intent_id: str) -> ReviewResult | None:
        """The challenge shown for this Intent while it is still live (no read)."""

        with self._lock:
            shown = self._shown.get(intent_id)
            if shown is None or shown.challenge is None or shown.intent is None:
                return None
            if not shown.challenge.is_live(_fmt(self._clock()), self._boot_id):
                del self._shown[intent_id]
                return None
            if shown.status is ReviewStatus.BUILT:
                # A repeat of a built challenge is the pending one.
                shown = replace(shown, status=ReviewStatus.PENDING)
                self._shown[intent_id] = shown
            return shown

    def _cached_review(self, intent_id: str, *, ignore_age: bool = False) -> ReviewResult | None:
        with self._lock:
            entry = self._reviewed.get(intent_id)
            if entry is None:
                return None
            at, result = entry
            if ignore_age or self._monotonic() - at < MIN_REVIEW_INTERVAL:
                return result
            del self._reviewed[intent_id]
            return None

    def invalidate(self, intent_id: str) -> None:
        """Forget what a review showed for an Intent (a CLAIM, IP-17c, calls
        this when it consumes the challenge)."""

        with self._lock:
            self._shown.pop(intent_id, None)
            self._reviewed.pop(intent_id, None)
            # The list / status facts are reused for 2 s: a claim ends that.
            self._facts_cache.clear()

    @staticmethod
    def _review_hold(intent_id: str, reason: R18ApprovalReason) -> ReviewResult:
        LOGGER.info("R18 approval review result=hold reason=%s", reason.value)
        return ReviewResult(ReviewStatus.HOLD, str(intent_id), reason)

    def _review(self, intent_id: str) -> ReviewResult:
        if not _valid_id(intent_id):
            raise _ApprovalHold(R18ApprovalReason.INTENT_UNKNOWN)
        if intent_id.startswith(INVERSE_INTENT_ID_PREFIX):
            raise _ApprovalHold(R18ApprovalReason.INVERSE_DEFERRED, intent_id)
        classified = self._policy._classify()
        now = self._clock()
        facts, expectation = self._forward_facts(classified, intent_id, now)
        intent = facts.intent
        self._policy._not_superseded(classified, intent, now)
        self._policy._target(intent)
        if not gate_open(self._gate):
            raise _ApprovalHold(R18ApprovalReason.ADMISSION_GATE_CLOSED, intent_id)
        window_end = intent.approval_window_end_utc
        live = self._live_written(facts, now)
        if live:
            attempt, written = live[0]
            return ReviewResult(
                ReviewStatus.WRITTEN, intent_id, None, written, attempt, window_end, intent
            )
        attempt = len(facts.challenges) + 1
        with self._lock:
            self._prune(now)
            pending = self._pending_challenges.get(intent_id)
            if pending is not None:
                return ReviewResult(
                    ReviewStatus.PENDING, intent_id, None, pending, attempt, window_end, intent
                )
            if len(facts.challenges) >= MAX_CHALLENGES_PER_INTENT:
                raise _ApprovalHold(R18ApprovalReason.ATTEMPTS_EXHAUSTED, intent_id)
            if len(self._pending_challenges) >= MAX_PENDING_CHALLENGES:
                raise _ApprovalHold(R18ApprovalReason.PENDING_FULL, intent_id)
            challenge = self._build(intent, expectation, now, classified)
            self._pending_challenges[intent_id] = challenge
        return ReviewResult(
            ReviewStatus.BUILT, intent_id, None, challenge, attempt, window_end, intent
        )

    def _build(
        self,
        intent: ActionIntentV1,
        expectation: str,
        now: datetime,
        classified: ReaderClassification,
    ) -> ApprovalChallengeV1:
        verdict = _verdict_for(classified, intent.intent_id)
        assert verdict is not None
        issued = now.replace(microsecond=0)
        expires = min(issued + self._ttl, intent.approval_window_end)
        deadline = self._commitment_deadline(intent)
        try:
            return ApprovalChallengeV1.for_forward(
                intent,
                expectation,
                verdict,
                nonce=secrets.token_hex(32),
                issued_utc=_fmt(issued),
                expires_utc=_fmt(expires),
                boot_id=self._boot_id,
                commitment_deadline_utc=None if deadline is None else _fmt(deadline),
            )
        except (TypeError, ValueError):
            raise _ApprovalHold(R18ApprovalReason.DEADLINE_ELAPSED, intent.intent_id) from None

    def _prune(self, now: datetime) -> None:
        """Drop the pending challenges that are dead (the lock is held)."""

        stamp = _fmt(now)
        for key in [
            key
            for key, value in self._pending_challenges.items()
            if not value.is_live(stamp, self._boot_id)
        ]:
            del self._pending_challenges[key]

    def pending_challenge(self, intent_id: str) -> ApprovalChallengeV1 | None:
        """The live pending in-memory challenge of an Intent, if any."""

        with self._lock:
            self._prune(self._clock())
            return self._pending_challenges.get(intent_id)

    # ---------------------------------------------------------- confirm side

    def issue(self, intent_id: str, nonce: str, code: str) -> ApprovalResult:
        """The challenge half of a confirm (the CLAIM is IP-17c).

        * a written, live, unconsumed nonce with the right code -> EXISTING:
          nothing is written, the caller goes straight to the CLAIM;
        * the pending in-memory nonce -> ``R18_CHALLENGE_ISSUE`` (a dead
          earlier challenge is expired first, once), then ISSUED;
        * anything else -> HOLD ``UNKNOWN_CHALLENGE`` (a wrong code is HOLD
          ``CODE_MISMATCH``; neither uses an attempt)."""

        if not self._enabled:
            return self._hold(R18ApprovalReason.R18_DISABLED, None)
        if not (_valid_id(intent_id) and type(nonce) is str and type(code) is str):
            return self._hold(R18ApprovalReason.UNKNOWN_CHALLENGE, None)
        if intent_id.startswith(INVERSE_INTENT_ID_PREFIX):
            return self._hold(R18ApprovalReason.INVERSE_DEFERRED, intent_id)
        # A wrong code never reaches the Runtime: it is compared here, in the
        # caller's thread, against the challenge shown (the pending one, or
        # the written one), with no read and no event (the handler checks it
        # again as defence in depth).
        pending = self.pending_challenge(intent_id)
        with self._lock:
            shown = self._shown.get(intent_id)
        for known in (pending, None if shown is None else shown.challenge):
            if (
                known is not None
                and known.nonce == nonce
                and not hmac.compare_digest(code, known.confirmation_code)
            ):
                return self._hold(R18ApprovalReason.CODE_MISMATCH, intent_id)
        existing = self._existing(intent_id, nonce, code)
        if existing is not None:
            return existing
        if pending is None or pending.nonce != nonce:
            return self._hold(R18ApprovalReason.UNKNOWN_CHALLENGE, intent_id)
        try:
            result = self._issue_event(pending, code)
            if (
                result.status is ApprovalStatus.HOLD
                and result.reason is R18ApprovalReason.PREVIOUS_CHALLENGE_OPEN
                and result.challenge_digest is not None
            ):
                # The earlier challenge is dead: end it first (§3 step 5), once.
                ended = self._expire_event(result.challenge_digest)
                if ended.status not in (ApprovalStatus.EXPIRED, ApprovalStatus.EXPIRE_EXISTING):
                    return ended
                result = self._issue_event(pending, code)
        except Exception:  # a refused band, a stopped Runtime, a timeout
            LOGGER.warning("R18 approval confirm could not run its event")
            return self._hold(R18ApprovalReason.JOURNAL_UNAVAILABLE, intent_id)
        if result.status is ApprovalStatus.HOLD and result.reason is R18ApprovalReason.NONCE_USED:
            # A lost race: another confirm wrote this very challenge first.
            existing = self._existing(intent_id, nonce, code)
            if existing is not None:
                result = existing
        if result.status is ApprovalStatus.HOLD:
            # What a review showed may be stale (drift, supersession): the next
            # review reads the ledger again (still rate limited), and the
            # pending challenge itself is untouched.
            with self._lock:
                self._shown.pop(intent_id, None)
        if result.status is ApprovalStatus.ISSUED and result.challenge is not None:
            with self._lock:
                window_end = None if shown is None else shown.window_end_utc
                self._shown[intent_id] = ReviewResult(
                    ReviewStatus.WRITTEN,
                    intent_id,
                    None,
                    result.challenge,
                    result.attempt,
                    window_end,
                    None if shown is None else shown.intent,
                )
                self._reviewed.pop(intent_id, None)
            self._arm(result.challenge, 0)
        return result

    def confirm(self, intent_id: str, nonce: str, code: str) -> ApprovalResult:
        """A whole operator confirm (§3 / §4 / §6): ``issue`` and then the
        CLAIM.  A nonce that is already written, live and unconsumed goes
        straight to the CLAIM (``issue`` returns EXISTING and writes nothing),
        so a retry after a CLAIM HOLD within the TTL writes no second challenge.
        Anything but ISSUED / EXISTING is returned as it is (a HOLD, nothing
        consumed)."""

        issued = self.issue(intent_id, nonce, code)
        if issued.status not in (ApprovalStatus.ISSUED, ApprovalStatus.EXISTING):
            return issued
        return self.claim(intent_id, nonce, code)

    def claim(self, intent_id: str, nonce: str, code: str) -> ApprovalResult:
        """The CLAIM half of a confirm: ``R18_EXECUTION_CLAIM`` for a nonce
        that is already written.  On a claim, what a review showed is
        forgotten (``invalidate``), the pending challenge is dropped and the
        expiry timer is cancelled."""

        if not self._enabled:
            return self._hold(R18ApprovalReason.R18_DISABLED, None)
        if not (_valid_id(intent_id) and _valid_id(nonce) and type(code) is str):
            return self._hold(R18ApprovalReason.UNKNOWN_CHALLENGE, None)
        if intent_id.startswith(INVERSE_INTENT_ID_PREFIX):
            return self._hold(R18ApprovalReason.INVERSE_DEFERRED, intent_id)
        try:
            future = self._submit_claim(intent_id, nonce, code)
            # §2-2 item 2: attached BEFORE the bounded wait, so a claim that
            # commits after ``result(timeout=60)`` gave up is still fed.
            self._feed_on_done(future)
            result = self._run(future)
        except Exception:  # a refused band, a stopped Runtime, a timeout
            LOGGER.warning("R18 approval claim could not run its event")
            return self._hold(R18ApprovalReason.JOURNAL_UNAVAILABLE, intent_id)
        if result.status is ApprovalStatus.CLAIMED:
            if result.claim is not None:
                self._notify_claimed(result.claim)
            with self._lock:
                self._pending_challenges.pop(intent_id, None)
                timer = (
                    None
                    if result.challenge is None
                    else self._timers.pop(result.challenge.challenge_digest, None)
                )
            if timer is not None:
                timer.cancel()
            self.invalidate(intent_id)
        else:
            # What a review showed may be stale (drift, supersession).
            with self._lock:
                self._shown.pop(intent_id, None)
        return result

    # ------------------------------------------------------- claim feeder

    def _feed_on_done(self, future: Future[AgentEventOutcome[object]]) -> None:
        """Done-callback of the claim future (IP-18a §2-2 item 2).

        It runs where the future is resolved: on the Runtime thread inside
        ``set_result``, or on the API thread when the future is already done.
        So it does no I/O, never blocks and never raises: it unwraps the
        value and calls ``on_claimed``, which is O(1)."""

        if self._on_claimed is None:
            return

        def done(finished: Future[AgentEventOutcome[object]]) -> None:
            try:
                if finished.cancelled():
                    return
                outcome = finished.result(timeout=0)
                value = outcome.value
                if (
                    type(value) is ApprovalResult
                    and value.status is ApprovalStatus.CLAIMED
                    and value.claim is not None
                ):
                    self._notify_claimed(value.claim)
            except Exception:  # a callback never escapes
                LOGGER.warning("R18 claim done-callback failed")

        future.add_done_callback(done)

    def _notify_claimed(self, claim: ExecutionClaimV1) -> None:
        callback = self._on_claimed
        if callback is None:
            return
        try:
            callback(claim)
        except Exception:
            LOGGER.warning("R18 on_claimed hook failed")

    # ------------------------------------------------------- facts (no event)

    def status(self, intent_id: str) -> FactsResult:
        """§4 ``status``: one Intent's ledger facts.  Reads only; no event."""

        if not self._enabled:
            return FactsResult(R18ApprovalReason.R18_DISABLED)
        if not _valid_id(intent_id) or intent_id.startswith(INVERSE_INTENT_ID_PREFIX):
            return FactsResult(R18ApprovalReason.INTENT_UNKNOWN)
        return self._facts_read(f"status:{intent_id}", lambda c, n: self._one_fact(c, intent_id, n))

    def open_approvals(self) -> FactsResult:
        """§4 ``list``: the open Intents (an ALLOW, ledger INTENT_ISSUED).
        Reads only; no event."""

        if not self._enabled:
            return FactsResult(R18ApprovalReason.R18_DISABLED)
        return self._facts_read("list", self._all_facts)

    def _facts_read(
        self,
        key: str,
        compute: Callable[[ReaderClassification, datetime], FactsResult],
    ) -> FactsResult:
        """Coalesced like a review (condition B): one ledger read at a time and
        a result reused for ``MIN_REVIEW_INTERVAL``."""

        def cached(ignore_age: bool) -> FactsResult | None:
            with self._lock:
                entry = self._facts_cache.get(key)
                if entry is None:
                    return None
                at, value = entry
                if ignore_age or self._monotonic() - at < MIN_REVIEW_INTERVAL:
                    return value
                del self._facts_cache[key]
                return None

        fresh = cached(False)
        if fresh is not None:
            return fresh
        if not self._facts_lock.acquire(blocking=False):
            return cached(True) or FactsResult(R18ApprovalReason.REVIEW_BUSY)
        try:
            try:
                classified = self._policy._classify()
                result = compute(classified, self._clock())
            except (HistoryStoreError, AgentStateLoadError):
                result = FactsResult(R18ApprovalReason.HISTORY_UNAVAILABLE)
            except _PolicyHold as hold:
                result = FactsResult(_map(hold.reason.name))
            except _ApprovalHold as hold:
                result = FactsResult(hold.reason)
        finally:
            self._facts_lock.release()
        with self._lock:
            if len(self._facts_cache) >= MAX_REVIEW_CACHE:
                self._facts_cache.clear()
            self._facts_cache[key] = (self._monotonic(), result)
        return result

    def _fact_of(
        self, classified: ReaderClassification, intent: ActionIntentV1, now: datetime
    ) -> ApprovalFacts:
        facts = self._facts(classified, intent.intent_id)
        ledger = derive_ledger(
            (RecordAt(item.record, item.processing_sequence) for item in classified.records),
            intent.obligation_group_key,
            now,
        )
        live = self._live_written(facts, now)
        challenge = live[0][1] if live else None
        ref = intent.artifact.source(SourceRole.COMMITMENT)
        try:
            deadline = self._commitment_deadline(intent)
        except _ApprovalHold:
            deadline = None
        return ApprovalFacts(
            intent.intent_id,
            ledger.state.value,
            intent.decision_id,
            intent.approval_window_end_utc,
            len(facts.challenges),
            challenge is not None,
            None if challenge is None else challenge.expires_utc,
            ref.ref_id,
            None if deadline is None else _fmt(deadline),
            intent.target_instance_id,
            intent.forward_key,
        )

    def _one_fact(
        self, classified: ReaderClassification, intent_id: str, now: datetime
    ) -> FactsResult:
        intent = self._policy._target_intent(classified, intent_id)
        return FactsResult(items=(self._fact_of(classified, intent, now),))

    def _all_facts(self, classified: ReaderClassification, now: datetime) -> FactsResult:
        items: list[ApprovalFacts] = []
        ordered = sorted(
            classified.records, key=lambda item: (item.processing_sequence, item.transaction_id)
        )
        for adopted in ordered:
            intent = adopted.record
            if type(intent) is not ActionIntentV1 or intent.incarnation_id != classified.incarnation:
                continue
            verdict = _verdict_for(classified, intent.intent_id)
            if verdict is None or verdict.verdict is not PolicyVerdict.ALLOW:
                continue
            fact = self._fact_of(classified, intent, now)
            if fact.ledger_state == LedgerState.INTENT_ISSUED.value:
                items.append(fact)
        return FactsResult(items=tuple(items[-MAX_LISTED_APPROVALS:]))

    def _existing(self, intent_id: str, nonce: str, code: str) -> ApprovalResult | None:
        """A written challenge with this nonce: EXISTING when live, unconsumed
        and the code is right; a HOLD when not; None when none is written."""

        try:
            classified = self._policy._classify()
            now = self._clock()
            facts = self._facts(classified, intent_id)
        except (HistoryStoreError, AgentStateLoadError):
            return self._hold(R18ApprovalReason.HISTORY_UNAVAILABLE, intent_id)
        except _PolicyHold as hold:
            return self._hold(_map(hold.reason.name), intent_id)
        written = [c for c in facts.challenges if c.nonce == nonce]
        if not written:
            return None
        challenge = written[0]
        if (
            challenge.challenge_digest in facts.expired
            or challenge.challenge_digest in facts.consumed
            or not challenge.is_live(_fmt(now), self._boot_id)
        ):
            return self._hold(R18ApprovalReason.CHALLENGE_NOT_LIVE, intent_id)
        if not hmac.compare_digest(code, challenge.confirmation_code):
            return self._hold(R18ApprovalReason.CODE_MISMATCH, intent_id)
        return ApprovalResult(
            ApprovalStatus.EXISTING,
            intent_id,
            None,
            facts.challenges.index(challenge) + 1,
            challenge,
        )

    def _issue_event(self, challenge: ApprovalChallengeV1, code: str) -> ApprovalResult:
        return self._run(self._submit_issue(challenge, code))

    def _expire_event(self, digest: str) -> ApprovalResult:
        return self._run(self._submit_expire(digest))

    @staticmethod
    def _run(future: Future[AgentEventOutcome[object]]) -> ApprovalResult:
        value = future.result(timeout=60).value
        assert type(value) is ApprovalResult
        return value

    # ------------------------------------------------------- R18_CHALLENGE_ISSUE

    def _issue_checked(self, event: AgentEvent, target: _Target) -> object:
        challenge = target.challenge
        assert challenge is not None and target.code is not None
        # 1.
        if not self._enabled:
            raise _ApprovalHold(R18ApprovalReason.R18_DISABLED)
        intent_id = challenge.intent_id
        if challenge.direction is not Direction.FORWARD or intent_id.startswith(
            INVERSE_INTENT_ID_PREFIX
        ):
            raise _ApprovalHold(R18ApprovalReason.INVERSE_DEFERRED, intent_id)
        initial = self._preview._state_snapshot()
        classified = self._policy._classify()
        now = self._clock()
        # §2 steps 2-3 for an approval.
        facts, expectation = self._forward_facts(classified, intent_id, now)
        intent = facts.intent
        if not self._chain_matches(challenge, intent, expectation):
            raise _ApprovalHold(R18ApprovalReason.CHALLENGE_STALE, intent_id)
        # The challenge is the pending in-memory one: never one a caller made.
        with self._lock:
            known = self._pending_challenges.get(intent_id)
        if known != challenge:
            raise _ApprovalHold(R18ApprovalReason.UNKNOWN_CHALLENGE, intent_id)
        # 3. the window and the boot / TTL.
        if not challenge.is_live(_fmt(now), self._boot_id):
            raise _ApprovalHold(R18ApprovalReason.CHALLENGE_NOT_LIVE, intent_id)
        if (
            _parse(challenge.expires_utc) > intent.approval_window_end
            or _parse(challenge.expires_utc) - _parse(challenge.issued_utc) > self._ttl
        ):
            raise _ApprovalHold(R18ApprovalReason.CHALLENGE_STALE, intent_id)
        # §2 steps 4-7.
        decision = self._policy._not_superseded(classified, intent, now)
        requalified = self._policy._sources(event, decision, intent)
        self._policy._target(intent)
        self._check_deadline(intent, requalified, challenge)
        if not gate_open(self._gate):
            raise _ApprovalHold(R18ApprovalReason.ADMISSION_GATE_CLOSED, intent_id)
        # 4. the typed code.
        if not hmac.compare_digest(target.code, challenge.confirmation_code):
            raise _ApprovalHold(R18ApprovalReason.CODE_MISMATCH, intent_id)
        # 5. an earlier written, unended challenge.
        for earlier in facts.challenges:
            if earlier.nonce == challenge.nonce or earlier.challenge_digest == (
                challenge.challenge_digest
            ):
                raise _ApprovalHold(R18ApprovalReason.NONCE_USED, intent_id)
        stamp = _fmt(now)
        for earlier in facts.challenges:
            if earlier.challenge_digest in facts.expired:
                continue
            if earlier.challenge_digest in facts.consumed:
                raise _ApprovalHold(R18ApprovalReason.LEDGER_UNSAFE, intent_id)
            if earlier.is_live(stamp, self._boot_id):
                raise _ApprovalHold(R18ApprovalReason.CHALLENGE_ALREADY_OPEN, intent_id)
            raise _ApprovalHold(
                R18ApprovalReason.PREVIOUS_CHALLENGE_OPEN,
                intent_id,
                earlier.challenge_digest,
            )
        # 6. the handler-supplied attempt and every insert condition (§3b).
        attempt = len(facts.challenges) + 1
        if attempt > MAX_CHALLENGES_PER_INTENT or len(facts.expired) != attempt - 1:
            raise _ApprovalHold(R18ApprovalReason.ATTEMPTS_EXHAUSTED, intent_id)
        operation = HistoryRecordOperationV1(
            HistoryOperationKind.R18_CHALLENGE_ISSUE,
            classified.incarnation,
            event.event_id,
            _sequence(event),
            TransactionKind.EVENT_MUTATION.value,
            RecordKind.APPROVAL_CHALLENGE,
            challenge,
            attempt,
        )
        if (
            intent.incarnation_id != classified.incarnation
            or _adopted_intent(classified, intent_id) is not intent
        ):
            raise _ApprovalHold(R18ApprovalReason.HISTORY_CONFLICT, intent_id)
        self._precheck_rows(classified, intent_id, operation)
        return self._prepare(
            event,
            classified,
            initial,
            operation,
            ApprovalResult(ApprovalStatus.ISSUED, intent_id, None, attempt, challenge),
            intent=intent,
            requalification=(decision, requalified),
        )

    @staticmethod
    def _chain_matches(
        challenge: ApprovalChallengeV1, intent: ActionIntentV1, expectation: str
    ) -> bool:
        """A1-4: the challenge binds exactly this Intent, forecast and target."""

        return (
            challenge.direction is Direction.FORWARD
            and challenge.intent_id == intent.intent_id
            and challenge.intent_digest == intent.intent_digest
            and challenge.operation_key == intent.forward_key
            and challenge.payload_digest == intent.payload_digest
            and challenge.target_instance_id == intent.target_instance_id
            and challenge.target_expected_revision == intent.target_expected_revision
            and challenge.expectation_digest == expectation
        )

    # ------------------------------------------------- R18_EXECUTION_CLAIM

    def _claim_checked(self, event: AgentEvent, target: _Target) -> object:
        """§6-1.  Consumes the nonce and writes the claim in ONE event; it
        causes no effect (IP-18 owns the executor, §7)."""

        assert target.nonce is not None and target.code is not None
        assert target.intent_id is not None
        # 1. r18.enabled, then FORWARD only.
        if not self._enabled:
            raise _ApprovalHold(R18ApprovalReason.R18_DISABLED)
        intent_id = target.intent_id
        if intent_id.startswith(INVERSE_INTENT_ID_PREFIX):
            raise _ApprovalHold(R18ApprovalReason.INVERSE_DEFERRED, intent_id)
        initial = self._preview._state_snapshot()
        classified = self._policy._classify()
        now = self._clock()
        # 2. the Intent, an ALLOW, the forecast and an open window ...
        facts, expectation = self._forward_facts(classified, intent_id, now)
        intent = facts.intent
        # ... the challenge (written by this or an earlier confirm) ...
        written = [item for item in facts.challenges if item.nonce == target.nonce]
        if len(written) != 1:
            raise _ApprovalHold(R18ApprovalReason.UNKNOWN_CHALLENGE, intent_id)
        challenge = written[0]
        digest = challenge.challenge_digest
        attempt = facts.challenges.index(challenge) + 1
        if digest in facts.consumed:
            raise _ApprovalHold(R18ApprovalReason.NONCE_USED, intent_id)
        stamp = _fmt(now)
        if digest in facts.expired or not challenge.is_live(stamp, self._boot_id):
            raise _ApprovalHold(R18ApprovalReason.CHALLENGE_NOT_LIVE, intent_id)
        # ... and the typed code.
        if not hmac.compare_digest(target.code, challenge.confirmation_code):
            raise _ApprovalHold(R18ApprovalReason.CODE_MISMATCH, intent_id)
        # 3. the A1-4 chain: Intent / forecast / ALLOW / challenge digests,
        # payload, target.
        if not self._chain_matches(challenge, intent, expectation):
            raise _ApprovalHold(R18ApprovalReason.CHALLENGE_STALE, intent_id)
        if (
            _parse(challenge.expires_utc) > intent.approval_window_end
            or _parse(challenge.expires_utc) - _parse(challenge.issued_utc) > self._ttl
        ):
            raise _ApprovalHold(R18ApprovalReason.CHALLENGE_STALE, intent_id)
        # 4. not superseded (6-2), the Commitment ACTIVE outside the margin and
        # the sources requalified (one code path with the Policy owner), then
        # 5. the ledger is INTENT_ISSUED (inside ``_not_superseded``).
        decision = self._policy._not_superseded(classified, intent, now)
        requalified = self._policy._sources(event, decision, intent)
        # 6. the target re-read; the AdmissionGate.
        self._policy._target(intent)
        self._check_deadline(intent, requalified, challenge)
        if not gate_open(self._gate):
            raise _ApprovalHold(R18ApprovalReason.ADMISSION_GATE_CLOSED, intent_id)
        # 7. the claim.
        try:
            claim = ExecutionClaimV1.claim(
                challenge,
                runtime_instance_id=self._runtime_instance_id,
                boot_id=self._boot_id,
                claim_event_id=event.event_id,
                claim_sequence=_sequence(event),
                claimed_utc=stamp,
                approval_channel=ApprovalChannel.LOCAL_LOOPBACK_TYPED_CODE_V1,
            )
        except (TypeError, ValueError):
            # Not live at the second, or at / after the Commitment deadline.
            raise _ApprovalHold(R18ApprovalReason.DEADLINE_ELAPSED, intent_id) from None
        # 8. ONE record operation: consumption + claim (§3b pre-checks: the
        # challenge is adopted with this nonce, not expired, not consumed; the
        # operation key and direction match; no claim for the key in any
        # incarnation (``claim:`` key); ``challenge-end:`` free).
        operation = HistoryRecordOperationV1(
            HistoryOperationKind.R18_EXECUTION_CLAIM,
            classified.incarnation,
            event.event_id,
            _sequence(event),
            TransactionKind.EVENT_MUTATION.value,
            RecordKind.EXECUTION_CLAIM,
            claim,
        )
        if (
            intent.incarnation_id != classified.incarnation
            or _adopted_intent(classified, intent_id) is not intent
        ):
            raise _ApprovalHold(R18ApprovalReason.HISTORY_CONFLICT, intent_id)
        self._precheck_rows(classified, intent_id, operation)
        return self._prepare(
            event,
            classified,
            initial,
            operation,
            ApprovalResult(
                ApprovalStatus.CLAIMED, intent_id, None, attempt, challenge, claim=claim
            ),
            intent=intent,
            requalification=(decision, requalified),
        )

    def _check_deadline(
        self, intent: ActionIntentV1, requalified: Requalified, challenge: ApprovalChallengeV1
    ) -> None:
        """A1-4: the challenge carries the qualified Commitment's deadline."""

        from suzka.r18.requalify import commitment_deadline

        try:
            at = commitment_deadline(self._preview, requalified.q)
        except RequalifyFailed:
            raise _ApprovalHold(R18ApprovalReason.SOURCE_NOT_CURRENT, intent.intent_id) from None
        expected = None if at is None else _fmt(at.replace(microsecond=0))
        if expected != challenge.commitment_deadline_utc:
            raise _ApprovalHold(R18ApprovalReason.CHALLENGE_STALE, intent.intent_id)

    def _precheck_rows(
        self,
        classified: ReaderClassification,
        intent_id: str,
        operation: HistoryRecordOperationV1,
    ) -> None:
        kinds = frozenset(
            {
                HistoryOperationKind.R18_CHALLENGE_ISSUE,
                HistoryOperationKind.R18_APPROVAL_EXPIRE,
                HistoryOperationKind.R18_EXECUTION_CLAIM,
            }
        )
        if self._in_flight(classified, intent_id, kinds):
            raise _ApprovalHold(R18ApprovalReason.IN_FLIGHT, intent_id)
        # UNIQUE(intent_id, attempt), nonce UNIQUE, challenge-end / claim keys.
        if self._store.live_keys(operation.reserved_keys):
            raise _ApprovalHold(R18ApprovalReason.IN_FLIGHT, intent_id)

    # ------------------------------------------------------ R18_APPROVAL_EXPIRE

    def _expire_checked(self, event: AgentEvent, target: _Target) -> object:
        if not self._enabled:
            raise _ApprovalHold(R18ApprovalReason.R18_DISABLED)
        initial = self._preview._state_snapshot()
        classified = self._policy._classify()
        now = self._clock()
        challenge = self._expire_target(classified, target, now)
        if challenge is None:
            LOGGER.info("R18 approval owner result=none_left")
            return ApprovalResult(ApprovalStatus.NONE_LEFT, None)
        intent_id = challenge.intent_id
        digest = challenge.challenge_digest
        if challenge.direction is not Direction.FORWARD:
            raise _ApprovalHold(R18ApprovalReason.INVERSE_DEFERRED, intent_id, digest)
        facts = self._facts(classified, intent_id)
        if digest in facts.consumed:
            raise _ApprovalHold(R18ApprovalReason.ALREADY_CONSUMED, intent_id, digest)
        for item in facts.expiries:
            if item.challenge_digest == digest:
                return ApprovalResult(
                    ApprovalStatus.EXPIRE_EXISTING,
                    intent_id,
                    None,
                    challenge=challenge,
                    expiration=item,
                    challenge_digest=digest,
                )
        # The reason must be true (§3).
        if challenge.boot_id != self._boot_id:
            reason = ExpirationReason.BOOT_CHANGED
        elif now >= _parse(challenge.expires_utc):
            reason = ExpirationReason.EXPIRED
        else:
            raise _ApprovalHold(R18ApprovalReason.STILL_LIVE, intent_id, digest)
        expiration = ApprovalExpirationV1.for_challenge(challenge, reason)
        operation = HistoryRecordOperationV1(
            HistoryOperationKind.R18_APPROVAL_EXPIRE,
            classified.incarnation,
            event.event_id,
            _sequence(event),
            TransactionKind.EVENT_MUTATION.value,
            RecordKind.APPROVAL_EXPIRATION,
            expiration,
        )
        self._precheck_rows(classified, intent_id, operation)
        return self._prepare(
            event,
            classified,
            initial,
            operation,
            ApprovalResult(
                ApprovalStatus.EXPIRED,
                intent_id,
                None,
                challenge=challenge,
                expiration=expiration,
                challenge_digest=digest,
            ),
        )

    def _expire_target(
        self, classified: ReaderClassification, target: _Target, now: datetime
    ) -> ApprovalChallengeV1 | None:
        ordered = sorted(
            classified.records, key=lambda item: (item.processing_sequence, item.transaction_id)
        )
        challenges = [
            item.record for item in ordered if type(item.record) is ApprovalChallengeV1
        ]
        if not target.scan:
            for challenge in challenges:
                if challenge.challenge_digest == target.challenge_digest:
                    return challenge
            raise _ApprovalHold(R18ApprovalReason.CHALLENGE_UNKNOWN)
        start = 0
        if target.after is not None:
            digests = [item.challenge_digest for item in challenges]
            if target.after not in digests:
                raise _ApprovalHold(R18ApprovalReason.CHALLENGE_UNKNOWN)
            start = digests.index(target.after) + 1
        ended = {
            item.record.challenge_digest
            for item in ordered
            if type(item.record) is ApprovalExpirationV1
        } | {
            item.record.challenge_digest
            for item in ordered
            if type(item.record) is ExecutionClaimV1
        }
        stamp = _fmt(now)
        for challenge in challenges[start:]:
            if challenge.challenge_digest not in ended and not challenge.is_live(
                stamp, self._boot_id
            ):
                return challenge
        return None

    # --------------------------------------------------------------- PREPARE

    def _prepare(
        self,
        event: AgentEvent,
        classified: ReaderClassification,
        initial: object,
        operation: HistoryRecordOperationV1,
        result: ApprovalResult,
        *,
        intent: ActionIntentV1 | None = None,
        requalification: tuple[object, Requalified] | None = None,
    ) -> object:
        intent_id = result.intent_id
        try:
            self._store.check_headroom(len(operation.canonical_bytes()))
        except HistoryStoreError:
            raise _ApprovalHold(R18ApprovalReason.HISTORY_CAPACITY, intent_id) from None
        if requalification is not None:
            # The reads above can take seconds: sources are checked again.
            from suzka.r18.requalify import RequalifyFailed, requalify

            decision, first = requalification
            try:
                again = requalify(self._preview, decision, event, self._clock)  # type: ignore[arg-type]
            except RequalifyFailed:
                raise _ApprovalHold(R18ApprovalReason.SOURCE_UNAVAILABLE, intent_id) from None
            if again.q != first.q or again.initial != first.initial:
                raise _ApprovalHold(R18ApprovalReason.SOURCE_UNAVAILABLE, intent_id)
        if self._preview._state_snapshot() != initial:
            raise _ApprovalHold(R18ApprovalReason.SOURCE_UNAVAILABLE, intent_id)
        if intent is not None and not intent.window_open(self._clock()):
            raise _ApprovalHold(R18ApprovalReason.WINDOW_CLOSED, intent_id)
        snapshot_hash, _as_of = self._decision._snapshot()
        transaction_id = TransactionCoordinator.derive_transaction_id(
            event, TransactionKind.EVENT_MUTATION
        )
        self._pending = _PendingOperation(
            event,
            classified.incarnation,
            snapshot_hash,
            operations_digest(self._store.operations(classified.incarnation)),
            transaction_id,
            operation,
            initial,
            operation.operation_kind,
        )
        LOGGER.info("R18 approval owner result=%s", result.status.value)
        participant = SubjectHistoryRecordParticipant(
            self._store,
            classified.incarnation,
            operation,
            JournalCommitProofSource(self._journal, self._recovery),
        )
        return CoordinatedResult(result, (participant,))

    # ---------------------------------------------------------------- commit

    def _validate_commit(self, event: AgentEvent) -> None:
        """After this event's PREPARE; any failure is a runtime fail-stop."""

        from suzka.r16.history_adoption import current_incarnation

        pending = self._pending
        if event.event_type not in _EVENT_TYPES:
            if pending is not None:
                raise _ApprovalCommitRefused("stale approval operation pending")
            return
        if pending is None:
            return
        if pending.event != event:
            raise _ApprovalCommitRefused("unbound approval operation pending")
        if self._preview._state_snapshot() != pending.initial_state:
            raise _ApprovalCommitRefused("approval state changed before WAL")
        if current_incarnation(self._journal, self._store) != pending.incarnation:
            raise _ApprovalCommitRefused("approval incarnation changed before WAL")
        if self._decision._snapshot()[0] != pending.snapshot_hash:
            raise _ApprovalCommitRefused("approval snapshot changed before WAL")
        rows = self._store.operations(pending.incarnation)
        own = [row for row in rows if row.transaction_id == pending.transaction_id]
        others = tuple(row for row in rows if row.transaction_id != pending.transaction_id)
        if (
            len(own) != 1
            or own[0].phase is not HistoryPhase.PREPARED
            or own[0].operation_kind is not pending.kind
            or own[0].pending_bytes != pending.operation.canonical_bytes()
            or own[0].operation_digest != pending.operation.operation_digest
            or operations_digest(others) != pending.rows_digest
        ):
            raise _ApprovalCommitRefused("SubjectHistory changed before WAL")

    def _clear(self) -> None:
        self._pending = None

    # ---------------------------------------------- timer and startup scan

    def _arm(
        self, challenge: ApprovalChallengeV1, retries: int, delay: float | None = None
    ) -> None:
        """A timer at ``expires + 1 s`` submits the expiry (§3); a transient
        HOLD is retried a bounded number of times."""

        if not self._enabled or self._stopped:
            return
        if delay is None:
            delay = (
                _parse(challenge.expires_utc) + EXPIRE_DELAY - self._clock()
            ).total_seconds()
        digest = challenge.challenge_digest

        def fire() -> None:
            with self._lock:
                self._timers.pop(digest, None)
            if self._stopped:
                return
            try:
                result = self._expire_event(digest)
            except Exception:  # the lazy path / window / restart scan cover it
                LOGGER.warning("R18 approval expiry timer could not submit")
                return
            if (
                result.status is ApprovalStatus.HOLD
                and result.reason in TRANSIENT_APPROVAL_REASONS
                and retries < MAX_TIMER_RETRIES
            ):
                self._arm(challenge, retries + 1, 1.0 + retries)

        timer = self._timer_factory(delay, fire)
        with self._lock:
            old = self._timers.pop(digest, None)
            self._timers[digest] = timer
        if old is not None:
            old.cancel()
        timer.start()

    def start(self) -> None:
        """§8 startup (r18.enabled only): the BOOT_CHANGED scan, once.  A store
        with no R17 / R18 operation has no challenge, so no event is posted."""

        if not self._enabled or self._scan_thread is not None:
            return
        self._stopped = False
        self._stop_event.clear()
        try:
            has_records = self._store.has_record_operations()
        except Exception:  # closed code only; the lazy path / window cover it
            LOGGER.warning("R18 approval startup scan skipped: history_unavailable")
            return
        if not has_records:
            return
        thread = threading.Thread(
            target=self._startup_scan, name="suzka-r18-approval-scan", daemon=True
        )
        self._scan_thread = thread
        thread.start()

    def _startup_scan(self) -> None:
        """Chain the scan over every dead challenge.  A transient HOLD that
        names no challenge (the journal or store was briefly unavailable) is
        retried ONCE after ``STARTUP_SCAN_RETRY_DELAY``; a second one ends the
        scan, and whatever it left unended is ended by the lazy path (the next
        confirm) or is closed by the window end (§8a)."""

        after: str | None = None
        retried = False
        for _ in range(MAX_EXPIRE_SCANS_PER_STARTUP):
            if self._stopped:
                return
            try:
                result = self._run(self._submit_expire_scan(after))
            except Exception:
                LOGGER.warning("R18 approval startup scan could not run")
                return
            if result.status is ApprovalStatus.NONE_LEFT:
                return
            if result.status is ApprovalStatus.HOLD and result.challenge_digest is None:
                if (
                    retried
                    or result.reason not in TRANSIENT_APPROVAL_REASONS
                    or self._stop_event.wait(STARTUP_SCAN_RETRY_DELAY)
                ):
                    return
                retried = True
                continue
            after = result.challenge_digest

    def stop(self) -> None:
        """Cancel the timers and join the startup scan (before the runtime stops)."""

        self._stopped = True
        self._stop_event.set()
        with self._lock:
            timers = list(self._timers.values())
            self._timers.clear()
        for timer in timers:
            timer.cancel()
        thread = self._scan_thread
        if thread is not None:
            thread.join(timeout=30)
            self._scan_thread = None


def _map(name: str) -> R18ApprovalReason:
    try:
        return R18ApprovalReason[name]
    except KeyError:
        return R18ApprovalReason.HISTORY_CONFLICT


__all__ = [
    "MAX_EXPIRE_SCANS_PER_STARTUP",
    "MAX_LISTED_APPROVALS",
    "MAX_PENDING_CHALLENGES",
    "TRANSIENT_APPROVAL_REASONS",
    "ApprovalChallengeOwnerV1",
    "ApprovalFacts",
    "ApprovalResult",
    "ApprovalStatus",
    "FactsResult",
    "R18ApprovalReason",
    "ReviewResult",
    "ReviewStatus",
]
