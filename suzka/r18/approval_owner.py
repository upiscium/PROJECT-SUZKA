"""R18 IP-17b: the Approval owner (#411 D1 r7 §1, §3, §3b, §8a, §12-1).

``ApprovalChallengeOwnerV1`` owns two events and is the only writer of
``approval_challenges`` and ``approval_expirations`` (through the
subject.history record participant):

* ``R18_CHALLENGE_ISSUE(prebuilt challenge, attempt, code)`` (source
  ``api.r18.approval``): written only at an operator confirm that carries the
  matching typed code.  Refused in the soft band (HP-8).
* ``R18_APPROVAL_EXPIRE(challenge | scan(after))`` (source
  ``runtime.r18.approval_expire``): written by a timer at ``expires + 1 s``,
  by the startup scan (BOOT_CHANGED) and by the lazy path.  ADMITTED in the
  soft band (HD-6 (a)): it only records that a challenge ended.

Both are gated by ``r18.enabled`` (HOLD ``R18_DISABLED`` before any read) and
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
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import hmac
import logging
import secrets
import threading
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
    Direction,
    ExecutionClaimV1,
    PolicyVerdict,
    SourceRole,
)
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
_MAX_ID: Final = 256
_EVENT_TYPES: Final = frozenset(
    {AgentEventType.R18_CHALLENGE_ISSUE, AgentEventType.R18_APPROVAL_EXPIRE}
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
        )


@dataclass(frozen=True, slots=True)
class ReviewResult:
    """What a review shows: a challenge (never written by the review) or a HOLD."""

    status: ReviewStatus
    intent_id: str
    reason: R18ApprovalReason | None = None
    challenge: ApprovalChallengeV1 | None = None
    attempt: int | None = None
    window_end_utc: str | None = None

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
        timer_factory: TimerFactory = _thread_timer,
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
        self._timer_factory = timer_factory
        self._inflight: tuple[object, _Target] | None = None
        self._pending: _PendingOperation | None = None
        self._lock = threading.Lock()
        # intent_id -> the one pending in-memory challenge (never written).
        self._pending_challenges: dict[str, ApprovalChallengeV1] = {}
        self._timers: dict[str, _Timer] = {}
        self._stopped = False
        self._scan_thread: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def boot_id(self) -> str:
        return self._boot_id

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
        intent_id = None if target.challenge is None else target.challenge.intent_id
        try:
            if target.kind is AgentEventType.R18_CHALLENGE_ISSUE:
                return self._issue_checked(event, target)
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
        the module note) and held in the bounded map."""

        if not self._enabled:
            return ReviewResult(ReviewStatus.HOLD, str(intent_id), R18ApprovalReason.R18_DISABLED)
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
            return ReviewResult(ReviewStatus.WRITTEN, intent_id, None, written, attempt, window_end)
        attempt = len(facts.challenges) + 1
        with self._lock:
            self._prune(now)
            pending = self._pending_challenges.get(intent_id)
            if pending is not None:
                return ReviewResult(
                    ReviewStatus.PENDING, intent_id, None, pending, attempt, window_end
                )
            if len(facts.challenges) >= MAX_CHALLENGES_PER_INTENT:
                raise _ApprovalHold(R18ApprovalReason.ATTEMPTS_EXHAUSTED, intent_id)
            if len(self._pending_challenges) >= MAX_PENDING_CHALLENGES:
                raise _ApprovalHold(R18ApprovalReason.PENDING_FULL, intent_id)
            challenge = self._build(intent, expectation, now, classified)
            self._pending_challenges[intent_id] = challenge
        return ReviewResult(ReviewStatus.BUILT, intent_id, None, challenge, attempt, window_end)

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
        try:
            classified = self._policy._classify()
            now = self._clock()
            facts = self._facts(classified, intent_id)
        except (HistoryStoreError, AgentStateLoadError):
            return self._hold(R18ApprovalReason.HISTORY_UNAVAILABLE, intent_id)
        except _PolicyHold as hold:
            return self._hold(_map(hold.reason.name), intent_id)
        written = [c for c in facts.challenges if c.nonce == nonce]
        if written:
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
        pending = self.pending_challenge(intent_id)
        if pending is None or pending.nonce != nonce:
            return self._hold(R18ApprovalReason.UNKNOWN_CHALLENGE, intent_id)
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
        if result.status is ApprovalStatus.ISSUED and result.challenge is not None:
            self._arm(result.challenge, 0)
        return result

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
        if (
            challenge.intent_digest != intent.intent_digest
            or challenge.operation_key != intent.forward_key
            or challenge.payload_digest != intent.payload_digest
            or challenge.target_instance_id != intent.target_instance_id
            or challenge.target_expected_revision != intent.target_expected_revision
            or challenge.expectation_digest != expectation
        ):
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

    def _check_deadline(
        self, intent: ActionIntentV1, requalified: Requalified, challenge: ApprovalChallengeV1
    ) -> None:
        """A1-4: the challenge carries the qualified Commitment's deadline."""

        from suzka.r18.requalify import commitment_deadline

        at = commitment_deadline(self._preview, requalified.q)
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
        if not self._store.has_record_operations():
            return
        thread = threading.Thread(
            target=self._startup_scan, name="suzka-r18-approval-scan", daemon=True
        )
        self._scan_thread = thread
        thread.start()

    def _startup_scan(self) -> None:
        after: str | None = None
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
                return
            after = result.challenge_digest

    def stop(self) -> None:
        """Cancel the timers and join the startup scan (before the runtime stops)."""

        self._stopped = True
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
    "MAX_PENDING_CHALLENGES",
    "TRANSIENT_APPROVAL_REASONS",
    "ApprovalChallengeOwnerV1",
    "ApprovalResult",
    "ApprovalStatus",
    "R18ApprovalReason",
    "ReviewResult",
    "ReviewStatus",
]
