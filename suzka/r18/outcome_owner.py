"""R18 IP-18a: the Outcome owner (#411 IP-18 D1 r4 §5).

``OutcomeOwnerV1`` owns ``R18_OUTCOME_COMMIT(claim_id)`` (source
``runtime.r18.outcome_commit``) and is the only writer of
``verified_outcomes``.  The executor and the reconciler supply claim ids;
there is no scan mode.  The event is ADMITTED in the soft band (HD-18-1): it
records a terminal fact and creates no obligation.

The handler decides from the target itself: it opens a ``TargetReader``
(``mode=ro``) and runs ``verify_forward``; the executor's receipt is never an
input (F08).  A HOLD writes nothing.  ``None`` from the verifier is never an
Outcome: no forward row is ``NOT_SETTLED`` (transient), anything else
``UNDECIDABLE`` (a key-level BLOCKED that stays CLAIMED, row 2).  IP-18a
writes revision 1 only.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging
from pathlib import Path

from suzka.r16.decision_issuer import _sequence
from suzka.r16.history_attestation_digest import operations_digest
from suzka.r16.history_participant import (
    JournalCommitProofSource,
    SubjectHistoryRecordParticipant,
)
from suzka.r16.history_reader import ReaderClassification
from suzka.r16.history_records import (
    HistoryOperationKind,
    HistoryRecordOperationV1,
    RecordKind,
)
from suzka.r16.history_store import HistoryPhase, HistoryStoreError
from suzka.r18.contracts import (
    ActionIntentV1,
    ApprovalChallengeV1,
    Direction,
    ExecutionClaimV1,
    VerifiedOutcomeV1,
)
from suzka.r18.policy_owner import ReviewArtifactPolicyV1, _adopted_intent, _PolicyHold
from suzka.r18.target import TargetCorrupt, TargetError, TargetReader, TargetUnavailable
from suzka.r18.verifier import verify_forward
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

LOGGER = logging.getLogger(__name__)

_MAX_ID = 256
_OUTCOME_REVISION = 1


class OutcomeStatus(str, Enum):
    COMMITTED = "outcome_committed"
    EXISTING = "outcome_existing"
    HOLD = "hold"


class R18OutcomeReason(str, Enum):
    """Why the Outcome owner held; a closed, source-free vocabulary."""

    R18_DISABLED = "r18_disabled"
    HISTORY_UNAVAILABLE = "history_unavailable"
    HISTORY_CONFLICT = "history_conflict"
    CLAIM_UNKNOWN = "claim_unknown"
    # No forward row on the target yet: transient (the settle comes first).
    NOT_SETTLED = "not_settled"
    # A mixed / inconsistent group: key-level BLOCKED, stays CLAIMED.
    UNDECIDABLE = "undecidable"
    TARGET_UNAVAILABLE = "target_unavailable"
    TARGET_CORRUPT = "target_corrupt"
    IN_FLIGHT = "in_flight"
    HISTORY_CAPACITY = "history_capacity"
    SOURCE_UNAVAILABLE = "source_unavailable"
    # The event could not be admitted or finish: nothing was written.
    JOURNAL_UNAVAILABLE = "journal_unavailable"


TRANSIENT_OUTCOME_REASONS: frozenset[R18OutcomeReason] = frozenset(
    {
        R18OutcomeReason.HISTORY_UNAVAILABLE,
        R18OutcomeReason.TARGET_UNAVAILABLE,
        R18OutcomeReason.NOT_SETTLED,
        R18OutcomeReason.IN_FLIGHT,
        R18OutcomeReason.JOURNAL_UNAVAILABLE,
    }
)


@dataclass(frozen=True, slots=True)
class _OutcomeResult:
    status: OutcomeStatus
    claim_id: str | None
    reason: R18OutcomeReason | None = None
    outcome: VerifiedOutcomeV1 | None = None

    def __post_init__(self) -> None:
        if type(self.status) is not OutcomeStatus:
            raise TypeError("outcome status must be exact")
        if (self.status is OutcomeStatus.HOLD) != (self.reason is not None):
            raise ValueError("exactly a HOLD carries a reason")
        if (self.status is not OutcomeStatus.HOLD) != (self.outcome is not None):
            raise ValueError("exactly a non-HOLD result carries an Outcome")


class _OutcomeHold(Exception):
    def __init__(self, reason: R18OutcomeReason, claim_id: str | None = None) -> None:
        super().__init__(reason.value)
        self.reason = reason
        self.claim_id = claim_id


class _OutcomeCommitRefused(RuntimeError):
    """The pre-WAL check failed: a runtime fail-stop (no source text)."""


@dataclass(frozen=True, slots=True)
class _PendingOutcome:
    event: AgentEvent
    incarnation: str
    snapshot_hash: str
    rows_digest: tuple[int, str]
    transaction_id: str
    operation: HistoryRecordOperationV1
    initial_state: object


def _valid_id(value: object) -> bool:
    return type(value) is str and 0 < len(value) <= _MAX_ID


class OutcomeOwnerV1:
    """The only writer of ``verified_outcomes`` (forward, revision 1)."""

    def __init__(
        self,
        policy_owner: ReviewArtifactPolicyV1,
        runtime: AgentRuntime,
        *,
        enabled: bool,
        target_path: Path,
    ) -> None:
        if (
            type(policy_owner) is not ReviewArtifactPolicyV1
            or type(runtime) is not AgentRuntime
            or policy_owner._runtime is not runtime
            or type(enabled) is not bool
            or not isinstance(target_path, Path)
        ):
            raise TypeError("unqualified Outcome owner binding")
        self._policy = policy_owner
        self._decision = policy_owner._decision
        self._preview = policy_owner._preview
        self._runtime = runtime
        self._store = policy_owner._store
        self._journal = policy_owner._journal
        self._recovery = policy_owner._recovery
        self._enabled = enabled
        self._target_path = target_path
        self._clock: Callable[[], datetime] = policy_owner._clock
        self._inflight: tuple[object, str] | None = None
        self._pending: _PendingOutcome | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------ submission

    def _submit_outcome(self, claim_id: str) -> Future[AgentEventOutcome[object]]:
        """``R18_OUTCOME_COMMIT(claim_id)``."""

        if not _valid_id(claim_id):
            raise ValueError("an Outcome names one claim id")
        nonce = object()

        def handle() -> object:
            if self._inflight is not None or self._pending is not None:
                raise _OutcomeCommitRefused("Outcome commit already pending")
            self._inflight = (nonce, claim_id)
            try:
                return self._commit(nonce)
            except BaseException:
                self._pending = None
                raise
            finally:
                self._inflight = None

        return self._runtime.submit(
            AgentEventType.R18_OUTCOME_COMMIT,
            AgentEventSource.RUNTIME_R18_OUTCOME_COMMIT,
            handle,
        )

    def _current_event(self, nonce: object) -> AgentEvent:
        event = self._runtime.current_event()
        if (
            self._inflight is None
            or self._inflight[0] is not nonce
            or event is None
            or event.event_type is not AgentEventType.R18_OUTCOME_COMMIT
            or event.source is not AgentEventSource.RUNTIME_R18_OUTCOME_COMMIT
            or type(event.processing_sequence) is not int
            or event.processing_sequence < 1
        ):
            raise _OutcomeCommitRefused("unbound Outcome commit event")
        return event

    # --------------------------------------------------------------- handler

    def _commit(self, nonce: object) -> object:
        self._pending = None
        event = self._current_event(nonce)
        assert self._inflight is not None
        claim_id = self._inflight[1]
        try:
            return self._commit_checked(event, claim_id)
        except (HistoryStoreError, AgentStateLoadError):
            return self._hold(R18OutcomeReason.HISTORY_UNAVAILABLE, claim_id)
        except _OutcomeHold as hold:
            return self._hold(hold.reason, hold.claim_id or claim_id)

    @staticmethod
    def _hold(reason: R18OutcomeReason, claim_id: str | None) -> _OutcomeResult:
        LOGGER.info("R18 outcome owner result=hold reason=%s", reason.value)
        return _OutcomeResult(OutcomeStatus.HOLD, claim_id, reason)

    def _commit_checked(self, event: AgentEvent, claim_id: str) -> object:
        # 1.
        if not self._enabled:
            raise _OutcomeHold(R18OutcomeReason.R18_DISABLED)
        initial = self._preview._state_snapshot()
        classified = self._classify()
        # 2. the adopted forward claim of this incarnation.
        claim = _adopted_claim(classified, claim_id)
        if claim is None:
            raise _OutcomeHold(R18OutcomeReason.CLAIM_UNKNOWN, claim_id)
        # 3. an Outcome revision 1 already exists: nothing is written.
        existing = _outcome_for(classified, claim)
        if existing is not None:
            LOGGER.info("R18 outcome owner result=existing")
            return _OutcomeResult(OutcomeStatus.EXISTING, claim_id, None, existing)
        # 4. the adopted chain claim -> Intent -> challenge is consistent.
        intent, challenge = _chain(classified, claim)
        # 5. verify, independently: the receipt is not an input (F08).
        outcome = self._verify(event, claim, intent, challenge)
        # 6. §3b: every insert condition and UNIQUE, from the same read.
        operation = HistoryRecordOperationV1(
            HistoryOperationKind.R18_OUTCOME_COMMIT,
            classified.incarnation,
            event.event_id,
            _sequence(event),
            TransactionKind.EVENT_MUTATION.value,
            RecordKind.VERIFIED_OUTCOME,
            outcome,
        )
        self._precheck(classified, claim, operation)
        return self._prepare(event, classified, initial, operation, outcome, claim_id)

    def _classify(self) -> ReaderClassification:
        try:
            return self._policy._classify()
        except _PolicyHold as hold:
            if hold.reason.name == "HISTORY_CONFLICT":
                raise _OutcomeHold(R18OutcomeReason.HISTORY_CONFLICT) from None
            raise _OutcomeHold(R18OutcomeReason.HISTORY_UNAVAILABLE) from None

    def _verify(
        self,
        event: AgentEvent,
        claim: ExecutionClaimV1,
        intent: ActionIntentV1,
        challenge: ApprovalChallengeV1,
    ) -> VerifiedOutcomeV1:
        try:
            reader = TargetReader.open(self._target_path)
        except TargetCorrupt:
            raise _OutcomeHold(R18OutcomeReason.TARGET_CORRUPT, claim.claim_id) from None
        except (TargetError, OSError, ValueError):
            raise _OutcomeHold(R18OutcomeReason.TARGET_UNAVAILABLE, claim.claim_id) from None
        try:
            outcome = verify_forward(
                reader,
                intent,
                challenge,
                consumed_nonce=claim.nonce,
                outcome_revision=_OUTCOME_REVISION,
                verifier_event_id=event.event_id,
                verifier_sequence=_sequence(event),
            )
            if outcome is None:
                status = reader.status(intent.obligation_group_key)
                if status.forward is None:
                    raise _OutcomeHold(R18OutcomeReason.NOT_SETTLED, claim.claim_id)
                raise _OutcomeHold(R18OutcomeReason.UNDECIDABLE, claim.claim_id)
            return outcome
        except _OutcomeHold:
            raise
        except TargetCorrupt:
            raise _OutcomeHold(R18OutcomeReason.TARGET_CORRUPT, claim.claim_id) from None
        except (TargetUnavailable, TargetError, OSError, ValueError):
            raise _OutcomeHold(R18OutcomeReason.TARGET_UNAVAILABLE, claim.claim_id) from None
        finally:
            reader.close()

    def _precheck(
        self,
        classified: ReaderClassification,
        claim: ExecutionClaimV1,
        operation: HistoryRecordOperationV1,
    ) -> None:
        """An unadopted Outcome operation of this claim holds its key."""

        adopted = [item.record for item in classified.records]
        for row in classified.rows:
            if (
                row.operation_kind is not HistoryOperationKind.R18_OUTCOME_COMMIT
                or row.phase not in (HistoryPhase.PREPARED, HistoryPhase.FINALIZED)
                or row.pending_bytes is None
            ):
                continue
            try:
                other = HistoryRecordOperationV1.from_bytes(row.pending_bytes)
            except (TypeError, ValueError):
                raise _OutcomeHold(R18OutcomeReason.HISTORY_CONFLICT, claim.claim_id) from None
            record = other.record
            if getattr(record, "operation_key", None) == claim.operation_key and record not in adopted:
                raise _OutcomeHold(R18OutcomeReason.IN_FLIGHT, claim.claim_id)
        if self._store.live_keys(operation.reserved_keys):
            raise _OutcomeHold(R18OutcomeReason.IN_FLIGHT, claim.claim_id)

    def _prepare(
        self,
        event: AgentEvent,
        classified: ReaderClassification,
        initial: object,
        operation: HistoryRecordOperationV1,
        outcome: VerifiedOutcomeV1,
        claim_id: str,
    ) -> object:
        try:
            self._store.check_headroom(len(operation.canonical_bytes()))
        except HistoryStoreError:
            raise _OutcomeHold(R18OutcomeReason.HISTORY_CAPACITY, claim_id) from None
        if self._preview._state_snapshot() != initial:
            raise _OutcomeHold(R18OutcomeReason.SOURCE_UNAVAILABLE, claim_id)
        snapshot_hash, _as_of = self._decision._snapshot()
        transaction_id = TransactionCoordinator.derive_transaction_id(
            event, TransactionKind.EVENT_MUTATION
        )
        self._pending = _PendingOutcome(
            event,
            classified.incarnation,
            snapshot_hash,
            operations_digest(self._store.operations(classified.incarnation)),
            transaction_id,
            operation,
            initial,
        )
        LOGGER.info("R18 outcome owner result=%s", OutcomeStatus.COMMITTED.value)
        participant = SubjectHistoryRecordParticipant(
            self._store,
            classified.incarnation,
            operation,
            JournalCommitProofSource(self._journal, self._recovery),
        )
        return CoordinatedResult(
            _OutcomeResult(OutcomeStatus.COMMITTED, claim_id, None, outcome),
            (participant,),
        )

    # ---------------------------------------------------------------- commit

    def _validate_commit(self, event: AgentEvent) -> None:
        """After this event's PREPARE; any failure is a runtime fail-stop."""

        from suzka.r16.history_adoption import current_incarnation

        pending = self._pending
        if event.event_type is not AgentEventType.R18_OUTCOME_COMMIT:
            if pending is not None:
                raise _OutcomeCommitRefused("stale Outcome commit pending")
            return
        if pending is None:
            return
        if pending.event != event:
            raise _OutcomeCommitRefused("unbound Outcome commit pending")
        if self._preview._state_snapshot() != pending.initial_state:
            raise _OutcomeCommitRefused("Outcome state changed before WAL")
        if current_incarnation(self._journal, self._store) != pending.incarnation:
            raise _OutcomeCommitRefused("Outcome incarnation changed before WAL")
        if self._decision._snapshot()[0] != pending.snapshot_hash:
            raise _OutcomeCommitRefused("Outcome snapshot changed before WAL")
        rows = self._store.operations(pending.incarnation)
        own = [row for row in rows if row.transaction_id == pending.transaction_id]
        others = tuple(row for row in rows if row.transaction_id != pending.transaction_id)
        if (
            len(own) != 1
            or own[0].phase is not HistoryPhase.PREPARED
            or own[0].operation_kind is not HistoryOperationKind.R18_OUTCOME_COMMIT
            or own[0].pending_bytes != pending.operation.canonical_bytes()
            or own[0].operation_digest != pending.operation.operation_digest
            or operations_digest(others) != pending.rows_digest
        ):
            raise _OutcomeCommitRefused("SubjectHistory changed before WAL")

    def _clear(self) -> None:
        self._pending = None


# ------------------------------------------------------------------- values


def _adopted_claim(
    classified: ReaderClassification, claim_id: str
) -> ExecutionClaimV1 | None:
    for adopted in classified.records:
        record = adopted.record
        if (
            adopted.record_kind is RecordKind.EXECUTION_CLAIM
            and type(record) is ExecutionClaimV1
            and record.claim_id == claim_id
            and record.direction is Direction.FORWARD
        ):
            return record
    return None


def _outcome_for(
    classified: ReaderClassification, claim: ExecutionClaimV1
) -> VerifiedOutcomeV1 | None:
    for adopted in classified.records:
        record = adopted.record
        if (
            adopted.record_kind is RecordKind.VERIFIED_OUTCOME
            and type(record) is VerifiedOutcomeV1
            and record.operation_key == claim.operation_key
            and record.direction is Direction.FORWARD
            and record.outcome_revision == _OUTCOME_REVISION
        ):
            return record
    return None


def _chain(
    classified: ReaderClassification, claim: ExecutionClaimV1
) -> tuple[ActionIntentV1, ApprovalChallengeV1]:
    """The ledger's checks (``ledger.py``): the claim's Intent and challenge
    are adopted and agree with it."""

    intent = _adopted_intent(classified, claim.intent_id)
    challenges = [
        adopted.record
        for adopted in classified.records
        if adopted.record_kind is RecordKind.APPROVAL_CHALLENGE
        and type(adopted.record) is ApprovalChallengeV1
        and adopted.record.challenge_digest == claim.challenge_digest
    ]
    if intent is None or len(challenges) != 1:
        raise _OutcomeHold(R18OutcomeReason.HISTORY_CONFLICT, claim.claim_id)
    challenge = challenges[0]
    assert type(challenge) is ApprovalChallengeV1
    if (
        intent.intent_digest != claim.intent_digest
        or intent.forward_key != claim.operation_key
        or intent.incarnation_id != classified.incarnation
        or challenge.intent_id != intent.intent_id
        or challenge.intent_digest != intent.intent_digest
        or challenge.nonce != claim.nonce
        or challenge.operation_key != claim.operation_key
        or challenge.direction is not Direction.FORWARD
    ):
        raise _OutcomeHold(R18OutcomeReason.HISTORY_CONFLICT, claim.claim_id)
    return intent, challenge


__all__ = [
    "TRANSIENT_OUTCOME_REASONS",
    "OutcomeOwnerV1",
    "OutcomeStatus",
    "R18OutcomeReason",
]
