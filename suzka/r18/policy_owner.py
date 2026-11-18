"""R18 IP-17a: the Policy owner (#411 D1 r7 §1, §2, §3a, §3b, §8a).

``ReviewArtifactPolicyV1`` owns ``R18_POLICY_COMMIT(intent_id | scan(after))``
and is the only writer of ``policy_verdicts`` (through the subject.history
record participant).  It is installed beside the Intent issuer (IP-16a-2),
whose Decision issuer's store, Journal, preview owner and clock it reuses;
while ``r18.enabled`` is false every event HOLDs ``R18_DISABLED`` before
anything is read, and the driver never submits one.  The event is refused in
the soft band (HP-8).

Inputs are the adopted forward Intent, its adopted forecast and the current
sources and target, never model output or a caller field.  The handler runs
before the event opens any R07 transaction, reads through one
``SubjectHistoryReader.classify()`` and, in the D1 §2 order:

1. ``r18.enabled``, else HOLD ``R18_DISABLED``;
2. the Intent: the explicit ``intent_id`` (an inverse id is HOLD
   ``INVERSE_DEFERRED``: IP-18's inverse is deferred, Human scope decision
   2026-11-18), or for ``scan(after)`` the first adopted forward Intent after
   the cursor (processing order) with an adopted forecast, no verdict and an
   open window, else ``NONE_LEFT``.  Not adopted -> ``NOT_YET_TERMINAL``;
   absent -> ``INTENT_UNKNOWN``; a verdict -> ``EXISTING``; the approval
   window has ended (§3a) -> ``CLOSED`` (nothing written);
3. the forecast: missing -> ``FORECAST_MISSING``; ``check_binding``, the
   payload digest and the target revision, else ``BINDING_MISMATCH``;
4. not superseded (§6-2): an ADOPTED head with another Decision -> DENY
   ``SOURCE_NOT_CURRENT``; a head that is not ADOPTED, a block, an
   unverifiable head, an unsafe ledger or a transient read -> HOLD;
5. the sources (``suzka.r18.requalify``): definite drift (moved sources, a
   self-origin Goal, a Commitment no longer ACTIVE at its sealed revision, or
   another obligation key) -> DENY ``SOURCE_NOT_CURRENT``; unavailable ->
   HOLD;
6. the target, read-only: another namespace / instance or any row for the
   group -> DENY ``TARGET_MISMATCH``; cannot be opened -> HOLD;
7. the ``AdmissionGate``: closed -> HOLD ``ADMISSION_GATE_CLOSED``;
8. §3b: every ``policy_verdicts_insert`` condition and UNIQUE is pre-checked
   (adopted Intent of this incarnation and direction, adopted forecast, no
   verdict, the ``verdict:`` key free), then headroom, a last source and
   window check, and a ``CoordinatedResult`` with one
   ``SubjectHistoryRecordParticipant`` carrying the ALLOW or DENY verdict.

A DENY found at steps 4-6 still passes steps 7 and 8 before it is written.
Any failure is a HOLD with a closed reason and writes nothing.

``_validate_commit`` (after this event's own PREPARE) checks the unchanged
domain snapshot, the incarnation, the R06 snapshot hash and that
SubjectHistory is exactly the evaluated rows plus this transaction's PREPARED
row, as the Intent issuer does; its failure is a runtime fail-stop.

No handler waits for an operator: an Intent nobody reviews closes at its
window end (APPROVAL_EXPIRED_UNCLAIMED, derived by the ledger reader, no
event) and a DENY closes it at once (POLICY_DENIED); either is row 2c for the
subject's next PLAN_PREVIEW (§8a).  The result is public-safe: a status, the
intent id and a closed reason, never request text.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Final

from suzka.r16.decision_issuer import _CONFLICT_HEADS, _DecisionIssuer, _sequence
from suzka.r16.history_attestation_digest import operations_digest
from suzka.r16.history_codec import CommittedDecisionV1
from suzka.r16.history_participant import (
    JournalCommitProofSource,
    SubjectHistoryRecordParticipant,
)
from suzka.r16.history_reader import (
    HeadStatus,
    ReaderClassification,
    ReaderTransient,
    SubjectHistoryReader,
)
from suzka.r16.history_records import (
    HistoryOperationKind,
    HistoryRecordOperationV1,
    RecordKind,
)
from suzka.r16.history_store import HistoryPhase, HistoryStoreError
from suzka.r16.successor_policy import LedgerState
from suzka.r17.contracts import ExpectedObservationV1
from suzka.r18.admission import ALWAYS_OPEN_UNTIL_IP18, AdmissionGate, gate_open
from suzka.r18.contracts import (
    INVERSE_INTENT_ID_PREFIX,
    ActionIntentV1,
    Direction,
    PolicyDenyReason,
    PolicyVerdict,
    PolicyVerdictV1,
    payload_digest,
)
from suzka.r18.ledger import RecordAt, derive_ledger
from suzka.r18.obligation import (
    TARGET_NAMESPACE,
    ObligationSecretError,
    ObligationSourceUnavailable,
    identity_for_source,
)
from suzka.r18.requalify import RequalifyFailed, RequalifyReason, requalify
from suzka.r18.target import TargetError, TargetReader
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

# The driver's startup bound on chained scans (D1 r7 §2).
MAX_POLICY_SCANS_PER_STARTUP: Final = 16
_MAX_ID: Final = 256


class PolicyCommitStatus(str, Enum):
    ALLOWED = "policy_allowed"
    DENIED = "policy_denied"
    EXISTING = "policy_existing"
    # §3a: the approval window ended; nothing is written.
    CLOSED = "policy_closed"
    NONE_LEFT = "policy_none_left"
    HOLD = "hold"


class R18PolicyReason(str, Enum):
    """Why the Policy owner held; a closed, source-free vocabulary."""

    R18_DISABLED = "r18_disabled"
    HISTORY_UNAVAILABLE = "history_unavailable"
    HISTORY_CONFLICT = "history_conflict"
    INTENT_UNKNOWN = "intent_unknown"
    # Transient: the Intent exists in this incarnation but is not adopted.
    NOT_YET_TERMINAL = "not_yet_terminal"
    # Human scope decision 2026-11-18: IP-17 handles FORWARD only.
    INVERSE_DEFERRED = "inverse_deferred"
    # Transient: the forecast is not adopted yet.
    FORECAST_MISSING = "forecast_missing"
    BINDING_MISMATCH = "binding_mismatch"
    # §2-4: the head is not ADOPTED / blocked / unverifiable, or the ledger
    # is not a pending Intent.
    HEAD_UNAVAILABLE = "head_unavailable"
    HEAD_CONFLICT = "head_conflict"
    LEDGER_UNSAFE = "ledger_unsafe"
    # §2-5 / §2-6: unavailable, never a DENY.
    SOURCE_UNAVAILABLE = "source_unavailable"
    SOURCE_CONFLICT = "source_conflict"
    DEADLINE_ELAPSED = "deadline_elapsed"
    OBLIGATION_SECRET_UNAVAILABLE = "obligation_secret_unavailable"
    TARGET_UNAVAILABLE = "target_unavailable"
    # §2-7.
    ADMISSION_GATE_CLOSED = "admission_gate_closed"
    # §3b: the ``verdict:`` key is reserved but not adopted (transient).
    IN_FLIGHT = "in_flight"
    HISTORY_CAPACITY = "history_capacity"


# The reasons the driver retries with its bounded delays.
TRANSIENT_POLICY_REASONS: Final = frozenset(
    {
        R18PolicyReason.NOT_YET_TERMINAL,
        R18PolicyReason.FORECAST_MISSING,
        R18PolicyReason.IN_FLIGHT,
    }
)
# requalify -> DENY (definite drift) or HOLD (unavailable), §2-5.
_DRIFT: Final = frozenset(
    {
        RequalifyReason.SOURCE_DRIFT,
        RequalifyReason.SELF_ORIGIN_GOAL,
        RequalifyReason.COMMITMENT_NOT_ACTIVE,
    }
)


@dataclass(frozen=True, slots=True)
class _PolicyResult:
    """The R18_POLICY_COMMIT value.

    Public-safe: a status, the intent id and a closed reason only.
    ``verdict`` is held privately for in-process inspection.
    """

    status: PolicyCommitStatus
    intent_id: str | None
    reason: R18PolicyReason | None
    verdict: PolicyVerdictV1 | None = None

    def __post_init__(self) -> None:
        if type(self.status) is not PolicyCommitStatus:
            raise TypeError("policy status must be exact")
        held = self.status is PolicyCommitStatus.HOLD
        if held != (self.reason is not None):
            raise ValueError("exactly a HOLD carries a reason")
        if self.reason is not None and type(self.reason) is not R18PolicyReason:
            raise TypeError("policy reason must be an exact R18PolicyReason")
        carries = self.status in (
            PolicyCommitStatus.ALLOWED,
            PolicyCommitStatus.DENIED,
            PolicyCommitStatus.EXISTING,
        )
        if carries != (self.verdict is not None):
            raise ValueError("only a written or existing result carries a verdict")
        if (self.status is PolicyCommitStatus.NONE_LEFT) != (
            self.intent_id is None and not held
        ):
            raise ValueError("only NONE_LEFT (or a HOLD) names no Intent")
        if self.verdict is not None:
            if self.verdict.intent_id != self.intent_id:
                raise ValueError("the result's intent id is not its verdict's")
            allowed = self.verdict.verdict is PolicyVerdict.ALLOW
            if self.status is PolicyCommitStatus.ALLOWED and not allowed:
                raise ValueError("an ALLOWED result carries an ALLOW")
            if self.status is PolicyCommitStatus.DENIED and allowed:
                raise ValueError("a DENIED result carries a DENY")

    @property
    def is_authoritative(self) -> bool:
        return self.status in (
            PolicyCommitStatus.ALLOWED,
            PolicyCommitStatus.DENIED,
            PolicyCommitStatus.EXISTING,
        )


@dataclass(frozen=True, slots=True)
class _PendingVerdict:
    event: AgentEvent
    incarnation: str
    snapshot_hash: str
    rows_digest: tuple[int, str]
    transaction_id: str
    operation: HistoryRecordOperationV1
    initial_state: object


@dataclass(frozen=True, slots=True)
class _Target:
    """The handler's request: one Intent, or a scan after a cursor."""

    intent_id: str | None
    after: str | None


class _PolicyHold(Exception):
    def __init__(self, reason: R18PolicyReason, intent_id: str | None = None) -> None:
        super().__init__(reason.value)
        self.reason = reason
        self.intent_id = intent_id


class _Deny(Exception):
    def __init__(self, reason: PolicyDenyReason) -> None:
        super().__init__(reason.value)
        self.reason = reason


class _PolicyCommitRefused(RuntimeError):
    """The pre-WAL check failed: a runtime fail-stop (no source text)."""


def _resolve_target_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else Path(os.getcwd()) / path


def _valid_id(value: object) -> bool:
    return type(value) is str and 0 < len(value) <= _MAX_ID


class ReviewArtifactPolicyV1:
    """The R18 Policy owner for one adopted forward ActionIntent."""

    def __init__(
        self,
        decision_issuer: _DecisionIssuer,
        runtime: AgentRuntime,
        *,
        enabled: bool,
        target_path: Path,
        admission_gate: AdmissionGate = ALWAYS_OPEN_UNTIL_IP18,
    ) -> None:
        if (
            type(decision_issuer) is not _DecisionIssuer
            or type(runtime) is not AgentRuntime
            or decision_issuer._preview._runtime is not runtime
            or decision_issuer._preview._issuer is not decision_issuer
            or type(enabled) is not bool
            or not isinstance(target_path, Path)
            or not isinstance(admission_gate, AdmissionGate)
        ):
            raise TypeError("unqualified Policy owner binding")
        self._decision = decision_issuer
        self._preview = decision_issuer._preview
        self._runtime = runtime
        self._store = decision_issuer._store
        self._journal = decision_issuer._journal
        self._recovery = decision_issuer._recovery
        self._enabled = enabled
        self._target_path = _resolve_target_path(target_path)
        self._clock: Callable[[], datetime] = decision_issuer._clock
        self._gate = admission_gate
        self._inflight: tuple[object, _Target] | None = None
        self._pending: _PendingVerdict | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------ submission

    def _submit_policy(self, intent_id: str) -> Future[AgentEventOutcome[object]]:
        """``R18_POLICY_COMMIT(intent_id)``."""

        if not _valid_id(intent_id):
            raise ValueError("a Policy verdict names one Intent id")
        return self._submit(_Target(intent_id, None))

    def _submit_scan(self, after: str | None) -> Future[AgentEventOutcome[object]]:
        """``R18_POLICY_COMMIT(scan(after))``: ``None`` scans from the start."""

        if after is not None and not _valid_id(after):
            raise ValueError("a Policy scan cursor is one Intent id")
        return self._submit(_Target(None, after))

    def _submit(self, target: _Target) -> Future[AgentEventOutcome[object]]:
        nonce = object()

        def handle() -> object:
            if self._inflight is not None or self._pending is not None:
                raise _PolicyCommitRefused("Policy commit already pending")
            self._inflight = (nonce, target)
            try:
                return self._commit(nonce)
            except BaseException:
                self._pending = None
                raise
            finally:
                self._inflight = None

        return self._runtime.submit(
            AgentEventType.R18_POLICY_COMMIT,
            AgentEventSource.RUNTIME_R18_POLICY_COMMIT,
            handle,
        )

    def _current_event(self, nonce: object) -> AgentEvent:
        event = self._runtime.current_event()
        if (
            self._inflight is None
            or self._inflight[0] is not nonce
            or event is None
            or event.event_type is not AgentEventType.R18_POLICY_COMMIT
            or event.source is not AgentEventSource.RUNTIME_R18_POLICY_COMMIT
            or type(event.processing_sequence) is not int
            or event.processing_sequence < 1
        ):
            raise _PolicyCommitRefused("unbound Policy commit event")
        return event

    # ------------------------------------------------------------ handler

    def _commit(self, nonce: object) -> object:
        self._pending = None
        event = self._current_event(nonce)
        assert self._inflight is not None
        target = self._inflight[1]
        try:
            return self._commit_checked(event, target)
        except (HistoryStoreError, AgentStateLoadError):
            return self._hold(R18PolicyReason.HISTORY_UNAVAILABLE, target.intent_id)
        except _PolicyHold as hold:
            return self._hold(hold.reason, hold.intent_id or target.intent_id)

    @staticmethod
    def _hold(reason: R18PolicyReason, intent_id: str | None) -> _PolicyResult:
        LOGGER.info("R18 policy owner result=hold reason=%s", reason.value)
        return _PolicyResult(PolicyCommitStatus.HOLD, intent_id, reason)

    def _commit_checked(self, event: AgentEvent, target: _Target) -> object:
        # 1.
        if not self._enabled:
            raise _PolicyHold(R18PolicyReason.R18_DISABLED)
        if target.intent_id is not None and target.intent_id.startswith(
            INVERSE_INTENT_ID_PREFIX
        ):
            raise _PolicyHold(R18PolicyReason.INVERSE_DEFERRED)
        # 2. one classification; the Intent.
        initial = self._preview._state_snapshot()
        classified = self._classify()
        now = self._clock()
        if target.intent_id is None:
            intent = self._scan_target(classified, target.after, now)
            if intent is None:
                LOGGER.info("R18 policy owner result=none_left")
                return _PolicyResult(PolicyCommitStatus.NONE_LEFT, None, None)
        else:
            intent = self._target_intent(classified, target.intent_id)
        intent_id = intent.intent_id
        existing = _verdict_for(classified, intent_id)
        if existing is not None:
            LOGGER.info("R18 policy owner result=existing")
            return _PolicyResult(PolicyCommitStatus.EXISTING, intent_id, None, existing)
        if not intent.window_open(now):
            LOGGER.info("R18 policy owner result=closed")
            return _PolicyResult(PolicyCommitStatus.CLOSED, intent_id, None)
        # 3. the forecast.
        forecast = _forecast_for(classified, intent_id)
        if forecast is None:
            raise _PolicyHold(R18PolicyReason.FORECAST_MISSING, intent_id)
        _bind(forecast, intent)
        # 4-6: DENY or pass.
        deny: PolicyDenyReason | None = None
        decision: CommittedDecisionV1 | None = None
        requalified: Requalified | None = None
        try:
            decision = self._not_superseded(classified, intent, now)
            requalified = self._sources(event, decision, intent)
            self._target(intent)
        except _Deny as denied:
            deny = denied.reason
        # 7. the AdmissionGate.
        if not gate_open(self._gate):
            raise _PolicyHold(R18PolicyReason.ADMISSION_GATE_CLOSED, intent_id)
        # 8. §3b pre-checks, then the verdict.
        verdict = PolicyVerdictV1(
            Direction.FORWARD,
            intent_id,
            intent.intent_digest,
            intent.forward_key,
            PolicyVerdict.ALLOW if deny is None else PolicyVerdict.DENY,
            deny,
            event.event_id,
            _sequence(event),
        )
        operation = HistoryRecordOperationV1(
            HistoryOperationKind.R18_POLICY_COMMIT,
            classified.incarnation,
            event.event_id,
            _sequence(event),
            TransactionKind.EVENT_MUTATION.value,
            RecordKind.POLICY_VERDICT,
            verdict,
        )
        self._precheck(classified, intent, operation)
        try:
            self._store.check_headroom(len(operation.canonical_bytes()))
        except HistoryStoreError:
            raise _PolicyHold(R18PolicyReason.HISTORY_CAPACITY, intent_id) from None
        # The target and secret reads above can take seconds: the sources
        # and the window are checked again, last before PREPARE.
        if requalified is not None and decision is not None:
            self._still_qualified(event, decision, requalified, intent_id)
        if self._preview._state_snapshot() != initial:
            raise _PolicyHold(R18PolicyReason.SOURCE_UNAVAILABLE, intent_id)
        if not intent.window_open(self._clock()):
            LOGGER.info("R18 policy owner result=closed")
            return _PolicyResult(PolicyCommitStatus.CLOSED, intent_id, None)
        snapshot_hash, _as_of = self._decision._snapshot()
        transaction_id = TransactionCoordinator.derive_transaction_id(
            event, TransactionKind.EVENT_MUTATION
        )
        self._pending = _PendingVerdict(
            event,
            classified.incarnation,
            snapshot_hash,
            operations_digest(self._store.operations(classified.incarnation)),
            transaction_id,
            operation,
            initial,
        )
        status = (
            PolicyCommitStatus.ALLOWED if deny is None else PolicyCommitStatus.DENIED
        )
        LOGGER.info(
            "R18 policy owner result=%s%s",
            status.value,
            "" if deny is None else f" reason={deny.value}",
        )
        participant = SubjectHistoryRecordParticipant(
            self._store,
            classified.incarnation,
            operation,
            JournalCommitProofSource(self._journal, self._recovery),
        )
        return CoordinatedResult(
            _PolicyResult(status, intent_id, None, verdict),
            (participant,),
        )

    # ------------------------------------------------------------ steps

    def _classify(self) -> ReaderClassification:
        reader = SubjectHistoryReader(self._journal, self._store)
        try:
            classified = reader.classify(inspection=self._journal.inspect())
        except ReaderTransient:
            raise _PolicyHold(R18PolicyReason.HISTORY_UNAVAILABLE) from None
        if classified.global_block is not None:
            raise _PolicyHold(
                R18PolicyReason.HISTORY_CONFLICT
                if classified.global_block in _CONFLICT_HEADS
                else R18PolicyReason.HISTORY_UNAVAILABLE
            )
        return classified

    @staticmethod
    def _scan_target(
        classified: ReaderClassification, after: str | None, now: datetime
    ) -> ActionIntentV1 | None:
        """The first adopted forward Intent after the cursor (processing
        order) with an adopted forecast, no verdict and an open window."""

        intents = [
            adopted
            for adopted in sorted(
                classified.records,
                key=lambda item: (item.processing_sequence, item.transaction_id),
            )
            if adopted.record_kind is RecordKind.ACTION_INTENT
            and type(adopted.record) is ActionIntentV1
        ]
        start = 0
        if after is not None:
            ids = [_intent_of(adopted).intent_id for adopted in intents]
            if after not in ids:
                # A cursor that is not an adopted Intent of this incarnation
                # cannot be continued: the driver stops on a HOLD with no id.
                raise _PolicyHold(R18PolicyReason.INTENT_UNKNOWN)
            start = ids.index(after) + 1
        forecasted = {
            record.record.action_intent_id
            for record in classified.records
            if record.record_kind is RecordKind.ACTION_FORECAST
            and type(record.record) is ExpectedObservationV1
        }
        decided = {
            record.record.intent_id
            for record in classified.records
            if record.record_kind is RecordKind.POLICY_VERDICT
            and type(record.record) is PolicyVerdictV1
        }
        for adopted in intents[start:]:
            intent = _intent_of(adopted)
            if (
                intent.intent_id in forecasted
                and intent.intent_id not in decided
                and intent.window_open(now)
            ):
                return intent
        return None

    @staticmethod
    def _target_intent(classified: ReaderClassification, intent_id: str) -> ActionIntentV1:
        for adopted in classified.records:
            record = adopted.record
            if (
                adopted.record_kind is RecordKind.ACTION_INTENT
                and type(record) is ActionIntentV1
                and record.intent_id == intent_id
            ):
                return record
        for row in classified.rows:
            if (
                row.operation_kind is not HistoryOperationKind.R18_INTENT_ISSUE
                or row.phase not in (HistoryPhase.PREPARED, HistoryPhase.FINALIZED)
                or row.pending_bytes is None
            ):
                continue
            try:
                operation = HistoryRecordOperationV1.from_bytes(row.pending_bytes)
            except (TypeError, ValueError):
                raise _PolicyHold(R18PolicyReason.HISTORY_CONFLICT, intent_id) from None
            record = operation.record
            if type(record) is ActionIntentV1 and record.intent_id == intent_id:
                raise _PolicyHold(R18PolicyReason.NOT_YET_TERMINAL, intent_id)
        raise _PolicyHold(R18PolicyReason.INTENT_UNKNOWN, intent_id)

    def _not_superseded(
        self, classified: ReaderClassification, intent: ActionIntentV1, now: datetime
    ) -> CommittedDecisionV1:
        """§2-4 (§6-2): the Intent's Decision is still its lineage's head and
        the group's ledger is this pending Intent."""

        intent_id = intent.intent_id
        head = classified.heads.get(intent.lineage_scope)
        if head is None or head.status is not HeadStatus.ADOPTED:
            if head is not None and head.reason in _CONFLICT_HEADS:
                raise _PolicyHold(R18PolicyReason.HEAD_CONFLICT, intent_id)
            raise _PolicyHold(R18PolicyReason.HEAD_UNAVAILABLE, intent_id)
        decision = head.decision
        assert decision is not None and head.transaction_id is not None
        if (decision.decision_id, decision.content_digest) != (
            intent.decision_id,
            intent.decision_committed_digest,
        ):
            raise _Deny(PolicyDenyReason.SOURCE_NOT_CURRENT)
        if intent.incarnation_id != classified.incarnation or not self._decision._head_verified(
            decision, classified.incarnation, head.transaction_id
        ):
            raise _PolicyHold(R18PolicyReason.HEAD_CONFLICT, intent_id)
        if self._store.live_key_owner(f"claim:{intent.forward_key}") is not None:
            raise _PolicyHold(R18PolicyReason.LEDGER_UNSAFE, intent_id)
        ledger = derive_ledger(
            (RecordAt(item.record, item.processing_sequence) for item in classified.records),
            intent.obligation_group_key,
            now,
        )
        if ledger.state is not LedgerState.INTENT_ISSUED or not any(
            ref.ref_id == intent_id for ref in ledger.refs
        ):
            raise _PolicyHold(R18PolicyReason.LEDGER_UNSAFE, intent_id)
        return decision

    def _sources(
        self, event: AgentEvent, decision: CommittedDecisionV1, intent: ActionIntentV1
    ) -> Requalified:
        """§2-5: the shared requalification, and the same obligation key."""

        intent_id = intent.intent_id
        try:
            requalified = requalify(self._preview, decision, event, self._clock)
        except RequalifyFailed as failed:
            if failed.reason in _DRIFT:
                raise _Deny(PolicyDenyReason.SOURCE_NOT_CURRENT) from None
            raise _PolicyHold(R18PolicyReason[failed.reason.name], intent_id) from None
        try:
            identity = identity_for_source(
                self._target_path.parent,
                source=self._preview._source,
                subject_id=self._store.meta().subject_id,
                q=requalified.q,
                scope=requalified.obligation_scope,
            )
        except ObligationSecretError:
            raise _PolicyHold(
                R18PolicyReason.OBLIGATION_SECRET_UNAVAILABLE, intent_id
            ) from None
        except ObligationSourceUnavailable:
            raise _PolicyHold(R18PolicyReason.SOURCE_UNAVAILABLE, intent_id) from None
        if identity.group_key != intent.obligation_group_key:
            raise _Deny(PolicyDenyReason.SOURCE_NOT_CURRENT)
        return requalified

    def _target(self, intent: ActionIntentV1) -> None:
        """§2-6: read-only; a missing target is never created."""

        intent_id = intent.intent_id
        try:
            reader = TargetReader.open(self._target_path)
        except (TargetError, OSError, ValueError):
            raise _PolicyHold(R18PolicyReason.TARGET_UNAVAILABLE, intent_id) from None
        try:
            status = reader.status(intent.obligation_group_key)
        except (TargetError, OSError, ValueError):
            raise _PolicyHold(R18PolicyReason.TARGET_UNAVAILABLE, intent_id) from None
        finally:
            reader.close()
        if (
            status.meta.target_namespace != TARGET_NAMESPACE
            or status.meta.target_namespace != intent.target_namespace
            or status.meta.target_instance_id != intent.target_instance_id
            or intent.target_expected_revision != 0
            or status.artifact is not None
            or status.forward is not None
            or status.inverse is not None
        ):
            raise _Deny(PolicyDenyReason.TARGET_MISMATCH)

    def _precheck(
        self,
        classified: ReaderClassification,
        intent: ActionIntentV1,
        operation: HistoryRecordOperationV1,
    ) -> None:
        """§3b: every ``policy_verdicts_insert`` condition and UNIQUE, from the
        same classification, so FINALIZE cannot refuse what PREPARE admits."""

        intent_id = intent.intent_id
        if (
            intent.incarnation_id != classified.incarnation
            or _adopted_intent(classified, intent_id) is not intent
        ):
            raise _PolicyHold(R18PolicyReason.HISTORY_CONFLICT, intent_id)
        if _forecast_for(classified, intent_id) is None:
            raise _PolicyHold(R18PolicyReason.FORECAST_MISSING, intent_id)
        if _verdict_for(classified, intent_id) is not None:
            raise _PolicyHold(R18PolicyReason.HISTORY_CONFLICT, intent_id)
        verdict = operation.record
        assert type(verdict) is PolicyVerdictV1
        if verdict.direction is not Direction.FORWARD or verdict.intent_id != intent_id:
            raise _PolicyHold(R18PolicyReason.HISTORY_CONFLICT, intent_id)
        # A verdict row of this intent that is not adopted yet (PREPARED, or
        # FINALIZED but not terminal) holds the key: UNIQUE(intent_id).
        if self._store.live_keys(operation.reserved_keys):
            raise _PolicyHold(R18PolicyReason.IN_FLIGHT, intent_id)

    def _still_qualified(
        self,
        event: AgentEvent,
        decision: CommittedDecisionV1,
        requalified: Requalified,
        intent_id: str,
    ) -> None:
        try:
            again = requalify(self._preview, decision, event, self._clock)
        except RequalifyFailed:
            raise _PolicyHold(R18PolicyReason.SOURCE_UNAVAILABLE, intent_id) from None
        if again.q != requalified.q or again.initial != requalified.initial:
            raise _PolicyHold(R18PolicyReason.SOURCE_UNAVAILABLE, intent_id)

    # ------------------------------------------------------------ commit

    def _validate_commit(self, event: AgentEvent) -> None:
        """After this event's PREPARE; any failure is a runtime fail-stop."""

        from suzka.r16.history_adoption import current_incarnation

        pending = self._pending
        if event.event_type is not AgentEventType.R18_POLICY_COMMIT:
            if pending is not None:
                raise _PolicyCommitRefused("stale Policy commit pending")
            return
        if pending is None:
            return
        if pending.event != event:
            raise _PolicyCommitRefused("unbound Policy commit pending")
        if self._preview._state_snapshot() != pending.initial_state:
            raise _PolicyCommitRefused("Policy state changed before WAL")
        if current_incarnation(self._journal, self._store) != pending.incarnation:
            raise _PolicyCommitRefused("Policy incarnation changed before WAL")
        if self._decision._snapshot()[0] != pending.snapshot_hash:
            raise _PolicyCommitRefused("Policy snapshot changed before WAL")
        rows = self._store.operations(pending.incarnation)
        own = [row for row in rows if row.transaction_id == pending.transaction_id]
        others = tuple(row for row in rows if row.transaction_id != pending.transaction_id)
        if (
            len(own) != 1
            or own[0].phase is not HistoryPhase.PREPARED
            or own[0].operation_kind is not HistoryOperationKind.R18_POLICY_COMMIT
            or own[0].pending_bytes != pending.operation.canonical_bytes()
            or own[0].operation_digest != pending.operation.operation_digest
            or operations_digest(others) != pending.rows_digest
        ):
            raise _PolicyCommitRefused("SubjectHistory changed before WAL")

    def _clear(self) -> None:
        self._pending = None


# ------------------------------------------------------------------- values


def _intent_of(adopted: object) -> ActionIntentV1:
    record = getattr(adopted, "record", None)
    assert type(record) is ActionIntentV1
    return record


def _bind(forecast: ExpectedObservationV1, intent: ActionIntentV1) -> None:
    """§2-3: the forecast is bound to this Intent, its Decision and target."""

    try:
        forecast.check_binding(
            decision_id=intent.decision_id,
            decision_committed_digest=intent.decision_committed_digest,
            policy_kind=intent.policy_kind,
            policy_row=intent.policy_row,
            action_intent_id=intent.intent_id,
            action_intent_digest=intent.intent_digest,
            forward_key=intent.forward_key,
            target_instance_id=intent.target_instance_id,
            target_namespace=intent.target_namespace,
            target_expected_revision=intent.target_expected_revision,
        )
    except (TypeError, ValueError):
        raise _PolicyHold(R18PolicyReason.BINDING_MISMATCH, intent.intent_id) from None
    if (
        forecast.predicted_payload_digest != intent.payload_digest
        or forecast.predicted_payload_digest
        != payload_digest(intent.artifact.payload_bytes())
        or forecast.incarnation_id != intent.incarnation_id
        or forecast.lineage_scope != intent.lineage_scope
    ):
        raise _PolicyHold(R18PolicyReason.BINDING_MISMATCH, intent.intent_id)


def _adopted_intent(
    classified: ReaderClassification, intent_id: str
) -> ActionIntentV1 | None:
    for adopted in classified.records:
        record = adopted.record
        if (
            adopted.record_kind is RecordKind.ACTION_INTENT
            and type(record) is ActionIntentV1
            and record.intent_id == intent_id
        ):
            return record
    return None


def _forecast_for(
    classified: ReaderClassification, intent_id: str
) -> ExpectedObservationV1 | None:
    for adopted in classified.records:
        record = adopted.record
        if (
            adopted.record_kind is RecordKind.ACTION_FORECAST
            and type(record) is ExpectedObservationV1
            and record.action_intent_id == intent_id
        ):
            return record
    return None


def _verdict_for(
    classified: ReaderClassification, intent_id: str
) -> PolicyVerdictV1 | None:
    for adopted in classified.records:
        record = adopted.record
        if (
            adopted.record_kind is RecordKind.POLICY_VERDICT
            and type(record) is PolicyVerdictV1
            and record.intent_id == intent_id
        ):
            return record
    return None


__all__ = [
    "MAX_POLICY_SCANS_PER_STARTUP",
    "TRANSIENT_POLICY_REASONS",
    "PolicyCommitStatus",
    "R18PolicyReason",
    "ReviewArtifactPolicyV1",
]
