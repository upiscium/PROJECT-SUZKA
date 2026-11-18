"""R18 IP-16a-2: the forward ActionIntent issuer (#411 comment 6916372789 §1-§3).

``ReviewArtifactIntentIssuerV1`` owns ``R18_ACTION_INTENT_ISSUE(lineage_scope)``
and is the only writer of ``action_intents`` (through the subject.history
record participant).  It is installed with the CommittedDecision issuer
(IP-10), whose preview owner, store and Journal it reuses; while
``r18.enabled`` is false every event HOLDs ``R18_DISABLED`` and writes
nothing, and the driver never submits one (D1 §1).

The handler runs before the event opens any R07 transaction, in the D1 §3
order; any failure is a HOLD with a closed reason and no artifact (A9-R1
R1-3), so a stored artifact always carries six PASS gates (r2 P3-11):

1. one ``SubjectHistoryReader.classify((scope,))`` gives the head and the
   adopted records (r2 P2-7).  The head must be ADOPTED (which the reader
   grants only to a terminal Decision), of policy row 4 (not 4b) and pass
   the IP-10 sealed replay; a blocked head maps like the Decision issuer's
   (DIVERGED / POISONED / gap / duplicate -> conflict, else unavailable);
2. the sources are requalified through the preview owner (twice, equal) and
   must equal the Decision's sealed ``source_refs``; the Goal is a user
   request; the Commitment is ACTIVE and not within ``DEADLINE_MARGIN`` of
   its deadline (the R16 rule, r2 P3-10);
3. the six gates are PASS (only reached when 1 and 2 passed);
4. the obligation identity (§3a / §8-1 (a)): the secret next to the target,
   the subject, X's Context, the ``user_input`` of X's committed episode and
   the Commitment scope.  The text is used only inside the HMAC;
5. an adopted Intent for this Decision -> EXISTING, before the target check
   (r3 P3-4);
6. the target (read-only, never created): its instance id, revision 0, and
   no row for the group (else ``PRIOR_OBLIGATION``, r2 P2-8);
7. the keys: this incarnation's Intent keys reserved but not adopted ->
   ``IN_FLIGHT`` (transient); adopted by another Decision, or the group's
   Claim key reserved at all -> ``PRIOR_OBLIGATION``;
8. headroom, a last requalification and deadline check, the
   ``AdmissionGate`` (IP-17a, closed -> ``ADMISSION_GATE_CLOSED``), then a
   ``CoordinatedResult`` with one
   ``SubjectHistoryRecordParticipant`` carrying
   ``HistoryRecordOperationV1(R18_INTENT_ISSUE, ACTION_INTENT, ActionIntentV1)``.

``_validate_commit`` (after this event's own PREPARE) checks the unchanged
domain snapshot, the incarnation, the R06 snapshot hash and that
SubjectHistory is exactly the evaluated rows plus this transaction's PREPARED
row, as the Decision issuer does; like it, it does not re-check the deadline
(that would turn a near deadline into a fail-stop; the margin covers the
handler -> ``_validate_commit`` window).  Its failure is a runtime fail-stop,
recovered at startup (PREPARED -> tombstone, internally committed -> roll
forward).

IP-17a (#411 D1 r7 §2-5, §3a, §12-1): the source requalification lives in
``suzka.r18.requalify`` (shared with the Policy owner), and the Intent carries
``approval_window_end_utc = min(now + r18.approval_window_seconds, Commitment
deadline - DEADLINE_MARGIN)`` (floored to the second).  A window shorter than
``MIN_APPROVAL_WINDOW`` is the existing ``DEADLINE_ELAPSED`` HOLD.

The forecast is IP-16a-3's; the Decision issuer's ledger reader (IP-16b)
holds a re-issued PLAN_PREVIEW at row 2 once this Intent exists, and the
obligation key is computed with the same shared helper.  A Claim key is unique across
incarnations, and a Claim of this incarnation implies an Intent for the same
group (refused by the Intent keys first), so any live Claim key is treated as
a prior obligation without telling the incarnations apart.  The result is
public-safe: a status, the intent id and a closed reason, never request text.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import logging
import os
from pathlib import Path
import re
from typing import TYPE_CHECKING, Final

from suzka.r16.decision_issuer import (
    DEADLINE_MARGIN,
    _CONFLICT_HEADS,
    _context_digest,
    _DecisionIssuer,
    _sequence,
)
from suzka.r16.history_attestation_digest import operations_digest
from suzka.r16.history_codec import (
    CommittedDecisionV1,
    PolicyKind,
    PolicyRow,
)
from suzka.r16.history_participant import (
    JournalCommitProofSource,
    SubjectHistoryRecordParticipant,
)
from suzka.r16.history_reader import (
    HeadBlockReason,
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
from suzka.r18.contracts import (
    ActionIntentV1,
    GateResult,
    GateStatus,
    GoalOrigin,
    ReviewArtifactV1,
    ReviewGate,
    ReviewSourceRef,
    SourceRole,
)
from suzka.r18.admission import ALWAYS_OPEN_UNTIL_IP18, AdmissionGate, gate_open
from suzka.r18.obligation import (
    TARGET_NAMESPACE,
    ObligationIdentity,
    ObligationSecretError,
    ObligationSourceUnavailable,
    identity_for_source,
)
from suzka.r18.requalify import (
    RequalifyFailed,
    active_commitment,
    commitment_deadline,
    qualify_twice,
    check_sources,
)
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
    from suzka.r16.plan_preview import _QualifiedPreview

LOGGER = logging.getLogger(__name__)

_SCOPE_PATTERN: Final = re.compile(r"x-[0-9a-f]{64}")
_INTENT_POLICY_KINDS: Final = frozenset(
    {PolicyKind.SUCCESSOR_POLICY_V2, PolicyKind.SUCCESSOR_POLICY_V3}
)
_TARGET_EXPECTED_REVISION: Final = 0
# IP-17a §3a: the default and bounds of r18.approval_window_seconds (HD-16
# (a)), and the shortest window an Intent may carry.
DEFAULT_APPROVAL_WINDOW_SECONDS: Final = 3600
MIN_APPROVAL_WINDOW: Final = timedelta(seconds=120)


class IntentIssueStatus(str, Enum):
    ISSUED = "intent_issued"
    EXISTING = "intent_existing"
    HOLD = "hold"


class R18IntentReason(str, Enum):
    """Why the Intent issuer held; a closed, source-free vocabulary (D1 §3)."""

    R18_DISABLED = "r18_disabled"
    # §3-1: the lineage head.
    HEAD_UNAVAILABLE = "head_unavailable"
    HEAD_CONFLICT = "head_conflict"
    NOT_ROW_4 = "not_row_4"
    # §3-2: the sources.
    SOURCE_UNAVAILABLE = "source_unavailable"
    SOURCE_CONFLICT = "source_conflict"
    SOURCE_DRIFT = "source_drift"
    SELF_ORIGIN_GOAL = "self_origin_goal"
    COMMITMENT_NOT_ACTIVE = "commitment_not_active"
    DEADLINE_ELAPSED = "deadline_elapsed"
    # §3-4 / §3a: the obligation secret is missing or untrusted (BLOCKED).
    OBLIGATION_SECRET_UNAVAILABLE = "obligation_secret_unavailable"
    # §3-6 / §3-7.
    TARGET_UNAVAILABLE = "target_unavailable"
    PRIOR_OBLIGATION = "prior_obligation"
    IN_FLIGHT = "in_flight"
    HISTORY_CAPACITY = "history_capacity"
    # IP-17a §2-7: the AdmissionGate is closed (IP-18's sweep).
    ADMISSION_GATE_CLOSED = "admission_gate_closed"


@dataclass(frozen=True, slots=True)
class _IntentIssueResult:
    """The R18_ACTION_INTENT_ISSUE value.

    Public-safe: a status, the intent id and a closed reason only.  ``intent``
    is held privately for in-process inspection.
    """

    status: IntentIssueStatus
    intent_id: str | None
    reason: R18IntentReason | None
    intent: ActionIntentV1 | None = None

    def __post_init__(self) -> None:
        if type(self.status) is not IntentIssueStatus:
            raise TypeError("intent issue status must be exact")
        held = self.status is IntentIssueStatus.HOLD
        if held != (self.reason is not None):
            raise ValueError("exactly a HOLD carries a reason")
        if self.reason is not None and type(self.reason) is not R18IntentReason:
            raise TypeError("intent reason must be an exact R18IntentReason")
        if held != (self.intent is None) or held != (self.intent_id is None):
            raise ValueError("only a non-HOLD result carries an Intent")
        if self.intent is not None and self.intent.intent_id != self.intent_id:
            raise ValueError("the result's intent id is not its Intent's")

    @property
    def is_authoritative(self) -> bool:
        return self.status is not IntentIssueStatus.HOLD


@dataclass(frozen=True, slots=True)
class _PendingIntent:
    event: AgentEvent
    incarnation: str
    snapshot_hash: str
    rows_digest: tuple[int, str]
    transaction_id: str
    operation: HistoryRecordOperationV1
    initial_state: object


class _IntentHold(Exception):
    def __init__(self, reason: R18IntentReason) -> None:
        super().__init__(reason.value)
        self.reason = reason


class _IntentCommitRefused(RuntimeError):
    """The pre-WAL check failed: a runtime fail-stop (no source text)."""


def _resolve_target_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else Path(os.getcwd()) / path


class ReviewArtifactIntentIssuerV1:
    """Source-admitted forward ActionIntent issuer (no target write)."""

    def __init__(
        self,
        decision_issuer: _DecisionIssuer,
        runtime: AgentRuntime,
        *,
        enabled: bool,
        target_path: Path,
        approval_window_seconds: int = DEFAULT_APPROVAL_WINDOW_SECONDS,
        admission_gate: AdmissionGate = ALWAYS_OPEN_UNTIL_IP18,
    ) -> None:
        if (
            type(decision_issuer) is not _DecisionIssuer
            or type(runtime) is not AgentRuntime
            or decision_issuer._preview._runtime is not runtime
            or decision_issuer._preview._issuer is not decision_issuer
            or type(enabled) is not bool
            or not isinstance(target_path, Path)
            or type(approval_window_seconds) is not int
            or not isinstance(admission_gate, AdmissionGate)
        ):
            raise TypeError("unqualified Intent issuer binding")
        if not 600 <= approval_window_seconds <= 86_400:
            raise ValueError("approval window must be 600..86400 seconds")
        self._decision = decision_issuer
        self._preview = decision_issuer._preview
        self._runtime = runtime
        self._store = decision_issuer._store
        self._journal = decision_issuer._journal
        self._recovery = decision_issuer._recovery
        self._enabled = enabled
        self._target_path = _resolve_target_path(target_path)
        self._clock: Callable[[], datetime] = decision_issuer._clock
        self._window = timedelta(seconds=approval_window_seconds)
        self._gate = admission_gate
        self._inflight: tuple[object, str] | None = None
        self._pending: _PendingIntent | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------ submission

    def _submit_intent(
        self, lineage_scope: str
    ) -> Future[AgentEventOutcome[object]]:
        if type(lineage_scope) is not str or _SCOPE_PATTERN.fullmatch(lineage_scope) is None:
            raise ValueError("an Intent names one review lineage scope")
        nonce = object()

        def handle() -> object:
            if self._inflight is not None or self._pending is not None:
                raise _IntentCommitRefused("Intent issue already pending")
            self._inflight = (nonce, lineage_scope)
            try:
                return self._issue(nonce)
            except BaseException:
                self._pending = None
                raise
            finally:
                self._inflight = None

        return self._runtime.submit(
            AgentEventType.R18_ACTION_INTENT_ISSUE,
            AgentEventSource.RUNTIME_R18_ACTION_INTENT_ISSUE,
            handle,
        )

    def _current_event(self, nonce: object) -> AgentEvent:
        event = self._runtime.current_event()
        if (
            self._inflight is None
            or self._inflight[0] is not nonce
            or event is None
            or event.event_type is not AgentEventType.R18_ACTION_INTENT_ISSUE
            or event.source is not AgentEventSource.RUNTIME_R18_ACTION_INTENT_ISSUE
            or type(event.processing_sequence) is not int
            or event.processing_sequence < 1
        ):
            raise _IntentCommitRefused("unbound Intent issue event")
        return event

    # ------------------------------------------------------------ handler

    def _issue(self, nonce: object) -> object:
        self._pending = None
        event = self._current_event(nonce)
        assert self._inflight is not None
        scope = self._inflight[1]
        try:
            return self._issue_checked(event, scope)
        except (HistoryStoreError, AgentStateLoadError):
            return self._hold(R18IntentReason.SOURCE_UNAVAILABLE)
        except _IntentHold as hold:
            return self._hold(hold.reason)

    @staticmethod
    def _hold(reason: R18IntentReason) -> _IntentIssueResult:
        LOGGER.info("R18 intent issuer result=hold reason=%s", reason.value)
        return _IntentIssueResult(IntentIssueStatus.HOLD, None, reason)

    def _issue_checked(self, event: AgentEvent, scope: str) -> object:
        if not self._enabled:
            raise _IntentHold(R18IntentReason.R18_DISABLED)
        # 1. head and lineage records from one classification.
        classified, decision, depth = self._head(scope)
        # 2. sources (IP-17a: the shared requalification).
        initial = self._preview._state_snapshot()
        q = self._qualified(event, decision, initial)
        try:
            check_sources(q, decision)
        except RequalifyFailed as failed:
            raise _IntentHold(R18IntentReason[failed.reason.name]) from None
        obligation_scope = self._active_commitment(q)
        # 3. gates: every check above passed, so each gate is PASS.
        gates = tuple(GateResult(gate, GateStatus.PASS) for gate in sorted(ReviewGate))
        # 4. obligation identity.
        identity = self._identity(q, obligation_scope)
        # 5. an adopted Intent for this Decision is returned before the target.
        existing = _intent_for(classified, lambda intent: intent.decision_id == decision.decision_id)
        if existing is not None:
            LOGGER.info("R18 intent issuer result=existing")
            return _IntentIssueResult(
                IntentIssueStatus.EXISTING, existing.intent_id, None, existing
            )
        # IP-17a §3a: the approval window, from the clock before the slow
        # target read (the deadline is checked again last, below).
        window_end = self._window_end(q)
        # 6. target.
        target_instance_id = self._target(identity.group_key)
        intent = self._intent(
            event,
            scope,
            classified.incarnation,
            decision,
            depth,
            q,
            identity,
            gates,
            target_instance_id,
            window_end,
        )
        operation = HistoryRecordOperationV1(
            HistoryOperationKind.R18_INTENT_ISSUE,
            classified.incarnation,
            event.event_id,
            _sequence(event),
            TransactionKind.EVENT_MUTATION.value,
            RecordKind.ACTION_INTENT,
            intent,
        )
        # 7. keys.
        self._check_keys(classified, operation, intent)
        # 8. headroom, then the sources must not have moved.
        try:
            self._store.check_headroom(len(operation.canonical_bytes()))
        except HistoryStoreError:
            raise _IntentHold(R18IntentReason.HISTORY_CAPACITY) from None
        if self._qualified(event, decision, initial) != q:
            raise _IntentHold(R18IntentReason.SOURCE_UNAVAILABLE)
        # The secret, episode and target reads above can take seconds (the
        # target's busy timeout): the deadline rule is applied again here,
        # last before PREPARE, as the Decision issuer does (D2 P2-2).
        self._active_commitment(q)
        # IP-17a §2-7: the AdmissionGate, last before PREPARE.
        if not gate_open(self._gate):
            raise _IntentHold(R18IntentReason.ADMISSION_GATE_CLOSED)
        snapshot_hash, _as_of = self._decision._snapshot()
        transaction_id = TransactionCoordinator.derive_transaction_id(
            event, TransactionKind.EVENT_MUTATION
        )
        self._pending = _PendingIntent(
            event,
            classified.incarnation,
            snapshot_hash,
            operations_digest(self._store.operations(classified.incarnation)),
            transaction_id,
            operation,
            initial,
        )
        LOGGER.info("R18 intent issuer result=issued")
        participant = SubjectHistoryRecordParticipant(
            self._store,
            classified.incarnation,
            operation,
            JournalCommitProofSource(self._journal, self._recovery),
        )
        return CoordinatedResult(
            _IntentIssueResult(IntentIssueStatus.ISSUED, intent.intent_id, None, intent),
            (participant,),
        )

    # ------------------------------------------------------------ steps

    def _head(self, scope: str) -> tuple[ReaderClassification, CommittedDecisionV1, int]:
        reader = SubjectHistoryReader(self._journal, self._store)
        try:
            classified = reader.classify((scope,), inspection=self._journal.inspect())
        except ReaderTransient:
            raise _IntentHold(R18IntentReason.HEAD_UNAVAILABLE) from None
        block = classified.global_block
        head = classified.heads.get(scope)
        if block is None and head is not None and head.status is HeadStatus.BLOCKED:
            block = head.reason
        if block is not None or head is None:
            raise _IntentHold(_blocked_reason(block))
        if head.status is not HeadStatus.ADOPTED:
            raise _IntentHold(R18IntentReason.NOT_ROW_4)
        decision = head.decision
        assert decision is not None and head.transaction_id is not None
        if (
            decision.policy_row is not PolicyRow.ROW_4
            or decision.policy_kind not in _INTENT_POLICY_KINDS
        ):
            raise _IntentHold(R18IntentReason.NOT_ROW_4)
        if decision.incarnation_id != classified.incarnation or not self._decision._head_verified(
            decision, classified.incarnation, head.transaction_id
        ):
            raise _IntentHold(R18IntentReason.HEAD_CONFLICT)
        return classified, decision, head.depth

    def _qualified(
        self, event: AgentEvent, decision: CommittedDecisionV1, initial: object
    ) -> _QualifiedPreview:
        try:
            return qualify_twice(self._preview, decision, event, initial)
        except RequalifyFailed as failed:
            raise _IntentHold(R18IntentReason[failed.reason.name]) from None

    def _active_commitment(self, q: _QualifiedPreview) -> tuple[str, ...]:
        """§3-2: ACTIVE and before ``deadline - DEADLINE_MARGIN``; its scope."""

        try:
            return active_commitment(self._preview, q, self._clock)
        except RequalifyFailed as failed:
            raise _IntentHold(R18IntentReason[failed.reason.name]) from None

    def _window_end(self, q: _QualifiedPreview) -> str:
        """IP-17a §3a: ``min(now + window, deadline - DEADLINE_MARGIN)``,
        floored to the second; shorter than ``MIN_APPROVAL_WINDOW`` is
        ``DEADLINE_ELAPSED``."""

        try:
            deadline = commitment_deadline(self._preview, q)
        except RequalifyFailed as failed:
            raise _IntentHold(R18IntentReason[failed.reason.name]) from None
        now = self._clock()
        end = now + self._window
        if deadline is not None:
            end = min(end, deadline - DEADLINE_MARGIN)
        end = end.astimezone(timezone.utc).replace(microsecond=0)
        if end - now < MIN_APPROVAL_WINDOW:
            raise _IntentHold(R18IntentReason.DEADLINE_ELAPSED)
        return end.strftime("%Y-%m-%dT%H:%M:%SZ")

    def _identity(
        self, q: _QualifiedPreview, obligation_scope: tuple[str, ...]
    ) -> ObligationIdentity:
        """§3a through the shared helper (the Decision issuer's ledger reader
        computes the same key with it, IP-16b)."""

        try:
            return identity_for_source(
                self._target_path.parent,
                source=self._preview._source,
                subject_id=self._store.meta().subject_id,
                q=q,
                scope=obligation_scope,
            )
        except ObligationSecretError:
            raise _IntentHold(R18IntentReason.OBLIGATION_SECRET_UNAVAILABLE) from None
        except ObligationSourceUnavailable:
            raise _IntentHold(R18IntentReason.SOURCE_UNAVAILABLE) from None

    def _target(self, group_key: str) -> str:
        """§3-6: read-only; a missing target is never created."""

        try:
            reader = TargetReader.open(self._target_path)
        except (TargetError, OSError, ValueError):
            raise _IntentHold(R18IntentReason.TARGET_UNAVAILABLE) from None
        try:
            status = reader.status(group_key)
        except (TargetError, OSError, ValueError):
            raise _IntentHold(R18IntentReason.TARGET_UNAVAILABLE) from None
        finally:
            reader.close()
        if status.meta.target_namespace != TARGET_NAMESPACE:
            raise _IntentHold(R18IntentReason.TARGET_UNAVAILABLE)
        if status.artifact is not None or status.forward is not None or status.inverse is not None:
            raise _IntentHold(R18IntentReason.PRIOR_OBLIGATION)
        return status.meta.target_instance_id

    @staticmethod
    def _intent(
        event: AgentEvent,
        scope: str,
        incarnation: str,
        decision: CommittedDecisionV1,
        depth: int,
        q: _QualifiedPreview,
        identity: ObligationIdentity,
        gates: tuple[GateResult, ...],
        target_instance_id: str,
        window_end: str,
    ) -> ActionIntentV1:
        assert q.commitment_id is not None
        assert q.commitment_revision is not None and q.commitment_digest is not None
        by_role = {
            SourceRole.COMMITMENT: (q.commitment_id, q.commitment_revision, q.commitment_digest),
            SourceRole.CONTEXT: (q.context_id, q.context_last_modified_revision, _context_digest(q)),
            SourceRole.DECISION: (decision.decision_id, depth, decision.content_digest),
            SourceRole.EXPERIENCE: (q.experience_id, q.experience_revision, q.experience_digest),
            SourceRole.GOAL: (q.goal_id, q.goal_revision, q.goal_digest),
            SourceRole.VALUE: (q.value_id, q.value_revision, q.value_digest),
        }
        selected = decision.decision.selected
        try:
            artifact = ReviewArtifactV1(
                TARGET_NAMESPACE,
                identity.group_key,
                tuple(ReviewSourceRef(role, *by_role[role]) for role in sorted(SourceRole)),
                gates,
            )
            return ActionIntentV1(
                lineage_scope=scope,
                incarnation_id=incarnation,
                admitting_event_id=event.event_id,
                admitting_sequence=_sequence(event),
                decision_id=decision.decision_id,
                decision_committed_digest=decision.content_digest,
                policy_kind=decision.policy_kind,
                policy_row=decision.policy_row,
                selected_candidate_id=selected.candidate_id,
                selected_candidate_digest=selected.candidate_digest,
                goal_origin=GoalOrigin.USER_REQUEST,
                target_instance_id=target_instance_id,
                target_expected_revision=_TARGET_EXPECTED_REVISION,
                artifact=artifact,
                approval_window_end_utc=window_end,
            )
        except (TypeError, ValueError):
            raise _IntentHold(R18IntentReason.SOURCE_UNAVAILABLE) from None

    def _check_keys(
        self,
        classified: ReaderClassification,
        operation: HistoryRecordOperationV1,
        intent: ActionIntentV1,
    ) -> None:
        """§3-7: read-only, before PREPARE."""

        claim = f"claim:{intent.forward_key}"
        keys = operation.reserved_keys
        live = self._store.live_keys((*keys, claim))
        if claim in live:
            raise _IntentHold(R18IntentReason.PRIOR_OBLIGATION)
        if live:
            other = _intent_for(
                classified,
                lambda adopted: adopted.obligation_group_key == intent.obligation_group_key,
            )
            raise _IntentHold(
                R18IntentReason.IN_FLIGHT if other is None else R18IntentReason.PRIOR_OBLIGATION
            )

    # ------------------------------------------------------------ commit

    def _validate_commit(self, event: AgentEvent) -> None:
        """After this event's PREPARE; any failure is a runtime fail-stop."""

        from suzka.r16.history_adoption import current_incarnation

        pending = self._pending
        if event.event_type is not AgentEventType.R18_ACTION_INTENT_ISSUE:
            if pending is not None:
                raise _IntentCommitRefused("stale Intent issue pending")
            return
        if pending is None:
            return
        if pending.event != event:
            raise _IntentCommitRefused("unbound Intent issue pending")
        if self._preview._state_snapshot() != pending.initial_state:
            raise _IntentCommitRefused("Intent state changed before WAL")
        if current_incarnation(self._journal, self._store) != pending.incarnation:
            raise _IntentCommitRefused("Intent incarnation changed before WAL")
        if self._decision._snapshot()[0] != pending.snapshot_hash:
            raise _IntentCommitRefused("Intent snapshot changed before WAL")
        rows = self._store.operations(pending.incarnation)
        own = [row for row in rows if row.transaction_id == pending.transaction_id]
        others = tuple(row for row in rows if row.transaction_id != pending.transaction_id)
        if (
            len(own) != 1
            or own[0].phase is not HistoryPhase.PREPARED
            or own[0].operation_kind is not HistoryOperationKind.R18_INTENT_ISSUE
            or own[0].pending_bytes != pending.operation.canonical_bytes()
            or own[0].operation_digest != pending.operation.operation_digest
            or operations_digest(others) != pending.rows_digest
        ):
            raise _IntentCommitRefused("SubjectHistory changed before WAL")

    def _clear(self) -> None:
        self._pending = None


# ------------------------------------------------------------------- values


def _blocked_reason(reason: HeadBlockReason | None) -> R18IntentReason:
    if reason in _CONFLICT_HEADS:
        return R18IntentReason.HEAD_CONFLICT
    return R18IntentReason.HEAD_UNAVAILABLE


def _intent_for(
    classified: ReaderClassification, match: Callable[[ActionIntentV1], bool]
) -> ActionIntentV1 | None:
    """An adopted forward Intent of this incarnation that matches."""

    for adopted in classified.records:
        record = adopted.record
        if (
            adopted.record_kind is RecordKind.ACTION_INTENT
            and type(record) is ActionIntentV1
            and match(record)
        ):
            return record
    return None


__all__ = [
    "DEFAULT_APPROVAL_WINDOW_SECONDS",
    "MIN_APPROVAL_WINDOW",
    "IntentIssueStatus",
    "R18IntentReason",
    "ReviewArtifactIntentIssuerV1",
]
