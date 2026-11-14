"""R16 IP-10: the CommittedDecision issuer (#371 D4-2/D5/D6, IP-10 D1/D2).

Runs inside the serialized PLAN_PREVIEW_REVIEW handler, before that event
opens any R07 transaction (the only place IP-8a's R15 read is allowed,
#374 D5 §3).  It is a separate owner from the non-authoritative U3-A
preview, which delegates to it only when SubjectHistory is installed.

Steps (IP-10 D1 §2 as replaced by D2):

1. qualify the sources through the preview owner (twice, equal);
2. build each SuccessorPolicyInputV2 state from its owner's result only;
3. verify the lineage head and compare its sealed input (D2 P1-1):
   identical -> the existing Decision read-only; a changed witness set
   (Commitment, interpretation, ledger, Feedback, Outcome) -> a successor;
   only the sources moved -> HOLD ``DECISION_SOURCE_DRIFT`` (#376 §C);
4. evaluate SuccessorPolicyV2; a HOLD row changes nothing;
5. a Decision row: check SubjectHistory headroom and the intent/lineage
   slots *before* PREPARE (D2 P1-3), then return a ``CoordinatedResult``
   with the ``subject.history`` participant.

``_validate_commit`` (after this event's own PREPARE) checks the unchanged
domain snapshot, the incarnation, the R06 snapshot hash and that
SubjectHistory is exactly the evaluated rows plus this transaction's PREPARED
row (D2 P1-2).  It cannot re-qualify the sources: source owners refuse to
read while this event's own R07 transaction is open.
Its failure is a runtime fail-stop, recovered at startup (IP-9 crash matrix).

Known limits: ledger/Feedback/Outcome are NONE/ABSENT (no R17/R18); HOLD is
not persisted (R19, #376 v2 §2); the R15 interpretation is sealed as its
closed enum only; the Context ref is the frame's last-modified revision, not
a digest of the Context's contents (#371 D6-3).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from hashlib import sha256
import logging
from typing import TYPE_CHECKING, Final

from suzka.r16.contracts import (
    MAX_JSON_BYTES,
    SCHEMA_VERSION,
    UNCERTAINTY_SCALE,
    CandidateKind,
    CandidateSelection,
    DecisionRecord,
    GoalEvidenceRef,
    HoldAssessment,
    PlanCandidate,
    PlanDAG,
    PlanStep,
    ReasonCode,
    StepKind,
    ValueEvidenceRef,
    checksum,
)
from suzka.r16.history_adoption import adopted_history_baseline, baseline_digest
from suzka.r16.history_attestation_digest import operations_digest
from suzka.r16.history_codec import (
    GENESIS,
    CommittedDecisionV1,
    HistorySourceRef,
    SourceRole,
    decision_id_for,
    policy_input_digest,
)
from suzka.r16.history_participant import (
    JournalCommitProofSource,
    SubjectHistoryParticipant,
)
from suzka.r16.history_reader import (
    HeadBlockReason,
    HeadStatus,
    HistoryHead,
    ReaderTransient,
    SubjectHistoryReader,
)
from suzka.r16.history_store import (
    HistoryOperationKind,
    HistoryPhase,
    HistoryStoreError,
    SubjectHistoryStore,
)
from suzka.r16.outcomes import R16OwnerReason
from suzka.r16.successor_policy import (
    NO_PREDECESSOR_HORIZON_KIND,
    POLICY_KIND,
    PREDECESSOR_DECISION_KIND,
    CommitmentState,
    FeedbackState,
    Interpretation,
    InterpretationStage,
    LedgerState,
    LineageRef,
    LineageState,
    PolicyOutputKind,
    PolicyRef,
    PredecessorState,
    RequiredSourceState,
    SuccessorPolicyInputV2,
    SuccessorPolicyResult,
    SuccessorRow,
    evaluate,
    interpretation_from_r15,
    utc_timestamp,
)
from suzka.runtime.agent_runtime import AgentEvent
from suzka.runtime.agent_state import AgentStateLoadError, AgentStateStore
from suzka.runtime.event_journal import (
    EventJournal,
    EventJournalInspection,
    TransactionKind,
)
from suzka.runtime.state_recovery import StateRecoveryCoordinator
from suzka.runtime.transaction_coordinator import (
    CoordinatedResult,
    TransactionCoordinator,
)

if TYPE_CHECKING:
    from suzka.r16.plan_preview import _PlanPreviewOwner, _QualifiedPreview

LOGGER = logging.getLogger(__name__)

_SCOPE_DOMAIN: Final = b"PROJECT-SUZKA:R16:REVIEW-LINEAGE-SCOPE:V1\0"
_INTENT_DOMAIN: Final = b"PROJECT-SUZKA:R16:REVIEW-INTENT:V1\0"
_HORIZON_DOMAIN: Final = b"PROJECT-SUZKA:R16:NO-PREDECESSOR-HORIZON:V1\0"
_CONTEXT_REF_DOMAIN: Final = b"PROJECT-SUZKA:R16:CONTEXT-FRAME-REF:V1\0"
_CLEARED_DOMAIN: Final = b"PROJECT-SUZKA:R16:CLEARED-RECORD-REF:V1\0"
_PLAN_DOMAIN: Final = b"PROJECT-SUZKA:R16:ISSUER-PLAN:V1\0"
# No-deadline Commitments are sealed with the latest representable instant.
NO_DEADLINE_UTC: Final = "9999-12-31T23:59:59Z"
# D2 P2-2: a Decision within this margin of its Commitment deadline is held
# before PREPARE (assumes handler -> _validate_commit is shorter; not proven).
DEADLINE_MARGIN: Final = timedelta(seconds=5)
_CONFLICT_HEADS: Final = frozenset(
    {
        HeadBlockReason.DIVERGED,
        HeadBlockReason.DUPLICATE_CHILD,
        HeadBlockReason.LINEAGE_GAP,
        HeadBlockReason.POISONED,
    }
)
_PLAN_A: Final = "source_first"
_PLAN_B: Final = "responsibility_review"
_VERIFY: Final = "verify_goal_source"
_COMPARE: Final = "compare_review_constraints"
_ASSESS: Final = "assess_review_readiness"

InterpretationRead = Callable[[AgentEvent, str, int], object]


class DecisionIssueStatus(str, Enum):
    ISSUED = "decision_issued"
    EXISTING = "decision_existing"
    HOLD = "hold"


@dataclass(frozen=True, slots=True)
class _DecisionIssueResult:
    """The PLAN_PREVIEW_REVIEW value when the issuer is installed.

    Public-safe: a status, the Decision ID, the policy row and closed reason
    codes only.  ``decision`` is held privately for in-process inspection.
    """

    status: DecisionIssueStatus
    decision_id: str | None
    row: SuccessorRow | None
    hold: HoldAssessment | None
    owner_reason: R16OwnerReason | None
    decision: CommittedDecisionV1 | None = None

    def __post_init__(self) -> None:
        if type(self.status) is not DecisionIssueStatus:
            raise TypeError("issue status must be exact")
        held = self.status is DecisionIssueStatus.HOLD
        if held != (self.decision_id is None) or held != (self.decision is None):
            raise ValueError("only a non-HOLD result carries a Decision")
        if held:
            # Either a policy HOLD row, or an issuer refusal before policy.
            by_policy = self.row is not None and self.hold is not None
            by_issuer = self.owner_reason is not None
            if by_policy == by_issuer or (by_issuer and (self.row or self.hold)):
                raise ValueError("a HOLD carries a row and assessment, or a reason")
        elif self.hold is not None or self.owner_reason is not None or self.row is None:
            raise ValueError("an issued Decision carries its row and no HOLD")

    @property
    def is_authoritative(self) -> bool:
        return self.status is not DecisionIssueStatus.HOLD


@dataclass(frozen=True, slots=True)
class _PendingIssue:
    event: AgentEvent
    qualification: _QualifiedPreview
    incarnation: str
    snapshot_hash: str
    rows_digest: tuple[int, str]
    transaction_id: str
    committed: CommittedDecisionV1
    initial_state: object


class _IssuerHold(Exception):
    def __init__(self, reason: R16OwnerReason) -> None:
        super().__init__(reason.value)
        self.reason = reason


def lineage_scope_for(experience_id: str) -> str:
    """One review lineage per committed Experience X (66 characters)."""

    return "x-" + sha256(_SCOPE_DOMAIN + experience_id.encode("ascii")).hexdigest()


def _ceil_utc(moment: datetime) -> str:
    """Seal a deadline rounded *up* to the second (D2 P2-2)."""

    moment = moment.astimezone(timezone.utc)
    if moment.microsecond:
        moment = moment.replace(microsecond=0) + timedelta(seconds=1)
    return utc_timestamp(moment)


def _witness(value: SuccessorPolicyInputV2) -> tuple[object, ...]:
    """IP-10 D2 P1-1: the policy witness set."""

    return (
        value.commitment,
        value.commitment_ref,
        value.commitment_deadline_utc,
        value.interpretation,
        value.interpretation_stage,
        value.ledger,
        value.ledger_refs,
        value.inverse_authorization,
        value.feedback,
        value.feedback_ref,
        value.verified_outcome_ref,
    )


class _DecisionIssuer:
    """Source-admitted CommittedDecision issuer (no Tool, no WRITE)."""

    def __init__(
        self,
        preview: _PlanPreviewOwner,
        store: SubjectHistoryStore,
        journal: EventJournal,
        state_recovery: StateRecoveryCoordinator,
        state_store: AgentStateStore,
        interpretation: InterpretationRead,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        from suzka.r16.plan_preview import _PlanPreviewOwner

        if (
            type(preview) is not _PlanPreviewOwner
            or type(store) is not SubjectHistoryStore
            or preview._journal is not journal
            or not callable(interpretation)
        ):
            raise TypeError("unqualified Decision issuer binding")
        self._preview = preview
        self._store = store
        self._journal = journal
        self._recovery = state_recovery
        self._state_store = state_store
        self._interpretation = interpretation
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._pending: _PendingIssue | None = None

    # ------------------------------------------------------------ handler

    def _issue(self, event: AgentEvent, goal_id: str) -> object:
        from suzka.r16.plan_preview import _PlanSourceConflict, _PlanSourceUnavailable

        self._pending = None
        initial = self._preview._state_snapshot()
        state: RequiredSourceState | None
        try:
            before = self._preview._qualify(goal_id, event)
            after = self._preview._qualify(goal_id, event)
            if after != before or self._preview._state_snapshot() != initial:
                raise _PlanSourceUnavailable("Decision source changed")
        except _PlanSourceConflict:
            state = RequiredSourceState.CONFLICT
        except _PlanSourceUnavailable:
            state = RequiredSourceState.UNAVAILABLE
        else:
            state = None
        try:
            if state is not None:
                return self._unqualified(event, state)
            return self._issue_qualified(event, before, initial)
        except (HistoryStoreError, AgentStateLoadError):
            # Review P2-3: a temporary store/state read failure is a HOLD.
            hold = _IssuerHold(R16OwnerReason.SOURCE_UNAVAILABLE)
            LOGGER.info("R16 issuer result=hold reason=%s", hold.reason.value)
            return _DecisionIssueResult(
                DecisionIssueStatus.HOLD, None, None, None, hold.reason
            )
        except _IssuerHold as hold:
            LOGGER.info("R16 issuer result=hold reason=%s", hold.reason.value)
            return _DecisionIssueResult(
                DecisionIssueStatus.HOLD, None, None, None, hold.reason
            )

    def _unqualified(
        self, event: AgentEvent, state: RequiredSourceState
    ) -> _DecisionIssueResult:
        """Sources did not qualify: still a policy row (0 or 1), no refs."""

        snapshot_hash, as_of = self._snapshot()
        value = SuccessorPolicyInputV2(
            required_source=state,
            source_refs=(),
            predecessor=PredecessorState.UNAVAILABLE,
            predecessor_ref=None,
            lineage=LineageState.UNAVAILABLE,
            lineage_ref=None,
            commitment=CommitmentState.ABSENT,
            commitment_ref=None,
            commitment_deadline_utc=None,
            ledger=LedgerState.NONE,
            ledger_refs=(),
            inverse_authorization=None,
            feedback=FeedbackState.ABSENT,
            feedback_ref=None,
            verified_outcome_ref=None,
            interpretation=Interpretation.UNAVAILABLE,
            interpretation_stage=InterpretationStage.NONE,
            as_of_r07_sequence=as_of,
            as_of_snapshot_hash=snapshot_hash,
            event_id=event.event_id,
            event_sequence=_sequence(event),
            now_utc=utc_timestamp(event.requested_at),
        )
        return self._hold_result(evaluate(value))

    @staticmethod
    def _hold_result(result: SuccessorPolicyResult) -> _DecisionIssueResult:
        LOGGER.info(
            "R16 issuer result=hold row=%s reason=%s",
            result.row.value,
            None if result.hold_reason is None else result.hold_reason.value,
        )
        return _DecisionIssueResult(
            DecisionIssueStatus.HOLD, None, result.row, result.hold_assessment(), None
        )

    def _snapshot(self) -> tuple[str, int]:
        """The committed R06 snapshot_before (D2 P2-1)."""

        snapshot = self._state_store.load()
        return (
            self._state_store.snapshot_hash(snapshot),
            snapshot.last_processed_event_sequence,
        )

    def _issue_qualified(
        self, event: AgentEvent, q: _QualifiedPreview, initial: object
    ) -> object:
        scope = lineage_scope_for(q.experience_id)
        snapshot_hash, as_of = self._snapshot()
        now_utc = utc_timestamp(event.requested_at)
        inspection = self._journal.inspect()
        lineage, lineage_ref, incarnation, head, attested = self._lineage(
            inspection, scope
        )
        commitment, commitment_ref, deadline = self._commitment(q)
        interpretation, stage = self._read_interpretation(event, q)
        predecessor, predecessor_ref = self._predecessor(
            head, incarnation, scope, as_of, snapshot_hash, attested
        )
        value = SuccessorPolicyInputV2(
            required_source=RequiredSourceState.QUALIFIED,
            source_refs=_required_refs(q),
            predecessor=predecessor,
            predecessor_ref=predecessor_ref,
            lineage=lineage,
            lineage_ref=lineage_ref,
            commitment=commitment,
            commitment_ref=commitment_ref,
            commitment_deadline_utc=deadline,
            ledger=LedgerState.NONE,
            ledger_refs=(),
            inverse_authorization=None,
            feedback=FeedbackState.ABSENT,
            feedback_ref=None,
            verified_outcome_ref=None,
            interpretation=interpretation,
            interpretation_stage=stage,
            as_of_r07_sequence=as_of,
            as_of_snapshot_hash=snapshot_hash,
            event_id=event.event_id,
            event_sequence=_sequence(event),
            now_utc=now_utc,
        )
        result = evaluate(value)
        if (
            predecessor is PredecessorState.QUALIFIED
            and head is not None
            and head.decision is not None
        ):
            existing = self._compare_head(head.decision, value, result)
            if existing is not None:
                return existing
        if result.is_hold:
            return self._hold_result(result)
        assert incarnation is not None
        committed = self._committed(event, q, value, result, incarnation, scope, head)
        self._check_slots(committed, incarnation)
        self._check_deadline(q)
        # The sources must not move while we built the Decision.
        if self._preview._qualify(q.goal_id, event) != q or (
            self._preview._state_snapshot() != initial
        ):
            raise _IssuerHold(R16OwnerReason.SOURCE_UNAVAILABLE)
        transaction_id = TransactionCoordinator.derive_transaction_id(
            event, TransactionKind.EVENT_MUTATION
        )
        self._pending = _PendingIssue(
            event,
            q,
            incarnation,
            snapshot_hash,
            operations_digest(self._store.operations(incarnation)),
            transaction_id,
            committed,
            initial,
        )
        LOGGER.info("R16 issuer result=issued row=%s", result.row.value)
        participant = SubjectHistoryParticipant(
            self._store,
            incarnation,
            committed,
            JournalCommitProofSource(self._journal, self._recovery),
        )
        return CoordinatedResult(
            _DecisionIssueResult(
                DecisionIssueStatus.ISSUED,
                committed.decision_id,
                result.row,
                None,
                None,
                committed,
            ),
            (participant,),
        )

    # ------------------------------------------------------------ adapters

    def _lineage(
        self, inspection: EventJournalInspection, scope: str
    ) -> tuple[
        LineageState, LineageRef | None, str | None, HistoryHead | None, int
    ]:
        reader = SubjectHistoryReader(self._journal, self._store)
        try:
            classified = reader.classify((scope,), inspection=inspection)
        except ReaderTransient:
            return LineageState.UNAVAILABLE, None, None, None, 0
        adopted = adopted_history_baseline(inspection)
        if adopted is None or inspection.journal_lineage_id is None:
            return LineageState.UNAVAILABLE, None, None, None, 0
        clear = inspection.terminal_gate_clear
        cleared = (
            None
            if clear is None or classified.horizon == 0
            else checksum(
                _CLEARED_DOMAIN,
                [
                    clear.baseline_id,
                    clear.recovery_id,
                    clear.processing_high_water,
                    clear.wal_record_hash,
                ],
            )
        )
        ref = LineageRef(
            inspection.journal_lineage_id,
            adopted.baseline_id,
            baseline_digest(adopted),
            cleared,
            classified.incarnation,
        )
        if classified.global_block is not None:
            head = HistoryHead(scope, HeadStatus.BLOCKED, classified.global_block)
        else:
            head = classified.heads[scope]
        return (
            LineageState.CONSISTENT,
            ref,
            classified.incarnation,
            head,
            classified.attested_high_water,
        )

    def _commitment(
        self, q: _QualifiedPreview
    ) -> tuple[CommitmentState, PolicyRef | None, str | None]:
        if q.commitment_id is None:
            return CommitmentState.ABSENT, None, None
        assert q.commitment_revision is not None and q.commitment_digest is not None
        record = self._preview._loop._commitment_system.get(q.commitment_id)
        if record is None:
            raise _IssuerHold(R16OwnerReason.SOURCE_UNAVAILABLE)
        at = record.deadline.at
        deadline = NO_DEADLINE_UTC if at is None else _ceil_utc(at)
        return (
            CommitmentState.ACTIVE,
            PolicyRef("commitment", q.commitment_id, q.commitment_revision, q.commitment_digest),
            deadline,
        )

    def _read_interpretation(
        self, event: AgentEvent, q: _QualifiedPreview
    ) -> tuple[Interpretation, InterpretationStage]:
        try:
            read = self._interpretation(
                event, q.experience_event_id, q.experience_event_sequence
            )
            return interpretation_from_r15(
                getattr(read, "outcome", None),
                getattr(read, "stage", None),
                getattr(read, "status", None),
            )
        except Exception:
            return Interpretation.UNAVAILABLE, InterpretationStage.NONE

    def _predecessor(
        self,
        head: HistoryHead | None,
        incarnation: str | None,
        scope: str,
        as_of: int,
        snapshot_hash: str,
        attested: int,
    ) -> tuple[PredecessorState, PolicyRef | None]:
        if head is None or incarnation is None:
            return PredecessorState.UNAVAILABLE, None
        if head.status is HeadStatus.EMPTY:
            horizon = checksum(
                _HORIZON_DOMAIN,
                [incarnation, scope, as_of, snapshot_hash, attested, "EMPTY"],
            )
            return PredecessorState.ABSENT_INITIAL, PolicyRef(
                NO_PREDECESSOR_HORIZON_KIND, scope, 0, horizon
            )
        if head.status is HeadStatus.BLOCKED:
            if head.reason in _CONFLICT_HEADS:
                return PredecessorState.CONFLICT, None
            return PredecessorState.UNAVAILABLE, None
        decision = head.decision
        assert decision is not None and head.transaction_id is not None
        if not self._head_verified(decision, incarnation, head.transaction_id):
            return PredecessorState.CONFLICT, None
        return PredecessorState.QUALIFIED, PolicyRef(
            PREDECESSOR_DECISION_KIND,
            decision.decision_id,
            head.depth,
            decision.content_digest,
        )

    def _head_verified(
        self, decision: CommittedDecisionV1, incarnation: str, transaction_id: str
    ) -> bool:
        """D2 P2-5: sealed replay, row/selection and as_of before use."""

        try:
            sealed = SuccessorPolicyInputV2.from_bytes(decision.policy_input_bytes)
            replay = evaluate(sealed)
            finalized = self._store.finalized_decision(incarnation, transaction_id)
        except (ValueError, HistoryStoreError):
            return False
        if finalized is None or replay.is_hold:
            return False
        selected = {
            item.candidate_id: item for item in decision.decision.candidates
        }.get(decision.decision.selected.candidate_id)
        expected_kind = {
            PolicyOutputKind.PLAN_A: (CandidateKind.PLAN, "candidate-a"),
            PolicyOutputKind.PLAN_B: (CandidateKind.PLAN, "candidate-b"),
            PolicyOutputKind.NO_OP: (CandidateKind.NO_OP, "candidate-noop"),
        }[replay.output]
        return (
            decision.policy_kind is POLICY_KIND
            and replay.history_row is decision.policy_row
            and selected is not None
            and (selected.kind, selected.candidate_id) == expected_kind
            and sealed.event_id == decision.issuer_event_id
            and sealed.event_sequence == decision.issuer_sequence
            and sealed.as_of_snapshot_hash
            == finalized.receipt.prepared_state_hash_before
        )

    def _compare_head(
        self,
        head: CommittedDecisionV1,
        value: SuccessorPolicyInputV2,
        result: SuccessorPolicyResult,
    ) -> _DecisionIssueResult | None:
        """D2 P1-1: identical -> existing; same witness, moved sources -> HOLD."""

        sealed = SuccessorPolicyInputV2.from_bytes(head.policy_input_bytes)
        if sealed.evidence_fingerprint() == value.evidence_fingerprint():
            if result.is_hold or result.history_row is not head.policy_row:
                # Same evidence, different row: only the predecessor/as_of
                # changed the table's answer, so the head is not reproducible
                # and no Decision is issued (reported as source drift).
                raise _IssuerHold(R16OwnerReason.DECISION_SOURCE_DRIFT)
            LOGGER.info("R16 issuer result=existing row=%s", result.row.value)
            return _DecisionIssueResult(
                DecisionIssueStatus.EXISTING,
                head.decision_id,
                result.row,
                None,
                None,
                head,
            )
        if _witness(sealed) == _witness(value):
            raise _IssuerHold(R16OwnerReason.DECISION_SOURCE_DRIFT)
        return None

    # ------------------------------------------------------------ Decision

    def _committed(
        self,
        event: AgentEvent,
        q: _QualifiedPreview,
        value: SuccessorPolicyInputV2,
        result: SuccessorPolicyResult,
        incarnation: str,
        scope: str,
        head: HistoryHead | None,
    ) -> CommittedDecisionV1:
        predecessor = (
            head.decision
            if head is not None and head.status is HeadStatus.ADOPTED
            else None
        )
        predecessor_key = GENESIS if predecessor is None else predecessor.decision_id
        policy_bytes = value.canonical_bytes()
        intent = checksum(
            _INTENT_DOMAIN,
            [incarnation, predecessor_key, value.evidence_fingerprint(), POLICY_KIND.value],
        )
        identity = decision_id_for(
            incarnation,
            event.event_id,
            _sequence(event),
            intent,
            policy_input_digest(policy_bytes),
        )
        record = _decision_record(identity, q, result)
        return CommittedDecisionV1(
            incarnation_id=incarnation,
            decision_id=identity,
            issuer_event_id=event.event_id,
            issuer_sequence=_sequence(event),
            lineage_scope=scope,
            review_intent_key=intent,
            predecessor_key=predecessor_key,
            predecessor_digest=None if predecessor is None else predecessor.content_digest,
            decision=record,
            policy_kind=POLICY_KIND,
            policy_row=result.history_row,
            policy_input_bytes=policy_bytes,
            source_refs=_history_refs(q),
        )

    def _check_slots(self, committed: CommittedDecisionV1, incarnation: str) -> None:
        """D2 P1-3: capacity and UNIQUE slots, read-only, before PREPARE."""

        try:
            self._store.check_headroom(len(committed.canonical_bytes()))
        except HistoryStoreError:
            raise _IssuerHold(R16OwnerReason.HISTORY_CAPACITY) from None
        rows = self._store.operations(incarnation)
        for row in rows:
            if (
                row.operation_kind is not HistoryOperationKind.R16_DECISION_ISSUE
                or row.phase is HistoryPhase.ABORTED
            ):
                continue
            if row.review_intent_key == committed.review_intent_key or (
                row.lineage_scope == committed.lineage_scope
                and row.predecessor_key == committed.predecessor_key
            ):
                raise _IssuerHold(R16OwnerReason.DECISION_SLOT_OCCUPIED)

    def _check_deadline(self, q: _QualifiedPreview) -> None:
        if q.commitment_id is None:
            return
        record = self._preview._loop._commitment_system.get(q.commitment_id)
        at = None if record is None else record.deadline.at
        if at is not None and at - self._clock() < DEADLINE_MARGIN:
            # Not yet elapsed, but too close to commit safely (D2 P2-2).
            raise _IssuerHold(R16OwnerReason.DEADLINE_ELAPSED)

    # ------------------------------------------------------------ commit

    def _validate_commit(self, event: AgentEvent) -> None:
        """After this event's PREPARE; any failure is a runtime fail-stop."""

        from suzka.r16.history_adoption import current_incarnation
        from suzka.r16.plan_preview import _PlanSourceUnavailable

        pending = self._pending
        if pending is None:
            return
        if pending.event != event:
            raise _PlanSourceUnavailable("unbound Decision issue pending")
        if self._preview._state_snapshot() != pending.initial_state:
            raise _PlanSourceUnavailable("Decision state changed before WAL")
        # Source owners refuse to read while any R07 transaction is open, and
        # this event's own transaction is open here.  The sources were
        # qualified twice in the handler; the serialized runtime admits no
        # other event in between, which the unchanged domain snapshot above,
        # the R06 snapshot hash and the SubjectHistory rows below confirm.
        if current_incarnation(self._journal, self._store) != pending.incarnation:
            raise _PlanSourceUnavailable("Decision incarnation changed before WAL")
        # Precondition check only: no R06 commit can occur between the handler
        # and this checkpoint in the serialized runtime (review P1-2 note).
        if self._snapshot()[0] != pending.snapshot_hash:
            raise _PlanSourceUnavailable("Decision snapshot changed before WAL")
        rows = self._store.operations(pending.incarnation)
        own = [row for row in rows if row.transaction_id == pending.transaction_id]
        others = tuple(row for row in rows if row.transaction_id != pending.transaction_id)
        if (
            len(own) != 1
            or own[0].phase is not HistoryPhase.PREPARED
            or own[0].review_intent_key != pending.committed.review_intent_key
            or own[0].lineage_scope != pending.committed.lineage_scope
            or own[0].predecessor_key != pending.committed.predecessor_key
            or operations_digest(others) != pending.rows_digest
        ):
            raise _PlanSourceUnavailable("SubjectHistory changed before WAL")

    def _clear(self) -> None:
        self._pending = None


# ------------------------------------------------------------------- values


def _sequence(event: AgentEvent) -> int:
    sequence = event.processing_sequence
    if type(sequence) is not int or sequence < 1:
        raise ValueError("Decision issue needs a sequenced event")
    return sequence


def _context_digest(q: _QualifiedPreview) -> str:
    return checksum(_CONTEXT_REF_DOMAIN, [q.context_id, q.context_last_modified_revision])


def _required_refs(q: _QualifiedPreview) -> tuple[PolicyRef, ...]:
    return (
        PolicyRef("context", q.context_id, q.context_last_modified_revision, _context_digest(q)),
        PolicyRef("experience", q.experience_id, q.experience_revision, q.experience_digest),
        PolicyRef("goal", q.goal_id, q.goal_revision, q.goal_digest),
        PolicyRef("value", q.value_id, q.value_revision, q.value_digest),
    )


def _history_refs(q: _QualifiedPreview) -> tuple[HistorySourceRef, ...]:
    refs = [
        HistorySourceRef(SourceRole.GOAL, "goal", q.goal_id, q.goal_revision, q.goal_digest),
        HistorySourceRef(SourceRole.VALUE, "value", q.value_id, q.value_revision, q.value_digest),
        HistorySourceRef(
            SourceRole.CONTEXT,
            "context",
            q.context_id,
            q.context_last_modified_revision,
            _context_digest(q),
        ),
        HistorySourceRef(
            SourceRole.EXPERIENCE,
            "experience",
            q.experience_id,
            q.experience_revision,
            q.experience_digest,
        ),
        HistorySourceRef(
            SourceRole.MOTIVATION,
            "motivation",
            q.motivation_id,
            0,
            q.motivation_revision_digest,
        ),
    ]
    if q.commitment_id is not None:
        assert q.commitment_revision is not None and q.commitment_digest is not None
        refs.append(
            HistorySourceRef(
                SourceRole.COMMITMENT,
                "commitment",
                q.commitment_id,
                q.commitment_revision,
                q.commitment_digest,
            )
        )
    return tuple(sorted(refs, key=lambda item: item.sort_key))


def _plan(goal: GoalEvidenceRef, value: ValueEvidenceRef, branch: str) -> PlanDAG:
    stamp = sha256(
        _PLAN_DOMAIN
        + "\x00".join(
            (goal.goal_id, goal.goal_digest, value.value_id, value.value_digest, branch)
        ).encode("ascii")
    ).hexdigest()
    steps: tuple[PlanStep, ...]
    if branch == _PLAN_A:
        steps = (
            PlanStep(SCHEMA_VERSION, _VERIFY, StepKind.CHECK, ()),
            PlanStep(SCHEMA_VERSION, _ASSESS, StepKind.ASSESS, (_VERIFY,)),
        )
    else:
        steps = (
            PlanStep(SCHEMA_VERSION, _VERIFY, StepKind.CHECK, ()),
            PlanStep(SCHEMA_VERSION, _COMPARE, StepKind.COMPARE, (_VERIFY,)),
            PlanStep(SCHEMA_VERSION, _ASSESS, StepKind.ASSESS, (_COMPARE,)),
        )
    return PlanDAG(SCHEMA_VERSION, stamp, steps)


def _decision_record(
    identity: str, q: _QualifiedPreview, result: SuccessorPolicyResult
) -> DecisionRecord:
    """Two Plans and one NO_OP; selection by the policy output (D5-2)."""

    goal = GoalEvidenceRef(SCHEMA_VERSION, q.goal_id, q.goal_revision, q.goal_digest)
    value = ValueEvidenceRef(SCHEMA_VERSION, q.value_id, q.value_revision, q.value_digest)
    a = PlanCandidate(
        SCHEMA_VERSION, "candidate-a", CandidateKind.PLAN, goal, value, _plan(goal, value, _PLAN_A)
    )
    b = PlanCandidate(
        SCHEMA_VERSION, "candidate-b", CandidateKind.PLAN, goal, value, _plan(goal, value, _PLAN_B)
    )
    noop = PlanCandidate(SCHEMA_VERSION, "candidate-noop", CandidateKind.NO_OP, goal, value, None)
    selected = {
        PolicyOutputKind.PLAN_A: a,
        PolicyOutputKind.PLAN_B: b,
        PolicyOutputKind.NO_OP: noop,
    }[result.output]
    reason = (
        ReasonCode.NO_OP_PREFERRED
        if result.output is PolicyOutputKind.NO_OP
        else ReasonCode.PLAN_PREFERRED
    )
    record = DecisionRecord(
        SCHEMA_VERSION,
        identity,
        (a, b, noop),
        CandidateSelection(SCHEMA_VERSION, selected.candidate_id, selected.content_digest),
        reason,
        UNCERTAINTY_SCALE,
    )
    if len(record.canonical_bytes()) > MAX_JSON_BYTES:
        raise ValueError("Decision exceeds the U1 bound")
    return record


__all__ = [
    "DEADLINE_MARGIN",
    "NO_DEADLINE_UTC",
    "DecisionIssueStatus",
    "lineage_scope_for",
]
