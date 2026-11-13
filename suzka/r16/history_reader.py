"""R16 U4-4: SubjectHistory lineage head reader and public-safe projection.

#376 §C, D1-4, D2-2(B) and D2-5.  A FINALIZED SQLite row is *not* by itself
an adopted Decision.  From one Journal inspection the reader first checks the
whole current incarnation, in both directions:

- no open R07 gate clear or recovery;
- every current-incarnation R07 transaction that requires ``subject.history``
  maps one-to-one to an operation row of this incarnation (committed ->
  FINALIZED, aborted -> ABORTED, open -> unresolved), and every row maps back
  to a retained R07 transaction.  A missing, extra or mismatched row blocks
  *every* scope: the scope of a missing row is unknowable, so there is no
  fallback to an older head.

A node of the requested scope then counts only when its transaction is
COMPLETED/RECONCILED with a ``subject.history`` FINALIZED/ALREADY_CONSISTENT
outcome, the event has exactly one committed terminal matching the stored
postcommit receipt (COMPLETED with the same snapshot and WAL identity, or a
startup RECOVERY_CLASSIFIED COMMITTED_BEFORE_CRASH with the same snapshot),
and the stored row columns and bytes decode to the same CommittedDecisionV1.

The head is the tip of the single predecessor chain from GENESIS; a gap,
duplicate child or digest mismatch blocks the scope.  This is historical
existence only (#376 D2-3): R16/R18 owners separately requalify current
Goal/Value/Context before any new action.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from suzka.r16.contracts import CandidateKind
from suzka.r16.history_adoption import resolve_incarnation
from suzka.r16.history_codec import (
    GENESIS,
    SUBJECT_HISTORY_PARTICIPANT_ID,
    CommittedDecisionV1,
    PolicyKind,
    PolicyRow,
    PostcommitReceiptV1,
    history_operation_digest,
)
from suzka.r16.history_store import (
    HistoryOperationKind,
    HistoryOperationRow,
    HistoryPhase,
    HistoryStoreError,
    SubjectHistoryStore,
)
from suzka.runtime.event_journal import (
    EventFailureCategory,
    EventJournal,
    EventJournalInspection,
    EventJournalRecord,
    EventJournalTransaction,
    EventLifecycle,
    ParticipantOutcome,
)


class HeadStatus(str, Enum):
    EMPTY = "EMPTY"
    ADOPTED = "ADOPTED"
    BLOCKED = "BLOCKED"


class HeadBlockReason(str, Enum):
    INCARNATION_UNAVAILABLE = "INCARNATION_UNAVAILABLE"
    STORE_UNAVAILABLE = "STORE_UNAVAILABLE"
    JOURNAL_UNAVAILABLE = "JOURNAL_UNAVAILABLE"
    UNRESOLVED_OPERATION = "UNRESOLVED_OPERATION"
    MISSING_JOURNAL_PROOF = "MISSING_JOURNAL_PROOF"
    DIVERGED = "DIVERGED"
    LINEAGE_GAP = "LINEAGE_GAP"
    DUPLICATE_CHILD = "DUPLICATE_CHILD"


@dataclass(frozen=True, slots=True)
class HistoryHead:
    lineage_scope: str
    status: HeadStatus
    reason: HeadBlockReason | None = None
    decision: CommittedDecisionV1 | None = None
    transaction_id: str | None = None
    depth: int = 0


class _Blocked(Exception):
    def __init__(self, reason: HeadBlockReason) -> None:
        super().__init__(reason.value)
        self.reason = reason


_ADOPTED_OUTCOMES = frozenset(
    {ParticipantOutcome.FINALIZED, ParticipantOutcome.ALREADY_CONSISTENT}
)
_EVENT_TERMINALS = frozenset(
    {EventLifecycle.COMPLETED, EventLifecycle.FAILED, EventLifecycle.RECOVERY_CLASSIFIED}
)

IncarnationResolver = Callable[[EventJournalInspection], tuple[str, int]]


def _requires_history(transaction: EventJournalTransaction) -> bool:
    return any(
        item.participant_id == SUBJECT_HISTORY_PARTICIPANT_ID
        for item in transaction.required_participants
    )


@dataclass(frozen=True, slots=True)
class _JournalView:
    committed: dict[str, EventJournalTransaction]
    aborted: dict[str, EventJournalTransaction]
    open: dict[str, EventJournalTransaction]
    event_terminals: dict[tuple[str, int], tuple[EventJournalRecord, ...]]

    @classmethod
    def of(cls, inspection: EventJournalInspection, horizon: int) -> _JournalView:
        def index(
            items: tuple[EventJournalTransaction, ...],
        ) -> dict[str, EventJournalTransaction]:
            # Only current-incarnation transactions that require history.
            return {
                item.transaction_id: item
                for item in items
                if _requires_history(item) and item.processing_sequence > horizon
            }

        terminals: dict[tuple[str, int], list[EventJournalRecord]] = {}
        for record in (*inspection.carried_records, *inspection.records):
            if (
                record.lifecycle in _EVENT_TERMINALS
                and record.event_id is not None
                and record.processing_sequence is not None
            ):
                terminals.setdefault(
                    (record.event_id, record.processing_sequence), []
                ).append(record)
        return cls(
            committed=index(
                (*inspection.completed_transactions, *inspection.reconciled_transactions)
            ),
            aborted=index(inspection.aborted_transactions),
            open=index(inspection.open_transactions),
            event_terminals={key: tuple(value) for key, value in terminals.items()},
        )

    def transaction(self, row: HistoryOperationRow) -> EventJournalTransaction | None:
        found = [
            item
            for item in (
                self.committed.get(row.transaction_id),
                self.aborted.get(row.transaction_id),
                self.open.get(row.transaction_id),
            )
            if item is not None
        ]
        if len(found) > 1:
            raise _Blocked(HeadBlockReason.DIVERGED)
        if not found:
            return None
        transaction = found[0]
        requirements = [
            item
            for item in transaction.required_participants
            if item.participant_id == SUBJECT_HISTORY_PARTICIPANT_ID
        ]
        if (
            transaction.event_id != row.event_id
            or transaction.processing_sequence != row.processing_sequence
            or transaction.kind.value != row.transaction_kind
            or len(requirements) != 1
            or requirements[0].operation_digest != row.operation_digest
        ):
            raise _Blocked(HeadBlockReason.DIVERGED)
        return transaction


class SubjectHistoryReader:
    """Read-only lineage head resolution for the current incarnation."""

    def __init__(
        self,
        journal: EventJournal,
        store: SubjectHistoryStore,
        incarnation: IncarnationResolver | None = None,
    ) -> None:
        self.journal = journal
        self.store = store
        self.incarnation: IncarnationResolver = incarnation or (
            lambda inspection: resolve_incarnation(inspection, store)
        )

    def head(self, lineage_scope: str) -> HistoryHead:
        try:
            return self._head(lineage_scope)
        except _Blocked as blocked:
            return HistoryHead(lineage_scope, HeadStatus.BLOCKED, blocked.reason)
        except HistoryStoreError:
            return HistoryHead(
                lineage_scope, HeadStatus.BLOCKED, HeadBlockReason.STORE_UNAVAILABLE
            )
        except (TypeError, ValueError):
            return HistoryHead(lineage_scope, HeadStatus.BLOCKED, HeadBlockReason.DIVERGED)

    def _head(self, lineage_scope: str) -> HistoryHead:
        try:
            inspection = self.journal.inspect()
        except Exception:
            raise _Blocked(HeadBlockReason.JOURNAL_UNAVAILABLE) from None
        # One inspection for both the incarnation and the evidence (no TOCTOU).
        try:
            incarnation, horizon = self.incarnation(inspection)
        except HistoryStoreError:
            raise
        except Exception:
            raise _Blocked(HeadBlockReason.INCARNATION_UNAVAILABLE) from None
        if inspection.open_gate_clear is not None or inspection.open_recoveries:
            raise _Blocked(HeadBlockReason.UNRESOLVED_OPERATION)
        view = _JournalView.of(inspection, horizon)
        rows = self.store.operations(incarnation)
        self._reconcile_all(rows, view)
        nodes: list[tuple[CommittedDecisionV1, str]] = []
        for row in rows:
            if (
                row.operation_kind is HistoryOperationKind.R16_DECISION_ISSUE
                and row.lineage_scope == lineage_scope
            ):
                node = self._node(incarnation, row, view)
                if node is not None:
                    nodes.append(node)
        return self._chain(lineage_scope, nodes)

    @staticmethod
    def _reconcile_all(
        rows: tuple[HistoryOperationRow, ...], view: _JournalView
    ) -> None:
        """Both directions over the whole incarnation (review P1-2)."""

        by_transaction = {row.transaction_id: row for row in rows}
        if len(by_transaction) != len(rows):
            raise _Blocked(HeadBlockReason.DIVERGED)
        for transaction_id in view.open:
            if transaction_id not in by_transaction:
                # TRANSACTION_PREPARED with no SQLite row yet: unresolved.
                raise _Blocked(HeadBlockReason.UNRESOLVED_OPERATION)
        for transaction_id in (*view.committed, *view.aborted):
            if transaction_id not in by_transaction:
                raise _Blocked(HeadBlockReason.DIVERGED)
        for row in rows:
            transaction = view.transaction(row)
            if transaction is None:
                # Every row needs retained R07 proof, tombstones included.
                raise _Blocked(HeadBlockReason.MISSING_JOURNAL_PROOF)
            if row.transaction_id in view.open:
                continue
            expected = (
                HistoryPhase.FINALIZED
                if row.transaction_id in view.committed
                else HistoryPhase.ABORTED
            )
            if row.phase is not expected:
                raise _Blocked(HeadBlockReason.DIVERGED)

    def _node(
        self, incarnation: str, row: HistoryOperationRow, view: _JournalView
    ) -> tuple[CommittedDecisionV1, str] | None:
        if row.transaction_id in view.open:
            raise _Blocked(HeadBlockReason.UNRESOLVED_OPERATION)
        if row.phase is HistoryPhase.ABORTED:
            return None
        transaction = view.committed[row.transaction_id]
        outcomes = [
            outcome
            for participant, outcome in transaction.participant_outcomes
            if participant == SUBJECT_HISTORY_PARTICIPANT_ID
        ]
        if len(outcomes) != 1 or outcomes[0] not in _ADOPTED_OUTCOMES:
            raise _Blocked(HeadBlockReason.DIVERGED)
        finalized = self.store.finalized_decision(incarnation, row.transaction_id)
        if finalized is None or row.pending_bytes != finalized.committed_bytes:
            raise _Blocked(HeadBlockReason.DIVERGED)
        decision = CommittedDecisionV1.from_json(finalized.committed_bytes)
        receipt = finalized.receipt
        if (
            decision.decision_id != finalized.decision_id
            or decision.content_digest != finalized.committed_digest
            or decision.incarnation_id != incarnation
            or history_operation_digest(decision) != row.operation_digest
            or decision.lineage_scope != row.lineage_scope
            or decision.review_intent_key != row.review_intent_key
            or decision.predecessor_key != row.predecessor_key
            or decision.issuer_event_id != row.event_id
            or decision.issuer_sequence != row.processing_sequence
            or finalized.receipt_digest != receipt.receipt_digest
            or receipt.event_id != row.event_id
            or receipt.processing_sequence != row.processing_sequence
        ):
            raise _Blocked(HeadBlockReason.DIVERGED)
        terminals = view.event_terminals.get((row.event_id, row.processing_sequence), ())
        if not terminals:
            raise _Blocked(HeadBlockReason.MISSING_JOURNAL_PROOF)
        if len(terminals) != 1 or not _terminal_matches(terminals[0], receipt):
            raise _Blocked(HeadBlockReason.DIVERGED)
        return decision, row.transaction_id

    @staticmethod
    def _chain(
        lineage_scope: str, nodes: list[tuple[CommittedDecisionV1, str]]
    ) -> HistoryHead:
        if not nodes:
            return HistoryHead(lineage_scope, HeadStatus.EMPTY)
        children: dict[str, list[tuple[CommittedDecisionV1, str]]] = {}
        for node in nodes:
            children.setdefault(node[0].predecessor_key, []).append(node)
        if any(len(items) > 1 for items in children.values()):
            raise _Blocked(HeadBlockReason.DUPLICATE_CHILD)
        key = GENESIS
        parent: CommittedDecisionV1 | None = None
        tip: tuple[CommittedDecisionV1, str] | None = None
        depth = 0
        while key in children:
            child = children[key][0]
            expected = None if parent is None else parent.content_digest
            if child[0].predecessor_digest != expected:
                raise _Blocked(HeadBlockReason.DIVERGED)
            depth += 1
            if depth > len(nodes):
                raise _Blocked(HeadBlockReason.DIVERGED)
            tip, parent, key = child, child[0], child[0].decision_id
        if depth != len(nodes) or tip is None:
            raise _Blocked(HeadBlockReason.LINEAGE_GAP)
        return HistoryHead(
            lineage_scope,
            HeadStatus.ADOPTED,
            decision=tip[0],
            transaction_id=tip[1],
            depth=depth,
        )


def _terminal_matches(record: EventJournalRecord, receipt: PostcommitReceiptV1) -> bool:
    if record.lifecycle is EventLifecycle.COMPLETED:
        return (
            record.snapshot_sequence,
            record.snapshot_hash,
            record.wal_record_id,
            record.wal_record_hash,
        ) == (
            receipt.snapshot_sequence,
            receipt.snapshot_hash,
            receipt.wal_record_id,
            receipt.wal_record_hash,
        )
    if record.lifecycle is EventLifecycle.RECOVERY_CLASSIFIED:
        # Startup classification of a pre-crash commit carries no WAL identity;
        # the stored receipt plus the R07 terminal cover that part (P1-1).
        return (
            record.failure_category is EventFailureCategory.COMMITTED_BEFORE_CRASH
            and record.snapshot_sequence == receipt.snapshot_sequence
            and record.snapshot_hash == receipt.snapshot_hash
        )
    return False  # FAILED or any other terminal is not a committed event




@dataclass(frozen=True, slots=True)
class HistoryHeadProjection:
    """Public-safe view: identities, digests and closed codes only.

    It never carries policy-input bytes, plan steps or candidate evidence;
    source references are digest-bound IDs, not fetched contents.
    """

    lineage_scope: str
    status: HeadStatus
    reason: HeadBlockReason | None
    decision_id: str | None
    depth: int
    policy_kind: PolicyKind | None
    policy_row: PolicyRow | None
    decision_reason: str | None
    selected_kind: CandidateKind | None
    source_refs: tuple[tuple[str, str, str, int, str], ...]

    def canonical_value(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "decision_reason": self.decision_reason,
            "depth": self.depth,
            "lineage_scope": self.lineage_scope,
            "policy_kind": None if self.policy_kind is None else self.policy_kind.value,
            "policy_row": None if self.policy_row is None else self.policy_row.value,
            "reason": None if self.reason is None else self.reason.value,
            "selected_kind": None if self.selected_kind is None else self.selected_kind.value,
            "source_refs": [list(item) for item in self.source_refs],
            "status": self.status.value,
        }


def project_head(head: HistoryHead) -> HistoryHeadProjection:
    decision = head.decision if head.status is HeadStatus.ADOPTED else None
    if decision is None:
        return HistoryHeadProjection(
            head.lineage_scope, head.status, head.reason, None, 0, None, None, None, None, ()
        )
    record = decision.decision
    selected = [
        item for item in record.candidates if item.candidate_id == record.selected.candidate_id
    ]
    return HistoryHeadProjection(
        lineage_scope=head.lineage_scope,
        status=head.status,
        reason=None,
        decision_id=decision.decision_id,
        depth=head.depth,
        policy_kind=decision.policy_kind,
        policy_row=decision.policy_row,
        decision_reason=record.reason.value,
        selected_kind=selected[0].kind if len(selected) == 1 else None,
        source_refs=tuple(
            (item.role.value, item.kind, item.ref_id, item.revision, item.digest)
            for item in decision.source_refs
        ),
    )


__all__ = [
    "HeadBlockReason",
    "HeadStatus",
    "HistoryHead",
    "HistoryHeadProjection",
    "SubjectHistoryReader",
    "project_head",
]
