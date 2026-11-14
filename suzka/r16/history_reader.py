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
from dataclasses import dataclass, replace
from enum import Enum

from suzka.r16.contracts import CandidateKind
from suzka.r16.history_adoption import resolve_incarnation
from suzka.r16.history_attestation_digest import (
    ChainNode,
    operations_digest,
    prefix_tips,
    scopes_digest,
)
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
    EventJournalHistoryAttestation,
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
    # C10: permanently blocked at compaction time and attested as such.
    POISONED = "POISONED"


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


class ReaderTransient(Exception):
    """A temporary inability to read (store/Journal/incarnation unavailable).

    Distinct from a BLOCKED classification so the C10 provider defers
    instead of poisoning (#376 C10 D3 P2-3).
    """

    def __init__(self, reason: HeadBlockReason) -> None:
        super().__init__(reason.value)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class ReaderClassification:
    """Whole-incarnation result (#376 C10 D3 §2).

    ``global_block`` comes from whole-incarnation reconciliation (no row for
    a transaction, phase mismatch, unproven tombstone, attestation mismatch,
    poisoned incarnation); ``heads`` holds every scope's own result.
    """

    incarnation: str
    horizon: int
    attested_high_water: int
    global_block: HeadBlockReason | None
    heads: dict[str, HistoryHead]
    chains: dict[str, tuple[ChainNode, ...]]
    rows: tuple[HistoryOperationRow, ...]


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
            result = self.classify((lineage_scope,))
        except ReaderTransient as transient:
            return HistoryHead(lineage_scope, HeadStatus.BLOCKED, transient.reason)
        if result.global_block is not None:
            return HistoryHead(lineage_scope, HeadStatus.BLOCKED, result.global_block)
        return result.heads[lineage_scope]

    def classify(
        self,
        scopes: tuple[str, ...] = (),
        *,
        inspection: EventJournalInspection | None = None,
    ) -> ReaderClassification:
        """Classify the whole current incarnation and the given scopes.

        Raises ``ReaderTransient`` only for temporary unavailability.
        """

        if inspection is None:
            try:
                inspection = self.journal.inspect()
            except Exception:
                raise ReaderTransient(HeadBlockReason.JOURNAL_UNAVAILABLE) from None
        # One inspection for both the incarnation and the evidence (no TOCTOU).
        try:
            incarnation, horizon = self.incarnation(inspection)
        except HistoryStoreError:
            raise ReaderTransient(HeadBlockReason.STORE_UNAVAILABLE) from None
        except Exception:
            raise ReaderTransient(HeadBlockReason.INCARNATION_UNAVAILABLE) from None
        try:
            rows = self.store.operations(incarnation)
        except HistoryStoreError:
            raise ReaderTransient(HeadBlockReason.STORE_UNAVAILABLE) from None
        attestation, high_water = _current_attestation(inspection, incarnation, horizon)
        result = ReaderClassification(
            incarnation, horizon, high_water, None, {}, {}, rows
        )
        try:
            return self._classify(
                inspection, incarnation, horizon, attestation, high_water, rows, scopes
            )
        except _Blocked as blocked:
            return replace(result, global_block=blocked.reason)
        except HistoryStoreError:
            raise ReaderTransient(HeadBlockReason.STORE_UNAVAILABLE) from None
        except (TypeError, ValueError):
            return replace(result, global_block=HeadBlockReason.DIVERGED)

    def _classify(
        self,
        inspection: EventJournalInspection,
        incarnation: str,
        horizon: int,
        attestation: EventJournalHistoryAttestation | None,
        high_water: int,
        rows: tuple[HistoryOperationRow, ...],
        requested: tuple[str, ...],
    ) -> ReaderClassification:
        if inspection.open_gate_clear is not None or inspection.open_recoveries:
            raise _Blocked(HeadBlockReason.UNRESOLVED_OPERATION)
        attested = tuple(row for row in rows if row.processing_sequence <= high_water)
        live = tuple(row for row in rows if row.processing_sequence > high_water)
        if attestation is not None:
            # C10: <= H is checked against the creation-time attestation; the
            # R07 facts it replaced were matched one to one when it was made.
            if operations_digest(attested) != (
                attestation.operations_count,
                attestation.operations_digest,
            ):
                raise _Blocked(HeadBlockReason.DIVERGED)
            if attestation.poisoned_all:
                raise _Blocked(HeadBlockReason.POISONED)
        view = _JournalView.of(inspection, max(horizon, high_water))
        self._reconcile_all(live, view)
        poisoned = frozenset(() if attestation is None else attestation.poisoned_scopes)
        scopes = sorted(
            {
                row.lineage_scope
                for row in rows
                if row.operation_kind is HistoryOperationKind.R16_DECISION_ISSUE
                and row.lineage_scope is not None
            }
            | set(requested)
        )
        heads: dict[str, HistoryHead] = {}
        chains: dict[str, tuple[ChainNode, ...]] = {}
        prefixes: dict[str, tuple[ChainNode, ...]] = {}
        for scope in scopes:
            scoped = tuple(
                row
                for row in rows
                if row.operation_kind is HistoryOperationKind.R16_DECISION_ISSUE
                and row.lineage_scope == scope
            )
            if scope in poisoned:
                heads[scope] = HistoryHead(scope, HeadStatus.BLOCKED, HeadBlockReason.POISONED)
                continue
            # The attested prefix must itself be a valid chain (global check).
            attested_nodes = [
                node
                for row in scoped
                if row.processing_sequence <= high_water
                for node in (self._internal_node(incarnation, row),)
                if node is not None
            ]
            if attested_nodes:
                prefixes[scope] = self._chain_nodes(attested_nodes)
            try:
                nodes = [
                    *attested_nodes,
                    *(
                        node
                        for row in scoped
                        if row.processing_sequence > high_water
                        for node in (self._node(incarnation, row, view),)
                        if node is not None
                    ),
                ]
                chain = self._chain_nodes(nodes)
            except _Blocked as blocked:
                heads[scope] = HistoryHead(scope, HeadStatus.BLOCKED, blocked.reason)
                continue
            chains[scope] = chain
            heads[scope] = (
                HistoryHead(
                    scope,
                    HeadStatus.ADOPTED,
                    decision=chain[-1].decision,
                    transaction_id=chain[-1].transaction_id,
                    depth=len(chain),
                )
                if chain
                else HistoryHead(scope, HeadStatus.EMPTY)
            )
        if attestation is not None and scopes_digest(
            prefix_tips(prefixes, high_water)
        ) != (attestation.scope_count, attestation.scopes_digest):
            raise _Blocked(HeadBlockReason.DIVERGED)
        return ReaderClassification(
            incarnation, horizon, high_water, None, heads, chains, rows
        )

    @staticmethod
    def _reconcile_all(
        rows: tuple[HistoryOperationRow, ...], view: _JournalView
    ) -> None:
        """Both directions over the retained part of the incarnation."""

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

    def _internal_node(
        self, incarnation: str, row: HistoryOperationRow
    ) -> ChainNode | None:
        """SQLite-internal checks only (attested rows, C10 D2 §1)."""

        if row.phase is HistoryPhase.ABORTED:
            return None
        if row.phase is not HistoryPhase.FINALIZED:
            raise _Blocked(HeadBlockReason.DIVERGED)
        node, _receipt = self._decoded(incarnation, row)
        return node

    def _decoded(
        self, incarnation: str, row: HistoryOperationRow
    ) -> tuple[ChainNode, PostcommitReceiptV1]:
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
        return ChainNode(decision, row.transaction_id, row.processing_sequence), receipt

    def _node(
        self, incarnation: str, row: HistoryOperationRow, view: _JournalView
    ) -> ChainNode | None:
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
        node, receipt = self._decoded(incarnation, row)
        terminals = view.event_terminals.get((row.event_id, row.processing_sequence), ())
        if not terminals:
            raise _Blocked(HeadBlockReason.MISSING_JOURNAL_PROOF)
        if len(terminals) != 1 or not _terminal_matches(terminals[0], receipt):
            raise _Blocked(HeadBlockReason.DIVERGED)
        return node

    @staticmethod
    def _chain_nodes(nodes: list[ChainNode]) -> tuple[ChainNode, ...]:
        if not nodes:
            return ()
        children: dict[str, list[ChainNode]] = {}
        for node in nodes:
            children.setdefault(node.decision.predecessor_key, []).append(node)
        if any(len(items) > 1 for items in children.values()):
            raise _Blocked(HeadBlockReason.DUPLICATE_CHILD)
        key = GENESIS
        parent: CommittedDecisionV1 | None = None
        chain: list[ChainNode] = []
        while key in children:
            child = children[key][0]
            expected = None if parent is None else parent.content_digest
            if child.decision.predecessor_digest != expected:
                raise _Blocked(HeadBlockReason.DIVERGED)
            chain.append(child)
            if len(chain) > len(nodes):
                raise _Blocked(HeadBlockReason.DIVERGED)
            parent, key = child.decision, child.decision.decision_id
        if len(chain) != len(nodes):
            raise _Blocked(HeadBlockReason.LINEAGE_GAP)
        return tuple(chain)


def _current_attestation(
    inspection: EventJournalInspection, incarnation: str, horizon: int
) -> tuple[EventJournalHistoryAttestation | None, int]:
    """The latest COMPACTION's attestation for this incarnation (D2 P2-2)."""

    compaction = inspection.latest_compaction
    if compaction is None or compaction.history_attestation is None:
        return None, 0
    attestation = compaction.history_attestation
    if attestation.incarnation_id != incarnation:
        return None, 0
    if attestation.horizon != horizon:
        raise _Blocked(HeadBlockReason.DIVERGED)
    return attestation, compaction.archived_processing_high_water


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
