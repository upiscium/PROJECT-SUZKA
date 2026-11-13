"""R16 U4-2: the ``subject.history`` R07 DURABLE_DOMAIN participant.

#376 D1-1/D1-3/D1-6/D2-2(A) and §C.  The participant is the only writer of
the SubjectHistoryDB.  Ordering for one operation:

1. R07 TRANSACTION_PREPARED, then ``prepare`` inserts the exact pending
   ``CommittedDecisionV1`` as PREPARED (refusing a reused transaction, a
   binding-only tombstone or a second child of the same predecessor).
2. R06 commits the event internally (StateWAL + canonical AgentState).
3. ``finalize`` first obtains a uniquely qualified ``INTERNALLY_COMMITTED``
   read-only proof for the exact *open* transaction (live finalize or
   startup roll-forward only), outside any SQLite lock, then performs one
   ``BEGIN IMMEDIATE`` FINALIZE storing the Decision, its sources and the
   postcommit receipt.  It never re-evaluates Goal/Value/Context.
4. R07 appends PARTICIPANT_FINALIZED and the transaction/event terminal.

Abort is only legal before an internal commit.  A binding-only ABORTED
tombstone is written when nothing was prepared, so the same transaction can
never be prepared later.  Any committed-but-missing, ambiguous or conflicting
evidence is ``ParticipantDivergedError`` / ``ParticipantUnavailableError``:
reconciliation fails closed, never aborts a committed Decision.
"""

from __future__ import annotations

from typing import Protocol

from suzka.runtime.event_journal import (
    AbortOutcome,
    EventJournal,
    EventJournalTransaction,
    EventLifecycle,
    ParticipantCapability,
    ParticipantOutcome,
    StartupParticipantOutcome,
)
from suzka.runtime.state_recovery import (
    InternalCommitClassification,
    StateRecoveryCoordinator,
)
from suzka.runtime.transaction_coordinator import (
    ParticipantDivergedError,
    ParticipantUnavailableError,
    TransactionBinding,
    validate_transaction_binding,
)
from suzka.r16.contracts import digest
from suzka.r16.history_codec import (
    SUBJECT_HISTORY_PARTICIPANT_ID,
    CommittedDecisionV1,
    PostcommitReceiptV1,
    history_operation_digest,
)
from suzka.r16.history_store import (
    HistoryOperationKind,
    HistoryOperationRow,
    HistoryPhase,
    HistoryStoreConflict,
    HistoryStoreCorrupt,
    HistoryStoreError,
    SubjectHistoryStore,
)


class CommitProofSource(Protocol):
    """Read-only live proof that the exact open transaction is committed."""

    def postcommit_receipt(self, binding: TransactionBinding) -> PostcommitReceiptV1: ...


class JournalCommitProofSource:
    """#376 D2-2(A): qualify only a current open R07 transaction.

    Historical reads (D2-2(B)) never call this: once later events or a WAL
    checkpoint move the tail, ``classify_transaction_commit`` correctly stops
    returning INTERNALLY_COMMITTED for an old transaction.
    """

    def __init__(
        self, journal: EventJournal, state_recovery: StateRecoveryCoordinator
    ) -> None:
        self.journal = journal
        self.state_recovery = state_recovery

    def postcommit_receipt(self, binding: TransactionBinding) -> PostcommitReceiptV1:
        try:
            inspection = self.journal.inspect()
        except Exception:
            raise ParticipantUnavailableError("Journal evidence is unavailable") from None
        matches = [
            item
            for item in inspection.open_transactions
            if item.transaction_id == binding.transaction_id
        ]
        if len(matches) != 1:
            raise ParticipantUnavailableError(
                "SubjectHistory proof requires the exact open transaction"
            )
        transaction = matches[0]
        self._require_binding(transaction, binding)
        proof = self.state_recovery.classify_transaction_commit(transaction)
        if proof.classification is InternalCommitClassification.AMBIGUOUS:
            raise ParticipantUnavailableError("Internal commit evidence is ambiguous")
        if proof.classification is not InternalCommitClassification.INTERNALLY_COMMITTED:
            raise ParticipantDivergedError("The transaction is not internally committed")
        if (
            proof.transaction_id != binding.transaction_id
            or proof.event_id != binding.event_id
            or proof.processing_sequence != binding.processing_sequence
            or proof.wal_generation_id is None
            or proof.wal_record_id is None
            or proof.wal_record_hash is None
        ):
            raise ParticipantDivergedError("Internal commit proof does not match")
        prepared = [
            record
            for record in inspection.records
            if record.lifecycle is EventLifecycle.PREPARED
            and record.event_id == binding.event_id
            and record.processing_sequence == binding.processing_sequence
        ]
        if (
            len(prepared) != 1
            or prepared[0].state_hash_before is None
            or prepared[0].state_hash_after is None
            or prepared[0].wal_generation_id != proof.wal_generation_id
        ):
            raise ParticipantDivergedError("Original PREPARED evidence does not match")
        try:
            return PostcommitReceiptV1(
                transaction_id=binding.transaction_id,
                event_id=binding.event_id,
                processing_sequence=binding.processing_sequence,
                prepared_state_hash_before=prepared[0].state_hash_before,
                prepared_state_hash_after=prepared[0].state_hash_after,
                snapshot_sequence=proof.snapshot_sequence,
                snapshot_hash=proof.snapshot_hash,
                wal_generation_id=proof.wal_generation_id,
                wal_record_id=proof.wal_record_id,
                wal_record_hash=proof.wal_record_hash,
            )
        except (TypeError, ValueError):
            raise ParticipantDivergedError("Internal commit proof is invalid") from None

    @staticmethod
    def _require_binding(
        transaction: EventJournalTransaction, binding: TransactionBinding
    ) -> None:
        requirements = [
            item
            for item in transaction.required_participants
            if item.participant_id == binding.participant_id
        ]
        if (
            transaction.event_id != binding.event_id
            or transaction.processing_sequence != binding.processing_sequence
            or transaction.kind is not binding.transaction_kind
            or len(requirements) != 1
            or requirements[0].operation_digest != binding.operation_digest
        ):
            raise ParticipantDivergedError("Journal transaction binding conflicts")


def _diverged(error: HistoryStoreError) -> Exception:
    if isinstance(error, (HistoryStoreConflict, HistoryStoreCorrupt)):
        return ParticipantDivergedError(str(error))
    return ParticipantUnavailableError(str(error))


def _require_binding(binding: TransactionBinding) -> None:
    if (
        not isinstance(binding, TransactionBinding)
        or binding.participant_id != SUBJECT_HISTORY_PARTICIPANT_ID
        or not validate_transaction_binding(binding)
    ):
        raise ParticipantDivergedError("SubjectHistory binding is invalid")
    try:
        digest(binding.operation_digest, "operation_digest")
    except ValueError:
        raise ParticipantDivergedError("SubjectHistory binding is invalid") from None


def _row_matches_binding(row: HistoryOperationRow, binding: TransactionBinding) -> bool:
    return (
        row.transaction_id == binding.transaction_id
        and row.event_id == binding.event_id
        and row.processing_sequence == binding.processing_sequence
        and row.transaction_kind == binding.transaction_kind.value
        and row.operation_digest == binding.operation_digest
    )


def _load(
    store: SubjectHistoryStore, incarnation_id: str, transaction_id: str
) -> HistoryOperationRow | None:
    try:
        return store.operation(incarnation_id, transaction_id)
    except HistoryStoreError as error:
        raise _diverged(error) from None


class SubjectHistoryParticipant:
    """Prepare/finalize/abort/reconcile authority for one Decision operation."""

    participant_id = SUBJECT_HISTORY_PARTICIPANT_ID
    capabilities = (
        ParticipantCapability.ABORT,
        ParticipantCapability.IDEMPOTENT_FINALIZE,
        ParticipantCapability.INSPECT_RECONCILE,
        ParticipantCapability.PREPARE,
    )

    def __init__(
        self,
        store: SubjectHistoryStore,
        incarnation_id: str,
        decision: CommittedDecisionV1,
        proof_source: CommitProofSource,
    ) -> None:
        if type(decision) is not CommittedDecisionV1:
            raise TypeError("SubjectHistory operation must be a CommittedDecisionV1")
        decision.__post_init__()
        if decision.incarnation_id != digest(incarnation_id, "incarnation_id"):
            raise ValueError("Decision incarnation is not the current incarnation")
        self.store = store
        self.incarnation_id = incarnation_id
        self.decision = decision
        self.proof_source = proof_source
        self.operation_digest = history_operation_digest(decision)

    # ---------------------------------------------------------- reconstruct

    @classmethod
    def from_pending(
        cls,
        store: SubjectHistoryStore,
        incarnation_id: str,
        proof_source: CommitProofSource,
        binding: TransactionBinding,
    ) -> SubjectHistoryParticipant:
        """Rebuild only from the exact retained PREPARED/FINALIZED bytes."""

        _require_binding(binding)
        row = _load(store, incarnation_id, binding.transaction_id)
        if row is None:
            # A committed transaction whose pending bytes are lost cannot be
            # reconstructed from a model/handler replay (#376 §B).
            raise ParticipantDivergedError("SubjectHistory pending operation is absent")
        if (
            row.operation_kind is not HistoryOperationKind.R16_DECISION_ISSUE
            or row.phase is HistoryPhase.ABORTED
            or row.pending_bytes is None
            or not _row_matches_binding(row, binding)
        ):
            raise ParticipantDivergedError("SubjectHistory pending operation conflicts")
        try:
            decision = CommittedDecisionV1.from_json(row.pending_bytes)
        except (TypeError, ValueError):
            raise ParticipantDivergedError("SubjectHistory pending bytes are invalid") from None
        participant = cls(store, incarnation_id, decision, proof_source)
        if participant.operation_digest != binding.operation_digest:
            raise ParticipantDivergedError("SubjectHistory pending digest conflicts")
        return participant

    # ---------------------------------------------------------------- live

    def prepare(self, binding: TransactionBinding) -> None:
        self._validate(binding)
        row = _load(self.store, self.incarnation_id, binding.transaction_id)
        if row is not None:
            if row.phase is HistoryPhase.PREPARED and self._row_is_operation(row, binding):
                return
            # A tombstone or a finalized row is never reissuable (#376 D1-1).
            raise ParticipantDivergedError("SubjectHistory transaction was already used")
        try:
            self.store.insert_prepared(
                transaction_id=binding.transaction_id,
                event_id=binding.event_id,
                processing_sequence=binding.processing_sequence,
                transaction_kind=binding.transaction_kind.value,
                operation_digest=self.operation_digest,
                decision=self.decision,
            )
        except HistoryStoreError as error:
            raise _diverged(error) from None

    def finalize(self, binding: TransactionBinding) -> ParticipantOutcome:
        self._validate(binding)
        row = _load(self.store, self.incarnation_id, binding.transaction_id)
        if row is None:
            raise ParticipantDivergedError("SubjectHistory pending operation is absent")
        if not self._row_is_operation(row, binding):
            raise ParticipantDivergedError("SubjectHistory operation conflicts")
        if row.phase is HistoryPhase.ABORTED:
            raise ParticipantDivergedError("An aborted SubjectHistory operation cannot finalize")
        if row.phase is HistoryPhase.FINALIZED:
            self._verify_finalized(binding)
            return ParticipantOutcome.ALREADY_CONSISTENT
        # Proof first, outside any SQLite write lock (#376 D1-5 lock order).
        receipt = self.proof_source.postcommit_receipt(binding)
        if (
            type(receipt) is not PostcommitReceiptV1
            or receipt.transaction_id != binding.transaction_id
            or receipt.event_id != binding.event_id
            or receipt.processing_sequence != binding.processing_sequence
        ):
            raise ParticipantDivergedError("Postcommit receipt does not match")
        try:
            self.store.finalize(
                transaction_id=binding.transaction_id,
                decision=self.decision,
                receipt=receipt,
            )
        except HistoryStoreError as error:
            raise _diverged(error) from None
        return ParticipantOutcome.FINALIZED

    def abort(self, binding: TransactionBinding) -> AbortOutcome:
        self._validate(binding)
        return self.abort_pending(self.store, self.incarnation_id, binding)

    @staticmethod
    def abort_pending(
        store: SubjectHistoryStore,
        incarnation_id: str,
        binding: TransactionBinding,
    ) -> AbortOutcome:
        """Abort before internal commit; callers must have proven PRE_INTERNAL.

        The live coordinator aborts only after a preparation failure, before
        R06 is attempted; startup aborts only after a PRE_INTERNAL proof.
        """

        _require_binding(binding)
        row = _load(store, incarnation_id, binding.transaction_id)
        try:
            if row is None:
                store.insert_abort_tombstone(
                    incarnation_id=incarnation_id,
                    transaction_id=binding.transaction_id,
                    event_id=binding.event_id,
                    processing_sequence=binding.processing_sequence,
                    transaction_kind=binding.transaction_kind.value,
                    operation_digest=binding.operation_digest,
                )
                return AbortOutcome.ALREADY_ABSENT
            if not _row_matches_binding(row, binding):
                raise ParticipantDivergedError("SubjectHistory abort binding conflicts")
            if row.phase is HistoryPhase.ABORTED:
                return AbortOutcome.ALREADY_ABSENT
            if row.phase is HistoryPhase.FINALIZED:
                raise ParticipantDivergedError("A finalized SubjectHistory operation cannot abort")
            store.mark_aborted(incarnation_id, binding.transaction_id)
            return AbortOutcome.ABORTED
        except HistoryStoreError as error:
            raise _diverged(error) from None

    # ------------------------------------------------------------- startup

    def inspect_reconciliation(self, binding: TransactionBinding) -> StartupParticipantOutcome:
        self._validate(binding)
        row = _load(self.store, self.incarnation_id, binding.transaction_id)
        if row is None or not self._row_is_operation(row, binding):
            raise ParticipantDivergedError("SubjectHistory operation conflicts")
        if row.phase is HistoryPhase.PREPARED:
            raise ParticipantUnavailableError("SubjectHistory operation needs roll-forward")
        if row.phase is not HistoryPhase.FINALIZED:
            raise ParticipantDivergedError("An aborted SubjectHistory operation is committed")
        self._verify_finalized(binding)
        return StartupParticipantOutcome.VERIFIED_CONSISTENT

    def reconcile(self, binding: TransactionBinding) -> StartupParticipantOutcome:
        outcome = self.finalize(binding)
        if outcome is ParticipantOutcome.ALREADY_CONSISTENT:
            return StartupParticipantOutcome.VERIFIED_CONSISTENT
        return StartupParticipantOutcome.ROLLED_FORWARD

    # ------------------------------------------------------------- helpers

    def _validate(self, binding: TransactionBinding) -> None:
        _require_binding(binding)
        if (
            binding.operation_digest != self.operation_digest
            or binding.event_id != self.decision.issuer_event_id
            or binding.processing_sequence != self.decision.issuer_sequence
        ):
            raise ParticipantDivergedError("SubjectHistory binding does not match its Decision")

    def _row_is_operation(self, row: HistoryOperationRow, binding: TransactionBinding) -> bool:
        return (
            row.operation_kind is HistoryOperationKind.R16_DECISION_ISSUE
            and _row_matches_binding(row, binding)
            and row.pending_bytes == self.decision.canonical_bytes()
        )

    def _verify_finalized(self, binding: TransactionBinding) -> None:
        try:
            finalized = self.store.finalized_decision(
                self.incarnation_id, binding.transaction_id
            )
        except HistoryStoreError as error:
            raise _diverged(error) from None
        if (
            finalized is None
            or finalized.decision_id != self.decision.decision_id
            or finalized.committed_bytes != self.decision.canonical_bytes()
            or finalized.committed_digest != self.decision.content_digest
            or finalized.receipt.transaction_id != binding.transaction_id
            or finalized.receipt.event_id != binding.event_id
            or finalized.receipt.processing_sequence != binding.processing_sequence
            or finalized.receipt_digest != finalized.receipt.receipt_digest
        ):
            raise ParticipantDivergedError("Finalized SubjectHistory evidence conflicts")


__all__ = [
    "CommitProofSource",
    "JournalCommitProofSource",
    "SubjectHistoryParticipant",
]
