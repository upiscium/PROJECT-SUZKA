"""Journal-aware retirement of Memory-owned Semantic operation receipts."""

from __future__ import annotations

from collections.abc import Iterable

from suzka.memory.semantic_participant import (
    MEMORY_SEMANTIC_PARTICIPANT_ID,
)
from suzka.memory.semantic_store import (
    SemanticStore,
    SemanticStoreConflict,
    SemanticStoreError,
)
from suzka.runtime.event_journal import EventJournal
from suzka.runtime.transaction_coordinator import (
    ParticipantRequirement,
)


class SemanticReceiptRetentionCoordinator:
    """Retire receipts only after Journal participant-finalized evidence."""

    def __init__(self, journal: EventJournal, store: SemanticStore) -> None:
        self.journal = journal
        self.store = store

    def _verified_checkpoint_horizon(self, inspection: object) -> int | None:
        checkpoint = self.store.load_checkpoint()
        if checkpoint is None:
            return None
        lineage = getattr(inspection, "journal_lineage_id", None)
        processing_high_water = getattr(inspection, "processing_high_water", None)
        records = getattr(inspection, "records", ())
        # After a compaction the tail may have been archived; the COMPACTION
        # then carries it verbatim (C8, #381 D1 P0-1).
        carried = getattr(inspection, "carried_records", ())
        tail_record = next(
            (
                record
                for record in (*records, *carried)
                if record.record_id == checkpoint.journal_tail_record_id
                and record.record_hash == checkpoint.journal_tail_record_hash
            ),
            None,
        )
        if tail_record is None:
            return None
        lifecycle = getattr(tail_record.lifecycle, "value", tail_record.lifecycle)
        if lifecycle in {"completed", "transaction_completed", "transaction_reconciled"}:
            tail_high_water = tail_record.processing_sequence
        elif lifecycle in {"participant_baseline_established", "cleared"}:
            tail_high_water = tail_record.recovery_processing_high_water
        else:
            tail_high_water = None
        if (
            checkpoint.journal_lineage_id != lineage
            or type(processing_high_water) is not int
            or checkpoint.processing_high_water > processing_high_water
            or tail_high_water != checkpoint.processing_high_water
        ):
            return None
        return checkpoint.processing_high_water

    def checkpoint_covers(self, transaction: object) -> bool:
        """Return whether a clean Semantic checkpoint covers this transaction."""

        try:
            inspection = self.journal.inspect()
            horizon = self._verified_checkpoint_horizon(inspection)
        except SemanticStoreError:
            return False
        sequence = getattr(transaction, "processing_sequence", None)
        return horizon is not None and type(sequence) is int and sequence <= horizon

    def advance_checkpoint(self) -> None:
        """Persist a cross-authority clean boundary after terminal completion."""

        inspection = self.journal.inspect()
        if (
            inspection.open_events
            or inspection.open_transactions
            or inspection.open_recoveries
            or inspection.external_reconciliation_required
            or inspection.journal_lineage_id is None
            or inspection.tail_record_id is None
            or inspection.tail_record_hash is None
        ):
            return
        self.store.write_checkpoint(
            processing_high_water=inspection.processing_high_water,
            journal_lineage_id=inspection.journal_lineage_id,
            journal_tail_record_id=inspection.tail_record_id,
            journal_tail_record_hash=inspection.tail_record_hash,
        )

    def _retired_transaction_proofs(self) -> dict[str, str]:
        inspection = self.journal.inspect()
        transactions = (
            *inspection.completed_transactions,
            *inspection.reconciled_transactions,
        )
        latest_baseline = inspection.baselines[-1] if inspection.baselines else None
        checkpoint_horizon = self._verified_checkpoint_horizon(inspection)
        baseline_horizon = (
            None
            if latest_baseline is None
            else latest_baseline.processing_high_water
        )
        horizons = tuple(
            horizon
            for horizon in (baseline_horizon, checkpoint_horizon)
            if horizon is not None
        )
        if not horizons:
            return {}
        safe_horizon = max(horizons)
        proofs: dict[str, str] = {}
        for transaction in transactions:
            if not any(
                participant_id == MEMORY_SEMANTIC_PARTICIPANT_ID
                for participant_id, _outcome in transaction.participant_outcomes
            ):
                continue
            requirement = next(
                (
                    item
                    for item in transaction.required_participants
                    if item.participant_id == MEMORY_SEMANTIC_PARTICIPANT_ID
                ),
                None,
            )
            if requirement is None:
                continue
            if transaction.processing_sequence > safe_horizon:
                continue
            proofs[transaction.transaction_id] = requirement.operation_digest
        return proofs

    def before_prepare(
        self, requirements: Iterable[ParticipantRequirement] | None = None
    ) -> None:
        """Retry safe retirement before admitting another coordinated mutation."""

        if requirements is not None and not any(
            requirement.participant_id == MEMORY_SEMANTIC_PARTICIPANT_ID
            for requirement in requirements
        ):
            return
        proofs = self._retired_transaction_proofs()
        for transaction_id, operation_digest in proofs.items():
            receipt = self.store.load_receipt(transaction_id)
            if receipt is not None and receipt.get("operation_digest") != operation_digest:
                raise SemanticStoreConflict("Semantic receipt proof conflicts")
        self.store.prune_receipts(proofs)

    def carried_tail(self) -> tuple[str, str] | None:
        """The Journal record the Semantic checkpoint names (C8), if any."""

        checkpoint = self.store.load_checkpoint()
        if checkpoint is None:
            return None
        return checkpoint.journal_tail_record_id, checkpoint.journal_tail_record_hash

    def compaction_blocked(self, archived_high_water: int) -> bool:
        """Retire receipts first; block if one still needs archived evidence.

        A receipt is retired only from Journal transaction evidence, which a
        compaction moves to the archive. So receipts for transactions at or
        below the archived high-water must be gone first (#381 D1 P0-2).
        """

        try:
            self.before_prepare()
            remaining = set(self.store.receipt_transaction_ids())
        except SemanticStoreError:
            return True
        if not remaining:
            return False
        inspection = self.journal.inspect()
        transactions = (
            *inspection.open_transactions,
            *inspection.reconciliation_required_transactions,
            *inspection.completed_transactions,
            *inspection.reconciled_transactions,
            *inspection.abort_required_transactions,
            *inspection.aborted_transactions,
        )
        return any(
            transaction.transaction_id in remaining
            and transaction.processing_sequence <= archived_high_water
            for transaction in transactions
        )

    def after_participant_finalized(self) -> None:
        """Best-effort cleanup after the Journal outcome is durable."""

        try:
            self.before_prepare()
        except SemanticStoreError:
            # The participant outcome is already durable.  Leaving a receipt
            # behind is safe; startup and the next admission retry cleanup.
            return

    def after_terminal_completion(self) -> None:
        """Advance the clean checkpoint, then retire newly unreachable receipts."""

        try:
            self.advance_checkpoint()
            self.before_prepare()
        except SemanticStoreError:
            # Terminal Journal/state evidence is already durable.  Keep
            # receipts until a later startup or admission retry can prove and
            # retire them.
            return


__all__ = ["SemanticReceiptRetentionCoordinator"]
