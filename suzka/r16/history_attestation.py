"""C10: the SubjectHistory attestation provider for Journal compaction.

#376 C10 D2/D3.  ``EventJournal.compact()`` calls this provider, under the
Journal lock and before anything is archived, whenever its own records say
``subject.history`` is registered.  The provider:

1. resolves the current incarnation from the same inspection the reader uses;
2. classifies the whole incarnation with the previous attestation;
3. matches the archived range's R07 ``subject.history`` tuples of the
   current incarnation to SubjectHistory rows one to one (binding and
   terminal phase);
4. returns an attestation of the operations and adopted chain prefixes at
   the new archived high-water, carrying permanently BLOCKED scopes (or the
   whole incarnation) as *poisoned* so compaction never starves.

Only *temporary* unavailability defers (store busy, Journal unreadable).
Owns no state and never writes SubjectHistory or the Journal.
"""

from __future__ import annotations

from suzka.r16.history_adoption import HistoryAdoptionError, resolve_incarnation
from suzka.r16.history_attestation_digest import (
    operations_digest,
    prefix_tips,
    scopes_digest,
)
from suzka.r16.history_reader import (
    HeadBlockReason,
    HeadStatus,
    ReaderTransient,
    SubjectHistoryReader,
)
from suzka.r16.history_store import (
    HistoryOperationRow,
    HistoryPhase,
    HistoryStoreError,
    SubjectHistoryStore,
)
from suzka.runtime.event_journal import (
    MAX_ATTESTED_POISONED_SCOPES,
    EventJournal,
    EventJournalHistoryAttestation,
    EventLifecycle,
    HistoryAttestationDefer,
    HistoryAttestationRefuse,
    HistoryAttestationRequest,
)
from suzka.r16.history_store import HistoryMetaPhase

# Resolvable by startup reconciliation: never poisoned (D3 §3).  Compaction
# runs only at a clean R07 boundary, so this is not expected to be reached
# there; deferring is the safe choice if it ever is (review P3-9).
_TRANSIENT_BLOCKS = frozenset({HeadBlockReason.UNRESOLVED_OPERATION})


class SubjectHistoryAttestationProvider:
    """Callable handed to ``EventJournal.compact(history_attestation=...)``."""

    def __init__(self, journal: EventJournal, store: SubjectHistoryStore) -> None:
        self.journal = journal
        self.store = store

    def __call__(
        self, request: HistoryAttestationRequest
    ) -> (
        EventJournalHistoryAttestation | HistoryAttestationDefer | HistoryAttestationRefuse
    ):
        try:
            inspection = self.journal.inspect()
        except Exception:
            return HistoryAttestationDefer("journal_unavailable")
        try:
            incarnation, horizon = resolve_incarnation(inspection, self.store)
            meta = self.store.meta()
        except HistoryStoreError:
            return HistoryAttestationDefer("store_unavailable")
        except HistoryAdoptionError:
            # Review P2-3: only a not-yet-active store or an open gate clear is
            # temporary; any other adoption mismatch is permanent.
            try:
                provisional = self.store.meta().phase is HistoryMetaPhase.PROVISIONAL
            except HistoryStoreError:
                return HistoryAttestationDefer("store_unavailable")
            if provisional or inspection.open_gate_clear is not None:
                return HistoryAttestationDefer("incarnation_unavailable")
            return HistoryAttestationRefuse("history_adoption_invalid")
        latest = inspection.latest_compaction
        if (None if latest is None else latest.history_attestation) != request.previous:
            # The request and the inspection describe different chains.
            raise ValueError("previous history attestation does not match")
        reader = SubjectHistoryReader(
            self.journal, self.store, lambda _inspection: (incarnation, horizon)
        )
        try:
            result = reader.classify(inspection=inspection)
        except ReaderTransient as transient:
            return HistoryAttestationDefer(transient.reason.value.lower())
        high_water = request.archived_high_water
        poisoned_all = False
        poisoned: set[str] = set()
        if result.global_block in _TRANSIENT_BLOCKS:
            return HistoryAttestationDefer("unresolved_operation")
        if result.global_block is not None:
            poisoned_all = True
        else:
            for scope, head in result.heads.items():
                if head.status is HeadStatus.BLOCKED:
                    if head.reason in _TRANSIENT_BLOCKS:
                        return HistoryAttestationDefer("unresolved_operation")
                    poisoned.add(scope)
            previous_high_water = (
                0 if latest is None else latest.archived_processing_high_water
            )
            if not _r07_matches(request, result.rows, horizon, previous_high_water):
                poisoned_all = True
        if len(poisoned) > MAX_ATTESTED_POISONED_SCOPES:
            poisoned_all = True
        rows = tuple(row for row in result.rows if row.processing_sequence <= high_water)
        operations_count, operations = operations_digest(rows)
        if poisoned_all:
            poisoned = set()
            tips = {}
        else:
            chains = {
                scope: chain
                for scope, chain in result.chains.items()
                if scope not in poisoned
            }
            try:
                tips = prefix_tips(chains, high_water)
            except ValueError:
                # Review P3-6: same treatment as the reader (a divergence).
                poisoned_all, poisoned, tips = True, set(), {}
        scope_count, scopes = scopes_digest(tips)
        return EventJournalHistoryAttestation(
            incarnation_id=incarnation,
            horizon=horizon,
            store_id=meta.store_id,
            subject_id=meta.subject_id,
            operations_count=operations_count,
            operations_digest=operations,
            scope_count=scope_count,
            scopes_digest=scopes,
            poisoned_all=poisoned_all,
            poisoned_scopes=tuple(sorted(poisoned)),
            previous_attestation_hash=(
                None if request.previous is None else request.previous.attestation_hash
            ),
        )


def _r07_matches(
    request: HistoryAttestationRequest,
    rows: tuple[HistoryOperationRow, ...],
    horizon: int,
    previous_high_water: int,
) -> bool:
    """One-to-one: archived R07 tuples of this incarnation <-> rows (D3 §1).

    Tuples at or below ``horizon`` belong to an earlier incarnation.  Rows at
    or below the previous archived high-water were attested before.
    """

    by_transaction = {row.transaction_id: row for row in rows}
    seen: set[str] = set()
    for item in request.r07_tuples:
        if len(item) != 8:
            return False
        transaction_id, event_id, sequence, _kind, digest, _outcomes, terminals, _ = item
        if not isinstance(sequence, int) or isinstance(sequence, bool):
            return False
        if sequence <= horizon:
            continue
        row = by_transaction.get(str(transaction_id))
        if (
            row is None
            or row.transaction_id in seen
            or row.event_id != event_id
            or row.processing_sequence != sequence
            or row.operation_digest != digest
            or not isinstance(terminals, tuple)
            or len(terminals) != 1
        ):
            return False
        expected = (
            HistoryPhase.ABORTED
            if terminals[0] == EventLifecycle.TRANSACTION_ABORTED.value
            else HistoryPhase.FINALIZED
        )
        if row.phase is not expected:
            return False
        seen.add(row.transaction_id)
    lower = max(horizon, previous_high_water)
    return all(
        row.transaction_id in seen
        for row in rows
        if lower < row.processing_sequence <= request.archived_high_water
    )


__all__ = ["SubjectHistoryAttestationProvider"]
