"""R16 U4-3: adopt ``subject.history`` as the fifth R07 participant (#376 §A).

Adoption is an explicit, one-way step at a clean R07 boundary:

1. the SubjectHistoryDB is created PROVISIONAL (only when the latest adopted
   baseline does not yet register ``subject.history``);
2. R07 publishes a strictly larger participant baseline (old four + explicit
   five) through the existing ``establish_participant_baseline``;
3. ``history_meta`` moves PROVISIONAL -> ACTIVE, fixing the adopted baseline
   ID/hash and the GENESIS incarnation.

A cold restart with an identical qualified baseline and a PROVISIONAL store
with zero operations resumes step 3.  A lost store after adoption, a store of
another Journal lineage or a different adopted baseline fails closed.

The current incarnation (#376 D1-4) is derived only from proven R07 Journal
facts: the adopted GENESIS baseline, or the latest terminal TRUE_ROLLBACK
CLEARED record taken under a baseline that already registers
``subject.history``.  It is never a SQLite-local counter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from suzka.r16.contracts import checksum
from suzka.r16.history_codec import SUBJECT_HISTORY_PARTICIPANT_ID
from suzka.r16.history_store import (
    HistoryMetaPhase,
    HistoryStoreConflict,
    HistoryStoreCorrupt,
    HistoryStoreUnavailable,
    SubjectHistoryStore,
)
from suzka.runtime.event_journal import (
    EventJournal,
    EventJournalGateClear,
    EventJournalInspection,
    EventJournalParticipantBaseline,
    ParticipantBaseline,
    ParticipantDomain,
)


SUBJECT_HISTORY_BASELINE: Final = ParticipantBaseline(
    participant_id=SUBJECT_HISTORY_PARTICIPANT_ID,
    domain=ParticipantDomain.DURABLE_DOMAIN,
)

_BASELINE_DOMAIN: Final = b"PROJECT-SUZKA:R16:SUBJECT-HISTORY-BASELINE:V1\0"
_INCARNATION_DOMAIN: Final = b"PROJECT-SUZKA:R16:SUBJECT-HISTORY-INCARNATION:V1\0"


class HistoryAdoptionError(Exception):
    """SubjectHistory adoption or incarnation evidence fails closed."""


def history_registry(
    registry: tuple[ParticipantBaseline, ...],
) -> tuple[ParticipantBaseline, ...]:
    """The strictly larger registry: the existing participants plus history."""

    if any(item.participant_id == SUBJECT_HISTORY_PARTICIPANT_ID for item in registry):
        raise HistoryAdoptionError("the base registry already names subject.history")
    return tuple(
        sorted((*registry, SUBJECT_HISTORY_BASELINE), key=lambda item: item.participant_id)
    )


def registers_history(baseline: EventJournalParticipantBaseline) -> bool:
    return any(
        item.participant_id == SUBJECT_HISTORY_PARTICIPANT_ID
        and item.domain is ParticipantDomain.DURABLE_DOMAIN
        for item in baseline.participant_registry
    )


def baseline_digest(baseline: EventJournalParticipantBaseline) -> str:
    """Digest of every field of one published R07 participant baseline."""

    return checksum(
        _BASELINE_DOMAIN,
        {
            "adoption_epoch": baseline.adoption_epoch,
            "baseline_id": baseline.baseline_id,
            "journal_lineage_id": baseline.journal_lineage_id,
            "participant_registry": [
                [item.participant_id, item.domain.value]
                for item in baseline.participant_registry
            ],
            "processing_high_water": baseline.processing_high_water,
            "snapshot_hash": baseline.snapshot_hash,
            "snapshot_sequence": baseline.snapshot_sequence,
            "wal_generation_id": baseline.wal_generation_id,
            "wal_record_hash": baseline.wal_record_hash,
            "wal_record_id": baseline.wal_record_id,
        },
    )


def genesis_incarnation_id(baseline: EventJournalParticipantBaseline) -> str:
    return checksum(
        _INCARNATION_DOMAIN,
        {
            "anchor": "GENESIS",
            "baseline_hash": baseline_digest(baseline),
            "baseline_id": baseline.baseline_id,
            "journal_lineage_id": baseline.journal_lineage_id,
        },
    )


def rollback_incarnation_id(clear: EventJournalGateClear) -> str:
    return checksum(
        _INCARNATION_DOMAIN,
        {
            "anchor": "TRUE_ROLLBACK_CLEARED",
            "baseline_id": clear.baseline_id,
            "journal_lineage_id": clear.journal_lineage_id,
            "processing_high_water": clear.processing_high_water,
            "reconciliation_id": clear.reconciliation_id,
            "recovery_id": clear.recovery_id,
            "snapshot_hash": clear.snapshot_hash,
            "snapshot_sequence": clear.snapshot_sequence,
            "wal_generation_id": clear.wal_generation_id,
            "wal_record_hash": clear.wal_record_hash,
            "wal_record_id": clear.wal_record_id,
        },
    )


def adopted_history_baseline(
    inspection: EventJournalInspection,
) -> EventJournalParticipantBaseline | None:
    """The first published baseline that registers ``subject.history``."""

    return next(
        (item for item in inspection.baselines if registers_history(item)), None
    )


def open_or_create_store(
    journal: EventJournal,
    path: Path,
    *,
    subject_id: str,
    max_page_count: int,
) -> SubjectHistoryStore:
    """Open the store, creating it PROVISIONAL only before adoption."""

    if not isinstance(path, Path):
        raise HistoryAdoptionError("SubjectHistory path must be a Path")
    inspection = journal.inspect()
    lineage = inspection.journal_lineage_id
    if lineage is None:
        raise HistoryAdoptionError("Journal lineage is unavailable")
    adopted = adopted_history_baseline(inspection)
    try:
        store = SubjectHistoryStore.open(path, max_page_count=max_page_count)
    except HistoryStoreUnavailable as error:
        if adopted is not None or path.exists() or path.is_symlink():
            # After adoption a missing store is lost authority, never recreated.
            raise HistoryAdoptionError("SubjectHistory store is unavailable") from error
    except (HistoryStoreConflict, HistoryStoreCorrupt) as error:
        raise HistoryAdoptionError("SubjectHistory store is not trusted") from error
    else:
        try:
            if store.meta().subject_id != subject_id:
                raise HistoryAdoptionError("SubjectHistory store belongs to another subject")
            if adopted is None:
                require_adoptable(store, inspection)
        except BaseException:
            store.close()
            raise
        return store
    try:
        return SubjectHistoryStore.create(
            path,
            subject_id=subject_id,
            journal_lineage_id=lineage,
            max_page_count=max_page_count,
        )
    except (HistoryStoreConflict, HistoryStoreCorrupt, HistoryStoreUnavailable) as error:
        raise HistoryAdoptionError("SubjectHistory store creation failed") from error


def require_adoptable(
    store: SubjectHistoryStore, inspection: EventJournalInspection
) -> None:
    """Before the one-way R07 adoption: a clean PROVISIONAL store of this Journal.

    Checked before ``establish_participant_baseline`` publishes the five
    participant baseline, so a foreign, stale or used store fails closed
    without any Journal write (#376 §A; IP-9 U4-3 review P1-1).
    """

    meta = store.meta()
    if meta.phase is not HistoryMetaPhase.PROVISIONAL:
        raise HistoryAdoptionError("an ACTIVE SubjectHistory store has no adoption")
    if meta.journal_lineage_id != inspection.journal_lineage_id:
        raise HistoryAdoptionError("SubjectHistory store belongs to another Journal")
    if store.has_operations():
        raise HistoryAdoptionError("a PROVISIONAL SubjectHistory store has rows")


def activate_adopted_store(
    store: SubjectHistoryStore, inspection: EventJournalInspection
) -> str:
    """Bind a PROVISIONAL store to the adopted baseline, or verify an ACTIVE one."""

    adopted = adopted_history_baseline(inspection)
    if adopted is None:
        raise HistoryAdoptionError("subject.history is not in an adopted baseline")
    meta = store.meta()
    if (
        meta.journal_lineage_id != inspection.journal_lineage_id
        or adopted.journal_lineage_id != inspection.journal_lineage_id
    ):
        raise HistoryAdoptionError("SubjectHistory store belongs to another Journal")
    adopted_hash = baseline_digest(adopted)
    genesis = genesis_incarnation_id(adopted)
    if meta.phase is HistoryMetaPhase.PROVISIONAL:
        # #376 §A: resume only with zero operations in any incarnation.
        if store.has_operations():
            raise HistoryAdoptionError("a PROVISIONAL SubjectHistory store has rows")
        try:
            store.activate(
                adopted_baseline_id=adopted.baseline_id,
                adopted_baseline_hash=adopted_hash,
                genesis_incarnation_id=genesis,
            )
        except (HistoryStoreConflict, HistoryStoreCorrupt, HistoryStoreUnavailable) as error:
            raise HistoryAdoptionError("SubjectHistory activation failed") from error
        return genesis
    if (
        meta.adopted_baseline_id != adopted.baseline_id
        or meta.adopted_baseline_hash != adopted_hash
        or meta.genesis_incarnation_id != genesis
    ):
        raise HistoryAdoptionError("SubjectHistory store was adopted for another baseline")
    return genesis


def current_incarnation(journal: EventJournal, store: SubjectHistoryStore) -> str:
    """#376 D1-4: the incarnation from proven Journal facts only."""

    inspection = journal.inspect()
    adopted = adopted_history_baseline(inspection)
    meta = store.meta()
    if (
        adopted is None
        or meta.phase is not HistoryMetaPhase.ACTIVE
        or meta.journal_lineage_id != inspection.journal_lineage_id
        or meta.adopted_baseline_id != adopted.baseline_id
        or meta.adopted_baseline_hash != baseline_digest(adopted)
        or meta.genesis_incarnation_id != genesis_incarnation_id(adopted)
    ):
        raise HistoryAdoptionError("SubjectHistory adoption evidence is not exact")
    if inspection.open_gate_clear is not None:
        raise HistoryAdoptionError("a true-rollback gate clear is still open")
    clear = inspection.terminal_gate_clear
    if clear is None:
        return meta.genesis_incarnation_id
    baselines = {item.baseline_id: item for item in inspection.baselines}
    clear_baseline = baselines.get(clear.baseline_id)
    if clear_baseline is None or clear.journal_lineage_id != inspection.journal_lineage_id:
        raise HistoryAdoptionError("true-rollback clear evidence is incomplete")
    if not registers_history(clear_baseline):
        # A rollback before adoption does not change the GENESIS incarnation.
        return meta.genesis_incarnation_id
    return rollback_incarnation_id(clear)


__all__ = [
    "SUBJECT_HISTORY_BASELINE",
    "HistoryAdoptionError",
    "activate_adopted_store",
    "adopted_history_baseline",
    "baseline_digest",
    "current_incarnation",
    "genesis_incarnation_id",
    "history_registry",
    "open_or_create_store",
    "registers_history",
    "require_adoptable",
    "rollback_incarnation_id",
]
