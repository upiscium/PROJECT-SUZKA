"""F15 DC-4: quarantine of the Experience / Semantic participants (#411).

D1 ``d-f15-slice`` r3 sec. 4.2 and 5.2, Human decisions HD-F15S-1 (a) (three
``QUARANTINED`` outcome values), HD-F15S-6 (a) (the whole store) and HD-new-2 (a)
(only ``DIVERGED`` / evidence-absent, never ``Unavailable``).

What this module is, and is not:

* ``ParticipantResolutionPolicy`` is the legacy default: it resolves nothing, so a
  coordinator without ``DomainQuarantinePolicy`` behaves exactly as before.
  ``DomainQuarantinePolicy`` is wired only when ``degraded_continuation.mode`` is
  ``"continue"``.
* The quarantine moves the damaged store directory aside by ``os.rename`` (the
  ``QuarantineWriter`` of DC-1), journals ``QUARANTINED`` through the EXISTING
  reconciliation-required path (never ``append_transaction_completed``), and
  leaves the empty store to the NEXT cycle (``CycleRestart``): the cycle is never
  resumed with the stale store objects.
* Nothing here repairs, rewrites, copies or deletes a damaged artifact.  The two
  things it creates are NEW: a manifest (DC-1) and the empty root with its
  marker (``create_empty_root``).

ONE-WAY: a Journal that contains a ``QUARANTINED`` outcome cannot be read by a
binary older than DC-4 (the enum value is unknown to it; it fails closed at load
with ``EventJournalLoadError``, S0-b).  Nothing here writes one unless a real
quarantine happens.

Known limits (stated, not hidden):

* A restored or hand-replaced store directory (FOREIGN_RESTORE, F-15) is only
  DETECTED (``domain_store_replaced``); no automatic action.  A directory that
  was deleted and is re-created by ordinary writes looks the same (no marker).
* ``quarantine_unavailable`` other than ``"never"`` is not implemented here: an
  ``Unavailable`` participant is never quarantined (HD-new-2 (a)).
* ``memory.semantic`` is implemented and tested but OFF by default
  (``DEFAULT_QUARANTINE_PARTICIPANTS``).  S0-c (D1 sec. 5.2 item 8, F-7) showed that
  after a Semantic quarantine EVERY chat fails while a DB2 projection row of the
  old store exists: ``DualMemorySystem._semantic_records_from_query`` verifies each
  R12 row against the (now empty) authoritative store and raises
  ``SemanticMemoryReadError``; the working-memory resolver does the same for a
  Semantic working-memory item.  D1 F-7 says "if unsafe, DC-4 narrows to Experience
  only"; the Semantic fault therefore ends ``RECONCILING`` exactly as before, until
  the read path is changed by a design decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import stat
import time
from typing import TYPE_CHECKING, Callable
from uuid import uuid4

from suzka.runtime.degraded_continuation import (
    DOMAIN_STORE_REPLACED_REASON,
    EPOCH_MARKER_NAME,
    DegradedContinuationStatus,
    ManifestError,
    ManifestState,
    QuarantineError,
    QuarantineManifest,
    QuarantineWriter,
    ResumeAction,
    classify_participant_error,
    is_quarantinable,
    IntegrityVerdict,
)
from suzka.runtime.event_journal import (
    AbortOutcome,
    EventJournalInspection,
    ParticipantCapability,
    ParticipantOutcome,
    ReconciliationReason,
    StartupParticipantOutcome,
)
from suzka.runtime.state_recovery import InternalCommitClassification

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from suzka.runtime.event_journal import EventJournalStartupReconciliation
    from suzka.runtime.startup_reconciliation import StartupReconciliationCoordinator

LOGGER = logging.getLogger(__name__)

MEMORY_EXPERIENCE = "memory.experience"
MEMORY_SEMANTIC = "memory.semantic"
#: participant id -> store directory name beside the Chroma directory
#: (``ExperienceStore.from_memory_root`` / ``SemanticStore.from_memory_root``).
STORE_DIRECTORY_NAMES: dict[str, str] = {
    MEMORY_EXPERIENCE: "experience",
    MEMORY_SEMANTIC: "semantic",
}
# Reasons the quarantine records (short labels, never a path or a payload).
REASON_OPEN_TRANSACTION = "diverged_open_transaction"
REASON_AGGREGATE = "diverged_aggregate"
#: One confirming re-read of a DIVERGED verdict after this many seconds (D1 sec. 3).
CONFIRM_DELAY_S = 1.0
_TMP_INFIX = ".dc-new-"
#: The participants a policy quarantines unless told otherwise.  Experience only:
#: see the module docstring (S0-c, D1 F-7).
DEFAULT_QUARANTINE_PARTICIPANTS: frozenset[str] = frozenset({MEMORY_EXPERIENCE})

__all__ = [
    "CONFIRM_DELAY_S",
    "DEFAULT_QUARANTINE_PARTICIPANTS",
    "DomainQuarantinePolicy",
    "ParticipantResolutionPolicy",
    "STORE_DIRECTORY_NAMES",
    "create_empty_root",
    "new_epoch_id",
    "participant_floor",
    "read_epoch_marker",
]


# --------------------------------------------------------------------------
# The policy interface (legacy default)
# --------------------------------------------------------------------------


class ParticipantResolutionPolicy:
    """Legacy behaviour: nothing is resolved.  A participant fault ends the
    reconciliation with ``external_participant_reconciliation_required``."""

    def begin(self, coordinator: "StartupReconciliationCoordinator") -> str | None:
        """Called once at the start of ``reconcile_open_transactions``.  Returns
        a ``degraded_reason`` to refuse with, or None.  May finish a crashed
        quarantine and then raise ``CycleRestart``."""

        return None

    def confirm_delay(self) -> None:
        """Wait before the one confirming re-read of a DIVERGED verdict."""

    def should_restart(self) -> bool:
        """True when a quarantine happened in this cycle: the store objects of
        the cycle are stale and the cycle must end (``CycleRestart``)."""

        return False

    def resolve_transaction_fault(
        self,
        coordinator: "StartupReconciliationCoordinator",
        participant_id: str | None,
        error: BaseException,
    ) -> bool:
        """True when the fault was resolved by a quarantine and every open
        transaction that needs the participant carries the outcome."""

        return False

    def resolve_aggregate_fault(
        self,
        coordinator: "StartupReconciliationCoordinator",
        participant_id: str,
        error: BaseException,
        reconciliation: "EventJournalStartupReconciliation",
    ) -> bool:
        return False

    def wants_confirmation(
        self, participant_id: str | None, error: BaseException
    ) -> bool:
        """True when ``error`` would be quarantined, so the caller re-reads it
        once before acting."""

        return False


# --------------------------------------------------------------------------
# Journal-derived helpers
# --------------------------------------------------------------------------


def _is_quarantined(value: object) -> bool:
    return getattr(value, "value", value) == "quarantined"


def participant_floor(inspection: EventJournalInspection, participant_id: str) -> int:
    """The highest processing sequence a quarantine of ``participant_id`` has
    covered, from JOURNAL RECORDS ONLY (a sidecar must never narrow
    verification): the maximum ``processing_sequence`` of any transaction that
    carries a ``QUARANTINED`` outcome for the participant, and the
    ``recovery_processing_high_water`` of any startup reconciliation that does.

    Transactions at or below it were checked against a store that no longer
    exists and are out of scope for a LATER aggregate verification of the new
    store.  It is used only inside ``_reconcile_aggregate_participant``: never
    an input of the transaction list, the requirement digest or the
    ``reconciliation_id`` (D1 r3 P1-A)."""

    floor = 0
    transactions = (
        *inspection.open_transactions,
        *inspection.completed_transactions,
        *inspection.reconciled_transactions,
        *inspection.aborted_transactions,
    )
    for transaction in transactions:
        if any(
            pid == participant_id and _is_quarantined(outcome)
            for pid, outcome in (
                *transaction.participant_outcomes,
                *transaction.abort_outcomes,
            )
        ):
            floor = max(floor, transaction.processing_sequence)
    for reconciliation in (
        *inspection.open_startup_reconciliations,
        *inspection.completed_startup_reconciliations,
    ):
        if any(
            pid == participant_id and _is_quarantined(outcome)
            for pid, _digest, outcome in reconciliation.participant_outcomes
        ):
            floor = max(floor, reconciliation.recovery_processing_high_water)
    return floor


def new_epoch_id(now: datetime | None = None) -> str:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid4().hex[:8]}"


def read_epoch_marker(root: Path) -> str | None:
    """The epoch id in ``<root>/.dc_epoch.json``, or None (absent, unreadable,
    not ours).  Read-only."""

    try:
        marker = json.loads((Path(root) / EPOCH_MARKER_NAME).read_bytes())
    except (OSError, ValueError):
        return None
    if isinstance(marker, dict) and isinstance(marker.get("epoch_id"), str):
        return marker["epoch_id"]
    return None


# --------------------------------------------------------------------------
# The empty root (step 4)
# --------------------------------------------------------------------------


def _mkdir_private(path: Path) -> None:
    os.mkdir(path, 0o700)
    os.chmod(path, 0o700)


def _write_marker(path: Path, data: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_root(temporary: Path, root: Path) -> None:
    os.rename(temporary, root)


def create_empty_root(
    root: Path,
    *,
    epoch_id: str,
    participant_id: str,
    created_at: datetime | None = None,
) -> None:
    """Create ``root`` as an empty store directory holding only the epoch
    marker, atomically (a sibling temporary directory, then ``rename``).

    Refuses (``QuarantineError``, nothing changed) when ``root`` exists in any
    form, when the parent is not owned by this process, or when a leftover
    temporary of this epoch is not exactly the marker.  A leftover temporary of
    THIS epoch that is exactly the marker is re-used (idempotent after a
    crash): it is never deleted.  The root has no ``records/`` / ``pending/`` /
    ``receipts/``: the store creates them on first use, exactly as for a store
    that never existed."""

    root = Path(root)
    parent = root.parent
    if os.path.lexists(root):
        raise QuarantineError("the empty root cannot be created: the path exists")
    try:
        if os.stat(parent).st_uid != os.geteuid():
            raise QuarantineError("the parent directory is not owned by this process")
    except OSError as error:
        raise QuarantineError("the parent directory cannot be inspected") from error
    temporary = parent / f"{root.name}{_TMP_INFIX}{epoch_id}"
    marker = temporary / EPOCH_MARKER_NAME
    stamp = (created_at or datetime.now(timezone.utc)).isoformat()
    payload = json.dumps(
        {"epoch_id": epoch_id, "created_at": stamp, "participant_id": participant_id},
        sort_keys=True,
    ).encode("utf-8")
    try:
        if os.path.lexists(temporary):
            status = os.lstat(temporary)
            if not stat.S_ISDIR(status.st_mode) or sorted(os.listdir(temporary)) != [
                EPOCH_MARKER_NAME
            ]:
                raise QuarantineError("a leftover temporary root is not the marker only")
            if read_epoch_marker(temporary) != epoch_id:
                raise QuarantineError("a leftover temporary root is another epoch's")
        else:
            _mkdir_private(temporary)
            _write_marker(marker, payload)
            _fsync_dir(temporary)
        _publish_root(temporary, root)
        _fsync_dir(parent)
    except OSError as error:
        raise QuarantineError("the empty root could not be created") from error


# --------------------------------------------------------------------------
# The policy
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Epochs:
    manifests: tuple[QuarantineManifest, ...]
    unreadable: bool


class DomainQuarantinePolicy(ParticipantResolutionPolicy):
    """Quarantine ``memory.experience`` / ``memory.semantic`` stores whose
    evidence is divergent or absent (D1 sec. 4.2)."""

    def __init__(
        self,
        *,
        memory_root: Path,
        quarantine_dir: Path,
        has_exclusive_authority: Callable[[], bool],
        status: DegradedContinuationStatus | None = None,
        t_hash_s: float = 30.0,
        confirm_delay_s: float = CONFIRM_DELAY_S,
        sleep: Callable[[float], None] = time.sleep,
        runtime_instance_id: str | None = None,
        mismatch_bound_s: float | None = None,
        wall_clock: Callable[[], datetime] | None = None,
        participants: frozenset[str] | None = None,
    ) -> None:
        enabled = DEFAULT_QUARANTINE_PARTICIPANTS if participants is None else participants
        if not enabled <= frozenset(STORE_DIRECTORY_NAMES):
            raise ValueError("only memory.experience / memory.semantic can be quarantined")
        #: What this policy ACTS on.  Existing manifests of any participant are
        #: still resumed (a crashed quarantine must always be finished).
        self.enabled_participants = frozenset(enabled)
        self._memory_root = Path(memory_root)
        self._status = status
        self._confirm_delay_s = confirm_delay_s
        self._sleep = sleep
        self._runtime_instance_id = runtime_instance_id
        extra: dict[str, object] = {}
        if mismatch_bound_s is not None:
            extra["mismatch_bound_s"] = mismatch_bound_s
        if wall_clock is not None:
            extra["wall_clock"] = wall_clock
        self._wall_clock = wall_clock or (lambda: datetime.now(timezone.utc))
        self.writer = QuarantineWriter(
            quarantine_dir,
            has_exclusive_authority=has_exclusive_authority,
            t_hash_s=t_hash_s,
            **extra,  # type: ignore[arg-type]
        )
        #: A reason to refuse the start with (set by ``start_cycle``).
        self.start_block: str | None = None
        #: Participants quarantined in THIS cycle.
        self.quarantined_this_cycle: list[str] = []

    # -- roots

    def root_for(self, participant_id: str) -> Path:
        return self._memory_root.parent / STORE_DIRECTORY_NAMES[participant_id]

    # -- the start of a cycle (before any store object is constructed)

    def start_cycle(self) -> None:
        """Finish what a crashed or restarted quarantine left, FIRST THING of a
        cycle: before ``ExperienceStore`` / ``SemanticStore`` exist and before
        any reconciliation (D1 sec. 4.2 step 4).  Never raises: a problem
        becomes ``start_block`` and the cycle ends RECONCILING with nothing
        deleted or guessed."""

        self.start_block = None
        try:
            epochs = self._load_epochs()
            if epochs.unreadable:
                self.start_block = "domain_quarantine_manifest_unreadable"
            for manifest in epochs.manifests:
                self._start_one(manifest)
            self._detect_replaced_stores()
            self._publish_overlay()
        except (QuarantineError, OSError) as error:
            LOGGER.warning("domain quarantine start check failed (%s)", type(error).__name__)
            self.start_block = self.start_block or "domain_quarantine_resume_failed"

    def _load_epochs(self) -> _Epochs:
        found: list[QuarantineManifest] = []
        unreadable = False
        for epoch_id in self.writer.list_epochs():
            try:
                found.append(self.writer.read_manifest(epoch_id))
            except ManifestError:
                unreadable = True
        found.sort(key=lambda item: (item.created_at, item.epoch_id))
        return _Epochs(tuple(found), unreadable)

    def _start_one(self, manifest: QuarantineManifest) -> None:
        participant_id = manifest.participant_id
        if participant_id not in STORE_DIRECTORY_NAMES:
            self.start_block = self.start_block or "domain_quarantine_manifest_unknown"
            return
        if manifest.state is ManifestState.COMMITTED:
            return
        root = self.root_for(participant_id)
        try:
            plan = self.writer.resume(manifest.epoch_id, root)
        except QuarantineError:
            self.start_block = self.start_block or "domain_quarantine_resume_failed"
            return
        if plan.action is ResumeAction.CREATE_EMPTY_ROOT:
            create_empty_root(
                root,
                epoch_id=manifest.epoch_id,
                participant_id=participant_id,
                created_at=self._wall_clock(),
            )
            self.writer.mark_committed(manifest.epoch_id)
        elif plan.action in (ResumeAction.MARK_COMMITTED, ResumeAction.DONE):
            return
        elif plan.action is ResumeAction.CONTINUE_JOURNAL:
            return  # PENDING became MOVED: step 3 follows in ``begin``
        else:  # BLOCKED
            if manifest.state is ManifestState.JOURNALED:
                self.start_block = self.start_block or DOMAIN_STORE_REPLACED_REASON
            else:
                self.start_block = self.start_block or "domain_quarantine_blocked"

    def _detect_replaced_stores(self) -> None:
        """F-15: a COMMITTED epoch exists for P and the live root lacks that
        epoch's marker -> block with no automatic action."""

        latest: dict[str, QuarantineManifest] = {}
        for manifest in self._load_epochs().manifests:
            if manifest.state is ManifestState.COMMITTED:
                latest[manifest.participant_id] = manifest
        for participant_id, manifest in latest.items():
            if participant_id not in STORE_DIRECTORY_NAMES:
                continue
            root = self.root_for(participant_id)
            if os.path.lexists(root) and read_epoch_marker(root) != manifest.epoch_id:
                self.start_block = self.start_block or DOMAIN_STORE_REPLACED_REASON

    def _publish_overlay(self) -> None:
        if self._status is None:
            return
        quarantined = [
            manifest
            for manifest in self._load_epochs().manifests
            if manifest.state is not ManifestState.PENDING
            and manifest.participant_id in STORE_DIRECTORY_NAMES
        ]
        self._status.note_domain_quarantine(
            (m.participant_id for m in quarantined),
            quarantined[-1].epoch_id if quarantined else None,
        )

    # -- reconciliation hooks

    def begin(self, coordinator: "StartupReconciliationCoordinator") -> str | None:
        if self.start_block is not None:
            return self.start_block
        try:
            if coordinator.journal.inspect().schema_version != 4:
                return None
            for manifest in self._load_epochs().manifests:
                if manifest.state is ManifestState.MOVED:
                    self._finish_step3(coordinator, manifest, None)
        except Exception as error:  # a Journal append or a manifest write failed
            LOGGER.warning("domain quarantine resume failed (%s)", type(error).__name__)
            return "external_participant_reconciliation_required"
        return None

    def confirm_delay(self) -> None:
        if self._confirm_delay_s > 0:
            self._sleep(self._confirm_delay_s)

    def should_restart(self) -> bool:
        return bool(self.quarantined_this_cycle)

    def wants_confirmation(
        self, participant_id: str | None, error: BaseException
    ) -> bool:
        return (
            participant_id is not None
            and participant_id in self.enabled_participants
            and is_quarantinable(
                participant_id, classify_participant_error(participant_id, error)
            )
        )

    def resolve_transaction_fault(
        self,
        coordinator: "StartupReconciliationCoordinator",
        participant_id: str | None,
        error: BaseException,
    ) -> bool:
        return self._resolve(coordinator, participant_id, error, None)

    def resolve_aggregate_fault(
        self,
        coordinator: "StartupReconciliationCoordinator",
        participant_id: str,
        error: BaseException,
        reconciliation: "EventJournalStartupReconciliation",
    ) -> bool:
        return self._resolve(coordinator, participant_id, error, reconciliation)

    # -- the action

    def _resolve(
        self,
        coordinator: "StartupReconciliationCoordinator",
        participant_id: str | None,
        error: BaseException,
        reconciliation: "EventJournalStartupReconciliation | None",
    ) -> bool:
        """Never raises (except the process-level ``BaseException`` kinds): a
        failure leaves the manifest where it got to (resumable) and returns
        False, so the legacy ``RECONCILING`` outcome follows."""

        if participant_id is None or participant_id not in self.enabled_participants:
            return False
        verdict = classify_participant_error(participant_id, error)
        if verdict is not IntegrityVerdict.DIVERGED or not is_quarantinable(
            participant_id, verdict
        ):
            return False
        try:
            journal = coordinator.journal
            if journal.inspect().schema_version != 4:
                return False  # a v3 Journal is migrated first; never quarantined
            existing = next(
                (
                    item
                    for item in self._load_epochs().manifests
                    if item.participant_id == participant_id
                    and item.state in (ManifestState.PENDING, ManifestState.MOVED)
                ),
                None,
            )
            if existing is not None:
                if existing.state is ManifestState.PENDING:
                    plan = self.writer.resume(
                        existing.epoch_id, self.root_for(participant_id)
                    )
                    if plan.action is not ResumeAction.CONTINUE_JOURNAL:
                        return False
                manifest = self.writer.read_manifest(existing.epoch_id)
            else:
                manifest = self._move(coordinator, participant_id, reconciliation)
            self._finish_step3(coordinator, manifest, reconciliation)
        except (QuarantineError, OSError, ManifestError) as failure:
            LOGGER.warning(
                "participant quarantine did not complete (%s)", type(failure).__name__
            )
            return False
        except Exception as failure:  # a Journal append failed mid-way
            LOGGER.warning(
                "participant quarantine stopped at the Journal (%s)",
                type(failure).__name__,
            )
            return False
        self.quarantined_this_cycle.append(participant_id)
        return True

    def _move(
        self,
        coordinator: "StartupReconciliationCoordinator",
        participant_id: str,
        reconciliation: "EventJournalStartupReconciliation | None",
    ) -> QuarantineManifest:
        inspection = coordinator.journal.inspect()
        root = self._store_root(coordinator, participant_id)
        transaction_ids = tuple(
            sorted(
                transaction.transaction_id
                for transaction in inspection.open_transactions
                if any(
                    item.participant_id == participant_id
                    for item in transaction.required_participants
                )
            )
        )
        common = {
            "reason": (
                REASON_AGGREGATE if reconciliation is not None else REASON_OPEN_TRANSACTION
            ),
            "participant_id": participant_id,
            "epoch_id": new_epoch_id(self._wall_clock()),
            "journal_lineage_id": inspection.journal_lineage_id,
            "journal_last_record_hash": inspection.tail_record_hash,
            "runtime_instance_id": self._runtime_instance_id,
            "transaction_ids": transaction_ids,
        }
        if os.path.lexists(root):
            return self.writer.quarantine_directory(root, **common)  # type: ignore[arg-type]
        return self.writer.record_absent_root(root, **common)  # type: ignore[arg-type]

    @staticmethod
    def _store_root(
        coordinator: "StartupReconciliationCoordinator", participant_id: str
    ) -> Path:
        store = (
            coordinator.experience_store
            if participant_id == MEMORY_EXPERIENCE
            else coordinator.semantic_store
        )
        return Path(store.root)

    def _finish_step3(
        self,
        coordinator: "StartupReconciliationCoordinator",
        manifest: QuarantineManifest,
        reconciliation: "EventJournalStartupReconciliation | None",
    ) -> None:
        """Step 3 (JOURNALED): attach the outcome AT ONCE to every open
        transaction (and open startup reconciliation) that needs the
        participant; idempotent, so a crash anywhere in it is re-run safely."""

        attached = self._attach(coordinator, manifest.participant_id)
        self.writer.mark_journaled(
            manifest.epoch_id,
            transaction_ids=tuple(sorted({*manifest.transaction_ids, *attached})),
        )
        if manifest.participant_id not in self.quarantined_this_cycle:
            self.quarantined_this_cycle.append(manifest.participant_id)
        self._publish_overlay()

    def _attach(
        self, coordinator: "StartupReconciliationCoordinator", participant_id: str
    ) -> tuple[str, ...]:
        journal = coordinator.journal
        attached: list[str] = []
        transactions = sorted(
            journal.inspect().open_transactions, key=lambda item: item.transaction_id
        )
        for transaction in transactions:
            requirement = next(
                (
                    item
                    for item in transaction.required_participants
                    if item.participant_id == participant_id
                ),
                None,
            )
            if requirement is None:
                continue
            proof = coordinator.state_recovery.inspect_transaction_commit(transaction)
            if proof.classification is InternalCommitClassification.AMBIGUOUS:
                continue  # left to the normal path, which refuses it
            event = coordinator._event(transaction)
            if proof.classification is InternalCommitClassification.PRE_INTERNAL:
                if (
                    ParticipantCapability.ABORT not in requirement.capabilities
                    or participant_id in {pid for pid, _ in transaction.abort_outcomes}
                ):
                    continue
                journal.append_participant_aborted(
                    event,
                    transaction.transaction_id,
                    participant_id,
                    requirement.operation_digest,
                    AbortOutcome.QUARANTINED,
                )
            else:
                known = {pid for pid, _ in transaction.participant_outcomes}
                if participant_id in known:
                    continue
                if transaction.reconciliation_reason is None:
                    required = {
                        item.participant_id for item in transaction.required_participants
                    }
                    journal.append_transaction_reconciliation_required(
                        event,
                        transaction.transaction_id,
                        ReconciliationReason.PARTICIPANT_DIVERGED,
                        tuple(sorted(required - known)),
                    )
                journal.append_participant_finalized(
                    event,
                    transaction.transaction_id,
                    participant_id,
                    requirement.operation_digest,
                    ParticipantOutcome.QUARANTINED,
                )
            attached.append(transaction.transaction_id)
        for reconciliation in journal.inspect().open_startup_reconciliations:
            requirement = next(
                (
                    item
                    for item in reconciliation.required_participants
                    if item.participant_id == participant_id
                ),
                None,
            )
            if requirement is None or participant_id in {
                item[0] for item in reconciliation.participant_outcomes
            }:
                continue
            journal.append_startup_participant_reconciled(
                reconciliation.reconciliation_id,
                reconciliation.recovery_id,
                reconciliation.snapshot_sequence,
                reconciliation.snapshot_hash,
                reconciliation.recovery_processing_high_water,
                reconciliation.wal_generation_id,
                reconciliation.wal_record_id,
                reconciliation.wal_record_hash,
                reconciliation.journal_lineage_id,
                participant_id,
                requirement.operation_digest,
                StartupParticipantOutcome.QUARANTINED,
            )
        return tuple(attached)
