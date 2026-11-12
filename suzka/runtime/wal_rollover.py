"""Ordinary StateWAL generation rollover at clean R07 boundaries (#382 A/B).

A rollover starts a new StateWAL generation whose baseline is the latest
committed snapshot. It never changes state: ``last_processed_event_sequence``
and the snapshot hash are the same before and after. Its only purpose is to
bound the size of the active generation (and with it inspection and startup
verification cost and disk use).

Protocol, with the Journal lock held for the whole sequence (lock order
Journal -> WAL, #382 D1 P2):

1. write and fsync the new generation (one baseline), not yet published;
2. Journal ``WAL_GENERATION_ROLLED_PREPARED``;
3. publish the manifest atomically;
4. Journal ``WAL_GENERATION_ROLLED_COMPLETED``;
5. delete generations other than active, its predecessor and the one the
   boot anchor points to (at most three are retained; Human decision).

A crash between 2 and 4 is resolved at the next startup
(``StateRecoveryCoordinator`` rolls forward if the manifest was published,
otherwise quarantines the file and records ``ROLLED_ABORTED``).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from uuid import UUID, uuid4

from suzka.runtime.event_journal import EventJournal
from suzka.runtime.state_wal import StateWAL, TransitionRecord

LOGGER = logging.getLogger(__name__)

DEFAULT_ROLLOVER_TRANSITIONS = 32
DEFAULT_ROLLOVER_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class StateWALRolloverPolicy:
    """Roll over once the active generation reaches either bound (#383 IP-3)."""

    max_transitions: int = DEFAULT_ROLLOVER_TRANSITIONS
    max_bytes: int = DEFAULT_ROLLOVER_BYTES

    def __post_init__(self) -> None:
        if self.max_transitions < 1 or self.max_bytes < 1:
            raise ValueError("rollover bounds must be positive")


@dataclass(frozen=True, slots=True)
class StateWALRolloverOutcome:
    performed: bool
    reason: str | None
    generation_id: UUID | None = None
    pruned: tuple[UUID, ...] = ()


class StateWALRolloverError(RuntimeError):
    """A rollover left the Journal with an unresolved preparation."""


class StateWALRolloverCoordinator:
    """Owns when and how an ordinary rollover runs; owns no state."""

    def __init__(
        self,
        journal: EventJournal,
        wal: StateWAL,
        policy: StateWALRolloverPolicy | None = None,
    ) -> None:
        self.journal = journal
        self.wal = wal
        self.policy = policy or StateWALRolloverPolicy()

    def maybe_rollover(self) -> StateWALRolloverOutcome:
        """Roll over if the boundary is clean and the generation is large enough.

        Failures before the Journal preparation are skipped (nothing durable
        refers to the new file; it is pruned later). Failures after it are
        compensated by quarantining the file and recording ``ROLLED_ABORTED``;
        if even that fails, ``StateWALRolloverError`` is raised so the caller
        fails closed, and startup resolves the preparation.
        """
        with self.journal.wal_rollover_boundary() as boundary:
            if not boundary.clean:
                return StateWALRolloverOutcome(False, boundary.reason)
            inspection = self.wal.inspect()
            manifest = inspection.active_manifest
            if (
                manifest is None
                or manifest.external_reconciliation_required
                or not inspection.records
                or str(manifest.active_generation_id) != boundary.wal_generation_id
                or str(inspection.records[-1].record_id) != boundary.wal_record_id
                or inspection.record_hashes[-1] != boundary.wal_record_hash
                or inspection.latest_snapshot_sequence != boundary.snapshot_sequence
                or inspection.latest_snapshot_hash != boundary.snapshot_hash
            ):
                # The WAL tail is not exactly what the Journal anchors: this is
                # not a clean boundary for a rollover.
                return StateWALRolloverOutcome(False, "wal_tail_not_anchored")
            transitions = sum(
                isinstance(record, TransitionRecord) for record in inspection.records
            )
            if (
                transitions < self.policy.max_transitions
                and self.wal.generation_size() < self.policy.max_bytes
            ):
                return StateWALRolloverOutcome(False, "below_threshold")
            generation_id = uuid4()
            rollover_id = uuid4()
            assert boundary.wal_generation_id is not None
            assert boundary.wal_record_hash is not None
            try:
                baseline, baseline_hash = self.wal.prepare_rollover_generation(
                    generation_id=generation_id,
                    journal_processing_high_water=boundary.processing_high_water,
                )
                self.journal.append_wal_rollover_prepared(
                    rollover_id=str(rollover_id),
                    wal_generation_id=str(generation_id),
                    wal_record_id=str(baseline.record_id),
                    wal_record_hash=baseline_hash,
                    wal_predecessor_generation_id=boundary.wal_generation_id,
                    wal_predecessor_record_hash=boundary.wal_record_hash,
                )
            except Exception:
                # Nothing durable refers to the new file yet; it is pruned by
                # a later rollover. Whether PREPARED reached the Journal is
                # decided from the Journal itself, never assumed.
                if self.journal.inspect().open_wal_rollover is not None:
                    self._abort(generation_id, rollover_id)
                return StateWALRolloverOutcome(False, "prepare_failed")
            try:
                self.wal.publish_rollover_generation(
                    generation_id=generation_id, baseline_record_hash=baseline_hash
                )
            except Exception:
                self._abort(generation_id, rollover_id)
                return StateWALRolloverOutcome(False, "publish_failed")
            self.journal.append_wal_rollover_completed(str(rollover_id))
            try:
                pruned = self.wal.prune_generations()
            except Exception:
                # Retention is best effort; the next rollover prunes again.
                LOGGER.warning("StateWAL generation pruning failed")
                pruned = ()
            return StateWALRolloverOutcome(True, None, generation_id, pruned)

    def _abort(self, generation_id: UUID, rollover_id: UUID) -> None:
        try:
            manifest = self.wal.inspect_optional().active_manifest
            if manifest is not None and manifest.active_generation_id == generation_id:
                # Published after all: roll forward instead of aborting.
                self.journal.append_wal_rollover_completed(str(rollover_id))
                return
            self.wal.quarantine_generation(generation_id, rollover_id)
            self.journal.append_wal_rollover_aborted(str(rollover_id))
        except Exception:
            raise StateWALRolloverError(
                "StateWAL rollover could not be resolved"
            ) from None
