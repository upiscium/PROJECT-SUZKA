"""When to compact the EventJournal (#381 D1/D2, IP-4a).

The Journal owns the compaction itself (``EventJournal.compact``). This
coordinator decides when to try it: at the clean boundary right after
terminal completion, after the StateWAL rollover, never concurrently with
either (#381 D2: rollover and COMPACTION run serially at clean boundaries).

Compaction is maintenance. A failure never fails a committed turn: if it
happens before the COMPACTION record exists nothing durable refers to the
new archive (it is quarantined at the next start); if it happens after,
the next start rolls the unlinks forward.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

from suzka.runtime.event_journal import EventJournal, EventJournalCompactionOutcome
from suzka.runtime.semantic_receipt_retention import (
    SemanticReceiptRetentionCoordinator,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_RETENTION_DAYS = 30


@dataclass(frozen=True, slots=True)
class JournalCompactionPolicy:
    retention_days: int = DEFAULT_RETENTION_DAYS

    def __post_init__(self) -> None:
        if (
            isinstance(self.retention_days, bool)
            or not isinstance(self.retention_days, int)
            or not 0 <= self.retention_days <= 36_500
        ):
            raise ValueError("retention_days is invalid")


class JournalCompactionCoordinator:
    """Owns when a compaction is attempted; owns no state."""

    def __init__(
        self,
        journal: EventJournal,
        policy: JournalCompactionPolicy | None = None,
        semantic: SemanticReceiptRetentionCoordinator | None = None,
    ) -> None:
        self.journal = journal
        self.policy = policy or JournalCompactionPolicy()
        self.semantic = semantic

    def maybe_compact(self) -> EventJournalCompactionOutcome:
        try:
            if not self.journal.compaction_due(
                retention_days=self.policy.retention_days
            ):
                return EventJournalCompactionOutcome(False, "nothing_due")
            semantic = self.semantic
            return self.journal.compact(
                retention_days=self.policy.retention_days,
                carried_tail=None if semantic is None else semantic.carried_tail(),
                guard=None if semantic is None else semantic.compaction_blocked,
            )
        except Exception:
            LOGGER.warning("EventJournal compaction skipped after a failure")
            return EventJournalCompactionOutcome(False, "compaction_failed")


__all__ = [
    "DEFAULT_RETENTION_DAYS",
    "JournalCompactionCoordinator",
    "JournalCompactionPolicy",
]
