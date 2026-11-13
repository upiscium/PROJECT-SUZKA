"""Three-stage source location over a compacted EventJournal (IP-4c).

#381 D1 P1-5/P1-6 and D2 P1-3: an owner that checks an earlier event against
the Journal first looks in the retained records. When the event is absent,
it is *archived* only if its owner-declared processing sequence lies inside
the archived range of the latest COMPACTION; otherwise it is *missing*.

The stage alone is never evidence that the event existed. What an owner may
do with an archived source depends on why it is reading it:

- **Admission** (a new source is being admitted: U2a, a proposal): the
  Journal check is the authority and must be retained. An archived source is
  refused as ``SOURCE_PAST_RETENTION`` when the owner's own durable record
  (for example the committed Experience ``experience_id_for_event``) proves
  that it existed, and as ``SOURCE_MISSING`` otherwise. A sequence merely
  claimed by a caller is never enough.
- **Continuation** (adoption, Plan, review of an already admitted record):
  the owner's committed revision, which was checked against the Journal at
  admission, is the authority. An archived source is accepted when that
  committed evidence backs it. Inside the retention window the independent
  Journal check is kept; outside it the evidence is intentionally derived
  (#376 D2-2). Owners never read the archive.
"""

from __future__ import annotations

from enum import Enum

from suzka.runtime.event_journal import EventJournalInspection, EventJournalRecord


class JournalSourceStage(str, Enum):
    RETAINED = "retained"
    ARCHIVED = "archived"
    MISSING = "missing"


class JournalSourceReason(str, Enum):
    """Closed refusal reasons for an earlier Journal source (#364)."""

    SOURCE_PAST_RETENTION = "source_past_retention"
    SOURCE_MISSING = "source_missing"


def archived_high_water(inspection: EventJournalInspection) -> int:
    """The highest processing sequence covered by archives, or 0."""

    compaction = inspection.latest_compaction
    return 0 if compaction is None else compaction.archived_processing_high_water


def locate_journal_source(
    inspection: EventJournalInspection, event_id: str, event_sequence: int
) -> tuple[JournalSourceStage, tuple[EventJournalRecord, ...]]:
    """Classify one earlier event; retained rows are returned for checking.

    Segments are archived whole and rotation happens only with no open event,
    so an event's records are either all retained or all archived.
    """

    rows = tuple(row for row in inspection.records if row.event_id == event_id)
    if rows:
        return JournalSourceStage.RETAINED, rows
    if (
        type(event_sequence) is int
        and 1 <= event_sequence <= archived_high_water(inspection)
    ):
        return JournalSourceStage.ARCHIVED, ()
    return JournalSourceStage.MISSING, ()


class SourceUnavailable(ValueError):
    """Base of the owners' coarse refusals; may carry a closed reason.

    The reason is a closed enum value: ``JournalSourceReason`` here, or an
    owner's own vocabulary (``suzka.r16.outcomes.R16OwnerReason``).
    """

    def __init__(self, message: str = "", *, reason: Enum | None = None) -> None:
        super().__init__(message)
        self.reason = reason


def reason_of(error: BaseException) -> Enum | None:
    """Carry a refusal reason across an owner's wrapping refusal."""

    return error.reason if isinstance(error, SourceUnavailable) else None
