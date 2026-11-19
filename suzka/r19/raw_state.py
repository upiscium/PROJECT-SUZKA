"""R19 IP-20c: raw store and Journal reads for the Wake owners (IP-20 D1 r3 §4-4, §4-5).

The pure eligibility functions live in ``suzka.r19.ledger``; what they need from
the raw store (the claim operations of one Wake, PREPARED and FINALIZED-not-yet-
terminal included) and from the retained Journal (the clock high-water floor and
the one-review-per-claim bound, HD-20-2) is read here, from rows and records the
caller passes in.  Nothing here reads a clock or writes.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from typing import Final

from suzka.r16.cycle_contracts import utc
from suzka.r16.history_records import HistoryOperationKind, HistoryRecordOperationV1
from suzka.r16.history_store import HistoryOperationRow, HistoryPhase
from suzka.r19.contracts import WakeClaimV1
from suzka.runtime.agent_runtime import AgentEventType
from suzka.runtime.event_journal import EventJournalRecord, EventLifecycle

# The longest lease a claim can ever have been given (``r19.claim_lease_s`` is
# capped at D-R19 §5's 120 s), so the claim-time lower bound below stays valid
# if the setting was lowered after the claim.
CLAIM_LEASE_CAP_S: Final = 120


def raw_claims_of(
    operations: Iterable[HistoryOperationRow], wakeup_id: str
) -> tuple[WakeClaimV1, ...]:
    """Every claim operation of one Wake in the **raw** store (PREPARED and
    FINALIZED-not-yet-terminal included, never ABORTED), ordered by attempt
    index. ``wake-attempt:`` keys carry no incarnation and the key index is
    global, so the next index is derived from this, never from the adopted view
    (D1 §4-4 step 3, forward note 4). Raises ``ValueError`` / ``TypeError`` on
    an undecodable pending envelope: the caller treats it as an unavailable
    history."""

    found: list[WakeClaimV1] = []
    for row in operations:
        if (
            row.operation_kind is not HistoryOperationKind.R19_WAKE_CLAIM
            or row.phase is HistoryPhase.ABORTED
            or row.pending_bytes is None
        ):
            continue
        record = HistoryRecordOperationV1.from_bytes(row.pending_bytes).record
        if type(record) is WakeClaimV1 and record.wakeup_id == wakeup_id:
            found.append(record)
    return tuple(sorted(found, key=lambda claim: claim.attempt_index))



def _stamp_of(record: EventJournalRecord) -> datetime:
    stamp = record.timestamp
    return stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp.astimezone(timezone.utc)


def journal_clock_floor(
    records: Iterable[EventJournalRecord], *, before_event_id: str | None = None
) -> datetime | None:
    """The high-water of D1 §4-5 (a): the **maximum** admission instant over the
    retained Journal, not the latest record (a rolled-back clock across a
    restart can make the latest older than an earlier one). The admission
    instant of an event is its ``ACCEPTED`` record's timestamp (the runtime's
    ``requested_at``, bound at the same serialized boundary); other lifecycle
    records carry the later times of their own work and would make a queued
    event look rolled back. ``before_event_id`` stops at that event's own
    ``ACCEPTED`` record, so events admitted after it (they queue behind it and
    are legitimately newer) never raise the floor. ``None``: no record."""

    highest: datetime | None = None
    for record in records:
        if record.lifecycle is not EventLifecycle.ACCEPTED:
            continue
        if before_event_id is not None and record.event_id == before_event_id:
            break
        stamp = _stamp_of(record)
        if highest is None or stamp > highest:
            highest = stamp
    return highest


def wake_clock_floor(
    records: Iterable[EventJournalRecord],
    claim: WakeClaimV1 | None,
    *,
    claim_lease_s: int,
    before_event_id: str | None = None,
) -> datetime | None:
    """``max`` of the Journal high-water and the previous claim's lower bound
    ``claim_lease_expires - claim_lease_s`` (a lease ends at most ``claim_lease_s``
    after its claim, so this bounds the claim time from below without the
    Journal, D1 §4-5 (b)). The Wake's creation time (c) and the claim event's own
    record are ``ACCEPTED`` records of the retained Journal, so the first term
    already covers them while they are retained."""

    floor = journal_clock_floor(records, before_event_id=before_event_id)
    if claim is not None:
        bound = utc(claim.claim_lease_expires_utc, "claim_lease_expires_utc") - timedelta(
            seconds=max(claim_lease_s, CLAIM_LEASE_CAP_S)
        )
        floor = bound if floor is None else max(floor, bound)
    return floor


def review_started_after(
    records: Iterable[EventJournalRecord],
    claim_sequence: int,
    *,
    exclude_event_id: str | None = None,
) -> bool:
    """HD-20-2 (a): whether a ``STARTED`` ``R16_WAKE_REVIEW`` record with
    ``processing_sequence > claim_sequence`` is in the retained Journal. An
    ``ACCEPTED`` record has no ``processing_sequence``, so an event admitted but
    lost before it started (no handler ran) may be repeated. The record does not
    name its claim, so a later review of another Wake also answers True
    (conservative: that attempt lapses at its lease)."""

    for record in records:
        if (
            record.lifecycle is EventLifecycle.STARTED
            and record.event_type is AgentEventType.R16_WAKE_REVIEW
            and record.processing_sequence is not None
            and record.processing_sequence > claim_sequence
            and record.event_id != exclude_event_id
        ):
            return True
    return False


__all__ = [
    "CLAIM_LEASE_CAP_S",
    "journal_clock_floor",
    "raw_claims_of",
    "review_started_after",
    "wake_clock_floor",
]
