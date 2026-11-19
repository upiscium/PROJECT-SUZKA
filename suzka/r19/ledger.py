"""R19 IP-20a: the pure derived Wake view (IP-20 D1 r3 §4-2, #373 A1-9).

Input is the adopted records of the current incarnation
(``SubjectHistoryReader.adopted_records().records``) and one instant; output,
per Wake, a state that is **derived and never stored**: no clock in SQL, so
"scheduled", "due", "claimed", "retryable", "expired" and "exhausted" are
always recomputed from rows and the instant (IP-14b §2-7). Nothing here reads
a store, the Journal or the clock, and nothing writes. IP-20b..d (the
materializer, the claim / review / terminal owners and the Scheduler) read
this view; IP-20a ships no writer and no scheduler.

State precedence (D1 §4-2). The first match wins:

1. ``FAILED`` / ``CANCELLED`` (a ``wake_terminals`` row exists);
2. ``CLOSED`` (a ``logical_cycle_resolutions`` row exists: a Decision, an
   EXISTING receipt or a HOLD receipt; a closed Wake is never ``EXPIRED``);
3. ``EXHAUSTED`` (3 claims and the latest lease over, or its outcome row exists;
   it wins over ``EXPIRED``, because ``ATTEMPTS_EXHAUSTED`` is the more
   specific reason);
4. ``EXPIRED_PENDING_LEASE`` / ``EXPIRED`` (``T >= min(expires, Commitment
   deadline)``; pending while the latest claim's lease is still live);
5. ``CLAIMED_LIVE`` (``T <`` the latest claim's lease end);
6. ``RETRYABLE`` (lease over, fewer than 3 attempts; ``T`` is before the expiry
   by step 4, so a lapsed claim of an expired Wake is ``EXPIRED``);
7. ``DUE`` (``not_before <= T <`` the expiry, no claim);
8. ``SCHEDULED`` (``T < not_before``).

A Wake whose rows contradict each other is ``CORRUPT``: the Scheduler leaves it
alone with a logged reason and never writes on top of it (§4-2, A-4).

``LEASE_EXPIRED`` is a derived attempt end (``AttemptOutcome.LEASE_EXPIRED``,
IP-14b §1-1): ``WakeEntry.lease_ended`` says it, and it is never stored.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Final, TypeVar

from suzka.r16.cycle_contracts import CycleClosureV1, LogicalCycleResolutionV1, utc
from suzka.r16.history_records import RecordKind

if TYPE_CHECKING:
    from suzka.r16.history_reader import AdoptedRecord

from suzka.r19.contracts import (
    MAX_ATTEMPTS,
    WakeAttemptOutcomeV1,
    WakeClaimV1,
    WakeTerminalKind,
    WakeTerminalV1,
    WakeV1,
    next_attempt_index,
)


_R = TypeVar("_R")


class WakeState(str, Enum):
    SCHEDULED = "SCHEDULED"
    DUE = "DUE"
    CLAIMED_LIVE = "CLAIMED_LIVE"
    RETRYABLE = "RETRYABLE"
    EXHAUSTED = "EXHAUSTED"
    EXPIRED_PENDING_LEASE = "EXPIRED_PENDING_LEASE"
    EXPIRED = "EXPIRED"
    CLOSED = "CLOSED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    # Contradicting rows (D1 §4-2): never written on top of.
    CORRUPT = "CORRUPT"


# States that need no further Wake action: a result or a terminal exists.
SETTLED_STATES: Final = frozenset(
    {WakeState.CLOSED, WakeState.FAILED, WakeState.CANCELLED}
)


@dataclass(frozen=True, slots=True)
class WakeEntry:
    """One Wake and what the adopted rows say about it at ``as_of_utc``."""

    wake: WakeV1
    state: WakeState
    as_of_utc: str
    # The materializing event (its Journal record bounds the clock floor, §4-5).
    wake_event_id: str
    wake_sequence: int
    claims: tuple[WakeClaimV1, ...]
    outcomes: tuple[WakeAttemptOutcomeV1, ...]
    closure: CycleClosureV1 | None
    terminal: WakeTerminalV1 | None
    corrupt_reason: str | None = None

    @property
    def wakeup_id(self) -> str:
        return self.wake.wakeup_id

    @property
    def attempts(self) -> int:
        return len(self.claims)

    @property
    def latest_claim(self) -> WakeClaimV1 | None:
        return self.claims[-1] if self.claims else None

    @property
    def resolution(self) -> LogicalCycleResolutionV1 | None:
        return None if self.closure is None else self.closure.resolution

    @property
    def next_attempt_index(self) -> int | None:
        """The next claim's index, or None when all 3 attempts are used."""
        return next_attempt_index(self.attempts)

    def outcome_of(self, claim_id: str) -> WakeAttemptOutcomeV1 | None:
        return next((item for item in self.outcomes if item.claim_id == claim_id), None)

    @property
    def latest_outcome(self) -> WakeAttemptOutcomeV1 | None:
        claim = self.latest_claim
        return None if claim is None else self.outcome_of(claim.claim_id)

    @property
    def expiry_utc(self) -> str:
        """``min(Wake expiry, Commitment deadline)``, the end of every window."""
        deadline = self.wake.commitment_deadline_utc
        return (
            self.wake.expires_at_utc
            if deadline is None
            else min(self.wake.expires_at_utc, deadline)
        )

    @property
    def lease_ended(self) -> bool:
        """The latest claim's lease is over at ``as_of_utc`` (derived, never
        stored; D1 §4-4 gap rule: the next claim waits for exactly this)."""
        claim = self.latest_claim
        return claim is None or utc(self.as_of_utc, "as_of_utc") >= utc(
            claim.claim_lease_expires_utc, "claim_lease_expires_utc"
        )

    @property
    def reason(self) -> str | None:
        """A closed code for the state, for logs; never a free text."""
        if self.corrupt_reason is not None:
            return self.corrupt_reason
        if self.terminal is not None:
            return self.terminal.reason.value
        if self.closure is not None:
            return self.closure.resolution.result_kind.value
        outcome = self.latest_outcome
        if outcome is not None:
            return (
                outcome.outcome.value
                if outcome.reason is None
                else outcome.reason.value
            )
        return None


@dataclass(frozen=True, slots=True)
class WakeView:
    """The derived Wake view at ``as_of_utc`` (D1 §4-2)."""

    as_of_utc: str
    # The Journal high-water floor of D1 §4-5 as given by the caller, if any.
    clock_floor_utc: str | None
    entries: tuple[WakeEntry, ...]
    # Records that name no adopted Wake (kind, id): logged, never acted on.
    orphans: tuple[tuple[RecordKind, str], ...] = field(default=())

    @property
    def clock_verifiable(self) -> bool:
        """False when the instant is earlier than the floor (D1 §4-5): a
        claim, review or terminal at such an instant is CLOCK_UNVERIFIABLE."""
        return self.clock_floor_utc is None or utc(self.as_of_utc, "as_of_utc") >= utc(
            self.clock_floor_utc, "clock_floor_utc"
        )

    def get(self, wakeup_id: str) -> WakeEntry | None:
        return next(
            (item for item in self.entries if item.wakeup_id == wakeup_id), None
        )

    def in_state(self, *states: WakeState) -> tuple[WakeEntry, ...]:
        return tuple(item for item in self.entries if item.state in states)

    def by_claim_id(self, claim_id: str) -> WakeEntry | None:
        return next(
            (
                item
                for item in self.entries
                if any(c.claim_id == claim_id for c in item.claims)
            ),
            None,
        )


def derive_wake_view(
    records: Iterable[AdoptedRecord], now_utc: str, *, clock_floor: str | None
) -> WakeView:
    """The view of ``records`` (adopted records) at ``now_utc``.

    ``records`` need only carry ``record_kind``, ``record``, ``event_id`` and
    ``processing_sequence`` (an ``AdoptedRecord``). Entries are ordered by
    ``(not_before_utc, cause_sequence, wakeup_id)``, the Scheduler's claim
    order (D1 §4-3 step 5).
    """

    now = utc(now_utc, "now_utc")
    if clock_floor is not None:
        utc(clock_floor, "clock_floor")
    wakes: dict[str, tuple[WakeV1, str, int]] = {}
    duplicate_wakes: set[str] = set()
    claims: list[WakeClaimV1] = []
    outcomes: list[WakeAttemptOutcomeV1] = []
    closures: list[CycleClosureV1] = []
    terminals: list[WakeTerminalV1] = []
    for item in records:
        kind = item.record_kind
        if kind is RecordKind.WAKE:
            wake = _expect(item.record, WakeV1)
            if wake.wakeup_id in wakes:
                duplicate_wakes.add(wake.wakeup_id)
                continue
            wakes[wake.wakeup_id] = (wake, item.event_id, item.processing_sequence)
        elif kind is RecordKind.WAKE_CLAIM:
            claims.append(_expect(item.record, WakeClaimV1))
        elif kind is RecordKind.WAKE_ATTEMPT_OUTCOME:
            outcomes.append(_expect(item.record, WakeAttemptOutcomeV1))
        elif kind is RecordKind.CYCLE_CLOSURE:
            closures.append(_expect(item.record, CycleClosureV1))
        elif kind is RecordKind.WAKE_TERMINAL:
            terminals.append(_expect(item.record, WakeTerminalV1))

    orphans: list[tuple[RecordKind, str]] = []
    claims_of: dict[str, list[WakeClaimV1]] = {key: [] for key in wakes}
    for claim in claims:
        if claim.wakeup_id in claims_of:
            claims_of[claim.wakeup_id].append(claim)
        else:
            orphans.append((RecordKind.WAKE_CLAIM, claim.claim_id))
    claim_wake = {
        claim.claim_id: claim.wakeup_id for key in claims_of for claim in claims_of[key]
    }
    outcomes_of: dict[str, list[WakeAttemptOutcomeV1]] = {key: [] for key in wakes}
    for outcome in outcomes:
        owner = claim_wake.get(outcome.claim_id)
        if owner is None:
            orphans.append((RecordKind.WAKE_ATTEMPT_OUTCOME, outcome.claim_id))
        else:
            outcomes_of[owner].append(outcome)
    cycle_owner = {
        (w.incarnation_id, w.logical_cycle_id): key
        for key, (w, _e, _s) in wakes.items()
    }
    closures_of: dict[str, list[CycleClosureV1]] = {key: [] for key in wakes}
    for closure in closures:
        owner = cycle_owner.get(
            (closure.resolution.incarnation_id, closure.resolution.logical_cycle_id)
        )
        if owner is None:
            orphans.append(
                (RecordKind.CYCLE_CLOSURE, closure.resolution.logical_cycle_id)
            )
        else:
            closures_of[owner].append(closure)
    terminals_of: dict[str, list[WakeTerminalV1]] = {key: [] for key in wakes}
    for terminal in terminals:
        if terminal.wakeup_id in terminals_of:
            terminals_of[terminal.wakeup_id].append(terminal)
        else:
            orphans.append((RecordKind.WAKE_TERMINAL, terminal.terminal_id))

    entries = [
        _entry(
            wake,
            event_id,
            sequence,
            now_utc,
            now,
            sorted(claims_of[key], key=lambda c: c.attempt_index),
            outcomes_of[key],
            closures_of[key],
            terminals_of[key],
            duplicate=key in duplicate_wakes,
        )
        for key, (wake, event_id, sequence) in wakes.items()
    ]
    entries.sort(
        key=lambda e: (e.wake.not_before_utc, e.wake.cause_sequence, e.wakeup_id)
    )
    return WakeView(now_utc, clock_floor, tuple(entries), tuple(orphans))


def _expect(record: object, expected: type[_R]) -> _R:
    if type(record) is not expected:
        raise TypeError(f"an adopted record of this kind must be a {expected.__name__}")
    return record


def _entry(
    wake: WakeV1,
    event_id: str,
    sequence: int,
    now_utc: str,
    now: datetime,
    claims: list[WakeClaimV1],
    outcomes: list[WakeAttemptOutcomeV1],
    closures: list[CycleClosureV1],
    terminals: list[WakeTerminalV1],
    *,
    duplicate: bool,
) -> WakeEntry:
    closure = closures[0] if closures else None
    terminal = terminals[0] if terminals else None
    corrupt = _corruption(wake, claims, outcomes, closures, terminals, duplicate)
    entry = WakeEntry(
        wake,
        WakeState.CORRUPT,
        now_utc,
        event_id,
        sequence,
        tuple(claims),
        tuple(outcomes),
        closure,
        terminal,
        corrupt,
    )
    if corrupt is not None:
        return entry
    state = _state(entry, now)
    return WakeEntry(
        wake,
        state,
        now_utc,
        event_id,
        sequence,
        entry.claims,
        entry.outcomes,
        closure,
        terminal,
    )


def _corruption(
    wake: WakeV1,
    claims: list[WakeClaimV1],
    outcomes: list[WakeAttemptOutcomeV1],
    closures: list[CycleClosureV1],
    terminals: list[WakeTerminalV1],
    duplicate: bool,
) -> str | None:
    """The D1 §4-2 consistency checks; a code, or None when the rows agree."""

    if duplicate:
        return "DUPLICATE_WAKE"
    for claim in claims:
        if (
            claim.wakeup_id != wake.wakeup_id
            or claim.logical_cycle_id != wake.logical_cycle_id
            or claim.incarnation_id != wake.incarnation_id
            or claim.evidence_fingerprint != wake.evidence_fingerprint
        ):
            return "CLAIM_CYCLE_MISMATCH"
    if [claim.attempt_index for claim in claims] != list(range(1, len(claims) + 1)):
        return "CLAIM_INDEX_GAP"
    if len(claims) > MAX_ATTEMPTS:
        return "CLAIM_INDEX_GAP"
    seen: set[str] = set()
    for outcome in outcomes:
        if outcome.claim_id in seen:
            return "DUPLICATE_OUTCOME"
        seen.add(outcome.claim_id)
    if len(terminals) > 1 or len(closures) > 1:
        return "DUPLICATE_CLOSURE"
    for terminal in terminals:
        # A-4: ``wake_terminals_insert`` requires the Wake's own ``cycle:`` key.
        if terminal.logical_cycle_id != wake.logical_cycle_id:
            return "TERMINAL_CYCLE_MISMATCH"
    if terminals and closures:
        return "TERMINAL_AND_RESOLUTION"
    for closure in closures:
        resolution = closure.resolution
        # A resolution is the result of the Wake's latest claim (one outcome or
        # one resolution per claim: both reserve ``attempt-end:<claim_id>``).
        if not claims or resolution.claim_id != claims[-1].claim_id:
            return "RESOLUTION_NOT_LATEST_CLAIM"
        if resolution.incarnation_id != wake.incarnation_id:
            return "RESOLUTION_INCARNATION"
        if resolution.logical_cycle_id != wake.logical_cycle_id:
            return "RESOLUTION_CYCLE_MISMATCH"
        if any(outcome.claim_id == resolution.claim_id for outcome in outcomes):
            return "RESOLUTION_AND_OUTCOME"
    return None


def _state(entry: WakeEntry, now: datetime) -> WakeState:
    wake = entry.wake
    if entry.terminal is not None:
        if entry.terminal.terminal is WakeTerminalKind.FAILED_TERMINAL:
            return WakeState.FAILED
        return WakeState.CANCELLED
    if entry.closure is not None:
        return WakeState.CLOSED
    claim = entry.latest_claim
    lease_ended = entry.lease_ended
    if entry.attempts >= MAX_ATTEMPTS and (
        lease_ended or entry.latest_outcome is not None
    ):
        return WakeState.EXHAUSTED
    if now >= utc(entry.expiry_utc, "expiry"):
        if claim is not None and not lease_ended:
            return WakeState.EXPIRED_PENDING_LEASE
        return WakeState.EXPIRED
    if claim is not None:
        return WakeState.CLAIMED_LIVE if not lease_ended else WakeState.RETRYABLE
    if now >= utc(wake.not_before_utc, "not_before_utc"):
        return WakeState.DUE
    return WakeState.SCHEDULED


__all__ = [
    "SETTLED_STATES",
    "WakeEntry",
    "WakeState",
    "WakeView",
    "derive_wake_view",
]
