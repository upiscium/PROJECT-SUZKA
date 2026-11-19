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
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Final, TypeVar

from suzka.r16.cycle_contracts import CycleClosureV1, LogicalCycleResolutionV1, utc
from suzka.r16.history_records import RecordKind

if TYPE_CHECKING:
    from suzka.r16.history_reader import AdoptedRecord

from suzka.r19.contracts import (
    ATTEMPT_MAX_SECONDS,
    MAX_ATTEMPTS,
    RefuseReason,
    WakeAttemptOutcomeV1,
    WakeClaimV1,
    WakeTerminalKind,
    WakeTerminalV1,
    TerminalReason,
    WakeV1,
    may_supersede,
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

    Raises ``TypeError`` when an adopted record's ``record`` is not the exact
    type its ``record_kind`` names (a programming or decoding error, never a
    state): the IP-20d tick must catch it, log it and do nothing that tick.

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




# ===========================================================================
# IP-20c: eligibility, closure admissibility and the Journal-based clock floor
#
# One pure function per decision, used by the claim / review / terminal owners
# (inside their serialized events) and by the IP-20d Scheduler's pre-filter, so
# the Scheduler submits only what a handler would pass (D1 §4-3). Nothing here
# reads a store, the Journal or the clock: the caller passes rows, Journal
# records and instants.
# ===========================================================================

# The live-clock wait a claim tolerates (HP-5's 10 s, §4-5); a review's wait is
# bounded by its lease instead.
QUEUE_WAIT_LIMIT: Final = timedelta(seconds=ATTEMPT_MAX_SECONDS)
# §6 step (4): the issuer's 5 s margin plus the handler's own 10 s budget.
REVIEW_DEADLINE_MARGIN: Final = timedelta(seconds=5 + ATTEMPT_MAX_SECONDS)


class WakeHold(str, Enum):
    """Why a Wake handler (or the Scheduler's pre-filter) writes nothing; a
    closed, source-free vocabulary. A hold never ends a cycle or an attempt:
    the attempt ends by its lease (D1 §6-1, last row)."""

    R19_DISABLED = "R19_DISABLED"
    HISTORY_UNAVAILABLE = "HISTORY_UNAVAILABLE"
    WAKE_UNKNOWN = "WAKE_UNKNOWN"
    WAKE_CORRUPT = "WAKE_CORRUPT"
    WAKE_TERMINAL = "WAKE_TERMINAL"
    WAKE_CLOSED = "WAKE_CLOSED"
    # A raw claim / closure operation of this Wake is PREPARED or FINALIZED but
    # not yet adopted, or a same-incarnation key is held by one (transient).
    IN_FLIGHT = "IN_FLIGHT"
    ATTEMPTS_USED = "ATTEMPTS_USED"
    # The gap rule (#373 A2-5): the latest claim's lease has not ended.
    LEASE_LIVE = "LEASE_LIVE"
    OUTSIDE_WAKE_WINDOW = "OUTSIDE_WAKE_WINDOW"
    CLOCK_UNVERIFIABLE = "CLOCK_UNVERIFIABLE"
    QUEUE_DELAY = "QUEUE_DELAY"
    CYCLE_RESERVED = "CYCLE_RESERVED"
    CLAIM_UNKNOWN = "CLAIM_UNKNOWN"
    CLAIM_NOT_LATEST = "CLAIM_NOT_LATEST"
    INCARNATION_MISMATCH = "INCARNATION_MISMATCH"
    ATTEMPT_ENDED = "ATTEMPT_ENDED"
    # HD-20-2 (a): a ``STARTED`` review after this claim is in the Journal.
    REVIEW_ALREADY_STARTED = "REVIEW_ALREADY_STARTED"
    HISTORY_CAPACITY = "HISTORY_CAPACITY"
    COLLISION = "COLLISION"
    NOT_DUE = "NOT_DUE"
    NOT_SUPERSEDABLE = "NOT_SUPERSEDABLE"
    # A result the closure builders cannot represent (defensive, never expected).
    UNREPRESENTABLE = "UNREPRESENTABLE"


# Holds that are transient states of the runtime or the store, not of the Wake
# (D1 §4-3 (ii), r3 P1-new): the Scheduler never counts them against its cap.
UNCAPPED_HOLDS: Final = frozenset({WakeHold.QUEUE_DELAY, WakeHold.IN_FLIGHT})


class ReviewStep(str, Enum):
    PROCEED = "PROCEED"
    # A resolution already exists: nothing is written (#373 A1-9).
    EXISTING = "EXISTING"
    HOLD = "HOLD"
    # A recordable refusal: the attempt counts (only when every mirror passed).
    REFUSE = "REFUSE"


@dataclass(frozen=True, slots=True)
class Eligibility:
    """A handler's pre-checks: proceed, ``existing`` (a resolution exists), or a
    hold."""

    hold: WakeHold | None = None
    existing: bool = False

    @property
    def proceed(self) -> bool:
        return self.hold is None and not self.existing


@dataclass(frozen=True, slots=True)
class ReviewCheck:
    step: ReviewStep
    hold: WakeHold | None = None
    refuse: RefuseReason | None = None

    def __post_init__(self) -> None:
        if (self.step is ReviewStep.HOLD) != (self.hold is not None) or (
            self.step is ReviewStep.REFUSE
        ) != (self.refuse is not None):
            raise ValueError("a hold or refusal step carries exactly its reason")


def cycle_key(incarnation_id: str, logical_cycle_id: str) -> str:
    """The key a closure or a terminal reserves (``history_records``)."""

    return f"cycle:{incarnation_id}:{logical_cycle_id}"


def attempt_end_key(claim_id: str) -> str:
    """The key an attempt outcome or a resolution reserves, once per claim."""

    return f"attempt-end:{claim_id}"


def _second(moment: datetime) -> datetime:
    """The instant as the canonical second-resolution stamps compare it."""

    return moment.astimezone(timezone.utc).replace(microsecond=0)


def _stamp(moment: datetime) -> str:
    return _second(moment).strftime("%Y-%m-%dT%H:%M:%SZ")


def _lease_ended(claim: WakeClaimV1 | None, now: datetime) -> bool:
    return claim is None or _second(now) >= utc(
        claim.claim_lease_expires_utc, "claim_lease_expires_utc"
    )


def _in_claim_window(wake: WakeV1, now: datetime) -> bool:
    """``not_before <= T < min(Wake expiry, Commitment deadline)`` (half-open)."""

    limits = [utc(wake.expires_at_utc, "expires_at_utc")]
    if wake.commitment_deadline_utc is not None:
        limits.append(utc(wake.commitment_deadline_utc, "commitment_deadline_utc"))
    return utc(wake.not_before_utc, "not_before_utc") <= _second(now) < min(limits)


def _clock_hold(
    now: datetime, live: datetime | None, floor: datetime | None
) -> WakeHold | None:
    """``L < T`` or ``T < floor`` (never clamp; D1 §4-5)."""

    if live is not None and live < now:
        return WakeHold.CLOCK_UNVERIFIABLE
    if floor is not None and now < floor:
        return WakeHold.CLOCK_UNVERIFIABLE
    return None


def claim_eligibility(
    entry: WakeEntry | None,
    *,
    incarnation_id: str,
    now: datetime,
    live: datetime | None,
    floor: datetime | None,
    raw_claims: tuple[WakeClaimV1, ...],
    cycle_key_live: bool,
) -> Eligibility:
    """The checks of ``R19_WAKE_CLAIM`` that need only rows and a clock (D1 §4-4
    steps 2-6; the PREPARE collision pre-check and headroom are the owner's).

    ``now`` is T (the event's admission instant), ``live`` the clock read at the
    handler's first line (the Scheduler passes ``None``: it has no queue wait),
    ``floor`` the Journal high-water of ``clock_floor`` and ``raw_claims`` this
    Wake's raw claim operations. Order: the Wake and its closure first, then
    ``IN_FLIGHT``, the attempt budget, the gap rule, the window, and last the
    clock (a claim that cannot be made is no attempt, so none of these records
    anything)."""

    if entry is None:
        return Eligibility(WakeHold.WAKE_UNKNOWN)
    if entry.state is WakeState.CORRUPT:
        return Eligibility(WakeHold.WAKE_CORRUPT)
    if entry.wake.incarnation_id != incarnation_id:
        # The trigger requires the Wake of this incarnation (a mismatch ABORTs).
        return Eligibility(WakeHold.INCARNATION_MISMATCH)
    if entry.terminal is not None:
        return Eligibility(WakeHold.WAKE_TERMINAL)
    if entry.closure is not None:
        return Eligibility(existing=True)
    if len(raw_claims) != entry.attempts:
        # Forward note 4: an operation of this Wake is in flight.
        return Eligibility(WakeHold.IN_FLIGHT)
    if cycle_key_live:
        return Eligibility(WakeHold.CYCLE_RESERVED)
    if next_attempt_index(len(raw_claims)) is None:
        return Eligibility(WakeHold.ATTEMPTS_USED)
    if not _lease_ended(entry.latest_claim, now):
        # P2-A: an outcome row never shortens the gap, only the lease end does.
        return Eligibility(WakeHold.LEASE_LIVE)
    if not _in_claim_window(entry.wake, now):
        return Eligibility(WakeHold.OUTSIDE_WAKE_WINDOW)
    hold = _clock_hold(now, live, floor)
    if hold is not None:
        return Eligibility(hold)
    if live is not None and live - now > QUEUE_WAIT_LIMIT:
        return Eligibility(WakeHold.QUEUE_DELAY)
    return Eligibility()


def closure_admissible(
    entry: WakeEntry,
    claim: WakeClaimV1,
    *,
    incarnation_id: str,
    raw_claims: tuple[WakeClaimV1, ...],
    live_keys: frozenset[str],
) -> WakeHold | None:
    """The mirror of every trigger condition of ``wake_attempt_outcomes_insert``
    and ``logical_cycle_resolutions_insert`` (D1 §6-4, r2 P1-1), over raw store
    state. ``None`` means the trigger would accept; a hold means it would ABORT,
    which in the participant is a runtime fail-stop, so the handler writes no
    row. ``live_keys`` holds the live ones among ``cycle_key(...)`` and
    ``attempt_end_key(claim_id)``.

    The conditions: the claim is adopted, belongs to this Wake's cycle and this
    incarnation; it is the Wake's latest claim in the raw store; the cycle key
    is free (no closure, terminal or PREPARED closure); the claim has no attempt
    outcome and no resolution (both reserve ``attempt-end:<claim_id>``); the
    Wake is not terminal (the terminal reserves the cycle key)."""

    wake = entry.wake
    if entry.terminal is not None:
        return WakeHold.WAKE_TERMINAL
    if (
        claim.incarnation_id != incarnation_id
        or wake.incarnation_id != incarnation_id
        or claim.logical_cycle_id != wake.logical_cycle_id
        or claim.wakeup_id != wake.wakeup_id
    ):
        return WakeHold.INCARNATION_MISMATCH
    adopted = {item.claim_id for item in entry.claims}
    if claim.claim_id not in adopted or not raw_claims:
        return WakeHold.CLAIM_UNKNOWN
    if raw_claims[-1].claim_id != claim.claim_id or len(raw_claims) != entry.attempts:
        return WakeHold.CLAIM_NOT_LATEST
    if cycle_key(incarnation_id, wake.logical_cycle_id) in live_keys:
        return WakeHold.CYCLE_RESERVED
    if attempt_end_key(claim.claim_id) in live_keys or entry.outcome_of(claim.claim_id):
        return WakeHold.ATTEMPT_ENDED
    return None


def review_eligibility(
    entry: WakeEntry | None,
    claim_id: str,
    *,
    now: datetime,
    live: datetime,
    floor: datetime | None,
    journal_lineage_id: str | None,
    incarnation_id: str,
    raw_claims: tuple[WakeClaimV1, ...],
    live_keys: frozenset[str],
    review_started: bool,
) -> ReviewCheck:
    """D1 §6 pre-steps 0-2 for ``R16_WAKE_REVIEW`` of one claim.

    No-row holds come first (the trigger would ABORT, or the attempt is
    already over): an unknown or terminal Wake, a failed ``closure_admissible``
    mirror, ``OUTSIDE_WAKE_WINDOW`` (the lease ended, so a row would add
    nothing), a ``STARTED`` review after this claim. Only if every mirror
    passed can a refusal be recorded: ``CLOCK_UNVERIFIABLE`` (``L < T`` or
    ``T <`` the floor) and ``LINEAGE_MISMATCH``. A resolution that already
    exists is ``EXISTING``, before any drift is evaluated (#373 A1-9)."""

    if entry is None:
        return ReviewCheck(ReviewStep.HOLD, WakeHold.WAKE_UNKNOWN)
    if entry.state is WakeState.CORRUPT:
        return ReviewCheck(ReviewStep.HOLD, WakeHold.WAKE_CORRUPT)
    if entry.closure is not None:
        return ReviewCheck(ReviewStep.EXISTING)
    if entry.terminal is not None:
        return ReviewCheck(ReviewStep.HOLD, WakeHold.WAKE_TERMINAL)
    claim = next((item for item in entry.claims if item.claim_id == claim_id), None)
    if claim is None:
        return ReviewCheck(ReviewStep.HOLD, WakeHold.CLAIM_UNKNOWN)
    hold = closure_admissible(
        entry, claim, incarnation_id=incarnation_id, raw_claims=raw_claims, live_keys=live_keys
    )
    if hold is not None:
        return ReviewCheck(ReviewStep.HOLD, hold)
    lease_end = utc(claim.claim_lease_expires_utc, "claim_lease_expires_utc")
    if live >= lease_end or not claim.in_window(entry.wake, _stamp(now)):
        # The attempt is over: no row (and no 10 s cap on queue wait, §4-5).
        return ReviewCheck(ReviewStep.HOLD, WakeHold.OUTSIDE_WAKE_WINDOW)
    if review_started:
        return ReviewCheck(ReviewStep.HOLD, WakeHold.REVIEW_ALREADY_STARTED)
    if _clock_hold(now, live, floor) is not None:
        return ReviewCheck(ReviewStep.REFUSE, refuse=RefuseReason.CLOCK_UNVERIFIABLE)
    if journal_lineage_id is None or entry.wake.journal_lineage_id != journal_lineage_id:
        return ReviewCheck(ReviewStep.REFUSE, refuse=RefuseReason.LINEAGE_MISMATCH)
    return ReviewCheck(ReviewStep.PROCEED)


def terminal_eligibility(
    entry: WakeEntry | None,
    reason: TerminalReason,
    *,
    incarnation_id: str,
    now: datetime,
    live: datetime | None,
    floor: datetime | None,
    raw_claims: tuple[WakeClaimV1, ...],
    cycle_key_live: bool,
    superseding: WakeEntry | None = None,
) -> Eligibility:
    """The checks of ``R19_WAKE_TERMINAL`` (D1 §4-4 last paragraph, A-4):
    ``EXPIRED_WAKE`` needs ``T >= min(expiry, Commitment deadline)`` and no
    live lease (``EXPIRED_PENDING_LEASE`` writes nothing, #373 A3-3);
    ``ATTEMPTS_EXHAUSTED`` needs all 3 claims in the raw store and the last
    lease over or its outcome row; ``SUPERSEDED`` needs ``may_supersede`` of an
    adopted newer Wake and an old Wake that no claim (raw) has touched. Never
    with a resolution (the cycle closes once), and never while the cycle key is
    held. A terminal has no queue-wait bound (an old T only delays it)."""

    if entry is None:
        return Eligibility(WakeHold.WAKE_UNKNOWN)
    if entry.state is WakeState.CORRUPT:
        return Eligibility(WakeHold.WAKE_CORRUPT)
    if entry.wake.incarnation_id != incarnation_id:
        return Eligibility(WakeHold.INCARNATION_MISMATCH)
    if entry.terminal is not None:
        return Eligibility(WakeHold.WAKE_TERMINAL)
    if entry.closure is not None:
        return Eligibility(WakeHold.WAKE_CLOSED)
    hold = _clock_hold(now, live, floor)
    if hold is not None:
        return Eligibility(hold)
    if cycle_key_live:
        return Eligibility(WakeHold.CYCLE_RESERVED)
    if len(raw_claims) != entry.attempts:
        return Eligibility(WakeHold.IN_FLIGHT)
    claim = entry.latest_claim
    if reason is TerminalReason.EXPIRED_WAKE:
        if _second(now) < utc(entry.expiry_utc, "expiry") or not _lease_ended(claim, now):
            return Eligibility(WakeHold.NOT_DUE)
    elif reason is TerminalReason.ATTEMPTS_EXHAUSTED:
        if len(raw_claims) != MAX_ATTEMPTS or not (
            _lease_ended(claim, now) or entry.latest_outcome is not None
        ):
            return Eligibility(WakeHold.NOT_DUE)
    else:
        if (
            raw_claims
            or entry.claims
            or superseding is None
            or superseding.state is WakeState.CORRUPT
            or not may_supersede(entry.wake, superseding.wake)
        ):
            return Eligibility(WakeHold.NOT_SUPERSEDABLE)
    return Eligibility()




__all__ = [
    "QUEUE_WAIT_LIMIT",
    "REVIEW_DEADLINE_MARGIN",
    "SETTLED_STATES",
    "UNCAPPED_HOLDS",
    "Eligibility",
    "ReviewCheck",
    "ReviewStep",
    "WakeEntry",
    "WakeHold",
    "WakeState",
    "WakeView",
    "attempt_end_key",
    "claim_eligibility",
    "closure_admissible",
    "cycle_key",
    "derive_wake_view",
    "review_eligibility",
    "terminal_eligibility",
]
