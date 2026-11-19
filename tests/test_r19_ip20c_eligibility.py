"""IP-20c: the pure eligibility, closure-admissibility and clock-floor functions
(IP-20 D1 r3 §4-4, §4-5, §6, §6-4, HD-20-2).

Everything is a synthetic fixture: the IP-13b Wake and the IP-20a view helpers.
These are the functions both the owners and the IP-20d Scheduler pre-filter use,
so each trigger condition of the attempt-outcome and closure operations has a
row here and an owner-level twin in ``test_r19_ip20c_owners.py``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from suzka.r16.history_records import HistoryOperationKind, HistoryRecordOperationV1, RecordKind
from suzka.r16.history_store import HistoryOperationRow, HistoryPhase
from suzka.r19.contracts import (
    ATTEMPT_MAX_EVENTS,
    ATTEMPT_MAX_SECONDS,
    CYCLE_MAX_EVENTS,
    MAX_ATTEMPTS,
    RefuseReason,
    TerminalReason,
    WakeV1,
)
from suzka.r19.ledger import (
    QUEUE_WAIT_LIMIT,
    REVIEW_DEADLINE_MARGIN,
    UNCAPPED_HOLDS,
    ReviewStep,
    WakeHold,
    attempt_end_key,
    claim_eligibility,
    closure_admissible,
    cycle_key,
    derive_wake_view,
    journal_clock_floor,
    raw_claims_of,
    review_eligibility,
    review_started_after,
    terminal_eligibility,
    wake_clock_floor,
)
from suzka.runtime.agent_runtime import AgentEventType
from suzka.runtime.event_journal import EventLifecycle

from test_r19_ip13b_contracts import EXPIRES, INC, NOT_BEFORE, TIMING, _wake
from test_r19_ip20a_ledger import (
    LEASE_END,
    _claim,
    _claim_rec,
    _closure_rec,
    _outcome_rec,
    _terminal,
    _terminal_rec,
    _wake_rec,
)

UTC = timezone.utc


def _at(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


T_DUE = _at("2026-11-16T01:30:00Z")
T_AFTER_LEASE = _at(LEASE_END[0]) + timedelta(seconds=1)
LINEAGE = "lineage-1"


def _entry(records, now: datetime, wake: WakeV1):  # type: ignore[no-untyped-def]
    view = derive_wake_view(records, now.strftime("%Y-%m-%dT%H:%M:%SZ"), clock_floor=None)
    return view.get(wake.wakeup_id)


def _claim_check(records, wake, now=T_DUE, *, live=None, floor=None, raw=None, cycle_live=False):  # type: ignore[no-untyped-def]
    entry = _entry(records, now, wake)
    if raw is None:
        raw = () if entry is None else entry.claims
    return claim_eligibility(
        entry, now=now, live=live, floor=floor, raw_claims=raw, cycle_key_live=cycle_live
    )


# ------------------------------------------------------------ the constants


def test_the_budget_arithmetic_holds_against_the_constants() -> None:
    """D1 §4-6: per attempt 1 claim + at most 1 review = 2 row-writing events;
    per cycle 1 materialize + 3 attempts x 2 + 1 terminal + 1 complete = 9."""

    per_attempt = 1 + 1
    assert per_attempt <= ATTEMPT_MAX_EVENTS
    per_cycle = 1 + MAX_ATTEMPTS * per_attempt + 1 + 1
    assert per_cycle == 9 and per_cycle <= CYCLE_MAX_EVENTS
    assert QUEUE_WAIT_LIMIT == timedelta(seconds=ATTEMPT_MAX_SECONDS)
    assert REVIEW_DEADLINE_MARGIN == timedelta(seconds=5 + ATTEMPT_MAX_SECONDS)
    assert UNCAPPED_HOLDS == {WakeHold.QUEUE_DELAY, WakeHold.IN_FLIGHT}


# ------------------------------------------------------------------- claim


def test_a_due_wake_is_claimable_and_its_window_is_half_open() -> None:
    wake = _wake()
    rows = [_wake_rec(wake)]
    assert _claim_check(rows, wake).proceed
    assert _claim_check(rows, wake, _at(NOT_BEFORE)).proceed
    early = _claim_check(rows, wake, _at(NOT_BEFORE) - timedelta(seconds=1))
    assert early.hold is WakeHold.OUTSIDE_WAKE_WINDOW
    assert _claim_check(rows, wake, _at(EXPIRES)).hold is WakeHold.OUTSIDE_WAKE_WINDOW
    assert _claim_check(rows, wake, _at(EXPIRES) - timedelta(seconds=1)).proceed


def test_an_unknown_terminal_or_closed_wake_is_not_claimed() -> None:
    wake = _wake()
    assert _claim_check([], wake).hold is WakeHold.WAKE_UNKNOWN
    terminal = [_wake_rec(wake), _terminal_rec(_terminal(wake))]
    assert _claim_check(terminal, wake).hold is WakeHold.WAKE_TERMINAL
    first = _claim(wake)
    closed = [_wake_rec(wake), _claim_rec(first), _closure_rec(first)]
    result = _claim_check(closed, wake, T_AFTER_LEASE)
    assert result.existing and not result.proceed  # EXISTING, nothing written


def test_the_gap_rule_waits_for_the_lease_even_with_an_outcome_row() -> None:
    wake = _wake()
    first = _claim(wake)
    rows = [_wake_rec(wake), _claim_rec(first)]
    inside = _at(LEASE_END[0]) - timedelta(seconds=1)
    assert _claim_check(rows, wake, inside).hold is WakeHold.LEASE_LIVE
    # An outcome row records a reason; it never shortens the gap (P2-A).
    refused = [*rows, _outcome_rec(first)]
    assert _claim_check(refused, wake, inside).hold is WakeHold.LEASE_LIVE
    assert _claim_check(refused, wake, _at(LEASE_END[0])).proceed  # half-open


def test_three_attempts_and_no_fourth() -> None:
    wake = _wake()
    claims = [_claim(wake, n) for n in (1, 2, 3)]
    rows = [_wake_rec(wake), *(_claim_rec(c) for c in claims)]
    assert _claim_check(rows, wake, _at(LEASE_END[2]) + timedelta(seconds=5)).hold in (
        WakeHold.ATTEMPTS_USED,
    )


def test_an_operation_in_flight_is_held_and_never_picks_an_index() -> None:
    """Forward note 4: the next attempt index comes from the RAW store.  A raw
    claim that is not adopted yet (PREPARED / FINALIZED-not-terminal) means an
    operation is in flight, and counting only the adopted claims would pick an
    index whose ``wake-attempt:`` key is already live (a PREPARE collision, a
    runtime fail-stop)."""

    wake = _wake()
    first, second = _claim(wake, 1), _claim(wake, 2)
    rows = [_wake_rec(wake), _claim_rec(first)]
    result = _claim_check(rows, wake, T_AFTER_LEASE, raw=(first, second))
    assert result.hold is WakeHold.IN_FLIGHT
    assert WakeHold.IN_FLIGHT in UNCAPPED_HOLDS


def test_a_reserved_cycle_key_blocks_the_claim() -> None:
    wake = _wake()
    rows = [_wake_rec(wake)]
    assert _claim_check(rows, wake, cycle_live=True).hold is WakeHold.CYCLE_RESERVED


def test_the_claim_clock_rules() -> None:
    wake = _wake()
    rows = [_wake_rec(wake)]
    # L < T: the clock went backward between admission and handling.
    assert _claim_check(rows, wake, live=T_DUE - timedelta(seconds=1)).hold is (
        WakeHold.CLOCK_UNVERIFIABLE
    )
    # T below the Journal high-water.
    assert _claim_check(rows, wake, floor=T_DUE + timedelta(seconds=1)).hold is (
        WakeHold.CLOCK_UNVERIFIABLE
    )
    assert _claim_check(rows, wake, floor=T_DUE).proceed
    # Queue wait: a claim must not carry an old T (> 10 s); exactly 10 s is fine.
    assert _claim_check(rows, wake, live=T_DUE + QUEUE_WAIT_LIMIT).proceed
    late = _claim_check(rows, wake, live=T_DUE + QUEUE_WAIT_LIMIT + timedelta(milliseconds=1))
    assert late.hold is WakeHold.QUEUE_DELAY
    # The Scheduler has no queue wait: live is None.
    assert _claim_check(rows, wake, live=None).proceed


# ------------------------------------------------------------- closure mirror


def _claimed():  # type: ignore[no-untyped-def]
    wake = _wake()
    first = _claim(wake)
    entry = _entry([_wake_rec(wake), _claim_rec(first)], T_DUE, wake)
    assert entry is not None
    return wake, first, entry


def _admissible(entry, claim, *, raw=None, live=frozenset(), incarnation=INC):  # type: ignore[no-untyped-def]
    return closure_admissible(
        entry,
        claim,
        incarnation_id=incarnation,
        raw_claims=(claim,) if raw is None else raw,
        live_keys=live,
    )


def test_a_closure_is_admissible_when_every_trigger_condition_passes() -> None:
    _wake_value, claim, entry = _claimed()
    assert _admissible(entry, claim) is None


def test_each_trigger_condition_of_the_outcome_and_resolution_is_mirrored() -> None:
    """D1 §6-4: one row per trigger condition; a violated one is a hold (no
    row), never a recorded refusal."""

    wake, claim, entry = _claimed()
    # the claim belongs to this incarnation
    assert _admissible(entry, claim, incarnation="c" * 64) is WakeHold.INCARNATION_MISMATCH
    # the claim is adopted
    other = _claim(wake, 2)
    assert _admissible(entry, other, raw=(claim, other)) is WakeHold.CLAIM_UNKNOWN
    # the claim is the Wake's latest claim in the RAW store (a PREPARED later
    # claim makes it stale)
    assert _admissible(entry, claim, raw=(claim, other)) is WakeHold.CLAIM_NOT_LATEST
    # the cycle key is free (a closure, a terminal or a PREPARED closure)
    live = frozenset({cycle_key(INC, wake.logical_cycle_id)})
    assert _admissible(entry, claim, live=live) is WakeHold.CYCLE_RESERVED
    # no attempt outcome and no resolution for the claim (``attempt-end:``)
    ended = frozenset({attempt_end_key(claim.claim_id)})
    assert _admissible(entry, claim, live=ended) is WakeHold.ATTEMPT_ENDED
    with_outcome = _entry(
        [_wake_rec(wake), _claim_rec(claim), _outcome_rec(claim)], T_DUE, wake
    )
    assert _admissible(with_outcome, claim) is WakeHold.ATTEMPT_ENDED
    # the Wake is not terminal
    done = _entry([_wake_rec(wake), _claim_rec(claim), _terminal_rec(_terminal(wake))], T_DUE, wake)
    assert _admissible(done, claim) is WakeHold.WAKE_TERMINAL


# ------------------------------------------------------------------ review


def _review(entry, claim, *, now=None, live=None, floor=None, lineage=LINEAGE, raw=None, keys=frozenset(), started=False, incarnation=INC):  # type: ignore[no-untyped-def]
    now = now or T_DUE
    return review_eligibility(
        entry,
        claim.claim_id,
        now=now,
        live=live or now,
        floor=floor,
        journal_lineage_id=lineage,
        incarnation_id=incarnation,
        raw_claims=(claim,) if raw is None else raw,
        live_keys=keys,
        review_started=started,
    )


def _claimed_now() -> tuple[WakeV1, object, object, datetime]:
    """A Wake claimed at 01:30:00; its lease ends at 01:32:00."""

    wake = _wake()
    from suzka.r19.contracts import WakeClaimV1
    from test_r19_ip13b_contracts import EVENT_C

    claim = WakeClaimV1.for_wake(
        wake, 1, claim_event_id=EVENT_C, claim_sequence=21,
        claimed_utc="2026-11-16T01:30:00Z", timing=TIMING,
    )
    entry = _entry([_wake_rec(wake), _claim_rec(claim)], _at("2026-11-16T01:30:30Z"), wake)
    return wake, claim, entry, _at("2026-11-16T01:30:30Z")


def test_a_live_claim_with_everything_passing_proceeds() -> None:
    wake, claim, entry, now = _claimed_now()
    check = _review(entry, claim, now=now, lineage=wake.journal_lineage_id)
    assert check.step is ReviewStep.PROCEED


def test_review_no_row_holds_come_before_any_recordable_refusal() -> None:
    wake, claim, entry, now = _claimed_now()
    lineage = wake.journal_lineage_id
    # A clock refusal AND a violated mirror: the mirror wins (no row).
    check = _review(
        entry, claim, now=now, lineage=lineage, floor=now + timedelta(hours=1),
        keys=frozenset({cycle_key(INC, wake.logical_cycle_id)}),
    )
    assert (check.step, check.hold) == (ReviewStep.HOLD, WakeHold.CYCLE_RESERVED)
    # Mirrors passing: the clock refusal is recordable.
    check = _review(entry, claim, now=now, lineage=lineage, floor=now + timedelta(hours=1))
    assert (check.step, check.refuse) == (ReviewStep.REFUSE, RefuseReason.CLOCK_UNVERIFIABLE)
    check = _review(entry, claim, now=now, lineage=lineage, live=now - timedelta(seconds=1))
    assert (check.step, check.refuse) == (ReviewStep.REFUSE, RefuseReason.CLOCK_UNVERIFIABLE)
    check = _review(entry, claim, now=now, lineage="another-lineage")
    assert (check.step, check.refuse) == (ReviewStep.REFUSE, RefuseReason.LINEAGE_MISMATCH)
    assert _review(entry, claim, now=now, lineage=None).refuse is RefuseReason.LINEAGE_MISMATCH


def test_the_review_wait_bound_is_the_lease_not_ten_seconds() -> None:
    wake, claim, entry, now = _claimed_now()
    lineage = wake.journal_lineage_id
    # A 60 s queue wait inside the lease is not refused and not DEFERRED_BUDGET.
    check = _review(entry, claim, now=now, live=now + timedelta(seconds=60), lineage=lineage)
    assert check.step is ReviewStep.PROCEED
    # At L >= the lease end the attempt is over: a no-row hold.
    over = _review(entry, claim, now=now, live=_at(claim.claim_lease_expires_utc), lineage=lineage)
    assert (over.step, over.hold) == (ReviewStep.HOLD, WakeHold.OUTSIDE_WAKE_WINDOW)
    # T at or after the lease end is outside the claim's window as well.
    late = _review(entry, claim, now=_at(claim.claim_lease_expires_utc), lineage=lineage)
    assert late.hold is WakeHold.OUTSIDE_WAKE_WINDOW


def test_the_review_checks_the_wake_before_the_claim() -> None:
    wake, claim, entry, now = _claimed_now()
    lineage = wake.journal_lineage_id
    assert review_eligibility(
        None, claim.claim_id, now=now, live=now, floor=None, journal_lineage_id=lineage,
        incarnation_id=INC, raw_claims=(), live_keys=frozenset(), review_started=False,
    ).hold is WakeHold.WAKE_UNKNOWN
    terminal = _entry(
        [_wake_rec(wake), _claim_rec(claim), _terminal_rec(_terminal(wake))], now, wake
    )
    assert _review(terminal, claim, now=now, lineage=lineage).hold is WakeHold.WAKE_TERMINAL
    closed = _entry([_wake_rec(wake), _claim_rec(claim), _closure_rec(claim)], now, wake)
    # A resolution that exists is EXISTING, before drift is evaluated (#373 A1-9).
    assert _review(closed, claim, now=now, lineage="other").step is ReviewStep.EXISTING
    unknown = _claim(wake, 2)
    assert review_eligibility(
        entry, unknown.claim_id, now=now, live=now, floor=None, journal_lineage_id=lineage,
        incarnation_id=INC, raw_claims=(claim,), live_keys=frozenset(), review_started=False,
    ).hold is WakeHold.CLAIM_UNKNOWN


def test_one_review_per_claim_holds_a_second() -> None:
    wake, claim, entry, now = _claimed_now()
    check = _review(entry, claim, now=now, lineage=wake.journal_lineage_id, started=True)
    assert (check.step, check.hold) == (ReviewStep.HOLD, WakeHold.REVIEW_ALREADY_STARTED)


# ---------------------------------------------------------------- terminals


def _terminal_check(records, wake, reason, now, *, raw=None, cycle_live=False, floor=None, newer=None):  # type: ignore[no-untyped-def]
    entry = _entry(records, now, wake)
    return terminal_eligibility(
        entry, reason, now=now, live=None, floor=floor,
        raw_claims=(entry.claims if raw is None else raw),
        cycle_key_live=cycle_live, superseding=newer,
    )


def test_expired_wake_needs_expiry_and_no_live_lease() -> None:
    wake = _wake()
    rows = [_wake_rec(wake)]
    E = TerminalReason.EXPIRED_WAKE
    assert _terminal_check(rows, wake, E, _at(EXPIRES) - timedelta(seconds=1)).hold is WakeHold.NOT_DUE
    assert _terminal_check(rows, wake, E, _at(EXPIRES)).proceed
    # EXPIRED_PENDING_LEASE: past expiry with a live lease writes nothing (A3-3).
    claim = _claim(wake)
    claimed = [*rows, _claim_rec(claim)]
    inside = _at(LEASE_END[0]) - timedelta(seconds=1)
    assert _terminal_check(claimed, wake, E, inside).hold is WakeHold.NOT_DUE  # before expiry
    from test_r19_ip13b_contracts import EVENT_C
    from suzka.r19.contracts import WakeClaimV1

    short = _wake(expires_at_utc="2026-11-16T02:01:00Z")
    c = WakeClaimV1.for_wake(
        short, 1, claim_event_id=EVENT_C, claim_sequence=21,
        claimed_utc="2026-11-16T02:00:00Z", timing=TIMING,
    )
    live_lease = [_wake_rec(short), _claim_rec(c)]
    assert c.claim_lease_expires_utc == "2026-11-16T02:01:00Z"  # clipped to the expiry
    assert _terminal_check(live_lease, short, E, _at("2026-11-16T02:01:00Z")).proceed


def test_attempts_exhausted_needs_all_three_claims_and_a_finished_last_attempt() -> None:
    wake = _wake()
    X = TerminalReason.ATTEMPTS_EXHAUSTED
    claims = [_claim(wake, n) for n in (1, 2, 3)]
    rows = [_wake_rec(wake), *(_claim_rec(c) for c in claims)]
    inside = _at("2026-11-16T02:21:00Z")  # the 3rd lease runs to 02:22
    assert _terminal_check(rows, wake, X, inside).hold is WakeHold.NOT_DUE
    # an outcome row on the last claim ends the attempt (the lease still bounds the next claim)
    with_outcome = [*rows, _outcome_rec(claims[2])]
    assert _terminal_check(with_outcome, wake, X, inside).proceed
    assert _terminal_check(rows, wake, X, _at(LEASE_END[2])).proceed
    two = [_wake_rec(wake), _claim_rec(claims[0]), _claim_rec(claims[1])]
    assert _terminal_check(two, wake, X, _at("2026-11-16T03:00:00Z")).hold is WakeHold.NOT_DUE


def test_superseded_needs_may_supersede_and_an_unclaimed_old_wake() -> None:
    from suzka.r19.contracts import WakeCause

    old = _wake(WakeCause.GOAL_ADMITTED)  # no U2c: may be superseded
    S_ = TerminalReason.SUPERSEDED
    new = _wake(
        WakeCause.FEEDBACK_ADMITTED,
        evidence_fingerprint="e" * 64,
        not_before_utc="2026-11-16T01:30:00Z",
    )
    new_entry = _entry([_wake_rec(new)], T_DUE, new)
    rows = [_wake_rec(old)]
    result = _terminal_check(rows, old, S_, T_DUE, newer=new_entry)
    from suzka.r19.contracts import may_supersede

    assert may_supersede(old, new) and result.proceed
    assert _terminal_check(rows, old, S_, T_DUE, newer=None).hold is WakeHold.NOT_SUPERSEDABLE
    claimed = [_wake_rec(old), _claim_rec(_claim(old))]
    assert _terminal_check(claimed, old, S_, T_DUE, newer=new_entry).hold is WakeHold.NOT_SUPERSEDABLE


def test_a_terminal_never_closes_a_closed_or_terminal_wake_nor_runs_on_a_reserved_cycle() -> None:
    wake = _wake()
    E = TerminalReason.EXPIRED_WAKE
    late = _at(EXPIRES) + timedelta(days=1)
    first = _claim(wake)
    closed = [_wake_rec(wake), _claim_rec(first), _closure_rec(first)]
    assert _terminal_check(closed, wake, E, late).hold is WakeHold.WAKE_CLOSED
    done = [_wake_rec(wake), _terminal_rec(_terminal(wake))]
    assert _terminal_check(done, wake, E, late).hold is WakeHold.WAKE_TERMINAL
    assert _terminal_check([_wake_rec(wake)], wake, E, late, cycle_live=True).hold is (
        WakeHold.CYCLE_RESERVED
    )
    # IN_FLIGHT: a raw claim that is not adopted yet.
    assert _terminal_check([_wake_rec(wake)], wake, E, late, raw=(first,)).hold is WakeHold.IN_FLIGHT
    # T below the floor: a terminal needs T >= the Journal high-water too.
    assert _terminal_check([_wake_rec(wake)], wake, E, late, floor=late + timedelta(seconds=1)).hold is (
        WakeHold.CLOCK_UNVERIFIABLE
    )


# --------------------------------------------------------------- the floor


def _record(lifecycle, stamp, event_id="e", event_type=None, sequence=None):  # type: ignore[no-untyped-def]
    return SimpleNamespace(
        lifecycle=lifecycle, timestamp=stamp, event_id=event_id, event_type=event_type,
        processing_sequence=sequence,
    )


A = EventLifecycle.ACCEPTED


def test_the_floor_is_the_maximum_admission_instant_not_the_latest_record() -> None:
    """A rolled-back clock across a restart can make the latest record older
    than an earlier one (D1 §4-5 (a))."""

    records = [
        _record(A, _at("2026-11-16T05:00:00Z"), "a"),
        _record(A, _at("2026-11-16T03:00:00Z"), "b"),  # the clock was rolled back
    ]
    assert journal_clock_floor(records) == _at("2026-11-16T05:00:00Z")
    assert journal_clock_floor([]) is None
    # Only the admission instants count: a later COMPLETED of the previous event
    # must not make a queued event look rolled back.
    records.append(_record(EventLifecycle.COMPLETED, _at("2026-11-16T09:00:00Z"), "b"))
    assert journal_clock_floor(records) == _at("2026-11-16T05:00:00Z")


def test_events_admitted_after_the_handling_event_never_raise_its_floor() -> None:
    records = [
        _record(A, _at("2026-11-16T05:00:00Z"), "a"),
        _record(A, _at("2026-11-16T05:00:10Z"), "me"),
        _record(A, _at("2026-11-16T05:00:20Z"), "later-chat-turn"),
    ]
    assert journal_clock_floor(records, before_event_id="me") == _at("2026-11-16T05:00:00Z")
    assert journal_clock_floor(records) == _at("2026-11-16T05:00:20Z")


def test_the_claim_lower_bound_is_used_without_the_journal() -> None:
    """A lease ends at most ``claim_lease_s`` after its claim, so
    ``lease_end - claim_lease_s`` bounds the claim time from below when the
    Journal record has been archived (D1 §4-5 (b))."""

    wake = _wake()
    claim = _claim(wake)
    bound = _at(claim.claim_lease_expires_utc) - timedelta(seconds=TIMING.claim_lease_s)
    assert wake_clock_floor([], claim, claim_lease_s=TIMING.claim_lease_s) == bound
    assert wake_clock_floor([], None, claim_lease_s=TIMING.claim_lease_s) is None
    # A lease setting lowered after the claim never raises the bound above the claim time.
    assert wake_clock_floor([], claim, claim_lease_s=30) == bound
    later = [_record(A, bound + timedelta(hours=1), "x")]
    assert wake_clock_floor(later, claim, claim_lease_s=TIMING.claim_lease_s) == (
        bound + timedelta(hours=1)
    )
    # The Journal's own term never lowers the bound.
    earlier = [_record(A, bound - timedelta(hours=1), "x")]
    assert wake_clock_floor(earlier, claim, claim_lease_s=TIMING.claim_lease_s) == bound


def test_a_started_review_after_the_claim_is_the_persistent_bound() -> None:
    """HD-20-2 (a): STARTED records only; an ACCEPTED one has no
    ``processing_sequence`` and ran no handler."""

    R = AgentEventType.R16_WAKE_REVIEW
    started = _record(EventLifecycle.STARTED, _at("2026-11-16T01:30:00Z"), "r1", R, 30)
    accepted_only = _record(A, _at("2026-11-16T01:30:00Z"), "r2", R, None)
    chat = _record(EventLifecycle.STARTED, _at("2026-11-16T01:30:00Z"), "c", AgentEventType.PLAN_PREVIEW_REVIEW, 31)
    assert review_started_after([started], 21)
    assert not review_started_after([accepted_only], 21)  # admitted but lost: may repeat
    assert not review_started_after([chat], 21)
    assert not review_started_after([started], 30)  # not after the claim itself
    assert not review_started_after([started], 21, exclude_event_id="r1")  # the review itself
    assert review_started_after([accepted_only, started], 29)


# ----------------------------------------------------------- the raw store


def _row(operation, phase=HistoryPhase.PREPARED, pending=True):  # type: ignore[no-untyped-def]
    return HistoryOperationRow(
        incarnation_id=INC,
        transaction_id="t" + operation.event_id[-4:] + str(operation.processing_sequence),
        event_id=operation.event_id,
        processing_sequence=operation.processing_sequence,
        transaction_kind="event_mutation",
        operation_kind=operation.operation_kind,
        operation_digest=operation.operation_digest,
        phase=phase,
        pending_bytes=operation.canonical_bytes() if pending else None,
        pending_digest=None,
        lineage_scope=None,
        review_intent_key=None,
        predecessor_key=None,
    )


def _claim_operation(claim):  # type: ignore[no-untyped-def]
    return HistoryRecordOperationV1(
        HistoryOperationKind.R19_WAKE_CLAIM, INC, claim.claim_event_id,
        claim.claim_sequence, "event_mutation", RecordKind.WAKE_CLAIM, claim,
    )


def test_raw_claims_count_prepared_and_finalized_rows_but_never_aborted_ones() -> None:
    wake = _wake()
    first, second = _claim(wake, 1), _claim(wake, 2)
    rows = [
        _row(_claim_operation(second), HistoryPhase.PREPARED),
        _row(_claim_operation(first), HistoryPhase.FINALIZED),
    ]
    assert raw_claims_of(rows, wake.wakeup_id) == (first, second)  # by attempt index
    aborted = _row(_claim_operation(second), HistoryPhase.ABORTED, pending=False)
    assert raw_claims_of([rows[1], aborted], wake.wakeup_id) == (first,)
    assert raw_claims_of(rows, "wakeup-" + "0" * 64) == ()


@pytest.mark.parametrize("value", [b"{}", b"not json"])
def test_an_undecodable_raw_envelope_is_an_error_not_a_count(value: bytes) -> None:
    wake = _wake()
    row = _row(_claim_operation(_claim(wake)))
    bad = HistoryOperationRow(
        row.incarnation_id, row.transaction_id, row.event_id, row.processing_sequence,
        row.transaction_kind, row.operation_kind, row.operation_digest, row.phase,
        value, None, None, None, None,
    )
    with pytest.raises((ValueError, TypeError)):
        raw_claims_of([bad], wake.wakeup_id)
